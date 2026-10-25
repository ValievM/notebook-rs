//! A fake HTTP server for the tests: it records what it receives and answers as told. Every
//! connection is served on its own thread, so a slow answer lets other requests overlap.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Received {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Received {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The value of a field of an `application/x-www-form-urlencoded` body.
    pub fn form(&self, name: &str) -> Option<String> {
        self.body.split('&').find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            (percent_decode(key) == name).then(|| percent_decode(value))
        })
    }
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                out.push(u8::from_str_radix(hex, 16).unwrap_or(b'?'));
                i += 2;
            }
            other => out.push(other),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub struct Reply {
    pub status: u16,
    pub content_type: &'static str,
    pub body: String,
}

impl Reply {
    pub fn json(status: u16, body: &str) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: body.into(),
        }
    }

    #[allow(dead_code)] // used by the tests of the Responses client
    pub fn events(body: &str) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            body: body.into(),
        }
    }
}

pub struct FakeServer {
    pub url: String,
    received: Arc<Mutex<Vec<Received>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FakeServer {
    pub fn start(handler: impl Fn(&Received) -> Reply + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let received = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let handler = Arc::new(handler);
        let thread = {
            let (received, stop) = (received.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let (received, handler) = (received.clone(), handler.clone());
                            std::thread::spawn(move || serve(stream, &received, &*handler));
                        }
                        Err(_) => std::thread::sleep(Duration::from_millis(2)),
                    }
                }
            })
        };
        Self {
            url,
            received,
            stop,
            thread: Some(thread),
        }
    }

    pub fn requests(&self) -> Vec<Received> {
        self.received.lock().unwrap().clone()
    }
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn serve(
    stream: TcpStream,
    received: &Mutex<Vec<Received>>,
    handler: &(dyn Fn(&Received) -> Reply + Send + Sync),
) {
    let _ = stream.set_nonblocking(false);
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
        return;
    }
    let mut parts = request_line.split_whitespace();
    let (method, path) = (
        parts.next().unwrap_or("").to_string(),
        parts.next().unwrap_or("").to_string(),
    );
    let mut headers = Vec::new();
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap_or(0);
            }
            headers.push((name.trim().to_string(), value.trim().to_string()));
        }
    }
    let mut body = vec![0u8; length];
    let _ = reader.read_exact(&mut body);
    let request = Received {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    };
    received.lock().unwrap().push(request.clone());
    let reply = handler(&request);
    let text = format!(
        "HTTP/1.1 {} X\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        reply.status,
        reply.content_type,
        reply.body.len(),
        reply.body
    );
    let mut stream = reader.into_inner();
    let _ = stream.write_all(text.as_bytes());
}
