//! Signing in through the browser: the person approves on OpenAI's page, the browser comes
//! back to a short-lived server on this machine, and the code it brings is exchanged for tokens.
//! PKCE (a secret made for this one attempt) keeps anyone who sees the code from using it, and
//! a `state` value keeps a stray request from being taken for the answer.

use super::{Credentials, DYNAMIC_CLIENT_ID, OAuth, OAuthError, RESOURCE};
use crate::error::{Error, Result};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::{Digest, Sha256};
use std::time::Duration;

/// The port the sign-in page is told to come back to (the one OpenAI's documentation uses).
pub const DEFAULT_PORT: u16 = 1455;
const CALLBACK_PATH: &str = "/auth/callback";
const SCOPE: &str = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";

pub struct LoginOptions<'a> {
    /// A previous sign-in: its client id and host id are kept, so the app stays one entry in
    /// the person's list of connected apps.
    pub existing: Option<&'a Credentials>,
    /// The name the app gets in that list (used at the first sign-in only).
    pub agent_name: &'a str,
    /// The port to come back to; 0 takes any free one.
    pub port: u16,
    /// Opens the address in the browser. Must return at once.
    pub open_browser: &'a (dyn Fn(&str) -> Result<()> + Sync),
    pub timeout: Duration,
}

/// `base64url(sha256(verifier))`, the PKCE challenge (RFC 7636).
pub(crate) fn challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

fn random_bytes(count: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; count];
    getrandom::fill(&mut bytes)
        .map_err(|e| Error::Io(std::io::Error::other(format!("no randomness: {e}"))))?;
    Ok(bytes)
}

/// `bytes` random bytes as base64url without padding.
pub(crate) fn random_base64url(bytes: usize) -> Result<String> {
    Ok(URL_SAFE_NO_PAD.encode(random_bytes(bytes)?))
}

/// A random UUID (version 4) written as `urn:uuid:...`, which names this machine to the server.
pub(crate) fn new_host_id() -> Result<String> {
    let mut id = random_bytes(16)?;
    id[6] = (id[6] & 0x0f) | 0x40;
    id[8] = (id[8] & 0x3f) | 0x80;
    let hex: String = id.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(format!(
        "urn:uuid:{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    ))
}

/// Percent-encodes everything except the characters that are safe in a URL (RFC 3986).
fn encode_component(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn decode_component(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let decoded = std::str::from_utf8(&bytes[i + 1..i + 3])
                    .ok()
                    .and_then(|hex| u8::from_str_radix(hex, 16).ok());
                match decoded {
                    Some(byte) => {
                        out.push(byte);
                        i += 2;
                    }
                    None => out.push(b'%'),
                }
            }
            other => out.push(other),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The address of the sign-in page, with everything the server needs to know.
#[allow(clippy::too_many_arguments)]
pub(crate) fn authorization_url(
    oauth: &OAuth,
    client_id: &str,
    host_id: &str,
    agent_name: Option<&str>,
    redirect_uri: &str,
    state: &str,
    nonce: &str,
    verifier: &str,
) -> String {
    let challenge = challenge(verifier);
    let mut pairs = vec![
        ("client_id", client_id),
        ("ext_agent_host_id", host_id),
        ("response_type", "code"),
        ("redirect_uri", redirect_uri),
        ("scope", SCOPE),
        ("resource", RESOURCE),
        ("state", state),
        ("nonce", nonce),
        ("code_challenge", challenge.as_str()),
        ("code_challenge_method", "S256"),
    ];
    if let Some(name) = agent_name {
        pairs.push(("agent_name_hint", name));
    }
    let query: Vec<String> = pairs
        .iter()
        .map(|(name, value)| format!("{name}={}", encode_component(value)))
        .collect();
    format!("{}?{}", oauth.authorize_url(), query.join("&"))
}
/// Signs in. Opens the browser, waits for the answer (up to `options.timeout`), checks it and
/// exchanges the code for tokens.
pub fn login(oauth: &OAuth, options: LoginOptions<'_>) -> Result<Credentials> {
    let verifier = random_base64url(48)?;
    let state = random_base64url(24)?;
    let nonce = random_base64url(24)?;
    let (client_id, host_id, agent_name) = match options.existing {
        Some(existing) => {
            let host = if existing.host_id.is_empty() {
                new_host_id()?
            } else {
                existing.host_id.clone()
            };
            (existing.client_id.clone(), host, None)
        }
        None => (
            DYNAMIC_CLIENT_ID.to_string(),
            new_host_id()?,
            Some(options.agent_name),
        ),
    };

    let listener = bind(options.port)?;
    let redirect_uri = format!(
        "http://127.0.0.1:{}{CALLBACK_PATH}",
        listener.local_addr()?.port()
    );
    let url = authorization_url(
        oauth,
        &client_id,
        &host_id,
        agent_name,
        &redirect_uri,
        &state,
        &nonce,
        &verifier,
    );
    (options.open_browser)(&url)?;

    let answer = wait_for_answer(&listener, options.timeout)?;
    let client_id = answer.validate(&state, &client_id)?;
    let mut credentials = oauth
        .exchange_code(&client_id, &answer.code, &verifier, &redirect_uri)
        .map_err(|error| match error {
            OAuthError::Network(_) => Error::ModelUnavailable(error.to_string()),
            _ => Error::SignIn(format!("the server did not accept the sign-in: {error}")),
        })?;
    credentials.host_id = host_id;
    if !credentials.has_plan_scope() {
        return Err(Error::SignIn(
            "plan usage was not granted: allow \"use my ChatGPT plan\" on the consent screen"
                .into(),
        ));
    }
    Ok(credentials)
}

/// Listens on the wanted port, or on any free one if it is taken (an address on the loopback
/// interface may use any port).
fn bind(port: u16) -> Result<std::net::TcpListener> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).or_else(|error| {
        if port == 0 {
            Err(error)
        } else {
            std::net::TcpListener::bind(("127.0.0.1", 0))
        }
    })?;
    listener.set_nonblocking(true)?;
    Ok(listener)
}

/// What the browser brought back to the local address.
struct Answer {
    code: String,
    state: String,
    client_id: String,
    error: String,
}

impl Answer {
    /// Rejects a refusal, a foreign `state` and a first sign-in with no issued client id.
    /// Returns the client id to exchange the code with.
    fn validate(&self, state: &str, known_client_id: &str) -> Result<String> {
        if !self.error.is_empty() {
            return Err(Error::SignIn(format!(
                "sign-in was refused: {}",
                self.error
            )));
        }
        if self.state != state {
            return Err(Error::SignIn(
                "the sign-in answer has an unexpected state; start again".into(),
            ));
        }
        let client_id = if self.client_id.is_empty() {
            known_client_id
        } else {
            &self.client_id
        };
        if client_id == DYNAMIC_CLIENT_ID {
            return Err(Error::SignIn("the server did not issue a client id".into()));
        }
        Ok(client_id.to_string())
    }
}

/// Serves the local address until the browser comes back to the callback path (other requests,
/// like the icon of the page, get "not found" and do not end the wait) or the time is up.
fn wait_for_answer(listener: &std::net::TcpListener, timeout: Duration) -> Result<Answer> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                if let Some(answer) = serve_one(stream) {
                    return Ok(answer);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if std::time::Instant::now() > deadline {
                    return Err(Error::SignIn(
                        "the sign-in was not completed in time: run `nb login` again".into(),
                    ));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(Error::Io(error)),
        }
    }
}

/// Answers one request. Returns the answer if it was the callback.
fn serve_one(stream: std::net::TcpStream) -> Option<Answer> {
    use std::io::{BufRead, BufReader, Write};
    stream.set_nonblocking(false).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).ok()?;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 || line.trim().is_empty() {
            break;
        }
    }
    let target = request_line.split_whitespace().nth(1).unwrap_or("");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut stream = reader.into_inner();
    if path != CALLBACK_PATH {
        let _ = stream
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return None;
    }
    let field = |name: &str| {
        query
            .split('&')
            .filter_map(|pair| pair.split_once('='))
            .find(|(key, _)| decode_component(key) == name)
            .map(|(_, value)| decode_component(value))
            .unwrap_or_default()
    };
    let answer = Answer {
        code: field("code"),
        state: field("state"),
        client_id: field("client_id"),
        error: field("error"),
    };
    let message = if answer.error.is_empty() {
        "Готово. Можно закрыть вкладку и вернуться в программу.".to_string()
    } else {
        format!("Вход отменён: {}", escape_html(&answer.error))
    };
    let page = format!("<!doctype html><meta charset=utf-8><title>Вход</title><p>{message}</p>");
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{page}",
            page.len()
        )
        .as_bytes(),
    );
    Some(answer)
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Opens an address in the default browser.
pub fn open_in_browser(url: &str) -> Result<()> {
    let (program, arguments): (&str, Vec<&str>) = if cfg!(target_os = "macos") {
        ("open", vec![url])
    } else if cfg!(target_os = "windows") {
        ("cmd", vec!["/C", "start", "", url])
    } else {
        ("xdg-open", vec![url])
    };
    let status = std::process::Command::new(program)
        .args(arguments)
        .status()
        .map_err(|error| {
            Error::Io(std::io::Error::other(format!(
                "cannot open the browser with `{program}`: {error}"
            )))
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(Error::Io(std::io::Error::other(format!(
            "`{program}` could not open the browser; open this address yourself: {url}"
        ))))
    }
}

#[cfg(test)]
mod tests {
    use super::super::PLAN_SCOPE;
    use super::super::testing::{FakeServer, Reply};
    use super::*;
    use sha2::{Digest, Sha256};
    use std::sync::{Arc, Mutex};

    #[test]
    fn the_challenge_is_the_one_of_the_standard() {
        // The example of RFC 7636, appendix B.
        assert_eq!(
            challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn random_strings_are_base64url_of_the_right_length_and_never_repeat() {
        let first = random_base64url(48).unwrap();
        let second = random_base64url(48).unwrap();
        assert_eq!(
            first.len(),
            64,
            "48 bytes are 64 characters without padding"
        );
        assert!(
            first
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        );
        assert_ne!(first, second);
    }

    #[test]
    fn a_host_id_is_a_version_4_uuid_urn() {
        let id = new_host_id().unwrap();
        assert_eq!(id.len(), "urn:uuid:".len() + 36, "{id}");
        assert!(id.starts_with("urn:uuid:"));
        let uuid = &id["urn:uuid:".len()..];
        let groups: Vec<&str> = uuid.split('-').collect();
        assert_eq!(
            groups.iter().map(|g| g.len()).collect::<Vec<_>>(),
            [8, 4, 4, 4, 12]
        );
        assert!(groups[2].starts_with('4'), "version: {id}");
        assert!(
            "89ab".contains(groups[3].chars().next().unwrap()),
            "variant: {id}"
        );
        assert_ne!(id, new_host_id().unwrap());
    }

    fn query_of(url: &str) -> Vec<(String, String)> {
        let query = url.split_once('?').expect("a query").1;
        query
            .split('&')
            .map(|pair| {
                let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
                (decode(k), decode(v))
            })
            .collect()
    }

    fn decode(text: &str) -> String {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'+' => out.push(b' '),
                b'%' if i + 2 < bytes.len() => {
                    out.push(u8::from_str_radix(&text[i + 1..i + 3], 16).unwrap());
                    i += 2;
                }
                other => out.push(other),
            }
            i += 1;
        }
        String::from_utf8(out).unwrap()
    }

    fn value<'a>(query: &'a [(String, String)], name: &str) -> Option<&'a str> {
        query
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    #[test]
    fn the_sign_in_address_carries_everything_the_server_needs() {
        let oauth = OAuth::with_urls(
            "https://auth.example/authorize",
            "https://auth.example/token",
        );

        let url = authorization_url(
            &oauth,
            DYNAMIC_CLIENT_ID,
            "urn:uuid:host",
            Some("Мой блокнот"),
            "http://127.0.0.1:1455/auth/callback",
            "STATE",
            "NONCE",
            "VERIFIER",
        );

        assert!(url.starts_with("https://auth.example/authorize?"), "{url}");
        let query = query_of(&url);
        for (name, expected) in [
            ("client_id", DYNAMIC_CLIENT_ID),
            ("ext_agent_host_id", "urn:uuid:host"),
            ("response_type", "code"),
            ("redirect_uri", "http://127.0.0.1:1455/auth/callback"),
            ("resource", RESOURCE),
            ("state", "STATE"),
            ("nonce", "NONCE"),
            ("code_challenge_method", "S256"),
            ("agent_name_hint", "Мой блокнот"),
        ] {
            assert_eq!(value(&query, name), Some(expected), "{name}");
        }
        assert_eq!(
            value(&query, "code_challenge"),
            Some(challenge("VERIFIER").as_str())
        );
        let scope = value(&query, "scope").unwrap();
        assert!(
            scope.split(' ').any(|s| s == PLAN_SCOPE) && scope.contains("offline_access"),
            "{scope}"
        );
        assert!(
            !url.contains(' ') && !url.contains("Мой"),
            "special characters must be encoded: {url}"
        );
    }

    #[test]
    fn a_repeated_sign_in_does_not_send_the_app_name_again() {
        let oauth = OAuth::new();
        let url = authorization_url(
            &oauth,
            "client-1",
            "urn:uuid:host",
            None,
            "http://127.0.0.1:1/cb",
            "S",
            "N",
            "V",
        );
        assert_eq!(value(&query_of(&url), "agent_name_hint"), None);
    }

    /// What the person's browser does: it goes to the address of the sign-in page, and then the
    /// server sends it back to `redirect_uri` with `extra` added to the query. Runs on its own
    /// thread, because `login` is waiting for it.
    fn browser(
        extra: &'static [(&'static str, &'static str)],
        seen: Arc<Mutex<Vec<String>>>,
    ) -> impl Fn(&str) -> Result<()> + Sync {
        move |url: &str| {
            let query = query_of(url);
            let redirect = value(&query, "redirect_uri").unwrap().to_string();
            let state = value(&query, "state").unwrap().to_string();
            seen.lock().unwrap().push(url.to_string());
            std::thread::spawn(move || {
                let mut target = format!("{redirect}?state={state}");
                for (name, content) in extra {
                    if *name == "state" {
                        target = format!("{redirect}?state={content}");
                        continue;
                    }
                    target.push_str(&format!("&{name}={content}"));
                }
                let agent: ureq::Agent = ureq::Agent::config_builder()
                    .http_status_as_error(false)
                    .build()
                    .into();
                let _ = agent.get(&target).call();
            });
            Ok(())
        }
    }

    fn token_server(scope: &'static str) -> FakeServer {
        FakeServer::start(move |_| {
            Reply::json(
                200,
                &format!(
                    r#"{{"access_token":"A","refresh_token":"R","expires_in":3600,"scope":"{scope}"}}"#
                ),
            )
        })
    }

    fn oauth_for(server: &FakeServer) -> OAuth {
        OAuth::with_urls(
            "https://auth.example/authorize",
            &format!("{}/token", server.url),
        )
    }

    fn options<'a>(
        open: &'a (dyn Fn(&str) -> Result<()> + Sync),
        existing: Option<&'a Credentials>,
    ) -> LoginOptions<'a> {
        LoginOptions {
            existing,
            agent_name: "notebook-rs",
            port: 0,
            open_browser: open,
            timeout: Duration::from_secs(10),
        }
    }

    #[test]
    fn a_sign_in_that_goes_well_ends_with_credentials_tied_to_this_attempt_by_pkce() {
        let server = token_server("openid chatgpt.tokens.use.direct");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let open = browser(
            &[("code", "THE-CODE"), ("client_id", "issued-client")],
            seen.clone(),
        );

        let credentials = login(&oauth_for(&server), options(&open, None)).unwrap();

        assert_eq!(credentials.client_id, "issued-client");
        assert_eq!(credentials.access_token, "A");
        assert!(credentials.has_plan_scope());
        assert!(credentials.host_id.starts_with("urn:uuid:"));
        // The secret that was sent to the token endpoint is the one whose hash went into the page.
        let page = query_of(&seen.lock().unwrap()[0]);
        let exchange = &server.requests()[0];
        assert_eq!(exchange.form("code").as_deref(), Some("THE-CODE"));
        assert_eq!(exchange.form("client_id").as_deref(), Some("issued-client"));
        let verifier = exchange.form("code_verifier").unwrap();
        let hashed = Sha256::digest(verifier.as_bytes());
        assert_eq!(
            Some(
                base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, hashed)
                    .as_str()
            ),
            value(&page, "code_challenge")
        );
        assert_eq!(
            exchange.form("redirect_uri").as_deref(),
            value(&page, "redirect_uri"),
            "the same address the page was told"
        );
    }

    #[test]
    fn an_answer_with_a_foreign_state_is_not_taken_and_the_token_server_is_not_asked() {
        let server = token_server("chatgpt.tokens.use.direct");
        let open = browser(
            &[("state", "FOREIGN"), ("code", "X"), ("client_id", "c")],
            Arc::default(),
        );

        let error = login(&oauth_for(&server), options(&open, None)).unwrap_err();

        assert!(matches!(error, Error::SignIn(_)), "{error}");
        assert!(error.to_string().contains("state"), "{error}");
        assert!(server.requests().is_empty());
    }

    #[test]
    fn a_refusal_in_the_browser_is_reported_with_its_reason() {
        let server = token_server("chatgpt.tokens.use.direct");
        let open = browser(&[("error", "access_denied")], Arc::default());
        let error = login(&oauth_for(&server), options(&open, None)).unwrap_err();
        assert!(error.to_string().contains("access_denied"), "{error}");
        assert!(server.requests().is_empty());
    }

    #[test]
    fn a_first_sign_in_without_an_issued_client_id_fails() {
        let server = token_server("chatgpt.tokens.use.direct");
        let open = browser(&[("code", "X")], Arc::default());
        let error = login(&oauth_for(&server), options(&open, None)).unwrap_err();
        assert!(error.to_string().contains("client id"), "{error}");
    }

    #[test]
    fn a_sign_in_that_did_not_grant_the_plan_is_refused_with_advice() {
        let server = token_server("openid");
        let open = browser(&[("code", "X"), ("client_id", "c")], Arc::default());
        let error = login(&oauth_for(&server), options(&open, None)).unwrap_err();
        assert!(matches!(error, Error::SignIn(_)));
        assert!(error.to_string().contains("ChatGPT plan"), "{error}");
    }

    #[test]
    fn a_refusal_of_the_token_server_is_a_sign_in_error() {
        let server = FakeServer::start(|_| Reply::json(400, r#"{"error":"invalid_grant"}"#));
        let open = browser(&[("code", "X"), ("client_id", "c")], Arc::default());
        let error = login(&oauth_for(&server), options(&open, None)).unwrap_err();
        assert!(matches!(error, Error::SignIn(_)), "{error}");
        assert!(error.to_string().contains("invalid_grant"), "{error}");
    }

    #[test]
    fn when_nobody_comes_back_it_gives_up_after_the_timeout() {
        let server = token_server("chatgpt.tokens.use.direct");
        let nothing = |_: &str| Ok(());
        let mut options = options(&nothing, None);
        options.timeout = Duration::from_millis(150);

        let started = std::time::Instant::now();
        let error = login(&oauth_for(&server), options).unwrap_err();

        assert!(
            error.to_string().contains("not completed in time"),
            "{error}"
        );
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn a_repeated_sign_in_keeps_the_client_and_the_host_and_does_not_name_the_app_again() {
        let server = token_server("chatgpt.tokens.use.direct");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let open = browser(&[("code", "X")], seen.clone());
        let existing = super::super::credentials::sample(0);

        let credentials = login(&oauth_for(&server), options(&open, Some(&existing))).unwrap();

        assert_eq!(credentials.client_id, existing.client_id);
        assert_eq!(credentials.host_id, existing.host_id);
        let page = query_of(&seen.lock().unwrap()[0]);
        assert_eq!(value(&page, "client_id"), Some("client-1"));
        assert_eq!(
            value(&page, "ext_agent_host_id"),
            Some(existing.host_id.as_str())
        );
        assert_eq!(value(&page, "agent_name_hint"), None);
    }

    #[test]
    fn the_page_shown_to_the_person_cannot_be_used_to_inject_markup() {
        let server = token_server("chatgpt.tokens.use.direct");
        let body = Arc::new(Mutex::new(String::new()));
        let captured = body.clone();
        let open = move |url: &str| -> Result<()> {
            let query = query_of(url);
            let redirect = value(&query, "redirect_uri").unwrap().to_string();
            let captured = captured.clone();
            std::thread::spawn(move || {
                let agent: ureq::Agent = ureq::Agent::config_builder()
                    .http_status_as_error(false)
                    .build()
                    .into();
                let target = format!("{redirect}?error=%3Cscript%3Ealert(1)%3C/script%3E");
                if let Ok(mut response) = agent.get(&target).call() {
                    *captured.lock().unwrap() =
                        response.body_mut().read_to_string().unwrap_or_default();
                }
            });
            Ok(())
        };

        let _ = login(&oauth_for(&server), options(&open, None));
        // The answer to the browser is written before `login` returns; give the thread a moment.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while body.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }

        let shown = body.lock().unwrap().clone();
        assert!(!shown.is_empty(), "no page came back");
        assert!(!shown.contains("<script>"), "{shown}");
        assert!(shown.contains("&lt;script&gt;"), "{shown}");
    }

    #[test]
    fn other_requests_to_the_local_server_do_not_end_the_wait() {
        let server = token_server("chatgpt.tokens.use.direct");
        let open = move |url: &str| -> Result<()> {
            let query = query_of(url);
            let redirect = value(&query, "redirect_uri").unwrap().to_string();
            let state = value(&query, "state").unwrap().to_string();
            std::thread::spawn(move || {
                let agent: ureq::Agent = ureq::Agent::config_builder()
                    .http_status_as_error(false)
                    .build()
                    .into();
                let base = redirect.trim_end_matches(CALLBACK_PATH).to_string();
                let favicon = agent
                    .get(&format!("{base}/favicon.ico"))
                    .call()
                    .map(|r| r.status().as_u16());
                assert_eq!(favicon.unwrap(), 404);
                let _ = agent
                    .get(&format!("{redirect}?state={state}&code=X&client_id=c"))
                    .call();
            });
            Ok(())
        };
        assert!(login(&oauth_for(&server), options(&open, None)).is_ok());
    }

    #[test]
    fn a_busy_port_falls_back_to_a_free_one() {
        let server = token_server("chatgpt.tokens.use.direct");
        let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let busy_port = busy.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let open = browser(&[("code", "X"), ("client_id", "c")], seen.clone());
        let mut options = options(&open, None);
        options.port = busy_port;

        assert!(login(&oauth_for(&server), options).is_ok());

        let page = query_of(&seen.lock().unwrap()[0]);
        let redirect = value(&page, "redirect_uri").unwrap();
        assert!(!redirect.contains(&format!(":{busy_port}/")), "{redirect}");
    }
}
