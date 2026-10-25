//! Where a source comes from. A `Reader` turns one kind of location (a web address, a PDF file,
//! a text file) into a title and text; `Sources` is a chain of readers and the first one that
//! accepts the location reads it. A new kind of source is a new `Reader`, nothing else changes.
//!
//! What comes out of a reader is untrusted text (web pages and PDFs may contain instructions
//! aimed at the model). The prompt tells the model to answer only from the passages, and every
//! citation is checked against them, but the text itself is never treated as instructions.

mod pdf;
mod text;
mod web;

pub use pdf::{PdfReader, tidy_pdf_text};
pub use text::TextReader;
pub use web::{WebReader, html_to_markdown};

use crate::error::{Error, Result};
use crate::metrics::Metrics;

/// A source ready to be split into passages: Markdown or plain text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub title: String,
    pub text: String,
}

pub trait Reader: Send + Sync {
    /// Whether this reader knows how to read the location (decided by its shape, no I/O).
    fn accepts(&self, location: &str) -> bool;
    fn read(&self, location: &str) -> Result<Document>;
}

/// A chain of responsibility: readers are tried in order.
pub struct Sources {
    readers: Vec<Box<dyn Reader>>,
    metrics: Metrics,
}

impl Sources {
    pub fn new(readers: Vec<Box<dyn Reader>>) -> Self {
        Self {
            readers,
            metrics: Metrics::off(),
        }
    }

    /// Measures every reading (`source.read`, named by the location): reading is where
    /// a particular file or page most often goes wrong.
    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Web addresses, then PDF files, then everything else as text.
    pub fn standard() -> Self {
        Self::new(vec![
            Box::new(WebReader::default()),
            Box::new(PdfReader),
            Box::new(TextReader),
        ])
    }

    pub fn read(&self, location: &str) -> Result<Document> {
        self.metrics
            .span("source.read")
            .subject(location)
            .run(|span| {
                let document = self
                    .readers
                    .iter()
                    .find(|reader| reader.accepts(location))
                    .ok_or_else(|| Error::Source(format!("no reader for {location}")))?
                    .read(location)?;
                span.set("chars", document.text.chars().count() as i64);
                Ok(document)
            })
    }
}

/// The last part of a path, used as the title of a file source.
pub(crate) fn file_name(location: &str) -> String {
    std::path::Path::new(location)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| location.to_string())
}

#[cfg(test)]
pub(crate) mod testing {
    /// A one-page PDF with `text` in a standard font: the smallest file `pdf-extract` can read.
    /// The offsets of the cross-reference table are computed, so the file is valid.
    pub fn minimal_pdf(text: &str) -> Vec<u8> {
        let stream = format!("BT /F1 24 Tf 72 700 Td ({text}) Tj ET");
        let objects = [
            "<</Type/Catalog/Pages 2 0 R>>".to_string(),
            "<</Type/Pages/Kids[3 0 R]/Count 1>>".to_string(),
            "<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>".to_string(),
            format!("<</Length {}>>\nstream\n{stream}\nendstream", stream.len()),
            "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>".to_string(),
        ];
        let mut pdf = String::from("%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.push_str(&format!("{} 0 obj\n{body}\nendobj\n", i + 1));
        }
        let xref = pdf.len();
        pdf.push_str(&format!(
            "xref\n0 {}\n0000000000 65535 f \n",
            objects.len() + 1
        ));
        for offset in offsets {
            pdf.push_str(&format!("{offset:010} 00000 n \n"));
        }
        pdf.push_str(&format!(
            "trailer\n<</Size {}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        ));
        pdf.into_bytes()
    }

    /// Serves one HTTP response on a free local port and returns the address to fetch.
    pub fn serve_once(status: &str, content_type: &str, body: Vec<u8>) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!(
            "http://127.0.0.1:{}/page",
            listener.local_addr().unwrap().port()
        );
        let head = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request = [0u8; 4096];
                let _ = stream.read(&mut request);
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        address
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        prefix: &'static str,
        title: &'static str,
    }

    impl Reader for Fake {
        fn accepts(&self, location: &str) -> bool {
            location.starts_with(self.prefix)
        }
        fn read(&self, _: &str) -> Result<Document> {
            Ok(Document {
                title: self.title.into(),
                text: String::new(),
            })
        }
    }

    fn chain() -> Sources {
        Sources::new(vec![
            Box::new(Fake {
                prefix: "a:",
                title: "first",
            }),
            Box::new(Fake {
                prefix: "a:b",
                title: "second",
            }),
            Box::new(Fake {
                prefix: "c:",
                title: "third",
            }),
        ])
    }

    #[test]
    fn the_first_reader_that_accepts_the_location_reads_it() {
        assert_eq!(chain().read("a:b").unwrap().title, "first");
        assert_eq!(chain().read("c:x").unwrap().title, "third");
    }

    #[test]
    fn a_location_nobody_accepts_is_an_error_that_names_it() {
        let error = chain().read("zzz").unwrap_err().to_string();
        assert!(error.contains("zzz"), "{error}");
    }

    use crate::metrics::{MemoryRecorder, Metrics};
    use std::sync::Arc;

    fn measured_chain() -> (Sources, Arc<MemoryRecorder>) {
        let recorder = Arc::new(MemoryRecorder::default());
        let sources = chain().with_metrics(Metrics::new(recorder.clone()));
        (sources, recorder)
    }

    #[test]
    fn every_reading_is_measured_by_its_location() {
        let (sources, recorder) = measured_chain();

        sources.read("a:договор.pdf").unwrap();

        let events = recorder.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "source.read");
        assert_eq!(events[0].subject.as_deref(), Some("a:договор.pdf"));
        assert!(events[0].ok);
        assert!(events[0].micros.is_some());
    }

    #[test]
    fn a_reading_that_fails_is_recorded_with_the_location_and_the_message() {
        let (sources, recorder) = measured_chain();

        assert!(sources.read("zzz особый файл").is_err());

        let event = &recorder.events()[0];
        assert!(!event.ok);
        assert_eq!(event.subject.as_deref(), Some("zzz особый файл"));
        assert_eq!(event.error.as_deref(), Some("source"));
        assert!(
            event
                .detail
                .as_deref()
                .unwrap()
                .contains("no reader for zzz")
        );
    }

    #[test]
    fn the_size_of_what_was_read_is_recorded() {
        struct Sized;
        impl Reader for Sized {
            fn accepts(&self, _: &str) -> bool {
                true
            }
            fn read(&self, _: &str) -> Result<Document> {
                Ok(Document {
                    title: "t".into(),
                    text: "четыре".into(),
                })
            }
        }
        let recorder = Arc::new(MemoryRecorder::default());
        let sources =
            Sources::new(vec![Box::new(Sized)]).with_metrics(Metrics::new(recorder.clone()));

        sources.read("x").unwrap();

        assert!(
            recorder.events()[0]
                .fields
                .contains(&("chars".to_string(), 6))
        );
    }

    #[test]
    fn the_standard_chain_sends_each_kind_to_its_reader() {
        let sources = Sources::standard();
        // A missing file is reported by the text reader, a missing PDF by the PDF reader.
        let text = sources
            .read("/definitely/not/here.txt")
            .unwrap_err()
            .to_string();
        let pdf = sources
            .read("/definitely/not/here.pdf")
            .unwrap_err()
            .to_string();
        assert!(text.contains("here.txt"), "{text}");
        assert!(pdf.contains("here.pdf"), "{pdf}");
    }
}
