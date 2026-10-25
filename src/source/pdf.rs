use super::{Document, Reader, file_name};
use crate::error::{Error, Result};

/// A PDF file with a text layer. Scanned pages (pictures of text) need OCR and are refused
/// with a clear message instead of becoming an empty source.
pub struct PdfReader;

impl Reader for PdfReader {
    fn accepts(&self, location: &str) -> bool {
        location.to_lowercase().ends_with(".pdf")
    }

    fn read(&self, location: &str) -> Result<Document> {
        let bytes =
            std::fs::read(location).map_err(|e| Error::Source(format!("{location}: {e}")))?;
        let text = text_from_bytes(&bytes).map_err(|e| match e {
            Error::Source(message) => Error::Source(format!("{location}: {message}")),
            other => other,
        })?;
        Ok(Document {
            title: file_name(location),
            text,
        })
    }
}

/// The text of a PDF held in memory, tidied. Also used for PDFs fetched from the web.
pub(crate) fn text_from_bytes(bytes: &[u8]) -> Result<String> {
    // The parser is third-party code fed with untrusted files and is known to panic on some
    // malformed ones: that must be an error for this source, not the end of the program.
    let raw = std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem(bytes))
        .map_err(|_| Error::Source("the PDF could not be parsed".into()))?
        .map_err(|e| Error::Source(format!("not a readable PDF: {e}")))?;
    let text = tidy_pdf_text(&raw);
    if text.is_empty() {
        return Err(Error::Source(
            "the PDF has no text layer (a scan needs OCR first)".into(),
        ));
    }
    Ok(text)
}

/// Raw PDF text breaks the line at the end of every printed line. Lines of one paragraph are
/// joined with spaces; blank lines and page breaks separate paragraphs.
pub fn tidy_pdf_text(raw: &str) -> String {
    raw.replace("\r\n", "\n")
        .replace('\x0c', "\n\n")
        .split("\n\n")
        .flat_map(|block| block.split("\n \n"))
        .map(|paragraph| {
            paragraph
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|paragraph| !paragraph.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::super::testing::minimal_pdf;
    use super::*;

    #[test]
    fn lines_of_one_paragraph_are_joined_and_paragraphs_stay_apart() {
        assert_eq!(
            tidy_pdf_text("Первая строка\nвторая строка\n\nНовый абзац"),
            "Первая строка вторая строка\n\nНовый абзац"
        );
    }

    #[test]
    fn a_page_break_separates_paragraphs() {
        assert_eq!(
            tidy_pdf_text("конец страницы\x0cначало следующей"),
            "конец страницы\n\nначало следующей"
        );
    }

    #[test]
    fn runs_of_blank_lines_and_trailing_spaces_collapse() {
        assert_eq!(tidy_pdf_text("a  \r\nb\n\n\n\n \nc\n"), "a b\n\nc");
    }

    #[test]
    fn only_pdf_files_are_accepted() {
        assert!(PdfReader.accepts("/tmp/Договор.PDF"));
        assert!(!PdfReader.accepts("/tmp/договор.txt"));
        assert!(!PdfReader.accepts("https://example.com/page"));
    }

    #[test]
    fn reads_the_text_of_a_pdf_and_titles_it_with_the_file_name() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("report.pdf");
        std::fs::write(&path, minimal_pdf("Hello notebook")).unwrap();
        let document = PdfReader.read(path.to_str().unwrap()).unwrap();
        assert_eq!(document.title, "report.pdf");
        assert!(
            document.text.contains("Hello notebook"),
            "{:?}",
            document.text
        );
    }

    #[test]
    fn a_file_that_is_not_a_pdf_is_an_error() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fake.pdf");
        std::fs::write(&path, "this is not a pdf").unwrap();
        assert!(PdfReader.read(path.to_str().unwrap()).is_err());
    }

    #[test]
    fn a_pdf_without_text_is_refused_with_a_hint_about_scans() {
        let error = text_from_bytes(&minimal_pdf("")).unwrap_err().to_string();
        assert!(error.contains("scan"), "{error}");
    }

    #[test]
    fn a_missing_file_is_an_error_that_names_the_file() {
        let error = PdfReader.read("/no/such/file.pdf").unwrap_err().to_string();
        assert!(error.contains("/no/such/file.pdf"), "{error}");
    }
}
