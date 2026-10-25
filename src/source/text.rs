use super::{Document, Reader, file_name};
use crate::error::{Error, Result};

/// A text or Markdown file. It accepts anything, so it goes last in the chain.
pub struct TextReader;

impl Reader for TextReader {
    fn accepts(&self, _location: &str) -> bool {
        true
    }

    fn read(&self, location: &str) -> Result<Document> {
        let text = std::fs::read_to_string(location)
            .map_err(|e| Error::Source(format!("{location}: {e}")))?;
        Ok(Document {
            title: file_name(location),
            text,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_file_and_titles_it_with_the_file_name() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("договор.md");
        std::fs::write(&path, "# Договор\n\nТекст.").unwrap();
        let document = TextReader.read(path.to_str().unwrap()).unwrap();
        assert_eq!(document.title, "договор.md");
        assert_eq!(document.text, "# Договор\n\nТекст.");
    }

    #[test]
    fn a_missing_file_is_an_error_that_names_the_file() {
        let error = TextReader
            .read("/no/such/file.txt")
            .unwrap_err()
            .to_string();
        assert!(error.contains("/no/such/file.txt"), "{error}");
    }

    #[test]
    fn a_file_that_is_not_text_is_an_error_not_garbage() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("picture.txt");
        std::fs::write(&path, [0xff, 0xfe, 0x00, 0x80]).unwrap();
        assert!(TextReader.read(path.to_str().unwrap()).is_err());
    }
}
