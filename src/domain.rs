//! The words of the domain. Newtype ids keep a notebook id from being used as a passage id.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NotebookId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassageId(pub i64);

/// A passage found for a question, with where it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct Passage {
    pub id: PassageId,
    pub source_title: String,
    pub ordinal: usize,
    pub text: String,
}
