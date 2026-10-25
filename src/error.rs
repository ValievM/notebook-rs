//! One error type for the library. The binary turns it into `anyhow` at the edge.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("storage error: {0}")]
    Storage(#[from] rusqlite::Error),
    #[error("the model is unavailable: {0}")]
    ModelUnavailable(String),
    #[error("the model reply cannot be used: {0}")]
    BadReply(String),
    #[error("cannot read the source: {0}")]
    Source(String),
    #[error("too much text for one request: {0}")]
    TooLarge(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("{0}")]
    Usage(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
