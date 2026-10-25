//! Finding the passages that can answer a question. Each way of searching is a `Retriever`;
//! they combine (see `HybridRetriever`) because they fail in different places.

mod bm25;
mod hybrid;
mod vector;

pub use bm25::Bm25Retriever;
pub use hybrid::{HybridRetriever, reciprocal_rank_fusion};
pub use vector::VectorRetriever;

use crate::domain::{NotebookId, Passage};
use crate::error::Result;

pub trait Retriever: Send + Sync {
    /// The best `limit` passages of one notebook for the question, best first.
    fn retrieve(&self, notebook: NotebookId, question: &str, limit: usize) -> Result<Vec<Passage>>;
}
