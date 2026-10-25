use super::Retriever;
use crate::domain::{NotebookId, Passage};
use crate::embed::{Embedder, cosine};
use crate::error::Result;
use crate::store::Store;
use std::sync::Arc;

/// Closeness in meaning: the question and every passage as vectors, compared by cosine.
/// Brute force over one notebook is instant at personal scale (tens of thousands of passages).
pub struct VectorRetriever {
    store: Arc<Store>,
    embedder: Arc<dyn Embedder>,
}

impl VectorRetriever {
    pub fn new(store: Arc<Store>, embedder: Arc<dyn Embedder>) -> Self {
        Self { store, embedder }
    }
}

impl Retriever for VectorRetriever {
    fn retrieve(&self, notebook: NotebookId, question: &str, limit: usize) -> Result<Vec<Passage>> {
        let query = self.embedder.embed_query(question)?;
        let embedded = self
            .store
            .embedded_passages(notebook, self.embedder.model())?;
        let mut scored: Vec<(f32, &Passage)> = embedded
            .iter()
            .map(|(passage, vector)| (cosine(&query, vector), passage))
            .filter(|(score, _)| *score > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        // Only the winners are copied out of the shared cache.
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(_, passage)| passage.clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashingEmbedder;
    use crate::store::NewPassage;

    #[test]
    fn ranks_the_closest_passage_first_and_skips_unrelated_ones() {
        let store = Arc::new(Store::in_memory().unwrap());
        let embedder: Arc<dyn Embedder> = Arc::new(HashingEmbedder::default());
        let nb = store.create_notebook("n").unwrap();
        for (title, text) in [
            ("a", "срок действия договора два года"),
            ("b", "купить молоко и хлеб"),
        ] {
            let vector = embedder.embed(&[text.to_string()]).unwrap().remove(0);
            store
                .add_source(
                    nb,
                    title,
                    &[NewPassage {
                        ordinal: 0,
                        text: text.into(),
                        embedding: Some(vector),
                    }],
                    embedder.model(),
                )
                .unwrap();
        }
        let found = VectorRetriever::new(store, embedder)
            .retrieve(nb, "срок договора", 5)
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].source_title, "a");
    }
}
