//! Real semantic embeddings, on this machine: fastembed (ONNX Runtime) with a multilingual E5
//! model that understands Russian. Built only with `--features fastembed`; the model is
//! downloaded once into the cache directory on first use.

use crate::embed::{Embedder, normalize};
use crate::error::{Error, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;
use std::sync::Mutex;

pub struct FastEmbedder {
    // `TextEmbedding::embed` needs `&mut self`; the lock keeps the embedder shareable.
    model: Mutex<TextEmbedding>,
}

impl FastEmbedder {
    /// Loads the model, downloading it into `cache_dir` if it is not there yet.
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        let options = TextInitOptions::new(EmbeddingModel::MultilingualE5Small)
            .with_cache_dir(cache_dir)
            .with_show_download_progress(true);
        let model = TextEmbedding::try_new(options).map_err(unavailable)?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn run(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut vectors = self
            .model
            .lock()
            .expect("the model lock is never held across a panic")
            .embed(texts, None)
            .map_err(unavailable)?;
        vectors.iter_mut().for_each(|v| normalize(v));
        Ok(vectors)
    }
}

fn unavailable(error: impl std::fmt::Display) -> Error {
    Error::ModelUnavailable(format!("embedding model: {error}"))
}

impl Embedder for FastEmbedder {
    fn model(&self) -> &str {
        "fastembed-multilingual-e5-small"
    }

    /// E5 models are trained with these markers: passages get "passage: ", questions "query: ".
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.run(texts.iter().map(|t| format!("passage: {t}")).collect())
    }

    fn embed_query(&self, question: &str) -> Result<Vec<f32>> {
        Ok(self.run(vec![format!("query: {question}")])?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::cosine;

    /// Downloads the model (about 120 MB) on the first run, so it is skipped by default:
    /// `cargo test --features fastembed -- --ignored`.
    #[test]
    #[ignore = "downloads the embedding model"]
    fn understands_meaning_not_just_shared_words() {
        let cache = std::env::temp_dir().join("notebook-rs-model-cache");
        let embedder = FastEmbedder::new(cache).unwrap();
        let passages = embedder
            .embed(&[
                "Арендатор обязан освободить помещение за тридцать дней до окончания срока.".into(),
                "Купить молоко и хлеб в магазине у дома.".into(),
            ])
            .unwrap();
        let question = embedder
            .embed_query("За сколько нужно съехать из квартиры?")
            .unwrap();
        let about_lease = cosine(&question, &passages[0]);
        let about_shopping = cosine(&question, &passages[1]);
        assert!(
            about_lease > about_shopping,
            "lease {about_lease} vs shopping {about_shopping}"
        );
    }
}
