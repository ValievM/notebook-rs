//! Turning text into vectors, so that a question can find a passage that says the same thing
//! in other words.

use crate::error::Result;

pub trait Embedder: Send + Sync {
    /// Names the model. Vectors made by different models are never compared.
    fn model(&self) -> &str;
    /// Vectors for passages that are being stored.
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// The vector of a question. Some models (E5) want questions and passages marked
    /// differently; the default treats them the same.
    fn embed_query(&self, question: &str) -> Result<Vec<f32>> {
        Ok(self.embed(&[question.to_string()])?.remove(0))
    }
}

/// Offline and instant: words (cut a little, like the search does) hashed into a fixed number of
/// buckets. It finds passages that share words in other forms, not synonyms. A real semantic
/// model (fastembed) plugs in behind the same trait.
pub struct HashingEmbedder {
    buckets: usize,
}

impl Default for HashingEmbedder {
    fn default() -> Self {
        Self { buckets: 512 }
    }
}

impl Embedder for HashingEmbedder {
    fn model(&self) -> &str {
        "hashing-512"
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| self.vector(text)).collect())
    }
}

impl HashingEmbedder {
    fn vector(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0_f32; self.buckets];
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.chars().count() >= 3)
        {
            let word = word.to_lowercase();
            let length = word.chars().count();
            let keep = if length > 5 { length - 2 } else { length };
            let stem: String = word.chars().take(keep).collect();
            vector[(fnv1a(stem.as_bytes()) % self.buckets as u64) as usize] += 1.0;
        }
        normalize(&mut vector);
        vector
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

/// Scales a vector to unit length, so that cosine similarity is a plain dot product.
pub fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Cosine similarity of two unit vectors is their dot product.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn similarity(a: &str, b: &str) -> f32 {
        let vectors = HashingEmbedder::default()
            .embed(&[a.into(), b.into()])
            .unwrap();
        cosine(&vectors[0], &vectors[1])
    }

    #[test]
    fn related_texts_are_closer_than_unrelated_ones() {
        let close = similarity(
            "срок действия договора",
            "Срок договора составляет два года",
        );
        let far = similarity("срок действия договора", "купить молоко и хлеб");
        assert!(close > 0.4, "close = {close}");
        assert!(far < 0.1, "far = {far}");
    }

    #[test]
    fn vectors_have_unit_length_and_empty_text_stays_zero() {
        let vectors = HashingEmbedder::default()
            .embed(&["слово другое".into(), "".into()])
            .unwrap();
        let norm = vectors[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!(vectors[1].iter().all(|x| *x == 0.0));
    }
}
