use super::Retriever;
use crate::domain::{NotebookId, Passage, PassageId};
use crate::error::Result;
use std::collections::HashMap;

/// A composite: asks several retrievers and fuses their rankings. A passage that several
/// of them like rises to the top, and one that only one of them found still gets in.
pub struct HybridRetriever {
    parts: Vec<Box<dyn Retriever>>,
}

impl HybridRetriever {
    pub fn new(parts: Vec<Box<dyn Retriever>>) -> Self {
        Self { parts }
    }
}

/// How many candidates each part gives per wanted passage; the fusion needs some room to work.
const CANDIDATES_PER_RESULT: usize = 3;

impl Retriever for HybridRetriever {
    fn retrieve(&self, notebook: NotebookId, question: &str, limit: usize) -> Result<Vec<Passage>> {
        let rankings = self
            .parts
            .iter()
            .map(|part| part.retrieve(notebook, question, limit * CANDIDATES_PER_RESULT))
            .collect::<Result<Vec<_>>>()?;
        let mut fused = reciprocal_rank_fusion(rankings);
        fused.truncate(limit);
        Ok(fused)
    }
}

/// Reciprocal rank fusion: a passage scores the sum of 1 / (60 + rank) over the rankings that
/// contain it. Only ranks count, so scores of different kinds (BM25, cosine) need no common scale.
pub fn reciprocal_rank_fusion(rankings: Vec<Vec<Passage>>) -> Vec<Passage> {
    const K: f64 = 60.0;
    let mut scores: HashMap<PassageId, f64> = HashMap::new();
    let mut by_id: HashMap<PassageId, Passage> = HashMap::new();
    for ranking in rankings {
        for (rank, passage) in ranking.into_iter().enumerate() {
            *scores.entry(passage.id).or_default() += 1.0 / (K + rank as f64 + 1.0);
            by_id.entry(passage.id).or_insert(passage);
        }
    }
    let mut ordered: Vec<(PassageId, f64)> = scores.into_iter().collect();
    // Ties are broken by id so the order never depends on hash-map iteration.
    ordered.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.0.cmp(&b.0.0)));
    ordered
        .into_iter()
        .filter_map(|(id, _)| by_id.remove(&id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(id: i64) -> Passage {
        Passage {
            id: PassageId(id),
            source_title: format!("s{id}"),
            ordinal: 0,
            text: format!("t{id}"),
        }
    }

    fn ids(passages: &[Passage]) -> Vec<i64> {
        passages.iter().map(|p| p.id.0).collect()
    }

    #[test]
    fn passages_both_rankings_like_beat_the_one_only_one_ranks() {
        let fused = reciprocal_rank_fusion(vec![vec![p(1), p(2), p(3)], vec![p(3), p(2)]]);
        // 3: 1/63 + 1/61, 2: 2/62, 1: 1/61 only.
        assert_eq!(ids(&fused), vec![3, 2, 1]);
    }

    #[test]
    fn a_passage_found_by_only_one_ranking_is_kept_once() {
        let fused = reciprocal_rank_fusion(vec![vec![p(1)], vec![p(9)], vec![p(9)]]);
        assert_eq!(ids(&fused), vec![9, 1]);
    }

    #[test]
    fn nothing_in_nothing_out() {
        assert!(reciprocal_rank_fusion(vec![vec![], vec![]]).is_empty());
    }

    struct Fixed(Vec<Passage>);
    impl Retriever for Fixed {
        fn retrieve(&self, _: NotebookId, _: &str, _: usize) -> Result<Vec<Passage>> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn the_hybrid_fuses_its_parts_and_respects_the_limit() {
        let hybrid = HybridRetriever::new(vec![
            Box::new(Fixed(vec![p(1), p(2)])),
            Box::new(Fixed(vec![p(2), p(3)])),
        ]);
        let found = hybrid.retrieve(NotebookId(1), "q", 2).unwrap();
        assert_eq!(ids(&found), vec![2, 1]);
    }
}
