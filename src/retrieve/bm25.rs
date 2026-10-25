use super::Retriever;
use crate::domain::{NotebookId, Passage};
use crate::error::Result;
use crate::store::Store;
use std::sync::Arc;

/// Words in common with the question, ranked by BM25.
pub struct Bm25Retriever {
    store: Arc<Store>,
}

impl Bm25Retriever {
    pub fn new(store: Arc<Store>) -> Self {
        Self { store }
    }
}

impl Retriever for Bm25Retriever {
    fn retrieve(&self, notebook: NotebookId, question: &str, limit: usize) -> Result<Vec<Passage>> {
        match fts_query(question) {
            Some(query) => self.store.search_text(notebook, &query, limit),
            None => Ok(Vec::new()),
        }
    }
}

/// Words of the question as prefix terms. Long words are cut a little so that
/// "договору" also finds "договор" and "договоре": a cheap stand-in for stemming.
fn fts_query(question: &str) -> Option<String> {
    let terms: Vec<String> = question
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3)
        .map(|w| {
            let word = w.to_lowercase();
            let length = word.chars().count();
            let keep = if length > 5 { length - 2 } else { length };
            format!("\"{}\"*", word.chars().take(keep).collect::<String>())
        })
        .collect();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::NewPassage;

    fn passage(text: &str) -> NewPassage {
        NewPassage {
            ordinal: 0,
            text: text.into(),
            embedding: None,
        }
    }

    #[test]
    fn finds_the_passage_even_in_another_word_form() {
        let store = Arc::new(Store::in_memory().unwrap());
        let nb = store.create_notebook("n").unwrap();
        store
            .add_source(
                nb,
                "договор.txt",
                &[passage("Срок действия договора составляет два года.")],
                "m",
            )
            .unwrap();
        store
            .add_source(nb, "заметки.txt", &[passage("Купить молоко и хлеб.")], "m")
            .unwrap();
        let found = Bm25Retriever::new(store)
            .retrieve(nb, "Какой срок у договору?", 3)
            .unwrap();
        assert_eq!(found[0].source_title, "договор.txt");
    }

    #[test]
    fn nothing_for_a_question_without_searchable_words() {
        let store = Arc::new(Store::in_memory().unwrap());
        let nb = store.create_notebook("n").unwrap();
        assert!(
            Bm25Retriever::new(store)
                .retrieve(nb, "? !", 3)
                .unwrap()
                .is_empty()
        );
    }
}
