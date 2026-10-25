//! The facade the binary (and later the window) talks to. It owns the wiring: how a source is
//! split and embedded on the way in, and how a question is searched and answered on the way out.

use crate::answer::{self, Answer};
use crate::chunk::Chunker;
use crate::chunk_markdown::MarkdownChunker;
use crate::deep;
use crate::digest::{self, Budget};
use crate::domain::{NotebookId, Passage};
use crate::embed::Embedder;
use crate::error::{Error, Result};
use crate::llm::Llm;
use crate::retrieve::{Bm25Retriever, HybridRetriever, Retriever, VectorRetriever};
use crate::store::{NewPassage, Store};
use std::sync::Arc;

const PASSAGES_PER_QUESTION: usize = 6;

pub struct Notebooks {
    store: Arc<Store>,
    chunker: Box<dyn Chunker>,
    embedder: Arc<dyn Embedder>,
    retriever: Box<dyn Retriever>,
    llm: Box<dyn Llm>,
    budget: Budget,
}

impl Notebooks {
    pub fn new(
        store: Arc<Store>,
        chunker: Box<dyn Chunker>,
        embedder: Arc<dyn Embedder>,
        retriever: Box<dyn Retriever>,
        llm: Box<dyn Llm>,
    ) -> Self {
        Self {
            store,
            chunker,
            embedder,
            retriever,
            llm,
            budget: Budget::default(),
        }
    }

    /// Changes how much text one model call may get in `summarize` and `compare`.
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// The usual wiring: Markdown-aware chunks (plain text works too), BM25 and vectors fused.
    pub fn standard(store: Store, embedder: Arc<dyn Embedder>, llm: Box<dyn Llm>) -> Self {
        let store = Arc::new(store);
        let retriever = HybridRetriever::new(vec![
            Box::new(Bm25Retriever::new(store.clone())),
            Box::new(VectorRetriever::new(store.clone(), embedder.clone())),
        ]);
        Self::new(
            store,
            Box::new(MarkdownChunker::default()),
            embedder,
            Box::new(retriever),
            llm,
        )
    }

    pub fn notebook(&self, name: &str) -> Result<NotebookId> {
        self.store.create_notebook(name)
    }

    /// Splits, embeds and stores a source; returns how many passages it became.
    pub fn add_source(&self, notebook: NotebookId, title: &str, text: &str) -> Result<usize> {
        let chunks = self.chunker.split(text);
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = self.embedder.embed(&texts)?;
        let passages: Vec<NewPassage> = chunks
            .into_iter()
            .zip(vectors)
            .map(|(chunk, vector)| NewPassage {
                ordinal: chunk.ordinal,
                text: chunk.text,
                embedding: Some(vector),
            })
            .collect();
        self.store
            .add_source(notebook, title, &passages, self.embedder.model())
    }

    pub fn ask(&self, notebook: NotebookId, question: &str) -> Result<Answer> {
        let passages = self
            .retriever
            .retrieve(notebook, question, PASSAGES_PER_QUESTION)?;
        answer::ask(self.llm.as_ref(), question, &passages)
    }

    /// The passages of a source (or of the whole notebook) in reading order; an unknown title is
    /// an error that lists the titles there are.
    fn passages_of(&self, notebook: NotebookId, title: Option<&str>) -> Result<Vec<Passage>> {
        if let Some(wanted) = title {
            let titles = self.store.source_titles(notebook)?;
            if !titles.iter().any(|known| known == wanted) {
                return Err(Error::NotFound(format!(
                    "no source «{wanted}» in the notebook; there are: {}",
                    titles.join(", ")
                )));
            }
        }
        self.store.source_passages(notebook, title)
    }

    /// A summary of one source, or of the whole notebook when `source` is `None`.
    pub fn summarize(&self, notebook: NotebookId, source: Option<&str>) -> Result<Answer> {
        let passages = self.passages_of(notebook, source)?;
        digest::summarize(self.llm.as_ref(), &passages, &self.budget)
    }

    /// What two sources of the notebook have in common, how they differ, where they contradict.
    pub fn compare(&self, notebook: NotebookId, first: &str, second: &str) -> Result<Answer> {
        if first == second {
            return Err(Error::Usage(
                "choose two different sources to compare".into(),
            ));
        }
        let first = self.passages_of(notebook, Some(first))?;
        let second = self.passages_of(notebook, Some(second))?;
        digest::compare(self.llm.as_ref(), &first, &second, &self.budget)
    }

    /// For questions that need several searches (comparisons, "and also..."): the model plans the
    /// searches first, so this takes two model calls instead of one.
    pub fn ask_deep(&self, notebook: NotebookId, question: &str) -> Result<Answer> {
        deep::research(
            self.llm.as_ref(),
            self.retriever.as_ref(),
            notebook,
            question,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashingEmbedder;
    use std::sync::Mutex;

    /// Remembers the prompt it was given and answers with a fixed reply.
    struct Recording {
        reply: &'static str,
        prompts: Arc<Mutex<Vec<String>>>,
    }
    impl Llm for Recording {
        fn complete(&self, prompt: &str) -> Result<String> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            Ok(self.reply.to_string())
        }
    }

    #[test]
    fn a_question_reaches_the_model_with_the_right_passage_and_the_answer_is_checked() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let llm = Recording {
            reply: r#"{"answer":"Два года [1].","citations":[{"n":1,"quote":"два года"},{"n":1,"quote":"выдумка"}]}"#,
            prompts: prompts.clone(),
        };
        let notebooks = Notebooks::standard(
            Store::in_memory().unwrap(),
            Arc::new(HashingEmbedder::default()),
            Box::new(llm),
        );
        let nb = notebooks.notebook("Диплом").unwrap();
        notebooks
            .add_source(
                nb,
                "договор.txt",
                "Срок действия договора составляет два года.",
            )
            .unwrap();
        notebooks
            .add_source(nb, "заметки.txt", "Купить молоко и хлеб.")
            .unwrap();

        let answer = notebooks.ask(nb, "Какой срок у договора?").unwrap();

        assert_eq!(answer.citations.len(), 1);
        assert_eq!(answer.rejected_quotes, 1);
        let prompt = prompts.lock().unwrap()[0].clone();
        assert!(prompt.contains("договор.txt"));
        assert!(!prompt.contains("молоко"));
    }

    /// Plan first, then the answer; remembers the prompts.
    struct PlanThenAnswer {
        calls: Mutex<Vec<String>>,
    }
    impl Llm for PlanThenAnswer {
        fn complete(&self, prompt: &str) -> Result<String> {
            let mut calls = self.calls.lock().unwrap();
            calls.push(prompt.to_string());
            Ok(if calls.len() == 1 {
                r#"{"searches":[{"term":"срок договора","instructions":"найди срок"},
                    {"term":"порядок оплаты","instructions":"найди оплату"}]}"#
            } else {
                r#"{"answer":"Два года, оплата ежемесячно [1][2].","citations":[
                    {"n":1,"quote":"два года"},{"n":2,"quote":"ежемесячно"}]}"#
            }
            .to_string())
        }
    }

    fn notebook_with_three_sources(prompts: &Arc<Mutex<Vec<String>>>) -> (Notebooks, NotebookId) {
        let llm = Recording {
            reply: r#"{"answer":"ок"}"#,
            prompts: prompts.clone(),
        };
        let notebooks = Notebooks::standard(
            Store::in_memory().unwrap(),
            Arc::new(HashingEmbedder::default()),
            Box::new(llm),
        );
        let nb = notebooks.notebook("Диплом").unwrap();
        for (title, text) in [
            ("договор.txt", "Срок действия договора два года."),
            ("памятка.txt", "О выезде предупредить за тридцать дней."),
            ("заметки.txt", "Купить молоко и хлеб."),
        ] {
            notebooks.add_source(nb, title, text).unwrap();
        }
        (notebooks, nb)
    }

    #[test]
    fn a_summary_of_the_notebook_reads_every_source_in_order() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = notebook_with_three_sources(&prompts);
        notebooks.summarize(nb, None).unwrap();
        let prompt = prompts.lock().unwrap()[0].clone();
        assert!(prompt.contains("Задание: Составь конспект"));
        let at = |needle: &str| prompt.find(needle).unwrap();
        assert!(at("договор.txt") < at("памятка.txt") && at("памятка.txt") < at("заметки.txt"));
    }

    #[test]
    fn a_summary_of_one_source_uses_only_that_source() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = notebook_with_three_sources(&prompts);
        notebooks.summarize(nb, Some("памятка.txt")).unwrap();
        let prompt = prompts.lock().unwrap()[0].clone();
        assert!(prompt.contains("тридцать дней"));
        assert!(!prompt.contains("договора") && !prompt.contains("молоко"));
    }

    #[test]
    fn an_unknown_source_is_not_found_and_the_error_lists_the_sources() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = notebook_with_three_sources(&prompts);
        let error = notebooks.summarize(nb, Some("нет.txt")).unwrap_err();
        assert!(matches!(error, crate::error::Error::NotFound(_)));
        let message = error.to_string();
        assert!(
            message.contains("нет.txt") && message.contains("договор.txt"),
            "{message}"
        );
        assert!(prompts.lock().unwrap().is_empty());
    }

    #[test]
    fn a_comparison_uses_exactly_the_two_chosen_sources() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = notebook_with_three_sources(&prompts);
        notebooks.compare(nb, "договор.txt", "памятка.txt").unwrap();
        let prompt = prompts.lock().unwrap()[0].clone();
        assert!(prompt.contains("Сравни источники «договор.txt» и «памятка.txt»"));
        assert!(prompt.contains("два года") && prompt.contains("тридцать дней"));
        assert!(!prompt.contains("молоко"));
    }

    #[test]
    fn comparing_asks_for_two_different_sources_that_exist() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = notebook_with_three_sources(&prompts);
        let same = notebooks
            .compare(nb, "договор.txt", "договор.txt")
            .unwrap_err();
        assert!(matches!(same, crate::error::Error::Usage(_)));
        let missing = notebooks.compare(nb, "договор.txt", "нет.txt").unwrap_err();
        assert!(matches!(missing, crate::error::Error::NotFound(_)));
        assert!(prompts.lock().unwrap().is_empty());
    }

    #[test]
    fn a_deep_question_searches_the_notebook_for_every_planned_term() {
        let llm = Arc::new(PlanThenAnswer {
            calls: Mutex::new(Vec::new()),
        });
        struct Shared(Arc<PlanThenAnswer>);
        impl Llm for Shared {
            fn complete(&self, prompt: &str) -> Result<String> {
                self.0.complete(prompt)
            }
        }
        let notebooks = Notebooks::standard(
            Store::in_memory().unwrap(),
            Arc::new(HashingEmbedder::default()),
            Box::new(Shared(llm.clone())),
        );
        let nb = notebooks.notebook("Диплом").unwrap();
        notebooks
            .add_source(
                nb,
                "договор.txt",
                "Срок действия договора составляет два года.",
            )
            .unwrap();
        notebooks
            .add_source(nb, "оплата.txt", "Оплата производится ежемесячно.")
            .unwrap();
        notebooks
            .add_source(nb, "заметки.txt", "Купить молоко и хлеб.")
            .unwrap();

        let answer = notebooks.ask_deep(nb, "Какой срок и как платить?").unwrap();

        let calls = llm.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert!(calls[1].contains("договор.txt") && calls[1].contains("оплата.txt"));
        assert!(!calls[1].contains("молоко"));
        assert_eq!(answer.citations.len(), 2);
    }
}
