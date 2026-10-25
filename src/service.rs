//! The facade the binary (and later the window) talks to. It owns the wiring: how a source is
//! split and embedded on the way in, and how a question is searched and answered on the way out.

use crate::answer::{self, Answer};
use crate::chunk::Chunker;
use crate::chunk_markdown::MarkdownChunker;
use crate::deep;
use crate::domain::NotebookId;
use crate::embed::Embedder;
use crate::error::Result;
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
        }
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
