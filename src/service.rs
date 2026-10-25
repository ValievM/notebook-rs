//! The facade the binary (and later the window) talks to. It owns the wiring: how a source is
//! split and embedded on the way in, and how a question is searched and answered on the way out.

use crate::answer::{self, Answer};
use crate::chunk::Chunker;
use crate::chunk_markdown::MarkdownChunker;
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
}
