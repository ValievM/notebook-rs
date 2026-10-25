//! Decorators: each wraps one of the ports (the model, the embedder, a retriever), does the same
//! thing as what it wraps, and measures it on the way. The code that uses a port never knows.

use super::Metrics;
use crate::domain::{NotebookId, Passage};
use crate::embed::Embedder;
use crate::error::Result;
use crate::llm::Llm;
use crate::retrieve::Retriever;
use std::sync::Arc;

/// Measures every call to the model: `llm.complete`.
pub struct InstrumentedLlm {
    inner: Box<dyn Llm>,
    metrics: Metrics,
}

impl InstrumentedLlm {
    pub fn new(inner: Box<dyn Llm>, metrics: Metrics) -> Self {
        Self { inner, metrics }
    }
}

impl Llm for InstrumentedLlm {
    fn describe(&self) -> String {
        self.inner.describe()
    }

    fn complete(&self, prompt: &str) -> Result<String> {
        self.metrics
            .span("llm.complete")
            .subject(&self.inner.describe())
            .field("prompt_chars", prompt.chars().count() as i64)
            .run(|span| {
                let reply = self.inner.complete(prompt)?;
                span.set("reply_chars", reply.chars().count() as i64);
                Ok(reply)
            })
    }
}

/// Measures embedding: `embed.passages` (stored text) and `embed.query` (a question).
pub struct InstrumentedEmbedder {
    inner: Arc<dyn Embedder>,
    metrics: Metrics,
}

impl InstrumentedEmbedder {
    pub fn new(inner: Arc<dyn Embedder>, metrics: Metrics) -> Self {
        Self { inner, metrics }
    }
}

impl Embedder for InstrumentedEmbedder {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let chars: usize = texts.iter().map(|t| t.chars().count()).sum();
        self.metrics
            .span("embed.passages")
            .field("texts", texts.len() as i64)
            .field("chars", chars as i64)
            .run(|_| self.inner.embed(texts))
    }

    fn embed_query(&self, question: &str) -> Result<Vec<f32>> {
        self.metrics
            .span("embed.query")
            .field("chars", question.chars().count() as i64)
            .run(|_| self.inner.embed_query(question))
    }
}

/// Measures a retriever under the name it was given (`retrieve.bm25`, `retrieve.vector`...).
pub struct InstrumentedRetriever {
    inner: Box<dyn Retriever>,
    metrics: Metrics,
    kind: String,
}

impl InstrumentedRetriever {
    pub fn new(inner: Box<dyn Retriever>, metrics: Metrics, kind: &str) -> Self {
        Self {
            inner,
            metrics,
            kind: kind.to_string(),
        }
    }
}

impl Retriever for InstrumentedRetriever {
    fn retrieve(&self, notebook: NotebookId, question: &str, limit: usize) -> Result<Vec<Passage>> {
        self.metrics
            .span(&self.kind)
            .field("limit", limit as i64)
            .run(|span| {
                let found = self.inner.retrieve(notebook, question, limit)?;
                span.set("results", found.len() as i64);
                Ok(found)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PassageId;
    use crate::error::Error;
    use crate::metrics::{Event, ManualClock, MemoryRecorder};

    fn fixture() -> (Metrics, Arc<MemoryRecorder>, Arc<ManualClock>) {
        let recorder = Arc::new(MemoryRecorder::default());
        let clock = Arc::new(ManualClock::starting_at(1_700_000_000_000));
        (
            Metrics::with_clock(recorder.clone(), clock.clone()),
            recorder,
            clock,
        )
    }

    fn only(recorder: &MemoryRecorder) -> Event {
        let events = recorder.events();
        assert_eq!(events.len(), 1, "{events:?}");
        events[0].clone()
    }

    fn field(event: &Event, name: &str) -> Option<i64> {
        event
            .fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
    }

    /// Takes 300 ms on the given clock and answers with the prompt reversed.
    struct SlowLlm(Arc<ManualClock>, bool);
    impl Llm for SlowLlm {
        fn complete(&self, prompt: &str) -> Result<String> {
            self.0.advance_micros(300_000);
            if self.1 {
                return Err(Error::ModelUnavailable("codex exited with status 1".into()));
            }
            Ok(prompt.chars().rev().collect())
        }
    }

    #[test]
    fn a_model_call_is_measured_with_the_sizes_of_the_prompt_and_the_reply() {
        let (metrics, recorder, clock) = fixture();
        let llm = InstrumentedLlm::new(Box::new(SlowLlm(clock, false)), metrics);

        let reply = llm.complete("вопрос").unwrap();

        assert_eq!(reply, "сорпов");
        let event = only(&recorder);
        assert_eq!(event.kind, "llm.complete");
        assert_eq!(event.micros, Some(300_000));
        assert_eq!(field(&event, "prompt_chars"), Some(6));
        assert_eq!(field(&event, "reply_chars"), Some(6));
        assert!(event.ok);
    }

    struct Named;
    impl Llm for Named {
        fn complete(&self, _: &str) -> Result<String> {
            Ok("x".into())
        }
        fn describe(&self) -> String {
            "chatgpt:special-model".into()
        }
    }

    #[test]
    fn a_model_call_is_measured_by_the_model_that_answered() {
        let (metrics, recorder, _) = fixture();
        let llm = InstrumentedLlm::new(Box::new(Named), metrics);

        llm.complete("x").unwrap();

        assert_eq!(
            only(&recorder).subject.as_deref(),
            Some("chatgpt:special-model")
        );
        assert_eq!(
            llm.describe(),
            "chatgpt:special-model",
            "the wrapper describes what it wraps"
        );
    }

    #[test]
    fn a_failing_model_is_recorded_and_the_error_reaches_the_caller() {
        let (metrics, recorder, clock) = fixture();
        let llm = InstrumentedLlm::new(Box::new(SlowLlm(clock, true)), metrics);

        let error = llm.complete("x").unwrap_err();

        assert!(matches!(error, Error::ModelUnavailable(_)));
        let event = only(&recorder);
        assert!(!event.ok);
        assert_eq!(event.error.as_deref(), Some("model_unavailable"));
        assert!(event.detail.as_deref().unwrap().contains("codex exited"));
        assert_eq!(field(&event, "reply_chars"), None);
    }

    struct FixedEmbedder;
    impl Embedder for FixedEmbedder {
        fn model(&self) -> &str {
            "fixed"
        }
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![1.0]; texts.len()])
        }
    }

    #[test]
    fn embedding_stored_text_and_a_question_are_measured_apart() {
        let (metrics, recorder, _) = fixture();
        let embedder = InstrumentedEmbedder::new(Arc::new(FixedEmbedder), metrics);

        let vectors = embedder.embed(&["раз".into(), "два три".into()]).unwrap();
        let query = embedder.embed_query("вопрос?").unwrap();

        assert_eq!((vectors.len(), query.len()), (2, 1));
        assert_eq!(embedder.model(), "fixed");
        let events = recorder.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, "embed.passages");
        assert_eq!(field(&events[0], "texts"), Some(2));
        assert_eq!(field(&events[0], "chars"), Some(10));
        assert_eq!(events[1].kind, "embed.query");
        assert_eq!(field(&events[1], "chars"), Some(7));
    }

    struct Found(usize);
    impl Retriever for Found {
        fn retrieve(&self, _: NotebookId, _: &str, _: usize) -> Result<Vec<Passage>> {
            Ok((0..self.0)
                .map(|i| Passage {
                    id: PassageId(i as i64),
                    source_title: "s".into(),
                    ordinal: i,
                    text: "t".into(),
                })
                .collect())
        }
    }

    #[test]
    fn a_search_is_measured_under_its_name_with_the_number_of_results() {
        let (metrics, recorder, _) = fixture();
        let retriever = InstrumentedRetriever::new(Box::new(Found(4)), metrics, "retrieve.bm25");

        let found = retriever.retrieve(NotebookId(1), "вопрос", 6).unwrap();

        assert_eq!(found.len(), 4);
        let event = only(&recorder);
        assert_eq!(event.kind, "retrieve.bm25");
        assert_eq!(field(&event, "limit"), Some(6));
        assert_eq!(field(&event, "results"), Some(4));
    }
}
