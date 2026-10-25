//! The facade the binary (and later the window) talks to. It owns the wiring: how a source is
//! split and embedded on the way in, and how a question is searched and answered on the way out.

use crate::answer::{self, Answer, Citation, Turn};
use crate::cards::{self, Deck};
use crate::chunk::Chunker;
use crate::chunk_markdown::MarkdownChunker;
use crate::deep;
use crate::diagram::{self, Diagram};
use crate::digest::{self, Budget};
use crate::domain::{NotebookId, Passage};
use crate::embed::Embedder;
use crate::error::{Error, Result};
use crate::llm::Llm;
use crate::metrics::{InstrumentedEmbedder, InstrumentedLlm, InstrumentedRetriever, Metrics};
use crate::mindmap::{self, MindMap};
use crate::retrieve::{Bm25Retriever, HybridRetriever, Retriever, VectorRetriever};
use crate::source::{Sources, folder};
use crate::store::{NewPassage, NotebookInfo, SourceInfo, Store};
use crate::view::{PassageView, source_view};
use std::sync::Arc;

const PASSAGES_PER_QUESTION: usize = 6;
const PASSAGES_PER_DIAGRAM: usize = 12;

/// How many earlier messages are looked at to build the dialog of a question, and how many are
/// shown in the window.
const HISTORY_LOOKBACK: usize = 12;
const HISTORY_SHOWN: usize = 200;
/// A follow-up this short ("and in more detail?") says nothing the search could use alone.
const SHORT_FOLLOW_UP_CHARS: usize = 60;

/// One message of the dialog of a notebook, as the window shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryEntry {
    pub id: i64,
    /// `true` for the person, `false` for the assistant.
    pub user: bool,
    /// `ask`, `summarize` or `compare`.
    pub kind: String,
    pub text: String,
    pub citations: Vec<Citation>,
    pub rejected_quotes: usize,
    /// The map, for an answer of kind `mindmap`.
    pub mind_map: Option<MindMap>,
    /// The diagram, for an answer of kind `diagram`.
    pub diagram: Option<Diagram>,
    /// The cards, for an answer of kind `cards`.
    pub deck: Option<Deck>,
}

/// What adding a location (a file, a page or a folder) did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct AddReport {
    /// Titles of the sources added, with their number of passages.
    pub added: Vec<(String, usize)>,
    /// Files of a folder that could not be added, with the reason. One bad file does not stop
    /// the others.
    pub failed: Vec<(String, String)>,
}

/// What `reindex` did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ReindexReport {
    /// Sources cut again from their kept text.
    pub recut: usize,
    /// Passages of the other sources that got vectors of the current model.
    pub reembedded_passages: usize,
    /// Sources that could not be cut again (added before texts were kept).
    pub without_text: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct AnswerPayload {
    citations: Vec<Citation>,
    rejected_quotes: usize,
    #[serde(default)]
    mind_map: Option<MindMap>,
    #[serde(default)]
    diagram: Option<Diagram>,
    #[serde(default)]
    deck: Option<Deck>,
}

/// How many passages go to the embedding model in one call. A bound on what is held and
/// computed at once, so that a long source is indexed in steps. (Measured: the memory of the
/// model is mostly decided by the model itself, not by this number; see DECISIONS D-14.)
pub const EMBED_BATCH: usize = 16;

pub struct Notebooks {
    store: Arc<Store>,
    chunker: Box<dyn Chunker>,
    embedder: Arc<dyn Embedder>,
    retriever: Box<dyn Retriever>,
    llm: Box<dyn Llm>,
    budget: Budget,
    metrics: Metrics,
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
            metrics: Metrics::off(),
        }
    }

    /// Measures the operations of the facade itself (`ask`, `source.add`...). To measure the
    /// model, the embedder and the searches as well, build with `standard_with_metrics`.
    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Changes how much text one model call may get in `summarize` and `compare`.
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// The usual wiring: Markdown-aware chunks (plain text works too), BM25 and vectors fused.
    pub fn standard(store: Store, embedder: Arc<dyn Embedder>, llm: Box<dyn Llm>) -> Self {
        Self::standard_with_metrics(store, embedder, llm, Metrics::off())
    }

    /// The usual wiring with everything measured: the facade, the model, the embedder, and
    /// each of the searches (see `metrics`).
    pub fn standard_with_metrics(
        store: Store,
        embedder: Arc<dyn Embedder>,
        llm: Box<dyn Llm>,
        metrics: Metrics,
    ) -> Self {
        let store = Arc::new(store);
        let embedder: Arc<dyn Embedder> =
            Arc::new(InstrumentedEmbedder::new(embedder, metrics.clone()));
        let llm = Box::new(InstrumentedLlm::new(llm, metrics.clone()));
        let measured = |retriever: Box<dyn Retriever>, kind: &str| -> Box<dyn Retriever> {
            Box::new(InstrumentedRetriever::new(retriever, metrics.clone(), kind))
        };
        let hybrid = HybridRetriever::new(vec![
            measured(Box::new(Bm25Retriever::new(store.clone())), "retrieve.bm25"),
            measured(
                Box::new(VectorRetriever::new(store.clone(), embedder.clone())),
                "retrieve.vector",
            ),
        ]);
        Self::new(
            store,
            Box::new(MarkdownChunker::default()),
            embedder,
            measured(Box::new(hybrid), "retrieve.hybrid"),
            llm,
        )
        .with_metrics(metrics)
    }

    pub fn notebook(&self, name: &str) -> Result<NotebookId> {
        self.store.create_notebook(name)
    }

    /// Splits, embeds and stores a source; returns how many passages it became.
    pub fn add_source(&self, notebook: NotebookId, title: &str, text: &str) -> Result<usize> {
        self.metrics
            .span("source.add")
            .subject(title)
            .field("chars", text.chars().count() as i64)
            .run(|span| {
                let chunks = self
                    .metrics
                    .span("source.chunk")
                    .subject(title)
                    .run(|step| {
                        let chunks = self.chunker.split(text);
                        step.set("passages", chunks.len() as i64);
                        Ok(chunks)
                    })?;
                let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
                let mut vectors = Vec::with_capacity(texts.len());
                for batch in texts.chunks(EMBED_BATCH) {
                    vectors.extend(self.embedder.embed(batch)?);
                }
                let passages: Vec<NewPassage> = chunks
                    .into_iter()
                    .zip(vectors)
                    .map(|(chunk, vector)| NewPassage {
                        ordinal: chunk.ordinal,
                        text: chunk.text,
                        embedding: Some(vector),
                    })
                    .collect();
                let count = self
                    .metrics
                    .span("store.write")
                    .subject(title)
                    .field("passages", passages.len() as i64)
                    .run(|_| {
                        self.store.add_source_with_text(
                            notebook,
                            title,
                            &passages,
                            self.embedder.model(),
                            text,
                        )
                    })?;
                span.set("passages", count as i64);
                self.metrics.sample_memory("after_add");
                Ok(count)
            })
    }

    /// Adds what the location points to: a file or a web page becomes one source (an error if it
    /// cannot be read); a folder becomes one source per supported file in it and in its
    /// subfolders, and the files that fail are listed in the report.
    pub fn add_location(
        &self,
        notebook: NotebookId,
        location: &str,
        sources: &Sources,
    ) -> Result<AddReport> {
        let mut report = AddReport::default();
        let path = std::path::Path::new(location);
        if !path.is_dir() {
            let document = sources.read(location)?;
            let count = self.add_source(notebook, &document.title, &document.text)?;
            report.added.push((document.title, count));
            return Ok(report);
        }
        for (title, file) in folder::files_in(path)? {
            let added = sources
                .read(&file.to_string_lossy())
                .and_then(|document| self.add_source(notebook, &title, &document.text));
            match added {
                Ok(count) => report.added.push((title, count)),
                Err(error) => report.failed.push((title, error.to_string())),
            }
        }
        Ok(report)
    }

    /// Cuts every source of the notebook again from its kept text and embeds it again (after the
    /// way of cutting or the embedding model changed). A source added before texts were kept
    /// cannot be cut again: its passages only get vectors of the current model.
    pub fn reindex(&self, notebook: NotebookId) -> Result<ReindexReport> {
        self.metrics.span("reindex").run(|span| {
            let mut report = ReindexReport::default();
            for (title, text) in self.store.source_texts(notebook)? {
                match text {
                    Some(text) => {
                        self.add_source(notebook, &title, &text)?;
                        report.recut += 1;
                    }
                    None => report.without_text.push(title),
                }
            }
            let stale = self
                .store
                .passages_without_vectors_of(notebook, self.embedder.model())?;
            for batch in stale.chunks(EMBED_BATCH) {
                let texts: Vec<String> = batch.iter().map(|p| p.text.clone()).collect();
                let vectors = self.embedder.embed(&texts)?;
                let given: Vec<_> = batch.iter().map(|p| p.id).zip(vectors).collect();
                self.store.set_vectors(&given, self.embedder.model())?;
                report.reembedded_passages += batch.len();
            }
            span.set("recut", report.recut as i64);
            span.set("reembedded", report.reembedded_passages as i64);
            Ok(report)
        })
    }

    pub fn ask(&self, notebook: NotebookId, question: &str) -> Result<Answer> {
        self.metrics
            .span("ask")
            .field("question_chars", question.chars().count() as i64)
            .run(|_| {
                let turns = self.earlier_turns(notebook)?;
                let passages = self.retriever.retrieve(
                    notebook,
                    &search_text(question, &turns),
                    PASSAGES_PER_QUESTION,
                )?;
                let answer = answer::ask_chat(self.llm.as_ref(), question, &[], &turns, &passages)?;
                self.remember(notebook, "ask", question, &answer)?;
                Ok(answer)
            })
    }

    /// The earlier questions and answers of the dialog, for the next question. Summaries and
    /// comparisons are in the dialog for the person but are not a conversation to continue.
    fn earlier_turns(&self, notebook: NotebookId) -> Result<Vec<Turn>> {
        Ok(self
            .store
            .messages(notebook, HISTORY_LOOKBACK)?
            .into_iter()
            .filter(|m| m.kind == "ask")
            .map(|m| Turn {
                user: m.role == "user",
                text: m.text,
            })
            .collect())
    }

    /// Keeps a question and its answer in the dialog of the notebook.
    fn remember(
        &self,
        notebook: NotebookId,
        kind: &str,
        question: &str,
        answer: &Answer,
    ) -> Result<()> {
        let payload = AnswerPayload {
            citations: answer.citations.clone(),
            rejected_quotes: answer.rejected_quotes,
            mind_map: None,
            diagram: None,
            deck: None,
        };
        self.keep(notebook, kind, question, &answer.text, payload)
    }

    fn keep(
        &self,
        notebook: NotebookId,
        kind: &str,
        question: &str,
        reply: &str,
        payload: AnswerPayload,
    ) -> Result<()> {
        let payload = serde_json::to_string(&payload).unwrap_or_default();
        self.store
            .add_message(notebook, "user", kind, question, "")?;
        self.store
            .add_message(notebook, "assistant", kind, reply, &payload)?;
        Ok(())
    }

    /// A diagram of a topic (boxes and arrows) from the passages found for it, kept in the
    /// dialog.
    pub fn diagram(&self, notebook: NotebookId, topic: &str) -> Result<Diagram> {
        self.metrics.span("diagram").run(|span| {
            let passages = self
                .retriever
                .retrieve(notebook, topic, PASSAGES_PER_DIAGRAM)?;
            let drawn = diagram::diagram(self.llm.as_ref(), &passages, topic)?;
            span.set("nodes", drawn.nodes.len() as i64);
            let payload = AnswerPayload {
                citations: Vec::new(),
                rejected_quotes: drawn.rejected_quotes,
                mind_map: None,
                diagram: Some(drawn.clone()),
                deck: None,
            };
            self.keep(
                notebook,
                "diagram",
                &format!("Схема: {topic}"),
                &drawn.title,
                payload,
            )?;
            Ok(drawn)
        })
    }

    /// The name of what a picture or a deck is made of: the source, or the notebook.
    fn material_title(&self, notebook: NotebookId, source: Option<&str>) -> Result<String> {
        Ok(match source {
            Some(title) => title.to_string(),
            None => self
                .store
                .list_notebooks()?
                .into_iter()
                .find(|info| info.id == notebook)
                .map_or_else(|| "Блокнот".to_string(), |info| info.name),
        })
    }

    /// Flashcards of one source (or of the whole notebook), kept in the dialog.
    pub fn flashcards(&self, notebook: NotebookId, source: Option<&str>) -> Result<Deck> {
        self.metrics
            .span("cards")
            .subject(source.unwrap_or("*"))
            .run(|span| {
                let passages = self.passages_of(notebook, source)?;
                let title = self.material_title(notebook, source)?;
                let deck = cards::flashcards(self.llm.as_ref(), &passages, &title, &self.budget)?;
                span.set("cards", deck.cards.len() as i64);
                let asked = match source {
                    Some(title) => format!("Карточки «{title}»"),
                    None => "Карточки блокнота".to_string(),
                };
                let payload = AnswerPayload {
                    citations: Vec::new(),
                    rejected_quotes: deck.dropped,
                    mind_map: None,
                    diagram: None,
                    deck: Some(deck.clone()),
                };
                self.keep(notebook, "cards", &asked, &deck.title, payload)?;
                Ok(deck)
            })
    }

    /// A mind map of one source (or of the whole notebook), kept in the dialog.
    pub fn mind_map(&self, notebook: NotebookId, source: Option<&str>) -> Result<MindMap> {
        self.metrics
            .span("mindmap")
            .subject(source.unwrap_or("*"))
            .run(|span| {
                let passages = self.passages_of(notebook, source)?;
                let title = self.material_title(notebook, source)?;
                let map = mindmap::mind_map(self.llm.as_ref(), &passages, &title, &self.budget)?;
                span.set("nodes", map.nodes as i64);
                let asked = match source {
                    Some(title) => format!("Ментальная карта «{title}»"),
                    None => "Ментальная карта блокнота".to_string(),
                };
                let payload = AnswerPayload {
                    citations: Vec::new(),
                    rejected_quotes: map.rejected_quotes,
                    mind_map: Some(map.clone()),
                    diagram: None,
                    deck: None,
                };
                self.keep(notebook, "mindmap", &asked, &map.root.title, payload)?;
                Ok(map)
            })
    }

    /// The dialog of the notebook, oldest first.
    pub fn history(&self, notebook: NotebookId) -> Result<Vec<HistoryEntry>> {
        Ok(self
            .store
            .messages(notebook, HISTORY_SHOWN)?
            .into_iter()
            .map(|m| {
                let payload: AnswerPayload = serde_json::from_str(&m.payload).unwrap_or_default();
                HistoryEntry {
                    id: m.id,
                    user: m.role == "user",
                    kind: m.kind,
                    text: m.text,
                    citations: payload.citations,
                    rejected_quotes: payload.rejected_quotes,
                    mind_map: payload.mind_map,
                    diagram: payload.diagram,
                    deck: payload.deck,
                }
            })
            .collect())
    }

    /// Forgets the dialog of the notebook; the sources stay.
    pub fn clear_history(&self, notebook: NotebookId) -> Result<()> {
        self.store.clear_messages(notebook)
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

    /// Every notebook with its number of sources.
    pub fn list_notebooks(&self) -> Result<Vec<NotebookInfo>> {
        self.store.list_notebooks()
    }

    /// The sources of the notebook with their number of passages.
    pub fn sources(&self, notebook: NotebookId) -> Result<Vec<SourceInfo>> {
        self.store.source_infos(notebook)
    }

    /// Removes a source; an unknown title is an error that lists the titles there are.
    pub fn delete_source(&self, notebook: NotebookId, title: &str) -> Result<()> {
        self.passages_of(notebook, Some(title))?;
        self.store.delete_source(notebook, title)?;
        Ok(())
    }

    /// A source laid out for reading. `cited` (position of the cited passage and the quote)
    /// marks the cited place.
    pub fn source_view(
        &self,
        notebook: NotebookId,
        title: &str,
        cited: Option<(usize, &str)>,
    ) -> Result<Vec<PassageView>> {
        let passages = self.passages_of(notebook, Some(title))?;
        Ok(source_view(&passages, cited))
    }

    /// A summary of one source, or of the whole notebook when `source` is `None`.
    pub fn summarize(&self, notebook: NotebookId, source: Option<&str>) -> Result<Answer> {
        self.metrics
            .span("summarize")
            .subject(source.unwrap_or("*"))
            .run(|_| {
                let passages = self.passages_of(notebook, source)?;
                let answer = digest::summarize(self.llm.as_ref(), &passages, &self.budget)?;
                let asked = match source {
                    Some(title) => format!("Конспект «{title}»"),
                    None => "Конспект блокнота".to_string(),
                };
                self.remember(notebook, "summarize", &asked, &answer)?;
                Ok(answer)
            })
    }

    /// A study plan of one source (or of the whole notebook): steps in the order to study them.
    pub fn study_plan(&self, notebook: NotebookId, source: Option<&str>) -> Result<Answer> {
        self.metrics
            .span("plan")
            .subject(source.unwrap_or("*"))
            .run(|_| {
                let passages = self.passages_of(notebook, source)?;
                let answer = digest::study_plan(self.llm.as_ref(), &passages, &self.budget)?;
                let asked = match source {
                    Some(title) => format!("План изучения «{title}»"),
                    None => "План изучения блокнота".to_string(),
                };
                self.remember(notebook, "plan", &asked, &answer)?;
                Ok(answer)
            })
    }

    /// What two sources of the notebook have in common, how they differ, where they contradict.
    pub fn compare(&self, notebook: NotebookId, first: &str, second: &str) -> Result<Answer> {
        self.metrics
            .span("compare")
            .subject(&format!("{first} + {second}"))
            .run(|_| {
                if first == second {
                    return Err(Error::Usage(
                        "choose two different sources to compare".into(),
                    ));
                }
                let first = self.passages_of(notebook, Some(first))?;
                let second = self.passages_of(notebook, Some(second))?;
                let answer = digest::compare(self.llm.as_ref(), &first, &second, &self.budget)?;
                let asked = format!(
                    "Сравнить «{}» и «{}»",
                    first.first().map_or("", |p| p.source_title.as_str()),
                    second.first().map_or("", |p| p.source_title.as_str())
                );
                self.remember(notebook, "compare", &asked, &answer)?;
                Ok(answer)
            })
    }

    /// For questions that need several searches (comparisons, "and also..."): the model plans the
    /// searches first, so this takes two model calls instead of one.
    pub fn ask_deep(&self, notebook: NotebookId, question: &str) -> Result<Answer> {
        self.metrics
            .span("ask_deep")
            .field("question_chars", question.chars().count() as i64)
            .run(|_| {
                let answer = deep::research(
                    self.llm.as_ref(),
                    self.retriever.as_ref(),
                    notebook,
                    question,
                )?;
                self.remember(notebook, "ask", question, &answer)?;
                Ok(answer)
            })
    }
}

/// What to search for: a short follow-up is searched together with the previous question, so
/// that "and in more detail?" still finds the passages of the topic.
fn search_text(question: &str, turns: &[Turn]) -> String {
    let previous = turns.iter().rev().find(|turn| turn.user);
    match previous {
        Some(turn) if question.chars().count() <= SHORT_FOLLOW_UP_CHARS => {
            format!("{} {question}", turn.text)
        }
        _ => question.to_string(),
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

    struct NoModel;
    impl Llm for NoModel {
        fn complete(&self, _: &str) -> Result<String> {
            Err(Error::ModelUnavailable("not needed".into()))
        }
    }

    fn with_chunker(store: &Arc<Store>, target: usize) -> Notebooks {
        Notebooks::new(
            store.clone(),
            Box::new(crate::chunk::ParagraphChunker { target }),
            Arc::new(HashingEmbedder::default()),
            Box::new(Bm25Retriever::new(store.clone())),
            Box::new(NoModel),
        )
    }

    #[test]
    fn reindexing_cuts_the_kept_texts_again_and_embeds_the_sources_that_have_none() {
        let store = Arc::new(Store::in_memory().unwrap());
        let nb = store.create_notebook("n").unwrap();
        let text = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац.";
        with_chunker(&store, 15)
            .add_source(nb, "a.txt", text)
            .unwrap();
        assert_eq!(store.source_passages(nb, Some("a.txt")).unwrap().len(), 3);
        // A source from before texts were kept, with a vector of another model.
        store
            .add_source(
                nb,
                "old.txt",
                &[NewPassage {
                    ordinal: 0,
                    text: "Старый текст.".into(),
                    embedding: Some(vec![1.0]),
                }],
                "other-model",
            )
            .unwrap();

        let report = with_chunker(&store, 900).reindex(nb).unwrap();

        assert_eq!(store.source_passages(nb, Some("a.txt")).unwrap().len(), 1);
        assert_eq!(report.recut, 1);
        assert_eq!(report.reembedded_passages, 1);
        assert_eq!(report.without_text, ["old.txt"]);
        assert!(
            store
                .passages_without_vectors_of(nb, "hashing-512")
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_folder_becomes_one_source_per_file_and_a_bad_file_does_not_stop_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.txt"), "Первый файл.").unwrap();
        std::fs::write(dir.path().join("sub/b.md"), "# Второй\n\nФайл.").unwrap();
        std::fs::write(dir.path().join("bad.pdf"), "это не PDF").unwrap();
        std::fs::write(dir.path().join("pic.png"), "не источник").unwrap();
        let store = Arc::new(Store::in_memory().unwrap());
        let nb = store.create_notebook("n").unwrap();

        let report = with_chunker(&store, 900)
            .add_location(nb, dir.path().to_str().unwrap(), &Sources::standard())
            .unwrap();

        let added: Vec<&str> = report.added.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(added, ["a.txt", "sub/b.md"]);
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].0, "bad.pdf");
        assert_eq!(store.source_titles(nb).unwrap(), ["a.txt", "sub/b.md"]);
    }

    #[test]
    fn a_single_file_is_added_as_before_and_its_error_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("one.txt");
        std::fs::write(&file, "Один файл.").unwrap();
        let store = Arc::new(Store::in_memory().unwrap());
        let nb = store.create_notebook("n").unwrap();
        let notebooks = with_chunker(&store, 900);

        let report = notebooks
            .add_location(nb, file.to_str().unwrap(), &Sources::standard())
            .unwrap();
        assert_eq!(report.added, [("one.txt".to_string(), 1)]);

        let missing = dir.path().join("missing.txt");
        assert!(
            notebooks
                .add_location(nb, missing.to_str().unwrap(), &Sources::standard())
                .is_err()
        );
    }

    #[test]
    fn reindexing_an_empty_notebook_does_nothing() {
        let store = Arc::new(Store::in_memory().unwrap());
        let nb = store.create_notebook("n").unwrap();
        let report = with_chunker(&store, 900).reindex(nb).unwrap();
        assert_eq!((report.recut, report.reembedded_passages), (0, 0));
        assert!(report.without_text.is_empty());
    }

    fn chat_notebooks(prompts: &Arc<Mutex<Vec<String>>>) -> (Notebooks, NotebookId) {
        let llm = Recording {
            reply: r#"{"answer":"Два года [1].","citations":[{"n":1,"quote":"два года"}]}"#,
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
        (notebooks, nb)
    }

    #[test]
    fn a_follow_up_sees_the_earlier_turns_and_the_dialog_is_kept_with_its_citations() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = chat_notebooks(&prompts);

        notebooks.ask(nb, "Какой срок у договора?").unwrap();
        notebooks.ask(nb, "А подробнее?").unwrap();

        let second = prompts.lock().unwrap()[1].clone();
        assert!(
            second.contains("Пользователь: Какой срок у договора?"),
            "{second}"
        );
        assert!(second.contains("Ассистент: Два года [1]."), "{second}");
        let first = prompts.lock().unwrap()[0].clone();
        assert!(!first.contains("История диалога"));

        let history = notebooks.history(nb).unwrap();
        let shape: Vec<(bool, &str)> = history.iter().map(|m| (m.user, m.text.as_str())).collect();
        assert_eq!(
            shape,
            [
                (true, "Какой срок у договора?"),
                (false, "Два года [1]."),
                (true, "А подробнее?"),
                (false, "Два года [1].")
            ]
        );
        assert_eq!(history[1].citations.len(), 1);
        assert_eq!(history[1].citations[0].source_title, "договор.txt");
    }

    #[test]
    fn a_short_follow_up_is_searched_together_with_the_previous_question() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = chat_notebooks(&prompts);
        notebooks.ask(nb, "Какой срок у договора?").unwrap();

        notebooks.ask(nb, "А подробнее?").unwrap();

        let second = prompts.lock().unwrap()[1].clone();
        assert!(second.contains("[1] Источник «договор.txt»"), "{second}");
    }

    #[test]
    fn clearing_the_dialog_makes_the_next_question_start_fresh() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = chat_notebooks(&prompts);
        notebooks.ask(nb, "Какой срок у договора?").unwrap();

        notebooks.clear_history(nb).unwrap();
        notebooks.ask(nb, "Какой срок у договора?").unwrap();

        assert!(notebooks.history(nb).unwrap().len() == 2);
        assert!(!prompts.lock().unwrap()[1].contains("История диалога"));
    }

    #[test]
    fn a_summary_and_a_comparison_are_kept_in_the_dialog_but_do_not_enter_later_questions() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = chat_notebooks(&prompts);

        notebooks.summarize(nb, None).unwrap();
        notebooks.compare(nb, "договор.txt", "заметки.txt").unwrap();
        notebooks.ask(nb, "Какой срок у договора?").unwrap();

        let kinds: Vec<String> = notebooks
            .history(nb)
            .unwrap()
            .into_iter()
            .map(|m| m.kind)
            .collect();
        assert_eq!(
            kinds,
            ["summarize", "summarize", "compare", "compare", "ask", "ask"]
        );
        let last = prompts.lock().unwrap().last().unwrap().clone();
        assert!(!last.contains("История диалога"), "{last}");
    }

    #[test]
    fn a_mind_map_of_the_notebook_is_built_and_kept_in_the_dialog_but_not_in_later_questions() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let llm = Recording {
            reply: r#"{"title":"","children":[{"title":"Срок","n":1,"quote":"два года"}]}"#,
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

        let map = notebooks.mind_map(nb, None).unwrap();

        assert_eq!(
            map.root.title, "Диплом",
            "the root is named after the notebook"
        );
        assert_eq!(
            map.root.children[0].cite.as_ref().unwrap().source_title,
            "договор.txt"
        );
        let history = notebooks.history(nb).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[1].kind, "mindmap");
        assert_eq!(history[1].mind_map.as_ref().unwrap().nodes, 2);
        assert_eq!(history[0].text, "Ментальная карта блокнота");

        notebooks.ask(nb, "Какой срок?").ok();
        let last = prompts.lock().unwrap().last().unwrap().clone();
        assert!(!last.contains("История диалога"), "{last}");
    }

    #[test]
    fn a_diagram_of_a_topic_is_drawn_from_found_passages_and_kept_in_the_dialog() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let llm = Recording {
            reply: r#"{"title":"Срок","nodes":[{"id":"a","label":"Подписание"},{"id":"b","label":"Два года","n":1,"quote":"два года"}],"edges":[{"from":"a","to":"b"}]}"#,
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

        let drawn = notebooks.diagram(nb, "срок договора").unwrap();

        assert_eq!(drawn.nodes.len(), 2);
        assert_eq!(
            drawn.nodes[1].cite.as_ref().unwrap().source_title,
            "договор.txt"
        );
        assert!(prompts.lock().unwrap()[0].contains("Тема схемы: срок договора"));
        let history = notebooks.history(nb).unwrap();
        assert_eq!(history[0].text, "Схема: срок договора");
        assert_eq!(history[1].kind, "diagram");
        assert_eq!(history[1].diagram.as_ref().unwrap().edges.len(), 1);

        let empty = notebooks.notebook("Пустой").unwrap();
        assert!(notebooks.diagram(empty, "что угодно").is_err());
    }

    #[test]
    fn a_study_plan_is_kept_in_the_dialog_as_a_plan_and_not_as_a_conversation() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = chat_notebooks(&prompts);

        let plan = notebooks.study_plan(nb, None).unwrap();

        assert!(!plan.text.is_empty());
        assert!(prompts.lock().unwrap()[0].contains("Составь план изучения"));
        let history = notebooks.history(nb).unwrap();
        assert_eq!(history[0].text, "План изучения блокнота");
        assert_eq!(history[1].kind, "plan");
        notebooks.ask(nb, "Какой срок?").unwrap();
        assert!(
            !prompts
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .contains("История диалога")
        );
    }

    #[test]
    fn flashcards_are_made_from_the_notebook_and_kept_in_the_dialog() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let llm = Recording {
            reply: r#"{"cards":[{"question":"Какой срок?","answer":"Два года","n":1,"quote":"два года"},{"question":"Выдумка?","answer":"да","n":1,"quote":"нет такого"}]}"#,
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

        let deck = notebooks.flashcards(nb, None).unwrap();

        assert_eq!(deck.title, "Диплом");
        assert_eq!(deck.cards.len(), 1);
        assert_eq!(deck.dropped, 1);
        let history = notebooks.history(nb).unwrap();
        assert_eq!(history[0].text, "Карточки блокнота");
        assert_eq!(history[1].kind, "cards");
        assert_eq!(
            history[1].deck.as_ref().unwrap().cards[0].answer,
            "Два года"
        );
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

    /// Remembers how many texts it got in each call.
    struct BatchSizes(Mutex<Vec<usize>>);
    impl Embedder for BatchSizes {
        fn model(&self) -> &str {
            "batch-sizes"
        }
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.0.lock().unwrap().push(texts.len());
            Ok(vec![vec![1.0]; texts.len()])
        }
    }

    use crate::metrics::{Event, ManualClock, MemoryRecorder};

    struct Failing;
    impl Llm for Failing {
        fn complete(&self, _prompt: &str) -> Result<String> {
            Err(Error::ModelUnavailable("codex exited with status 1".into()))
        }
    }

    fn measured(llm: Box<dyn Llm>) -> (Notebooks, Arc<MemoryRecorder>, NotebookId) {
        let recorder = Arc::new(MemoryRecorder::default());
        let metrics = Metrics::with_clock(recorder.clone(), Arc::new(ManualClock::starting_at(1)))
            .with_memory_probe(|| Some(777));
        let notebooks = Notebooks::standard_with_metrics(
            Store::in_memory().unwrap(),
            Arc::new(HashingEmbedder::default()),
            llm,
            metrics,
        );
        let nb = notebooks.notebook("n").unwrap();
        notebooks
            .add_source(
                nb,
                "договор.txt",
                "Срок действия договора составляет два года.",
            )
            .unwrap();
        notebooks
            .add_source(nb, "памятка.txt", "О выезде предупредить за тридцать дней.")
            .unwrap();
        recorder.events(); // the set-up is not what the test looks at
        (notebooks, recorder, nb)
    }

    fn reply() -> Box<dyn Llm> {
        Box::new(Recording {
            reply: r#"{"answer":"ок"}"#,
            prompts: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn kinds(events: &[Event]) -> Vec<String> {
        let mut kinds: Vec<String> = events.iter().map(|e| e.kind.clone()).collect();
        kinds.sort();
        kinds.dedup();
        kinds
    }

    fn events_after(recorder: &MemoryRecorder, skip: usize) -> Vec<Event> {
        recorder.events().into_iter().skip(skip).collect()
    }

    #[test]
    fn a_question_is_measured_end_to_end_and_at_every_step() {
        let (notebooks, recorder, nb) = measured(reply());
        let before = recorder.events().len();

        notebooks.ask(nb, "Какой срок договора?").unwrap();

        let events = events_after(&recorder, before);
        assert_eq!(
            kinds(&events),
            [
                "ask",
                "embed.query",
                "llm.complete",
                "retrieve.bm25",
                "retrieve.hybrid",
                "retrieve.vector"
            ]
        );
        let ask = events.iter().find(|e| e.kind == "ask").unwrap();
        assert!(ask.fields.contains(&("question_chars".to_string(), 20)));
    }

    #[test]
    fn adding_a_source_is_measured_by_its_name_with_its_steps_and_the_memory_after() {
        let (notebooks, recorder, nb) = measured(reply());
        let before = recorder.events().len();

        notebooks
            .add_source(nb, "новый.md", "# Раздел\n\nТекст источника.")
            .unwrap();

        let events = events_after(&recorder, before);
        assert_eq!(
            kinds(&events),
            [
                "embed.passages",
                "memory.after_add",
                "source.add",
                "source.chunk",
                "store.write"
            ]
        );
        let add = events.iter().find(|e| e.kind == "source.add").unwrap();
        assert_eq!(add.subject.as_deref(), Some("новый.md"));
        assert!(add.fields.contains(&("passages".to_string(), 1)));
        // "# Раздел" (8) + two line breaks (2) + "Текст источника." (16)
        assert!(add.fields.contains(&("chars".to_string(), 26)));
        let memory = events
            .iter()
            .find(|e| e.kind == "memory.after_add")
            .unwrap();
        assert_eq!(memory.fields, vec![("rss_bytes".to_string(), 777)]);
    }

    #[test]
    fn a_failed_question_is_recorded_with_the_error_at_each_level() {
        let (notebooks, recorder, nb) = measured(Box::new(Failing));
        let before = recorder.events().len();

        assert!(notebooks.ask(nb, "вопрос про срок").is_err());

        let events = events_after(&recorder, before);
        for kind in ["ask", "llm.complete"] {
            let event = events.iter().find(|e| e.kind == kind).unwrap();
            assert!(!event.ok, "{kind}");
            assert_eq!(event.error.as_deref(), Some("model_unavailable"));
        }
        let ask = events.iter().find(|e| e.kind == "ask").unwrap();
        assert!(ask.detail.as_deref().unwrap().contains("codex exited"));
    }

    #[test]
    fn a_missing_source_is_recorded_with_its_name() {
        let (notebooks, recorder, nb) = measured(reply());
        let before = recorder.events().len();

        assert!(notebooks.summarize(nb, Some("нет.txt")).is_err());

        let events = events_after(&recorder, before);
        let event = events.iter().find(|e| e.kind == "summarize").unwrap();
        assert!(!event.ok);
        assert_eq!(event.subject.as_deref(), Some("нет.txt"));
        assert_eq!(event.error.as_deref(), Some("not_found"));
    }

    #[test]
    fn a_summary_of_the_whole_notebook_and_a_comparison_are_measured_by_what_they_were_about() {
        let (notebooks, recorder, nb) = measured(reply());
        let before = recorder.events().len();

        notebooks.summarize(nb, None).unwrap();
        notebooks.compare(nb, "договор.txt", "памятка.txt").unwrap();

        let events = events_after(&recorder, before);
        let summarize = events.iter().find(|e| e.kind == "summarize").unwrap();
        assert_eq!(summarize.subject.as_deref(), Some("*"));
        let compare = events.iter().find(|e| e.kind == "compare").unwrap();
        assert_eq!(
            compare.subject.as_deref(),
            Some("договор.txt + памятка.txt")
        );
    }

    #[test]
    fn a_long_source_is_embedded_in_small_batches_and_nothing_is_lost() {
        let embedder = Arc::new(BatchSizes(Mutex::new(Vec::new())));
        let notebooks = Notebooks::standard(
            Store::in_memory().unwrap(),
            embedder.clone(),
            Box::new(Recording {
                reply: "{}",
                prompts: Arc::new(Mutex::new(Vec::new())),
            }),
        );
        let nb = notebooks.notebook("n").unwrap();
        // 70 paragraphs of 600 characters: two never fit one passage, so 70 passages.
        let text: Vec<String> = (0..70)
            .map(|i| format!("{i} {}", "слово ".repeat(100)))
            .collect();

        let count = notebooks
            .add_source(nb, "книга.txt", &text.join("\n\n"))
            .unwrap();

        assert_eq!(count, 70);
        let sizes = embedder.0.lock().unwrap().clone();
        assert!(sizes.iter().all(|size| *size <= EMBED_BATCH), "{sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 70);
        assert_eq!(notebooks.sources(nb).unwrap()[0].passages, 70);
    }

    #[test]
    fn the_notebook_lists_its_sources_and_a_deleted_one_is_gone() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = notebook_with_three_sources(&prompts);
        let titles: Vec<String> = notebooks
            .sources(nb)
            .unwrap()
            .into_iter()
            .map(|s| s.title)
            .collect();
        assert_eq!(titles, vec!["договор.txt", "памятка.txt", "заметки.txt"]);
        assert_eq!(notebooks.list_notebooks().unwrap()[0].sources, 3);

        notebooks.delete_source(nb, "памятка.txt").unwrap();

        let titles: Vec<String> = notebooks
            .sources(nb)
            .unwrap()
            .into_iter()
            .map(|s| s.title)
            .collect();
        assert_eq!(titles, vec!["договор.txt", "заметки.txt"]);
        let error = notebooks.delete_source(nb, "памятка.txt").unwrap_err();
        assert!(matches!(error, crate::error::Error::NotFound(_)));
    }

    #[test]
    fn a_citation_opens_its_source_at_the_quoted_place() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let (notebooks, nb) = notebook_with_three_sources(&prompts);
        let view = notebooks
            .source_view(nb, "договор.txt", Some((0, "ДВА года")))
            .unwrap();
        assert_eq!(view[0].hit.as_deref(), Some("два года"));
        assert_eq!(view[0].before, "Срок действия договора ");
        let missing = notebooks.source_view(nb, "нет.txt", None).unwrap_err();
        assert!(matches!(missing, crate::error::Error::NotFound(_)));
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
