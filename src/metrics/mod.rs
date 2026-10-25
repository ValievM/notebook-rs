//! Measuring what the program does and how long it takes, for the person who runs it: where the
//! time goes, how often things fail, how much memory the process holds.
//!
//! Recorded: durations, sizes, counts, bytes, the name of the source an operation was about
//! (a title, a file name or an address: an error is often caused by one particular file) and
//! the error with its message. Never recorded: the text of a source, a question or an answer.
//! (The program has one user, who owns the file the metrics go to, so names are kept.)
//!
//! The pieces: `Recorder` is where events go (a trait, like the model and the search);
//! `Clock` says what time it is (so that tests can set it); `Metrics` is the handle the code
//! holds; `Span` measures one stretch of work.

pub mod instrument;
pub mod report;
pub mod sampler;
pub mod sqlite;

pub use instrument::{InstrumentedEmbedder, InstrumentedLlm, InstrumentedRetriever};
pub use sampler::MemorySampler;
pub use sqlite::{RETENTION_DAYS, SqliteRecorder, Summary, summarize};

use crate::error::Result;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// One measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// When it started, in milliseconds since the Unix epoch.
    pub at_ms: i64,
    /// What was measured, as `area.operation`: `llm.complete`, `embed.query`, `source.add`...
    pub kind: String,
    /// How long it took; `None` for a sample taken at a moment (memory).
    pub micros: Option<u64>,
    pub ok: bool,
    /// Sizes and counts that describe the work: `passages`, `chars`, `rss_bytes`...
    pub fields: Vec<(String, i64)>,
    /// The source the operation was about: a title, a file name or an address.
    pub subject: Option<String>,
    /// The kind of the error if it failed (see `Error::kind`).
    pub error: Option<String>,
    /// The message of the error, cut to `DETAIL_LIMIT` characters.
    pub detail: Option<String>,
}

/// How much of an error message is kept.
pub const DETAIL_LIMIT: usize = 300;

/// Where events go.
pub trait Recorder: Send + Sync {
    fn record(&self, event: Event);
}

/// Drops everything: what the code uses when nobody asked for metrics.
pub struct NullRecorder;

impl Recorder for NullRecorder {
    fn record(&self, _event: Event) {}
}

/// Keeps events in memory, for tests.
#[derive(Default)]
pub struct MemoryRecorder {
    events: Mutex<Vec<Event>>,
}

impl MemoryRecorder {
    pub fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }
}

impl Recorder for MemoryRecorder {
    fn record(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }
}

/// What time it is. Two clocks, because they answer different questions: the wall clock says
/// *when* (it can jump when the person changes the time), the monotonic one says *how long*.
pub trait Clock: Send + Sync {
    fn wall_ms(&self) -> i64;
    fn mono_micros(&self) -> u64;
}

pub struct SystemClock {
    started: Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self {
            started: Instant::now(),
        }
    }
}

impl Clock for SystemClock {
    fn wall_ms(&self) -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_millis() as i64)
    }

    fn mono_micros(&self) -> u64 {
        self.started.elapsed().as_micros() as u64
    }
}

/// A clock that only moves when told to, so that a test can say "this took 250 ms" exactly.
#[derive(Default)]
pub struct ManualClock {
    state: Mutex<(i64, u64)>,
}

impl ManualClock {
    pub fn starting_at(wall_ms: i64) -> Self {
        Self {
            state: Mutex::new((wall_ms, 0)),
        }
    }

    pub fn advance_micros(&self, micros: u64) {
        let mut state = self.state.lock().unwrap();
        state.0 += (micros / 1000) as i64;
        state.1 += micros;
    }
}

impl Clock for ManualClock {
    fn wall_ms(&self) -> i64 {
        self.state.lock().unwrap().0
    }

    fn mono_micros(&self) -> u64 {
        self.state.lock().unwrap().1
    }
}

/// The handle the code holds. Cheap to clone; does nothing when built with `off()`.
#[derive(Clone)]
pub struct Metrics {
    recorder: Arc<dyn Recorder>,
    clock: Arc<dyn Clock>,
    memory: fn() -> Option<u64>,
}

impl Metrics {
    pub fn new(recorder: Arc<dyn Recorder>) -> Self {
        Self::with_clock(recorder, Arc::new(SystemClock::default()))
    }

    pub fn with_clock(recorder: Arc<dyn Recorder>, clock: Arc<dyn Clock>) -> Self {
        Self {
            recorder,
            clock,
            memory: process_rss_bytes,
        }
    }

    pub fn off() -> Self {
        Self::new(Arc::new(NullRecorder))
    }

    /// Replaces how the memory of the process is read (tests give a fixed number).
    pub fn with_memory_probe(mut self, probe: fn() -> Option<u64>) -> Self {
        self.memory = probe;
        self
    }

    /// Starts measuring a stretch of work.
    pub fn span(&self, kind: &str) -> Span<'_> {
        Span {
            metrics: self,
            kind: kind.to_string(),
            at_ms: self.clock.wall_ms(),
            started: self.clock.mono_micros(),
            fields: Vec::new(),
            subject: None,
        }
    }

    /// Records the memory the process holds right now, under `memory.<label>`.
    pub fn sample_memory(&self, label: &str) {
        let Some(bytes) = (self.memory)() else {
            return;
        };
        self.recorder.record(Event {
            at_ms: self.clock.wall_ms(),
            kind: format!("memory.{label}"),
            micros: None,
            ok: true,
            fields: vec![("rss_bytes".to_string(), bytes as i64)],
            subject: None,
            error: None,
            detail: None,
        });
    }
}

/// One stretch of work being measured. Finish it with `run`, which records it whether the
/// work succeeded or failed.
pub struct Span<'m> {
    metrics: &'m Metrics,
    kind: String,
    at_ms: i64,
    started: u64,
    fields: Vec<(String, i64)>,
    subject: Option<String>,
}

impl Span<'_> {
    /// The source this work is about.
    pub fn subject(mut self, subject: &str) -> Self {
        self.subject = Some(subject.to_string());
        self
    }

    /// A number that describes the work, known before it starts (how many texts, how long).
    pub fn field(mut self, name: &str, value: i64) -> Self {
        self.fields.push((name.to_string(), value));
        self
    }

    /// Adds a number known only after the work (how many results came out).
    pub fn set(&mut self, name: &str, value: i64) {
        self.fields.push((name.to_string(), value));
    }

    /// Runs the work and records how long it took and whether it failed. The work may add
    /// fields with `set`. The result is passed through unchanged.
    pub fn run<T>(mut self, work: impl FnOnce(&mut Span<'_>) -> Result<T>) -> Result<T> {
        let outcome = work(&mut self);
        let micros = self
            .metrics
            .clock
            .mono_micros()
            .saturating_sub(self.started);
        let (ok, error, detail) = match &outcome {
            Ok(_) => (true, None, None),
            Err(error) => (
                false,
                Some(error.kind().to_string()),
                Some(error.to_string().chars().take(DETAIL_LIMIT).collect()),
            ),
        };
        self.metrics.recorder.record(Event {
            at_ms: self.at_ms,
            kind: self.kind,
            micros: Some(micros),
            ok,
            fields: self.fields,
            subject: self.subject,
            error,
            detail,
        });
        outcome
    }
}

/// The memory the process holds now, in bytes (resident set size), or `None` if it cannot be read.
pub fn process_rss_bytes() -> Option<u64> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kilobytes: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    Some(kilobytes * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    fn fixture() -> (Metrics, Arc<MemoryRecorder>, Arc<ManualClock>) {
        let recorder = Arc::new(MemoryRecorder::default());
        let clock = Arc::new(ManualClock::starting_at(1_700_000_000_000));
        let metrics = Metrics::with_clock(recorder.clone(), clock.clone());
        (metrics, recorder, clock)
    }

    #[test]
    fn a_stretch_of_work_is_recorded_with_its_start_and_its_exact_duration() {
        let (metrics, recorder, clock) = fixture();
        clock.advance_micros(5_000);

        let result = metrics
            .span("llm.complete")
            .field("prompt_chars", 1200)
            .run(|_| {
                clock.advance_micros(250_000);
                Ok(42)
            });

        assert_eq!(result.unwrap(), 42);
        assert_eq!(
            recorder.events(),
            vec![Event {
                at_ms: 1_700_000_000_005,
                kind: "llm.complete".into(),
                micros: Some(250_000),
                ok: true,
                fields: vec![("prompt_chars".into(), 1200)],
                subject: None,
                error: None,
                detail: None,
            }]
        );
    }

    #[test]
    fn numbers_known_only_after_the_work_are_recorded_too() {
        let (metrics, recorder, _) = fixture();
        metrics
            .span("retrieve.bm25")
            .run(|span| {
                span.set("results", 6);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            recorder.events()[0].fields,
            vec![("results".to_string(), 6)]
        );
    }

    #[test]
    fn a_failure_is_recorded_with_its_kind_the_source_and_the_message_and_passed_on() {
        let (metrics, recorder, clock) = fixture();

        let result: Result<()> = metrics.span("source.add").subject("scan.pdf").run(|_| {
            clock.advance_micros(1_000);
            Err(Error::Source("scan.pdf: the PDF has no text layer".into()))
        });

        assert!(matches!(result, Err(Error::Source(_))));
        let event = &recorder.events()[0];
        assert!(!event.ok);
        assert_eq!(event.micros, Some(1_000));
        assert_eq!(event.subject.as_deref(), Some("scan.pdf"));
        assert_eq!(event.error.as_deref(), Some("source"));
        assert_eq!(
            event.detail.as_deref(),
            Some("cannot read the source: scan.pdf: the PDF has no text layer")
        );
    }

    #[test]
    fn a_long_error_message_is_cut_on_a_character_boundary() {
        let (metrics, recorder, _) = fixture();
        let long = "ошибка ".repeat(100);

        let _ = metrics
            .span("x")
            .run(|_| -> Result<()> { Err(Error::Usage(long.clone())) });

        let detail = recorder.events()[0].detail.clone().unwrap();
        assert_eq!(detail.chars().count(), DETAIL_LIMIT);
        assert!(long.starts_with(&detail));
    }

    #[test]
    fn a_success_carries_its_subject_and_no_error() {
        let (metrics, recorder, _) = fixture();
        metrics
            .span("source.add")
            .subject("договор.md")
            .run(|_| Ok(()))
            .unwrap();
        let event = &recorder.events()[0];
        assert_eq!(event.subject.as_deref(), Some("договор.md"));
        assert_eq!((event.error.clone(), event.detail.clone()), (None, None));
    }

    #[test]
    fn every_error_has_a_short_stable_kind() {
        let kinds = [
            (Error::ModelUnavailable("x".into()), "model_unavailable"),
            (Error::BadReply("x".into()), "bad_reply"),
            (Error::Source("x".into()), "source"),
            (Error::SignIn("x".into()), "sign_in"),
            (Error::TooLarge("x".into()), "too_large"),
            (Error::NotFound("x".into()), "not_found"),
            (Error::Usage("x".into()), "usage"),
            (Error::Io(std::io::Error::other("x")), "io"),
            (
                Error::Storage(rusqlite::Error::QueryReturnedNoRows),
                "storage",
            ),
        ];
        for (error, kind) in kinds {
            assert_eq!(error.kind(), kind);
        }
    }

    #[test]
    fn with_metrics_off_the_work_still_runs_once_and_nothing_is_kept() {
        let mut runs = 0;
        let result = Metrics::off().span("anything").run(|_| {
            runs += 1;
            Ok("done")
        });
        assert_eq!(result.unwrap(), "done");
        assert_eq!(runs, 1);
    }

    #[test]
    fn a_memory_sample_is_an_event_without_a_duration() {
        let (metrics, recorder, _) = fixture();
        let metrics = metrics.with_memory_probe(|| Some(123_456_789));

        metrics.sample_memory("after_start");

        let event = &recorder.events()[0];
        assert_eq!(event.kind, "memory.after_start");
        assert_eq!(event.micros, None);
        assert!(event.ok);
        assert_eq!(event.fields, vec![("rss_bytes".to_string(), 123_456_789)]);
    }

    #[test]
    fn a_memory_that_cannot_be_read_records_nothing() {
        let (metrics, recorder, _) = fixture();
        metrics.with_memory_probe(|| None).sample_memory("x");
        assert!(recorder.events().is_empty());
    }

    #[test]
    fn the_system_clock_tells_a_real_date_and_only_moves_forward() {
        let clock = SystemClock::default();
        // After 2024-01-01 and before 2100.
        assert!((1_704_067_200_000..4_102_444_800_000).contains(&clock.wall_ms()));
        let first = clock.mono_micros();
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert!(clock.mono_micros() > first);
    }

    #[test]
    fn the_memory_of_this_process_can_be_read() {
        let bytes = process_rss_bytes().expect("`ps` is available");
        assert!(bytes > 1_000_000, "{bytes}");
    }
}
