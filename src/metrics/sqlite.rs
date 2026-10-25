//! Keeping events in a SQLite file, and reading them back as a summary.
//!
//! Writing must never slow the program down or break it, so `record` only puts the event into
//! a channel; one thread owns the connection and writes what arrives, many events in one
//! transaction. (A channel fits here, unlike in the search: many threads produce events, one
//! consumer writes them, nobody waits for an answer.) If the disk fails, events are lost, the
//! program carries on.

use super::{Event, Recorder};
use crate::error::Result;
use rusqlite::{Connection, OpenFlags, params};
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::thread::JoinHandle;
use std::time::Duration;

const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS metric (
        id INTEGER PRIMARY KEY,
        at_ms INTEGER NOT NULL,
        kind TEXT NOT NULL,
        micros INTEGER,
        ok INTEGER NOT NULL,
        subject TEXT,
        error TEXT,
        detail TEXT,
        fields TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS metric_by_time ON metric(at_ms);
    CREATE INDEX IF NOT EXISTS metric_by_kind ON metric(kind, at_ms);";

/// How long events are kept; older ones are forgotten when the file is opened.
pub const RETENTION_DAYS: u32 = 30;

const DAY_MS: i64 = 86_400_000;
/// At most this many events go into one transaction.
const BATCH: usize = 256;

enum Message {
    Event(Event),
    /// Answers when everything sent before it has been written.
    Flush(SyncSender<()>),
}

pub struct SqliteRecorder {
    sender: Option<Sender<Message>>,
    writer: Option<JoinHandle<()>>,
}

impl SqliteRecorder {
    /// Opens (creating if needed) the metrics file and forgets what is older than `keep_days`
    /// before `now_ms`.
    pub fn open(path: &Path, keep_days: u32, now_ms: i64) -> Result<Self> {
        let db = Connection::open(path)?;
        db.busy_timeout(Duration::from_secs(2))?;
        // WAL: `nb stats` can read the file while the window is writing to it.
        db.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
        db.execute_batch(SCHEMA)?;
        db.execute(
            "DELETE FROM metric WHERE at_ms < ?1",
            [now_ms - i64::from(keep_days) * DAY_MS],
        )?;
        let (sender, receiver) = mpsc::channel();
        let writer = std::thread::Builder::new()
            .name("metrics-writer".into())
            .spawn(move || write_loop(db, receiver))?;
        Ok(Self {
            sender: Some(sender),
            writer: Some(writer),
        })
    }

    /// Waits until every event recorded so far is in the file.
    pub fn flush(&self) {
        let Some(sender) = &self.sender else { return };
        let (reply, answered) = mpsc::sync_channel(1);
        if sender.send(Message::Flush(reply)).is_ok() {
            let _ = answered.recv();
        }
    }
}

impl Recorder for SqliteRecorder {
    fn record(&self, event: Event) {
        if let Some(sender) = &self.sender {
            // A writer that is gone loses the event; the program goes on.
            let _ = sender.send(Message::Event(event));
        }
    }
}

impl Drop for SqliteRecorder {
    /// Closes the channel and waits for the writer to finish what is in it.
    fn drop(&mut self) {
        self.sender.take();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

/// The writer thread: takes what has arrived (up to a batch), writes it in one transaction,
/// then answers the flushes that were waiting.
fn write_loop(mut db: Connection, receiver: Receiver<Message>) {
    while let Ok(first) = receiver.recv() {
        let mut messages = vec![first];
        while messages.len() < BATCH {
            match receiver.try_recv() {
                Ok(message) => messages.push(message),
                Err(_) => break,
            }
        }
        let mut events = Vec::new();
        let mut flushes = Vec::new();
        for message in messages {
            match message {
                Message::Event(event) => events.push(event),
                Message::Flush(reply) => flushes.push(reply),
            }
        }
        if let Err(error) = insert_all(&mut db, &events) {
            eprintln!("notebook-rs: metrics were not written: {error}");
        }
        for reply in flushes {
            let _ = reply.send(());
        }
    }
}

fn insert_all(db: &mut Connection, events: &[Event]) -> rusqlite::Result<()> {
    let tx = db.transaction()?;
    {
        let mut insert = tx.prepare(
            "INSERT INTO metric(at_ms, kind, micros, ok, subject, error, detail, fields)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )?;
        for event in events {
            let fields: serde_json::Map<String, serde_json::Value> = event
                .fields
                .iter()
                .map(|(name, value)| (name.clone(), serde_json::Value::from(*value)))
                .collect();
            insert.execute(params![
                event.at_ms,
                event.kind,
                event.micros.map(|m| m as i64),
                event.ok,
                event.subject,
                event.error,
                event.detail,
                serde_json::Value::Object(fields).to_string(),
            ])?;
        }
    }
    tx.commit()
}

/// How long operations of one kind took, and how often they failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KindStats {
    pub kind: String,
    pub count: u64,
    pub errors: u64,
    pub p50_micros: Option<u64>,
    pub p95_micros: Option<u64>,
    pub max_micros: Option<u64>,
    pub total_micros: u64,
}

/// The memory samples of one moment of the program (`memory.<label>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryStats {
    pub label: String,
    pub last_bytes: u64,
    pub max_bytes: u64,
    pub samples: u64,
}

/// An operation that failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureRow {
    pub at_ms: i64,
    pub kind: String,
    pub subject: Option<String>,
    pub error: String,
    pub detail: String,
}

/// One of the slowest operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlowRow {
    pub at_ms: i64,
    pub kind: String,
    pub subject: Option<String>,
    pub micros: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    pub since_ms: i64,
    pub kinds: Vec<KindStats>,
    pub memory: Vec<MemoryStats>,
    /// The latest failures, newest first.
    pub failures: Vec<FailureRow>,
    /// The slowest operations, slowest first.
    pub slowest: Vec<SlowRow>,
}

/// Reads the file and summarizes what happened since `since_ms`.
pub fn summarize(path: &Path, since_ms: i64) -> Result<Summary> {
    // Read-write without creating: a file that is not there is an error, and a file in WAL mode
    // that the window is writing to can still be read.
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    db.busy_timeout(Duration::from_secs(2))?;
    Ok(Summary {
        since_ms,
        kinds: kind_stats(&db, since_ms)?,
        memory: memory_stats(&db, since_ms)?,
        failures: failures(&db, since_ms)?,
        slowest: slowest(&db, since_ms)?,
    })
}

fn kind_stats(db: &Connection, since_ms: i64) -> Result<Vec<KindStats>> {
    let mut statement = db.prepare(
        "SELECT kind, micros, ok FROM metric WHERE at_ms >= ?1 AND kind NOT LIKE 'memory.%'",
    )?;
    let rows = statement.query_map([since_ms], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, Option<i64>>(1)?,
            r.get::<_, bool>(2)?,
        ))
    })?;
    // kind -> (how many, how many failed, the durations)
    let mut grouped: HashMap<String, (u64, u64, Vec<u64>)> = HashMap::new();
    for row in rows {
        let (kind, micros, ok) = row?;
        let entry = grouped.entry(kind).or_default();
        entry.0 += 1;
        entry.1 += u64::from(!ok);
        entry.2.extend(micros.map(|m| m as u64));
    }
    let mut stats: Vec<KindStats> = grouped
        .into_iter()
        .map(|(kind, (count, errors, mut durations))| {
            durations.sort_unstable();
            KindStats {
                kind,
                count,
                errors,
                p50_micros: percentile(&durations, 50.0),
                p95_micros: percentile(&durations, 95.0),
                max_micros: durations.last().copied(),
                total_micros: durations.iter().sum(),
            }
        })
        .collect();
    stats.sort_by(|a, b| {
        b.total_micros
            .cmp(&a.total_micros)
            .then(a.kind.cmp(&b.kind))
    });
    Ok(stats)
}

fn memory_stats(db: &Connection, since_ms: i64) -> Result<Vec<MemoryStats>> {
    let mut statement = db.prepare(
        "SELECT kind, fields FROM metric WHERE at_ms >= ?1 AND kind LIKE 'memory.%' ORDER BY at_ms",
    )?;
    let rows = statement.query_map([since_ms], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
    })?;
    let mut by_label: HashMap<String, MemoryStats> = HashMap::new();
    for row in rows {
        let (kind, fields) = row?;
        let Some(bytes) = serde_json::from_str::<serde_json::Value>(&fields)
            .ok()
            .and_then(|f| f["rss_bytes"].as_u64())
        else {
            continue;
        };
        let label = kind.trim_start_matches("memory.").to_string();
        let entry = by_label.entry(label.clone()).or_insert(MemoryStats {
            label,
            last_bytes: 0,
            max_bytes: 0,
            samples: 0,
        });
        entry.last_bytes = bytes; // rows come oldest first, so the last one wins
        entry.max_bytes = entry.max_bytes.max(bytes);
        entry.samples += 1;
    }
    let mut stats: Vec<MemoryStats> = by_label.into_values().collect();
    stats.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(stats)
}

fn failures(db: &Connection, since_ms: i64) -> Result<Vec<FailureRow>> {
    let mut statement = db.prepare(
        "SELECT at_ms, kind, subject, error, detail FROM metric
         WHERE at_ms >= ?1 AND ok = 0 ORDER BY at_ms DESC LIMIT 10",
    )?;
    let rows = statement.query_map([since_ms], |r| {
        Ok(FailureRow {
            at_ms: r.get(0)?,
            kind: r.get(1)?,
            subject: r.get(2)?,
            error: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
            detail: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
        })
    })?;
    Ok(rows.collect::<std::result::Result<_, _>>()?)
}

fn slowest(db: &Connection, since_ms: i64) -> Result<Vec<SlowRow>> {
    let mut statement = db.prepare(
        "SELECT at_ms, kind, subject, micros FROM metric
         WHERE at_ms >= ?1 AND micros IS NOT NULL AND kind NOT LIKE 'memory.%'
         ORDER BY micros DESC LIMIT 5",
    )?;
    let rows = statement.query_map([since_ms], |r| {
        Ok(SlowRow {
            at_ms: r.get(0)?,
            kind: r.get(1)?,
            subject: r.get(2)?,
            micros: r.get::<_, i64>(3)? as u64,
        })
    })?;
    Ok(rows.collect::<std::result::Result<_, _>>()?)
}

/// The value below which `percent` percent of the sorted values lie (nearest rank).
pub fn percentile(sorted: &[u64], percent: f64) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = ((percent / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    Some(sorted[rank.min(sorted.len()) - 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    const NOW: i64 = 1_700_000_000_000;
    const DAY: i64 = 86_400_000;

    fn event(kind: &str, at_ms: i64, micros: Option<u64>) -> Event {
        Event {
            at_ms,
            kind: kind.into(),
            micros,
            ok: true,
            fields: vec![],
            subject: None,
            error: None,
            detail: None,
        }
    }

    fn failed(kind: &str, at_ms: i64, subject: &str, error: &str, detail: &str) -> Event {
        Event {
            ok: false,
            subject: Some(subject.into()),
            error: Some(error.into()),
            detail: Some(detail.into()),
            ..event(kind, at_ms, Some(1_000))
        }
    }

    fn rows(path: &Path) -> i64 {
        Connection::open(path)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM metric", [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn percentiles_use_the_nearest_rank() {
        let hundred: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile(&hundred, 50.0), Some(50));
        assert_eq!(percentile(&hundred, 95.0), Some(95));
        assert_eq!(percentile(&hundred, 100.0), Some(100));
        assert_eq!(percentile(&[7], 95.0), Some(7));
        assert_eq!(percentile(&[1, 2, 3, 4], 50.0), Some(2));
        assert_eq!(percentile(&[], 50.0), None);
    }

    #[test]
    fn events_are_written_with_all_their_fields() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.db");
        let recorder = SqliteRecorder::open(&path, 30, NOW).unwrap();

        recorder.record(Event {
            fields: vec![("passages".into(), 12), ("chars".into(), 3400)],
            ..failed("source.add", NOW, "договор.pdf", "source", "no text layer")
        });
        recorder.flush();

        let db = Connection::open(&path).unwrap();
        let column = |name: &str| -> Option<String> {
            db.query_row(
                &format!("SELECT CAST({name} AS TEXT) FROM metric"),
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(column("kind").as_deref(), Some("source.add"));
        assert_eq!(column("micros").as_deref(), Some("1000"));
        assert_eq!(column("ok").as_deref(), Some("0"));
        assert_eq!(column("subject").as_deref(), Some("договор.pdf"));
        assert_eq!(column("error").as_deref(), Some("source"));
        assert_eq!(column("detail").as_deref(), Some("no text layer"));
        let fields: serde_json::Value = serde_json::from_str(&column("fields").unwrap()).unwrap();
        assert_eq!(fields, serde_json::json!({"passages": 12, "chars": 3400}));
    }

    #[test]
    fn events_not_yet_flushed_are_still_written_when_the_recorder_is_dropped() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.db");
        {
            let recorder = SqliteRecorder::open(&path, 30, NOW).unwrap();
            // Enough events that the writer is still busy when the recorder is dropped: only
            // a drop that waits for it finds all of them in the file.
            for i in 0..20_000 {
                recorder.record(event("a", NOW + i, Some(1)));
            }
        }
        assert_eq!(rows(&path), 20_000);
    }

    #[test]
    fn many_threads_record_at_the_same_time_and_nothing_is_lost() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.db");
        let recorder = SqliteRecorder::open(&path, 30, NOW).unwrap();
        std::thread::scope(|scope| {
            for t in 0..8 {
                let recorder = &recorder;
                scope.spawn(move || {
                    for i in 0..50 {
                        recorder.record(event("a", NOW + t * 1000 + i, Some(1)));
                    }
                });
            }
        });
        recorder.flush();
        assert_eq!(rows(&path), 400);
    }

    #[test]
    fn what_is_older_than_the_retention_is_forgotten_when_the_file_is_opened() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.db");
        {
            let recorder = SqliteRecorder::open(&path, 30, NOW).unwrap();
            recorder.record(event("old", NOW - 31 * DAY, Some(1)));
            recorder.record(event("recent", NOW - 29 * DAY, Some(1)));
        }
        let _reopened = SqliteRecorder::open(&path, 30, NOW).unwrap();
        let db = Connection::open(&path).unwrap();
        let kinds: Vec<String> = db
            .prepare("SELECT kind FROM metric")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|k| k.unwrap())
            .collect();
        assert_eq!(kinds, vec!["recent"]);
    }

    #[test]
    fn a_file_that_cannot_be_opened_is_an_error_not_a_panic() {
        let directory = tempfile::tempdir().unwrap();
        assert!(SqliteRecorder::open(directory.path(), 30, NOW).is_err());
    }

    fn filled() -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.db");
        let recorder = SqliteRecorder::open(&path, 30, NOW).unwrap();
        for micros in 1..=100u64 {
            recorder.record(event(
                "llm.complete",
                NOW + micros as i64,
                Some(micros * 1000),
            ));
        }
        recorder.record(failed(
            "llm.complete",
            NOW + 200,
            "a.txt",
            "model_unavailable",
            "codex exited",
        ));
        recorder.record(event("embed.query", NOW + 300, Some(40_000)));
        recorder.record(event("old.thing", NOW - 5 * DAY, Some(9)));
        for (i, bytes) in [900, 1500, 1200].into_iter().enumerate() {
            let mut sample = event("memory.after_start", NOW + 400 + i as i64, None);
            sample.fields = vec![("rss_bytes".into(), bytes)];
            recorder.record(sample);
        }
        recorder.flush();
        (directory, path)
    }

    #[test]
    fn the_summary_counts_failures_and_gives_percentiles_per_kind() {
        let (_directory, path) = filled();
        let summary = summarize(&path, NOW - DAY).unwrap();

        let llm = summary
            .kinds
            .iter()
            .find(|k| k.kind == "llm.complete")
            .unwrap();
        assert_eq!((llm.count, llm.errors), (101, 1));
        // 101 events took 1..=100 ms and 1 ms (the failed one): sorted, the median is the 51st.
        assert_eq!(llm.p50_micros, Some(50_000));
        assert_eq!(llm.p95_micros, Some(95_000));
        assert_eq!(llm.max_micros, Some(100_000));
        assert_eq!(llm.total_micros, (1..=100u64).sum::<u64>() * 1000 + 1_000);
        assert!(
            summary.kinds.iter().all(|k| !k.kind.starts_with("memory")),
            "memory has its own list"
        );
    }

    #[test]
    fn the_summary_leaves_out_what_happened_before_the_window() {
        let (_directory, path) = filled();
        let summary = summarize(&path, NOW - DAY).unwrap();
        assert!(summary.kinds.iter().all(|k| k.kind != "old.thing"));
        let all = summarize(&path, NOW - 10 * DAY).unwrap();
        assert!(all.kinds.iter().any(|k| k.kind == "old.thing"));
    }

    #[test]
    fn the_summary_lists_memory_by_moment_with_the_last_and_the_highest_value() {
        let (_directory, path) = filled();
        let summary = summarize(&path, NOW - DAY).unwrap();
        assert_eq!(
            summary.memory,
            vec![MemoryStats {
                label: "after_start".into(),
                last_bytes: 1200,
                max_bytes: 1500,
                samples: 3
            }]
        );
    }

    #[test]
    fn the_summary_names_the_failed_source_and_the_message_and_the_slowest_operations() {
        let (_directory, path) = filled();
        let summary = summarize(&path, NOW - DAY).unwrap();
        assert_eq!(
            summary.failures,
            vec![FailureRow {
                at_ms: NOW + 200,
                kind: "llm.complete".into(),
                subject: Some("a.txt".into()),
                error: "model_unavailable".into(),
                detail: "codex exited".into(),
            }]
        );
        assert_eq!(summary.slowest[0].micros, 100_000);
        assert!(summary.slowest.len() <= 5);
        assert!(
            summary
                .slowest
                .windows(2)
                .all(|w| w[0].micros >= w[1].micros)
        );
    }

    #[test]
    fn only_the_ten_latest_failures_are_listed_newest_first() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.db");
        {
            let recorder = SqliteRecorder::open(&path, 30, NOW).unwrap();
            for i in 0..12 {
                recorder.record(failed("x", NOW + i, &format!("f{i}"), "io", "boom"));
            }
        }
        let summary = summarize(&path, NOW - DAY).unwrap();
        let subjects: Vec<&str> = summary
            .failures
            .iter()
            .map(|f| f.subject.as_deref().unwrap())
            .collect();
        assert_eq!(
            subjects,
            ["f11", "f10", "f9", "f8", "f7", "f6", "f5", "f4", "f3", "f2"]
        );
    }

    #[test]
    fn a_missing_file_is_an_error() {
        assert!(summarize(Path::new("/no/such/metrics.db"), 0).is_err());
    }
}
