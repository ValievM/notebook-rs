use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use notebook_rs::{
    answer::Answer,
    digest::Budget,
    embed::{Embedder, HashingEmbedder},
    llm::CodexExec,
    metrics::{Clock, Metrics, RETENTION_DAYS, SqliteRecorder, SystemClock, report, summarize},
    service::Notebooks,
    source::Sources,
    store::Store,
};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "nb",
    about = "Your sources, a strong model, answers with checked citations"
)]
struct Cli {
    /// Where the notebooks live (default: ~/.notebook-rs/notebooks.db).
    #[arg(long, global = true)]
    db: Option<PathBuf>,
    /// How passages and questions become vectors. Vectors of one kind are never mixed with
    /// another: after switching, add the sources again.
    #[arg(long, global = true, value_enum, default_value_t = EmbedderKind::default_kind())]
    embedder: EmbedderKind,
    /// How many characters of sources one model call may get in `summarize` and `compare`;
    /// longer material is cut into parts that are summarized at the same time and merged.
    #[arg(long, global = true)]
    max_chars: Option<usize>,
    /// Do not record durations and memory (they go to metrics.db next to the notebooks).
    #[arg(long, global = true)]
    no_metrics: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum EmbedderKind {
    /// Offline word hashing: instant, no download, no synonyms.
    Hashing,
    /// Multilingual E5 model on this machine (built with `--features fastembed`).
    #[cfg(feature = "fastembed")]
    Fastembed,
}

impl EmbedderKind {
    fn default_kind() -> Self {
        #[cfg(feature = "fastembed")]
        return Self::Fastembed;
        #[cfg(not(feature = "fastembed"))]
        Self::Hashing
    }
}

#[cfg_attr(not(feature = "fastembed"), allow(unused_variables))]
fn embedder(kind: EmbedderKind, db_path: &std::path::Path) -> Result<Arc<dyn Embedder>> {
    match kind {
        EmbedderKind::Hashing => Ok(Arc::new(HashingEmbedder::default())),
        #[cfg(feature = "fastembed")]
        EmbedderKind::Fastembed => {
            let cache = db_path.with_file_name("models");
            Ok(Arc::new(notebook_rs::embed_fast::FastEmbedder::new(cache)?))
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Add a source: a text or Markdown file, a PDF file, or a web address (http/https).
    /// The same title replaces the old source.
    Add { notebook: String, location: String },
    /// Ask a question; the answer uses only this notebook's sources.
    Ask {
        notebook: String,
        question: String,
        /// Plan several searches first (two model calls instead of one): for comparisons and
        /// questions with several parts.
        #[arg(long)]
        deep: bool,
    },
    /// A summary of one source (`--source <title>`) or of the whole notebook.
    Summarize {
        notebook: String,
        /// The title of one source; without it every source of the notebook is summarized.
        #[arg(long)]
        source: Option<String>,
    },
    /// What two sources have in common, how they differ and where they contradict.
    Compare {
        notebook: String,
        first: String,
        second: String,
    },
    /// How long things took, how often they failed, how much memory the program held.
    Stats {
        /// How far back to look.
        #[arg(long, default_value_t = 24)]
        hours: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let path = match cli.db {
        Some(path) => path,
        None => {
            let home = std::env::var_os("HOME").context("no HOME")?;
            let directory = PathBuf::from(home).join(".notebook-rs");
            std::fs::create_dir_all(&directory)?;
            directory.join("notebooks.db")
        }
    };
    let metrics_path = path.with_file_name("metrics.db");
    let clock = SystemClock::default();
    if let Command::Stats { hours } = cli.command {
        return print_stats(&metrics_path, hours, clock.wall_ms());
    }
    // Kept in this scope: dropping it at the end of `main` writes what is still waiting.
    let recorder = (!cli.no_metrics)
        .then(|| SqliteRecorder::open(&metrics_path, RETENTION_DAYS, clock.wall_ms()))
        .and_then(|opened| {
            opened
                .inspect_err(|error| eprintln!("metrics are off: {error}"))
                .ok()
        })
        .map(Arc::new);
    let metrics = recorder
        .clone()
        .map_or_else(Metrics::off, |r| Metrics::new(r));
    let embedder = metrics.span("engine.model_load").run(|_| {
        embedder(cli.embedder, &path)
            .map_err(|e| notebook_rs::error::Error::ModelUnavailable(e.to_string()))
    })?;
    metrics.sample_memory("after_model_load");
    let mut notebooks = Notebooks::standard_with_metrics(
        Store::open(&path)?,
        embedder,
        Box::new(CodexExec::default()),
        metrics.clone(),
    );
    if let Some(chars_per_call) = cli.max_chars {
        notebooks = notebooks.with_budget(Budget {
            chars_per_call,
            ..Budget::default()
        });
    }
    match cli.command {
        Command::Add { notebook, location } => {
            let id = notebooks.notebook(&notebook)?;
            let document = Sources::standard()
                .with_metrics(metrics.clone())
                .read(&location)?;
            let count = notebooks.add_source(id, &document.title, &document.text)?;
            println!(
                "«{}» добавлен в «{notebook}»: {count} фрагментов.",
                document.title
            );
        }
        Command::Ask {
            notebook,
            question,
            deep,
        } => {
            let id = notebooks.notebook(&notebook)?;
            let reply = if deep {
                notebooks.ask_deep(id, &question)?
            } else {
                notebooks.ask(id, &question)?
            };
            print_answer(&reply);
        }
        Command::Summarize { notebook, source } => {
            let id = notebooks.notebook(&notebook)?;
            print_answer(&notebooks.summarize(id, source.as_deref())?);
        }
        Command::Compare {
            notebook,
            first,
            second,
        } => {
            let id = notebooks.notebook(&notebook)?;
            print_answer(&notebooks.compare(id, &first, &second)?);
        }
        Command::Stats { .. } => unreachable!("handled before the notebooks are opened"),
    }
    Ok(())
}

/// The answer, then the citations that were found in the sources, then how many were not.
fn print_answer(reply: &Answer) {
    println!("{}\n", reply.text);
    for citation in &reply.citations {
        println!(
            "  [{}] {} — «{}»",
            citation.n, citation.source_title, citation.quote
        );
    }
    if reply.rejected_quotes > 0 {
        println!(
            "\n(отброшено цитат, которых нет в источниках: {})",
            reply.rejected_quotes
        );
    }
}

fn print_stats(path: &std::path::Path, hours: u64, now_ms: i64) -> Result<()> {
    if !path.exists() {
        println!(
            "Метрик ещё нет: они появятся после первой операции ({}).",
            path.display()
        );
        return Ok(());
    }
    let summary = summarize(path, now_ms - (hours as i64) * 3_600_000)
        .with_context(|| format!("cannot read {}", path.display()))?;
    print!("{}", report::render(&summary, hours, now_ms));
    Ok(())
}
