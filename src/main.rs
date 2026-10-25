use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use notebook_rs::{
    answer::Answer,
    digest::Budget,
    embed::{Embedder, HashingEmbedder},
    llm::CodexExec,
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
    let embedder = embedder(cli.embedder, &path)?;
    let mut notebooks = Notebooks::standard(
        Store::open(&path)?,
        embedder,
        Box::new(CodexExec::default()),
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
            let document = Sources::standard().read(&location)?;
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
