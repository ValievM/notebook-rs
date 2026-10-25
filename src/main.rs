use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use notebook_rs::{
    embed::{Embedder, HashingEmbedder},
    llm::CodexExec,
    service::Notebooks,
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
    /// Add a text or markdown file as a source (the same title replaces the old one).
    Add { notebook: String, file: PathBuf },
    /// Ask a question; the answer uses only this notebook's sources.
    Ask {
        notebook: String,
        question: String,
        /// Plan several searches first (two model calls instead of one): for comparisons and
        /// questions with several parts.
        #[arg(long)]
        deep: bool,
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
    let notebooks = Notebooks::standard(
        Store::open(&path)?,
        embedder,
        Box::new(CodexExec::default()),
    );
    match cli.command {
        Command::Add { notebook, file } => {
            let id = notebooks.notebook(&notebook)?;
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("cannot read {}", file.display()))?;
            let title = file
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| file.display().to_string());
            let count = notebooks.add_source(id, &title, &text)?;
            println!("«{title}» добавлен в «{notebook}»: {count} фрагментов.");
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
    }
    Ok(())
}
