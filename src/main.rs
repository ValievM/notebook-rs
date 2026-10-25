use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use notebook_rs::{answer, llm::CodexExec, store::Store};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nb", about = "Your sources, a strong model, answers with checked citations")]
struct Cli {
    /// Where the notebooks live (default: ~/.notebook-rs/notebooks.db).
    #[arg(long, global = true)]
    db: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Add a text or markdown file as a source (the same title replaces the old one).
    Add { notebook: String, file: PathBuf },
    /// Ask a question; the answer uses only this notebook's sources.
    Ask { notebook: String, question: String },
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
    let mut store = Store::open(&path)?;
    match cli.command {
        Command::Add { notebook, file } => {
            let id = store.create_notebook(&notebook)?;
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("cannot read {}", file.display()))?;
            let title = file
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| file.display().to_string());
            let count = store.add_source(id, &title, &text)?;
            println!("«{title}» добавлен в «{notebook}»: {count} фрагментов.");
        }
        Command::Ask { notebook, question } => {
            let id = store.create_notebook(&notebook)?;
            let passages = store.search(id, &question, 6)?;
            let reply = answer::ask(&CodexExec::default(), &question, &passages)?;
            println!("{}\n", reply.text);
            for citation in &reply.citations {
                println!("  [{}] {} — «{}»", citation.n, citation.source_title, citation.quote);
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
