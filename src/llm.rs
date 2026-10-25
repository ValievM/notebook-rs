//! The model behind the answers. One method, so a stronger or a local model can be swapped in.

use anyhow::{bail, Context, Result};
use std::io::Write;
use std::process::{Command, Stdio};

pub trait Llm {
    /// One prompt in, the model's final message out.
    fn complete(&self, prompt: &str) -> Result<String>;
}

/// Codex on the owner's ChatGPT subscription, through `codex exec`.
/// The login is the owner's own (`codex login`); no token passes through this program.
pub struct CodexExec {
    pub binary: String,
}

impl Default for CodexExec {
    fn default() -> Self {
        Self { binary: "codex".into() }
    }
}

impl Llm for CodexExec {
    fn complete(&self, prompt: &str) -> Result<String> {
        let directory = tempfile_dir()?;
        let out = directory.join("answer.txt");
        let mut child = Command::new(&self.binary)
            .args(["exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-C"])
            .arg(&directory)
            .arg("-o")
            .arg(&out)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("cannot start `codex`: is it installed and in PATH?")?;
        child
            .stdin
            .take()
            .context("no stdin")?
            .write_all(prompt.as_bytes())?;
        let status = child.wait()?;
        let text = std::fs::read_to_string(&out).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&directory);
        if !status.success() {
            bail!("codex exited with {status}; try `codex login status`");
        }
        Ok(text)
    }
}

fn tempfile_dir() -> Result<std::path::PathBuf> {
    let directory = std::env::temp_dir().join(format!("notebook-rs-{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    Ok(directory)
}
