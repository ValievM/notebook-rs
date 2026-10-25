//! The model behind the answers. One method, so a stronger or a local model can be swapped in.

use crate::error::{Error, Result};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

pub trait Llm: Send + Sync {
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
        Self {
            binary: "codex".into(),
        }
    }
}

impl Llm for CodexExec {
    fn complete(&self, prompt: &str) -> Result<String> {
        let directory = tempfile_dir()?;
        let out = directory.join("answer.txt");
        let mut child = Command::new(&self.binary)
            .args([
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "-s",
                "read-only",
                "-C",
            ])
            .arg(&directory)
            .arg("-o")
            .arg(&out)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| Error::ModelUnavailable(format!("cannot start `{}`: {e}", self.binary)))?;
        child
            .stdin
            .take()
            .ok_or_else(|| Error::ModelUnavailable("no stdin".into()))?
            .write_all(prompt.as_bytes())?;
        let status = child.wait()?;
        let text = std::fs::read_to_string(&out).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&directory);
        if !status.success() {
            return Err(Error::ModelUnavailable(format!(
                "codex exited with {status}; try `codex login status`"
            )));
        }
        Ok(text)
    }
}

/// A folder of its own for every call: calls may run at the same time (see `digest`), and
/// two of them must never share the file the answer is written to.
fn tempfile_dir() -> Result<std::path::PathBuf> {
    // Only uniqueness matters, not ordering between threads, so `Relaxed` is enough.
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!("notebook-rs-{}-{call}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    Ok(directory)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::parallel::map_ordered;
    use std::os::unix::fs::PermissionsExt;

    /// A stand-in for `codex`: waits a little (so that parallel calls really overlap), then
    /// writes the prompt it got on stdin into the file named after `-o`.
    fn fake_codex(directory: &std::path::Path) -> String {
        let path = directory.join("codex");
        std::fs::write(
            &path,
            "#!/bin/sh\nout=\"\"\nwhile [ $# -gt 0 ]; do\n  if [ \"$1\" = \"-o\" ]; then out=\"$2\"; fi\n  shift\ndone\nprompt=$(cat)\nsleep 0.3\nprintf '%s' \"$prompt\" > \"$out\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn parallel_calls_do_not_mix_up_their_answers() {
        let directory = tempfile::tempdir().unwrap();
        let llm = CodexExec {
            binary: fake_codex(directory.path()),
        };
        let prompts: Vec<String> = (0..4).map(|i| format!("prompt number {i}")).collect();
        let answers = map_ordered(&prompts, |prompt| llm.complete(prompt).unwrap());
        assert_eq!(answers, prompts);
    }

    #[test]
    fn a_failing_codex_is_an_unavailable_model() {
        let llm = CodexExec {
            binary: "/usr/bin/false".into(),
        };
        assert!(matches!(llm.complete("x"), Err(Error::ModelUnavailable(_))));
    }
}
