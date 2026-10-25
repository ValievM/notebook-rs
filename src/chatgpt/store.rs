use super::Credentials;
use crate::error::{Error, Result};
use std::path::PathBuf;

/// Where the credentials are kept. A trait so that the refresh logic can be tested without a disk.
pub trait Store: Send + Sync {
    /// The saved credentials, or `Error::SignIn` if there are none.
    fn load(&self) -> Result<Credentials>;
    fn save(&self, credentials: &Credentials) -> Result<()>;
}

/// One JSON file readable by its owner only.
pub struct FileStore {
    path: PathBuf,
}

impl FileStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Store for FileStore {
    fn load(&self) -> Result<Credentials> {
        let raw = match std::fs::read(&self.path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::SignIn("not signed in yet: run `nb login`".into()));
            }
            Err(error) => return Err(Error::Io(error)),
        };
        // Only the position of a parse error is reported: its text could quote the tokens.
        serde_json::from_slice(&raw).map_err(|error| {
            Error::SignIn(format!(
                "{} cannot be read (line {}, column {}): sign in again with `nb login`",
                self.path.display(),
                error.line(),
                error.column()
            ))
        })
    }

    /// Writes under a temporary name with the owner-only mode and renames it into place, so
    /// that an interruption never leaves half-written tokens (the refresh token is single-use:
    /// losing a new pair means signing in again).
    fn save(&self, credentials: &Credentials) -> Result<()> {
        let directory = self
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        create_private_directory(directory)?;
        let temporary = directory.join(format!(".auth-{}.tmp", std::process::id()));
        let written = write_private_file(
            &temporary,
            &serde_json::to_vec_pretty(credentials).map_err(|e| Error::Io(e.into()))?,
        )
        .and_then(|()| std::fs::rename(&temporary, &self.path));
        if written.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        Ok(written?)
    }
}

/// Creates the folder (and its parents) closed to everyone but the owner when it is new.
fn create_private_directory(directory: &std::path::Path) -> Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    Ok(builder.create(directory)?)
}

/// Writes a new file that only the owner can read, and makes sure the bytes are on the disk.
fn write_private_file(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::super::credentials::sample;
    use super::*;

    #[test]
    fn saved_credentials_are_loaded_back_and_the_folder_is_created() {
        let directory = tempfile::tempdir().unwrap();
        let store = FileStore::new(directory.path().join("deep/er/auth.json"));

        store.save(&sample(42)).unwrap();

        assert_eq!(store.load().unwrap(), sample(42));
    }

    #[test]
    fn a_missing_file_means_not_signed_in() {
        let directory = tempfile::tempdir().unwrap();
        let error = FileStore::new(directory.path().join("auth.json"))
            .load()
            .unwrap_err();
        assert!(matches!(error, Error::SignIn(_)), "{error}");
        assert!(error.to_string().contains("nb login"), "{error}");
    }

    #[test]
    fn a_corrupt_file_is_an_error_that_names_the_file_but_not_its_content() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("auth.json");
        std::fs::write(&path, "{ SECRET-GARBAGE").unwrap();

        let error = FileStore::new(path.clone()).load().unwrap_err().to_string();

        assert!(error.contains("auth.json"), "{error}");
        assert!(!error.contains("SECRET-GARBAGE"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn the_file_can_be_read_by_its_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("auth.json");
        let store = FileStore::new(path.clone());

        store.save(&sample(1)).unwrap();
        store.save(&sample(2)).unwrap(); // saving again must keep the mode

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "{mode:o}");
    }

    #[test]
    fn saving_leaves_no_temporary_files_behind_and_replaces_the_old_content() {
        let directory = tempfile::tempdir().unwrap();
        let store = FileStore::new(directory.path().join("auth.json"));
        store.save(&sample(1)).unwrap();
        store.save(&sample(2)).unwrap();

        let names: Vec<String> = std::fs::read_dir(directory.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["auth.json"]);
        assert_eq!(store.load().unwrap().expires_at_ms, 2);
    }
}
