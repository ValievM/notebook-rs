//! The choices the person made for the model (which one, how much it thinks), kept in a small
//! JSON file next to the notebooks so that the command line and the window agree.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// How much the model thinks before it answers.
pub const EFFORTS: [&str; 4] = ["minimal", "low", "medium", "high"];

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

impl Settings {
    /// Reads the file; a missing file means no choices yet.
    pub fn load(path: &Path) -> Result<Self> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(Error::Io(error)),
        };
        serde_json::from_str(&text)
            .map_err(|e| Error::Usage(format!("{} is not valid settings: {e}", path.display())))
    }

    /// The saved choices, with a flag or an environment variable taking precedence.
    pub fn overridden_by(self, model: Option<String>, effort: Option<String>) -> Self {
        Self {
            model: model.or(self.model),
            effort: effort.or(self.effort),
        }
    }

    /// Writes the file whole or not at all.
    pub fn save(&self, path: &Path) -> Result<()> {
        let temporary = path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| Error::Usage(format!("cannot write settings: {e}")))?;
        std::fs::write(&temporary, text)?;
        std::fs::rename(&temporary, path)?;
        Ok(())
    }
}

/// Checks that `effort` is one of [`EFFORTS`].
pub fn check_effort(effort: &str) -> Result<()> {
    if EFFORTS.contains(&effort) {
        Ok(())
    } else {
        Err(Error::Usage(format!(
            "unknown effort \"{effort}\": use one of {}",
            EFFORTS.join(", ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_means_no_choices() {
        let directory = tempfile::tempdir().unwrap();
        let loaded = Settings::load(&directory.path().join("settings.json")).unwrap();
        assert_eq!(loaded, Settings::default());
    }

    #[test]
    fn what_was_saved_is_loaded_back() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.json");
        let settings = Settings {
            model: Some("m-1".into()),
            effort: Some("medium".into()),
        };
        settings.save(&path).unwrap();
        assert_eq!(Settings::load(&path).unwrap(), settings);
        let leftovers: Vec<_> = std::fs::read_dir(directory.path()).unwrap().collect();
        assert_eq!(leftovers.len(), 1, "no temporary file stays behind");
    }

    #[test]
    fn a_broken_file_is_an_error_that_names_the_file_not_silently_ignored() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.json");
        std::fs::write(&path, "{not json").unwrap();
        let error = Settings::load(&path).unwrap_err();
        assert!(matches!(error, Error::Usage(_)), "{error}");
        assert!(error.to_string().contains("settings.json"), "{error}");
    }

    #[test]
    fn a_flag_beats_the_saved_choice_and_the_rest_stays() {
        let saved = Settings {
            model: Some("saved".into()),
            effort: Some("medium".into()),
        };
        let merged = saved.overridden_by(Some("flag".into()), None);
        assert_eq!(merged.model.as_deref(), Some("flag"));
        assert_eq!(merged.effort.as_deref(), Some("medium"));
    }

    #[test]
    fn only_known_efforts_pass() {
        for effort in EFFORTS {
            check_effort(effort).unwrap();
        }
        let error = check_effort("extreme").unwrap_err();
        assert!(matches!(error, Error::Usage(_)));
        assert!(error.to_string().contains("medium"), "{error}");
    }
}
