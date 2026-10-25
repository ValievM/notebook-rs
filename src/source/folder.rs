//! A folder as a source: every text, Markdown and PDF file in it and in its subfolders becomes a
//! source of its own, titled with its path inside the folder (so `a/notes.md` and `b/notes.md`
//! stay apart).

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

/// Files a folder is searched for.
pub const EXTENSIONS: [&str; 4] = ["txt", "md", "markdown", "pdf"];
/// How many files one folder may bring: a guard against pointing the program at a whole disk.
pub const MAX_FILES: usize = 500;
const MAX_DEPTH: usize = 8;

/// The supported files of the folder as (title, path), sorted by title. Hidden files and folders
/// (starting with a dot) are skipped and links are not followed.
pub fn files_in(folder: &Path) -> Result<Vec<(String, PathBuf)>> {
    files_in_limited(folder, MAX_FILES)
}

fn files_in_limited(folder: &Path, max: usize) -> Result<Vec<(String, PathBuf)>> {
    let mut found = Vec::new();
    walk(folder, folder, 0, max, &mut found)?;
    found.sort();
    if found.is_empty() {
        return Err(Error::Source(format!(
            "{}: no files of a supported kind ({})",
            folder.display(),
            EXTENSIONS.join(", ")
        )));
    }
    Ok(found)
}

fn walk(
    root: &Path,
    folder: &Path,
    depth: usize,
    max: usize,
    found: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    let entries = std::fs::read_dir(folder)
        .map_err(|e| Error::Source(format!("{}: {e}", folder.display())))?;
    for entry in entries {
        let entry = entry.map_err(|e| Error::Source(format!("{}: {e}", folder.display())))?;
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_dir() && depth < MAX_DEPTH {
            walk(root, &path, depth + 1, max, found)?;
        } else if kind.is_file() && supported(&path) {
            if found.len() >= max {
                return Err(Error::TooLarge(format!(
                    "{}: more than {max} files; add a smaller folder",
                    root.display()
                )));
            }
            let title = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .components()
                .map(|part| part.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            found.push((title, path));
        }
    }
    Ok(())
}

fn supported(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.contains(&e.to_lowercase().as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(root: &Path, name: &str) {
        let path = root.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, "текст").unwrap();
    }

    #[test]
    fn supported_files_are_found_in_subfolders_titled_by_their_path_and_sorted() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "b.txt",
            "a/notes.md",
            "c/notes.md",
            "doc.PDF",
            "pic.png",
            ".hidden.txt",
            ".git/x.md",
        ] {
            touch(dir.path(), name);
        }

        let titles: Vec<String> = files_in(dir.path())
            .unwrap()
            .into_iter()
            .map(|(title, _)| title)
            .collect();

        assert_eq!(titles, ["a/notes.md", "b.txt", "c/notes.md", "doc.PDF"]);
    }

    #[cfg(unix)]
    #[test]
    fn links_are_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        touch(outside.path(), "secret.txt");
        touch(dir.path(), "own.txt");
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("secret.txt"),
            dir.path().join("link.txt"),
        )
        .unwrap();

        let titles: Vec<String> = files_in(dir.path())
            .unwrap()
            .into_iter()
            .map(|f| f.0)
            .collect();

        assert_eq!(titles, ["own.txt"]);
    }

    #[test]
    fn a_folder_without_supported_files_is_an_error_that_says_what_is_supported() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "pic.png");
        let error = files_in(dir.path()).unwrap_err().to_string();
        assert!(error.contains("pdf") && error.contains("md"), "{error}");
    }

    #[test]
    fn too_many_files_are_refused_not_cut_silently() {
        let dir = tempfile::tempdir().unwrap();
        for n in 0..4 {
            touch(dir.path(), &format!("{n}.txt"));
        }
        assert!(files_in_limited(dir.path(), 4).is_ok());
        let error = files_in_limited(dir.path(), 3).unwrap_err();
        assert!(matches!(error, Error::TooLarge(_)), "{error}");
    }
}
