//! Notebooks, sources and their passages, in one SQLite file, searched with FTS5 (BM25).

use crate::chunk::{self, Chunk};
use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::Path;

const TARGET_CHUNK_CHARS: usize = 900;

/// A passage found for a question, with where it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct Passage {
    pub source_title: String,
    pub ordinal: usize,
    pub text: String,
}

pub struct Store {
    db: Connection,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_connection(Connection::open(path).context("cannot open the notebook file")?)
    }

    pub fn in_memory() -> Result<Self> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(db: Connection) -> Result<Self> {
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS notebook (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
             CREATE TABLE IF NOT EXISTS source (
               id INTEGER PRIMARY KEY,
               notebook_id INTEGER NOT NULL REFERENCES notebook(id),
               title TEXT NOT NULL,
               UNIQUE (notebook_id, title));
             CREATE TABLE IF NOT EXISTS passage (
               id INTEGER PRIMARY KEY,
               source_id INTEGER NOT NULL REFERENCES source(id),
               ordinal INTEGER NOT NULL,
               text TEXT NOT NULL);
             CREATE VIRTUAL TABLE IF NOT EXISTS passage_index USING fts5(
               text, content='passage', content_rowid='id', tokenize='unicode61 remove_diacritics 2');",
        )?;
        Ok(Self { db })
    }

    pub fn create_notebook(&self, name: &str) -> Result<i64> {
        self.db
            .execute("INSERT OR IGNORE INTO notebook(name) VALUES (?1)", [name])?;
        Ok(self
            .db
            .query_row("SELECT id FROM notebook WHERE name = ?1", [name], |r| r.get(0))?)
    }

    /// Adds a source, or replaces it if one with the same title is already there (no copies).
    pub fn add_source(&mut self, notebook: i64, title: &str, text: &str) -> Result<usize> {
        let chunks = chunk::split(text, TARGET_CHUNK_CHARS);
        let tx = self.db.transaction()?;
        let old: Option<i64> = tx
            .query_row(
                "SELECT id FROM source WHERE notebook_id = ?1 AND title = ?2",
                params![notebook, title],
                |r| r.get(0),
            )
            .ok();
        if let Some(id) = old {
            tx.execute(
                "INSERT INTO passage_index(passage_index, rowid, text)
                 SELECT 'delete', id, text FROM passage WHERE source_id = ?1",
                [id],
            )?;
            tx.execute("DELETE FROM passage WHERE source_id = ?1", [id])?;
            tx.execute("DELETE FROM source WHERE id = ?1", [id])?;
        }
        tx.execute(
            "INSERT INTO source(notebook_id, title) VALUES (?1, ?2)",
            params![notebook, title],
        )?;
        let source_id = tx.last_insert_rowid();
        for Chunk { ordinal, text } in &chunks {
            tx.execute(
                "INSERT INTO passage(source_id, ordinal, text) VALUES (?1, ?2, ?3)",
                params![source_id, *ordinal as i64, text],
            )?;
            let id = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO passage_index(rowid, text) VALUES (?1, ?2)",
                params![id, text],
            )?;
        }
        tx.commit()?;
        Ok(chunks.len())
    }

    pub fn source_count(&self, notebook: i64) -> Result<usize> {
        Ok(self.db.query_row(
            "SELECT COUNT(*) FROM source WHERE notebook_id = ?1",
            [notebook],
            |r| r.get::<_, i64>(0),
        )? as usize)
    }

    /// The best `limit` passages for a question, best first. Only this notebook is searched.
    pub fn search(&self, notebook: i64, question: &str, limit: usize) -> Result<Vec<Passage>> {
        let Some(query) = fts_query(question) else {
            return Ok(Vec::new());
        };
        let mut statement = self.db.prepare(
            "SELECT s.title, p.ordinal, p.text
             FROM passage_index
             JOIN passage p ON p.id = passage_index.rowid
             JOIN source s ON s.id = p.source_id
             WHERE passage_index MATCH ?1 AND s.notebook_id = ?2
             ORDER BY bm25(passage_index) LIMIT ?3",
        )?;
        let rows = statement.query_map(params![query, notebook, limit as i64], |r| {
            Ok(Passage {
                source_title: r.get(0)?,
                ordinal: r.get::<_, i64>(1)? as usize,
                text: r.get(2)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }
}

/// Words of the question as prefix terms. Long words are cut a little so that
/// "договору" also finds "договор" and "договоре": a cheap stand-in for stemming.
fn fts_query(question: &str) -> Option<String> {
    let terms: Vec<String> = question
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3)
        .map(|w| {
            let word = w.to_lowercase();
            let length = word.chars().count();
            let keep = if length > 5 { length - 2 } else { length };
            format!("\"{}\"*", word.chars().take(keep).collect::<String>())
        })
        .collect();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with_sources() -> (Store, i64) {
        let mut store = Store::in_memory().unwrap();
        let notebook = store.create_notebook("Диплом").unwrap();
        store
            .add_source(notebook, "договор.txt", "Срок действия договора составляет два года.\n\nОплата производится ежемесячно.")
            .unwrap();
        store
            .add_source(notebook, "заметки.txt", "Купить молоко и хлеб.\n\nПозвонить научному руководителю.")
            .unwrap();
        (store, notebook)
    }

    #[test]
    fn finds_the_passage_that_answers_the_question_even_in_another_word_form() {
        let (store, notebook) = store_with_sources();
        let found = store.search(notebook, "Какой срок у договору?", 3).unwrap();
        assert_eq!(found[0].source_title, "договор.txt");
        assert!(found[0].text.contains("два года"));
    }

    #[test]
    fn searches_only_the_chosen_notebook() {
        let (mut store, notebook) = store_with_sources();
        let other = store.create_notebook("Другое").unwrap();
        store.add_source(other, "чужое.txt", "Срок годности продукта: неделя.").unwrap();
        let found = store.search(notebook, "срок", 5).unwrap();
        assert!(found.iter().all(|p| p.source_title != "чужое.txt"));
    }

    #[test]
    fn adding_the_same_title_again_replaces_it_without_copies() {
        let (mut store, notebook) = store_with_sources();
        store
            .add_source(notebook, "договор.txt", "Срок действия договора составляет пять лет.")
            .unwrap();
        assert_eq!(store.source_count(notebook).unwrap(), 2);
        let found = store.search(notebook, "срок договора", 5).unwrap();
        let texts: Vec<&str> = found.iter().map(|p| p.text.as_str()).collect();
        assert!(texts.iter().any(|t| t.contains("пять лет")));
        assert!(texts.iter().all(|t| !t.contains("два года")));
    }

    #[test]
    fn returns_nothing_for_a_question_without_searchable_words() {
        let (store, notebook) = store_with_sources();
        assert!(store.search(notebook, "? !", 3).unwrap().is_empty());
        assert!(store.search(notebook, "квантовая хромодинамика", 3).unwrap().is_empty());
    }
}
