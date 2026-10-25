//! Notebooks, sources and their passages in one SQLite file: text searched with FTS5 (BM25),
//! vectors kept next to the passages. This is the only module that knows SQL.

use crate::domain::{NotebookId, Passage, PassageId};
use crate::error::Result;
use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

/// A passage on its way in: text, and the vector if one was made.
pub struct NewPassage {
    pub ordinal: usize,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
}

/// A notebook with the number of its sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotebookInfo {
    pub id: NotebookId,
    pub name: String,
    pub sources: usize,
}

/// One message of the dialog of a notebook, as it was kept. `payload` is opaque text for the
/// caller (the citations of an answer, a map...): this module only keeps it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredMessage {
    pub id: i64,
    /// `user` or `assistant`.
    pub role: String,
    /// What it was: `ask`, `summarize`, `compare`...
    pub kind: String,
    pub text: String,
    pub payload: String,
}

/// A source with the number of its passages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceInfo {
    pub title: String,
    pub passages: usize,
}

/// The vectors of one notebook for one model, shared between readers without copying.
pub type EmbeddedPassages = Arc<Vec<(Passage, Vec<f32>)>>;

/// Loaded vectors, so that a question does not read and decode them from the file again.
/// `generation` counts the writes; a loader may store its result only if no write happened
/// while it was loading (otherwise it would put an outdated copy into the cache).
#[derive(Default)]
struct VectorCache {
    generation: u64,
    entries: HashMap<(NotebookId, String), EmbeddedPassages>,
}

/// A repository over one SQLite connection.
///
/// Two primitives, two jobs. The `Mutex` serializes the connection: SQLite takes one statement
/// at a time on a connection, and a writer must not interleave with a reader. The `RwLock`
/// guards the vector cache, which is read by every question and cleared only by a write: many
/// readers at once, one writer. No method holds one lock while taking the other.
pub struct Store {
    db: Mutex<Connection>,
    vectors: RwLock<VectorCache>,
}

const SCHEMA_VERSION: i32 = 1;
const CACHE_LOCK: &str = "the cache lock is never held across a panic";

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_connection(Connection::open(path)?)
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
             CREATE TABLE IF NOT EXISTS message (
               id INTEGER PRIMARY KEY,
               notebook_id INTEGER NOT NULL REFERENCES notebook(id),
               role TEXT NOT NULL,
               kind TEXT NOT NULL,
               text TEXT NOT NULL,
               payload TEXT NOT NULL DEFAULT '');
             CREATE INDEX IF NOT EXISTS message_by_notebook ON message(notebook_id, id);
             CREATE VIRTUAL TABLE IF NOT EXISTS passage_index USING fts5(
               text, content='passage', content_rowid='id', tokenize='unicode61 remove_diacritics 2');",
        )?;
        migrate(&db)?;
        Ok(Self {
            db: Mutex::new(db),
            vectors: RwLock::default(),
        })
    }

    fn db(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.db
            .lock()
            .expect("the database lock is never held across a panic")
    }

    /// Finds the notebook by name or creates it.
    pub fn create_notebook(&self, name: &str) -> Result<NotebookId> {
        let db = self.db();
        db.execute("INSERT OR IGNORE INTO notebook(name) VALUES (?1)", [name])?;
        Ok(NotebookId(db.query_row(
            "SELECT id FROM notebook WHERE name = ?1",
            [name],
            |r| r.get(0),
        )?))
    }

    /// Adds a source, or replaces it if one with the same title is already there (no copies).
    /// `model` names the embedding model of the vectors in `passages`.
    pub fn add_source(
        &self,
        notebook: NotebookId,
        title: &str,
        passages: &[NewPassage],
        model: &str,
    ) -> Result<usize> {
        let count = self.write_source(notebook, title, passages, model)?;
        self.forget_vectors();
        Ok(count)
    }

    fn write_source(
        &self,
        notebook: NotebookId,
        title: &str,
        passages: &[NewPassage],
        model: &str,
    ) -> Result<usize> {
        let db = self.db();
        let tx = db.unchecked_transaction()?;
        let old: Option<i64> = tx
            .query_row(
                "SELECT id FROM source WHERE notebook_id = ?1 AND title = ?2",
                params![notebook.0, title],
                |r| r.get(0),
            )
            .ok();
        if let Some(id) = old {
            remove_source_rows(&tx, id)?;
        }
        tx.execute(
            "INSERT INTO source(notebook_id, title) VALUES (?1, ?2)",
            params![notebook.0, title],
        )?;
        let source_id = tx.last_insert_rowid();
        for passage in passages {
            let blob = passage.embedding.as_deref().map(encode);
            tx.execute(
                "INSERT INTO passage(source_id, ordinal, text, embedding, embedding_model)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    source_id,
                    passage.ordinal as i64,
                    passage.text,
                    blob,
                    blob.as_ref().map(|_| model)
                ],
            )?;
            let id = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO passage_index(rowid, text) VALUES (?1, ?2)",
                params![id, passage.text],
            )?;
        }
        tx.commit()?;
        Ok(passages.len())
    }

    /// Every notebook with its number of sources, in the order they were created.
    pub fn list_notebooks(&self) -> Result<Vec<NotebookInfo>> {
        let db = self.db();
        let mut statement = db.prepare(
            "SELECT n.id, n.name, (SELECT COUNT(*) FROM source s WHERE s.notebook_id = n.id)
             FROM notebook n ORDER BY n.id",
        )?;
        let rows = statement.query_map([], |r| {
            Ok(NotebookInfo {
                id: NotebookId(r.get(0)?),
                name: r.get(1)?,
                sources: r.get::<_, i64>(2)? as usize,
            })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// The sources of the notebook with their number of passages, in the order they were added.
    pub fn source_infos(&self, notebook: NotebookId) -> Result<Vec<SourceInfo>> {
        let db = self.db();
        let mut statement = db.prepare(
            "SELECT s.title, (SELECT COUNT(*) FROM passage p WHERE p.source_id = s.id)
             FROM source s WHERE s.notebook_id = ?1 ORDER BY s.id",
        )?;
        let rows = statement.query_map([notebook.0], |r| {
            Ok(SourceInfo {
                title: r.get(0)?,
                passages: r.get::<_, i64>(1)? as usize,
            })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Removes a source with its passages. Returns whether there was such a source.
    pub fn delete_source(&self, notebook: NotebookId, title: &str) -> Result<bool> {
        let removed = {
            let db = self.db();
            let tx = db.unchecked_transaction()?;
            let id: Option<i64> = tx
                .query_row(
                    "SELECT id FROM source WHERE notebook_id = ?1 AND title = ?2",
                    params![notebook.0, title],
                    |r| r.get(0),
                )
                .ok();
            if let Some(id) = id {
                remove_source_rows(&tx, id)?;
            }
            tx.commit()?;
            id.is_some()
        };
        if removed {
            self.forget_vectors();
        }
        Ok(removed)
    }

    /// Appends a message to the dialog of the notebook.
    pub fn add_message(
        &self,
        notebook: NotebookId,
        role: &str,
        kind: &str,
        text: &str,
        payload: &str,
    ) -> Result<i64> {
        let db = self.db();
        db.execute(
            "INSERT INTO message(notebook_id, role, kind, text, payload) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![notebook.0, role, kind, text, payload],
        )?;
        Ok(db.last_insert_rowid())
    }

    /// The last `limit` messages of the notebook, oldest first.
    pub fn messages(&self, notebook: NotebookId, limit: usize) -> Result<Vec<StoredMessage>> {
        let db = self.db();
        let mut statement = db.prepare(
            "SELECT id, role, kind, text, payload FROM message
             WHERE notebook_id = ?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = statement.query_map(params![notebook.0, limit as i64], |r| {
            Ok(StoredMessage {
                id: r.get(0)?,
                role: r.get(1)?,
                kind: r.get(2)?,
                text: r.get(3)?,
                payload: r.get(4)?,
            })
        })?;
        let mut messages: Vec<StoredMessage> = rows.collect::<std::result::Result<_, _>>()?;
        messages.reverse();
        Ok(messages)
    }

    /// Forgets the dialog of the notebook (other notebooks keep theirs).
    pub fn clear_messages(&self, notebook: NotebookId) -> Result<()> {
        self.db()
            .execute("DELETE FROM message WHERE notebook_id = ?1", [notebook.0])?;
        Ok(())
    }

    pub fn source_count(&self, notebook: NotebookId) -> Result<usize> {
        Ok(self.db().query_row(
            "SELECT COUNT(*) FROM source WHERE notebook_id = ?1",
            [notebook.0],
            |r| r.get::<_, i64>(0),
        )? as usize)
    }

    /// Titles of the notebook's sources, in the order they were added.
    pub fn source_titles(&self, notebook: NotebookId) -> Result<Vec<String>> {
        let db = self.db();
        let mut statement =
            db.prepare("SELECT title FROM source WHERE notebook_id = ?1 ORDER BY id")?;
        let rows = statement.query_map([notebook.0], |r| r.get(0))?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// The passages of one source (or of every source if `title` is `None`) in reading order:
    /// sources in the order they were added, passages by position.
    pub fn source_passages(
        &self,
        notebook: NotebookId,
        title: Option<&str>,
    ) -> Result<Vec<Passage>> {
        let db = self.db();
        let mut statement = db.prepare(
            "SELECT p.id, s.title, p.ordinal, p.text
             FROM passage p JOIN source s ON s.id = p.source_id
             WHERE s.notebook_id = ?1 AND (?2 IS NULL OR s.title = ?2)
             ORDER BY s.id, p.ordinal",
        )?;
        let rows = statement.query_map(params![notebook.0, title], passage_from_row)?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Passages matching an FTS5 query, best first. Only this notebook is searched.
    pub fn search_text(
        &self,
        notebook: NotebookId,
        fts_query: &str,
        limit: usize,
    ) -> Result<Vec<Passage>> {
        let db = self.db();
        let mut statement = db.prepare(
            "SELECT p.id, s.title, p.ordinal, p.text
             FROM passage_index
             JOIN passage p ON p.id = passage_index.rowid
             JOIN source s ON s.id = p.source_id
             WHERE passage_index MATCH ?1 AND s.notebook_id = ?2
             ORDER BY bm25(passage_index) LIMIT ?3",
        )?;
        let rows = statement.query_map(
            params![fts_query, notebook.0, limit as i64],
            passage_from_row,
        )?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Every passage of the notebook that has a vector from `model`.
    pub fn embedded_passages(&self, notebook: NotebookId, model: &str) -> Result<EmbeddedPassages> {
        self.embedded_passages_with(notebook, model, || {})
    }

    /// `after_load` runs between loading from the file and storing in the cache: the place
    /// where a concurrent write can slip in. Tests use it to make that race happen on purpose.
    fn embedded_passages_with(
        &self,
        notebook: NotebookId,
        model: &str,
        after_load: impl FnOnce(),
    ) -> Result<EmbeddedPassages> {
        let key = (notebook, model.to_string());
        let seen_generation = {
            let cache = self.vectors.read().expect(CACHE_LOCK);
            if let Some(hit) = cache.entries.get(&key) {
                return Ok(hit.clone());
            }
            cache.generation
        };
        // Neither lock is held while the file is read. Two readers that miss at the same time
        // both load; the second store just replaces the first with an identical copy.
        let loaded: EmbeddedPassages = Arc::new(self.load_embedded(notebook, model)?);
        after_load();
        let mut cache = self.vectors.write().expect(CACHE_LOCK);
        if cache.generation == seen_generation {
            cache.entries.insert(key, loaded.clone());
        }
        Ok(loaded)
    }

    /// Called after every committed write: loaded copies may be outdated now.
    fn forget_vectors(&self) {
        let mut cache = self.vectors.write().expect(CACHE_LOCK);
        cache.generation += 1;
        cache.entries.clear();
    }

    fn load_embedded(&self, notebook: NotebookId, model: &str) -> Result<Vec<(Passage, Vec<f32>)>> {
        let db = self.db();
        let mut statement = db.prepare(
            "SELECT p.id, s.title, p.ordinal, p.text, p.embedding
             FROM passage p JOIN source s ON s.id = p.source_id
             WHERE s.notebook_id = ?1 AND p.embedding_model = ?2",
        )?;
        let rows = statement.query_map(params![notebook.0, model], |r| {
            Ok((passage_from_row(r)?, decode(&r.get::<_, Vec<u8>>(4)?)))
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }
}

/// Deletes a source, its passages and their entries in the text index.
fn remove_source_rows(tx: &rusqlite::Transaction<'_>, source_id: i64) -> rusqlite::Result<()> {
    tx.execute(
        "INSERT INTO passage_index(passage_index, rowid, text)
         SELECT 'delete', id, text FROM passage WHERE source_id = ?1",
        [source_id],
    )?;
    tx.execute("DELETE FROM passage WHERE source_id = ?1", [source_id])?;
    tx.execute("DELETE FROM source WHERE id = ?1", [source_id])?;
    Ok(())
}

fn passage_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Passage> {
    Ok(Passage {
        id: PassageId(r.get(0)?),
        source_title: r.get(1)?,
        ordinal: r.get::<_, i64>(2)? as usize,
        text: r.get(3)?,
    })
}

/// Files made by the first prototype have no vector columns: add them, keep the data.
fn migrate(db: &Connection) -> Result<()> {
    let version: i32 = db.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version < 1 {
        let has_vectors = db
            .prepare("SELECT 1 FROM pragma_table_info('passage') WHERE name = 'embedding'")?
            .exists([])?;
        if !has_vectors {
            db.execute_batch(
                "ALTER TABLE passage ADD COLUMN embedding BLOB;
                 ALTER TABLE passage ADD COLUMN embedding_model TEXT;",
            )?;
        }
    }
    db.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    Ok(())
}

fn encode(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passage(ordinal: usize, text: &str, embedding: Option<Vec<f32>>) -> NewPassage {
        NewPassage {
            ordinal,
            text: text.into(),
            embedding,
        }
    }

    #[test]
    fn a_dialog_is_kept_per_notebook_in_order_and_the_last_messages_are_returned() {
        let store = Store::in_memory().unwrap();
        let a = store.create_notebook("a").unwrap();
        let b = store.create_notebook("b").unwrap();
        store.add_message(a, "user", "ask", "первый", "").unwrap();
        store
            .add_message(a, "assistant", "ask", "ответ", r#"{"x":1}"#)
            .unwrap();
        store.add_message(a, "user", "ask", "второй", "").unwrap();
        store.add_message(b, "user", "ask", "чужой", "").unwrap();

        let last_two = store.messages(a, 2).unwrap();
        let texts: Vec<&str> = last_two.iter().map(|m| m.text.as_str()).collect();
        assert_eq!(texts, ["ответ", "второй"]);
        assert_eq!(last_two[0].payload, r#"{"x":1}"#);
        assert_eq!(
            (last_two[0].role.as_str(), last_two[0].kind.as_str()),
            ("assistant", "ask")
        );

        store.clear_messages(a).unwrap();
        assert!(store.messages(a, 10).unwrap().is_empty());
        assert_eq!(store.messages(b, 10).unwrap().len(), 1);
    }

    #[test]
    fn finds_text_only_in_the_chosen_notebook() {
        let store = Store::in_memory().unwrap();
        let mine = store.create_notebook("Диплом").unwrap();
        let other = store.create_notebook("Другое").unwrap();
        store
            .add_source(
                mine,
                "a.txt",
                &[passage(0, "Срок действия договора два года", None)],
                "m",
            )
            .unwrap();
        store
            .add_source(
                other,
                "b.txt",
                &[passage(0, "Срок годности неделя", None)],
                "m",
            )
            .unwrap();
        let found = store.search_text(mine, "\"срок\"*", 5).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].source_title, "a.txt");
    }

    #[test]
    fn the_same_title_replaces_the_source_without_copies() {
        let store = Store::in_memory().unwrap();
        let nb = store.create_notebook("n").unwrap();
        store
            .add_source(nb, "a.txt", &[passage(0, "два года", None)], "m")
            .unwrap();
        store
            .add_source(nb, "a.txt", &[passage(0, "пять лет", None)], "m")
            .unwrap();
        assert_eq!(store.source_count(nb).unwrap(), 1);
        assert!(store.search_text(nb, "\"два\"*", 5).unwrap().is_empty());
        assert_eq!(store.search_text(nb, "\"пять\"*", 5).unwrap().len(), 1);
    }

    #[test]
    fn vectors_round_trip_and_are_returned_only_for_their_own_model() {
        let store = Store::in_memory().unwrap();
        let nb = store.create_notebook("n").unwrap();
        store
            .add_source(
                nb,
                "a.txt",
                &[passage(0, "текст", Some(vec![0.25, -1.5]))],
                "model-a",
            )
            .unwrap();
        let found = store.embedded_passages(nb, "model-a").unwrap();
        assert_eq!(found[0].1, vec![0.25, -1.5]);
        assert!(store.embedded_passages(nb, "model-b").unwrap().is_empty());
    }

    fn notebook_with_two_sources() -> (Store, NotebookId) {
        let store = Store::in_memory().unwrap();
        let nb = store.create_notebook("n").unwrap();
        store
            .add_source(
                nb,
                "a.txt",
                &[passage(0, "a-ноль", None), passage(1, "a-один", None)],
                "m",
            )
            .unwrap();
        store
            .add_source(nb, "b.txt", &[passage(0, "b-ноль", None)], "m")
            .unwrap();
        (store, nb)
    }

    #[test]
    fn lists_source_titles_in_the_order_they_were_added() {
        let (store, nb) = notebook_with_two_sources();
        assert_eq!(store.source_titles(nb).unwrap(), vec!["a.txt", "b.txt"]);
    }

    #[test]
    fn passages_of_all_sources_come_in_reading_order() {
        let (store, nb) = notebook_with_two_sources();
        let texts: Vec<String> = store
            .source_passages(nb, None)
            .unwrap()
            .into_iter()
            .map(|p| p.text)
            .collect();
        assert_eq!(texts, vec!["a-ноль", "a-один", "b-ноль"]);
    }

    #[test]
    fn passages_of_one_source_are_only_its_own() {
        let (store, nb) = notebook_with_two_sources();
        let found = store.source_passages(nb, Some("b.txt")).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].source_title, "b.txt");
        assert!(
            store
                .source_passages(nb, Some("нет.txt"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn another_notebook_has_its_own_sources() {
        let (store, nb) = notebook_with_two_sources();
        let other = store.create_notebook("other").unwrap();
        assert!(store.source_titles(other).unwrap().is_empty());
        assert!(store.source_passages(other, None).unwrap().is_empty());
        assert_eq!(store.source_passages(nb, None).unwrap().len(), 3);
    }

    #[test]
    fn lists_notebooks_with_their_number_of_sources() {
        let (store, nb) = notebook_with_two_sources();
        let empty = store.create_notebook("пустой").unwrap();
        assert_eq!(
            store.list_notebooks().unwrap(),
            vec![
                NotebookInfo {
                    id: nb,
                    name: "n".into(),
                    sources: 2
                },
                NotebookInfo {
                    id: empty,
                    name: "пустой".into(),
                    sources: 0
                },
            ]
        );
    }

    #[test]
    fn lists_sources_with_their_number_of_passages() {
        let (store, nb) = notebook_with_two_sources();
        assert_eq!(
            store.source_infos(nb).unwrap(),
            vec![
                SourceInfo {
                    title: "a.txt".into(),
                    passages: 2
                },
                SourceInfo {
                    title: "b.txt".into(),
                    passages: 1
                },
            ]
        );
    }

    #[test]
    fn a_deleted_source_is_gone_from_listings_search_and_vectors() {
        let store = Store::in_memory().unwrap();
        let nb = store.create_notebook("n").unwrap();
        store
            .add_source(
                nb,
                "a.txt",
                &[passage(0, "красная лиса", Some(vec![1.0]))],
                "m",
            )
            .unwrap();
        store
            .add_source(
                nb,
                "b.txt",
                &[passage(0, "синяя лиса", Some(vec![1.0]))],
                "m",
            )
            .unwrap();
        let before = store.embedded_passages(nb, "m").unwrap();
        assert_eq!(before.len(), 2);

        assert!(store.delete_source(nb, "a.txt").unwrap());

        assert_eq!(store.source_titles(nb).unwrap(), vec!["b.txt"]);
        let found = store.search_text(nb, "\"лиса\"*", 5).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].source_title, "b.txt");
        assert_eq!(store.embedded_passages(nb, "m").unwrap().len(), 1);
    }

    #[test]
    fn deleting_an_unknown_source_or_one_of_another_notebook_changes_nothing() {
        let (store, nb) = notebook_with_two_sources();
        let other = store.create_notebook("other").unwrap();
        assert!(!store.delete_source(nb, "нет.txt").unwrap());
        assert!(!store.delete_source(other, "a.txt").unwrap());
        assert_eq!(store.source_titles(nb).unwrap(), vec!["a.txt", "b.txt"]);
    }

    fn with_vector(title: &str, store: &Store, nb: NotebookId) {
        store
            .add_source(nb, title, &[passage(0, title, Some(vec![1.0]))], "m")
            .unwrap();
    }

    #[test]
    fn repeated_reads_share_one_loaded_copy() {
        let store = Store::in_memory().unwrap();
        let nb = store.create_notebook("n").unwrap();
        with_vector("a.txt", &store, nb);
        let first = store.embedded_passages(nb, "m").unwrap();
        let second = store.embedded_passages(nb, "m").unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn another_notebook_or_model_has_its_own_entry() {
        let store = Store::in_memory().unwrap();
        let mine = store.create_notebook("n").unwrap();
        let other = store.create_notebook("o").unwrap();
        with_vector("a.txt", &store, mine);
        let here = store.embedded_passages(mine, "m").unwrap();
        assert!(!Arc::ptr_eq(
            &here,
            &store.embedded_passages(other, "m").unwrap()
        ));
        assert!(
            store
                .embedded_passages(mine, "other-model")
                .unwrap()
                .is_empty()
        );
        assert_eq!(here.len(), 1);
    }

    #[test]
    fn a_write_makes_readers_load_again() {
        let store = Store::in_memory().unwrap();
        let nb = store.create_notebook("n").unwrap();
        with_vector("a.txt", &store, nb);
        let before = store.embedded_passages(nb, "m").unwrap();
        with_vector("b.txt", &store, nb);
        let after = store.embedded_passages(nb, "m").unwrap();
        assert_eq!(before.len(), 1);
        assert_eq!(after.len(), 2);
        assert!(!Arc::ptr_eq(&before, &after));
    }

    #[test]
    fn a_copy_loaded_before_a_concurrent_write_is_not_kept_in_the_cache() {
        let store = Store::in_memory().unwrap();
        let nb = store.create_notebook("n").unwrap();
        with_vector("a.txt", &store, nb);
        // The reader has loaded one passage; before it can store the copy, a write happens.
        let stale = store
            .embedded_passages_with(nb, "m", || with_vector("b.txt", &store, nb))
            .unwrap();
        assert_eq!(stale.len(), 1);
        assert_eq!(store.embedded_passages(nb, "m").unwrap().len(), 2);
    }

    #[test]
    fn many_threads_writing_and_reading_always_see_their_own_writes() {
        let store = Store::in_memory().unwrap();
        let nb = store.create_notebook("n").unwrap();
        std::thread::scope(|scope| {
            for i in 0..8 {
                let store = &store;
                scope.spawn(move || {
                    let title = format!("source-{i}");
                    with_vector(&title, store, nb);
                    let seen = store.embedded_passages(nb, "m").unwrap();
                    assert!(seen.iter().any(|(p, _)| p.text == title));
                });
            }
        });
        assert_eq!(store.embedded_passages(nb, "m").unwrap().len(), 8);
    }

    #[test]
    fn a_file_from_the_first_prototype_is_upgraded_and_keeps_its_data() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("old.db");
        {
            let db = Connection::open(&path).unwrap();
            db.execute_batch(
                "CREATE TABLE notebook (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
                 CREATE TABLE source (id INTEGER PRIMARY KEY, notebook_id INTEGER NOT NULL, title TEXT NOT NULL, UNIQUE (notebook_id, title));
                 CREATE TABLE passage (id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL);
                 INSERT INTO notebook VALUES (1, 'старый');",
            )
            .unwrap();
        }
        let store = Store::open(&path).unwrap();
        let nb = store.create_notebook("старый").unwrap();
        assert_eq!(nb, NotebookId(1));
        store
            .add_source(nb, "a.txt", &[passage(0, "текст", Some(vec![1.0]))], "m")
            .unwrap();
    }
}
