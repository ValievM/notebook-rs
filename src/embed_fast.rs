//! Real semantic embeddings, on this machine: fastembed (ONNX Runtime) with a multilingual E5
//! model that understands Russian. Built only with `--features fastembed`.
//!
//! The model is the 8-bit quantized E5-small (113 MB). The full-size file of the same model
//! (448 MB) needed about 3.8 GB of memory just to be loaded, and the window shared a 16 GB
//! machine with a browser, an editor and Docker; the quantized one needs a fraction of that.
//! Its files are downloaded once into the cache directory.
//!
//! A dynamically quantized model scales its numbers to the batch it is given, so the vector of
//! a text would depend on the texts it was computed together with. Every text is therefore
//! embedded on its own: a passage gets the same vector in any company, and a question is
//! computed exactly like a passage.

use crate::embed::{Embedder, normalize};
use crate::error::{Error, Result};
use fastembed::{
    InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

const REPO: &str = "Xenova/multilingual-e5-small";
const FOLDER: &str = "e5-small-q8";
const MODEL_FILE: &str = "onnx/model_quantized.onnx";
const TOKENIZER_FILE: &str = "tokenizer.json";
const CONFIG_FILE: &str = "config.json";
const SPECIAL_TOKENS_FILE: &str = "special_tokens_map.json";
const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
const FILES: [&str; 5] = [
    MODEL_FILE,
    TOKENIZER_FILE,
    CONFIG_FILE,
    SPECIAL_TOKENS_FILE,
    TOKENIZER_CONFIG_FILE,
];

pub struct FastEmbedder {
    // `TextEmbedding::embed` needs `&mut self`; the lock keeps the embedder shareable.
    model: Mutex<TextEmbedding>,
}

impl FastEmbedder {
    /// Loads the model, downloading its files into `cache_dir` if they are not there yet.
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        let directory = cache_dir.join(FOLDER);
        ensure_files(&directory, &HuggingFace)?;
        let read = |path: &str| {
            std::fs::read(directory.join(path)).map_err(|e| unavailable(format!("{path}: {e}")))
        };
        let tokenizer_files = TokenizerFiles {
            tokenizer_file: read(TOKENIZER_FILE)?,
            config_file: read(CONFIG_FILE)?,
            special_tokens_map_file: read(SPECIAL_TOKENS_FILE)?,
            tokenizer_config_file: read(TOKENIZER_CONFIG_FILE)?,
        };
        let model = UserDefinedEmbeddingModel::new(read(MODEL_FILE)?, tokenizer_files)
            .with_pooling(Pooling::Mean)
            .with_quantization(QuantizationMode::Dynamic);
        let model =
            TextEmbedding::try_new_from_user_defined(model, InitOptionsUserDefined::default())
                .map_err(unavailable)?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    /// One vector per text, each computed on its own.
    fn run(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut model = self
            .model
            .lock()
            .expect("the model lock is never held across a panic");
        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            let mut vector = model.embed(&[text], None).map_err(unavailable)?.remove(0);
            normalize(&mut vector);
            vectors.push(vector);
        }
        Ok(vectors)
    }
}

fn unavailable(error: impl std::fmt::Display) -> Error {
    Error::ModelUnavailable(format!("embedding model: {error}"))
}

impl Embedder for FastEmbedder {
    fn model(&self) -> &str {
        "fastembed-multilingual-e5-small-q8"
    }

    /// E5 models are trained with these markers: passages get "passage: ", questions "query: ".
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.run(texts.iter().map(|t| format!("passage: {t}")).collect())
    }

    fn embed_query(&self, question: &str) -> Result<Vec<f32>> {
        Ok(self.run(vec![format!("query: {question}")])?.remove(0))
    }
}

/// Where the model files come from. A trait so that the tests need no network.
trait Fetch {
    /// Writes the file `path` of the model repository to `to`.
    fn fetch(&self, path: &str, to: &Path) -> Result<()>;
}

struct HuggingFace;

impl Fetch for HuggingFace {
    fn fetch(&self, path: &str, to: &Path) -> Result<()> {
        eprintln!("notebook-rs: downloading {path} of the search model");
        let url = format!("https://huggingface.co/{REPO}/resolve/main/{path}");
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(1800)))
            .build()
            .into();
        let mut response = agent
            .get(&url)
            .call()
            .map_err(|e| unavailable(format!("{url}: {e}")))?;
        let mut reader = response
            .body_mut()
            .with_config()
            .limit(1_000_000_000)
            .reader();
        let mut file = std::fs::File::create(to)?;
        std::io::copy(&mut reader, &mut file).map_err(|e| unavailable(format!("{url}: {e}")))?;
        Ok(())
    }
}

/// Makes sure every model file is in `directory`, fetching the missing ones. A file is written
/// under a temporary name and renamed when complete, so an interrupted download never leaves
/// a half-written file that would be taken for the real one next time.
fn ensure_files(directory: &Path, fetch: &dyn Fetch) -> Result<()> {
    for path in FILES {
        let target = directory.join(path);
        if target.exists() {
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let partial = target.with_file_name(format!(
            "{}.part",
            target.file_name().unwrap_or_default().to_string_lossy()
        ));
        match fetch.fetch(path, &partial) {
            Ok(()) => std::fs::rename(&partial, &target)?,
            Err(error) => {
                let _ = std::fs::remove_file(&partial);
                return Err(error);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::cosine;

    /// Writes the name of the file as its content and remembers what was asked for. A file
    /// named in `fail_on` is written halfway and then the "network" breaks.
    struct Fake {
        asked: Mutex<Vec<String>>,
        fail_on: Option<&'static str>,
    }

    impl Fake {
        fn new(fail_on: Option<&'static str>) -> Self {
            Self {
                asked: Mutex::new(Vec::new()),
                fail_on,
            }
        }

        fn asked(&self) -> Vec<String> {
            self.asked.lock().unwrap().clone()
        }
    }

    impl Fetch for Fake {
        fn fetch(&self, path: &str, to: &Path) -> Result<()> {
            self.asked.lock().unwrap().push(path.to_string());
            if self.fail_on == Some(path) {
                std::fs::write(to, "half")?;
                return Err(unavailable("the network broke"));
            }
            std::fs::write(to, path)?;
            Ok(())
        }
    }

    #[test]
    fn every_model_file_is_fetched_once_and_lands_where_it_is_read_from() {
        let directory = tempfile::tempdir().unwrap();
        let fake = Fake::new(None);

        ensure_files(directory.path(), &fake).unwrap();

        assert_eq!(fake.asked().len(), 5);
        for path in FILES {
            assert_eq!(
                std::fs::read_to_string(directory.path().join(path)).unwrap(),
                path
            );
        }
        ensure_files(directory.path(), &fake).unwrap();
        assert_eq!(fake.asked().len(), 5, "nothing is fetched twice");
    }

    #[test]
    fn only_the_missing_files_are_fetched() {
        let directory = tempfile::tempdir().unwrap();
        ensure_files(directory.path(), &Fake::new(None)).unwrap();
        std::fs::remove_file(directory.path().join(CONFIG_FILE)).unwrap();

        let fake = Fake::new(None);
        ensure_files(directory.path(), &fake).unwrap();

        assert_eq!(fake.asked(), vec![CONFIG_FILE]);
    }

    #[test]
    fn a_broken_download_leaves_no_half_written_file_to_be_mistaken_for_the_real_one() {
        let directory = tempfile::tempdir().unwrap();

        let error = ensure_files(directory.path(), &Fake::new(Some(TOKENIZER_FILE))).unwrap_err();

        assert!(error.to_string().contains("network broke"), "{error}");
        assert!(
            directory.path().join(MODEL_FILE).exists(),
            "files finished before it stay"
        );
        assert!(!directory.path().join(TOKENIZER_FILE).exists());
        assert!(!directory.path().join("tokenizer.json.part").exists());
        // The next start fetches only what is missing.
        let fake = Fake::new(None);
        ensure_files(directory.path(), &fake).unwrap();
        assert_eq!(
            fake.asked(),
            vec![
                TOKENIZER_FILE,
                CONFIG_FILE,
                SPECIAL_TOKENS_FILE,
                TOKENIZER_CONFIG_FILE
            ]
        );
    }

    /// Downloads the model (113 MB) on the first run, so it is skipped by default:
    /// `cargo test --features fastembed -- --ignored`.
    #[test]
    #[ignore = "downloads the embedding model"]
    fn understands_meaning_not_just_shared_words() {
        let cache = std::env::temp_dir().join("notebook-rs-model-cache");
        let embedder = FastEmbedder::new(cache).unwrap();
        let passages = embedder
            .embed(&[
                "Арендатор обязан освободить помещение за тридцать дней до окончания срока.".into(),
                "Купить молоко и хлеб в магазине у дома.".into(),
            ])
            .unwrap();
        let question = embedder
            .embed_query("За сколько нужно съехать из квартиры?")
            .unwrap();
        let about_lease = cosine(&question, &passages[0]);
        let about_shopping = cosine(&question, &passages[1]);
        assert!(
            about_lease > about_shopping,
            "lease {about_lease} vs shopping {about_shopping}"
        );
    }

    /// A passage must get the same vector whatever it is computed together with.
    #[test]
    #[ignore = "downloads the embedding model"]
    fn a_text_gets_the_same_vector_alone_and_in_company() {
        let cache = std::env::temp_dir().join("notebook-rs-model-cache");
        let embedder = FastEmbedder::new(cache).unwrap();
        let text = "Срок действия договора составляет два года.".to_string();
        let alone = embedder
            .embed(std::slice::from_ref(&text))
            .unwrap()
            .remove(0);
        let company = embedder
            .embed(&["Совсем другой текст про погоду и море.".into(), text])
            .unwrap()
            .remove(1);
        assert!(
            cosine(&alone, &company) > 0.9999,
            "{}",
            cosine(&alone, &company)
        );
    }
}
