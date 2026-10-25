//! The desktop window of notebook-rs: a thin layer between the page and `notebook_rs`. Every
//! command takes the engine, does its work on a worker thread (a model call takes seconds and
//! must never freeze the window) and returns plain data (see `dto`).

mod dto;

use dto::{AnswerDto, NotebookDto, PassageViewDto, SourceDto, StatusDto, user_message};
use notebook_rs::domain::NotebookId;
use notebook_rs::embed_fast::FastEmbedder;
use notebook_rs::error::Result as NotebookResult;
use notebook_rs::llm::CodexExec;
use notebook_rs::service::Notebooks;
use notebook_rs::source::Sources;
use notebook_rs::store::Store;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tauri::{Manager, State};

/// The notebooks and the embedding model, built in the background: the first start downloads
/// the model, and the window must open at once and say so instead of hanging.
#[derive(Default)]
struct Engine {
    slot: OnceLock<Result<Arc<Notebooks>, String>>,
}

impl Engine {
    fn get(&self) -> Result<Arc<Notebooks>, String> {
        match self.slot.get() {
            None => Err("Поиск ещё готовится: при первом запуске скачивается модель (около 120 МБ). Подождите немного.".into()),
            Some(Ok(engine)) => Ok(engine.clone()),
            Some(Err(message)) => Err(message.clone()),
        }
    }
}

/// The same folder as the command line tool, so both see the same notebooks.
fn start_engine() -> Result<Arc<Notebooks>, String> {
    let home = std::env::var_os("HOME").ok_or("Не найдена домашняя папка.")?;
    let directory = PathBuf::from(home).join(".notebook-rs");
    std::fs::create_dir_all(&directory).map_err(|e| format!("{}: {e}", directory.display()))?;
    let store = Store::open(&directory.join("notebooks.db")).map_err(|e| user_message(&e))?;
    let embedder = FastEmbedder::new(directory.join("models")).map_err(|e| user_message(&e))?;
    Ok(Arc::new(Notebooks::standard(
        store,
        Arc::new(embedder),
        Box::new(CodexExec::default()),
    )))
}

/// Runs blocking work off the window's thread and turns its error into text for the person.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> NotebookResult<T> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(work)
        .await
        .map_err(|e| format!("Внутренняя ошибка: {e}"))?
        .map_err(|e| user_message(&e))
}

/// A notebook name as the person typed it: trimmed, and not empty.
fn clean_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Название блокнота не может быть пустым.".into());
    }
    Ok(name.to_string())
}

#[tauri::command]
fn status(engine: State<'_, Engine>) -> StatusDto {
    match engine.slot.get() {
        None => StatusDto {
            ready: false,
            error: None,
        },
        Some(Ok(_)) => StatusDto {
            ready: true,
            error: None,
        },
        Some(Err(message)) => StatusDto {
            ready: false,
            error: Some(message.clone()),
        },
    }
}

#[tauri::command]
async fn list_notebooks(engine: State<'_, Engine>) -> Result<Vec<NotebookDto>, String> {
    let notebooks = engine.get()?;
    blocking(move || {
        Ok(notebooks
            .list_notebooks()?
            .into_iter()
            .map(Into::into)
            .collect())
    })
    .await
}

#[tauri::command]
async fn create_notebook(engine: State<'_, Engine>, name: String) -> Result<NotebookDto, String> {
    let notebooks = engine.get()?;
    let name = clean_name(&name)?;
    blocking(move || {
        let id = notebooks.notebook(&name)?;
        let info = notebooks
            .list_notebooks()?
            .into_iter()
            .find(|info| info.id == id)
            .expect("a notebook that was just created is listed");
        Ok(info.into())
    })
    .await
}

#[tauri::command]
async fn list_sources(
    engine: State<'_, Engine>,
    notebook_id: i64,
) -> Result<Vec<SourceDto>, String> {
    let notebooks = engine.get()?;
    blocking(move || {
        Ok(notebooks
            .sources(NotebookId(notebook_id))?
            .into_iter()
            .map(Into::into)
            .collect())
    })
    .await
}

/// `location` is a file path (text, Markdown, PDF) or a web address.
#[tauri::command]
async fn add_source(
    engine: State<'_, Engine>,
    notebook_id: i64,
    location: String,
) -> Result<usize, String> {
    let notebooks = engine.get()?;
    blocking(move || {
        let document = Sources::standard().read(location.trim())?;
        notebooks.add_source(NotebookId(notebook_id), &document.title, &document.text)
    })
    .await
}

#[tauri::command]
async fn delete_source(
    engine: State<'_, Engine>,
    notebook_id: i64,
    title: String,
) -> Result<(), String> {
    let notebooks = engine.get()?;
    blocking(move || notebooks.delete_source(NotebookId(notebook_id), &title)).await
}

#[tauri::command]
async fn ask(
    engine: State<'_, Engine>,
    notebook_id: i64,
    question: String,
    deep: bool,
) -> Result<AnswerDto, String> {
    let notebooks = engine.get()?;
    blocking(move || {
        let id = NotebookId(notebook_id);
        let answer = if deep {
            notebooks.ask_deep(id, &question)?
        } else {
            notebooks.ask(id, &question)?
        };
        Ok(answer.into())
    })
    .await
}

#[tauri::command]
async fn summarize(
    engine: State<'_, Engine>,
    notebook_id: i64,
    source: Option<String>,
) -> Result<AnswerDto, String> {
    let notebooks = engine.get()?;
    blocking(move || {
        Ok(notebooks
            .summarize(NotebookId(notebook_id), source.as_deref())?
            .into())
    })
    .await
}

#[tauri::command]
async fn compare(
    engine: State<'_, Engine>,
    notebook_id: i64,
    first: String,
    second: String,
) -> Result<AnswerDto, String> {
    let notebooks = engine.get()?;
    blocking(move || {
        Ok(notebooks
            .compare(NotebookId(notebook_id), &first, &second)?
            .into())
    })
    .await
}

/// A source laid out for reading; `ordinal` and `quote` mark a cited place.
#[tauri::command]
async fn source_view(
    engine: State<'_, Engine>,
    notebook_id: i64,
    title: String,
    ordinal: Option<usize>,
    quote: Option<String>,
) -> Result<Vec<PassageViewDto>, String> {
    let notebooks = engine.get()?;
    blocking(move || {
        let cited = ordinal.zip(quote.as_deref());
        let view = notebooks.source_view(NotebookId(notebook_id), &title, cited)?;
        Ok(view.into_iter().map(Into::into).collect())
    })
    .await
}

/// The commands the page may call. The app and the tests build their windows from this.
fn configure<R: tauri::Runtime>(builder: tauri::Builder<R>) -> tauri::Builder<R> {
    builder.invoke_handler(tauri::generate_handler![
        status,
        list_notebooks,
        create_notebook,
        list_sources,
        add_source,
        delete_source,
        ask,
        summarize,
        compare,
        source_view
    ])
}

pub fn run() {
    configure(tauri::Builder::default())
        .plugin(tauri_plugin_dialog::init())
        .manage(Engine::default())
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn_blocking(move || {
                let result = start_engine();
                match &result {
                    Ok(_) => eprintln!("notebook-rs: the engine is ready"),
                    Err(message) => eprintln!("notebook-rs: the engine did not start: {message}"),
                }
                let _ = handle.state::<Engine>().slot.set(result);
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("the window could not start");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_notebook_name_is_trimmed_and_may_not_be_empty() {
        assert_eq!(clean_name("  Диплом \n").unwrap(), "Диплом");
        assert!(clean_name("   ").is_err());
        assert!(clean_name("").is_err());
    }

    #[test]
    fn before_the_engine_is_built_commands_say_what_is_going_on() {
        let engine = Engine::default();
        let message = engine.get().err().unwrap();
        assert!(message.contains("готовится"), "{message}");
    }

    #[test]
    fn a_failed_start_is_reported_as_it_is() {
        let engine = Engine::default();
        engine.slot.set(Err("нет места на диске".into())).ok();
        assert_eq!(engine.get().err().unwrap(), "нет места на диске");
    }
}

/// The commands called the way the page calls them: through the real IPC of Tauri (in a
/// runtime without a window), with JSON arguments and answers.
#[cfg(test)]
mod ipc_tests {
    use super::*;
    use notebook_rs::embed::HashingEmbedder;
    use notebook_rs::error::Result as NotebookResult;
    use notebook_rs::llm::Llm;
    use serde_json::{Value, json};
    use tauri::test::{INVOKE_KEY, MockRuntime, mock_builder, mock_context, noop_assets};
    use tauri::webview::InvokeRequest;
    use tauri::{WebviewWindow, WebviewWindowBuilder};

    /// Answers the same to everything and counts the calls.
    struct Fixed(Arc<std::sync::atomic::AtomicUsize>);
    impl Llm for Fixed {
        fn complete(&self, _prompt: &str) -> NotebookResult<String> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(r#"{"answer":"Два года [1].","citations":[{"n":1,"quote":"два года"}]}"#.into())
        }
    }

    fn window(engine: Engine) -> WebviewWindow<MockRuntime> {
        let app = configure(mock_builder())
            .manage(engine)
            .build(mock_context(noop_assets()))
            .expect("the test app builds");
        // The window keeps the app alive for the length of the test.
        let window = WebviewWindowBuilder::new(&app, "main", Default::default())
            .build()
            .expect("the test window builds");
        std::mem::forget(app);
        window
    }

    fn ready_engine() -> Engine {
        ready_engine_counting().0
    }

    fn ready_engine_counting() -> (Engine, Arc<std::sync::atomic::AtomicUsize>) {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let engine = Engine::default();
        let notebooks = Notebooks::standard(
            Store::in_memory().unwrap(),
            Arc::new(HashingEmbedder::default()),
            Box::new(Fixed(calls.clone())),
        );
        engine.slot.set(Ok(Arc::new(notebooks))).ok();
        (engine, calls)
    }

    fn call(
        window: &WebviewWindow<MockRuntime>,
        command: &str,
        arguments: Value,
    ) -> Result<Value, Value> {
        tauri::test::get_ipc_response(
            window,
            InvokeRequest {
                cmd: command.into(),
                callback: tauri::ipc::CallbackFn(0),
                error: tauri::ipc::CallbackFn(1),
                // The address of the app's own page. Anything else counts as remote content, and Tauri
                // refuses to let remote content call custom commands (a protection to keep on).
                url: "tauri://localhost".parse().unwrap(),
                body: tauri::ipc::InvokeBody::Json(arguments),
                headers: Default::default(),
                invoke_key: INVOKE_KEY.to_string(),
            },
        )
        .map(|body| body.deserialize::<Value>().unwrap())
    }

    #[test]
    fn a_notebook_is_created_trimmed_and_listed() {
        let window = window(ready_engine());
        let created = call(&window, "create_notebook", json!({"name": "  Диплом "})).unwrap();
        assert_eq!(created["name"], "Диплом");
        assert_eq!(created["sources"], 0);
        let listed = call(&window, "list_notebooks", json!({})).unwrap();
        assert_eq!(listed, json!([created]));
    }

    #[test]
    fn a_source_goes_in_a_question_comes_back_cited_and_the_citation_opens_the_source() {
        let window = window(ready_engine());
        let id = call(&window, "create_notebook", json!({"name": "Аренда"})).unwrap()["id"].clone();
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("договор.txt");
        std::fs::write(&file, "Срок действия договора составляет два года.").unwrap();

        let added = call(
            &window,
            "add_source",
            json!({"notebookId": id, "location": file.to_str().unwrap()}),
        );
        assert_eq!(added.unwrap(), json!(1));
        let sources = call(&window, "list_sources", json!({"notebookId": id})).unwrap();
        assert_eq!(sources, json!([{"title": "договор.txt", "passages": 1}]));

        let answer = call(
            &window,
            "ask",
            json!({"notebookId": id, "question": "Какой срок?", "deep": false}),
        )
        .unwrap();
        assert_eq!(answer["text"], "Два года [1].");
        assert_eq!(answer["rejectedQuotes"], 0);
        let citation = &answer["citations"][0];
        assert_eq!(citation["sourceTitle"], "договор.txt");

        let view = call(
            &window,
            "source_view",
            json!({"notebookId": id, "title": citation["sourceTitle"], "ordinal": citation["ordinal"], "quote": citation["quote"]}),
        )
        .unwrap();
        assert_eq!(view[0]["hit"], "два года");
        assert_eq!(view[0]["before"], "Срок действия договора составляет ");

        call(
            &window,
            "delete_source",
            json!({"notebookId": id, "title": "договор.txt"}),
        )
        .unwrap();
        assert_eq!(
            call(&window, "list_sources", json!({"notebookId": id})).unwrap(),
            json!([])
        );
    }

    #[test]
    fn a_deep_question_takes_two_model_calls_and_a_plain_one_takes_one() {
        let (engine, calls) = ready_engine_counting();
        let window = window(engine);
        let id = call(&window, "create_notebook", json!({"name": "n"})).unwrap()["id"].clone();
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("a.txt");
        std::fs::write(&file, "Срок действия договора составляет два года.").unwrap();
        call(
            &window,
            "add_source",
            json!({"notebookId": id, "location": file.to_str().unwrap()}),
        )
        .unwrap();

        call(
            &window,
            "ask",
            json!({"notebookId": id, "question": "Какой срок договора?", "deep": false}),
        )
        .unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        call(
            &window,
            "ask",
            json!({"notebookId": id, "question": "Какой срок договора?", "deep": true}),
        )
        .unwrap();
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "a plan, then the answer"
        );
    }

    #[test]
    fn a_location_pasted_with_spaces_around_it_is_still_read() {
        let window = window(ready_engine());
        let id = call(&window, "create_notebook", json!({"name": "n"})).unwrap()["id"].clone();
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("a.txt");
        std::fs::write(&file, "Текст источника.").unwrap();
        let pasted = format!("  {}\n", file.to_str().unwrap());
        let added = call(
            &window,
            "add_source",
            json!({"notebookId": id, "location": pasted}),
        );
        assert_eq!(added.unwrap(), json!(1));
    }

    #[test]
    fn optional_arguments_may_be_null_as_the_page_sends_them() {
        let window = window(ready_engine());
        let id = call(&window, "create_notebook", json!({"name": "n"})).unwrap()["id"].clone();
        let view = call(
            &window,
            "source_view",
            json!({"notebookId": id, "title": "нет.txt", "ordinal": null, "quote": null}),
        );
        assert!(view.unwrap_err().as_str().unwrap().contains("Не найдено"));
        let summary = call(
            &window,
            "summarize",
            json!({"notebookId": id, "source": null}),
        )
        .unwrap();
        assert!(summary["text"].as_str().unwrap().contains("нечего"));
    }

    #[test]
    fn arguments_are_camel_case_so_snake_case_ones_do_not_reach_the_command() {
        let window = window(ready_engine());
        let wrong = call(&window, "list_sources", json!({"notebook_id": 1}));
        assert!(wrong.is_err());
    }

    #[test]
    fn errors_reach_the_page_as_russian_text() {
        let window = window(ready_engine());
        let id = call(&window, "create_notebook", json!({"name": "n"})).unwrap()["id"].clone();
        let error = call(
            &window,
            "compare",
            json!({"notebookId": id, "first": "a", "second": "a"}),
        )
        .unwrap_err();
        assert!(error.as_str().unwrap().starts_with("Так нельзя"), "{error}");
        let empty_name = call(&window, "create_notebook", json!({"name": "   "})).unwrap_err();
        assert_eq!(empty_name, "Название блокнота не может быть пустым.");
    }

    #[test]
    fn before_the_engine_is_ready_the_page_is_told_to_wait() {
        let window = window(Engine::default());
        assert_eq!(
            call(&window, "status", json!({})).unwrap(),
            json!({"ready": false, "error": null})
        );
        let error = call(&window, "list_notebooks", json!({})).unwrap_err();
        assert!(error.as_str().unwrap().contains("готовится"), "{error}");
    }

    #[test]
    fn a_failed_start_is_visible_in_the_status() {
        let engine = Engine::default();
        engine.slot.set(Err("нет места".into())).ok();
        let window = window(engine);
        assert_eq!(
            call(&window, "status", json!({})).unwrap(),
            json!({"ready": false, "error": "нет места"})
        );
    }
}
