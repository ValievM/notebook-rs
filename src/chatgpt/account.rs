use super::{
    API_BASE, AutoLlm, ChatGptLlm, DEFAULT_PORT, FileStore, LoginOptions, OAuth, Store, TokenSource,
};
use crate::error::{Error, Result};
use crate::llm::Llm;
use crate::metrics::SystemClock;
use crate::settings::{Settings, check_effort};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long the person has to approve in the browser.
const LOCK: &str = "the choice lock is never held across a panic";

const SIGN_IN_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// The ChatGPT sign-in of this machine, as the command line and the window use it: where the
/// tokens are kept, how to sign in and out, and the model that uses them.
pub struct ChatGptAccount {
    file: PathBuf,
    store: Arc<FileStore>,
    oauth: Arc<OAuth>,
    tokens: Arc<TokenSource>,
    api_base: String,
    port: u16,
    /// The model and the effort in use now; shared with the model, so a change applies at once.
    choice: Arc<Mutex<Settings>>,
    /// Where a choice is saved, if it is saved at all.
    settings_file: Option<PathBuf>,
}

impl ChatGptAccount {
    pub fn open(file: PathBuf) -> Self {
        Self::with_oauth(file, OAuth::new())
    }

    /// With another authorization server, for the tests.
    pub fn with_oauth(file: PathBuf, oauth: OAuth) -> Self {
        let store = Arc::new(FileStore::new(file.clone()));
        let oauth = Arc::new(oauth);
        let tokens = Arc::new(TokenSource::new(
            store.clone(),
            oauth.clone(),
            Arc::new(SystemClock::default()),
        ));
        Self {
            file,
            store,
            oauth,
            tokens,
            api_base: API_BASE.to_string(),
            port: DEFAULT_PORT,
            choice: Arc::new(Mutex::new(Settings::default())),
            settings_file: None,
        }
    }

    /// The choices in use at the start (the saved ones, maybe overridden by a flag) and the file
    /// where a new choice is saved.
    pub fn with_settings(mut self, file: PathBuf, initial: Settings) -> Self {
        *self.choice.lock().expect(LOCK) = initial;
        self.settings_file = Some(file);
        self
    }

    /// The model and the effort in use now (`None`: the account's first model, the model's own
    /// effort).
    pub fn choice(&self) -> Settings {
        self.choice.lock().expect(LOCK).clone()
    }

    /// Chooses the model and/or the effort, saves the choice and applies it to the next call.
    /// With a sign-in, a model that the account does not list is refused.
    pub fn choose(&self, model: Option<&str>, effort: Option<&str>) -> Result<Settings> {
        if let Some(effort) = effort {
            check_effort(effort)?;
        }
        if let Some(model) = model.filter(|_| self.signed_in()) {
            let models = self.models()?;
            if !models.iter().any(|m| m.slug == model) {
                let names: Vec<_> = models.iter().map(|m| m.slug.as_str()).collect();
                return Err(Error::Usage(format!(
                    "the model \"{model}\" is not in the list of the account: {}",
                    names.join(", ")
                )));
            }
        }
        if let Some(file) = &self.settings_file {
            // The saved file is changed, not the overridden values in use.
            let mut saved = Settings::load(file)?;
            saved.model = model.map(str::to_string).or(saved.model);
            saved.effort = effort.map(str::to_string).or(saved.effort);
            saved.save(file)?;
        }
        let mut current = self.choice.lock().expect(LOCK);
        current.model = model.map(str::to_string).or(current.model.take());
        current.effort = effort.map(str::to_string).or(current.effort.take());
        Ok(current.clone())
    }

    /// Another API address, for the tests.
    pub fn with_api_base(mut self, api_base: &str) -> Self {
        self.api_base = api_base.to_string();
        self
    }

    /// Another port for the browser to come back to (0 takes any free one), for the tests.
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Whether a usable sign-in is saved.
    pub fn signed_in(&self) -> bool {
        self.tokens.is_signed_in()
    }

    /// Signs in through the browser and saves the result. Keeps the client and the host of a
    /// previous sign-in, so the app stays one entry in the person's list of connected apps.
    pub fn sign_in(
        &self,
        agent_name: &str,
        open_browser: &(dyn Fn(&str) -> Result<()> + Sync),
    ) -> Result<()> {
        let existing = self.store.load().ok();
        let credentials = super::login(
            &self.oauth,
            LoginOptions {
                existing: existing.as_ref(),
                agent_name,
                port: self.port,
                open_browser,
                timeout: SIGN_IN_TIMEOUT,
            },
        )?;
        self.tokens.sign_in(credentials)
    }

    /// Forgets the sign-in on this machine (the file is deleted). The connection also shows in
    /// the person's ChatGPT settings, where it can be removed too.
    pub fn sign_out(&self) -> Result<()> {
        match std::fs::remove_file(&self.file) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(Error::Io(error)),
        }
    }

    /// The models this account can choose from, as the server lists them.
    pub fn models(&self) -> Result<Vec<super::ModelEntry>> {
        ChatGptLlm::new(self.tokens.clone())
            .with_base_url(&self.api_base)
            .models()
    }

    /// The model: the ChatGPT plan when signed in, `fallback` when not. It uses the current
    /// [`choice`](Self::choice).
    pub fn llm(&self, fallback: Box<dyn Llm>) -> AutoLlm {
        let chatgpt = ChatGptLlm::new(self.tokens.clone())
            .with_base_url(&self.api_base)
            .with_shared_choice(self.choice.clone());
        AutoLlm::new(self.tokens.clone(), chatgpt, fallback)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{FakeServer, Reply};
    use super::*;

    struct Codex;
    impl Llm for Codex {
        fn complete(&self, _: &str) -> Result<String> {
            Ok("from codex".into())
        }
    }

    fn auth_server() -> FakeServer {
        FakeServer::start(|_| {
            Reply::json(
                200,
                r#"{"access_token":"A","refresh_token":"R","expires_in":3600,"scope":"openid chatgpt.tokens.use.direct"}"#,
            )
        })
    }

    fn account(directory: &std::path::Path, auth: &FakeServer) -> ChatGptAccount {
        let oauth = OAuth::with_urls(
            "https://auth.example/authorize",
            &format!("{}/token", auth.url),
        );
        ChatGptAccount::with_oauth(directory.join("chatgpt-auth.json"), oauth).with_port(0)
    }

    /// The person approving in the browser: the redirect comes back with a code and a client id.
    fn approving(url: &str) -> Result<()> {
        let pair = |name: &str| {
            let start = url.find(&format!("{name}=")).unwrap() + name.len() + 1;
            let end = url[start..].find('&').map_or(url.len(), |e| start + e);
            url[start..end].to_string()
        };
        let redirect = pair("redirect_uri").replace("%3A", ":").replace("%2F", "/");
        let state = pair("state");
        std::thread::spawn(move || {
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .http_status_as_error(false)
                .build()
                .into();
            let _ = agent
                .get(&format!("{redirect}?state={state}&code=X&client_id=issued"))
                .call();
        });
        Ok(())
    }

    #[test]
    fn signing_in_saves_the_tokens_in_a_file_only_the_owner_can_read() {
        let directory = tempfile::tempdir().unwrap();
        let auth = auth_server();
        let account = account(directory.path(), &auth);
        assert!(!account.signed_in());

        account.sign_in("notebook-rs", &approving).unwrap();

        assert!(account.signed_in());
        let saved = FileStore::new(directory.path().join("chatgpt-auth.json"))
            .load()
            .unwrap();
        assert_eq!(
            (saved.client_id.as_str(), saved.access_token.as_str()),
            ("issued", "A")
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(directory.path().join("chatgpt-auth.json"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn signing_in_again_keeps_the_host_so_the_app_stays_one_entry() {
        let directory = tempfile::tempdir().unwrap();
        let auth = auth_server();
        let account = account(directory.path(), &auth);
        account.sign_in("notebook-rs", &approving).unwrap();
        let first = FileStore::new(directory.path().join("chatgpt-auth.json"))
            .load()
            .unwrap();

        account.sign_in("notebook-rs", &approving).unwrap();

        let second = FileStore::new(directory.path().join("chatgpt-auth.json"))
            .load()
            .unwrap();
        assert_eq!(second.host_id, first.host_id);
        assert_eq!(second.client_id, first.client_id);
    }

    #[test]
    fn signing_out_deletes_the_file_and_twice_is_fine() {
        let directory = tempfile::tempdir().unwrap();
        let auth = auth_server();
        let account = account(directory.path(), &auth);
        account.sign_in("notebook-rs", &approving).unwrap();
        assert!(directory.path().join("chatgpt-auth.json").exists());

        account.sign_out().unwrap();

        assert!(!account.signed_in());
        assert!(!directory.path().join("chatgpt-auth.json").exists());
        account.sign_out().unwrap();
    }

    #[test]
    fn the_model_is_codex_before_signing_in_and_chatgpt_with_the_chosen_model_and_effort_after() {
        let directory = tempfile::tempdir().unwrap();
        let auth = auth_server();
        let api = FakeServer::start(|_| {
            Reply::events(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"from chatgpt\"}\n\ndata: {\"type\":\"response.completed\"}\n\n",
            )
        });
        let account = account(directory.path(), &auth)
            .with_api_base(&api.url)
            .with_settings(
                directory.path().join("settings.json"),
                Settings {
                    model: Some("gpt-test".into()),
                    effort: Some("high".into()),
                },
            );
        let llm = account.llm(Box::new(Codex));
        assert_eq!(llm.complete("x").unwrap(), "from codex");
        assert!(api.requests().is_empty());

        account.sign_in("notebook-rs", &approving).unwrap();

        assert_eq!(llm.complete("x").unwrap(), "from chatgpt");
        let body: serde_json::Value = serde_json::from_str(&api.requests()[0].body).unwrap();
        assert_eq!(body["model"], "gpt-test");
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(api.requests()[0].header("authorization"), Some("Bearer A"));
    }

    #[test]
    fn the_account_lists_its_models() {
        let directory = tempfile::tempdir().unwrap();
        let auth = auth_server();
        let api = FakeServer::start(|_| {
            Reply::json(200, r#"{"models":[{"slug":"m-1","visibility":"list"}]}"#)
        });
        let account = account(directory.path(), &auth).with_api_base(&api.url);
        account.sign_in("notebook-rs", &approving).unwrap();

        let models = account.models().unwrap();

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].slug, "m-1");
    }

    #[test]
    fn a_choice_is_saved_applied_at_once_and_checked() {
        let directory = tempfile::tempdir().unwrap();
        let auth = auth_server();
        let api = FakeServer::start(|request| {
            if request.path == "/models" {
                Reply::json(
                    200,
                    r#"{"models":[{"slug":"m-1","visibility":"list"},{"slug":"m-2","visibility":"list"}]}"#,
                )
            } else {
                Reply::events(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\ndata: {\"type\":\"response.completed\"}\n\n",
                )
            }
        });
        let file = directory.path().join("settings.json");
        let account = account(directory.path(), &auth)
            .with_api_base(&api.url)
            .with_settings(file.clone(), Settings::default());
        account.sign_in("notebook-rs", &approving).unwrap();
        let llm = account.llm(Box::new(Codex));

        account.choose(Some("m-2"), Some("medium")).unwrap();
        account.choose(None, Some("high")).unwrap();

        assert_eq!(
            account.choice(),
            Settings {
                model: Some("m-2".into()),
                effort: Some("high".into())
            }
        );
        assert_eq!(Settings::load(&file).unwrap(), account.choice());
        llm.complete("x").unwrap();
        let sent = api.requests();
        let body: serde_json::Value = serde_json::from_str(&sent.last().unwrap().body).unwrap();
        assert_eq!(body["model"], "m-2");
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(account.choose(Some("nope"), None).is_err());
        assert!(account.choose(None, Some("extreme")).is_err());
        assert_eq!(account.choice().model.as_deref(), Some("m-2"));
    }

    #[test]
    fn a_flag_in_use_is_not_saved_over_the_file() {
        let directory = tempfile::tempdir().unwrap();
        let auth = auth_server();
        let file = directory.path().join("settings.json");
        Settings {
            model: Some("saved".into()),
            effort: None,
        }
        .save(&file)
        .unwrap();
        let account = account(directory.path(), &auth).with_settings(
            file.clone(),
            Settings {
                model: Some("from-flag".into()),
                effort: None,
            },
        );

        account.choose(None, Some("low")).unwrap();

        assert_eq!(
            Settings::load(&file).unwrap().model.as_deref(),
            Some("saved")
        );
        assert_eq!(account.choice().model.as_deref(), Some("from-flag"));
    }
}
