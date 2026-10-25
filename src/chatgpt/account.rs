use super::{
    API_BASE, AutoLlm, ChatGptLlm, DEFAULT_PORT, FileStore, LoginOptions, OAuth, Store, TokenSource,
};
use crate::error::{Error, Result};
use crate::llm::Llm;
use crate::metrics::SystemClock;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// How long the person has to approve in the browser.
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
        }
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

    /// The model: the ChatGPT plan when signed in, `fallback` when not. `model` and `effort`
    /// choose the model and how much it thinks; without them the account's first model is used.
    pub fn llm(
        &self,
        fallback: Box<dyn Llm>,
        model: Option<&str>,
        effort: Option<&str>,
    ) -> AutoLlm {
        let mut chatgpt = ChatGptLlm::new(self.tokens.clone()).with_base_url(&self.api_base);
        if let Some(model) = model {
            chatgpt = chatgpt.with_model(model);
        }
        if let Some(effort) = effort {
            chatgpt = chatgpt.with_reasoning_effort(effort);
        }
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
        let account = account(directory.path(), &auth).with_api_base(&api.url);
        let llm = account.llm(Box::new(Codex), Some("gpt-test"), Some("high"));
        assert_eq!(llm.complete("x").unwrap(), "from codex");
        assert!(api.requests().is_empty());

        account.sign_in("notebook-rs", &approving).unwrap();

        assert_eq!(llm.complete("x").unwrap(), "from chatgpt");
        let body: serde_json::Value = serde_json::from_str(&api.requests()[0].body).unwrap();
        assert_eq!(body["model"], "gpt-test");
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(api.requests()[0].header("authorization"), Some("Bearer A"));
    }
}
