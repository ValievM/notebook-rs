use super::{ChatGptLlm, TokenSource};
use crate::error::Result;
use crate::llm::Llm;
use std::sync::Arc;

/// Uses the ChatGPT sign-in when the person has signed in, and the fallback (Codex through its
/// command line tool) when not. The choice is made at every call, so signing in or out takes
/// effect at once, without a restart.
pub struct AutoLlm {
    tokens: Arc<TokenSource>,
    chatgpt: ChatGptLlm,
    fallback: Box<dyn Llm>,
}

impl AutoLlm {
    pub fn new(tokens: Arc<TokenSource>, chatgpt: ChatGptLlm, fallback: Box<dyn Llm>) -> Self {
        Self {
            tokens,
            chatgpt,
            fallback,
        }
    }

    /// Which one answers now: `"chatgpt"` or `"fallback"`.
    pub fn active(&self) -> &'static str {
        if self.tokens.is_signed_in() {
            "chatgpt"
        } else {
            "fallback"
        }
    }
}

impl Llm for AutoLlm {
    fn complete(&self, prompt: &str) -> Result<String> {
        if self.tokens.is_signed_in() {
            self.chatgpt.complete(prompt)
        } else {
            self.fallback.complete(prompt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::credentials::sample;
    use super::super::testing::{FakeServer, Reply};
    use super::super::{Credentials, OAuth, Store};
    use super::*;
    use crate::metrics::ManualClock;
    use std::sync::Mutex;

    struct SwitchableStore(Mutex<Option<Credentials>>);
    impl Store for SwitchableStore {
        fn load(&self) -> Result<Credentials> {
            self.0.lock().unwrap().clone().ok_or_else(|| {
                crate::error::Error::SignIn("not signed in yet: run `nb login`".into())
            })
        }
        fn save(&self, credentials: &Credentials) -> Result<()> {
            *self.0.lock().unwrap() = Some(credentials.clone());
            Ok(())
        }
    }

    struct Codex;
    impl Llm for Codex {
        fn complete(&self, _: &str) -> Result<String> {
            Ok("from codex".into())
        }
    }

    fn fixture(signed_in: bool) -> (AutoLlm, Arc<SwitchableStore>, FakeServer) {
        let api = FakeServer::start(|_| {
            Reply::events(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"from chatgpt\"}\n\ndata: {\"type\":\"response.completed\"}\n\n",
            )
        });
        let store = Arc::new(SwitchableStore(Mutex::new(
            signed_in.then(|| sample(10_000_000)),
        )));
        let clock = Arc::new(ManualClock::starting_at(0));
        let tokens = Arc::new(TokenSource::new(
            store.clone(),
            Arc::new(OAuth::with_urls(
                "https://example.invalid/a",
                "http://127.0.0.1:9/token",
            )),
            clock,
        ));
        let chatgpt = ChatGptLlm::new(tokens.clone())
            .with_base_url(&api.url)
            .with_model("m");
        (AutoLlm::new(tokens, chatgpt, Box::new(Codex)), store, api)
    }

    #[test]
    fn when_signed_in_the_chatgpt_plan_answers() {
        let (llm, _, api) = fixture(true);
        assert_eq!(llm.active(), "chatgpt");
        assert_eq!(llm.complete("x").unwrap(), "from chatgpt");
        assert_eq!(api.requests().len(), 1);
    }

    #[test]
    fn when_not_signed_in_the_fallback_answers_and_the_api_is_not_touched() {
        let (llm, _, api) = fixture(false);
        assert_eq!(llm.active(), "fallback");
        assert_eq!(llm.complete("x").unwrap(), "from codex");
        assert!(api.requests().is_empty());
    }

    #[test]
    fn signing_in_takes_effect_at_the_next_question_without_a_restart() {
        let (llm, store, _api) = fixture(false);
        assert_eq!(llm.complete("x").unwrap(), "from codex");

        store.save(&sample(10_000_000)).unwrap();

        assert_eq!(llm.complete("x").unwrap(), "from chatgpt");
    }

    #[test]
    fn a_sign_in_that_expired_for_good_is_reported_not_silently_replaced_by_the_fallback() {
        // Silently switching to another model would hide that the sign-in needs attention.
        let (llm, store, _api) = fixture(true);
        let mut expired = sample(0);
        expired.expires_at_ms = 0;
        store.save(&expired).unwrap();
        // The refresh endpoint is unreachable here: that is a model-unavailable error, not a switch.
        let error = llm.complete("x").unwrap_err();
        assert!(!error.to_string().contains("from codex"));
    }
}
