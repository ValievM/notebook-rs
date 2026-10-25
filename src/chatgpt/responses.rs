//! The model through the Responses API, paid by the ChatGPT plan: one request with
//! `stream: true` and `store: false` (both are required for the plan to be charged), and the
//! answer is assembled from the events of the stream.

use super::{API_BASE, TokenSource};
use crate::error::{Error, Result};
use crate::llm::Llm;
use crate::settings::Settings;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How many times a request that failed for a temporary reason is tried in all.
const ATTEMPTS: usize = 3;

pub struct ChatGptLlm {
    base_url: String,
    /// What the person chose; shared, so that it can change while the program runs.
    choice: Arc<Mutex<Settings>>,
    /// The first model of the account, asked for once when the person chose none.
    default_model: Mutex<Option<String>>,
    tokens: Arc<TokenSource>,
    agent: ureq::Agent,
    retry_pause: Duration,
}

impl ChatGptLlm {
    pub fn new(tokens: Arc<TokenSource>) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            // An answer can take minutes; the request is cut only if it hangs for much longer.
            .timeout_global(Some(Duration::from_secs(900)))
            // A refusal carries its reason in the body: it must be readable, not an error.
            .http_status_as_error(false)
            .build()
            .into();
        Self {
            base_url: API_BASE.to_string(),
            choice: Arc::new(Mutex::new(Settings::default())),
            default_model: Mutex::new(None),
            tokens,
            agent,
            retry_pause: Duration::from_secs(1),
        }
    }

    /// Another address, for the tests.
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.trim_end_matches('/').to_string();
        self
    }

    /// Asks for this model instead of the first one of the account.
    pub fn with_model(self, model: &str) -> Self {
        self.choice.lock().expect(LOCK).model = Some(model.to_string());
        self
    }

    /// Reads the model and the effort from this shared choice at every call.
    pub fn with_shared_choice(mut self, choice: Arc<Mutex<Settings>>) -> Self {
        self.choice = choice;
        self
    }

    /// How much the model thinks (`low`, `medium`, `high`).
    pub fn with_reasoning_effort(self, effort: &str) -> Self {
        self.choice.lock().expect(LOCK).effort = Some(effort.to_string());
        self
    }

    pub fn with_retry_pause(mut self, pause: Duration) -> Self {
        self.retry_pause = pause;
        self
    }

    /// The model that is asked for now, if it is known yet.
    pub fn model_name(&self) -> Option<String> {
        let chosen = self.choice.lock().expect(LOCK).model.clone();
        chosen.or_else(|| self.default_model.lock().expect(LOCK).clone())
    }
}

const LOCK: &str = "the model lock is never held across a panic";

impl Llm for ChatGptLlm {
    fn describe(&self) -> String {
        match self.model_name() {
            Some(model) => format!("chatgpt:{model}"),
            None => "chatgpt".to_string(),
        }
    }

    fn complete(&self, prompt: &str) -> Result<String> {
        let model = self.resolve_model()?;
        let effort = self.choice.lock().expect(LOCK).effort.clone();
        let body = request_body(&model, prompt, effort.as_deref());
        for attempt in 1..=ATTEMPTS {
            match self.stream_once(&body) {
                Ok(text) => return Ok(text),
                Err(failure) if failure.retryable && attempt < ATTEMPTS => {
                    std::thread::sleep(self.retry_pause * attempt as u32);
                }
                Err(failure) => return Err(failure.error),
            }
        }
        unreachable!("the last attempt returns")
    }
}

/// A failed attempt, and whether trying again can help.
struct Failure {
    error: Error,
    retryable: bool,
}

impl Failure {
    fn fatal(error: Error) -> Self {
        Self {
            error,
            retryable: false,
        }
    }

    fn temporary(error: Error) -> Self {
        Self {
            error,
            retryable: true,
        }
    }
}

/// A model the account can use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    pub slug: String,
    /// The name shown to people, when the server gives one.
    pub display_name: Option<String>,
}

impl ChatGptLlm {
    /// The models of the account that are meant to be chosen (visibility `list`), in the order
    /// the server gives them.
    pub fn models(&self) -> Result<Vec<ModelEntry>> {
        let token = self.tokens.token()?;
        let mut response = self
            .agent
            .get(&format!("{}/models", self.base_url))
            .header("Authorization", &format!("Bearer {token}"))
            .call()
            .map_err(|e| Error::ModelUnavailable(format!("cannot reach the ChatGPT API: {e}")))?;
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .with_config()
            .limit(MAX_ERROR_BODY * 8)
            .read_to_string()
            .map_err(|e| Error::ModelUnavailable(format!("cannot read the model list: {e}")))?;
        if status != 200 {
            return Err(api_failure(status, &body).error);
        }
        let listing: serde_json::Value = serde_json::from_str(&body)
            .map_err(|_| Error::ModelUnavailable("the model list is not JSON".into()))?;
        Ok(listing["models"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|m| m["visibility"] == "list")
            .filter_map(|m| {
                let slug = m["slug"].as_str().filter(|s| !s.is_empty())?;
                Some(ModelEntry {
                    slug: slug.to_string(),
                    display_name: m["display_name"].as_str().map(str::to_string),
                })
            })
            .collect())
    }

    /// The model to ask for: the chosen one, or the first listed model of the account (asked
    /// for once and remembered).
    fn resolve_model(&self) -> Result<String> {
        if let Some(model) = self.model_name() {
            return Ok(model);
        }
        // The lock is not held while the server is asked; two first calls at once may both ask,
        // which is harmless.
        let model = self
            .models()?
            .into_iter()
            .next()
            .ok_or_else(|| {
                Error::ModelUnavailable("no models are available for this account".into())
            })?
            .slug;
        *self.default_model.lock().expect(LOCK) = Some(model.clone());
        Ok(model)
    }

    /// One request, and the stream of its answer read to the end.
    fn stream_once(&self, body: &str) -> std::result::Result<String, Failure> {
        let token = self.tokens.token().map_err(|error| match error {
            Error::ModelUnavailable(_) => Failure::temporary(error),
            other => Failure::fatal(other),
        })?;
        let response = self
            .agent
            .post(&format!("{}/responses", self.base_url))
            .header("Authorization", &format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .send(body)
            .map_err(|e| {
                Failure::temporary(Error::ModelUnavailable(format!(
                    "cannot reach the ChatGPT API: {e}"
                )))
            })?;
        let status = response.status().as_u16();
        let mut body = response.into_body();
        if status != 200 {
            let text = body
                .with_config()
                .limit(MAX_ERROR_BODY)
                .read_to_string()
                .unwrap_or_default();
            return Err(api_failure(status, &text));
        }
        read_events(std::io::BufReader::new(body.into_reader().take(MAX_STREAM)))
    }
}

/// How much of an error answer is read, and how much of a stream (a guard, not a limit anyone meets).
const MAX_ERROR_BODY: u64 = 16 * 1024;
const MAX_STREAM: u64 = 64 * 1024 * 1024;

fn request_body(model: &str, prompt: &str, effort: Option<&str>) -> String {
    let mut body = serde_json::json!({
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "store": false,
        "stream": true,
    });
    if let Some(effort) = effort {
        body["reasoning"] = serde_json::json!({ "effort": effort });
    }
    body.to_string()
}

/// An HTTP refusal as an error. The message is the code and text the server named, never the
/// request (which holds the token).
fn api_failure(status: u16, body: &str) -> Failure {
    let parsed: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let message = if let Some(code) = parsed["error"]["code"].as_str().filter(|c| !c.is_empty()) {
        format!(
            "{code}: {}",
            parsed["error"]["message"].as_str().unwrap_or("")
        )
    } else if let Some(text) = parsed["error"]["message"].as_str() {
        text.to_string()
    } else if let Some(detail) = parsed["detail"].as_str() {
        detail.to_string()
    } else {
        body.trim().chars().take(300).collect()
    };
    if status == 401 {
        return Failure::fatal(Error::SignIn(
            "ChatGPT did not accept the sign-in token: run `nb login` again".into(),
        ));
    }
    let plan_used_up = message.contains("usage_limit_exceeded");
    let error = Error::ModelUnavailable(if plan_used_up {
        format!("ChatGPT API {status}: {message} (the plan limit is used up: try later)")
    } else {
        format!("ChatGPT API {status}: {message}")
    });
    if (status >= 500 || status == 429) && !plan_used_up {
        Failure::temporary(error)
    } else {
        Failure::fatal(error)
    }
}

/// Reads the server-sent events of the stream: the text comes in `output_text.delta` events and
/// `response.completed` ends it; a failure event ends it with an error.
fn read_events(reader: impl std::io::BufRead) -> std::result::Result<String, Failure> {
    let mut text = String::new();
    let mut completed = false;
    for line in reader.lines() {
        let line = line.map_err(|e| {
            Failure::temporary(Error::ModelUnavailable(format!("the stream broke: {e}")))
        })?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let event: serde_json::Value = serde_json::from_str(data).map_err(|_| {
            Failure::fatal(Error::ModelUnavailable(
                "a stream event could not be decoded".into(),
            ))
        })?;
        match event["type"].as_str().unwrap_or("") {
            "response.output_text.delta" => text.push_str(event["delta"].as_str().unwrap_or("")),
            "response.completed" => completed = true,
            "response.failed" | "error" | "response.incomplete" => {
                return Err(stream_failure(&event, data));
            }
            _ => {}
        }
    }
    if !completed {
        return Err(Failure::fatal(Error::ModelUnavailable(
            "the ChatGPT stream ended without response.completed".into(),
        )));
    }
    Ok(text)
}

/// A failure event as an error with the code OpenAI gave.
fn stream_failure(event: &serde_json::Value, raw: &str) -> Failure {
    let text = |value: &serde_json::Value| value.as_str().unwrap_or("").to_string();
    let (mut code, mut message) = (text(&event["code"]), text(&event["message"]));
    if event["error"].is_object() {
        (code, message) = (
            text(&event["error"]["code"]),
            text(&event["error"]["message"]),
        );
    }
    if event["response"]["error"].is_object() {
        (code, message) = (
            text(&event["response"]["error"]["code"]),
            text(&event["response"]["error"]["message"]),
        );
    }
    if event["type"] == "response.incomplete" && event["response"]["incomplete_details"].is_object()
    {
        code = "incomplete".to_string();
        message = text(&event["response"]["incomplete_details"]["reason"]);
    }
    if code.is_empty() {
        code = text(&event["type"]);
    }
    if message.is_empty() {
        message = raw.chars().take(300).collect();
    }
    let error = Error::ModelUnavailable(format!("{code}: {message}"));
    let temporary = matches!(
        code.as_str(),
        "error"
            | "server_error"
            | "internal_error"
            | "service_unavailable"
            | "overloaded"
            | "subscription_sharing_usage_unavailable"
    );
    if temporary {
        Failure::temporary(error)
    } else {
        Failure::fatal(error)
    }
}

#[cfg(test)]
mod tests {
    use super::super::credentials::sample;
    use super::super::testing::{FakeServer, Reply};
    use super::super::{OAuth, Store};
    use super::*;
    use crate::metrics::ManualClock;

    struct StaticStore(Credentials);
    use super::super::Credentials;
    impl Store for StaticStore {
        fn load(&self) -> Result<Credentials> {
            Ok(self.0.clone())
        }
        fn save(&self, _: &Credentials) -> Result<()> {
            Ok(())
        }
    }

    const NOW: i64 = 1_000_000;

    fn tokens_with(credentials: Credentials, oauth_url: &str) -> Arc<TokenSource> {
        let clock = Arc::new(ManualClock::starting_at(NOW));
        let oauth = OAuth::with_urls("https://example.invalid/a", &format!("{oauth_url}/token"))
            .with_clock(clock.clone());
        Arc::new(TokenSource::new(
            Arc::new(StaticStore(credentials)),
            Arc::new(oauth),
            clock,
        ))
    }

    fn signed_in() -> Arc<TokenSource> {
        tokens_with(sample(NOW + 3_600_000), "http://127.0.0.1:9")
    }

    fn client(server: &FakeServer) -> ChatGptLlm {
        ChatGptLlm::new(signed_in())
            .with_base_url(&server.url)
            .with_model("test-model")
            .with_retry_pause(Duration::from_millis(1))
    }

    fn delta(text: &str) -> String {
        format!(
            "event: response.output_text.delta\ndata: {}\n\n",
            serde_json::json!({"type":"response.output_text.delta","delta":text})
        )
    }

    const COMPLETED: &str =
        "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n";

    fn stream_of(parts: &[&str]) -> Reply {
        let mut body: String = parts.iter().map(|p| delta(p)).collect();
        body.push_str(COMPLETED);
        Reply::events(&body)
    }

    #[test]
    fn the_request_has_the_shape_that_makes_the_plan_pay() {
        let server = FakeServer::start(|_| stream_of(&["ок"]));

        client(&server).complete("Какой срок?").unwrap();

        let request = &server.requests()[0];
        assert_eq!(
            (request.method.as_str(), request.path.as_str()),
            ("POST", "/responses")
        );
        assert_eq!(
            request.header("authorization"),
            Some("Bearer ACCESS-SECRET")
        );
        assert_eq!(request.header("accept"), Some("text/event-stream"));
        assert_eq!(request.header("content-type"), Some("application/json"));
        let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(
            body["input"],
            serde_json::json!([{"role": "user", "content": "Какой срок?"}])
        );
        assert!(body.get("reasoning").is_none());
    }

    #[test]
    fn it_describes_itself_by_its_model_once_the_model_is_known() {
        let server = FakeServer::start(|_| stream_of(&["ок"]));
        assert_eq!(client(&server).describe(), "chatgpt:test-model");
        let unknown = ChatGptLlm::new(signed_in()).with_base_url(&server.url);
        assert_eq!(unknown.describe(), "chatgpt");
    }

    #[test]
    fn the_reasoning_effort_is_sent_when_it_is_chosen() {
        let server = FakeServer::start(|_| stream_of(&["ок"]));
        client(&server)
            .with_reasoning_effort("high")
            .complete("x")
            .unwrap();
        let body: serde_json::Value = serde_json::from_str(&server.requests()[0].body).unwrap();
        assert_eq!(body["reasoning"]["effort"], "high");
    }

    #[test]
    fn a_change_of_the_shared_choice_applies_to_the_next_call() {
        let server = FakeServer::start(|_| stream_of(&["ок"]));
        let choice = Arc::new(Mutex::new(Settings::default()));
        let llm = ChatGptLlm::new(signed_in())
            .with_base_url(&server.url)
            .with_shared_choice(choice.clone());
        *choice.lock().unwrap() = Settings {
            model: Some("one".into()),
            effort: Some("low".into()),
        };
        llm.complete("x").unwrap();
        *choice.lock().unwrap() = Settings {
            model: Some("two".into()),
            effort: Some("high".into()),
        };
        llm.complete("x").unwrap();

        let sent: Vec<serde_json::Value> = server
            .requests()
            .iter()
            .map(|r| serde_json::from_str(&r.body).unwrap())
            .collect();
        assert_eq!(
            (
                sent[0]["model"].as_str(),
                sent[0]["reasoning"]["effort"].as_str()
            ),
            (Some("one"), Some("low"))
        );
        assert_eq!(
            (
                sent[1]["model"].as_str(),
                sent[1]["reasoning"]["effort"].as_str()
            ),
            (Some("two"), Some("high"))
        );
        assert_eq!(llm.describe(), "chatgpt:two");
    }

    #[test]
    fn the_answer_is_put_together_from_the_pieces_of_the_stream() {
        let server = FakeServer::start(|_| {
            let mut body =
                String::from("event: response.created\ndata: {\"type\":\"response.created\"}\n\n");
            body.push_str(&delta("Два "));
            body.push_str("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\"}}\n\n");
            body.push_str(&delta("года"));
            body.push_str(COMPLETED);
            body.push_str("data: [DONE]\n\n");
            Reply::events(&body)
        });
        assert_eq!(client(&server).complete("x").unwrap(), "Два года");
    }

    #[test]
    fn the_model_is_taken_from_the_list_of_the_account_once_skipping_hidden_ones() {
        let server = FakeServer::start(|request| {
            if request.path == "/models" {
                Reply::json(
                    200,
                    r#"{"models":[{"slug":"hidden-one","visibility":"hide"},{"slug":"the-first-listed","visibility":"list"},{"slug":"another","visibility":"list"}]}"#,
                )
            } else {
                stream_of(&["ок"])
            }
        });
        let llm = ChatGptLlm::new(signed_in()).with_base_url(&server.url);

        llm.complete("a").unwrap();
        llm.complete("b").unwrap();

        let paths: Vec<String> = server.requests().iter().map(|r| r.path.clone()).collect();
        assert_eq!(
            paths,
            ["/models", "/responses", "/responses"],
            "the list is asked for once"
        );
        assert_eq!(
            server.requests()[0].header("authorization"),
            Some("Bearer ACCESS-SECRET")
        );
        let body: serde_json::Value = serde_json::from_str(&server.requests()[1].body).unwrap();
        assert_eq!(body["model"], "the-first-listed");
        assert_eq!(llm.model_name().as_deref(), Some("the-first-listed"));
    }

    #[test]
    fn the_models_of_the_account_are_listed_in_order_without_the_hidden_ones() {
        let server = FakeServer::start(|_| {
            Reply::json(
                200,
                r#"{"models":[
                {"slug":"first-model","display_name":"First","visibility":"list"},
                {"slug":"hidden-model","visibility":"hide"},
                {"slug":"second-model","visibility":"list"},
                {"slug":"","visibility":"list"}]}"#,
            )
        });

        let models = ChatGptLlm::new(signed_in())
            .with_base_url(&server.url)
            .models()
            .unwrap();

        assert_eq!(
            models,
            vec![
                ModelEntry {
                    slug: "first-model".into(),
                    display_name: Some("First".into())
                },
                ModelEntry {
                    slug: "second-model".into(),
                    display_name: None
                },
            ]
        );
        assert_eq!(
            server.requests()[0].header("authorization"),
            Some("Bearer ACCESS-SECRET")
        );
    }

    #[test]
    fn the_list_of_models_reports_a_rejected_token_like_any_other_call() {
        let server = FakeServer::start(|_| Reply::json(401, r#"{"error":{"message":"bad"}}"#));
        let error = ChatGptLlm::new(signed_in())
            .with_base_url(&server.url)
            .models()
            .unwrap_err();
        assert!(matches!(error, Error::SignIn(_)), "{error}");
    }

    #[test]
    fn an_account_without_listed_models_is_an_error_that_says_so() {
        let server = FakeServer::start(|_| {
            Reply::json(200, r#"{"models":[{"slug":"x","visibility":"hide"}]}"#)
        });
        let error = ChatGptLlm::new(signed_in())
            .with_base_url(&server.url)
            .complete("a")
            .unwrap_err();
        assert!(matches!(error, Error::ModelUnavailable(_)), "{error}");
        assert!(error.to_string().contains("no models"), "{error}");
    }

    #[test]
    fn a_rejected_token_asks_for_a_new_sign_in() {
        let server = FakeServer::start(|_| {
            Reply::json(401, r#"{"error":{"code":"invalid_token","message":"bad"}}"#)
        });
        let error = client(&server).complete("x").unwrap_err();
        assert!(matches!(error, Error::SignIn(_)), "{error}");
        assert!(error.to_string().contains("nb login"), "{error}");
        assert_eq!(server.requests().len(), 1);
    }

    #[test]
    fn an_exhausted_plan_is_reported_clearly_and_not_retried() {
        let server = FakeServer::start(|_| {
            Reply::json(
                429,
                r#"{"error":{"code":"usage_limit_exceeded","message":"The usage limit has been reached"}}"#,
            )
        });
        let error = client(&server).complete("x").unwrap_err();
        assert!(matches!(error, Error::ModelUnavailable(_)), "{error}");
        assert!(
            error.to_string().contains("usage_limit_exceeded"),
            "{error}"
        );
        assert_eq!(server.requests().len(), 1, "waiting does not refill a plan");
    }

    #[test]
    fn a_temporary_failure_is_tried_again_and_a_lasting_one_gives_up_after_three_tries() {
        let calls = Arc::new(Mutex::new(0));
        let counted = calls.clone();
        let flaky = FakeServer::start(move |_| {
            let mut n = counted.lock().unwrap();
            *n += 1;
            if *n == 1 {
                Reply::json(503, "later")
            } else {
                stream_of(&["ок"])
            }
        });
        assert_eq!(client(&flaky).complete("x").unwrap(), "ок");
        assert_eq!(flaky.requests().len(), 2);

        let down = FakeServer::start(|_| Reply::json(500, "broken"));
        let error = client(&down).complete("x").unwrap_err();
        assert!(matches!(error, Error::ModelUnavailable(_)), "{error}");
        assert!(error.to_string().contains("500"), "{error}");
        assert_eq!(down.requests().len(), 3);
    }

    fn failure_event(code: &str, message: &str) -> Reply {
        Reply::events(&format!(
            "data: {}\n\n",
            serde_json::json!({"type": "response.failed", "response": {"error": {"code": code, "message": message}}})
        ))
    }

    #[test]
    fn a_failure_inside_the_stream_is_an_error_with_its_code_and_a_lasting_one_is_not_retried() {
        let server = FakeServer::start(|_| failure_event("context_length_exceeded", "too long"));
        let error = client(&server).complete("x").unwrap_err();
        assert!(
            error.to_string().contains("context_length_exceeded"),
            "{error}"
        );
        assert_eq!(server.requests().len(), 1);
    }

    #[test]
    fn a_temporary_failure_inside_the_stream_is_tried_again() {
        let server = FakeServer::start(|_| failure_event("server_error", "oops"));
        assert!(client(&server).complete("x").is_err());
        assert_eq!(server.requests().len(), 3);
    }

    #[test]
    fn an_incomplete_answer_says_why() {
        let server = FakeServer::start(|_| {
            Reply::events(
                "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
            )
        });
        let error = client(&server).complete("x").unwrap_err();
        assert!(error.to_string().contains("max_output_tokens"), "{error}");
    }

    #[test]
    fn a_stream_that_ends_without_completion_is_an_error_not_a_half_answer() {
        let server = FakeServer::start(|_| Reply::events(&delta("полови")));
        let error = client(&server).complete("x").unwrap_err();
        assert!(error.to_string().contains("response.completed"), "{error}");
    }

    #[test]
    fn the_token_is_refreshed_first_when_it_is_about_to_expire_and_the_new_one_is_used() {
        let auth = FakeServer::start(|_| {
            Reply::json(
                200,
                r#"{"access_token":"REFRESHED","refresh_token":"R2","expires_in":3600}"#,
            )
        });
        let api = FakeServer::start(|_| stream_of(&["ок"]));
        let tokens = tokens_with(sample(NOW + 1_000), &auth.url);
        let llm = ChatGptLlm::new(tokens)
            .with_base_url(&api.url)
            .with_model("m");

        llm.complete("x").unwrap();

        assert_eq!(auth.requests().len(), 1);
        assert_eq!(
            api.requests()[0].header("authorization"),
            Some("Bearer REFRESHED")
        );
    }

    #[test]
    fn no_error_message_ever_contains_the_token() {
        let server = FakeServer::start(|_| Reply::json(500, "boom"));
        let error = client(&server).complete("x").unwrap_err().to_string();
        assert!(
            !error.contains("ACCESS-SECRET") && !error.contains("REFRESH-SECRET"),
            "{error}"
        );
    }
}
