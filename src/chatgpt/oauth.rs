use super::{Credentials, RESOURCE};
use crate::metrics::{Clock, SystemClock};
use serde::Deserialize;
use std::sync::Arc;

/// The server refused: its status and the error it named. The body is never kept beyond these.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenError {
    pub status: u16,
    pub code: String,
    pub description: String,
}

impl std::fmt::Display for TokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "token endpoint returned {}: {} {}",
            self.status, self.code, self.description
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OAuthError {
    /// The server answered with a refusal.
    Rejected(TokenError),
    /// The server could not be reached.
    Network(String),
    /// The server answered, but not with tokens.
    Malformed(String),
}

impl std::fmt::Display for OAuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OAuthError::Rejected(error) => write!(f, "{error}"),
            OAuthError::Network(detail) => write!(f, "cannot reach the sign-in server: {detail}"),
            OAuthError::Malformed(detail) => {
                write!(f, "unexpected answer of the sign-in server: {detail}")
            }
        }
    }
}

impl OAuthError {
    /// Whether the refresh token is no good any more and the person has to sign in again.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OAuthError::Rejected(error) if matches!(
                error.code.as_str(),
                "invalid_grant"
                    | "invalid_refresh_token"
                    | "token_expired"
                    | "refresh_token_expired"
                    | "refresh_token_invalidated"
                    | "refresh_token_reused"
            )
        )
    }
}

/// The client of the OpenAI authorization server.
pub struct OAuth {
    authorize_url: String,
    token_url: String,
    agent: ureq::Agent,
    clock: Arc<dyn Clock>,
}

impl OAuth {
    pub fn new() -> Self {
        Self::with_urls(super::AUTHORIZE_URL, super::TOKEN_URL)
    }

    /// Other addresses, for the tests.
    pub fn with_urls(authorize_url: &str, token_url: &str) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            // A refusal carries the reason in its body: it must be readable, not an error.
            .http_status_as_error(false)
            .build()
            .into();
        Self {
            authorize_url: authorize_url.to_string(),
            token_url: token_url.to_string(),
            agent,
            clock: Arc::new(SystemClock::default()),
        }
    }

    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    pub fn authorize_url(&self) -> &str {
        &self.authorize_url
    }

    /// Exchanges the code from the browser for tokens.
    pub fn exchange_code(
        &self,
        client_id: &str,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
    ) -> Result<Credentials, OAuthError> {
        let form = [
            ("grant_type", "authorization_code"),
            ("client_id", client_id),
            ("code", code),
            ("code_verifier", verifier),
            ("redirect_uri", redirect_uri),
            ("resource", RESOURCE),
        ];
        let base = Credentials {
            client_id: client_id.to_string(),
            host_id: String::new(),
            access_token: String::new(),
            refresh_token: String::new(),
            expires_at_ms: 0,
            scope: String::new(),
        };
        self.request_tokens(&form, base)
    }

    /// Exchanges the refresh token for a new pair. No scope is sent, so the granted rights stay.
    pub fn refresh(&self, current: &Credentials) -> Result<Credentials, OAuthError> {
        let form = [
            ("grant_type", "refresh_token"),
            ("client_id", current.client_id.as_str()),
            ("refresh_token", current.refresh_token.as_str()),
            ("resource", RESOURCE),
        ];
        self.request_tokens(&form, current.clone())
    }

    /// Sends the form and reads the tokens. Fields the server does not return stay as they are
    /// in `base`. Nothing of the body, except the named error, is kept in a failure.
    fn request_tokens(
        &self,
        form: &[(&str, &str)],
        base: Credentials,
    ) -> Result<Credentials, OAuthError> {
        let mut response = self
            .agent
            .post(&self.token_url)
            .send_form(form.iter().copied())
            .map_err(|e| OAuthError::Network(e.to_string()))?;
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .with_config()
            .limit(MAX_BODY)
            .read_to_string()
            .map_err(|e| OAuthError::Network(e.to_string()))?;
        if status != 200 {
            let failure: Failure = serde_json::from_str(&body).unwrap_or_default();
            return Err(OAuthError::Rejected(TokenError {
                status,
                code: failure.error,
                description: failure.error_description,
            }));
        }
        let tokens: Tokens = serde_json::from_str(&body)
            .map_err(|_| OAuthError::Malformed("the answer is not JSON with tokens".into()))?;
        if tokens.access_token.is_empty() {
            return Err(OAuthError::Malformed(
                "the answer has no access_token".into(),
            ));
        }
        let mut updated = base;
        updated.access_token = tokens.access_token;
        if !tokens.refresh_token.is_empty() {
            updated.refresh_token = tokens.refresh_token;
        }
        if !tokens.scope.is_empty() {
            updated.scope = tokens.scope;
        }
        updated.expires_at_ms = self.clock.wall_ms() + tokens.expires_in * 1000;
        Ok(updated)
    }
}

/// How much of an answer is read.
const MAX_BODY: u64 = 64 * 1024;

#[derive(Default, Deserialize)]
struct Failure {
    #[serde(default)]
    error: String,
    #[serde(default)]
    error_description: String,
}

#[derive(Deserialize)]
struct Tokens {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    refresh_token: String,
    #[serde(default)]
    expires_in: i64,
    #[serde(default)]
    scope: String,
}

impl Default for OAuth {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::credentials::sample;
    use super::super::testing::{FakeServer, Reply};
    use super::*;
    use crate::metrics::ManualClock;

    fn oauth_for(server: &FakeServer, now_ms: i64) -> OAuth {
        OAuth::with_urls(
            "https://example.invalid/authorize",
            &format!("{}/token", server.url),
        )
        .with_clock(Arc::new(ManualClock::starting_at(now_ms)))
    }

    fn tokens(access: &str, refresh: Option<&str>, expires_in: i64, scope: Option<&str>) -> String {
        let mut json = serde_json::json!({"access_token": access, "expires_in": expires_in});
        if let Some(refresh) = refresh {
            json["refresh_token"] = refresh.into();
        }
        if let Some(scope) = scope {
            json["scope"] = scope.into();
        }
        json.to_string()
    }

    #[test]
    fn the_code_is_exchanged_with_every_field_the_server_needs() {
        let server = FakeServer::start(|_| {
            Reply::json(
                200,
                &tokens(
                    "A1",
                    Some("R1"),
                    3600,
                    Some("openid chatgpt.tokens.use.direct"),
                ),
            )
        });
        let credentials = oauth_for(&server, 1_000_000)
            .exchange_code(
                "client-9",
                "the-code",
                "the-verifier",
                "http://127.0.0.1:1455/auth/callback",
            )
            .unwrap();

        let request = &server.requests()[0];
        assert_eq!(
            (request.method.as_str(), request.path.as_str()),
            ("POST", "/token")
        );
        assert_eq!(
            request.header("content-type"),
            Some("application/x-www-form-urlencoded")
        );
        for (name, value) in [
            ("grant_type", "authorization_code"),
            ("client_id", "client-9"),
            ("code", "the-code"),
            ("code_verifier", "the-verifier"),
            ("redirect_uri", "http://127.0.0.1:1455/auth/callback"),
            ("resource", RESOURCE),
        ] {
            assert_eq!(request.form(name).as_deref(), Some(value), "{name}");
        }
        assert_eq!(credentials.client_id, "client-9");
        assert_eq!(credentials.access_token, "A1");
        assert_eq!(credentials.refresh_token, "R1");
        assert_eq!(credentials.scope, "openid chatgpt.tokens.use.direct");
        assert_eq!(credentials.expires_at_ms, 1_000_000 + 3_600_000);
    }

    #[test]
    fn a_refresh_sends_the_refresh_token_and_no_scope_and_keeps_what_the_server_does_not_return() {
        let server = FakeServer::start(|_| Reply::json(200, &tokens("A2", None, 60, None)));
        let current = sample(0);

        let refreshed = oauth_for(&server, 500).refresh(&current).unwrap();

        let request = &server.requests()[0];
        assert_eq!(request.form("grant_type").as_deref(), Some("refresh_token"));
        assert_eq!(request.form("client_id").as_deref(), Some("client-1"));
        assert_eq!(
            request.form("refresh_token").as_deref(),
            Some("REFRESH-SECRET")
        );
        assert_eq!(request.form("resource").as_deref(), Some(RESOURCE));
        assert_eq!(
            request.form("scope"),
            None,
            "sending a scope could narrow the rights"
        );
        assert_eq!(refreshed.access_token, "A2");
        assert_eq!(
            refreshed.refresh_token, "REFRESH-SECRET",
            "kept when not rotated"
        );
        assert_eq!(refreshed.scope, current.scope);
        assert_eq!(refreshed.host_id, current.host_id);
        assert_eq!(refreshed.client_id, current.client_id);
        assert_eq!(refreshed.expires_at_ms, 500 + 60_000);
    }

    #[test]
    fn a_rotated_refresh_token_replaces_the_old_one() {
        let server = FakeServer::start(|_| Reply::json(200, &tokens("A2", Some("R2"), 60, None)));
        assert_eq!(
            oauth_for(&server, 0)
                .refresh(&sample(0))
                .unwrap()
                .refresh_token,
            "R2"
        );
    }

    #[test]
    fn a_refusal_is_reported_with_its_code_and_is_final_for_a_dead_refresh_token() {
        let server = FakeServer::start(|_| {
            Reply::json(
                400,
                r#"{"error":"invalid_grant","error_description":"refresh token reused"}"#,
            )
        });

        let error = oauth_for(&server, 0).refresh(&sample(0)).unwrap_err();

        assert_eq!(
            error,
            OAuthError::Rejected(TokenError {
                status: 400,
                code: "invalid_grant".into(),
                description: "refresh token reused".into()
            })
        );
        assert!(error.is_terminal());
        assert!(!error.to_string().contains("REFRESH-SECRET"));
    }

    #[test]
    fn a_server_failure_is_not_final() {
        let server = FakeServer::start(|_| Reply::json(500, "oops"));
        let error = oauth_for(&server, 0).refresh(&sample(0)).unwrap_err();
        assert!(matches!(&error, OAuthError::Rejected(e) if e.status == 500));
        assert!(!error.is_terminal());
    }

    #[test]
    fn every_code_that_means_sign_in_again_is_final_and_others_are_not() {
        let rejected = |code: &str| {
            OAuthError::Rejected(TokenError {
                status: 400,
                code: code.into(),
                description: String::new(),
            })
        };
        for code in [
            "invalid_grant",
            "invalid_refresh_token",
            "token_expired",
            "refresh_token_expired",
            "refresh_token_invalidated",
            "refresh_token_reused",
        ] {
            assert!(rejected(code).is_terminal(), "{code}");
        }
        assert!(!rejected("temporarily_unavailable").is_terminal());
        assert!(!OAuthError::Network("down".into()).is_terminal());
        assert!(!OAuthError::Malformed("x".into()).is_terminal());
    }

    #[test]
    fn an_answer_without_an_access_token_or_not_json_is_malformed() {
        let server = FakeServer::start(|request| {
            if request.form("refresh_token").is_some() {
                Reply::json(200, r#"{"expires_in": 60}"#)
            } else {
                Reply::json(200, "not json at all")
            }
        });
        let oauth = oauth_for(&server, 0);
        assert!(matches!(
            oauth.refresh(&sample(0)),
            Err(OAuthError::Malformed(_))
        ));
        assert!(matches!(
            oauth.exchange_code("c", "code", "v", "http://127.0.0.1:1/cb"),
            Err(OAuthError::Malformed(_))
        ));
    }

    #[test]
    fn a_server_that_cannot_be_reached_is_a_network_error() {
        let oauth = OAuth::with_urls("https://example.invalid/a", "http://127.0.0.1:9/token");
        assert!(matches!(
            oauth.refresh(&sample(0)),
            Err(OAuthError::Network(_))
        ));
    }
}
