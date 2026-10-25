use super::PLAN_SCOPE;
use serde::{Deserialize, Serialize};

/// What the sign-in leaves behind. Secret: never printed, never logged.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Credentials {
    pub client_id: String,
    /// Names this machine to the server (`urn:uuid:...`), kept across sign-ins.
    pub host_id: String,
    pub access_token: String,
    pub refresh_token: String,
    /// When the access token stops working, in milliseconds since the Unix epoch.
    pub expires_at_ms: i64,
    /// The rights that were granted, separated by spaces.
    pub scope: String,
}

impl Credentials {
    /// Whether the person allowed spending the subscription.
    pub fn has_plan_scope(&self) -> bool {
        self.scope
            .split_whitespace()
            .any(|right| right == PLAN_SCOPE)
    }

    /// Whether the token has expired or will within `window_ms`.
    pub fn expires_within(&self, now_ms: i64, window_ms: i64) -> bool {
        now_ms + window_ms >= self.expires_at_ms
    }
}

/// The tokens must never reach a log or an error message by accident.
impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("client_id", &self.client_id)
            .field("host_id", &self.host_id)
            .field("expires_at_ms", &self.expires_at_ms)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
pub(crate) fn sample(expires_at_ms: i64) -> Credentials {
    Credentials {
        client_id: "client-1".into(),
        host_id: "urn:uuid:11111111-1111-4111-8111-111111111111".into(),
        access_token: "ACCESS-SECRET".into(),
        refresh_token: "REFRESH-SECRET".into(),
        expires_at_ms,
        scope: format!("openid offline_access {PLAN_SCOPE}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_scope_must_be_among_the_granted_rights() {
        assert!(sample(0).has_plan_scope());
        let mut without = sample(0);
        without.scope = "openid offline_access".into();
        assert!(!without.has_plan_scope());
        // A right that merely contains the name is not the right.
        without.scope = format!("x{PLAN_SCOPE} {PLAN_SCOPE}y");
        assert!(!without.has_plan_scope());
    }

    #[test]
    fn a_token_is_due_when_it_expires_within_the_window_or_already_has() {
        let credentials = sample(10_000);
        assert!(
            !credentials.expires_within(0, 9_999),
            "still 10 s left, window is shorter"
        );
        assert!(
            credentials.expires_within(0, 10_000),
            "exactly at the edge counts as due"
        );
        assert!(credentials.expires_within(0, 60_000));
        assert!(credentials.expires_within(20_000, 0), "already expired");
    }

    #[test]
    fn debug_output_never_shows_the_tokens() {
        let shown = format!("{:?}", sample(1));
        assert!(
            !shown.contains("ACCESS-SECRET") && !shown.contains("REFRESH-SECRET"),
            "{shown}"
        );
        assert!(shown.contains("client-1"), "{shown}");
    }

    #[test]
    fn credentials_survive_a_json_round_trip() {
        let original = sample(123);
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(
            serde_json::from_str::<Credentials>(&json).unwrap(),
            original
        );
    }
}
