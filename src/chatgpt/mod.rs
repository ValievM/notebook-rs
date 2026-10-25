//! Using the ChatGPT subscription as the model, with the person's own sign-in, following
//! OpenAI's "Sign in with ChatGPT" flow for open-source projects: OAuth with PKCE in the
//! browser, a token that spends the person's plan (`chatgpt.tokens.use.direct`), and the
//! Responses API with `store: false` and `stream: true`.
//!
//! The program never sees a password. The token lives in one file that only the owner can
//! read, and it is refreshed by itself.
//!
//! This is a port of the sign-in that the owner's family-expense bot used before it was
//! removed from that project (see DECISIONS D-16).

mod account;
mod auto;
mod credentials;
mod login;
mod oauth;
mod responses;
mod store;
mod token_source;

#[cfg(test)]
mod testing;

pub use account::ChatGptAccount;
pub use auto::AutoLlm;
pub use credentials::Credentials;
pub use login::{DEFAULT_PORT, LoginOptions, login, open_in_browser};
pub use oauth::{OAuth, OAuthError, TokenError};
pub use responses::{ChatGptLlm, ModelEntry};
pub use store::{FileStore, Store};
pub use token_source::TokenSource;

/// The right to spend the subscription on model calls.
pub const PLAN_SCOPE: &str = "chatgpt.tokens.use.direct";
/// What the token is for.
pub const RESOURCE: &str = "https://api.openai.com/v1";
/// Used for the first sign-in only: the server then issues the client id that is kept.
pub const DYNAMIC_CLIENT_ID: &str = "dynamic_agent_client";
pub const AUTHORIZE_URL: &str = "https://auth.openai.com/api/accounts/authorize";
pub const TOKEN_URL: &str = "https://auth.openai.com/api/accounts/oauth/token";
/// A token is refreshed this long before it expires.
pub const REFRESH_WINDOW_MS: i64 = 2 * 60 * 1000;
/// The address of the API the token is for.
pub const API_BASE: &str = "https://api.openai.com/v1";
