use super::{Credentials, OAuth, REFRESH_WINDOW_MS, Store};
use crate::error::{Error, Result};
use crate::metrics::Clock;
use std::sync::{Arc, Mutex};

const LOCK: &str = "the refresh lock is never held across a panic";

/// Hands out a valid access token and refreshes it by itself.
///
/// The refresh token is single-use: the server replaces it with every refresh. Two refreshes at
/// once would spend the same token twice and lose the sign-in, so refreshing happens under a
/// lock that is held while the server is called. This is a deliberate exception to the rule of
/// never holding a lock across an outside call: here the waiting is the point.
pub struct TokenSource {
    store: Arc<dyn Store>,
    oauth: Arc<OAuth>,
    clock: Arc<dyn Clock>,
    /// Held for the whole of a refresh, so that refreshes follow one another.
    refreshing: Mutex<()>,
}

impl TokenSource {
    pub fn new(store: Arc<dyn Store>, oauth: Arc<OAuth>, clock: Arc<dyn Clock>) -> Self {
        Self {
            store,
            oauth,
            clock,
            refreshing: Mutex::new(()),
        }
    }

    /// Whether a usable sign-in is saved (read from the store each time, so a sign-in made by
    /// another process is seen at once).
    pub fn is_signed_in(&self) -> bool {
        self.store
            .load()
            .is_ok_and(|credentials| credentials.has_plan_scope())
    }

    /// Saves the credentials of a fresh sign-in.
    pub fn sign_in(&self, credentials: Credentials) -> Result<()> {
        let _guard = self.refreshing.lock().expect(LOCK);
        self.store.save(&credentials)
    }

    /// An access token that works now. The new pair of a refresh is saved before the new token
    /// is handed out.
    pub fn token(&self) -> Result<String> {
        let _guard = self.refreshing.lock().expect(LOCK);
        // Read from the store every time: a sign-in or a refresh made by another process (the
        // window and the command line share the file) is seen at once.
        let credentials = self.store.load()?;
        if !credentials.has_plan_scope() {
            return Err(Error::SignIn(
                "this sign-in does not allow using the ChatGPT plan: run `nb login` and allow it"
                    .into(),
            ));
        }
        if !credentials.expires_within(self.clock.wall_ms(), REFRESH_WINDOW_MS) {
            return Ok(credentials.access_token);
        }
        let refreshed = match self.oauth.refresh(&credentials) {
            Ok(refreshed) => refreshed,
            Err(error) if error.is_terminal() => {
                // Another process may have refreshed in the meantime; its pair is the good one.
                if let Ok(stored) = self.store.load()
                    && stored != credentials
                    && !stored.expires_within(self.clock.wall_ms(), REFRESH_WINDOW_MS)
                {
                    return Ok(stored.access_token);
                }
                return Err(Error::SignIn(format!(
                    "the sign-in has expired ({error}): run `nb login` again"
                )));
            }
            Err(error) => {
                return Err(Error::ModelUnavailable(format!(
                    "could not refresh the ChatGPT token: {error}"
                )));
            }
        };
        // The new pair goes to the disk before anyone uses it: the refresh token has changed,
        // and a pair that was only in memory would be lost with the process.
        self.store.save(&refreshed)?;
        Ok(refreshed.access_token)
    }
}

#[cfg(test)]
mod tests {
    use super::super::credentials::sample;
    use super::super::testing::{FakeServer, Reply};
    use super::*;
    use crate::metrics::ManualClock;

    /// A store in memory; it can be told to fail on save.
    struct MemoryStore {
        saved: Mutex<Option<Credentials>>,
        saves: Mutex<usize>,
        fail_save: bool,
    }

    impl MemoryStore {
        fn with(credentials: Option<Credentials>) -> Arc<Self> {
            Arc::new(Self {
                saved: Mutex::new(credentials),
                saves: Mutex::new(0),
                fail_save: false,
            })
        }

        fn failing_on_save(credentials: Credentials) -> Arc<Self> {
            Arc::new(Self {
                saved: Mutex::new(Some(credentials)),
                saves: Mutex::new(0),
                fail_save: true,
            })
        }
    }

    impl Store for MemoryStore {
        fn load(&self) -> Result<Credentials> {
            self.saved
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| Error::SignIn("not signed in yet: run `nb login`".into()))
        }

        fn save(&self, credentials: &Credentials) -> Result<()> {
            if self.fail_save {
                return Err(Error::Io(std::io::Error::other("disk full")));
            }
            *self.saves.lock().unwrap() += 1;
            *self.saved.lock().unwrap() = Some(credentials.clone());
            Ok(())
        }
    }

    const NOW: i64 = 1_000_000;

    fn source_for(server: &FakeServer, store: Arc<MemoryStore>) -> TokenSource {
        let clock = Arc::new(ManualClock::starting_at(NOW));
        let oauth = OAuth::with_urls(
            "https://example.invalid/a",
            &format!("{}/token", server.url),
        )
        .with_clock(clock.clone());
        TokenSource::new(store, Arc::new(oauth), clock)
    }

    fn refreshing_server() -> FakeServer {
        FakeServer::start(|_| {
            Reply::json(
                200,
                r#"{"access_token":"NEW-ACCESS","refresh_token":"NEW-REFRESH","expires_in":3600}"#,
            )
        })
    }

    #[test]
    fn a_token_that_is_still_good_is_handed_out_without_asking_the_server() {
        let server = refreshing_server();
        let store = MemoryStore::with(Some(sample(NOW + 10 * 60_000)));

        let token = source_for(&server, store).token().unwrap();

        assert_eq!(token, "ACCESS-SECRET");
        assert!(server.requests().is_empty());
    }

    #[test]
    fn a_token_about_to_expire_is_refreshed_and_the_new_pair_is_saved() {
        let server = refreshing_server();
        let store = MemoryStore::with(Some(sample(NOW + 60_000))); // one minute left, window is two
        let source = source_for(&server, store.clone());

        assert_eq!(source.token().unwrap(), "NEW-ACCESS");

        let saved = store.load().unwrap();
        assert_eq!(
            (saved.access_token.as_str(), saved.refresh_token.as_str()),
            ("NEW-ACCESS", "NEW-REFRESH")
        );
        assert_eq!(saved.expires_at_ms, NOW + 3_600_000);
        assert_eq!(source.token().unwrap(), "NEW-ACCESS");
        assert_eq!(
            server.requests().len(),
            1,
            "the next call uses the refreshed token"
        );
    }

    #[test]
    fn if_the_new_pair_cannot_be_saved_the_new_token_is_not_handed_out() {
        let server = refreshing_server();
        let store = MemoryStore::failing_on_save(sample(NOW + 1_000));
        let source = source_for(&server, store);

        assert!(source.token().is_err());
    }

    #[test]
    fn many_threads_asking_for_an_expired_token_cause_exactly_one_refresh() {
        let server = FakeServer::start(|_| {
            // Slow enough that the other threads arrive while this refresh is under way.
            std::thread::sleep(std::time::Duration::from_millis(150));
            Reply::json(
                200,
                r#"{"access_token":"NEW-ACCESS","refresh_token":"NEW-REFRESH","expires_in":3600}"#,
            )
        });
        let source = source_for(&server, MemoryStore::with(Some(sample(NOW - 1))));

        let tokens: Vec<String> =
            crate::parallel::map_ordered(&[0; 8], |_| source.token().unwrap());

        assert!(tokens.iter().all(|t| t == "NEW-ACCESS"), "{tokens:?}");
        assert_eq!(
            server.requests().len(),
            1,
            "a second refresh would spend the same refresh token twice"
        );
    }

    #[test]
    fn a_dead_refresh_token_asks_for_a_new_sign_in_and_touches_nothing() {
        let server = FakeServer::start(|_| Reply::json(400, r#"{"error":"invalid_grant"}"#));
        let store = MemoryStore::with(Some(sample(NOW - 1)));

        let error = source_for(&server, store.clone()).token().unwrap_err();

        assert!(matches!(error, Error::SignIn(_)), "{error}");
        assert!(error.to_string().contains("nb login"), "{error}");
        assert_eq!(*store.saves.lock().unwrap(), 0);
    }

    #[test]
    fn if_another_process_refreshed_first_its_new_pair_is_used_instead_of_asking_to_sign_in() {
        // The window and the command line share the file. When this process asks to refresh with
        // a refresh token that the other one has just spent, the server refuses, but the file
        // already holds the new pair: that is the good one.
        let store = MemoryStore::with(Some(sample(NOW - 1)));
        let other_process = store.clone();
        let server = FakeServer::start(move |_| {
            let mut fresh = sample(NOW + 3_600_000);
            fresh.access_token = "OTHER-PROCESS-ACCESS".into();
            fresh.refresh_token = "OTHER-PROCESS-REFRESH".into();
            other_process.save(&fresh).unwrap();
            Reply::json(
                400,
                r#"{"error":"invalid_grant","error_description":"already used"}"#,
            )
        });

        let token = source_for(&server, store).token().unwrap();

        assert_eq!(token, "OTHER-PROCESS-ACCESS");
    }

    #[test]
    fn a_server_failure_is_not_taken_for_a_lost_sign_in() {
        let server = FakeServer::start(|_| Reply::json(503, "later"));
        let error = source_for(&server, MemoryStore::with(Some(sample(NOW - 1))))
            .token()
            .unwrap_err();
        assert!(matches!(error, Error::ModelUnavailable(_)), "{error}");
    }

    #[test]
    fn a_sign_in_without_the_plan_right_is_refused() {
        let server = refreshing_server();
        let mut credentials = sample(NOW + 600_000);
        credentials.scope = "openid".into();
        let error = source_for(&server, MemoryStore::with(Some(credentials)))
            .token()
            .unwrap_err();
        assert!(matches!(error, Error::SignIn(_)), "{error}");
    }

    #[test]
    fn nobody_signed_in_means_a_request_to_sign_in() {
        let server = refreshing_server();
        let error = source_for(&server, MemoryStore::with(None))
            .token()
            .unwrap_err();
        assert!(matches!(error, Error::SignIn(_)), "{error}");
    }

    #[test]
    fn being_signed_in_is_read_from_the_store_each_time() {
        let server = refreshing_server();
        let store = MemoryStore::with(None);
        let source = source_for(&server, store.clone());
        assert!(!source.is_signed_in());

        store.save(&sample(NOW + 600_000)).unwrap();

        assert!(source.is_signed_in());
    }

    #[test]
    fn a_fresh_sign_in_is_saved_and_used_at_once() {
        let server = refreshing_server();
        let store = MemoryStore::with(None);
        let source = source_for(&server, store.clone());

        source.sign_in(sample(NOW + 600_000)).unwrap();

        assert_eq!(store.load().unwrap(), sample(NOW + 600_000));
        assert_eq!(source.token().unwrap(), "ACCESS-SECRET");
    }
}
