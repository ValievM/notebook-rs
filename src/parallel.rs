//! Fork-join for a short list of independent jobs: every job gets its own scoped thread, the
//! results come back in the order of the jobs, and a panic in a job is re-raised in the caller.
//!
//! One thread per job is fine here because the lists are short by construction (a handful of
//! retrievers, at most five planned searches). It is not a thread pool: do not hand it a
//! list whose length comes from user data.

pub fn map_ordered<T, R>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .iter()
            .map(|item| scope.spawn(|| work(item)))
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    })
}

/// Test helper: proves that jobs really overlap in time, without sleeping or measuring.
#[cfg(test)]
pub(crate) mod testing {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// Each job calls `meet`; it returns true only if all `expected` jobs are inside it at the
    /// same moment. Run one after another, the first job times out and gets false.
    pub struct Rendezvous {
        arrived: AtomicUsize,
        expected: usize,
    }

    impl Rendezvous {
        pub fn new(expected: usize) -> Self {
            Self {
                arrived: AtomicUsize::new(0),
                expected,
            }
        }

        pub fn meet(&self) -> bool {
            self.arrived.fetch_add(1, Ordering::SeqCst);
            let deadline = Instant::now() + Duration::from_secs(5);
            while self.arrived.load(Ordering::SeqCst) < self.expected {
                if Instant::now() > deadline {
                    return false;
                }
                std::thread::yield_now();
            }
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::Rendezvous;
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn jobs_run_at_the_same_time() {
        let rendezvous = Rendezvous::new(3);
        let met = map_ordered(&[1, 2, 3], |_| rendezvous.meet());
        assert_eq!(met, vec![true, true, true]);
    }

    #[test]
    fn results_keep_the_order_of_the_jobs_even_when_a_later_job_finishes_first() {
        let second_done = AtomicBool::new(false);
        let results = map_ordered(&[0, 1], |job| {
            if *job == 0 {
                // The first job waits (at most five seconds) until the second has finished.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while !second_done.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
                    std::thread::yield_now();
                }
                ("first", second_done.load(Ordering::SeqCst))
            } else {
                second_done.store(true, Ordering::SeqCst);
                ("second", true)
            }
        });
        assert_eq!(results, vec![("first", true), ("second", true)]);
    }

    #[test]
    fn an_empty_list_gives_an_empty_result() {
        let results: Vec<i32> = map_ordered(&[] as &[i32], |x| *x);
        assert!(results.is_empty());
    }

    #[test]
    #[should_panic(expected = "job three failed")]
    fn a_panic_in_a_job_reaches_the_caller() {
        map_ordered(&[1, 2, 3], |job| {
            if *job == 3 {
                panic!("job three failed");
            }
        });
    }
}
