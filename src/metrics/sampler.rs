//! A background thread that records the memory of the process at a fixed interval, for as long
//! as it lives: the memory samples taken after an operation show the steps, these show the
//! trend (does it grow while the window sits open?).

use super::Metrics;
use crate::error::Result;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

/// Records `memory.periodic` immediately and then every `every`, until dropped.
pub struct MemorySampler {
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl MemorySampler {
    pub fn start(metrics: Metrics, every: Duration) -> Result<Self> {
        // Waiting on a channel instead of sleeping: closing the channel wakes the thread at once.
        let (stop, stopped) = mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("memory-sampler".into())
            .spawn(move || {
                loop {
                    metrics.sample_memory("periodic");
                    if !matches!(stopped.recv_timeout(every), Err(RecvTimeoutError::Timeout)) {
                        break;
                    }
                }
            })?;
        Ok(Self {
            stop: Some(stop),
            thread: Some(thread),
        })
    }
}

impl Drop for MemorySampler {
    /// Stops the thread at once (it does not finish waiting for the next sample) and joins it.
    fn drop(&mut self) {
        self.stop.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{MemoryRecorder, SystemClock};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    /// Waits (at most five seconds) until `condition` holds, without sleeping a fixed time.
    fn wait_until(condition: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition() {
            if Instant::now() > deadline {
                return false;
            }
            std::thread::yield_now();
        }
        true
    }

    fn metrics_with(probe: fn() -> Option<u64>) -> (Metrics, Arc<MemoryRecorder>) {
        let recorder = Arc::new(MemoryRecorder::default());
        let metrics = Metrics::with_clock(recorder.clone(), Arc::new(SystemClock::default()))
            .with_memory_probe(probe);
        (metrics, recorder)
    }

    #[test]
    fn samples_come_at_the_interval_and_stop_when_the_sampler_is_dropped() {
        static READS: AtomicUsize = AtomicUsize::new(0);
        fn probe() -> Option<u64> {
            Some(1_000 + READS.fetch_add(1, Ordering::SeqCst) as u64)
        }
        let (metrics, recorder) = metrics_with(probe);

        let sampler = MemorySampler::start(metrics, Duration::from_millis(5)).unwrap();
        assert!(
            wait_until(|| recorder.events().len() >= 3),
            "no samples came"
        );
        drop(sampler);

        let events = recorder.events();
        assert!(
            events
                .iter()
                .all(|e| e.kind == "memory.periodic" && e.micros.is_none())
        );
        // The probe is read once per sample, in order.
        let values: Vec<i64> = events.iter().map(|e| e.fields[0].1).collect();
        assert_eq!(values[..3], [1_000, 1_001, 1_002]);
        let kept = events.len();
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(
            recorder.events().len(),
            kept,
            "the thread kept sampling after the drop"
        );
    }

    #[test]
    fn the_first_sample_is_taken_at_once_not_after_the_first_interval() {
        fn probe() -> Option<u64> {
            Some(7)
        }
        let (metrics, recorder) = metrics_with(probe);

        let _sampler = MemorySampler::start(metrics, Duration::from_secs(3600)).unwrap();

        assert!(
            wait_until(|| recorder.events().len() == 1),
            "the first sample did not come"
        );
    }

    #[test]
    fn dropping_does_not_wait_for_the_next_sample() {
        fn probe() -> Option<u64> {
            Some(1)
        }
        let (metrics, recorder) = metrics_with(probe);
        let sampler = MemorySampler::start(metrics, Duration::from_secs(3600)).unwrap();
        assert!(wait_until(|| !recorder.events().is_empty()));

        let started = Instant::now();
        drop(sampler);

        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_sample_in_progress_is_finished_before_the_drop_returns() {
        use std::sync::atomic::AtomicBool;
        static STARTED: AtomicBool = AtomicBool::new(false);
        fn slow_probe() -> Option<u64> {
            STARTED.store(true, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(60));
            Some(1)
        }
        let (metrics, recorder) = metrics_with(slow_probe);
        let sampler = MemorySampler::start(metrics, Duration::from_secs(3600)).unwrap();
        assert!(wait_until(|| STARTED.load(Ordering::SeqCst)));

        drop(sampler);

        // The reading that was under way when the drop came is in; nothing comes after.
        assert_eq!(recorder.events().len(), 1);
        std::thread::sleep(Duration::from_millis(120));
        assert_eq!(recorder.events().len(), 1);
    }

    #[test]
    fn a_memory_that_cannot_be_read_records_nothing_and_the_sampler_goes_on() {
        static READS: AtomicUsize = AtomicUsize::new(0);
        fn probe() -> Option<u64> {
            // Fails on the first reads, then works.
            (READS.fetch_add(1, Ordering::SeqCst) >= 3).then_some(5)
        }
        let (metrics, recorder) = metrics_with(probe);

        let _sampler = MemorySampler::start(metrics, Duration::from_millis(5)).unwrap();

        assert!(wait_until(|| !recorder.events().is_empty()));
        assert!(READS.load(Ordering::SeqCst) >= 4);
    }
}
