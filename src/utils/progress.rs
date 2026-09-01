//! Progress tracking — atomic counter for parallel evaluation.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Atomic counter for tracking evaluation progress across rayon workers.
pub(crate) struct ProgressTracker {
    done: AtomicUsize,
}

impl ProgressTracker {
    pub(crate) fn new() -> Self {
        ProgressTracker {
            done: AtomicUsize::new(0),
        }
    }

    /// Workers call this after scoring. Just counts — no printing.
    pub(crate) fn increment(&self) {
        self.done.fetch_add(1, Ordering::Relaxed);
    }

    /// Final count (no-op — logging handled elsewhere).
    pub(crate) fn finish(&self) {}
}
