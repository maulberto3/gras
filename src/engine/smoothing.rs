//! Fitness smoothing: the rolling buffer, its mean, and the window it uses.
//!
//! Every ranking decision in the engine reads *smoothed* fitness, not the raw
//! per-step value — raw values bounce with batch difficulty, the trend is what
//! matters. The consumers are the current best/worst net (cull victim
//! selection, crossover gate replacement), the roulette parent ranking, the
//! checkpoint ledger's `pop_mean_fitness`, and the `RaceSnapshot` handed to a
//! custom stop closure.

/// A small fixed-capacity rolling buffer — the in-memory equivalent of reading
/// back the last K `NetMetrics` snapshots from a net's state file.
#[derive(Clone, Debug, Default)]
pub struct RollingBuffer {
    inner: std::collections::VecDeque<f32>,
    cap: usize,
}

impl RollingBuffer {
    pub fn new(cap: usize) -> Self {
        RollingBuffer {
            inner: std::collections::VecDeque::with_capacity(cap),
            cap,
        }
    }
    /// A buffer pre-filled with the given values (capped at `cap`, oldest
    /// dropped first). Test convenience: fixtures push a whole history at
    /// once instead of looping `push`.
    pub fn push(&mut self, v: f32) {
        self.inner.push_back(v);
        if self.inner.len() > self.cap {
            self.inner.pop_front();
        }
    }
    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, f32> {
        self.inner.iter()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn clear(&mut self) {
        self.inner.clear();
    }
    /// Pre-filled constructor: `RollingBuffer::from_values(10, &[0.5, 1.0])`.
    pub fn from_values(cap: usize, values: &[f32]) -> Self {
        let mut buf = RollingBuffer::new(cap);
        for &v in values {
            buf.push(v);
        }
        buf
    }
}

/// Rolling mean of a buffer's values. Returns 0.0 for an empty buffer.
pub fn rolling_mean(buf: &RollingBuffer) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let sum: f32 = buf.iter().sum();
    sum / buf.len() as f32
}

/// The per-net smoothing window, in steps (K).
pub use crate::engine::config::SMOOTHING_WINDOW;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_mean_empty_is_zero() {
        let buf = RollingBuffer::new(5);
        assert_eq!(rolling_mean(&buf), 0.0);
    }

    #[test]
    fn rolling_mean_is_windowed() {
        let mut buf = RollingBuffer::new(3);
        for v in [1.0, 2.0, 3.0, 4.0] {
            buf.push(v);
        }
        // Only the last 3 values remain: (2+3+4)/3 = 3.
        assert_eq!(rolling_mean(&buf), 3.0);
    }
}
