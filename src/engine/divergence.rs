//! Divergence machinery: the rolling fitness buffer, its mean, and the
//! built-in divergence policy a pluggable closure can replace.


/// A small fixed-capacity rolling buffer — the in-memory equivalent of reading
/// back the last K `NetMetrics` snapshots from a net's state file. Used only
/// for divergence smoothing.
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
}

/// Rolling mean of a buffer's values. Returns 0.0 for an empty buffer.
pub fn rolling_mean(buf: &RollingBuffer) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let sum: f32 = buf.iter().sum();
    sum / buf.len() as f32
}

/// Per-step divergence computation (D3). Rolling mean K=10 over each net's
/// per-step fitness, then `(max-min)/max` over the live population.
pub use crate::engine::config::DIVERGENCE_WINDOW;

/// The built-in divergence policy: `(max - min) / max` over the population's
/// smoothed fitness — bounded in [0, 1] (0 = identical population, 1 = worst
/// member at zero and the best positive). 0 when the slice is empty or max is
/// non-positive. This is the policy a configured `divergence_fn` closure
/// replaces.
pub fn built_in_divergence(smoothed: &[f32]) -> f32 {
    if smoothed.is_empty() {
        return 0.0;
    }
    let max = smoothed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = smoothed.iter().cloned().fold(f32::INFINITY, f32::min);
    if max <= 0.0 {
        return 0.0;
    }
    (max - min) / max
}

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

    #[test]
    fn built_in_divergence_edges() {
        assert_eq!(built_in_divergence(&[]), 0.0);
        assert_eq!(built_in_divergence(&[0.0, 0.0]), 0.0);
        // Spread relative to the (highest) leader: 0.6/0.8 = 0.75, bounded ≤ 1.
        assert!((built_in_divergence(&[0.2, 0.8]) - 0.75).abs() < 1e-6);
    }
}
