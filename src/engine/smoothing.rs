//! Fitness smoothing: the rolling buffer, its mean, its spread, and the window
//! it uses.
//!
//! Every ranking decision in the engine reads *smoothed* fitness, not the raw
//! per-step value — raw values bounce with batch difficulty, the trend is what
//! matters. Since FIT the ranking value is not the mean but the **risk-adjusted
//! score**: the window mean shifted by `λ` window standard deviations toward the
//! bad side (see [`Direction::risk_adjusted`]), so a steady net outranks a
//! volatile one with the same mean. The readable mean and its spread travel
//! alongside it (logs, `history.csv`, `engine.json`). The consumers are the
//! current best/worst net (cull victim selection), the roulette parent ranking,
//! the elite/champion set, and the `RaceSnapshot` handed to a custom stop
//! closure.
//!
//! Only the arithmetic mean is kept here: the geometric / harmonic family was
//! removed with FIT. Both punish the LOW tail, which is the risk-averse choice
//! for a Maximize fitness but the risk-SEEKING one for a Minimize fitness (its
//! low readings are the lucky ones), so neither is a direction-symmetric
//! objective. The spread term is symmetric, which is why one λ covers both
//! directions.

use crate::engine::fitness::Direction;

/// A small fixed-capacity rolling buffer — the in-memory equivalent of reading
/// back the last K `NetMetrics` snapshots from a net's state file.
#[derive(Clone, Debug, Default)]
pub struct RollingBuffer {
    inner: std::collections::VecDeque<f32>,
    cap: usize,
}

impl RollingBuffer {
    /// New buffer keeping at most `cap` readings (oldest dropped first).
    pub fn new(cap: usize) -> Self {
        RollingBuffer {
            inner: std::collections::VecDeque::with_capacity(cap),
            cap,
        }
    }
    /// Push one reading, dropping the oldest past `cap`.
    pub fn push(&mut self, v: f32) {
        self.inner.push_back(v);
        if self.inner.len() > self.cap {
            self.inner.pop_front();
        }
    }
    /// Iterate the buffered readings, oldest first.
    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, f32> {
        self.inner.iter()
    }
    /// True when no reading has been pushed.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    /// Number of buffered readings (at most `cap`).
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    /// Drop every buffered reading.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
        /// The most recent value pushed (the net's LATEST reading); the window
        /// mean lags it by up to `cap` steps — the guardrail prints both.
    pub fn last(&self) -> Option<f32> {
        self.inner.back().copied()
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
///
/// This is the READABLE window value ("the turns it survives"), reported as `μ`
/// in the logs and JSON. It is also the center the risk-adjusted score starts
/// from — ranking itself reads [`WindowStats::risk_adjusted`], not this.
pub fn rolling_mean(buf: &RollingBuffer) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let sum: f32 = buf.iter().sum();
    sum / buf.len() as f32
}

/// A smoothed reading WITH its own error bar: how many readings it averages and
/// how much they spread — the race's number is a mean of K small samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowStats {
    /// Readings behind `mean` (buffer length, ≤ the smoothing window).
    pub n: usize,
        /// Their mean — the readable smoothed level, and the center the ranking score is
        /// computed from. Use [`WindowStats::risk_adjusted`] for the value ranking uses.
    pub mean: f32,
    /// Population std of the readings: their step-to-step spread (`σ` in the logs).
    pub std: f32,
        /// Extreme readings in the window — the readable form of `std` (e.g. one
        /// 500 among fizzles).
    pub min: f32,
    /// Largest reading in the window.
    pub max: f32,
    /// The buffer's most recent value (see [`RollingBuffer::last`]): the reading
    /// that describes the weights the run actually ended with.
    pub latest: f32,
}

impl WindowStats {
        /// Standard error of `mean` (`std / √n`). `None` for zero readings;
        /// `Some(0.0)` for one reading (unobservable spread, not precision).
    pub fn se(&self) -> Option<f32> {
        (self.n > 0).then(|| self.std / (self.n as f32).sqrt())
    }

        /// The RISK-ADJUSTED score (FIT) this window ranks by: the mean shifted `lambda`
        /// window-stds toward the bad side for `direction`. Thin wrapper over
        /// [`Direction::risk_adjusted`] so the fitness units, the direction rule and the
        /// spread all stay in ONE place; `lambda = 0.0` returns `mean` exactly.
        ///
        /// A single-reading window has `std = 0`, so it scores as its own reading — a lucky
        /// first step looks perfectly safe. The guard against that is an eligibility rule
        /// (minimum readings before ranking), not this formula.
    pub fn risk_adjusted(&self, direction: Direction, lambda: f32) -> f32 {
        direction.risk_adjusted(self.mean, self.std, lambda)
    }
}

/// Stats of a rolling buffer: `None` when it is empty (a net that was never
/// measured has no reading, and must not read as a confident 0.0).
pub fn rolling_stats(buf: &RollingBuffer) -> Option<WindowStats> {
    if buf.is_empty() {
        return None;
    }
    let n = buf.len();
    let mean = rolling_mean(buf);
    let var = buf
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n as f32;
    Some(WindowStats {
        n,
        mean,
        std: var.sqrt(),
        min: buf.iter().copied().fold(f32::INFINITY, f32::min),
        max: buf.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        latest: buf.last().unwrap_or(mean),
    })
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

    #[test]
    fn window_stats_risk_adjusted_reads_the_window() {
        // The bimodal champion: one lucky 500 among four fizzles. The mean is a
        // readable 132, the spread is 184 — the score must charge for it.
        let buf = RollingBuffer::from_values(5, &[40.0, 500.0, 40.0, 40.0, 40.0]);
        let s = rolling_stats(&buf).unwrap();
        assert_eq!(s.mean, 132.0);
        assert!((s.risk_adjusted(Direction::Maximize, 0.0) - 132.0).abs() < 1e-4);
        let scored = s.risk_adjusted(Direction::Maximize, 1.0);
        assert!(
            scored < s.mean,
            "a maximizer pays for its spread: {scored} must be under {}",
            s.mean
        );
        // Minimize: the same spread is added, never subtracted.
        let scored_min = s.risk_adjusted(Direction::Minimize, 1.0);
        assert!(scored_min > s.mean);
        // Both arms pay the SAME λσ; they differ only in which side of the mean
        // they are on.
        let (paid_hi, paid_lo) = (s.mean - scored, scored_min - s.mean);
        assert!(
            (paid_hi - paid_lo).abs() < 1e-4,
            "one λ covers both directions: paid {paid_hi} vs {paid_lo}"
        );
    }

    #[test]
    fn window_stats_risk_adjusted_keeps_a_steady_window_on_top() {
        // Same mean, different spread: the steady window must score better in BOTH
        // directions — the whole point of FIT.
        let steady = rolling_stats(&RollingBuffer::from_values(4, &[300.0; 4])).unwrap();
        let volatile =
            rolling_stats(&RollingBuffer::from_values(4, &[250.0, 350.0, 250.0, 350.0])).unwrap();
        assert_eq!(steady.mean, volatile.mean);
        for dir in [Direction::Maximize, Direction::Minimize] {
            let a = steady.risk_adjusted(dir, 1.0);
            let b = volatile.risk_adjusted(dir, 1.0);
            assert!(
                dir.is_better(a, b),
                "{dir:?}: steady {a} must beat volatile {b}"
            );
        }
    }

    #[test]
    fn rolling_stats_empty_is_none() {
        assert_eq!(rolling_stats(&RollingBuffer::new(5)), None);
    }

    #[test]
    fn rolling_stats_reports_spread_and_se() {
        // The race window of a bimodal champion: one fluke solve among fizzles.
        let buf = RollingBuffer::from_values(5, &[40.0, 500.0, 40.0, 40.0, 40.0]);
        let s = rolling_stats(&buf).unwrap();
        assert_eq!(s.n, 5);
        assert_eq!(s.mean, 132.0);
        assert_eq!((s.min, s.max), (40.0, 500.0));
        // The latest value is the LAST pushed one (40.0 here), not the fluke.
        assert_eq!(s.latest, 40.0);
        // Population std: sqrt((4·92² + 368²)/5) = 184 — the deviations are
        // −92 (×4) and +368 (×1), so one solve dominates the spread.
        assert!((s.std - 184.0).abs() < 0.1, "std was {}", s.std);
            // SE = std/√n ≈ 82.3 — the error bar that makes this 132-mean race
            // reading look far shakier than a 100-game exam's ±2.
        let se = s.se().unwrap();
        assert!((se - s.std / 5.0_f32.sqrt()).abs() < 1e-4, "se was {se}");
    }

    #[test]
    fn rolling_stats_single_reading_has_no_observable_spread() {
        let buf = RollingBuffer::from_values(5, &[300.0]);
        let s = rolling_stats(&buf).unwrap();
        assert_eq!((s.n, s.mean, s.std), (1, 300.0, 0.0));
        assert_eq!((s.min, s.max), (300.0, 300.0));
        assert_eq!(s.latest, 300.0);
        // A zero SE here is "only one reading", not "perfectly precise" — the
        // verdict's noisy branch is the caller that must not trust it.
        assert_eq!(s.se(), Some(0.0));
    }
}
