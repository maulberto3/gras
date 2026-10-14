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
    /// New buffer keeping at most `cap` readings (oldest dropped first).
    pub fn new(cap: usize) -> Self {
        RollingBuffer {
            inner: std::collections::VecDeque::with_capacity(cap),
            cap,
        }
    }
        /// A buffer pre-filled with `values` (capped at `cap`); test convenience.
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
pub fn rolling_mean(buf: &RollingBuffer) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let sum: f32 = buf.iter().sum();
    sum / buf.len() as f32
}

/// Geometric mean of a buffer's values. For a maximizer's readings this is the
/// "count the low readings" mean: it shrinks the influence of lucky high spikes.
pub fn rolling_geometric_mean(buf: &RollingBuffer, floor: f32) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let n = buf.len() as f32;
    let sum_log: f32 = buf.iter().map(|v| v.max(floor).ln()).sum();
    (sum_log / n).exp()
}

/// Harmonic mean of a buffer's values. For a minimizer's readings this is the
/// "count the high readings" mean: it shrinks the influence of lucky low spikes.
pub fn rolling_harmonic_mean(buf: &RollingBuffer, floor: f32) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let n = buf.len() as f32;
    let v = floor.max(f32::EPSILON);
    let sum_recip: f32 = buf.iter().map(|x| 1.0 / x.max(v)).sum();
    n / sum_recip
}

/// A smoothed reading WITH its own error bar: how many readings it averages and
/// how much they spread — the race's number is a mean of K small samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowStats {
    /// Readings behind `mean` (buffer length, ≤ the smoothing window).
    pub n: usize,
    /// Their mean — the smoothed reading every ranking decision uses.
    pub mean: f32,
    /// Population std of the readings: their step-to-step spread.
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
    fn geometric_mean_weights_the_low_readings() {
            // The cartpole champion's window: one lucky 500 among four fizzles;
            // the geometric mean pulls toward the fizzles.
        let buf = RollingBuffer::from_values(5, &[40.0, 500.0, 40.0, 40.0, 40.0]);
        // exp(mean(ln(40), ln(500), ln(40), ln(40), ln(40)))
        let geo = rolling_geometric_mean(&buf, f32::EPSILON);
        // independent check: product^(1/5)
        let product: f32 = buf.iter().product::<f32>();
        // Tolerance, not identity: the two paths (log-sum vs product-root)
        // round differently in f32.
        assert!(
            (product.powf(0.2) - geo).abs() < 1e-4,
            "geo mismatch: product-root {} vs geo {}",
            product.powf(0.2),
            geo
        );
        assert!(geo < 500.0, "must shrink the 500");
        assert!(geo > 40.0, "must stay above the floor");
    }

    #[test]
    fn harmonic_mean_weights_the_high_readings() {
        // For a minimizer the worst step is the one you most want to count.
        let buf = RollingBuffer::from_values(5, &[40.0, 500.0, 40.0, 40.0, 40.0]);
        let hm = rolling_harmonic_mean(&buf, f32::EPSILON);
        // n / sum(1/x) with one 500 and four 40s.
        let expected = 5.0 / (4.0 / 40.0 + 1.0 / 500.0);
        assert_eq!(hm, expected, "harmonic mismatch {}", hm);
        // For a maximizer this is the wrong choice: it counts the high 500 more.
        assert!(hm > 40.0);
    }

    #[test]
    fn geometric_and_harmonic_handle_zeros_via_the_floor() {
        // A window with a true 0 — a net that died immediately — must not
        // crash (ln(0)) or dominate (1/0).
        let buf = RollingBuffer::from_values(3, &[0.0, 0.0, 50.0]);
        // With floor 1.0, the two 0.0s count as 1.0 each.
        let geo = rolling_geometric_mean(&buf, 1.0);
        let expected_geo = (1.0_f32 * 1.0 * 50.0).powf(1.0 / 3.0);
        assert!(
            (geo - expected_geo).abs() < 1e-5,
            "geo mismatch: {} vs {}",
            geo,
            expected_geo
        );
        let hm = rolling_harmonic_mean(&buf, 1.0);
        assert_eq!(hm, 3.0 / (2.0 / 1.0 + 1.0 / 50.0));
        // Floor 0.0 would crash: ln(0) or 1/0. Provide a positive floor at the
        // call site (see WindowStats::smoothed_mean for the default).
        let geo_floor = rolling_geometric_mean(&buf, f32::EPSILON);
        assert!(geo_floor > 0.0);
        assert!(geo_floor < 50.0);
    }

    #[test]
    fn geometric_mean_is_strictly_between_arithmetic_and_floor_under_spread() {
        let buf = RollingBuffer::from_values(5, &[40.0, 500.0, 40.0, 40.0, 40.0]);
        let arith = rolling_mean(&buf);
        let geo = rolling_geometric_mean(&buf, f32::EPSILON);
        assert!(geo < arith, "geometric < arithmetic when readings spread");
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
