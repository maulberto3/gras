//! Fitness smoothing: the rolling buffer, its mean, its spread, and the window it uses.

use crate::engine::config::MeanType;
use crate::utils::fitness::Direction;

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

/// Rolling mean of a buffer's values.
pub fn rolling_mean(buf: &RollingBuffer) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let sum: f32 = buf.iter().sum();
    sum / buf.len() as f32
}

/// A smoothed reading WITH its own error bar: how many readings it averages
/// how much they spread — the race's number is a mean of K small samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowStats {
    /// Readings behind `mean` (buffer length, ≤ the smoothing window).
    pub n: usize,
    /// Their mean — the readable smoothed level, and the center the ranking score
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
    /// that describes the weights the run actually ended
    pub latest: f32,
    /// The family [`Self::fam_center`] / [`Self::fam_spread`] are measured
    /// (`set_fitness_mean_type`).
    pub family: MeanType,
    /// The window's centre IN `family`'s space — what [`Self::score`] shifts by `λ`.
    pub fam_center: f32,
    /// The window's spread about `fam_center`, in the same space: the std
    /// transformed readings, or the scaled MAD for [`MeanType::Median`].
    pub fam_spread: f32,
}

impl WindowStats {
    /// Standard error of `mean` (`std / √n`). `None` for zero readings;
    /// `Some(0.0)` for one reading (unobservable spread, not precision).
    pub fn se(&self) -> Option<f32> {
        (self.n > 0).then(|| self.std / (self.n as f32).sqrt())
    }

    /// The RISK-ADJUSTED score (FIT) this window ranks by: the mean shifted `lambda` window-stds
    /// toward the bad side for `direction`.
    pub fn risk_adjusted(&self, direction: Direction, lambda: f32) -> f32 {
        direction.risk_adjusted(self.mean, self.std, lambda)
    }

    /// The RANKING score in the window's chosen [`MeanType`] family: the family centre shifted
    /// `lambda` of its own spread toward the bad side, mapped back into fitness units.
    pub fn score(&self, direction: Direction, lambda: f32) -> f32 {
        self.family
            .restore(direction.risk_adjusted(self.fam_center, self.fam_spread, lambda))
    }
}

impl MeanType {
    /// Map one reading into the family's averaging space.
    fn transform(self, x: f32) -> f32 {
        match self {
            MeanType::Arithmetic | MeanType::Median => x,
            // `signum(0.0)` is `1.0`, and `ln(1 + 0) = 0`, so 0 maps to 0 either way.
            MeanType::SignedLog => x.signum() * (1.0 + x.abs()).ln(),
        }
    }

    /// Map a family-space value back into fitness units (the inverse of `transform`).
    fn restore(self, y: f32) -> f32 {
        match self {
            MeanType::Arithmetic | MeanType::Median => y,
            MeanType::SignedLog => y.signum() * (y.abs().exp() - 1.0),
        }
    }
}

/// `1.4826`: the scale factor that makes the MAD of a normal sample estimate its `σ`,
/// a [`MeanType::Median`] run's λ stays comparable with an arithmetic one's.
const MAD_TO_SIGMA: f32 = 1.4826;

/// The weighted median of `(value, weight)` pairs: the first value whose cumulative weight reaches
/// half the total.
fn weighted_median(mut pairs: Vec<(f32, f32)>) -> f32 {
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let half = pairs.iter().map(|(_, w)| *w).sum::<f32>() / 2.0;
    let mut acc = 0.0;
    for (i, (v, w)) in pairs.iter().enumerate() {
        acc += *w;
        if acc >= half || i + 1 == pairs.len() {
            return *v;
        }
    }
    0.0
}

/// Stats of a rolling buffer: `None` when it is empty (a net that was never measured has no
/// reading, and must not read as a confident 0.0).
pub fn rolling_stats(buf: &RollingBuffer) -> Option<WindowStats> {
    rolling_stats_with(buf, MeanType::Arithmetic)
}

/// [`rolling_stats`] in `family`'s averaging space (`set_fitness_mean_type`): the centre and the spread
/// are measured after the family's transform, so a spike cannot drag the level the way it drags the
/// arithmetic mean.
pub fn rolling_stats_with(buf: &RollingBuffer, family: MeanType) -> Option<WindowStats> {
    rolling_stats_all(buf, family)
}

/// [`rolling_stats_with`] over the LAST `take` readings only — the POLICY-SCOPED window
/// (`set_fitness_policy_scoped`): the readings taken on the weights the net holds right now, so
/// no reading from another policy enters the statistics. `take`
/// clamps to the buffer; `0` is `None` (nothing measured on this policy yet), and `1` is that
/// value with zero spread, so `mean ∓ λ·0` stays clean instead of claiming a confidence nobody has.
pub fn rolling_stats_tail(
    buf: &RollingBuffer,
    take: usize,
    family: MeanType,
) -> Option<WindowStats> {
    let take = take.min(buf.len());
    if take == 0 {
        return None;
    }
    if take == buf.len() {
        return rolling_stats_all(buf, family);
    }
    // A private tail buffer keeps ONE implementation of the weighted mean/spread: the epoch is
    // just a shorter window, never a second arithmetic.
    let mut tail = RollingBuffer::new(take);
    for v in buf.iter().skip(buf.len() - take) {
        tail.push(*v);
    }
    rolling_stats_all(&tail, family)
}

/// The one rolling-stats arithmetic both entry points share.
fn rolling_stats_all(buf: &RollingBuffer, family: MeanType) -> Option<WindowStats> {
    if buf.is_empty() {
        return None;
    }
    let n = buf.len();
    let mean = rolling_mean(buf);
    let var = buf.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    // The centre and the spread the SCORE is built from, in the family's own space.
    let (fam_center, fam_spread) = match family {
        MeanType::Arithmetic => (mean, var.sqrt()),
        MeanType::SignedLog => {
            // One transform, one log further out: a spike enters `g` compressed, so it cannot drag
            // the centre the way it drags the arithmetic mean, and λ charges the spread
            let gmean = buf.iter().map(|v| family.transform(*v)).sum::<f32>() / n as f32;
            let gvar = buf
                .iter()
                .map(|v| {
                    let d = family.transform(*v) - gmean;
                    d * d
                })
                .sum::<f32>()
                / n as f32;
            (gmean, gvar.sqrt())
        }
        MeanType::Median => {
            let pairs: Vec<(f32, f32)> = buf.iter().map(|v| (*v, 1.0)).collect();
            let center = weighted_median(pairs.clone());
            let mad = weighted_median(
                pairs
                    .iter()
                    .map(|(v, w)| ((v - center).abs(), *w))
                    .collect(),
            );
            (center, mad * MAD_TO_SIGMA)
        }
    };
    Some(WindowStats {
        n,
        mean,
        std: var.sqrt(),
        min: buf.iter().copied().fold(f32::INFINITY, f32::min),
        max: buf.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        latest: buf.last().unwrap_or(mean),
        family,
        fam_center,
        fam_spread,
    })
}

/// The per-net smoothing window, in steps (K).
pub use crate::engine::config::SMOOTHING_WINDOW;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::config::MeanType;

    /// The epoch window is the SAME arithmetic over a shorter tail — never a second formula.
    #[test]
    fn rolling_stats_tail_reads_only_the_last_take() {
        let buf = RollingBuffer::from_values(5, &[10.0, 10.0, 10.0, 4.0, 6.0]);
        let all = rolling_stats_with(&buf, MeanType::Arithmetic).unwrap();
        assert_eq!(
            (all.n, all.mean),
            (5, 8.0),
            "the whole buffer is the old reading"
        );
        let epoch = rolling_stats_tail(&buf, 2, MeanType::Arithmetic).unwrap();
        assert_eq!((epoch.n, epoch.mean, epoch.std), (2, 5.0, 1.0));
        assert_eq!(
            epoch.latest, 6.0,
            "the tail keeps the newest reading, not the tail's own order"
        );
        assert_eq!(
            rolling_stats_tail(&buf, 0, MeanType::Arithmetic),
            None,
            "no reading on this policy ⇒ no reading, never a fabricated 0"
        );
        // ONE reading: its value IS the mean, and zero spread keeps `mean ∓ λ·0` clean.
        let one = rolling_stats_tail(&buf, 1, MeanType::Arithmetic).unwrap();
        assert_eq!((one.n, one.mean, one.std), (1, 6.0, 0.0));
        assert_eq!(
            one.score(crate::utils::fitness::Direction::Maximize, 0.5),
            6.0
        );
    }

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
        // The bimodal champion: one lucky 500 among four fizzles. The mean
        // readable 132, the spread is 184 — the score must charge
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
        // they
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
        let volatile = rolling_stats(&RollingBuffer::from_values(
            4,
            &[250.0, 350.0, 250.0, 350.0],
        ))
        .unwrap();
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
        // Population std: sqrt((4·92² + 368²)/5) = 184 — the deviations
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
        // A zero SE here is "only one reading", not "perfectly precise" —
        // verdict's noisy branch is the caller that must not trust
        assert_eq!(s.se(), Some(0.0));
    }

    #[test]
    fn signed_log_damps_a_spike_without_dropping_it() {
        // The bimodal champion again: one 500 among four 40s.
        let buf = RollingBuffer::from_values(5, &[40.0, 500.0, 40.0, 40.0, 40.0]);
        let arith = rolling_stats_with(&buf, MeanType::Arithmetic).unwrap();
        let damped = rolling_stats_with(&buf, MeanType::SignedLog).unwrap();
        let score = damped.score(Direction::Maximize, 0.0);
        assert!(
            score < arith.score(Direction::Maximize, 0.0),
            "score was {score}"
        );
        assert!(score > 40.0, "the spike must still count: {score}");
        // λ charges the spread of the SAME space, so a maximizer still pays
        assert!(damped.score(Direction::Maximize, 1.0) < score);
        assert!(damped.score(Direction::Minimize, 1.0) > score);
        // One reading maps through `g` and back unchanged, negative readings included.
        let single =
            rolling_stats_with(&RollingBuffer::from_values(1, &[-3.5]), MeanType::SignedLog)
                .unwrap();
        assert!((single.score(Direction::Minimize, 0.0) + 3.5).abs() < 1e-4);
    }

    #[test]
    fn median_ignores_one_fluke_entirely() {
        // Four 40s and one 500: the median is 40 and the MAD about it is 0, so NO λ charge
        // can move the score — a single jump cannot buy (or cost) a rank.
        let buf = RollingBuffer::from_values(5, &[40.0, 500.0, 40.0, 40.0, 40.0]);
        let s = rolling_stats_with(&buf, MeanType::Median).unwrap();
        assert_eq!(s.fam_center, 40.0);
        for lambda in [0.0, 0.5, 2.0] {
            for dir in [Direction::Maximize, Direction::Minimize] {
                assert_eq!(s.score(dir, lambda), 40.0);
            }
        }
        // A genuinely split window: the median sits on the middle reading and the scaled
        // MAD reads the half-window it disagrees
        let mixed = RollingBuffer::from_values(3, &[10.0, 40.0, 90.0]);
        let m = rolling_stats_with(&mixed, MeanType::Median).unwrap();
        assert_eq!(m.fam_center, 40.0);
        assert!(
            (m.fam_spread - 30.0 * 1.4826).abs() < 1e-3,
            "spread was {}",
            m.fam_spread
        );
        // The default family is the pre-knob ranking, and `rolling_stats` is its shortcut.
        assert_eq!(MeanType::default(), MeanType::Arithmetic);
        assert_eq!(
            rolling_stats_with(&buf, MeanType::Arithmetic)
                .unwrap()
                .score(Direction::Maximize, 0.5),
            rolling_stats(&buf).unwrap().score(Direction::Maximize, 0.5)
        );
    }
}
