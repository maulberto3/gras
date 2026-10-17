//! Scoring — fitness function for ranking individuals.
//!
//! [`Fitness`] is a pure scoring function. Training loss is the user's
//! responsibility (lives inside the [`Trainer`](crate::trainer::Trainer)).

use flodl::Variable;
use flodl::tensor::Result;
use serde::{Deserialize, Serialize};

/// The engine-computed scoring signature: `(pred, target) → f32`.
type ScoreFn = dyn Fn(&Variable, &Variable) -> Result<f32> + Send + Sync;

// ── Direction — lower or higher is better ─────────────────────────────────

/// Whether lower or higher is better.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum Direction {
    /// Lower is better (losses).
    #[default]
    Minimize,
    /// Higher is better (metrics).
    Maximize,
}

impl Direction {
    /// Is `new` better than `current`?
    pub fn is_better(&self, new: f32, current: f32) -> bool {
        match self {
            Direction::Minimize => new < current,
            Direction::Maximize => new > current,
        }
    }

    /// Compact arrow for logs: `↓` or `↑`.
    pub fn arrow(&self) -> &'static str {
        match self {
            Direction::Minimize => "↓",
            Direction::Maximize => "↑",
        }
    }

    /// Total-order comparison under this direction (for `max_by`).
    /// NaN treated as equal.
    pub fn cmp(&self, a: f32, b: f32) -> std::cmp::Ordering {
        match self {
            Direction::Minimize => b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal),
            Direction::Maximize => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
        }
    }

    /// The RISK-ADJUSTED ranking value: the window `mean` shifted by `lambda`
    /// window standard deviations TOWARD THE BAD SIDE, in the fitness's own
    /// units. This is the single place the smoothing window's spread becomes a
    /// ranking decision (FIT).
    ///
    /// - `Maximize` — a weak window drags the net DOWN (`mean − λσ`): the net is
    ///   credited only with the value it is still expected to beat after one
    ///   window of bad luck.
    /// - `Minimize` — a costly window drags the net UP (`mean + λσ`).
    ///
    /// `std` is direction-INVARIANT (spread is spread), which is why ONE lambda
    /// covers both directions — no per-direction knob exists or is needed. The
    /// shift is expressed in fitness units, so [`Direction::cmp`] and every
    /// call site that sorts with it keep working unchanged: `lambda = 0.0`
    /// returns the plain mean exactly (the pre-FIT ranking).
    pub fn risk_adjusted(&self, mean: f32, std: f32, lambda: f32) -> f32 {
        match self {
            Direction::Maximize => mean - lambda * std,
            Direction::Minimize => mean + lambda * std,
        }
    }
}

// ── Fitness — computed or reported ───────────────────────────────────────

/// Ranking signal for the race. [`Fitness::Computed`] = the engine scores
/// `(pred, target)`; [`Fitness::Reported`] = the trainer ships the scalar.
///
/// `Clone` is cheap and shares ONE scorer: cloning a [`Fitness::Computed`]
/// clones an `Arc`, so the same scoring function can travel into several runs
/// (and several `RunSpec`s) without being rebuilt per copy.
#[derive(Clone)]
pub enum Fitness {
    /// Engine-computed: a pure `(pred, target) → f32` scoring function,
    /// plus the direction and label that travel with it.
    Computed {
        /// Ranking metric: `(pred, target) → f32`.
        score_fn: std::sync::Arc<ScoreFn>,
        /// Score direction.
        direction: Direction,
        /// Label for logs.
        label: String,
    },
    /// Trainer-reported: the value arrives via `StepReport.fitness`; the
    /// engine never scores. Carries only the direction and label.
    Reported {
        /// Score direction.
        direction: Direction,
        /// Label for logs.
        label: String,
    },
}

impl Fitness {
    /// Create a new engine-computed fitness function (supervised style).
    pub fn new<F>(score_fn: F, direction: Direction, label: &str) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<f32> + Send + Sync + 'static,
    {
        Fitness::Computed {
            score_fn: std::sync::Arc::new(score_fn) as std::sync::Arc<ScoreFn>,
            direction,
            label: label.to_string(),
        }
    }

        /// Trainer-reported fitness (RL style): the trainer supplies the scalar
        /// each step and the engine ranks on it.
    pub fn reported(direction: Direction, label: &str) -> Self {
        Fitness::Reported {
            direction,
            label: label.to_string(),
        }
    }

        /// Lower-is-better scorer — shorthand for
        /// `Fitness::new(score_fn, Direction::Minimize, label)`.
    pub fn minimize<F>(label: &str, score_fn: F) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<f32> + Send + Sync + 'static,
    {
        Fitness::new(score_fn, Direction::Minimize, label)
    }

        /// Higher-is-better scorer — shorthand for
        /// `Fitness::new(score_fn, Direction::Maximize, label)`.
    pub fn maximize<F>(label: &str, score_fn: F) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<f32> + Send + Sync + 'static,
    {
        Fitness::new(score_fn, Direction::Maximize, label)
    }

        /// Score prediction against target — engine-computed ranking metric.
        /// Errors on a [`Fitness::Reported`] (nothing to score there).
    pub fn score(&self, pred: &Variable, target: &Variable) -> Result<f32> {
        match self {
            Fitness::Computed { score_fn, .. } => score_fn(pred, target),
            Fitness::Reported { label, .. } => Err(flodl::tensor::TensorError::new(&format!(
                "fitness `{label}` is Reported — the trainer supplies the value in StepReport; there is no (pred, target) scorer to call"
            ))),
        }
    }

    /// Score direction.
    pub fn direction(&self) -> Direction {
        match self {
            Fitness::Computed { direction, .. } => *direction,
            Fitness::Reported { direction, .. } => *direction,
        }
    }

    /// Label for logs.
    pub fn label(&self) -> &str {
        match self {
            Fitness::Computed { label, .. } => label,
            Fitness::Reported { label, .. } => label,
        }
    }

    /// Does the engine compute the score itself (`true` = Computed)?
    pub fn is_computed(&self) -> bool {
        matches!(self, Fitness::Computed { .. })
    }

    /// Does the trainer report the score (`true` = Reported)?
    pub fn is_reported(&self) -> bool {
        matches!(self, Fitness::Reported { .. })
    }
}

// ── Re-exports for backward compat ────────────────────────────────────────

// Historical note: old code used FitnessKind in the pre-split engine config;
// we keep a serializable label string instead.

/// Serialized fitness label for engine.json (e.g. "mse", "accuracy").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitnessLabel(pub String);

/// An informative (non-ranking) metric for a run — a label plus its closure;
/// values live in `NetMetrics::informative` in the run's `Vec<Metric>` order.
#[derive(Clone)]
pub struct Metric {
    /// Metric label, recorded in engine.json/history.csv.
    pub label: String,
    score_fn: std::sync::Arc<ScoreFn>,
}

impl Metric {
        /// Build a metric: a label + your own `(pred, target) -> f32` closure.
    pub fn custom<F>(label: impl Into<String>, score_fn: F) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<f32> + Send + Sync + 'static,
    {
        Metric {
            label: label.into(),
            score_fn: std::sync::Arc::new(score_fn),
        }
    }

    /// Label for logs / engine.json / history.csv header.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Score this metric with its own closure.
    pub fn score(&self, pred: &Variable, y: &Variable) -> Result<f32> {
        (self.score_fn)(pred, y)
    }
}

impl PartialEq for Metric {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label // closures don't compare; identity is the label
    }
}

impl std::fmt::Debug for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metric")
            .field("label", &self.label)
            .finish()
    }
}

impl Default for FitnessLabel {
    fn default() -> Self {
        FitnessLabel("loss".to_string())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::network::Network;
    use crate::graph::node::Node;
    use crate::graph::topology::Topology;
    use crate::utils::score::*;
    use flodl::nn::Module;
    use proptest::prelude::*;

    fn tiny_net() -> Network {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_output(1, 1, 1));
        graph.finalize();
        Network::build(&graph, flodl::Device::CPU).unwrap()
    }

    fn input(xs: &[f32]) -> Variable {
        Variable::new(
            flodl::Tensor::from_f32(xs, &[xs.len() as i64, 1], flodl::Device::CPU).unwrap(),
            false,
        )
    }

    /// `Fitness` is `Clone` so one scorer can be shared across runs: the clone
    /// must run the SAME closure (not a rebuilt one), and carry the same
    /// direction and label.
    #[test]
    fn fitness_clone_shares_the_scorer() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let f = Fitness::maximize("counter", move |_p, _t| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(1.0)
        });
        let g = f.clone();
        assert!(f.is_computed() && g.is_computed());
        assert_eq!(g.label(), "counter");
        assert_eq!(g.direction(), Direction::Maximize);
        let x = input(&[3.5]);
        assert_eq!(f.score(&x, &x).unwrap(), 1.0);
        assert_eq!(g.score(&x, &x).unwrap(), 1.0);
        // Both copies hit ONE counter: a re-built closure would have its own.
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_direction_semantics() {
        assert!(Direction::Minimize.is_better(0.5, 1.0));
        assert!(!Direction::Minimize.is_better(1.0, 0.5));
        assert!(Direction::Maximize.is_better(1.0, 0.5));
        assert!(!Direction::Maximize.is_better(0.5, 1.0));
    }

    #[test]
    fn risk_adjusted_at_zero_lambda_is_the_plain_mean() {
        // The pre-FIT ranking, exactly — for BOTH directions and both signs of
        // value (a negative-mean fitness must not drift either).
        for (mean, std) in [(320.4, 41.0), (-0.75, 0.25), (0.0, 3.0)] {
            assert_eq!(Direction::Maximize.risk_adjusted(mean, std, 0.0), mean);
            assert_eq!(Direction::Minimize.risk_adjusted(mean, std, 0.0), mean);
        }
    }

    #[test]
    fn risk_adjusted_shifts_toward_the_bad_side() {
        let (mean, std) = (320.0, 40.0);
        let lambda = 1.0;
        // Maximize: the penalty is SUBTRACTED (a weak window is less attractive).
        let hi = Direction::Maximize.risk_adjusted(mean, std, lambda);
        assert_eq!(hi, 280.0);
        // Minimize: the same spread is ADDED (a costly window is less attractive).
        let lo = Direction::Minimize.risk_adjusted(mean, std, lambda);
        assert_eq!(lo, 360.0);
        // The two directions move in OPPOSITE absolute directions but the SAME
        // relative one: each is worse than its own mean.
        assert!(hi < mean, "a maximizer must lose value from spread");
        assert!(lo > mean, "a minimizer must lose value from spread");
        // Wider lambda penalizes monotonically, and never better than λ=0.
        for lambda in [0.5, 1.0, 2.0] {
            assert!(Direction::Maximize.risk_adjusted(mean, std, lambda) <= mean);
            assert!(Direction::Minimize.risk_adjusted(mean, std, lambda) >= mean);
        }
        // λ = 2 = two window-stds, not a clamp or a re-scale.
        assert_eq!(Direction::Maximize.risk_adjusted(mean, std, 2.0), 240.0);
    }

    #[test]
    fn risk_adjusted_is_mirrored_between_directions() {
        // Negating the readings and flipping the direction must give the negated
        // score: the two arms are one rule, not two rules that happen to agree.
        for (mean, std) in [(320.0_f32, 40.0_f32), (1e-4, 2e-5), (-12.5, 3.5)] {
            let up = Direction::Maximize.risk_adjusted(mean, std, 1.5);
            let down = Direction::Minimize.risk_adjusted(-mean, std, 1.5);
            assert!(
                (up + down).abs() < 1e-5,
                "Maximize({mean},{std}) = {up} must mirror Minimize({},{std}) = {down}",
                -mean
            );
        }
    }

    #[test]
    fn risk_adjustment_breaks_a_mean_tie_toward_the_steadier_net() {
        // THE FIT CONTRACT: two nets with the SAME window mean, one steady one
        // volatile. Under Maximize the steady net must rank better; the volatile
        // one still wins on the plain mean (a tie, decided by the tie-break).
        let steady = (300.0_f32, 5.0_f32); // always ~300
        let volatile = (300.0_f32, 120.0_f32); // 40s and 500s
        let dir = Direction::Maximize;
        assert_eq!(dir.cmp(steady.0, volatile.0), std::cmp::Ordering::Equal);
        let s = dir.risk_adjusted(steady.0, steady.1, 1.0);
        let v = dir.risk_adjusted(volatile.0, volatile.1, 1.0);
        assert!(
            dir.is_better(s, v),
            "the steady net ({s}) must outrank the volatile one ({v})"
        );
        // And λ = 0 removes the preference entirely (back to the mean).
        assert_eq!(
            dir.risk_adjusted(steady.0, steady.1, 0.0),
            dir.risk_adjusted(volatile.0, volatile.1, 0.0)
        );
        // A minimizer: same numbers, spread is now the cost — the steady net wins
        // on the SAME sort (`is_better` reads the smaller value as better).
        let dir = Direction::Minimize;
        let s = dir.risk_adjusted(steady.0, steady.1, 1.0);
        let v = dir.risk_adjusted(volatile.0, volatile.1, 1.0);
        assert!(dir.is_better(s, v), "minimizer: steady {s} beats volatile {v}");
    }

    #[test]
    fn test_reported_fitness_contract() {
        let f = Fitness::reported(Direction::Maximize, "episode_reward");
        assert!(f.is_reported());
        assert!(!f.is_computed());
        assert_eq!(f.direction(), Direction::Maximize);
        assert_eq!(f.label(), "episode_reward");
        // No scorer exists — scoring must error loudly, not return a value.
        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let pred = net.forward(&x).unwrap();
        assert!(f.score(&pred, &pred).is_err());
    }

    #[test]
    fn test_computed_fitness_flags() {
        let f = Fitness::new(mse_loss_score, Direction::Minimize, "mse");
        assert!(f.is_computed());
        assert!(!f.is_reported());
    }

    #[test]
    fn test_fitness_minimize_maximize_shorthand() {
        let lo = Fitness::minimize("mse", mse_loss_score);
        assert!(lo.is_computed());
        assert_eq!(lo.direction(), Direction::Minimize);
        assert_eq!(lo.label(), "mse");

        let hi = Fitness::maximize("f1", f1_score);
        assert!(hi.is_computed());
        assert_eq!(hi.direction(), Direction::Maximize);
        assert_eq!(hi.label(), "f1");

        // The shorthand must score identically to the long form.
        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let y = input(&[1.0, 2.0, 3.0]);
        let pred = net.forward(&x).unwrap();
        let short = Fitness::minimize("mse", mse_loss_score)
            .score(&pred, &y)
            .unwrap();
        let long = Fitness::new(mse_loss_score, Direction::Minimize, "mse")
            .score(&pred, &y)
            .unwrap();
        assert!((short - long).abs() < 1e-9);
    }

    #[test]
    fn test_fitness_new() {
        let f = Fitness::new(
            |pred, y| {
                let diff = pred.data().sub(&y.data())?;
                let sq = diff.mul(&diff)?;
                Ok(sq.mean()?.item()? as f32)
            },
            Direction::Minimize,
            "mse",
        );
        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let y = input(&[1.0, 2.0, 3.0]);
        let pred = net.forward(&x).unwrap();
        let score = f.score(&pred, &y).unwrap();
        assert!(score.is_finite());
        assert!(score >= 0.0);
        assert_eq!(f.direction(), Direction::Minimize);
        assert_eq!(f.label(), "mse");
    }

    // TODO: restore `test_fitness_from_loss_with_other` when
    // `from_loss_with_other` is re-enabled.

    #[test]
    fn test_categorical_helpers() {
        let mut graph = Topology::new(0, None);
        graph.options.input_dim = Some(2);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 2, 2));
        graph.nodes.push(Node::new_output(2, 2, 2));
        graph.nodes[2].hidden_dim = Some(2);
        graph.finalize();
        let net = Network::build(&graph, flodl::Device::CPU).unwrap();
        let x = Variable::new(
            flodl::Tensor::from_f32(&[2.0, 0.5, 0.1, 0.9, 1.2, 0.3], &[3, 2], flodl::Device::CPU)
                .unwrap(),
            false,
        );
        let pred = net.forward(&x).unwrap();
        let pa = pred.data().argmax(1, false).unwrap().to_i64_vec().unwrap();
        let y = Variable::new(
            crate::utils::tabular_data::one_hot(
                &pa.iter().map(|&v| v as usize).collect::<Vec<_>>(),
                2,
                flodl::Device::CPU,
            )
            .unwrap(),
            false,
        );
        let acc = accuracy_score(&pred, &y).unwrap();
        let ce = cross_entropy_onehot(&pred, &y).unwrap();
        assert!((acc - 1.0).abs() < 1e-6, "acc = {acc}");
        assert!(ce > 0.0 && ce.is_finite(), "ce = {ce}");
    }

    #[test]
    fn test_f1_and_precision_from_vecs() {
        assert_eq!(f1_from_vecs(&[0, 1, 0, 1], &[0, 1, 0, 1], 2), 1.0);
        assert_eq!(precision_from_vecs(&[0, 1, 0, 1], &[0, 1, 0, 1], 2), 1.0);
        let pa = [0i64, 0, 0, 0];
        let ta = [0i64, 1, 0, 1];
        assert!((precision_from_vecs(&pa, &ta, 2) - 0.25).abs() < 1e-9);
        assert!((f1_from_vecs(&pa, &ta, 2) - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(f1_from_vecs(&[], &[], 2), 0.0);
        assert_eq!(precision_from_vecs(&[0], &[0], 0), 0.0);
    }

    #[test]
    fn test_all_scoring_helpers() {
        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let pred = net.forward(&x).unwrap();
        // Use pred as both pred and y for perfect-fit case.
        let y = Variable::new(pred.data().clone(), false);

        let mse = mse_loss_score(&pred, &y).unwrap();
        assert!(mse < 1e-6, "mse_score = {mse}");
        let l1 = l1_loss_score(&pred, &y).unwrap();
        assert!(l1 < 1e-6, "l1_score = {l1}");
        let rmse = rmse_score(&pred, &y).unwrap();
        assert!(rmse < 1e-3, "rmse_score = {rmse}");
        let r2 = r2_score(&pred, &y).unwrap();
        assert!((r2 - 1.0).abs() < 1e-3, "r2_score = {r2}");

        // Cross-entropy: > 0 for valid predictions
        let mut graph2 = Topology::new(0, None);
        graph2.options.input_dim = Some(2);
        graph2.nodes.push(Node::new_input(0, 2));
        graph2.nodes.push(Node::new_hidden(1, 2, 2));
        graph2.nodes.push(Node::new_output(2, 2, 3));
        graph2.nodes[2].hidden_dim = Some(3);
        graph2.finalize();
        let cat_net = Network::build(&graph2, flodl::Device::CPU).unwrap();
        let cx = Variable::new(
            flodl::Tensor::from_f32(&[1.0, 0.5, 0.1, 0.9], &[2, 2], flodl::Device::CPU).unwrap(),
            false,
        );
        let cpred = cat_net.forward(&cx).unwrap();
        // One-hot targets
        let ct = Variable::new(
            flodl::Tensor::from_f32(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], flodl::Device::CPU)
                .unwrap(),
            false,
        );
        let ce = cross_entropy_onehot(&cpred, &ct).unwrap();
        assert!(ce > 0.0 && ce.is_finite(), "cross_entropy = {ce}");
        let acc = accuracy_score(&cpred, &ct).unwrap();
        assert!((0.0..=1.0).contains(&acc), "accuracy = {acc}");
        let f1 = f1_score(&cpred, &ct).unwrap();
        assert!((0.0..=1.0).contains(&f1), "f1 = {f1}");
        let prec = precision_score(&cpred, &ct).unwrap();
        assert!((0.0..=1.0).contains(&prec), "precision = {prec}");
        let (pa, ta) = argmax_classes(&cpred, &ct).unwrap();
        assert_eq!(pa.len(), 2);
        assert_eq!(ta.len(), 2);
    }

    proptest! {
        #[test]
        fn prop_direction_is_better_is_antisymmetric(a in -100.0f32..100.0, b in -100.0f32..100.0) {
            if a != b {
                if Direction::Minimize.is_better(a, b) {
                    prop_assert!(!Direction::Minimize.is_better(b, a));
                }
                if Direction::Maximize.is_better(a, b) {
                    prop_assert!(!Direction::Maximize.is_better(b, a));
                }
            }
        }


    }
}
