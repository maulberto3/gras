//! Scoring strategies 🎯 — the fitness/loss split and direction.
//!
//! The engine consumes a [`Fitness`] that separates **scoring** (ranking
//! individuals) from **loss** (training via backward). The user always
//! provides a scoring function; the loss function is optional — when
//! absent, MSE is the default training signal.
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  Fitness                                        │
//! │  ├─ score_fn:  (pred, y) → f32    ← ranking    │
//! │  └─ loss_fn:   (pred, y) → Variable ← training │
//! │               (None → MSE default)              │
//! └─────────────────────────────────────────────────┘
//! ```

use flodl::Variable;
use flodl::tensor::Result;
use serde::{Deserialize, Serialize};

// ── Direction — lower or higher is better ─────────────────────────────────

/// Whether a **lower** or a **higher** score is better.
///
/// - Losses (`Mse`, `Mae`, ...) → [`Minimize`](Direction::Minimize).
/// - Metrics (`R2`, `Accuracy`, `F1`) → [`Maximize`](Direction::Maximize).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum Direction {
    /// Lower scores are better (losses).
    #[default]
    Minimize,
    /// Higher scores are better (metrics like accuracy).
    Maximize,
}

impl Direction {
    /// Is `new` better than `current` under this direction?
    pub fn is_better(&self, new: f32, current: f32) -> bool {
        match self {
            Direction::Minimize => new < current,
            Direction::Maximize => new > current,
        }
    }

    /// A compact arrow for logs: `↓` = lower is better, `↑` = higher.
    pub fn arrow(&self) -> &'static str {
        match self {
            Direction::Minimize => "↓",
            Direction::Maximize => "↑",
        }
    }

    /// Total-order comparison under this direction, ready for `max_by`:
    /// returns [`Ordering::Greater`](std::cmp::Ordering::Greater) when `a` is
    /// **better** than `b`. NaN is treated as equal (deterministic, never
    /// panics).
    pub fn cmp(&self, a: f32, b: f32) -> std::cmp::Ordering {
        match self {
            Direction::Minimize => b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal),
            Direction::Maximize => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
        }
    }
}

// ── Fitness — the scoring + training interface ────────────────────────────

/// The fitness interface — separates **scoring** (ranking individuals)
/// from **training** (backward pass). The engine calls `score()` on held-out
/// data to rank, and `train_metric()` on training data to update weights.
///
/// ```text
/// Fitness::from_loss(f)                        ← same function for both
/// Fitness::from_loss_with_other(score, train_metric, ...) ← separate
/// ```
///
/// **The user must always provide an explicit train metric.** The engine
/// cannot guess the right training signal for different output formats
/// (regression, binary, multi-class, etc.).
pub struct Fitness {
    /// The ranking metric — `(pred, target) → f32` score.
    /// Called on **eval** batches to rank individuals.
    score_fn: Box<dyn Fn(&Variable, &Variable) -> Result<f32> + Send + Sync>,
    /// The training metric — `(pred, target) → Variable` for backward.
    /// Always required — the engine cannot guess the right signal for different
    /// output formats (regression, binary, multi-class, etc.).
    train_metric_fn: Box<dyn Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync>,
    /// Which direction is better for the **score** (ranking individuals).
    score_direction: Direction,
    /// Which direction is better for the **train metric** (training signal).
    train_metric_direction: Direction,
    /// Label for the fitness/scoring function (e.g. "accuracy", "r2").
    fitness_label: String,
    /// Label for the train metric (e.g. "cross_entropy", "mse").
    train_metric_label: String,
}

impl Fitness {
    // TODO(fitness): separate ranking + training when multi-objective selection
    // is implemented (Pareto or weighted). The tension: evolution ranks on
    // fitness (e.g. accuracy) but training minimizes a differentiable loss
    // (e.g. cross-entropy). Without multi-objective selection, the two can
    // drift apart — evolution picks nets that don't train well, or training
    // improves the loss but not the fitness. Implementing this requires either
    // Pareto-dominance selection or a weighted scalar projection.
    //
    // pub fn from_loss_with_other<S, L>(
    //     score_fn: S,
    //     train_metric_fn: L,
    //     score_direction: Direction,
    //     train_metric_direction: Direction,
    //     fitness_label: &str,
    //     train_metric_label: &str,
    // ) -> Self
    // where
    //     S: Fn(&Variable, &Variable) -> Result<f32> + Send + Sync + 'static,
    //     L: Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static,
    // {
    //     Fitness {
    //         score_fn: Box::new(score_fn),
    //         train_metric_fn: Box::new(train_metric_fn),
    //         score_direction,
    //         train_metric_direction,
    //         fitness_label: fitness_label.to_string(),
    //         train_metric_label: train_metric_label.to_string(),
    //     }
    // }

    /// Same function for both scoring and training.
    ///
    /// ```text
    /// Fitness::from_loss(|pred, y| mse_loss(pred, y), Direction::Minimize, "mse")
    /// ```
    pub fn from_loss<F>(f: F, direction: Direction, label: &str) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static,
    {
        let f = std::sync::Arc::new(f);
        let f2 = f.clone();
        Fitness {
            score_fn: Box::new(move |pred, y| {
                Ok(f(pred, y)?.item()? as f32)
            }),
            train_metric_fn: Box::new(move |pred, y| f2(pred, y)),
            score_direction: direction,
            train_metric_direction: direction,
            fitness_label: label.to_string(),
            train_metric_label: label.to_string(),
        }
    }

    /// Score a prediction against its target — the scalar ranking metric.
    /// Called on **eval** batches to rank individuals.
    pub fn score(&self, pred: &Variable, target: &Variable) -> Result<f32> {
        (self.score_fn)(pred, target)
    }

    /// Compute the training metric tensor — used for backward (training).
    pub fn train_metric(&self, pred: &Variable, target: &Variable) -> Result<Variable> {
        (self.train_metric_fn)(pred, target)
    }

    /// Which direction is better for the **score** (ranking individuals).
    pub fn direction(&self) -> Direction {
        self.score_direction
    }

    /// Which direction is better for the **train metric** (training signal).
    pub fn train_metric_direction(&self) -> Direction {
        self.train_metric_direction
    }

    /// Fitness/scoring label for logs.
    pub fn fitness_label(&self) -> &str {
        &self.fitness_label
    }

    /// Train metric label for logs.
    pub fn train_metric_label(&self) -> &str {
        &self.train_metric_label
    }

    /// Whether fitness and train metric use the same function.
    pub fn train_metric_is_fitness(&self) -> bool {
        self.fitness_label == self.train_metric_label
    }
}

// ── Scoring helpers — public utility functions ────────────────────────────

/// `1 − SS_res/SS_tot` — fraction of target variance explained (higher =
/// better; `1.0` = perfect).
pub fn r2_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let t = y.data().to_f32_vec()?;
    let p = pred.data().to_f32_vec()?;
    let n = t.len() as f32;
    if n == 0.0 {
        return Ok(0.0);
    }
    let mean_t = t.iter().sum::<f32>() / n;
    let ss_tot: f32 = t.iter().map(|&v| (v - mean_t).powi(2)).sum();
    let ss_res: f32 = t.iter().zip(&p).map(|(&a, &b)| (a - b).powi(2)).sum();
    if ss_tot.abs() < f32::EPSILON {
        return Ok(if ss_res.abs() < f32::EPSILON { 1.0 } else { 0.0 });
    }
    Ok(1.0 - ss_res / ss_tot)
}

/// Argmax class index per row, as plain `Vec<i64>`.
pub fn argmax_classes(pred: &Variable, y: &Variable) -> Result<(Vec<i64>, Vec<i64>)> {
    let pa = pred.data().argmax(1, false)?.to_i64_vec()?;
    let ta = y.data().argmax(1, false)?.to_i64_vec()?;
    Ok((pa, ta))
}

/// Fraction of rows whose argmax class matches the target (higher = better).
pub fn accuracy_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let n = pa.len() as f32;
    if n == 0.0 {
        return Ok(0.0);
    }
    Ok(pa.iter().zip(&ta).filter(|(a, b)| a == b).count() as f32 / n)
}

/// Macro F1 from argmax class vectors (per-class F1 averaged over classes).
pub fn f1_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let classes = y.data().shape()[1] as usize;
    Ok(f1_from_vecs(&pa, &ta, classes))
}

/// Macro F1 from precomputed argmax vectors.
pub fn f1_from_vecs(pa: &[i64], ta: &[i64], classes: usize) -> f32 {
    if classes == 0 || pa.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for c in 0..classes as i64 {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;
        for (p, t) in pa.iter().zip(ta.iter()) {
            match (*p == c, *t == c) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
        let denom = (tp + fp + tp + fn_) as f32;
        if denom > 0.0 {
            sum += 2.0 * tp as f32 / denom;
        }
    }
    sum / classes as f32
}

/// Macro precision from argmax class vectors.
pub fn precision_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let classes = y.data().shape()[1] as usize;
    Ok(precision_from_vecs(&pa, &ta, classes))
}

/// MSE score (lower = better) — `mse_loss(pred, y).item()`.
pub fn mse_loss_score(pred: &Variable, y: &Variable) -> Result<f32> {
    Ok(flodl::nn::loss::mse_loss(pred, y)?.item()? as f32)
}

/// L1/MAE score (lower = better) — `l1_loss(pred, y).item()`.
pub fn l1_loss_score(pred: &Variable, y: &Variable) -> Result<f32> {
    Ok(flodl::nn::loss::l1_loss(pred, y)?.item()? as f32)
}

/// RMSE score (lower = better) — `sqrt(mse_loss(pred, y).item())`.
pub fn rmse_score(pred: &Variable, y: &Variable) -> Result<f32> {
    Ok(mse_loss_score(pred, y)?.sqrt())
}

/// Macro precision from precomputed argmax vectors.
pub fn precision_from_vecs(pa: &[i64], ta: &[i64], classes: usize) -> f32 {
    if classes == 0 || pa.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for c in 0..classes as i64 {
        let mut tp = 0usize;
        let mut fp = 0usize;
        for (p, t) in pa.iter().zip(ta.iter()) {
            if *p == c {
                if *t == c {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }
        let denom = tp + fp;
        if denom > 0 {
            sum += tp as f32 / denom as f32;
        }
    }
    sum / classes as f32
}

/// Softmax cross-entropy against a **one-hot** target, computed as
/// `−mean(one_hot · log_softmax(pred))` (lower = better).
pub fn cross_entropy_onehot(pred: &Variable, y: &Variable) -> Result<f32> {
    let n = y.data().shape()[0] as f32;
    if n == 0.0 {
        return Ok(0.0);
    }
    let ls = pred.data().log_softmax(1)?;
    let masked = ls.mul(&y.data())?;
    Ok(-masked.sum()?.item()? as f32 / n)
}

/// Softmax cross-entropy loss tensor (for backward).
pub fn cross_entropy_onehot_loss(pred: &Variable, y: &Variable) -> Result<Variable> {
    let ls = pred.data().log_softmax(1)?;
    let masked = ls.mul(&y.data())?;
    let neg = flodl::Tensor::from_f32(&[-1.0], &[1], masked.device())?;
    let n = flodl::Tensor::from_f32(
        &[y.data().shape()[0] as f32], &[1], y.data().device(),
    )?;
    Ok(Variable::new(masked.sum()?.mul(&neg)?.div(&n)?, false))
}

// ── BestIndividual — the current champion ─────────────────────────────────

/// The best individual seen so far — scored by a [`Fitness`].
#[derive(Clone, Debug)]
pub struct BestIndividual {
    pub fitness: f32,
    /// Loss on eval batches (only when Fitness has an explicit loss).
    pub loss: Option<f32>,
    /// Index in the population (`pop[i]`) that produced this best.
    pub pop_index: usize,
    /// The blueprint that scored best — `to_json` it to replicate the net.
    pub topology: crate::topology::Topology,
}

// ── Re-exports for backward compat ────────────────────────────────────────

// Backward-compat: old code used FitnessKind in EngineOptions.
// We keep a serializable label string instead.

/// Serialized fitness label for engine.json (e.g. "mse", "accuracy").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitnessLabel(pub String);

impl Default for FitnessLabel {
    fn default() -> Self {
        FitnessLabel("loss".to_string())
    }
}

impl std::fmt::Display for FitnessLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Network;
    use crate::node::Node;
    use crate::topology::Topology;
    use flodl::nn::Module;

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

    #[test]
    fn test_direction_semantics() {
        assert!(Direction::Minimize.is_better(0.5, 1.0));
        assert!(!Direction::Minimize.is_better(1.0, 0.5));
        assert!(Direction::Maximize.is_better(1.0, 0.5));
        assert!(!Direction::Maximize.is_better(0.5, 1.0));
    }

    #[test]
    fn test_fitness_from_loss() {
        let f = Fitness::from_loss(|pred, y| flodl::nn::loss::mse_loss(pred, y), Direction::Minimize, "mse");
        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let y = input(&[1.0, 2.0, 3.0]);
        let pred = net.forward(&x).unwrap();
        let score = f.score(&pred, &y).unwrap();
        assert!(score.is_finite());
        assert!(score >= 0.0);
        assert_eq!(f.direction(), Direction::Minimize);
        // train_metric() and score() should agree (score = train_metric.item())
        let tm = f.train_metric(&pred, &y).unwrap();
        let tm_v = tm.item().unwrap() as f32;
        assert!((score - tm_v).abs() < 1e-6);
    }

    // TODO: uncomment when from_loss_with_other is re-enabled.
    // #[test]
    // fn test_fitness_from_loss_with_other() {
    //     let f = Fitness::from_loss_with_other(
    //         accuracy_score,
    //         |pred, y| flodl::nn::loss::mse_loss(pred, y),
    //         Direction::Maximize,
    //         Direction::Minimize,
    //         "accuracy",
    //         "mse",
    //     );
    //     ...
    // }

    #[test]
    fn test_scoring_helpers() {
        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let pred = net.forward(&x).unwrap();
        // Use pred as both pred and y for perfect fit.
        let y = Variable::new(pred.data().clone(), false);
        let r2 = r2_score(&pred, &y).unwrap();
        assert!((r2 - 1.0).abs() < 1e-3, "perfect fit → R2 ≈ 1, got {r2}");
        let mse = flodl::nn::loss::mse_loss(&pred, &y).unwrap().item().unwrap() as f32;
        assert!(mse < 1e-6);
    }

    #[test]
    fn test_categorical_helpers() {
        let mut graph = Topology::new(0, None);
        graph.options.input_dim = 2;
        graph.options.hidden_dim = 4;
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
            crate::utils::synthetic::one_hot(
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
}
