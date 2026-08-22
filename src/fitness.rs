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

/// The fitness/loss interface — separates **scoring** (ranking individuals)
/// from **training** (backward pass). The engine calls `score()` on held-out
/// data to rank, and `loss()` on training data to update weights.
///
/// ```text
/// Fitness::new(score_fn)                       ← loss defaults to MSE
/// Fitness::new_with_loss(score_fn, loss_fn)    ← explicit training loss
/// ```
pub struct Fitness {
    /// The ranking metric — `(pred, target) → f32` score.
    /// Called on **eval** batches to rank individuals.
    score_fn: Box<dyn Fn(&Variable, &Variable) -> Result<f32> + Send + Sync>,
    /// The training loss — `(pred, target) → Variable` for backward.
    /// If None, defaults to MSE loss.
    loss_fn: Option<Box<dyn Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync>>,
    /// Which direction is better for the **score** (ranking individuals).
    score_direction: Direction,
    /// Which direction is better for the **loss** (training signal).
    /// Always Minimize in practice (backward minimizes), but explicit
    /// here for API clarity and logging.
    loss_direction: Direction,
    /// Human label for logs (e.g. "accuracy", "mse").
    label: String,
}

impl Fitness {
    /// Create a fitness with a scoring function. The training loss defaults
    /// to MSE when `loss()` is called.
    ///
    /// ```text
    /// Fitness::new(|pred, y| accuracy(pred, y))
    /// // ranking: accuracy (↑ higher is better)
    /// // training: MSE (↓ lower is better)
    /// ```
    pub fn new<F>(score_fn: F, score_direction: Direction, label: &str) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<f32> + Send + Sync + 'static,
    {
        Fitness {
            score_fn: Box::new(score_fn),
            loss_fn: None,
            score_direction,
            loss_direction: Direction::Minimize, // MSE default, always minimize
            label: label.to_string(),
        }
    }

    /// Create a fitness with both a scoring function and a training loss.
    ///
    /// ```text
    /// Fitness::new_with_loss(
    ///     |pred, y| accuracy(pred, y),           // ranking metric
    ///     |pred, y| cross_entropy(pred, y),       // training loss
    ///     Direction::Maximize,
    ///     "accuracy",
    /// )
    /// ```
    pub fn new_with_loss<S, L>(
        score_fn: S,
        loss_fn: L,
        score_direction: Direction,
        loss_direction: Direction,
        label: &str,
    ) -> Self
    where
        S: Fn(&Variable, &Variable) -> Result<f32> + Send + Sync + 'static,
        L: Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static,
    {
        Fitness {
            score_fn: Box::new(score_fn),
            loss_fn: Some(Box::new(loss_fn)),
            score_direction,
            loss_direction,
            label: label.to_string(),
        }
    }

    /// Convenience: same function for both scoring and loss (returns Variable,
    /// score = `.item()`). Direction defaults to Minimize.
    pub fn from_loss<F>(f: F) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static,
    {
        let f = std::sync::Arc::new(f);
        let f2 = f.clone();
        Fitness {
            score_fn: Box::new(move |pred, y| {
                Ok(f(pred, y)?.item()? as f32)
            }),
            loss_fn: Some(Box::new(move |pred, y| f2(pred, y))),
            score_direction: Direction::Minimize,
            loss_direction: Direction::Minimize,
            label: "loss".to_string(),
        }
    }

    /// Score a prediction against its target — the scalar ranking metric.
    /// Called on **eval** batches to rank individuals.
    pub fn score(&self, pred: &Variable, target: &Variable) -> Result<f32> {
        (self.score_fn)(pred, target)
    }

    /// Compute the loss tensor — used for backward (training).
    /// If no explicit loss was provided, defaults to MSE.
    pub fn loss(&self, pred: &Variable, target: &Variable) -> Result<Variable> {
        match &self.loss_fn {
            Some(f) => f(pred, target),
            None => flodl::nn::loss::mse_loss(pred, target),
        }
    }

    /// Which direction is better for the **score** (ranking individuals).
    pub fn direction(&self) -> Direction {
        self.score_direction
    }

    /// Which direction is better for the **loss** (training signal).
    pub fn loss_direction(&self) -> Direction {
        self.loss_direction
    }

    /// Human label for logs.
    pub fn label(&self) -> &str {
        &self.label
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
        let f = Fitness::from_loss(|pred, y| flodl::nn::loss::mse_loss(pred, y));
        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let y = input(&[1.0, 2.0, 3.0]);
        let pred = net.forward(&x).unwrap();
        let score = f.score(&pred, &y).unwrap();
        assert!(score.is_finite());
        assert!(score >= 0.0);
        assert_eq!(f.direction(), Direction::Minimize);
        // loss() and score() should agree (score = loss.item())
        let loss_t = f.loss(&pred, &y).unwrap();
        let loss_v = loss_t.item().unwrap() as f32;
        assert!((score - loss_v).abs() < 1e-6);
    }

    #[test]
    fn test_fitness_new_with_loss() {
        let f = Fitness::new_with_loss(
            accuracy_score,
            |pred, y| flodl::nn::loss::mse_loss(pred, y),
            Direction::Maximize,  // score: accuracy
            Direction::Minimize,  // loss: MSE
            "accuracy",
        );
        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let y = input(&[1.0, 2.0, 3.0]);
        let pred = net.forward(&x).unwrap();
        let score = f.score(&pred, &y).unwrap();
        assert!(score.is_finite());
        assert!(score >= 0.0 && score <= 1.0);
        assert_eq!(f.direction(), Direction::Maximize);
        assert_eq!(f.label(), "accuracy");
        // loss() returns MSE, not accuracy
        let loss_t = f.loss(&pred, &y).unwrap();
        assert!(loss_t.item().unwrap().is_finite());
    }

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
