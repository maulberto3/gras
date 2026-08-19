//! Scoring strategies 🎯 — built-ins, the drop-in custom path, and the
//! direction (lower or higher = better).
//!
//! The engine consumes a scorer through the [`Fitness`] enum: either a
//! built-in ([`FitnessKind`], serializable so a run's `engine.json` records
//! which one was used) or your own — a plain closure via
//! [`Fitness::loss_fn`]. Every route flows through both
//! [`Fitness::compute_loss`] (training: returns `Variable` for backward)
//! and [`Fitness::evaluate`] (scoring: returns scalar for ranking).
//!
//! Every scorer knows its [`Direction`]: regression losses are
//! [`Direction::Minimize`], metrics like accuracy are [`Direction::Maximize`].
//! The engine compares candidates with that direction (internally normalized,
//! presented in user space), and custom scorers pick theirs at construction.

use flodl::Variable;
use flodl::nn::loss::{cross_entropy_loss, l1_loss, mse_loss};
use flodl::tensor::Result;
use serde::{Deserialize, Serialize};

/// Whether a **lower** or a **higher** score is better.
///
/// - Losses (`Mse`, `Mae`, ...) → [`Minimize`](Direction::Minimize).
/// - Metrics (`R2`, `Accuracy`, `F1`) → [`Maximize`](Direction::Maximize).
///
/// The engine compares candidates with this direction; the run's logging and
/// `improvements/` filenames always show the raw user-space score.
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
    pub fn is_better(&self, new: f64, current: f64) -> bool {
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
    pub fn cmp(&self, a: f64, b: f64) -> std::cmp::Ordering {
        match self {
            // Lower is better: a beats b when a < b.
            Direction::Minimize => b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal),
            // Higher is better: a beats b when a > b.
            Direction::Maximize => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
        }
    }
}

/// Built-in scoring strategies — the "use this path, go for 'mse'" path.
/// Serializable, so the run's `engine.json` records which one was used.
///
/// The serializable config: a copyable enum (`mse`, `accuracy`, ...) —
/// the one you name in `EngineOptions`, print in logs, and record in
/// `engine.json`. The runtime type is [`Fitness`] (wraps `FitnessKind`
/// or a custom closure).
///
/// Four continuous kinds (targets `[n, 1]`) and four categorical kinds
/// (targets one-hot `[n, C]`). Each declares its [`Direction`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitnessKind {
    /// Mean squared error between prediction and target (lower = better).
    #[default]
    Mse,
    /// Mean absolute error (lower = better).
    Mae,
    /// Root mean squared error (lower = better).
    Rmse,
    /// Coefficient of determination, `1 − SS_res/SS_tot` (higher = better).
    R2,
    /// Fraction of argmax-matched rows (higher = better; one-hot targets).
    Accuracy,
    /// Softmax cross-entropy on logits (lower = better; one-hot targets).
    CrossEntropy,
    /// Macro F1 over classes (higher = better; one-hot targets).
    F1,
    /// Macro precision over classes (higher = better; one-hot targets).
    Precision,
}

impl FitnessKind {
    /// Scalar score for this built-in — the ranking metric. For losses
    /// (Mse, Mae, Rmse, CrossEntropy) this is the loss value; for metrics
    /// (R2, Accuracy, F1, Precision) this is the specialized scoring math
    /// (argmax, confusion matrix, etc.).
    pub fn score(&self, pred: &Variable, target: &Variable) -> Result<f64> {
        match self {
            FitnessKind::Mse => mse_loss(pred, target)?.item(),
            FitnessKind::Mae => l1_loss(pred, target)?.item(),
            FitnessKind::Rmse => Ok(mse_loss(pred, target)?.item()?.sqrt()),
            FitnessKind::R2 => r2_score(pred, target),
            FitnessKind::Accuracy => accuracy_score(pred, target),
            FitnessKind::CrossEntropy => cross_entropy_onehot(pred, target),
            FitnessKind::F1 => f1_score(pred, target),
            FitnessKind::Precision => precision_score(pred, target),
        }
    }

    /// Which direction is better for this built-in.
    pub fn direction(self) -> Direction {
        match self {
            FitnessKind::Mse | FitnessKind::Mae | FitnessKind::Rmse | FitnessKind::CrossEntropy => {
                Direction::Minimize
            }
            FitnessKind::R2 | FitnessKind::Accuracy | FitnessKind::F1 | FitnessKind::Precision => {
                Direction::Maximize
            }
        }
    }

    /// A short human label for logs.
    pub fn label(self) -> &'static str {
        match self {
            FitnessKind::Mse => "mse",
            FitnessKind::Mae => "mae",
            FitnessKind::Rmse => "rmse",
            FitnessKind::R2 => "r2",
            FitnessKind::Accuracy => "acc",
            FitnessKind::CrossEntropy => "xent",
            FitnessKind::F1 => "f1",
            FitnessKind::Precision => "prec",
        }
    }
}

// ── scorer math (continuous) ─────────────────────────────────────────────

/// `1 − SS_res/SS_tot` — fraction of target variance explained (higher =
/// better; `1.0` = perfect). Handles the degenerate `ss_tot == 0` case by
/// scoring `1.0` when predictions match exactly, `0.0` otherwise.
fn r2_score(pred: &Variable, y: &Variable) -> Result<f64> {
    // Accumulate in f64 over the raw f32 values — no scalar-broadcast tensor
    // hack, no precision lost to an intermediate f32 cast.
    let t = y.data().to_f32_vec()?;
    let p = pred.data().to_f32_vec()?;
    let n = t.len() as f64;
    if n == 0.0 {
        return Ok(0.0);
    }
    let mean_t = t.iter().map(|&v| v as f64).sum::<f64>() / n;
    let ss_tot: f64 = t.iter().map(|&v| (v as f64 - mean_t).powi(2)).sum();
    let ss_res: f64 = t
        .iter()
        .zip(&p)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum();
    if ss_tot.abs() < f64::EPSILON {
        return Ok(if ss_res.abs() < f64::EPSILON {
            1.0
        } else {
            0.0
        });
    }
    Ok(1.0 - ss_res / ss_tot)
}

// ── scorer math (categorical, one-hot targets [n, C]) ────────────────────

/// Argmax class index per row, as a plain `Vec<i64>`.
fn argmax_classes(pred: &Variable, y: &Variable) -> Result<(Vec<i64>, Vec<i64>)> {
    let pa = pred.data().argmax(1, false)?.to_i64_vec()?;
    let ta = y.data().argmax(1, false)?.to_i64_vec()?;
    Ok((pa, ta))
}

/// Fraction of rows whose argmax class matches the target (higher = better).
fn accuracy_score(pred: &Variable, y: &Variable) -> Result<f64> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let n = pa.len() as f64;
    if n == 0.0 {
        return Ok(0.0);
    }
    Ok(pa.iter().zip(&ta).filter(|(a, b)| a == b).count() as f64 / n)
}

/// Macro F1 from argmax class vectors (per-class F1 averaged over classes;
/// higher = better; a class with no predictions contributes 0).
fn f1_from_vecs(pa: &[i64], ta: &[i64], classes: usize) -> f64 {
    if classes == 0 || pa.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for c in 0..classes as i64 {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;
        for (p, t) in pa.iter().zip(ta.iter()) {
            let is_p = *p == c;
            let is_t = *t == c;
            match (is_p, is_t) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
        let denom = (tp + fp + tp + fn_) as f64; // 2·tp + fp + fn
        if denom > 0.0 {
            sum += 2.0 * tp as f64 / denom;
        }
    }
    sum / classes as f64
}

fn f1_score(pred: &Variable, y: &Variable) -> Result<f64> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let classes = y.data().shape()[1] as usize;
    Ok(f1_from_vecs(&pa, &ta, classes))
}

/// Softmax cross-entropy against a **one-hot** target, computed as
/// `−mean(one_hot · log_softmax(pred))` (lower = better). Implemented
/// manually because flodl's `cross_entropy_loss` wrapper only accepts
/// class-index targets.
fn cross_entropy_onehot(pred: &Variable, y: &Variable) -> Result<f64> {
    let n = y.data().shape()[0] as f64;
    if n == 0.0 {
        return Ok(0.0);
    }
    let ls = pred.data().log_softmax(1)?;
    let masked = ls.mul(&y.data())?;
    Ok(-masked.sum()?.item()? / n)
}

/// Macro precision from argmax class vectors (per-class `tp/(tp+fp)`
/// averaged over classes; higher = better; a class with no predictions
/// contributes 0).
fn precision_from_vecs(pa: &[i64], ta: &[i64], classes: usize) -> f64 {
    if classes == 0 || pa.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
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
            sum += tp as f64 / denom as f64;
        }
    }
    sum / classes as f64
}

fn precision_score(pred: &Variable, y: &Variable) -> Result<f64> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let classes = y.data().shape()[1] as usize;
    Ok(precision_from_vecs(&pa, &ta, classes))
}

/// Adapt a plain closure returning `Variable` into a loss that can do
/// both backward (training) and score (ranking via `.item()`).
struct ClosureLoss<F>(F);

impl<F> Loss for ClosureLoss<F>
where
    F: Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync,
{
    fn loss(&self, pred: &Variable, target: &Variable) -> Result<Variable> {
        (self.0)(pred, target)
    }
}

/// The loss/scoring interface — used for both **backward** (training) and
/// **scoring** (ranking). Built-in variants call flodl's loss functions;
/// custom closures wrap a user-provided function. The engine never
/// distinguishes between the two: one `Fitness` drives the whole pipeline.
///
/// - [`Fitness::mse`] — the one-liner built-in.
/// - [`Fitness::loss_fn`] — "use this path, but I want THIS loss function":
///   a closure `(pred, target) -> Variable` — the loss tensor, ready for
///   `.backward()`. The scalar score is extracted via `.item()`.
///   Defaults to [`Direction::Minimize`]; use [`Fitness::loss_directed`]
///   for maximizers.
pub enum Fitness {
    Builtin(FitnessKind),
    Custom {
        #[allow(private_interfaces)]
        loss_fn: Box<dyn Loss>,
        direction: Direction,
    },
}

/// Internal trait for loss functions that return a `Variable` (for
/// backward). Implemented by `FitnessKind` (built-ins) and by closures
/// via `ClosureLoss`.
#[allow(private_interfaces)]
trait Loss: Send + Sync {
    fn loss(&self, pred: &Variable, target: &Variable) -> Result<Variable>;
}

impl Loss for FitnessKind {
    fn loss(&self, pred: &Variable, target: &Variable) -> Result<Variable> {
        match self {
            FitnessKind::Mse => mse_loss(pred, target),
            FitnessKind::Mae => l1_loss(pred, target),
            FitnessKind::Rmse => Ok(mse_loss(pred, target)?),
            FitnessKind::CrossEntropy => cross_entropy_loss(pred, target),
            // R2, Accuracy, F1, Precision: differentiable approximations
            // for backward compatibility — the real scoring uses `.item()`.
            // R2, Accuracy, F1, Precision: use MSE as proxy for backward
            _ => Ok(mse_loss(pred, target)?),
        }
    }
}

impl Fitness {
    /// Built-in mean-squared-error scoring (lower = better).
    pub fn mse() -> Self {
        Fitness::Builtin(FitnessKind::Mse)
    }

    /// Built-in softmax cross-entropy on logits (lower = better;
    /// one-hot targets). The canonical loss for classification — for
    /// MNIST with 10 classes, random guessing starts at ≈ 2.3.
    pub fn cross_entropy() -> Self {
        Fitness::Builtin(FitnessKind::CrossEntropy)
    }

    /// Built-in accuracy scoring (higher = better; one-hot targets).
    pub fn accuracy() -> Self {
        Fitness::Builtin(FitnessKind::Accuracy)
    }

    /// A built-in kind by name — the engine's "provided 'mse', 'accuracy',
    /// ..." path.
    pub fn from_kind(kind: FitnessKind) -> Self {
        Fitness::Builtin(kind)
    }

    /// Drop in your own loss as a closure. `f(pred, target) -> loss_tensor`
    /// — the Variable is used for `.backward()` (training) and `.item()`
    /// gives the scalar score (ranking). Defaults to tracking the
    /// **minimum** as the current best (lower = better). For
    /// higher-is-better, use [`Fitness::loss_directed`].
    ///
    /// # Example
    /// ```ignore
    /// use flodl::mse_loss;
    ///
    /// let fitness = Fitness::loss(|pred, target| mse_loss(pred, target));
    /// ```
    pub fn loss_fn<F>(f: F) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static,
    {
        Self::loss_directed(f, Direction::Minimize)
    }

    /// Like [`Fitness::loss_fn`], with an explicit direction.
    pub fn loss_directed<F>(f: F, direction: Direction) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static,
    {
        Fitness::Custom {
            loss_fn: Box::new(ClosureLoss(f)),
            direction,
        }
    }

    /// Which direction is better for this scorer — drives the engine's
    /// best-tracking, logging and selection.
    pub fn direction(&self) -> Direction {
        match self {
            Fitness::Builtin(kind) => kind.direction(),
            Fitness::Custom { direction, .. } => *direction,
        }
    }

    /// The built-in kind, if any — `None` for custom closures.
    /// Used by the engine to sync `options.fitness` with the runtime scorer.
    pub fn kind(&self) -> Option<FitnessKind> {
        match self {
            Fitness::Builtin(kind) => Some(*kind),
            Fitness::Custom { .. } => None,
        }
    }

    /// Compute the loss tensor — used for backward (training) and score
    /// (ranking via `.item()`). This is the single entry point that both
    /// the trainer and the evaluator use.
    pub fn compute_loss(&self, pred: &Variable, target: &Variable) -> Result<Variable> {
        match self {
            Fitness::Builtin(kind) => kind.loss(pred, target),
            Fitness::Custom { loss_fn, .. } => loss_fn.loss(pred, target),
        }
    }

    /// Score one prediction against its target batch — the scalar score
    /// used for ranking. For built-ins, this uses the specialized scoring
    /// math (argmax for Accuracy, etc.). For custom closures, this
    /// extracts `.item()` from the loss tensor.
    pub fn evaluate(&self, pred: &Variable, target: &Variable) -> Result<f64> {
        match self {
            Fitness::Builtin(kind) => kind.score(pred, target),
            Fitness::Custom { loss_fn, .. } => loss_fn.loss(pred, target)?.item(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Network;
    use flodl::nn::Module;

    fn tiny_net() -> Network {
        let mut graph = crate::topology::Topology::new(0, None);
        graph.nodes.push(crate::node::Node::new_input(0, 1));
        graph.nodes.push(crate::node::Node::new_output(1, 1, 1));
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
    fn test_fitness_kind_scores_with_mse() {
        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let y = input(&[1.0, 2.0, 3.0]);
        let pred = net.forward(&x).unwrap();
        let score = Fitness::mse().evaluate(&pred, &y).unwrap();
        assert!(score.is_finite());
        assert!(score >= 0.0);
        assert_eq!(Fitness::mse().direction(), Direction::Minimize);
    }

    #[test]
    fn test_all_kinds_score_and_direction() {
        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let y = input(&[1.0, 2.0, 3.0]);
        let pred = net.forward(&x).unwrap();
        for kind in [
            FitnessKind::Mse,
            FitnessKind::Mae,
            FitnessKind::Rmse,
            FitnessKind::R2,
            FitnessKind::Accuracy,
            FitnessKind::CrossEntropy,
            FitnessKind::F1,
            FitnessKind::Precision,
        ] {
            let score = Fitness::from_kind(kind).evaluate(&pred, &y).unwrap();
            assert!(score.is_finite(), "{kind:?} must score finite");
        }
    }

    #[test]
    fn test_perfect_prediction_scores_ideal_values() {
        let net = tiny_net();
        let x = input(&[0.5, 0.8, 0.1]);
        let pred = net.forward(&x).unwrap();
        let y = Variable::new(pred.data().clone(), false);
        let r2 = Fitness::from_kind(FitnessKind::R2)
            .evaluate(&pred, &y)
            .unwrap();
        assert!((r2 - 1.0).abs() < 1e-3, "perfect fit → R2 ≈ 1, got {r2}");
        let mse = Fitness::from_kind(FitnessKind::Mse)
            .evaluate(&pred, &y)
            .unwrap();
        assert!(mse < 1e-6);
    }

    #[test]
    fn test_categorical_perfect_matches() {
        let mut graph = crate::topology::Topology::new(0, None);
        graph.options.input_dim = 2;
        graph.options.hidden_dim = 4;
        graph.nodes.push(crate::node::Node::new_input(0, 2));
        graph.nodes.push(crate::node::Node::new_hidden(1, 2, 2));
        graph.nodes.push(crate::node::Node::new_output(2, 2, 2));
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
            crate::synthetic::one_hot(
                &pa.iter().map(|&v| v as usize).collect::<Vec<_>>(),
                2,
                flodl::Device::CPU,
            )
            .unwrap(),
            false,
        );
        let acc = Fitness::from_kind(FitnessKind::Accuracy)
            .evaluate(&pred, &y)
            .unwrap();
        let ce = Fitness::from_kind(FitnessKind::CrossEntropy)
            .evaluate(&pred, &y)
            .unwrap();
        assert!((acc - 1.0).abs() < 1e-6, "acc = {acc}");
        assert!(ce > 0.0 && ce.is_finite(), "ce = {ce}");
        assert_eq!(FitnessKind::Accuracy.direction(), Direction::Maximize);
        assert_eq!(FitnessKind::CrossEntropy.direction(), Direction::Minimize);
        assert_eq!(FitnessKind::F1.direction(), Direction::Maximize);
        assert_eq!(FitnessKind::Precision.direction(), Direction::Maximize);
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
    fn test_fitness_loss_fn_closure_routes() {
        let custom = Fitness::loss_fn(|pred, y| flodl::l1_loss(pred, y));

        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let y = input(&[1.5, 2.5, 3.5]);
        let pred = net.forward(&x).unwrap();
        let a = custom.evaluate(&pred, &y).unwrap();
        let loss_t = custom.compute_loss(&pred, &y).unwrap();
        let b = loss_t.item().unwrap();
        assert!((a - b).abs() < 1e-9);
        assert!(a.is_finite());
        assert_eq!(custom.direction(), Direction::Minimize);
        let max = Fitness::loss_directed(
            |_, _| {
                Ok(flodl::Variable::new(
                    flodl::Tensor::from_f32(&[1.0], &[1], flodl::Device::CPU).unwrap(),
                    true,
                ))
            },
            Direction::Maximize,
        );
        assert_eq!(max.direction(), Direction::Maximize);
    }
}
