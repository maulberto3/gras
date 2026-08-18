//! Scoring strategies 🎯 — built-ins, the drop-in custom path, and the
//! direction (lower or higher = better).
//!
//! The engine consumes a scorer through the [`Fitness`] enum: either a
//! built-in ([`FitnessKind`], serializable so a run's `engine.json` records
//! which one was used) or your own — a plain closure via
//! [`Fitness::custom`], or a reusable named scorer implementing the
//! [`FitnessScorer`] trait via [`Fitness::scorer`]. Every route flows
//! through [`Fitness::evaluate`] identically, so built-ins and custom
//! scorers get the same evaluation/checkpoint/logging treatment.
//!
//! Every scorer knows its [`Direction`]: regression losses are
//! [`Direction::Minimize`], metrics like accuracy are [`Direction::Maximize`].
//! The engine compares candidates with that direction (internally normalized,
//! presented in user space), and custom scorers pick theirs at construction.
//!
//! This module also owns the **canonical datasets** for the built-ins — one
//! synthetic generator per scorer — so each built-in's data lives next to
//! its scorer. (The generic tensor I/O contract itself stays in
//! [`crate::data`].)
//!
//! # Data contracts
//!
//! - **Continuous** kinds ([`FitnessKind::Mse`], `Mae`, `Rmse`, `R2`):
//!   targets `[n, 1]`.
//! - **Categorical** kinds (`Accuracy`, `CrossEntropy`, `F1`, `Nll`):
//!   targets are **one-hot** `[n, C]`.

use flodl::nn::loss::{l1_loss, mse_loss};
use flodl::tensor::Result;
use flodl::{Device, Tensor, Variable};
use serde::{Deserialize, Serialize};

use crate::data::Dataset;

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

/// Built-in scoring strategies — the \"use this path, go for 'mse'\" path.
/// Serializable, so the run's `engine.json` records which one was used.
///
/// The **config** of the fitness trio: a serializable, copyable enum
/// (`mse`, `accuracy`, ...) — the one you name in `EngineOptions`, print in
/// logs, and record in `engine.json`. The other two types are [`Fitness`]
/// (the runtime wrapper the engine evaluates with) and [`FitnessScorer`]
/// (the trait a scorer implements).
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

/// The scorer contract: a metric over a **prediction vs its target batch**.
///
/// The engine runs the network's forward pass itself, so a scorer never sees
/// the [`Network`](crate::network::Network), the data path, or the batches —
/// just the two tensors a metric actually compares: what the net predicted
/// and what it should have predicted. Built-ins implement this
/// ([`FitnessKind`]), and [`Fitness::custom`] adapts a plain closure to it.
/// Implement it yourself for a reusable, named scorer:
///
/// ```ignore
/// struct MyScorer;
/// impl FitnessScorer for MyScorer { /* ... */ }
/// Engine::new(opts, path, Fitness::scorer(MyScorer))?;
/// ```
///
/// The `Send + Sync` bounds let the engine score the whole population in
/// parallel (each individual's scorer call runs on its own rayon task).
pub trait FitnessScorer: Send + Sync {
    /// Score `pred` (the network's output for a batch) against `target`.
    /// The engine tracks the best according to the scorer's [`Direction`]
    /// (default: lower = better).
    fn score(&self, pred: &Variable, target: &Variable) -> Result<f64>;
}

impl FitnessScorer for FitnessKind {
    fn score(&self, pred: &Variable, target: &Variable) -> Result<f64> {
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

/// Adapt a plain closure into a [`FitnessScorer`].
struct ClosureScorer<F>(F);

impl<F> FitnessScorer for ClosureScorer<F>
where
    F: Fn(&Variable, &Variable) -> Result<f64> + Send + Sync,
{
    fn score(&self, pred: &Variable, target: &Variable) -> Result<f64> {
        (self.0)(pred, target)
    }
}

/// The **runtime** type of the fitness trio: the scorer the engine actually
/// runs (`Fitness::Builtin(kind)` or a wrapped closure/trait scorer). Pick
/// one with [`Fitness::mse`], [`Fitness::custom`], or [`Fitness::scorer`],
/// and pass it to [`Engine::new`](crate::engine::Engine::new). The other
/// two types: [`FitnessKind`] (the serializable config you name in
/// `EngineOptions`) and [`FitnessScorer`] (the trait a scorer implements).
///
/// - [`Fitness::mse`] — the one-liner built-in.
/// - [`Fitness::custom`] — \"use this path, but I want THIS fitness function\":
///   a closure `(&Variable, &Variable) -> Result<f64>` — the **prediction**
///   and the **target batch** (the engine runs the forward pass itself), so
///   you only write the metric. Defaults to [`Direction::Minimize`]; use
///   [`Fitness::custom_directed`] for maximizers.
/// - [`Fitness::scorer`] — the trait route: a named, reusable
///   [`FitnessScorer`] implementation (default direction Minimize, or
///   [`Fitness::scorer_directed`]).
pub enum Fitness {
    Builtin(FitnessKind),
    Custom {
        scorer: Box<dyn FitnessScorer>,
        direction: Direction,
    },
}

impl Fitness {
    /// Built-in mean-squared-error scoring (lower = better).
    pub fn mse() -> Self {
        Fitness::Builtin(FitnessKind::Mse)
    }

    /// A built-in kind by name — the engine's \"provided 'mse', 'accuracy',
    /// ...\" path
    pub fn from_kind(kind: FitnessKind) -> Self {
        Fitness::Builtin(kind)
    }

    /// Drop in your own scorer as a closure. `f(pred, target) -> score` —
    /// the prediction and the target batch, nothing else (the engine runs
    /// the forward pass). Defaults to tracking the **minimum** as the
    /// current best (lower = better). For higher-is-better, use
    /// [`Fitness::custom_directed`].
    pub fn custom<F>(f: F) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<f64> + Send + Sync + 'static,
    {
        Self::custom_directed(f, Direction::Minimize)
    }

    /// Like [`Fitness::custom`], with an explicit direction.
    pub fn custom_directed<F>(f: F, direction: Direction) -> Self
    where
        F: Fn(&Variable, &Variable) -> Result<f64> + Send + Sync + 'static,
    {
        Fitness::Custom {
            scorer: Box::new(ClosureScorer(f)),
            direction,
        }
    }

    /// Drop in your own scorer as a named [`FitnessScorer`] implementation
    /// (default direction: lower = better).
    pub fn scorer<S>(s: S) -> Self
    where
        S: FitnessScorer + 'static,
    {
        Self::scorer_directed(s, Direction::Minimize)
    }

    /// Like [`Fitness::scorer`], with an explicit direction.
    pub fn scorer_directed<S>(s: S, direction: Direction) -> Self
    where
        S: FitnessScorer + 'static,
    {
        Fitness::Custom {
            scorer: Box::new(s),
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

    /// Score one prediction against its target batch — the single
    /// evaluation path every route (built-in or custom) flows through.
    /// Callers run the network's forward pass first and hand in the result.
    pub fn evaluate(&self, pred: &Variable, target: &Variable) -> Result<f64> {
        match self {
            Fitness::Builtin(kind) => kind.score(pred, target),
            Fitness::Custom { scorer, .. } => scorer.score(pred, target),
        }
    }
}

// ── canonical synthetic datasets ─────────────────────────────────────────
//
// One generator per built-in family, saved through [`crate::data::save_dataset`]
// so the engine consumes them via the same path contract as any real data.
// Continuous generators produce targets `[n, 1]`; categorical ones produce
// **one-hot** targets `[n, C]`.

fn one_hot(classes: &[usize], num_classes: usize, device: Device) -> Result<Tensor> {
    let mut flat = vec![0.0f32; classes.len() * num_classes];
    for (r, &c) in classes.iter().enumerate() {
        flat[r * num_classes + c] = 1.0;
    }
    Tensor::from_f32(&flat, &[classes.len() as i64, num_classes as i64], device)
}

/// Synthetic `y = sin(2πx)`, `x ∈ [-1, 1]` — the canonical smoke-test data
/// for the regression built-ins (input_dim 1).
pub fn synthetic_sine(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.f64() * 2.0 - 1.0;
        xs.push(x as f32);
        ys.push((2.0 * std::f64::consts::PI * x).sin() as f32);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 1], device)?,
        targets: Tensor::from_f32(&ys, &[n as i64, 1], device)?,
    })
}

/// Synthetic cubic `y = x³ − x + 0.5`, `x ∈ [-2, 2]` — a non-monotonic
/// regression target (input_dim 1).
pub fn synthetic_poly3(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.f64() * 4.0 - 2.0;
        xs.push(x as f32);
        ys.push((x * x * x - x + 0.5) as f32);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 1], device)?,
        targets: Tensor::from_f32(&ys, &[n as i64, 1], device)?,
    })
}

/// Synthetic sigmoid `y = 1/(1 + e^{−3x})`, `x ∈ [-2, 2]` — a bounded
/// regression target (input_dim 1).
pub fn synthetic_sigmoid(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.f64() * 4.0 - 2.0;
        xs.push(x as f32);
        ys.push((1.0 / (1.0 + (-3.0 * x).exp())) as f32);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 1], device)?,
        targets: Tensor::from_f32(&ys, &[n as i64, 1], device)?,
    })
}

/// Synthetic 2-output regression `y = [sin(2πx), cos(2πx)]` — exercises
/// multi-dim targets (input_dim 1, output_dim 2).
pub fn synthetic_multi_sine(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let x = rng.f64() * 2.0 - 1.0;
        xs.push(x as f32);
        let a = 2.0 * std::f64::consts::PI * x;
        ys.push(a.sin() as f32);
        ys.push(a.cos() as f32);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 1], device)?,
        targets: Tensor::from_f32(&ys, &[n as i64, 2], device)?,
    })
}

/// Synthetic XOR — 2 features, 2 classes (one-hot `[n, 2]`).
pub fn synthetic_xor(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n * 2);
    let mut classes = Vec::with_capacity(n);
    for _ in 0..n {
        let a = if rng.f64() < 0.5 { 0.0 } else { 1.0 };
        let b = if rng.f64() < 0.5 { 0.0 } else { 1.0 };
        xs.push(a as f32);
        xs.push(b as f32);
        classes.push(if (a as usize) != (b as usize) { 1 } else { 0 });
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 2], device)?,
        targets: one_hot(&classes, 2, device)?,
    })
}

/// Synthetic Gaussian blobs — 3 well-separated 2-D clusters, 3 classes
/// (one-hot `[n, 3]`).
pub fn synthetic_blobs(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let centers = [(0.0f64, 0.0f64), (4.0, 0.0), (2.0, 4.0)];
    let mut xs = Vec::with_capacity(n * 2);
    let mut classes = Vec::with_capacity(n);
    for _ in 0..n {
        let c = rng.usize(0..centers.len());
        let (cx, cy) = centers[c];
        xs.push((cx + gauss(&mut rng)) as f32);
        xs.push((cy + gauss(&mut rng)) as f32);
        classes.push(c);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 2], device)?,
        targets: one_hot(&classes, 3, device)?,
    })
}

/// Synthetic 2-arm spiral — 2 classes intertwined (one-hot `[n, 2]`); a
/// harder benchmark than blobs for a fixed topology.
pub fn synthetic_spiral(n: usize, _seed: u64, device: Device) -> Result<Dataset> {
    // `_seed` — the geometry is deterministic, no RNG needed (kept for a
    // uniform dataset-generator signature).
    let mut xs = Vec::with_capacity(n * 2);
    let mut classes = Vec::with_capacity(n);
    let n2 = (n / 2).max(1);
    for c in 0..2usize {
        for i in 0..n2 {
            let t = i as f64 / n2 as f64 * 4.0 * std::f64::consts::PI;
            let r = t / (4.0 * std::f64::consts::PI);
            let sign = if c == 0 { 1.0 } else { -1.0 };
            let x = r * t.cos() + sign * 0.1;
            let y = r * t.sin() + sign * 0.1;
            let j = if c == 0 { i } else { n2 + i };
            if j < n {
                xs.push(x as f32);
                xs.push(y as f32);
                classes.push(c);
            }
        }
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[xs.len() as i64 / 2, 2], device)?,
        targets: one_hot(&classes, 2, device)?,
    })
}

/// Synthetic iris-like — 4 features, 3 classes (one-hot `[n, 3]`).
pub fn synthetic_iris_like(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let centers = [
        [5.0f64, 3.4, 1.5, 0.2],
        [5.9, 2.8, 4.2, 1.3],
        [6.5, 3.0, 5.5, 2.0],
    ];
    let mut xs = Vec::with_capacity(n * 4);
    let mut classes = Vec::with_capacity(n);
    for _ in 0..n {
        let c = rng.usize(0..centers.len());
        for &v in &centers[c] {
            xs.push((v + 0.2 * gauss(&mut rng)) as f32);
        }
        classes.push(c);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 4], device)?,
        targets: one_hot(&classes, 3, device)?,
    })
}

/// Standard normal sample via the Box–Muller transform.
fn gauss(rng: &mut fastrand::Rng) -> f64 {
    let u1 = (rng.f64() + 1e-9).min(1.0 - 1e-9);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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
        Network::build(&graph, Device::CPU).unwrap()
    }

    fn input(xs: &[f32]) -> Variable {
        Variable::new(
            Tensor::from_f32(xs, &[xs.len() as i64, 1], Device::CPU).unwrap(),
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
        // Sanity: the built-in Mse scorer returns a finite scalar for a
        // tiny hand-built network.
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
        // Deterministic: use the network's own prediction as the target, so
        // every "perfect" metric must hit its ideal — R2 ≈ 1, accuracy 1.0.
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
        // Pred == target ⇒ accuracy 1.0 and a strictly positive
        // cross-entropy (finite logits). F1/precision need class coverage,
        // so they're tested at the vec level below.
        let mut graph = crate::topology::Topology::new(0, None);
        graph.options.input_dim = 2;
        graph.options.hidden_dim = 4;
        graph.nodes.push(crate::node::Node::new_input(0, 2));
        graph.nodes.push(crate::node::Node::new_hidden(1, 2, 2));
        graph.nodes.push(crate::node::Node::new_output(2, 2, 2));
        // The output tensor's width is hidden_dim (ports only fan out the
        // same tensor) — cap it at 2 so logits are 2-class.
        graph.nodes[2].hidden_dim = Some(2);
        graph.finalize();
        let net = Network::build(&graph, Device::CPU).unwrap();
        let x = Variable::new(
            Tensor::from_f32(&[2.0, 0.5, 0.1, 0.9, 1.2, 0.3], &[3, 2], Device::CPU).unwrap(),
            false,
        );
        let pred = net.forward(&x).unwrap();
        // Target = the network's own argmax as one-hot → accuracy must be 1.
        let pa = pred.data().argmax(1, false).unwrap().to_i64_vec().unwrap();
        let y = Variable::new(
            one_hot(
                &pa.iter().map(|&v| v as usize).collect::<Vec<_>>(),
                2,
                Device::CPU,
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
        // Perfect agreement over both classes → 1.0.
        assert_eq!(f1_from_vecs(&[0, 1, 0, 1], &[0, 1, 0, 1], 2), 1.0);
        assert_eq!(precision_from_vecs(&[0, 1, 0, 1], &[0, 1, 0, 1], 2), 1.0);
        // All predictions class 0 against a mixed target: class 0 prec 0.5,
        // class 1 prec 0 (no predictions) → macro 0.25.
        let pa = [0i64, 0, 0, 0];
        let ta = [0i64, 1, 0, 1];
        assert!((precision_from_vecs(&pa, &ta, 2) - 0.25).abs() < 1e-9);
        // F1: class 0 → tp 2, fp 2, fn 2 → 2·2/6 = 2/3; class 1 → 0 → 1/3.
        assert!((f1_from_vecs(&pa, &ta, 2) - 1.0 / 3.0).abs() < 1e-9);
        // Empty / no classes are degenerate but safe.
        assert_eq!(f1_from_vecs(&[], &[], 2), 0.0);
        assert_eq!(precision_from_vecs(&[0], &[0], 0), 0.0);
    }

    #[test]
    fn test_synthetic_datasets_contracts() {
        // Continuous: [n, 1]; categorical: one-hot [n, C] rows sum to 1.
        for ds in [
            synthetic_sine(64, 1, Device::CPU).unwrap(),
            synthetic_poly3(64, 1, Device::CPU).unwrap(),
            synthetic_sigmoid(64, 1, Device::CPU).unwrap(),
            synthetic_multi_sine(64, 1, Device::CPU).unwrap(),
        ] {
            assert_eq!(ds.inputs.shape()[0], ds.targets.shape()[0]);
            assert_eq!(ds.inputs.dtype(), flodl::DType::Float32);
        }
        let multi = synthetic_multi_sine(16, 1, Device::CPU).unwrap();
        assert_eq!(multi.targets.shape(), &[16, 2]);

        let xor = synthetic_xor(32, 1, Device::CPU).unwrap();
        assert_eq!(xor.targets.shape(), &[32, 2]);
        let row_sums: f32 = xor.targets.sum().unwrap().item().unwrap() as f32;
        assert!(
            (row_sums - 32.0).abs() < 1e-4,
            "one-hot rows sum to 1 per row"
        );

        let blobs = synthetic_blobs(32, 1, Device::CPU).unwrap();
        assert_eq!(blobs.targets.shape(), &[32, 3]);
        let spiral = synthetic_spiral(32, 1, Device::CPU).unwrap();
        assert_eq!(spiral.targets.shape(), &[32, 2]);
        let iris = synthetic_iris_like(32, 1, Device::CPU).unwrap();
        assert_eq!(iris.targets.shape(), &[32, 3]);
        assert_eq!(iris.inputs.shape(), &[32, 4]);
    }

    #[test]
    fn test_fitness_custom_closure_and_trait_routes() {
        // Both custom routes — the closure adapter and the trait impl — must
        // score through the same evaluate() path as the built-in.
        let closure = Fitness::custom(|pred, y| {
            let diff = pred.data().sub(&y.data())?;
            diff.abs()?.mean()?.item()
        });
        struct AbsMean;
        impl FitnessScorer for AbsMean {
            fn score(&self, pred: &Variable, target: &Variable) -> Result<f64> {
                let diff = pred.data().sub(&target.data())?;
                diff.abs()?.mean()?.item()
            }
        }
        let via_trait = Fitness::scorer(AbsMean);

        let net = tiny_net();
        let x = input(&[1.0, 2.0, 3.0]);
        let y = input(&[1.5, 2.5, 3.5]);
        let pred = net.forward(&x).unwrap();
        let a = closure.evaluate(&pred, &y).unwrap();
        let b = via_trait.evaluate(&pred, &y).unwrap();
        assert!((a - b).abs() < 1e-9);
        assert!(a.is_finite());
        // Default direction is Minimize; the directed variant overrides.
        assert_eq!(closure.direction(), Direction::Minimize);
        let max = Fitness::custom_directed(|_, _| Ok(1.0), Direction::Maximize);
        assert_eq!(max.direction(), Direction::Maximize);
    }
}
