//! Training — the [`Trainer`] trait is the primary extension point.
//!
//! Implement [`Trainer`] for full control over your training loop (RL,
//! segmentation, early stopping, anything). For quick custom loops, wrap a
//! single closure with [`from_fn`] — no trait boilerplate.
//! [`supervised::SupervisedTrainer`] is an optional built-in convenience
//! for standard SGD/Adam supervised learning; it is not the intended path.

pub mod supervised;
// Flat re-export so `gras::trainer::{SupervisedTrainer, TrainingConfig,
// OptimizerKind}` works alongside `gras::trainer::supervised::…`.
pub use supervised::{OptimizerKind, SupervisedTrainer, TrainingConfig};

use flodl::tensor::Result;
use flodl::{DType, Device};

use crate::engine::fitness::Fitness;
use crate::graph::network::Network;

// ── EvalOutcome — what one evaluate() produced ──────────────────────────────

/// Everything one [`Trainer::evaluate`] call returns: the ranking score plus
/// optional per-step train/test loss curves (overfitting tracking). The
/// engine records the curves on each individual in the gen JSONs; leave the
/// vectors empty when your trainer can't produce them.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EvalOutcome {
    /// Fitness score — used for evolutionary ranking (respect the engine's
    /// [`Fitness`] direction).
    pub score: f32,
    /// Held-out loss after training (final epoch), for logs/robustness only.
    pub eval_loss: Option<f32>,
    /// Parameter count, used for the robustness table.
    pub param_count: usize,
    /// Training loss **per step (batch)**, oldest first. With 1 epoch × 8
    /// train batches → 8 values.
    pub train_losses: Vec<f32>,
    /// Mean held-out loss **per eval pass** (one per epoch), oldest first —
    /// all eval batches see the same fixed model, so only the pass mean is
    /// meaningful. The last entry equals `eval_loss`. Plot against
    /// `train_losses` to spot overfitting.
    pub eval_losses: Vec<f32>,
}

impl EvalOutcome {
    /// A score-only outcome with no loss curves.
    pub fn new(score: f32, eval_loss: Option<f32>, param_count: usize) -> Self {
        EvalOutcome {
            score,
            eval_loss,
            param_count,
            train_losses: Vec::new(),
            eval_losses: Vec::new(),
        }
    }
}

/// Keep the old tuple return working: `Ok((score, loss, params))` closures
/// still compile through [`from_fn`] — they just produce no loss curves.
impl From<(f32, Option<f32>, usize)> for EvalOutcome {
    fn from((score, eval_loss, param_count): (f32, Option<f32>, usize)) -> Self {
        EvalOutcome::new(score, eval_loss, param_count)
    }
}

// ── Trainer trait ───────────────────────────────────────────────────────────

/// Trait for training and evaluating a single network.
///
/// The engine builds a network from a topology, then calls `evaluate()`.
/// You own the whole pipeline: data, training loop, scoring.
///
/// # Implementors
///
/// - [`Trainer::from_fn`] — wrap a single closure (no boilerplate)
/// - [`supervised::SupervisedTrainer`] — optional built-in for standard
///   supervised learning
/// - Custom: implement this trait for your own use case
pub trait Trainer: Send + Sync {
    /// Input dimension — must match the data you feed `forward`.
    fn input_dim(&self) -> usize;
    /// Output dimension — must match your targets.
    fn output_dim(&self) -> usize;
    /// Device for network building and data placement.
    fn device(&self) -> Device;
    /// Data type for network weights and data.
    fn dtype(&self) -> DType;
    /// Train this network and return the [`EvalOutcome`].
    ///
    /// - `score` — used for evolutionary selection.
    /// - `eval_loss` — optional, used for logging/robustness only.
    /// - `param_count` — optional, used for the robustness table.
    /// - `train_losses` / `eval_losses` — optional curves recorded verbatim
    ///   on each individual in the gen JSONs: per-step train loss, and the
    ///   per-epoch eval-pass mean (all eval batches see the same fixed
    ///   model, so only the pass mean is meaningful).
    ///
    /// `gen_seed` — engine's current generation seed for data shuffling.
    /// Same seed + same options must produce the same outcome.
    fn evaluate(&self, net: Network, fitness: &Fitness, gen_seed: u64) -> Result<EvalOutcome>;

    /// Dropout probability this trainer expects on hidden nodes (0.0 = none).
    /// The engine bakes it into every network it builds and records it in
    /// each topology, so the knob lives with the training config — the
    /// evolution engine itself does not own a dropout setting. Custom
    /// trainers default to 0.0 (they own their regularization); override to
    /// opt in, or use [`from_fn`] + [`ClosureTrainer::with_dropout`].
    fn dropout_prob(&self) -> f32 {
        0.0
    }
}

/// `Box<T>` (including `Box<dyn Trainer>`) acts as a [`Trainer`] — so
/// `Engine::new` accepts concrete trainers, closure adapters, and explicitly
/// boxed trait objects alike.
impl<T: Trainer + ?Sized> Trainer for Box<T> {
    fn input_dim(&self) -> usize {
        (**self).input_dim()
    }
    fn output_dim(&self) -> usize {
        (**self).output_dim()
    }
    fn device(&self) -> Device {
        (**self).device()
    }
    fn dtype(&self) -> DType {
        (**self).dtype()
    }
    fn evaluate(&self, net: Network, fitness: &Fitness, gen_seed: u64) -> Result<EvalOutcome> {
        (**self).evaluate(net, fitness, gen_seed)
    }
    fn dropout_prob(&self) -> f32 {
        (**self).dropout_prob()
    }
}

// ── Closure adapter ─────────────────────────────────────────────────────────

/// Build a trainer from a single closure that owns the whole training
/// pipeline — your data, your loss, your schedule.
///
/// The closure may return either the classic `(score, eval_loss,
/// param_count)` tuple or a full [`EvalOutcome`] (when you want per-epoch
/// loss curves persisted on each individual). The engine ranks on `score`.
///
/// Returns a [`ClosureTrainer`], which implements [`Trainer`].
pub fn from_fn<F, O>(
    input_dim: usize,
    output_dim: usize,
    device: Device,
    dtype: DType,
    evaluate_fn: F,
) -> ClosureTrainer
where
    F: Fn(Network, u64) -> Result<O> + Send + Sync + 'static,
    O: Into<EvalOutcome>,
{
    ClosureTrainer::new(input_dim, output_dim, device, dtype, evaluate_fn)
}

/// A [`Trainer`] built from a single closure — see [`from_fn`].
pub struct ClosureTrainer {
    input_dim: usize,
    output_dim: usize,
    device: Device,
    dtype: DType,
    dropout_prob: f32,
    evaluate_fn: Box<dyn Fn(Network, u64) -> Result<EvalOutcome> + Send + Sync>,
}

impl ClosureTrainer {
    /// Wrap a closure as a [`Trainer`]. Equivalent to [`from_fn`].
    pub fn new<F, O>(
        input_dim: usize,
        output_dim: usize,
        device: Device,
        dtype: DType,
        evaluate_fn: F,
    ) -> Self
    where
        F: Fn(Network, u64) -> Result<O> + Send + Sync + 'static,
        O: Into<EvalOutcome>,
    {
        ClosureTrainer {
            input_dim,
            output_dim,
            device,
            dtype,
            dropout_prob: 0.0,
            evaluate_fn: Box::new(move |net, gen_seed| evaluate_fn(net, gen_seed).map(Into::into)),
        }
    }

    /// Opt into dropout on hidden nodes for this closure trainer. The engine
    /// bakes the value into every network it builds (and records it in each
    /// topology). Default: 0.0 — a custom training loop owns its own
    /// regularization.
    pub fn with_dropout(mut self, dropout_prob: f32) -> Self {
        self.dropout_prob = dropout_prob.clamp(0.0, 1.0);
        self
    }
}

impl Trainer for ClosureTrainer {
    fn input_dim(&self) -> usize {
        self.input_dim
    }
    fn output_dim(&self) -> usize {
        self.output_dim
    }
    fn device(&self) -> Device {
        self.device
    }
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn evaluate(&self, net: Network, _fitness: &Fitness, gen_seed: u64) -> Result<EvalOutcome> {
        (self.evaluate_fn)(net, gen_seed)
    }
    fn dropout_prob(&self) -> f32 {
        self.dropout_prob
    }
}
