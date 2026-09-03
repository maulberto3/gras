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
    /// Train this network and return `(score, eval_loss, param_count)`.
    ///
    /// - `score` — used for evolutionary selection (respect the engine's
    ///   [`Fitness`] direction).
    /// - `eval_loss` — optional, used for logging/robustness only.
    /// - `param_count` — optional, used for the robustness table.
    ///
    /// `gen_seed` — engine's current generation seed for data shuffling.
    fn evaluate(&self, net: Network, fitness: &Fitness, gen_seed: u64) -> Result<(f32, Option<f32>, usize)>;
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
    fn evaluate(&self, net: Network, fitness: &Fitness, gen_seed: u64) -> Result<(f32, Option<f32>, usize)> {
        (**self).evaluate(net, fitness, gen_seed)
    }
}

// ── Closure adapter ─────────────────────────────────────────────────────────

/// Build a trainer from a single closure that owns the whole training
/// pipeline — your data, your loss, your schedule. The closure returns
/// `(score, eval_loss, param_count)`; the engine ranks on `score`.
///
/// Returns a [`ClosureTrainer`], which implements [`Trainer`].
pub fn from_fn<F>(
    input_dim: usize,
    output_dim: usize,
    device: Device,
    dtype: DType,
    evaluate_fn: F,
) -> ClosureTrainer
where
    F: Fn(Network, u64) -> Result<(f32, Option<f32>, usize)> + Send + Sync + 'static,
{
    ClosureTrainer::new(input_dim, output_dim, device, dtype, evaluate_fn)
}

/// A [`Trainer`] built from a single closure — see [`from_fn`].
pub struct ClosureTrainer {
    input_dim: usize,
    output_dim: usize,
    device: Device,
    dtype: DType,
    evaluate_fn: Box<dyn Fn(Network, u64) -> Result<(f32, Option<f32>, usize)> + Send + Sync>,
}

impl ClosureTrainer {
    /// Wrap a closure as a [`Trainer`]. Equivalent to [`from_fn`].
    pub fn new<F>(
        input_dim: usize,
        output_dim: usize,
        device: Device,
        dtype: DType,
        evaluate_fn: F,
    ) -> Self
    where
        F: Fn(Network, u64) -> Result<(f32, Option<f32>, usize)> + Send + Sync + 'static,
    {
        ClosureTrainer {
            input_dim,
            output_dim,
            device,
            dtype,
            evaluate_fn: Box::new(evaluate_fn),
        }
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

    fn evaluate(&self, net: Network, _fitness: &Fitness, gen_seed: u64) -> Result<(f32, Option<f32>, usize)> {
        (self.evaluate_fn)(net, gen_seed)
    }
}