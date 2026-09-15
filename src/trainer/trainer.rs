//! The Trainer trait and associated step execution contexts.
//! Sourced from the engine's data-agnostic training contract.

use crate::graph::network::Network;
use crate::trainer::stream::BatchStream;
use crate::utils::tabular_data::Dataset;

/// The evaluation/metrics report returned by `Trainer::train_step` or
/// `Trainer::eval_step` at the end of each clock-step.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StepReport {
    /// The training loss achieved on the train batch (for logging).
    pub train_loss: f32,
    /// Optional validation loss evaluated on the shared eval batch.
    pub eval_loss: Option<f32>,
    /// The ranking fitness score. Higher is better if `Direction::Maximize`,
    /// lower is better if `Direction::Minimize`.
    pub fitness: f32,
    /// Extra non-ranking/informative scores configured for the run, if any.
    pub informative: Vec<f32>,
}

/// Rich read-only context handed to the trainer on every step.
///
/// Contains the shared batch stream (already seeded and offset to this step),
/// the current step's epoch parameters, and identity metrics.
pub struct StepContext<'a> {
    /// The dataset and batch stream, if available.
    pub data: Option<&'a RunData<'a>>,
    /// The ranking fitness metric (gives the direction and label).
    pub fitness: Option<&'a crate::engine::fitness::Fitness>,
    /// Informative non-ranking metrics configured for the run.
    pub metrics: &'a [crate::engine::fitness::Metric],
    /// Lightweight snapshot of current run/epoch metadata.
    pub env: StepEnv,
    /// Canonical hash of the net being trained (for logs or tracking).
    pub net_hash: &'a str,
    /// Deterministic net-specific weight initialization/dropout seed.
    pub net_seed: u64,
}

/// Lightweight engine-metadata snapshot handed to the trainer each step via
/// `StepContext::env`. Read-only — the trainer inspects, the engine owns.
#[derive(Clone, Copy, Debug)]
pub struct StepEnv {
    /// The current step clock (same as the `step` argument of `train_step`).
    pub step: usize,
    /// The run's seed (as resolved at construction, recorded in engine.json).
    pub run_seed: u64,
    /// Configured population size (the run's target, not the live count).
    pub pop_size: usize,
    /// Live nets in the population right now.
    pub live_count: usize,
    /// Checkpoint cadence (crossover-gate bar is recorded every N steps).
    pub checkpoint_every: usize,
    /// The per-net fitness smoothing window, in steps (K).
    pub smoothing_window: usize,
}

/// The dataset + shared batch stream, bundled. The engine constructs one for
/// the run; a trainer that uses the shared stream accesses both through it.
pub struct RunData<'a> {
    /// The run's dataset (already on the right device/dtype).
    pub dataset: &'a Dataset,
    /// The shared deterministic batch stream.
    pub stream: &'a BatchStream,
}

impl<'a> RunData<'a> {
    /// Draw a training batch for the given step.
    pub fn train_batch(&self, step: u64) -> flodl::tensor::Result<(flodl::Tensor, flodl::Tensor)> {
        self.stream.train_batch(self.dataset, step)
    }

    /// Draw an evaluation batch for the given step.
    pub fn eval_batch(&self, step: u64) -> flodl::tensor::Result<(flodl::Tensor, flodl::Tensor)> {
        self.stream.eval_batch(self.dataset, step)
    }
}

/// The batch shape requested by the trainer (defaults to batch_size).
#[derive(Clone, Copy, Debug)]
pub struct StreamShape {
    pub batch_size: usize,
    pub eval_batch_size: usize,
}

impl StreamShape {
    /// No override — use the engine's default stream shape.
    pub fn none() -> Self {
        Self {
            batch_size: 16,
            eval_batch_size: 16,
        }
    }

    /// Convenience for the common case: same size for train + eval.
    pub fn uniform(n: usize) -> Self {
        Self {
            batch_size: n,
            eval_batch_size: n,
        }
    }
}

/// The core training contract.
///
/// Decouples the backpropagation updates, metrics scoring, and loss functions
/// from the evolutionary logic of `RaceEngine`.
pub trait Trainer: Send {
    /// Supply the loss function to use when training.
    fn loss(&self) -> Option<crate::trainer::LossFn<'_>> {
        None
    }

    /// Optional LR schedule: the learning rate to use at `step`, or `None`
    /// for the optimizer's own (fixed) LR. Applied by the trainer itself at
    /// the top of `train_step` via `optimizer.set_lr(lr)`.
    ///
    /// **Replay contract:** the schedule MUST be a pure function of `step`.
    /// Catch-up children replay past steps through `train_step`, so a pure
    /// schedule hands them the exact LR the population saw. A stateful
    /// schedule (e.g. metrics-driven plateau decay) would advance its state
    /// on every replay and silently break determinism.
    fn scheduled_lr(&self, _step: usize) -> Option<f64> {
        None
    }

    /// The optional batch-shape override requested by the trainer.
    fn stream_shape(&self) -> Option<StreamShape> {
        None
    }

    /// Optional self-description of this scheme's hyperparameters, recorded in
    /// `engine.json` under `"trainer"`.
    ///
    /// Return a JSON object so a run stays reproducible without reading the
    /// caller's source — e.g.
    /// `{"trainer":"tabular","learning_rate":0.001,"grad_clip":1.0}`.
    /// The engine never interprets the contents; it only persists them.
    /// `None` (the default) means "no description available".
    fn describe(&self) -> Option<serde_json::Value> {
        None
    }

    /// Make the optimizer for a newly built or reloaded network.
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer>;

    /// Train the network for one clock-step in place.
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &StepContext<'_>,
    ) -> flodl::tensor::Result<StepReport>;
}

impl Trainer for Box<dyn Trainer> {
    fn loss(&self) -> Option<crate::trainer::LossFn<'_>> {
        self.as_ref().loss()
    }
    fn scheduled_lr(&self, step: usize) -> Option<f64> {
        self.as_ref().scheduled_lr(step)
    }
    fn stream_shape(&self) -> Option<StreamShape> {
        self.as_ref().stream_shape()
    }
    fn describe(&self) -> Option<serde_json::Value> {
        self.as_ref().describe()
    }
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.as_ref().make_optimizer(net)
    }
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &StepContext<'_>,
    ) -> flodl::tensor::Result<StepReport> {
        self.as_mut().train_step(net, optimizer, step, ctx)
    }
}

/// Helper for trait object boxing.
pub trait IntoBoxedTrainer {
    fn into_boxed(self) -> Box<dyn Trainer>;
}

impl<T: Trainer + 'static> IntoBoxedTrainer for T {
    fn into_boxed(self) -> Box<dyn Trainer> {
        Box::new(self)
    }
}
