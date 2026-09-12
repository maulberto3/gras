//! Training schemes for the step-race engine.
//!
//! The engine is **training-agnostic**: it orchestrates the population
//! lifecycle (step clock, divergence, cull/insert, checkpoint gates, stop
//! criteria) and delegates *how one net trains on one step* to a
//! caller-supplied [`Trainer`]. The user brings their own training recipe;
//! the only contract is [`Trainer::train_step`] returning a [`StepReport`].
//!
//! - [`stream`] — the shared deterministic `BatchStream` + `PoolSplit`.
//! - [`supervised`] — the reference implementation (`TabularTrainer`),
//!   extracted verbatim from the engine so behavior is unchanged.

pub mod stream;
pub mod supervised;

pub use supervised::TabularTrainer;

use flodl::nn::optim::Optimizer;
use flodl::tensor::Result;
use flodl::Variable;

/// The loss-function signature, aliased for readability in the trait and
/// context.
pub type LossFn<'a> = &'a (dyn Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync);

use crate::engine::fitness::{Fitness, Metric};
use crate::graph::network::Network;
use crate::utils::data::Dataset;
use stream::BatchStream;

/// What one training episode reports back to the engine. Same shape the
/// engine computed inline before the extraction — a drop-in contract.
#[derive(Clone, Debug)]
pub struct StepReport {
    /// Training loss for this step (batch the trainer chose to train on).
    pub train_loss: f32,
    /// Held-out eval loss, if the scheme produced one (`None` = no eval this
    /// step; the engine stores it as-is and the rollups skip missing evals).
    pub eval_loss: Option<f32>,
    /// Ranking fitness for this step (drives cull/insert selection).
    pub fitness: f32,
    /// Informative (non-ranking) metric values, same order as the run's
    /// `Vec<Metric>`.
    pub informative: Vec<f32>,
}

/// Everything the engine offers a training scheme for one step. Borrowed —
/// the engine keeps ownership; the trainer just reads.
///
/// # Offered, not imposed
/// The scoring and data handles are `Option`s because they are the *engine's*
/// opinions about how to train, not requirements. A supervised scheme uses
/// them all; an RL scheme has no `loss_fn` and supplies its own reward logic;
/// a scheme with its own dataloader ignores the shared stream. What is NOT
/// optional is identity: `step`, `net_hash`, `net_seed` — the engine always
/// provides them and they are what catch-up replay and resume parity hang
/// off.
pub struct StepContext<'a> {
    // ── engine-provided handles (use what fits your paradigm) ──
    /// The run's fitness function (direction-aware ranking score), if defined.
    /// (The loss is NOT here: it's the trainer's own business — schemes that
    /// have one keep it internally, e.g. `TabularTrainer::new(loss_fn)`.)
    pub fitness: Option<&'a Fitness>,
    /// Informative metric labels, in report order (empty when none defined).
    pub metrics: &'a [Metric],
    /// The shared deterministic batch stream + dataset (train + eval pools,
    /// pure function of `(run_seed, step)`). A scheme with its own data
    /// source ignores this; the reference scheme draws
    /// `train_batch(step)`/`eval_batch(step)` from it for comparability.
    pub data: Option<&'a RunData<'a>>,
    /// Snapshot of the engine's run state at this step — for schemes that
    /// want to know engine metadata (what step, pop size, run seed, best
    /// held-out eval so far) and branch their recipe on it. Borrowed view,
    /// never owned.
    pub env: StepEnv,
    // ── identity (always present — the replay contract hangs off these) ──
    /// Hash of the net being trained (lets schemes derive per-net streams).
    pub net_hash: &'a str,
    /// The net's seed (per-net determinism input).
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
    /// Steps between divergence/cull checks.
    pub divergence_window: usize,
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
    /// This step's shared train batch — pure function of `(run_seed, step)`.
    pub fn train_batch(&self, step: u64) -> Result<(flodl::Tensor, flodl::Tensor)> {
        self.stream.train_batch(self.dataset, step)
    }

    /// This step's shared eval batch — pure function of `(run_seed, step)`.
    pub fn eval_batch(&self, step: u64) -> Result<(flodl::Tensor, flodl::Tensor)> {
        self.stream.eval_batch(self.dataset, step)
    }
}

/// Shape of the shared batch stream a scheme wants, when it uses the
/// engine's stream. Returned from [`Trainer::stream_shape`] to override the
/// run defaults at construction time — deterministic, fixed for the run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamShape {
    /// Rows per shared train batch.
    pub batch_size: usize,
    /// Rows per shared eval batch.
    pub eval_batch_size: usize,
}

/// The training contract. The engine hands over the net (mutated in place —
/// coefficients must evolve across steps, never rebuilt) plus a
/// [`StepContext`]; the scheme trains and reports what happened.
///
/// # The per-step clause
/// `train_step` is **one step-clock of training for this net**, and the
/// returned [`StepReport`] must describe **the net's state at the end of
/// this step**: `fitness`/`eval_loss` scored after this step's updates, on
/// this step's data. The engine records the report under the step clock,
/// feeds it to cull/insert ranking, and replays it during catch-up — a
/// report about any other state silently corrupts the race. In debug builds
/// the engine *probes* the contract: it re-scores the net on the step's eval
/// batch and panics if the reported `eval_loss` doesn't match (cheap during
/// development; zero cost in release). Freedom beyond that is real: a scheme
/// may run any number of optimizer steps inside `train_step`, choose its own
/// batches, or skip eval (return `eval_loss: None`) — image, RL, and NLP
/// schemes subclass this same trait; nothing about the recipe is fixed.
///
/// Determinism is the implementor's responsibility: the reference seeds via
/// `crate::utils::race_steps::seed_step_randomness` so catch-up replay parity
/// holds; custom schemes that break seeding will fail the parity assertions
/// on resume.
pub trait Trainer: Send {
    /// The loss function this scheme trains against, if its paradigm has
    /// one. The engine offers it back through `StepContext::loss_fn` and uses
    /// it for its own bookkeeping evals (best-net snapshot). Supervised
    /// schemes return `Some`; RL schemes return `None` (rewards are the
    /// scheme's internal business).
    fn loss(&self) -> Option<LossFn<'_>> {
        None
    }

    /// Shape of the shared batch stream this scheme wants, if it uses the
    /// engine's stream. `None` (default) = run defaults from `RunSpec::stream`.
    /// The engine applies this once at construction — deterministic and
    /// fixed for the run. The train/eval **split ratio is NOT overridable**:
    /// which rows are held out is engine bookkeeping that protects fitness
    /// comparability across every net, whatever the recipe.
    fn stream_shape(&self) -> Option<StreamShape> {
        None
    }

    /// Build the optimizer a net trains with. Called by the engine when a net
    /// enters the population (initial seed, crossover child, immigrant) and on
    /// resume rebuilds — the engine owns *when* optimizers exist, the scheme
    /// owns *how they're configured* (Adam vs SGD, learning rate, weight
    /// decay, …). Fresh state per call — children start their optimizer at
    /// step 0 of their catch-up, matching the group's step-0 state.
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer>;

    /// Train `net` for one step-clock and report the step's metrics.
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &StepContext<'_>,
    ) -> Result<StepReport>;
}

/// Ergonomic constructor for a boxed [`Trainer`]: anything trainable
/// converts into a boxed trainer, so call sites can write
/// `trainer: TabularTrainer::new(loss).with_learning_rate(1e-3)` in a
/// `RunSpec` field without `Box::new(...)`.
///
/// (No separate maker type — the concrete trainer structs *are* the nice
/// API; `RunSpec` accepts anything convertible into `Box<dyn Trainer>`.)
pub trait IntoBoxedTrainer {
    fn into_boxed(self) -> Box<dyn Trainer>;
}

impl<T: Trainer + 'static> IntoBoxedTrainer for T {
    fn into_boxed(self) -> Box<dyn Trainer> {
        Box::new(self)
    }
}impl IntoBoxedTrainer for Box<dyn Trainer> {
    fn into_boxed(self) -> Box<dyn Trainer> {
        self
    }
}
