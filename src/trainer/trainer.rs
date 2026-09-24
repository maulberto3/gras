//! The trainer contracts, split per run mode.
//!
//! The engine runs either **Tabular** (dataset-driven, supervised) or **RL**
//! (environment-driven, trainer-reported fitness) — see
//! [`crate::engine::RunSpec`]. The trainer surface is split the same way:
//!
//! - [`StepTrainer`] — the mode-agnostic core every scheme implements
//!   (optimizer creation, self-description).
//! - [`TabularStep`] — the dataset-driven contract: a **required**
//!   `(pred, target)` loss, access to the shared batch stream, per-step eval.
//! - [`RlStep`] — the environment-driven contract: **no loss method at all**
//!   (the training signal lives inside `train_step`), no data in the context,
//!   the fitness is whatever the trainer reports in `StepReport.fitness`.
//!
//! The split makes wrong flavor combinations a **compile error**: an RL
//! scheme cannot return a `(pred, target)` loss, and a tabular scheme cannot
//! be built without one. `RaceEngine::new` enforces the mode pairing via
//! trait bounds (`RunSpec::tabular` demands `T: TabularStep`,
//! `RunSpec::RL` demands `T: RlStep`).

use crate::graph::network::Network;
use crate::utils::tabular_data::Dataset;

/// The evaluation/metrics report returned by a trainer's `train_step` at the
/// end of each clock-step.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StepReport {
    /// The training loss achieved on the train batch (for logging).
    pub train_loss: f32,
    /// Optional validation loss evaluated on the shared eval batch.
    /// (RL schemes always report `None` — there is no held-out batch.)
    pub eval_loss: Option<f32>,
    /// The ranking fitness score. Higher is better if `Direction::Maximize`,
    /// lower is better if `Direction::Minimize`. Tabular: computed by the
    /// engine's fitness fn on the eval batch. RL: **reported** — the scalar
    /// the trainer derived from its environment.
    pub fitness: f32,
    /// Extra non-ranking/informative scores configured for the run, if any.
    pub informative: Vec<f32>,
    /// RL-only environment volume for this step — see [`RlStepMeta`]. `None`
    /// in Tabular (no environment) and from RL trainers that don't report it
    /// (the log then prints `—` in the RL columns).
    #[serde(default)]
    pub rl: Option<RlStepMeta>,
}

/// How much environment this step actually played — the RL counterpart of
/// tabular's eval-loss column, which has nothing to report in a mode with no
/// held-out batch.
///
/// Vocabulary (used by the engine log and every RL example): **match** = one
/// episode (one full game, one bandit pull), **turn** = one environment step
/// inside a match. CartPole calls them episodes/timesteps; the engine log says
/// match/turn throughout.
///
/// The engine sums these over the live population per clock-step and prints
/// `matches <total> │ turns <total> │ turns/match <mean>` in the per-step
/// rollup — so a long RL step shows *why* it took long, and a changing match
/// length (e.g. `MatchLength::RandomNumTurns`) is visible as it happens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RlStepMeta {
    /// Matches (episodes) this net's trainer played this step.
    pub matches: usize,
    /// Environment turns played across those matches.
    pub turns: usize,
}

/// Lightweight engine-metadata snapshot handed to the trainer each step.
/// Read-only — the trainer inspects, the engine owns.
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

/// The dataset + shared batch stream, bundled. Only ever present in a
/// [`TabularContext`] — RL trainers never see data.
pub struct RunData<'a> {
    /// The run's dataset (already on the right device/dtype).
    pub dataset: &'a Dataset,
    /// The shared deterministic batch stream.
    pub stream: &'a crate::trainer::stream::BatchStream,
}

impl RunData<'_> {
    /// Draw a training batch for the given step.
    pub fn train_batch(&self, step: u64) -> flodl::tensor::Result<(flodl::Tensor, flodl::Tensor)> {
        self.stream.train_batch(self.dataset, step)
    }

    /// Draw an evaluation batch for the given step.
    pub fn eval_batch(&self, step: u64) -> flodl::tensor::Result<(flodl::Tensor, flodl::Tensor)> {
        self.stream.eval_batch(self.dataset, step)
    }
}

/// The context handed to a [`TabularStep`] on every step. Data is **not**
/// optional — a tabular run always has a dataset and stream, by construction.
pub struct TabularContext<'a> {
    /// The run's dataset and shared batch stream.
    pub data: RunData<'a>,
    /// The ranking fitness metric (gives the direction and label).
    pub fitness: &'a crate::engine::fitness::Fitness,
    /// Informative non-ranking metrics configured for the run.
    pub metrics: &'a [crate::engine::fitness::Metric],
    /// Lightweight snapshot of current run/epoch metadata.
    pub env: StepEnv,
    /// Canonical hash of the net being trained (for logs or tracking).
    pub net_hash: &'a str,
    /// Deterministic net-specific weight initialization/dropout seed.
    pub net_seed: u64,
}

/// The context handed to an [`RlStep`] on every step. There is **no** data
/// field: the trainer owns its environment. The fitness direction/label is
/// still exposed (for logging the reported score in the trainer's own style).
pub struct RlContext<'a> {
    /// The ranking fitness metric (direction + label; the VALUE is the
    /// trainer's to produce in `StepReport.fitness`).
    pub fitness: &'a crate::engine::fitness::Fitness,
    /// Informative non-ranking metrics configured for the run.
    pub metrics: &'a [crate::engine::fitness::Metric],
    /// Lightweight snapshot of current run/epoch metadata.
    pub env: StepEnv,
    /// Canonical hash of the net being trained (for logs or tracking).
    pub net_hash: &'a str,
    /// Deterministic net-specific weight initialization/dropout seed.
    pub net_seed: u64,
}

/// The batch shape requested by a tabular trainer (defaults to batch_size).
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

// ── The traits ───────────────────────────────────────────────────────────────

/// The mode-agnostic core every trainer implements: build an optimizer for a
/// net and describe the recipe. The mode-specific contracts
/// ([`TabularStep`], [`RlStep`]) require this as a supertrait.
pub trait StepTrainer: Send {
    /// Make the optimizer for a newly built or reloaded network.
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer>;

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
}

/// The dataset-driven (supervised) training contract.
///
/// Owns the loss — it is REQUIRED here, not an `Option` — and optionally
/// shapes the shared batch stream. Data arrives via [`TabularContext::data`],
/// always present.
pub trait TabularStep: StepTrainer {
    /// The loss this scheme trains against: `(pred, target) -> loss`.
    fn loss(&self) -> crate::trainer::LossFn<'_>;

    /// Optional LR schedule: the learning rate to use at `step`, or `None`
    /// for the optimizer's own (fixed) LR. Applied by the trainer itself at
    /// the top of `train_step` via `optimizer.set_lr(lr)`.
    ///
    /// **Replay contract:** the schedule MUST be a pure function of `step`.
    /// Catch-up children replay past steps through `train_step`, so a pure
    /// schedule hands them the exact LR the population saw. A stateful
    /// schedule would advance its state on every replay and silently break
    /// determinism.
    fn scheduled_lr(&self, _step: usize) -> Option<f64> {
        None
    }

    /// The optional batch-shape override requested from the engine stream.
    fn stream_shape(&self) -> Option<StreamShape> {
        None
    }

    /// Train the network for one clock-step in place.
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &TabularContext<'_>,
    ) -> flodl::tensor::Result<StepReport>;
}

/// The environment-driven (RL) training contract.
///
/// There is deliberately NO `loss()` method and no data in [`RlContext`]:
/// the training signal (rewards, trajectories, advantages) is the trainer's
/// internal business, produced inside `train_step`. The engine's ONLY ranking
/// input is the fitness value the trainer puts in
/// [`StepReport::fitness`] each step — which is why RL runs are built with
/// [`crate::engine::fitness::Fitness::reported`].
///
/// See `examples/cartpole.rs` for the canonical implementation.
pub trait RlStep: StepTrainer {
    /// Train the network for one clock-step in place: run the environment,
    /// apply the update, and report the reward-derived fitness.
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> flodl::tensor::Result<StepReport>;

    /// Pop-wide phase: called ONCE per step with ALL live nets (mutable,
    /// read order = live order) BEFORE any per-net `train_step`. Default:
    /// no-op (the historical per-net-isolated behavior).
    ///
    /// This is the engine-side vantage point for loop-breaking schemes that
    /// need the whole population: pop-mean action anchors, distillation,
    /// shared-referee matches. The trainer may READ every net here and
    /// cache whatever its `train_step` needs (e.g. the pop's mean action
    /// distribution per observation); per-net learning stays in
    /// `train_step`. Implementations must not rely on the order or on
    /// Optimizer state — only on the forward pass.
    fn pop_phase(&mut self, _nets: &mut [(String, &mut Network)], _step: usize) {}
}

// ── Trait-object plumbing ────────────────────────────────────────────────────

/// The engine's boxed trainer, tagged by mode. Built from the matching
/// [`RunSpec`](crate::engine::RunSpec) variant; each step dispatches on the
/// arm so a Tabular step always carries data and an RL step never does.
pub enum ModeTrainer {
    /// Dataset-driven scheme (from `RunSpec::tabular`).
    Tabular(Box<dyn TabularStep>),
    /// Environment-driven scheme (from `RunSpec::RL`).
    Rl(Box<dyn RlStep>),
}

impl ModeTrainer {
    /// Which mode this trainer serves.
    pub fn is_tabular(&self) -> bool {
        matches!(self, ModeTrainer::Tabular(_))
    }

    /// The Tabular arm, if this is one. The engine's tabular-only paths
    /// (stream shape resolution, the loss-based checkpoint exam) use this.
    pub fn as_tabular(&self) -> Option<&dyn TabularStep> {
        match self {
            ModeTrainer::Tabular(t) => Some(t.as_ref()),
            ModeTrainer::Rl(_) => None,
        }
    }

    /// Batch-shape request from a Tabular trainer; RL has no stream, so it
    /// is always `None` there.
    pub fn stream_shape(&self) -> Option<StreamShape> {
        self.as_tabular().and_then(|t| t.stream_shape())
    }

    pub fn describe(&self) -> Option<serde_json::Value> {
        match self {
            ModeTrainer::Tabular(t) => t.describe(),
            ModeTrainer::Rl(t) => t.describe(),
        }
    }

    /// Pop-wide phase dispatch (RL only): see [`RlStep::pop_phase`]. Tabular
    /// has no such phase — its group step IS shared by construction (the
    /// shared batch stream).
    pub(crate) fn pop_phase(&mut self, nets: &mut [(String, &mut Network)], step: usize) {
        if let ModeTrainer::Rl(t) = self {
            t.pop_phase(nets, step);
        }
    }

    pub fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        match self {
            ModeTrainer::Tabular(t) => t.make_optimizer(net),
            ModeTrainer::Rl(t) => t.make_optimizer(net),
        }
    }

    /// Run one training step with an optional data handle. The mode decides
    /// which context type is constructed: Tabular requires `data` to be
    /// `Some` (the engine guarantees this — it is a construction invariant,
    /// not a per-step check); RL ignores it entirely.
    pub(crate) fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        data: Option<&RunData<'_>>,
        fitness: &crate::engine::fitness::Fitness,
        metrics: &[crate::engine::fitness::Metric],
        env: StepEnv,
        net_hash: &str,
        net_seed: u64,
    ) -> flodl::tensor::Result<StepReport> {
        match self {
            ModeTrainer::Tabular(t) => {
                let data = data.expect(
                    "ModeTrainer::Tabular step without data — engine construction invariant violated",
                );
                let ctx = TabularContext {
                    data: RunData {
                        dataset: data.dataset,
                        stream: data.stream,
                    },
                    fitness,
                    metrics,
                    env,
                    net_hash,
                    net_seed,
                };
                t.train_step(net, optimizer, step, &ctx)
            }
            ModeTrainer::Rl(t) => {
                let ctx = RlContext {
                    fitness,
                    metrics,
                    env,
                    net_hash,
                    net_seed,
                };
                t.train_step(net, optimizer, step, &ctx)
            }
        }
    }
}

impl StepTrainer for Box<dyn StepTrainer> {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.as_ref().make_optimizer(net)
    }
    fn describe(&self) -> Option<serde_json::Value> {
        self.as_ref().describe()
    }
}

// Boxed trait objects satisfy the mode traits too, so the default spec
// generics (`Box<dyn TabularStep>` / `Box<dyn RlStep>`) resolve.
impl TabularStep for Box<dyn TabularStep> {
    fn loss(&self) -> crate::trainer::LossFn<'_> {
        self.as_ref().loss()
    }
    fn scheduled_lr(&self, step: usize) -> Option<f64> {
        self.as_ref().scheduled_lr(step)
    }
    fn stream_shape(&self) -> Option<StreamShape> {
        self.as_ref().stream_shape()
    }
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &TabularContext<'_>,
    ) -> flodl::tensor::Result<StepReport> {
        self.as_mut().train_step(net, optimizer, step, ctx)
    }
}

impl RlStep for Box<dyn RlStep> {
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> flodl::tensor::Result<StepReport> {
        self.as_mut().train_step(net, optimizer, step, ctx)
    }
    fn pop_phase(&mut self, nets: &mut [(String, &mut Network)], step: usize) {
        self.as_mut().pop_phase(nets, step)
    }
}

impl StepTrainer for Box<dyn TabularStep> {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.as_ref().make_optimizer(net)
    }
    fn describe(&self) -> Option<serde_json::Value> {
        self.as_ref().describe()
    }
}

impl StepTrainer for Box<dyn RlStep> {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.as_ref().make_optimizer(net)
    }
    fn describe(&self) -> Option<serde_json::Value> {
        self.as_ref().describe()
    }
}

/// Helper for trait object boxing.
pub trait IntoBoxedTrainer {
    fn into_boxed(self) -> Box<dyn StepTrainer>;
}

impl<T: StepTrainer + 'static> IntoBoxedTrainer for T {
    fn into_boxed(self) -> Box<dyn StepTrainer> {
        Box::new(self)
    }
}
