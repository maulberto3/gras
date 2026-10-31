//! The trainer contracts — ONE trait per run mode.
//!
//! The engine runs either **Tabular** (dataset-driven, supervised) or **RL**
//! (environment-driven, trainer-reported fitness) — see
//! [`crate::engine::RunSpec`]. A scheme implements exactly ONE of:
//!
//! - [`TabularStep`] — the dataset-driven contract: a **required**
//!   `(pred, target)` loss, access to the shared batch stream, per-step eval.
//! - [`RlStep`] — the environment-driven contract: **no loss method at all**
//!   (the training signal lives inside `train_step`), no data in the context,
//!   the fitness is whatever the trainer reports in [`RlStepReport::fitness`].
//!
//! Each trait is self-sufficient: it carries the shared lifecycle
//! (`make_optimizer`, optional `describe`, optional `restore_optimizer`) next to
//! its mode-specific step, so a trainer is ONE `impl` block. Wrong flavor
//! combinations stay a **compile error**: `RunSpec::tabular` demands
//! `T: TabularStep` and `RunSpec::rl` demands `T: RlStep`, and the two traits
//! expose disjoint step methods.
//!
//! Reports are mode-split too: [`TabularStep`] returns [`TabularStepReport`],
//! [`RlStep`] returns [`RlStepReport`] — no always-`None` fields. Both
//! normalize into the engine-internal [`StepReport`] union (the persisted
//! `NetMetrics` stays one type: wire format, history.csv, replay parity).
//!
//! The engine stores either mode behind one [`AnyTrainer`] enum. There is no
//! adapter trait and no hand-written `Box<dyn _>` forwarding impls: an
//! exhaustive `match` cannot silently fall back to a trait default.

use crate::graph::network::Network;
use crate::utils::tabular_data::Dataset;

/// The evaluation/metrics report returned by a trainer's `train_step` at the end
/// of each step. `Default` derived for `StepReport { fitness, ..Default::default() }`.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct StepReport {
    /// The training loss achieved on the train batch (for logging).
    pub train_loss: f32,
    /// Optional validation loss evaluated on the shared eval batch.
    /// (RL schemes always report `None` — there is no held-out batch.)
    pub eval_loss: Option<f32>,
        /// The ranking fitness. Tabular: computed by the engine's fitness fn; RL:
        /// **reported** — the scalar the trainer derived from its environment.
    pub fitness: f32,
    /// Extra non-ranking/informative scores configured for the run, if any.
    pub informative: Vec<f32>,
        /// RL-only environment volume for this step — see [`RlStepMeta`]. `None` in
        /// Tabular (the log prints `—`).
    #[serde(default)]
    pub rl: Option<RlStepMeta>,
}

/// The report a [`TabularStep`] returns: no RL volume field (a dataset trainer
/// has no environment). `Default` derived for `..Default::default()`.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TabularStepReport {
    /// The training loss achieved on this step's train batch (logging only;
    /// ranking reads `fitness`).
    pub train_loss: f32,
    /// Held-out loss on this step's shared eval batch. `None` = not measured.
    pub eval_loss: Option<f32>,
    /// The ranking fitness — computed by the run's fitness scorer on the
    /// eval batch (a [`TabularStep`] reports the value, the run owns the fn).
    pub fitness: f32,
    /// Extra non-ranking/informative scores configured for the run, if any.
    pub informative: Vec<f32>,
}

/// The report an [`RlStep`] returns: no `eval_loss` (evaluation happens inside
/// `train_step`, and its result IS `fitness`).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RlStepReport {
    /// The training objective your update produced (REINFORCE loss, TD error,
    /// …). Logging only — the engine never ranks on it.
    pub train_loss: f32,
    /// The ranking scalar derived from your environment. MANDATORY meaning:
    /// this is the ONLY ranking input the engine gets from an RL trainer.
    pub fitness: f32,
    /// Extra non-ranking/informative scores configured for the run, if any.
    pub informative: Vec<f32>,
    /// Environment volume for this step — feeds the `matches`/`train`/`eval`
    /// rollup columns. `None` = not reported (the log prints `—`).
    pub rl: Option<RlStepMeta>,
}

    // ── Report normalization: the two mode reports collapse into the engine-internal
    // union `NetMetrics` consumes — the only place the two types meet. ──

impl From<TabularStepReport> for StepReport {
    fn from(r: TabularStepReport) -> Self {
        StepReport {
            train_loss: r.train_loss,
            eval_loss: r.eval_loss,
            fitness: r.fitness,
            informative: r.informative,
            rl: None, // tabular: no environment, so no volume to report
        }
    }
}

impl From<RlStepReport> for StepReport {
    fn from(r: RlStepReport) -> Self {
        StepReport {
            train_loss: r.train_loss,
            eval_loss: None, // RL: no (pred, target) held-out loss exists
            fitness: r.fitness,
            informative: r.informative,
            rl: r.rl,
        }
    }
}

/// How much environment this step played — the RL counterpart of tabular's
/// eval-loss column. **match** = one episode, **turn** = one env step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RlStepMeta {
    /// Matches (episodes) this net's trainer played this step (train + eval).
    pub matches: usize,
    /// Turns played in this step's TRAIN matches.
    #[serde(default)]
    pub train_turns: usize,
    /// Turns played in this step's EVAL matches — evaluation must keep
    /// measuring the net's own policy.
    #[serde(default)]
    pub eval_turns: usize,
}

impl RlStepMeta {
        /// Total environment turns this step (train + eval) — the rollup's
        /// `turns/match` denominator and `ms/turn` cost cell.
    pub fn turns(&self) -> usize {
        self.train_turns + self.eval_turns
    }
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
    /// Checkpoint cadence — the durability tick (frontier + elite weight
    /// snapshots) and the stream's eval-permutation era fire every N steps.
    /// **`0` = the tick is off** and the eval permutation never rotates (one eternal
    /// era); a trainer must not `% 0` it.
    pub checkpoint_every: usize,
    /// The per-net fitness smoothing window, in steps (K).
    pub smoothing_window: usize,
        /// The run's step budget (`None` = unbounded), carried so a trainer can anneal
        /// schedules without a new config knob or a private counter.
    pub max_steps: Option<usize>,
}

impl StepEnv {
        /// Fraction of the step budget consumed (`0.0` → `1.0`), or `None` when
        /// unbounded — pure arithmetic on the clock, so replays match.
    pub fn progress(&self) -> Option<f32> {
        crate::utils::training::run_progress(self.step, self.max_steps)
    }
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
    pub fitness: &'a crate::utils::fitness::Fitness,
    /// Informative non-ranking metrics configured for the run.
    pub metrics: &'a [crate::utils::fitness::Metric],
    /// Lightweight snapshot of current run/epoch metadata.
    pub env: StepEnv,
    /// Canonical hash of the net being trained (for logs or tracking).
    pub net_hash: &'a str,
    /// Deterministic net-specific weight initialization/dropout seed.
    pub net_seed: u64,
}

/// The context handed to an [`RlStep`] each step. No data field: the trainer owns
/// its environment. Fitness direction/label is exposed for logging.
pub struct RlContext<'a> {
    /// The ranking fitness metric (direction + label; the VALUE is the
    /// trainer's to produce in `RlStepReport.fitness`).
    pub fitness: &'a crate::utils::fitness::Fitness,
    /// Informative non-ranking metrics configured for the run.
    pub metrics: &'a [crate::utils::fitness::Metric],
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
    /// Train batch size this trainer requests from the shared stream.
    pub batch_size: usize,
    /// Eval batch size this trainer requests from the shared stream.
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

/// The dataset-driven training contract — the ONE trait a tabular scheme
/// implements. Owns the loss (required, not `Option`) and shapes the stream.
///
/// # Randomness — the replay contract
///
/// The engine re-seeds BOTH generators (`fastrand` + libtorch) once per
/// `(net, step)` before calling [`TabularStep::train_step`]:
/// [`seed_step_randomness`](crate::utils::race_steps::seed_step_randomness)
/// mixes `ctx.net_seed` with `ctx.env.step`. A trainer that draws its own
/// randomness (dropout, sampling, augmentation) must derive it from those same
/// inputs — the `(net_seed, step)` pair is the whole address of a step — and
/// never from a hidden global counter, or the run stops being replayable. The
/// same rule as [`TabularStep::scheduled_lr`].
pub trait TabularStep: Send {
    /// Make the optimizer for a newly built or reloaded network.
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer>;

    /// Restore this net's optimizer state from `path`, or `None` to build a fresh one.
    ///
    /// The hook exists because flodl's `Stateful::load_state_file` needs the CONCRETE
    /// optimizer type while the engine holds `Box<dyn Optimizer>` — only the trainer
    /// knows which optimizer it makes. Default `None`: the engine calls `make_optimizer`
    /// and logs that the moments restarted (option A).
    fn restore_optimizer(
        &self,
        _net: &Network,
        _path: &std::path::Path,
    ) -> Option<Box<dyn flodl::nn::optim::Optimizer>> {
        None
    }

        /// Per-net bookkeeping to persist with this net's checkpoint (R9).
        ///
        /// The engine calls this once per LIVE net at every checkpoint and stores the
        /// result in that net's `nets/<hash>.json` envelope (field `trainer_state`), so it
        /// travels with the net instead of guessing which state belongs to which hash.
        /// Return whatever the scheme must remember about `net_hash` that the WEIGHTS do
        /// not hold — the example is `last_accuracy` in `examples/mnist.rs`, which decides
        /// whether the model is still improving. Default `None`: no state, no field, and
        /// the file stays byte-identical to a pre-R9 run.
        ///
        /// Keep it small and JSON-only: it is rewritten at every tick and read back by
        /// [`Self::restore`] after a resume, possibly in a different build.
        fn state(&self, _net_hash: &str) -> Option<serde_json::Value> {
            None
        }

        /// Hand back what [`Self::state`] recorded for this net, once, when a run resumes.
        ///
        /// Called AFTER the net's weights (and optimizer, when saved) are loaded and BEFORE
        /// the next step, so a decision at the resumed clock sees what the uninterrupted run
        /// would have seen. The default does nothing; a scheme that returns `Some` from
        /// `state` and ignores the value here diverges from its twin — the two hooks are a
        /// pair.
        fn restore(&mut self, _net_hash: &str, _state: &serde_json::Value) {}

        /// Optional self-description of this scheme's hyperparameters, recorded in
        /// `engine.json` under `"trainer"`. The engine never interprets the contents.
    fn describe(&self) -> Option<serde_json::Value> {
        None
    }

    /// The loss this scheme trains against: `(pred, target) -> loss`.
    fn loss(&self) -> crate::trainer::LossFn<'_>;

        /// Optional LR schedule for `step` (`None` = the optimizer's fixed LR), applied
        /// by the trainer. MUST be a pure function of `step` (replay contract).
        ///
        /// `max_steps` is the run's budget, so a progress-shaped schedule
        /// (`cosine`, `onecycle`) can be a pure function of the clock instead of
        /// keeping a private horizon — `None` means an unbounded run.
    fn scheduled_lr(&self, _step: usize, _max_steps: Option<usize>) -> Option<f64> {
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
    ) -> flodl::tensor::Result<TabularStepReport>;
}

/// The environment-driven training contract — the ONE trait an RL scheme
/// implements. No `loss()`; the engine's only ranking input is `fitness`.
///
/// # Randomness — the replay contract
///
/// The engine re-seeds BOTH generators (`fastrand` + libtorch) once per
/// `(net, step)` before calling [`RlStep::train_step`]:
/// [`seed_step_randomness`](crate::utils::race_steps::seed_step_randomness)
/// mixes `ctx.net_seed` with `ctx.env.step`. An environment that randomizes
/// (initial state, resets, exploration) must derive its draws from those same
/// inputs so a replay reproduces the episode — never from a hidden global.
pub trait RlStep: Send {
    /// Make the optimizer for a newly built or reloaded network.
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer>;

    /// Restore this net's optimizer state from `path` (see
    /// [`TabularStep::restore_optimizer`] for why this is a trainer hook).
    fn restore_optimizer(
        &self,
        _net: &Network,
        _path: &std::path::Path,
    ) -> Option<Box<dyn flodl::nn::optim::Optimizer>> {
        None
    }

    /// Optional self-description of this scheme's hyperparameters, recorded in
    /// `engine.json` under `"trainer"`. `None` (the default) = "no description".
    fn describe(&self) -> Option<serde_json::Value> {
        None
    }

    /// Train the network for one clock-step in place: run the environment,
    /// apply the update, and report the reward-derived fitness.
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> flodl::tensor::Result<RlStepReport>;

        /// Pop-wide phase: called ONCE per step with ALL live nets, BEFORE any per-net
        /// `train_step`. Default: no-op; for loop-breaking schemes needing the population.
        /// A scheme that implements this MUST also return `true` from
        /// [`Self::wants_pop_phase`], or the engine will never call it.
    fn pop_phase(&mut self, _nets: &mut [(String, &mut Network)], _step: usize) {}

        /// Does this scheme use [`Self::pop_phase`]? The engine asks this BEFORE gathering
        /// the live-net slice — an O(population) pass per step — so `false` is what keeps a
        /// scheme that ignores the hook free. Return `true` only alongside a real
        /// `pop_phase` override.
    fn wants_pop_phase(&self) -> bool {
        false
    }
}

// ── The engine's trainer seam ────────────────────────────────────────────────────

/// The ENGINE's view of any trainer — one value, two modes. Every method is an
/// exhaustive `match`, so adding one cannot fall through to a trait default.
pub enum AnyTrainer {
    /// A tabular (dataset-driven) trainer.
    Tabular(Box<dyn TabularStep>),
    /// An RL (environment-driven) trainer.
    Rl(Box<dyn RlStep>),
}

impl AnyTrainer {
    /// Build the tabular arm, boxing the concrete trainer.
    pub fn tabular(trainer: impl TabularStep + 'static) -> Self {
        Self::Tabular(Box::new(trainer))
    }

    /// Build the RL arm, boxing the concrete trainer.
    pub fn rl(trainer: impl RlStep + 'static) -> Self {
        Self::Rl(Box::new(trainer))
    }

    /// Mode tag (drives the engine's log columns and RL-only phases).
    pub fn is_rl(&self) -> bool {
        matches!(self, Self::Rl(_))
    }

    /// Restore optimizer state through the trainer's own concrete optimizer type.
    pub fn restore_optimizer(
        &self,
        net: &Network,
        path: &std::path::Path,
    ) -> Option<Box<dyn flodl::nn::optim::Optimizer>> {
        match self {
            Self::Tabular(t) => t.restore_optimizer(net, path),
            Self::Rl(t) => t.restore_optimizer(net, path),
        }
    }

    /// Build the optimizer this scheme trains with.
    pub fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        match self {
            Self::Tabular(t) => t.make_optimizer(net),
            Self::Rl(t) => t.make_optimizer(net),
        }
    }

    /// This scheme's hyperparameter blob for `engine.json` (`None` = none).
    pub fn describe(&self) -> Option<serde_json::Value> {
        match self {
            Self::Tabular(t) => t.describe(),
            Self::Rl(t) => t.describe(),
        }
    }

    /// R9 — this net's trainer bookkeeping for a checkpoint. The RL arm has no state hook:
    /// its per-episode bookkeeping is the environment's, and an env that draws its own
    /// randomness is already documented as non-resumable.
    pub fn trainer_state(&self, net_hash: &str) -> Option<serde_json::Value> {
        match self {
            Self::Tabular(t) => t.state(net_hash),
            Self::Rl(_) => None,
        }
    }

    /// R9 — hand a resumed net's bookkeeping back to the scheme before its next step.
    pub fn restore_trainer_state(&mut self, net_hash: &str, state: &serde_json::Value) {
        if let Self::Tabular(t) = self {
            t.restore(net_hash, state);
        }
    }

    /// RL pop-wide phase (a no-op for the tabular arm): see [`RlStep::pop_phase`].
    pub fn pop_phase(&mut self, nets: &mut [(String, &mut Network)], step: usize) {
        match self {
            Self::Tabular(_) => {}
            Self::Rl(t) => t.pop_phase(nets, step),
        }
    }

    /// Does the pop-wide phase need to run this step? `false` for the tabular arm and for
    /// any RL scheme that leaves [`RlStep::pop_phase`] alone, so the engine skips building
    /// the live-net slice entirely: see [`RlStep::wants_pop_phase`].
    pub fn wants_pop_phase(&self) -> bool {
        match self {
            Self::Tabular(_) => false,
            Self::Rl(t) => t.wants_pop_phase(),
        }
    }

    /// One training step — the arm decides which context type to build.
    #[allow(clippy::too_many_arguments)]
    pub fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        data: Option<&RunData<'_>>,
        fitness: &crate::utils::fitness::Fitness,
        metrics: &[crate::utils::fitness::Metric],
        env: StepEnv,
        net_hash: &str,
        net_seed: u64,
    ) -> flodl::tensor::Result<StepReport> {
        match self {
            Self::Tabular(t) => {
                let data = data
                    .expect("Tabular step without data — engine construction invariant violated");
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
                Ok(t.train_step(net, optimizer, step, &ctx)?.into())
            }
            Self::Rl(t) => {
                let ctx = RlContext {
                    fitness,
                    metrics,
                    env,
                    net_hash,
                    net_seed,
                };
                Ok(t.train_step(net, optimizer, step, &ctx)?.into())
            }
        }
    }

    /// The tabular loss, when this is a tabular trainer (checkpoint exam).
    pub fn tabular_loss(&self) -> Option<super::LossFn<'_>> {
        match self {
            Self::Tabular(t) => Some(t.loss()),
            Self::Rl(_) => None,
        }
    }

    /// Requested batch geometry (tabular only; `None` for RL).
    pub fn stream_shape(&self) -> Option<StreamShape> {
        match self {
            Self::Tabular(t) => t.stream_shape(),
            Self::Rl(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // `Module` provides `Network::parameters()`, used by `make_optimizer`.
    use flodl::Module;

    /// The minimal RL scheme: exists to exercise the enum's defaults.
    struct SentinelRl;

    impl RlStep for SentinelRl {
        fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }
        fn train_step(
            &mut self,
            _net: &mut Network,
            _optimizer: &mut dyn flodl::nn::optim::Optimizer,
            _step: usize,
            _ctx: &RlContext<'_>,
        ) -> flodl::tensor::Result<RlStepReport> {
            unimplemented!("the pop-phase test never trains")
        }
    }

        /// An RL scheme that DOES use the pop-wide hook — the opt-in half of the
    /// `pop_phase_is_opt_in` test.
    struct PopPhaseRl;

    impl RlStep for PopPhaseRl {
        fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }
        fn train_step(
            &mut self,
            _net: &mut Network,
            _optimizer: &mut dyn flodl::nn::optim::Optimizer,
            _step: usize,
            _ctx: &RlContext<'_>,
        ) -> flodl::tensor::Result<RlStepReport> {
            unimplemented!("the pop-phase test never trains")
        }
        fn pop_phase(&mut self, _nets: &mut [(String, &mut Network)], _step: usize) {}
        fn wants_pop_phase(&self) -> bool {
            true
        }
    }

        /// REGRESSION: the pop-wide phase is opt-in. The engine asks `wants_pop_phase`
        /// BEFORE gathering the live nets, so the tabular arm and any RL scheme that
        /// leaves the hook alone must say `false` and skip the per-step allocation.
    #[test]
    fn pop_phase_is_opt_in() {
        let tabular = crate::trainer::TabularTrainer::new(
            |_pred, _y| -> flodl::tensor::Result<flodl::Variable> {
                unimplemented!("the pop-phase test never trains")
            },
        );
        assert!(
            !AnyTrainer::tabular(tabular).wants_pop_phase(),
            "the tabular arm has no pop-wide phase"
        );
        assert!(
            !AnyTrainer::rl(SentinelRl).wants_pop_phase(),
            "an RL scheme that leaves pop_phase alone must not trigger the gather"
        );
        assert!(
            AnyTrainer::rl(PopPhaseRl).wants_pop_phase(),
            "a scheme that implements pop_phase must opt in"
        );
    }

    /// The ctx carries the run clock (`step` + `max_steps`), so a trainer can anneal
        /// its own schedule; `None` = unbounded, the same convention as challenge decay.
    #[test]
    fn step_env_progress_reads_the_run_clock() {
        let env = StepEnv {
            step: 25,
            run_seed: 1,
            pop_size: 2,
            live_count: 2,
            checkpoint_every: 10,
            smoothing_window: 5,
            max_steps: Some(100),
        };
        assert_eq!(env.progress(), Some(0.25));
        // Past the budget clamps rather than overshooting.
        assert_eq!(StepEnv { step: 400, ..env }.progress(), Some(1.0));
        // Unbounded: the trainer gets no schedule signal at all.
        assert_eq!(
            StepEnv {
                max_steps: None,
                ..env
            }
            .progress(),
            None
        );
    }


}
