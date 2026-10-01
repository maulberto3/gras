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
//!   the fitness is whatever the trainer reports in `RlStepReport.fitness`.
//!
//! Reports are mode-split too: [`TabularStep`] returns [`TabularStepReport`],
//! [`RlStep`] returns [`RlStepReport`] — no always-`None` fields. Both
//! normalize into the engine-internal [`StepReport`] union (the persisted
//! `NetMetrics` stays one type: wire format, history.csv, replay parity).
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
///
/// `Default` is derived so a mostly-zero report stays short to spell:
/// `StepReport { fitness, ..Default::default() }`.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
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
    /// Tabular-only challenge volume: how many individual INPUT VALUES this
    /// net jittered this step (the trainer's report — see
    /// [`TabularStepReport::challenged_inputs`]). `0` when no challenge fired.
    #[serde(default)]
    pub challenged_inputs: usize,
    /// RL-only challenge volume: turns played under a FORCED action this step
    /// (see [`RlStepReport::challenged_turns`]). `0` when no challenge fired.
    #[serde(default)]
    pub challenged_turns: usize,
    /// RL-only environment volume for this step — see [`RlStepMeta`]. `None`
    /// in Tabular (no environment) and from RL trainers that don't report it
    /// (the log then prints `—` in the RL columns).
    #[serde(default)]
    pub rl: Option<RlStepMeta>,
}

/// The report a [`TabularStep`] trainer returns. Tabular-shaped: no RL
/// volume field exists here (a dataset trainer has no environment to
/// report) — that dishonesty of the old union struct is gone.
///
/// `Default` is derived for `..Default::default()` convenience.
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
    /// Challenge volume: how many individual INPUT VALUES this net jittered
    /// this step. The engine only keeps the books — it sums what the trainer
    /// reports, exactly like RL's `RlStepReport.challenged_turns`. Seed it from
    /// [`TabularContext::challenge_prob`], which is the effective (decayed)
    /// per-input-feature probability; leave `0` when the knob is off.
    #[serde(default)]
    pub challenged_inputs: usize,
}

/// The report an [`RlStep`] trainer returns. RL-shaped: no `eval_loss` field
/// exists here (there is no (pred, target) held-out batch in RL — evaluation
/// happens inside `train_step` and its result IS `fitness`).
///
/// `Default` is derived for `..Default::default()` convenience.
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
    /// Challenge volume: turns played under a FORCED action this step (the
    /// `ctx.challenged` signal honored). The engine sums these into the
    /// rollup's `⚔` column; normally a subset of `rl.train_turns`. `0` when no
    /// challenge fired (or the trainer ignores the signal). Direct scalar, the
    /// same shape as [`TabularStepReport::challenged_inputs`].
    #[serde(default)]
    pub challenged_turns: usize,
    /// Environment volume for this step — feeds the `matches`/`train`/`eval`
    /// rollup columns. `None` = not reported (the log prints `—`).
    pub rl: Option<RlStepMeta>,
}

// ── Report normalization: the mode reports collapse into the engine-internal
// union the persisted `NetMetrics` consumes. This is the ONLY place the two
// report types meet.

impl From<TabularStepReport> for StepReport {
    fn from(r: TabularStepReport) -> Self {
        StepReport {
            train_loss: r.train_loss,
            eval_loss: r.eval_loss,
            fitness: r.fitness,
            informative: r.informative,
            challenged_inputs: r.challenged_inputs,
            challenged_turns: 0, // tabular: challenge volume is in input values
            rl: None,            // tabular: no environment, so no volume to report
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
            challenged_inputs: 0, // RL reports turns, not input values
            challenged_turns: r.challenged_turns,
            rl: r.rl,
        }
    }
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
/// Train and eval turns are reported SEPARATELY, because a challenge can only
/// ever force TRAIN turns: the trigger is a per-(net, step) probability, so
/// the expected `⚔` is `p_eff × train_turns` — read against the total it
/// would overstate the denominator by the eval share. The rollup prints both
/// halves and the expectation itself (`⚔ <n> (exp <m>)`).
///
/// The engine sums these over the live population per clock-step and prints
/// `matches <total> │ train <total> │ eval <total> │ turns/match <mean>` in
/// the per-step rollup — so a long RL step shows *why* it took long, and a
/// changing match length (e.g. `MatchLength::RandomNumTurns`) is visible as it
/// happens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RlStepMeta {
    /// Matches (episodes) this net's trainer played this step (train + eval).
    pub matches: usize,
    /// Turns played in this step's TRAIN matches — the only turns a challenge
    /// can force, and the denominator of the rollup's expected `⚔`.
    #[serde(default)]
    pub train_turns: usize,
    /// Turns played in this step's EVAL matches. Never challenged — evaluation
    /// must keep measuring the net's own policy.
    #[serde(default)]
    pub eval_turns: usize,
}

impl RlStepMeta {
    /// Total environment turns this step (train + eval) — what the rollup's
    /// `turns/match` mean averages over.
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
    /// Checkpoint cadence (crossover-gate bar is recorded every N steps).
    pub checkpoint_every: usize,
    /// The per-net fitness smoothing window, in steps (K).
    pub smoothing_window: usize,
    /// The run's step budget as configured (`None` = unbounded). Carried so a
    /// trainer can derive its own progress and anneal schedule-shaped things
    /// (learning rate, entropy bonus, epsilon, exploration temperature)
    /// WITHOUT a new config knob and without inventing a private counter.
    pub max_steps: Option<usize>,
}

impl StepEnv {
    /// Fraction of the run's step budget consumed (`0.0` → `1.0`), or `None`
    /// when the run is unbounded — see
    /// [`crate::engine::config::run_progress`], the single definition both the
    /// engine and trainers share.
    ///
    /// Pure arithmetic on the clock: no stored state, so a replayed step,
    /// a catch-up step and a resumed step all compute the identical value —
    /// which is what keeps an annealed schedule replay-safe.
    pub fn progress(&self) -> Option<f32> {
        crate::engine::config::run_progress(self.step, self.max_steps)
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
    pub fitness: &'a crate::engine::fitness::Fitness,
    /// Informative non-ranking metrics configured for the run.
    pub metrics: &'a [crate::engine::fitness::Metric],
    /// Lightweight snapshot of current run/epoch metadata.
    pub env: StepEnv,
    /// Canonical hash of the net being trained (for logs or tracking).
    pub net_hash: &'a str,
    /// Deterministic net-specific weight initialization/dropout seed.
    pub net_seed: u64,
    /// The challenge SIGNAL (anti-plateau): true when the engine's seeded
    /// trigger fired for this (net, step) — `set_run_challenge_prob` > 0.
    /// The TRAINER owns the response: jitter its batch, swap in harder rows,
    /// drop features, … (or ignore the flag entirely — keep the knob at 0 so
    /// it never fires). Replay-safe: the flag is a pure function of
    /// run/net/step, so catch-up re-derives it identically.
    pub challenged: bool,
    /// The EFFECTIVE (decay-adjusted) challenge probability for this step —
    /// the per-(input, feature) rate for an element-level response. Roll it
    /// yourself (seeded from `net_seed`/`step` so replay matches) and report
    /// how many values you jittered in
    /// [`TabularStepReport::challenged_inputs`]. `0.0` = challenges off.
    pub challenge_prob: f32,
}

/// The context handed to an [`RlStep`] on every step. There is **no** data
/// field: the trainer owns its environment. The fitness direction/label is
/// still exposed (for logging the reported score in the trainer's own style).
pub struct RlContext<'a> {
    /// The ranking fitness metric (direction + label; the VALUE is the
    /// trainer's to produce in `RlStepReport.fitness`).
    pub fitness: &'a crate::engine::fitness::Fitness,
    /// Informative non-ranking metrics configured for the run.
    pub metrics: &'a [crate::engine::fitness::Metric],
    /// Lightweight snapshot of current run/epoch metadata.
    pub env: StepEnv,
    /// Canonical hash of the net being trained (for logs or tracking).
    pub net_hash: &'a str,
    /// Deterministic net-specific weight initialization/dropout seed.
    pub net_seed: u64,
    /// The challenge SIGNAL (anti-plateau): true when the engine's seeded
    /// trigger fired for this (net, step) — `set_run_challenge_prob` > 0.
    /// Never true for a frozen (act-and-measure) step or in the post-race
    /// pruner's solo phase: the knob is a race device.
    /// The TRAINER decides what to do with it: branch to a forced/Exploration
    /// action inside `train_step` (replay-safe — the flag is a pure function
    /// of run/net/step, so catch-up re-derives it identically), or ignore it
    /// (a trainer without challenge logic never looks at the flag — keep the
    /// knob at 0 so it never fires). Whatever the trainer reports in
    /// `RlStepReport.challenged_turns` is what the log shows; the engine
    /// does not verify the flag was honored.
    pub challenged: bool,
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
///
/// # Contract: BOTH traits, always
/// Implementing a mode contract means complying with BOTH halves — the
/// supertrait [`StepTrainer`] (shared lifecycle: `make_optimizer`, optional
/// `describe`) AND the mode trait itself (the step: [`RlStep::train_step`] or
/// [`TabularStep::train_step`]). Rust requires two `impl` blocks; that is the
/// contract, not boilerplate. `RlStep` alone does not compile as a trainer.
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

    /// The guardrail's measurement half: score ONE fresh holdout game with
    /// the given (champion-reloaded) net, in the SAME units and estimator
    /// as the fitness reported in the step report. Implement it to make the
    /// trainer guardrail-capable — the engine's `guardrail()` then needs no
    /// second scorer object: `engine.guardrail()` is the whole call.
    ///
    /// How MANY games are played is the run's business, not the trainer's:
    /// the config's `set_elite_guardrail_matches` (default 16). The contract is
    /// index-based (`game_i`), so any count works — seed the draw from
    /// `game_i` and the games stay independent.
    ///
    /// Default: unimplemented (calling `guardrail()` without a scorer
    /// panics with this contract text — loud, not silent).
    fn holdout_score(&mut self, _net: &mut Network, _game_i: usize) -> flodl::tensor::Result<f32> {
        unimplemented!(
            "holdout_score is not implemented for this trainer — implement it (one fresh \
             game, same units as the reported fitness) to use engine.guardrail()"
        )
    }
}

/// The dataset-driven (supervised) training contract. A trainer complies
/// with BOTH [`StepTrainer`] and this trait — two `impl` blocks, both
/// required (see [`StepTrainer`]).
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
    ) -> flodl::tensor::Result<TabularStepReport>;
}

/// The environment-driven (RL) training contract.
///
/// # Contract: BOTH traits, always
/// A trainer complies with BOTH [`StepTrainer`] (supertrait: optimizer +
/// recipe) and `RlStep` (this step contract) — two `impl` blocks, both
/// required. See [`StepTrainer`] for the full statement.
///
/// There is deliberately NO `loss()` method and no data in [`RlContext`]:
/// the training signal (rewards, trajectories, advantages) is the trainer's
/// internal business, produced inside `train_step`. The engine's ONLY ranking
/// input is the fitness value the trainer puts in
/// [`RlStepReport::fitness`] each step — which is why RL runs are built with
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
    ) -> flodl::tensor::Result<RlStepReport>;

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

/// The ENGINE's view of any trainer — the seam the split rests on
/// (TODO.md step 8). The engine core never knows which mode it serves; it
/// talks to this trait. Implemented by [`ModeAdapter`], the per-mode
/// adapter whose type carries the mode tag. Sealed: only this module can
/// add implementations.
pub trait EngineTrainer: Send {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer>;
    fn describe(&self) -> Option<serde_json::Value>;
    /// RL pop-wide phase (no-op for tabular): see [`RlStep::pop_phase`].
    fn pop_phase(&mut self, nets: &mut [(String, &mut Network)], step: usize);
    /// One training step — the mode decides which context type to build.
    /// `challenged` is the engine's challenge SIGNAL (both modes; `false`
    /// when the knob is off) — the adapter folds it into the context it
    /// builds.
    #[allow(clippy::too_many_arguments)]
    fn train_step(
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
        challenged: bool,
        challenge_prob: f32,
    ) -> flodl::tensor::Result<StepReport>;
    /// Guardrail dispatch — forwards to the inner trainer's
    /// `StepTrainer::holdout_score` (see the adapters).
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> flodl::tensor::Result<f32> {
        let _ = (net, game_i);
        unimplemented!(
            "holdout_score is not implemented for this trainer — implement it (one fresh \
             game, same units as the reported fitness) to use engine.guardrail()"
        )
    }
    /// Mode tag (drives the engine's log columns and RL-only phases).
    fn is_rl(&self) -> bool;
    /// The tabular loss, when this is a tabular trainer (checkpoint exam).
    fn tabular_loss(&self) -> Option<super::LossFn<'_>>;
    /// Requested batch geometry (tabular only; `None` for RL).
    fn stream_shape(&self) -> Option<StreamShape> {
        None
    }
}

/// Per-mode engine adapter: statically knows its mode, wraps the boxed
/// user trainer. The mode tag lives in the TYPE, not a runtime enum arm
/// (TODO.md step 8/9 — replaces the deleted ModeTrainer enum).
pub struct ModeAdapter<T: Send + 'static> {
    inner: T,
}

impl ModeAdapter<Box<dyn TabularStep>> {
    pub fn tabular(inner: Box<dyn TabularStep>) -> Self {
        Self { inner }
    }
}

impl ModeAdapter<Box<dyn RlStep>> {
    pub fn rl(inner: Box<dyn RlStep>) -> Self {
        Self { inner }
    }
}

impl EngineTrainer for ModeAdapter<Box<dyn TabularStep>> {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.inner.make_optimizer(net)
    }
    fn describe(&self) -> Option<serde_json::Value> {
        self.inner.describe()
    }
    fn pop_phase(&mut self, _nets: &mut [(String, &mut Network)], _step: usize) {}
    fn train_step(
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
        challenged: bool,
        challenge_prob: f32,
    ) -> flodl::tensor::Result<StepReport> {
        let data =
            data.expect("Tabular step without data — engine construction invariant violated");
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
            challenged,
            challenge_prob,
        };
        Ok(self.inner.train_step(net, optimizer, step, &ctx)?.into())
    }
    /// Guardrail dispatch — WITHOUT this the `EngineTrainer` default runs
    /// (`unimplemented!()`), so `engine.guardrail()` panicked even for a
    /// library trainer that HAD a scorer installed
    /// (`TabularTrainer::with_holdout_scorer`). The RL adapter below always
    /// forwarded these; this one was missing them.
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> flodl::tensor::Result<f32> {
        self.inner.holdout_score(net, game_i)
    }
    fn is_rl(&self) -> bool {
        false
    }
    fn tabular_loss(&self) -> Option<super::LossFn<'_>> {
        Some(self.inner.loss())
    }
    fn stream_shape(&self) -> Option<StreamShape> {
        self.inner.stream_shape()
    }
}

impl EngineTrainer for ModeAdapter<Box<dyn RlStep>> {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.inner.make_optimizer(net)
    }
    fn describe(&self) -> Option<serde_json::Value> {
        self.inner.describe()
    }
    fn pop_phase(&mut self, nets: &mut [(String, &mut Network)], step: usize) {
        self.inner.pop_phase(nets, step)
    }
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        _data: Option<&RunData<'_>>,
        fitness: &crate::engine::fitness::Fitness,
        metrics: &[crate::engine::fitness::Metric],
        env: StepEnv,
        net_hash: &str,
        net_seed: u64,
        challenged: bool,
        _challenge_prob: f32,
    ) -> flodl::tensor::Result<StepReport> {
        let ctx = RlContext {
            fitness,
            metrics,
            env,
            net_hash,
            net_seed,
            challenged,
        };
        Ok(self.inner.train_step(net, optimizer, step, &ctx)?.into())
    }
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> flodl::tensor::Result<f32> {
        self.inner.holdout_score(net, game_i)
    }
    fn is_rl(&self) -> bool {
        true
    }
    fn tabular_loss(&self) -> Option<super::LossFn<'_>> {
        None
    }
}
impl StepTrainer for Box<dyn StepTrainer> {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.as_ref().make_optimizer(net)
    }
    fn describe(&self) -> Option<serde_json::Value> {
        self.as_ref().describe()
    }
    /// Guardrail: forward through the box. Omitting this is a silent trap —
    /// `self.inner.holdout_score(..)` on a `Box<dyn _>` resolves to the BOX's
    /// impl (found before the autoderef to `dyn _`), whose default is
    /// `unimplemented!()`. Every boxed impl below must forward it.
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> flodl::tensor::Result<f32> {
        self.as_mut().holdout_score(net, game_i)
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
    ) -> flodl::tensor::Result<TabularStepReport> {
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
    ) -> flodl::tensor::Result<RlStepReport> {
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
    /// Guardrail forwarding — see the `Box<dyn StepTrainer>` impl above.
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> flodl::tensor::Result<f32> {
        self.as_mut().holdout_score(net, game_i)
    }
}

impl StepTrainer for Box<dyn RlStep> {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.as_ref().make_optimizer(net)
    }
    fn describe(&self) -> Option<serde_json::Value> {
        self.as_ref().describe()
    }
    /// Guardrail forwarding — see the `Box<dyn StepTrainer>` impl above. This
    /// was the hole that made `examples/cartpole` panic AFTER its run:
    /// `ModeAdapter<Box<dyn RlStep>>::holdout_score` forwards to here, and an
    /// absent override here means the trait default `unimplemented!()` runs.
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> flodl::tensor::Result<f32> {
        self.as_mut().holdout_score(net, game_i)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::node::Node;
    use crate::graph::topology::Topology;
    use flodl::Device;
    // `Module` provides `Network::parameters()`, used by `make_optimizer`.
    use flodl::Module;

    /// A trainer that ANSWERS the guardrail with a sentinel, so a test can tell
    /// "the override ran" from "the trait default panicked".
    struct SentinelRl;

    impl StepTrainer for SentinelRl {
        fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }
        fn holdout_score(
            &mut self,
            _net: &mut Network,
            game_i: usize,
        ) -> flodl::tensor::Result<f32> {
            Ok(100.0 + game_i as f32)
        }
    }

    impl RlStep for SentinelRl {
        fn train_step(
            &mut self,
            _net: &mut Network,
            _optimizer: &mut dyn flodl::nn::optim::Optimizer,
            _step: usize,
            _ctx: &RlContext<'_>,
        ) -> flodl::tensor::Result<RlStepReport> {
            unimplemented!("the guardrail dispatch test never trains")
        }
    }

    /// The ctx carries the run clock (`step` + `max_steps`), so a trainer can
    /// derive progress and anneal its own schedules without a private counter —
    /// and `None` means "unbounded: no schedule", the same convention the
    /// engine's challenge decay uses.
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

    /// The smallest net that satisfies the guardrail's signature — the
    /// sentinel never touches it, but a `&mut Network` is required to call.
    fn tiny_net() -> Network {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.refresh_labels();
        graph.finalize();
        Network::build(&graph, Device::CPU).unwrap()
    }

    /// REGRESSION (cartpole's post-run abort): `engine.guardrail()` must reach
    /// the trainer's OWN `holdout_score` through every box in the chain. It did
    /// not — `Box<dyn RlStep>` resolves `holdout_score` to its own
    /// `StepTrainer` impl (found before the autoderef to `dyn RlStep`), and that
    /// impl forwarded only `make_optimizer`/`describe`, so the trait DEFAULT
    /// ran and `unimplemented!()` killed a race that had already finished.
    #[test]
    fn guardrail_forwarding_survives_boxing_rl() {
        let mut net = tiny_net();
        // ModeAdapter<Box<dyn RlStep>> -> Box<dyn RlStep> -> SentinelRl.
        let mut adapter = ModeAdapter::rl(Box::new(SentinelRl));
        assert_eq!(
            EngineTrainer::holdout_score(&mut adapter, &mut net, 1).unwrap(),
            101.0,
            "the boxed trainer's override must run, not the trait default"
        );
    }

    /// The tabular half of the same bug: `ModeAdapter<Box<dyn TabularStep>>`
    /// omitted the holdout forward entirely, so even a library trainer WITH a
    /// scorer installed (`TabularTrainer::with_holdout_scorer` — the MNIST
    /// path) hit the `EngineTrainer` default panic.
    #[test]
    fn guardrail_forwarding_survives_boxing_tabular() {
        let trainer = crate::trainer::TabularTrainer::new(
            |_pred, _y| -> flodl::tensor::Result<flodl::Variable> {
                unimplemented!("the guardrail dispatch test never trains")
            },
        )
        .with_holdout_scorer(|_net, game_i| -> flodl::tensor::Result<f32> {
            Ok(7.0 + game_i as f32)
        });
        let mut adapter = ModeAdapter::tabular(Box::new(trainer));
        let mut net = tiny_net();
        assert_eq!(
            EngineTrainer::holdout_score(&mut adapter, &mut net, 2).unwrap(),
            9.0,
            "the installed scorer must run through the tabular adapter"
        );
    }
}
