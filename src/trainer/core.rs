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
//! (`make_optimizer`, optional `describe`, optional `holdout_score`) next to
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
}

/// The context handed to an [`RlStep`] each step. No data field: the trainer owns
/// its environment. Fitness direction/label is exposed for logging.
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
pub trait TabularStep: Send {
    /// Make the optimizer for a newly built or reloaded network.
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer>;

        /// Optional self-description of this scheme's hyperparameters, recorded in
        /// `engine.json` under `"trainer"`. The engine never interprets the contents.
    fn describe(&self) -> Option<serde_json::Value> {
        None
    }

        /// Guardrail measurement half: score ONE fresh holdout game with the given net,
        /// in the run's fitness units. `game_i` seeds your draw. Default: unimplemented.
    fn holdout_score(&mut self, _net: &mut Network, _game_i: usize) -> flodl::tensor::Result<f32> {
        unimplemented!(
            "holdout_score is not implemented for this trainer — implement it (one fresh \
             game, same units as the reported fitness) to use engine.guardrail()"
        )
    }

    /// The loss this scheme trains against: `(pred, target) -> loss`.
    fn loss(&self) -> crate::trainer::LossFn<'_>;

        /// Optional LR schedule for `step` (`None` = the optimizer's fixed LR), applied
        /// by the trainer. MUST be a pure function of `step` (replay contract).
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

/// The environment-driven training contract — the ONE trait an RL scheme
/// implements. No `loss()`; the engine's only ranking input is `fitness`.
pub trait RlStep: Send {
    /// Make the optimizer for a newly built or reloaded network.
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer>;

    /// Optional self-description of this scheme's hyperparameters, recorded in
    /// `engine.json` under `"trainer"`. `None` (the default) = "no description".
    fn describe(&self) -> Option<serde_json::Value> {
        None
    }

        /// Guardrail measurement half: score ONE fresh holdout game with the given net,
        /// in the run's fitness units. `game_i` seeds your draw. Default: unimplemented.
    fn holdout_score(&mut self, _net: &mut Network, _game_i: usize) -> flodl::tensor::Result<f32> {
        unimplemented!(
            "holdout_score is not implemented for this trainer — implement it (one fresh \
             game, same units as the reported fitness) to use engine.guardrail()"
        )
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
    fn pop_phase(&mut self, _nets: &mut [(String, &mut Network)], _step: usize) {}
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

    /// RL pop-wide phase (a no-op for the tabular arm): see [`RlStep::pop_phase`].
    pub fn pop_phase(&mut self, nets: &mut [(String, &mut Network)], step: usize) {
        match self {
            Self::Tabular(_) => {}
            Self::Rl(t) => t.pop_phase(nets, step),
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
        fitness: &crate::engine::fitness::Fitness,
        metrics: &[crate::engine::fitness::Metric],
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

    /// Guardrail dispatch — forwards to the trainer's own `holdout_score`.
    pub fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> flodl::tensor::Result<f32> {
        match self {
            Self::Tabular(t) => t.holdout_score(net, game_i),
            Self::Rl(t) => t.holdout_score(net, game_i),
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
    use crate::graph::node::Node;
    use crate::graph::topology::Topology;
    use flodl::Device;
    // `Module` provides `Network::parameters()`, used by `make_optimizer`.
    use flodl::Module;

    /// A trainer that ANSWERS the guardrail with a sentinel, so a test can tell
    /// "the override ran" from "the trait default panicked".
    struct SentinelRl;

    impl RlStep for SentinelRl {
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

        /// REGRESSION: `engine.guardrail()` must reach the trainer's OWN
        /// `holdout_score` through enum dispatch, not the trait DEFAULT (`unimplemented!()`).
    #[test]
    fn enum_dispatch_reaches_the_rl_override() {
        let mut net = tiny_net();
        // AnyTrainer::Rl -> Box<dyn RlStep> -> SentinelRl.
        let mut trainer = AnyTrainer::rl(SentinelRl);
        assert_eq!(
            trainer.holdout_score(&mut net, 1).unwrap(),
            101.0,
            "the boxed trainer's override must run, not the trait default"
        );
    }

        /// The tabular half of the same bug: a trainer WITH a scorer installed must
        /// reach it through enum dispatch, not a default panic.
    #[test]
    fn enum_dispatch_reaches_the_tabular_scorer() {
        let trainer = crate::trainer::TabularTrainer::new(
            |_pred, _y| -> flodl::tensor::Result<flodl::Variable> {
                unimplemented!("the guardrail dispatch test never trains")
            },
        )
        .with_holdout_scorer(|_net, game_i| -> flodl::tensor::Result<f32> {
            Ok(7.0 + game_i as f32)
        });
        let mut any = AnyTrainer::tabular(trainer);
        let mut net = tiny_net();
        assert_eq!(
            any.holdout_score(&mut net, 2).unwrap(),
            9.0,
            "the installed scorer must run through enum dispatch"
        );
    }
}
