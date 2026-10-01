//! Step-race scheduler for the continuous step-race engine.
//!
//! One global `step_clock`. Every live net — original population or
//! caught-up newcomer — is at the **same** step at the start of every loop
//! iteration. One iteration = one step_clock for the whole group:
//!
//! 1. The shared `BatchStream` produces one `(train_batch, eval_batch)` for
//!    this step — identical for every net.
//! 2. Each live net (deterministic hash order) does `train_one_step` +
//!    `eval_one_step` on its **live** `Network` + `Optimizer` (coefs evolve
//!    in place; optimizer state carries forward across steps), records its
//!    metrics into `nets/<hash>.json`, and logs one per-net line.
//! 3. After **all** nets stepped: every `checkpoint_every` steps, record the
//!    population-mean smoothed fitness in the checkpoint ledger — the bar a
//!    crossover child must clear.
//! 4. Evolve — two independent roll groups. Crossover rolls: a checkpoint-gated
//!    child replaces the worst net only if it clears every gate, otherwise the
//!    attempt is discarded and the population is unchanged. Immigrant rolls: a
//!    random entrant replaces an inverse-fitness victim (pop stays constant).
//! 5. One per-step rollup line (or the `Minimal` frame) describing the step
//!    that just ended — logged LAST on purpose, so it always sits at the bottom
//!    of the terminal.
//! 6. Stop criteria checked (max_steps, custom_stop) —
//!    log which fired.
//! 7. Repeat — the loop's clock is the nets' recorded steps (every net is at
//!    the same step after the group steps, so reading any live net's step
//!    gives the clock).
//!
//! Determinism: the children's catch-up uses the same `seed_step_randomness` +
//! same step primitives as the group loop, so two identical-seed runs produce
//! identical cull/insertion sequences.
//!
//! Memory model: **all live nets + their optimizers live inside the loop** for
//! the whole run. Pop 5 → 5 networks + 5 optimizers in memory at once. On
//! cull, the culled net's entries are dropped (memory freed). On insert, the
//! child's entries are added. This is simple, not minimal-memory — fine for
//! the small default pop; the user runs bigger pops when it matters.
//!
//! Persistence:
//! - `engine.json` — written once at run start (the run header; `RunHeader`).
//! - `nets/<hash>.json` — written **once** when a net leaves the live pop
//!   (cull write — its final state snapshot), and written **once** at run end
//!   for every still-live net (stop write — the live pop's final state).
//!   During the run, live nets are held only in memory. An interrupted run
//!   leaves `engine.json` + the culled nets' JSONs on disk; resume
//!   reconstructs the live frontier from those + replays each net's stream.
//!   `history.csv` (per-step metric rows) and `attempts.csv` (evolution
//!   events) are the run's logs, flushed at checkpoints and at stop.

use flodl::nn::Optimizer;
use flodl::tensor::Result;
use std::collections::HashMap;
use tracing::info;

use crate::engine::fitness::{Fitness, FitnessLabel, Metric};
use crate::graph::network::Network;
use crate::state::{
    ConfigSnapshot, NetState, RaceState, RunConfig, RunHeader, write_engine_json, write_net_state,
};
use crate::trainer::stream::{BatchStream, PoolSplit};

/// Placeholder swapped into `self.trainer` for the duration of the pop-wide
/// phase, so the real trainer can be taken out of `self` without aliasing
/// the `&mut Network` borrows. Its `pop_phase` is the trait's no-op default
/// and its other methods are never called (it lives inside `run` for a few
/// lines only).
struct NoopPopTrainer;
impl crate::trainer::EngineTrainer for NoopPopTrainer {
    fn make_optimizer(&self, _net: &Network) -> Box<dyn Optimizer> {
        unreachable!("NoopPopTrainer never builds optimizers")
    }
    fn describe(&self) -> Option<serde_json::Value> {
        None
    }
    fn pop_phase(&mut self, _nets: &mut [(String, &mut Network)], _step: usize) {}
    fn train_step(
        &mut self,
        _net: &mut Network,
        _optimizer: &mut dyn Optimizer,
        _step: usize,
        _data: Option<&crate::trainer::RunData<'_>>,
        _fitness: &Fitness,
        _metrics: &[Metric],
        _env: crate::trainer::StepEnv,
        _net_hash: &str,
        _net_seed: u64,
        _challenged: bool,
        _challenge_prob: f32,
    ) -> flodl::tensor::Result<crate::trainer::StepReport> {
        unreachable!("NoopPopTrainer is a placeholder only for the pop-wide phase")
    }
    fn is_rl(&self) -> bool {
        true
    }
    fn tabular_loss(&self) -> Option<crate::trainer::LossFn<'_>> {
        None
    }
}
use super::config::{RaceConfig, StopReason};
use super::format::{build_stamp, csv_field};
use super::smoothing::{RollingBuffer, rolling_mean};

/// One entry in the checkpoint ledger: the population-mean smoothed fitness
/// recorded at a checkpoint step. A crossover child must BEAT this value at
/// every checkpoint it passes through during catch-up, or it is discarded.
/// `exam_mean_fitness` is the population's mean score on that era's
/// surprise-exam batch (rows never seen in training or per-step eval) — an
/// anti-memorization diagnostic recorded alongside the gate bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Checkpoint {
    pub step: usize,
    pub pop_mean_fitness: f32,
    /// Mean exam fitness across all live nets at this checkpoint.
    pub exam_mean_fitness: f32,
}

// ── CoreEngine ───────────────────────────────────────────────────────────────

/// The continuous step-race loop.
///
/// Owns the global step clock (implicit — the nets' recorded steps **are** the
/// clock), the shared batch stream, the in-memory live population (`RaceState`
/// with per-net `Network` + per-net `Optimizer`), and the per-net rolling
/// fitness buffers (ranking smoothing: cull/insert, parent roulette,
/// checkpoint means).
///
/// Memory model: **all live nets + their optimizers live inside the loop** for
/// the whole run. Pop 5 means 5 networks + 5 optimizers in memory at once. On
/// cull, the culled net's entries are dropped (memory freed). On insert, the
/// child's entries are added. This is simple, not minimal-memory — fine for
/// the small default pop; the user runs bigger pops when it matters.
pub struct CoreEngine {
    pub(crate) run_dir: std::path::PathBuf,
    pub(crate) header: RunHeader,
    pub(crate) config: RaceConfig,
    /// Run-level context stamped into each net's meta block.
    pub(crate) meta_ctx: crate::engine::config::RunMetaCtx,
    /// Shared batch stream (Tabular mode). `None` in RL mode — the trainer
    /// drives its own environment; `ctx.data` is `None` there.
    pub(crate) stream: Option<BatchStream>,
    /// The run's dataset (Tabular mode). `None` in RL mode.
    pub(crate) dataset: Option<crate::utils::tabular_data::Dataset>,
    /// Explicit two-dataset layout marker: `Some(test_len)` when the run was
    /// built from `{train, test}` dirs (test rows concatenated after train
    /// rows in `dataset`). Drives the pool construction AND the header
    /// record so resume rebuilds identical pools. `None` = single-pool split.
    pub(crate) explicit_test_rows: Option<usize>,
    pub(crate) fitness: Fitness,
    pub(crate) metrics: Vec<Metric>,
    /// The caller-supplied training scheme, boxed per mode. The engine never
    /// trains — it only orchestrates the population lifecycle and delegates
    /// one net/one step to this contract (see `crate::trainer::{TabularStep,
    /// RlStep}`). The mode arm decides which context (data or no data) the
    /// step receives.
    ///
    /// Engine-split note (TODO.md step 5): `TabularEngine`/`RlEngine` wrap
    /// this core and are the public types; this field stays the internal
    /// dispatch enum until step 8/9 deletes it in favor of per-mode concrete
    /// trainers.
    pub(crate) trainer: Box<dyn crate::trainer::EngineTrainer>,
    /// Per-step log verbosity for the engine.
    pub(crate) log_level: crate::engine::config::LogLevel,
    /// Graceful Ctrl+C flag (set by `run`'s signal handler, consumed by
    /// `check_stop`). `None` outside `run` — tests construct without it.
    pub(crate) interrupt_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,

    /// Per-step step-log counters captured during the evolve phase and
    /// consumed by the SAME step's rollup line / `Minimal` framed table (both
    /// render after the evolve phase, so the reader's last line always
    /// describes the step that just finished).
    pub(crate) step_evolve: StepEvolve,
    /// Per-step RL volume (matches + turns), summed over every net that
    /// stepped this clock. Reset at the top of each step; consumed by the
    /// same step's rollup line / `Minimal` table. Stays zero in Tabular mode.
    pub(crate) step_rl: RlVolume,
    /// Run-total of challenged turns (live path only — a resume's replay
    /// mirrors history and does not re-count). Surfaced in the stop summary
    /// next to the turns actually played, so the challenge rate is directly
    /// readable as a fraction. Not persisted: a resumed run counts only its
    /// own post-resume turns.
    pub(crate) total_challenged_turns: usize,
    /// Run-total of the TRAIN turns those challenged turns came out of — the
    /// denominator `total_challenged_turns` is only meaningful against (eval
    /// turns are never challengeable). Same live-only accounting.
    pub(crate) total_train_turns: usize,
    /// Run-total of the turns the trigger was expected to force: summed
    /// `p_eff(step) × train_turns(step)`. Decay-aware, so it is directly
    /// comparable to `total_challenged_turns` — how far the run's realized
    /// challenge footprint sits from what the knob asked for.
    pub(crate) expected_challenged_turns: f32,
    /// Per-step count of individual INPUT VALUES challenged (jittered) by the
    /// tabular trainers this clock — the element-level volume the trainers
    /// reported. Reset each step.
    pub(crate) step_challenged_inputs: usize,
    /// Run-total challenged input values (live path only). Surfaced in the
    /// stop summary for tabular runs, where environment turns don't exist.
    pub(crate) total_challenged_inputs: usize,
    /// Run-total of the EXPECTED challenged input values: summed
    /// `p_eff(step) × rows × features` over every tabular net-step, directly
    /// comparable to `total_challenged_inputs`.
    pub(crate) expected_challenged_inputs: f32,
    /// Full hashes of the elite set exported at stop (top-`elite_count` by
    /// smoothed fitness) — the single source of truth for "which nets are the
    /// champions". Populated by both `write_champion_*` writers so post-race
    /// tooling (e.g. the examples' holdout guardrails) reads THE champions
    /// instead of guessing from file mtimes. Empty until the first champion
    /// dump; empty when `elite_count` = 0 or no live net has metrics.
    pub(crate) champions: Vec<String>,
    /// In-memory state of the live population (topology JSON, seed, step,
    /// last_metrics, lineage). The source of truth for resume + tooling.
    pub(crate) state: RaceState,
    /// Per-net `Network` — built once on insert / catch-up complete, mutated
    /// in place each step. **Coefficients live here**, not on disk.
    pub(crate) networks: HashMap<String, Network>,
    /// Per-net `Optimizer` — created on insert / catch-up complete, mutated in
    /// place each step. Optimizer state (Adam momentum/variance) carries
    /// forward across steps.
    pub(crate) optimizers: HashMap<String, Box<dyn Optimizer>>,
    /// Rolling fitness history per net (in-memory; equivalent to reading back
    /// the last K `NetMetrics` snapshots). Every ranking decision reads the
    /// smoothed mean of this buffer, not the raw per-step value.
    pub(crate) rolling_fitness: HashMap<String, RollingBuffer>,
    pub(crate) rolling_train: HashMap<String, RollingBuffer>,
    pub(crate) rolling_eval: HashMap<String, RollingBuffer>,
    /// Wall-clock start for the wall-clock stop criterion.
    pub(crate) started_at_wall: std::time::Instant,
    /// Wall-clock seconds accumulated by EARLIER sessions of this run (stop
    /// → resume cycles). `elapsed_seconds` reports `elapsed_base_secs +
    /// started_at_wall.elapsed()`, so a resumed run keeps its true age. 0 on
    /// a fresh run; restored from `engine.json` on resume.
    pub(crate) elapsed_base_secs: u64,
    /// Wall-clock start of the CURRENT step (set at the top of the run loop).
    /// Consumed by the per-step logs (`Summ` rollup line / `Minimal` table) so
    /// long runs surface per-step cost — e.g. RL bridges where one step plays
    /// many full matches through a Python subprocess.
    pub(crate) step_started_at_wall: std::time::Instant,
    /// Cumulative culls so far (for max_culls stop).
    pub(crate) culls: usize,
    /// Children born per step clock. Disambiguates same-clock siblings across
    /// all evolution rounds at one step (crossover rolls + immigrant rolls),
    /// so same-clock children derive distinct seeds.
    pub(crate) children_born_at_clock: HashMap<usize, usize>,
    /// The checkpoint ledger: population-mean smoothed fitness recorded every
    /// `checkpoint_every` steps. A crossover child must BEAT the recorded
    /// mean at every checkpoint it passes through during catch-up, or it is
    /// discarded. Persisted to `checkpoints.json` so resume reconstructs
    /// identical gates.
    pub(crate) checkpoints: Vec<Checkpoint>,
    /// Anti-devolution A: the nets currently holding the freeze crown (the
    /// top-`elite_count` when `freeze_elites` is on — named by the ★ badge on
    /// the rollup line). Empty until the first frozen step. Tracked so the
    /// "crowned / crown moved" lines fire on TRANSITIONS only, not every
    /// step, and so the badge can name every champion (with `elite_count > 1`
    /// there is more than one).
    pub(crate) frozen_crown: std::collections::HashSet<String>,
    /// Set TRUE only inside the post-race pruner phase: the solo steps are
    /// the finishing workout, so elite freeze is BYPASSED — survivors train
    /// with the real optimizer regardless of `freeze_elites` (they are all
    /// elites by construction, so freezing would mean the champion never
    /// takes a weight update after the race). Not persisted: the pruner
    /// phase runs inside a single `finish_race` call, and its history rows
    /// are plain trained steps that replay with the real optimizer.
    pub(crate) pruner_solo_active: bool,
    /// `Minimal` mode: last step's population means (train, eval, fitness)
    /// so the framed table can show what changed this step. `None` until the
    /// first table render (the table itself starts at step 2 for this reason).
    pub(crate) minimal_prev_means: Option<(f32, f32, f32)>,
    /// Buffer history rows in memory to minimize slow disk I/O writes.
    /// `history.csv`: one row per live net per step (`append_metrics_csv`).
    pub(crate) history_csv_buffer: String,
    /// Buffer for evolution-event rows. SEPARATE file (`attempts.csv`) with
    /// its own header: an attempt row has none of a metric row's columns
    /// (train_loss/eval_loss/informative) and a metric row has none of an
    /// attempt's, so sharing one schema meant padding empty cells — and a
    /// padded cell count that drifts silently shifts every later column.
    pub(crate) attempts_csv_buffer: String,
}

// ── Construction ─────────────────────────────────────────────────────────────

/// One evolution roll's outcome, formatted for the single per-roll log line.
/// `survived`: crossover-only — did the child clear the checkpoint gate
/// (retries stop on survival; the last attempt's outcome is the one logged).
pub(crate) struct RollOutcome {
    pub survived: bool,
    pub detail: String,
}
impl RollOutcome {
    pub fn survived(detail: impl Into<String>) -> Self {
        Self {
            survived: true,
            detail: detail.into(),
        }
    }
    pub fn spent(detail: impl Into<String>) -> Self {
        Self {
            survived: false,
            detail: detail.into(),
        }
    }
}

/// Per-step evolve counters, reused by the `Minimal` table. Reset each step.
#[derive(Clone, Copy, Default)]
pub(crate) struct StepEvolve {
    pub culls: usize,
    pub inserts: usize,
    pub cross_fired: usize,
    pub cross_survived: usize,
    pub cross_discarded: usize,
    pub mutate_fired: usize,
}

/// A do-nothing optimizer — the act-and-measure freeze pass runs the net's
/// normal trainer step through this so gradients compute (the report carries
/// a real train_loss) but apply nothing: frozen weights stay frozen, and the
/// net's REAL optimizer (with its momentum state) is never touched.
pub(crate) struct NoopOptimizer;

impl flodl::nn::optim::Optimizer for NoopOptimizer {
    fn step(&mut self) -> flodl::tensor::Result<()> {
        Ok(()) // gradients accumulate; nothing applies
    }
    fn zero_grad(&self) {}
    fn lr(&self) -> f64 {
        0.0
    }
    fn set_lr(&mut self, _lr: f64) {}
}

/// Per-step RL volume, summed over every net that stepped this clock — the
/// environment work the population actually did. `matches`/`train_turns`/
/// `eval_turns` come from `StepReport.rl` (see
/// [`crate::trainer::RlStepMeta`]); `nets` counts the nets that reported, so a
/// trainer which forgets to report shows as `—` instead of a silent `0`.
/// Always zero in Tabular mode.
///
/// The train/eval split is kept apart because the challenge can only force
/// train turns: the rollup's expected `⚔` is `p_chall × train_turns`, so the
/// two halves must be summed separately to state that denominator honestly.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RlVolume {
    /// Nets that reported `StepReport.rl` this step.
    pub nets: usize,
    /// Matches played by those nets, summed (train + eval).
    pub matches: usize,
    /// Train turns played by those nets, summed — the challengeable turns.
    pub train_turns: usize,
    /// Eval turns played by those nets, summed (never challenged).
    pub eval_turns: usize,
    /// Turns played under a forced action this step, summed (what the
    /// trainers reported in `RlStepReport.challenged_turns`). 0 = none.
    pub challenged_turns: usize,
}

impl RlVolume {
    /// Add this step's challenged turns to the volume (called from the step
    /// path as reports land).
    pub(crate) fn add_challenged(&mut self, turns: usize) {
        self.challenged_turns += turns;
    }

    /// Total environment turns this step (train + eval) — the `turns/match`
    /// numerator.
    pub(crate) fn turns(&self) -> usize {
        self.train_turns + self.eval_turns
    }

    /// The RL middle column of the per-step rollup, e.g.
    /// `matches 300 │ train 1331 │ eval 2662 │ p_chall 0.100 │ ⚔ 75 (exp 133)
    /// │ turns/match 13`. `—` when no net reported any match (Tabular mode, or
    /// a mis-wired RL trainer).
    ///
    /// `eff_prob` is the CURRENT effective challenge probability (post-decay).
    /// Two derived cells hang off it, each hidden when it would lie: `p_chall`
    /// only when the knob is on, and `⚔ <n> (exp <m>)` only when the trainer
    /// reported challenged turns — where `<m> = eff_prob × train_turns` is the
    /// expectation under the per-(net, step) trigger. That lets the reader
    /// judge the draw without doing decay math (and watch the knob decay
    /// toward 0 step by step).
    pub(crate) fn label(&self, eff_prob: f32) -> String {
        if self.nets == 0 || self.matches == 0 {
            return "matches — │ train — │ eval — │ turns/match —".to_string();
        }
        let prob = if eff_prob > 0.0 {
            format!(" │ p_chall {:.3}", eff_prob)
        } else {
            String::new()
        };
        // Only train turns can be challenged, so the expectation is taken over
        // them alone — never over the train+eval total.
        let challenged = if self.challenged_turns > 0 {
            let expected = if eff_prob > 0.0 {
                format!(" (exp {:.0})", eff_prob * self.train_turns as f32)
            } else {
                String::new()
            };
            format!(" │ ⚔ {}{}", self.challenged_turns, expected)
        } else {
            String::new()
        };
        format!(
            "matches {} │ train {} │ eval {}{}{} │ turns/match {:.0}",
            self.matches,
            self.train_turns,
            self.eval_turns,
            prob,
            challenged,
            self.turns() as f32 / self.matches as f32
        )
    }
}

impl CoreEngine {
    /// The CURRENT effective challenge probability (post-decay). Delegates to
    /// [`crate::engine::config::effective_challenge_prob`] — the SAME function
    /// `challenge_fires` decides with — so the rollup line, the checkpoint
    /// line and the trigger can never disagree about the knob's value.
    ///
    /// `0.0` in the pruner solo phase: the challenge knob is a RACE device
    /// (anti-plateau exploration), and the phase is the final polish — the
    /// trigger is likewise suppressed there, so the logged/expected counts
    /// zero out with it.
    pub(crate) fn effective_challenge_prob(&self, clock: usize) -> f32 {
        if self.pruner_solo_active {
            return 0.0;
        }
        crate::engine::config::effective_challenge_prob(
            self.config.challenge_prob,
            clock,
            self.config.max_steps,
            self.config.challenge_decay,
        )
    }

    /// True when per-step detail lines (evolve rollup, per-child gate/cull/
    /// insert lines, checkpoint line, race-start line) should be emitted —
    /// i.e. every level except `Minimal` (table only) and `None` (silent).
    pub(crate) fn verbose_detail(&self) -> bool {
        !matches!(
            self.log_level,
            crate::engine::config::LogLevel::Minimal | crate::engine::config::LogLevel::None
        )
    }

    /// The run directory this engine writes to (chosen by `RunSpec.run_dir`
    /// or defaulted to `results/<timestamp>` — read back here after
    /// construction, e.g. to print or to resume later).
    pub fn run_dir(&self) -> &std::path::Path {
        &self.run_dir
    }

    /// Start a race from a [`RunSpec`] — the entire argument surface.
    ///
    /// The engine does the rest of the bookkeeping itself: loads the dataset
    /// from `spec.data_dir`, pulls the loss from the trainer (training
    /// business), derives a timestamped run id + `results/<run_id>` directory
    /// (unless `spec.run_dir` overrides), resolves a random seed when
    /// `spec.seed` is `None` (the resolved seed is recorded in
    /// `engine.json` for repro), and writes the self-contained run header.
    ///
    /// After construction, `engine.run_dir()` and `engine.run_seed()` expose
    /// what the engine chose.
    pub fn from_spec(
        spec: crate::engine::run_spec::RunSpec<
            Box<dyn crate::trainer::TabularStep>,
            Box<dyn crate::trainer::RlStep>,
        >,
    ) -> Result<Self> {
        use crate::engine::run_spec::RunSpec as RS;
        let (config, fitness, trainer, seed, run_dir, tabular_data_dir) = match spec {
            RS::Tabular(s) => (
                s.config,
                s.fitness,
                Box::new(crate::trainer::ModeAdapter::tabular(s.trainer))
                    as Box<dyn crate::trainer::EngineTrainer>,
                s.seed,
                s.run_dir,
                Some(s.data_dir),
            ),
            RS::RL(s) => (
                s.config,
                s.fitness,
                Box::new(crate::trainer::ModeAdapter::rl(s.trainer))
                    as Box<dyn crate::trainer::EngineTrainer>,
                s.seed,
                s.run_dir,
                None,
            ),
        };
        // The spec variant DECIDES the mode; the config's `mode` field is
        // DERIVED (was: user-declared via `set_run_mode(..)` and validated —
        // deleted, the derivation makes disagreement unrepresentable). The
        // spec's mechanics (data_dir vs env) are the single source of truth.
        let derived_mode = if tabular_data_dir.is_some() {
            crate::engine::config::RunMode::Tabular
        } else {
            crate::engine::config::RunMode::Rl
        };
        let mut config = config;
        config.mode = derived_mode;
        // RL mode REQUIRES a reported fitness: there is no dataset, so a
        // (pred, target) scorer has nothing to score. Fail before the run
        // starts, not three steps in.
        if tabular_data_dir.is_none() && fitness.is_computed() {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                "RL spec requires Fitness::reported(direction, label) — a Computed (pred, target) scorer has no dataset to score against. The trainer must supply the fitness value in StepReport.".to_string(),
            )
            .into());
        }
        // Catch-up toggles are RL-effective only: catch-up exists so a net is
        // comparable to the population, and in Tabular that comparability is
        // sacred (one shared data stream — starting mid-stream silently skips
        // training rows). Tabular runs ALWAYS catch up; the flags are ignored
        // there (documented on the setters) rather than erroring — the
        // DEFAULTS themselves (mutation off, crossover on) must not make every
        // tabular config fail construction.
        //
        // Data layout resolution: a directory with BOTH `train/` and `test/`
        // subdirs (each the usual inputs+targets shape) is the EXPLICIT
        // two-dataset layout — `resolve_train_test_datasets` reads them and
        // the engine guarantees the test rows never touch training. Anything
        // else falls back to the single-pool `resolve_dataset` (seeded
        // internal split). Dims are validated against the TRAIN side (the
        // side the networks actually fit).
        let (dataset, explicit_test_rows) = match &tabular_data_dir {
            Some(data_dir) => {
                let train_dir = data_dir.join("train");
                let test_dir = data_dir.join("test");
                if train_dir.exists() && test_dir.exists() {
                    let (train, test) =
                        crate::utils::tabular_data::resolve_train_test_datasets(data_dir)?;
                    let test_len = test.len();
                    let combined = train.concat(&test)?.to_device(config.device())?;
                    (Some(combined), Some(test_len))
                } else {
                    (
                        Some(
                            crate::utils::tabular_data::resolve_dataset(data_dir)?
                                .to_device(config.device())?,
                        ),
                        None,
                    )
                }
            }
            None => (None, None),
        };
        // Random seed when omitted: time ^ fastrand (recorded in engine.json).
        let run_seed = seed.unwrap_or_else(|| {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            t ^ fastrand::u64(..)
        });
        let run_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
            .to_string();
        let run_dir = run_dir.unwrap_or_else(|| std::path::Path::new("results").join(&run_id));
        // Infer dims from data_dir when TopologyOptions leaves them unset
        // (Tabular only). input_dim/output_dim are now Option<usize> — None
        // means "fill from dataset at run start", Some(v) means user set it
        // (and the engine validates v against the dataset below). RL specs
        // have no dataset: dims MUST be set explicitly by the user.
        let mut topology_options = config.topology_options;
        let mut topology_errors: Vec<String> = Vec::new();
        if let Some(dataset) = &dataset {
            let inferred_input_dim = dataset.inputs.shape()[1] as usize;
            let inferred_output_dim = dataset.targets.shape()[1] as usize;
            if let Some(user_in) = topology_options.input_dim {
                if user_in != inferred_input_dim {
                    topology_errors.push(format!(
                        "config.topology_options.input_dim = {user_in} but data_dir has {inferred_input_dim} features — set to None to infer, or fix either side"
                    ));
                }
            } else {
                topology_options.input_dim = Some(inferred_input_dim);
            }
            if let Some(user_out) = topology_options.output_dim {
                if user_out != inferred_output_dim {
                    topology_errors.push(format!(
                        "config.topology_options.output_dim = {user_out} but data_dir has {inferred_output_dim} target columns — set to None to infer, or fix either side"
                    ));
                }
            } else {
                topology_options.output_dim = Some(inferred_output_dim);
            }
        } else {
            if topology_options.input_dim.is_none() {
                topology_errors.push(
                    "RL spec has no dataset — set set_topology_input_dim(..) explicitly"
                        .to_string(),
                );
            }
            if topology_options.output_dim.is_none() {
                topology_errors.push(
                    "RL spec has no dataset — set set_topology_output_dim(..) explicitly"
                        .to_string(),
                );
            }
        }
        if !topology_errors.is_empty() {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                topology_errors.join("; "),
            )
            .into());
        }
        let metrics = config.metrics.clone();
        // Challenge contract: `challenge_prob` is validated at the TRIGGER,
        // not here — a fired trigger is handed to the trainer as
        // `ctx.challenged`, and the trainer owns the response (RL forces an
        // action, tabular jitters its batch). A trainer that ignores the flag
        // keeps the knob at its default 0.
        let train_eval_split_ratio = 0.2f32;
        let held_out_eval_rows = 256usize;
        let default_batch_size = 16usize;
        // The trainer owns batch geometry (Tabular only — RL has no stream).
        // Resolve it BEFORE the header so engine.json records the stream
        // shape the run actually used. `None` = no stream at all (RL), which
        // stays `None` in the record — a tabular default here would read as a
        // fact about a run that never draws a batch.
        let (batch_size, eval_batch_size) = match trainer.stream_shape() {
            Some(shape) => (Some(shape.batch_size), Some(shape.eval_batch_size)),
            None => (None, None),
        };

        let (header_input_dim, header_output_dim) = match &dataset {
            Some(d) => (d.inputs.shape()[1] as usize, d.targets.shape()[1] as usize),
            None => (
                topology_options.input_dim.unwrap_or(0),
                topology_options.output_dim.unwrap_or(0),
            ),
        };
        let header = RunHeader::from_race_options_at(
            RunConfig {
                run_id: run_id.clone(),
                engine_mode: Some(
                    if tabular_data_dir.is_some() {
                        "tabular"
                    } else {
                        "rl"
                    }
                    .to_string(),
                ),
                run_seed,
                fitness_label: FitnessLabel(fitness.label().to_string()),
                fitness_direction: fitness.direction(),
                input_dim: header_input_dim,
                output_dim: header_output_dim,
                topology_options,
                hidden_dim_pool: config.hidden_dim_pool.clone().unwrap_or(4..=8),
                hidden_dim_stride: config.hidden_dim_stride,
                combine_op_pool: config.combine_op_pool.clone(),
                activation_pool: config.activation_pool.clone(),
                standardize_op_pool: config.standardize_op_pool.clone(),
                informative_metrics: metrics.clone(),
                max_steps: config.max_steps,
                train_eval_split_ratio: Some(train_eval_split_ratio),
                held_out_eval_rows: Some(held_out_eval_rows),
                explicit_test_rows,
                culls: 0,
                run_elapsed_secs: 0,
                children_born_at_clock: HashMap::new(),
                config: ConfigSnapshot::from_config(&config, batch_size, eval_batch_size),
                trainer: trainer.describe(),
            },
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
        );
        write_engine_json(&run_dir, &header)?;
        // Settle the log sinks now that the run dir exists: `trace_file` opens
        // the sidecar and flushes the buffered start-up block into it, the
        // default closes that buffer instead. Either way the knob is the dir's
        // only consumer — logging itself never guesses it.
        super::logging::settle_run_dir(&run_dir, config.trace_file);

        // Seed the initial population internally — transparent to the user,
        // like the old generational engine where `run` was the only call.
        // Generated before the engine consumes the config.
        let initial_topologies = super::population::initial_population(&config, run_seed);
        // User run-topologies fill slots FIRST (blueprint-only, fresh weights);
        // the random batch fills the rest and re-rolls any draw that collides
        // with a run topology (same dedupe discipline as the random batch itself).
        let run_topology_count = config.run_topologies.len();
        let initial_topologies = if run_topology_count > 0 {
            let mut mixed = config.run_topologies.clone();
            mixed.extend(initial_topologies);
            mixed.truncate(config.pop_size.max(run_topology_count));
            mixed
        } else {
            initial_topologies
        };

        // Shared batch stream — engine infrastructure (Tabular only). The
        // trainer may shape batch sizes (`Trainer::stream_shape`) but the
        // split ratio is NOT overridable: which rows are held out protects
        // fitness comparability across every net, whatever the recipe.
        // RL mode has no dataset — `ctx.data` is `None` and the trainer is
        // fully responsible for its own experience (the env it drives).
        let batch_stream = match &dataset {
            Some(dataset) => {
                // Two stream layouts (see the resolution above):
                // - explicit `{train, test}` dirs → pools BY CONSTRUCTION:
                //   train pool = every train row, eval+gating = seeded split
                //   of the test rows (no training row can leak into eval).
                // - single pool → the classic seeded shuffle split.
                let split = match explicit_test_rows {
                    Some(test_len) => {
                        PoolSplit::explicit(dataset.len() - test_len, test_len, run_seed)
                    }
                    None => PoolSplit::of(dataset, train_eval_split_ratio, run_seed),
                };
                Some(
                    BatchStream::new(run_seed, batch_size.unwrap_or(default_batch_size), split)
                        .with_eval_batch_size(eval_batch_size.unwrap_or(default_batch_size))
                        .with_held_out_eval_rows(held_out_eval_rows)
                        .with_checkpoint_every(config.checkpoint_every),
                )
            }
            None => None,
        };

        let log_level = config.log_level;
        let meta_ctx = crate::engine::config::RunMetaCtx {
            input_dim: header.input_dim,
            output_dim: header.output_dim,
            batch_size: batch_stream.as_ref().map(|s| s.batch_size()),
            dropout_prob: header.topology_options.dropout_prob,
            fitness_label: header.fitness_label.0.clone(),
            direction: format!("{:?}", header.fitness_direction).to_lowercase(),
            pop_size: config.pop_size,
            run_seed,
        };

        let mut engine = CoreEngine {
            run_dir,
            header,
            dataset,
            explicit_test_rows,
            config: RaceConfig { ..config },
            meta_ctx,
            stream: batch_stream,
            fitness,
            metrics,
            state: RaceState::new(),
            history_csv_buffer: String::new(),
            attempts_csv_buffer: String::new(),
            networks: HashMap::new(),
            optimizers: HashMap::new(),
            rolling_fitness: HashMap::new(),
            rolling_train: HashMap::new(),
            rolling_eval: HashMap::new(),
            started_at_wall: std::time::Instant::now(),
            elapsed_base_secs: 0,
            step_started_at_wall: std::time::Instant::now(),
            culls: 0,
            children_born_at_clock: HashMap::new(),
            step_evolve: StepEvolve::default(),
            step_rl: RlVolume::default(),
            total_challenged_turns: 0,
            total_train_turns: 0,
            expected_challenged_turns: 0.0,
            step_challenged_inputs: 0,
            total_challenged_inputs: 0,
            expected_challenged_inputs: 0.0,
            champions: Vec::new(),
            checkpoints: Vec::new(),
            frozen_crown: std::collections::HashSet::new(),
            pruner_solo_active: false,
            minimal_prev_means: None,
            log_level,
            interrupt_flag: None,
            trainer,
        };

        // Seed the initial population internally — transparent to the user,
        // like the old generational engine where `run` was the only call.
        engine.seed_population_internal(initial_topologies, None)?;

        Ok(engine)
    }

    /// Runtime default for the best-net eval size, for configs that predate
    /// the field (legacy `0` sentinel ⇒ batch_size). No-op for current configs.
    fn apply_eval_defaults(&mut self) {
        // held_out_eval_rows now lives on RunSpec::stream, not RaceConfig.
        // The engine stores the stream directly, so this is a no-op for current
        // code paths — the stream's held_out_eval_rows is read from the engine
        // field where needed.
    }

    /// The run id back to the caller (for logs / tool use).
    pub fn run_id(&self) -> &str {
        &self.header.run_id
    }

    /// The run seed back to the caller.
    pub fn run_seed(&self) -> u64 {
        self.header.run_seed
    }

    /// The current step clock — read from any live net's recorded step. After
    /// the group steps, every live net is at the same step, so any live net
    /// gives the clock. Returns 0 when the population is empty.
    pub fn step_clock(&self) -> usize {
        // The next clock is one past the last clock any live net actually
        // trained — read from the RECORDED clock (`last_metrics.step`), not
        // from the training count (`state.step`). The two differ for every
        // net inserted mid-run: it skips its birth clock, so its count stays
        // one below the clock it reached. A resumed run that keys off the
        // count therefore restarts one clock too early and RE-TRAINS a clock
        // that is already done (observed as duplicate metric rows at the same
        // clock: `['1','2','3','3']`).
        //
        // Max is order-independent, and a freshly caught-up child simply
        // trails (its last catch-up clock is older than the population's), so
        // it never drags the clock backwards.
        self.state
            .live_hashes()
            .iter()
            .filter_map(|h| self.state.net(h))
            .filter_map(|s| s.last_metrics.as_ref().map(|m| m.step))
            .max()
            .map(|last| last + 1)
            .unwrap_or(0)
    }

    /// The live population count.
    pub fn live_count(&self) -> usize {
        self.state.live_count()
    }

    /// Cumulative culls so far (tombstones written + slots freed).
    pub fn cull_count(&self) -> usize {
        self.culls
    }

    /// Cumulative successful insertions (crossover survivors + immigrants).
    /// NOTE: `children_born_at_clock` counts every *generated* child,
    /// including gate-rejected crossover attempts — this accessor counts only
    /// the ones that actually joined. Should equal `cull_count()` (every
    /// insertion pairs with one cull; pop stays at pop_size).
    pub fn insert_count(&self) -> usize {
        self.culls
    }

    /// Expose the underlying population state.
    pub fn state(&self) -> &RaceState {
        &self.state
    }

    /// Load every live net from `nets/` and replay it to its recorded step.
    /// Shared by both resume flavors (the only mode-dependent part is the
    /// trainer + stream the engine already holds). Also reloads the checkpoint
    /// ledger so crossover gates compare children against the SAME historical
    /// bars the original run recorded — without this, a resumed run's gates
    /// start empty and children born before the resume point face no gate.
    pub(crate) fn load_live_frontier(&mut self) -> Result<()> {
        // Replay-relevant knob guard: the smoothing window shapes every
        // ranking decision, so a resumed run must use the SAME window the
        // original recorded — a mismatch would silently re-rank history.
        // (Same contract as pop_size; checked before any replay work.)
        if self.config.smoothing_window != self.header.config.smoothing_window {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: smoothing_window mismatch — run recorded {} but the resume config sets {} — \
                 ranking semantics would change; set set_run_smoothing_window({}) (or omit) to resume",
                self.header.config.smoothing_window,
                self.config.smoothing_window,
                self.header.config.smoothing_window
            )).into());
        }
        // Same contract for the challenge trigger: it re-fires from
        // (run_seed, net_seed, step, challenge_prob), so a resumed prob that
        // differs from the recorded one would replay a DIFFERENT challenge
        // pattern and break bit-exact parity.
        if self.config.challenge_prob != self.header.config.challenge_prob {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: challenge_prob mismatch — run recorded {} but the resume config sets {} — \
                 the challenged-step replay pattern would diverge; set set_run_challenge_prob({}) \
                 (or omit) to resume",
                self.header.config.challenge_prob,
                self.config.challenge_prob,
                self.header.config.challenge_prob
            ))
            .into());
        }
        // The decay SHAPE is replay state too: the trigger re-derives its
        // probability from (prob, shape, step), so a changed shape would
        // re-fire a different challenge pattern. Same contract as above.
        if self.config.challenge_decay != self.header.config.challenge_decay {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: challenge_decay mismatch — run recorded '{}' but the resume config sets '{}' — \
                 the challenged-step replay pattern would diverge; set set_run_challenge_decay(ChallengeDecay::{:?}) \
                 (or omit) to resume",
                self.header.config.challenge_decay,
                self.config.challenge_decay,
                self.header.config.challenge_decay
            ))
            .into());
        }
        let run_dir = self.run_dir.clone();
        let nets_dir = run_dir.join("nets");
        let mut loaded = 0usize;
        let entries = std::fs::read_dir(&nets_dir).map_err(|source| {
            crate::utils::error::EngineError::Io {
                path: nets_dir.display().to_string(),
                source,
            }
        })?;
        for entry in entries {
            let path = entry
                .map_err(|source| crate::utils::error::EngineError::Io {
                    path: nets_dir.display().to_string(),
                    source,
                })?
                .path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let raw = std::fs::read_to_string(&path).map_err(|source| {
                crate::utils::error::EngineError::Io {
                    path: path.display().to_string(),
                    source,
                }
            })?;
            let state = NetState::from_json(&raw)?;
            if !state.is_alive {
                tracing::debug!(
                    "race resume: skipping tombstone net {} (culled)",
                    state.hash
                );
                continue;
            }
            // Ghost guard (defensive): a gate-rejected child once leaked as a
            // live state file whose counters came from its replay window — a
            // net born at step N cannot have trained clocks < N (its birth
            // clock's group step ran before it existed). Such a state is a
            // structural impossibility for a real member; skip it with a
            // warning instead of failing the whole resume on one ghost.
            let entered = state.entered_at_step;
            if entered > 0 {
                if let Some(lm) = state.last_metrics.as_ref() {
                    if entered > lm.step + 1 {
                        tracing::warn!(
                            "race resume: skipping GHOST net {} (entered_at_step {} but last trained clock {} — a gate-rejected provisional child that leaked into nets/)",
                            state.hash,
                            entered,
                            lm.step
                        );
                        continue;
                    }
                }
            }
            let hash = state.hash.clone();
            let step = state.step;
            let child = self.replay_loaded_net(state)?;

            // Insert the replayed net into the live maps. Its rolling buffers
            // are seeded from the replayed trajectory so the smoothed stats are
            // warm on resume (not cold) — ranking resumes where it left off.
            let mut buf = RollingBuffer::new(self.config.smoothing_window);
            let mut train_buf = RollingBuffer::new(self.config.smoothing_window);
            let mut eval_buf = RollingBuffer::new(self.config.smoothing_window);
            if let Some(m) = self.state.net(&hash).and_then(|s| s.last_metrics.as_ref()) {
                buf.push(m.fitness);
                train_buf.push(m.train_loss);
                if let Some(e) = m.eval_loss {
                    eval_buf.push(e);
                }
            }
            self.state.insert(child.state.clone(), None);
            self.networks.insert(child.state.hash.clone(), child.net);
            self.optimizers
                .insert(child.state.hash.clone(), child.optimizer);
            self.rolling_fitness.insert(hash.clone(), buf);
            self.rolling_train.insert(hash.clone(), train_buf);
            self.rolling_eval.insert(hash.clone(), eval_buf);
            loaded += 1;
            info!(
                "race resume: net {} replayed to step {} (parity ok)",
                hash, step
            );
        }
        if loaded != self.config.pop_size {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: expected {} live nets in population (per config.pop_size), but found {} live nets in {}",
                self.config.pop_size, loaded, nets_dir.display()
            )).into());
        }
        info!(
            "race resume: loaded {} nets from {}",
            loaded,
            run_dir.display()
        );
        self.load_checkpoints()?;
        Ok(())
    }

    // ── Population setup ────────────────────────────────────────────────────
    /// Populate the population from finalized topologies (internal — `new`
    /// calls this with the freshly generated initial population; resume
    /// replays from disk instead).
    ///
    /// Each topology is built on the engine's device with its own embedded
    /// seed + dropout_prob (read from `topology.options` by `Network::build`),
    /// and gets a fresh Adam optimizer. The topologies must already be
    /// finalized (the engine calls `Network::build`, which re-validates).
    fn seed_population_internal(
        &mut self,
        topologies: Vec<crate::graph::topology::Topology>,
        initial_fitness: Option<f32>,
    ) -> Result<()> {
        for (ordinal, topo) in topologies.iter().enumerate() {
            let net = Network::build(topo, self.config.device())?;
            // Lineage: founders are labeled `founder-<ordinal>` — a readable
            // "part of the original population" marker, distinct from
            // `random`/`crossover-*` origins children carry.
            let mut state = NetState::new(topo, 0, Some(format!("founder-{ordinal}")))?;
            state.stamp_meta(&net, &self.meta_ctx);
            let hash = state.hash.clone();
            let opt = self.trainer.make_optimizer(&net);
            self.networks.insert(hash.clone(), net);
            self.optimizers.insert(hash.clone(), opt);
            self.rolling_fitness.insert(
                hash.clone(),
                RollingBuffer::new(self.config.smoothing_window),
            );
            self.rolling_train.insert(
                hash.clone(),
                RollingBuffer::new(self.config.smoothing_window),
            );
            self.rolling_eval.insert(
                hash.clone(),
                RollingBuffer::new(self.config.smoothing_window),
            );
            self.state.insert(state, initial_fitness);
        }
        let count = self.state.live_count();
        info!(
            "race init: populated {} live nets into {}",
            count,
            self.run_dir.display()
        );
        Ok(())
    }

    // ── The run loop ────────────────────────────────────────────────────────

    /// Run the step-race loop until a stop criterion fires.
    ///
    /// Each iteration = one step_clock for the whole group. The loop is the
    /// heart of the step-race model: every live net trains on the same batch,
    /// is evaluated on the same held-out batch, records its metrics, then the
    /// evolve rolls cull and replace by smoothed fitness.
    ///
    /// This is the step-race analog of the generational ``Engine::run``: same
    /// ``info``/``debug`` log discipline, same deterministic seed discipline,
    /// same ``no dead air`` rule from AGENTS.md §5 — only the cadence changes
    /// from per-generation to per-step and selection runs as per-step evolve
    /// rolls instead of an end-of-generation sweep.
    ///
    /// Deterministic: two runs with the same `run_seed` + config produce the
    /// same cull/insertion sequence and the same per-step metrics (tested in
    /// `tests::race_determinism`).
    pub fn run(&mut self) -> Result<StopReason> {
        self.apply_eval_defaults();
        // Keep the stream's eval-rotation cadence in lockstep with the gate
        // cadence — a post-construction config edit (tests, tooling) must not
        // desync era numbering between the replay and the gate ledger.
        if let Some(stream) = self.stream.as_mut() {
            stream.set_run_checkpoint_every(self.config.checkpoint_every);
        }
        // Run-start config resolution line: everything that shapes the search,
        // logged once so the log is self-describing (empty pools ⇒ all ops).
        if self.verbose_detail() {
            let ops = self
                .config
                .resolved_crossover_ops()
                .map(|ops| {
                    ops.iter()
                        .map(|o| match o {
                            crate::engine::config::CrossoverOp::OnePoint => "one_point",
                            crate::engine::config::CrossoverOp::Uniform => "uniform",
                        })
                        .collect::<Vec<&str>>()
                        .join(",")
                })
                .unwrap_or_else(|e| format!("INVALID({e})"));
            let acts = if self.config.activation_pool.is_empty() {
                "all".to_string()
            } else {
                self.config.activation_pool.join(",")
            };
            let combines = if self.config.combine_op_pool.is_empty() {
                "default-safe(add,mean,max,min)".to_string()
            } else {
                self.config.combine_op_pool.join(",")
            };
            let std_ops = if self.config.standardize_op_pool.is_empty() {
                "all".to_string()
            } else {
                self.config.standardize_op_pool.join(",")
            };
            info!(
                "search space: crossover_ops=[{}] │ activations=[{}] │ combine=[{}] │ standardize=[{}] │ cull_policy={:?} │ elite_count={}",
                ops,
                acts,
                combines,
                std_ops,
                self.config.crossover_cull_policy,
                self.config.elite_count,
            );
        }
        // `None` mode: one compact start line, then silence until stop.
        if self.verbose_detail() {
            info!(
                "race start: run={} seed={} checkpoint_every={} crossover_rolls={} mutate_rolls={} K={} pop={} pid={} (kill this pid to end the race; Ctrl+C = graceful)",
                self.header.run_id,
                self.header.run_seed,
                self.config.checkpoint_every,
                self.config.crossover_rolls,
                self.config.mutate_rolls,
                self.config.smoothing_window,
                self.config.pop_size,
                std::process::id(),
            );
            // The knobs that silently change what a run MEANS, printed up
            // front so a resumed command can be reconstructed from the log
            // alone (and so a stale binary is visible before it misleads).
            info!(
                "race knobs: gate={:?} cull_policy={:?} retries={} dropout={} freeze_elites={} fresh_immigrants={} elite_save={} worst_save={} probation_steps={}",
                self.config.crossover_gate,
                self.config.crossover_cull_policy,
                self.config.crossover_retries,
                self.header.topology_options.dropout_prob,
                self.config.freeze_elites,
                self.config.mode_specific.mutation_catch_up(),
                self.config.elite_save_topology,
                self.config.worst_save_topology,
                self.config.mutation_probation_steps,
            );
            info!("  {}", build_stamp());
        }

        // ── Graceful Ctrl+C (always on) ─────────────────────────────────
        // One SIGINT sets the flag; the loop abandons the in-flight step at
        // the next boundary and flows through the SAME artifact path as a
        // natural stop (pruner → frontier → champion → guardrail). A second
        // Ctrl+C during shutdown force-kills (ctrlc's default handler is
        // replaced; we re-raise via std::process::exit).
        let interrupted = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let flag = interrupted.clone();
            let first = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let _ = ctrlc::set_handler(move || {
                if first.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    // Second press: the user is done waiting.
                    std::process::exit(130);
                }
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                eprintln!(
                    "\n── Ctrl+C received: shutting down gracefully at the next step boundary (press again to force-kill) …"
                );
            });
        }
        self.interrupt_flag = Some(interrupted);

        loop {
            let clock = self.step_clock();
            self.step_started_at_wall = std::time::Instant::now();
            // Per-step accumulators start empty; the rollup at the END of this
            // step reads them (it is the last line printed for the step).
            self.step_rl = RlVolume::default();
            self.step_challenged_inputs = 0;

            // ── 1. group step: every live net on the same shared batch ──────
            let hashes = self.state.live_hashes();
            if hashes.is_empty() {
                info!("race: population empty — stopping");
                return Ok(StopReason::MaxSteps);
            }
            // 1a. pop-wide phase (RL only): hand the trainer the whole live
            // population ONCE per step, before any per-net step. Pop-level
            // schemes (pop-mean action anchors, distillation) build their
            // shared state here; the default is a no-op. Tabular never calls
            // it — its group sharing is the batch stream itself.
            if self.trainer.is_rl() {
                // Take the trainer OUT of self so the &mut Network borrows
                // (self.networks) and the &mut trainer call don't alias.
                let mut trainer = std::mem::replace(&mut self.trainer, Box::new(NoopPopTrainer));
                // One pass over the map — multiple get_mut borrows stored at
                // once don't compile (NLL), but iter_mut does.
                let live: std::collections::HashSet<&String> = hashes.iter().collect();
                let mut nets: Vec<(String, &mut Network)> = self
                    .networks
                    .iter_mut()
                    .filter(|(h, _)| live.contains(h))
                    .map(|(h, net)| (h.clone(), net))
                    .collect();
                trainer.pop_phase(&mut nets, clock);
                self.trainer = trainer;
            }
            for hash in &hashes {
                // Mid-step Ctrl+C: abandon the step immediately (user choice:
                // "current step just gets wiped out, faster easier"). Nothing
                // of this step is recorded — history.csv, checkpoints and the
                // metrics buffer stay consistent at step-1; the shutdown path
                // below exports from there.
                if let Some(flag) = &self.interrupt_flag {
                    if flag.load(std::sync::atomic::Ordering::SeqCst) {
                        info!(
                            "step {} │ interrupted mid-step — abandoning (no partial records), shutting down gracefully",
                            clock
                        );
                        // Skip to the post-race sequence with Interrupted.
                        // The pruner/champion/frontier path treats this like
                        // any stop reason at clock-1's state.
                        return self.finish_race(StopReason::Interrupted, clock.saturating_sub(1));
                    }
                }
                // A trainer error DURING an interrupt is the interrupt, not a
                // fault: Ctrl+C signals the whole process group, so a bridge
                // child (or anything downstream of one) typically dies with
                // the same SIGINT before the flag check above can fire. That
                // surfaced as `race error: match failed: runner.py exited…` —
                // the error path racing the graceful path to the exit. Route
                // it into the SAME graceful shutdown: abandon the step, run
                // finish_race(Interrupted), artifacts still land on disk.
                if let Err(e) = self.step_one_net(hash, clock) {
                    let flagged = self
                        .interrupt_flag
                        .as_ref()
                        .is_some_and(|f| f.load(std::sync::atomic::Ordering::SeqCst));
                    if flagged {
                        info!(
                            "step {} │ interrupted mid-step (child died with the signal: {e}) — abandoning, shutting down gracefully",
                            clock
                        );
                        return self.finish_race(StopReason::Interrupted, clock.saturating_sub(1));
                    }
                    return Err(e);
                }
            }
            self.append_metrics_csv(clock)?;

            // ── 1b. stepped log: who trained this clock, who didn't ─────────
            // Crossover/mutation rolls get one line each; the GROUP step (the
            // thing every net does) got only the rollup stats. This line names
            // the participation: `stepped 5/6` plus the frozen nets that were
            // skipped — with act-and-measure freeze the skips are the frozen
            // elites, so a reader can see exactly who did not learn.
            if self.log_level == crate::engine::config::LogLevel::Summ
                && self.config.freeze_elites
                && !self.frozen_crown.is_empty()
            {
                let mut skipped: Vec<String> = self
                    .frozen_crown
                    .iter()
                    .map(|h| h[..8.min(h.len())].to_string())
                    .collect();
                skipped.sort();
                info!(
                    "step {} │ stepped {}/{} │ frozen (act+measure, no update): {}",
                    clock,
                    self.state.live_count() - self.frozen_crown.len(),
                    self.state.live_count(),
                    skipped.join(" "),
                );
            }

            // ── 2. record checkpoint (every `checkpoint_every` steps) ───────
            if clock > 0 && clock % self.config.checkpoint_every == 0 {
                let mean = self.population_mean_smoothed_fitness();
                // Surprise exam: score every live net on this era's gating-pool
                // batch — rows never touched by training or per-step eval. This
                // is diagnostic (recorded in the ledger), not ranking input, so
                // it cannot perturb selection or the replay contract.
                let era = (clock / self.config.checkpoint_every) as u64;
                let exam_mean = self.run_checkpoint_exam(era)?;
                self.checkpoints.push(Checkpoint {
                    step: clock,
                    pop_mean_fitness: mean,
                    exam_mean_fitness: exam_mean,
                });
                if let Err(e) = self.write_checkpoints() {
                    tracing::warn!("checkpoint ledger write failed: {e}");
                }
                if let Err(e) = self.write_live_frontier_states() {
                    tracing::warn!("checkpoint live states write failed: {e}");
                }
                // Elite weight snapshot (hard-kill durability): the frontier
                // states above already carry topology + step counters every
                // checkpoint; this adds the WEIGHTS of the top-k so a
                // `kill -9` loses at most one checkpoint interval of them.
                if self.config.elite_checkpoint_weights {
                    if let Err(e) = self.write_checkpoint_elite_safetensors() {
                        tracing::warn!("checkpoint elite snapshot failed: {e}");
                    }
                }
                // Flush metrics at the checkpoint too: an interrupted run then
                // keeps the per-step history up to its last checkpoint instead
                // of losing the whole in-memory buffer.
                if let Err(e) = self.flush_csv_exports() {
                    tracing::warn!("checkpoint CSV flush failed: {e}");
                }
                if self.verbose_detail() {
                    let arrow = self.fitness.direction().arrow();
                    // The bar the gate will actually compare against: for Soft
                    // it is the mean of ALL checkpoint means (a historical
                    // average, typically BELOW a rising `pop_mean_fitness`),
                    // for Hard the newest checkpoint's mean. Shown here so the
                    // two numbers are never mistaken for each other.
                    let bar_txt = match self.current_gate_bar(clock) {
                        Some(bar) => format!(
                            "gate bar({}) {} {:.4} │ ",
                            match self.config.crossover_gate {
                                crate::engine::config::CrossoverGate::Hard => "hard",
                                crate::engine::config::CrossoverGate::Soft => "soft",
                            },
                            arrow,
                            bar,
                        ),
                        None => String::new(),
                    };
                    // RL has no exam batch — `—`, like the eval_loss column,
                    // instead of a fake 0.0000.
                    let exam_txt = if self.exam_available() {
                        format!("{} {:.4}", arrow, exam_mean)
                    } else {
                        "—".to_string()
                    };
                    // The challenge knob's CURRENT effective probability —
                    // it decays linearly to 0 at the run's step budget (see
                    // `challenge_fires`), so each checkpoint line shows the
                    // reader where in the decay they are.
                    let challenge_txt = if self.config.challenge_prob > 0.0 {
                        format!(
                            " │ challenge_prob {:.3}",
                            self.effective_challenge_prob(clock)
                        )
                    } else {
                        String::new()
                    };
                    info!(
                        "step {} │ checkpoint │ pop_mean_fitness {} {:.4} │ {}exam_mean_fitness {} (ledger: {} entries){}",
                        clock,
                        arrow,
                        mean,
                        bar_txt,
                        exam_txt,
                        self.checkpoints.len(),
                        challenge_txt,
                    );
                }
            }

            // ── 3. evolve — always, two independent roll groups ─────────────
            // ONE log line per roll, logged here from the loop — outcome
            // details travel up from the evolve helpers as plain strings.
            // 15 rolls ⇒ 15 lines, each starting with its group name.
            let mut cross_fired = 0usize;
            let mut cross_survived = 0usize;
            let mut cross_discarded = 0usize;
            let mut mutate_fired = 0usize;
            let culls_before = self.culls;
            let cross_total = self.config.crossover_rolls;
            let mutate_total = self.config.mutate_rolls;
            // 3a. crossover rolls: checkpoint-gated children.
            for roll in 0..cross_total {
                if self.state.live_count() < 2 {
                    info!(
                        "step {} │ crossover {}/{} skipped (pop < 2)",
                        clock,
                        roll + 1,
                        cross_total,
                    );
                    break; // not enough population to evolve
                }
                if fastrand::f32() >= self.config.crossover_prob {
                    info!(
                        "step {} │ crossover {}/{} did not fire (p={:.2})",
                        clock,
                        roll + 1,
                        cross_total,
                        self.config.crossover_prob,
                    );
                    continue;
                }
                cross_fired += 1;
                // cx_retry_full: on a gate rejection, retry the FULL attempt
                // (fresh parents, fresh generate + gate replay).
                // `crossover_retries` is the TOTAL attempts budget per roll:
                // retries=2 ⇒ at most 2 generate+gate tries, then the roll is
                // spent. (1 = no retries — single attempt, pass or no-op.)
                let max_attempts = self.config.crossover_retries.max(1);
                let mut attempt = 0usize;
                let outcome = loop {
                    attempt += 1;
                    let out = self.evolve_crossover_child(clock, roll, attempt)?;
                    if out.survived {
                        cross_survived += 1;
                        break out;
                    }
                    if attempt >= max_attempts {
                        cross_discarded += 1;
                        break out;
                    }
                    // retry — the prior attempt's detail is superseded
                    let _ = &out;
                };
                info!(
                    "step {} │ crossover {}/{} ({} attempt(s)): {} │ pop now {}",
                    clock,
                    roll + 1,
                    cross_total,
                    attempt,
                    outcome.detail,
                    self.state.live_count(),
                );
            }
            // 3b. mutation rolls: random immigrants (no gate).
            for roll in 0..mutate_total {
                if self.state.live_count() == 0 {
                    info!(
                        "step {} │ mutation {}/{} skipped (pop empty)",
                        clock,
                        roll + 1,
                        mutate_total,
                    );
                    break;
                }
                if fastrand::f32() >= self.config.mutate_prob {
                    info!(
                        "step {} │ mutation {}/{} did not fire (p={:.2})",
                        clock,
                        roll + 1,
                        mutate_total,
                        self.config.mutate_prob,
                    );
                    continue;
                }
                mutate_fired += 1;
                let detail = self.evolve_random_immigrant(clock, roll)?;
                info!(
                    "step {} │ mutation {}/{}: {} │ pop now {}",
                    clock,
                    roll + 1,
                    mutate_total,
                    detail,
                    self.state.live_count(),
                );
            }
            // Evolve-phase summary for the `Minimal` table (rendered as part
            // of the NEXT step's frame, so the reader sees "what happened last
            // step" in the same table that shows the new stats). Reuses the
            // roll counts already computed above — no extra instrumentation.
            self.step_evolve = StepEvolve {
                culls: self.culls - culls_before,
                inserts: cross_survived + mutate_fired,
                cross_fired,
                cross_survived,
                cross_discarded,
                mutate_fired,
            };

            // ── 3c. per-step rollup — LAST line of the step ─────────────────
            // Ordered train → checkpoint → evolve → rollup so the summary
            // always sits at the bottom of the terminal: no scrolling back to
            // find "how is this run doing", and `took` covers the whole step.
            // `Minimal` renders its framed table here instead. `None` mode
            // stays silent per step.
            if self.log_level != crate::engine::config::LogLevel::None {
                self.log_step_rollup(clock);
            }

            // ── 4. stop criteria ────────────────────────────────────────────
            if let Some(reason) = self.check_stop(clock) {
                return self.finish_race(reason, clock);
            }
        }
    }

    /// The shared post-race sequence — pruner transition, champion artifacts,
    /// stop summary, frontier write, metrics flush. Every stop reason flows
    /// through here (natural bars AND Ctrl+C), so an interrupted run leaves
    /// exactly the artifact trail a natural one does.
    fn finish_race(&mut self, reason: StopReason, clock: usize) -> Result<StopReason> {
        // Post-race pruner (pop_pruner): the stop reason becomes a
        // TRANSITION, not an exit — cull everything except the top
        // `elite_count` nets (min 1) and keep training them solo for
        // `pruner.steps` more steps. Evolution, stop criteria and per-step
        // challenges are phase-locked OFF: they are race-phase concerns and
        // the surviving nets race no one.
        if let Some(pruner) = self.config.pop_pruner {
            let reason = self.run_pruner_phase(reason, clock, pruner)?;
            return Ok(reason);
        }
        // The champion markdown dump is an ARTIFACT, not a log line —
        // it must land on disk at every log level, even `None`.
        self.write_champion_markdown()?;
        self.write_champion_safetensors()?;
        self.write_worst_artifacts()?;
        // `None` mode stays silent until the caller's final stop print.
        if self.verbose_detail() {
            info!("── stop ──");
            info!("  race stop: {:?} at step {}", reason, clock);
            self.log_stop_summary(clock)?;
        }
        self.write_live_frontier_states()?;
        self.flush_csv_exports()?;
        self.persist_run_counters();
        Ok(reason)
    }

    /// Stamp the run-level counters (`culls`, accumulated wall time,
    /// `children_born_at_clock`) into `engine.json` so a resume restores
    /// them instead of starting fresh — a resumed run continues the ORIGINAL
    /// cull budget, wall-clock age, and child-seed disambiguation state.
    /// Best-effort: a write failure is a warning, not a stop failure (the
    /// frontier and ledger are already safely on disk at this point).
    pub(crate) fn persist_run_counters(&self) {
        let mut header = self.header.clone();
        header.culls = self.culls;
        header.run_elapsed_secs = self.elapsed_base_secs + self.started_at_wall.elapsed().as_secs();
        header.children_born_at_clock = self.children_born_at_clock.clone();
        if let Err(e) = crate::state::write_engine_json(&self.run_dir, &header) {
            tracing::warn!("engine.json counter persistence failed: {e}");
        }
    }

    // ── Smoothed fitness ────────────────────────────────────────────────────

    /// Per-net rolling-mean fitness over the live population, in
    /// `live_hashes()` order — the shared input for the checkpoint ledger,
    /// cull/insert ranking, and `RaceSnapshot`.
    pub(crate) fn smoothed_fitness_values(&self) -> Vec<f32> {
        // Only nets with a non-empty rolling buffer get a smoothed value. A
        // freshly caught-up child's buffer is populated during catch-up
        // (replayed fitness for steps 0..clock), so it IS included immediately
        // after insertion, not after a group step. Nets with empty buffers
        // (shouldn't exist in normal operation) are excluded so phantom 0.0
        // values don't drag the population mean down.
        self.state
            .live_hashes()
            .iter()
            .filter(|h| {
                self.rolling_fitness
                    .get(*h)
                    .map(|b| b.iter().count() > 0)
                    .unwrap_or(false)
            })
            .map(|h| rolling_mean(self.rolling_fitness.get(h).unwrap()))
            .collect()
    }

    /// The net's smoothed fitness (rolling mean over the last
    /// `smoothing_window` step fitnesses — the same value every ranking
    /// decision reads). `None` = no recorded fitness yet (never stepped, or
    /// the buffer was empty). This is what the final-elites line prints; the
    /// guardrail's "race smoothed" note should carry THIS number, not the
    /// last raw step's fitness.
    pub fn smoothed_fitness_of(&self, hash: &str) -> Option<f32> {
        self.rolling_fitness.get(hash).map(rolling_mean)
    }

    /// The post-race holdout guardrail: reload the champion's TRAINED
    /// weights from this run's exports and play fresh unseen games through
    /// the run's OWN trainer (`holdout_score` — one game per call, same
    /// units as the reported fitness). No scorer object to wire: the engine
    /// owns the trainer, so `engine.guardrail(device)` is the whole call.
    /// How many games it plays is the run's `set_elite_guardrail_matches`
    /// (default 16) — one knob, set where the rest of the run is configured.
    ///
    /// `None` = no champion was ever exported, the champion could not be
    /// reloaded (missing/unloadable weights — the verdict would be about a
    /// stranger), or every game failed; each logs a warning.
    pub fn guardrail(
        &mut self,
        device: flodl::Device,
    ) -> Option<crate::engine::guardrail::GuardrailVerdict> {
        let champion = self.champions.first()?;
        let matches = self.config.guardrail_matches;
        let race_smoothed = self.smoothed_fitness_of(champion);
        crate::engine::guardrail::score_with(
            &self.run_dir,
            champion,
            race_smoothed,
            matches,
            device,
            &mut self.trainer,
            |t, net, game_i| crate::trainer::EngineTrainer::holdout_score(&mut **t, net, game_i),
        )
    }

    /// The elite set: the top `config.elite_count` live nets by smoothed
    /// fitness (direction-aware). Protected from ALL culls — crossover (any
    /// policy) and mutation alike. Always leaves at least one cullable net:
    /// the effective guard size is `min(elite_count, live − 1)`.
    pub(crate) fn elite_hashes(&self) -> Vec<String> {
        let k = self
            .config
            .elite_count
            .min(self.state.live_count().saturating_sub(1));
        if k == 0 {
            return Vec::new();
        }
        let direction = self.fitness.direction();
        let mut ranked: Vec<(String, f32)> = self
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| {
                self.rolling_fitness
                    .get(h)
                    .filter(|b| !b.is_empty())
                    .map(|b| (h.clone(), rolling_mean(b)))
            })
            .collect();
        ranked.sort_by(|a, b| direction.cmp(b.1, a.1));
        ranked.into_iter().take(k).map(|(h, _)| h).collect()
    }

    /// The net's 1-based rank among ALL live nets (not just the elite seats)
    /// by smoothed fitness — the same direction-aware sort `elite_hashes`
    /// uses, so `rank_position(h) <= elite_count` ⇔ `h` is elite. `None` when
    /// the hash is unknown or has no fitness yet. Used by the dethrone log
    /// (was-rank → out) and any future rank-delta reporting.
    pub(crate) fn rank_position(&self, hash: &str) -> Option<usize> {
        let direction = self.fitness.direction();
        let mut ranked: Vec<(String, f32)> = self
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| {
                self.rolling_fitness
                    .get(h)
                    .filter(|b| !b.is_empty())
                    .map(|b| (h.clone(), rolling_mean(b)))
            })
            .collect();
        ranked.sort_by(|a, b| direction.cmp(b.1, a.1));
        ranked.iter().position(|(h, _)| h == hash).map(|p| p + 1)
    }

    // ── Selection helpers ───────────────────────────────────────────────────

    /// The live net hash with the best smoothed fitness (highest for Maximize,
    /// lowest for Minimize). Test-only for now: the random-topology fallback
    /// replaced the clone-fittest path, and culling uses `worst_nets`. Keep
    /// for the determinism tests (and any future elite-preservation policy).
    #[cfg(test)]
    pub(crate) fn fittest_net_hash(
        &self,
    ) -> std::result::Result<String, crate::utils::error::EngineError> {
        let hashes = self.state.live_hashes();
        if hashes.is_empty() {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                "race: no live nets to select fittest from".into(),
            ));
        }
        let direction = self.fitness.direction();
        let mut best_hash = hashes[0].clone();
        let mut best_val = rolling_mean(self.rolling_fitness.get(&best_hash).unwrap());
        for hash in &hashes[1..] {
            let val = rolling_mean(self.rolling_fitness.get(hash).unwrap());
            if direction.is_better(val, best_val) {
                best_val = val;
                best_hash = hash.clone();
            }
        }
        Ok(best_hash)
    }

    /// The worst live net hashes by smoothed fitness (opposite of fittest).
    /// Uses the live smoothed fitness; only cullable nets are candidates.
    pub(crate) fn worst_nets_by_smoothed_fitness(&self, count: usize) -> Result<Vec<String>> {
        let hashes = self.state.live_hashes();
        if hashes.is_empty() {
            return Ok(Vec::new());
        }
        let direction = self.fitness.direction();
        // The elite guard applies HERE too: this selector feeds the
        // crossover-Worst victim (and the mutation fallback), and an elite
        // must never be a victim under either. (Its comment used to claim
        // this exclusion while the code didn't do it.)
        let elite = self.elite_hashes();
        let mut scored: Vec<(String, f32)> = hashes
            .iter()
            // Nets with an empty rolling buffer (never trained at this
            // clock) are not cullable — no verdict yet.
            .filter(|h| {
                self.rolling_fitness
                    .get(*h)
                    .map(|b| b.iter().count() > 0)
                    .unwrap_or(false)
            })
            .filter(|h| !elite.contains(*h))
            .map(|h| {
                (
                    h.clone(),
                    rolling_mean(self.rolling_fitness.get(h).unwrap()),
                )
            })
            .collect();
        // Sort worst-first: `Direction::cmp(a, b)` orders ascending for
        // Maximize (smallest = worst first) and descending for Minimize
        // (largest = worst first). `take(count)` then culls the worst.
        scored.sort_by(|a, b| direction.cmp(a.1, b.1));
        Ok(scored.into_iter().map(|(h, _)| h).take(count).collect())
    }

    // ── Shared batch materialization ────────────────────────────────────────

    // ── Instrumentals ───────────────────────────────────────────────────────

    /// Write the NetState file for every currently active, live net.
    pub(crate) fn write_live_frontier_states(&self) -> Result<()> {
        let live = self.state.live_hashes();
        for h in &live {
            if let Some(state) = self.state.net(h) {
                write_net_state(&self.run_dir, state)?;
            }
        }
        Ok(())
    }

    /// Append a step's metrics for all live nets to the history buffer.
    /// `history.csv` carries ONLY these rows — evolution events live in their
    /// own `attempts.csv` (`record_attempt`), so neither schema needs the
    /// other's columns padded out.
    pub(crate) fn append_metrics_csv(&mut self, step: usize) -> Result<()> {
        if !self.config.csv_export {
            return Ok(());
        }
        for hash in &self.state.live_hashes() {
            if let Some(net_state) = self.state.net(hash) {
                if let Some(m) = &net_state.last_metrics {
                    let origin = net_state.created_from.clone().unwrap_or_default();
                    let mut row = format!(
                        "{},{},{},{},{},{},{},{}",
                        step,
                        hash,
                        net_state.net_seed,
                        csv_field(&origin),
                        net_state.entered_at_step,
                        m.train_loss,
                        m.eval_loss
                            .map(|val| val.to_string())
                            .unwrap_or_else(|| "nan".to_string()),
                        m.fitness
                    );
                    for &val in &m.informative {
                        row.push(',');
                        row.push_str(&val.to_string());
                    }
                    row.push('\n');
                    self.history_csv_buffer.push_str(&row);
                }
            }
        }
        Ok(())
    }

    /// Flush both buffered CSV exports — `history.csv` (per-step metric rows)
    /// and `attempts.csv` (evolution events). Each file has its own header, so
    /// neither schema pads out the other's columns. Same cadence: checkpoints
    /// + stop, gated by `csv_export`.
    pub(crate) fn flush_csv_exports(&mut self) -> Result<()> {
        if !self.config.csv_export {
            return Ok(());
        }
        // `net_seed` completes the individual identity: (hash, net_seed)
        // uniquely distinguishes re-born individuals that share a topology
        // hash across eras.
        let mut history_header =
            "step,hash,net_seed,origin,entered_at_step,train_loss,eval_loss,fitness".to_string();
        for m in &self.metrics {
            history_header.push(',');
            history_header.push_str(m.label());
        }
        let attempts_header = "step,branch,attempt,outcome,child_hash,child_net_seed,child_origin,gate_index,child_fitness,bar,victim,victim_net_seed,pop_size";
        let history_body = std::mem::take(&mut self.history_csv_buffer);
        let attempts_body = std::mem::take(&mut self.attempts_csv_buffer);
        self.append_csv_file("history.csv", &history_header, &history_body)?;
        self.append_csv_file("attempts.csv", attempts_header, &attempts_body)
    }

    /// Append `body` to `<run_dir>/<file>`, writing `header` first when the
    /// file does not exist yet. A no-op for an empty body.
    fn append_csv_file(&self, file: &str, header: &str, body: &str) -> Result<()> {
        if body.is_empty() {
            return Ok(());
        }
        let path = self.run_dir.join(file);
        let exists = path.exists();

        use std::fs::OpenOptions;
        use std::io::Write;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                crate::utils::error::EngineError::Io {
                    path: parent.display().to_string(),
                    source,
                }
            })?;
        }

        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            })?;

        if !exists {
            let mut h = header.to_string();
            h.push('\n');
            f.write_all(h.as_bytes())
                .map_err(|source| crate::utils::error::EngineError::Io {
                    path: path.display().to_string(),
                    source,
                })?;
        }

        f.write_all(body.as_bytes())
            .map_err(|source| crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            })?;
        Ok(())
    }

    /// Rebuild the run header from the current config and re-write `engine.json`.
    ///
    /// This is an instrumental for later iters, not a required path for Item 4.
    /// Today the header is written once at ``CoreEngine::from_spec`` and the run config
    /// is not mutated mid-run, so this method is a no-op in the current layout.
    /// It exists so a future iter can re-serialize the header if/when it adds a
    /// mid-run config change (for example a live ``max_steps`` or
    /// ``stagnation_window`` adjustment) without rebuilding the whole engine.
    pub fn refresh_header(&mut self) -> Result<()> {
        let cfg = RunConfig {
            run_id: self.header.run_id.clone(),
            engine_mode: self.header.engine_mode.clone(),
            run_seed: self.header.run_seed,
            fitness_label: self.header.fitness_label.clone(),
            fitness_direction: self.header.fitness_direction,
            input_dim: self.header.input_dim,
            output_dim: self.header.output_dim,
            topology_options: self.header.topology_options,
            hidden_dim_pool: self.config.hidden_dim_pool.clone().unwrap_or(4..=8),
            hidden_dim_stride: self.config.hidden_dim_stride,
            combine_op_pool: self.config.combine_op_pool.clone(),
            activation_pool: self.config.activation_pool.clone(),
            standardize_op_pool: self.config.standardize_op_pool.clone(),
            informative_metrics: self.metrics.clone(),
            max_steps: self.config.max_steps,
            train_eval_split_ratio: Some(
                self.stream
                    .as_ref()
                    .map(|s| s.train_eval_split_ratio())
                    .unwrap_or(0.2),
            ),
            held_out_eval_rows: Some(
                self.stream
                    .as_ref()
                    .map(|s| s.held_out_eval_rows())
                    .unwrap_or(256),
            ),
            explicit_test_rows: self.explicit_test_rows,
            culls: self.culls,
            run_elapsed_secs: self.elapsed_base_secs + self.started_at_wall.elapsed().as_secs(),
            children_born_at_clock: self.children_born_at_clock.clone(),
            config: ConfigSnapshot::from_config(
                &self.config,
                self.stream.as_ref().map(|s| s.batch_size()),
                self.stream.as_ref().map(|s| s.eval_batch_size()),
            ),
            trainer: self.trainer.describe(),
        };
        let header = RunHeader::from_race_options(cfg);
        write_engine_json(&self.run_dir, &header)
    }
}

/// Trainer-blob validation, the resume gate.
///
/// Every trainer can describe its own hyperparameters as a JSON blob
/// (`Trainer::describe`) — learning rate, update scheme, matches per step…
/// The run persists that blob in `engine.json` → `"trainer"`. On resume the
/// caller hands the engine a NEWLY built trainer from the current source
/// consts; if any const changed since the run started, the replay parity
/// assert will eventually fail — but with the useless message "reconstruction
/// is not bit-identical (seed/primitive/config drift)" and no hint which knob
/// moved. This function compares the two blobs FIRST and names the drift:
///
/// ```text
/// resume: trainer blob mismatch (consts changed since the run started?) —
///   "matches_per_step": run had 2, incoming has 1
/// ```
///
/// `None` on either side skips the check silently: the trainer may not
/// implement the hook, and legacy `engine.json` files carry no blob. The
/// check is a courtesy diagnostic, not a replay guarantee — a blob that
/// matches does not prove bit-identical replay, it only catches the
/// *declared* drift early. The parity assert remains the hard gate.
pub(crate) fn assert_trainer_blob_matches(
    recorded: &Option<serde_json::Value>,
    incoming_trainer: &dyn crate::trainer::EngineTrainer,
    caller: &str,
) -> Result<()> {
    let Some(recorded) = recorded else {
        return Ok(()); // legacy run or hook-less trainer: nothing to compare
    };
    let Some(incoming) = incoming_trainer.describe() else {
        return Ok(()); // run recorded a blob; the incoming trainer no longer describes itself
    };
    let mut diffs = Vec::new();
    diff_json("", recorded, &incoming, &mut diffs);
    if diffs.is_empty() {
        return Ok(());
    }
    Err(crate::utils::error::EngineError::InvalidOptions(format!(
        "{caller}: trainer blob mismatch (consts changed since the run started?) — {}",
        diffs.join("; ")
    ))
    .into())
}

/// Collect human-readable differences between two JSON blobs, recursing into
/// objects (path-prefixed keys like `match_length.random.min`) and comparing
/// everything else by debug representation. Arrays compare wholesale (a
/// reordered pool is still a difference worth naming).
fn diff_json(path: &str, a: &serde_json::Value, b: &serde_json::Value, out: &mut Vec<String>) {
    match (a, b) {
        (serde_json::Value::Object(am), serde_json::Value::Object(bm)) => {
            for (k, av) in am {
                let key = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                match bm.get(k) {
                    Some(bv) => diff_json(&key, av, bv, out),
                    None => out.push(format!(
                        "\"{key}\": run had {av}, incoming doesn't declare it"
                    )),
                }
            }
            for (k, bv) in bm {
                if !am.contains_key(k) {
                    let key = if path.is_empty() {
                        k.clone()
                    } else {
                        format!("{path}.{k}")
                    };
                    out.push(format!(
                        "\"{key}\": incoming declares {bv}, run didn't record it"
                    ));
                }
            }
        }
        _ => {
            if a != b {
                out.push(format!("\"{path}\": run had {a}, incoming has {b}"));
            }
        }
    }
}

// ── Optimizer construction ───────────────────────────────────────────────────

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    // Direction only appears in test helpers, so it is imported here to keep
    // the lib-only `cargo check` warning-free.
    use crate::engine::fitness::Direction;
    use crate::graph::node::Node;
    use crate::graph::topology::Topology;
    use crate::graph::topology::TopologyOptions;
    use crate::utils::tabular_data::synthetic_classification;
    use flodl::{Device, Variable};

    use crate::engine::config::DEFAULT_CHECKPOINT_EVERY;

    /// The RL volume label is the RL middle column of the per-step rollup: the
    /// train/eval turn split, the mean turns/match, and — when the challenge
    /// fired — the observed ⚔ next to what the trigger promised. A population
    /// that reported nothing (Tabular, or a mis-wired RL trainer) must read as
    /// `—`, never as a silent `0`.
    #[test]
    fn rl_volume_label_formats_and_never_lies() {
        let empty = RlVolume::default();
        assert!(empty.label(0.1).contains('—'), "{}", empty.label(0.1));
        let pop = RlVolume {
            nets: 3,
            matches: 3,
            train_turns: 432,
            eval_turns: 108,
            challenged_turns: 0,
        };
        assert_eq!(
            pop.label(0.0),
            "matches 3 │ train 432 │ eval 108 │ turns/match 180",
            "p_chall cell hidden when the knob is off"
        );
        // Matches with zero turns is still a real report (every match died on
        // turn 0) — it reads as a mean of 0, not as `—`.
        let zero_turns = RlVolume {
            nets: 1,
            matches: 1,
            train_turns: 0,
            eval_turns: 0,
            challenged_turns: 0,
        };
        assert_eq!(
            zero_turns.label(0.0),
            "matches 1 │ train 0 │ eval 0 │ turns/match 0"
        );
        // The ⚔ cell carries the expectation too: p_eff × TRAIN turns, not the
        // whole volume — eval turns are never challengeable.
        let challenged = RlVolume {
            nets: 50,
            matches: 300,
            train_turns: 1331,
            eval_turns: 2662,
            challenged_turns: 75,
        };
        assert_eq!(
            challenged.label(0.1),
            "matches 300 │ train 1331 │ eval 2662 │ p_chall 0.100 │ ⚔ 75 (exp 133) │ turns/match 13"
        );
    }

    fn tiny_dataset() -> crate::utils::tabular_data::Dataset {
        synthetic_classification(64, 2, 2, 7, Device::CPU).unwrap()
    }

    fn tiny_topology(seed: usize) -> Topology {
        let mut topo = Topology::new(
            seed,
            Some(TopologyOptions {
                topology_seed: seed,
                min_hidden_num_nodes: 1,
                max_hidden_num_nodes: 1,
                min_hidden_inputs_per_node: 1,
                max_hidden_inputs_per_node: 1,
                min_hidden_outputs_per_node: 1,
                max_hidden_outputs_per_node: 1,
                input_dim: Some(2),
                output_dim: Some(2),
                dropout_prob: 0.0,
            }),
        );
        topo.nodes.push(Node::new_input(0, 2));
        topo.nodes.push(Node::new_hidden(1, 2, 4));
        topo.nodes.push(Node::new_output(2, 4, 2));
        // Wire the chain: input(2 outs) → hidden(2 in, 4 outs) → output(4 in, 2 outs).
        let port = |node: usize, index: usize| crate::graph::topology::Port { node, index };
        let conn = |from: (usize, usize), to: (usize, usize)| crate::graph::topology::Connection {
            from: port(from.0, from.1),
            to: port(to.0, to.1),
        };
        topo.connections.push(conn((0, 0), (1, 0)));
        topo.connections.push(conn((0, 1), (1, 1)));
        for i in 0..4 {
            topo.connections.push(conn((1, i), (2, i)));
        }
        topo.finalize();
        topo
    }

    fn loss_fn() -> impl Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static {
        |pred, y| {
            let diff = pred.data().sub(&y.data())?;
            let sq = diff.mul(&diff)?;
            Ok(Variable::new(sq.mean()?, true))
        }
    }

    fn fitness() -> Fitness {
        Fitness::new(
            |pred, y| {
                let diff = pred.data().sub(&y.data())?;
                let sq = diff.mul(&diff)?;
                Ok(sq.mean()?.item()? as f32)
            },
            Direction::Minimize,
            "mse",
        )
    }

    /// Persist the tiny dataset so RunSpec.data_dir can point at it. Unique
    /// per call — tests run in parallel threads and must not share a dir.
    fn tiny_dataset_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("gras-test-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        crate::utils::tabular_data::save_dataset(&dir, &tiny_dataset()).unwrap();
        dir
    }

    fn engine(run_dir: &std::path::Path, seed: u64) -> Result<crate::engine::TabularEngine> {
        // pop_size 0 ⇒ new() auto-seeds nothing; each test seeds its own
        // tiny topologies explicitly.
        let data_dir = tiny_dataset_dir("engine");
        let config = RaceConfig {
            pop_size: 0,
            ..RaceConfig::defaults()
        };
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            data_dir,
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(seed),
            Some(run_dir.to_path_buf()),
        ))
    }

    #[test]
    fn champion_hashes_match_ranked_elites() {
        // The regression guard for the guardrail bug: post-race tooling must
        // read THE champions from the engine, and the set must equal the
        // top-elite_count of rank_live — never a byproduct of file ordering.
        let run_dir = std::env::temp_dir().join("gras-champion-hashes");
        let mut eng = engine(&run_dir, 7).unwrap();
        eng.config.elite_count = 2;
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        // Distinct smoothed fitness per net. The test fitness is Minimize
        // (LOWER is better), so the best net is the one with the LOWEST value.
        seed_raw_fitness(&mut eng, 0.5);
        let mut hashes = eng.state.live_hashes();
        hashes.sort();
        assert_eq!(hashes.len(), 3);
        for (i, h) in hashes.iter().enumerate() {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(0.5 + i as f32 * 0.1); // 0.5, 0.6, 0.7 across the three nets
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        }
        eng.record_champions();
        let champs = eng.champion_hashes();
        assert_eq!(champs.len(), 2, "elite_count champions recorded");
        assert_eq!(
            champs[0], hashes[0],
            "lowest smoothed fitness first (Minimize)"
        );
        assert_eq!(champs[1], hashes[1], "second-lowest second");
        assert!(!champs.contains(&hashes[2]), "worst net excluded");
        let _ = std::fs::remove_dir_all(&run_dir);
    }

    #[test]
    fn race_config_defaults_are_conservative() {
        let cfg = RaceConfig::defaults();
        assert_eq!(cfg.pop_size, 5);
        assert_eq!(cfg.checkpoint_every, DEFAULT_CHECKPOINT_EVERY);
        assert_eq!(cfg.crossover_rolls, 1);
        assert_eq!(cfg.mutate_rolls, 1);
        assert_eq!(
            cfg.crossover_gate,
            crate::engine::config::CrossoverGate::Hard
        );
        assert_eq!(
            cfg.crossover_cull_policy,
            crate::engine::config::CrossCullPolicy::Worst
        );
        assert_eq!(cfg.checkpoint_every, DEFAULT_CHECKPOINT_EVERY);
        // batch_size and held_out_eval_rows now live on RunSpec::stream
        // (engine infrastructure), not on RaceConfig — verified by the
        // stream_shape / stream_info contract instead.
        assert_eq!(cfg.max_steps, None, "budgets inactive by default");
        assert!(cfg.hidden_dim_pool.is_some());
        assert_eq!(cfg.pop_size, 5);
    }

    #[test]
    fn select_random_victim_is_deterministic_and_alive() {
        let dir = std::env::temp_dir().join("race_random_victim");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine.config.crossover_cull_policy = crate::engine::config::CrossCullPolicy::Random;
        engine
            .seed_population_internal(
                vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
                Some(0.5),
            )
            .unwrap();
        let live = engine.state.live_hashes();
        // Deterministic: same clock ⇒ same victim.
        let v1 = engine.select_random_victim(3).unwrap();
        let v2 = engine.select_random_victim(3).unwrap();
        assert_eq!(v1, v2, "victim is a pure function of (run_seed, clock)");
        // And the victim is always a live net.
        assert!(live.contains(&v1));
        // Different clock ⇒ (very likely) a different draw is possible; at
        // minimum the draw stays in-bounds, which the contains() assert covers.
        let _ = engine.select_random_victim(4).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn elite_guard_protects_top_k_from_all_culls() {
        let dir = std::env::temp_dir().join("race_elite_guard");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine.config.elite_count = 1;
        engine.config.crossover_cull_policy = crate::engine::config::CrossCullPolicy::Random;
        engine
            .seed_population_internal(
                vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
                Some(0.5),
            )
            .unwrap();
        // Give the nets distinct fitness verdicts: seed buffers with fake
        // smoothed scores so ranking is well-defined. Map hash → score so the
        // expected elite can be derived without assuming hash order.
        let hashes = engine.state.live_hashes();
        let mut scores: Vec<(String, f32)> = Vec::new();
        for (i, h) in hashes.iter().enumerate() {
            let s = 0.1 + i as f32 * 0.3;
            engine.rolling_fitness.get_mut(h).unwrap().push(s);
            scores.push((h.clone(), s));
        }
        // NOTE: the test harness fitness is Minimize (mse) — the fittest net
        // has the LOWEST score, so the elite is the minimum, not the maximum.
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let expected_elite = scores[0].0.clone();
        let elite = engine.elite_hashes();
        assert_eq!(elite.len(), 1, "guard of 1 protects exactly one net");
        assert_eq!(
            elite[0], expected_elite,
            "the fittest net (highest seeded score) is the elite"
        );
        // Random victim draw NEVER lands on the elite.
        for clock in 0..20 {
            let v = engine.select_random_victim(clock).unwrap();
            assert_ne!(v, elite[0], "elite must never be a crossover victim");
        }
        // Mutation roulette also skips the elite: worst-net weight is largest,
        // elite is excluded entirely.
        if let Some(victim) = engine.select_inverse_proportional().unwrap() {
            assert_ne!(victim, elite[0], "elite must never be a mutation victim");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mutation_targets_worst_most_often() {
        // Regression test for the inverted-weights bug: with 3 nets of very
        // different fitness, the fitness-inverse roulette must pick the WORST
        // net most often, never the best.
        let dir = std::env::temp_dir().join("race_mutation_targets_worst");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(
                vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
                Some(0.5),
            )
            .unwrap();
        let hashes = engine.state.live_hashes();
        // Minimize direction: lowest score = fittest (best), highest = worst.
        let mut worst = hashes[0].clone();
        let mut best = hashes[0].clone();
        let (mut min_s, mut max_s) = (f32::INFINITY, f32::NEG_INFINITY);
        for h in &hashes {
            let s = 0.1 + hashes.iter().position(|x| x == h).unwrap() as f32 * 0.3;
            engine.rolling_fitness.get_mut(h).unwrap().push(s);
            if s < min_s {
                min_s = s;
                best = h.clone();
            }
            if s > max_s {
                max_s = s;
                worst = h.clone();
            }
        }
        let mut worst_picks = 0usize;
        let mut best_picks = 0usize;
        for _ in 0..200 {
            if let Some(v) = engine.select_inverse_proportional().unwrap() {
                if v == worst {
                    worst_picks += 1;
                }
                if v == best {
                    best_picks += 1;
                }
            }
        }
        assert!(
            worst_picks > best_picks * 5,
            "worst net picked {} times vs best {} — roulette must favor the worst",
            worst_picks,
            best_picks
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn step_clock_is_zero_when_empty() {
        let dir = std::env::temp_dir().join("race_clock_empty_test");
        let engine = engine(&dir, 42).unwrap();
        assert_eq!(engine.step_clock(), 0);
    }

    #[test]
    fn step_clock_reads_from_live_net_after_populate() {
        let dir = std::env::temp_dir().join("race_clock_pop_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], None)
            .unwrap();
        // After populate, all nets are at step 0.
        assert_eq!(engine.step_clock(), 0);
    }

    /// The clock must come from the RECORDED clock, not the training count.
    /// A net inserted mid-run trains one fewer time than the clock it reached
    /// (it skips its birth clock), so a resumed run keying off `state.step`
    /// would restart one clock early and re-train a finished clock.
    #[test]
    fn step_clock_continues_past_the_recorded_clock_not_the_training_count() {
        let dir = std::env::temp_dir().join("race_clock_joiner_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], None)
            .unwrap();
        let hashes = engine.state.live_hashes();
        let metrics = |step: usize| crate::state::NetMetrics {
            step,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.5,
            informative: vec![],
            frozen: false,
        };
        // Founder-shaped: 4 trainings, last clock 3 (trained every clock).
        for _ in 0..4 {
            engine.state.record_step(&hashes[0], metrics(3)).unwrap();
        }
        // Joiner-shaped: 3 trainings, SAME last clock 3 (skipped one clock).
        for _ in 0..3 {
            engine.state.record_step(&hashes[1], metrics(3)).unwrap();
        }
        // Both shapes agree: the population is done with clock 3, next is 4.
        assert_eq!(engine.step_clock(), 4, "next clock is last recorded + 1");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Anti-devolution guards ──────────────────────────────────────────────

    #[test]
    fn freeze_elite_act_and_measure_no_weight_update() {
        // A (act-and-measure freeze): the frozen elite plays its normal step
        // against the current clock — the trainer RUNS (fresh fitness is
        // recorded into state + buffers, so its standing stays honest) — but
        // through a no-op optimizer, so its weights never change. The
        // observable contract: metrics advance with a FRESH fitness (not the
        // carried one), and the step clock advances like everyone else's.
        let dir = std::env::temp_dir().join("gras_freeze_elite_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 11).unwrap();
        eng.config.freeze_elites = true;
        eng.config.elite_count = 1;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        // Fixture is Minimize: lower smoothed = better (see worst-culls test).
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.5);
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.1);
        assert_eq!(eng.elite_hashes(), vec![hs[1].clone()]);
        // Give the elite a recorded skill state, then step it while frozen.
        let m = crate::state::NetMetrics {
            step: 3,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.5,
            informative: vec![],
            frozen: false,
        };
        eng.state.record_step(&hs[1], m).unwrap();
        let before_len = eng.rolling_fitness.get(&hs[1]).unwrap().len();
        eng.step_one_net(&hs[1], 5).unwrap();
        let after = eng.rolling_fitness.get(&hs[1]).unwrap();
        assert_eq!(after.len(), before_len + 1, "fresh measurement appended");
        let last_metrics = eng
            .state
            .net(&hs[1])
            .unwrap()
            .last_metrics
            .as_ref()
            .unwrap();
        assert_eq!(
            last_metrics.step, 5,
            "frozen net acted at the CURRENT clock (act-and-measure)"
        );
        assert_ne!(
            last_metrics.fitness, 0.5,
            "the fitness is a fresh measurement, not the carried skill"
        );
        assert_eq!(
            eng.state.net(&hs[1]).unwrap().step,
            2,
            "frozen net's step count advanced (it acted)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dethroned_elite_resumes_stepping_without_catch_up() {
        // Dethrone under act-and-measure freeze: the former elite never fell
        // behind the clock (it acted every frozen step), so losing the crown
        // means simply RESUMING normal training — no replay, no gauntlet. Its
        // next step is exactly one clock later than its last frozen step.
        let dir = std::env::temp_dir().join("gras_dethrone_resume_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 13).unwrap();
        eng.config.freeze_elites = true;
        eng.config.elite_count = 1;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        // Net B is the frozen elite (fixture is Minimize).
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.1);
        let m = crate::state::NetMetrics {
            step: 3,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.1,
            informative: vec![],
            frozen: false,
        };
        eng.state.record_step(&hs[1], m).unwrap();
        eng.step_one_net(&hs[1], 5).unwrap();
        assert!(eng.frozen_crown.contains(&hs[1]));
        assert_eq!(
            eng.state
                .net(&hs[1])
                .unwrap()
                .last_metrics
                .as_ref()
                .unwrap()
                .step,
            5,
            "frozen net acted at clock 5 (kept current)"
        );
        // Net A overtakes: the crown moves to A, B is dethroned. (A's value
        // 0.05 is a deep win: after B resumes TRAINING — a real weight update
        // on the test topology — its fresh fitness stays well above A's.)
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.05);
        let m2 = crate::state::NetMetrics {
            step: 6,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.05,
            informative: vec![],
            frozen: false,
        };
        eng.state.record_step(&hs[0], m2).unwrap();
        eng.step_one_net(&hs[0], 7).unwrap();
        assert!(eng.frozen_crown.contains(&hs[0]), "crown moved to A");
        // B's next step is a NORMAL train step at the current clock — one
        // clock past its last acted clock (no catch-up replay span).
        eng.step_one_net(&hs[1], 8).unwrap();
        let b_last = eng
            .state
            .net(&hs[1])
            .unwrap()
            .last_metrics
            .as_ref()
            .unwrap();
        assert_eq!(
            b_last.step, 8,
            "dethroned net stepped once at the current clock (no catch-up)"
        );
        // NOTE: B may legitimately re-crown if its trained fitness beats A's
        // 0.05 — that is correct dethrone semantics, not a bug. Assert the
        // crown is consistent with B's current standing instead of a fixed
        // membership:
        assert_eq!(
            eng.frozen_crown.contains(&hs[1]),
            eng.elite_hashes().contains(&hs[1]),
            "crown membership mirrors the elite set after B's resumed step"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn freeze_crown_logs_on_transition_only() {
        // A: the "crowned" event fires once at first crowning and once on
        // migration — never on stable steps. Asserted via frozen_crown state
        // transitions (the log lines are info-level; the state is the testable
        // contract driving them).
        let dir = std::env::temp_dir().join("gras_freeze_crown_transition");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 23).unwrap();
        eng.config.freeze_elites = true;
        eng.config.elite_count = 1;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        assert!(
            eng.frozen_crown.is_empty(),
            "no crown before any frozen step"
        );
        // Net B (lower smoothed = fitter under Minimize) is the freeze target.
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.1);
        assert_eq!(eng.elite_hashes(), vec![hs[1].clone()]);
        // First frozen step: crown set (info line "champion crowned" fires).
        let m = crate::state::NetMetrics {
            step: 3,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.1,
            informative: vec![],
            frozen: false,
        };
        eng.state.record_step(&hs[1], m).unwrap();
        eng.step_one_net(&hs[1], 5).unwrap();
        assert!(eng.frozen_crown.contains(&hs[1]));
        // Second frozen step, same champion: NO transition — crown unchanged.
        eng.step_one_net(&hs[1], 6).unwrap();
        assert!(eng.frozen_crown.contains(&hs[1]));
        // Crown migration: net A trains past the frozen champion, becomes
        // elite (and thus frozen) — crown moves ("crown moved" line fires).
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.05);
        let m2 = crate::state::NetMetrics {
            step: 6,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.05,
            informative: vec![],
            frozen: false,
        };
        eng.state.record_step(&hs[0], m2).unwrap();
        eng.step_one_net(&hs[0], 7).unwrap();
        assert!(
            eng.frozen_crown.contains(&hs[0]),
            "crown moved to the new champion"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mutation_catch_up_off_is_the_default_no_handicap() {
        // Default (mutation_catch_up = false): a mutation immigrant keeps
        // step 0 / no metrics (no catch-up replay) and gets the NO-catch-up
        // detail line. With the flag on, catch-up runs and the immigrant
        // trains to clock (RL runs only; the fixture IS RL).
        let dir = std::env::temp_dir().join("gras_mutation_catch_up_toggle");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 31).unwrap();
        eng.config.mutate_rolls = 1;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.9);
        // Default: no catch-up. A fresh immigrant is inserted; its step count
        // is 0 and it has no recorded metrics (no replay happened).
        eng.evolve_random_immigrant(5, 0).unwrap();
        assert_eq!(eng.state.live_hashes().len(), 2, "cull+insert keeps size");
        let newcomer = eng
            .state
            .live_hashes()
            .into_iter()
            .find(|h| !hs.contains(h))
            .expect("a new immigrant hash exists");
        let s = eng.state.net(&newcomer).unwrap();
        assert_eq!(
            s.step, 0,
            "catch-up off (default): no replayed training count"
        );
        assert!(
            s.last_metrics.is_none(),
            "catch-up off (default): no replayed metrics"
        );
        // Flip the flag: the next immigrant catches up to the clock.
        eng.config.mode_specific =
            crate::engine::config::ModeConfig::Rl(crate::engine::config::RlConfig {
                mutation_catch_up: true,
                ..crate::engine::config::RlConfig::default()
            });
        let before = eng.state.live_hashes();
        eng.evolve_random_immigrant(7, 0).unwrap();
        let newcomer2 = eng
            .state
            .live_hashes()
            .into_iter()
            .find(|h| !before.contains(h))
            .expect("a second immigrant hash exists");
        let s2 = eng.state.net(&newcomer2).unwrap();
        assert_eq!(s2.step, 7, "catch-up on: immigrant replayed to the clock");
        assert!(s2.last_metrics.is_some(), "catch-up on: replayed metrics");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fittest_net_is_deterministic_given_same_state() {
        let dir = std::env::temp_dir().join("race_fittest_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        // Both have empty rolling buffers ⇒ smoothed fitness 0 ⇒ either is fittest,
        // but the choice is deterministic for the same engine state.
        let h1 = engine.fittest_net_hash().unwrap();
        let h2 = engine.fittest_net_hash().unwrap();
        assert_eq!(h1, h2);
    }
    #[test]
    fn worst_nets_returns_empty_for_empty_pop() {
        let dir = std::env::temp_dir().join("race_worst_empty_test");
        let engine = engine(&dir, 42).unwrap();
        let worst = engine.worst_nets_by_smoothed_fitness(1).unwrap();
        assert!(worst.is_empty());
    }

    #[test]
    fn worst_nets_culls_the_worst_under_both_directions() {
        // Minimize (default fixture): smoothed 0.2 vs 0.8 → worst = 0.8.
        let dir = std::env::temp_dir().join("race_worst_min_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut e = engine(&dir, 5).unwrap();
        e.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = e.state.live_hashes();
        e.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.2);
        e.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.8);
        assert_eq!(
            e.worst_nets_by_smoothed_fitness(1).unwrap()[0],
            hs[1],
            "Minimize: highest loss is worst"
        );

        // Maximize: worst = lowest score.
        let dir = std::env::temp_dir().join("race_worst_max_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut e = engine(&dir, 5).unwrap();
        e.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = e.state.live_hashes();
        e.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.2);
        e.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.8);
        e.fitness = Fitness::new(|_, _| Ok(0.0), Direction::Maximize, "fixture");
        assert_eq!(
            e.worst_nets_by_smoothed_fitness(1).unwrap()[0],
            hs[0],
            "Maximize: lowest score is worst"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn catch_up_replays_steps_zero_to_clock() {
        let dir = std::env::temp_dir().join("race_catchup_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine.config.crossover_prob = 0.0;
        engine.config.mutate_prob = 0.0;
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.0))
            .unwrap();
        let clock = 5;
        // generate_child can now return Ok(None) (spent roll) — this test
        // exercises catch-up mechanics, so build the child directly.
        let mut child = engine.random_child(clock, 0).unwrap();
        assert!(
            !child
                .state
                .created_from
                .as_deref()
                .unwrap()
                .contains("crossover"),
            "catch-up replay test should use the random branch so the child nets fit the tiny harness"
        );
        let topo = child.state.topology().unwrap();
        assert_eq!(topo.options.input_dim, Some(2));
        assert_eq!(topo.options.output_dim, Some(2));
        // Catch-up replays steps 0..clock-1 and leaves the child at the clock.
        engine.catch_up(&mut child, clock).unwrap();
        assert_eq!(child.state.step, clock);
        let m = child.state.last_metrics.as_ref().unwrap();
        assert_eq!(m.step, clock - 1);
    }

    #[test]
    fn catch_up_is_deterministic() {
        let dir_a = std::env::temp_dir().join("race_catchup_det_a");
        let dir_b = std::env::temp_dir().join("race_catchup_det_b");
        let mut engine_a = engine(&dir_a, 42).unwrap();
        let mut engine_b = engine(&dir_b, 42).unwrap();
        engine_a.config.crossover_prob = 0.0;
        engine_b.config.crossover_prob = 0.0;
        engine_a.config.mutate_prob = 0.0;
        engine_b.config.mutate_prob = 0.0;
        engine_a
            .seed_population_internal(vec![tiny_topology(7)], Some(0.0))
            .unwrap();
        engine_b
            .seed_population_internal(vec![tiny_topology(7)], Some(0.0))
            .unwrap();
        let clock = 5;
        let mut child_a = engine_a.random_child(clock, 0).unwrap();
        let mut child_b = engine_b.random_child(clock, 0).unwrap();
        engine_a.catch_up(&mut child_a, clock).unwrap();
        engine_b.catch_up(&mut child_b, clock).unwrap();
        // Same seed ⇒ same catch-up metrics.
        let m_a = child_a.state.last_metrics.as_ref().unwrap();
        let m_b = child_b.state.last_metrics.as_ref().unwrap();
        assert_eq!(m_a.train_loss, m_b.train_loss);
        assert_eq!(m_a.fitness, m_b.fitness);
        assert_eq!(m_a.eval_loss, m_b.eval_loss);
    }

    #[test]
    fn step_one_net_advances_the_net_step_and_records_metrics() {
        let dir = std::env::temp_dir().join("race_step_one_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], None)
            .unwrap();
        let hash = engine.state.live_hashes()[0].clone();
        engine.step_one_net(&hash, 0).unwrap();
        let state = engine.state.net(&hash).unwrap();
        assert_eq!(state.step, 1);
        let m = state.last_metrics.as_ref().unwrap();
        assert_eq!(m.step, 0);
        assert!(m.train_loss.is_finite());
        assert!(m.fitness.is_finite());
    }

    #[test]
    fn step_one_net_writes_state_file() {
        let dir = std::env::temp_dir().join("race_step_one_file_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], None)
            .unwrap();
        let hash = engine.state.live_hashes()[0].clone();
        engine.step_one_net(&hash, 0).unwrap();
        engine.write_live_frontier_states().unwrap();
        let loaded = crate::state::load_net_state(&dir, &hash).unwrap();
        assert_eq!(loaded.step, 1);
        assert!(loaded.last_metrics.as_ref().unwrap().train_loss.is_finite());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── RL mode (RunSpec::rl) ─────────────────────────────────────────

    /// Minimal RlStep for construction-validation tests: reports a constant
    /// fitness, touches nothing else.
    struct TestRlTrainer;
    impl crate::trainer::StepTrainer for TestRlTrainer {
        fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
            use flodl::nn::Module;
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }
    }
    impl crate::trainer::RlStep for TestRlTrainer {
        fn train_step(
            &mut self,
            _net: &mut Network,
            _optimizer: &mut dyn Optimizer,
            _step: usize,
            _ctx: &crate::trainer::RlContext<'_>,
        ) -> flodl::tensor::Result<crate::trainer::RlStepReport> {
            Ok(crate::trainer::RlStepReport {
                train_loss: 0.0,
                fitness: 1.0,
                informative: Vec::new(),
                challenged_turns: 0,
                rl: None,
            })
        }
    }

    /// RL trainer whose report depends on `(net_seed, step)` — a stand-in for
    /// "a deterministic env": replay can only reproduce the recorded metrics
    /// if the engine feeds the trainer the same clock + seed it did live.
    struct SeededRlTrainer;
    impl crate::trainer::StepTrainer for SeededRlTrainer {
        fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
            use flodl::nn::Module;
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }
    }
    impl crate::trainer::RlStep for SeededRlTrainer {
        fn train_step(
            &mut self,
            _net: &mut Network,
            _optimizer: &mut dyn Optimizer,
            step: usize,
            ctx: &crate::trainer::RlContext<'_>,
        ) -> flodl::tensor::Result<crate::trainer::RlStepReport> {
            let fitness = (ctx.net_seed % 97) as f32 + step as f32;
            Ok(crate::trainer::RlStepReport {
                train_loss: step as f32 * 0.5,
                fitness,
                informative: Vec::new(),
                challenged_turns: 0,
                rl: Some(crate::trainer::RlStepMeta {
                    matches: 2,
                    train_turns: 12 + step,
                    eval_turns: 8,
                }),
            })
        }
    }

    /// An RL-mode config (`RaceConfig` is not `Clone`, so tests rebuild it by
    /// value: `pop == 0` only for the harness that seeds its own topologies).
    fn rl_config(pop: usize) -> RaceConfig {
        let topo = crate::graph::topology::TopologyOptions {
            input_dim: Some(2),
            output_dim: Some(2),
            ..Default::default()
        };
        RaceConfig {
            pop_size: pop,
            mode: crate::engine::config::RunMode::Rl,
            topology_options: topo,
            ..RaceConfig::defaults()
        }
    }

    /// RL trainer that HONORS the challenge SIGNAL: when `ctx.challenged` is
    /// true it reports a fitness of −1000 (a sentinel no normal step ever
    /// reports) plus 5 challenged turns, so tests can observe exactly which
    /// steps ran challenged.
    struct ChallengingRlTrainer;
    impl crate::trainer::StepTrainer for ChallengingRlTrainer {
        fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
            use flodl::nn::Module;
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }
    }
    impl crate::trainer::RlStep for ChallengingRlTrainer {
        fn train_step(
            &mut self,
            _net: &mut Network,
            _optimizer: &mut dyn Optimizer,
            step: usize,
            ctx: &crate::trainer::RlContext<'_>,
        ) -> flodl::tensor::Result<crate::trainer::RlStepReport> {
            let (fitness, challenged_turns) = if ctx.challenged {
                (-1000.0, 5)
            } else {
                (step as f32 + 1.0, 0)
            };
            Ok(crate::trainer::RlStepReport {
                train_loss: step as f32 * 0.5,
                fitness,
                informative: Vec::new(),
                challenged_turns,
                rl: Some(crate::trainer::RlStepMeta {
                    matches: 1,
                    train_turns: 5,
                    eval_turns: 0,
                }),
            })
        }
    }
    /// RL-mode engine harness: no dataset, no stream, `RunMode::Rl`.
    fn rl_engine(run_dir: &std::path::Path, seed: u64) -> Result<crate::engine::RlEngine> {
        crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
            rl_config(0),
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            SeededRlTrainer,
            Some(seed),
            Some(run_dir.to_path_buf()),
        ))
    }

    /// The RL resume contract, mirroring the tabular twin test: an interrupted
    /// run resumed and continued N more steps lands exactly where an
    /// uninterrupted run of the same length lands (replay is bit-exact, and
    /// the frontier + checkpoint ledger come back from disk).
    #[test]
    fn rl_resume_then_continue_matches_uninterrupted_twin() {
        let dir_a = std::env::temp_dir().join("gras-rl-resume-a");
        let dir_b = std::env::temp_dir().join("gras-rl-resume-b");
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);

        let run = |dir: &std::path::Path,
                   steps: usize|
         -> Vec<(String, Option<crate::state::NetMetrics>)> {
            let mut eng = rl_engine(dir, 4242).unwrap();
            eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
                .unwrap();
            let hashes = eng.state.live_hashes();
            for clock in 0..steps {
                for h in &hashes {
                    eng.step_one_net(h, clock).unwrap();
                }
            }
            let out = hashes
                .iter()
                .map(|h| (h.clone(), eng.state.net(h).unwrap().last_metrics.clone()))
                .collect();
            // Persist the frontier the way a stop does. (`engine.json` is
            // already on disk — `from_spec` writes the header.)
            for h in &hashes {
                let state = eng.state.net(h).cloned().unwrap();
                crate::state::write_net_state(dir, &state).unwrap();
            }
            out
        };

        // Uninterrupted twin: 5 steps in one sitting.
        let want = run(&dir_a, 5);

        // Interrupted: 3 steps, drop, resume, 2 more.
        let interrupted = run(&dir_b, 3);
        assert_eq!(interrupted[0].1.as_ref().unwrap().step, 2);
        let mut resumed = crate::engine::RlEngine::resume(
            dir_b.clone(),
            rl_config(2),
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            SeededRlTrainer,
        )
        .unwrap();
        assert_eq!(resumed.state.live_count(), 2, "RL frontier restored");
        let hashes = resumed.state.live_hashes();
        for clock in 3..5 {
            for h in &hashes {
                resumed.step_one_net(h, clock).unwrap();
            }
        }
        for (hash, metrics) in &want {
            let got = resumed.state.net(hash).unwrap().last_metrics.clone();
            assert_eq!(&got, metrics, "RL resume diverged for net {hash}");
        }

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    /// The challenge SIGNAL: with prob 1.0 every step fires, so a trainer
    /// honoring `ctx.challenged` reports its sentinel fitness and challenged
    /// turns on EVERY step; a trainer ignoring the signal is simply never
    /// challenged (no error — the knob is a signal, the trainer decides).
    #[test]
    fn challenge_signal_reaches_the_trainer_and_ignoring_is_legal() {
        let dir = std::env::temp_dir().join("gras-challenge-signal");
        let _ = std::fs::remove_dir_all(&dir);
        let mut cfg = rl_config(0);
        cfg.challenge_prob = 1.0; // every step fires
        let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
            cfg,
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            ChallengingRlTrainer, // honors the flag → sentinel on every step
            Some(1),
            Some(dir.clone()),
        ))
        .unwrap();
        eng.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let hash = eng.state.live_hashes()[0].clone();
        eng.step_one_net(&hash, 0).unwrap();
        let lm = eng.state.net(&hash).unwrap().last_metrics.clone().unwrap();
        assert_eq!(lm.fitness, -1000.0, "honored flag → challenged step");
        // The rollup's ⚔ accounting counts what the trainer reported.
        assert_eq!(eng.step_rl.challenged_turns, 5);
        assert_eq!(eng.total_challenged_turns, 5);
        let _ = std::fs::remove_dir_all(&dir);

        // Ignoring the signal is legal: SeededRlTrainer never reads the flag.
        let dir2 = std::env::temp_dir().join("gras-challenge-ignored");
        let _ = std::fs::remove_dir_all(&dir2);
        let mut cfg2 = rl_config(0);
        cfg2.challenge_prob = 1.0;
        let mut eng2 = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
            cfg2,
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            SeededRlTrainer,
            Some(1),
            Some(dir2.clone()),
        ))
        .unwrap();
        eng2.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let hash2 = eng2.state.live_hashes()[0].clone();
        eng2.step_one_net(&hash2, 0).unwrap(); // must NOT error
        let lm2 = eng2
            .state
            .net(&hash2)
            .unwrap()
            .last_metrics
            .clone()
            .unwrap();
        assert_ne!(
            lm2.fitness, -1000.0,
            "an ignoring trainer is never challenged"
        );
        assert_eq!(
            eng2.total_challenged_turns, 0,
            "no challenged turns were reported"
        );
        let _ = std::fs::remove_dir_all(&dir2);
    }

    /// A challenge is just another step: its fitness (measured under the
    /// forced action) RANKS like any other — recorded in metrics AND fed to
    /// the ranking buffers.
    #[test]
    fn challenged_fitness_ranks_like_any_other_step() {
        let dir = std::env::temp_dir().join("gras-challenge-honesty");
        let _ = std::fs::remove_dir_all(&dir);
        let mut cfg = rl_config(0);
        cfg.challenge_prob = 1.0; // every step fires → fully deterministic
        cfg.freeze_elites = false; // isolate the challenge from the freeze
        let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
            cfg,
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            ChallengingRlTrainer,
            Some(7),
            Some(dir.clone()),
        ))
        .unwrap();
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes = eng.state.live_hashes();
        for clock in 0..3 {
            for h in &hashes {
                eng.step_one_net(h, clock).unwrap();
            }
        }
        for h in &hashes {
            // Recorded: the challenged sentinel fitness is on every step.
            let lm = eng.state.net(h).unwrap().last_metrics.clone().unwrap();
            assert_eq!(lm.fitness, -1000.0, "challenged metrics must be recorded");
            // Ranks: the sentinel IS in the ranking buffer — a challenge is
            // just another step, no exclusion.
            let buf = eng.rolling_fitness.get(h.as_str()).unwrap();
            assert_eq!(
                buf.iter().count(),
                3,
                "every step (challenged included) must feed the ranking buffer"
            );
            assert!(
                buf.iter().all(|&f| f == -1000.0),
                "the challenged sentinel must be the buffered value"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Replay parity under challenges: an interrupted run with
    /// `challenge_prob = 1.0` (every step challenged) resumed mid-run lands
    /// on the twin's metrics — the catch-up mirror re-fires the same
    /// challenges through the same `challenge_step`.
    #[test]
    fn challenge_resume_then_continue_matches_uninterrupted_twin() {
        let dir_a = std::env::temp_dir().join("gras-rl-challenge-a");
        let dir_b = std::env::temp_dir().join("gras-rl-challenge-b");
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);

        let challenge_cfg = |pop: usize| {
            let mut cfg = rl_config(pop);
            cfg.challenge_prob = 1.0;
            cfg.freeze_elites = false;
            cfg
        };
        let run = |dir: &std::path::Path,
                   steps: usize|
         -> Vec<(String, Option<crate::state::NetMetrics>)> {
            let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
                challenge_cfg(0),
                crate::engine::fitness::Fitness::reported(
                    crate::engine::fitness::Direction::Maximize,
                    "reward",
                ),
                ChallengingRlTrainer,
                Some(4242),
                Some(dir.to_path_buf()),
            ))
            .unwrap();
            eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
                .unwrap();
            let hashes = eng.state.live_hashes();
            for clock in 0..steps {
                for h in &hashes {
                    eng.step_one_net(h, clock).unwrap();
                }
            }
            let out = hashes
                .iter()
                .map(|h| (h.clone(), eng.state.net(h).unwrap().last_metrics.clone()))
                .collect();
            for h in &hashes {
                let state = eng.state.net(h).cloned().unwrap();
                crate::state::write_net_state(dir, &state).unwrap();
            }
            out
        };

        let want = run(&dir_a, 5);
        // Sanity: the twin itself ran challenged steps (sentinel fitness).
        assert_eq!(want[0].1.as_ref().unwrap().fitness, -1000.0);

        run(&dir_b, 3);
        let mut resumed = crate::engine::RlEngine::resume(
            dir_b.clone(),
            challenge_cfg(2),
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            ChallengingRlTrainer,
        )
        .unwrap();
        assert_eq!(resumed.state.live_count(), 2, "RL frontier restored");
        let hashes = resumed.state.live_hashes();
        for clock in 3..5 {
            for h in &hashes {
                resumed.step_one_net(h, clock).unwrap();
            }
        }
        for (hash, metrics) in &want {
            let got = resumed.state.net(hash).unwrap().last_metrics.clone();
            assert_eq!(&got, metrics, "challenge resume diverged for net {hash}");
        }

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    #[test]
    fn rl_resume_rejects_a_tabular_run_dir() {
        // Wrong mode must be refused by name, not silently replayed.
        let dir = std::env::temp_dir().join("gras-rl-resume-wrong-mode");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 3).unwrap();
        eng.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        drop(eng);
        let err = match crate::engine::RlEngine::resume(
            dir.clone(),
            rl_config(1),
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            SeededRlTrainer,
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("resume_rl on a tabular run must fail"),
        };
        assert!(
            err.contains("not \"rl\""),
            "error must name the mismatch: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rl_spec_derives_rl_mode_without_declaration() {
        // The spec variant DECIDES the mode: an RL spec runs as RL with no
        // `set_run_mode(..)` call (the setter is deleted — disagreement is
        // unrepresentable). The engine derives `config.mode` from the spec.
        let dir = std::env::temp_dir().join("gras-rl-mode-derived");
        let _ = std::fs::remove_dir_all(&dir);
        let config = RaceConfig {
            pop_size: 2,
            max_steps: Some(1),
            ..RaceConfig::defaults()
        };
        let spec = crate::engine::run_spec::RunSpec::rl(
            config,
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            TestRlTrainer,
            Some(7),
            Some(dir.clone()),
        );
        let spec = match spec {
            crate::engine::run_spec::RunSpec::RL(mut s) => {
                s.config.topology_options.input_dim = Some(1);
                s.config.topology_options.output_dim = Some(1);
                crate::engine::run_spec::RunSpec::RL(s)
            }
            other => other,
        };
        let eng = CoreEngine::from_spec(spec)
            .expect("RL spec must derive RunMode::Rl from the spec variant itself");
        assert_eq!(
            eng.config.mode,
            crate::engine::config::RunMode::Rl,
            "mode is derived from the spec variant, not user-declared"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rl_spec_rejects_computed_fitness() {
        // No dataset exists in RL mode, so a (pred, target) scorer has
        // nothing to score — construction must fail loudly, not mid-run.
        let config = RaceConfig {
            pop_size: 2,
            max_steps: Some(1),
            mode: crate::engine::config::RunMode::Rl, // declared correctly; the FITNESS is the bug under test
            ..RaceConfig::defaults()
        };
        let topo = crate::graph::topology::TopologyOptions {
            input_dim: Some(1),
            output_dim: Some(1),
            ..Default::default()
        };
        let config = RaceConfig {
            topology_options: topo,
            ..config
        };
        let spec = crate::engine::run_spec::RunSpec::rl(
            config,
            fitness(), // Computed — WRONG for RL
            TestRlTrainer,
            Some(7),
            None,
        );
        let err = match CoreEngine::from_spec(spec) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("RL spec with Computed fitness must be rejected at construction"),
        };
        assert!(
            err.contains("Fitness::reported"),
            "error must name the fix: {err}"
        );
    }

    // ── Iter-5: child generation contract ──────────────────────────────

    fn seed_two_parents(dir: &std::path::Path, seed: u64) -> crate::engine::TabularEngine {
        let mut engine = engine(dir, seed).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        engine
    }

    #[test]
    fn child_generation_is_deterministic_per_run_seed_and_clock() {
        let dir_a = std::env::temp_dir().join("race_child_det_a");
        let dir_b = std::env::temp_dir().join("race_child_det_b");
        let mut a = seed_two_parents(&dir_a, 123);
        let mut b = seed_two_parents(&dir_b, 123);
        a.config.crossover_prob = 1.0;
        b.config.crossover_prob = 1.0;
        // Determinism ⇒ the SAME outcome, including `None` (all pairing
        // attempts incompatible). The self-pairing redraw shares the rng, so
        // it must fire identically in both engines.
        let ca = a.generate_child(3, 0).unwrap();
        let cb = b.generate_child(3, 0).unwrap();
        let (ca, cb) = match (ca, cb) {
            (Some(ca), Some(cb)) => (ca, cb),
            (None, None) => return,
            _ => panic!("deterministic engines must agree on Some/None"),
        };
        assert_eq!(
            ca.state.hash, cb.state.hash,
            "same seed + clock ⇒ same child topology"
        );
        assert_eq!(ca.state.net_seed, cb.state.net_seed);
        assert_eq!(ca.state.created_from, cb.state.created_from);
    }

    #[test]
    fn crossover_prob_zero_means_spent_roll() {
        let dir = std::env::temp_dir().join("race_child_cx0");
        let mut engine = seed_two_parents(&dir, 7);
        engine.config.crossover_prob = 0.0;
        // crossover_prob=0 ⇒ the roll never fires ⇒ Ok(None) (spent roll).
        // Random immigrants come only from the mutation rolls.
        for clock in 0..4 {
            let child = engine.generate_child(clock, 0).unwrap();
            assert!(
                child.is_none(),
                "crossover_prob=0 ⇒ no child at clock {clock}"
            );
        }
    }

    #[test]
    fn mutation_prob_zero_and_one_flips_mut_suffix() {
        // The mutation roll now belongs to the CROSSOVER branch only: the
        // exploit path may get a perturbation; the immigrant path is pure
        // exploration and never carries the '+mut' suffix.
        let dir_no = std::env::temp_dir().join("race_child_mut0");
        let mut no = seed_two_parents(&dir_no, 21);
        no.config.mutate_prob = 0.0;
        let c = no.random_child(1, 0).unwrap();
        assert_eq!(
            c.state.created_from.as_deref(),
            Some("random"),
            "no '+mut' suffix"
        );

        let dir_yes = std::env::temp_dir().join("race_child_mut1");
        let mut yes = seed_two_parents(&dir_yes, 21);
        yes.config.mutate_prob = 1.0;
        let c = yes.random_child(1, 0).unwrap();
        assert_eq!(
            c.state.created_from.as_deref(),
            Some("random"),
            "immigrant path is mutation-free regardless of mutate_prob"
        );
    }

    fn flat_topology(seed: usize) -> Topology {
        // A topology with **no hidden nodes**: input → output directly.
        // Crossover requires hidden nodes to match pivots on, so any pairing
        // of flat parents fails all 3 attempts deterministically — the exact
        // precondition for the clone-fittest fallback.
        let mut topo = Topology::new(
            seed,
            Some(TopologyOptions {
                topology_seed: seed,
                min_hidden_num_nodes: 0,
                max_hidden_num_nodes: 0,
                min_hidden_inputs_per_node: 1,
                max_hidden_inputs_per_node: 1,
                min_hidden_outputs_per_node: 1,
                max_hidden_outputs_per_node: 1,
                input_dim: Some(2),
                output_dim: Some(2),
                dropout_prob: 0.0,
            }),
        );
        topo.nodes.push(Node::new_input(0, 2));
        topo.nodes.push(Node::new_output(1, 2, 2));
        let conn = |from: (usize, usize), to: (usize, usize)| crate::graph::topology::Connection {
            from: crate::graph::topology::Port {
                node: from.0,
                index: from.1,
            },
            to: crate::graph::topology::Port {
                node: to.0,
                index: to.1,
            },
        };
        topo.connections.push(conn((0, 0), (1, 0)));
        topo.connections.push(conn((0, 1), (1, 1)));
        topo.finalize();
        topo
    }

    #[test]
    fn failed_crossover_after_three_attempts_is_a_noop() {
        // Hidden-less parents make every crossover attempt a no-op
        // (cx_one_point/cx_uniform both bail with zero hidden nodes). After
        // 3 attempts the roll is SPENT — no random fallback (random whole
        // nets enter only via the mutation path), signaled by a clean error
        // the caller skips, never a stall.
        let dir = std::env::temp_dir().join("race_child_fallback");
        let mut engine = engine(&dir, 55).unwrap();
        engine
            .seed_population_internal(vec![flat_topology(7), flat_topology(8)], Some(0.5))
            .unwrap();
        engine.config.crossover_prob = 1.0;
        engine.config.mutate_prob = 0.0;
        let before = engine.state.live_count();
        let result = engine.generate_child(2, 0).unwrap();
        assert!(
            result.is_none(),
            "3 failed crossovers ⇒ spent roll (no child), not a random fallback"
        );
        assert_eq!(
            engine.state.live_count(),
            before,
            "population must be untouched by a spent crossover roll"
        );
    }

    #[test]
    fn crossover_never_self_pairs_the_same_net() {
        // The two roulette draws are independent, so without the redraw guard
        // the same net could be both parents (a no-op self-cross that burns
        // the roll on a clone). Sweep several (seed, clock) pairs — with one
        // net dominating the roulette wheel the naive draw self-pairs often —
        // and assert no surviving lineage reads `parents=X,X`.
        for seed in [3u64, 17, 55, 123, 9001] {
            let dir = std::env::temp_dir().join(format!("race_self_pair_{seed}"));
            let mut eng = engine(&dir, seed).unwrap();
            eng.seed_population_internal(
                vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
                Some(0.5),
            )
            .unwrap();
            eng.config.crossover_prob = 1.0;
            // Fixture is Minimize: make net 0 the overwhelming favorite.
            let hs = eng.state.live_hashes();
            for (i, h) in hs.iter().enumerate() {
                let v = 0.01 + i as f32 * 10.0;
                eng.rolling_fitness.get_mut(h).unwrap().push(v);
            }
            for clock in 0..12 {
                if let Some(Some(child)) = eng.generate_child(clock, 0).unwrap().into() {
                    let lineage = child.state.created_from.clone().unwrap_or_default();
                    let parents = lineage
                        .split_once("parents=")
                        .map(|(_, p)| p)
                        .unwrap_or_default();
                    let pair: Vec<&str> = parents.split(',').collect();
                    assert!(
                        !(pair.len() == 2 && pair[0] == pair[1]),
                        "seed {seed} clock {clock}: self-paired crossover ({lineage})"
                    );
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn resume_restores_run_counters() {
        // Stop → resume must continue the ORIGINAL cull budget, wall-clock
        // age, and child-seed ordinals — a resumed run is the same race, not
        // a fresh one with reset budgets. `held_out_eval_rows` also rides the
        // header: the resumed stream must use the run's recorded geometry.
        let dir = std::env::temp_dir().join("race_resume_counters");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes = engine.state.live_hashes();
        for clock in 0..2 {
            for h in &hashes {
                engine.step_one_net(h, clock).unwrap();
            }
        }
        // Simulate a run history: some culls, some elapsed wall time, a
        // couple of child ordinals.
        engine.culls = 7;
        engine.elapsed_base_secs = 120;
        engine.children_born_at_clock.insert(1, 2);
        engine.children_born_at_clock.insert(3, 5);
        // A stop stamps the counters into engine.json.
        engine.persist_run_counters();
        for h in &hashes {
            let state = engine.state.net(h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        drop(engine);

        // The persisted header carries them.
        let header = crate::state::load_engine_json(&dir).unwrap();
        assert_eq!(header.culls, 7);
        assert_eq!(header.run_elapsed_secs, 120);
        assert_eq!(header.children_born_at_clock.get(&3), Some(&5));

        let resumed = crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("resume_counters"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 2;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        assert_eq!(resumed.culls, 7, "cull budget continues, not resets");
        assert_eq!(
            resumed.elapsed_base_secs, 120,
            "wall-clock age restored as base offset"
        );
        assert_eq!(
            resumed.children_born_at_clock.get(&1),
            Some(&2),
            "child ordinals restored"
        );
        // elapsed_seconds now reports base + current-session time.
        let snap = resumed.snapshot(2);
        assert!(
            snap.elapsed_seconds >= 120,
            "elapsed_seconds keeps the run's true age (got {})",
            snap.elapsed_seconds
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_held_out_eval_rows_ride_the_header() {
        // The header's held_out_eval_rows is data geometry: the resumed
        // stream must use the run's recorded value, not re-decide it.
        let dir = std::env::temp_dir().join("race_resume_held_out");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h = engine.state.live_hashes()[0].clone();
        engine.step_one_net(&h, 0).unwrap();
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
        // Stamp a distinctive value into the header (as a run with a custom
        // stream would have recorded).
        engine.header.held_out_eval_rows = Some(64);
        crate::state::write_engine_json(&dir, &engine.header).unwrap();
        drop(engine);

        let resumed = crate::engine::TabularEngine::resume(
            dir,
            tiny_dataset_dir("resume_held_out"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 1;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        assert_eq!(
            resumed.stream.as_ref().unwrap().held_out_eval_rows(),
            64,
            "resumed stream honors the header's held_out_eval_rows"
        );
    }

    #[test]
    fn engine_mode_discriminator_roundtrip() {
        // Engine split (TODO.md step 6): new headers carry `engine_mode` at
        // the root; legacy shared-shape headers (absent root field) are
        // migrated on load by deriving from `config.mode`, so old run dirs
        // keep resuming. New dirs get the discriminator stamped.
        let dir = std::env::temp_dir().join("race_engine_mode_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h = engine.state.live_hashes()[0].clone();
        engine.step_one_net(&h, 0).unwrap();
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();

        // New header: root discriminator present and correct.
        let header = crate::state::load_engine_json(&dir).unwrap();
        assert_eq!(
            header.engine_mode.as_deref(),
            Some("tabular"),
            "from_spec stamps the mode at the header root"
        );

        // Legacy shape: strip the root field, keep `config.mode` — must still
        // load, deriving the mode, and must still resume.
        let mut legacy: serde_json::Value = serde_json::from_str(
            &crate::state::load_engine_json(&dir)
                .unwrap()
                .to_json()
                .unwrap(),
        )
        .unwrap();
        legacy.as_object_mut().unwrap().remove("engine_mode");
        assert!(legacy.get("config").unwrap().get("mode").is_some());
        std::fs::write(
            dir.join("engine.json"),
            serde_json::to_string_pretty(&legacy).unwrap(),
        )
        .unwrap();
        let migrated = crate::state::load_engine_json(&dir).unwrap();
        assert_eq!(
            migrated.engine_mode.as_deref(),
            Some("tabular"),
            "legacy header migrates via config.mode"
        );
        let mut resumed = crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("engine_mode_rt"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 1;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        resumed.step_one_net(&h, 1).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_replays_nets_with_metric_parity() {
        let dir = std::env::temp_dir().join("race_resume_parity");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes: Vec<String> = engine.state.live_hashes();

        // Drive both nets 4 steps by hand (deterministic stream).
        for clock in 0..4 {
            for h in &hashes {
                engine.step_one_net(h, clock).unwrap();
            }
        }
        // Persist the live frontier as a stop would.
        for h in &hashes {
            let state = engine.state.net(h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        let recorded: Vec<_> = hashes
            .iter()
            .map(|h| engine.state.net(h).unwrap().last_metrics.clone().unwrap())
            .collect();
        drop(engine);

        // Reconstruct via resume — parity is asserted inside.
        let mut resumed = crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("resume"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 2;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        assert_eq!(resumed.state.live_count(), 2, "both live nets restored");
        for (h, rec) in hashes.iter().zip(recorded) {
            let state = resumed.state.net(h).unwrap();
            assert_eq!(state.step, 4, "net {h} replayed to its recorded step");
            assert_eq!(state.last_metrics, Some(rec), "bit-identical metrics");
        }

        // The resumed engine continues stepping normally.
        resumed.step_one_net(&hashes[0], 4).unwrap();
        assert_eq!(resumed.state.net(&hashes[0]).unwrap().step, 5);
    }

    #[test]
    fn resume_then_continue_matches_uninterrupted_twin() {
        // Tier C exit proof: run 3 steps → drop → resume → run 2 more must
        // land exactly where an uninterrupted 5-step twin lands.
        let dir_a = std::env::temp_dir().join("race_twin_uninterrupted");
        let _ = std::fs::remove_dir_all(&dir_a);
        let mut full = engine(&dir_a, 123).unwrap();
        full.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h_full = full.state.live_hashes()[0].clone();
        for clock in 0..5 {
            full.step_one_net(&h_full, clock).unwrap();
        }
        let final_metrics_full = full.state.net(&h_full).unwrap().last_metrics.clone();

        // Interrupted twin: 3 steps, persist, drop, resume, 2 more.
        let dir_b = std::env::temp_dir().join("race_twin_interrupted");
        let _ = std::fs::remove_dir_all(&dir_b);
        let mut part = engine(&dir_b, 123).unwrap();
        part.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h_part = part.state.live_hashes()[0].clone();
        for clock in 0..3 {
            part.step_one_net(&h_part, clock).unwrap();
        }
        let state = part.state.net(&h_part).cloned().unwrap();
        crate::state::write_net_state(&dir_b, &state).unwrap();
        drop(part);

        let mut resumed = crate::engine::TabularEngine::resume(
            dir_b,
            tiny_dataset_dir("resume2"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 1;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        // After resume the hash is the same (same topology), so step 3/4
        // continue the identical trajectory.
        for clock in 3..5 {
            resumed.step_one_net(&h_part, clock).unwrap();
        }
        assert_eq!(
            resumed.state.net(&h_part).unwrap().last_metrics,
            final_metrics_full,
            "interrupt+resume == uninterrupted twin (bit-identical)"
        );
    }

    #[test]
    fn immigrant_evolution_keeps_pop_constant() {
        let dir = std::env::temp_dir().join("race_immigrant_pop");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal((0..4).map(tiny_topology).collect(), Some(0.5))
            .unwrap();
        // Three sequential immigrant rounds at the same clock: each culls one
        // fitness-inverse-selected net and inserts a random immigrant. If the
        // child ordinal restarts per round, two children collide on hash →
        // dedupe → pop shrinks below 4.
        for _ in 0..3 {
            engine.evolve_random_immigrant(7, 0).unwrap();
        }
        assert_eq!(
            engine.state.live_count(),
            4,
            "3 immigrant rounds × cull1+birth1 must leave pop unchanged"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_checkpoint_ledger_gate_parity() {
        let dir = std::env::temp_dir().join("race_checkpoint_resume_parity");
        let _ = std::fs::remove_dir_all(&dir);

        let mut engine = engine(&dir, 42).unwrap();
        engine.config.checkpoint_every = 2;
        // Keep the stream's rotation cadence in sync, exactly as run() does —
        // this test steps nets manually, bypassing run().
        let ckpt_every = engine.config.checkpoint_every;
        engine
            .stream
            .as_mut()
            .unwrap()
            .set_run_checkpoint_every(ckpt_every);
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();

        // Step 1 & 2 to create a checkpoint at step 2
        let hashes = engine.state.live_hashes();
        for clock in 0..3 {
            for h in &hashes {
                engine.step_one_net(h, clock).unwrap();
            }
            // Trigger checkpoint write in engine.run() equivalent
            if clock > 0 && clock % engine.config.checkpoint_every == 0 {
                let era = (clock / engine.config.checkpoint_every) as u64;
                let mean = engine.population_mean_smoothed_fitness();
                let exam = engine.run_checkpoint_exam(era).unwrap();
                engine.checkpoints.push(Checkpoint {
                    step: clock,
                    pop_mean_fitness: mean,
                    exam_mean_fitness: exam,
                });
                engine.write_checkpoints().unwrap();
            }
        }
        assert_eq!(engine.checkpoints.len(), 1);
        let expected_mean = engine.checkpoints[0].pop_mean_fitness;

        // Persist the live states
        for h in &hashes {
            let state = engine.state.net(h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        drop(engine);

        // Resume engine
        let resumed = crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("chk_parity"),
            {
                let mut c = RaceConfig::defaults();
                c.checkpoint_every = 2;
                c.pop_size = 2;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();

        assert_eq!(resumed.checkpoints.len(), 1, "ledger reloaded");
        assert_eq!(resumed.checkpoints[0].step, 2);
        assert_eq!(resumed.checkpoints[0].pop_mean_fitness, expected_mean);
        assert!(
            !resumed.checkpoints[0].exam_mean_fitness.is_nan(),
            "exam reading persisted"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_validates_pop_size_mismatch() {
        let dir = std::env::temp_dir().join("race_resume_pop_validation");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes = engine.state.live_hashes();

        // Write both as live
        for h in &hashes {
            let state = engine.state.net(h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        drop(engine);

        // Resume with pop_size 3 (mismatch, expects 3, only 2 found)
        let mut config = RaceConfig::defaults();
        config.pop_size = 3;
        let resumed = crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("pop_validation"),
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        );
        assert!(resumed.is_err());
        let err_msg = resumed.err().unwrap().to_string();
        assert!(err_msg.contains("resume: expected 3 live nets"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Trainer-blob validation on resume ───────────────────────────────

    #[test]
    fn resume_rejects_changed_trainer_const_naming_the_key() {
        // The drift detector: a run trained with LR 0.001, resumed with LR
        // 0.5, must fail at RESUME TIME naming "learning_rate" and both
        // values — not later, as a generic parity-assert failure.
        let dir = std::env::temp_dir().join("race_blob_mismatch");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h = engine.state.live_hashes().remove(0);
        engine.step_one_net(&h, 0).unwrap();
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
        drop(engine);

        let resumed = crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("blob_mismatch"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 1;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer {
                learning_rate: 0.5, // the run recorded the default (0.001)
                ..crate::trainer::TabularTrainer::new(loss_fn())
            },
        );
        assert!(resumed.is_err(), "changed LR must be caught at resume");
        let msg = resumed.err().unwrap().to_string();
        assert!(
            msg.contains("learning_rate") && msg.contains("0.5") && msg.contains("0.001"),
            "error must NAME the drifted key and both values, got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_allows_identical_trainer_blob() {
        // Control for the mismatch test: same consts → resume proceeds.
        let dir = std::env::temp_dir().join("race_blob_match");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h = engine.state.live_hashes().remove(0);
        engine.step_one_net(&h, 0).unwrap();
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
        drop(engine);

        let resumed = crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("blob_match"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 1;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()), // identical consts
        );
        assert!(
            resumed.is_ok(),
            "identical blob must resume: {:?}",
            resumed.err()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_skips_blob_check_for_legacy_headers() {
        // A legacy run (or hook-less trainer) records no blob: the check
        // must stay silent, not hard-fail every old run_dir.
        let recorded = None;
        let trainer = crate::trainer::ModeAdapter::tabular(Box::new(
            crate::trainer::TabularTrainer::new(loss_fn()),
        ));
        assert!(
            super::assert_trainer_blob_matches(&recorded, &trainer, "resume").is_ok(),
            "no recorded blob ⇒ nothing to compare ⇒ resume proceeds"
        );
    }

    #[test]
    fn trainer_blob_diff_reports_nested_and_missing_keys() {
        // The diff helper's shape guarantees: nested objects report
        // path-prefixed keys; keys present on one side only are named.
        let run = serde_json::json!({"update": "reinforce", "match_length": {"random": {"min": 48, "max": 240}}});
        let incoming = serde_json::json!({"update": "value_head", "match_length": {"random": {"min": 48, "max": 120}}, "grad_clip": 1.0});
        let mut diffs = Vec::new();
        super::diff_json("", &run, &incoming, &mut diffs);
        let joined = diffs.join("; ");
        assert!(
            joined.contains("\"update\": run had \"reinforce\", incoming has \"value_head\""),
            "{joined}"
        );
        assert!(
            joined.contains("\"match_length.random.max\": run had 240, incoming has 120"),
            "{joined}"
        );
        assert!(
            joined.contains("\"grad_clip\": incoming declares 1.0, run didn't record it"),
            "{joined}"
        );
        // And no false positive on the equal key.
        assert!(!joined.contains("random.min"), "{joined}");
    }

    /// Seed every live net with a RAW last-step fitness of `v`.
    fn seed_raw_fitness(engine: &mut crate::engine::TabularEngine, v: f32) {
        for h in engine.state.live_hashes() {
            engine
                .state
                .record_step(
                    &h,
                    crate::state::NetMetrics {
                        step: 0,
                        train_loss: 0.0,
                        eval_loss: None,
                        fitness: v,
                        informative: vec![],
                        frozen: false,
                    },
                )
                .unwrap();
        }
    }

    #[test]
    fn stop_criteria_single_each_fires() {
        // max_steps alone: fires at its own step.
        let run_dir = std::env::temp_dir().join("gras-race-steps");
        let mut eng = engine(&run_dir, 5).unwrap();
        eng.config.max_steps = Some(20);
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        seed_raw_fitness(&mut eng, 0.5);
        assert_eq!(eng.check_stop(19), None, "not fired yet");
        assert_eq!(eng.check_stop(20), Some(StopReason::MaxSteps));
    }

    #[test]
    fn single_stop_criterion_still_works() {
        let run_dir = std::env::temp_dir().join("gras-race-single");
        let mut eng = engine(&run_dir, 5).unwrap();
        eng.config.max_steps = Some(7);
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        seed_raw_fitness(&mut eng, 0.5);
        assert_eq!(eng.check_stop(6), None);
        assert_eq!(eng.check_stop(7), Some(StopReason::MaxSteps));
    }

    // ── Post-race pruner (pop_pruner) ─────────────────────────────────────

    fn pruned_engine(
        run_dir: &std::path::Path,
        seed: u64,
        keep: usize,
        solo: usize,
    ) -> crate::engine::TabularEngine {
        let data_dir = tiny_dataset_dir("pruner");
        let config = RaceConfig {
            pop_size: 0,
            elite_count: keep,
            max_steps: Some(2),
            pop_pruner: Some(crate::engine::config::PopPruner {
                method: crate::engine::config::PopPrunerMethod::Hard,
                steps: solo,
            }),
            ..RaceConfig::defaults()
        };
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            data_dir,
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(seed),
            Some(run_dir.to_path_buf()),
        ))
        .unwrap()
    }

    #[test]
    fn pruner_disabled_stops_at_max_steps() {
        let run_dir = std::env::temp_dir().join("gras-pruner-off");
        let _ = std::fs::remove_dir_all(&run_dir);
        let mut eng = engine(&run_dir, 42).unwrap();
        eng.config.max_steps = Some(2);
        eng.config.pop_pruner = None;
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let reason = eng.run().unwrap();
        assert_eq!(reason, StopReason::MaxSteps);
        assert_eq!(
            eng.state.live_count(),
            3,
            "pruner off: nobody is culled at stop"
        );
    }

    #[test]
    fn pruner_hard_culls_to_elites_and_trains_solo_steps() {
        let run_dir = std::env::temp_dir().join("gras-pruner-hard");
        let _ = std::fs::remove_dir_all(&run_dir);
        let mut eng = pruned_engine(&run_dir, 42, 1, 3);
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let reason = eng.run().unwrap();
        assert_eq!(reason, StopReason::MaxSteps);
        // Race steps 0..=2 (stop checks fire AFTER the step at max_steps) +
        // 3 solo steps — the elites trained THROUGH the phase.
        let live = eng.state.live_hashes();
        assert_eq!(live.len(), 1, "Hard pruner keeps only the top-1");
        let survivor = eng.state.net(&live[0]).unwrap();
        assert_eq!(survivor.step, 3 + 3, "survivor advanced through solo phase");
        // The pruner phase is recorded too: culls as `pruner` rows in
        // attempts.csv, solo steps as ordinary history.csv metric rows.
        let history = std::fs::read_to_string(run_dir.join("history.csv")).unwrap();
        let attempts = std::fs::read_to_string(run_dir.join("attempts.csv")).unwrap();
        assert!(
            attempts.lines().any(|l| l.contains("pruner")),
            "culls are recorded as pruner attempt rows"
        );
        for solo_step in [3usize, 4, 5] {
            assert!(
                history
                    .lines()
                    .any(|l| l.starts_with(&format!("{solo_step},"))),
                "solo step {solo_step} appears as a history.csv metric row"
            );
        }
        // Freeze bypass: with freeze_elites on (the default), the survivor is
        // an elite by construction — the pruner phase must still TRAIN it,
        // not act-and-measure it. No frozen spans may cover the solo steps.
        let survivor = eng.state.net(&live[0]).unwrap();
        for solo_step in [3usize, 4, 5] {
            assert!(
                !survivor.is_frozen_step(solo_step),
                "solo step {solo_step} must be a REAL trained step, not frozen"
            );
        }
        // The pruner path RETURNS from `finish_race` before its own counter
        // stamp, so the phase persists them itself: the culls it performed
        // must be in engine.json, not only in the in-memory counter.
        let json = std::fs::read_to_string(run_dir.join("engine.json")).unwrap();
        let header: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            header["culls"].as_u64().unwrap() as usize,
            eng.culls,
            "pruner culls are stamped into engine.json"
        );
        assert!(eng.culls > 0, "the pruner did cull someone");
    }

    #[test]
    fn pruner_solo_steps_train_even_when_freeze_elites_is_on() {
        let run_dir = std::env::temp_dir().join("gras-pruner-freeze-on");
        let _ = std::fs::remove_dir_all(&run_dir);
        // freeze_elites defaults to TRUE — exactly the configuration where
        // the old bug froze every solo step (all survivors are elites). The
        // optimizer-moves proxy: the survivor takes REAL steps, so its
        // trained-step count (step − frozen) advances through the phase.
        let mut eng = pruned_engine(&run_dir, 42, 1, 3);
        assert!(eng.config.freeze_elites, "freeze_elites default is on");
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let live = {
            let reason = eng.run().unwrap();
            assert_eq!(reason, StopReason::MaxSteps);
            eng.state.live_hashes()
        };
        let survivor = eng.state.net(&live[0]).unwrap();
        // Dual assertion: freeze DOES engage during the race (the top net is
        // frozen from the first step where an elite set exists — proving the
        // flag below is a real bypass, not freeze being off) but NONE of the
        // pruner solo steps are frozen — they took real optimizer steps.
        let race_frozen = (0..=2).filter(|s| survivor.is_frozen_step(*s)).count();
        assert!(
            race_frozen > 0,
            "freeze engaged during the race (the top net was frozen at least once)"
        );
        for solo_step in [3usize, 4, 5] {
            assert!(
                !survivor.is_frozen_step(solo_step),
                "solo step {solo_step} must train (real optimizer) even with freeze_elites on"
            );
        }
        assert!(
            !eng.frozen_crown.contains(&live[0]),
            "no crown membership should persist from the bypassed phase"
        );
    }

    #[test]
    fn pruner_keeps_top_elite_count_nets() {
        let run_dir = std::env::temp_dir().join("gras-pruner-two");
        let _ = std::fs::remove_dir_all(&run_dir);
        let mut eng = pruned_engine(&run_dir, 42, 2, 1);
        eng.seed_population_internal(
            vec![
                tiny_topology(7),
                tiny_topology(8),
                tiny_topology(9),
                tiny_topology(10),
            ],
            Some(0.5),
        )
        .unwrap();
        eng.run().unwrap();
        assert_eq!(
            eng.state.live_count(),
            2,
            "elite_count=2 keeps two nets racing"
        );
    }

    #[test]
    fn pruner_phase_never_fires_challenges() {
        let dir = std::env::temp_dir().join("gras-pruner-no-challenge");
        let _ = std::fs::remove_dir_all(&dir);
        // The decay ALONE would not save us here: `max_steps = None` keeps
        // `p_eff` at the full knob for every step, and the race stops early
        // via `custom_stop`. So any solo challenge would have to come from an
        // explicit suppression, which is what this pins.
        let mut config = rl_config(0);
        config.pop_size = 2;
        config.challenge_prob = 1.0;
        config.max_steps = None;
        config.crossover_rolls = 0;
        config.mutate_rolls = 0;
        config.custom_stop = Some(Box::new(|snap: &crate::engine::config::RaceSnapshot| {
            snap.step >= 2
        }));
        config.pop_pruner = Some(crate::engine::config::PopPruner {
            method: crate::engine::config::PopPrunerMethod::Hard,
            steps: 3,
        });
        let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
            config,
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            ChallengingRlTrainer,
            Some(11),
            Some(dir.clone()),
        ))
        .unwrap();
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        assert_eq!(eng.run().unwrap(), StopReason::CustomStop);
        // history.csv carries a fitness per (step, net): the sentinel −1000
        // marks a challenged step (ChallengingRlTrainer), and the solo steps
        // 3..=5 must be free of it. No evolve rolls ⇒ every row is a founder
        // (plain `founder-N` origin, no quoting to parse around).
        let history = std::fs::read_to_string(dir.join("history.csv")).unwrap();
        let mut race_sentinels = 0usize;
        let mut solo_rows = 0usize;
        for line in history.lines().skip(1) {
            let cols: Vec<&str> = line.split(',').collect();
            let step: usize = cols[0].parse().unwrap();
            let fitness: f32 = cols[7].parse().unwrap();
            if step >= 3 {
                solo_rows += 1;
                assert_ne!(
                    fitness, -1000.0,
                    "pruner solo step {step} fired a challenge"
                );
            } else if fitness == -1000.0 {
                race_sentinels += 1;
            }
        }
        assert!(solo_rows > 0, "the pruner phase recorded solo rows");
        assert!(
            race_sentinels > 0,
            "the knob WAS live during the race (p_eff = 1.0)"
        );
        assert_eq!(
            eng.total_challenged_turns,
            race_sentinels * 5,
            "challenged-turn total counts race steps only"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Both CSV exports must be self-consistent: every data row has exactly as
    /// many fields as its own header. The old single-file layout padded
    /// attempt rows with a hand-counted run of commas; one comma too many
    /// shifted `branch`…`pop_size` a column left for every attempt row (and
    /// left the real live count with no header). Field-count equality is the
    /// invariant that catches that class of bug.
    #[test]
    fn csv_exports_have_aligned_columns() {
        fn fields(line: &str) -> usize {
            // Quote-aware counter: `csv_field` wraps origins containing commas.
            let mut n = 1;
            let mut quoted = false;
            for ch in line.chars() {
                match ch {
                    '"' => quoted = !quoted,
                    ',' if !quoted => n += 1,
                    _ => {}
                }
            }
            n
        }
        let run_dir = std::env::temp_dir().join("gras-csv-alignment");
        let _ = std::fs::remove_dir_all(&run_dir);
        let mut eng = pruned_engine(&run_dir, 42, 1, 2);
        // A configured metric widens the history header — the shape most
        // likely to drift from the attempt layout.
        eng.metrics = vec![crate::engine::fitness::Metric::custom("dummy", |_p, _y| {
            Ok(0.0)
        })];
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        eng.run().unwrap();
        for file in ["history.csv", "attempts.csv"] {
            let text = std::fs::read_to_string(run_dir.join(file)).unwrap();
            let mut lines = text.lines();
            let expected = fields(lines.next().expect("header"));
            assert_eq!(expected, fields(lines.clone().next().unwrap_or_default()));
            let mut rows = 0;
            for line in lines {
                assert_eq!(
                    fields(line),
                    expected,
                    "{file} row has the wrong field count: {line}"
                );
                rows += 1;
            }
            assert!(rows > 0, "{file} has data rows");
        }
        // The attempt ledger really landed in its own file (the pruner culls).
        let attempts = std::fs::read_to_string(run_dir.join("attempts.csv")).unwrap();
        assert!(attempts.lines().any(|l| l.contains("pruner")));
        let _ = std::fs::remove_dir_all(&run_dir);
    }

    /// Smoke test for the explicit `{train, test}` two-dataset layout: a
    /// data dir holding both subdirs resolves through
    /// `resolve_train_test_datasets`, the stream's eval/gating pools contain
    /// ONLY test rows (train/eval disjointness is guaranteed by the layout,
    /// not a shuffle), and the header records `explicit_test_rows` so resume
    /// rebuilds the same pools.
    #[test]
    fn explicit_train_test_dirs_build_disjoint_pools() {
        // Build a data dir with train/ (80 rows) + test/ (20 rows) — same
        // distribution, disjoint ROW SETS (train rows 0..80, test rows 80..100
        // of one synthetic pool; the point is the DIRECTORY layout, not the
        // values).
        let pool =
            crate::utils::tabular_data::synthetic_classification(100, 2, 2, 9, flodl::Device::CPU)
                .unwrap();
        let split_at = 80;
        let inputs = pool.inputs.to_f32_vec().unwrap();
        let targets = pool.targets.to_f32_vec().unwrap();
        let in_dim = 2usize;
        let out_dim = 2usize;
        let slice = |rows: std::ops::Range<usize>| {
            let xi: Vec<f32> = rows
                .clone()
                .flat_map(|r| inputs[r * in_dim..(r + 1) * in_dim].to_vec())
                .collect();
            let yi: Vec<f32> = rows
                .clone()
                .flat_map(|r| targets[r * out_dim..(r + 1) * out_dim].to_vec())
                .collect();
            crate::utils::tabular_data::Dataset {
                inputs: flodl::Tensor::from_f32(
                    &xi,
                    &[(rows.len()) as i64, in_dim as i64],
                    flodl::Device::CPU,
                )
                .unwrap(),
                targets: flodl::Tensor::from_f32(
                    &yi,
                    &[(rows.len()) as i64, out_dim as i64],
                    flodl::Device::CPU,
                )
                .unwrap(),
            }
        };
        let data_dir = std::env::temp_dir().join(format!("gras-train-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data_dir);
        crate::utils::tabular_data::save_dataset(&data_dir.join("train"), &slice(0..split_at))
            .unwrap();
        crate::utils::tabular_data::save_dataset(&data_dir.join("test"), &slice(split_at..100))
            .unwrap();

        let run_dir = std::env::temp_dir().join("gras-train-test-run");
        let _ = std::fs::remove_dir_all(&run_dir);
        let config = RaceConfig {
            pop_size: 0,
            max_steps: Some(2),
            ..RaceConfig::defaults()
        };
        let mut eng =
            crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
                data_dir.clone(),
                config,
                fitness(),
                crate::trainer::TabularTrainer::new(loss_fn()),
                Some(7),
                Some(run_dir.clone()),
            ))
            .unwrap();
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        eng.run().unwrap();

        // The engine took the explicit path: header records the test-row count.
        let header = crate::state::load_engine_json(&run_dir).unwrap();
        assert_eq!(
            header.explicit_test_rows,
            Some(100 - split_at),
            "engine.json must record the explicit layout"
        );
        // The stream's pools are disjoint BY CONSTRUCTION: eval pool rows all
        // sit in the test range [80..100) of the combined dataset.
        let stream = eng.stream.as_ref().unwrap();
        let (train_pool, eval_pool, gating_pool) = stream.pools();
        let test_start = split_at as i64;
        for row in eval_pool {
            assert!(
                *row >= test_start,
                "eval pool row {row} leaks into the train range"
            );
        }
        for row in gating_pool {
            assert!(*row >= test_start, "gating row {row} leaks into train");
        }
        for row in train_pool {
            assert!(*row < test_start, "train pool row {row} enters test range");
        }
        // Resume rebuilds the SAME pools from the header (no re-detection).
        let resumed = crate::engine::TabularEngine::resume(
            run_dir.clone(),
            data_dir.clone(),
            RaceConfig {
                pop_size: 2,
                ..RaceConfig::defaults()
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        let rs = resumed.stream.as_ref().unwrap().pools();
        assert_eq!(eval_pool, rs.1, "resume: identical eval pool");
        assert_eq!(gating_pool, rs.2, "resume: identical gating pool");
        assert_eq!(train_pool, rs.0);

        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&run_dir);
    }

    #[test]
    fn pruner_is_deterministic() {
        let (dir_a, dir_b) = (
            std::env::temp_dir().join("gras-pruner-det-a"),
            std::env::temp_dir().join("gras-pruner-det-b"),
        );
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
        let mut a = pruned_engine(&dir_a, 7, 1, 2);
        let mut b = pruned_engine(&dir_b, 7, 1, 2);
        let pop = vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)];
        a.seed_population_internal(pop.clone(), Some(0.5)).unwrap();
        b.seed_population_internal(pop, Some(0.5)).unwrap();
        a.run().unwrap();
        b.run().unwrap();
        assert_eq!(
            a.state.live_hashes(),
            b.state.live_hashes(),
            "same seed ⇒ same survivor"
        );
        let ha = a.state.net(&a.state.live_hashes()[0]).unwrap();
        let hb = b.state.net(&b.state.live_hashes()[0]).unwrap();
        assert_eq!(ha.step, hb.step);
        assert_eq!(
            ha.last_metrics.as_ref().map(|m| m.fitness),
            hb.last_metrics.as_ref().map(|m| m.fitness),
            "same seed ⇒ identical solo trajectory"
        );
    }

    // ── Founder topologies ──────────────────────────────────────────────

    #[test]
    fn founders_fill_slots_first_and_random_fills_the_rest() {
        let dir = std::env::temp_dir().join("gras-runtopos-mix");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.config.pop_size = 3;
        eng.config.run_topologies = vec![tiny_topology(101), tiny_topology(102)];
        // Rebuild the population the way from_spec mixes founders: founders
        // first, random batch after, truncated to pop_size.
        let random_batch =
            crate::engine::population::initial_population(&eng.config, eng.run_seed());
        let mut mixed = eng.config.run_topologies.clone();
        mixed.extend(random_batch);
        mixed.truncate(eng.config.pop_size);
        eng.seed_population_internal(mixed, Some(0.5)).unwrap();
        assert_eq!(eng.state.live_count(), 3);
        // The first two slots carry the founders' exact topology JSON.
        let founder_json: Vec<String> = eng
            .config
            .run_topologies
            .iter()
            .map(|t| t.to_json().unwrap())
            .collect();
        let live: Vec<String> = eng
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| eng.state.net(h).map(|s| s.topology.clone()))
            .collect();
        assert!(
            live.contains(&founder_json[0]) && live.contains(&founder_json[1]),
            "both founder blueprints must be in the population"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_topology_count_is_recorded_in_engine_json() {
        let dir = std::env::temp_dir().join("gras-runtopos-count");
        let _ = std::fs::remove_dir_all(&dir);
        let data_dir = tiny_dataset_dir("runtopos-count");
        let config = RaceConfig {
            run_topologies: vec![tiny_topology(11)],
            ..RaceConfig::defaults()
        };
        let eng =
            crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
                data_dir,
                config,
                fitness(),
                crate::trainer::TabularTrainer::new(loss_fn()),
                Some(7),
                Some(dir.clone()),
            ))
            .unwrap();
        let header = crate::state::load_engine_json(&dir).unwrap();
        assert_eq!(header.config.run_topology_count, 1);
        // Founder labels: the ordinal order of seeding is founders-first.
        assert_eq!(eng.step_clock(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn founder_topology_loader_reads_netstate_json() {
        // Round-trip: a net state written by the engine's own persistence
        // must be loadable back as a founder blueprint.
        let dir = std::env::temp_dir().join("gras-runtopos-loader");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        eng.write_live_frontier_states().unwrap();
        let nets_dir = dir.join("nets");
        let state_path = std::fs::read_dir(&nets_dir)
            .unwrap()
            .flatten()
            .next()
            .unwrap()
            .path();
        let topo = crate::engine::population::run_topology_from_json_file(&state_path).unwrap();
        assert_eq!(
            topo.to_json().unwrap(),
            tiny_topology(7).to_json().unwrap(),
            "loader must reproduce the exact blueprint"
        );
        // Run-dir loader: one ranked founder from the same run.
        let found = crate::engine::population::run_topologies_from_run_dir(&dir, 1).unwrap();
        assert_eq!(found.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Checkpoint elite weight snapshot ────────────────────────────────

    #[test]
    fn checkpoint_elite_snapshot_lands_and_flag_off_skips() {
        for (enabled, expect_file) in [(true, true), (false, false)] {
            let dir = std::env::temp_dir().join(format!("gras-ckpt-elite-{enabled}"));
            let _ = std::fs::remove_dir_all(&dir);
            let mut eng = engine(&dir, 7).unwrap();
            eng.config.checkpoint_every = 2;
            eng.config.max_steps = Some(4);
            eng.config.elite_checkpoint_weights = enabled;
            eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
                .unwrap();
            eng.run().unwrap();
            if expect_file {
                // Champion hash from the engine — the same source the
                // snapshot writer used.
                let champ = eng.champion_hashes()[0].clone();
                let short = &champ[..8];
                let snap = dir.join(format!("checkpoint-elite-{short}.safetensors"));
                assert!(
                    snap.exists(),
                    "checkpoint snapshot missing for champion {short}"
                );
            } else {
                let snaps = std::fs::read_dir(&dir)
                    .unwrap()
                    .flatten()
                    .filter(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .starts_with("checkpoint-elite-")
                    })
                    .count();
                assert_eq!(snaps, 0, "flag off ⇒ no checkpoint elite files");
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    // ── Mutation probation ─────────────────────────────────────────────

    #[test]
    fn probation_shields_fresh_nets_from_inverse_roulette() {
        let dir = std::env::temp_dir().join("gras-probation-shield");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.config.mutation_probation_steps = 3;
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        // Give everyone a verdict, one clearly worst (Minimize: highest =
        // worst). Without probation, the inverse roulette would always
        // target it.
        let mut hashes = eng.state.live_hashes();
        hashes.sort();
        for (i, h) in hashes.iter().enumerate() {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(0.5 + i as f32 * 0.1);
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        }
        eng.state.net_mut(&hashes[2]).unwrap().entered_at_step = 10; // fresh
        let clock = 11; // within the 3-step window
        for _ in 0..20 {
            let victim = eng.select_mutation_victim(clock, 0).unwrap();
            assert_ne!(
                victim, hashes[2],
                "a net inside its probation steps must never be the victim"
            );
        }
        // Past the window the protection lifts — the fresh net becomes an
        // eligible candidate again (the roulette is probabilistic, so we
        // assert eligibility via the predicate, not a single draw).
        let late = 13; // 13 − 10 = 3 ≥ window 3
        assert!(
            !eng.on_probation(&hashes[2], late),
            "protection expires after k clocks"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn probation_breaks_when_every_non_elite_is_fresh() {
        // A firing roll must ALWAYS find a slot: with the whole non-elite
        // population inside its window, the last resort picks the worst
        // on-probation net rather than failing.
        let dir = std::env::temp_dir().join("gras-probation-break");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.config.mutation_probation_steps = 100; // effectively everyone
        eng.config.elite_count = 1;
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let mut hashes = eng.state.live_hashes();
        hashes.sort();
        for (i, h) in hashes.iter().enumerate() {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(0.5 + i as f32 * 0.1);
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        }
        let victim = eng.select_mutation_victim(5, 0).unwrap();
        // The elite (best = hashes[0]) must STILL be protected; the worst of
        // the two remaining probation nets takes the slot.
        assert_ne!(
            victim, hashes[0],
            "probation never overrides the elite guard"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn probation_steps_is_recorded_in_engine_json() {
        let dir = std::env::temp_dir().join("gras-probation-json");
        let _ = std::fs::remove_dir_all(&dir);
        let data_dir = tiny_dataset_dir("probation-json");
        let config = RaceConfig {
            mutation_probation_steps: 4,
            ..RaceConfig::defaults()
        };
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            data_dir,
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(7),
            Some(dir.clone()),
        ))
        .unwrap();
        let header = crate::state::load_engine_json(&dir).unwrap();
        assert_eq!(header.config.mutation_probation_steps, 4);
        assert!(header.config.elite_checkpoint_weights);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn crossover_gate_window_limits_bars_and_persists() {
        // The window must (a) slice the ledger to the last k RECORDED
        // checkpoints for both gates and (b) round-trip through engine.json.
        let dir = std::env::temp_dir().join("gras-cx-gate-window");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 7).unwrap();
        engine.config.crossover_gate_window = 2;
        // Seed a 5-entry ledger with distinct, ordered bars (Minimize: lower
        // = better — so the last-2 mean differs sharply from the all-5 mean).
        for (i, step) in [10usize, 20, 30, 40, 50].iter().enumerate() {
            engine.checkpoints.push(Checkpoint {
                step: *step,
                pop_mean_fitness: 1.0 + i as f32,
                exam_mean_fitness: 0.0,
            });
        }
        let all: Vec<(usize, Checkpoint)> = engine
            .checkpoints
            .iter()
            .enumerate()
            .map(|(i, c)| (i, *c))
            .collect();
        let gate_k = engine.config.crossover_gate_window;
        let windowed = &all[all.len() - gate_k..];
        assert_eq!(windowed.len(), 2);
        assert_eq!(windowed.len(), 2);
        // The all-bars mean (3.0) vs windowed mean: window actually changed
        // the verdict input.
        let all_mean = all.iter().map(|(_, c)| c.pop_mean_fitness).sum::<f32>() / 5.0;
        assert!((all_mean - 3.0).abs() < 1e-6);
        let win_mean = windowed
            .iter()
            .map(|(_, c)| c.pop_mean_fitness)
            .sum::<f32>()
            / 2.0;
        assert!(
            (win_mean - 4.5).abs() < 1e-6,
            "last two bars are 4.0 and 5.0 → mean 4.5 (got {win_mean})"
        );
        // Persist + reload: the knob is replay-relevant.
        let cfg = crate::engine::config::RaceConfig {
            crossover_gate_window: 7,
            ..crate::engine::config::RaceConfig::defaults()
        };
        let snap = crate::state::ConfigSnapshot::from_config(&cfg, Some(32), Some(32));
        assert_eq!(snap.crossover_gate_window, 7);
        assert_eq!(
            crate::engine::config::RaceConfig::defaults().crossover_gate_window,
            0,
            "default = unbounded (legacy behavior)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
