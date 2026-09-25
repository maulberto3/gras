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
//! 6. Stop criteria checked (max_steps, max_target_fitness, custom_stop) —
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
//!   There is no per-step file write during the run, and no separate
//!   `history.csv` or `culled.log` artifact.

use flodl::nn::Optimizer;
use flodl::tensor::Result;
use log::info;
use std::collections::HashMap;

use crate::engine::fitness::{Fitness, FitnessLabel, Metric};
use crate::graph::network::Network;
use crate::state::{
    ConfigSnapshot, NetMetrics, NetState, RaceState, RunConfig, RunHeader, write_engine_json,
    write_net_state,
};
use crate::trainer::stream::{BatchStream, PoolSplit};

/// Placeholder swapped into `self.trainer` for the duration of the pop-wide
/// phase, so the real trainer can be taken out of `self` without aliasing
/// the `&mut Network` borrows. Its `pop_phase` is the trait's no-op default
/// and its other methods are never called (it lives inside `run` for a few
/// lines only).
struct NoopPopTrainer;
impl crate::trainer::StepTrainer for NoopPopTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
    }
}
impl crate::trainer::RlStep for NoopPopTrainer {
    fn train_step(
        &mut self,
        _net: &mut Network,
        _optimizer: &mut dyn Optimizer,
        _step: usize,
        _ctx: &crate::trainer::RlContext<'_>,
    ) -> flodl::tensor::Result<crate::trainer::StepReport> {
        unreachable!("NoopPopTrainer is a placeholder only for the pop-wide phase")
    }
}
// Used by the debug contract probe in `step_one_net` and by the checkpoint
// surprise exam (`run_checkpoint_exam`) — the exam runs in release too, so
// this import is NOT debug-gated.
use crate::utils::race_steps::eval_one_step;

// ── Build identity ──────────────────────────────────────────────────────────

/// One-line description of the RUNNING binary: crate version, profile, and
/// the executable path + its mtime. Printed at start and recorded in
/// `engine.json`.
///
/// Why the mtime matters: an artifact error (or a "parity failed") is often a
/// STALE PROCESS, not stale logic — a long run started before a fix keeps the
/// old behavior for hours, and its artifacts look wrong for no visible
/// reason. Stamping the exe + build time makes "was this the new code?" a
/// one-glance question instead of an investigation.
pub(crate) fn build_stamp() -> String {
    let exe = std::env::current_exe().ok();
    let built = exe
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| {
            let secs = d.as_secs();
            // Compact, timezone-free UTC stamp (YYYY-MM-DD HH:MM:SS).
            let (days, rem) = (secs / 86_400, secs % 86_400);
            let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
            let (y, mo, d) = civil_from_days(days as i64);
            format!("{y:04}-{mo:02}-{d:02} {h:02}:{m:02}:{s:02}Z")
        })
        .unwrap_or_else(|| "unknown".into());
    let path = exe
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "-".into());
    format!(
        "gras {} [{}] built {built} — exe {path}",
        env!("CARGO_PKG_VERSION"),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    )
}

/// Days-since-epoch → (year, month, day), proleptic Gregorian (Howard Hinnant's
/// `civil_from_days`). Self-contained so the engine needs no date dependency.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ── Display helpers ─────────────────────────────────────────────────────────

/// Format a float to 2 decimals for the **console log only**. Artifacts
/// (`engine.json`, `nets/<hash>.json`, `history.csv`, `checkpoints.json`) are
/// always written from the raw `f32` — serde and `Display` emit the shortest
/// decimal that round-trips, i.e. full precision. Never route anything
/// persisted through this function.
fn fmt2(v: f32) -> String {
    format!("{v:.2}")
}

/// Format an optional float to 2 decimals — plain `—` when unset (never
/// `Some(...)` in user-facing logs).
fn fmt_opt2(v: &Option<f32>) -> String {
    v.as_ref()
        .map(|x| format!("{x:.2}"))
        .unwrap_or_else(|| "—".into())
}

/// RFC4180-quote a CSV field when it holds a delimiter, quote, or newline.
/// Lineage strings (`crossover:parents=h1,h2`) contain commas, so this is not
/// optional — an unquoted lineage would shift every later column.
fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

use super::child::RaceChild;
use super::config::{RaceConfig, RaceSnapshot, StopReason};
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
    pub(crate) trainer: crate::trainer::ModeTrainer,
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
    /// Anti-devolution D: per-net entry fitness floor. Set lazily at the
    /// net's first verdict (smoothed fitness right after birth / insertion /
    /// gate pass — the moment its "birth-right" is measured). Cleared on
    /// cull. Consulted after each scoring pass when `regression_tol` is on.
    pub(crate) fitness_floors: HashMap<String, f32>,
    /// Anti-devolution D: hashes demoted THIS step (regressed below floor).
    /// Elite status is denied while demoted (a collapsed net can't hold the
    /// crown); culling needs NO special queue — the collapsed fitness already
    /// up-weights the net in the victim roulette. Rebuilt each
    /// step after scoring.
    pub(crate) demoted: std::collections::HashSet<String>,
    /// Anti-devolution A: the nets currently holding the freeze crown (the
    /// top-`elite_count` when `freeze_elites` is on — named by the ★ badge on
    /// the rollup line). Empty until the first frozen step. Tracked so the
    /// "crowned / crown moved" lines fire on TRANSITIONS only, not every
    /// step, and so the badge can name every champion (with `elite_count > 1`
    /// there is more than one).
    pub(crate) frozen_crown: std::collections::HashSet<String>,
    /// `Minimal` mode: last step's population means (train, eval, fitness)
    /// so the framed table can show what changed this step. `None` until the
    /// first table render (the table itself starts at step 2 for this reason).
    pub(crate) minimal_prev_means: Option<(f32, f32, f32)>,
    /// Buffer history rows in memory to minimize slow disk I/O writes. One
    /// unified event log (`history.csv`): per-step live-net metric rows AND
    /// evolution attempt rows (inserted or rejected), distinguished by the
    /// leading `type` column.
    pub(crate) history_csv_buffer: String,
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

/// Per-step RL volume, summed over every net that stepped this clock — the
/// environment work the population actually did. `matches`/`turns` come from
/// `StepReport.rl` (see [`crate::trainer::RlStepMeta`]); `nets` counts the nets
/// that reported, so a trainer which forgets to report shows as `—` instead of
/// a silent `0`. Always zero in Tabular mode.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RlVolume {
    /// Nets that reported `StepReport.rl` this step.
    pub nets: usize,
    /// Matches played by those nets, summed.
    pub matches: usize,
    /// Environment turns played by those nets, summed.
    pub turns: usize,
}

impl RlVolume {
    /// The RL middle column of the per-step rollup, e.g.
    /// `matches 100 │ turns 14400 │ turns/match 144`. `—` when no net reported
    /// any match (Tabular mode, or a mis-wired RL trainer).
    fn label(&self) -> String {
        if self.nets == 0 || self.matches == 0 {
            return "matches — │ turns — │ turns/match —".to_string();
        }
        format!(
            "matches {} │ turns {} │ turns/match {:.0}",
            self.matches,
            self.turns,
            self.turns as f32 / self.matches as f32
        )
    }
}

impl CoreEngine {
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
                crate::trainer::ModeTrainer::Tabular(s.trainer),
                s.seed,
                s.run_dir,
                Some(s.data_dir),
            ),
            RS::RL(s) => (
                s.config,
                s.fitness,
                crate::trainer::ModeTrainer::Rl(s.trainer),
                s.seed,
                s.run_dir,
                None,
            ),
        };
        // The config's RunMode and the spec variant must AGREE. The spec
        // variant decides the mechanics (data vs env); `set_run_mode(..)` decides
        // the recorded/validated intent. A mismatch is a config bug — fail
        // loudly at construction rather than running the wrong paradigm.
        // (Image/NLP modes are future variants of RunSpec — they must be
        // declared in config but have no spec variant yet, hence they're
        // rejected too. The DEFAULT Tabular mode is exempt: it matches both
        // variants' zero-config story — an RL spec with an unset mode fails,
        // forcing the user to declare intent.)
        if tabular_data_dir.is_none() && config.mode != crate::engine::config::RunMode::Rl {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                "RunSpec::rl requires .set_run_mode(RunMode::Rl) on the config — the declared mode and the spec variant must agree".to_string(),
            )
            .into());
        }
        if tabular_data_dir.is_some() && config.mode != crate::engine::config::RunMode::Tabular {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                format!(
                    "RunSpec::tabular (a data_dir was given) requires .set_run_mode(RunMode::Tabular) — config declares {:?} with no way to serve it; Image/NLP spec variants are future work",
                    config.mode
                ),
            )
            .into());
        }
        // RL mode REQUIRES a reported fitness: there is no dataset, so a
        // (pred, target) scorer has nothing to score. Fail before the run
        // starts, not three steps in.
        if tabular_data_dir.is_none() && fitness.is_computed() {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                "RL spec requires Fitness::reported(direction, label) — a Computed (pred, target) scorer has no dataset to score against. The trainer must supply the fitness value in StepReport.".to_string(),
            )
            .into());
        }
        // Fresh-start immigrants are an RL-only idea: catch-up exists so a
        // mid-run insertion is comparable to the population, and in Tabular
        // that comparability is sacred (one shared data stream — starting
        // mid-stream silently skips training rows). Only the engine knows the
        // mode AND the knob, so this is where the conflict fails loudly.
        if tabular_data_dir.is_some() && config.immigrant_fresh_start {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                "set_immigrant_fresh_start(true) requires RL mode: in tabular, an immigrant that skips catch-up misses shared training data, breaking fitness comparability with the population".to_string(),
            )
            .into());
        }
        let dataset = match &tabular_data_dir {
            Some(data_dir) => Some(
                crate::utils::tabular_data::resolve_dataset(data_dir)?
                    .to_device(config.device())?,
            ),
            None => None,
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

        // Seed the initial population internally — transparent to the user,
        // like the old generational engine where `run` was the only call.
        // Generated before the engine consumes the config.
        let initial_topologies = super::population::initial_population(&config, run_seed);

        // Shared batch stream — engine infrastructure (Tabular only). The
        // trainer may shape batch sizes (`Trainer::stream_shape`) but the
        // split ratio is NOT overridable: which rows are held out protects
        // fitness comparability across every net, whatever the recipe.
        // RL mode has no dataset — `ctx.data` is `None` and the trainer is
        // fully responsible for its own experience (the env it drives).
        let batch_stream = match &dataset {
            Some(dataset) => {
                let split = PoolSplit::of(dataset, train_eval_split_ratio, run_seed);
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
            config: RaceConfig { ..config },
            meta_ctx,
            stream: batch_stream,
            fitness,
            metrics,
            state: RaceState::new(),
            history_csv_buffer: String::new(),
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
            champions: Vec::new(),
            checkpoints: Vec::new(),
            fitness_floors: HashMap::new(),
            demoted: std::collections::HashSet::new(),
            frozen_crown: std::collections::HashSet::new(),
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
                 ranking semantics would change; set set_fitness_smoothing_window({}) (or omit) to resume",
                self.header.config.smoothing_window,
                self.config.smoothing_window,
                self.header.config.smoothing_window
            )).into());
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
                log::debug!(
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
                        log::warn!(
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
        for ordinal in 0..topologies.len() {
            let topo = &topologies[ordinal];
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
            stream.set_checkpoint_every(self.config.checkpoint_every);
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
                "race knobs: gate={:?} cull_policy={:?} retries={} dropout={} freeze_elites={} regression_tol={} fresh_immigrants={} elite_save={} worst_save={}",
                self.config.crossover_gate,
                self.config.crossover_cull_policy,
                self.config.crossover_retries,
                self.header.topology_options.dropout_prob,
                self.config.freeze_elites,
                self.config
                    .regression_tol
                    .map(|t| format!("{t}"))
                    .unwrap_or_else(|| "off".into()),
                self.config.immigrant_fresh_start,
                self.config.elite_save_topology,
                self.config.worst_save_topology,
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
            if !self.trainer.is_tabular() {
                // Take the trainer OUT of self so the &mut Network borrows
                // (self.networks) and the &mut trainer call don't alias.
                let mut trainer = std::mem::replace(
                    &mut self.trainer,
                    crate::trainer::ModeTrainer::Rl(Box::new(NoopPopTrainer)),
                );
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
                    log::warn!("checkpoint ledger write failed: {e}");
                }
                if let Err(e) = self.write_live_frontier_states() {
                    log::warn!("checkpoint live states write failed: {e}");
                }
                // Flush metrics at the checkpoint too: an interrupted run then
                // keeps the per-step history up to its last checkpoint instead
                // of losing the whole in-memory buffer.
                if let Err(e) = self.flush_metrics_csv() {
                    log::warn!("checkpoint metrics flush failed: {e}");
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
                    info!(
                        "step {} │ checkpoint │ pop_mean_fitness {} {:.4} │ {}exam_mean_fitness {} (ledger: {} entries)",
                        clock,
                        arrow,
                        mean,
                        bar_txt,
                        exam_txt,
                        self.checkpoints.len(),
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
        // `pruner.steps` more steps. Evolution and stop criteria are
        // phase-locked OFF: they are evolution-phase concerns and the
        // surviving nets race no one.
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
        self.flush_metrics_csv()?;
        self.persist_run_counters();
        Ok(reason)
    }

    /// Stamp the run-level counters (`culls`, accumulated wall time,
    /// `children_born_at_clock`) into `engine.json` so a resume restores
    /// them instead of starting fresh — a resumed run continues the ORIGINAL
    /// cull budget, wall-clock age, and child-seed disambiguation state.
    /// Best-effort: a write failure is a warning, not a stop failure (the
    /// frontier and ledger are already safely on disk at this point).
    fn persist_run_counters(&self) {
        let mut header = self.header.clone();
        header.culls = self.culls;
        header.run_elapsed_secs = self.elapsed_base_secs + self.started_at_wall.elapsed().as_secs();
        header.children_born_at_clock = self.children_born_at_clock.clone();
        if let Err(e) = crate::state::write_engine_json(&self.run_dir, &header) {
            log::warn!("engine.json counter persistence failed: {e}");
        }
    }

    /// Post-race pruner phase (Hard method): cull all but the top
    /// `elite_count` live nets, then keep training the survivors for
    /// `pruner.steps` extra steps with evolution and stop criteria off.
    ///
    /// The survivors continue through the SAME per-net step path (`step_one_net`)
    /// with the SAME trainer, optimizer state, and shared stream — only the
    /// orchestration differs: no evolve rolls fire (the race is over), and
    /// `check_stop` is not consulted (its signals are population-level and
    /// meaningless on 1–2 nets; std on a 1-net population is literally 0).
    /// Every solo step is recorded exactly like a race step — history.csv
    /// metric rows and the survivors' `nets/<hash>.json` step counters — so
    /// the post-race extension is a visible, replayable part of the run.
    fn run_pruner_phase(
        &mut self,
        reason: StopReason,
        clock: usize,
        pruner: crate::engine::config::PopPruner,
    ) -> Result<StopReason> {
        // Keep the top-k elites (k = max(1, elite_count)) — the same ranking
        // the elite guard uses, so "who survives" is exactly "who was elite".
        let keep = {
            let k = self.config.elite_count.max(1).min(self.state.live_count());
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
            ranked
                .into_iter()
                .take(k)
                .map(|(h, _)| h)
                .collect::<Vec<_>>()
        };
        // Worst-net dump BEFORE the cull: the anti-champion only exists
        // while the full field is alive — after culling to elites there is
        // no "worst" left to distinguish from the elite.
        self.write_worst_artifacts()?;
        let victims: Vec<String> = self
            .state
            .live_hashes()
            .into_iter()
            .filter(|h| !keep.contains(h))
            .collect();
        if self.verbose_detail() {
            info!(
                "── pruner phase ── race stop: {:?} at step {} → keeping top-{} ({}), culling {} → solo training for {} steps",
                reason,
                clock,
                keep.len(),
                keep.iter()
                    .map(|h| h[..8].to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                victims.len(),
                pruner.steps,
            );
        }
        for v in &victims {
            // Attempt row first (victim identity readable), then the cull —
            // same discipline as the crossover insert path.
            let victim_seed = self.state.net(v).map(|s| s.net_seed);
            self.record_attempt(
                clock,
                "pruner",
                0,
                None,
                None,
                "pruned",
                "pruned",
                None,
                None,
                None,
                Some(v.as_str()),
                victim_seed,
            );
            self.cull_net(v, clock, "pruned")?;
        }
        if !victims.is_empty() {
            self.flush_metrics_csv()?;
            self.write_live_frontier_states()?;
        }
        if pruner.steps == 0 {
            self.write_champion_markdown()?;
            self.write_champion_safetensors()?;
            if self.verbose_detail() {
                info!("  pruner phase: 0 steps configured — nothing further to train");
            }
            return Ok(reason);
        }
        // Solo extension: the same group-step shape, minus evolution and stop
        // checks. Optimizer state is intact (nothing rebuilt) — the elites
        // simply keep learning at the same LR/hyperparams.
        for offset in 0..pruner.steps {
            let step = clock + 1 + offset;
            // Solo steps evolve nothing: both per-step accumulators start
            // clean, so the rollup reports this step's own RL volume and the
            // (already logged) pruner culls are not re-counted every step.
            self.step_evolve = StepEvolve::default();
            self.step_rl = RlVolume::default();
            // The rollup's `took` must be THIS step's cost. The race loop
            // resets the clock at the top of every iteration; the pruner
            // loop had no reset, so its lines reported cumulative wall time
            // since the race started (a pop-2 solo step showing "took 52.3s"
            // when it really took ~0.9s). Same reset, same meaning.
            self.step_started_at_wall = std::time::Instant::now();
            let hashes = self.state.live_hashes();
            if hashes.is_empty() {
                break;
            }
            for hash in &hashes {
                self.step_one_net(hash, step)?;
            }
            self.append_metrics_csv(step)?;
            if self.log_level != crate::engine::config::LogLevel::None {
                self.log_step_rollup(step);
            }
        }
        let final_step = clock + pruner.steps;
        self.write_champion_markdown()?;
        self.write_champion_safetensors()?;
        // NOTE: no worst dump here — the population was already culled to
        // elites above (the worst was dumped pre-cull, before that).
        if self.verbose_detail() {
            info!(
                "── pruner phase complete ── trained to step {} ({} solo step(s))",
                final_step, pruner.steps,
            );
            self.log_stop_summary(final_step)?;
        }
        self.write_live_frontier_states()?;
        self.flush_metrics_csv()?;
        Ok(reason)
    }

    // ── One net's step ──────────────────────────────────────────────────────

    /// Stop-time summary: which JSONs were written as the live frontier and
    /// the resume command to continue from here.
    fn log_stop_summary(&self, clock: usize) -> Result<()> {
        let live = self.state.live_hashes();
        info!(
            "  wrote {} frontier snapshot(s) → nets/<hash>.json (one per live net)",
            live.len(),
        );
        // The population size is part of the resume call (the engine refuses a
        // mismatch instead of silently patching it), so print the number the
        // caller must pass — not just the directory.
        info!(
            "  {} live net(s) at step {} → resume with: --resume {} --pop {}",
            live.len(),
            clock,
            self.run_dir.display(),
            live.len(),
        );

        // Final elite report: the top-k nets by smoothed fitness, with their
        // metadata — the run's champions at stop time. Elite = rank, not
        // identity, so this is "who holds the top-k right now".
        let k = self.config.elite_count.max(1); // report at least the champion
        let mut ranked: Vec<(String, f32, usize, Option<f32>)> = live // (hash, smoothed, step, eval_loss)
            .iter()
            .filter_map(|h| {
                let buf = self.rolling_fitness.get(h)?;
                if buf.is_empty() {
                    return None;
                }
                let step = self
                    .state
                    .net(h)
                    .and_then(|s| s.last_metrics.as_ref())
                    .map(|m| m.step)
                    .unwrap_or(0);
                let eval_loss = self
                    .state
                    .net(h)
                    .and_then(|s| s.last_metrics.as_ref())
                    .and_then(|m| m.eval_loss);
                Some((h.clone(), rolling_mean(buf), step, eval_loss))
            })
            .collect();
        let direction = self.fitness.direction();
        ranked.sort_by(|a, b| direction.cmp(b.1, a.1));
        let arrow = direction.arrow();
        info!("  ── final elites (top-{k}, fitness{arrow} smoothed) ──");
        for (rank, (h, fit, step, eval_loss)) in ranked.iter().take(k).enumerate() {
            // A frozen elite's last_metrics.step stays at its FIRST record
            // (training is skipped, so nothing advances the recorded clock).
            // Plain `last_step 0` therefore reads like "never played again"
            // — say `frozen@0` instead: measured once, carried since.
            let step_label = if self.config.freeze_elites && self.frozen_crown.contains(h) {
                format!("frozen@{step}")
            } else {
                format!("last_step {step}")
            };
            let meta = self.state.net(h).map(|s| {
                format!(
                    "origin={} born@{} params={}",
                    s.created_from.clone().unwrap_or_else(|| "?".into()),
                    s.entered_at_step,
                    s.meta
                        .params
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "?".into()),
                )
            });
            info!(
                "  #{} {} fitness{arrow} {:.4} │ eval_loss {:?} │ {} │ {}",
                rank + 1,
                &h[..8.min(h.len())],
                fit,
                eval_loss,
                step_label,
                meta.unwrap_or_default(),
            );
        }
        Ok(())
    }

    /// Write the top-k elites' topology markdown to the run dir, one file
    /// per elite (`elite-<hash>.md`), k = `elite_count` (min 1). The user
    /// configured more than one elite — they get more than one artifact.
    /// UNCONDITIONAL — called at stop regardless of log level: the `.md` file
    /// is a run artifact (like `nets/<hash>.json` and `history.csv`), not a
    /// log line. Prints a confirmation only when per-step detail is on.
    fn write_champion_markdown(&mut self) -> Result<()> {
        if !self.config.elite_save_topology {
            return Ok(());
        }
        self.record_champions();
        let ranked = self.champion_ranked();
        let k = ranked.len();
        for (rank, (hash, fitness)) in ranked.iter().enumerate() {
            let Some(state) = self.state.net(hash) else {
                continue;
            };
            let Ok(topo) = state.topology() else {
                continue;
            };
            let short = &hash[..8.min(hash.len())];
            let path = self.run_dir.join(format!("elite-{short}.md"));
            let label = if k <= 1 {
                "Elite".to_string()
            } else {
                format!("Elite #{}", rank + 1)
            };
            let md = format!(
                "**{label} · fitness {} (smoothed) = {:.4}**\n\n{}",
                self.fitness.direction().arrow(),
                fitness,
                crate::utils::markdown::topology_markdown(&topo, None),
            );
            match std::fs::write(&path, &md) {
                Ok(()) => {
                    if self.verbose_detail() {
                        info!(
                            "  elite topology → {} (markdown, ready to view)",
                            path.display()
                        );
                    }
                }
                Err(source) => log::warn!(
                    "elite markdown write failed: {}",
                    crate::utils::error::EngineError::Io {
                        path: path.display().to_string(),
                        source,
                    }
                ),
            }
        }
        Ok(())
    }

    /// Write the top-k elites' weights as `.safetensors`, one file per elite
    /// (`elite-<hash>.safetensors`), k = `elite_count` (min 1). UNCONDITIONAL
    /// — same artifact discipline as [`Self::write_champion_markdown`]: lands
    /// on disk at every log level. Exports each elite's live in-memory
    /// `Network` — byte-faithful coefficients, exactly what the race left
    /// them with.
    fn write_champion_safetensors(&mut self) -> Result<()> {
        if !self.config.elite_save_safetensors {
            return Ok(());
        }
        self.record_champions();
        let ranked = self.champion_ranked();
        for (hash, _) in &ranked {
            let short = &hash[..8.min(hash.len())];
            let path = self.run_dir.join(format!("elite-{short}.safetensors"));
            // The elite's live Network is still in memory at stop — export
            // it directly, no rebuild/replay needed. Its coefficients are the
            // exact ones the race left it with (byte-faithful by construction).
            match self.networks.get(hash.as_str()) {
                Some(net) => {
                    if let Err(e) = crate::utils::safetensors::export_safetensors(net, &path) {
                        log::warn!("elite safetensors export failed: {e}");
                    } else if self.verbose_detail() {
                        info!("  elite weights → {} (safetensors)", path.display());
                    }
                }
                None => log::warn!("elite safetensors export failed: live network missing"),
            }
        }
        Ok(())
    }

    /// Full hashes of the elite set exported at stop — the single source of
    /// truth for "which nets are the champions". Read this (not file mtimes,
    /// not glob order) when post-race tooling needs the champion: the frontier
    /// snapshot writes all live nets in HashMap order, so "latest file" is
    /// arbitrary. Empty until the first champion dump.
    pub fn champion_hashes(&self) -> &[String] {
        &self.champions
    }

    /// The champion set: top-`elite_count` live hashes by smoothed fitness.
    /// The ONE ranking both elite writers consume, so `elite-<hash>.md`,
    /// `elite-<hash>.safetensors` and `champion_hashes()` can never disagree.
    fn champion_ranked(&self) -> Vec<(String, f32)> {
        let k = self.config.elite_count.max(1);
        self.rank_live().into_iter().take(k).collect()
    }

    /// Snapshot the current champion set into `self.champions` before any
    /// artifact writing, so `champion_hashes()` reflects exactly the set the
    /// writers iterate — even if an individual file write fails.
    fn record_champions(&mut self) {
        self.champions = self.champion_ranked().into_iter().map(|(h, _)| h).collect();
    }

    /// Rank live nets by smoothed fitness, best first. Shared by the elite
    /// artifact writers and (with `.last()`) the worst-net dump.
    fn rank_live(&self) -> Vec<(String, f32)> {
        let mut ranked: Vec<(String, f32)> = self
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| {
                let buf = self.rolling_fitness.get(h)?;
                if buf.is_empty() {
                    return None;
                }
                Some((h.clone(), rolling_mean(buf)))
            })
            .collect();
        let direction = self.fitness.direction();
        ranked.sort_by(|a, b| direction.cmp(b.1, a.1));
        ranked
    }

    /// When any `worst_save_*` flag is set: save the WORST live net's
    /// artifacts — `worst-<hash>.md` (topology markdown) and/or
    /// `worst-<hash>.safetensors` (weights), each behind its own flag. The
    /// anti-champion is the net the search avoided — its topology is often
    /// the cheapest way to see what the fitness signal rejected.
    ///
    /// MUST be called BEFORE any cull shrinks the population: the worst is
    /// ranked over the full pre-cull field (e.g. the pop_pruner culls to
    /// elites, so a post-cull call would find no worst to dump).
    fn write_worst_artifacts(&self) -> Result<()> {
        if !self.config.worst_save_topology && !self.config.worst_save_safetensors {
            return Ok(());
        }
        let ranked = self.rank_live();
        let Some((worst, worst_fitness)) = ranked.last() else {
            return Ok(()); // nothing ever scored — nothing to dump
        };
        let short = &worst[..8.min(worst.len())];
        if self.config.worst_save_topology {
            if let Some(state) = self.state.net(worst) {
                if let Ok(topo) = state.topology() {
                    let path = self.run_dir.join(format!("worst-{short}.md"));
                    let md = format!(
                        "**Worst · fitness {} (smoothed) = {:.4}**\n\n{}",
                        self.fitness.direction().arrow(),
                        worst_fitness,
                        crate::utils::markdown::topology_markdown(&topo, None),
                    );
                    if let Err(source) = std::fs::write(&path, &md) {
                        log::warn!(
                            "worst markdown write failed: {}",
                            crate::utils::error::EngineError::Io {
                                path: path.display().to_string(),
                                source,
                            }
                        );
                    }
                }
            }
        }
        if self.config.worst_save_safetensors {
            let path = self.run_dir.join(format!("worst-{short}.safetensors"));
            match self.networks.get(worst.as_str()) {
                Some(net) => {
                    if let Err(e) = crate::utils::safetensors::export_safetensors(net, &path) {
                        log::warn!("worst safetensors export failed: {e}");
                    } else if self.verbose_detail() {
                        info!("  worst-net artifacts → worst-{short}.md / .safetensors");
                    }
                }
                None => log::warn!("worst safetensors export failed: live network missing"),
            }
        }
        Ok(())
    }

    /// Step one live net at the given step_clock: train on the shared train
    /// batch, eval on the shared eval batch, record metrics into `RaceState`,
    /// push fitness into the rolling buffer, overwrite `nets/<hash>.json`,
    /// log one per-net line.
    ///
    /// The net's `Network` + `Optimizer` live in the engine's maps and are
    /// mutated in place — coefficients evolve across steps, optimizer state
    /// carries forward. **Not rebuilt each step** (that would lose optimizer
    /// state and forget prior training).
    fn step_one_net(&mut self, hash: &str, clock: usize) -> Result<()> {
        let net_seed = {
            let state = self.state.net(hash).unwrap();
            state.net_seed as u64
        };
        // ANTI-DEVOLUTION A — elite freeze (computed BEFORE the optimizer
        // borrow): top-k nets skip the trainer call entirely. Their skill is
        // the frozen state, so the "report" is the last recorded metrics —
        // re-measuring would require re-running the trainer, which is
        // exactly what we're skipping. Scoring, ranking, parent selection
        // all proceed as normal off the carried metrics.
        let frozen = self.config.freeze_elites && self.elite_hashes().contains(&hash.to_string());
        if frozen && !self.frozen_crown.contains(hash) {
            // Crown transition: this net newly joined the frozen set (first
            // crowning or a child trained past a frozen elite). One info
            // line, only on the transition; stable steps are silent (the ★
            // badge on the rollup line carries the state).
            let fitness = self
                .state
                .net(hash)
                .and_then(|s| s.last_metrics.as_ref().map(|m| m.fitness));
            let seat = self.elite_hashes().len();
            let pos = self
                .elite_hashes()
                .iter()
                .position(|h| h == hash)
                .map(|p| p + 1)
                .unwrap_or(0);
            if seat == 1 {
                // Single-elite wording (the common case): crown moves as one.
                if self.frozen_crown.is_empty() {
                    info!(
                        "step {} │ freeze │ champion {} crowned (fitness{} {}) — training skipped from here",
                        clock,
                        &hash[..8.min(hash.len())],
                        self.fitness.direction().arrow(),
                        fitness.map(fmt2).unwrap_or_else(|| "—".into()),
                    );
                } else {
                    let prev = self.frozen_crown.iter().next().cloned().unwrap_or_default();
                    info!(
                        "step {} │ freeze │ crown moved {} → {} (fitness{} {}) — previous champion resumes training",
                        clock,
                        &prev[..8.min(prev.len())],
                        &hash[..8.min(hash.len())],
                        self.fitness.direction().arrow(),
                        fitness.map(fmt2).unwrap_or_else(|| "—".into()),
                    );
                }
            } else {
                // Multi-elite: say WHICH seat the net just took. No
                // "crown moved" language — with k seats, one joining does
                // not imply another was dethroned.
                info!(
                    "step {} │ freeze │ elite seat {pos}/{seat}: {} frozen (fitness{} {}) — training skipped from here",
                    clock,
                    &hash[..8.min(hash.len())],
                    self.fitness.direction().arrow(),
                    fitness.map(fmt2).unwrap_or_else(|| "—".into()),
                );
            }
            self.frozen_crown.insert(hash.to_string());
        }
        if frozen {
            if let Some(metrics) = self.state.net(hash).and_then(|s| s.last_metrics.clone()) {
                if let Some(buf) = self.rolling_fitness.get_mut(hash) {
                    buf.push(metrics.fitness);
                }
                if let Some(buf) = self.rolling_eval.get_mut(hash) {
                    if let Some(e) = metrics.eval_loss {
                        buf.push(e);
                    }
                }
                log::debug!(
                    "step {} │ net {} │ FROZEN (elite) — carrying fitness{} {}",
                    clock,
                    &hash[..8.min(hash.len())],
                    self.fitness.direction().arrow(),
                    fmt2(metrics.fitness),
                );
                return Ok(());
            }
            // No last_metrics yet (frozen before its first verdict — only
            // possible at step 0): fall through and train once so the net
            // has a skill state worth freezing.
        }
        // One step = whatever the caller's training scheme does for one step
        // clock. The engine only consumes the returned report.
        let optimizer = self.optimizers.get_mut(hash).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Optimizer in memory)"
            ))
        })?;
        // The mode decides the context: Tabular receives the run's data
        // handle (guaranteed present by construction), RL receives none. The
        // context itself is built inside `ModeTrainer::train_step`.
        let run_data = self
            .dataset
            .as_ref()
            .zip(self.stream.as_ref())
            .map(|(dataset, stream)| crate::trainer::RunData { dataset, stream });
        let env = crate::trainer::StepEnv {
            step: clock,
            run_seed: self.header.run_seed,
            pop_size: self.config.pop_size,
            live_count: self.state.live_count(),
            checkpoint_every: self.config.checkpoint_every,
            smoothing_window: self.config.smoothing_window,
        };
        let net = self.networks.get_mut(hash).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Network in memory)"
            ))
        })?;
        // Seed BOTH RNGs for this (net, step) before the trainer runs: gras's
        // fastrand stream and libtorch's global RNG (dropout masks). Without
        // the libtorch seed a `dropout_prob > 0` net redraws different masks
        // on replay and breaks resume parity. The engine owns this because
        // the trainer — any trainer — may draw masks inside `train_step`.
        crate::utils::race_steps::seed_step_randomness(net_seed, clock as u64, 0);
        let report = self.trainer.train_step(
            net,
            optimizer.as_mut(),
            clock,
            run_data.as_ref(),
            &self.fitness,
            &self.metrics,
            env,
            hash,
            net_seed,
        )?;
        // Debug-only contract probe: the Trainer's per-step clause says the
        // report describes the net AFTER this step's training. Re-score the
        // net on the step's eval batch and check the reported eval loss is
        // what this net actually scores — catches a stale/fabricated report
        // at development time. Zero cost in release.
        #[cfg(debug_assertions)]
        // Tabular-only: the probe needs an eval batch + a computed fitness
        // scorer. RL mode has neither (reported fitness, no dataset).
        if let (Some(reported), Some(net), Some(stream), Some(dataset), true) = (
            report.eval_loss,
            self.networks.get_mut(hash),
            self.stream.as_ref(),
            self.dataset.as_ref(),
            self.fitness.is_computed(),
        ) {
            let eval_batch = stream
                .eval_batch(dataset, clock as u64)
                .expect("probe: eval batch");
            if let (Some(loss), Some(fit)) = (
                self.trainer.as_tabular().map(|t| t.loss()),
                Some(&self.fitness),
            ) {
                if let Ok(actual) = eval_one_step(net, loss, fit, &self.metrics, &eval_batch) {
                    if let Some(actual_loss) = actual.eval_loss {
                        let rel = ((reported - actual_loss).abs()) / actual_loss.abs().max(1e-6);
                        assert!(
                            rel < 1e-3,
                            "trainer contract violated: step {clock} net {hash} reported eval_loss {reported:.6} but the net scores {actual_loss:.6} — the StepReport must describe the net's state at the end of this step"
                        );
                    }
                }
            }
        }
        // RL volume: the trainer's reported environment work for this step
        // accumulates across the group so the rollup can show the step's real
        // workload (matches + turns). Tabular trainers report `None`.
        if let Some(rl) = report.rl {
            self.step_rl.nets += 1;
            self.step_rl.matches += rl.matches;
            self.step_rl.turns += rl.turns;
        }
        let metrics = NetMetrics {
            step: clock,
            train_loss: report.train_loss,
            eval_loss: report.eval_loss,
            fitness: report.fitness,
            informative: report.informative,
        };
        self.state.record_step(hash, metrics.clone())?;
        if let Some(buf) = self.rolling_fitness.get_mut(hash) {
            buf.push(metrics.fitness);
        }
        if let Some(buf) = self.rolling_train.get_mut(hash) {
            buf.push(metrics.train_loss);
        }
        if let Some(buf) = self.rolling_eval.get_mut(hash) {
            if let Some(e) = metrics.eval_loss {
                buf.push(e);
            }
        }
        // ANTI-DEVOLUTION D — regression demotion: lazily set the net's
        // floor at its first verdict (the "birth-right" measurement), then
        // each step check whether it has collapsed below floor × tol.
        if self.config.regression_tol.is_some() {
            self.update_regression_guard(hash, clock);
        }
        // Per-net detail is debug-only (Plan A: the user log shows one rollup
        // line per step — see `run`). All values remain in nets/<hash>.json.
        let dir = self.fitness.direction().arrow();
        log::debug!(
            "step {} │ net {} │ train_loss↓ {} │ eval_loss↓ {} │ fitness{} {}",
            clock,
            hash,
            fmt2(metrics.train_loss),
            fmt_opt2(&metrics.eval_loss),
            dir,
            fmt2(metrics.fitness),
        );
        Ok(())
    }

    /// One rollup line per step: pop size, the train-loss mean ± std, the
    /// mode's middle column, and the fitness column in the same shape. Two
    /// decimals throughout — the spread is the signal worth reading, the
    /// extremes are noise. Printed LAST for the step (see the run loop), so
    /// it lands at the bottom of the terminal.
    ///
    /// Middle column by mode: Tabular = held-out `eval_loss`; RL = the step's
    /// environment volume (`matches`/`turns`, from [`RlVolume`]) — an RL net
    /// has no held-out batch, and the volume is what explains the wall time.
    fn log_step_rollup(&mut self, clock: usize) {
        // Smoothed (K-step rolling mean) population stats — raw per-step
        // values bounce with batch difficulty; the trend is what matters.
        let mut trains = Vec::new();
        let mut evals = Vec::new();
        for h in self.state.live_hashes() {
            if let Some(buf) = self.rolling_train.get(h.as_str()) {
                trains.push(rolling_mean(buf));
            }
            if let Some(buf) = self.rolling_eval.get(h.as_str()) {
                if !buf.is_empty() {
                    evals.push(rolling_mean(buf));
                }
            }
        }
        // Population mean ± population std, 2 decimals.
        let stats = |v: &[f32]| {
            if v.is_empty() {
                "—".to_string()
            } else {
                let n = v.len() as f32;
                let mean = v.iter().sum::<f32>() / n;
                let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
                format!("{mean:.2} ± {:.2}", var.sqrt())
            }
        };
        // Per-step wall time: the step's own cost (train + evolve), not time
        // since run start. Matters most in RL (one step = many full matches).
        let step_secs = self.step_started_at_wall.elapsed().as_secs_f32();
        // Fitness column: same mean ± std shape so all three read alike.
        let fits: Vec<f32> = self
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| {
                self.rolling_fitness
                    .get(h.as_str())
                    .filter(|b| !b.is_empty())
            })
            .map(rolling_mean)
            .collect();
        let fit_stats = stats(&fits);
        // The mode's middle column: tabular reports the held-out eval loss,
        // RL reports the environment volume it actually played.
        let mid = if self.trainer.is_tabular() {
            format!("eval_loss↓ {}", stats(&evals))
        } else {
            self.step_rl.label()
        };
        // Per-step rollup: Summ = compact one-liner; Minimal = framed table
        // (from step 2 — deltas need a prior step). The ★ badge names the
        // CURRENT elite set (top-`elite_count` by smoothed fitness this
        // step) — it explains who is immune to culls right now. With
        // `freeze_elites` on, every elite seat is frozen each step, so the
        // set needs no per-net marker: the badge lists the names alone.
        let freeze_badge = {
            let elites = self.elite_hashes();
            if elites.is_empty() {
                String::new()
            } else {
                let mut names: Vec<String> = elites
                    .iter()
                    .map(|h| h[..8.min(h.len())].to_string())
                    .collect();
                names.sort();
                format!(" │ ★ {}", names.join(" "))
            }
        };
        if self.log_level == crate::engine::config::LogLevel::Minimal {
            self.log_minimal_table(&trains, &evals, &fits, step_secs);
        } else {
            info!(
                "step {} │ pop {} │ train_loss↓ {} │ {} │ fitness{} {} │ took {:.1}s{}",
                clock,
                self.state.live_count(),
                stats(&trains),
                mid,
                self.fitness.direction().arrow(),
                fit_stats,
                step_secs,
                freeze_badge,
            );
        }
    }

    /// `Minimal` mode: one comfy framed table per step (starting at step 2 —
    /// step 1 has no prior step to diff against). Values "update in place":
    /// the shape is identical every step, only the numbers move. Deltas reuse
    /// what the engine already reports (population-mean smoothed values vs the
    /// last step); a zero delta drops the parenthetical. The evolve counters
    /// and the best-net footer are plain current values — no delta on the
    /// footer. Nothing else is printed per step in this mode.
    ///
    /// Row 1's tail is mode-dependent, exactly like the `Summ` line: tabular
    /// shows the held-out `eval_loss`, RL shows the step's environment volume
    /// (`matches`/`turns`). Rendered at the END of the step, so the evolve
    /// counters in the frame are this step's own.
    fn log_minimal_table(&mut self, trains: &[f32], evals: &[f32], fits: &[f32], step_secs: f32) {
        let mean = |v: &[f32]| {
            if v.is_empty() {
                f32::NAN
            } else {
                v.iter().sum::<f32>() / v.len() as f32
            }
        };
        let (train_m, eval_m, fit_m) = (mean(trains), mean(evals), mean(fits));
        // Delta vs the previous step's mean; omitted entirely when 0. The
        // first table render (step 2) sets the baseline without deltas.
        // Only surface a delta that survives 2-decimal rounding, so the table
        // never shows a meaningless `(∆+0.00)`.
        let delta = |cur: f32, prev: Option<f32>| match prev {
            Some(p) if (cur - p).abs() >= 5e-3 => format!(" (∆{:+.2})", cur - p),
            _ => String::new(),
        };
        let (pt, pe, pf) = self
            .minimal_prev_means
            .unwrap_or((f32::NAN, f32::NAN, f32::NAN));
        let pop = self.state.live_count();
        let e = self.step_evolve;
        // Build the rows first, THEN size the box: a fixed width overflowed
        // on long delta lines, swallowed the right border, and glued rows
        // together on one terminal line. Padding is char-count based — the
        // content itself is ASCII (the ↓/↑/∆ glyphs are terminal-drawn in
        // the labels, and the pad uses the raw byte length which matches
        // since every glyph here is 3-byte UTF-8 but consistently present
        // per column) — so alignment holds.
        let mut rows: Vec<String> = vec![
            if self.trainer.is_tabular() {
                format!(
                    "pop {:>3} │ train_loss↓ {:.2}{} │ eval_loss↓ {:.2}{}",
                    pop,
                    train_m,
                    delta(train_m, Some(pt)),
                    eval_m,
                    delta(eval_m, Some(pe)),
                )
            } else {
                format!(
                    "pop {:>3} │ train_loss↓ {:.2}{} │ {}",
                    pop,
                    train_m,
                    delta(train_m, Some(pt)),
                    self.step_rl.label(),
                )
            },
            format!(
                "fitness{} {:.2}{} │ culls {} │ inserts {}",
                self.fitness.direction().arrow(),
                fit_m,
                delta(fit_m, Some(pf)),
                e.culls,
                e.inserts,
            ),
            format!(
                "crossover fired {} ({} inserted, {} spent) │ mutation rolled {} ({} immigrant(s))",
                e.cross_fired, e.cross_survived, e.cross_discarded, e.mutate_fired, e.mutate_fired,
            ),
            format!("step took {:.1}s", step_secs),
        ];
        // Anti-devolution row — only when a knob is on (zero noise by
        // default): the elite set and/or the current demotion count. Same
        // annotation style as the Summ line's badge.
        let mut guard_cells: Vec<String> = Vec::new();
        if self.config.freeze_elites || self.config.elite_count > 0 {
            let elites = self.elite_hashes();
            let mut names: Vec<String> = elites
                .iter()
                .map(|h| h[..8.min(h.len())].to_string())
                .collect();
            names.sort();
            guard_cells.push(format!("elite: {}", names.join(" ")));
        }
        if self.config.regression_tol.is_some() {
            guard_cells.push(format!("demoted: {}", self.demoted.len()));
        }
        if !guard_cells.is_empty() {
            rows.push(guard_cells.join(" │ "));
        }
        // Display width: count chars, not bytes (↓/↑/∆ are 3 bytes, 1 char).
        let inner = rows
            .iter()
            .map(|r| r.chars().count())
            .max()
            .unwrap_or(0)
            .max(20);
        let pad = |s: &String| {
            let visible = s.chars().count();
            format!("│ {}{}│", s, " ".repeat(inner.saturating_sub(visible)))
        };
        let top = format!("┌{}┐", "-".repeat(inner + 2));
        let bot = format!("└{}┘", "-".repeat(inner + 2));

        println!("\x1b[2K\r{}", top);
        for r in &rows {
            println!("\x1b[2K\r{}", pad(r));
        }
        println!("\x1b[2K\r{}", bot);
        // Baseline for the next step's deltas.
        self.minimal_prev_means = Some((train_m, eval_m, fit_m));
    }

    // ── Smoothed fitness ────────────────────────────────────────────────────

    /// Per-net rolling-mean fitness over the live population, in
    /// `live_hashes()` order — the shared input for the checkpoint ledger,
    /// cull/insert ranking, and `RaceSnapshot`.
    fn smoothed_fitness_values(&self) -> Vec<f32> {
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

    /// The best smoothed fitness in the live population (under the fitness
    /// direction).
    fn best_smoothed_fitness(&self) -> f32 {
        let hashes = self.state.live_hashes();
        if hashes.is_empty() {
            return 0.0;
        }
        let direction = self.fitness.direction();
        let mut best = rolling_mean(self.rolling_fitness.get(&hashes[0]).unwrap());
        for h in &hashes[1..] {
            let v = rolling_mean(self.rolling_fitness.get(h).unwrap());
            if direction.is_better(v, best) {
                best = v;
            }
        }
        best
    }

    // ── Cull + insert ───────────────────────────────────────────────────────

    /// Fresh empty rolling buffer for a net about to be caught up, so
    /// catch-up's per-step pushes land somewhere — a live net must never be
    /// without a buffer entry.
    fn pre_insert_buffer(&mut self, hash: &str) {
        self.rolling_fitness.insert(
            hash.to_string(),
            RollingBuffer::new(self.config.smoothing_window),
        );
        self.rolling_train.insert(
            hash.to_string(),
            RollingBuffer::new(self.config.smoothing_window),
        );
        self.rolling_eval.insert(
            hash.to_string(),
            RollingBuffer::new(self.config.smoothing_window),
        );
    }

    /// A guaranteed-fresh random child (mutation path): roll `random_child`
    /// directly (bypassing the crossover/mutation dispatch, which the caller
    /// has already resolved), re-rolling on duplicate hashes (bounded).
    fn generate_random_at(&mut self, clock: usize, start_idx: usize) -> Result<RaceChild> {
        for attempt in 0..8 {
            let child = self.random_child(clock, start_idx + attempt)?;
            if !self.state.net(&child.state.hash).is_some() {
                // consume the ordinal(s) we used
                *self.children_born_at_clock.get_mut(&clock).unwrap() = start_idx + attempt + 1;
                return Ok(child);
            }
            if self.verbose_detail() {
                info!(
                    "step {} │ random child {} is a duplicate topology → re-rolling unique seed",
                    clock,
                    &child.state.hash[..8],
                );
            }
        }
        Err(flodl::tensor::TensorError::new(
            "race: could not generate a unique random child after 8 attempts",
        ))
    }

    /// Insert a caught-up child into the live maps (state, network, optimizer;
    /// its rolling buffer was pre-inserted before catch-up and is NOT reset).
    fn insert_child(&mut self, child: RaceChild, _clock: usize, _group: &str) {
        // NO per-event log here: inserts are reported on the calling roll's
        // single line — one roll, one line.
        let child_fitness = child.state.last_metrics.as_ref().map(|m| m.fitness);
        self.state.insert(child.state.clone(), child_fitness);
        self.networks.insert(child.state.hash.clone(), child.net);
        self.optimizers
            .insert(child.state.hash.clone(), child.optimizer);
    }

    /// One evolution roll of the crossover branch: generate a child, run the
    /// checkpoint-gated catch-up (early-out discard on the first failed
    /// gate), and on success cull one slot per the cull policy + insert the
    /// child. A failed child culls nothing — the gate IS the cull, and a
    /// failed attempt is a SPENT ROLL: no random fallback (random whole nets
    /// enter exclusively via the mutation rolls).
    /// Returns the roll outcome (survived + log detail); `attempt` is the
    /// 1-based retry ordinal (for history.csv).
    fn evolve_crossover_child(
        &mut self,
        clock: usize,
        _roll: usize,
        attempt: usize,
    ) -> Result<RollOutcome> {
        let idx = self.next_child_ordinal(clock);
        // Ok(spent) = spent roll (crossover produced nothing viable or a
        // duplicate of a live net): no cull, no insert, move on.
        let mut child = match self.generate_child(clock, idx)? {
            Some(c) => c,
            None => {
                return Ok(RollOutcome::spent(format!(
                    "no child: {} parent-pairing draw(s) × {} gate attempt(s) all incompatible (no compatible pivot/dims)",
                    crate::engine::child::MAX_PARENT_PAIRINGS,
                    attempt,
                )));
            }
        };
        if self.state.net(&child.state.hash).is_some() {
            // Duplicate of a live topology (crossover recreated an existing
            // net) → the roll is spent: discard, no replacement. Random
            // whole nets enter only via the mutation path.
            self.record_attempt(
                clock,
                "crossover",
                attempt,
                None,
                None,
                "duplicate-discarded",
                "rejected_duplicate",
                None,
                None,
                None,
                None,
                None,
            );
            return Ok(RollOutcome::spent(format!(
                "child {} duplicates a live topology → discarded",
                &child.state.hash[..8.min(child.state.hash.len())],
            )));
        }

        // Checkpoint-gated catch-up: the child must beat the recorded
        // population mean at the checkpoints between its birth and now
        // (Hard: every gate; Soft: the aggregate mean of the gate means).
        // Children born before any checkpoint exists skip the gate entirely.
        self.pre_insert_buffer(&child.state.hash);
        let mut replayed_to = 0usize;
        let mut last_checkpoint_step = 0usize;
        let mut child_fit_at_gate = f32::NAN;
        let relevant: Vec<(usize, Checkpoint)> = self
            .checkpoints
            .iter()
            .enumerate()
            .filter(|(_, chk)| chk.step <= clock)
            .map(|(i, chk)| (i, *chk))
            .collect();
        let checkpoint_count = relevant.len();
        let mut failed_gate: Option<(usize, usize, f32, f32)> = None; // (i+1, chk.step, child, mean)
        match self.config.crossover_gate {
            crate::engine::config::CrossoverGate::Hard => {
                // Evaluate gate-by-gate: replay to each checkpoint, compare.
                // PROVISIONAL replay: a rejected candidate must leave no
                // state file behind (the ghost-file resume bug).
                for (i, chk) in &relevant {
                    self.catch_up_range_provisional(&mut child, replayed_to, chk.step)?;
                    replayed_to = chk.step;
                    last_checkpoint_step = chk.step;
                    let child_fit = self
                        .rolling_fitness
                        .get(&child.state.hash)
                        .map(rolling_mean)
                        .unwrap_or(f32::NAN);
                    child_fit_at_gate = child_fit;
                    let beat = self
                        .fitness
                        .direction()
                        .is_better(child_fit, chk.pop_mean_fitness);
                    if !beat {
                        failed_gate = Some((i + 1, chk.step, child_fit, chk.pop_mean_fitness));
                        break;
                    }
                }
            }
            crate::engine::config::CrossoverGate::Soft => {
                // One aggregate bar: beat the mean of the checkpoint means.
                // Replay straight to the last relevant checkpoint, compare once.
                if let Some((_, last)) = relevant.last() {
                    self.catch_up_range_provisional(&mut child, replayed_to, last.step)?;
                    replayed_to = last.step;
                    last_checkpoint_step = last.step;
                    let mean_of_means = relevant
                        .iter()
                        .map(|(_, c)| c.pop_mean_fitness)
                        .sum::<f32>()
                        / checkpoint_count as f32;
                    let child_fit = self
                        .rolling_fitness
                        .get(&child.state.hash)
                        .map(rolling_mean)
                        .unwrap_or(f32::NAN);
                    child_fit_at_gate = child_fit;
                    let beat = self.fitness.direction().is_better(child_fit, mean_of_means);
                    if !beat {
                        failed_gate = Some((checkpoint_count, last.step, child_fit, mean_of_means));
                    }
                }
            }
        }
        if let Some((gate_i, gate_step, child_fit, bar)) = failed_gate {
            // "discarded" = the child was rejected by the checkpoint gate and
            // never joined the population; the pop did NOT shrink.
            // Record the rejected attempt (cx_retry_full measurement).
            self.record_attempt(
                clock,
                "crossover",
                attempt,
                Some(&child.state.hash),
                Some(child.state.net_seed),
                &child.state.created_from.clone().unwrap_or_default(),
                "rejected_gate",
                Some(gate_i),
                Some(child_fit),
                Some(bar),
                None,
                None,
            );
            // Drop the child's provisional buffers — it never joined.
            self.rolling_fitness.remove(&child.state.hash);
            self.rolling_train.remove(&child.state.hash);
            self.rolling_eval.remove(&child.state.hash);
            return Ok(RollOutcome::spent(format!(
                "child {} rejected by {} gate {}/{} ({}{:.4} vs bar {:.4}) → discarded at step {}",
                &child.state.hash[..8.min(child.state.hash.len())],
                match self.config.crossover_gate {
                    crate::engine::config::CrossoverGate::Hard => "hard",
                    crate::engine::config::CrossoverGate::Soft => "soft",
                },
                gate_i,
                checkpoint_count,
                self.fitness.direction().arrow(),
                child_fit,
                bar,
                gate_step,
            )));
        }
        let _ = last_checkpoint_step;
        // Passed every gate — finish the replay to the clock, resolve the
        // victim, record the attempt, then cull + insert.
        self.catch_up_range(&mut child, replayed_to, clock)?;
        // Slot eviction per CrossCullPolicy (crossover-only — the immigrant
        // channel has its own fitness-inverse victim selection): Worst =
        // merit-based (default); Random = uniform (diversity-first). In both
        // cases the elite guard excludes the top-k nets from victim status.
        // The victim is resolved BEFORE recording, so the attempt row can
        // carry its identity (fills the previously-empty `victim` column).
        let victim_for_log: Option<String> = match self.config.crossover_cull_policy {
            crate::engine::config::CrossCullPolicy::Worst => {
                self.worst_nets_by_smoothed_fitness(1)?.pop()
            }
            crate::engine::config::CrossCullPolicy::Random => {
                Some(self.select_random_victim(clock)?)
            }
        };
        // Record the admitted attempt (cx_retry_full measurement) — before
        // the cull/insert, while both victim and child states are readable.
        let victim_net_seed = victim_for_log
            .as_ref()
            .and_then(|v| self.state.net(v))
            .map(|s| s.net_seed);
        self.record_attempt(
            clock,
            "crossover",
            attempt,
            Some(&child.state.hash),
            Some(child.state.net_seed),
            &child.state.created_from.clone().unwrap_or_default(),
            "inserted",
            None,
            Some(child_fit_at_gate),
            None,
            victim_for_log.as_deref(),
            victim_net_seed,
        );
        match self.config.crossover_cull_policy {
            crate::engine::config::CrossCullPolicy::Worst => {
                if let Some(victim) = &victim_for_log {
                    self.cull_net(victim, clock, "crossover")?;
                }
            }
            crate::engine::config::CrossCullPolicy::Random => {
                if let Some(victim) = &victim_for_log {
                    self.cull_net(victim, clock, "crossover-random")?;
                }
            }
        }
        // The gate verdict belongs on the SUCCESS line too: what the child
        // scored vs the bar it beat is the whole story of the admission.
        let gate_note = if checkpoint_count > 0 {
            let bar = match self.config.crossover_gate {
                crate::engine::config::CrossoverGate::Hard => {
                    // Hard gate: the LAST checkpoint's bar was the final one beaten.
                    relevant
                        .last()
                        .map(|(_, c)| c.pop_mean_fitness)
                        .unwrap_or(f32::NAN)
                }
                crate::engine::config::CrossoverGate::Soft => {
                    // Soft gate: one aggregate bar — the mean of the checkpoint means.
                    relevant
                        .iter()
                        .map(|(_, c)| c.pop_mean_fitness)
                        .sum::<f32>()
                        / checkpoint_count.max(1) as f32
                }
            };
            format!(
                " | gate {} {:.4} vs bar {:.4}",
                match self.config.crossover_gate {
                    crate::engine::config::CrossoverGate::Hard => "hard",
                    crate::engine::config::CrossoverGate::Soft => "soft",
                },
                child_fit_at_gate,
                bar,
            )
        } else {
            String::new() // no checkpoints yet — gate skipped, nothing to show
        };
        let inserted_hash = child.state.hash[..8.min(child.state.hash.len())].to_string();
        let lineage = child
            .state
            .created_from
            .clone()
            .unwrap_or_else(|| "?".into());
        self.insert_child(child, clock, "crossover");
        Ok(RollOutcome::survived(format!(
            "child {} ({}) inserted{}, victim {} culled",
            inserted_hash,
            lineage,
            gate_note,
            victim_for_log
                .as_deref()
                .map(|v| v[..8.min(v.len())].to_string())
                .unwrap_or_else(|| "none".into()),
        )))
    }

    /// Append one evolution-event row to the history.csv buffer (cx_retry_full
    /// measurement). Every crossover/mutation attempt is recorded — inserted or
    /// rejected — so gate-failure rate, operator yield, and retry economics are
    /// queryable after the run. Buffered, flushed at the same points as the
    /// per-step metric rows (they share the unified history.csv).
    fn record_attempt(
        &mut self,
        step: usize,
        branch: &str,
        attempt: usize,
        child_hash: Option<&str>,
        child_net_seed: Option<usize>,
        origin: &str,
        outcome: &str,
        gate_index: Option<usize>,
        child_fitness: Option<f32>,
        bar: Option<f32>,
        victim: Option<&str>,
        victim_net_seed: Option<usize>,
    ) {
        if !self.config.csv_export {
            return;
        }
        // Attempt rows align with the unified header: type,step,hash,net_seed,
        // origin, then empty metric columns (entered_at_step/train_loss/
        // eval_loss/... — attempts have no per-step training state), then the
        // attempt tail (branch,attempt,outcome,gate_index,child_fitness,bar,
        // victim,victim_net_seed,pop).
        // FULL hashes + net_seed here (no truncation): (hash, net_seed) is the
        // unique individual key — the same topology hash can legitimately
        // appear in multiple attempt rows (regenerated children) and across
        // eras, so the log must not manufacture collisions.
        // entered_at_step + 3 metric cols + informative cols = empty slots.
        let empty_metrics = ",".repeat(4 + self.metrics.len());
        let row = format!(
            "attempt,{},{},{},{},{},\"{}\",{},{},{},{},{},{},{},{}\n",
            step,
            child_hash.unwrap_or_default(),
            child_net_seed.map(|s| s.to_string()).unwrap_or_default(),
            csv_field(origin),
            empty_metrics,
            branch,
            attempt,
            outcome,
            gate_index.map(|g| g.to_string()).unwrap_or_default(),
            child_fitness.map(|v| v.to_string()).unwrap_or_default(),
            bar.map(|v| v.to_string()).unwrap_or_default(),
            victim.unwrap_or_default(),
            victim_net_seed.map(|s| s.to_string()).unwrap_or_default(),
            self.state.live_count(),
        );
        self.history_csv_buffer.push_str(&row);
    }

    /// Uniformly random **cullable** live net — the `Random` arm of BOTH cull
    /// policies (crossover children and mutation immigrants). Deterministic:
    /// derived from `(run_seed, clock, salt)` so replays and resume pick the
    /// same victim. Elite nets are excluded from the draw (no child —
    /// crossover or immigrant — can evict an elite).
    ///
    /// `salt` separates the several rolls that fire within one clock: without
    /// it every roll of the same step would draw the same index.
    fn select_random_victim_at(&self, clock: usize, salt: usize) -> Result<String> {
        let elite = self.elite_hashes();
        let hashes: Vec<String> = self
            .state
            .live_hashes()
            .into_iter()
            .filter(|h| !elite.contains(h))
            .collect();
        if hashes.is_empty() {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                "random cull: no cullable (non-elite) live net".into(),
            )
            .into());
        }
        let seed = crate::utils::seed::derive_seed(
            self.header.run_seed,
            clock.wrapping_mul(7919).wrapping_add(salt),
        );
        let idx = fastrand::Rng::with_seed(seed).usize(..hashes.len());
        Ok(hashes[idx].clone())
    }

    /// [`Self::select_random_victim_at`] with salt 0 — the crossover path's
    /// original derivation, kept verbatim so its victim draws (and the tests
    /// pinning them) are unchanged.
    fn select_random_victim(&self, clock: usize) -> Result<String> {
        self.select_random_victim_at(clock, 0)
    }

    /// The mutation channel's victim, per [`crate::engine::config::MutationCullPolicy`].
    /// Every arm falls back to the same two last resorts (worst-net when no
    /// net has a fitness verdict yet, then first live) so a firing roll ALWAYS
    /// finds a slot for its immigrant. `roll` salts the `Random` arm so the
    /// rolls of one step don't all draw the same net.
    fn select_mutation_victim(&self, clock: usize, roll: usize) -> Result<String> {
        use crate::engine::config::MutationCullPolicy;
        let picked = match self.config.mutation_cull_policy {
            MutationCullPolicy::InverseFitness => self
                .select_inverse_proportional()?
                .or_else(|| self.mutation_victim_fallback()),
            MutationCullPolicy::Worst => self.mutation_victim_fallback(),
            // Salt `roll + 1`: the crossover path draws with salt 0, so a
            // mutation roll never silently mirrors a crossover victim.
            MutationCullPolicy::Random => self
                .select_random_victim_at(clock, roll + 1)
                .ok()
                .or_else(|| self.mutation_victim_fallback()),
        };
        picked.ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(
                "mutation cull: no live net to evict".into(),
            )
            .into()
        })
    }

    /// Mutation victim last resorts: the worst net by smoothed fitness, else
    /// the first live hash. `None` only when the population is empty (a firing
    /// roll cannot happen then).
    fn mutation_victim_fallback(&self) -> Option<String> {
        self.worst_nets_by_smoothed_fitness(1)
            .ok()
            .and_then(|mut w| w.pop())
            .or_else(|| self.state.live_hashes().first().cloned())
    }

    /// One evolution roll of the mutation branch: cull a net selected
    /// inversely-proportionate to fitness and insert a fully-random
    /// immigrant (no checkpoint gate — a random topology could never clear
    /// historical means; its job is diversity).
    ///
    /// When no net has a fitness verdict yet (fresh population, empty rolling
    /// buffers), falls back to culling the first live net — the immigrant
    /// still needs a slot, and a random cull is the only honest option when
    /// nothing distinguishes the population yet.
    fn evolve_random_immigrant(&mut self, clock: usize, roll: usize) -> Result<String> {
        // Pick a victim per the mutation cull policy (`InverseFitness` by
        // default — the historical fitness-inverse roulette; `Worst` for a
        // deterministic merit cull; `Random` for uniform turnover). The elite
        // guard makes the top-k immune in every arm.
        let victim = self.select_mutation_victim(clock, roll)?;
        // Anti-devolution D attribution: when the victim is a demoted net,
        // the tombstone says WHY — "regressed" instead of the generic slot
        // eviction reason. NOT a special cull lane (the guard no longer
        // front-loads the roulette): the collapsed fitness chose this victim
        // through the ordinary inverse-proportional draw; the floor check
        // just names it.
        let reason = if self.demoted.contains(&victim) {
            "regressed"
        } else {
            "immigrant-slot"
        };
        self.cull_net(&victim, clock, reason)?;
        let idx = self.next_child_ordinal(clock);
        let mut child = self.generate_random_at(clock, idx)?;
        self.pre_insert_buffer(&child.state.hash);
        // Catch-up (the default) replays steps 0..clock so the newborn is
        // comparable to the population. With `immigrant_fresh_start` it is
        // skipped instead: the immigrant keeps its fresh-init weights and
        // trains from the current clock on. Sound in RL (no shared data stream
        // to have missed — see the knob's docs); rejected for Tabular at
        // construction. Its empty rolling buffers already mean "no verdict
        // yet", so it cannot be culled or crowned before its first step.
        let fresh_start = self.config.immigrant_fresh_start;
        if fresh_start {
            child.state.step = 0;
            child.state.last_metrics = None;
        } else {
            self.catch_up(&mut child, clock)?;
        }
        let detail = if fresh_start {
            format!(
                "victim {} culled ({reason}), immigrant {} inserted (no gate, FRESH-START at step {clock} — no catch-up)",
                &victim[..8.min(victim.len())],
                &child.state.hash[..8.min(child.state.hash.len())],
            )
        } else {
            format!(
                "victim {} culled ({reason}), immigrant {} inserted (no gate)",
                &victim[..8.min(victim.len())],
                &child.state.hash[..8.min(child.state.hash.len())],
            )
        };
        let victim_net_seed = self.state.net(&victim).map(|s| s.net_seed);
        self.record_attempt(
            clock,
            "mutation",
            1,
            Some(&child.state.hash),
            Some(child.state.net_seed),
            &child.state.created_from.clone().unwrap_or_default(),
            "inserted",
            None,
            child.state.last_metrics.as_ref().map(|m| m.fitness),
            None,
            Some(&victim),
            victim_net_seed,
        );
        self.insert_child(child, clock, "mutation");
        Ok(detail)
    }

    /// The next child ordinal at this clock (across all evolution branches).
    fn next_child_ordinal(&mut self, clock: usize) -> usize {
        let idx = *self.children_born_at_clock.entry(clock).or_insert(0);
        *self.children_born_at_clock.get_mut(&clock).unwrap() = idx + 1;
        idx
    }

    /// Cull one net: final state snapshot to disk, drop from all live maps.
    fn cull_net(&mut self, hash: &str, clock: usize, reason: &str) -> Result<()> {
        // NO per-event log here: culls are reported on the calling roll's
        // single line (crossover/mutation/pruned) — one roll, one line.
        let smoothed = self
            .rolling_fitness
            .get(hash)
            .map(rolling_mean)
            .unwrap_or(f32::NAN);
        if let Some(mut state) = self.state.net(hash).cloned() {
            state.is_alive = false;
            // Cull metadata turns the tombstone into a complete record of
            // when/why this net left the population, not just a dead snapshot.
            state.culled_at_step = Some(clock);
            state.cull_reason = Some(reason.to_string());
            state.final_smoothed_fitness = Some(smoothed);
            write_net_state(&self.run_dir, &state)?;
        }
        self.state.remove(hash);
        self.networks.remove(hash);
        self.optimizers.remove(hash);
        self.rolling_fitness.remove(hash);
        self.rolling_train.remove(hash);
        self.rolling_eval.remove(hash);
        // Decision-lag bookkeeping dies with the net: the shadow and its
        // promotion clock are per-individual state.
        // Anti-devolution bookkeeping: the net is gone — its floor and any
        // demotion flag die with it.
        self.fitness_floors.remove(hash);
        self.demoted.remove(hash);
        self.culls += 1;
        Ok(())
    }

    /// Population-mean smoothed fitness — the value recorded at each
    /// checkpoint and the bar a crossover child must beat.
    fn population_mean_smoothed_fitness(&self) -> f32 {
        let values = self.smoothed_fitness_values();
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f32>() / values.len() as f32
    }

    /// Whether this run has a surprise-exam batch at all: it needs BOTH a
    /// dataset/stream and a tabular trainer, so RL runs (and any run without
    /// data) do not. The ledger still stores 0.0 for those; this predicate is
    /// what lets the log print `—` instead of a fabricated score.
    fn exam_available(&self) -> bool {
        self.stream.is_some() && self.dataset.is_some() && self.trainer.as_tabular().is_some()
    }

    /// The gate bar a crossover child faces at `clock` — the exact quantity
    /// [`Self::evolve_crossover_child`] compares against (`relevant` = every
    /// checkpoint with `step <= clock`; Hard takes the last one, Soft the mean
    /// of them). `None` when no checkpoint exists yet. Printed on the
    /// checkpoint line so the ledger entry and the gate lines can be read
    /// together (a Soft bar is a historical AVERAGE, not the newest mean).
    fn current_gate_bar(&self, clock: usize) -> Option<f32> {
        let relevant: Vec<&Checkpoint> = self
            .checkpoints
            .iter()
            .filter(|c| c.step <= clock)
            .collect();
        if relevant.is_empty() {
            return None;
        }
        Some(match self.config.crossover_gate {
            crate::engine::config::CrossoverGate::Hard => relevant
                .last()
                .map(|c| c.pop_mean_fitness)
                .unwrap_or(f32::NAN),
            crate::engine::config::CrossoverGate::Soft => {
                relevant.iter().map(|c| c.pop_mean_fitness).sum::<f32>() / relevant.len() as f32
            }
        })
    }

    /// Run the checkpoint "surprise exam": score every live net on the given
    /// era's gating-pool batch (rows never used for training or per-step
    /// eval). Returns the population's mean exam fitness — a generalization
    /// diagnostic recorded in the checkpoint ledger, never used for ranking,
    /// culling, or gating (so the replay contract is untouched). Nets are
    /// scored in eval mode (no gradients); a net whose eval-mode forward has
    /// side effects would violate the Trainer contract anyway.
    ///
    /// Returns `0.0` when [`Self::exam_available`] is false (RL): the ledger
    /// field is unconditional, but the log omits it rather than showing it.
    fn run_checkpoint_exam(&mut self, era: u64) -> Result<f32> {
        // RL mode has no dataset to examine — the exam is a generalization
        // diagnostic over held-out DATA rows, meaningless without data. Also
        // requires a Tabular trainer (the loss to score with).
        let (stream, dataset, loss) = match (
            self.stream.as_ref(),
            self.dataset.as_ref(),
            self.trainer.as_tabular(),
        ) {
            (Some(s), Some(d), Some(t)) => (s, d, t.loss()),
            _ => return Ok(0.0),
        };
        let exam_batch = stream.exam_batch(dataset, era)?;
        let direction = self.fitness.direction();
        let mut scores: Vec<f32> = Vec::new();
        // Collect hashes first to avoid borrowing self.networks while calling
        // eval_one_step (which needs &mut Network).
        let hashes = self.state.live_hashes();
        for hash in &hashes {
            if let Some(net) = self.networks.get_mut(hash) {
                if let Ok(report) =
                    eval_one_step(net, loss, &self.fitness, &self.metrics, &exam_batch)
                {
                    scores.push(report.fitness);
                }
            }
        }
        let _ = direction; // direction-aware comparison happens upstream if needed
        if scores.is_empty() {
            return Ok(0.0);
        }
        Ok(scores.iter().sum::<f32>() / scores.len() as f32)
    }

    /// Select a live net inversely-proportionate to smoothed fitness (worst
    /// nets most likely) — the `MutationCullPolicy::InverseFitness` arm. Elite
    /// nets (top-`config.elite_count`) are excluded entirely — they can never
    /// be mutation victims. Returns `None` when no cullable candidate remains
    /// (empty/single-net pop, or all elite).
    fn select_inverse_proportional(&self) -> Result<Option<String>> {
        let hashes = self.state.live_hashes();
        if hashes.len() < 2 {
            return Ok(None);
        }
        let direction = self.fitness.direction();
        let elite = self.elite_hashes();
        // Inverse fitness: weight = (adjusted best) − (adjusted value) ≥ 0 —
        // the worst net gets the largest weight, the best gets zero. (The
        // previous `value − worst` weighting was inverted: it targeted the
        // FITTEST net — fixed; the confused "wait, inverted" comment is gone.)
        let scored: Vec<(String, f32)> = hashes
            .iter()
            .filter(|h| !elite.contains(h))
            .filter(|h| {
                self.rolling_fitness
                    .get(*h)
                    .map(|b| b.iter().count() > 0)
                    .unwrap_or(false)
            })
            .map(|h| {
                (
                    h.clone(),
                    rolling_mean(self.rolling_fitness.get(h).unwrap()),
                )
            })
            .collect();
        if scored.len() < 2 {
            return Ok(None);
        }
        let adjusted = |v: f32| match direction {
            crate::engine::fitness::Direction::Maximize => v,
            crate::engine::fitness::Direction::Minimize => -v,
        };
        let adj_best = scored
            .iter()
            .map(|(_, v)| adjusted(*v))
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<(String, f32)> = scored
            .into_iter()
            .map(|(h, v)| (h, adj_best - adjusted(v)))
            .collect();
        // Anti-devolution D — SOFT, no special queue: a demoted (regressed
        // below floor × tol) net is NOT evicted deterministically. Its
        // collapsed fitness already up-weights it in this roulette (weight =
        // adj_best − fitness, so a cliff-fall is a fat cull target by itself).
        // The guard's remaining teeth are status, not culling: no elite seat
        // (see `elite_hashes`), and the "regressed" tombstone attribution
        // below. A recovering net keeps a shot at surviving and jumping back.
        let total: f32 = weights.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            // All-equal (non-elite) population: uniform draw.
            let i = fastrand::usize(0..weights.len());
            return Ok(weights.into_iter().nth(i).map(|(h, _)| h));
        }
        let mut pick = fastrand::f32() * total;
        for (h, w) in &weights {
            pick -= w;
            if pick <= 0.0 {
                return Ok(Some(h.clone()));
            }
        }
        Ok(weights.last().map(|(h, _)| h.clone()))
    }

    /// ANTI-DEVOLUTION D — the regression check, factored out of
    /// `step_one_net` (which only calls it when `regression_tol` is set).
    /// Lazily set the net's floor at its first verdict (the "birth-right"
    /// measurement), then each step check whether the net has FALLEN more
    /// than `(1 − tol)` of the floor's magnitude. A collapsed net loses elite
    /// protection (denied in [`Self::elite_hashes`]); culling stays the
    /// ordinary inverse-fitness roulette — the collapsed fitness up-weights
    /// the net there naturally.
    ///
    /// The comparison is a signed DISTANCE (`floor − smoothed` for Maximize,
    /// `smoothed − floor` for Minimize), not a ratio (`floor × tol`). A ratio
    /// silently inverts for negative fitness: with floor −8.96 and tol 0.7,
    /// `floor × tol = −6.27` — a threshold ABOVE the floor, so a net at
    /// −8.41 (an improvement!) was condemned as "collapsed". kagi's earned
    /// money is routinely negative early in a race; distance is sign-agnostic.
    /// `max(1.0)` keeps a floor near zero from firing on float dust.
    /// Ordinary wobble never trips the guard: the comparison is on the
    /// smoothed value against a generous tolerance, and recovering clears
    /// the flag (the seat is contestable again).
    fn update_regression_guard(&mut self, hash: &str, clock: usize) {
        let Some(buf) = self.rolling_fitness.get(hash) else {
            return;
        };
        if buf.is_empty() {
            return;
        }
        let smoothed = rolling_mean(buf);
        let direction = self.fitness.direction();
        match self.fitness_floors.get_mut(hash) {
            None => {
                self.fitness_floors.insert(hash.to_string(), smoothed);
            }
            Some(floor) => {
                let tol = self.config.regression_tol.unwrap();
                let (fallen, allowed) = match direction {
                    crate::engine::fitness::Direction::Maximize => {
                        (*floor - smoothed, (1.0 - tol) * floor.abs().max(1.0))
                    }
                    crate::engine::fitness::Direction::Minimize => {
                        (smoothed - *floor, (1.0 - tol) * floor.abs().max(1.0))
                    }
                };
                let collapsed = fallen > allowed;
                if collapsed {
                    if !self.demoted.contains(hash) {
                        log::info!(
                            "step {} │ net {} │ REGRESSED below floor (fell {:.2}, allowed {:.2}; floor {:.2}) — demoted (no elite seat; up-weighted in cull roulette by its collapsed fitness)",
                            clock,
                            &hash[..8.min(hash.len())],
                            fallen,
                            allowed,
                            *floor,
                        );
                    }
                    self.demoted.insert(hash.to_string());
                } else {
                    if self.demoted.remove(hash) {
                        // Mirror event: recovery is as important as the fall
                        // — it tells the reader the guard isn't stuck and
                        // the net re-earned its standing.
                        info!(
                            "step {} │ net {} │ recovered (smoothed {:.2} back within floor × tol) — demotion cleared",
                            clock,
                            &hash[..8.min(hash.len())],
                            smoothed,
                        );
                    }
                }
            }
        }
    }

    /// The elite set: the top `config.elite_count` live nets by smoothed
    /// fitness (direction-aware). Protected from ALL culls — crossover (any
    /// policy) and mutation alike. Always leaves at least one cullable net:
    /// the effective guard size is `min(elite_count, live − 1)`.
    fn elite_hashes(&self) -> Vec<String> {
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
            // Anti-devolution D: a REGRESSED net cannot hold elite status,
            // even if its (collapsed) smoothed fitness still ranks top-k.
            .filter(|h| !self.demoted.contains(*h))
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
    fn worst_nets_by_smoothed_fitness(&self, count: usize) -> Result<Vec<String>> {
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

    // ── Stop criteria ───────────────────────────────────────────────────────

    /// Check stop criteria at the given step. Returns `Some(reason)` if one
    /// fires, `None` if the run should continue.
    ///
    /// `max_steps` and `max_target_fitness` are mutually exclusive (enforced
    /// at `build()` — exactly one may be set). `custom_stop` is independent
    /// and always evaluated last, joining whichever built-in was chosen.
    fn check_stop(&mut self, step: usize) -> Option<StopReason> {
        // 0. Ctrl+C (checked FIRST — the user's intent outranks every bar;
        // abandoning flows through the SAME artifact path as a natural stop).
        // The in-flight step IS completed (we are between steps here), then
        // the whole post-race sequence runs.
        if let Some(flag) = &self.interrupt_flag {
            if flag.load(std::sync::atomic::Ordering::SeqCst) {
                info!(
                    "race: Ctrl+C — shutting down gracefully after step {}",
                    step
                );
                return Some(StopReason::Interrupted);
            }
        }
        // 1. Explicit step budget (fires only if set).
        if let Some(max) = self.config.max_steps {
            if step >= max {
                return Some(StopReason::MaxSteps);
            }
        }
        // 2. Best smoothed fitness reached the target (fires only if set).
        if let Some(target) = self.config.max_target_fitness {
            let best = self.best_smoothed_fitness();
            if self.fitness.direction().is_better(best, target) {
                return Some(StopReason::TargetScore);
            }
        }
        // 3. Custom stop: joins the race **in addition** to the built-ins, so
        // a user policy can stop the run on criteria the built-ins don't
        // model.
        if let Some(stop) = self.config.custom_stop.as_ref() {
            if stop(&self.snapshot(step)) {
                return Some(StopReason::CustomStop);
            }
        }
        None
    }

    /// Build the read-only `RaceSnapshot` handed to a custom stop closure.
    /// `best`/`worst` are direction-aware (the fitness direction decides which
    /// end of the spread is "best"); `mean` is the plain arithmetic mean.
    fn snapshot(&self, step: usize) -> RaceSnapshot {
        let smoothed = self.smoothed_fitness_values();
        let (best, worst, mean) = if smoothed.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let direction = self.fitness.direction();
            let mut best = smoothed[0];
            let mut worst = smoothed[0];
            for &v in &smoothed[1..] {
                if direction.is_better(v, best) {
                    best = v;
                }
                if direction.is_better(worst, v) {
                    worst = v;
                }
            }
            let mean = smoothed.iter().sum::<f32>() / smoothed.len() as f32;
            (best, worst, mean)
        };
        RaceSnapshot {
            live_count: self.state.live_count(),
            step,
            best_smoothed_fitness: best,
            worst_smoothed_fitness: worst,
            mean_smoothed_fitness: mean,
            culls: self.culls,
            elapsed_seconds: self.elapsed_base_secs + self.started_at_wall.elapsed().as_secs(),
        }
    }

    // ── Instrumentals ───────────────────────────────────────────────────────

    /// Persist the checkpoint ledger to `checkpoints.json` (sidecar next to
    /// `engine.json`). Overwritten on every record — small file, atomic
    /// enough for analysis purposes.
    fn write_checkpoints(&self) -> Result<()> {
        let path = self.run_dir.join("checkpoints.json");
        let v: Vec<serde_json::Value> = self
            .checkpoints
            .iter()
            .map(|c| {
                serde_json::Value::Object({
                    let mut m = serde_json::Map::new();
                    m.insert("step".into(), serde_json::Value::from(c.step));
                    m.insert(
                        "pop_mean_fitness".into(),
                        serde_json::Value::from(c.pop_mean_fitness),
                    );
                    m.insert(
                        "exam_mean_fitness".into(),
                        serde_json::Value::from(c.exam_mean_fitness),
                    );
                    m
                })
            })
            .collect();
        let raw = serde_json::to_string_pretty(&v).map_err(|e| {
            crate::utils::error::EngineError::Json(format!("checkpoints serialize: {e}"))
        })?;
        std::fs::write(&path, raw).map_err(|source| crate::utils::error::EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Ok(())
    }

    /// Write the NetState file for every currently active, live net.
    fn write_live_frontier_states(&self) -> Result<()> {
        let live = self.state.live_hashes();
        for h in &live {
            if let Some(state) = self.state.net(h) {
                write_net_state(&self.run_dir, state)?;
            }
        }
        Ok(())
    }

    /// Append a step's metrics for all live nets to the history buffer.
    /// Rows are typed `metric` in the unified history.csv (evolution events
    /// are typed `attempt` — see `record_attempt`).
    fn append_metrics_csv(&mut self, step: usize) -> Result<()> {
        if !self.config.csv_export {
            return Ok(());
        }
        for hash in &self.state.live_hashes() {
            if let Some(net_state) = self.state.net(hash) {
                if let Some(m) = &net_state.last_metrics {
                    let origin = net_state.created_from.clone().unwrap_or_default();
                    let mut row = format!(
                        "metric,{},{},{},{},{},{},{},{}",
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

    /// Flush the buffered history rows to `history.csv` — the unified event
    /// log (per-step `metric` rows + evolution `attempt` rows). Same cadence:
    /// checkpoints + stop, gated by `csv_export`.
    fn flush_metrics_csv(&mut self) -> Result<()> {
        if !self.config.csv_export || self.history_csv_buffer.is_empty() {
            return Ok(());
        }
        let path = self.run_dir.join("history.csv");
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

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            })?;

        if !exists {
            // Unified header: the `type` column discriminates the row shape
            // (`metric` = per-step live-net snapshot; `attempt` = evolution
            // event). Downstream columns overlap where they can; a reader
            // filters by type first.
            // `net_seed` completes the individual identity: (hash, net_seed)
            // uniquely distinguishes re-born individuals that share a
            // topology hash across eras.
            let mut headers =
                "type,step,hash,net_seed,origin,entered_at_step,train_loss,eval_loss,fitness"
                    .to_string();
            for m in &self.metrics {
                headers.push(',');
                headers.push_str(m.label());
            }
            headers.push_str(",branch,attempt,outcome,gate_index,child_fitness,bar,victim,victim_net_seed,pop_size");
            headers.push('\n');
            file.write_all(headers.as_bytes()).map_err(|source| {
                crate::utils::error::EngineError::Io {
                    path: path.display().to_string(),
                    source,
                }
            })?;
        }

        file.write_all(self.history_csv_buffer.as_bytes())
            .map_err(|source| crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            })?;

        self.history_csv_buffer.clear();
        Ok(())
    }

    /// Load a checkpoint ledger written by [`Self::write_checkpoints`]
    /// (resume path). Missing file = fresh run, empty ledger.
    pub(crate) fn load_checkpoints(&mut self) -> Result<()> {
        let path = self.run_dir.join("checkpoints.json");
        if !path.exists() {
            return Ok(());
        }
        let raw = std::fs::read_to_string(&path).map_err(|source| {
            crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            }
        })?;
        let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
            crate::utils::error::EngineError::Json(format!("checkpoints parse: {e}"))
        })?;
        let mut checkpoints = Vec::new();
        if let Some(arr) = v.as_array() {
            for entry in arr {
                let step = entry.get("step").and_then(|f| f.as_u64()).unwrap_or(0) as usize;
                let pop_mean_fitness = entry
                    .get("pop_mean_fitness")
                    .and_then(|f| f.as_f64())
                    .unwrap_or(0.0) as f32;
                let exam_mean_fitness = entry
                    .get("exam_mean_fitness")
                    .and_then(|f| f.as_f64())
                    .unwrap_or(0.0) as f32;
                checkpoints.push(Checkpoint {
                    step,
                    pop_mean_fitness,
                    exam_mean_fitness,
                });
            }
        }
        self.checkpoints = checkpoints;
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
    incoming_trainer: &crate::trainer::ModeTrainer,
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

    /// The RL volume label is the RL middle column of the per-step rollup:
    /// totals plus the mean turns/match. A population that reported nothing
    /// (Tabular, or a mis-wired RL trainer) must read as `—`, never as a
    /// silent `0`.
    #[test]
    fn rl_volume_label_formats_and_never_lies() {
        let empty = RlVolume::default();
        assert!(empty.label().contains('—'), "{}", empty.label());
        let pop = RlVolume {
            nets: 3,
            matches: 3,
            turns: 432,
        };
        assert_eq!(pop.label(), "matches 3 │ turns 432 │ turns/match 144");
        // Matches with zero turns is still a real report (every match died on
        // turn 0) — it reads as a mean of 0, not as `—`.
        let zero_turns = RlVolume {
            nets: 1,
            matches: 1,
            turns: 0,
        };
        assert_eq!(zero_turns.label(), "matches 1 │ turns 0 │ turns/match 0");
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
        assert!(cfg.max_target_fitness.is_none());
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
        let metrics = |step: usize| crate::state::state::NetMetrics {
            step,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.5,
            informative: vec![],
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
    fn freeze_elites_excludes_top_k_from_trainer_call() {
        // A: with freeze on, the top-k net by smoothed fitness carries its
        // last metrics forward instead of training. Observed through the
        // freeze branch's contract: last_metrics exists ⇒ buffer grows from
        // the carried value and the trainer is never invoked (a real run's
        // log shows the FROZEN debug line; here we assert the mechanism).
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
        // Give the elite a recorded skill state, then verify the freeze
        // branch carries it into the rolling buffer without any training.
        let m = crate::state::state::NetMetrics {
            // fitness 0.5 — a plausible frozen skill
            step: 3,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.5,
            informative: vec![],
        };
        eng.state.record_step(&hs[1], m).unwrap();
        let before_len = eng.rolling_fitness.get(&hs[1]).unwrap().len();
        eng.step_one_net(&hs[1], 5).unwrap();
        let after = eng.rolling_fitness.get(&hs[1]).unwrap();
        assert_eq!(after.len(), before_len + 1, "carried metric appended");
        assert_eq!(
            *after.iter().last().unwrap(),
            0.5,
            "carried the frozen skill value"
        );
        assert_eq!(
            eng.state.net(&hs[1]).unwrap().step,
            1,
            "frozen net's clock advanced only by the record_step call (no trainer step)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn regression_demotion_denies_elite_and_fronts_cull_line() {
        // D: a net that collapses below floor × tol loses elite status and
        // is chosen FIRST by the inverse-proportional victim selector,
        // ahead of worse-looking healthy nets.
        let dir = std::env::temp_dir().join("gras_regression_demote_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 13).unwrap();
        eng.config.regression_tol = Some(0.7);
        eng.config.elite_count = 1;
        // FOUR nets: with elite_count=1 three stay cullable. (With only 2
        // cullable nets the inverse-fitness roulette is degenerate — the
        // better cullable carries weight 0, so the worst is picked with
        // probability 1 no matter what demotion does — and no soft-vs-hard
        // distinction is observable.)
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
        let hs = eng.state.live_hashes();
        // Fixture is Minimize: lower smoothed = fitter. Net A: floor 0.1
        // (was great), then collapses toward 1.0 — 1.0 > 0.1/0.7 ⇒ demoted.
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
        eng.update_regression_guard(&hs[0], 4); // floor = 0.1
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(1.0);
        eng.update_regression_guard(&hs[0], 4); // collapse detected
        // Net B: healthy best — floor 0.5, never collapses. Holds the elite
        // seat (excluded from the cull roulette).
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.5);
        eng.update_regression_guard(&hs[1], 4);
        // Net C: healthy middling — floor 0.7, never collapses. Becomes the
        // best CULLABLE net (weight 0 in the roulette).
        eng.rolling_fitness.get_mut(&hs[2]).unwrap().push(0.7);
        eng.update_regression_guard(&hs[2], 4);
        // Net D: healthy bad — floor 0.75, never collapses.
        eng.rolling_fitness.get_mut(&hs[3]).unwrap().push(0.75);
        eng.update_regression_guard(&hs[3], 4);
        assert!(eng.demoted.contains(&hs[0]), "collapsed net is demoted");
        assert!(!eng.demoted.contains(&hs[1]));
        assert!(!eng.demoted.contains(&hs[2]));
        // Recovery clears the flag: A's next smoothed (0.4) is below the
        // collapse line again (0.4 > 0.1/0.7 is FALSE) — undemoted.
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.4);
        eng.update_regression_guard(&hs[0], 4);
        // Hmm: 0.4 vs floor 0.1 under Minimize: collapsed = smoothed > floor/tol
        // = 0.143. 0.4 > 0.143 ⇒ still demoted (floor was set while strong).
        assert!(
            eng.demoted.contains(&hs[0]),
            "staying 4× worse than birth-floor keeps the demotion (by design)"
        );
        // Elite denial: A's smoothed (mean 0.5 of 0.1,1.0) would rank top-1
        // among {0.5, 0.5, 0.9} under Minimize — but demotion bars it.
        let elite = eng.elite_hashes();
        assert!(!elite.contains(&hs[0]), "demoted net cannot hold elite");
        // SOFT culling (no front-of-line queue): the plain inverse-
        // proportional roulette favors the demoted net once its collapse
        // accumulates in the rolling window (the window is why a ONE-step
        // dip is deliberately not a cull offense). Push the collapse a few
        // more steps so A's smoothed mean is clearly the worst, then draw:
        // A is picked far more often than any healthy peer, but NOT always
        // (a recovering net keeps a shot at surviving).
        for _ in 0..5 {
            eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(1.0);
            eng.update_regression_guard(&hs[0], 4); // still demoted
        }
        // Now A's smoothed ≈ 0.81 (worst), C 0.7 and D 0.75 healthy. Cullable
        // weights (best cullable C = weight 0): A ≈ 0.11, D ≈ 0.05 → A is
        // picked ~69% of draws: strongly favored, never guaranteed.
        let mut demoted_picks = 0usize;
        let mut healthy_picks = 0usize;
        let runs = 400;
        for seed in 0..runs {
            fastrand::seed(seed);
            match eng.select_inverse_proportional().unwrap().as_deref() {
                Some(h) if h == hs[0] => demoted_picks += 1,
                Some(h) if h == hs[3] => healthy_picks += 1,
                _ => {}
            }
        }
        assert!(
            demoted_picks > healthy_picks,
            "collapsed fitness must up-weight the demoted net in the roulette \
             (demoted {demoted_picks} vs healthy {healthy_picks} of {runs})"
        );
        assert!(
            demoted_picks < runs as usize,
            "demotion must NOT be a guaranteed eviction — the roulette still draws"
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
        let m = crate::state::state::NetMetrics {
            step: 3,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.1,
            informative: vec![],
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
        let m2 = crate::state::state::NetMetrics {
            step: 6,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.05,
            informative: vec![],
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
    fn demoted_victim_cull_reason_says_regressed() {
        // D: when the mutation roulette's victim is a demoted net, the cull
        // reason is "regressed" (attribution in tombstone + history.csv);
        // healthy victims keep the generic "immigrant-slot". NOTE: no
        // special lane — the collapsed fitness wins the ordinary draw, the
        // guard only names it.
        let dir = std::env::temp_dir().join("gras_regressed_cull_reason");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 29).unwrap();
        eng.config.regression_tol = Some(0.7);
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let hs = eng.state.live_hashes();
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
        eng.update_regression_guard(&hs[0], 4); // floor 0.1
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(1.0);
        eng.update_regression_guard(&hs[0], 4); // collapse → demoted
        // Let the collapse accumulate in the rolling window so the demoted
        // net's smoothed fitness is the WORST (i.e. the fattest cull weight):
        for _ in 0..4 {
            eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(1.0);
            eng.update_regression_guard(&hs[0], 4);
        } // smoothed ≈ 0.85 vs B 0.5 / C 0.8 ⇒ weight 0.35 vs 0.3 vs 0.0
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.5);
        eng.update_regression_guard(&hs[1], 4); // healthy
        eng.rolling_fitness.get_mut(&hs[2]).unwrap().push(0.8);
        eng.update_regression_guard(&hs[2], 4); // healthy
        assert!(eng.demoted.contains(&hs[0]));
        // The demoted net has the fattest cull weight (fitness 1.0 vs
        // 0.5/0.8), so most seeds pick it — but the draw is probabilistic;
        // find a seed where the roulette agrees, then attribute.
        let victim = (0..1000)
            .find_map(|s| {
                fastrand::seed(s);
                let v = eng.select_inverse_proportional().unwrap().unwrap();
                (v == hs[0]).then_some(v)
            })
            .expect("demoted net is the fattest target — some seed must pick it");
        let reason = if eng.demoted.contains(&victim) {
            "regressed"
        } else {
            "immigrant-slot"
        };
        assert_eq!(
            reason, "regressed",
            "demoted victim is attributed to the guard"
        );
        // Healthy net → generic reason.
        let reason2 = if eng.demoted.contains(&hs[1]) {
            "regressed"
        } else {
            "immigrant-slot"
        };
        assert_eq!(reason2, "immigrant-slot");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn regression_off_by_default_leaves_tabular_untouched() {
        // Both guards off (defaults): no floors recorded, nobody demoted,
        // elite ranking identical to the pre-feature behavior.
        let dir = std::env::temp_dir().join("gras_regress_off_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 17).unwrap();
        assert!(!eng.config.freeze_elites);
        assert!(eng.config.regression_tol.is_none());
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(1.0); // a "collapse"
        eng.step_one_net(&hs[0], 4).unwrap();
        assert!(eng.fitness_floors.is_empty(), "no floors when guard off");
        assert!(eng.demoted.is_empty(), "no demotions when guard off");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fresh_start_immigrant_skips_catch_up() {
        // With the knob on, a mutation immigrant keeps step 0 / no metrics
        // (no catch-up replay) and gets the FRESH-START detail line. With it
        // off (default), catch-up runs and the immigrant trains to clock.
        let dir = std::env::temp_dir().join("gras_fresh_start_immigrant");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 31).unwrap();
        eng.config.immigrant_fresh_start = true;
        eng.config.mutate_rolls = 1;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.9);
        // A fresh immigrant is inserted; its step count is 0 and it has no
        // recorded metrics (no catch-up replay happened).
        eng.evolve_random_immigrant(5, 0).unwrap();
        assert_eq!(eng.state.live_hashes().len(), 2, "cull+insert keeps size");
        let newcomer = eng
            .state
            .live_hashes()
            .into_iter()
            .find(|h| !hs.contains(h))
            .expect("a new immigrant hash exists");
        let s = eng.state.net(&newcomer).unwrap();
        assert_eq!(s.step, 0, "fresh-start: no replayed training count");
        assert!(s.last_metrics.is_none(), "fresh-start: no replayed metrics");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fresh_start_rejected_on_tabular_at_construction() {
        // Tabular + fresh-start is a construction error: catch-up protects
        // shared-stream comparability there, and skipping it would silently
        // skip training rows.
        let dir = std::env::temp_dir().join("gras_fresh_start_tabular_reject");
        let _ = std::fs::remove_dir_all(&dir);
        let data_dir = tiny_dataset_dir("fresh-reject");
        let config = RaceConfig {
            pop_size: 2,
            immigrant_fresh_start: true,
            ..RaceConfig::defaults()
        };
        let result =
            crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
                data_dir,
                config,
                fitness(),
                crate::trainer::TabularTrainer::new(loss_fn()),
                Some(9),
                Some(dir.clone()),
            ));
        let err_text = match result {
            Ok(_) => panic!("tabular + fresh-start must fail at construction"),
            Err(e) => format!("{e}"),
        };
        assert!(
            err_text.contains("requires RL mode"),
            "error names the RL-only rule: {err_text}"
        );
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
        ) -> flodl::tensor::Result<crate::trainer::StepReport> {
            Ok(crate::trainer::StepReport {
                train_loss: 0.0,
                eval_loss: None,
                fitness: 1.0,
                informative: Vec::new(),
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
        ) -> flodl::tensor::Result<crate::trainer::StepReport> {
            let fitness = (ctx.net_seed % 97) as f32 + step as f32;
            Ok(crate::trainer::StepReport {
                train_loss: step as f32 * 0.5,
                eval_loss: None,
                fitness,
                informative: Vec::new(),
                rl: Some(crate::trainer::RlStepMeta {
                    matches: 2,
                    turns: 20 + step,
                }),
            })
        }
    }

    /// An RL-mode config (`RaceConfig` is not `Clone`, so tests rebuild it by
    /// value: `pop == 0` only for the harness that seeds its own topologies).
    fn rl_config(pop: usize) -> RaceConfig {
        let mut topo = crate::graph::topology::TopologyOptions::default();
        topo.input_dim = Some(2);
        topo.output_dim = Some(2);
        RaceConfig {
            pop_size: pop,
            mode: crate::engine::config::RunMode::Rl,
            topology_options: topo,
            ..RaceConfig::defaults()
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
         -> Vec<(String, Option<crate::state::state::NetMetrics>)> {
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
    fn rl_spec_requires_declared_rl_mode() {
        // The spec variant and the config's set_run_mode(..) must agree: an RL
        // spec with the default Tabular mode is a config bug.
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
            None,
        );
        let err = match CoreEngine::from_spec(spec) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("RL spec without set_run_mode(RunMode::Rl) must be rejected"),
        };
        assert!(
            err.contains("set_run_mode(RunMode::Rl)"),
            "error must name the fix: {err}"
        );
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
        let mut topo = crate::graph::topology::TopologyOptions::default();
        topo.input_dim = Some(1);
        topo.output_dim = Some(1);
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
        let ca = a.generate_child(3, 0).unwrap().unwrap();
        let cb = b.generate_child(3, 0).unwrap().unwrap();
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

    // ── Iter-6 Tier B/C: resume replay + parity ──────────────────────

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
            .set_checkpoint_every(ckpt_every);
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
        let trainer = crate::trainer::ModeTrainer::Tabular(Box::new(
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
                    crate::state::state::NetMetrics {
                        step: 0,
                        train_loss: 0.0,
                        eval_loss: None,
                        fitness: v,
                        informative: vec![],
                    },
                )
                .unwrap();
        }
    }

    #[test]
    #[should_panic(expected = "only one stop criteria can be used at a time")]
    fn stop_criteria_both_set_panics_at_build() {
        // max_steps + max_target_fitness together are a config error — a
        // HAND-BUILT config can still carry both (the field setters each
        // clear their sibling, but a struct literal bypasses them), so
        // `build()` panics with the exclusive-criteria message.
        let mut cfg = RaceConfig::defaults();
        cfg.max_steps = Some(20);
        cfg.max_target_fitness = Some(0.5);
        // Route the hand-built config through build()'s validation by
        // rebuilding it from the same fields the builder would have written.
        let b = RaceConfig::builder().set_pop_size(2);
        let mut rebuilt = b.build();
        rebuilt.max_steps = cfg.max_steps;
        rebuilt.max_target_fitness = cfg.max_target_fitness;
        crate::engine::config::RaceConfigBuilder::validate_single_stop(&rebuilt)
            .unwrap_or_else(|e| panic!("invalid RaceConfig: {e}"));
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

        // max_target_fitness alone: fires once best smoothed crosses it.
        let run_dir = std::env::temp_dir().join("gras-race-target");
        let mut eng = engine(&run_dir, 5).unwrap();
        eng.config.max_target_fitness = Some(0.6); // Minimize: fires once best smoothed (0.5) < 0.6
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        seed_raw_fitness(&mut eng, 0.5);
        assert_eq!(eng.check_stop(10), Some(StopReason::TargetScore));
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
        // History recorded the pruner phase too (culls marked `pruned`).
        let history = std::fs::read_to_string(run_dir.join("history.csv")).unwrap();
        assert!(
            history.contains("pruner"),
            "culls are recorded as pruner attempt rows"
        );
        for solo_step in [3usize, 4, 5] {
            assert!(
                history.contains(&format!("metric,{solo_step},")),
                "solo step {solo_step} appears as a metric row"
            );
        }
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
}
