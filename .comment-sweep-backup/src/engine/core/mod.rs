//! Step-race scheduler for the continuous step-race engine.
//!
//! One global `step_clock`. Every live net steps together each loop iteration,
//! pulling from the same shared batch stream. One iteration = one step_clock:
//!
//! 1. Evolve first (from clock 1 on) — two independent roll groups. Crossover
//!    rolls: a recombined child evicts a victim per `CrossCullPolicy`; a draw
//!    that produces nothing is discarded and the population is unchanged.
//!    Immigrant rolls: a random entrant replaces an inverse-fitness
//!    victim (pop stays constant). Acting on the metrics the previous clock
//!    recorded, this is what makes a clock's log read in the order the clock
//!    happened: evolve → crown → step.
//! 2. Deal the crown (`refresh_crown`): the eligible elite take(k) is frozen
//!    for this clock, so the seats that evolved population produced print
//!    before anyone steps.
//! 3. The shared `BatchStream` produces one `(train_batch, eval_batch)` for
//!    this step — identical for every net. Each live net (deterministic hash
//!    order) does `train_one_step` + `eval_one_step` on its **live** `Network`
//!    + `Optimizer` (coefs evolve in place; optimizer state carries forward
//!    across steps), records its metrics into `nets/<hash>.json`, and logs one
//!    per-net line. A child born at the head of this clock sits it out (see
//!    the birth-clock invariant below).
//! 4. After **all** nets stepped: every `checkpoint_every` steps, snapshot the
//!    live frontier (states + elite weights) and flush the CSVs — the durability
//!    tick, which also advances the stream's eval-permutation era.
//! 5. One per-step rollup line (or the `Minimal` frame) describing the step
//!    that just ended — logged LAST on purpose, so it always sits at the bottom
//!    of the terminal.
//! 6. Stop criteria checked (max_steps, custom_stop) —
//!    log which fired.
//! 7. Repeat — the loop's clock is the nets' recorded steps (every net is at
//!    the same step after the group steps, so reading any live net's step
//!    gives the clock).
//!
//! Determinism: resume replay uses the same `seed_step_randomness` + same step
//! primitives as the group loop, so two identical-seed runs produce identical
//! cull/insertion sequences.
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

pub(crate) use flodl::nn::Optimizer;
pub(crate) use flodl::tensor::Result;
pub(crate) use std::collections::HashMap;
pub(crate) use tracing::{debug, info};

pub(crate) use crate::engine::evolve::RollKind;
pub(crate) use crate::evolution::selection::SelectionMethod;
pub(crate) use crate::graph::network::Network;
pub(crate) use crate::state::{
    ConfigSnapshot, NetState, RaceState, RunConfig, RunHeader, write_engine_json, write_net_state,
};
pub(crate) use crate::trainer::stream::{BatchStream, PoolSplit};
pub(crate) use crate::utils::fitness::{Fitness, FitnessLabel, Metric};

pub(crate) use super::config::{RaceConfig, StopReason};
pub(crate) use super::format::{build_stamp, csv_field, fmt_cfg, fmt_line};
pub(crate) use super::smoothing::{RollingBuffer, WindowStats, rolling_stats_with};

// ── CoreEngine ───────────────────────────────────────────────────────────────

/// The continuous step-race loop: owns the step clock (the nets' recorded steps
/// ARE the clock), the batch stream, the live population and the rolling buffers.
// One concern per file (AGENTS.md rule 19): these hold `impl CoreEngine`
// blocks, not new types. A split is a MOVE — fields and method names are
// unchanged, so the public surface is identical.
mod durability;
mod evolution;
mod ranking;
mod resume;
pub(crate) mod roles;
// Re-exported so `crate::engine::core::RlVolume` and friends keep their paths.
pub(crate) use evolution::*;

/// Attempts a crossover roll may make (fresh parents each) before it is discarded.
const CROSSOVER_ATTEMPTS: usize = 5;

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
    /// Explicit two-dataset layout marker: `Some(test_len)` for a `{train, test}`
    /// build (test rows concatenated after train). Rebuilds identical pools on resume.
    pub(crate) explicit_test_rows: Option<usize>,
    /// The EXAM store — a dataset the race never reads. Deliberately its own field and
    /// NOT concatenated into `dataset`: no stream index, no metric and no callback can
    /// reach it, so "forgetting it" is structural rather than a policy. Scored once at
    /// stop, for the reported nets only (`exam.rs`), and reported, never fed back.
    pub(crate) exam: Option<crate::utils::tabular_data::Dataset>,
    pub(crate) fitness: Fitness,
    pub(crate) metrics: Vec<Metric>,
    /// The caller-supplied training scheme, boxed per mode: the engine never trains,
    /// it hands one net/one step to this contract. The mode tag lives in the enum arm.
    pub(crate) trainer: crate::trainer::AnyTrainer,
    /// Per-step log verbosity for the engine.
    pub(crate) log_level: crate::engine::config::LogLevel,
    /// Graceful Ctrl+C flag (set by `run`'s signal handler, consumed by
    /// `check_stop`). `None` outside `run` — tests construct without it.
    pub(crate) interrupt_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// True when the frontier and clock came from DISK (`resume` / `resume_with`).
    /// `run` then checks the stop BEFORE its first clock: a budget the recorded run
    /// already spent must re-examine the frontier as recorded, not advance it one
    /// clock further. A fresh race keeps the bottom-of-loop check, so its clock
    /// numbering is unchanged.
    pub(crate) resumed: bool,

    /// Per-step step-log counters captured during the evolve phase and consumed by
    /// the SAME step's rollup line / `Minimal` table (both render after the evolve).
    pub(crate) step_evolve: StepEvolve,
    /// Per-step RL volume (matches + turns), summed over every net that stepped this
    /// clock: reset each step, consumed by the same step's rollup. Zero in Tabular mode.
    pub(crate) step_rl: RlVolume,
    /// Run-total of the TRAIN turns (live path only — a resume's replay mirrors
    /// history and does not re-count): the denominator for `turns/match` / `ms/turn`.
    pub(crate) total_train_turns: usize,
    /// Full hashes of the elite set exported at stop (top-`elite_count` by smoothed
    /// fitness) — the single source of truth for "which nets are the champions".
    pub(crate) champions: Vec<String>,
    /// In-memory state of the live population (topology JSON, seed, step,
    /// last_metrics, lineage). The source of truth for resume + tooling.
    pub(crate) state: RaceState,
    /// Per-net `Network` — built once on insert, mutated in place each step.
    /// **Coefficients live here**, not on disk.
    pub(crate) networks: HashMap<String, Network>,
    /// Per-net `Optimizer` — created on insert, mutated in place each step;
    /// Adam momentum/variance carries forward across steps.
    pub(crate) optimizers: HashMap<String, Box<dyn Optimizer>>,
    /// Rolling fitness history per net (in-memory): every ranking decision reads the
    /// smoothed mean of this buffer, not the raw per-step value.
    pub(crate) rolling_fitness: HashMap<String, RollingBuffer>,
    pub(crate) rolling_train: HashMap<String, RollingBuffer>,
    pub(crate) rolling_eval: HashMap<String, RollingBuffer>,
    /// The ticket probability the coin last COMPARED for each net, captured AT THE DRAW
    /// (`step_one_net_in_frame`). The lottery summary line averages THESE, so its
    /// `expected` comes from the same number the coin drew against, not a re-derivation
    /// from merits that have moved on since.
    ///
    /// Reporting state only, never replay state: it describes the last clock, so it is
    /// not persisted and a resume simply refills it on its first graded clock.
    pub(crate) last_ticket_p: HashMap<String, f32>,
    /// Wall-clock start for the wall-clock stop criterion.
    pub(crate) started_at_wall: std::time::Instant,
    /// Wall-clock seconds accumulated by EARLIER sessions (stop → resume cycles):
    /// `elapsed_seconds` reports `elapsed_base_secs + started_at_wall.elapsed()`.
    pub(crate) elapsed_base_secs: u64,
    /// Wall-clock start of the CURRENT step (set at the top of the run loop), so the
    /// per-step logs surface per-step cost — e.g. RL bridges playing full matches.
    pub(crate) step_started_at_wall: std::time::Instant,
    /// Cumulative culls so far (for max_culls stop).
    pub(crate) culls: usize,
    /// The run's tally of evolution events (see [`crate::state::RunAccounting`]):
    /// printed at stop and persisted under `accounting` in `engine.json`.
    pub(crate) accounting: crate::state::RunAccounting,
    /// Set once, at the first evolve clock, when the initial population is
    /// counted into `accounting` (see the evolve phase).
    pub(crate) initial_population_counted: bool,
    /// Children born per step clock — disambiguates same-clock siblings across all
    /// evolution rounds, so same-clock children derive distinct seeds.
    pub(crate) children_born_at_clock: HashMap<usize, usize>,
    /// Anti-devolution A: the nets holding the freeze crown — the seats the elite gate
    /// dealt, a SUBSET of the count-top nets; tracked so crown lines fire on transitions.
    pub(crate) frozen_crown: std::collections::HashSet<String>,
    /// `Minimal` mode: last step's population means (train, eval, fitness) so the
    /// framed table can show what changed. `None` until the first render.
    pub(crate) minimal_prev_means: Option<(f32, f32, f32)>,
    /// Buffer history rows in memory to minimize slow disk I/O writes.
    /// `history.csv`: one row per live net per step (`append_metrics_csv`).
    pub(crate) history_csv_buffer: String,
    /// Buffer for evolution-event rows — its own `attempts.csv` with its own header:
    /// the two row kinds share no columns, and padding shifted every later column.
    pub(crate) attempts_csv_buffer: String,
}

/// The inputs a constructor actually CHOOSES. Everything else on [`CoreEngine`]
/// is per-run scratch (buffers, caches, counters) and starts empty/fresh.
///
/// Exists because the two engines used to spell the whole 30-field struct literal
/// twice (`tabular_engine.rs` and `rl_engine.rs`), so adding a field was a two-file
/// edit with no compiler reminder until one side failed. One list, one place.
pub(crate) struct CoreEngineParts {
    pub(crate) run_dir: std::path::PathBuf,
    pub(crate) header: RunHeader,
    pub(crate) config: RaceConfig,
    pub(crate) meta_ctx: crate::engine::config::RunMetaCtx,
    pub(crate) stream: Option<BatchStream>,
    pub(crate) dataset: Option<crate::utils::tabular_data::Dataset>,
    pub(crate) explicit_test_rows: Option<usize>,
    pub(crate) exam: Option<crate::utils::tabular_data::Dataset>,
    pub(crate) fitness: Fitness,
    pub(crate) metrics: Vec<Metric>,
    pub(crate) trainer: crate::trainer::AnyTrainer,
    pub(crate) log_level: crate::engine::config::LogLevel,
    pub(crate) resumed: bool,
}

impl CoreEngine {
    /// Assemble the engine from the chosen parts; every scratch field is initialised
    /// HERE, so a new field cannot be added to one engine and forgotten in the other.
    /// Pure construction — no I/O, no RNG, no record writes.
    pub(crate) fn assemble(p: CoreEngineParts) -> Self {
        Self {
            run_dir: p.run_dir,
            header: p.header,
            config: p.config,
            meta_ctx: p.meta_ctx,
            stream: p.stream,
            dataset: p.dataset,
            explicit_test_rows: p.explicit_test_rows,
            exam: p.exam,
            fitness: p.fitness,
            metrics: p.metrics,
            trainer: p.trainer,
            log_level: p.log_level,
            interrupt_flag: None,
            resumed: p.resumed,
            step_evolve: StepEvolve::default(),
            step_rl: RlVolume::default(),
            total_train_turns: 0,
            champions: Vec::new(),
            state: RaceState::new(),
            networks: HashMap::new(),
            optimizers: HashMap::new(),
            rolling_fitness: HashMap::new(),
            rolling_train: HashMap::new(),
            rolling_eval: HashMap::new(),
            last_ticket_p: HashMap::new(),
            started_at_wall: std::time::Instant::now(),
            elapsed_base_secs: 0,
            step_started_at_wall: std::time::Instant::now(),
            culls: 0,
            accounting: crate::state::RunAccounting::default(),
            initial_population_counted: false,
            children_born_at_clock: HashMap::new(),
            frozen_crown: std::collections::HashSet::new(),
            minimal_prev_means: None,
            history_csv_buffer: String::new(),
            attempts_csv_buffer: String::new(),
        }
    }
}

// ── Construction ─────────────────────────────────────────────────────────────

impl CoreEngine {
    /// True when per-step detail lines (evolve rollup, per-child lines, checkpoint,
    /// race-start) are emitted — every level except `Minimal` and `None`.
    pub(crate) fn verbose_detail(&self) -> bool {
        !matches!(
            self.log_level,
            crate::engine::config::LogLevel::Minimal | crate::engine::config::LogLevel::None
        )
    }

    /// The run directory this engine writes to (`RunSpec.run_dir` or
    /// `results/<timestamp>`) — read back here after construction, e.g. to resume later.
    pub fn run_dir(&self) -> &std::path::Path {
        &self.run_dir
    }

    /// Start a race from a [`RunSpec`] — the entire argument surface. The engine loads
    /// data, derives the run dir, resolves a seed and writes the self-contained header.
    pub fn from_spec(spec: crate::engine::run_spec::RunSpec) -> Result<Self> {
        use crate::engine::run_spec::RunSpec as RS;
        let (config, fitness, trainer, seed, run_dir, tabular_data_dir) = match spec {
            RS::Tabular(s) => (
                s.config,
                s.fitness,
                crate::trainer::AnyTrainer::Tabular(s.trainer),
                s.seed,
                s.run_dir,
                Some(s.data_dir),
            ),
            RS::RL(s) => (
                s.config,
                s.fitness,
                crate::trainer::AnyTrainer::Rl(s.trainer),
                s.seed,
                s.run_dir,
                None,
            ),
        };
        // The spec variant DECIDES the mode; the config's `mode` field is DERIVED (the
        // `set_run_mode` setter was deleted — disagreement is now unrepresentable).
        let derived_mode = if tabular_data_dir.is_some() {
            crate::engine::config::RunMode::Tabular
        } else {
            crate::engine::config::RunMode::Rl
        };
        let mut config = config;
        config.mode = derived_mode;
        // A pool NAME is refused here, before data loads and before the first draw: the
        // resolvers' `empty ⇒ all` fallback cannot tell a typo from an empty pool, and a
        // typo that widens the search silently is a run that describes itself wrongly.
        config
            .validate_search_pools()
            .map_err(|e| crate::utils::error::EngineError::InvalidOptions(e))?;
        // RL mode REQUIRES a reported fitness: there is no dataset, so a (pred, target)
        // scorer has nothing to score. Fail before the run starts, not three steps in.
        if tabular_data_dir.is_none() && fitness.is_computed() {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                "RL spec requires Fitness::reported(direction, label) — a Computed (pred, target) scorer has no dataset to score against. The trainer must supply the fitness value in StepReport.".to_string(),
            )
            .into());
        }
        // Catch-up toggles are RL-effective only — Tabular always catches up (one shared
        // stream). Data ROLES come from the subdirectories, through ONE resolver shared
        // with resume (`core/roles.rs`): `val/` ranks, `test/` reports, a lone pool splits.
        let mut roles = match &tabular_data_dir {
            Some(data_dir) => Some(roles::resolve(data_dir, config.device())?),
            None => None,
        };
        let explicit_test_rows = roles.as_ref().and_then(|r| r.explicit_test_rows);
        let eval_split = roles.as_ref().and_then(|r| r.eval_split.clone());
        let dataset = roles.as_mut().and_then(|r| r.dataset.take());
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
        // Infer dims from data_dir when TopologyOptions leaves them unset (Tabular only):
        // `None` means "fill from the dataset at run start". RL specs must set them.
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
        // The EVAL share (`eval_share`), clamped by the same bounds the setter
        // uses: this value builds the pools (and, for a single-pool dir, decides
        // how much of the data the race is judged on). It was a hardcoded `0.2`
        // until 2026-11-01, so no caller could choose the split at all.
        let train_eval_split_ratio = config.eval_share.clamp(
            crate::engine::config::EVAL_SHARE_MIN,
            crate::engine::config::EVAL_SHARE_MAX,
        );
        let default_batch_size = 16usize;
        // The trainer owns batch geometry (Tabular only), resolved BEFORE the header so
        // engine.json records the stream shape the run used. `None` = no stream (RL).
        let (batch_size, eval_batch_size) = match trainer.stream_shape() {
            Some(shape) => (Some(shape.batch_size), Some(shape.eval_batch_size)),
            None => (None, None),
        };

        // The EXAM: resolved by `roles` (a sibling `exam/`, or `test/` beside a
        // ranking `val/`) and NEVER concatenated into `dataset` (see
        // `CoreEngine::exam`). Its row count rides the header so a resume can prove
        // the same exam is still on disk.
        let exam = roles.as_mut().and_then(|r| r.exam.take());

        let (header_input_dim, header_output_dim) = match &dataset {
            Some(d) => (d.inputs.shape()[1] as usize, d.targets.shape()[1] as usize),
            None => (
                topology_options.input_dim.unwrap_or(0),
                topology_options.output_dim.unwrap_or(0),
            ),
        };
        let mut header = RunHeader::from_race_options_at(
            RunConfig {
                run_id: run_id.clone(),
                // One-argument resume reads this back (`TabularEngine::resume_from`).
                data_dir: tabular_data_dir.as_ref().map(|p| p.display().to_string()),
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
                // Legacy key, EVAL share (see `DEFAULT_EVAL_SHARE`): this is the
                // value a resume reads back to rebuild the same two pools.
                train_eval_split_ratio: Some(train_eval_split_ratio),
                explicit_test_rows,
                eval_split,
                exam_rows: exam.as_ref().map(|d| d.len()),
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
        // The eval-permutation era is `checkpoint_every` at birth, and from then on
        // it is whatever THIS file says. A resume may change the tick cadence; it may not
        // change the clocks at which the eval batches were drawn.
        header.eval_era_every = Some(config.checkpoint_every);
        write_engine_json(&run_dir, &header)?;
        // Settle the log sinks now that the run dir exists: `trace_file` opens the sidecar
        // and flushes the buffered start-up block into it; the default closes that buffer.
        super::logging::settle_run_dir(&run_dir, config.trace_file);

        // Seed the initial population internally — [imported elites?] + random draws —
        // deduped and re-rolled so the population always lands at `pop_size`.
        let mut founders: Vec<crate::graph::topology::Topology> = Vec::with_capacity(1);
        let mut seed_source: Option<crate::state::SeedSourceRecord> = None;
        if config.elite_previous_load {
            // The compatibility gate reads the candidate's engine.json header,
            // so this run's own header (mode, dims, trainer name) is the input.
            let trainer_name = header
                .trainer
                .as_ref()
                .and_then(|t| t.get("trainer"))
                .and_then(|v| v.as_str());
            // `max_seeds = pop_size`: the WHOLE elite list comes over in rank order, so the
            // founders are the source's own ranking, never a reshuffle of it.
            match super::seed::resolve_plan(
                config.elite_previous_dir.as_deref(),
                // This run's own dir already exists on disk (the header is
                // written above), so it would otherwise be "the newest run".
                Some(run_dir.as_path()),
                header.engine_mode.as_deref().unwrap_or(""),
                header.input_dim,
                header.output_dim,
                trainer_name,
                config.pop_size,
            ) {
                super::seed::SeedOutcome::Seeded(plan) => {
                    let champ = plan.champion();
                    let short = &champ.champion_hash[..8.min(champ.champion_hash.len())];
                    let n = plan.picks.len();
                    // Deliberately a plain emit (not level-filtered), like the skip
                    // branch: the inheritance receipt must be visible at any log level.
                    crate::utils::cfy::emit_always(&crate::utils::cfy::card(
                        "Inheritance",
                        &crate::utils::cfy::kv(&[
                            ("source run".to_string(), plan.source_run_id.clone()),
                            (
                                "source dir".to_string(),
                                plan.source_dir.display().to_string(),
                            ),
                            ("champion".to_string(), short.to_string()),
                            (
                                "ranks inherited".to_string(),
                                if n == 1 {
                                    "rank 1".to_string()
                                } else {
                                    format!("ranks 1..{n}")
                                },
                            ),
                            (
                                "founder slots".to_string(),
                                if n == 1 {
                                    "0".to_string()
                                } else {
                                    format!("0-{}", n - 1)
                                },
                            ),
                        ]),
                        Some(
                            "rank 1 of the source run's final table; the rest follow that table's order",
                        ),
                    ));
                    let elite_hashes: Vec<String> =
                        plan.picks.iter().map(|p| p.champion_hash.clone()).collect();
                    for pick in &plan.picks {
                        founders.push(pick.topology.clone());
                    }
                    seed_source = Some(crate::state::SeedSourceRecord {
                        source_run_id: plan.source_run_id.clone(),
                        source_dir: plan.source_dir.display().to_string(),
                        champion_hash: champ.champion_hash.clone(),
                        champion_params: champ.champion_params,
                        elite_hashes,
                    });
                }
                super::seed::SeedOutcome::Skipped(why) => {
                    // Deliberately a plain emit: a fact about what the run INHERITED must be
                    // visible at any log level. The run continues all-random either way.
                    crate::utils::cfy::emit_always(&crate::utils::cfy::card(
                        "Inheritance",
                        &crate::utils::cfy::kv(&[(
                            "elite_previous_load".to_string(),
                            "skipped".to_string(),
                        )]),
                        Some(&format!("{why} — starting from an all-random population")),
                    ));
                }
            }
        }
        let initial_topologies =
            super::population::fill_unique(&config, run_seed, founders, config.pop_size);
        // Every seeded slot is labeled by origin, so `nets/*.json` and `attempts.csv` say
        // which source elite it came from instead of a bare `founder-N`.
        let initial_topologies: Vec<(crate::graph::topology::Topology, Option<String>)> =
            initial_topologies
                .into_iter()
                .enumerate()
                .map(|(ordinal, topo)| {
                    let label = seed_source.as_ref().and_then(|rec| {
                        rec.elite_hashes.get(ordinal).map(|hash| {
                            format!("seed:{}:{}", rec.source_run_id, &hash[..8.min(hash.len())])
                        })
                    });
                    (topo, label)
                })
                .collect();
        // Provenance goes in the header — including on disk, hence the second
        // write (the first happened before the population was resolved).
        if seed_source.is_some() {
            header.run_topology_source = seed_source.clone();
            write_engine_json(&run_dir, &header)?;
        }

        // Shared batch stream — engine infrastructure (Tabular only). The trainer may
        // shape batch sizes, but the split ratio is not overridable: held-out comparability.
        let batch_stream = match &dataset {
            Some(dataset) => {
                // Two stream layouts: explicit `{train, test}` dirs → pools by construction (no
                // leak possible), or a single pool → the classic seeded shuffle split.
                let split = match explicit_test_rows {
                    Some(test_len) => {
                        PoolSplit::explicit(dataset.len() - test_len, test_len, run_seed)
                    }
                    None => PoolSplit::of(dataset, train_eval_split_ratio, run_seed),
                };
                Some(
                    BatchStream::new(run_seed, batch_size.unwrap_or(default_batch_size), split)
                        .with_eval_batch_size(eval_batch_size.unwrap_or(default_batch_size))
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
            exam,
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
            last_ticket_p: HashMap::new(),
            started_at_wall: std::time::Instant::now(),
            elapsed_base_secs: 0,
            step_started_at_wall: std::time::Instant::now(),
            culls: 0,
            accounting: crate::state::RunAccounting::default(),
            initial_population_counted: false,
            children_born_at_clock: HashMap::new(),
            step_evolve: StepEvolve::default(),
            step_rl: RlVolume::default(),
            total_train_turns: 0,
            champions: Vec::new(),
            frozen_crown: std::collections::HashSet::new(),
            minimal_prev_means: None,
            log_level,
            interrupt_flag: None,
            resumed: false,
            trainer,
        };

        // Seed the initial population internally — transparent to the user,
        // like the old generational engine where `run` was the only call.
        engine.seed_population_labeled(initial_topologies, None)?;

        Ok(engine)
    }

    /// Runtime default for the best-net eval size, for configs that predate
    /// the field (legacy `0` sentinel ⇒ batch_size). No-op for current configs.
    fn apply_eval_defaults(&mut self) {
        // Batch geometry now lives on RunSpec::stream, not RaceConfig — the engine
        // stores the stream directly, so this is a no-op for current code paths.
    }

    /// The run id back to the caller (for logs / tool use).
    pub fn run_id(&self) -> &str {
        &self.header.run_id
    }

    /// The run seed back to the caller.
    pub fn run_seed(&self) -> u64 {
        self.header.run_seed
    }

    /// The eval-permutation era length this engine's stream was built with.
    ///
    /// On a fresh run it equals `checkpoint_every`; on a resume it is whatever
    /// `engine.json` recorded, so a caller who changes the tick cadence still draws the
    /// eval batches the first half drew. `0` = one eternal era, or no stream (RL).
    pub fn eval_era_every(&self) -> usize {
        self.stream
            .as_ref()
            .map(|s| s.checkpoint_every())
            .unwrap_or(0)
    }

    /// The current step clock — read from any live net's recorded step (after the
    /// group steps every live net is at the same step). 0 when the population is empty.
    pub fn step_clock(&self) -> usize {
        // The next clock is one past the last clock any live net trained — read from the
        // RECORDED clock, not `state.step`: a mid-run insert skips its birth clock.
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

    /// Expose the underlying population state.
    pub fn state(&self) -> &RaceState {
        &self.state
    }

    /// Populate the population from finalized topologies (internal — `new` calls
    /// this; resume replays from disk): each built on the device, fresh Adam.
    #[cfg(test)]
    fn seed_population_internal(
        &mut self,
        topologies: Vec<crate::graph::topology::Topology>,
        initial_fitness: Option<f32>,
    ) -> Result<()> {
        let labeled = topologies.into_iter().map(|t| (t, None)).collect();
        self.seed_population_labeled(labeled, initial_fitness)
    }

    /// [`Self::seed_population_internal`] with per-slot lineage labels: slot 0 of a
    /// seeded run carries `seed:<run id>:<hash>` instead of `founder-0`.
    fn seed_population_labeled(
        &mut self,
        topologies: Vec<(crate::graph::topology::Topology, Option<String>)>,
        initial_fitness: Option<f32>,
    ) -> Result<()> {
        for (ordinal, (topo, label)) in topologies.iter().enumerate() {
            let net = Network::build(topo, self.config.device())?;
            // Lineage: founders are labeled `founder-<ordinal>`, distinct from `random` /
            // `crossover-*` origins; a label supplied by the caller (the champion) wins.
            let origin = label
                .clone()
                .unwrap_or_else(|| format!("founder-{ordinal}"));
            let mut state = NetState::new(topo, 0, Some(origin))?;
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
        // The init line names the inheritance — "populated 100 live nets" says nothing
        // about whether one came from a previous experiment. Silent when all locks.
        let seeded_slots = topologies
            .iter()
            .filter(|(_, label)| label.is_some())
            .count();
        // CFY: the init line is a table; inheritance provenance becomes its own rows.
        let mut rows: Vec<(String, String)> = vec![
            ("live nets".to_string(), count.to_string()),
            ("run dir".to_string(), self.run_dir.display().to_string()),
            ("seeded slots".to_string(), seeded_slots.to_string()),
        ];
        if let Some(src) = self.header.run_topology_source.as_ref() {
            rows.push(("source run".to_string(), src.source_run_id.clone()));
            rows.push((
                "champion".to_string(),
                src.champion_hash[..8.min(src.champion_hash.len())].to_string(),
            ));
            rows.push((
                "ranks inherited".to_string(),
                if seeded_slots == 1 {
                    "rank 1".to_string()
                } else {
                    format!("ranks 1..{seeded_slots}")
                },
            ));
        }
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            "Race init",
            &crate::utils::cfy::kv(&rows),
            Some(if seeded_slots == 0 {
                "all-random founder batch"
            } else {
                "seeded slots inherit a previous experiment's champion topology"
            }),
        ));
        Ok(())
    }

    // ── The run loop ────────────────────────────────────────────────────────

    /// Run the step-race loop until a stop criterion fires: one step_clock per
    /// iteration — same batch, same eval, then the evolve rolls cull and replace.
    pub fn run(&mut self) -> Result<StopReason> {
        self.apply_eval_defaults();
        // Keep the stream's eval-rotation cadence in lockstep with the checkpoint cadence —
        // a post-construction config edit must not desync era numbering — but ONLY on a
        // fresh run. On a resume the era is the RECORD: `TabularEngine::resume` built the
        // stream from `header.eval_era_every`, and overwriting it here would rotate the eval
        // permutation at this call's cadence instead of the one the first half used.
        if !self.resumed {
            if let Some(stream) = self.stream.as_mut() {
                stream.set_run_checkpoint_every(self.config.checkpoint_every);
            }
            // The record follows the run that is actually starting: a config edit between
            // construction and `run` would otherwise leave `engine.json` naming an era this
            // run never used, and a later resume would draw the wrong eval batches.
            if self.header.eval_era_every != Some(self.config.checkpoint_every) {
                self.header.eval_era_every = Some(self.config.checkpoint_every);
                if let Err(e) = crate::state::write_engine_json(&self.run_dir, &self.header) {
                    tracing::warn!("engine.json eval-era record failed: {e}");
                }
            }
        }
        // CFY: the whole start-up block is comfy tables — a run manifest (B), a
        // search-space section (A), a trainer section, and the log legend as a
        // card footnote (E). None of these lines start with `step`, so none
        // stays a one-liner; every field the old prose carried is a cell here.
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
            let mut manifest: Vec<(String, String)> = vec![
                ("run_id".to_string(), self.header.run_id.clone()),
                ("seed".to_string(), self.header.run_seed.to_string()),
                ("pid".to_string(), std::process::id().to_string()),
                (
                    "mode".to_string(),
                    self.header
                        .engine_mode
                        .clone()
                        .unwrap_or_else(|| "—".to_string()),
                ),
                ("pop_size".to_string(), self.config.pop_size.to_string()),
                (
                    "max_steps".to_string(),
                    self.config
                        .max_steps
                        .map(|m| m.to_string())
                        .unwrap_or_else(|| "unbounded".to_string()),
                ),
                (
                    "smoothing_window (K)".to_string(),
                    self.config.smoothing_window.to_string(),
                ),
                (
                    "risk_aversion (λ)".to_string(),
                    fmt_cfg(self.config.risk_aversion),
                ),
                (
                    "window_recency (r)".to_string(),
                    fmt_cfg(self.config.window_recency),
                ),
                ("mean_type".to_string(), self.config.mean_type.to_string()),
                (
                    "train_age".to_string(),
                    match self.config.train_age {
                        true => "armed".to_string(),
                        false => "off".to_string(),
                    },
                ),
                (
                    "checkpoint_every".to_string(),
                    self.config.checkpoint_every.to_string(),
                ),
                (
                    "crossover_rolls".to_string(),
                    self.config.crossover_rolls.to_string(),
                ),
                (
                    "mutate_rolls".to_string(),
                    self.config.mutate_rolls.to_string(),
                ),
                (
                    "crossover_prob".to_string(),
                    fmt_cfg(self.config.crossover_prob),
                ),
                (
                    "crossover_selection".to_string(),
                    self.config.crossover_selection.to_string(),
                ),
                ("mutate_prob".to_string(), fmt_cfg(self.config.mutate_prob)),
                (
                    "elite_count".to_string(),
                    self.config.elite_count.to_string(),
                ),
                (
                    "freeze_elites".to_string(),
                    self.config.freeze_elites.to_string(),
                ),
                (
                    "train_merit (Ω)".to_string(),
                    fmt_cfg(self.config.train_merit),
                ),
                (
                    "train_merit_p_min (p_min)".to_string(),
                    fmt_cfg(self.config.train_merit_p_min),
                ),
                (
                    "train_merit_p_max (p_max)".to_string(),
                    fmt_cfg(self.config.train_merit_p_max),
                ),
                // Which end of the field the tickets favour, spelled out so a reader
                // does not have to interpret a bare `false`.
                (
                    "lottery tilt".to_string(),
                    if self.config.train_merit_inverted {
                        "worst-first".to_string()
                    } else {
                        "best-first".to_string()
                    },
                ),
                (
                    "feature_split_pool".to_string(),
                    self.config.feature_split_pool_names().join(", "),
                ),
                // The one knob that edits an INHERITED gene after crossover: masks,
                // node ops, port activations and dropout. `0` = exact inheritance, so
                // the manifest says so instead of printing a bare number.
                (
                    "topology drift".to_string(),
                    if self.config.drift_prob > 0.0 {
                        fmt_cfg(self.config.drift_prob)
                    } else {
                        "0 (exact inheritance)".to_string()
                    },
                ),
                (
                    "fitness".to_string(),
                    format!(
                        "{} ({})",
                        self.meta_ctx.fitness_label, self.meta_ctx.direction
                    ),
                ),
                ("input_dim".to_string(), self.meta_ctx.input_dim.to_string()),
                (
                    "output_dim".to_string(),
                    self.meta_ctx.output_dim.to_string(),
                ),
                (
                    "batch_size".to_string(),
                    self.meta_ctx
                        .batch_size
                        .map(|b| b.to_string())
                        .unwrap_or_else(|| "—".to_string()),
                ),
                (
                    "dropout".to_string(),
                    format!("{}", self.meta_ctx.dropout_prob),
                ),
                ("build".to_string(), build_stamp()),
            ];
            // Knobs that silently change what a run MEANS — kept beside the identity so a
            // resumed command stays reconstructible from the log alone.
            for (k, v) in [
                (
                    "crossover_cull_policy",
                    self.config.crossover_cull_policy.to_string(),
                ),
                (
                    "mutation_cull_policy",
                    self.config.mutation_cull_policy.to_string(),
                ),
                ("elite_save", self.config.elite_save_topology.to_string()),
                (
                    "probation_steps",
                    self.config.mutation_probation_steps.to_string(),
                ),
                (
                    "crossover_probation_steps",
                    self.config.crossover_probation_steps.to_string(),
                ),
                (
                    "elite_join_probation_steps",
                    self.config.elite_join_probation_steps.to_string(),
                ),
                (
                    "elite_crown_lifetime_steps",
                    self.config.elite_crown_lifetime_steps.to_string(),
                ),
            ] {
                manifest.push((k.to_string(), v));
            }
            crate::utils::cfy::emit(&crate::utils::cfy::card(
                "Run manifest",
                &crate::utils::cfy::kv(&manifest),
                None,
            ));
            // The legend: every word and symbol the step lines use, defined ONCE
            // next to the arithmetic it names. Later lines can stay short because
            // the reader met the identities here.
            let sign = match self.fitness.direction() {
                crate::utils::fitness::Direction::Maximize => "−",
                crate::utils::fitness::Direction::Minimize => "+",
            };
            let legend = crate::utils::cfy::kv(&[
                (
                    "numbers".to_string(),
                    "|v| ≥ 10 → 1 decimal · 1 ≤ |v| < 10 → 2 · |v| < 1 → 4".to_string(),
                ),
                (
                    "reading".to_string(),
                    format!(
                        "avg = mean of the last K={} readings · spread = how variable those readings are · SE = spread/√n = how well the mean is known · n = readings behind it",
                        self.config.smoothing_window
                    ),
                ),
                (
                    "score".to_string(),
                    format!(
                        "score{} = window avg {} λ × window spread (what the ranking sorts on; the rollup's `±` is the `σ` ACROSS the live scores, not this window spread) · λ={}",
                        self.fitness.direction().arrow(),
                        sign,
                        fmt_cfg(self.config.risk_aversion)
                    ),
                ),
                (
                    "counts".to_string(),
                    "ran = ticketed + crown · ticketed = won + exempt · graded = won + idled · live = ran + held back · expected = graded × avg ticket chance, drawn = won · ★ = elite rank + seat term"
                        .to_string(),
                ),
                (
                    "seats".to_string(),
                    "seat = an elite reserve · wait = clocks to serve before sitting · term = clocks a seat may be held; a completed term is a scheduled rotation, not a loss"
                        .to_string(),
                ),
                (
                    "words".to_string(),
                    "idled = the lottery withheld this clock's train+eval · exempt = crown or born shield, always runs · ticket = one clock of train+eval · trained n of m = optimizer updates over clocks lived"
                        .to_string(),
                ),
            ]);
            crate::utils::cfy::emit(&crate::utils::cfy::card("Legend", &legend, None));

            // The genotype's structural bounds: without these the card names the
            // pools but hides how big a net may get, which is half the search space.
            let topo = self.config.topology_options;
            let search = crate::utils::cfy::kv(&[
                ("crossover_ops".to_string(), ops),
                ("activations".to_string(), acts),
                ("combine".to_string(), combines),
                ("standardize".to_string(), std_ops),
                (
                    "hidden_dim_pool".to_string(),
                    match &self.config.hidden_dim_pool {
                        Some(r) => format!(
                            "{}..={} (stride {})",
                            r.start(),
                            r.end(),
                            self.config.hidden_dim_stride
                        ),
                        None => "default 4..=8".to_string(),
                    },
                ),
                (
                    "hidden_nodes".to_string(),
                    format!(
                        "{}..={}",
                        topo.min_hidden_num_nodes, topo.max_hidden_num_nodes
                    ),
                ),
                (
                    "hidden_inputs".to_string(),
                    format!(
                        "{}..={}",
                        topo.min_hidden_inputs_per_node, topo.max_hidden_inputs_per_node
                    ),
                ),
                (
                    "hidden_outputs".to_string(),
                    format!(
                        "{}..={}",
                        topo.min_hidden_outputs_per_node, topo.max_hidden_outputs_per_node
                    ),
                ),
            ]);
            crate::utils::cfy::emit(&crate::utils::cfy::card("Search space", &search, None));

            // The trainer's own self-description: the only visible record of trainer-side
            // knobs (challenge trigger, schedules, clipping), which left with the trainer.
            if let Some(trainer) = self.header.trainer.as_ref() {
                let rows = crate::engine::reporting::summarize_rows(trainer);
                if !rows.is_empty() {
                    crate::utils::cfy::emit(&crate::utils::cfy::card(
                        "Trainer",
                        &crate::utils::cfy::kv(&rows),
                        None,
                    ));
                }
            }
        }

        // ── Graceful Ctrl+C (always on) ─────────────────────────────────

        // One SIGINT sets the flag: the loop abandons the in-flight step at the next
        // boundary and takes the same artifact path as a natural stop. A second force-kills.
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
                // CFY: the interrupt notice is a table, kept on stderr so it does not
                // disturb the `Minimal` frame's stdout redraw.
                eprintln!(
                    "{}",
                    crate::utils::cfy::card(
                        "Ctrl+C",
                        &crate::utils::cfy::kv(&[(
                            "action".to_string(),
                            "graceful shutdown at the next step boundary".to_string(),
                        )]),
                        Some("press again to force-kill"),
                    )
                );
            });
        }
        self.interrupt_flag = Some(interrupted);

        loop {
            let clock = self.step_clock();
            // Pre-flight stop for a RESUMED race: the clock came from disk, so the budget
            // may ALREADY be spent (`--max-steps` at or under the recorded clock). Checking
            // here keeps a guardrail-only re-exam looking at the frontier as RECORDED — no
            // clock is trained, evolved or re-ranked before the check. A fresh race starts
            // at clock 0 and never takes this branch, so its numbering is unchanged.
            if self.resumed {
                // `check_stop_before_clock`, not `check_stop`: at the TOP of a
                // clock the budget predicate is `clock >= max`, one clock apart
                // from the bottom-of-loop form. Sharing the bottom form here
                // silently ate the last allowed clock of every top-up.
                if let Some(reason) = self.check_stop_before_clock(clock) {
                    return self.finish_race(reason, clock.saturating_sub(1));
                }
            }
            self.step_started_at_wall = std::time::Instant::now();
            // Per-step accumulators start empty; the rollup at the END of this
            // step reads them (it is the last line printed for the step).
            self.step_rl = RlVolume::default();

            // ── 0. evolve — now BEFORE the crown deal and the group step ────

            // The pipeline order is evolve → deal → step, so a clock's logs read in order.
            // Clock 0 is skipped, and a child born here is excluded from this clock's group.
            if clock > 0 {
                // ── 3. evolve — always, two independent roll groups ─────────────

                // ONE log line per roll, emitted here from the loop — outcome details travel up
                // from the evolve helpers as plain strings. 15 rolls ⇒ 15 lines.
                let mut cross_fired = 0usize;
                let mut cross_survived = 0usize;
                let mut cross_discarded = 0usize;
                let mut cross_attempts = 0usize;
                let mut mutate_fired = 0usize;
                let culls_before = self.culls;
                let cross_total = self.config.crossover_rolls;
                let mutate_total = self.config.mutate_rolls;
                // The cull shield is a POPULATION-level fact: when nobody is cull-eligible, every
                // roll would be spent for nothing — say it once. Derived draws shift no later roll.
                let evolve_allowed = self.has_cull_eligible();
                if !evolve_allowed {
                    self.accounting.deferred_clocks += 1;
                    // The windows are NAMED, with their lengths: "no cull-eligible net"
                    // alone reads as a stuck engine, while `k=5` at clock 3 makes it
                    // arithmetic — the shield is per-child and outlives a short run, so
                    // every clock of it defers by construction.
                    let shielded = self.config.crossover_probation_steps >= clock
                        || self.config.mutation_probation_steps >= clock;
                    info!(
                        "step {} │ evolution deferred: no cull-eligible net — every non-elite net is inside its cull shield window (crossover k={}, mutation k={}) or elite ({} crossover + {} mutation roll(s) forgone; nothing culled/born){}",
                        clock,
                        self.config.crossover_probation_steps,
                        self.config.mutation_probation_steps,
                        cross_total,
                        mutate_total,
                        if shielded {
                            " · a window is at least as long as the run so far, so no clock could evolve yet"
                        } else {
                            ""
                        },
                    );
                }
                // The initial batch is whatever is alive at the FIRST evolve clock — nothing is
                // inserted yet and this clock's culls come later, so it is the founder batch.
                if !self.initial_population_counted {
                    self.initial_population_counted = true;
                    let initial: Vec<String> = self.state.live_hashes();
                    let inherited = initial
                        .iter()
                        .filter(|h| {
                            self.state
                                .net(h)
                                .and_then(|s| s.created_from.as_deref())
                                .is_some_and(|c| c.starts_with("seed:"))
                        })
                        .count();
                    self.accounting.founders = initial.len() - inherited;
                    self.accounting.inherited_founders = inherited;
                    self.accounting.born = initial.len();
                }
                self.accounting.crossover_rolls += cross_total;
                self.accounting.mutation_rolls += mutate_total;
                // 3a. crossover rolls: recombined children.
                // Non-firing rolls are the NORMAL case (p < 1). Collected here and
                // reported as ONE summary line per clock, so a reader does not mistake
                // `1/5, 3/5, 5/5` for lost output.
                let mut cross_missed: Vec<usize> = Vec::new();
                let mut mutate_missed: Vec<usize> = Vec::new();
                for roll in 0..cross_total {
                    if !evolve_allowed {
                        break; // nobody to cull: the whole batch is forgone
                    }
                    if self.state.live_count() < 2 {
                        info!(
                            "step {} │ crossover roll {} of {} skipped (pop < 2)",
                            clock,
                            roll + 1,
                            cross_total,
                        );
                        break; // not enough population to evolve
                    }
                    if self.roll_fires(RollKind::Crossover, clock, roll) {
                        // roll fired — fall through to the attempt loop
                    } else {
                        // debug: a roll that missed is the NORMAL case (p < 1), so it is per-roll
                        // detail, not a per-clock event; `--log-level debug` shows them all.
                        cross_missed.push(roll + 1);
                        debug!(
                            "step {} │ crossover roll {} of {} missed (p={})",
                            clock,
                            roll + 1,
                            cross_total,
                            fmt_cfg(self.config.crossover_prob),
                        );
                        continue;
                    }
                    cross_fired += 1;
                    // A crossover roll retries the FULL attempt (fresh parents) up to a
                    // fixed budget; a roll that still produces nothing is discarded.
                    let max_attempts = CROSSOVER_ATTEMPTS;
                    let mut attempt = 0usize;
                    let outcome = loop {
                        attempt += 1;
                        let out = self.evolve_crossover_child(clock, roll, attempt)?;
                        if out.survived {
                            cross_survived += 1;
                            break out;
                        }
                        if out.no_slot {
                            // Hard shield: no cull-eligible victim. Retrying cannot change that within this
                            // clock, and it is not a discard (no child was ever built).
                            break out;
                        }
                        if attempt >= max_attempts {
                            cross_discarded += 1;
                            break out;
                        }
                        // retry — the prior attempt's detail is superseded
                        let _ = &out;
                    };
                    // Attempts are the roll cost, so they count whichever way
                    // the roll ended (a spent roll paid for its tries too).
                    cross_attempts += attempt;
                    // A roll that BUILT a child is a real event (info); one that paid its attempts
                    // and produced nothing is per-roll detail, already carrying its counters.
                    if outcome.survived {
                        let draws = if attempt == 1 {
                            "1 parent draw".to_string()
                        } else {
                            format!("{attempt} parent draws")
                        };
                        info!(
                            "step {} │ crossover roll {} of {} → {} ({draws})",
                            clock,
                            roll + 1,
                            cross_total,
                            outcome.detail,
                        );
                    } else {
                        debug!(
                            "step {} │ crossover roll {} of {}: {}",
                            clock,
                            roll + 1,
                            cross_total,
                            outcome.detail,
                        );
                    }
                }
                // 3b. mutation rolls: random immigrants.
                for roll in 0..mutate_total {
                    if !evolve_allowed {
                        break; // nobody to cull: the whole batch is forgone
                    }
                    if self.state.live_count() == 0 {
                        info!(
                            "step {} │ mutation roll {} of {} skipped (pop empty)",
                            clock,
                            roll + 1,
                            mutate_total,
                        );
                        break;
                    }
                    if !self.roll_fires(RollKind::Mutation, clock, roll) {
                        // Same as the crossover no-fire line: per-roll detail.
                        mutate_missed.push(roll + 1);
                        debug!(
                            "step {} │ mutation roll {} of {} missed (p={})",
                            clock,
                            roll + 1,
                            mutate_total,
                            fmt_cfg(self.config.mutate_prob),
                        );
                        continue;
                    }
                    match self.evolve_random_immigrant(clock, roll)? {
                        Some(detail) => {
                            mutate_fired += 1;
                            info!(
                                "step {} │ mutation roll {} of {} → {}",
                                clock,
                                roll + 1,
                                mutate_total,
                                detail,
                            );
                        }
                        // Hard shield: the roll is spent with nothing culled and no immigrant built — a
                        // fresh net is never evicted. Per-roll detail; the rollup carries the verdict.
                        None => {
                            self.accounting.mutation_blocked += 1;
                            // Same shape as the crossover shield's line — one
                            // fact, one spelling (`{group} n/N: detail`).
                            debug!(
                                "step {} │ mutation roll {} of {}: {}",
                                clock,
                                roll + 1,
                                mutate_total,
                                self.no_cull_roll_detail("mutation"),
                            )
                        }
                    }
                }
                // The rolls that did NOT fire, named in one line: their absence is
                // the probability working, not dropped output.
                if self.verbose_detail() && (!cross_missed.is_empty() || !mutate_missed.is_empty())
                {
                    let list = |v: &[usize]| {
                        v.iter()
                            .map(|n| n.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                    let mut bits = Vec::new();
                    let plural = |v: &[usize]| if v.len() == 1 { "" } else { "s" };
                    if !cross_missed.is_empty() {
                        bits.push(format!(
                            "crossover roll{} {} missed (p={})",
                            plural(&cross_missed),
                            list(&cross_missed),
                            fmt_cfg(self.config.crossover_prob)
                        ));
                    }
                    if !mutate_missed.is_empty() {
                        bits.push(format!(
                            "mutation roll{} {} missed (p={})",
                            plural(&mutate_missed),
                            list(&mutate_missed),
                            fmt_cfg(self.config.mutate_prob)
                        ));
                    }
                    info!("step {} │ {}", clock, bits.join(" │ "));
                }
                // The run's accounting rides the same counters the step
                // summary uses — one place to update, no second bookkeeping.
                self.accounting.crossover_fired += cross_fired;
                self.accounting.crossover_attempts += cross_attempts;
                self.accounting.crossover_discarded += cross_discarded;
                self.accounting.crossover_children += cross_survived;
                self.accounting.mutation_fired += mutate_fired;
                self.accounting.immigrants += mutate_fired;
                self.accounting.born += cross_survived + mutate_fired;
                // Evolve-phase summary for the `Minimal` table (rendered in the NEXT step's
                // frame), reusing the roll counts already computed — no extra instrumentation.
                self.step_evolve = StepEvolve {
                    culls: self.culls - culls_before,
                    inserts: cross_survived + mutate_fired,
                    cross_fired,
                    cross_survived,
                    cross_discarded,
                    mutate_fired,
                };
            }

            // ── 1. group step: every live net on the same shared batch ──────

            // A child born above is deliberately NOT in this clock's group (it skipped its
            // birth clock), keeping the `entered_at_step + 1` metric-row contract.
            let hashes: Vec<String> = self
                .state
                .live_hashes()
                .into_iter()
                .filter(|h| {
                    clock == 0
                        || self
                            .state
                            .net(h)
                            .map(|s| s.entered_at_step != clock)
                            .unwrap_or(true)
                })
                .collect();
            if hashes.is_empty() {
                crate::utils::cfy::emit(&crate::utils::cfy::card(
                    "Race",
                    &crate::utils::cfy::kv(&[("live nets".to_string(), "0".to_string())]),
                    Some("population empty — stopping"),
                ));
                return Ok(StopReason::MaxSteps);
            }
            // 1-pre. KISS crown: the crown IS the elite take(k) set, recomputed fresh every
            // step (refresh_crown prints the freeze/dethrone diffs up front).
            self.refresh_crown(clock);
            // 1a. pop-wide phase (RL only): hand the trainer the whole live population once
            // per step, before any per-net step. The trainer is asked FIRST whether it uses
            // the hook — the gather below is O(population), so a mode (or scheme) that
            // ignores `pop_phase` must not pay for it every step.
            if self.trainer.wants_pop_phase() {
                // One pass over the map — multiple get_mut borrows stored at
                // once don't compile (NLL), but iter_mut does.
                let live: std::collections::HashSet<&String> = hashes.iter().collect();
                let mut nets: Vec<(String, &mut Network)> = self
                    .networks
                    .iter_mut()
                    .filter(|(h, _)| live.contains(h))
                    .map(|(h, net)| (h.clone(), net))
                    .collect();
                self.trainer.pop_phase(&mut nets, clock);
            }
            // One lottery field per clock, built BEFORE anyone steps: every net is
            // graded against the same pre-clock readings (order-independent) and the
            // walk over live nets happens once, not once per net. `None` when the
            // lottery is off — zero cost on the default path.
            let lottery = self.lottery_frame(clock);
            // THE COIN NEVER STARVES THE WHOLE GROUP. The clock is read back from the
            // records a step writes, so a clock whose every draw lost would retake the
            // same draws at the same clock forever — one unlucky clock would pin the
            // run. A flat field already abstains (nobody measured = nobody to starve);
            // this is that rule one level up: if nobody in the group wins the coin, the
            // HIGHEST chance takes the clock. Ties break on the hash, so the pick is a
            // function of the coin and not of `live_hashes()` order.
            let forced_ticket: Option<(String, f32)> = lottery.and_then(|frame| {
                let mut best: Option<(String, f32)> = None;
                for hash in &hashes {
                    let p = self.train_lottery_p_eff(hash, clock, Some(frame));
                    if self.train_lottery_draws_at(hash, clock, p) {
                        return None; // a real ticket exists: nothing to force
                    }
                    let better = match &best {
                        None => true,
                        Some((held, top)) => p > *top || (p == *top && hash < held),
                    };
                    if better {
                        best = Some((hash.clone(), p));
                    }
                }
                best
            });
            let forced_ticket = forced_ticket.map(|(hash, _)| hash);
            if let Some(hash) = &forced_ticket {
                info!(
                    "step {} │ lottery │ every net in the group lost its draw — {} takes the clock on the highest chance (p={}) so the clock is not pinned",
                    clock,
                    &hash[..8.min(hash.len())],
                    fmt_cfg(self.train_lottery_p_eff(hash, clock, lottery)),
                );
            }
            // The rollup is the only consumer of the skipped-update tally, so its two
            // O(n) sums only run when a line will actually be printed.
            //
            // It does NOT ride the crown's switch: the line also carries the training
            // budget (ran / ticketed / idled) and the population identity, which a run with
            // `freeze_elites = false` needs just as much. The seat legs render themselves
            // off instead (`crown_armed`), so silencing the crown never silences the budget.
            let rollup_on = self.log_level == crate::engine::config::LogLevel::Summ;
            // The lottery's own tally for this clock: a per-net snapshot of the IDLE
            // counters (clocks it has left the net untrained and unmeasured), so
            // after the loop the rollup can say how many tickets it denied AND WHICH
            // nets got none (the delta, not the cumulative total).
            // Gated on the LOTTERY's own line, not the rollup's: `freeze_elites` off
            // silences the crown rollup, never the train-budget report.
            let lottery_line_on = self.lottery_line_on(lottery.is_some());
            // The lottery's per-net card, printed BEFORE anyone steps: every number in
            // it is an input to this clock's coin — the same `frame` handed to
            // `step_one_net_in_frame` below — so `p` and the field bounds printed with
            // it explain the draw rather than being rebuilt from post-step readings.
            // What the clock actually skipped is the who-line, after the loop.
            if lottery_line_on {
                if let Some(frame) = lottery {
                    // The per-net card is the DEBUG view: at Summ a population of 88 would
                    // redraw 88 rows every clock. Debug/trace still gets it in full; Summ
                    // gets the one aggregate line after the loop.
                    if tracing::enabled!(tracing::Level::DEBUG) {
                        self.log_lottery_card(clock, frame);
                    }
                }
            }
            let idle_before: Vec<usize> = if lottery_line_on {
                hashes
                    .iter()
                    .map(|h| self.state.net(h).map(|s| s.idle_clocks()).unwrap_or(0))
                    .collect()
            } else {
                Vec::new()
            };
            for hash in &hashes {
                // Mid-step Ctrl+C: abandon the step immediately. Nothing of it is recorded —
                // history, checkpoints and buffers stay consistent at step-1.
                if let Some(flag) = &self.interrupt_flag {
                    if flag.load(std::sync::atomic::Ordering::SeqCst) {
                        info!(
                            "step {} │ interrupted mid-step — abandoning (no partial records), shutting down gracefully",
                            clock
                        );
                        // Skip to the post-race sequence with Interrupted: the champion/frontier
                        // path treats this like any stop reason at clock-1's state.
                        return self.finish_race(StopReason::Interrupted, clock.saturating_sub(1));
                    }
                }
                // A trainer error DURING an interrupt IS the interrupt (SIGINT hits the whole
                // process group, so a bridge child dies first): route it into the graceful path.
                let forced = forced_ticket.as_deref() == Some(hash.as_str());
                if let Err(e) = self.step_one_net_in_frame(hash, clock, lottery, forced) {
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

            // ── 1b. per-step rollup: trained / frozen / newborn + elite gate ──

            // One line per GROUP step, printed LAST so it lands at the bottom of the
            // terminal; what each field means is documented once on `step_rollup_line`.
            // Per-net idle deltas: ONE read, consumed by BOTH the rollup (as a count)
            // and the who-line (as hashes). Independent of the rollup's gate, so a
            // crown-less run still reports who the budget denied.
            let (trained, withheld) = self.lottery_clock_report(&hashes, &idle_before);
            let lottery_skips = withheld.len();

            if rollup_on {
                let vitals = self.step_vitals(clock, &hashes, lottery_skips);
                info!("step {} │ {}", clock, step_rollup_line(&vitals));
            }
            // The who-line rides the LOTTERY's gate, not the rollup's: a run with the
            // elite crown off (`freeze_elites = false`) still reports which nets got no
            // ticket. Printed only when something actually did not.
            if lottery_line_on {
                if let Some(frame) = lottery {
                    // The Summ view is the aggregate: `p̄` vs the realized `won`, the
                    // field, and the budget means. One line, whatever the population size.
                    self.log_lottery_summary(clock, frame, trained.len(), withheld.len());
                    // WHO got the tickets is DEBUG: a Summ terminal cannot carry a
                    // wrapped hash list per clock.
                    if lottery_skips > 0 {
                        let (mut trained, mut withheld) = (trained, withheld);
                        trained.sort();
                        withheld.sort();
                        tracing::debug!(
                            "step {} │ lottery │ {}",
                            clock,
                            lottery_who_line(&trained, &withheld)
                        );
                    }
                }
            }

            // ── 2. durability tick (every `checkpoint_every` steps) ─────────
            // `> 0` is load-bearing twice: `checkpoint_every = 0` is the documented OFF
            // (no tick, no era rotation) and `clock % 0` would panic.
            if self.config.checkpoint_every > 0
                && clock > 0
                && clock % self.config.checkpoint_every == 0
            {
                // Snapshot the live frontier. NOTE: this is not a point-in-time SET — a
                // net culled after this tick overwrites its own file with a tombstone, so a
                // hard kill can leave a frontier that no longer resumes (measured: 2 live of
                // 8, refused as a pop_size mismatch). notes/RESUME.md §4 has the numbers and
                // §6 the fix (an atomic `ckpt.json` manifest).
                if let Err(e) = self.write_live_frontier_states() {
                    tracing::warn!("checkpoint live states write failed: {e}");
                }
                // EVERY live net's WEIGHTS — the frontier states above carry topology and
                // counters, this completes them. Stale files are swept here, never on
                // cull: see `sweep_stale_net_weights`.
                if let Err(e) = self.write_live_net_weights() {
                    tracing::warn!("checkpoint net weights write failed: {e}");
                }
                self.sweep_stale_net_weights();
                // The frontier as a SET, after the files it names: until this lands, a
                // kill leaves `nets/` as the only record, and a net culled after the
                // tick has already overwritten its own file with a tombstone. Written
                // here — nets first, manifest second — so a manifest never names a net
                // whose weights are absent. A manifest that LAGS the weights is refused on
                // resume: a loaded weight cannot be rewound.
                if let Err(e) = self.write_frontier_manifest(clock + 1) {
                    tracing::warn!("checkpoint frontier manifest write failed: {e}");
                }
                // Flush metrics at the tick too, so an interrupted run keeps per-step
                // history up to its last tick instead of the whole in-memory buffer.
                if let Err(e) = self.flush_csv_exports() {
                    tracing::warn!("checkpoint CSV flush failed: {e}");
                }
                if self.verbose_detail() {
                    // Name the files, not the internals: "frontier" and "flushed"
                    // are engine vocabulary a reader cannot map to a path.
                    info!(
                        "step {} │ checkpoint │ saved: live net states + weights (nets/*.json, nets/*.safetensors) + CSV buffers written",
                        clock
                    );
                }
            }

            // ── 3c. per-step rollup — LAST line of the step ─────────────────

            // Ordered evolve → crown → train → checkpoint → rollup so the summary always
            // sits at the bottom; `took` covers the whole step. `Minimal` renders the table.
            // Diversity rides the checkpoint cadence, ABOVE the rollup (the rollup stays
            // the clock's last line).
            self.log_diversity(clock);
            // Same cadence, same reason: the mixing rate is a slow reading, and it sits
            // above the rollup so the clock still ends on its summary line.
            self.log_crossover_mix(clock);
            if self.log_level != crate::engine::config::LogLevel::None {
                self.log_step_rollup(clock);
            }

            // ── 4. stop criteria ────────────────────────────────────────────
            if let Some(reason) = self.check_stop(clock) {
                return self.finish_race(reason, clock);
            }
        }
    }

    /// The shared post-race sequence — champion artifacts, stop summary, frontier
    /// write, metrics flush. Every stop reason flows through here.
    fn finish_race(&mut self, reason: StopReason, clock: usize) -> Result<StopReason> {
        // The champion markdown dump is an ARTIFACT, not a log line —
        // it must land on disk at every log level, even `None`.
        self.write_elite_index()?;
        self.write_champion_markdown()?;
        // `None` mode stays silent until the caller's final stop print.
        if self.verbose_detail() {
            // CFY: the stop banner is a table too.
            crate::utils::cfy::emit(&crate::utils::cfy::card(
                "Stop",
                &crate::utils::cfy::kv(&[
                    ("reason".to_string(), format!("{reason:?}")),
                    ("step".to_string(), clock.to_string()),
                ]),
                None,
            ));
            self.log_stop_summary(clock)?;
        }
        // After the stop cards, and OUTSIDE the gate: the final-field TABLE is an
        // artifact, written at every log level — `None` included, since that is the
        // scripted caller who wants the table as a file. `log_final_field` gates only
        // its card (and the ancestry under it) on the same `verbose_detail`.
        self.log_final_field(clock, self.verbose_detail());
        // Stamp the live nets' final readings first: the frontier write below
        // then carries `final_*` on every net file, culled or not.
        self.stamp_live_final_scores();
        self.write_live_frontier_states()?;
        // …and their weights, so the stop-time frontier is as loadable as a tick's.
        self.write_live_net_weights()?;
        self.sweep_stale_net_weights();
        // The same point-in-time set at the stop clock, so a finished (or Ctrl+C'd)
        // dir resumes from exactly here — its manifest and its CSV agree.
        self.write_frontier_manifest(clock + 1)?;
        self.flush_csv_exports()?;
        self.persist_run_counters();
        Ok(reason)
    }

    /// Stamp the run-level counters (`culls`, accumulated wall time,
    /// `children_born_at_clock`) into `engine.json` so resume restores them.
    pub(crate) fn persist_run_counters(&self) {
        let mut header = self.header.clone();
        header.culls = self.culls;
        header.run_elapsed_secs = self.elapsed_base_secs + self.started_at_wall.elapsed().as_secs();
        header.children_born_at_clock = self.children_born_at_clock.clone();
        // The run's tally of evolution events: part of "what happened", not re-derivable
        // from the config — stamped here, the last write of the run. Resume ignores it.
        header.accounting = Some(self.accounting.clone());
        if let Err(e) = crate::state::write_engine_json(&self.run_dir, &header) {
            tracing::warn!("engine.json counter persistence failed: {e}");
        }
    }

    /// Stamp the SEARCH SPACE from a loaded header onto a config. Shared by both
    /// `resume_from` constructors so the two arms cannot drift: these fields decide WHICH
    /// topologies exist (options, pools, hidden-dim range), which metrics are recorded,
    /// and where the run stops — none of them belongs to the caller on a resume.
    pub(crate) fn apply_header_search_space(
        config: &mut RaceConfig,
        header: &crate::state::RunHeader,
    ) {
        config.topology_options = header.topology_options;
        // A header written BEFORE the split moved onto `TopologyOptions` recorded it on
        // `config.feature_split`; restore it here so an old run keeps drawing the same
        // masks for its newborns instead of silently reverting to "no split".
        if config.topology_options.feature_split.is_none() {
            if let Some(raw) = header.config.feature_split.as_deref() {
                config.topology_options.feature_split =
                    crate::engine::config::parse_feature_split_wire(raw);
            }
        }
        if let Some(pool) = header
            .hidden_dim_pool
            .split_once("..=")
            .and_then(|(start, end)| {
                Some(start.trim().parse::<usize>().ok()?..=end.trim().parse::<usize>().ok()?)
            })
        {
            config.hidden_dim_pool = Some(pool);
        }
        config.hidden_dim_stride = header.hidden_dim_stride.max(1);
        config.combine_op_pool = header.combine_op_pool.clone();
        config.activation_pool = header.activation_pool.clone();
        config.standardize_op_pool = header.standardize_op_pool.clone();
        // The metric set comes from the RECORD: a resume that drops a metric appends
        // `history.csv` rows one column short of its own header.
        config.metrics = header
            .informative_metrics
            .iter()
            .filter_map(|name| crate::trainer::menu::metric_for(name))
            .collect();
    }

    /// Change the step budget of a LIVE engine — the resume-and-extend door. The budget is
    /// ABSOLUTE (`max_steps` counts clocks, not clocks-this-process), so extending it is the
    /// one stop knob a resume may move; the caller's value wins over the recorded one.
    ///
    /// The ONE thing this can quietly change is the shape of a budget-relative LR
    /// schedule, and the check lives here because this is the single funnel every
    /// language's "extend the run" goes through — see
    /// [`warn_budget_changed_under_a_budget_schedule`].
    pub fn set_stop_max_steps(&mut self, steps: usize) {
        if self.resumed && self.header.max_steps != Some(steps) {
            warn_budget_changed_under_a_budget_schedule(&self.header, Some(steps));
        }
        self.config.max_steps = Some(steps);
    }
}

/// The per-step participation rollup line (the `1b.` block of `run()`) as a pure
/// function of this clock's counts — named fields, so the wording is testable.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StepVitals {
    /// Nets that RAN the trainer this clock: an update, or the crown's act-and-measure
    /// pass. Nets the lottery denied a ticket are NOT in here — they idled.
    pub stepped: usize,
    /// Nets that got a REAL weight update (`would_train - lottery_skips`).
    pub trained: usize,
    /// Live nets at the start of this clock.
    pub live: usize,
    /// Crown members in this clock's group that acted + measured but skipped the update.
    pub frozen: usize,
    /// Nets that WOULD have updated without the lottery (`group - crown`).
    pub would_train: usize,
    /// Updates the training lottery withheld this clock (0 when it is off).
    pub lottery_skips: usize,
    /// Nets that entered the pool this clock; they train from the NEXT clock.
    pub newborn: usize,
    /// Nets that served the join window and are not age-blocked: seat-eligible.
    pub eligible: usize,
    /// Nets still inside the join window (`elite_join_probation_steps`).
    pub probation: usize,
    /// Nets past the crown lifetime (`elite_crown_lifetime_steps`); 0 disarms it.
    pub age_blocked: usize,
    /// Seats actually dealt this clock (the crown's size).
    pub seats_held: usize,
    /// Dealable seats: `elite_count`, capped by the cullable population.
    pub seats_capacity: usize,
    /// The join window `k` this clock was graded against.
    pub join_k: usize,
    /// The crown lifetime `j` this clock was graded against (`0` = disarmed).
    pub lifetime_j: usize,
    /// Whether the crown is armed at all (`freeze_elites`). When it is not, the seat and
    /// gate legs are meaningless — every live net is seat-eligible and no seat can be
    /// dealt — so the line says so instead of printing a gate tally nobody is subject to.
    pub crown_armed: bool,
}

/// WHO the ticket lottery graded this clock, in one compact line: both counts are
/// always stated, and the SMALLER side is listed in full (capped at
/// [`LOTTERY_LIST_MAX`]) so the other side is the explicit complement — a clock where
/// 4 of 28 trained names the 4; a clock where 24 of 28 idled names the 24.
fn lottery_who_line(trained: &[String], withheld: &[String]) -> String {
    /// Hashes listed before the `+N` tail.
    const LOTTERY_LIST_MAX: usize = 12;
    let list = |v: &[String]| {
        let mut s = v
            .iter()
            .take(LOTTERY_LIST_MAX)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        if v.len() > LOTTERY_LIST_MAX {
            s.push_str(&format!(" +{}", v.len() - LOTTERY_LIST_MAX));
        }
        s
    };
    if withheld.len() <= trained.len() {
        format!(
            "{} trained · {} idled: {}",
            trained.len(),
            withheld.len(),
            list(withheld)
        )
    } else {
        format!(
            "{} trained: {} · {} idled",
            trained.len(),
            list(trained),
            withheld.len()
        )
    }
}

/// The per-step participation line, every composite printed with the arithmetic
/// that produced it: `ran A of L = T ticketed + C crown`, `held back H = …`,
/// `L = A + H`, the seat gate as a partition of the live count, and the lottery's
/// `would-train` budget.
///
/// Invariant: `stepped + held = live`. The crown is INSIDE `ran` (it ran the
/// trainer) and never in the held-back list, so the two counts cannot overshoot
/// `live`.
fn step_rollup_line(v: &StepVitals) -> String {
    let mut parts = Vec::new();
    // `ticketed` and `updating` are the same number: the nets that got a real
    // weight update. The word changes with the lottery, the count does not.
    let update_word = if v.lottery_skips > 0 {
        "ticketed"
    } else {
        "updating"
    };
    let head = if v.frozen > 0 {
        format!(
            "ran {} of {} = {} {} + {} crown",
            v.stepped, v.live, v.trained, update_word, v.frozen
        )
    } else {
        format!(
            "ran {} of {} = {} {}",
            v.stepped, v.live, v.trained, update_word
        )
    };
    parts.push(head);
    // Nets that did not RUN at all: the lottery's idle clocks and the newborns
    // sitting their birth clock out. The crown is absent by construction (it acted).
    let mut not_run = Vec::new();
    if v.lottery_skips > 0 {
        not_run.push(format!("{} idled (no train, no eval)", v.lottery_skips));
    }
    if v.newborn > 0 {
        not_run.push(format!("{} newborn (first step next clock)", v.newborn));
    }
    let held = v.lottery_skips + v.newborn;
    if !not_run.is_empty() {
        parts.push(format!("held back {} = {}", held, not_run.join(" + ")));
    }
    parts.push(format!("{} = {} + {}", v.live, v.stepped, held));
    if !v.crown_armed {
        // No crown ⇒ no seats and no gates: the two legs below would print a tally for a
        // restriction that is not in force (and a `seats 0 of k` that reads like a bug).
        parts.push(format!(
            "crown off (freeze_elites = false): 0 of {} seats, no gate — every live net is seat-eligible",
            v.live
        ));
    } else {
        parts.push(format!("seats {} of {}", v.seats_held, v.seats_capacity));
        // The seat gates partition the live count: eligible + join window/cull shield
        // + age-blocked = live. Printing the subtraction lets a reader check it.
        let lifetime = if v.lifetime_j == 0 {
            "age block off".to_string()
        } else {
            format!("{}-clock lifetime", v.lifetime_j)
        };
        parts.push(format!(
            "seat gates, over all {} nets: {} may sit = {} − {} in a join window or cull shield ({} clocks) − {} age-blocked ({})",
            v.live, v.eligible, v.live, v.probation, v.join_k, v.age_blocked, lifetime
        ));
    }
    if v.lottery_skips > 0 {
        // The lottery's budget, in its own units: the ticketed plus the idled are
        // exactly the nets that would have trained without it.
        parts.push(format!(
            "would-train {} = {} ticketed + {} idled",
            v.would_train, v.trained, v.lottery_skips
        ));
    }
    parts.join(" │ ")
}

/// Trainer-blob validation, the resume gate: compares `Trainer::describe` against
/// the blob in `engine.json` and names the drifted const. `None` on either side skips.
pub(crate) fn assert_trainer_blob_matches(
    recorded: &Option<serde_json::Value>,
    incoming_trainer: &crate::trainer::AnyTrainer,
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

/// Warn when a resume changes the BUDGET under a budget-relative LR schedule.
///
/// `cosine` and `onecycle` are functions of `step / max_steps` (see
/// `crate::trainer::schedule`), so extending a 4-clock run to 6 does not merely add
/// two clocks: the clocks already on disk were trained on a curve drawn against 4,
/// and the new ones are drawn against 6. The prefix stays honest and the frontier
/// stays exact — but the file is no longer the twin of a run made at the new budget,
/// and that is the sort of silent difference this engine warns about instead of
/// leaving to a diff someone runs a week later.
///
/// Not a refusal: extending a run is legitimate, and a resume that keeps the budget
/// (the crash case) is unaffected.
pub(crate) fn warn_budget_changed_under_a_budget_schedule(
    header: &RunHeader,
    new_budget: Option<usize>,
) {
    let Some(blob) = header.trainer.as_ref() else {
        return;
    };
    // The key is arm-specific: tabular records `lr_schedule_name` beside the legacy
    // boolean, RL records `lr_schedule` as the label. Accept either spelling.
    let label = blob
        .get("lr_schedule_name")
        .or_else(|| blob.get("lr_schedule"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let budget_relative = label.starts_with("cosine(") || label.starts_with("onecycle(");
    if !budget_relative {
        return;
    }
    let recorded = header.max_steps;
    if recorded != new_budget {
        tracing::warn!(
            "race resume: the step budget changes ({}) while the run trains with the budget-relative \
             schedule {label} — every clock on disk was drawn against the RECORDED budget, so this \
             file is not the twin of a run made at the new one; pass max_steps={} to keep the curve, \
             or launch the whole run at the budget you want",
            match (recorded, new_budget) {
                (Some(a), Some(b)) => format!("{a} → {b}"),
                (Some(a), None) => format!("{a} → unbounded"),
                (None, Some(b)) => format!("unbounded → {b}"),
                (None, None) => "unbounded".to_string(),
            },
            recorded.map(|n| n.to_string()).unwrap_or_default(),
        );
    }
}

/// Collect human-readable differences between two JSON blobs, recursing into
/// objects (path-prefixed keys) and comparing everything else by debug repr.
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
mod tests;
