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

use flodl::nn::Optimizer;
use flodl::tensor::Result;
use std::collections::HashMap;
use tracing::{debug, info};

use crate::utils::fitness::{Fitness, FitnessLabel, Metric};
use crate::graph::network::Network;
use crate::state::{
    ConfigSnapshot, NetState, RaceState, RunConfig, RunHeader, write_engine_json, write_net_state,
};
use crate::engine::evolve::RollKind;
use crate::evolution::selection::SelectionMethod;
use crate::trainer::stream::{BatchStream, PoolSplit};

use super::config::{RaceConfig, StopReason};
use super::format::{
    ReadingStyle, build_stamp, csv_field, fmt_cell, fmt_cell_opt, fmt_cfg, fmt_line,
    fmt_reading,
};
use super::smoothing::{RollingBuffer, WindowStats, rolling_stats};

// ── CoreEngine ───────────────────────────────────────────────────────────────

/// The continuous step-race loop: owns the step clock (the nets' recorded steps
/// ARE the clock), the batch stream, the live population and the rolling buffers.
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

// ── Construction ─────────────────────────────────────────────────────────────

/// One evolution roll's outcome, formatted for the single per-roll log line.
/// `survived`: crossover-only — did the child get inserted?
pub(crate) struct RollOutcome {
    /// Crossover-only: did the child get inserted?
    pub survived: bool,
    /// No cull-eligible slot existed (hard probation shield): the roll ended WITHOUT
    /// generating or inserting a child. `run` tells this apart from a spent roll.
    pub no_slot: bool,
    /// Human-readable outcome, rendered as the per-roll log line.
    pub detail: String,
}
impl RollOutcome {
    /// A crossover child that was admitted.
    pub fn survived(detail: impl Into<String>) -> Self {
        Self {
            survived: true,
            no_slot: false,
            detail: detail.into(),
        }
    }
    /// A roll that paid its attempts and produced no child.
    pub fn spent(detail: impl Into<String>) -> Self {
        Self {
            survived: false,
            no_slot: false,
            detail: detail.into(),
        }
    }
    /// A roll skipped: no cull-eligible victim existed this clock.
    pub fn no_slot(detail: impl Into<String>) -> Self {
        Self {
            survived: false,
            no_slot: true,
            detail: detail.into(),
        }
    }
}

/// Per-step evolve counters, reused by the `Minimal` table. Reset each step.
#[derive(Clone, Copy, Default)]
pub(crate) struct StepEvolve {
    /// Nets culled this step.
    pub culls: usize,
    /// Nets inserted this step (crossover survivors + immigrants).
    pub inserts: usize,
    /// Crossover rolls that fired this step.
    pub cross_fired: usize,
    /// Fired crossover rolls whose child was admitted.
    pub cross_survived: usize,
    /// Fired crossover rolls whose child was discarded.
    pub cross_discarded: usize,
    /// Immigrant rolls that fired this step.
    pub mutate_fired: usize,
}

/// A do-nothing optimizer — the freeze pass runs the net's normal trainer step
/// through this: gradients compute, nothing applies, the real optimizer is untouched.
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

/// Per-step RL volume summed over every net that stepped this clock — the env work
/// actually done. `nets` counts reporters (a silent trainer shows `—`), zero in Tabular.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RlVolume {
    /// Nets that reported `StepReport.rl` this step.
    pub nets: usize,
    /// Matches played by those nets, summed (train + eval).
    pub matches: usize,
    /// Train turns played by those nets, summed.
    pub train_turns: usize,
    /// Eval turns played by those nets, summed.
    pub eval_turns: usize,
}

impl RlVolume {
    /// Total environment turns this step (train + eval) — the `turns/match`
    /// numerator.
    pub(crate) fn turns(&self) -> usize {
        self.train_turns + self.eval_turns
    }

    /// The RL middle column of the rollup: the episode count, the turn identity
    /// `turns = train + eval`, the mean turns per episode, and the per-turn wall
    /// cost — `took` alone cannot answer "did the engine get slower?".
    pub(crate) fn label(&self, step_secs: f32) -> String {
        if self.nets == 0 || self.matches == 0 {
            return "episodes — │ turns — = train — + eval — │ turns/episode — │ ms/turn —"
                .to_string();
        }
        let turns = self.turns();
        let per_turn = if turns == 0 {
            "—".to_string()
        } else {
            fmt_line(step_secs * 1000.0 / turns as f32)
        };
        format!(
            "episodes {} │ turns {} = train {} + eval {} │ turns/episode {} │ {} ms/turn",
            self.matches,
            turns,
            self.train_turns,
            self.eval_turns,
            fmt_line(turns as f32 / self.matches as f32),
            per_turn
        )
    }
}

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
        // RL mode REQUIRES a reported fitness: there is no dataset, so a (pred, target)
        // scorer has nothing to score. Fail before the run starts, not three steps in.
        if tabular_data_dir.is_none() && fitness.is_computed() {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                "RL spec requires Fitness::reported(direction, label) — a Computed (pred, target) scorer has no dataset to score against. The trainer must supply the fitness value in StepReport.".to_string(),
            )
            .into());
        }
        // Catch-up toggles are RL-effective only — Tabular always catches up (one shared
        // stream). Data layout: `{train, test}` dirs are explicit, anything else is one pool.
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
        let train_eval_split_ratio = 0.2f32;
        let default_batch_size = 16usize;
        // The trainer owns batch geometry (Tabular only), resolved BEFORE the header so
        // engine.json records the stream shape the run used. `None` = no stream (RL).
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
        let mut header = RunHeader::from_race_options_at(
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
        // Settle the log sinks now that the run dir exists: `trace_file` opens the sidecar
        // and flushes the buffered start-up block into it; the default closes that buffer.
        super::logging::settle_run_dir(&run_dir, config.trace_file);

        // Seed the initial population internally — [imported champion?] + run-topologies +
        // random draws — deduped and re-rolled so the population always lands at `pop_size`.
        let run_topology_count = config.run_topologies.len();
        let mut founders: Vec<crate::graph::topology::Topology> =
            Vec::with_capacity(run_topology_count + 1);
        let mut seed_source: Option<crate::state::SeedSourceRecord> = None;
        // Manual founders WIN: `set_run_topologies` is the user naming the batch, so the
        // engine must not inject a net they did not ask for (or shift every ordinal).
        let manual_founders = !config.run_topologies.is_empty();
        if config.elite_previous_load && manual_founders {
            // CFY + always-visible: the inheritance receipt is a table that must show at
            // any log level (it is a fact about what the run INHERITED).
            crate::utils::cfy::emit_always(&crate::utils::cfy::card(
                "Inheritance",
                &crate::utils::cfy::kv(&[(
                    "founder blueprints".to_string(),
                    config.run_topologies.len().to_string(),
                )]),
                Some("elite_previous_load skipped — manual founders win (set_run_topologies)"),
            ));
        } else if config.elite_previous_load {
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
                                "champion holdout".to_string(),
                                champ
                                    .guardrail_mean
                                    .map(fmt_cell)
                                    .unwrap_or_else(|| "—".to_string()),
                            ),
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
                        Some(if n == 1 {
                            "champion certified by the guardrail"
                        } else {
                            "champion rank 1 certified by the guardrail; the rest by the source run's own elite ranking"
                        }),
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
                        guardrail_mean: champ.guardrail_mean,
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
        founders.extend(config.run_topologies.clone());
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

    /// Load every live net from `nets/` and replay it to its recorded step. Also
    /// reloads the checkpoint ledger so the history is the SAME.
    pub(crate) fn load_live_frontier(&mut self) -> Result<()> {
        // Replay-relevant knob guard: the smoothing window shapes every ranking decision,
        // so a resumed run must use the SAME window the original recorded.
        if self.config.smoothing_window != self.header.config.smoothing_window {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: smoothing_window mismatch — run recorded {} but the resume config sets {} — \
                 ranking semantics would change; set set_run_smoothing_window({}) (or omit) to resume",
                self.header.config.smoothing_window,
                self.config.smoothing_window,
                self.header.config.smoothing_window
            )).into());
        }
        // λ is replay-relevant for the same reason: it decides how much of a net's
        // spread is charged before it ranks. A run recorded BEFORE FIT stored no λ
        // (⇒ 0.0) because it ranked on the plain mean, so resuming it needs
        // `set_run_risk_aversion(0.0)` — the refusal names that value.
        if (self.config.risk_aversion - self.header.config.risk_aversion).abs() > 1e-6 {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: risk_aversion mismatch — run recorded λ={} but the resume config sets λ={} — \
                 ranking semantics would change; set set_run_risk_aversion({}) (or omit) to resume \
                 (a run that predates FIT recorded 0.00: it ranked on the plain mean)",
                fmt_cfg(self.header.config.risk_aversion),
                fmt_cfg(self.config.risk_aversion),
                fmt_cfg(self.header.config.risk_aversion)
            )).into());
        }
        // The parent draw is replay-relevant for the same class of reason: roulette and
        // tournament consume the roll's RNG stream differently AND prefer different
        // parents, so a resume at another method is a different search. A run from
        // BEFORE the knob recorded nothing (`""`) — it drew roulette, the default, so
        // that reads equal rather than refusing every legacy run.
        let recorded_selection = if self.header.config.crossover_selection.is_empty() {
            SelectionMethod::Roulette.to_string()
        } else {
            self.header.config.crossover_selection.clone()
        };
        let current_selection = self.config.crossover_selection.to_string();
        if recorded_selection != current_selection {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: crossover_selection mismatch — run recorded {recorded_selection} but the \
                 resume config sets {current_selection} — parent draws would change; call \
                 set_crossover_selection(...) with the recorded method (or omit it) to resume"
            )).into());
        }
        // The training lottery decides WHICH clocks were act-only, and those clocks are
        // persisted per net (`idle_spans`) for replay. A resume at a different Ω would
        // disagree with that ledger, so it is refused like K and λ — by the value, not
        // merely by "on/off": Ω = 0.5 idles a different set from Ω = 1.
        if (self.config.train_merit - self.header.config.train_merit).abs() > 1e-6 {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: train_merit (Ω) mismatch — run recorded Ω={} but the resume \
                 config sets Ω={} — which nets received a train-eval ticket would change; \
                 call set_run_train_merit({}) (or omit it) to resume",
                fmt_cfg(self.header.config.train_merit),
                fmt_cfg(self.config.train_merit),
                fmt_cfg(self.header.config.train_merit)
            ))
            .into());
        }
            // Cull policies decide which net the NEXT roll removes, so a resume at a
            // different policy is a different search — refused like the parent draw above.
            // This is the ONE rule for this gate: guard a knob when it changes the future
            // draws the engine will consume. The rest stay forward-only on purpose (rolls,
            // probs, pools, probation windows, elite_count, freeze_elites): they change
            // what happens next but never what the recorded ledger says happened, and
            // their numeric headers predate any safe legacy default (a missing f32 reads
            // 0.0, not the 0.5 those runs used).
            // `""` = a header written before the field existed; those runs culled `worst`
            // (`RaceConfig::defaults()`), so it reads equal instead of refusing.
        for (label, setter, recorded, current) in [
            (
                "crossover_cull_policy",
                "set_crossover_cull_policy",
                self.header.config.crossover_cull_policy.clone(),
                self.config.crossover_cull_policy.to_string(),
            ),
            (
                "mutation_cull_policy",
                "set_mutation_cull_policy",
                self.header.config.mutation_cull_policy.clone(),
                self.config.mutation_cull_policy.to_string(),
            ),
        ] {
            let recorded = if recorded.is_empty() {
                "worst".to_string()
            } else {
                recorded
            };
            if recorded != current {
                return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                    "resume: {label} mismatch — run recorded {recorded} but the resume config \
                     sets {current} — which net the next cull removes would change; call \
                     {setter}(...) with the recorded policy (or omit it) to resume"
                ))
                .into());
            }
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
        // CFY: per-net replay rows collected here, emitted as one table after the load.
        let mut replayed: Vec<Vec<String>> = Vec::new();
        // The smoothing windows themselves are not persisted per net (only its LAST
        // reading is), but `history.csv` holds every reading the run took. Replaying it
        // keeps each window warm, so the ranking score, the elite set and the guardrail's
        // race-reading baseline resume where the recorded run stopped instead of
        // restarting from a single cold reading.
        let history = Self::load_history_windows(&run_dir, self.config.smoothing_window);
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
            // Ghost guard (defensive): a rejected child once leaked as a live state; a net
            // born at step N cannot have trained clocks < N — warn and skip, don't fail.
            let entered = state.entered_at_step;
            if entered > 0 {
                if let Some(lm) = state.last_metrics.as_ref() {
                    if entered > lm.step + 1 {
                        tracing::warn!(
                            "race resume: skipping GHOST net {} (entered_at_step {} but last trained clock {} — a rejected child that leaked into nets/)",
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

            // Insert the replayed net into the live maps, its rolling buffers seeded from the
            // replayed trajectory so the smoothed stats are warm on resume, not cold.
            let mut state = child.state.clone();
            state.elite_probation_age = child.state.elite_probation_age;
            state.elite_seat_age = child.state.elite_seat_age;
            self.state.insert(state, None);
            self.networks.insert(child.state.hash.clone(), child.net);
            self.optimizers
                .insert(child.state.hash.clone(), child.optimizer);
            // Rolling buffers seeded from the run's own recorded readings — the smoothed
            // stats are warm on resume, not cold; ranking resumes where it left off.
            let mut buf = RollingBuffer::new(self.config.smoothing_window);
            let mut train_buf = RollingBuffer::new(self.config.smoothing_window);
            let mut eval_buf = RollingBuffer::new(self.config.smoothing_window);
            match history.get(&hash) {
                // The recorded runway, oldest → newest: the same window the live run held.
                Some(rows) if !rows.is_empty() => {
                    for (fitness, train_loss, eval_loss) in rows {
                        buf.push(*fitness);
                        train_buf.push(*train_loss);
                        if let Some(eval) = eval_loss {
                            eval_buf.push(*eval);
                        }
                    }
                }
                // No `history.csv` (csv_export off, or a run recorded before it): fall back
                // to the single last reading, which is all the net state carries.
                _ => {
                    if let Some(m) = self.state.net(&hash).and_then(|s| s.last_metrics.as_ref()) {
                        buf.push(m.fitness);
                        train_buf.push(m.train_loss);
                        if let Some(e) = m.eval_loss {
                            eval_buf.push(e);
                        }
                    }
                }
            }
            self.rolling_fitness.insert(hash.clone(), buf);
            self.rolling_train.insert(hash.clone(), train_buf);
            self.rolling_eval.insert(hash.clone(), eval_buf);
            loaded += 1;
            replayed.push(vec![hash.clone(), step.to_string()]);
        }
        if loaded != self.config.pop_size {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: expected {} live nets in population (per config.pop_size), but found {} live nets in {}",
                self.config.pop_size, loaded, nets_dir.display()
            )).into());
        }
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            "Race resume",
            &crate::utils::cfy::table(&["hash", "replayed step"], &replayed),
            Some(&format!("loaded {loaded} nets from {}", run_dir.display())),
        ));
        Ok(())
    }

    /// The last `window` readings per net, oldest → newest, read back from
    /// `<run_dir>/history.csv` (`fitness`, `train_loss`, `eval_loss` columns, located
    /// by the header so appended FIT/informative columns cannot shift them). Empty
    /// when the run exported no CSV — the caller then falls back to the one reading
    /// the net state carries.
    fn load_history_windows(
        run_dir: &std::path::Path,
        window: usize,
    ) -> HashMap<String, Vec<(f32, f32, Option<f32>)>> {
        let Ok(raw) = std::fs::read_to_string(run_dir.join("history.csv")) else {
            return HashMap::new();
        };
        let mut lines = raw.lines();
        let header = lines.next().unwrap_or_default();
        let cols = crate::engine::format::csv_split_line(header);
        let find = |name: &str| cols.iter().position(|c| *c == name);
        let (Some(hash_i), Some(train_i), Some(eval_i), Some(fit_i)) = (
            find("hash"),
            find("train_loss"),
            find("eval_loss"),
            find("fitness"),
        ) else {
            return HashMap::new();
        };
        let mut out: HashMap<String, std::collections::VecDeque<(f32, f32, Option<f32>)>> =
            HashMap::new();
        for line in lines {
            let fields = crate::engine::format::csv_split_line(line);
            let needed = hash_i.max(train_i).max(eval_i).max(fit_i);
            if fields.len() <= needed {
                continue;
            }
            let Ok(train_loss) = fields[train_i].parse::<f32>() else {
                continue;
            };
            let Ok(fitness) = fields[fit_i].parse::<f32>() else {
                continue;
            };
            let eval_loss = fields[eval_i].parse::<f32>().ok();
            let q = out.entry(fields[hash_i].to_string()).or_default();
            q.push_back((fitness, train_loss, eval_loss));
            if q.len() > window {
                q.pop_front();
            }
        }
        out.into_iter()
            .map(|(hash, queue)| (hash, queue.into_iter().collect()))
            .collect()
    }

    // ── Population setup ────────────────────────────────────────────

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
            (
                "run dir".to_string(),
                self.run_dir.display().to_string(),
            ),
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
            if let Some(m) = src.guardrail_mean {
                rows.push(("champion holdout".to_string(), fmt_cell(m)));
            }
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
        // a post-construction config edit must not desync era numbering.
        if let Some(stream) = self.stream.as_mut() {
            stream.set_run_checkpoint_every(self.config.checkpoint_every);
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
                    "checkpoint_every".to_string(),
                    self.config.checkpoint_every.to_string(),
                ),
                (
                    "crossover_rolls".to_string(),
                    self.config.crossover_rolls.to_string(),
                ),
                ("mutate_rolls".to_string(), self.config.mutate_rolls.to_string()),
                (
                    "crossover_prob".to_string(),
                    fmt_cfg(self.config.crossover_prob),
                ),
                (
                    "crossover_selection".to_string(),
                    self.config.crossover_selection.to_string(),
                ),
                ("mutate_prob".to_string(), fmt_cfg(self.config.mutate_prob)),
                ("elite_count".to_string(), self.config.elite_count.to_string()),
                (
                    "freeze_elites".to_string(),
                    self.config.freeze_elites.to_string(),
                ),
                (
                    "train_merit (Ω)".to_string(),
                    fmt_cfg(self.config.train_merit),
                ),
                (
                    "guardrail_matches".to_string(),
                    self.config.guardrail_matches.to_string(),
                ),
                (
                    "fitness".to_string(),
                    format!("{} ({})", self.meta_ctx.fitness_label, self.meta_ctx.direction),
                ),
                ("input_dim".to_string(), self.meta_ctx.input_dim.to_string()),
                ("output_dim".to_string(), self.meta_ctx.output_dim.to_string()),
                (
                    "batch_size".to_string(),
                    self.meta_ctx
                        .batch_size
                        .map(|b| b.to_string())
                        .unwrap_or_else(|| "—".to_string()),
                ),
                ("dropout".to_string(), format!("{}", self.meta_ctx.dropout_prob)),
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
                ("worst_save", self.config.worst_save_topology.to_string()),
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
                        "score{} = window avg {} λ × window spread (what the ranking sorts on) · λ={}",
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
                if let Some(reason) = self.check_stop(clock) {
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
                if self.verbose_detail() && (!cross_missed.is_empty() || !mutate_missed.is_empty()) {
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
                if let Err(e) = self.step_one_net_in_frame(hash, clock, lottery) {
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
                // `history_flush_each`: push the CSV buffers to disk every clock instead of
                // waiting for the checkpoint tick. Off by default — buffering is the fast
                // path, this is the "a kill -9 must not lose the last step" knob.
            if self.config.history_flush_each {
                self.flush_csv_exports()?;
            }

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
            if self.config.checkpoint_every > 0 && clock > 0 && clock % self.config.checkpoint_every == 0
            {
                // Snapshot the live frontier so a hard kill loses at most one tick.
                if let Err(e) = self.write_live_frontier_states() {
                    tracing::warn!("checkpoint live states write failed: {e}");
                }
                // Elite weight snapshot (hard-kill durability): the frontier states carry
                // topology + counters every tick; this adds the top-k WEIGHTS.
                if self.config.elite_checkpoint_weights {
                    if let Err(e) = self.write_checkpoint_elite_safetensors() {
                        tracing::warn!("checkpoint elite snapshot failed: {e}");
                    }
                }
                // Flush metrics at the tick too, so an interrupted run keeps per-step
                // history up to its last tick instead of the whole in-memory buffer.
                if let Err(e) = self.flush_csv_exports() {
                    tracing::warn!("checkpoint CSV flush failed: {e}");
                }
                if self.verbose_detail() {
                    // Name the files, not the internals: "frontier" and "flushed"
                    // are engine vocabulary a reader cannot map to a path.
                    let elite = if self.config.elite_checkpoint_weights {
                        " + elite weights (checkpoint-elite-*.safetensors)"
                    } else {
                        ""
                    };
                    info!(
                        "step {} │ checkpoint │ saved: live net states (nets/*.json){elite} + CSV buffers written",
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
        self.write_champion_safetensors()?;
        self.write_worst_artifacts()?;
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
        // Stamp the live nets' final readings first: the frontier write below
        // then carries `final_*` on every net file, culled or not.
        self.stamp_live_final_scores();
        self.write_live_frontier_states()?;
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

    // ── Smoothed fitness ────────────────────────────────────────────────────

    /// Per-net RISK-ADJUSTED fitness over the live population, in `live_hashes()`
    /// order — the shared input for ranking and `RaceSnapshot` (FIT: the window
    /// mean charged `λ` window-stds for its own spread).
    pub(crate) fn smoothed_fitness_values(&self) -> Vec<f32> {
        // Only nets with a non-empty rolling buffer get a value — a caught-up child
        // qualifies at once, and phantom 0.0 values never drag the mean down.
        let direction = self.fitness.direction();
        let lambda = self.config.risk_aversion;
        self.state
            .live_hashes()
            .into_iter()
            .filter_map(|h| self.race_reading_of(&h))
            .map(|s| s.risk_adjusted(direction, lambda))
            .collect()
    }

        /// The net's ranking value (FIT): the risk-adjusted window score, `mean ∓ λ·σ`
        /// over `smoothing_window` step fitnesses — what every ranking decision reads.
        /// `None` = no fitness yet. The READABLE level behind it is
        /// [`CoreEngine::race_reading_of`]'s `mean`; at `λ = 0` the two are identical.
    pub fn smoothed_fitness_of(&self, hash: &str) -> Option<f32> {
        let direction = self.fitness.direction();
        let lambda = self.config.risk_aversion;
        self.race_reading_of(hash)
            .map(|s| s.risk_adjusted(direction, lambda))
    }

    /// The net's RACE READING with its own error bar: the smoothed mean plus how many
    /// step readings are behind it and how far apart they were ([`WindowStats`]).
    pub(crate) fn race_reading_of(&self, hash: &str) -> Option<WindowStats> {
        self.rolling_fitness.get(hash).and_then(rolling_stats)
    }

    /// The post-race holdout guardrail: score EVERY elite (rank 1 = the champion) on
    /// fresh unseen draws through the run's own trainer — `elite_count` nets ×
    /// `guardrail_matches` draws each (the core's mode-neutral word for one exam unit:
    /// a holdout batch here, an episode in an RL run). Returns the champion's verdict; the rest are
    /// printed in the card's table and recorded in `engine.json`.
    /// `None` = no champion (each case logs a warning), the champion produced no
    /// verdict, or the exam is switched off (`guardrail_matches = 0`).
    pub fn guardrail(&mut self) -> Option<crate::engine::guardrail::GuardrailVerdict> {
        // `0` matches = the exam is OFF (see `set_elite_guardrail_matches`): return
        // BEFORE anything expensive or observable — no champion lookup, no holdout
        // game, no verdict. `engine.json` keeps any earlier record, and the
        // next run's inheritance never depended on the verdict (`elites.json`).
        if self.config.guardrail_matches == 0 {
            crate::utils::cfy::emit(&crate::utils::cfy::card(
                "Guardrail",
                &crate::utils::cfy::kv(&[(
                    "guardrail_matches".to_string(),
                    "0".to_string(),
                )]),
                Some("skipped — no holdout exam configured; nothing recorded"),
            ));
            return None;
        }
        // The champion set is written by the stop-time writers; recompute it when empty
        // so the guardrail never depends on whether a writer happened to run.
        if self.champions.is_empty() {
            self.record_champions();
        }
        let champion = self.champion_for_guardrail()?;
        let matches = self.config.guardrail_matches;
        // EVERY elite takes the exam, not just rank 1: `champions` is the stop-time
        // elite snapshot (top `elite_count`), i.e. exactly the nets the run exported.
        // The champion leads, so its verdict is the one that certifies the run.
        let mut exam_set: Vec<String> = Vec::with_capacity(self.champions.len() + 1);
        exam_set.push(champion.clone());
        for hash in &self.champions {
            if !exam_set.contains(hash) {
                exam_set.push(hash.clone());
            }
        }
        let mut exams: Vec<crate::engine::guardrail::GuardrailVerdict> =
            Vec::with_capacity(exam_set.len());
        for hash in &exam_set {
            // The race reading AND its error bar, read from the LIVE rolling buffer
            // before scoring: the spread is what separates "collapsed" from "an
            // unresolvable mean".
            let race = self.race_reading_of(hash);
            let Some(net) = self.networks.get_mut(hash.as_str()) else {
                // Not in memory (already gone from the population): there is nothing
                // honest to score, so this net's exam is skipped rather than run on a
                // net rebuilt from disk — a fresh-init net measures like noise.
                tracing::warn!(
                    "guardrail: elite {} is not one of the live nets — exam skipped",
                    &hash[..8.min(hash.len())]
                );
                continue;
            };
            // `eval()` first: the resting state a reloaded net would get for free, and the
            // state the holdout MUST be measured in (dropout would sample a random policy).
            flodl::Module::eval(net);
            let Some(mut verdict) = crate::engine::guardrail::score_net(
                net,
                hash,
                race.map(|r| r.mean),
                matches,
                &mut self.trainer,
                |t, net, game_i| t.holdout_score(net, game_i),
            ) else {
                tracing::warn!(
                    "guardrail: no holdout game scored on elite {} — exam skipped",
                    &hash[..8.min(hash.len())]
                );
                continue;
            };
            // Carry the race side's noise onto the verdict, so the example's print
            // and this module's own table quote the same error bar.
            verdict.race_se = race.and_then(|r| r.se());
            verdict.race_samples = race.map(|r| r.n).unwrap_or(0);
            verdict.race_latest = race.map(|r| r.latest);
            exams.push(verdict);
        }
        // Rank 1 IS the certification: with no champion verdict nothing is recorded,
        // exactly like the old single-champion exam.
        let Some(verdict) = exams.iter().find(|v| v.hash == champion).cloned() else {
            tracing::warn!(
                "guardrail: the champion produced no holdout verdict — nothing recorded"
            );
            return None;
        };
        let direction = self.fitness.direction();
        let champion_race = self.race_reading_of(&verdict.hash);
        // One row per examined elite, in exam (rank) order: the race reading on the
        // left, the fresh-game exam on the right, the graded tag last. The champion's
        // full sentence — which carries the mechanism — rides as the card's footnote.
        let mut rows: Vec<Vec<String>> = Vec::with_capacity(exams.len());
        let mut elite_records: Vec<crate::state::GuardrailElite> = Vec::with_capacity(exams.len());
        let mut champion_line = String::new();
        for (rank0, v) in exams.iter().enumerate() {
            let race = self.race_reading_of(&v.hash);
            // The per-elite window context (the champion's context line, for EVERY
            // elite): its min..max, and the SECOND reference line — the exam graded
            // against this net's LATEST reading, which the window mean lags. Both read
            // the same `race`, so a row can never pair one net's window with another's
            // numbers.
            let latest_tag = crate::engine::guardrail::latest_agreement_grade(
                race.map(|r| r.latest),
                race.map(|r| r.std),
                v.mean(),
                v.se(),
                v.scores.len(),
                direction,
            );
            // ONE grading for the table and the sentence (`agreement_grade`), so a row
            // can never contradict its own verdict text.
            let grade = crate::engine::guardrail::agreement_grade(
                v.race_smoothed,
                v.race_se,
                v.mean(),
                v.se(),
                v.scores.len(),
                direction,
            );
            let line = crate::engine::guardrail::agreement_verdict_noisy(
                v.race_smoothed,
                v.race_se,
                Some(v.race_samples).filter(|n| *n > 0),
                v.race_latest,
                v.mean(),
                v.se(),
                v.scores.len(),
                direction,
            );
            if v.hash == verdict.hash {
                champion_line = line.clone();
            }
                // GUARDRAIL DATA ONLY. The race side (`race μ`, its n, its SE, the
                // window, the latest reading) is ALREADY in the `Final field` table
                // above, in the same rank order and under the same hash — repeating it
                // here made this table a wide restatement of that one. Every row below
                // is a number the exam itself produced (plus its verdict tag); the
                // elite table's `reading` cell is this row's race reference.
            rows.push(vec![
                (rank0 + 1).to_string(),
                v.hash[..8.min(v.hash.len())].to_string(),
                fmt_cell_opt(&v.mean()),
                fmt_cell_opt(&v.std()),
                fmt_cell_opt(&v.se()),
                // `scored/played`: `score_net` skips a game whose scorer errors, so
                // a degraded exam must be visible next to the full one.
                format!("{}/{}", v.scores.len(), matches),
                grade.tag().to_string(),
            ]);
            elite_records.push(crate::state::GuardrailElite {
                rank: rank0 + 1,
                hash: v.hash.clone(),
                params: self.net_params(&v.hash),
                race_mean: v.race_smoothed,
                race_std: race.map(|r| r.std),
                race_score: self.smoothed_fitness_of(&v.hash),
                race_se: v.race_se,
                race_samples: v.race_samples,
                race_latest: v.race_latest,
                race_min: race.map(|r| r.min),
                race_max: race.map(|r| r.max),
                latest_agreement: latest_tag.map(|t| t.tag().to_string()),
                holdout_mean: v.mean(),
                holdout_std: v.std(),
                matches_scored: v.scores.len(),
            matches_attempted: matches,
                agreement: grade.tag().to_string(),
                verdict: line,
            });
        }
        // NO champion-context block above the table. Every elite now carries its own
        // window and its own exam grade in the rows below, so a champion-only block
        // read as if rank 1 were the best net — and the exam is exactly the measurement
        // that can contradict that (a lower-ranked elite can out-score it). The
        // champion's numbers stay in the footnote AND in `engine.json`; the identifiers
        // the old block carried (hash, rank 1 of N) lead the footnote instead.
            // The core engine's wording stays MODE-NEUTRAL: one exam unit is a DRAW here
            // (a holdout batch in a tabular run, an episode in an RL one), and the RL
            // examples describe their own units (`turns`, `games`) in their own prose.
            // Naming the core's unit "game" made every tabular run read as if it had a
            // simulator somewhere in it.
        let title = format!(
            "Guardrail (elite set on fresh unseen draws — {} net(s) × {} draw(s) each)",
            exams.len(),
            matches
        );
        let body = crate::utils::cfy::table(
            &["rank", "hash", "exam avg", "exam spread", "exam SE", "draws", "status"],
            &rows,
        )
        .to_string();
        // The footnote names WHICH net it certifies — the rank-1 identity the dropped
        // context block used to carry, at no layout cost — and carries the column legend.
        // It is the SAME place every other card puts its legend; the old one-cell
        // `columns` table above the data was the only card in the log shaped that way.
        let champion_note = format!(
            "{} (rank 1 of {}) — {champion_line} · exam avg = mean unseen-draw score · exam spread = per-draw spread · exam SE = spread/√draws · draws = scored/played · status = the exam graded against this net's race reading (`Final field` above, same ranks and hashes)",
            &verdict.hash[..8.min(verdict.hash.len())],
            self.champions.len().max(1)
        );
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            &title,
            &body,
            Some(&champion_note),
        ));
        // Record it: the guardrail is a run OUTPUT (the only reading on unseen draws), so
        // it lives in engine.json and the next experiment inherits its champion. Rank 1
        // keeps the top-level block (the seeding contract); every exam is in `elites`.
        let record = crate::state::GuardrailRecord {
            champion_hash: verdict.hash.clone(),
            champion_params: self.net_params(&verdict.hash),
            race_smoothed: verdict.race_smoothed,
            race_std: champion_race.map(|r| r.std),
            race_score: self.smoothed_fitness_of(&verdict.hash),
            race_se: verdict.race_se,
            race_samples: verdict.race_samples,
            race_latest: verdict.race_latest,
            holdout_mean: verdict.mean(),
            holdout_std: verdict.std(),
            matches_scored: verdict.scores.len(),
            verdict: champion_line,
            elites: elite_records,
        };
        self.header.set_guardrail(Some(record.clone()));
        if let Err(e) = crate::state::write_guardrail_record(&self.run_dir, &record) {
            tracing::warn!("engine.json guardrail record failed: {e}");
        }
        Some(verdict)
    }

    /// A live net's parameter count (`NetMeta::params`) — the guardrail
    /// tie-break key.
    fn net_params(&self, hash: &str) -> Option<i64> {
        self.state.net(hash).and_then(|s| s.meta.params)
    }

        /// The net the guardrail certifies: the HEAD of the recorded champion ranking —
        /// the same order the `Final field` table prints, so the table's rank 1 and the
        /// certified champion cannot disagree (they did: the guardrail used to re-rank
        /// the ties by fewest parameters while the table kept the sort's own order, so a
        /// 500.0000/500.0000 pair could show rank 1 in one card and rank 2 in the other).
        ///
        /// The tie-break itself lives in [`Self::ranked_live_impl`], which owns the one
        /// order every card reads.
    fn champion_for_guardrail(&self) -> Option<String> {
        self.champions.first().cloned()
    }

    /// THE ranking primitive: live nets paired with smoothed fitness, ordered
    /// best-first. `require_measured` is the eligibility contract (true = ranked_live).
    fn ranked_live_impl(&self, require_measured: bool) -> Vec<(String, f32)> {
        let direction = self.fitness.direction();
        let lambda = self.config.risk_aversion;
            // `(hash, score, params)`: the size rides along so the comparator below is a
            // pure function of the tuple — ONE map lookup per net instead of two per
            // comparison, and no borrow gymnastics inside the sort.
        let mut ranked: Vec<(String, f32, i64)> = self
            .state
            .live_hashes()
            .into_iter()
            .filter_map(|h| match self.rolling_fitness.get(&h).and_then(rolling_stats) {
                // FIT: the ranking value is the risk-adjusted score, so a net with a
                // wild window pays for its spread before it is ever crowned, culled
                // or drawn as a parent.
                Some(s) if !require_measured || s.n > 0 => {
                    Some((h, s.risk_adjusted(direction, lambda)))
                }
                Some(_) => None,
                // Never measured: the parent-selection contract counts it as 0.0.
                None if !require_measured => Some((h, 0.0)),
                None => None,
            })
            .map(|(h, score)| {
                let params = self.net_params(&h).unwrap_or(i64::MAX);
                (h, score, params)
            })
            .collect();
            // A TOTAL order, so no card's numbering depends on hash-map or insertion
            // order: the risk-adjusted score first, then the SMALLER net (equal score ⇒
            // prefer the cheaper one to export — the rule the guardrail champion used to
            // apply on its own, now shared by every card and by parent/cull selection),
            // then the hash, which is unique and makes the order total even for two
            // identical-score, identical-size nets. Ties here are EXACT equality: a
            // tolerance would not be a total order, and the sort is free to reorder
            // anything its comparator calls equal.
        ranked.sort_by(|a, b| {
            direction
                .cmp(b.1, a.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.into_iter().map(|(h, score, _)| (h, score)).collect()
    }

    /// Live nets that have a measured (non-empty) rolling buffer, best-first.
    pub(crate) fn ranked_live(&self) -> Vec<(String, f32)> {
        self.ranked_live_impl(true)
    }

    /// Every live net, best-first; a net with no measurement yet counts as
    /// 0.0 — the parent-selection contract.
    pub(crate) fn ranked_live_all(&self) -> Vec<(String, f32)> {
        self.ranked_live_impl(false)
    }

    /// The elite set: the top `config.elite_count` live nets by smoothed fitness that
    /// pass the gate (probation `≥ k`, seat age `< j`); any seat exit resets probation.
    pub(crate) fn elite_hashes(&self) -> Vec<String> {
        let k = self
            .config
            .elite_count
            .min(self.state.live_count().saturating_sub(1));
        if k == 0 {
            return Vec::new();
        }
        let probation = self.config.elite_join_probation_steps;
        let lifetime = self.config.elite_crown_lifetime_steps;
        // The REGULAR shield is a precondition, checked first: a net may not be elite
        // before it could be culled (same born-based channel window).
        let clock = self.step_clock();
        self.ranked_live()
            .into_iter()
            .filter(|(h, _)| {
                self.state.net(h).is_some_and(|s| {
                    // (a) cull shield passed (born-based, never resets) — Rust `on_probation`.
                    !self.on_probation(h, clock)
                        // (b) join window served (resets on every seat exit) — Rust `elite_probation_age`.
                        && s.elite_probation_age >= probation
                        // (c) seat still inside its lifetime (0 = no lifetime).
                        && (lifetime == 0 || s.elite_seat_age < lifetime)
                })
            })
            .take(k)
            .map(|(h, _)| h)
            .collect()
    }

    /// KISS crown refresh (the FREEZE-ELITES model, in one place): the crown IS the
    /// elite take(k) set, recomputed each step; the diff prints one line per transition.
    pub(crate) fn refresh_crown(&mut self, clock: usize) {
        let enabled = self.config.freeze_elites;
        let new_crown: std::collections::HashSet<String> = if enabled {
            self.elite_hashes().into_iter().collect()
        } else {
            // Freeze off: nobody holds a seat.
            std::collections::HashSet::new()
        };
        let mut entering: Vec<String> = new_crown
            .iter()
            .filter(|h| !self.frozen_crown.contains(*h))
            .cloned()
            .collect();
        let mut leaving: Vec<String> = self
            .frozen_crown
            .iter()
            .filter(|h| !new_crown.contains(*h))
            .cloned()
            .collect();
        // Print order: seats then exits, each in rank order — the log reads
        // as one deal of the crown, not k scattered events.
        entering.sort_by(|a, b| {
            self.rank_position(a)
                .unwrap_or(usize::MAX)
                .cmp(&self.rank_position(b).unwrap_or(usize::MAX))
        });
        leaving.sort();
        // 3. Seat exits: record WHY each net left before resetting probation (ANY exit
        // restarts it) — inferring the cause from the rank contradicted itself.
        let lifetime = self.config.elite_crown_lifetime_steps;
        let mut exit_reason: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for h in &leaving {
            if let Some(s) = self.state.net(h) {                let short = &h[..8.min(h.len())];
                let reason = if lifetime > 0 && s.elite_seat_age >= lifetime {
                    // A rotation, not a defeat: the seat term simply ran out.
                    format!(
                        "seat term complete → {short} turned over after {} of {} clocks",
                        s.elite_seat_age, lifetime
                    )
                } else {
                    format!(
                        "seat term complete → {short} turned over, outranked among the eligible nets"
                    )
                };
                exit_reason.insert(h.clone(), reason);
            }
        }
        for h in &leaving {
            if let Some(s) = self.state.net_mut(h) {
                s.elite_probation_age = 0;
                s.elite_seat_age = 0;
            }
        }
        // 4. Settle the crown BEFORE the lines: every field they print is the
        //    outcome of THIS deal, not the previous clock's state.
        self.frozen_crown = new_crown;
        let seat = self.frozen_crown.len();
        let capacity = self
            .config
            .elite_count
            .min(self.state.live_count().saturating_sub(1));
        // 5. One freeze line per net entering (rank order), one dethrone line
        //    per net leaving; a net holding rank is silent.
        for (idx, h) in entering.iter().enumerate() {
            let pos = self.rank_position(h).unwrap_or(0);
            // `seat N/M` is the SEAT INDEX inside this deal, NOT the net's global rank: with
            // probation on, rank 18 can take the 10th seat — the rank is named separately.
            let rank_note = if pos == 0 || pos == idx + 1 {
                String::new()
            } else {
                format!(" (rank {pos})")
            };
            let fit = self.smoothed_fitness_of(h);
            let age = self
                .state
                .net(h)
                .map(|s| clock.saturating_sub(s.entered_at_step));
            let params = self
                .networks
                .get(h)
                .map(|n| n.topology_blueprint().param_estimate());
            // Trainings vs clocks LIVED, and the gate in words. The old pair looked like a
            // training count but was the age of the WEIGHTS (children inherit it).
            let trainings = self.state.net(h).map(|s| s.step).unwrap_or(0);
            let age_now = age.unwrap_or(0);
            // The two counters that both look like `n/5` get their own names: the
            // join window (`wait`) and the seat lifetime (`this seat held`).
            let (join_age, seat_age) = self
                .state
                .net(h)
                .map(|s| (s.elite_probation_age, s.elite_seat_age))
                .unwrap_or((0, 0));
            let join_k = self.config.elite_join_probation_steps;
            let lifetime = self.config.elite_crown_lifetime_steps;
            let wait = format!("wait served {} of {join_k}", join_age.min(join_k));
            let held = if lifetime == 0 {
                format!("this seat held {seat_age} clocks (lifetime off)")
            } else {
                format!("this seat held {seat_age} of {lifetime} clocks")
            };
                // The reading behind the rank: `smt` on the line is the risk-adjusted
                // score (FIT); this is `μ ±σ (SE, n)` — the level it was charged from, the
                // spread it paid, and how well that level is pinned down. `—` with no window.
            let reading = self.race_reading_of(h);
            let reading_txt = reading.map_or_else(
                || "—".to_string(),
                |r| {
                    fmt_reading(
                        Some(r.mean),
                        (r.n > 1).then_some(r.std),
                        r.n,
                        ReadingStyle::Line,
                    )
                },
            );
            let params_txt = params
                .map(|p| format!("{p} parameters"))
                .unwrap_or_else(|| "parameters —".to_string());
            // The score, with the arithmetic that produced it: `score = avg ∓ λ·spread`.
            let score_cell = crate::engine::reporting::seat_score_cell(
                fit,
                reading.as_ref(),
                self.fitness.direction(),
            );
            info!(
                "step {} │ seat {} of {seat} → {}{} │ {} │ reading {} │ age {} clocks │ {} │ {} │ {wait} │ {held}",
                clock,
                idx + 1,
                &h[..8.min(h.len())],
                rank_note,
                score_cell,
                reading_txt,
                age_now,
                crate::engine::reporting::trained_note(trainings, age_now),
                params_txt,
            );
        }
        for h in &leaving {
            // Freeze/dethrone chatter belongs to the elite process only —
            // a disabled refresh (freeze off) just clears.
            if !enabled {
                continue;
            }
            let fit = self.smoothed_fitness_of(h);
            // k == 1 keeps the crown-moved pointer (the new #1 really did replace the old).
            // Multi-seat crowns use MEMBERSHIP language: with k seats churn has no one taker.
            let detail = if seat <= 1 {
                let taker = self.frozen_crown.iter().next().cloned().unwrap_or_default();
                let taker_fit = if taker.is_empty() {
                    None
                } else {
                    self.smoothed_fitness_of(&taker)
                };
                // The scores themselves; the line's `score = …` cell already names
                // the quantity, so repeating the tag here would read twice.
                let gap_txt = match (fit, taker_fit) {
                    (Some(a), Some(b)) => format!(
                        " ({}{} → {}{})",
                        self.fitness.direction().arrow(),
                        fmt_line(a),
                        self.fitness.direction().arrow(),
                        fmt_line(b)
                    ),
                    (Some(a), None) => format!(
                        " ({}{})",
                        self.fitness.direction().arrow(),
                        fmt_line(a)
                    ),
                    _ => String::new(),
                };
                if taker.is_empty() {
                    String::new() // crown is empty now — no successor to name
                } else {
                    format!(
                        " — crown moved {} → {}{}",
                        &h[..8.min(h.len())],
                        &taker[..8.min(taker.len())],
                        gap_txt
                    )
                }
            } else {
                // Multi-seat crowns add nothing here: the REASON and the net's current rank are
                // printed by the caller, and guessing a cause produced self-contradictory lines.
                String::new()
            };
            // Trainings vs clocks LIVED for this net — the same pairing the seat line prints
            // (see `trained_note`); the old pair was weight staleness, not a training count.
            let (trainings, age) = match self.state.net(h) {
                Some(s) => (s.step, clock.saturating_sub(s.entered_at_step)),
                None => (0, 0),
            };
            let reason = exit_reason
                .get(h)
                .cloned()
                .unwrap_or_else(|| "seat not re-dealt".to_string());
            // Where the net stands as it leaves: a net aged out at peak can still rank #1,
            // so its rank was never the reason it lost the seat.
            let rank_note = match self.rank_position(h) {
                Some(r) => format!(" (still rank {r} of {} overall)", self.state.live_count()),
                None => String::new(),
            };
            // The ex-elite just re-entered the JOIN WINDOW: it trains like any
            // non-elite until it has re-served the k-clock window.
            info!(
                "step {} │ {}{}{} │ {} │ seat wait restarts at 0 of {} │ weights and optimizer kept, next step resumes │ crown {}/{}",
                clock,
                reason,
                rank_note,
                detail,
                crate::engine::reporting::trained_note(trainings, age),
                self.config.elite_join_probation_steps,
                self.frozen_crown.len(),
                capacity,
            );
        }
        // 5b. Why the elite RANKING and the crown disagree, said on the clocks that
        // changed the deal: the ★ badge shows the STATE, this line shows the REASON.
        if enabled && !(entering.is_empty() && leaving.is_empty()) {
            let crown = &self.frozen_crown;
            let ranked = self.ranked_live();
            let unseated_total = ranked.iter().filter(|(h, _)| !crown.contains(h)).count();
            let shown: Vec<String> = ranked
                .iter()
                .filter(|(h, _)| !crown.contains(h))
                .take(3)
                .map(|(h, _)| {
                    format!(
                        "{} ({})",
                        &h[..8.min(h.len())],
                        self.unseated_reason(h, clock)
                    )
                })
                .collect();
            if !shown.is_empty() {
                let more = unseated_total.saturating_sub(shown.len());
                let tail = if more > 0 {
                    format!(" … +{more} more")
                } else {
                    String::new()
                };
                info!(
                    "step {} │ freeze │ ranked without a seat: {}{tail}",
                    clock,
                    shown.join(" │ ")
                );
            }
        }

        // 6. TICK both counters for the clock about to run: probation ages every live
        // net, the seat age only whoever holds a seat.
        let crown = &self.frozen_crown;
        for (h, s) in self.state.live_mut() {
            s.elite_probation_age += 1;
            if crown.contains(h) {
                s.elite_seat_age += 1;
            }
        }
    }

    /// Why `hash` — a net in the elite RANKING — holds no seat right now: barred by
    /// the join window (`a/k`) or the cull shield, or `outranked` (the seats ran out).
    fn unseated_reason(&self, hash: &str, clock: usize) -> String {
        let k = self.config.elite_join_probation_steps;
        match self.state.net(hash) {
            Some(s) if s.elite_probation_age < k => {
                format!("join window {}/{k}", s.elite_probation_age)
            }
            Some(_) if self.on_probation(hash, clock) => "cull shield".to_string(),
            Some(_) => "outranked".to_string(),
            None => "no state".to_string(),
        }
    }

    /// The elite gate in numbers over the live population: `(eligible, on_probation,
    /// age_blocked)`. `clock` MUST be the deal's clock, never `step_clock()`.
    pub(crate) fn elite_gate_counts(&self, clock: usize) -> (usize, usize, usize) {
        let k = self.config.elite_join_probation_steps;
        let j = self.config.elite_crown_lifetime_steps;
        let (mut eligible, mut probation, mut age_blocked) = (0, 0, 0);
        for h in self.state.live_hashes() {
            let Some(s) = self.state.net(&h) else {
                continue;
            };
            if self.on_probation(&h, clock) || s.elite_probation_age < k {
                probation += 1;
            } else if j > 0 && s.elite_seat_age >= j {
                age_blocked += 1;
            } else {
                eligible += 1;
            }
        }
        (eligible, probation, age_blocked)
    }

        /// Whether this clock prints the `lottery │ who` line: the `Summ` reporting shape
        /// with the lottery actually ARMED (`lottery_frame` returned a field).
        ///
        /// Deliberately independent of `freeze_elites`: that knob gates the CROWN rollup
        /// (its `crown H/C seats` tally and the `stepped A/B` line), and switching the
        /// crown off must not hide who the training lottery let update — the two rules
        /// answer different questions about the same clock.
    pub(crate) fn lottery_line_on(&self, armed: bool) -> bool {
        armed && self.log_level == crate::engine::config::LogLevel::Summ
    }

        /// Split one clock's nets into `(trained, idled)` from the IDLE-ledger snapshot
        /// taken BEFORE the clock ran: a net whose `idle_clocks` grew got no ticket,
        /// everyone else in the group ran the trainer. Crown members land in NEITHER
        /// list — they acted for the elite freeze, which the rollup names separately.
        ///
        /// Reads the ledger, not a counter: `idle_spans` is what replay obeys, so the
        /// report and the resume plan cannot disagree about which clocks were idle.
        ///
        /// Returns two empty vectors when there was no snapshot (lottery off, or the
        /// line off), so the caller's `lottery_skips` is 0 and nothing is reported.
    pub(crate) fn lottery_clock_report(
        &self,
        group: &[String],
        idle_before: &[usize],
    ) -> (Vec<String>, Vec<String>) {
        let mut trained: Vec<String> = Vec::new();
        let mut withheld: Vec<String> = Vec::new();
        for (i, h) in group.iter().enumerate() {
            let Some(&before) = idle_before.get(i) else {
                continue;
            };
            let now = self.state.net(h).map(|s| s.idle_clocks()).unwrap_or(0);
            if now > before {
                withheld.push(h[..8.min(h.len())].to_string());
            } else if !self.frozen_crown.contains(h) {
                trained.push(h[..8.min(h.len())].to_string());
            }
        }
        (trained, withheld)
    }

    /// Every count the per-step rollup needs, from ONE walk of the live population
    /// plus the seats the crown knows: `newborn = live − group.len()` (a net born this
    /// clock sits the clock out entirely), `would_train = group − crown`,
    /// `stepped = group − lottery skips` (idle nets did nothing) and
    /// `trained = would_train − lottery skips`.
    pub(crate) fn step_vitals(
        &self,
        clock: usize,
        group: &[String],
        lottery_skips: usize,
    ) -> StepVitals {
        let live = self.state.live_count();
        let frozen = group
            .iter()
            .filter(|h| self.frozen_crown.contains(*h))
            .count();
        let would_train = group.len().saturating_sub(frozen);
        let (eligible, probation, age_blocked) = self.elite_gate_counts(clock);
        StepVitals {
            // An idle net did not step at all (no train, no eval): `stepped` counts the
            // updaters plus the crown's act-and-measure passes only.
            stepped: group.len().saturating_sub(lottery_skips),
            trained: would_train.saturating_sub(lottery_skips),
            live,
            frozen,
            would_train,
            lottery_skips,
            newborn: live.saturating_sub(group.len()),
            eligible,
            probation,
            age_blocked,
            // The crown IS the seat set, so the seats held are the frozen nets.
            seats_held: frozen,
            seats_capacity: self.config.elite_count.min(live.saturating_sub(1)),
            join_k: self.config.elite_join_probation_steps,
            lifetime_j: self.config.elite_crown_lifetime_steps,
            crown_armed: self.config.freeze_elites,
        }
    }

    /// The net's 1-based rank among ALL live nets by smoothed fitness (the same sort
    /// `elite_hashes` uses); `None` when unknown or unmeasured. Feeds the logs.
    pub(crate) fn rank_position(&self, hash: &str) -> Option<usize> {
        self.ranked_live()
            .iter()
            .position(|(h, _)| h == hash)
            .map(|p| p + 1)
    }

    // ── Selection helpers ───────────────────────────────────────────────────

    /// The live net with the best smoothed fitness (direction-aware). Test-only
    /// since the random-topology fallback replaced the clone-fittest path.
    #[cfg(test)]
    pub(crate) fn fittest_net_hash(
        &self,
    ) -> std::result::Result<String, crate::utils::error::EngineError> {
        match self.ranked_live_all().into_iter().next() {
            Some((h, _)) => Ok(h),
            None => Err(crate::utils::error::EngineError::InvalidOptions(
                "race: no live nets to select fittest from".into(),
            )),
        }
    }

    /// The worst live net hashes by smoothed fitness (opposite of fittest).
    /// Uses the live smoothed fitness; only cullable nets are candidates.
    pub(crate) fn worst_nets_by_smoothed_fitness(&self, count: usize) -> Result<Vec<String>> {
        // The elite guard applies HERE too: this selector feeds the crossover-Worst
        // victim, and an elite must never be a victim under either policy.
        let elite = self.elite_hashes();
        // `ranked_live()` is best-first, so the worst is its reverse.
        Ok(self
            .ranked_live()
            .into_iter()
            .rev()
            .filter(|(h, _)| !elite.contains(h))
            .map(|(h, _)| h)
            .take(count)
            .collect())
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

    /// Stamp the tombstone fields onto LIVE nets at stop: `final_fitness_mean`,
    /// `final_fitness_std`, `final_fitness_samples` from the net's race reading and
    /// `final_risk_score` from the FIT score it ranked by — the same three legs a
    /// cull records. Without this a net still alive at stop carries `None`, and
    /// [`crate::engine::population::run_topologies_from_run_dir`]'s seeding cannot
    /// rank it (it would fall to file order). Called from [`Self::finish_race`]
    /// BEFORE the frontier write, so one write carries the stamped fields;
    /// durability ticks during the run stay unstamped (the fields mean "final").
    pub(crate) fn stamp_live_final_scores(&mut self) {
        for h in self.state.live_hashes() {
            let reading = self.race_reading_of(&h);
            let smoothed = reading.map(|r| r.mean).unwrap_or(f32::NAN);
            let smoothed_std = reading.map(|r| r.std);
            let risk_score = self.smoothed_fitness_of(&h);
            let samples = reading.map(|r| r.n).unwrap_or(0);
            if let Some(state) = self.state.net_mut(&h) {
                state.final_fitness_mean = Some(smoothed);
                state.final_fitness_std = smoothed_std;
                state.final_risk_score = risk_score;
                state.final_fitness_samples = samples;
            }
        }
    }

        /// Append a step's metrics for the live nets that were trained and scored this
        /// clock to the history buffer (idle nets contribute no row: nothing happened to
        /// them). `history.csv` carries ONLY these rows — evolution events live in
        /// `attempts.csv`.
    pub(crate) fn append_metrics_csv(&mut self, step: usize) -> Result<()> {
        if !self.config.csv_export {
            return Ok(());
        }
        for hash in &self.state.live_hashes() {
            if let Some(net_state) = self.state.net(hash) {
                if let Some(m) = &net_state.last_metrics {
                        // Only a clock that was TRAINED AND SCORED owns a row. An idle
                        // lottery clock (or a newborn's birth clock) has no metrics for
                        // THIS step, and repeating the last measurement under a new step
                        // label would fabricate one — so the row is simply absent.
                    if m.step != step {
                        continue;
                    }
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
                        // FIT columns, appended LAST so every existing column keeps its
                        // position (positional readers stay valid): the risk-adjusted
                        // score this net ranked by and the window spread it paid for.
                        // `nan` = no reading yet, the same convention as `eval_loss`.
                    let reading = self.race_reading_of(hash);
                    let score = self.smoothed_fitness_of(hash);
                    row.push(',');
                    row.push_str(
                        &score
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "nan".to_string()),
                    );
                    row.push(',');
                    row.push_str(
                        &reading
                            .map(|r| r.std.to_string())
                            .unwrap_or_else(|| "nan".to_string()),
                    );
                    row.push('\n');
                    self.history_csv_buffer.push_str(&row);
                }
            }
        }
        Ok(())
    }

    /// Flush both buffered CSV exports — `history.csv` (metric rows) and
    /// `attempts.csv` (evolution events) — at checkpoints and stop, gated by `csv_export`.
    pub(crate) fn flush_csv_exports(&mut self) -> Result<()> {
        if !self.config.csv_export {
            return Ok(());
        }
        // `net_seed` completes the individual identity: (hash, net_seed) uniquely
        // distinguishes re-born individuals that share a topology hash across eras.
        let mut history_header =
            "step,hash,net_seed,origin,entered_at_step,train_loss,eval_loss,fitness".to_string();
        for m in &self.metrics {
            history_header.push(',');
            history_header.push_str(m.label());
        }
        // The window's other two legs (`fitness` above stays the RAW per-step value):
        // the risk-adjusted score the net ranked by, and its spread `σ`.
        history_header.push_str(",fitness_smooth,fitness_smooth_std");
        let attempts_header = "step,branch,attempt,outcome,child_hash,child_net_seed,child_origin,child_fitness,victim,victim_net_seed,pop_size";
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
    /// A no-op today (the header is written once and config is never mutated mid-run).
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
        let mut header = RunHeader::from_race_options(cfg);
        // Post-race facts are NOT launch config: re-deriving the header must never wipe
        // the recorded guardrail champion or the seed receipt.
        header.guardrail = self.header.guardrail.clone();
        header.run_topology_source = self.header.run_topology_source.clone();
        write_engine_json(&self.run_dir, &header)
    }
}

/// Attempts a crossover roll may make (fresh parents each) before it is discarded.
const CROSSOVER_ATTEMPTS: usize = 5;

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
        not_run.push(format!(
            "{} newborn (first step next clock)",
            v.newborn
        ));
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
mod tests {
    use super::*;
    // Direction only appears in test helpers, so it is imported here to keep
    // the lib-only `cargo check` warning-free.
    use crate::utils::fitness::Direction;
    use crate::graph::node::Node;
    use crate::graph::topology::Topology;
    use crate::graph::topology::TopologyOptions;
    use crate::utils::tabular_data::synthetic_classification;
    use flodl::{Device, Variable};

    use crate::engine::config::DEFAULT_CHECKPOINT_EVERY;

    /// The RL volume label — the RL middle column of the rollup: the episode
    /// count, `turns = train + eval`, turns/episode, per-turn cost. An unmeasured
    /// cell reads `—`, never a fake 0.
    #[test]
    fn rl_volume_label_formats_and_never_lies() {
        let empty = RlVolume::default();
        assert!(empty.label(1.0).contains('—'), "{}", empty.label(1.0));
        let pop = RlVolume {
            nets: 3,
            matches: 3,
            train_turns: 432,
            eval_turns: 108,
        };
        // 540 turns in 1.0s ⇒ 1000/540 = 1.85… ⇒ `1.85` ms/turn (the 1..10 band
        // prints 2 decimals). 540/3 = 180.0 turns per episode.
        assert_eq!(
            pop.label(1.0),
            "episodes 3 │ turns 540 = train 432 + eval 108 │ turns/episode 180.0 │ 1.85 ms/turn"
        );
        // Matches with zero turns is still a real report (every match died on turn 0) —
        // it reads as a mean of 0, with the cost cell `—` rather than an infinity.
        let zero_turns = RlVolume {
            nets: 1,
            matches: 1,
            train_turns: 0,
            eval_turns: 0,
        };
        assert_eq!(
            zero_turns.label(1.0),
            "episodes 1 │ turns 0 = train 0 + eval 0 │ turns/episode 0.0000 │ — ms/turn"
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

    // ── Test helpers ─────────────────────────────────────────────────────

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

    /// FIT: the ranking value is the risk-adjusted score, so a steady net outranks a
    /// volatile one with the SAME window mean — and `λ = 0` puts them back on the mean.
    /// The test fitness is Minimize (lower is better), i.e. the direction where the
    /// penalty is ADDED, which is the half a naive implementation gets wrong.
    #[test]
    fn risk_aversion_ranks_the_steadier_net_first() {
        let dir = std::env::temp_dir().join("gras-fit-ranking");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let mut hashes = eng.state.live_hashes();
        hashes.sort();
        assert_eq!(hashes.len(), 2);
        // Equal window means (0.50) so ONLY the spread can separate them: `hashes[0]`
        // never moves, `hashes[1]` swings ±0.30 (population σ = 0.30).
        for (h, values) in [
            (&hashes[0], [0.50_f32, 0.50, 0.50, 0.50]),
            (&hashes[1], [0.20_f32, 0.80, 0.20, 0.80]),
        ] {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            for v in values {
                buf.push(v);
            }
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        }

        // λ = 0 (the pre-FIT ranking): equal means ⇒ equal scores, so this pair is a
        // tie and the ORDER is the ranking's own tie-break (fewest params, then hash —
        // see `ranking_ties_go_to_the_smaller_net_then_the_hash`): pin the values here,
        // not the order, and the tie-break there.
        eng.config.risk_aversion = 0.0;
        let flat = eng.ranked_live();
        assert_eq!(flat.len(), 2);
        assert_eq!(flat[0].1, flat[1].1, "λ=0 must rank on the plain mean");
        assert!((flat[0].1 - 0.50).abs() < 1e-6);

        // λ = 1: Minimize pays `mean + σ`, so the steady net scores 0.50 and the
        // volatile one 0.80 — the steady net ranks first and is strictly better.
        eng.config.risk_aversion = 1.0;
        let ranked = eng.ranked_live();
        assert_eq!(ranked[0].0, hashes[0], "the steady net must rank first");
        assert!((ranked[0].1 - 0.50).abs() < 1e-6, "steady score was {}", ranked[0].1);
        assert!(
            (ranked[1].1 - 0.80).abs() < 1e-6,
            "volatile score was {}",
            ranked[1].1
        );
        // The readable window mean is untouched by λ — the score is derived from it.
        let reading = eng.race_reading_of(&hashes[1]).unwrap();
        assert_eq!((reading.mean, reading.std), (0.50, 0.30));

        // The ELITE take-k reads the same ranking: the steadier net holds the seat.
        eng.config.elite_count = 1;
        eng.config.elite_join_probation_steps = 0;
        eng.config.elite_crown_lifetime_steps = 0;
        eng.config.mutation_probation_steps = 0;
        eng.config.crossover_probation_steps = 0;
        assert_eq!(
            eng.elite_hashes(),
            vec![hashes[0].clone()],
            "the elite must be the steadier net"
        );
        // …and so does cull-victim selection: the volatile net is the one dropped.
        assert_eq!(
            eng.worst_nets_by_smoothed_fitness(1).unwrap(),
            vec![hashes[1].clone()],
            "the cull victim must be the volatile net"
        );
    }

        /// The OFF path in the direction that makes the penalty ADDITIVE: with λ = 0 a
        /// Minimize run must ignore spread completely (score == window mean, and the net
        /// with the better mean wins even when it is the volatile one) — finite, no NaN,
        /// no inversion. The `+ λσ` arm only bites for λ > 0.
    #[test]
    fn minimize_with_lambda_zero_ignores_spread_entirely() {
        let dir = std::env::temp_dir().join("gras-fit-minimize-off");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 13).unwrap();
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let mut hashes = eng.state.live_hashes();
        hashes.sort();
        // Minimize (the test fitness): lower is better. `hashes[0]` has the BETTER mean
        // (0.40) but is volatile; `hashes[1]` is perfectly steady at a worse 0.60.
        for (h, values) in [
            (&hashes[0], [0.20_f32, 0.60, 0.20, 0.60]),
            (&hashes[1], [0.60_f32, 0.60, 0.60, 0.60]),
        ] {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            for v in values {
                buf.push(v);
            }
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        }
        // λ = 0 — also what a clamped negative λ becomes at the builder (pinned by
        // `config::tests::risk_aversion_defaults_on_and_clamps_at_off`).
        eng.config.risk_aversion = 0.0;
        let ranked = eng.ranked_live();
        for (h, score) in &ranked {
            let reading = eng.race_reading_of(h).unwrap();
            assert!(score.is_finite(), "λ=0 must never produce a NaN score: {h}");
            assert_eq!(*score, reading.mean, "λ=0 must score exactly the mean");
        }
        assert_eq!(
            ranked[0].0, hashes[0],
            "with the risk term off the better MEAN wins — steadiness earns nothing"
        );
        assert!((ranked[0].1 - 0.40).abs() < 1e-6, "mean was {}", ranked[0].1);
        assert!((ranked[1].1 - 0.60).abs() < 1e-6, "mean was {}", ranked[1].1);
        // Switched on, the same direction charges the spread ON TOP of the mean: the
        // steady net's 0.60 beats the volatile net's 0.40 + 0.20σ = 0.60 in a Minimize
        // sort, and both readings are untouched.
        eng.config.risk_aversion = 1.0;
        let ranked = eng.ranked_live();
        assert!((ranked[1].1 - 0.60).abs() < 1e-6, "volatile score was {}", ranked[1].1);
        assert!((ranked[0].1 - 0.60).abs() < 1e-6, "steady score was {}", ranked[0].1);
        assert_eq!(
            eng.race_reading_of(&hashes[0]).unwrap().mean,
            0.40,
            "the readable mean is not perturbed by λ"
        );
        assert_eq!(ranked.len(), 2, "a score tie drops nobody");
    }

    /// The tombstone a cull writes records all three legs of the reading the net died
    /// with: the readable mean, its spread, and the score it was ranked by.
    #[test]
    fn cull_tombstone_records_mean_spread_and_risk_score() {
        let dir = std::env::temp_dir().join("gras-fit-tombstone");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 11).unwrap();
        eng.config.risk_aversion = 2.0;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes = eng.state.live_hashes();
        let victim = hashes[1].clone();
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        for v in [0.20_f32, 0.80, 0.20, 0.80] {
            buf.push(v);
        }
        *eng.rolling_fitness.get_mut(&victim).unwrap() = buf;
        eng.cull_net(&victim, 3, "crossover").unwrap();
        let state = crate::state::load_net_state(&dir, &victim).unwrap();
        assert_eq!(state.final_fitness_mean, Some(0.50), "the mean is readable");
        assert_eq!(state.final_fitness_std, Some(0.30), "the spread is kept");
        // Minimize at λ=2: 0.50 + 2(0.30) = 1.10 — the value that actually ranked it.
        let score = state.final_risk_score.unwrap();
        assert!((score - 1.10).abs() < 1e-6, "risk score was {score}");
        assert_eq!(state.final_fitness_samples, 4);
        assert_eq!(state.cull_reason.as_deref(), Some("crossover"));
    }

    #[test]
    fn champion_hashes_match_ranked_elites() {
        // Regression guard for the guardrail bug: post-race tooling must read THE
        // champions from the engine — the top-elite_count of rank_live, not file order.
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
        // Evolution semantics default to the cartpole reference (only the
        // SIZE axis stays small/conservative).
        assert_eq!(cfg.mutate_prob, 0.5);
        assert_eq!(
            cfg.mutation_cull_policy,
            crate::engine::config::MutationCullPolicy::Worst
        );
        assert_eq!(cfg.mutation_probation_steps, 5);
        assert_eq!(cfg.crossover_probation_steps, 5);
        assert_eq!(cfg.elite_join_probation_steps, 5);
        assert_eq!(cfg.elite_crown_lifetime_steps, 5);
        assert_eq!(
            cfg.crossover_cull_policy,
            crate::engine::config::CrossCullPolicy::Worst
        );
        assert_eq!(cfg.checkpoint_every, DEFAULT_CHECKPOINT_EVERY);
        // batch_size lives on RunSpec::stream (engine infrastructure), not
        // RaceConfig — verified by the stream_shape contract.
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
        // Channel probation 0: this test pins the DRAW's determinism, not the shield
        // (see `probation_shields_*`).
        engine.config.mutation_probation_steps = 0;
        engine.config.crossover_probation_steps = 0;
        engine
            .seed_population_internal(
                vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
                Some(0.5),
            )
            .unwrap();
        let live = engine.state.live_hashes();
        // Deterministic: same clock ⇒ same victim.
        let v1 = engine
            .select_random_victim(3)
            .unwrap()
            .expect("a cull-eligible victim");
        let v2 = engine
            .select_random_victim(3)
            .unwrap()
            .expect("a cull-eligible victim");
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
        // Give the nets distinct fitness verdicts: seed buffers with fake smoothed
        // scores (map hash → score) so ranking is well-defined without hash order.
        let hashes = engine.state.live_hashes();
        let mut scores: Vec<(String, f32)> = Vec::new();
        for (i, h) in hashes.iter().enumerate() {
            let s = 0.1 + i as f32 * 0.3;
            engine.rolling_fitness.get_mut(h).unwrap().push(s);
            scores.push((h.clone(), s));
        }
        // NOTE: the harness fitness is Minimize (mse), so the fittest net has the
        // LOWEST score. Join probation k=5 would bar all three at clock 0 — k=0 here.
        engine.config.elite_join_probation_steps = 0;
        // The born-based CHANNEL windows gate the seat too, so they must be 0 as well
        // for an elite to exist at clock 0.
        engine.config.mutation_probation_steps = 0;
        engine.config.crossover_probation_steps = 0;
        // Refresh the crown at clock 0: only refresh_crown writes `frozen_crown`, and
        // `elite_hashes()` is a pure query — frozen-path asserts read the settled set.
        engine.refresh_crown(0);
        // NOTE: the test harness fitness is Minimize (mse) — the fittest net
        // has the LOWEST score, so the elite is the minimum, not the maximum.
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let expected_elite = scores[0].0.clone();
        let elite = engine.elite_hashes();
        assert_eq!(elite.len(), 1, "guard of 1 protects exactly one net");
        assert_eq!(
            elite[0], expected_elite,
            "the fittest net (lowest seeded score) is the elite"
        );
        for clock in 0..20 {
            let v = engine
                .select_random_victim(clock)
                .unwrap()
                .expect("the two non-elite, off-probation nets are cullable");
            assert_ne!(v, elite[0], "elite must never be a crossover victim");
        }
        // Mutation roulette also skips the elite: worst-net weight is largest,
        // elite is excluded entirely.
        if let Some(victim) = engine.select_inverse_proportional_at(0, 0).unwrap() {
            assert_ne!(victim, elite[0], "elite must never be a mutation victim");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mutation_targets_worst_most_often() {
        // Regression test for the inverted-weights bug: with 3 nets of very different
        // fitness the inverse roulette must pick the WORST most often, never the best.
        let dir = std::env::temp_dir().join("race_mutation_targets_worst");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        // Isolate the ROULETTE WEIGHTS: the default probation windows would
        // otherwise shield every fresh net and the draw would return None.
        engine.config.mutation_probation_steps = 0;
        engine.config.elite_join_probation_steps = 0;
        engine.config.elite_crown_lifetime_steps = 5;
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
        // 200 distinct `(clock, roll)` coordinates: the roulette draw is DERIVED from
        // them, so this distribution is reproducible, not entropy-seeded.
        for i in 0..200usize {
            if let Some(v) = engine.select_inverse_proportional_at(i / 8, i % 8).unwrap() {
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

    /// The clock reads 0 before anything ran AND stays 0 after a populate that has not
    /// stepped: it comes from the live nets' RECORDED clocks, not from a counter a
    /// population change could move (the joiner test below is the other half).
    #[test]
    fn step_clock_is_zero_until_a_net_actually_steps() {
        let dir = std::env::temp_dir().join("race_clock_empty_test");
        let mut engine = engine(&dir, 42).unwrap();
        assert_eq!(engine.step_clock(), 0, "nothing populated yet");
        engine
            .seed_population_internal(vec![tiny_topology(7)], None)
            .unwrap();
        assert_eq!(engine.step_clock(), 0, "populated, nobody has stepped");
    }

    /// The clock must come from the RECORDED clock, not the training count: a net
    /// inserted mid-run has one fewer training than the clock it reached.
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
        // A (act-and-measure freeze): the frozen elite plays its normal step — fitness
        // is fresh and the buffer grows — through a no-op optimizer. Weight unchanged.
        let dir = std::env::temp_dir().join("gras_freeze_elite_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 11).unwrap();
        // Join window 0 so net B can be crowned on arrival: the subject here is the
        // frozen-branch contract, not the probation window.
        open_crown(&mut eng);
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        // Fixture is Minimize: lower smoothed = better. Mirror the end-of-group-step
        // model (train, record_step, push fitness) so refresh_crown has a verdict.
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.5);
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.1);
        assert_eq!(
            eng.elite_hashes(),
            vec![hs[1].clone()],
            "net B is the best net"
        );
        // Let net A carry a verdict too, so the population is symmetric and ranking
        // stays honest.
        let m0 = crate::state::NetMetrics {
            step: 0,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.5,
            informative: vec![],
            frozen: false,
        };
        eng.state.record_step(&hs[0], m0).unwrap();
        // KISS: refresh_crown DEALT the crown at group-step start; `step_one_net` only
        // obeys it. Join probation is 0 here, so the best net (B) takes the seat.
        eng.refresh_crown(0);
        let before_len = eng.rolling_fitness.get(&hs[1]).unwrap().len();
        eng.step_one_net(&hs[1], 0).unwrap();
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
            last_metrics.step, 0,
            "frozen net's recorded last_metrics.step is the clock it acted at (act-and-measure)"
        );
        assert_ne!(
            last_metrics.fitness, 0.5,
            "the fitness is a fresh measurement, not the carried skill"
        );
        // Freeze-member check reads the canonical `frozen_crown` (what refresh_crown
        // writes), not `elite_hashes()` — the frozen step just pushed a fresh value.
        assert!(
            eng.frozen_crown.contains(&hs[1]),
            "net B is in the frozen crown after the freeze step"
        );
        assert_eq!(
            eng.state.net(&hs[1]).unwrap().step,
            0,
            "frozen net's step count is unchanged (act-and-measure is not a training step)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn dethroned_elite_resumes_stepping_without_replay() {
        // Dethrone under act-and-measure freeze: the former elite never fell behind the
        // clock, so losing the crown is simply RESUMING normal training (no gauntlet).
        let dir = std::env::temp_dir().join("gras_dethrone_resume_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 13).unwrap();
        // All windows 0: the subject is dethrone/resume, not the join window —
        // the crown must be dealable on the first refresh.
        open_crown(&mut eng);
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
        eng.refresh_crown(5);
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
        // Net A overtakes: the crown moves to A, B is dethroned. (A's 0.05 is a deep
        // win — B resumes TRAINING and its fresh fitness stays well above it.)
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
        eng.refresh_crown(7);
        assert!(eng.frozen_crown.contains(&hs[0]), "crown moved to A");
        // B's next step is a NORMAL train step at the current clock — one
        // clock past its last acted clock (no replay span).
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
            "dethroned net stepped once at the current clock (no replay)"
        );
        // NOTE: B may legitimately re-crown if its trained fitness beats A's 0.05 —
        // correct dethrone semantics. Assert consistency with B's standing, not a set.
        assert_eq!(
            eng.frozen_crown.contains(&hs[1]),
            eng.elite_hashes().contains(&hs[1]),
            "crown membership mirrors the elite set after B's resumed step"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn freeze_crown_logs_on_transition_only() {
        // A: the "crowned" event fires once at first crowning and once on migration,
        // never on stable steps — asserted via frozen_crown state transitions.
        let dir = std::env::temp_dir().join("gras_freeze_crown_transition");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 23).unwrap();
        // All windows 0 so B can be crowned on the FIRST frozen step; the transition-
        // logging contract under test is independent of the windows.
        open_crown(&mut eng);
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
        // First frozen step: the crown set is dealt, so the "elite seat" line fires. We
        // call `refresh_crown` directly (it writes `frozen_crown`) and never record_step.
        eng.refresh_crown(5);
        assert!(eng.frozen_crown.contains(&hs[1]));
        // Second refresh, same champion: NO transition — crown unchanged and
        // no new line would print (membership diff is empty).
        eng.refresh_crown(6);
        assert!(eng.frozen_crown.contains(&hs[1]));
        // Crown migration: net A trains past the frozen champion, becomes
        // elite — the diff moves the crown ("elite seat" + "dethroned" pair).
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
        // Refresh at clock 7 (one past A's recorded clock 6) so refresh_crown
        // sees both nets' latest verdicts and can re-deal the crown.
        eng.refresh_crown(7);
        assert!(
            eng.frozen_crown.contains(&hs[0]),
            "crown moved to the new champion"
        );
        assert!(
            !eng.frozen_crown.contains(&hs[1]),
            "old champion left the crown in the same deal"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn newborn_immigrant_starts_fresh_without_replay() {
        // A mutation immigrant joins at step 0 with no recorded metrics: there is no
        // replay, and its first optimizer update is the next group step.
        let dir = std::env::temp_dir().join("gras_mutation_newborn_fresh");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 31).unwrap();
        eng.config.mutate_rolls = 1;
        // Cull shield 0: this test pins the immigrant's fresh-insert bookkeeping,
        // and with the default window nobody would be cull-eligible yet.
        eng.config.mutation_probation_steps = 0;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.9);
        // A fresh immigrant is inserted; its step count is 0 and it has no
        // recorded metrics (no replay happened).
        eng.evolve_random_immigrant(5, 0).unwrap();
        assert_eq!(eng.state.live_hashes().len(), 2, "cull+insert keeps size");
        let newcomer = eng
            .state
            .live_hashes()
            .into_iter()
            .find(|h| !hs.contains(h))
            .expect("a new immigrant hash exists");
        let s = eng.state.net(&newcomer).unwrap();
        assert_eq!(s.step, 0, "newborn: no replayed training count");
        assert!(s.last_metrics.is_none(), "newborn: no replayed metrics");
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

    /// Both ends of the ranking keep the SAME number of nets: `elite_count` drives the
    /// worst writers too, the order is the probe's own rule (spent × gap, hash as the
    /// tie-break), a champion is never also written as a worst, and a topology-only run
    /// still lands its files.
    #[test]
    fn worst_artifacts_mirror_the_elite_count() {
        let dir = std::env::temp_dir().join("race_worst_artifacts_count");
        let _ = std::fs::remove_dir_all(&dir);
        let mut e = engine(&dir, 11).unwrap();
        let topos: Vec<_> = (7..=12).map(tiny_topology).collect();
        e.seed_population_internal(topos, Some(0.5)).unwrap();
        let mut hs = e.state.live_hashes();
        hs.sort();
        let n = hs.len();
        assert!(n >= 3, "fixture needs a few nets, got {n}");
        // Index the field: `hs[i]` scores `1.0 + i` (Minimize fixture) and trained `i`
        // clocks — so BOTH legs of `spent × gap` rise with the index and the worst-first
        // order is exactly the reverse of the sorted hash order, with no ties.
        for (h, s) in e.state.live_mut() {
            s.step = hs.iter().position(|x| x == h).unwrap();
        }
        for (i, h) in hs.iter().enumerate() {
            e.rolling_fitness.get_mut(h).unwrap().push(1.0 + i as f32);
        }

        // k = 2: the two worst by the probe ranking, worst first — and NOT the top-2
        // champions, which the pick excludes.
        e.config.elite_count = 2;
        let picks = e.worst_ranked();
        assert_eq!(picks.len(), 2, "the bottom end keeps elite_count nets");
        assert_eq!(picks[0].hash, hs[n - 1], "worst first");
        assert_eq!(picks[1].hash, hs[n - 2]);
        assert!(
            picks[0].spent * picks[0].gap > picks[1].spent * picks[1].gap,
            "the ranking is spent × gap descending"
        );
        e.record_champions();
        for pick in &picks {
            assert!(
                !e.champions.contains(&pick.hash),
                "a net saved as an elite must not also be saved as a worst"
            );
        }

        // k = 1 keeps the same WORST net (the count changes how many, never which is
        // worst), and `elite_count` is the only knob: the flags decide whether, not how
        // many.
        e.config.elite_count = 1;
        let one = e.worst_ranked();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].hash, picks[0].hash, "same worst, fewer files");

        // Topology-only saving still writes one `worst-<hash>.md` per pick (and used to
        // print nothing at all — the card lived inside the safetensors branch).
        e.config.elite_count = 2;
        e.config.worst_save_topology = true;
        e.write_worst_artifacts().unwrap();
        let mut written: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with("worst-"))
            .collect();
        written.sort();
        let mut expected: Vec<String> = picks
            .iter()
            .map(|p| format!("worst-{}.md", &p.hash[..8]))
            .collect();
        expected.sort();
        assert_eq!(written, expected, "one file per pick, named by hash");
        let _ = std::fs::remove_dir_all(&dir);
    }

        /// λ must actually MOVE a decision, not just be carried around: the same two nets
        /// rank opposite ways at λ = 0 (mean decides) and λ = 2 (spread is charged). This
        /// pins that the risk term reaches the cull, not only the printed score.
    #[test]
    fn lambda_moves_the_cull_victim_from_the_low_mean_to_the_risky_net() {
        let dir = std::env::temp_dir().join("race_lambda_cull_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut e = engine(&dir, 5).unwrap();
        e.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = e.state.live_hashes();
        e.fitness = Fitness::new(|_, _| Ok(0.0), Direction::Maximize, "fixture");
        // Same population, two readings each: A steady at 10, B swinging 6↔16.
        for (h, vals) in [(&hs[0], [10.0_f32, 10.0]), (&hs[1], [6.0, 16.0])] {
            let buf = e.rolling_fitness.get_mut(h).unwrap();
            for v in vals {
                buf.push(v);
            }
        }
        // λ = 0 ranks on the mean: 10 < 11, so the steady net is the worst.
        e.config.risk_aversion = 0.0;
        assert_eq!(
            e.worst_nets_by_smoothed_fitness(1).unwrap()[0],
            hs[0],
            "λ=0: the plain mean decides, so the 10 is worst"
        );
        // λ = 2 charges 2σ: 11 − 2·5 = 1 < 10, so the RISKY net is now the worst.
        e.config.risk_aversion = 2.0;
        assert_eq!(
            e.worst_nets_by_smoothed_fitness(1).unwrap()[0],
            hs[1],
            "λ=2: the spread charge flips the victim to the risky net"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stop stamps the tombstone fields onto LIVE nets too: the frontier alone
    /// ranks a finished run's seeding (seeding previously fell to file order for
    /// any net still alive at stop).
    #[test]
    fn stop_stamps_final_scores_on_live_nets() {
        let dir = std::env::temp_dir().join("race_final_stamp_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine.config.crossover_prob = 0.0;
        engine.config.mutate_prob = 0.0;
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.0))
            .unwrap();
        for clock in 0..3 {
            let hashes = engine.state.live_hashes();
            for h in &hashes {
                engine.step_one_net(h, clock).unwrap();
            }
        }
        let hashes = engine.state.live_hashes();
        engine.stamp_live_final_scores();
        for h in &hashes {
            let state = engine.state.net(h).unwrap();
            assert!(state.final_fitness_mean.is_some(), "mean stamped: {h}");
            assert!(state.final_risk_score.is_some(), "FIT score stamped: {h}");
            assert!(state.final_fitness_samples > 0, "samples stamped: {h}");
        }
        // What a stop writes: the stamped fields end up in nets/<hash>.json.
        engine.write_live_frontier_states().unwrap();
        let raw = std::fs::read_to_string(
            dir.join("nets").join(format!("{}.json", hashes[0])),
        )
        .unwrap();
        let loaded: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(
            loaded["final_risk_score"].is_number(),
            "stamped score must be a number, not null/absent"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn replay_range_replays_steps_zero_to_clock() {
        let dir = std::env::temp_dir().join("race_catchup_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine.config.crossover_prob = 0.0;
        engine.config.mutate_prob = 0.0;
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.0))
            .unwrap();
        let clock = 5;
        // generate_child can now return Ok(None) (spent roll) — this test
        // exercises replay mechanics, so build the child directly.
        let mut child = engine.random_child(clock, 0).unwrap();
        assert!(
            !child
                .state
                .created_from
                .as_deref()
                .unwrap()
                .contains("crossover"),
            "the replay test should use the random branch so the child nets fit the tiny harness"
        );
        let topo = child.state.topology().unwrap();
        assert_eq!(topo.options.input_dim, Some(2));
        assert_eq!(topo.options.output_dim, Some(2));
        // Replay runs steps 0..clock-1 and leaves the child at the clock.
        engine.replay_range(&mut child, 0, clock).unwrap();
        assert_eq!(child.state.step, clock);
        let m = child.state.last_metrics.as_ref().unwrap();
        assert_eq!(m.step, clock - 1);
    }

    #[test]
    fn replay_range_is_deterministic() {
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
        engine_a.replay_range(&mut child_a, 0, clock).unwrap();
        engine_b.replay_range(&mut child_b, 0, clock).unwrap();
        // Same seed ⇒ same replayed metrics.
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
    impl crate::trainer::RlStep for TestRlTrainer {
        fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
            use flodl::nn::Module;
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }

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
                rl: None,
            })
        }
    }

    /// RL trainer whose report depends on `(net_seed, step)` — a stand-in for "a
    /// deterministic env": replay matches only if the engine feeds the same clock.
    struct SeededRlTrainer;
    impl crate::trainer::RlStep for SeededRlTrainer {
        fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
            use flodl::nn::Module;
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }

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

    /// RL trainer that rolls the challenge trigger itself (the trainer-side recipe;
    /// fires → sentinel fitness −1000, so tests see exactly which steps challenged).
    struct ChallengingRlTrainer;
    impl crate::trainer::RlStep for ChallengingRlTrainer {
        fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
            use flodl::nn::Module;
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }

        fn train_step(
            &mut self,
            _net: &mut Network,
            _optimizer: &mut dyn Optimizer,
            step: usize,
            ctx: &crate::trainer::RlContext<'_>,
        ) -> flodl::tensor::Result<crate::trainer::RlStepReport> {
            let challenged = crate::trainer::challenge::challenge_fires(
                ctx.env.run_seed,
                ctx.net_seed,
                step,
                1.0, // the trainer's own knob: every step fires
                ctx.env.max_steps,
                crate::trainer::challenge::ChallengeDecay::Linear,
            );
            let fitness = if challenged {
                -1000.0
            } else {
                step as f32 + 1.0
            };
            Ok(crate::trainer::RlStepReport {
                train_loss: step as f32 * 0.5,
                fitness,
                informative: Vec::new(),
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
            crate::utils::fitness::Fitness::reported(
                crate::utils::fitness::Direction::Maximize,
                "reward",
            ),
            SeededRlTrainer,
            Some(seed),
            Some(run_dir.to_path_buf()),
        ))
    }

    /// The RL resume contract, mirroring the tabular twin: an interrupted run
    /// resumed for N more steps lands exactly where an uninterrupted one does.
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
            crate::utils::fitness::Fitness::reported(
                crate::utils::fitness::Direction::Maximize,
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

    /// The trainer-owned challenge trigger: knob 1.0 fires every step, so its
    /// sentinel lands on every step — the engine never sees the knob.
    #[test]
    fn trainer_owned_challenge_trigger_fires_inside_train_step() {
        let dir = std::env::temp_dir().join("gras-challenge-signal");
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = rl_config(0);
        let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
            cfg,
            crate::utils::fitness::Fitness::reported(
                crate::utils::fitness::Direction::Maximize,
                "reward",
            ),
            ChallengingRlTrainer, // rolls its own trigger → sentinel every step
            Some(1),
            Some(dir.clone()),
        ))
        .unwrap();
        eng.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let hash = eng.state.live_hashes()[0].clone();
        eng.step_one_net(&hash, 0).unwrap();
        let lm = eng.state.net(&hash).unwrap().last_metrics.clone().unwrap();
        assert_eq!(
            lm.fitness, -1000.0,
            "the trainer-side trigger fired → sentinel step"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A challenge is just another step: its fitness ranks like any other — recorded
    /// in metrics AND fed to the ranking buffers.
    #[test]
    fn challenged_fitness_ranks_like_any_other_step() {
        let dir = std::env::temp_dir().join("gras-challenge-honesty");
        let _ = std::fs::remove_dir_all(&dir);
        let mut cfg = rl_config(0);
        cfg.freeze_elites = false; // isolate the challenge from the freeze
        let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
            cfg,
            crate::utils::fitness::Fitness::reported(
                crate::utils::fitness::Direction::Maximize,
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

    /// Replay parity under the TRAINER-OWNED trigger: a resumed run re-derives the
    /// identical trigger from (run_seed, net_seed, step) and lands on the twin.
    #[test]
    fn challenge_resume_then_continue_matches_uninterrupted_twin() {
        let dir_a = std::env::temp_dir().join("gras-rl-challenge-a");
        let dir_b = std::env::temp_dir().join("gras-rl-challenge-b");
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);

        let challenge_cfg = |pop: usize| {
            let mut cfg = rl_config(pop);
            cfg.freeze_elites = false;
            cfg
        };
        let run = |dir: &std::path::Path,
                   steps: usize|
         -> Vec<(String, Option<crate::state::NetMetrics>)> {
            let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
                challenge_cfg(0),
                crate::utils::fitness::Fitness::reported(
                    crate::utils::fitness::Direction::Maximize,
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
            crate::utils::fitness::Fitness::reported(
                crate::utils::fitness::Direction::Maximize,
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
            crate::utils::fitness::Fitness::reported(
                crate::utils::fitness::Direction::Maximize,
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
        // The spec variant DECIDES the mode — an RL spec runs as RL with no
        // `set_run_mode(..)` call (deleted: disagreement is unrepresentable).
        let dir = std::env::temp_dir().join("gras-rl-mode-derived");
        let _ = std::fs::remove_dir_all(&dir);
        let config = RaceConfig {
            pop_size: 2,
            max_steps: Some(1),
            ..RaceConfig::defaults()
        };
        let spec = crate::engine::run_spec::RunSpec::rl(
            config,
            crate::utils::fitness::Fitness::reported(
                crate::utils::fitness::Direction::Maximize,
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
        // Determinism ⇒ the SAME outcome, including `None` (all pairing attempts
        // incompatible). The self-pairing redraw shares the rng, so it must too.
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
        // The mutation roll belongs to the CROSSOVER branch only: the exploit path may
        // get a perturbation; the immigrant path never carries the '+mut' suffix.
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
        // A topology with NO hidden nodes: input → output directly. Crossover needs
        // hidden nodes to match pivots, so every pairing fails — the clone-fittest case.
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
        // Hidden-less parents make every crossover attempt a no-op; after 3 attempts the
        // roll is SPENT (no random fallback) via a clean error the caller skips.
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
        // The two roulette draws are independent, so without the redraw guard one net
        // could be both parents. Sweep (seed, clock) pairs and assert no `parents=X,X`.
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

    /// REGRESSION: a crossover child INHERITS its parents' feature split. `cx_*`
    /// re-finalizes both parents, which drops port-count-bound masks — so the engine
    /// snapshots them before the mix and reinstalls the blend afterwards. With the
    /// operator pinned to one-point, every child's rows must be A's prefix + B's
    /// suffix; a fresh policy draw would satisfy that only by coincidence.
    #[test]
    fn crossover_children_inherit_the_parents_feature_split() {
        let dir = std::env::temp_dir().join("gras-mask-inherit");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 11).unwrap();
        eng.config.feature_split = Some(crate::graph::topology::FeatureSplitPolicy::Random);
        eng.config.crossover_prob = 1.0;
        eng.config.crossover_ops_pool = vec!["one_point".to_string()];
        // Founders carry a split, installed after their finalize exactly as a real run
        // does; distinct rows per founder make the mix observable.
        let founders: Vec<crate::graph::topology::Topology> = [7usize, 8, 9]
            .into_iter()
            .map(|seed| {
                let mut topo = tiny_topology(seed);
                topo.install_input_feature_masks(&[
                    vec![1, 0],
                    vec![(seed % 2) as u8, ((seed + 1) % 2) as u8],
                ]);
                topo
            })
            .collect();
        eng.seed_population_internal(founders, Some(0.5)).unwrap();

        let live = eng.state.live_hashes();
        let masks_of = |hash: &str| {
            eng.state
                .net(hash)
                .expect("parent is live")
                .topology()
                .expect("topology parses")
                .input_feature_masks()
                .expect("a founder of a masked run carries a split")
        };
        let parents: Vec<Vec<Vec<u8>>> = live.iter().map(|h| masks_of(h)).collect();

        let mut checked = 0usize;
        for clock in 0..16 {
            let Some(child) = eng.generate_child(clock, 0).unwrap() else {
                continue;
            };
            let lineage = child.state.created_from.clone().unwrap_or_default();
            let Some(pair) = lineage.strip_prefix("crossover-one-point:parents=") else {
                continue;
            };
            let pair: Vec<&str> = pair.split(',').collect();
            let (Some(ia), Some(ib)) = (
                live.iter().position(|h| h == pair[0]),
                live.iter().position(|h| h == pair[1]),
            ) else {
                panic!("clock {clock}: lineage names an unknown parent ({lineage})");
            };
            let (a, b) = (&parents[ia], &parents[ib]);
            let child_rows = child
                .state
                .topology()
                .expect("topology parses")
                .input_feature_masks()
                .expect("the child keeps the inherited split");
            assert_eq!(child_rows.len(), a.len(), "one row per input port");
            assert!(
                (1..child_rows.len()).any(|p| child_rows[..p] == a[..p] && child_rows[p..] == b[p..]),
                "clock {clock}: {child_rows:?} is not A's prefix + B's suffix of {a:?} / {b:?}"
            );
            checked += 1;
        }
        assert!(checked > 0, "no crossover child was generated to check");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_restores_run_counters() {
        // Stop → resume continues the ORIGINAL cull budget, wall-clock age and child
        // ordinals; the stream geometry rides the header so it holds across resume.
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
    fn engine_mode_discriminator_roundtrip() {
        // Engine split: new headers carry `engine_mode` at the root; legacy shared-shape
        // headers (field absent) are migrated on load by deriving from `config.mode`.
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
    fn run_culls_continue_across_a_resume_boundary() {
        // The loop-level shape the step-level resume tests cannot see: culls born
        // INSIDE run()'s evolve phase, on a race that stops mid-run and resumes,
        // must leave the frontier exactly where an uninterrupted twin lands.
        let cfg = |max_steps: Option<usize>, stop_after_clock_zero: bool| {
            let mut c = RaceConfig::defaults();
            c.pop_size = 2;
            c.mutate_rolls = 1;
            c.mutate_prob = 1.0; // every step's mutation roll fires → a cull per clock
            // defaults() shields founders for 5 clocks; the twin must be able to cull.
            c.mutation_probation_steps = 0;
            c.crossover_probation_steps = 0;
            c.checkpoint_every = 1; // frontier survives the stop without a manual flush
            c.max_steps = max_steps;
            if stop_after_clock_zero {
                c.custom_stop = Some(Box::new(|_| true)); // fires at the FIRST check
            }
            c
        };

        // Uninterrupted twin: 4 clocks in one sitting (budget = clocks RUN).
        let dir_a = std::env::temp_dir().join("race_run_boundary_twin");
        let _ = std::fs::remove_dir_all(&dir_a);
        let mut twin = engine(&dir_a, 123).unwrap();
        twin.config = cfg(Some(4), false);
        twin.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let reason_twin = twin.run().unwrap();
        assert_eq!(reason_twin, StopReason::MaxSteps);
        assert!(
            twin.culls >= 2,
            "the twin must have actually evolved (got {} culls)",
            twin.culls
        );
        let twin_final: Vec<(String, Option<crate::state::NetMetrics>)> = twin
            .state
            .live_hashes()
            .iter()
            .map(|h| (h.clone(), twin.state.net(h).unwrap().last_metrics.clone()))
            .collect();
        let twin_culls = twin.culls;
        drop(twin);

        // Interrupted twin: run() stops via custom_stop after clock 0, and the
        // STOP writes the frontier (finish_race → write_live_frontier_states).
        let dir_b = std::env::temp_dir().join("race_run_boundary_split");
        let _ = std::fs::remove_dir_all(&dir_b);
        let mut part = engine(&dir_b, 123).unwrap();
        part.config = cfg(Some(4), true);
        part.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let reason_part = part.run().unwrap();
        assert_eq!(reason_part, StopReason::CustomStop);
        drop(part);

        // Resume through run() with the SAME config (the budget stays 4; the
        // recorded clock is 0, so the pre-flight stop does not fire) — the
        // remainder runs the remaining 3 clocks and their evolution.
        let mut resumed = crate::engine::TabularEngine::resume(
            dir_b.clone(),
            tiny_dataset_dir("run_boundary_resume"),
            cfg(Some(4), false),
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        assert_eq!(resumed.run().unwrap(), StopReason::MaxSteps);
        assert_eq!(
            resumed.culls, twin_culls,
            "the cull budget continues, not resets, and evolution replays identically"
        );
        let split_final: Vec<(String, Option<crate::state::NetMetrics>)> = resumed
            .state
            .live_hashes()
            .iter()
            .map(|h| (h.clone(), resumed.state.net(h).unwrap().last_metrics.clone()))
            .collect();
        assert_eq!(
            split_final, twin_final,
            "interrupt+resume == uninterrupted twin (bit-identical frontier)"
        );
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    #[test]
    fn immigrant_evolution_keeps_pop_constant() {
        let dir = std::env::temp_dir().join("race_immigrant_pop");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal((0..4).map(tiny_topology).collect(), Some(0.5))
            .unwrap();
        // Three sequential immigrant rounds at one clock each cull and insert: if the
        // child ordinal restarted per round, two children would collide → pop shrinks.
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

    /// The durability tick writes the live frontier states; a resume rebuilds the
    /// SAME frontier from them (no ledger involved — the ledger was removed as a
    /// diagnostic that nothing read).
    #[test]
    fn resume_rebuilds_the_frontier_from_durability_states() {
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
        let hashes = engine.state.live_hashes();
        for clock in 0..3 {
            for h in &hashes {
                engine.step_one_net(h, clock).unwrap();
            }
            if clock > 0 && clock % engine.config.checkpoint_every == 0 {
                engine.write_live_frontier_states().unwrap();
            }
        }
        let live_before = engine.state.live_count();
        drop(engine);

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
        assert_eq!(
            resumed.state.live_count(),
            live_before,
            "the tick's frontier states must rebuild the same live population"
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

        /// λ is replay-relevant like K: a run that recorded one λ cannot be resumed at
        /// another, and a run from BEFORE FIT (no λ in `engine.json` ⇒ 0.0, i.e. it ranked
        /// on the plain mean) is refused BY NAME until the caller passes 0.0 back.
    #[test]
    fn resume_refuses_a_risk_aversion_mismatch_and_names_the_recorded_value() {
        let dir = std::env::temp_dir().join("race_resume_risk_aversion");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        for h in engine.state.live_hashes() {
            let state = engine.state.net(&h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        drop(engine);

        // Strip the field: exactly what an `engine.json` written before FIT looks like.
        let raw = std::fs::read_to_string(dir.join("engine.json")).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        json["config"]
            .as_object_mut()
            .unwrap()
            .remove("risk_aversion");
        std::fs::write(dir.join("engine.json"), json.to_string()).unwrap();

        let resume = |config: RaceConfig| {
            crate::engine::TabularEngine::resume(
                dir.clone(),
                tiny_dataset_dir("risk_aversion"),
                config,
                fitness(),
                crate::trainer::TabularTrainer::new(loss_fn()),
            )
        };
        // The NEW default (λ = 1.0) against a run that recorded none is refused, and the
        // message names the value that reproduces the recorded ranking.
        let mut current = RaceConfig::defaults();
        current.pop_size = 2;
        let err = resume(current)
            .err()
            .expect("a λ mismatch must be refused, not patched")
            .to_string();
        assert!(err.contains("risk_aversion mismatch"), "{err}");
        assert!(
            err.contains("set_run_risk_aversion(0.00)"),
            "the refusal must name the recorded λ: {err}"
        );
        // …and passing the recorded λ back resumes the run.
        let mut legacy = RaceConfig::defaults();
        legacy.pop_size = 2;
        legacy.risk_aversion = 0.0;
        assert!(
            resume(legacy).is_ok(),
            "a pre-FIT run must resume at the λ it actually ranked with"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

        /// The parent draw is replay-relevant like K and λ: roulette and tournament
        /// prefer different parents AND consume the roll's RNG differently, so a resume
        /// at another method is a different search. A pre-knob header recorded nothing
        /// (`""`), which reads as roulette — the draw those runs made — not as a refusal.
    #[test]
    fn resume_refuses_a_crossover_selection_mismatch_and_reads_legacy_as_roulette() {
        let dir = std::env::temp_dir().join("race_resume_crossover_selection");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        for h in engine.state.live_hashes() {
            let state = engine.state.net(&h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        drop(engine);

        let resume = |config: RaceConfig| {
            crate::engine::TabularEngine::resume(
                dir.clone(),
                tiny_dataset_dir("crossover_selection"),
                config,
                fitness(),
                crate::trainer::TabularTrainer::new(loss_fn()),
            )
        };
        // A different method is refused, and the message names BOTH draws.
        let mut tournament = RaceConfig::defaults();
        tournament.pop_size = 2;
        tournament.crossover_selection = SelectionMethod::Tournament { tournament_size: 3 };
        let err = resume(tournament)
            .err()
            .expect("a selection mismatch must be refused, not patched")
            .to_string();
        assert!(err.contains("crossover_selection mismatch"), "{err}");
        assert!(err.contains("roulette"), "names the recorded draw: {err}");
        assert!(
            err.contains("tournament(size=3)"),
            "names the requested draw: {err}"
        );

        // A header written before the knob: strip the key, resume at the default.
        let raw = std::fs::read_to_string(dir.join("engine.json")).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        json["config"]
            .as_object_mut()
            .unwrap()
            .remove("crossover_selection");
        std::fs::write(dir.join("engine.json"), json.to_string()).unwrap();
        let mut legacy = RaceConfig::defaults();
        legacy.pop_size = 2;
        assert!(
            resume(legacy).is_ok(),
            "a pre-knob run must resume at roulette, the draw it actually made"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Training lottery (experimental) ─────────────────────────

        /// The fraction is bounded and abstains on bad data: a flat field, a NaN merit or
        /// an infinite bound must yield `1.0` (train), never a silent starvation.
    #[test]
    fn merit_fraction_is_bounded_and_abstains_on_bad_data() {
        assert_eq!(CoreEngine::merit_fraction(2.0, 3.0, 1.0), 0.5);
        assert_eq!(CoreEngine::merit_fraction(3.0, 3.0, 1.0), 1.0);
        assert_eq!(CoreEngine::merit_fraction(1.0, 3.0, 1.0), 0.0);
        // Out-of-field values clamp rather than overshoot.
        assert_eq!(CoreEngine::merit_fraction(9.0, 3.0, 1.0), 1.0);
        assert_eq!(CoreEngine::merit_fraction(-9.0, 3.0, 1.0), 0.0);
        // No spread / non-finite: the lottery abstains (everyone trains).
        assert_eq!(CoreEngine::merit_fraction(2.0, 2.0, 2.0), 1.0);
        assert_eq!(CoreEngine::merit_fraction(f32::NAN, 3.0, 1.0), 1.0);
        assert_eq!(CoreEngine::merit_fraction(2.0, f32::INFINITY, 1.0), 1.0);
    }

        /// The lottery is OFF by default, stays DISARMED until the starting population
        /// has served its born shield, then grades each net by level-per-TICKET
        /// relative to the clock's field. The coin itself is deterministic per (net,
        /// clock), and a net inside its shield always trains no matter how it grades.
    #[test]
    fn train_lottery_arms_after_the_born_shield_and_grades_level_per_ticket() {
        let dir = std::env::temp_dir().join("gras-train-lottery");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 21).unwrap();
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let hashes = eng.state.live_hashes();

        // OFF (the default): no field, everybody trains.
        assert!(eng.lottery_frame(50).is_none());
        eng.config.train_merit = 1.0;

        // Means 100 / 200 / 300 over the SAME budget (each net has spent one ticket):
        // the shared divisor cannot change the ORDER, only the scale — merits are
        // 100/1 … 300/1. A reading with no ticket would be ungradable, so the fixture
        // stamps the clock the reading came from.
        for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
            let h = &hashes[i];
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(mean);
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
            eng.state.net_mut(h).unwrap().step = 1;
        }

        // ON: still disarmed while the founders are inside their born shield (same data,
        // different clock — the arming is what changes)…
        eng.config.mutation_probation_steps = 5;
        eng.config.crossover_probation_steps = 5;
        assert!(
            eng.lottery_frame(4).is_none(),
            "founders are shielded until clock 5"
        );
        // …and armed once every founder is out of it.
        let frame = eng.lottery_frame(5).expect("armed at clock 5");
        // The fixture fitness is Minimize: the LOWEST level-per-ticket is the best.
        let p = |eng: &CoreEngine, h: &str| eng.train_lottery_p(h, 5, Some(frame));
        assert!((p(&eng, &hashes[0]) - 1.0).abs() < 1e-6);
        assert!((p(&eng, &hashes[1]) - 0.5).abs() < 1e-6);
        assert!((p(&eng, &hashes[2]) - 0.0).abs() < 1e-6);

        // p == 1 always trains, p == 0 never does, and the same (net, clock) always
        // lands the same way — a resume replays the identical decisions.
        assert!(eng.train_lottery_draws(&hashes[0], 5, Some(frame)));
        assert!(!eng.train_lottery_draws(&hashes[2], 5, Some(frame)));
        assert_eq!(
            eng.train_lottery_draws(&hashes[1], 5, Some(frame)),
            eng.train_lottery_draws(&hashes[1], 5, Some(frame))
        );

        // The routing contract: a denied ticket IDLES the net — no trainer call, so no
        // metric row, no update and no reading (p = 0 above is this clock's coin).
        eng.step_one_net_in_frame(&hashes[2], 5, Some(frame))
            .expect("idle step");
        let worst = eng.state.net(&hashes[2]).unwrap();
        assert_eq!(worst.step, 1, "idle does not advance the training clock");
        assert!(worst.last_metrics.is_none(), "idle records no metric row");
        assert_eq!(worst.idle_clocks(), 1, "the clock lands in the idle ledger");
        assert!(worst.idle_spans.contains(&(5, 5)), "as this clock's span");

        // A newborn keeps its own shield: it trains regardless of how it grades.
        eng.state.net_mut(&hashes[2]).unwrap().entered_at_step = 5;
        assert_eq!(eng.train_lottery_p(&hashes[2], 5, Some(frame)), 1.0);

        let _ = std::fs::remove_dir_all(&dir);
    }

        /// Ω is a dial on the WITHHOLDING, not a scale on the merit: `p_eff = 1 − Ω(1 − p)`.
        /// The endpoints are exact — `0` is fully off (no field is even built, so the
        /// engine never consults the lottery) and `1` is the full merit tilt — and the
        /// middle sits linearly between them, which is what makes `0.5` mean "half the
        /// idling" instead of behaving like `1`. Scaling the merits could not do that:
        /// `p` is a min-max FRACTION of the field, so a common factor cancels.
    #[test]
    fn train_merit_omega_blends_the_withholding_linearly() {
        let dir = std::env::temp_dir().join("gras-train-merit-omega");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 21).unwrap();
        eng.config.mutation_probation_steps = 0;
        eng.config.crossover_probation_steps = 0;
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let hashes = eng.state.live_hashes();
        for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
            let h = &hashes[i];
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(mean);
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
            eng.state.net_mut(h).unwrap().step = 1;
        }

        // OFF: no field at all, so every net trains and no draw is consumed.
        eng.config.train_merit = 0.0;
        assert!(eng.lottery_frame(5).is_none(), "Ω = 0 must not build a field");
        for h in &hashes {
            assert_eq!(eng.train_lottery_p_eff(h, 5, None), 1.0);
            assert!(eng.train_lottery_draws(h, 5, None));
        }

        // Fixture fitness is Minimize with merits 100/200/300 per ticket: raw fractions
        // are 1, 0.5 and 0 — the best, the middle and the worst.
        eng.config.train_merit = 1.0;
        let frame = eng.lottery_frame(5).expect("armed at clock 5");
        let raw: Vec<f32> = hashes
            .iter()
            .map(|h| eng.train_lottery_p(h, 5, Some(frame)))
            .collect();
        assert!((raw[0] - 1.0).abs() < 1e-6);
        assert!((raw[1] - 0.5).abs() < 1e-6);
        assert!((raw[2] - 0.0).abs() < 1e-6);
        let eff = |eng: &CoreEngine, i: usize| eng.train_lottery_p_eff(&hashes[i], 5, Some(frame));

        // Ω = 1 is exactly the raw fraction — today's behavior, unchanged.
        for i in 0..3 {
            assert!((eff(&eng, i) - raw[i]).abs() < 1e-6);
        }
        // Half strength: the middle net withholds half as often (0.75, not 0.5), and the
        // worst keeps a standing 0.5 chance instead of the permanent p = 0.
        eng.config.train_merit = 0.5;
        assert!((eff(&eng, 0) - 1.0).abs() < 1e-6, "the best always trains");
        assert!((eff(&eng, 1) - 0.75).abs() < 1e-6);
        assert!((eff(&eng, 2) - 0.5).abs() < 1e-6);
        // A quarter strength is a quarter of the withholding.
        eng.config.train_merit = 0.25;
        assert!((eff(&eng, 1) - 0.875).abs() < 1e-6);
        assert!((eff(&eng, 2) - 0.75).abs() < 1e-6);
        // The dial does not make a coin flip at p_eff = 1: the best net consumes none.
        eng.config.train_merit = 0.5;
        assert!(eng.train_lottery_draws(&hashes[0], 5, Some(frame)));
        // Out-of-range strengths are clamped, so a bad flag cannot invert the dial.
        eng.config.train_merit = 4.0;
        assert!((eff(&eng, 2) - 0.0).abs() < 1e-6, "Ω clamps to 1");
        eng.config.train_merit = -1.0;
        assert!((eff(&eng, 2) - 1.0).abs() < 1e-6, "Ω clamps to 0");

        let _ = std::fs::remove_dir_all(&dir);
    }

        /// The denominator is TICKETS, not clocks lived: two nets with the SAME score
        /// divide by the clocks that actually ran the trainer and scored them, so the
        /// net that spent 10 tickets grades better than the one that spent 1. This is
        /// also what keeps an idle clock free — it is not a ticket, so the ratio can
        /// never decay while the numerator stands still.
    #[test]
    fn lottery_merit_divides_by_tickets_not_clocks_lived() {
        let dir = std::env::temp_dir().join("gras-lottery-tickets");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 21).unwrap();
        eng.config.train_merit = 1.0;
        // No shields: this test is about the divisor, not the born gate.
        eng.config.mutation_probation_steps = 0;
        eng.config.crossover_probation_steps = 0;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes = eng.state.live_hashes();
        // Identical score, different age AND different budget spent.
        for h in &hashes {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(100.0);
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        }
        eng.state.net_mut(&hashes[0]).unwrap().step = 1;
        eng.state.net_mut(&hashes[1]).unwrap().entered_at_step = 3;
        eng.state.net_mut(&hashes[1]).unwrap().step = 10;

        // Same reading, different budget: 100/1 vs 100/10. Age plays no part.
        assert_eq!(eng.merit_per_ticket(&hashes[0]), Some(100.0));
        assert_eq!(eng.merit_per_ticket(&hashes[1]), Some(10.0));

        let frame = eng.lottery_frame(5).expect("armed at clock 5");
        // Minimize: the LOWEST merit is best, so the net that spent 10 tickets takes
        // p = 1.0 and the one that spent 1 takes p = 0.0 — the clock-lived divisor
        // inverted this, which is why it went.
        assert!((eng.train_lottery_p(&hashes[0], 5, Some(frame)) - 0.0).abs() < 1e-6);
        assert!((eng.train_lottery_p(&hashes[1], 5, Some(frame)) - 1.0).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }

        /// An idle clock moves NOTHING the lottery grades: with the ticket divisor the
        /// reading and the denominator both stand still, so `smt/tkt` is bit-identical
        /// and only the idle ledger grows. (The clock-lived divisor this replaced made
        /// the ratio decay for idling — a penalty for a decision the net did not make.)
    #[test]
    fn an_idle_clock_leaves_the_merit_bit_identical() {
        let dir = std::env::temp_dir().join("gras-lottery-idle-invariant");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 21).unwrap();
        eng.config.train_merit = 1.0;
        eng.config.mutation_probation_steps = 0;
        eng.config.crossover_probation_steps = 0;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes = eng.state.live_hashes();
        for h in &hashes {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(100.0);
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
            eng.state.net_mut(h).unwrap().step = 5;
        }
        let before = eng.merit_per_ticket(&hashes[1]).expect("graded");
        assert_eq!(before, 20.0, "5 tickets spent, 100 scored");

        // One clock later: hashes[0] trained, hashes[1] idled. Readings unchanged.
        eng.state.net_mut(&hashes[0]).unwrap().step += 1;
        eng.idle_step(&hashes[1], 6).unwrap();
        assert_eq!(
            eng.merit_per_ticket(&hashes[1]),
            Some(before),
            "an idle clock changes neither the score nor the ticket count"
        );
        let idled = eng.state.net(&hashes[1]).unwrap();
        assert_eq!(idled.step, 5, "no weight update");
        assert!(idled.last_metrics.is_none(), "no metric row");
        assert_eq!(idled.idle_clocks(), 1);
        assert_eq!(idled.idle_spans, vec![(6, 6)]);
        // The trained peer's denominator grew, so the field CAN separate them — that
        // is the budget rotating, not the idle net climbing.
        let peer = eng.merit_per_ticket(&hashes[0]).expect("graded");
        assert!((peer - 100.0 / 6.0).abs() < 1e-5, "6 tickets spent: {peer}");

        let _ = std::fs::remove_dir_all(&dir);
    }

        /// A run can end on idle clocks (the lottery withheld the last tickets). Resume
        /// must skip them: they have no update and no draw, so replaying one would train
        /// a net the run never trained — a different `state.step` and different weights,
        /// with only the recorded row standing between it and silent drift.
    #[test]
    fn resume_does_not_replay_idle_clocks() {
        let dir = std::env::temp_dir().join("race_resume_idle");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h = engine.state.live_hashes()[0].clone();
        for clock in 0..3 {
            engine.step_one_net(&h, clock).unwrap();
        }
        // Two trailing idle clocks, exactly what two denied tickets record.
        engine.idle_step(&h, 3).unwrap();
        engine.idle_step(&h, 4).unwrap();
        let trained = engine.state.net(&h).unwrap().step;
        let last = engine.state.net(&h).unwrap().last_metrics.clone();
        assert_eq!(trained, 3, "three tickets, two idles");
        assert_eq!(last.as_ref().map(|m| m.step), Some(2));
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
        drop(engine);

        // Metric parity is asserted INSIDE resume: replaying an idle clock would move
        // the recorded row (and `state.step` to 5 instead of 3) and fail there.
        let mut resumed = crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("resume_idle"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 1;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        let restored = resumed.state.net(&h).unwrap();
        assert_eq!(restored.step, 3, "idle clocks are not replayed as trainings");
        assert_eq!(restored.last_metrics, last, "the last ticket's row survived");
        assert_eq!(restored.idle_clocks(), 2, "the ledger round-trips");
        // The next clock is a normal training step again.
        resumed.step_one_net(&h, 5).unwrap();
        assert_eq!(resumed.state.net(&h).unwrap().step, 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

        /// `history.csv` carries one row per clock a net was TRAINED AND SCORED on. An
        /// idle net has no metrics for the new step, and writing its carried snapshot
        /// under the loop's step would fabricate a measurement.
    #[test]
    fn history_csv_writes_only_scored_clocks() {
        let dir = std::env::temp_dir().join("gras-history-idle");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 21).unwrap();
        eng.config.csv_export = true;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes = eng.state.live_hashes();
        // Clock 4: both nets were trained and scored.
        for h in &hashes {
            let m = crate::state::NetMetrics {
                step: 4,
                train_loss: 1.0,
                eval_loss: Some(2.0),
                fitness: 3.0,
                informative: vec![],
                frozen: false,
            };
            eng.state.record_step(h, m).unwrap();
        }
        eng.append_metrics_csv(4).unwrap();
        assert_eq!(eng.history_csv_buffer.lines().count(), 2, "one row per scored net");

        // Clock 5: hashes[0] idles (no metrics), hashes[1] is scored.
        eng.idle_step(&hashes[0], 5).unwrap();
        let m = crate::state::NetMetrics {
            step: 5,
            train_loss: 1.0,
            eval_loss: Some(2.0),
            fitness: 3.0,
            informative: vec![],
            frozen: false,
        };
        eng.state.record_step(&hashes[1], m).unwrap();
        eng.append_metrics_csv(5).unwrap();
        let rows: Vec<&str> = eng.history_csv_buffer.lines().collect();
        assert_eq!(rows.len(), 3, "exactly one new row");
        let new = rows.last().unwrap();
        assert!(
            new.starts_with(&format!("5,{}", hashes[1])),
            "the scored net owns the row: {new}"
        );
        assert!(
            !eng.history_csv_buffer
                .contains(&format!("5,{}", hashes[0])),
            "an idle net must not repeat its last measurement under the new step"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

        /// The lottery line's gate is `Summ` + an ARMED lottery, nothing else: switching
        /// the elite crown off (`freeze_elites = false`, which suppresses the step
        /// rollup) must not silence the train-budget report, and `Minimal`/`None` stay
        /// quiet.
    #[test]
    fn lottery_line_prints_at_summ_without_the_crown() {
        let dir = std::env::temp_dir().join("gras-lottery-line-gate");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 21).unwrap();
        eng.log_level = crate::engine::config::LogLevel::Summ;
        // The crown OFF — the who-line's gate must not ride it. (The ROLLUP's gate no
        // longer rides it either: the rollup prints `crown off …` on the same line, so
        // this flag only silences the seat legs, never a report.)
        eng.config.freeze_elites = false;
        assert!(
            eng.lottery_line_on(true),
            "a summ run without the crown still reports the lottery's budget"
        );
        // Not armed yet (the field is `None` until the founders clear their shield).
        assert!(!eng.lottery_line_on(false));
        // The other reporting shapes carry the lottery only in nets/*.json.
        eng.log_level = crate::engine::config::LogLevel::Minimal;
        assert!(!eng.lottery_line_on(true));
        eng.log_level = crate::engine::config::LogLevel::None;
        assert!(!eng.lottery_line_on(true));
        let _ = std::fs::remove_dir_all(&dir);
    }

        /// The clock's split comes from the IDLE ledger alone: a net whose idle clocks
        /// grew appears under `withheld` (idled), the rest of the group under `trained`,
        /// and a crown member under NEITHER (it acted for the elite freeze, a different
        /// cause the rollup names by itself). No snapshot ⇒ no report, which is how the
        /// lottery's OFF path stays silent without a second flag.
    #[test]
    fn lottery_clock_report_splits_by_idle_delta_and_excludes_the_crown() {
        let dir = std::env::temp_dir().join("gras-lottery-clock-report");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 21).unwrap();
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let hashes = eng.state.live_hashes();
        let short = |h: &str| h[..8.min(h.len())].to_string();
        // No snapshot (lottery off / line off): nothing is claimed either way.
        let (t, w) = eng.lottery_clock_report(&hashes, &[]);
        assert!(t.is_empty() && w.is_empty());
        // hashes[1] had already idled twice; this clock it idles again and hashes[2]
        // idles for the first time. hashes[0] trains.
        let before = [0usize, 2, 1];
        eng.state.net_mut(&hashes[1]).unwrap().record_idle_step(0);
        eng.state.net_mut(&hashes[1]).unwrap().record_idle_step(1);
        eng.state.net_mut(&hashes[2]).unwrap().record_idle_step(0);
        eng.state.net_mut(&hashes[1]).unwrap().record_idle_step(6);
        eng.state.net_mut(&hashes[2]).unwrap().record_idle_step(6);
        let (t, w) = eng.lottery_clock_report(&hashes, &before);
        assert_eq!(t, vec![short(&hashes[0])], "the non-skipping net trained");
        assert_eq!(w, vec![short(&hashes[1]), short(&hashes[2])]);
        // A crown seat is neither trained nor withheld — the elite freeze owns it.
        eng.frozen_crown.insert(hashes[0].clone());
        let (t, w) = eng.lottery_clock_report(&hashes, &before);
        assert!(t.is_empty(), "a frozen crown member never counts as training");
        assert_eq!(w.len(), 2, "and it is not withheld by the lottery either");
        let _ = std::fs::remove_dir_all(&dir);
    }

        /// The coin's seed source is `(run_seed, net hash, clock)`: the ENGINE's run seed
        /// mixed with the individual's identity — never the net's own `net_seed`, which
        /// seeds the weight init and the trainer's per-step stream instead. Consequences
        /// pinned here: the same triple replays bit-identically, sample ORDER cannot move
        /// a decision (no shared stream is consumed), rewriting a net's `net_seed` cannot
        /// flip its coin, and the run seed is the lever that can.
    #[test]
    fn lottery_coin_is_seeded_by_run_seed_and_net_hash_not_net_seed() {
        let dir = std::env::temp_dir().join("gras-lottery-coin-seed");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 21).unwrap();
        eng.config.train_merit = 1.0;
        eng.config.mutation_probation_steps = 5;
        eng.config.crossover_probation_steps = 5;
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let hashes = eng.state.live_hashes();
        // Merits 10 / 20 / 30 per ticket ⇒ the middle net grades at p = 0.5, the only
        // interesting point of the interval (0 and 1 short-circuit without a draw).
        for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
            let h = &hashes[i];
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(mean);
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
            eng.state.net_mut(h).unwrap().step = 10;
        }
        let frame = eng.lottery_frame(5).expect("armed at clock 5");
        let mid = hashes[1].clone();
        let sample = |eng: &CoreEngine, h: &str| -> Vec<bool> {
            (5..69).map(|c| eng.train_lottery_draws(h, c, Some(frame))).collect()
        };
        let first = sample(&eng, &mid);
        assert!(
            first.iter().any(|&d| d) && first.iter().any(|&d| !d),
            "p = 0.5 over 64 clocks must decide both ways"
        );
        // Sampling the others in between consumes nothing: the results are unchanged.
        let _ = sample(&eng, &hashes[0]);
        let _ = sample(&eng, &hashes[2]);
        assert_eq!(
            sample(&eng, &mid),
            first,
            "the same (net, clock) must always land the same way"
        );
        // The net's OWN seed is not read by the coin (it seeds weights + trainer noise).
        let net_seed = eng.state.net(&mid).unwrap().net_seed;
        eng.state.net_mut(&mid).unwrap().net_seed = net_seed ^ 0xDEAD_BEEF;
        assert_eq!(
            sample(&eng, &mid),
            first,
            "the coin must not read net_seed — only the hash identifies the individual"
        );
        // The ENGINE's run seed is the lever.
        eng.header.run_seed ^= 1;
        assert_ne!(
            sample(&eng, &mid),
            first,
            "a different run seed must move the coin"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

        /// The training lottery's strength Ω decides which clocks were idle, and those
        /// clocks are the replay ledger: a run recorded at one Ω cannot be resumed at
        /// another (including on/off), and the refusal names the recorded value. Ω is a
        /// DIAL, not a switch, so 0.5 vs 1.0 must be refused too — they idle a
        /// different set of nets.
    #[test]
    fn resume_refuses_a_train_merit_mismatch() {
        let dir = std::env::temp_dir().join("race_resume_train_merit");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine.config.train_merit = 0.5;
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        for h in engine.state.live_hashes() {
            let state = engine.state.net(&h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        // Record the strength in the header (the engine wrote the default at construction,
        // before the knob was set for this test).
        let raw = std::fs::read_to_string(dir.join("engine.json")).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        json["config"]["train_merit"] = serde_json::json!(0.5);
        std::fs::write(dir.join("engine.json"), json.to_string()).unwrap();
        drop(engine);

        let resume = |config: RaceConfig| {
            crate::engine::TabularEngine::resume(
                dir.clone(),
                tiny_dataset_dir("train_merit"),
                config,
                fitness(),
                crate::trainer::TabularTrainer::new(loss_fn()),
            )
        };
        let mut off = RaceConfig::defaults();
        off.pop_size = 2;
        let err = resume(off)
            .err()
            .expect("a lottery mismatch must be refused, not patched")
            .to_string();
        assert!(err.contains("train_merit (\u{3a9}) mismatch"), "{err}");
        assert!(
            err.contains("set_run_train_merit(0.50)"),
            "the refusal must name the recorded strength: {err}"
        );
        // On/off is not the only mismatch a dial can have: the neighbouring strength
        // idles a DIFFERENT set of nets, so it is refused on the same terms.
        let mut full = RaceConfig::defaults();
        full.pop_size = 2;
        full.train_merit = 1.0;
        assert!(resume(full).is_err(), "0.5 vs 1.0 must be refused");
        let mut recorded = RaceConfig::defaults();
        recorded.pop_size = 2;
        recorded.train_merit = 0.5;
        assert!(
            resume(recorded).is_ok(),
            "the recorded setting must resume"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

        /// A header written before Ω recorded the knob as the bool
        /// `evolution_train_lottery`: `true` ran at the full merit weighting, so it must
        /// normalize to Ω = 1.0 (and stay resumable at that strength), while `false` —
        /// and a header with no key at all — normalize to Ω = 0.0, the plain path.
    #[test]
    fn a_legacy_bool_lottery_header_normalizes_to_omega() {
        let snap = |json: &str| {
            let header: crate::state::RunHeader = crate::state::RunHeader::from_json(json)
                .expect("legacy header loads");
            header.config.train_merit
        };
        assert_eq!(
            snap(r#"{"config":{"evolution_train_lottery":true}}"#),
            1.0,
            "a run recorded as ON trained at the full merit weighting"
        );
        assert_eq!(
            snap(r#"{"config":{"evolution_train_lottery":false}}"#),
            0.0
        );
        assert_eq!(snap(r#"{"config":{}}"#), 0.0, "pre-knob = always train");
        // One truth per header: the legacy key is consumed, not carried forward.
        let header = crate::state::RunHeader::from_json(
            r#"{"config":{"evolution_train_lottery":true}}"#,
        )
        .unwrap();
        assert_eq!(header.config.legacy_train_lottery, None);
        let written = header.to_json().unwrap();
        assert!(!written.contains("evolution_train_lottery"), "{written}");
        assert!(written.contains("\"train_merit\""), "{written}");
    }

        /// A resume at a different cull policy would evict different nets from the very
        /// next roll: refused by name, on BOTH channels, while the recorded policy still
        /// resumes. The header's crossover policy is blanked first, so the legacy `""` →
        /// `worst` path is exercised too.
    #[test]
    fn resume_refuses_a_cull_policy_mismatch_on_both_channels() {
        let dir = std::env::temp_dir().join("race_resume_cull_policy");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        for h in engine.state.live_hashes() {
            let state = engine.state.net(&h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        drop(engine);
        // Blank the recorded crossover policy: a pre-field header.
        let raw = std::fs::read_to_string(dir.join("engine.json")).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        json["config"]["crossover_cull_policy"] = serde_json::json!("");
        std::fs::write(dir.join("engine.json"), json.to_string()).unwrap();

        let resume = |config: RaceConfig| {
            crate::engine::TabularEngine::resume(
                dir.clone(),
                tiny_dataset_dir("cull_policy"),
                config,
                fitness(),
                crate::trainer::TabularTrainer::new(loss_fn()),
            )
        };
        // Legacy `""` reads `worst`, which is what the defaults resume at: allowed.
        let mut same = RaceConfig::defaults();
        same.pop_size = 2;
        assert!(resume(same).is_ok(), "a blanked legacy policy must resume");

        let mut cross = RaceConfig::defaults();
        cross.pop_size = 2;
        cross.crossover_cull_policy = crate::engine::config::CrossCullPolicy::Random;
        let err = resume(cross)
            .err()
            .expect("a crossover cull mismatch must be refused")
            .to_string();
        assert!(err.contains("crossover_cull_policy mismatch"), "{err}");
        assert!(err.contains("set_crossover_cull_policy"), "{err}");

        let mut mutp = RaceConfig::defaults();
        mutp.pop_size = 2;
        mutp.mutation_cull_policy = crate::engine::config::MutationCullPolicy::InverseFitness;
        let err = resume(mutp)
            .err()
            .expect("a mutation cull mismatch must be refused")
            .to_string();
        assert!(err.contains("mutation_cull_policy mismatch"), "{err}");
        assert!(err.contains("set_mutation_cull_policy"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Trainer-blob validation on resume ────────────────────────────────────────

    #[test]
    fn resume_rejects_changed_trainer_const_naming_the_key() {
        // The drift detector: a run trained at LR 0.001 and resumed at 0.5 must fail at
        // RESUME time naming "learning_rate" and both values, not as a parity assert.
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
        let trainer = crate::trainer::AnyTrainer::tabular(crate::trainer::TabularTrainer::new(
            loss_fn(),
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
    fn single_stop_criterion_still_works() {
        let run_dir = std::env::temp_dir().join("gras-race-single");
        let mut eng = engine(&run_dir, 5).unwrap();
        eng.config.max_steps = Some(7);
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        seed_raw_fitness(&mut eng, 0.5);
        assert_eq!(eng.check_stop(5), None);
        assert_eq!(eng.check_stop(6), Some(StopReason::MaxSteps));
    }

    // ── Elite crown fixtures ──────────────────────────────────────────────

    /// Crown fixtures: every probation window (channel + elite join) zeroed, since
    /// the gate logic itself is pinned by `elite_join_probation_crown_lifetime`.
    fn open_crown(eng: &mut crate::engine::TabularEngine) {
        eng.config.freeze_elites = true;
        eng.config.elite_count = 1;
        eng.config.mutation_probation_steps = 0;
        eng.config.crossover_probation_steps = 0;
        eng.config.elite_join_probation_steps = 0;
    }

    /// Both CSV exports must be self-consistent: every data row has exactly as
    /// many fields as its own header (the old shared layout shifted columns).
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
        let mut eng = engine(&run_dir, 42).unwrap();
        // The ledger is fed by EVOLUTION events, so force both roll channels to
        // fire on every step: probation windows zeroed (a shield would swallow
        // the early rolls) and freeze off (it hides the non-elite nets a cull
        // needs). `max_steps` buys 3 clocks, so the rolls have measured nets to
        // choose victims from.
        eng.config.max_steps = Some(3);
        eng.config.crossover_rolls = 1;
        eng.config.crossover_prob = 1.0;
        eng.config.mutate_rolls = 1;
        eng.config.mutate_prob = 1.0;
        eng.config.mutation_probation_steps = 0;
        eng.config.crossover_probation_steps = 0;
        eng.config.elite_join_probation_steps = 0;
        eng.config.freeze_elites = false;
        // A configured metric widens the history header — the shape most
        // likely to drift from the attempt layout.
        eng.metrics = vec![crate::utils::fitness::Metric::custom("dummy", |_p, _y| {
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
        // The attempt ledger really landed in its own file — populated by the
        // evolution rolls above, not by a stop-time writer.
        let attempts = std::fs::read_to_string(run_dir.join("attempts.csv")).unwrap();
        assert!(
            attempts.lines().any(|l| l.contains("inserted")),
            "the attempt ledger records a real evolution insert"
        );
        let _ = std::fs::remove_dir_all(&run_dir);
    }

    /// Smoke test for the explicit `{train, test}` layout: eval/gating pools hold
    /// ONLY test rows, and the header records `explicit_test_rows` for resume.
    #[test]
    fn explicit_train_test_dirs_build_disjoint_pools() {
        // A data dir with train/ (80 rows) + test/ (20 rows): same distribution,
        // disjoint row sets — the DIRECTORY layout is the point, not the values.
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
        let (train_pool, eval_pool) = stream.pools();
        let test_start = split_at as i64;
        for row in eval_pool {
            assert!(
                *row >= test_start,
                "eval pool row {row} leaks into the train range"
            );
        }
        for row in train_pool {
            assert!(*row < test_start, "train pool row {row} enters test range");
        }
        // Every test row is evaluated
        assert_eq!(
            eval_pool.len(),
            100 - split_at,
            "all held-out rows are in the eval pool"
        );
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
        assert_eq!(train_pool, rs.0);

        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&run_dir);
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

    // ── Seed from a previous run's guardrail champion ───────────────────

    /// Write a prior-run fixture (`engine.json` + champion `nets/<hash>.json`,
    /// optionally with a `guardrail` block); no `trainer` blob, so that check skips.
    fn write_prior_run(dir: &std::path::Path, run_id: &str, with_guardrail: bool) -> Topology {
        std::fs::create_dir_all(dir.join("nets")).unwrap();
        let topo = tiny_topology(999);
        let hash = "feedfacefeedface";
        std::fs::write(
            dir.join("nets").join(format!("{hash}.json")),
            serde_json::json!({ "topology": topo.to_json().unwrap() }).to_string(),
        )
        .unwrap();
        let mut header = serde_json::json!({
            "run_id": run_id,
            "engine_mode": "tabular",
            "run_seed": 7,
            "fitness_label": "mse",
            "fitness_direction": "Minimize",
            "input_dim": 2,
            "output_dim": 2,
            "config": { "mode": "tabular" },
        });
        if with_guardrail {
            header["guardrail"] = serde_json::json!({
                "champion_hash": hash,
                "champion_params": 4242,
                "race_smoothed": 0.25,
                "holdout_mean": 0.30,
                "matches_scored": 16,
            });
        }
        std::fs::write(dir.join("engine.json"), header.to_string()).unwrap();
        topo
    }

    fn seeded_engine(
        run_dir: &std::path::Path,
        seed_root: &std::path::Path,
    ) -> Result<crate::engine::TabularEngine> {
        let data_dir = tiny_dataset_dir("seeded");
        let config = RaceConfig {
            pop_size: 3,
            elite_previous_load: true,
            elite_previous_dir: Some(seed_root.to_path_buf()),
            ..RaceConfig::defaults()
        };
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            data_dir,
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(7),
            Some(run_dir.to_path_buf()),
        ))
    }

    /// The happy path: slot 0 carries the champion's blueprint, the population is
    /// full, lineage says where it came from, and the receipt is recorded.
    #[test]
    fn elite_previous_load_imports_the_certified_champion() {
        let prior = std::env::temp_dir().join("gras-seedprior-good");
        let _ = std::fs::remove_dir_all(&prior);
        std::fs::create_dir_all(&prior).unwrap();
        let champion = write_prior_run(&prior.join("1000"), "1000", true);

        let dir = std::env::temp_dir().join("gras-seedprior-run");
        let _ = std::fs::remove_dir_all(&dir);
        let eng = seeded_engine(&dir, &prior).unwrap();

        // 1. Population: full size, and the champion blueprint is IN it.
        assert_eq!(eng.state.live_count(), 3, "seeding must not shrink the pop");
        let champion_json = champion.to_json().unwrap();
        let seeded: Vec<String> = eng
            .state
            .live_hashes()
            .iter()
            .filter(|h| {
                eng.state
                    .net(h)
                    .map(|s| s.topology == champion_json)
                    .unwrap_or(false)
            })
            .map(|h| h.clone())
            .collect();
        assert_eq!(seeded.len(), 1, "exactly one slot carries the champion");

        // 2. Lineage: the imported founder names its source, not `founder-0`.
        let origin = eng.state.net(&seeded[0]).unwrap().created_from.clone();
        assert_eq!(origin.as_deref(), Some("seed:1000:feedface"));

        // 3. Receipt: the header names the inherited experiment, and the run-topology
        //    count stays user-only (the import is not a `run_topologies` entry).
        let header = crate::state::load_engine_json(&dir).unwrap();
        let source = header
            .run_topology_source
            .expect("a seeded run must record its source");
        assert_eq!(source.source_run_id, "1000");
        assert_eq!(source.champion_hash, "feedfacefeedface");
        assert_eq!(source.champion_params, Some(4242));
        assert_eq!(source.guardrail_mean, Some(0.30));
        assert_eq!(header.config.run_topology_count, 0);
        assert!(header.config.elite_previous_load);

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&prior);
    }

    /// Every failure mode degrades to an all-random population of the right size —
    /// no panic, no short population, no false `run_topology_source`.
    #[test]
    fn elite_previous_load_failures_fall_back_to_random() {
        // A prior run with NO guardrail champion recorded.
        let prior = std::env::temp_dir().join("gras-seedprior-none");
        let _ = std::fs::remove_dir_all(&prior);
        std::fs::create_dir_all(&prior).unwrap();
        write_prior_run(&prior.join("1000"), "1000", false);
        let dir = std::env::temp_dir().join("gras-seedprior-none-run");
        let _ = std::fs::remove_dir_all(&dir);
        let eng = seeded_engine(&dir, &prior).unwrap();
        assert_eq!(eng.state.live_count(), 3, "fallback still fills the pop");
        assert!(
            crate::state::load_engine_json(&dir)
                .unwrap()
                .run_topology_source
                .is_none(),
            "a skipped seed must not be recorded as an inheritance"
        );

        // A root that does not exist at all.
        let dir2 = std::env::temp_dir().join("gras-seedprior-missing-run");
        let _ = std::fs::remove_dir_all(&dir2);
        let eng2 = seeded_engine(&dir2, &prior.join("nowhere")).unwrap();
        assert_eq!(eng2.state.live_count(), 3);

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
        let _ = std::fs::remove_dir_all(&prior);
    }

    /// Certification needs no weight file: the live champion is scored in memory
    /// and recorded in `engine.json`, so `elite_save_safetensors` can stay off.
    #[test]
    fn guardrail_certifies_a_live_champion_without_weight_files() {
        let dir = std::env::temp_dir().join("gras-guardrail-live");
        let _ = std::fs::remove_dir_all(&dir);
        let data_dir = tiny_dataset_dir("guardrail-live");
        let config = RaceConfig {
            // pop_size 0 ⇒ `from_spec` seeds nothing (the default pools draw topologies a
            // 2-dim dataset cannot feed); the test seeds two tiny ones explicitly below.
            pop_size: 0,
            max_steps: Some(2),
            guardrail_matches: 3,
            // The point of the test: NO weight export at all.
            elite_save_safetensors: false,
            elite_checkpoint_weights: false,
            ..RaceConfig::defaults()
        };
        let trainer = crate::trainer::TabularTrainer::new(loss_fn())
            .with_holdout_scorer(|_net, game_i| -> Result<f32> { Ok(game_i as f32 + 10.0) });
        let mut eng =
            crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
                data_dir,
                config,
                fitness(),
                trainer,
                Some(7),
                Some(dir.clone()),
            ))
            .unwrap();
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        eng.run().unwrap();
        let exported = std::fs::read_dir(&dir).unwrap().flatten().any(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.starts_with("elite-") && n.ends_with(".safetensors")
        });
        assert!(
            !exported,
            "this test must prove certification without a weight file"
        );

        let verdict = eng
            .guardrail()
            .expect("the live champion must be scoreable in memory");
        assert_eq!(verdict.scores.len(), 3);
        let record = crate::state::load_engine_json(&dir)
            .unwrap()
            .guardrail
            .expect("the verdict must still be recorded");
        assert_eq!(record.champion_hash, verdict.hash);
        assert_eq!(record.matches_scored, 3);
        assert_eq!(record.holdout_mean, verdict.mean());
        let _ = std::fs::remove_dir_all(&dir);
    }

        /// The exam covers the WHOLE elite set, not just rank 1: every elite is scored
        /// `guardrail_matches` times and recorded with its own grade. The champion keeps
        /// the top-level block (the seeding contract).
    #[test]
    fn the_guardrail_examines_every_elite() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dir = std::env::temp_dir().join("gras-guardrail-elites");
        let _ = std::fs::remove_dir_all(&dir);
        let data_dir = tiny_dataset_dir("guardrail-elites");
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let config = RaceConfig {
            pop_size: 0,
            max_steps: Some(2),
            guardrail_matches: 2,
            elite_count: 3,
            ..RaceConfig::defaults()
        };
        let trainer = crate::trainer::TabularTrainer::new(loss_fn()).with_holdout_scorer(
            move |_net, game_i| -> Result<f32> {
                // The 6th call is the LAST elite's first game: fail it, so the test
                // covers a DEGRADED exam (`scored < played`) beside a full one.
                if counted.fetch_add(1, Ordering::SeqCst) == 5 {
                    return Err(flodl::tensor::TensorError::new("holdout game failed"));
                }
                Ok(game_i as f32)
            },
        );
        let mut eng =
            crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
                data_dir,
                config,
                fitness(),
                trainer,
                Some(7),
                Some(dir.clone()),
            ))
            .unwrap();
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        eng.run().unwrap();
        assert_eq!(
            eng.champion_hashes().len(),
            3,
            "the run must export three elites for this test to mean anything"
        );

        let verdict = eng
            .guardrail()
            .expect("the champion must be scored on fresh games");
        assert_eq!(verdict.scores.len(), 2, "the champion's own exam");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3 * 2,
            "every elite × every holdout game, not just the champion's"
        );

        let header = crate::state::load_engine_json(&dir).unwrap();
        assert_eq!(
            header.config.feature_split, None,
            "the run's split policy is recorded even when it is off"
        );
        let record = header.guardrail.expect("the exam must be recorded");
        assert_eq!(record.champion_hash, verdict.hash);
        assert_eq!(record.elites.len(), 3, "one record per examined elite");
        assert_eq!(record.elites[0].rank, 1);
        assert_eq!(record.elites[0].hash, record.champion_hash);
        assert_eq!(record.elites[0].matches_scored, 2, "the champion's exam is full");
        // `played` is the denominator: every elite was asked for both games…
        assert!(
            record.elites.iter().all(|e| e.matches_attempted == 2),
            "each elite plays the whole exam: {:?}",
            record.elites.iter().map(|e| e.matches_attempted).collect::<Vec<_>>()
        );
        // …while a scorer error shows up as `scored < played` on that one elite.
        let degraded: Vec<usize> = record
            .elites
            .iter()
            .filter(|e| e.matches_scored < e.matches_attempted)
            .map(|e| e.rank)
            .collect();
        assert_eq!(
            degraded,
            vec![3],
            "the failed holdout game must read as scored < played"
        );
        assert!(
            record.elites.iter().all(|e| !e.agreement.is_empty()),
            "every exam is graded, even a degraded one"
        );
        // Every elite carries its own window context — not just the champion: the
        // min..max the card prints, in reading order.
        assert!(
            record
                .elites
                .iter()
                .all(|e| matches!((e.race_min, e.race_max), (Some(lo), Some(hi)) if lo <= hi)),
            "every elite's card row needs its window: {:?}",
            record
                .elites
                .iter()
                .map(|e| (e.race_min, e.race_max))
                .collect::<Vec<_>>()
        );
        // …and the SECOND reference line is graded against that net's latest reading.
        assert!(
            record
                .elites
                .iter()
                .all(|e| e.latest_agreement.as_deref().is_some_and(|t| !t.is_empty())),
            "the latest-vs-exam line is graded for every examined elite: {:?}",
            record
                .elites
                .iter()
                .map(|e| e.latest_agreement.clone())
                .collect::<Vec<_>>()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `set_run_topologies` takes precedence: no extra net is injected above the
    /// user's list, ordinals do not shift, and nothing is recorded as inherited.
    #[test]
    fn manual_founders_win_over_the_previous_run_seed() {
        let prior = std::env::temp_dir().join("gras-seedprior-manual");
        let _ = std::fs::remove_dir_all(&prior);
        std::fs::create_dir_all(&prior).unwrap();
        let champion = write_prior_run(&prior.join("1000"), "1000", true);

        let dir = std::env::temp_dir().join("gras-seedprior-manual-run");
        let _ = std::fs::remove_dir_all(&dir);
        let data_dir = tiny_dataset_dir("seeded-manual");
        let manual = tiny_topology(11);
        let config = RaceConfig {
            pop_size: 3,
            elite_previous_load: true,
            elite_previous_dir: Some(prior.clone()),
            run_topologies: vec![manual.clone()],
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

        assert_eq!(eng.state.live_count(), 3, "the batch stays pop_size strong");
        let contains = |topo: &Topology| {
            let json = topo.to_json().unwrap();
            eng.state.live_hashes().iter().any(|h| {
                eng.state
                    .net(h)
                    .map(|s| s.topology == json)
                    .unwrap_or(false)
            })
        };
        assert!(
            contains(&manual),
            "the manual blueprint must be in the batch"
        );
        assert!(
            !contains(&champion),
            "the auto seed must NOT be injected when founders are named manually"
        );
        assert!(
            crate::state::load_engine_json(&dir)
                .unwrap()
                .run_topology_source
                .is_none(),
            "a skipped auto seed must not be recorded as an inheritance"
        );
        // The manual blueprint is slot 0 / `founder-0`, unshifted.
        assert_eq!(
            eng.champions.len(),
            0,
            "no champions recorded before any step"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&prior);
    }

    /// Inheritance is AUTOMATIC and ON by default: an unconfigured run looks for a
    /// compatible previous experiment by itself, and records nothing (falls back to
    /// an all-random batch) when there is none. `set_elite_previous_load(false)`
    /// opts out entirely.
    #[test]
    fn elite_previous_load_is_on_by_default_and_can_be_turned_off() {
        let defaults = RaceConfig::defaults();
        assert!(defaults.elite_previous_load);
        assert!(defaults.elite_previous_dir.is_none());

        // On by default: this tabular run finds no compatible tabular source (only
        // the engine's own `results/`), so nothing is inherited and the batch is
        // still full and all-random.
        let dir = std::env::temp_dir().join("gras-seedprior-on");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.config.pop_size = 2;
        let topos = crate::engine::population::fill_unique(
            &eng.config,
            eng.run_seed(),
            Vec::new(),
            eng.config.pop_size,
        );
        assert_eq!(topos.len(), 2);
        assert!(
            crate::state::load_engine_json(&dir)
                .unwrap()
                .run_topology_source
                .is_none()
        );
        let _ = std::fs::remove_dir_all(&dir);

        // Explicit opt-out: no lookup, no receipt, still a full random population.
        let dir = std::env::temp_dir().join("gras-seedprior-optout");
        let _ = std::fs::remove_dir_all(&dir);
        let config = RaceConfig {
            pop_size: 2,
            elite_previous_load: false,
            ..RaceConfig::defaults()
        };
        let eng =
            crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
                tiny_dataset_dir("seedprior-optout"),
                config,
                fitness(),
                crate::trainer::TabularTrainer::new(loss_fn()),
                Some(7),
                Some(dir.clone()),
            ))
            .unwrap();
        assert_eq!(eng.state.live_count(), 2, "opt-out still fills the pop");
        assert!(
            crate::state::load_engine_json(&dir)
                .unwrap()
                .run_topology_source
                .is_none()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

        /// The one ranking order, and the invariant the two stop-time cards need: an
        /// equal-score tie goes to the SMALLER net (unknown size last), an equal-score,
        /// equal-size tie to the smaller hash — and the guardrail's certified champion is
        /// exactly the elite table's rank 1, which is where the two cards used to flip.
    #[test]
    fn ranking_ties_go_to_the_smaller_net_then_the_hash() {
        let dir = std::env::temp_dir().join("gras-ranking-tie-break");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 21).unwrap();
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let hashes = eng.state.live_hashes();
        // The solved-cartpole case: identical windows ⇒ EXACTLY equal scores.
        for h in &hashes {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            for _ in 0..5 {
                buf.push(500.0);
            }
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
            eng.state.net_mut(h).unwrap().step = 5;
        }
        // Params decide: the largest net is ranked LAST even though it is first by hash.
        for (i, params) in [9_000_i64, 1_000, 5_000].into_iter().enumerate() {
            eng.state.net_mut(&hashes[i]).unwrap().meta.params = Some(params);
        }
        let ranked: Vec<String> = eng.ranked_live().into_iter().map(|(h, _)| h).collect();
        assert_eq!(
            ranked,
            vec![hashes[1].clone(), hashes[2].clone(), hashes[0].clone()],
            "equal fitness ⇒ fewest parameters first"
        );
        eng.config.elite_count = 1;
        eng.record_champions();
        assert_eq!(
            eng.champion_for_guardrail().as_ref(),
            Some(&hashes[1]),
            "the certified champion is the elite table's rank 1"
        );
        // Same size as well ⇒ the hash breaks the tie, so the order is still total.
        for h in &hashes {
            eng.state.net_mut(h).unwrap().meta.params = Some(4_000);
        }
        let ranked: Vec<String> = eng.ranked_live().into_iter().map(|(h, _)| h).collect();
        let mut by_hash = hashes.clone();
        by_hash.sort();
        assert_eq!(ranked, by_hash, "a full tie falls back to the hash order");
        // An unknown size loses to a known one (a net with no meta yet is not preferred).
        eng.state.net_mut(&hashes[0]).unwrap().meta.params = None;
        assert_eq!(eng.ranked_live()[2].0, hashes[0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The guardrail record survives `refresh_header` (what a resume-then-stop
    /// does): a resumed run must never erase the champion the next run inherits.
    #[test]
    fn guardrail_record_rides_the_header_and_survives_a_refresh() {
        let dir = std::env::temp_dir().join("gras-guardrail-record");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        let record = crate::state::GuardrailRecord {
            champion_hash: "abcdefghabcdefgh".into(),
            champion_params: Some(77),
            race_smoothed: Some(0.5),
            race_std: Some(0.08),
            race_score: Some(0.42),
            race_se: Some(0.02),
            race_samples: 5,
            race_latest: Some(0.52),
            holdout_mean: Some(0.55),
            holdout_std: Some(0.01),
            matches_scored: 16,
            verdict: "consistent with the race signal ✅".into(),
            elites: vec![crate::state::GuardrailElite {
                rank: 1,
                hash: "abcdefghabcdefgh".into(),
                holdout_mean: Some(0.55),
                matches_scored: 16,
                agreement: "consistent".into(),
                ..Default::default()
            }],
        };
        eng.header.set_guardrail(Some(record.clone()));
        crate::state::write_guardrail_record(&dir, &record).unwrap();

        // On disk…
        let loaded = crate::state::load_engine_json(&dir).unwrap();
        assert_eq!(loaded.guardrail, Some(record.clone()));
        // …and still there after a header rebuild.
        eng.refresh_header().unwrap();
        let again = crate::state::load_engine_json(&dir).unwrap();
        assert_eq!(again.guardrail, Some(record));

        // A legacy header with no block loads as None (not as a fake champion).
        std::fs::write(dir.join("engine.json"), r#"{"run_id":"legacy"}"#).unwrap();
        let legacy = crate::state::load_engine_json(&dir).unwrap();
        assert!(legacy.guardrail.is_none() && legacy.run_topology_source.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

        /// `guardrail_matches = 0` is the OFF switch: the exam returns `None` and records
        /// nothing. The engine's test trainer has NO holdout scorer (its default panics if
        /// called), so a clean `None` here is proof the skip happens BEFORE any game —
        /// a clamped-to-1 implementation would panic.
    #[test]
    fn zero_guardrail_matches_skips_the_exam_entirely() {
        let dir = std::env::temp_dir().join("gras-guardrail-off");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.config.guardrail_matches = 0;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        seed_raw_fitness(&mut eng, 0.5);
        assert!(
            eng.guardrail().is_none(),
            "0 matches must skip the exam, not run one game"
        );
        // Nothing recorded: a later reader sees "not scored", never a fake verdict.
        assert!(
            eng.header.guardrail.is_none(),
            "a skipped exam must not stamp a verdict into engine.json"
        );
        // And the OFF path does not even need a champion: an empty field is fine.
        let empty_dir = std::env::temp_dir().join("gras-guardrail-off-empty");
        let _ = std::fs::remove_dir_all(&empty_dir);
        let mut empty = engine(&empty_dir, 8).unwrap();
        empty.config.guardrail_matches = 0;
        assert!(empty.guardrail().is_none());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&empty_dir);
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
            // The ledger was removed: the tick writes durability snapshots, never
            // a checkpoints.json diagnostic.
            assert!(
                !dir.join("checkpoints.json").exists(),
                "the removed checkpoint ledger must not be written"
            );
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

        /// `checkpoint_every = 0` is the OFF switch for the durability tick: no
        /// `checkpoint-elite-*` snapshots, no per-tick CSV flush — but the run is still a
        /// COMPLETE run (stop-time artifacts and frontier states land).
    #[test]
    fn zero_checkpoint_every_skips_the_tick_but_still_finishes_the_run() {
        let dir = std::env::temp_dir().join("gras-ckpt-off");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.config.checkpoint_every = 0;
        eng.config.max_steps = Some(4);
        eng.config.elite_checkpoint_weights = true; // would snapshot on any tick
        // Both OFF switches at once, through a REAL completed run: the exam skip must
        // also survive the run path (not just a hand-built engine).
        eng.config.guardrail_matches = 0;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        assert_eq!(eng.run().unwrap(), StopReason::MaxSteps);
        assert!(
            eng.guardrail().is_none(),
            "guardrail_matches = 0 must skip after a real run too"
        );
        let snaps: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with("checkpoint-elite-"))
            .collect();
        assert!(
            snaps.is_empty(),
            "0 cadence must write no durability snapshots, got {snaps:?}"
        );
        // …and the stop-time writers still ran: the run is complete, not truncated.
        assert!(dir.join("elites.json").exists(), "elite index is a stop artifact");
        assert!(dir.join("engine.json").exists());
        let nets = std::fs::read_dir(dir.join("nets"))
            .unwrap()
            .flatten()
            .count();
        assert!(nets > 0, "frontier states land at stop even with the tick off");
        let _ = std::fs::remove_dir_all(&dir);
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
        // Give everyone a verdict, one clearly worst (Minimize: highest = worst), so
        // the inverse roulette would always target it without probation.
        let mut hashes = eng.state.live_hashes();
        hashes.sort();
        for (i, h) in hashes.iter().enumerate() {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(0.5 + i as f32 * 0.1);
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        }
        eng.state.net_mut(&hashes[2]).unwrap().entered_at_step = 10; // fresh
        // Advance the ENGINE clock to 11: the probation check reads `step_clock()`,
        // not the `clock` argument, and pushing buffers alone leaves it at 0.
        for h in [&hashes[0], &hashes[1]] {
            eng.state
                .record_step(
                    h,
                    crate::state::NetMetrics {
                        step: 10,
                        train_loss: 0.0,
                        eval_loss: None,
                        fitness: 0.5,
                        informative: vec![],
                        frozen: false,
                    },
                )
                .unwrap();
        }
        assert_eq!(eng.step_clock(), 11, "engine clock matches the test's");
        let clock = 11; // within the 3-step window
        for _ in 0..20 {
            let victim = eng
                .select_mutation_victim(clock, 0)
                .unwrap()
                .expect("the two off-probation nets remain cullable");
            assert_ne!(
                victim, hashes[2],
                "a net inside its probation steps must never be the victim"
            );
        }
        // Past the window the protection lifts and the net is eligible again — assert
        // via the predicate, since the roulette draw is probabilistic.
        let late = 13; // 13 − 10 = 3 ≥ window 3
        assert!(
            !eng.on_probation(&hashes[2], late),
            "protection expires after k clocks"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn probation_is_never_broken_when_every_non_elite_is_fresh() {
        // HARD SHIELD: a firing roll must not buy its slot by culling a fresh net. With
        // everyone inside its channel window the roll is SKIPPED — nothing is culled.
        let dir = std::env::temp_dir().join("gras-probation-no-slot");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.config.mutation_probation_steps = 100; // effectively everyone
        eng.config.elite_count = 1;
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        // The population-level check agrees: nobody is cull-eligible.
        assert!(
            !eng.has_cull_eligible(),
            "every net is inside its channel window"
        );
        // …and every victim selector returns None rather than a fresh net.
        assert_eq!(
            eng.select_mutation_victim(5, 0).unwrap(),
            None,
            "no cull-eligible victim → the pick is skipped, never a fresh net"
        );
        assert_eq!(
            eng.select_random_victim(5).unwrap(),
            None,
            "the random arm skips too"
        );
        // The channel entry point refuses to build an immigrant at all.
        assert_eq!(
            eng.evolve_random_immigrant(5, 0).unwrap(),
            None,
            "the mutation roll is spent without inserting anything"
        );
        assert_eq!(eng.state.live_count(), 3, "nobody was culled");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn probation_steps_is_recorded_in_engine_json() {
        let dir = std::env::temp_dir().join("gras-probation-json");
        let _ = std::fs::remove_dir_all(&dir);
        let data_dir = tiny_dataset_dir("probation-json");
        let config = RaceConfig {
            mutation_probation_steps: 4,
            crossover_probation_steps: 6,
            elite_join_probation_steps: 5,
            elite_crown_lifetime_steps: 5,
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
        assert_eq!(header.config.crossover_probation_steps, 6);
        assert_eq!(header.config.elite_join_probation_steps, 5);
        assert_eq!(header.config.elite_crown_lifetime_steps, 5);
        assert!(header.config.elite_checkpoint_weights);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn elite_join_probation_crown_lifetime() {
        // A caught-up child arrives rankable (full buffer), but elite probation means a
        // fresh net trains instead of freezing on arrival. Minimize: lowest is best.
        let dir = std::env::temp_dir().join("gras-elite-probation");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.config.elite_count = 1;
        eng.config.elite_join_probation_steps = 3;
        eng.config.elite_crown_lifetime_steps = 5;
        // The born-based CHANNEL windows gate a seat too, so 0 them out: this
        // test pins the elite JOIN window (k=3), not the cull shield.
        eng.config.mutation_probation_steps = 0;
        eng.config.crossover_probation_steps = 0;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let mut hashes = eng.state.live_hashes();
        hashes.sort();
        // Step a few clocks so both nets carry a verdict (`step_clock` is one past the
        // last recorded clock: 4 group clocks → 4). The gate reads the counters.
        for clock in 0..4 {
            for h in &hashes {
                eng.step_one_net(h, clock).unwrap();
            }
        }
        assert_eq!(
            eng.step_clock(),
            4,
            "clock advanced by the group steps (one recorded clock per net per clock)"
        );
        // `hashes[0]` is the BEST net but FRESH; `hashes[1]` is worse but
        // already through its probation.
        let best_vals = [0.1_f32, 0.9];
        for (i, h) in hashes.iter().enumerate() {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(best_vals[i]);
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        }
        eng.state.net_mut(&hashes[0]).unwrap().elite_probation_age = 0; // k served: 0
        eng.state.net_mut(&hashes[1]).unwrap().elite_probation_age = 4; // k served: 4
        // Fresh best net is BARRED; the served worst net takes the seat.
        assert_eq!(
            eng.elite_hashes(),
            vec![hashes[1].clone()],
            "a fresh net cannot be an elite (would freeze on arrival)"
        );
        // Serve the window: the genuinely-best net becomes the elite.
        eng.state.net_mut(&hashes[0]).unwrap().elite_probation_age = 4;
        assert_eq!(
            eng.elite_hashes(),
            vec![hashes[0].clone()],
            "protection lifts once the net has served k clocks"
        );
        // If EVERY rankable net is fresh the crown is EMPTY (no fallback) — nothing is
        // crowned or frozen before it has served its k clocks.
        for h in &hashes {
            eng.state.net_mut(h).unwrap().elite_probation_age = 0;
        }
        assert!(
            eng.elite_hashes().is_empty(),
            "no net may be crowned/frozen while every net is younger than the window"
        );

        // ── Seat lifetime + probation restart, through refresh_crown ──────
        // Both nets have served probation: the best takes the seat.
        for h in &hashes {
            eng.state.net_mut(h).unwrap().elite_probation_age = 4;
        }
        eng.refresh_crown(0);
        assert!(
            eng.frozen_crown.contains(&hashes[0]),
            "the best net takes the seat once probation is served"
        );
        // Sit the full lifetime: the entry refresh plus j-1 = 4 refreshes keep the seat
        // (seat ages 0..4), and the 5th age-eliminates it at peak even while ranked #1.
        let mut clock = 1;
        for _ in 0..4 {
            eng.refresh_crown(clock);
            clock += 1;
            assert!(
                eng.frozen_crown.contains(&hashes[0]),
                "seat is younger than the lifetime: it keeps the seat"
            );
        }
        eng.refresh_crown(clock);
        clock += 1;
        assert!(
            !eng.frozen_crown.contains(&hashes[0]),
            "the lifetime expires: the top net is age-eliminated at peak"
        );
        // ANY seat exit restarts probation: the ex-elite is back at 0 (+ this refresh's
        // tick) and trains like any non-elite until it has re-served k clocks.
        let ex = eng.state.net(&hashes[0]).unwrap();
        assert_eq!(
            ex.elite_probation_age, 1,
            "seat exit restarts probation at 0 (+1 tick)"
        );
        assert_eq!(ex.elite_seat_age, 0, "seat age dies with the seat");
        // Re-serve k = 3: the exit refresh counted the first clock (age 1), so two more
        // complete the window; throughout the ex-elite is NOT seated.
        eng.refresh_crown(clock);
        clock += 1;
        assert_eq!(
            eng.state.net(&hashes[0]).unwrap().elite_probation_age,
            2,
            "probation re-serves one clock per refresh"
        );
        assert!(
            !eng.elite_hashes().contains(&hashes[0]),
            "still barred at 2 of k = 3"
        );
        eng.refresh_crown(clock);
        clock += 1;
        assert_eq!(
            eng.state.net(&hashes[0]).unwrap().elite_probation_age,
            3,
            "probation re-served: k = 3"
        );
        assert!(
            !eng.frozen_crown.contains(&hashes[0]),
            "the ex-elite kept training right up to the eligibility clock"
        );
        // The next refresh's CHECK (age 3 >= k) seats it again.
        eng.refresh_crown(clock);
        assert!(
            eng.frozen_crown.contains(&hashes[0]),
            "…and it reclaims the seat (it is still the best net)"
        );
        assert_eq!(
            eng.state.net(&hashes[0]).unwrap().elite_seat_age,
            1,
            "the reclaimed seat starts a FRESH lifetime (seat age 0 + the tick)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn crossover_probation_shields_crossover_children_with_its_own_window() {
        let dir = std::env::temp_dir().join("gras-crossover-probation");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 7).unwrap();
        eng.config.mutation_probation_steps = 3;
        eng.config.crossover_probation_steps = 7;
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let mut hashes = eng.state.live_hashes();
        hashes.sort();
        // Same birth clock for all three; only the ORIGIN picks the window.
        for h in &hashes {
            eng.state.net_mut(h).unwrap().entered_at_step = 10;
        }
        eng.state.net_mut(&hashes[0]).unwrap().created_from =
            Some("crossover-one-point:parents=a,b".into());
        eng.state.net_mut(&hashes[1]).unwrap().created_from = Some("random".into());
        eng.state.net_mut(&hashes[2]).unwrap().created_from = Some("founder-2".into());

        // 1 clock fresh: each net's own window covers it.
        assert!(eng.on_probation(&hashes[0], 11), "crossover window is 7");
        assert!(eng.on_probation(&hashes[1], 11), "mutation window is 3");
        // 3 clocks fresh: the mutation window has expired, the crossover one has not.
        let c = 13;
        assert!(
            eng.on_probation(&hashes[0], c),
            "crossover child still fresh at 3 (window 7)"
        );
        assert!(
            !eng.on_probation(&hashes[1], c),
            "random immigrant's 3-clock window is done"
        );
        assert!(
            !eng.on_probation(&hashes[2], c),
            "a founder uses the mutation window, not the crossover one"
        );
        // 7 clocks fresh: the crossover window expires too.
        assert!(
            !eng.on_probation(&hashes[0], 17),
            "crossover window expires after k clocks"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }    /// The lottery's DEBUG who-line: both counts always stated, and the side the
    /// reader needs (the smaller one) listed in full — the other is its complement.
    /// The Summ view is the aggregate line (`log_lottery_summary`), not this list.
    #[test]
    fn lottery_who_line_names_the_smaller_side_and_caps_the_list() {
        let owned = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // Few trained, many idled → name the trained ones.
        assert_eq!(
            lottery_who_line(&owned(&["a", "b"]), &owned(&["c", "d", "e"])),
            "2 trained: a b · 3 idled"
        );
        // Few idled → name those instead.
        assert_eq!(
            lottery_who_line(&owned(&["a", "b", "c", "d", "e"]), &owned(&["z"])),
            "5 trained · 1 idled: z"
        );
        // Long lists are capped with a `+N` tail on the named side only.
        let trained: Vec<String> = (0..13).map(|i| format!("t{i}")).collect();
        let withheld: Vec<String> = (0..20).map(|i| format!("w{i}")).collect();
        let line = lottery_who_line(&trained, &withheld);
        assert!(line.starts_with("13 trained: t0 t1 "), "{line}");
        assert!(line.ends_with(" +1 · 20 idled"), "{line}");
    }

    /// The rollup wording contract: `trained + frozen + idled + newborn = live` always
    /// holds, `stepped` counts updaters + crown act-and-measure passes (an idle net did
    /// not act at all), the sat-out causes stay distinguishable, and an all-trained
    /// clock carries no sat-out segment.
    #[test]
    fn step_rollup_line_accounts_for_every_live_net() {
        // Test shorthand for the eleven vitals, in the old positional order: these
        // cases are about the WORDING contract, not about re-listing every field.
        fn rollup(s: [usize; 11]) -> String {
            step_rollup_line(&StepVitals {
                stepped: s[0],
                live: s[1],
                frozen: s[2],
                // The fixtures model the pre-lottery path: every non-crown net updated.
                trained: s[0].saturating_sub(s[2]),
                would_train: s[0].saturating_sub(s[2]),
                lottery_skips: 0,
                newborn: s[3],
                eligible: s[4],
                probation: s[5],
                age_blocked: s[6],
                seats_held: s[7],
                seats_capacity: s[8],
                join_k: s[9],
                lifetime_j: s[10],
                crown_armed: true,
            })
        }

        // 84 ran (74 updated + 10 crown), 16 newborn, 100 live.
        let line = rollup([84, 100, 10, 16, 74, 26, 0, 10, 10, 5, 5]);
        assert_eq!(
            line,
            "ran 84 of 100 = 74 updating + 10 crown │ held back 16 = 16 newborn (first step next clock) │ 100 = 84 + 16 │ seats 10 of 10 │ seat gates, over all 100 nets: 74 may sit = 100 − 26 in a join window or cull shield (5 clocks) − 0 age-blocked (5-clock lifetime)"
        );
        // The rollup is ONE log line: the crown tally rides inline, never an indent.
        assert!(!line.contains('\n'), "rollup stays one line: {line}");
        // The identities the format now PRINTS, not just upholds: 74 + 10 = 84,
        // 84 + 16 = 100, and 74 + 26 + 0 = 100.
        assert!(74 + 10 == 84 && 84 + 16 == 100 && 74 + 26 + 0 == 100);

        // Newborns only (no crown yet): one held-back cause, no stray '+'.
        let newborn_only = rollup([4, 6, 0, 2, 0, 6, 0, 0, 1, 5, 5]);
        assert!(
            newborn_only.contains(
                "ran 4 of 6 = 4 updating │ held back 2 = 2 newborn (first step next clock)"
            ),
            "{newborn_only}"
        );
        assert!(!newborn_only.contains("crown"), "{newborn_only}");

        // Frozen only: every live net RAN (one of them was the crown), so there is no
        // held-back segment at all — the crown rides the `ran` identity.
        let frozen_only = rollup([6, 6, 1, 0, 5, 1, 0, 1, 1, 5, 5]);
        assert!(
            frozen_only.contains("ran 6 of 6 = 5 updating + 1 crown"),
            "{frozen_only}"
        );
        assert!(!frozen_only.contains("newborn"), "{frozen_only}");
        assert!(
            !frozen_only.contains("held back"),
            "a crown that acted is not a held-back net: {frozen_only}"
        );

        // The lottery names its budget explicitly: tickets / eligible / idled. An idle
        // net did not run at all, so `stepped` is the tickets plus the crown (5 + 1) —
        // never the group.
        let lottery = step_rollup_line(&StepVitals {
            stepped: 6,
            trained: 5,
            live: 10,
            frozen: 1,
            would_train: 9,
            lottery_skips: 4,
            newborn: 0,
            eligible: 9,
            probation: 1,
            age_blocked: 0,
            seats_held: 1,
            seats_capacity: 1,
            join_k: 5,
            lifetime_j: 5,
            crown_armed: true,
        });
        assert!(
            lottery.contains("ran 6 of 10 = 5 ticketed + 1 crown"),
            "{lottery}"
        );
        assert!(
            lottery.contains("held back 4 = 4 idled (no train, no eval)"),
            "{lottery}"
        );
        assert!(lottery.contains("10 = 6 + 4"), "{lottery}");
        assert!(
            lottery.contains("would-train 9 = 5 ticketed + 4 idled"),
            "{lottery}"
        );
        // `ran 6` = 5 tickets + 1 crown, and 6 + 4 idled = 10 live — the head and
        // the held-back list account for every net exactly once.

        // All trained: NO did-not-run segment at all (no births, no idles), and no crown
        // note either — a zero would be noise.
        let clean = rollup([4, 4, 0, 0, 0, 4, 0, 0, 1, 5, 5]);
        assert_eq!(
            clean,
            "ran 4 of 4 = 4 updating │ 4 = 4 + 0 │ seats 0 of 1 │ seat gates, over all 4 nets: 0 may sit = 4 − 4 in a join window or cull shield (5 clocks) − 0 age-blocked (5-clock lifetime)"
        );

        // Crown OFF: the line still prints — the budget and the population identity are
        // not the crown's business — but the seat and gate legs are replaced by one
        // statement of the fact, never a `seats 0 of 5` that reads like a broken crown.
        let crownless = step_rollup_line(&StepVitals {
            stepped: 20,
            trained: 20,
            live: 24,
            frozen: 0,
            would_train: 20,
            lottery_skips: 0,
            newborn: 4,
            eligible: 24,
            probation: 0,
            age_blocked: 0,
            seats_held: 0,
            seats_capacity: 5,
            join_k: 5,
            lifetime_j: 5,
            crown_armed: false,
        });
        assert!(
            crownless.contains("ran 20 of 24 = 20 updating"),
            "a crown-less run still reports its training budget: {crownless}"
        );
        assert!(
            crownless.contains(
                "crown off (freeze_elites = false): 0 of 24 seats, no gate — every live net is seat-eligible"
            ),
            "{crownless}"
        );
        assert!(
            !crownless.contains("seat gates") && !crownless.contains("seats 0 of 5"),
            "{crownless}"
        );
    }
}
