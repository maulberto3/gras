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
//! 5. Stop criteria checked (max_steps, max_target_fitness, custom_stop) —
//!    custom stop) — log which fired.
//! 6. Repeat — the loop's clock is the nets' recorded steps (every net is at
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
use crate::trainer::Trainer;
use crate::trainer::stream::{BatchStream, PoolSplit};
// Used by the debug contract probe in `step_one_net` and by the checkpoint
// surprise exam (`run_checkpoint_exam`) — the exam runs in release too, so
// this import is NOT debug-gated.
use crate::utils::race_steps::eval_one_step;

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
use super::smoothing::{SMOOTHING_WINDOW, RollingBuffer, rolling_mean};

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

// ── RaceEngine ───────────────────────────────────────────────────────────────

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
pub struct RaceEngine {
    pub(crate) run_dir: std::path::PathBuf,
    pub(crate) header: RunHeader,
    pub(crate) config: RaceConfig,
    /// Run-level context stamped into each net's meta block.
    pub(crate) meta_ctx: crate::engine::config::RunMetaCtx,
    pub(crate) stream: BatchStream,
    pub(crate) dataset: crate::utils::tabular_data::Dataset,
    pub(crate) fitness: Fitness,
    pub(crate) metrics: Vec<Metric>,
    /// The caller-supplied training scheme. The engine never trains — it only
    /// orchestrates the population lifecycle and delegates one net/one step
    /// to this contract (see `crate::trainer::Trainer`).
    pub(crate) trainer: Box<dyn crate::trainer::Trainer>,
    /// Per-step log verbosity for the engine.
    pub(crate) log_level: crate::engine::config::LogLevel,

    /// Per-step step-log counters captured during the evolve phase and
    /// consumed by the `Minimal` framed table at the start of the NEXT step
    /// (so the table shows "what happened last step" without extra lines).
    pub(crate) step_evolve: StepEvolve,
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

impl RaceEngine {
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
    pub fn new<T: Trainer + 'static>(spec: crate::engine::run_spec::RunSpec<T>) -> Result<Self> {
        let crate::engine::run_spec::RunSpec {
            data_dir,
            config,
            fitness,
            trainer,
            seed,
            run_dir,
        } = spec;
        let trainer: Box<dyn Trainer> = Box::new(trainer);
        let dataset = crate::utils::tabular_data::resolve_dataset(&data_dir)?
            .to_device(config.device())?;
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
        // Infer dims from data_dir when TopologyOptions leaves them unset.
        // input_dim/output_dim are now Option<usize> — None means
        // "fill from dataset at run start", Some(v) means user set it (and
        // the engine validates v against the dataset below).
        let inferred_input_dim = dataset.inputs.shape()[1] as usize;
        let inferred_output_dim = dataset.targets.shape()[1] as usize;
        let mut topology_options = config.topology_options;
        let mut topology_errors: Vec<String> = Vec::new();
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
        // The trainer owns batch geometry. Resolve it BEFORE the header so
        // engine.json records the stream shape the run actually used.
        let (batch_size, eval_batch_size) = match trainer.stream_shape() {
            Some(shape) => (shape.batch_size, shape.eval_batch_size),
            None => (default_batch_size, default_batch_size),
        };

        let header = RunHeader::from_race_options_at(
            RunConfig {
                run_id: run_id.clone(),
                run_seed,
                fitness_label: FitnessLabel(fitness.label().to_string()),
                fitness_direction: fitness.direction(),
                input_dim: dataset.inputs.shape()[1] as usize,
                output_dim: dataset.targets.shape()[1] as usize,
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

        // Shared batch stream — engine infrastructure. The
        // trainer may shape batch sizes (`Trainer::stream_shape`) but the
        // split ratio is NOT overridable: which rows are held out protects
        // fitness comparability across every net, whatever the recipe.
        let split = PoolSplit::of(&dataset, train_eval_split_ratio, run_seed);
        let batch_stream = BatchStream::new(run_seed, batch_size, split)
            .with_eval_batch_size(eval_batch_size)
            .with_held_out_eval_rows(held_out_eval_rows)
            .with_checkpoint_every(config.checkpoint_every);

        let log_level = config.log_level;
        let meta_ctx = crate::engine::config::RunMetaCtx {
            input_dim: header.input_dim,
            output_dim: header.output_dim,
            batch_size: batch_stream.batch_size(),
            dropout_prob: header.topology_options.dropout_prob,
            fitness_label: header.fitness_label.0.clone(),
            loss_label: "cross_entropy".to_string(),
            direction: format!("{:?}", header.fitness_direction).to_lowercase(),
            pop_size: config.pop_size,
            run_seed,
        };

        let mut engine = RaceEngine {
            run_dir,
            header,
            dataset,
            config: RaceConfig {
                ..config
            },
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
            culls: 0,
            children_born_at_clock: HashMap::new(),
            step_evolve: StepEvolve::default(),
            checkpoints: Vec::new(),
            minimal_prev_means: None,
            log_level,
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
        // The population clock is the MAX step across live nets — the step the
        // original population has reached. Reading an arbitrary net (e.g. the
        // first hash) is wrong: a crossover child inserted mid-evolve is
        // caught up to `clock` while the population sits at `clock + 1`, so a
        // first-hash read can return the child's step and REPEAT the whole
        // iteration (duplicate rollup lines, extra training, budget skew).
        // Max is order-independent: children trail by one, originals define
        // the clock.
        self.state
            .live_hashes()
            .iter()
            .filter_map(|h| self.state.net(h).map(|s| s.step))
            .max()
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

    // ── Population setup ────────────────────────────────────────────────────

    /// Populate the initial population from a list of finalized topologies.
    ///
    /// Each topology is built on the engine's device with its own embedded
    /// seed + dropout_prob (read from `topology.options` by `Network::build`),
    /// and gets a fresh Adam optimizer. The topologies must already be
    /// finalized (the engine calls `Network::build`, which re-validates).
    ///
    /// This mirrors the generational engine's population bootstrap and log
    /// shape, adjusted for the step-race model: no generations, no per-gen
    /// snapshots, but the same per-net build diagnostics and the same
    /// deterministic per-individual seed derivation.
    /// Iter-6 Tier B (Tier A's loader is pinned): reconstruct a run's live
    /// population from disk and replay each net to its recorded step.
    ///
    /// Reads `engine.json` for the run identity and every `nets/<hash>.json`
    /// for the live frontier. Culled nets' JSONs are final tombstones and are
    /// ignored — only the still-alive frontier files (the ones the last run
    /// wrote at cull/stop time) are loaded. This constructor assumes the run
    /// directory holds a **complete, consistent snapshot**: pop-size checking
    /// and grace-counter reconstruction land with Tier A (pinned).
    ///
    /// Each loaded net is rebuilt (topology + weight seed → `Network` +
    /// `Optimizer`) and replayed through `replay_loaded_net`, which asserts
    /// metric parity with the recorded `last_metrics` — a silent
    /// reconstruction drift fails loudly here.
    /// Resume a run from its run directory + the same spec shape as `new`
    /// (minus seed/run_dir — both come from the persisted `engine.json` and
    /// the directory itself). The trainer must be the same scheme the run
    /// started with; split ratio / eval rows also come from the header.
    pub fn resume<T: Trainer + 'static>(
        run_dir: std::path::PathBuf,
        data_dir: std::path::PathBuf,
        config: RaceConfig,
        fitness: Fitness,
        trainer: T,
    ) -> Result<Self> {
        let trainer: Box<dyn Trainer> = Box::new(trainer);
        let header = crate::state::load_engine_json(&run_dir)?;
        let dataset = crate::utils::tabular_data::resolve_dataset(&data_dir)?
            .to_device(config.device())?;
        let metrics = config.metrics.clone();
        let run_seed = header.run_seed;
        // On resume, stream shape comes from the run header (data integrity),
        // but the trainer's stream_shape() can still override batch sizes.
        let header_stream_ratio = header.train_eval_split_ratio.unwrap_or(0.2); // legacy default if header missing it
        let _header_held_out = header.held_out_eval_rows.unwrap_or(256); // legacy default if header missing it

        // Same rule on resume: the trainer's stream shape (same trainer the
        // run started with) overrides batch sizes; the split ratio comes from
        // the run header — data integrity is not re-decided on resume.
        let (batch_size, eval_batch_size) = match trainer.stream_shape() {
            Some(shape) => (shape.batch_size, shape.eval_batch_size),
            None => (16, 16), // conservative default when no stream_shape override
        };
        let split = PoolSplit::of(&dataset, header_stream_ratio, run_seed);
        let stream = BatchStream::new(run_seed, batch_size, split)
            .with_eval_batch_size(eval_batch_size)
            .with_checkpoint_every(config.checkpoint_every);

        let log_level = config.log_level;
        let meta_ctx = crate::engine::config::RunMetaCtx {
            input_dim: header.input_dim,
            output_dim: header.output_dim,
            batch_size,
            dropout_prob: header.topology_options.dropout_prob,
            fitness_label: header.fitness_label.0.clone(),
            loss_label: "cross_entropy".to_string(),
            direction: format!("{:?}", header.fitness_direction).to_lowercase(),
            pop_size: config.pop_size,
            run_seed,
        };

        let mut engine = RaceEngine {
            run_dir: run_dir.clone(),
            header,
            config,
            meta_ctx,
            stream,
            dataset,
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
            culls: 0,
            children_born_at_clock: HashMap::new(),
            step_evolve: StepEvolve::default(),
            checkpoints: Vec::new(),
            minimal_prev_means: None,
            log_level,
            trainer,
        };

        // Load every persisted net snapshot from `nets/` and replay it.
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
            let hash = state.hash.clone();
            let step = state.step;
            let child = engine.replay_loaded_net(state)?;

            // Insert the replayed net into the live maps. Its rolling buffers
            // are seeded from the replayed trajectory so the smoothed stats are
            // warm on resume (not cold) — ranking resumes where it left off.
            let mut buf = RollingBuffer::new(SMOOTHING_WINDOW);
            let mut train_buf = RollingBuffer::new(SMOOTHING_WINDOW);
            let mut eval_buf = RollingBuffer::new(SMOOTHING_WINDOW);
            if let Some(m) = engine
                .state
                .net(&hash)
                .and_then(|s| s.last_metrics.as_ref())
            {
                buf.push(m.fitness);
                train_buf.push(m.train_loss);
                if let Some(e) = m.eval_loss {
                    eval_buf.push(e);
                }
            }
            engine.state.insert(child.state.clone(), None);
            engine.networks.insert(child.state.hash.clone(), child.net);
            engine
                .optimizers
                .insert(child.state.hash.clone(), child.optimizer);
            engine.rolling_fitness.insert(hash.clone(), buf);
            engine.rolling_train.insert(hash.clone(), train_buf);
            engine.rolling_eval.insert(hash.clone(), eval_buf);
            loaded += 1;
            info!(
                "race resume: net {} replayed to step {} (parity ok)",
                hash, step
            );
        }
        if loaded != engine.config.pop_size {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: expected {} live nets in population (per config.pop_size), but found {} live nets in {}",
                engine.config.pop_size, loaded, nets_dir.display()
            )).into());
        }
        info!(
            "race resume: loaded {} nets from {}",
            loaded,
            run_dir.display()
        );
        // Reload the checkpoint ledger so the crossover gates compare children
        // against the SAME historical bars the original run recorded. Without
        // this, a resumed run's gates start empty and children born before the
        // resume point would face no gate at all.
        engine.load_checkpoints()?;
        Ok(engine)
    }

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
            self.rolling_fitness
                .insert(hash.clone(), RollingBuffer::new(SMOOTHING_WINDOW));
            self.rolling_train
                .insert(hash.clone(), RollingBuffer::new(SMOOTHING_WINDOW));
            self.rolling_eval
                .insert(hash.clone(), RollingBuffer::new(SMOOTHING_WINDOW));
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
        self.stream.set_checkpoint_every(self.config.checkpoint_every);
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
                "race start: run={} seed={} checkpoint_every={} crossover_rolls={} mutate_rolls={} K={} pop={}",
                self.header.run_id,
                self.header.run_seed,
                self.config.checkpoint_every,
                self.config.crossover_rolls,
                self.config.mutate_rolls,
                SMOOTHING_WINDOW,
                self.config.pop_size,
            );
        }

        loop {
            let clock = self.step_clock();

            // ── 1. group step: every live net on the same shared batch ──────
            let hashes = self.state.live_hashes();
            if hashes.is_empty() {
                info!("race: population empty — stopping");
                return Ok(StopReason::MaxSteps);
            }
            for hash in &hashes {
                self.step_one_net(hash, clock)?;
            }
            self.append_metrics_csv(clock)?;
            // `None` mode: no per-step logging at all (log_step_rollup's
            // best-net eval still runs — it feeds snapshot bookkeeping).
            // `Minimal` is handled INSIDE log_step_rollup (table only).
            if self.log_level != crate::engine::config::LogLevel::None {
                self.log_step_rollup(clock);
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
                    info!(
                        "step {} │ checkpoint │ pop_mean_fitness {} {:.4} │ exam_mean_fitness {} {:.4} (ledger: {} entries)",
                        clock,
                        self.fitness.direction().arrow(),
                        mean,
                        self.fitness.direction().arrow(),
                        exam_mean,
                        self.checkpoints.len(),
                    );
                }
            }

            // ── 3. evolve — always, two independent roll groups ─────────────
            let mut cross_fired = 0usize;
            let mut cross_survived = 0usize;
            let mut cross_discarded = 0usize;
            let mut mutate_fired = 0usize;
            let culls_before = self.culls;
            // 3a. crossover rolls: checkpoint-gated children.
            for roll in 0..self.config.crossover_rolls {
                if self.state.live_count() < 2 {
                    break; // not enough population to evolve
                }
                if fastrand::f32() >= self.config.crossover_prob {
                    continue;
                }
                cross_fired += 1;
                // cx_retry_full: on a gate rejection, retry the FULL attempt
                // (fresh parents, fresh generate + gate replay) up to
                // `crossover_retries` extra times. Compute spent per attempt
                // is the price of a chance at a child that clears the bars.
                let max_attempts = 1 + self.config.crossover_retries;
                let mut attempt = 0usize;
                loop {
                    attempt += 1;
                    if self.evolve_crossover_child(clock, roll, attempt)? {
                        cross_survived += 1;
                        break;
                    }
                    if attempt >= max_attempts {
                        cross_discarded += 1;
                        break;
                    }
                    if self.verbose_detail() {
                        info!(
                            "step {} │ crossover roll {} retry {}/{} (gate rejected prior attempt)",
                            clock,
                            roll,
                            attempt,
                            max_attempts - 1,
                        );
                    }
                }
            }
            // 3b. mutation rolls: random immigrants (no gate).
            for roll in 0..self.config.mutate_rolls {
                if self.state.live_count() == 0 {
                    break;
                }
                if fastrand::f32() >= self.config.mutate_prob {
                    continue;
                }
                mutate_fired += 1;
                self.evolve_random_immigrant(clock, roll)?;
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
            // Evolve rollup — plain-language, no roll ordinals. "discarded"
            // = a crossover child was rejected by the checkpoint gate (the
            // population did NOT shrink; only the attempt was dropped). Rolls
            // that did NOT fire are also shown, so a quiet step is explained:
            // "fired 0/2» means the probability roll said no (normal), while
            // "fired 2 → 0 inserted" means the gate said no (selection).
            if self.verbose_detail() {
                let cross_total = self.config.crossover_rolls;
                let mutate_total = self.config.mutate_rolls;
                info!(
                    "step {} │ evolve │ crossover fired {}/{} → {} inserted, {} discarded by gate │ mutation fired {}/{} → {} immigrant(s) inserted (no gate) │ pop now {}",
                    clock,
                    cross_fired,
                    cross_total,
                    cross_survived,
                    cross_discarded,
                    mutate_fired,
                    mutate_total,
                    mutate_fired,
                    self.state.live_count(),
                );
                // Per-roll detail in Full mode: say WHY each roll stayed cold.
                if self.log_level == crate::engine::config::LogLevel::Full {
                    if cross_fired < cross_total {
                        info!(
                            "step {} │ evolve │ crossover: {} roll(s) did not fire (probability roll, p={:.2}){}",
                            clock,
                            cross_total - cross_fired,
                            self.config.crossover_prob,
                            if self.state.live_count() < 2 { " — also pop < 2 blocks crossover" } else { "" },
                        );
                    }
                    if mutate_fired < mutate_total {
                        info!(
                            "step {} │ evolve │ mutation: {} roll(s) did not fire (probability roll, p={:.2})",
                            clock,
                            mutate_total - mutate_fired,
                            self.config.mutate_prob,
                        );
                    }
                }
            }

            // ── 4. stop criteria ────────────────────────────────────────────
            if let Some(reason) = self.check_stop(clock) {
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
                return Ok(reason);
            }
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
            ranked.into_iter().take(k).map(|(h, _)| h).collect::<Vec<_>>()
        };
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
                keep.iter().map(|h| h[..8].to_string()).collect::<Vec<_>>().join(","),
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
            self.write_worst_artifacts()?;
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
        self.write_worst_artifacts()?;
        if self.verbose_detail() {
            info!(
                "── pruner phase complete ── trained to step {} ({} solo step(s))",
                final_step,
                pruner.steps,
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
        info!(
            "  {} live net(s) at step {} → resume with: --resume {}",
            live.len(),
            clock,
            self.run_dir.display(),
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
            let meta = self.state.net(h).map(|s| {
                format!(
                    "origin={} born@{} params={}",
                    s.created_from.clone().unwrap_or_else(|| "?".into()),
                    s.entered_at_step,
                    s.meta.params.map(|p| p.to_string()).unwrap_or_else(|| "?".into()),
                )
            });
            info!(
                "  #{} {} fitness{arrow} {:.4} │ eval_loss {:?} │ last_step {} │ {}",
                rank + 1,
                &h[..8.min(h.len())],
                fit,
                eval_loss,
                step,
                meta.unwrap_or_default(),
            );
        }
        Ok(())
    }

    /// Write the champion's (elite rank #1) topology markdown to the run dir.
    /// UNCONDITIONAL — called at stop regardless of log level: the `.md` file
    /// is a run artifact (like `nets/<hash>.json` and `history.csv`), not a
    /// log line. Prints a confirmation only when per-step detail is on.
    fn write_champion_markdown(&self) -> Result<()> {
        let ranked = self.rank_live();
        let Some((champ, _)) = ranked.first() else {
            return Ok(()); // nothing ever scored — no champion to dump
        };
        if !self.config.elite_save_topology {
            return Ok(());
        }
        let Some(state) = self.state.net(champ) else {
            return Ok(());
        };
        let Ok(topo) = state.topology() else {
            return Ok(());
        };
        let short = &champ[..8.min(champ.len())];
        let path = self.run_dir.join(format!("elite-{short}.md"));
        let md = crate::utils::markdown::topology_markdown(&topo, None);
        match std::fs::write(&path, &md) {
            Ok(()) => {
                if self.verbose_detail() {
                    info!("  champion topology → {} (markdown, ready to view)", path.display());
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
        Ok(())
    }

    /// Write the champion's (elite rank #1) weights as `.safetensors` to the
    /// run dir. UNCONDITIONAL — same artifact discipline as
    /// [`Self::write_champion_markdown`]: lands on disk at every log level.
    /// Exports the champion's live in-memory `Network` — byte-faithful
    /// coefficients, exactly what the race left it with.
    fn write_champion_safetensors(&self) -> Result<()> {
        let ranked = self.rank_live();
        let Some((champ, _)) = ranked.first() else {
            return Ok(()); // nothing ever scored — no champion to dump
        };
        if !self.config.elite_save_safetensors {
            return Ok(());
        }
        let short = &champ[..8.min(champ.len())];
        let path = self.run_dir.join(format!("elite-{short}.safetensors"));
        // The champion's live Network is still in memory at stop — export
        // it directly, no rebuild/replay needed. Its coefficients are the
        // exact ones the race left it with (byte-faithful by construction).
        match self.networks.get(champ.as_str()) {
            Some(net) => {
                if let Err(e) = crate::utils::safetensors::export_safetensors(net, &path) {
                    log::warn!("elite safetensors export failed: {e}");
                } else if self.verbose_detail() {
                    info!("  champion weights → {} (safetensors)", path.display());
                }
            }
            None => log::warn!("elite safetensors export failed: live network missing"),
        }
        Ok(())
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
    /// artifacts too — `worst-<hash>.md` (topology markdown) and/or
    /// `worst-<hash>.safetensors` (weights), each behind its own flag. Same
    /// artifact discipline as the elite dumps: unconditional on log level,
    /// called right after them at every stop path. The anti-champion is the
    /// net the search avoided — its topology is often the cheapest way to see
    /// what the fitness signal rejected.
    fn write_worst_artifacts(&self) -> Result<()> {
        if !self.config.worst_save_topology && !self.config.worst_save_safetensors {
            return Ok(());
        }
        let ranked = self.rank_live();
        let Some((worst, _)) = ranked.last() else {
            return Ok(()); // nothing ever scored — nothing to dump
        };
        let short = &worst[..8.min(worst.len())];
        if self.config.worst_save_topology {
            if let Some(state) = self.state.net(worst) {
                if let Ok(topo) = state.topology() {
                    let path = self.run_dir.join(format!("worst-{short}.md"));
                    let md = crate::utils::markdown::topology_markdown(&topo, None);
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
        // One step = whatever the caller's training scheme does for one step
        // clock. The engine only consumes the returned report.
        let optimizer = self.optimizers.get_mut(hash).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Optimizer in memory)"
            ))
        })?;
        // Build the context from engine-owned state only. The loss comes
        // from the trainer; to avoid borrowing `self.trainer` immutably here
        // (conflicts with the mutable `train_step` below), we pass a fresh
        // loss *option* resolved via a one-time split: trainer.loss() borrows
        // self.trainer, so instead we lend the loss through a scoped raw
        // pointer-free trick — call loss() first, keep the borrow in `ctx`,
        // and make train_step take `&mut self` *before* ctx borrow ends by
        // reborrowing through a local. Rust's NLL handles this because ctx's
        // loss lifetime is tied to the trainer, and train_step reborrows
        // through the same path — so we split the borrows explicitly:
        // compute the loss borrow BEFORE the mutable trainer borrow begins,
        // by taking trainer out of self for the duration.
        let run_data = crate::trainer::RunData {
            dataset: &self.dataset,
            stream: &self.stream,
        };
        let ctx = crate::trainer::StepContext {
            data: Some(&run_data),
            fitness: Some(&self.fitness),
            metrics: &self.metrics,
            env: crate::trainer::StepEnv {
                step: clock,
                run_seed: self.header.run_seed,
                pop_size: self.config.pop_size,
                live_count: self.state.live_count(),
                checkpoint_every: self.config.checkpoint_every,
                smoothing_window: SMOOTHING_WINDOW,
            },
            net_hash: hash,
            net_seed,
        };
        let net = self.networks.get_mut(hash).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Network in memory)"
            ))
        })?;
        let report = self
            .trainer
            .train_step(net, optimizer.as_mut(), clock, &ctx)?;
        // Debug-only contract probe: the Trainer's per-step clause says the
        // report describes the net AFTER this step's training. Re-score the
        // net on the step's eval batch and check the reported eval loss is
        // what this net actually scores — catches a stale/fabricated report
        // at development time. Zero cost in release.
        #[cfg(debug_assertions)]
        if let (Some(reported), Some(net)) = (report.eval_loss, self.networks.get_mut(hash)) {
            let eval_batch = self
                .stream
                .eval_batch(&self.dataset, clock as u64)
                .expect("probe: eval batch");
            if let (Some(loss), Some(fit)) = (self.trainer.loss(), Some(&self.fitness)) {
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

    /// One rollup line per step: pop size, train/eval loss means ± std, and
    /// the fitness column in the same shape. Two decimals throughout — the
    /// spread is the signal worth reading, the extremes are noise.
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
        // Per-step rollup: Summ = compact one-liner; Minimal = framed table
        // (from step 2 — deltas need a prior step); Full = one-liner + per-net
        // detail + checkpoint + divergence lines.
        if self.log_level == crate::engine::config::LogLevel::Minimal {
            self.log_minimal_table(&trains, &evals, &fits);
        } else {
            info!(
                "step {} │ pop {} │ train_loss↓ {} │ eval_loss↓ {} │ fitness{} {}",
                clock,
                self.state.live_count(),
                stats(&trains),
                stats(&evals),
                self.fitness.direction().arrow(),
                fit_stats,
            );
        }
        if self.log_level == crate::engine::config::LogLevel::Full {
            // Per-net detail for this step.
            for h in self.state.live_hashes() {
                if let Some(m) = self.state.net(&h).and_then(|s| s.last_metrics.as_ref()) {
                    log::info!(
                        "step {} │ net {} │ train_loss↓ {} │ eval_loss↓ {} │ fitness{} {}",
                        clock,
                        h,
                        fmt2(m.train_loss),
                        fmt_opt2(&m.eval_loss),
                        self.fitness.direction().arrow(),
                        fmt2(m.fitness),
                    );
                }
            }
            // Checkpoint ledger diagnostic.
            if !self.checkpoints.is_empty() {
                let last = &self.checkpoints[self.checkpoints.len() - 1];
                log::info!(
                    "step {} │ checkpoint │ step {} │ pop_mean_fitness {} {:.4}",
                    clock,
                    last.step,
                    self.fitness.direction().arrow(),
                    last.pop_mean_fitness,
                );
            }
        }
    }

    /// `Minimal` mode: one comfy framed table per step (starting at step 2 —
    /// step 1 has no prior step to diff against). Values "update in place":
    /// the shape is identical every step, only the numbers move. Deltas reuse
    /// what the engine already reports (population-mean smoothed values vs the
    /// last step); a zero delta drops the parenthetical. The evolve counters
    /// and the best-net footer are plain current values — no delta on the
    /// footer. Nothing else is printed per step in this mode.
    fn log_minimal_table(&mut self, trains: &[f32], evals: &[f32], fits: &[f32]) {
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
        let rows: Vec<String> = vec![
            format!(
                "pop {:>3} │ train_loss↓ {:.2}{} │ eval_loss↓ {:.2}{}",
                pop,
                train_m,
                delta(train_m, Some(pt)),
                eval_m,
                delta(eval_m, Some(pe)),
            ),
            format!(
                "fitness{} {:.2}{} │ culls {} │ inserts {}",
                self.fitness.direction().arrow(),
                fit_m,
                delta(fit_m, Some(pf)),
                e.culls,
                e.inserts,
            ),
            format!(
                "evolve │ crossover fired {} ({} inserted, {} discarded) │ mutation {}",
                e.cross_fired, e.cross_survived, e.cross_discarded, e.mutate_fired,
            ),
        ];
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
        self.rolling_fitness
            .insert(hash.to_string(), RollingBuffer::new(SMOOTHING_WINDOW));
        self.rolling_train
            .insert(hash.to_string(), RollingBuffer::new(SMOOTHING_WINDOW));
        self.rolling_eval
            .insert(hash.to_string(), RollingBuffer::new(SMOOTHING_WINDOW));
    }

    /// A guaranteed-fresh random child: bump the clock ordinal until the
    /// topology hash is not live (bounded retries; log when a dup is hit).
    fn generate_random_at(&mut self, clock: usize, start_idx: usize) -> Result<RaceChild> {
        for attempt in 0..8 {
            let child = self.generate_child(clock, start_idx + attempt)?;
            if !self.state.net(&child.state.hash).is_some() {
                // consume the ordinal(s) we used
                *self.children_born_at_clock.get_mut(&clock).unwrap() = start_idx + attempt + 1;
                return Ok(child);
            }
            if self.verbose_detail() {
                info!(
                    "step {} │ crossover child {} is a duplicate topology → discarded, random replacement",
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
    fn insert_child(&mut self, child: RaceChild, clock: usize) {
        let lineage = child.state.created_from.clone().unwrap_or("?".into());
        let child_fitness = child.state.last_metrics.as_ref().map(|m| m.fitness);
        self.state.insert(child.state.clone(), child_fitness);
        self.networks.insert(child.state.hash.clone(), child.net);
        self.optimizers
            .insert(child.state.hash.clone(), child.optimizer);
        if self.verbose_detail() {
            info!(
                "step {} │ inserted {} {} │ caught-up {} → json",
                clock, child.state.hash, lineage, clock,
            );
        }
    }

    /// One evolution roll of the crossover branch: generate a child, run the
    /// checkpoint-gated catch-up (early-out discard on the first failed
    /// gate), and on success cull the current worst net + insert the child.
    /// A failed child culls nothing — the gate IS the cull.
    /// Returns `true` when the child survived the gate and was inserted.
    /// `attempt` is the 1-based retry ordinal (for history.csv).
    fn evolve_crossover_child(&mut self, clock: usize, _roll: usize, attempt: usize) -> Result<bool> {
        let idx = self.next_child_ordinal(clock);
        let mut child = self.generate_child(clock, idx)?;
        if self.state.net(&child.state.hash).is_some() {
            if self.verbose_detail() {
                info!(
                    "step {} │ crossover child {} is a duplicate topology → discarded, random replacement",
                    clock,
                    &child.state.hash[..8],
                );
            }
            let idx = self.next_child_ordinal(clock);
            child = self.generate_random_at(clock, idx)?;
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
                for (i, chk) in &relevant {
                    self.catch_up_range(&mut child, replayed_to, chk.step)?;
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
                    self.catch_up_range(&mut child, replayed_to, last.step)?;
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
            if self.verbose_detail() {
                info!(
                    "step {} │ crossover child {} rejected by {} gate {}/{} ({}{:.4} vs bar {:.4}) → discarded at step {}, pop unchanged",
                    clock,
                    &child.state.hash[..8],
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
                );
            }
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
            return Ok(false);
        }
        let _ = last_checkpoint_step;
        let _ = child_fit_at_gate;
        // Passed every gate — finish the replay to the clock and insert.
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
        self.insert_child(child, clock);
        Ok(true)
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

    /// Uniformly random **cullable** live net — the `CrossCullPolicy::Random`
    /// victim for a crossover replacement. Deterministic: derived from
    /// `(run_seed, clock)` so replays and resume pick the same victim. Elite
    /// nets are excluded from the draw (a crossover child can never evict an
    /// elite, even under the Random policy).
    fn select_random_victim(&self, clock: usize) -> Result<String> {
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
        let seed = crate::utils::seed::derive_seed(self.header.run_seed, clock.wrapping_mul(7919));
        let idx = fastrand::Rng::with_seed(seed).usize(..hashes.len());
        Ok(hashes[idx].clone())
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
    fn evolve_random_immigrant(&mut self, clock: usize, roll: usize) -> Result<()> {
        // Pick a victim: fitness-inverse among NON-ELITE nets when possible
        // (the elite guard makes the top-k immune to mutation culls too),
        // worst-net when buffers are cold, first-live as last resort.
        let victim = self.select_inverse_proportional()?.unwrap_or_else(|| {
            // No fitness verdict available — fall back to worst-net, then
            // first-live as last resort.
            let worst = match self.worst_nets_by_smoothed_fitness(1) {
                Ok(mut w) => w.pop(),
                Err(_) => None,
            };
            worst.unwrap_or_else(|| self.state.live_hashes().first().unwrap().clone())
        });
        self.cull_net(&victim, clock, "immigrant-slot")?;
        let idx = self.next_child_ordinal(clock);
        let mut child = self.generate_random_at(clock, idx)?;
        self.pre_insert_buffer(&child.state.hash);
        self.catch_up(&mut child, clock)?;
        let _ = roll; // roll index kept for determinism only, not logged
        if self.verbose_detail() {
            info!(
                "step {} │ mutation │ inserted {} random immigrant (no gate) → caught up to step {}",
                clock,
                &child.state.hash[..8],
                clock,
            );
        }
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
        self.insert_child(child, clock);
        Ok(())
    }

    /// The next child ordinal at this clock (across all evolution branches).
    fn next_child_ordinal(&mut self, clock: usize) -> usize {
        let idx = *self.children_born_at_clock.entry(clock).or_insert(0);
        *self.children_born_at_clock.get_mut(&clock).unwrap() = idx + 1;
        idx
    }

    /// Cull one net: final state snapshot to disk, drop from all live maps.
    fn cull_net(&mut self, hash: &str, clock: usize, reason: &str) -> Result<()> {
        let dir = self.fitness.direction().arrow();
        let smoothed = self
            .rolling_fitness
            .get(hash)
            .map(rolling_mean)
            .unwrap_or(f32::NAN);
        if self.verbose_detail() {
            info!(
                "step {} │ culled {} smoothed{} {} ({}) → tombstone, pop now {}",
                clock,
                hash,
                dir,
                fmt2(smoothed),
                reason,
                self.state.live_count() - 1,
            );
        }
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

    /// Run the checkpoint "surprise exam": score every live net on the given
    /// era's gating-pool batch (rows never used for training or per-step
    /// eval). Returns the population's mean exam fitness — a generalization
    /// diagnostic recorded in the checkpoint ledger, never used for ranking,
    /// culling, or gating (so the replay contract is untouched). Nets are
    /// scored in eval mode (no gradients); a net whose eval-mode forward has
    /// side effects would violate the Trainer contract anyway.
    fn run_checkpoint_exam(&mut self, era: u64) -> Result<f32> {
        let exam_batch = self.stream.exam_batch(&self.dataset, era)?;
        let direction = self.fitness.direction();
        let mut scores: Vec<f32> = Vec::new();
        // Collect hashes first to avoid borrowing self.networks while calling
        // eval_one_step (which needs &mut Network).
        let hashes = self.state.live_hashes();
        for hash in &hashes {
            if let Some(net) = self.networks.get_mut(hash) {
                if let Some(loss) = self.trainer.loss() {
                    if let Ok(report) =
                        eval_one_step(net, loss, &self.fitness, &self.metrics, &exam_batch)
                    {
                        scores.push(report.fitness);
                    }
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
    /// nets most likely). Elite nets (top-`config.elite_count`) are excluded
    /// entirely — they can never be mutation victims. Returns `None` when no
    /// cullable candidate remains (empty/single-net pop, or all elite).
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
            elapsed_seconds: self.started_at_wall.elapsed().as_secs(),
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
            let mut headers = "type,step,hash,net_seed,origin,entered_at_step,train_loss,eval_loss,fitness".to_string();
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
    /// Today the header is written once at ``RaceEngine::new`` and the run config
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
            train_eval_split_ratio: Some(self.stream.train_eval_split_ratio()),
            held_out_eval_rows: Some(self.stream.held_out_eval_rows()),
            config: ConfigSnapshot::from_config(
                &self.config,
                self.stream.batch_size(),
                self.stream.eval_batch_size(),
            ),
            trainer: self.trainer.describe(),
        };
        let header = RunHeader::from_race_options(cfg);
        write_engine_json(&self.run_dir, &header)
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

    fn engine(run_dir: &std::path::Path, seed: u64) -> Result<RaceEngine> {
        // pop_size 0 ⇒ new() auto-seeds nothing; each test seeds its own
        // tiny topologies explicitly.
        let data_dir = tiny_dataset_dir("engine");
        let config = RaceConfig {
            pop_size: 0,
            ..RaceConfig::defaults()
        };
        RaceEngine::new(crate::engine::run_spec::RunSpec {
            data_dir,
            config,
            fitness: fitness(),
            trainer: crate::trainer::TabularTrainer::new(loss_fn()),
            seed: Some(seed),
            run_dir: Some(run_dir.to_path_buf()),
        })
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
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)], Some(0.5))
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
        let mut child = engine.generate_child(clock, 0).unwrap();
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
        let mut child_a = engine_a.generate_child(clock, 0).unwrap();
        let mut child_b = engine_b.generate_child(clock, 0).unwrap();
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

    // ── Iter-5: child generation contract ──────────────────────────────

    fn seed_two_parents(dir: &std::path::Path, seed: u64) -> RaceEngine {
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
        let ca = a.generate_child(3, 0).unwrap();
        let cb = b.generate_child(3, 0).unwrap();
        assert_eq!(
            ca.state.hash, cb.state.hash,
            "same seed + clock ⇒ same child topology"
        );
        assert_eq!(ca.state.net_seed, cb.state.net_seed);
        assert_eq!(ca.state.created_from, cb.state.created_from);
    }

    #[test]
    fn crossover_prob_zero_means_always_random_branch() {
        let dir = std::env::temp_dir().join("race_child_cx0");
        let mut engine = seed_two_parents(&dir, 7);
        engine.config.crossover_prob = 0.0;
        // One child per culled slot; every one must be lineage "random".
        for clock in 0..4 {
            let child = engine.generate_child(clock, 0).unwrap();
            let lineage = child.state.created_from.as_deref().unwrap();
            assert!(
                lineage == "random" || lineage == "random+mut",
                "crossover_prob=0 ⇒ random branch at clock {clock}, got {lineage}"
            );
        }
    }

    #[test]
    fn mutation_prob_zero_and_one_flips_mut_suffix() {
        let dir_no = std::env::temp_dir().join("race_child_mut0");
        let mut no = seed_two_parents(&dir_no, 21);
        no.config.crossover_prob = 0.0;
        no.config.mutate_prob = 0.0;
        let c = no.generate_child(1, 0).unwrap();
        assert_eq!(
            c.state.created_from.as_deref(),
            Some("random"),
            "no '+mut' suffix"
        );

        let dir_yes = std::env::temp_dir().join("race_child_mut1");
        let mut yes = seed_two_parents(&dir_yes, 21);
        yes.config.crossover_prob = 0.0;
        yes.config.mutate_prob = 1.0;
        let c = yes.generate_child(1, 0).unwrap();
        assert_eq!(
            c.state.created_from.as_deref(),
            Some("random+mut"),
            "mutate_prob=1 ⇒ '+mut' suffix"
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
    fn fallback_after_three_failed_crossovers_is_random_topology() {
        // Hidden-less parents make every crossover attempt a no-op
        // (cx_one_point/cx_uniform both bail with zero hidden nodes), so
        // after 3 attempts the engine must fall back to a fresh random
        // topology (lineage "random"), never stall — and never clone the
        // fittest (cloning would reinforce the leader and starve diversity).
        let dir = std::env::temp_dir().join("race_child_fallback");
        let mut engine = engine(&dir, 55).unwrap();
        engine
            .seed_population_internal(vec![flat_topology(7), flat_topology(8)], Some(0.5))
            .unwrap();
        engine.config.crossover_prob = 1.0;
        engine.config.mutate_prob = 0.0;
        let child = engine.generate_child(2, 0).unwrap();
        let lineage = child.state.created_from.as_deref().unwrap();
        assert!(
            lineage == "random",
            "3 failed crossovers ⇒ random topology fallback, got {lineage}"
        );
        // Child must still be trainable: dataset dims stamped, seed set.
        assert!(child.state.net_seed != 0);
    }

    // ── Iter-6 Tier B/C: resume replay + parity ──────────────────────

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
        let mut resumed = RaceEngine::resume(
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

        let mut resumed = RaceEngine::resume(
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
        engine.stream.set_checkpoint_every(engine.config.checkpoint_every);
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
                let mean = engine.population_mean_smoothed_fitness();
                let exam = engine.run_checkpoint_exam((clock / engine.config.checkpoint_every) as u64).unwrap();
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
        let resumed = RaceEngine::resume(
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
        assert!(!resumed.checkpoints[0].exam_mean_fitness.is_nan(), "exam reading persisted");
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
        let resumed = RaceEngine::resume(
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

    /// Seed every live net with a RAW last-step fitness of `v`.
    fn seed_raw_fitness(engine: &mut RaceEngine, v: f32) {
        for h in engine.state.live_hashes() {
            engine.state.record_step(
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
    fn stop_criteria_race_first_fire_wins() {
        // Two criteria set → both are live; whichever fires first ends the run.
        // max_steps fires at step 20 while target (2.0) is unreachable — the
        // step budget must win at its own step because it is checked first.
        let run_dir = std::env::temp_dir().join("gras-race-steps");
        let mut eng = engine(&run_dir, 5).unwrap();
        eng.config.max_steps = Some(20);
        eng.config.max_target_fitness = Some(-1.0); // unreachable under Minimize (fires when best < target)
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        seed_raw_fitness(&mut eng, 0.5);
        assert_eq!(eng.check_stop(19), None, "neither has fired yet");
        assert_eq!(eng.check_stop(20), Some(StopReason::MaxSteps));

        // Reverse the race: target fires first while the step budget is far.
        let run_dir = std::env::temp_dir().join("gras-race-target");
        let mut eng = engine(&run_dir, 5).unwrap();
        eng.config.max_steps = Some(1000);
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

    fn pruned_engine(run_dir: &std::path::Path, seed: u64, keep: usize, solo: usize) -> RaceEngine {
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
        RaceEngine::new(crate::engine::run_spec::RunSpec {
            data_dir,
            config,
            fitness: fitness(),
            trainer: crate::trainer::TabularTrainer::new(loss_fn()),
            seed: Some(seed),
            run_dir: Some(run_dir.to_path_buf()),
        })
        .unwrap()
    }

    #[test]
    fn pruner_disabled_stops_at_max_steps() {
        let run_dir = std::env::temp_dir().join("gras-pruner-off");
        let _ = std::fs::remove_dir_all(&run_dir);
        let mut eng = engine(&run_dir, 42).unwrap();
        eng.config.max_steps = Some(2);
        eng.config.pop_pruner = None;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)], Some(0.5))
            .unwrap();
        let reason = eng.run().unwrap();
        assert_eq!(reason, StopReason::MaxSteps);
        assert_eq!(eng.state.live_count(), 3, "pruner off: nobody is culled at stop");
    }

    #[test]
    fn pruner_hard_culls_to_elites_and_trains_solo_steps() {
        let run_dir = std::env::temp_dir().join("gras-pruner-hard");
        let _ = std::fs::remove_dir_all(&run_dir);
        let mut eng = pruned_engine(&run_dir, 42, 1, 3);
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)], Some(0.5))
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
        assert!(history.contains("pruner"), "culls are recorded as pruner attempt rows");
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
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8), tiny_topology(9), tiny_topology(10)], Some(0.5))
            .unwrap();
        eng.run().unwrap();
        assert_eq!(eng.state.live_count(), 2, "elite_count=2 keeps two nets racing");
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
        assert_eq!(ha.last_metrics.as_ref().map(|m| m.fitness), hb.last_metrics.as_ref().map(|m| m.fitness),
            "same seed ⇒ identical solo trajectory");
    }
}
