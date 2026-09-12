//! Step-race scheduler — Iter 4 of the continuous step-race revamp
//! (see `RACE_REVAMP.md`).
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
//! 3. After **all** nets stepped: if `step_clock >= grace_steps`, compute
//!    divergence over the population's smoothed eval fitness (rolling mean
//!    K=10, `(max-min)/max`, 0 if max==0). On breach: cull worst `cull_count`
//!    nets (drop their `Network` + `Optimizer` from memory + remove from
//!    `RaceState`) → generate `cull_count` children (stub: clone fittest
//!    parent) → catch each up solo through steps `0..step_clock` → insert
//!    into the live maps → log. **No grace restart** — grace is one-time at
//!    run start.
//! 4. Stop criteria checked (max_steps, wall-clock, max_culls, stagnation,
//!    target_score) — log which fired.
//! 5. Repeat — the loop's clock is the nets' recorded steps (every net is at
//!    the same step after the group steps, so reading any live net's step
//!    gives the clock).
//!
//! Determinism: the children's catch-up uses the same `seed_step_randomness` +
//! same step primitives as the group loop, so two identical-seed runs produce
//! identical cull/insertion sequences (the Iter-4 exit proof).
//!
//! Memory model: **all live nets + their optimizers live inside the loop** for
//! the whole run. Pop 5 → 5 networks + 5 optimizers in memory at once. On
//! cull, the culled net's entries are dropped (memory freed). On insert, the
//! child's entries are added. This is simple, not minimal-memory — fine for
//! the small default pop; the user runs bigger pops when it matters.
//!
//! Persistence (Iter 4 shape):
//! - `engine.json` — written once at run start (the run header; `RunHeader`).
//! - `nets/<hash>.json` — written **once** when a net leaves the live pop
//!   (cull write — its final state snapshot), and written **once** at run end
//!   for every still-live net (stop write — the live pop's final state).
//!   During the run, live nets are held only in memory. An interrupted run
//!   leaves `engine.json` + the culled nets' JSONs on disk; resume (Item 6)
//!   reconstructs the live frontier from those + replays each net's stream.
//!   There is no per-step file write during the run, and no separate
//!   `metrics.csv` or `culled.log` artifact (per D8 of RACE_REVAMP.md).

use flodl::nn::Optimizer;
use flodl::tensor::Result;
use log::info;
use std::collections::HashMap;

use crate::engine::fitness::{Fitness, FitnessLabel, Metric};
use crate::graph::network::Network;
use crate::state::{NetMetrics, NetState, RaceState, RunConfig, RunHeader, write_engine_json, write_net_state};
use crate::trainer::stream::{BatchStream, PoolSplit};
use crate::utils::race_steps::eval_one_step;

// ── Display helpers ─────────────────────────────────────────────────────────

/// Format a float to 4 decimals (all per-step metrics print through this).
fn fmt4(v: f32) -> String {
    format!("{v:.4}")
}

/// Format an optional float to 4 decimals — plain `—` when unset (never
/// `Some(...)` in user-facing logs).
fn fmt_opt4(v: &Option<f32>) -> String {
    v.as_ref().map(|x| format!("{x:.4}")).unwrap_or_else(|| "—".into())
}

use super::config::{RaceConfig, RaceSnapshot, StopReason};
use super::child::RaceChild;
use super::divergence::{DIVERGENCE_WINDOW, RollingBuffer, rolling_mean};

/// One entry in the checkpoint ledger: the population-mean smoothed fitness
/// recorded at a checkpoint step. A crossover child must BEAT this value at
/// every checkpoint it passes through during catch-up, or it is discarded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Checkpoint {
    pub step: usize,
    pub pop_mean_fitness: f32,
}

// ── RaceEngine ───────────────────────────────────────────────────────────────

/// The continuous step-race loop.
///
/// Owns the global step clock (implicit — the nets' recorded steps **are** the
/// clock), the shared batch stream, the in-memory live population (`RaceState`
/// with per-net `Network` + per-net `Optimizer`), and the per-net rolling
/// fitness buffers (for divergence smoothing).
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
    pub(crate) dataset: crate::utils::data::Dataset,
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
    /// the last K `NetMetrics` snapshots). Used only for divergence smoothing.
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
    /// Buffer metrics rows in memory to minimize slow disk I/O writes.
    pub(crate) metrics_csv_buffer: String,
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
    pub fn new(spec: crate::engine::run_spec::RunSpec) -> Result<Self> {
        let crate::engine::run_spec::RunSpec {
            data_dir,
            config,
            fitness,
            trainer,
            seed,
            run_dir,
        } = spec;
        let dataset = crate::utils::data::resolve_dataset(&data_dir)?;
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
        let run_dir = run_dir
            .unwrap_or_else(|| std::path::Path::new("results").join(&run_id));
        // Infer dims from data_dir when TopologyOptions leaves them unset.
        // input_dim/output_dim/hidden_dim are now Option<usize> — None means
        // "fill from dataset at run start", Some(v) means user set it (and
        // the engine validates v against the dataset below).
        let inferred_input_dim = dataset.inputs.shape()[1] as usize;
        let inferred_output_dim = dataset.targets.shape()[1] as usize;
        let inferred_hidden_dim = (dataset.inputs.shape()[1] as usize).max(8);
        let mut topology_options = config.topology_options.clone();
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
        if let Some(user_hidden) = topology_options.hidden_dim {
            // hidden_dim is a per-node internal dim — we don't veer the user's
            // choice, but warn if it's implausibly small relative to input dim.
            if user_hidden < inferred_input_dim && inferred_input_dim > user_hidden * 2 {
                topology_errors.push(format!(
                    "config.topology_options.hidden_dim = {user_hidden} is small relative to the data's input dim {inferred_input_dim} — consider >= {inferred_input_dim}"
                ));
            }
        } else {
            topology_options.hidden_dim = Some(inferred_hidden_dim);
        }
        if !topology_errors.is_empty() {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                topology_errors.join("; "),
            ).into());
        }
        let metrics = config.metrics.clone();
        let train_eval_split_ratio = 0.2f32;
        let held_out_eval_rows = 256usize;
        let default_batch_size = 16usize;

        let header = RunHeader::from_race_options_at(
            RunConfig {
                run_id: run_id.clone(),
                run_seed,
                fitness_label: FitnessLabel(fitness.label().to_string()),
                fitness_direction: fitness.direction(),
                input_dim: dataset.inputs.shape()[1] as usize,
                output_dim: dataset.targets.shape()[1] as usize,
                topology_options: topology_options,
                hidden_dim_pool: config.hidden_dim_pool.clone().unwrap_or(4..=8),
                hidden_dim_stride: config.hidden_dim_stride,
                combine_op_pool: config.combine_op_pool.clone(),
                activation_pool: config.activation_pool.clone(),
                standardize_op_pool: config.standardize_op_pool.clone(),
                informative_metrics: metrics.clone(),
                max_steps: config.max_steps,
                grace_steps: None,
                divergence_threshold: None,
                train_eval_split_ratio: Some(train_eval_split_ratio),
                held_out_eval_rows: Some(held_out_eval_rows),
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
        let (batch_size, eval_batch_size) = match trainer.stream_shape() {
            Some(shape) => (shape.batch_size, shape.eval_batch_size),
            None => (default_batch_size, default_batch_size),
        };
        let split = PoolSplit::of(&dataset, train_eval_split_ratio, run_seed);
        let batch_stream = BatchStream::new(run_seed, batch_size, split)
            .with_eval_batch_size(eval_batch_size)
            .with_held_out_eval_rows(held_out_eval_rows);

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
                // fill in the crossover_fallback_to_immigrant field with the
                // value from the user's config (default false)
                crossover_fallback_to_immigrant: config.crossover_fallback_to_immigrant,
                ..config
            },
            meta_ctx,
            stream: batch_stream,
            fitness,
            metrics,
            state: RaceState::new(),
            metrics_csv_buffer: String::new(),
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
        self.state.live_hashes().first().and_then(|h| {
            self.state.net(h).map(|s| s.step)
        }).unwrap_or(0)
    }

    /// The live population count.
    pub fn live_count(&self) -> usize {
        self.state.live_count()
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
    pub fn resume(
        run_dir: std::path::PathBuf,
        data_dir: std::path::PathBuf,
        config: RaceConfig,
        fitness: Fitness,
        trainer: Box<dyn crate::trainer::Trainer>,
    ) -> Result<Self> {
        let header = crate::state::load_engine_json(&run_dir)?;
        let dataset = crate::utils::data::resolve_dataset(&data_dir)?;
        let metrics = config.metrics.clone();
        let run_seed = header.run_seed;
            // On resume, stream shape comes from the run header (data integrity),
            // but the trainer's stream_shape() can still override batch sizes.
            let header_stream_ratio = header
                .train_eval_split_ratio
                .unwrap_or(0.2); // legacy default if header missing it
            let _header_held_out = header
                .held_out_eval_rows
                .unwrap_or(256); // legacy default if header missing it

        // Same rule on resume: the trainer's stream shape (same trainer the
        // run started with) overrides batch sizes; the split ratio comes from
        // the run header — data integrity is not re-decided on resume.
        let (batch_size, eval_batch_size) = match trainer.stream_shape() {
            Some(shape) => (shape.batch_size, shape.eval_batch_size),
            None => (16, 16), // conservative default when no stream_shape override
        };
        let split = PoolSplit::of(&dataset, header_stream_ratio, run_seed);
        let stream = BatchStream::new(run_seed, batch_size, split)
            .with_eval_batch_size(eval_batch_size);

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
            metrics_csv_buffer: String::new(),
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
        let entries = std::fs::read_dir(&nets_dir).map_err(|source| crate::utils::error::EngineError::Io {
            path: nets_dir.display().to_string(),
            source,
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
            let raw = std::fs::read_to_string(&path).map_err(|source| crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            })?;
            let state = NetState::from_json(&raw)?;
            if !state.is_alive {
                log::debug!("race resume: skipping tombstone net {} (culled)", state.hash);
                continue;
            }
            let hash = state.hash.clone();
            let step = state.step;
            let child = engine.replay_loaded_net(state)?;

            // Insert the replayed net into the live maps. Its rolling fitness
            // buffer is seeded from the replayed trajectory so divergence is
            // warm on resume (not cold).
            // Warm all three rolling buffers from the replayed last metrics
            // (resume resumes the smoothed stats, not just divergence's).
            let mut buf = RollingBuffer::new(DIVERGENCE_WINDOW);
            let mut train_buf = RollingBuffer::new(DIVERGENCE_WINDOW);
            let mut eval_buf = RollingBuffer::new(DIVERGENCE_WINDOW);
            if let Some(m) = engine.state.net(&hash).and_then(|s| s.last_metrics.as_ref()) {
                buf.push(m.fitness);
                train_buf.push(m.train_loss);
                if let Some(e) = m.eval_loss {
                    eval_buf.push(e);
                }
            }
            engine.state.insert(child.state.clone(), None);
            engine.networks.insert(child.state.hash.clone(), child.net);
            engine.optimizers.insert(child.state.hash.clone(), child.optimizer);
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
        for topo in topologies {
            let net = Network::build(&topo, self.config.device())?;
            let mut state = NetState::new(&topo, 0, None)?;
            state.stamp_meta(&net, &self.meta_ctx);
            let hash = state.hash.clone();
            let opt = self.trainer.make_optimizer(&net);
            self.networks.insert(hash.clone(), net);
            self.optimizers.insert(hash.clone(), opt);
            self.rolling_fitness.insert(
                hash.clone(),
                RollingBuffer::new(DIVERGENCE_WINDOW),
            );
            self.rolling_train.insert(hash.clone(), RollingBuffer::new(DIVERGENCE_WINDOW));
            self.rolling_eval.insert(hash.clone(), RollingBuffer::new(DIVERGENCE_WINDOW));
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
    /// is evaluated on the same held-out batch, records its metrics, and then
    /// (after grace) is subject to divergence-based culling.
    ///
    /// This is the step-race analog of the generational ``Engine::run``: same
    /// ``info``/``debug`` log discipline, same deterministic seed discipline,
    /// same ``no dead air`` rule from AGENTS.md §5 — only the cadence changes
    /// from per-generation to per-step and the selection/cull logic moves from
    /// the end of a generation to a live divergence check.
    ///
    /// Deterministic: two runs with the same `run_seed` + config produce the
    /// same cull/insertion sequence and the same per-step metrics (the Iter-4
    /// exit proof — tested in `tests::race_determinism`).
    pub fn run(&mut self) -> Result<StopReason> {
        self.apply_eval_defaults();
        self.write_options_csv()?;
        // `None` mode: one compact start line, then silence until stop.
        if self.verbose_detail() {
            info!(
                "race start: run={} seed={} checkpoint_every={} cross_rolls={} mutate_rolls={} K={} pop={}",
                self.header.run_id,
                self.header.run_seed,
                self.config.checkpoint_every,
                self.config.cross_rolls,
                self.config.mutate_rolls,
                DIVERGENCE_WINDOW,
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
                self.checkpoints.push(Checkpoint {
                    step: clock,
                    pop_mean_fitness: mean,
                });
                if let Err(e) = self.write_checkpoints() {
                    log::warn!("checkpoint ledger write failed: {e}");
                }
                if let Err(e) = self.write_live_frontier_states() {
                    log::warn!("checkpoint live states write failed: {e}");
                }
                if self.verbose_detail() {
                    info!(
                        "step {} │ checkpoint │ pop_mean_fitness {} {:.4} (ledger: {} entries)",
                        clock,
                        self.fitness.direction().arrow(),
                        mean,
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
            for roll in 0..self.config.cross_rolls {
                if self.state.live_count() < 2 {
                    break; // not enough population to evolve
                }
                if fastrand::f32() >= self.config.crossover_prob {
                    continue;
                }
                cross_fired += 1;
                if self.evolve_crossover_child(clock, roll)? {
                    cross_survived += 1;
                } else {
                    cross_discarded += 1;
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
            // population did NOT shrink; only the attempt was dropped).
            if self.verbose_detail() {
                info!(
                    "step {} │ evolve │ crossover fired {} → {} inserted, {} discarded by gate │ mutation → {} immigrant(s) │ pop now {}",
                    clock,
                    cross_fired,
                    cross_survived,
                    cross_discarded,
                    mutate_fired,
                    self.state.live_count(),
                );
            }

            // ── 4. stop criteria ────────────────────────────────────────────
            if let Some(reason) = self.check_stop(clock) {
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

    // ── One net's step ──────────────────────────────────────────────────────

    /// Stop-time summary: which JSONs were written as the live frontier and
    /// the resume command to continue from here.
    fn log_stop_summary(&self, clock: usize) -> Result<()> {
        let live = self.state.live_hashes();
        for h in &live {
            info!("  wrote nets/{h}.json (frontier snapshot)");
        }
        info!(
            "  {} live net(s) at step {} → resume with: --resume {}",
            live.len(),
            clock,
            self.run_dir.display(),
        );
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
        let optimizer = self
            .optimizers
            .get_mut(hash)
            .ok_or_else(|| crate::utils::error::EngineError::InvalidOptions(
                format!("race: net {hash} not live (no Optimizer in memory)"),
            ))?;
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
                divergence_window: DIVERGENCE_WINDOW,
            },
            net_hash: hash,
            net_seed,
        };
        let net = self
            .networks
            .get_mut(hash)
            .ok_or_else(|| crate::utils::error::EngineError::InvalidOptions(
                format!("race: net {hash} not live (no Network in memory)"),
            ))?;
        let report = self.trainer.train_step(net, optimizer.as_mut(), clock, &ctx)?;
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
            fmt4(metrics.train_loss),
            fmt_opt4(&metrics.eval_loss),
            dir,
            fmt4(metrics.fitness),
        );
        Ok(())
    }

    /// One rollup line per step (Plan A): pop size, train/eval loss ranges
    /// with means, and the current best net under the fitness direction.
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
        let stats = |v: &[f32]| {
            if v.is_empty() {
                "—".to_string()
            } else {
                let min = v.iter().cloned().fold(f32::MAX, f32::min);
                let max = v.iter().cloned().fold(f32::MIN, f32::max);
                let mean = v.iter().sum::<f32>() / v.len() as f32;
                format!("{:.4}…{:.4} (μ {:.4})", min, max, mean)
            }
        };
        // Fitness column: smoothed-fitness spread across the population,
        // same shape as the train/eval columns (min…max (μ mean)) so all
        // three read alike.
        let fits: Vec<f32> = self
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| self.rolling_fitness.get(h.as_str()).filter(|b| !b.is_empty()))
            .map(rolling_mean)
            .collect();
        let fit_stats = if fits.is_empty() {
            "—".to_string()
        } else {
            let direction = self.fitness.direction();
            let min = fits.iter().cloned().fold(f32::MAX, f32::min);
            let max = fits.iter().cloned().fold(f32::MIN, f32::max);
            let mean = fits.iter().sum::<f32>() / fits.len() as f32;
            let (bad, good) = match direction {
                crate::engine::fitness::Direction::Maximize => (min, max),
                crate::engine::fitness::Direction::Minimize => (max, min),
            };
            format!("{bad:.4}…{good:.4} (μ {mean:.4})")
        };
        // Per-step rollup: Summ = compact one-liner; Minimal = framed table
        // (from step 2 — deltas need a prior step); Full = one-liner + per-net
        // detail + checkpoint + divergence lines.
        if self.log_level == crate::engine::config::LogLevel::Minimal {
            self.log_minimal_table(clock, &trains, &evals, &fits);
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
                        clock, h, fmt4(m.train_loss),
                        fmt_opt4(&m.eval_loss),
                        self.fitness.direction().arrow(),
                        fmt4(m.fitness),
                    );
                }
            }
            // Checkpoint ledger + divergence diagnostic.
            if !self.checkpoints.is_empty() {
                let last = &self.checkpoints[self.checkpoints.len() - 1];
                log::info!(
                    "step {} │ checkpoint │ step {} │ pop_mean_fitness {} {:.4}",
                    clock, last.step, self.fitness.direction().arrow(), last.pop_mean_fitness,
                );
            }
            let div = self.divergence();
            log::info!(
                "step {} │ divergence │ K={} │ {:.4}",
                clock, DIVERGENCE_WINDOW, div,
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
    fn log_minimal_table(&mut self, clock: usize, trains: &[f32], evals: &[f32], fits: &[f32]) {
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
        let delta = |cur: f32, prev: Option<f32>| match prev {
            Some(p) if (cur - p).abs() >= 1e-6 => format!(" (∆{:+.4})", cur - p),
            _ => String::new(),
        };
        let (pt, pe, pf) = self.minimal_prev_means.unwrap_or((f32::NAN, f32::NAN, f32::NAN));
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
                "pop {:>3} │ train_loss↓ {:.4}{} │ eval_loss↓ {:.4}{}",
                pop, train_m, delta(train_m, Some(pt)), eval_m, delta(eval_m, Some(pe)),
            ),
            format!(
                "fitness{} {:.4}{} │ culls {} │ inserts {}",
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
        let inner = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0).max(20);
        let pad = |s: &String| {
            let visible = s.chars().count();
            format!("│ {}{}│", s, " ".repeat(inner.saturating_sub(visible)))
        };
        let line = |s: String| {
            let visible = s.chars().count();
            let dashes = "-".repeat(inner.saturating_sub(visible) + 2);
            println!("\x1b[2K\r{}{}{}", s, dashes, s.chars().last().unwrap_or('-'));
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

    // ── Divergence ──────────────────────────────────────────────────────────

    /// Compute population divergence at the current step (D3).
    ///
    /// Each net's smoothed fitness = rolling mean of its last K per-step
    /// fitness values (from the in-memory `rolling_fitness` buffer, kept in
    /// sync with `record_step` calls). Divergence = `(max - min) / max` over
    /// the live population's smoothed fitness; 0 if max is 0 (avoids
    /// div-by-zero early in a run).
    /// Diagnostic only since the checkpoint-gate model: the loop no longer
    /// acts on divergence, but the metric is still informative.
    pub fn divergence(&self) -> f32 {
        super::divergence::built_in_divergence(&self.smoothed_fitness_values())
    }

    /// Per-net rolling-mean fitness over the live population, in
    /// `live_hashes()` order — the shared input for the divergence diagnostic,
    /// the checkpoint ledger, and `RaceSnapshot`.
    fn smoothed_fitness_values(&self) -> Vec<f32> {
        // Only nets with a non-empty rolling buffer earn a divergence verdict.
        // A freshly caught-up child's buffer is populated during catch-up (replayed
        // fitness for steps 0..clock), so it IS included in the post-insert
        // re-check — the new child enters the divergence population immediately
        // after insertion, not after a group step. Nets with empty buffers
        // (shouldn't exist in normal operation) are excluded to avoid phantom
        // 0.0 values pinning divergence at 1.0.
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

    // ── Breach handling ─────────────────────────────────────────────────────

    /// On a divergence breach: cull the worst `cull_count` nets (drop their
    /// `Network` + `Optimizer` from memory + remove from `RaceState`), generate
    /// `cull_count` children (stub: clone fittest parent each), catch each up
    /// solo through steps `0..step_clock`, insert them into the live maps, log.
    ///
    /// This is the step-race analog of the generational ``next_generation``
    /// path (select → crossover → dedup → mutate → refill). The differences:
    /// selection is divergence-based rather than fitness-ranked, only one child
    /// type is produced per breach here (stub), and the child is caught up
    /// before rejoining instead of being dropped into the next generation.
    ///
    /// Generates **one child per culled slot** so the population count stays at
    /// `pop_size` across the run (the user's "pop size should be the same along
    /// all experiment run" rule). With the default `cull_count == 1` this is a
    /// non-issue; the general case keeps pop constant.
    ///
    /// Each culled net's state is written to `nets/<hash>.json` **before** it
    /// is dropped (cull write — final snapshot). Each inserted child is caught
    /// up solo and its state is written **once** at the end of its catch-up
    /// (see `catch_up`'s doc on the asymmetry with the group loop).
    /// Fresh empty rolling buffer for a net about to be caught up (so
    /// catch-up's per-step pushes land somewhere — a live net must never be
    /// without a buffer entry).
    fn pre_insert_buffer(&mut self, hash: &str) {
        self.rolling_fitness
            .insert(hash.to_string(), RollingBuffer::new(DIVERGENCE_WINDOW));
        self.rolling_train
            .insert(hash.to_string(), RollingBuffer::new(DIVERGENCE_WINDOW));
        self.rolling_eval
            .insert(hash.to_string(), RollingBuffer::new(DIVERGENCE_WINDOW));
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
                "step {} │ inserted {} {} │ caught-up {} (parity ok) → json",
                clock, child.state.hash, lineage, clock,
            );
        }
    }

    /// One evolution roll of the crossover branch: generate a child, run the
    /// checkpoint-gated catch-up (early-out discard on the first failed
    /// gate), and on success cull the current worst net + insert the child.
    /// A failed child culls nothing — the gate IS the cull.
    /// Returns `true` when the child survived the gate and was inserted.
    fn evolve_crossover_child(&mut self, clock: usize, _roll: usize) -> Result<bool> {
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
        match self.config.check {
            crate::engine::config::CheckMode::Hard => {
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
            crate::engine::config::CheckMode::Soft => {
                // One aggregate bar: beat the mean of the checkpoint means.
                // Replay straight to the last relevant checkpoint, compare once.
                if let Some((_, last)) = relevant.last() {
                    self.catch_up_range(&mut child, replayed_to, last.step)?;
                    replayed_to = last.step;
                    last_checkpoint_step = last.step;
                    let mean_of_means =
                        relevant.iter().map(|(_, c)| c.pop_mean_fitness).sum::<f32>()
                            / checkpoint_count as f32;
                    let child_fit = self
                        .rolling_fitness
                        .get(&child.state.hash)
                        .map(rolling_mean)
                        .unwrap_or(f32::NAN);
                    child_fit_at_gate = child_fit;
                    let beat = self
                        .fitness
                        .direction()
                        .is_better(child_fit, mean_of_means);
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
                    match self.config.check {
                        crate::engine::config::CheckMode::Hard => "hard",
                        crate::engine::config::CheckMode::Soft => "soft",
                    },
                    gate_i,
                    checkpoint_count,
                    self.fitness.direction().arrow(),
                    child_fit,
                    bar,
                    gate_step,
                );
            }
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
        self.cull_worst(clock, "crossover")?;
        self.insert_child(child, clock);
        Ok(true)
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
        // Pick a victim: fitness-inverse when possible, worst-net when
        // buffers are cold, first-live as last resort.
        let victim = self.select_inverse_proportional()?
            .unwrap_or_else(|| {
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
                "step {} │ inserted {} immigrant (no gate) → caught up to step {}",
                clock,
                &child.state.hash[..8],
                clock,
            );
        }
        self.insert_child(child, clock);
        Ok(())
    }

    /// The next child ordinal at this clock (across all evolution branches).
    fn next_child_ordinal(&mut self, clock: usize) -> usize {
        let idx = *self.children_born_at_clock.entry(clock).or_insert(0);
        *self.children_born_at_clock.get_mut(&clock).unwrap() = idx + 1;
        idx
    }

    /// Cull the current worst net (smoothed fitness) — the slot a surviving
    /// crossover child takes.
    fn cull_worst(&mut self, clock: usize, reason: &str) -> Result<()> {
        let worst = self.worst_nets_by_smoothed_fitness(1)?;
        for hash in worst {
            self.cull_net(&hash, clock, reason)?;
        }
        Ok(())
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
                clock, hash, dir, fmt4(smoothed), reason, self.state.live_count() - 1,
            );
        }
        if let Some(mut state) = self.state.net(hash).cloned() {
            state.is_alive = false;
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

    /// Select a live net inversely-proportionate to smoothed fitness (worst
    /// nets most likely). Returns `None` for an empty or single-net pop
    /// (nothing to sacrifice).
    fn select_inverse_proportional(&self) -> Result<Option<String>> {
        let hashes = self.state.live_hashes();
        if hashes.len() < 2 {
            return Ok(None);
        }
        let direction = self.fitness.direction();
        // Inverse fitness: worst nets get the biggest weight. Shift by the
        // population min so weights are non-negative and the worst net has
        // the largest share.
        let scored: Vec<(String, f32)> = hashes
            .iter()
            .filter(|h| {
                self.rolling_fitness
                    .get(*h)
                    .map(|b| b.iter().count() > 0)
                    .unwrap_or(false)
            })
            .map(|h| (h.clone(), rolling_mean(self.rolling_fitness.get(h).unwrap())))
            .collect();
        if scored.len() < 2 {
            return Ok(None);
        }
        let worst = scored
            .iter()
            .min_by(|a, b| direction.cmp(a.1, b.1))
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        // weight = (direction-adjusted worst) - (direction-adjusted value):
        // for Maximize, worst is the min ⇒ weight = worst − value ≥ 0, worst
        // net gets weight 0... wait, inverted: we want the WORST net to have
        // the LARGEST weight, so weight = value − worst in adjusted space.
        let adjusted = |v: f32| match direction {
            crate::engine::fitness::Direction::Maximize => v,
            crate::engine::fitness::Direction::Minimize => -v,
        };
        let adj_worst = adjusted(worst);
        let weights: Vec<(String, f32)> = scored
            .into_iter()
            .map(|(h, v)| (h, adjusted(v) - adj_worst))
            .collect();
        let total: f32 = weights.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            // All-equal population: uniform draw.
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

    // ── Selection helpers ───────────────────────────────────────────────────

    /// The live net hash with the best smoothed fitness (highest for Maximize,
    /// lowest for Minimize). Test-only for now: the random-topology fallback
    /// replaced the clone-fittest path, and culling uses `worst_nets`. Keep
    /// for the determinism tests (and any future elite-preservation policy).
    #[cfg(test)]
    pub(crate) fn fittest_net_hash(&self) -> std::result::Result<String, crate::utils::error::EngineError> {
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
            .map(|h| (h.clone(), rolling_mean(self.rolling_fitness.get(h).unwrap())))
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
    fn check_stop(&mut self, step: usize) -> Option<StopReason> {
        // max_steps first (explicit budget).
        if let Some(max) = self.config.max_steps {
            if step >= max {
                return Some(StopReason::MaxSteps);
            }
        }
        // wall-clock.
        if let Some(seconds) = self.config.wall_clock_seconds {
            if self.started_at_wall.elapsed().as_secs() >= seconds {
                return Some(StopReason::WallClock);
            }
        }
        // max_culls.
        if let Some(max) = self.config.max_culls {
            if self.culls >= max {
                return Some(StopReason::MaxCulls);
            }
        }
        // target_score.
        if let Some(target) = self.config.target_score {
            let best = self.best_smoothed_fitness();
            if self.fitness.direction().is_better(best, target) {
                return Some(StopReason::TargetScore);
            }
        }
        // Custom stop (Iter 5 pluggable contract): consulted **in addition**
        // to the built-ins, always last, so a user policy can stop the run on
        // criteria the built-ins don't model.
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
                    m.insert("pop_mean_fitness".into(), serde_json::Value::from(c.pop_mean_fitness));
                    m
                })
            })
            .collect();
        let raw = serde_json::to_string_pretty(&v)
            .map_err(|e| crate::utils::error::EngineError::Json(format!("checkpoints serialize: {e}")))?;
        std::fs::write(&path, raw).map_err(|source| {
            crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            }
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

    /// Write the options/configuration CSV once at startup.
    fn write_options_csv(&self) -> Result<()> {
        if !self.config.csv_export {
            return Ok(());
        }
        let path = self.run_dir.join("options.csv");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| crate::utils::error::EngineError::Io {
                path: parent.display().to_string(),
                source,
            })?;
        }
        let headers = "run_id,run_seed,pop_size,checkpoint_every,crossover_prob,mutate_prob,crossover_parents,mode,crossover_fallback_to_immigrant\n";
        let mode_str = format!("{:?}", self.config.mode).to_lowercase();
        let row = format!(
            "{},{},{},{},{},{},{},{},{}\n",
            self.header.run_id,
            self.header.run_seed,
            self.config.pop_size,
            self.config.checkpoint_every,
            self.config.crossover_prob,
            self.config.mutate_prob,
            self.config.crossover_parents,
            mode_str,
            self.config.crossover_fallback_to_immigrant
        );
        let mut content = String::with_capacity(headers.len() + row.len());
        content.push_str(headers);
        content.push_str(&row);
        std::fs::write(&path, content).map_err(|source| {
            crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            }
        })?;
        Ok(())
    }

    /// Append a step's metrics for all live nets to our in-memory CSV buffer.
    fn append_metrics_csv(&mut self, step: usize) -> Result<()> {
        if !self.config.csv_export {
            return Ok(());
        }
        for hash in &self.state.live_hashes() {
            if let Some(net_state) = self.state.net(hash) {
                if let Some(m) = &net_state.last_metrics {
                    let mut row = format!(
                        "{},{},{},{},{}",
                        step,
                        hash,
                        m.train_loss,
                        m.eval_loss.map(|val| val.to_string()).unwrap_or_else(|| "nan".to_string()),
                        m.fitness
                    );
                    for &val in &m.informative {
                        row.push(',');
                        row.push_str(&val.to_string());
                    }
                    row.push('\n');
                    self.metrics_csv_buffer.push_str(&row);
                }
            }
        }
        Ok(())
    }

    /// Flush the buffered metrics CSV rows from memory to disk.
    fn flush_metrics_csv(&mut self) -> Result<()> {
        if !self.config.csv_export || self.metrics_csv_buffer.is_empty() {
            return Ok(());
        }
        let path = self.run_dir.join("metrics.csv");
        let exists = path.exists();
        
        use std::fs::OpenOptions;
        use std::io::Write;
        
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| crate::utils::error::EngineError::Io {
                path: parent.display().to_string(),
                source,
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
            let mut headers = "step,hash,train_loss,eval_loss,fitness".to_string();
            for m in &self.metrics {
                headers.push(',');
                headers.push_str(m.label());
            }
            headers.push('\n');
            file.write_all(headers.as_bytes()).map_err(|source| crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            })?;
        }
        
        file.write_all(self.metrics_csv_buffer.as_bytes()).map_err(|source| crate::utils::error::EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;
        
        self.metrics_csv_buffer.clear();
        Ok(())
    }

    /// Load a checkpoint ledger written by [`Self::write_checkpoints`]
    /// (resume path). Missing file = fresh run, empty ledger.
    pub(crate) fn load_checkpoints(&mut self) -> Result<()> {
        let path = self.run_dir.join("checkpoints.json");
        if !path.exists() {
            return Ok(());
        }
        let raw = std::fs::read_to_string(&path).map_err(|source| crate::utils::error::EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| crate::utils::error::EngineError::Json(format!("checkpoints parse: {e}")))?;
        let mut checkpoints = Vec::new();
        if let Some(arr) = v.as_array() {
            for entry in arr {
                let step = entry.get("step").and_then(|f| f.as_u64()).unwrap_or(0) as usize;
                let pop_mean_fitness =
                    entry.get("pop_mean_fitness").and_then(|f| f.as_f64()).unwrap_or(0.0) as f32;
                checkpoints.push(Checkpoint { step, pop_mean_fitness });
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
    pub fn refresh_header(&mut self) -> Result<()> {        let cfg = RunConfig {
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
            max_steps: self.config.max_steps,                grace_steps: None,
                divergence_threshold: None,
            train_eval_split_ratio: Some(self.stream.train_eval_split_ratio()),
            held_out_eval_rows: Some(self.stream.held_out_eval_rows()),
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
    use crate::utils::data::synthetic_classification;
    use flodl::{Device, Variable};

    use crate::engine::config::DEFAULT_CHECKPOINT_EVERY;

    fn tiny_dataset() -> crate::utils::data::Dataset {
        synthetic_classification(64, 2, 2, 7, Device::CPU).unwrap()
    }

    fn tiny_topology(seed: usize) -> Topology {
        let mut topo = Topology::new(seed, Some(TopologyOptions {
            topology_seed: seed,
            min_hidden_num_nodes: 1,
            max_hidden_num_nodes: 1,
            min_hidden_inputs_per_node: 1,
            max_hidden_inputs_per_node: 1,
            min_hidden_outputs_per_node: 1,
            max_hidden_outputs_per_node: 1,
            input_dim: Some(2),
            hidden_dim: Some(4),
            output_dim: Some(2),
            dropout_prob: 0.0,
        }));
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
        crate::utils::data::save_dataset(&dir, &tiny_dataset()).unwrap();
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
            trainer: Box::new(crate::trainer::TabularTrainer::new(loss_fn())),
            seed: Some(seed),
            run_dir: Some(run_dir.to_path_buf()),
        })
    }

    #[test]
    fn race_config_defaults_are_conservative() {
        let cfg = RaceConfig::defaults();
        assert_eq!(cfg.pop_size, 5);
        assert_eq!(cfg.checkpoint_every, DEFAULT_CHECKPOINT_EVERY);
        assert_eq!(cfg.cross_rolls, 1);
        assert_eq!(cfg.mutate_rolls, 1);
        assert_eq!(cfg.check, crate::engine::config::CheckMode::Hard);
        assert_eq!(cfg.checkpoint_every, DEFAULT_CHECKPOINT_EVERY);
        // batch_size and held_out_eval_rows now live on RunSpec::stream
        // (engine infrastructure), not on RaceConfig — verified by the
        // stream_shape / stream_info contract instead.
        assert_eq!(cfg.max_steps, None, "budgets inactive by default");
        assert!(cfg.wall_clock_seconds.is_none());
        assert!(cfg.max_culls.is_none());
        assert!(cfg.target_score.is_none());
        assert!(cfg.hidden_dim_pool.is_some());
        assert_eq!(cfg.pop_size, 5);
    }

    #[test]
    fn divergence_is_zero_when_empty() {
        let dir = std::env::temp_dir().join("race_div_empty_test");
        let engine = engine(&dir, 42).unwrap();
        assert_eq!(engine.divergence(), 0.0);
    }

    #[test]
    fn divergence_is_zero_for_single_net() {
        let dir = std::env::temp_dir().join("race_div_one_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine.seed_population_internal(vec![tiny_topology(7)], Some(0.5)).unwrap();
        assert_eq!(engine.divergence(), 0.0, "one net ⇒ no spread");
    }

    #[test]
    fn divergence_is_zero_when_all_smoothed_fitness_equal() {
        let dir = std::env::temp_dir().join("race_div_equal_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5)).unwrap();
        // Manually fill both rolling buffers with the same fitness to simulate
        // a spread-less population.
        if let Some(buf) = engine.rolling_fitness.get_mut(&engine.state.live_hashes()[0]) {
            buf.push(0.5);
            buf.push(0.5);
        }
        if let Some(buf) = engine.rolling_fitness.get_mut(&engine.state.live_hashes()[1]) {
            buf.push(0.5);
            buf.push(0.5);
        }
        assert_eq!(engine.divergence(), 0.0, "equal smoothed fitness ⇒ no spread");
    }

    #[test]
    fn divergence_spikes_with_spread() {
        let dir = std::env::temp_dir().join("race_div_spread_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5)).unwrap();
        // Fill buffers with different fitnesses to simulate a spread.
        if let Some(buf) = engine.rolling_fitness.get_mut(&engine.state.live_hashes()[0]) {
            buf.push(0.9);
        }
        if let Some(buf) = engine.rolling_fitness.get_mut(&engine.state.live_hashes()[1]) {
            buf.push(0.1);
        }
        let div = engine.divergence();
        assert!(div > 0.0, "different smoothed fitness ⇒ spread ({div}))");
        // /max form: (0.9-0.1)/0.9 ≈ 0.889, bounded ≤ 1.
        assert!((div - 0.8888).abs() < 1e-4, "expected (0.9-0.1)/0.9 ≈ 0.889, got {div}");
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
        engine.seed_population_internal(vec![tiny_topology(7)], None).unwrap();
        // After populate, all nets are at step 0.
        assert_eq!(engine.step_clock(), 0);
    }

    #[test]
    fn fittest_net_is_deterministic_given_same_state() {
        let dir = std::env::temp_dir().join("race_fittest_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5)).unwrap();
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
        engine.seed_population_internal(vec![tiny_topology(7)], Some(0.0)).unwrap();
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
        engine_a.seed_population_internal(vec![tiny_topology(7)], Some(0.0)).unwrap();
        engine_b.seed_population_internal(vec![tiny_topology(7)], Some(0.0)).unwrap();
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
        engine.seed_population_internal(vec![tiny_topology(7)], None).unwrap();
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
        engine.seed_population_internal(vec![tiny_topology(7)], None).unwrap();
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
        engine.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5)).unwrap();
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
        assert_eq!(ca.state.hash, cb.state.hash, "same seed + clock ⇒ same child topology");
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
        assert_eq!(c.state.created_from.as_deref(), Some("random"), "no '+mut' suffix");

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
                hidden_dim: Some(4),
                output_dim: Some(2),
                dropout_prob: 0.0,
            }),
        );
        topo.nodes.push(Node::new_input(0, 2));
        topo.nodes.push(Node::new_output(1, 2, 2));
        let conn = |from: (usize, usize), to: (usize, usize)| crate::graph::topology::Connection {
            from: crate::graph::topology::Port { node: from.0, index: from.1 },
            to: crate::graph::topology::Port { node: to.0, index: to.1 },
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
        engine.seed_population_internal(vec![flat_topology(7), flat_topology(8)], Some(0.5)).unwrap();
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
        engine.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5)).unwrap();
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
            Box::new(crate::trainer::TabularTrainer::new(loss_fn())),
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
        full.seed_population_internal(vec![tiny_topology(7)], Some(0.5)).unwrap();
        let h_full = full.state.live_hashes()[0].clone();
        for clock in 0..5 {
            full.step_one_net(&h_full, clock).unwrap();
        }
        let final_metrics_full = full.state.net(&h_full).unwrap().last_metrics.clone();

        // Interrupted twin: 3 steps, persist, drop, resume, 2 more.
        let dir_b = std::env::temp_dir().join("race_twin_interrupted");
        let _ = std::fs::remove_dir_all(&dir_b);
        let mut part = engine(&dir_b, 123).unwrap();
        part.seed_population_internal(vec![tiny_topology(7)], Some(0.5)).unwrap();
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
            Box::new(crate::trainer::TabularTrainer::new(loss_fn())),
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
            .seed_population_internal(
                (0..4).map(tiny_topology).collect(),
                Some(0.5),
            )
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
        engine.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5)).unwrap();

        // Step 1 & 2 to create a checkpoint at step 2
        let hashes = engine.state.live_hashes();
        for clock in 0..3 {
            for h in &hashes {
                engine.step_one_net(h, clock).unwrap();
            }
            // Trigger checkpoint write in engine.run() equivalent
            if clock > 0 && clock % engine.config.checkpoint_every == 0 {
                let mean = engine.population_mean_smoothed_fitness();
                engine.checkpoints.push(Checkpoint {
                    step: clock,
                    pop_mean_fitness: mean,
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
            Box::new(crate::trainer::TabularTrainer::new(loss_fn())),
        )
        .unwrap();

        assert_eq!(resumed.checkpoints.len(), 1, "ledger reloaded");
        assert_eq!(resumed.checkpoints[0].step, 2);
        assert_eq!(resumed.checkpoints[0].pop_mean_fitness, expected_mean);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_validates_pop_size_mismatch() {
        let dir = std::env::temp_dir().join("race_resume_pop_validation");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5)).unwrap();
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
            Box::new(crate::trainer::TabularTrainer::new(loss_fn())),
        );
        assert!(resumed.is_err());
        let err_msg = resumed.err().unwrap().to_string();
        assert!(err_msg.contains("resume: expected 3 live nets"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
