//! Step-race run state — replaces the generational `gen_XX.json` + per-gen
//! `robustness.csv` artifact shape with a present-tense model:
//!
//! - `engine.json` — the run header, overwritten when reproducible config
//!   changes. Still named `engine.json` so it reads as "the engine run's data."
//! - `nets/<hash>.json` — one per live net, overwritten each step the net
//!   survives. Contains everything needed to hand that net back to the stream
//!   later (topology JSON, weight-init seed, current step, lineage placeholder).
//!
//! No per-gen snapshots, no default `history.csv` (each net state file carries
//! its own metrics snapshot), no default `culled.log` (a net that stops being
//! stepped simply stops being updated; an explicit tombstone log can be added
//! later if an audit trail is wanted).

use std::collections::HashMap;
use std::path::Path;

use flodl::tensor::Result;
use serde::{Deserialize, Serialize};

use crate::engine::config::{RaceConfig, SMOOTHING_WINDOW};
use crate::engine::fitness::{Direction, FitnessLabel, Metric};
use crate::graph::topology::{Topology, TopologyOptions};
use crate::utils::error::EngineError;
use crate::utils::seed::topo_hash;

// ── Run header (engine.json) ───────────────────────────────────────────────

/// The run's **complete** configuration, serialized into `engine.json` under
/// `"config"`.
///
/// `engine.json` is the single source of truth for a run's settings: every knob
/// that can change the trajectory is recorded here, so the file alone is enough
/// to reproduce the run — no parallel `options.csv` is written. To add a knob,
/// add it to this struct and to [`ConfigSnapshot::from_config`].
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ConfigSnapshot {
    /// Population size the race held constant.
    pub pop_size: usize,
    pub checkpoint_every: usize,
    pub crossover_rolls: usize,
    pub mutate_rolls: usize,
    /// Checkpoint gate strictness (`"hard"` | `"soft"`).
    pub crossover_gating: String,
    /// Extra gate-aware retries per crossover roll (0 = none).
    pub crossover_retries: usize,
    /// Crossover replacement policy (`"worst"` | `"random"`). Crossover-only:
    /// mutation immigrants always evict via fitness-inverse roulette.
    pub crossover_cull_policy: String,
    /// Elite guard size: top-k nets by smoothed fitness immune to all culls.
    pub elite_count: usize,
    /// Crossover operator pool (`"one_point"`/`"uniform"`); empty ⇒ both.
    pub crossover_ops_pool: Vec<String>,
    pub crossover_prob: f32,
    pub mutate_prob: f32,
    /// Stop budgets as configured (`None` = no limit).
    pub max_target_fitness: Option<f32>,
    /// Whether a pluggable `custom_stop` closure was installed. The closure is
    /// code, not data, so only its presence is recordable.
    pub has_custom_stop: bool,
    /// Per-step verbosity (`"none"` | `"summ"` | `"minimal"` | `"full"`).
    pub log_level: String,
    /// Problem-space target (`"tabular"` | `"onecimage"` | `"threecimage"` |
    /// `"nlp"` | `"rl"`).
    pub mode: String,
    /// Effective shared-stream geometry — after the trainer's `stream_shape()`
    /// override, i.e. the shape the run actually used.
    pub batch_size: usize,
    pub eval_batch_size: usize,
    /// Per-net fitness smoothing window (K).
    pub smoothing_window: usize,
    /// Device the run executed on (`"cpu"` | `"cuda:0"`).
    pub device: String,
    /// Whether the lossless per-step `metrics.csv` was written.
    pub csv_export: bool,
    /// Post-race pruner: `(method, extra_steps)`. `None` = off. `#[serde(default)]`
    /// so legacy `engine.json` headers load cleanly.
    #[serde(default)]
    pub pop_pruner: Option<(String, usize)>,
}

impl ConfigSnapshot {
    /// Capture every knob from a run's config plus the effective stream shape.
    pub fn from_config(cfg: &RaceConfig, batch_size: usize, eval_batch_size: usize) -> Self {
        ConfigSnapshot {
            pop_size: cfg.pop_size,
            checkpoint_every: cfg.checkpoint_every,
            crossover_rolls: cfg.crossover_rolls,
            mutate_rolls: cfg.mutate_rolls,
            crossover_gating: format!("{:?}", cfg.crossover_gating).to_lowercase(),
            crossover_retries: cfg.crossover_retries,
            crossover_cull_policy: format!("{:?}", cfg.crossover_cull_policy).to_lowercase(),
            elite_count: cfg.elite_count,
            crossover_ops_pool: cfg.crossover_ops_pool.clone(),
            crossover_prob: cfg.crossover_prob,
            mutate_prob: cfg.mutate_prob,
            max_target_fitness: cfg.max_target_fitness,
            has_custom_stop: cfg.custom_stop.is_some(),
            log_level: format!("{:?}", cfg.log_level).to_lowercase(),
            mode: format!("{:?}", cfg.mode).to_lowercase(),
            batch_size,
            eval_batch_size,
            smoothing_window: SMOOTHING_WINDOW,
            device: if cfg!(feature = "cuda") { "cuda:0" } else { "cpu" }.to_string(),
            csv_export: cfg.csv_export,
            pop_pruner: cfg.pop_pruner.map(|p| (format!("{:?}", p.method).to_lowercase(), p.steps)),
        }
    }
}

/// The reproducible run header, serialized as `engine.json` in the run dir.
///
/// This is the "who are we, what rules are we running under" snapshot. It is
/// overwritten whenever the config that affects reproducibility changes — it
/// does **not** accumulate history. The live population lives in
/// `nets/<hash>.json` files, not here.
#[derive(Clone, Debug, Serialize)]
pub struct RunHeader {
    /// User-visible run id (directory name or CLI-chosen label).
    pub run_id: String,
    /// Single deterministic root for the whole run. When the user passes
    /// `--seed N` this is that N; when no seed is given the engine generates
    /// one and records it here so an interrupted run can be resumed exactly.
    pub run_seed: u64,
    /// Fitness label used for ranking (e.g. `"f1"`, `"cross_entropy"`).
    pub fitness_label: FitnessLabel,
    /// Whether lower or higher fitness is better.
    pub fitness_direction: Direction,
    /// Dataset dims the run was launched with (informational; the stream
    /// derives its pool from the actual dataset on disk).
    pub input_dim: usize,
    pub output_dim: usize,
    /// Topology template + GP-pool knobs the run was launched with.
    pub topology_options: TopologyOptions,
    /// Hidden-dim sampling range.
    pub hidden_dim_pool: String,
    pub hidden_dim_stride: usize,
    /// Combine/activation/standardize pools actually used by the run.
    pub combine_op_pool: Vec<String>,
    pub activation_pool: Vec<String>,
    pub standardize_op_pool: Vec<String>,
    /// Informative (non-ranking) metrics configured for the run, if any.
    /// Their labels determine the extra columns a reader may expect in a
    /// per-net metrics snapshot.
    pub informative_metrics: Vec<String>,
    /// Stop criteria / budgets configured for the run (informational;
    /// the scheduler honors them, this is the recorded shape for later tools).
    pub max_steps: Option<usize>,
    /// When the run was started (wall clock), for human-readability.
    pub started_at: Option<String>,
    pub train_eval_split_ratio: Option<f32>,
    pub held_out_eval_rows: Option<usize>,
    /// The run's complete configuration (see [`ConfigSnapshot`]). This is the
    /// authoritative settings record — `engine.json` alone reproduces a run.
    #[serde(default)]
    pub config: ConfigSnapshot,
    /// The training scheme's self-description, from `Trainer::describe()`
    /// (learning rate, grad clip, optimizer, schedule, …). Free-form JSON:
    /// the engine persists it verbatim and never interprets it. `None` when
    /// the trainer doesn't implement the hook (or on legacy `engine.json`).
    #[serde(default)]
    pub trainer: Option<serde_json::Value>,
}

/// Everything that defines a step-race run's reproducible configuration —
/// the field-by-field input to [`RunHeader::from_race_options`].
#[derive(Debug)]
pub struct RunConfig {
    pub run_id: String,
    pub run_seed: u64,
    pub fitness_label: FitnessLabel,
    pub fitness_direction: Direction,
    pub input_dim: usize,
    pub output_dim: usize,
    pub topology_options: TopologyOptions,
    pub hidden_dim_pool: std::ops::RangeInclusive<usize>,
    pub hidden_dim_stride: usize,
    pub combine_op_pool: Vec<String>,
    pub activation_pool: Vec<String>,
    pub standardize_op_pool: Vec<String>,
    pub informative_metrics: Vec<Metric>,
    pub max_steps: Option<usize>,
    pub train_eval_split_ratio: Option<f32>,
    pub held_out_eval_rows: Option<usize>,
    /// The run's complete configuration (see [`ConfigSnapshot`]).
    pub config: ConfigSnapshot,
    /// The training scheme's self-description (see `Trainer::describe`).
    pub trainer: Option<serde_json::Value>,
}

impl RunHeader {
    /// Build the header from the run's launch config.
    pub fn from_race_options(cfg: RunConfig) -> Self {
        let RunConfig {
            run_id,
            run_seed,
            fitness_label,
            fitness_direction,
            input_dim,
            output_dim,
            topology_options,
            hidden_dim_pool,
            hidden_dim_stride,
            combine_op_pool,
            activation_pool,
            standardize_op_pool,
            informative_metrics,
            max_steps,
            train_eval_split_ratio,
            held_out_eval_rows,
            config,
            trainer,
        } = cfg;
        RunHeader {
            run_id,
            run_seed,
            fitness_label,
            fitness_direction,
            input_dim,
            output_dim,
            topology_options,
            hidden_dim_pool: format!("{}..={}", hidden_dim_pool.start(), hidden_dim_pool.end()),
            hidden_dim_stride,
            combine_op_pool,
            activation_pool,
            standardize_op_pool,
            informative_metrics: informative_metrics
                .into_iter()
                .map(|m| m.label().to_string())
                .collect(),
            max_steps,
            started_at: None,
            train_eval_split_ratio,
            held_out_eval_rows,
            config,
            trainer,
        }
    }

    /// Like `from_race_options`, but also stamps `started_at`.
    pub fn from_race_options_at(cfg: RunConfig, started_at: String) -> Self {
        let mut out = Self::from_race_options(cfg);
        out.started_at = Some(started_at);
        out
    }

    /// Serialize to pretty JSON for writing `engine.json`.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)
            .map_err(|e| EngineError::Json(format!("engine.json serialize: {e}")))?)
    }

    /// Load a previously written `engine.json`.
    pub fn from_json(raw: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| EngineError::Json(format!("engine.json parse: {e}")))?;
        let fitness_direction = v
            .get("fitness_direction")
            .and_then(|f| f.as_str())
            .and_then(|s| match s {
                "Minimize" => Some(Direction::Minimize),
                "Maximize" => Some(Direction::Maximize),
                _ => None,
            })
            .unwrap_or(Direction::Maximize);
        let fitness_label = serde_json::from_value(
            v.get("fitness_label")
                .cloned()
                .unwrap_or_else(|| serde_json::Value::String("loss".into())),
        )
        .unwrap_or(FitnessLabel("loss".into()));
        let input_dim = v.get("input_dim").and_then(|f| f.as_u64()).unwrap_or(0) as usize;
        let output_dim = v.get("output_dim").and_then(|f| f.as_u64()).unwrap_or(0) as usize;
        let topology_options = v
            .get("topology_options")
            .map(|t| serde_json::from_value(t.clone()).unwrap_or_default())
            .unwrap_or_default();
        let hidden_dim_pool = v
            .get("hidden_dim_pool")
            .and_then(|f| f.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "4..=8".into());
        let hidden_dim_stride: usize = v
            .get("hidden_dim_stride")
            .and_then(|f| f.as_u64())
            .map(|n| n as usize)
            .unwrap_or(1);
        let combine_op_pool: Vec<String> = serde_json::from_value(
            v.get("combine_op_pool")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
        .unwrap_or_default();
        let activation_pool: Vec<String> = serde_json::from_value(
            v.get("activation_pool")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
        .unwrap_or_default();
        let standardize_op_pool: Vec<String> = serde_json::from_value(
            v.get("standardize_op_pool")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
        .unwrap_or_default();
        let informative_metrics: Vec<String> = serde_json::from_value(
            v.get("informative_metrics")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
        .unwrap_or_default();
        let max_steps = v
            .get("max_steps")
            .and_then(|f| f.as_u64())
            .map(|n| n as usize);
        let started_at = v
            .get("started_at")
            .and_then(|f| f.as_str())
            .map(|s| s.to_string());
        let train_eval_split_ratio = v
            .get("train_eval_split_ratio")
            .and_then(|f| f.as_f64())
            .map(|n| n as f32);
        let held_out_eval_rows = v
            .get("held_out_eval_rows")
            .and_then(|f| f.as_u64())
            .map(|n| n as usize);
        // Missing on legacy engine.json files ⇒ Default (all-zero snapshot).
        let config: ConfigSnapshot = v
            .get("config")
            .cloned()
            .and_then(|c| serde_json::from_value(c).ok())
            .unwrap_or_default();
        // Trainer self-description (see `Trainer::describe`). Absent on legacy
        // files and on trainers that don't implement the hook.
        let trainer = v.get("trainer").cloned();
        Ok(RunHeader {
            run_id: v
                .get("run_id")
                .and_then(|f| f.as_str())
                .unwrap_or("unknown")
                .to_string(),
            run_seed: v.get("run_seed").and_then(|f| f.as_u64()).unwrap_or(0),
            fitness_label,
            fitness_direction,
            input_dim,
            output_dim,
            topology_options,
            hidden_dim_pool,
            hidden_dim_stride,
            combine_op_pool,
            activation_pool,
            standardize_op_pool,
            informative_metrics,
            max_steps,
            started_at,
            train_eval_split_ratio,
            held_out_eval_rows,
            config,
            trainer,
        })
    }
}

fn default_alive() -> bool {
    true
}

// ── Per-net state (nets/<hash>.json) ──────────────────────────────────────

/// The last-known state of one live net. Written to `nets/<hash>.json` and
/// overwritten each step the net survives (and on cull-enter, if a tombstone
/// path is enabled later).
///
/// This is the resume anchor for that net: load it, rebuild the topology +
/// network from the embedded blueprint + weight-init seed, then replay the
/// deterministic stream to `step`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetState {
    /// Canonical xxh3 hash of this individual's topology (16 hex chars).
    pub hash: String,
    /// The topology blueprint, serialized. Self-describing: `Network::build`
    /// reads `options.topology_seed` (weight init) and `options.dropout_prob` (the
    /// regularization the run applied) straight from here.
    pub topology: String,
    /// Weight-init seed embedded in the topology's `options.topology_seed`. Stored
    /// here redundantly so a reader doesn't have to re-parse the topology to
    /// recover the seed the run used for this individual.
    pub net_seed: usize,
    /// The step this net has been trained/evaluated through. A resume replays
    /// the stream from step 0 up to (but not including) this step, then
    /// continues from here.
    pub step: usize,
    /// Whether this net is currently alive in the active population. Culled nets
    /// are written as final tombstones to disk with `is_alive: false`.
    #[serde(default = "default_alive")]
    pub is_alive: bool,
    /// Step at which this net was culled (`None` while it is alive). With
    /// `cull_reason` and `final_smoothed_fitness` this makes a tombstone a
    /// complete record of when and why a net left the population — not just
    /// the surviving frontier.
    #[serde(default)]
    pub culled_at_step: Option<usize>,
    /// Why it left (`"crossover"` | `"immigrant-slot"`).
    #[serde(default)]
    pub cull_reason: Option<String>,
    /// Its smoothed fitness at the moment of culling.
    #[serde(default)]
    pub final_smoothed_fitness: Option<f32>,
    /// When this individual entered the live population. For lifetime tracking
    /// and the robustness-equivalent (appearances / span / final score).
    pub entered_at_step: usize,
    /// Optional lineage record for Iter 5: which parent(s) / mutation produced
    /// this individual. Left as an optional string so the field exists today
    /// without inventing the crossover/mutation schema ahead of Iter 5.
    pub created_from: Option<String>,
    /// Last-seen metrics snapshot. Enough for a resume/time-travel read
    /// without walking a separate CSV. Schema: loss + fitness + any configured
    /// informative metrics. Empty until the first persisted step.
    pub last_metrics: Option<NetMetrics>,
    /// Rich per-individual metadata (params, dims, training regime, run
    /// context). Written once at construction; missing on legacy JSONs
    /// (serde default) and backfilled on first rewrite.
    #[serde(default)]
    pub meta: NetMeta,
}

/// Rich metadata for one individual, embedded in `nets/<hash>.json` under
/// the `meta` key. Everything a reader (human or tool) needs to understand
/// the net *and* the conditions it trained under, without touching code.
/// Populated once at construction; `params` is stamped right after the
/// `Network` is built (cheap, computed once — never re-serialized per step).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NetMeta {
    /// Trainable parameter count (elements) of the built network.
    #[serde(default)]
    pub params: Option<i64>,
    /// Number of parameter tensors.
    #[serde(default)]
    pub param_tensors: Option<usize>,
    /// Dataset dims the run was launched with.
    #[serde(default)]
    pub input_dim: Option<usize>,
    #[serde(default)]
    pub output_dim: Option<usize>,
    /// Shared training regime for the run.
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Deprecated: the learning rate is trainer-owned. The authoritative value
    /// is `engine.json` → `"trainer"` (from `Trainer::describe`). Stays `None`
    /// here because the engine cannot see inside the user's trainer.
    #[serde(default)]
    pub learning_rate: Option<f32>,
    /// Deprecated: see `learning_rate` — trainer-owned, recorded in
    /// `engine.json` → `"trainer"`.
    #[serde(default)]
    pub grad_clip: Option<f32>,
    /// Dropout probability stamped into the topology (also lives in the
    /// topology JSON; duplicated here for quick scanning).
    #[serde(default)]
    pub dropout_prob: Option<f32>,
    /// Ranking fitness + loss function labels for this run.
    #[serde(default)]
    pub fitness_label: Option<String>,
    #[serde(default)]
    pub loss_label: Option<String>,
    /// What ranking direction this run uses ("minimize"/"maximize").
    #[serde(default)]
    pub direction: Option<String>,
    /// The step this individual was born into the run (== entered_at_step;
    /// duplicated so the meta block is self-contained).
    #[serde(default)]
    pub born_at_step: Option<usize>,
    /// Population size + run seed context (inherited from the header).
    #[serde(default)]
    pub pop_size: Option<usize>,
    #[serde(default)]
    pub run_seed: Option<u64>,
}

/// One snapshot of metrics for a single net, at a single step.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NetMetrics {
    pub step: usize,
    /// Training loss for this step's train batch.
    pub train_loss: f32,
    /// Held-out loss for this step's eval batch (eval mode).
    pub eval_loss: Option<f32>,
    /// Ranking fitness for this step's eval batch (the signal the run evolves on).
    pub fitness: f32,
    /// Informative-only metric values, in the same order the run configured them.
    pub informative: Vec<f32>,
}

impl NetState {
    /// Stamp the `meta` block from the materialized network + the run context.
    /// Called once per net right after its `Network` is built (initial seed,
    /// child finalize, resume rebuild) — never per step.
    pub fn stamp_meta(
        &mut self,
        net: &crate::graph::network::Network,
        run: &crate::engine::config::RunMetaCtx,
    ) {
        use flodl::Module;
        let params = net.parameters();
        self.meta.param_tensors = Some(params.len());
        self.meta.params = Some(params.iter().map(|p| p.variable.numel()).sum());
        self.meta.input_dim = Some(run.input_dim);
        self.meta.output_dim = Some(run.output_dim);
        self.meta.batch_size = Some(run.batch_size);
        // learning_rate/grad_clip are trainer-owned, so the engine leaves them
        // None in the per-net meta; the trainer's `describe()` block in
        // engine.json is the authoritative record.
        self.meta.fitness_label = Some(run.fitness_label.clone());
        self.meta.loss_label = Some(run.loss_label.clone());
        self.meta.direction = Some(run.direction.clone());
        self.meta.born_at_step = Some(self.entered_at_step);
        self.meta.pop_size = Some(run.pop_size);
        self.meta.run_seed = Some(run.run_seed);
        // Dropout lives in the topology too — duplicate for quick scanning.
        self.meta.dropout_prob = Some(run.dropout_prob);
    }

    /// Build a new live individual's state from its topology and entry step.
    pub fn new(topo: &Topology, step: usize, created_from: Option<String>) -> Result<Self> {
        let topo_json = topo
            .to_json()
            .map_err(|e| EngineError::Json(format!("net state topology json: {e}")))?;
        let net_seed = topo.options.topology_seed;
        Ok(NetState {
            hash: topo_hash(&topo_json),
            topology: topo_json,
            net_seed,
            step,
            is_alive: true,
            culled_at_step: None,
            cull_reason: None,
            final_smoothed_fitness: None,
            entered_at_step: step,
            created_from,
            last_metrics: None,
            meta: NetMeta::default(),
        })
    }

    /// Serialize to pretty JSON for writing `nets/<hash>.json`.
    /// Backfill `meta` on legacy JSONs so every written record carries a
    /// self-contained metadata block. We rebuild a throwaway `Network` from
    /// the topology just to count params (cheap, once per write flight); the
    /// `RunMetaCtx::default()` is fine here because we only need the network
    /// shape, not the run regime (that comes from the callers who call
    /// `stamp_meta` directly with the real run context).
    pub fn to_json(&self) -> Result<String> {
        let mut to_write = self.clone();
        if to_write.meta.params.is_none() {
            let topo = crate::graph::topology::Topology::from_json(&to_write.topology)
                .map_err(|e| EngineError::Json(format!("net state meta backfill: {e}")))?;
            let net = crate::graph::network::Network::build(&topo, crate::Device::CPU)
                .map_err(|e| EngineError::Json(format!("net state meta backfill build: {e}")))?;
            to_write.stamp_meta(&net, &crate::engine::config::RunMetaCtx::default());
        }
        Ok(serde_json::to_string_pretty(&to_write)
            .map_err(|e| EngineError::Json(format!("net state json: {e}")))?)
    }

    /// Load a previously written `nets/<hash>.json`.
    pub fn from_json(raw: &str) -> Result<Self> {
        Ok(serde_json::from_str(raw)
            .map_err(|e| EngineError::Json(format!("net state parse: {e}")))?)
    }

    /// Update the last-metrics snapshot.
    pub fn record_metrics(&mut self, metrics: NetMetrics) {
        self.last_metrics = Some(metrics);
    }

    /// Advance the step counter (called after a net's step is fully recorded).
    pub fn advance_step(&mut self) {
        self.step += 1;
    }

    /// Recover the topology from the embedded JSON.
    pub fn topology(&self) -> Result<Topology> {
        Ok(Topology::from_json(&self.topology)
            .map_err(|e| EngineError::Json(format!("net state topology parse: {e}")))?)
    }
}

// ── Runtime population state (in memory) ──────────────────────────────────

/// The in-memory view of the live population that the scheduler reads/writes.
///
/// Backed by a hash map from canonical topology hash to `NetState`, plus a
/// small robustness-equivalent accumulator that tracks per-topology lifetime
/// and final score over the run (the replacement for the old per-generation
/// `robustness.csv` semantics — here it is one accumulator in the run, not a
/// separate CSV written at generation boundaries).
pub struct RaceState {
    /// Live nets, indexed by canonical topology hash.
    nets: HashMap<String, NetState>,
    /// Per-topology lifetime/score accumulator (robustness-equivalent).
    topology_lifetime: HashMap<String, TopologyLifetime>,
}

/// Per-topology lifetime tracking — the robustness-equivalent for the step-race
/// model. Tracks how many times a topology entered the live population, the
/// step span it covered, and its final recorded fitness.
#[derive(Clone, Debug)]
pub struct TopologyLifetime {
    /// Times this topology entered the live population.
    appearances: usize,
    /// First step this topology was live at.
    first_step: usize,
    /// Last step this topology was live at (updated as it survives).
    last_step: usize,
    /// Last recorded fitness for this topology (the final one seen in the run).
    last_fitness: Option<f32>,
}

impl TopologyLifetime {
    pub fn new(first_step: usize, fitness: Option<f32>) -> Self {
        TopologyLifetime {
            appearances: 1,
            first_step,
            last_step: first_step,
            last_fitness: fitness,
        }
    }

    pub fn record_entry(&mut self, step: usize, fitness: Option<f32>) {
        self.appearances = self.appearances.saturating_add(1);
        self.last_step = step;
        self.last_fitness = fitness;
    }

    /// Advance the liveness window: `last_step` always moves forward;
    /// fitness only overwrites when a fresh finite value is given.
    pub fn record_progress(&mut self, step: usize, fitness: Option<f32>) {
        self.last_step = self.last_step.max(step);
        if let Some(f) = fitness {
            self.last_fitness = Some(f);
        }
    }

    pub fn appearances(&self) -> usize {
        self.appearances
    }

    pub fn first_step(&self) -> usize {
        self.first_step
    }

    pub fn last_step(&self) -> usize {
        self.last_step
    }

    pub fn span(&self) -> usize {
        self.last_step.saturating_sub(self.first_step)
    }

    pub fn last_fitness(&self) -> Option<f32> {
        self.last_fitness
    }
}

impl Default for RaceState {
    fn default() -> Self {
        Self::new()
    }
}

impl RaceState {
    pub fn new() -> Self {
        RaceState {
            nets: HashMap::new(),
            topology_lifetime: HashMap::new(),
        }
    }

    /// Register a newly inserted net (after catch-up or initial insertion).
    pub fn insert(&mut self, state: NetState, initial_fitness: Option<f32>) {
        let key = state.hash.clone();
        if let Some(entry) = self.topology_lifetime.get_mut(&key) {
            entry.record_entry(state.entered_at_step, initial_fitness);
        } else {
            self.topology_lifetime.insert(
                key.clone(),
                TopologyLifetime::new(state.entered_at_step, initial_fitness),
            );
        }
        self.nets.insert(key, state);
    }

    /// Record that a live net advanced one step with the given metrics.
    pub fn record_step(&mut self, hash: &str, metrics: NetMetrics) -> Result<()> {
        let state = self.nets.get_mut(hash).ok_or_else(|| {
            EngineError::InvalidOptions(format!("race state: net {hash} not live"))
        })?;
        state.record_metrics(metrics.clone());
        state.advance_step();
        let fitness = metrics.fitness;
        let fitness = if fitness.is_finite() {
            Some(fitness)
        } else {
            None
        };
        let entry = self
            .topology_lifetime
            .entry(hash.to_string())
            .or_insert_with(|| TopologyLifetime::new(state.entered_at_step, fitness));
        entry.record_progress(metrics.step, fitness);
        Ok(())
    }

    pub fn net(&self, hash: &str) -> Option<&NetState> {
        self.nets.get(hash)
    }

    pub fn net_mut(&mut self, hash: &str) -> Option<&mut NetState> {
        self.nets.get_mut(hash)
    }

    pub fn live_hashes(&self) -> Vec<String> {
        let mut hashes: Vec<String> = self.nets.keys().cloned().collect();
        hashes.sort();
        hashes
    }

    pub fn live_count(&self) -> usize {
        self.nets.len()
    }

    /// Remove a net from the live set. The state is returned so the caller can
    /// decide whether to tombstone it (off by default today) or drop it.
    pub fn remove(&mut self, hash: &str) -> Option<NetState> {
        let state = self.nets.remove(hash);
        if let Some(state) = &state {
            self.topology_lifetime.remove(&state.hash);
        }
        state
    }

    /// Iterate over every topology lifetime entry.
    pub fn lifetimes(&self) -> Vec<(&String, &TopologyLifetime)> {
        let mut out: Vec<_> = self.topology_lifetime.iter().collect();
        out.sort_by_key(|(a, _)| *a);
        out
    }

    /// Clear the entire state (used by tests / fresh-run init).
    pub fn clear(&mut self) {
        self.nets.clear();
        self.topology_lifetime.clear();
    }
}

// ── File I/O ───────────────────────────────────────────────────────────────

/// Write the run header to `engine.json` in `dir`.
pub fn write_engine_json(dir: &Path, header: &RunHeader) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|source| EngineError::Io {
        path: dir.display().to_string(),
        source,
    })?;
    let path = dir.join("engine.json");
    let json = header.to_json()?;
    std::fs::write(&path, json).map_err(|source| EngineError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(())
}

/// Load the run header from `engine.json` in `dir`.
pub fn load_engine_json(dir: &Path) -> Result<RunHeader> {
    let path = dir.join("engine.json");
    let raw = std::fs::read_to_string(&path).map_err(|source| EngineError::Io {
        path: path.display().to_string(),
        source,
    })?;
    RunHeader::from_json(&raw)
}

/// Ensure the `nets/` directory exists under `dir`.
pub fn ensure_nets_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir.join("nets")).map_err(|source| EngineError::Io {
        path: dir.display().to_string(),
        source,
    })?;
    Ok(())
}

/// Write (overwrite) one net's state to `nets/<hash>.json`.
pub fn write_net_state(dir: &Path, state: &NetState) -> Result<()> {
    ensure_nets_dir(dir)?;
    let path = dir.join("nets").join(format!("{}.json", state.hash));
    let json = state.to_json()?;
    std::fs::write(&path, json).map_err(|source| EngineError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(())
}

/// Load one net's state from `nets/<hash>.json`.
pub fn load_net_state(dir: &Path, hash: &str) -> Result<NetState> {
    let path = dir.join("nets").join(format!("{}.json", hash));
    let raw = std::fs::read_to_string(&path).map_err(|source| EngineError::Io {
        path: path.display().to_string(),
        source,
    })?;
    NetState::from_json(&raw)
}

/// Remove one net's state file. Returns whether a file was removed.
pub fn remove_net_state(dir: &Path, hash: &str) -> Result<bool> {
    let path = dir.join("nets").join(format!("{}.json", hash));
    if path.exists() {
        std::fs::remove_file(&path).map_err(|source| EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Ok(true)
    } else {
        Ok(false)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::fitness::{Direction, FitnessLabel};
    use crate::graph::topology::TopologyOptions;
    use std::path::PathBuf;

    fn fresh_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gras_race_state_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn baseline_header(run_id: &str) -> RunHeader {
        RunHeader::from_race_options(RunConfig {
            run_id: run_id.to_string(),
            run_seed: 42,
            fitness_label: FitnessLabel("f1".to_string()),
            fitness_direction: Direction::Maximize,
            input_dim: 784,
            output_dim: 10,
            topology_options: TopologyOptions::default(),
            hidden_dim_pool: 4..=8,
            hidden_dim_stride: 1,
            combine_op_pool: vec!["Mean".into(), "Min".into()],
            activation_pool: vec!["ReLU".into(), "SELU".into()],
            standardize_op_pool: vec!["Identity".into()],
            informative_metrics: vec![Metric("dummy".to_string())],
            max_steps: Some(10_000),
            train_eval_split_ratio: Some(0.2),
            held_out_eval_rows: Some(256),
            config: ConfigSnapshot::default(),
            trainer: None,
        })
    }

    fn tiny_topology() -> Topology {
        let mut topo = crate::graph::topology::Topology::new(
            0,
            Some(crate::graph::topology::TopologyOptions {
                topology_seed: 7,
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
        topo.nodes.push(crate::graph::node::Node::new_input(0, 2));
        topo.nodes
            .push(crate::graph::node::Node::new_hidden(1, 2, 4));
        topo.nodes
            .push(crate::graph::node::Node::new_output(2, 4, 2));
        topo
    }

    #[test]
    fn engine_json_write_and_read_back_identical() {
        let dir = fresh_dir();
        let header = baseline_header("run-1");
        write_engine_json(&dir, &header).unwrap();
        let loaded = load_engine_json(&dir).unwrap();
        assert_eq!(loaded.run_id, "run-1");
        assert_eq!(loaded.run_seed, 42);
        assert_eq!(loaded.fitness_label.0, "f1");
        assert_eq!(loaded.fitness_direction, Direction::Maximize);
        assert_eq!(loaded.input_dim, 784);
        assert_eq!(loaded.output_dim, 10);
        assert_eq!(loaded.max_steps, Some(10_000));
        // Overwrite changes the header.
        let updated = baseline_header("run-2");
        write_engine_json(&dir, &updated).unwrap();
        let again = load_engine_json(&dir).unwrap();
        assert_eq!(again.run_id, "run-2");
    }

    #[test]
    fn net_state_round_trip_keeps_hash_and_seed_and_step() {
        let dir = fresh_dir();
        let topo = tiny_topology();
        let mut state = NetState::new(&topo, 0, Some("parent-a x parent-b".into())).unwrap();
        state.record_metrics(NetMetrics {
            step: 5,
            train_loss: 1.25,
            eval_loss: Some(1.40),
            fitness: 0.61,
            informative: vec![0.4, 0.6],
        });
        write_net_state(&dir, &state).unwrap();
        let loaded = load_net_state(&dir, &state.hash).unwrap();
        assert_eq!(loaded.hash, state.hash);
        assert_eq!(loaded.net_seed, 7);
        assert_eq!(loaded.step, 0);
        assert_eq!(loaded.entered_at_step, 0);
        assert_eq!(loaded.created_from.as_deref(), Some("parent-a x parent-b"));
        let m = loaded.last_metrics.as_ref().unwrap();
        assert_eq!(m.step, 5);
        assert_eq!(m.train_loss, 1.25);
        assert_eq!(m.eval_loss, Some(1.40));
        assert_eq!(m.fitness, 0.61);
        assert_eq!(m.informative, vec![0.4, 0.6]);
        // Topology round-trips exactly (topo_hash is canonical — same JSON ⇒ same hash).
        assert_eq!(loaded.hash, topo_hash(&loaded.topology));
        let rebuilt = loaded.topology().unwrap();
        let rebuilt_json = rebuilt.to_json().unwrap();
        assert_eq!(rebuilt_json, loaded.topology);
    }

    #[test]
    fn overwrite_replaces_previous_step() {
        let dir = fresh_dir();
        let topo = tiny_topology();
        let mut state = NetState::new(&topo, 0, None).unwrap();
        state.record_metrics(NetMetrics {
            step: 0,
            train_loss: 2.0,
            eval_loss: Some(2.1),
            fitness: 0.1,
            informative: vec![],
        });
        write_net_state(&dir, &state).unwrap();

        state.advance_step();
        state.record_metrics(NetMetrics {
            step: 1,
            train_loss: 1.5,
            eval_loss: Some(1.6),
            fitness: 0.2,
            informative: vec![],
        });
        write_net_state(&dir, &state).unwrap();

        let loaded = load_net_state(&dir, &state.hash).unwrap();
        assert_eq!(loaded.step, 1);
        assert_eq!(loaded.last_metrics.as_ref().unwrap().fitness, 0.2);
    }

    #[test]
    fn race_state_insert_and_remove() {
        let mut st = RaceState::new();
        let topo = tiny_topology();
        let state = NetState::new(&topo, 0, None).unwrap();
        st.insert(state, Some(0.3));

        assert_eq!(st.live_count(), 1);
        let hashes = st.live_hashes();
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes[0].len(), 16);

        let removed = st.remove(&hashes[0]);
        assert!(removed.is_some());
        assert_eq!(st.live_count(), 0);
    }

    #[test]
    fn race_state_lifetime_tracks_appearances_and_span() {
        let mut st = RaceState::new();
        let topo = tiny_topology();
        // Insert a net at step 10 with an initial fitness — the lifetime
        // accumulator seeds from that entry call.
        let state = NetState::new(&topo, 10, None).unwrap();
        st.insert(state, Some(0.3));

        let h = st.live_hashes()[0].clone();
        // Each record_step advances the net's own step counter, and seeds or
        // updates the per-topology lifetime from the metrics' step and fitness.
        st.record_step(
            &h,
            NetMetrics {
                step: 10,
                train_loss: 0.0,
                eval_loss: None,
                fitness: 0.4,
                informative: vec![],
            },
        )
        .unwrap();
        st.record_step(
            &h,
            NetMetrics {
                step: 11,
                train_loss: 0.0,
                eval_loss: None,
                fitness: 0.5,
                informative: vec![],
            },
        )
        .unwrap();

        let (_, life) = st.lifetimes().into_iter().next().unwrap();
        assert_eq!(life.appearances(), 1);
        assert_eq!(life.first_step(), 10);
        assert_eq!(life.last_step(), 11);
        assert_eq!(life.span(), 1);
        assert_eq!(life.last_fitness(), Some(0.5));
        assert_eq!(st.live_count(), 1);
    }

    #[test]
    fn removed_net_file_is_gone() {
        let dir = fresh_dir();
        let topo = tiny_topology();
        let state = NetState::new(&topo, 0, None).unwrap();
        write_net_state(&dir, &state).unwrap();
        assert!(
            dir.join("nets")
                .join(format!("{}.json", state.hash))
                .exists()
        );
        remove_net_state(&dir, &state.hash).unwrap();
        assert!(
            !dir.join("nets")
                .join(format!("{}.json", state.hash))
                .exists()
        );
    }

    #[test]
    fn load_nonexistent_net_errors() {
        let dir = fresh_dir();
        let err = load_net_state(&dir, "deadbeef").unwrap_err();
        assert!(err.to_string().contains("cannot access"));
    }
}
