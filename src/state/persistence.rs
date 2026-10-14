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

use crate::engine::config::RaceConfig;
use crate::engine::fitness::{Direction, FitnessLabel, Metric};
use crate::graph::topology::{Topology, TopologyOptions};
use crate::utils::error::EngineError;
use crate::utils::seed::topo_hash;

// ── Run header (engine.json) ───────────────────────────────────────────────

/// The run's **complete** configuration, serialized into `engine.json` under
/// `"config"` — the single source of truth; the file alone reproduces the run.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ConfigSnapshot {
    /// Population size the race held constant.
    pub pop_size: usize,
    /// Steps between population checkpoints.
    pub checkpoint_every: usize,
    /// Crossover rolls offered per step.
    pub crossover_rolls: usize,
    /// Mutation (immigrant) rolls offered per step.
    pub mutate_rolls: usize,
    /// Crossover replacement policy (`"worst"` | `"random"`). Crossover-only:
    /// mutation immigrants always evict via fitness-inverse roulette.
    pub crossover_cull_policy: String,
    /// Elite guard size: top-k nets by smoothed fitness immune to all culls.
    pub elite_count: usize,
    /// Crossover operator pool (`"one_point"`/`"uniform"`); empty ⇒ both.
    pub crossover_ops_pool: Vec<String>,
    /// Probability a crossover roll fires.
    pub crossover_prob: f32,
    /// Probability a mutation roll fires.
    pub mutate_prob: f32,
        /// Stop budget as configured (`None` = no limit). Duplicated at the header root;
        /// this copy is authoritative.
    #[serde(default)]
    pub max_steps: Option<usize>,
    /// Whether a pluggable `custom_stop` closure was installed. The closure is
    /// code, not data, so only its presence is recordable.
    pub has_custom_stop: bool,
    /// Per-step verbosity (`"none"` | `"summ"` | `"minimal"`).
    pub log_level: String,
    /// Whether the run wrote the `telemetry.jsonl` event sidecar
    /// (`#[serde(default)]` so pre-knob `engine.json` files still load).
    #[serde(default)]
    pub trace_file: bool,
    /// Problem-space target (`"tabular"` | `"onecimage"` | `"threecimage"` |
    /// `"nlp"` | `"rl"`).
    pub mode: String,
        /// Effective shared-stream geometry after the trainer's `stream_shape()` override.
        /// `None` in modes with no shared stream (RL).
    pub batch_size: Option<usize>,
    /// Effective eval batch size (`None` when the mode has no shared stream).
    pub eval_batch_size: Option<usize>,
    /// Per-net fitness smoothing window (K).
    pub smoothing_window: usize,
        /// Risk aversion λ (FIT) the run ranked with — window stds charged against a net's
        /// smoothed mean. `#[serde(default)]` (⇒ `0.0`) because every run recorded before
        /// FIT ranked on the PLAIN MEAN: resuming one must reproduce that exact ranking, so
        /// the legacy default is the pre-FIT behavior, not the new default.
    #[serde(default)]
    pub risk_aversion: f32,
    /// Device the run executed on (`"cpu"` | `"cuda:0"`).
    pub device: String,
    /// Whether the lossless per-step `metrics.csv` was written.
    pub csv_export: bool,
    /// Post-race pruner: `(method, extra_steps)`. `None` = off. `#[serde(default)]`
    /// so legacy `engine.json` headers load cleanly.
    #[serde(default)]
    pub pop_pruner: Option<(String, usize)>,
    /// Human label for the experiment (`set_run_name`), verbatim. Purely
    /// informative — never affects the results folder name.
    pub run_name: Option<String>,
        /// Anti-devolution A (`set_elite_freeze`): top-k nets skip the trainer call.
        /// Replay-relevant, so it must match on resume.
    #[serde(default)]
    pub freeze_elites: bool,
        /// How many initial slots were user-supplied run topologies. Informational (lineage
        /// reads: ordinals below this are imports).
    #[serde(default)]
    pub run_topology_count: usize,
        /// Per-checkpoint elite weight snapshot flag. Informational — describes artifact
        /// cadence, not replay semantics.
    #[serde(default = "default_true")]
    pub elite_checkpoint_weights: bool,
        /// Mutation probation steps: cull-immunity for an immigrant's first k clocks.
        /// Replay-relevant.
    #[serde(default)]
    pub mutation_probation_steps: usize,
        /// Crossover probation steps: cull-immunity for a crossover child's first k clocks.
        /// Replay-relevant.
    #[serde(default)]
    pub crossover_probation_steps: usize,
        /// Elite join probation steps: clocks a fresh net must train before it may hold a
        /// seat. Replay-relevant.
    #[serde(default)]
    pub elite_join_probation_steps: usize,
        /// Elite crown lifetime steps: a net on the crown is culled out of it after k clocks
        /// at peak (`0` disables). Replay-relevant.
    #[serde(default)]
    pub elite_crown_lifetime_steps: usize,
        /// Fresh holdout games the post-race guardrail plays. Diagnostic — no resume guard.
    #[serde(default = "default_guardrail_matches")]
    pub guardrail_matches: usize,
        /// Load the newest compatible previous experiment's elites as the first
        /// founders. Recorded because it decides what the founder batch IS. Default
        /// false. `alias` reads headers written before the rename.
    #[serde(default, alias = "seed_previous_run")]
    pub elite_previous_load: bool,
        /// Where that previous experiment is looked up: a run dir or a dir of run dirs
        /// (newest wins). `None` ⇒ `results/`.
    #[serde(default, alias = "seed_previous_run_dir")]
    pub elite_previous_dir: Option<String>,
        /// One-line identity of the binary that produced the run (version, profile, exe
        /// path, build time). Diagnostics only.
    #[serde(default)]
    pub build: String,
}

impl ConfigSnapshot {
    /// Capture every knob from a run's config plus the effective stream shape
    /// (`None`/`None` for modes with no shared stream — see the fields).
    pub fn from_config(
        cfg: &RaceConfig,
        batch_size: Option<usize>,
        eval_batch_size: Option<usize>,
    ) -> Self {
        ConfigSnapshot {
            pop_size: cfg.pop_size,
            checkpoint_every: cfg.checkpoint_every,
            crossover_rolls: cfg.crossover_rolls,
            mutate_rolls: cfg.mutate_rolls,
            crossover_cull_policy: format!("{:?}", cfg.crossover_cull_policy).to_lowercase(),
            elite_count: cfg.elite_count,
            crossover_ops_pool: cfg.crossover_ops_pool.clone(),
            crossover_prob: cfg.crossover_prob,
            mutate_prob: cfg.mutate_prob,
            max_steps: cfg.max_steps,
            has_custom_stop: cfg.custom_stop.is_some(),
            log_level: format!("{:?}", cfg.log_level).to_lowercase(),
            trace_file: cfg.trace_file,
            mode: format!("{:?}", cfg.mode).to_lowercase(),
            batch_size,
            eval_batch_size,
            smoothing_window: cfg.smoothing_window,
            risk_aversion: cfg.risk_aversion,
            device: if cfg!(feature = "cuda") {
                "cuda:0"
            } else {
                "cpu"
            }
            .to_string(),
            csv_export: cfg.csv_export,
            pop_pruner: cfg
                .pop_pruner
                .map(|p| (format!("{:?}", p.method).to_lowercase(), p.steps)),
            run_name: cfg.run_name.clone(),
            freeze_elites: cfg.freeze_elites,
            run_topology_count: cfg.run_topologies.len(),
            elite_previous_load: cfg.elite_previous_load,
            elite_previous_dir: cfg
                .elite_previous_dir
                .as_ref()
                .map(|p| p.display().to_string()),
            elite_checkpoint_weights: cfg.elite_checkpoint_weights,
            mutation_probation_steps: cfg.mutation_probation_steps,
            crossover_probation_steps: cfg.crossover_probation_steps,
            elite_join_probation_steps: cfg.elite_join_probation_steps,
            elite_crown_lifetime_steps: cfg.elite_crown_lifetime_steps,
            guardrail_matches: cfg.guardrail_matches,
            build: crate::engine::format::build_stamp(),
        }
    }
}

/// The post-race guardrail's record of the run's certified champion — the one field
/// of `engine.json` that is not launch config, and what makes seeding possible.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GuardrailRecord {
    /// Full hash of the champion that was scored.
    pub champion_hash: String,
        /// Trainable parameter count of that net (elements) — the tie-break key: on equal
        /// fitness the SMALLEST champion is scored and recorded.
    #[serde(default)]
    pub champion_params: Option<i64>,
        /// The champion's smoothed race fitness — the READABLE window mean, kept in the
        /// fitness's own units so it stays directly comparable with `holdout_mean`. `None` if
        /// the run recorded no verdict.
    #[serde(default)]
    pub race_smoothed: Option<f32>,
        /// Spread of the champion's race window (σ). The holdout's own `holdout_std` is the
        /// same quantity measured on fresh games, so the two spreads are comparable.
    #[serde(default)]
    pub race_std: Option<f32>,
        /// The RISK-ADJUSTED score (FIT) that ranked the champion first — `race_smoothed`
        /// penalized by `λ · race_std` in the run's direction. Recorded so a reader can see
        /// how much of the champion's rank came from steadiness rather than raw level.
    #[serde(default)]
    pub race_score: Option<f32>,
        /// Standard error of `race_smoothed`; `None` when the scoring path could not see the
        /// champion's rolling window (saved-run scoring).
    #[serde(default)]
    pub race_se: Option<f32>,
    /// Step readings behind `race_smoothed` (the smoothing window's length).
    /// `0` = unknown; `1` means the spread could not be observed at all.
    #[serde(default)]
    pub race_samples: usize,
        /// The champion's MOST RECENT step reading; the smoothed reference lags it by up to
        /// the window length, so the verdict quotes both when they disagree.
    #[serde(default)]
    pub race_latest: Option<f32>,
    /// Mean holdout score over the games that actually produced one.
    #[serde(default)]
    pub holdout_mean: Option<f32>,
    /// Population std of those scores.
    #[serde(default)]
    pub holdout_std: Option<f32>,
    /// Games that produced a score (per-game failures are skipped, so this can
    /// be less than `config.guardrail_matches`).
    #[serde(default)]
    pub matches_scored: usize,
        /// Unit-free agreement between the race reading and the holdout — which reading to
        /// trust and why. Informational.
    #[serde(default)]
    pub verdict: String,
}

/// Which previous experiment this run's founder batch was seeded from — the inheritance
/// receipt (informational; the copied topology is recorded per net like any founder).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SeedSourceRecord {
    /// The source run's id (its `results/<id>` directory name).
    pub source_run_id: String,
    /// Directory the source was read from (as resolved).
    pub source_dir: String,
    /// The guardrail-certified champion that was copied.
    pub champion_hash: String,
    /// That champion's parameter count (the tie-break key).
    #[serde(default)]
    pub champion_params: Option<i64>,
    /// Its holdout mean, carried for readability (the champion's pedigree).
    #[serde(default)]
    pub guardrail_mean: Option<f32>,
        /// Every elite reinstated here, ranked (best first, `[0]` == `champion_hash`);
        /// one entry when the source predates `elites.json`, empty if nothing inherited.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub elite_hashes: Vec<String>,
}

/// The reproducible run header, serialized as `engine.json`. Overwritten when the
/// reproducibility-affecting config changes; the post-race guardrail record is preserved.
#[derive(Clone, Debug, Serialize)]
pub struct RunHeader {
    /// User-visible run id (directory name or CLI-chosen label).
    pub run_id: String,
        /// Mode discriminator (`"tabular"` | `"rl"`) at the header ROOT; legacy headers
        /// lack it and `from_json` derives it from `config.mode`.
    #[serde(default)]
    pub engine_mode: Option<String>,
        /// Single deterministic root for the whole run; generated and recorded when no seed
        /// is given, so an interrupted run can be resumed exactly.
    pub run_seed: u64,
    /// Fitness label used for ranking (e.g. `"f1"`, `"cross_entropy"`).
    pub fitness_label: FitnessLabel,
    /// Whether lower or higher fitness is better.
    pub fitness_direction: Direction,
    /// Dataset dims the run was launched with (informational; the stream
    /// derives its pool from the actual dataset on disk).
    pub input_dim: usize,
    /// Output dimension the run was built for.
    pub output_dim: usize,
    /// Topology template + GP-pool knobs the run was launched with.
    pub topology_options: TopologyOptions,
    /// Hidden-dim sampling range.
    pub hidden_dim_pool: String,
    /// Hidden-dim stride used when sampling.
    pub hidden_dim_stride: usize,
    /// Combine/activation/standardize pools actually used by the run.
    pub combine_op_pool: Vec<String>,
    /// Activation pool the run drew from.
    pub activation_pool: Vec<String>,
    /// Standardize-op pool the run drew from.
    pub standardize_op_pool: Vec<String>,
        /// Informative (non-ranking) metrics configured for the run; their labels determine
        /// the extra per-net metric columns.
    pub informative_metrics: Vec<String>,
        /// Stop criteria / budgets. The scheduler honors them and `config` carries the
        /// authoritative copy; the stopped-by fact is in `history.csv` / the stop banner.
    pub max_steps: Option<usize>,
    /// When the run was started (wall clock), for human-readability.
    pub started_at: Option<String>,
    /// Train/eval split ratio recorded for the run.
    pub train_eval_split_ratio: Option<f32>,
        /// Cumulative culls at the last stop — restored on resume so the cull budget
        /// continues. `0` on legacy files.
    #[serde(default)]
    pub culls: usize,
        /// Accumulated wall-clock seconds at the last stop — restored on resume as the base
        /// offset. `0` on legacy files.
    #[serde(default)]
    pub run_elapsed_secs: u64,
    /// Children born per step clock at the last stop (same-clock siblings get
    /// distinct child seeds). Restored on resume; empty on legacy files.
    #[serde(default)]
    pub children_born_at_clock: std::collections::HashMap<usize, usize>,
        /// Data layout: `None` = single-pool seeded split; `Some(n)` = explicit `{train,test}`
        /// with `n` test rows after the train rows. Recorded so resume rebuilds the pools.
    #[serde(default)]
    pub explicit_test_rows: Option<usize>,
    /// The run's complete configuration (see [`ConfigSnapshot`]). This is the
    /// authoritative settings record — `engine.json` alone reproduces a run.
    #[serde(default)]
    pub config: ConfigSnapshot,
        /// The training scheme's self-description from `Trainer::describe()` — persisted
        /// verbatim, never interpreted. `None` on legacy files.
    #[serde(default)]
    pub trainer: Option<serde_json::Value>,
        /// The post-race guardrail's certified champion. Written after the race; preserved by
        /// `refresh_header`/resume.
    #[serde(default)]
    pub guardrail: Option<GuardrailRecord>,
    /// The previous experiment this run's founder batch was seeded from (see
    /// [`SeedSourceRecord`]). `None` = an all-random + user-blueprint batch.
    #[serde(default)]
    pub run_topology_source: Option<SeedSourceRecord>,
    /// What the run DID to its population (see [`RunAccounting`]). Written at
    /// stop; absent on a running or legacy run.
    #[serde(default)]
    pub accounting: Option<RunAccounting>,
}

/// The run's tally of what its evolution actually did — births, deaths and what the roll
/// batches bought. Counters of the WRITING process; the frontier is the cross-session record.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunAccounting {
    /// Every net this process created: the initial batch plus each insert.
    pub born: usize,
    /// Initial-batch split: random founders vs nets inherited from a previous
    /// run's `elites.json` (`created_from = seed:<run>:<hash>`).
    pub founders: usize,
    /// Founders that came from an inherited run.
    pub inherited_founders: usize,
    /// Nets born from an evolution roll, by channel.
    pub crossover_children: usize,
    /// Random immigrants inserted.
    pub immigrants: usize,
    /// Culls by the slot they freed (`crossover` / `immigrant-slot`).
    pub culls: usize,
    /// Culls that freed a slot for a crossover child.
    pub culls_crossover: usize,
    /// Culls that freed a slot for a mutation immigrant.
    pub culls_immigrant: usize,
        /// Crossover roll batch: rolls offered/fired, attempts made, rolls discarded after the
        /// retry budget.
    pub crossover_rolls: usize,
    /// Crossover rolls that fired.
    pub crossover_fired: usize,
    /// Crossover generate attempts made.
    pub crossover_attempts: usize,
    /// Crossover rolls discarded after the retry budget.
    pub crossover_discarded: usize,
    /// Mutation roll batch: rolls offered, rolls that fired and built an
    /// immigrant, and rolls the hard shield refused (no cull-eligible victim).
    pub mutation_rolls: usize,
    /// Mutation rolls that fired.
    pub mutation_fired: usize,
    /// Mutation rolls skipped by the shield (no cullable victim).
    pub mutation_blocked: usize,
        /// Clocks whose whole roll batch was forgone because no live net was cull-eligible.
    pub deferred_clocks: usize,
}

/// Everything that defines a step-race run's reproducible configuration —
/// the field-by-field input to [`RunHeader::from_race_options`].
#[derive(Debug)]
pub struct RunConfig {
    /// Timestamped run id (also the run dir name).
    pub run_id: String,
    /// Mode discriminator (`"tabular"` | `"rl"`) — see [`RunHeader::engine_mode`].
    pub engine_mode: Option<String>,
    /// Deterministic seed for the whole run.
    pub run_seed: u64,
    /// Ranking fitness label.
    pub fitness_label: FitnessLabel,
    /// Ranking direction (maximize/minimize).
    pub fitness_direction: Direction,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// The topology template options.
    pub topology_options: TopologyOptions,
    /// Hidden-dim pool the run sampled from.
    pub hidden_dim_pool: std::ops::RangeInclusive<usize>,
    /// Hidden-dim stride.
    pub hidden_dim_stride: usize,
    /// Combine-op pool.
    pub combine_op_pool: Vec<String>,
    /// Activation pool.
    pub activation_pool: Vec<String>,
    /// Standardize-op pool.
    pub standardize_op_pool: Vec<String>,
    /// Informative (non-ranking) metrics.
    pub informative_metrics: Vec<Metric>,
    /// Step budget as configured (`None` = no limit).
    pub max_steps: Option<usize>,
    /// Train/eval split ratio.
    pub train_eval_split_ratio: Option<f32>,
    /// See [`RunHeader::explicit_test_rows`].
    pub explicit_test_rows: Option<usize>,
    /// Cumulative culls at the last stop.
    pub culls: usize,
    /// Accumulated wall-clock seconds at the last stop.
    pub run_elapsed_secs: u64,
    /// Children born per clock (persisted counter).
    pub children_born_at_clock: std::collections::HashMap<usize, usize>,
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
            engine_mode,
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
            explicit_test_rows,
            culls,
            run_elapsed_secs,
            children_born_at_clock,
            config,
            trainer,
        } = cfg;
        RunHeader {
            run_id,
            engine_mode,
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
            explicit_test_rows,
            culls,
            run_elapsed_secs,
            children_born_at_clock,
            config,
            trainer,
                // Both are post-construction facts: the guardrail hasn't run at launch, and
                // the seed source is stamped in right after the population is resolved.
            guardrail: None,
            run_topology_source: None,
            accounting: None,
        }
    }

        /// Record the guardrail's certified champion — called by `CoreEngine::guardrail` so
        /// every later header rewrite keeps it.
    pub fn set_guardrail(&mut self, record: Option<GuardrailRecord>) {
        self.guardrail = record;
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
        let explicit_test_rows = v
            .get("explicit_test_rows")
            .and_then(|f| f.as_u64())
            .map(|n| n as usize);
        // Counter persistence (written at stop; legacy files read as fresh).
        let culls = v.get("culls").and_then(|f| f.as_u64()).unwrap_or(0) as usize;
        let run_elapsed_secs = v
            .get("run_elapsed_secs")
            .and_then(|f| f.as_u64())
            .unwrap_or(0);
        let children_born_at_clock: std::collections::HashMap<usize, usize> = v
            .get("children_born_at_clock")
            .and_then(|c| c.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, val)| {
                        let clock: usize = k.parse().ok()?;
                        let n = val.as_u64()? as usize;
                        Some((clock, n))
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Missing on legacy engine.json files ⇒ Default (all-zero snapshot).
        let config: ConfigSnapshot = v
            .get("config")
            .cloned()
            .and_then(|c| serde_json::from_value(c).ok())
            .unwrap_or_default();
        // Trainer self-description (see `Trainer::describe`). Absent on legacy
        // files and on trainers that don't implement the hook.
        let trainer = v.get("trainer").cloned();
        // Post-race facts. Absent on a running run, on legacy files, and on
        // runs whose guardrail never scored a champion.
        let guardrail: Option<GuardrailRecord> = v
            .get("guardrail")
            .cloned()
            .and_then(|g| serde_json::from_value(g).ok());
        let run_topology_source: Option<SeedSourceRecord> = v
            .get("run_topology_source")
            .cloned()
            .and_then(|s| serde_json::from_value(s).ok());
            // Migration: legacy headers have no root `engine_mode` — derive it from
            // `config.mode`. New headers carry it so a reader picks the engine first.
        let engine_mode = v
            .get("engine_mode")
            .and_then(|m| m.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                v.get("config")
                    .and_then(|c| c.get("mode"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            });
        Ok(RunHeader {
            run_id: v
                .get("run_id")
                .and_then(|f| f.as_str())
                .unwrap_or("unknown")
                .to_string(),
            engine_mode,
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
            explicit_test_rows,
            culls,
            run_elapsed_secs,
            children_born_at_clock,
            config,
            trainer,
            guardrail,
            run_topology_source,
            accounting: v
                .get("accounting")
                .cloned()
                .and_then(|x| serde_json::from_value(x).ok()),
        })
    }
}

/// Serde default: a net with no `alive` field is alive.
fn default_alive() -> bool {
    true
}

/// Serde default: `true`.
fn default_true() -> bool {
    true
}

    /// Pre-knob headers carry no `guardrail_matches`; anchor the serde fallback on the
    /// config's own default rather than 0.
fn default_guardrail_matches() -> usize {
    crate::engine::config::DEFAULT_GUARDRAIL_MATCHES
}

// ── Per-net state (nets/<hash>.json) ──────────────────────────────────────

/// The last-known state of one live net, written to `nets/<hash>.json` and overwritten each
/// surviving step. The resume anchor: rebuild from the blueprint + seed, then replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetState {
    /// Canonical xxh3 hash of this individual's topology (16 hex chars).
    pub hash: String,
        /// The topology blueprint, serialized (self-describing: `Network::build` reads the
        /// seed and dropout from here).
    pub topology: String,
        /// Weight-init seed, stored redundantly so a reader needn't re-parse the topology.
    pub net_seed: usize,
        /// The step this net has trained/evaluated through; a resume replays up to it, then
        /// continues.
    pub step: usize,
    /// Whether this net is currently alive in the active population. Culled nets
    /// are written as final tombstones to disk with `is_alive: false`.
    #[serde(default = "default_alive")]
    pub is_alive: bool,
        /// Step this net was culled (`None` while alive). With `cull_reason` and
        /// `final_smoothed_fitness` it makes a complete tombstone.
    #[serde(default)]
    pub culled_at_step: Option<usize>,
    /// Why it left (`"crossover"` | `"immigrant-slot"`).
    #[serde(default)]
    pub cull_reason: Option<String>,
        /// Its smoothed fitness at the moment of culling — the READABLE window mean (e.g. the
        /// turns it survived), NOT the risk-adjusted score that ranked it.
    #[serde(default)]
    pub final_smoothed_fitness: Option<f32>,
        /// Spread of that window (population std of the step readings). With the mean it tells
        /// "always that good" from "one lucky step" at a glance; `None` on legacy tombstones.
    #[serde(default)]
    pub final_fitness_std: Option<f32>,
        /// The RISK-ADJUSTED score (FIT: `mean ∓ λσ`) the net was actually ranked by at the
        /// moment of culling. Equals `final_smoothed_fitness` when λ = 0.
    #[serde(default)]
    pub final_risk_score: Option<f32>,
        /// Step readings behind the three values above. `0` on legacy tombstones.
    #[serde(default)]
    pub final_fitness_samples: usize,
    /// When this individual entered the live population. For lifetime tracking
    /// and the robustness-equivalent (appearances / span / final score).
    pub entered_at_step: usize,
        /// Optional lineage record: which parents/mutation produced this individual.
    pub created_from: Option<String>,
        /// Last-seen metrics snapshot (loss + fitness + informative metrics) — enough for a
        /// resume/time-travel read. Empty until the first persisted step.
    pub last_metrics: Option<NetMetrics>,
        /// Inclusive clock ranges this net executed as FROZEN (act-and-measure). Replay-
        /// relevant: resume re-runs them through a no-op optimizer. Run-length, disjoint.
    #[serde(default)]
    pub frozen_spans: Vec<(usize, usize)>,
        /// Rich per-individual metadata (params, dims, regime, run context). Missing on
        /// legacy JSONs; backfilled on first rewrite.
    #[serde(default)]
    pub meta: NetMeta,
        /// When this net's weights were last updated with a real optimizer. Freeze-log only,
        /// never written to disk; rebuilt from `last_metrics` on resume.
    #[serde(skip)]
    pub last_update_step: usize,
        /// Elite probation clock: starts at 0 on birth and any seat exit, ticked +1 per clock;
        /// a seat needs `elite_probation_age >= elite_join_probation_steps`.
    #[serde(default)]
    pub elite_probation_age: usize,
        /// Elite seat clock: 0 while unseated, +1 per seated clock; at
        /// `elite_crown_lifetime_steps` the net ages out (`0` disables).
    #[serde(default)]
    pub elite_seat_age: usize,
}

/// Rich metadata for one individual (embedded in `nets/<hash>.json` under `meta`) — enough
/// to understand the net and its training conditions without touching code.
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
    /// Output dimension (`None` in modes without a shared stream).
    pub output_dim: Option<usize>,
        /// Shared training regime. `None` in modes with no shared stream (RL) — LR, grad clip
        /// and volume are trainer-owned (`engine.json` → `"trainer"`).
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Dropout probability stamped into the topology (also lives in the
    /// topology JSON; duplicated here for quick scanning).
    #[serde(default)]
    pub dropout_prob: Option<f32>,
        /// Ranking fitness label. No `loss_label` beside it: the loss/update scheme is
        /// trainer-owned, so `engine.json` → `"trainer"` is the only honest record.
    #[serde(default)]
    pub fitness_label: Option<String>,
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
    /// Run seed recorded for this net's regime (`None` if unknown).
    pub run_seed: Option<u64>,
}

/// One snapshot of metrics for a single net, at a single step.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NetMetrics {
    /// Step this metric snapshot was recorded at.
    pub step: usize,
    /// Training loss for this step's train batch.
    pub train_loss: f32,
    /// Held-out loss for this step's eval batch (eval mode).
    pub eval_loss: Option<f32>,
    /// Ranking fitness for this step's eval batch (the signal the run evolves on).
    pub fitness: f32,
    /// Informative-only metric values, in the same order the run configured them.
    pub informative: Vec<f32>,
        /// True when this step was an ACT-AND-MEASURE pass (frozen elite). Replay-relevant:
        /// resume re-runs it through a no-op optimizer.
    #[serde(default)]
    pub frozen: bool,
}

impl NetState {
        /// Stamp the `meta` block from the materialized network + run context — once per net
        /// right after its `Network` is built, never per step.
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
        self.meta.batch_size = run.batch_size;
            // LR, grad clip and the loss scheme are trainer-owned, so the engine records none
            // of them here; the trainer's `describe()` block is the authoritative record.
        self.meta.fitness_label = Some(run.fitness_label.clone());
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
            final_fitness_std: None,
            final_risk_score: None,
            final_fitness_samples: 0,
            entered_at_step: step,
            created_from,
            last_metrics: None,
            frozen_spans: Vec::new(),
            meta: NetMeta::default(),
            last_update_step: step,
            elite_probation_age: 0,
            elite_seat_age: 0,
        })
    }

        /// Record clock `step` as an act-and-measure (frozen) step, extending the last span
        /// when contiguous. Does NOT advance `last_update_step`.
    pub fn record_frozen_step(&mut self, step: usize) {
        match self.frozen_spans.last_mut() {
            Some((_, end)) if *end + 1 == step => *end = step,
            _ => self.frozen_spans.push((step, step)),
        }
    }

    /// Whether `step` lies inside a recorded frozen span.
    pub fn is_frozen_step(&self, step: usize) -> bool {
        self.frozen_spans
            .iter()
            .any(|&(s, e)| step >= s && step <= e)
    }

        /// Serialize to pretty JSON for `nets/<hash>.json`, backfilling `meta` on legacy
        /// records with a throwaway network build (cheap, once per write flight).
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

        /// Advance the local step clock and record it as the last weight-update step — the
        /// LIVE-path hook (a frozen step uses the separate freeze-only path).
    pub fn advance_step(&mut self) {
        self.step += 1;
        self.last_update_step = self.step;
    }

    /// Recover the topology from the embedded JSON.
    pub fn topology(&self) -> Result<Topology> {
        Ok(Topology::from_json(&self.topology)
            .map_err(|e| EngineError::Json(format!("net state topology parse: {e}")))?)
    }
}

// ── Runtime population state (in memory) ──────────────────────────────────

/// The in-memory view of the live population the scheduler reads/writes: a hash map of
/// `NetState` plus a per-topology lifetime accumulator (the robustness-equivalent).
pub struct RaceState {
    /// Live nets, indexed by canonical topology hash.
    nets: HashMap<String, NetState>,
    /// Per-topology lifetime/score accumulator (robustness-equivalent).
    topology_lifetime: HashMap<String, TopologyLifetime>,
}

        /// Per-topology lifetime tracking — entries, step span, and final recorded fitness.
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
        /// New accumulator seeded from the first entry.
    pub fn new(first_step: usize, fitness: Option<f32>) -> Self {
        TopologyLifetime {
            appearances: 1,
            first_step,
            last_step: first_step,
            last_fitness: fitness,
        }
    }

        /// Record one entry at `step` with `fitness`.
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

        /// How many times this topology entered the population.
    pub fn appearances(&self) -> usize {
        self.appearances
    }

        /// First step it was seen.
    pub fn first_step(&self) -> usize {
        self.first_step
    }

        /// Last step it was seen.
    pub fn last_step(&self) -> usize {
        self.last_step
    }

        /// Steps covered from first to last.
    pub fn span(&self) -> usize {
        self.last_step.saturating_sub(self.first_step)
    }

        /// Final recorded fitness.
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
        /// New empty population state.
    pub fn new() -> Self {
        RaceState {
            nets: HashMap::new(),
            topology_lifetime: HashMap::new(),
        }
    }

    /// Register a newly inserted net (after initial insertion).
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

        /// Record an ACT-AND-MEASURE (frozen) step: metrics, lifetime progress and the frozen
        /// span — but WITHOUT advancing the clock. Counterpart to [`Self::record_step`].
    pub fn record_frozen_act(&mut self, hash: &str, metrics: NetMetrics) -> Result<()> {
        let state = self.nets.get_mut(hash).ok_or_else(|| {
            EngineError::InvalidOptions(format!("race state: net {hash} not live"))
        })?;
        state.record_metrics(metrics.clone());
        // The span is how replay_step re-runs this clock through the
        // no-op optimizer on resume.
        state.record_frozen_step(metrics.step);
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

        /// Immutable access to a live net by hash.
    pub fn net(&self, hash: &str) -> Option<&NetState> {
        self.nets.get(hash)
    }

        /// Mutable access to a live net by hash.
    pub fn net_mut(&mut self, hash: &str) -> Option<&mut NetState> {
        self.nets.get_mut(hash)
    }

        /// Hashes of every live (non-tombstoned) net.
    pub fn live_hashes(&self) -> Vec<String> {
        let mut hashes: Vec<String> = self.nets.keys().cloned().collect();
        hashes.sort();
        hashes
    }

        /// Number of live nets.
    pub fn live_count(&self) -> usize {
        self.nets.len()
    }

        /// Mutable access to every live net keyed by hash — for the commutative counter tick,
        /// so iteration order doesn't matter.
    pub fn live_mut(&mut self) -> impl Iterator<Item = (&String, &mut NetState)> {
        self.nets.iter_mut()
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

/// Stamp the post-race guardrail record into an EXISTING `engine.json` via raw read-modify-
/// write, so unmodeled fields and hand-edited notes survive. Missing file errors.
pub fn write_guardrail_record(dir: &Path, record: &GuardrailRecord) -> Result<()> {
    stamp_engine_json_key(
        dir,
        "guardrail",
        serde_json::to_value(record)
            .map_err(|e| EngineError::Json(format!("guardrail record serialize: {e}")))?,
    )
}

/// Insert/replace one key in `engine.json`, preserving everything else
/// verbatim. Shared by the post-race stamps.
fn stamp_engine_json_key(dir: &Path, key: &str, value: serde_json::Value) -> Result<()> {
    let path = dir.join("engine.json");
    let raw = std::fs::read_to_string(&path).map_err(|source| EngineError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut doc: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| EngineError::Json(format!("engine.json parse: {e}")))?;
    let obj = doc
        .as_object_mut()
        .ok_or_else(|| EngineError::Json("engine.json is not a JSON object".into()))?;
    obj.insert(key.to_string(), value);
    let json = serde_json::to_string_pretty(&doc)
        .map_err(|e| EngineError::Json(format!("engine.json serialize: {e}")))?;
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
            explicit_test_rows: None,
            run_id: run_id.to_string(),
            engine_mode: Some("tabular".into()),
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
            informative_metrics: vec![Metric::custom("dummy", |_p, _y| Ok(0.0))],
            max_steps: Some(10_000),
            train_eval_split_ratio: Some(0.2),
            culls: 0,
            run_elapsed_secs: 0,
            children_born_at_clock: std::collections::HashMap::new(),
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

        /// FIT legacy contract: a run recorded BEFORE the λ knob carries no
        /// `risk_aversion` in its `engine.json` — and it ranked on the PLAIN MEAN, so the
        /// serde default must be `0.0`, NOT the new default `1.0`. Anything else would
        /// silently re-rank a resumed legacy run against its recorded history.
    #[test]
    fn legacy_config_snapshot_defaults_risk_aversion_to_the_plain_mean() {
        let snapshot = ConfigSnapshot::from_config(&crate::engine::config::RaceConfig::defaults(), None, None);
        assert_eq!(snapshot.risk_aversion, crate::engine::config::RISK_AVERSION);
        // Round-trips at the new default…
        let json = serde_json::to_string(&snapshot).unwrap();
        let back: ConfigSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.risk_aversion, crate::engine::config::RISK_AVERSION);
        // …and a value JSON with the key REMOVED (exactly a pre-FIT engine.json) loads
        // as λ = 0.0 while every other knob still comes through.
        let mut legacy: serde_json::Value = serde_json::from_str(&json).unwrap();
        legacy.as_object_mut().unwrap().remove("risk_aversion");
        let loaded: ConfigSnapshot = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            loaded.risk_aversion, 0.0,
            "a pre-FIT run ranked on the plain mean"
        );
        assert_eq!(loaded.smoothing_window, snapshot.smoothing_window);
        assert_eq!(loaded.pop_size, snapshot.pop_size);
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
            frozen: false,
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

        /// Schema separation: run-level facts in the header's `config`, per-net facts in
        /// `nets/<hash>.json` — neither invents a value for a mode that has none.
    #[test]
    fn schema_records_mode_appropriate_geometry_and_no_trainer_owned_fields() {
        // RL header: no stream, so geometry is genuinely absent (`null`), and
        // the step budget sits beside the fitness target in `config`.
        let rl_cfg = crate::engine::RaceConfig {
            mode: crate::engine::config::RunMode::Rl,
            max_steps: Some(25),
            ..Default::default()
        };
        let rl = ConfigSnapshot::from_config(&rl_cfg, None, None);
        assert_eq!(rl.batch_size, None);
        assert_eq!(rl.eval_batch_size, None);
        assert_eq!(
            rl.max_steps,
            Some(25),
            "step budget recorded with the stop criteria"
        );

        // Tabular header: the stream shape the run actually used is recorded.
        let tab =
            ConfigSnapshot::from_config(&crate::engine::RaceConfig::default(), Some(16), Some(128));
        assert_eq!(tab.batch_size, Some(16));
        assert_eq!(tab.eval_batch_size, Some(128));

        // In the written header, absent means absent — not a 16 default.
        let dir = fresh_dir();
        let mut header = baseline_header("rl-run");
        header.config = rl;
        write_engine_json(&dir, &header).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("engine.json")).unwrap())
                .unwrap();
        assert_eq!(v["config"]["batch_size"], serde_json::Value::Null);
        assert_eq!(v["config"]["max_steps"], serde_json::json!(25));

        // Per-net meta follows the same rule, and carries no trainer-owned knob.
        let topo = tiny_topology();
        let net = crate::graph::network::Network::build(&topo, crate::Device::CPU).unwrap();
        let mut state = NetState::new(&topo, 0, None).unwrap();
        state.stamp_meta(&net, &crate::engine::config::RunMetaCtx::default());
        assert_eq!(state.meta.batch_size, None);
        let net_json: serde_json::Value = serde_json::from_str(&state.to_json().unwrap()).unwrap();
        for absent in ["loss_label", "learning_rate", "grad_clip"] {
            assert!(
                net_json["meta"].get(absent).is_none(),
                "`{absent}` is trainer-owned (engine.json → trainer), not per-net schema"
            );
        }
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
            frozen: false,
        });
        write_net_state(&dir, &state).unwrap();

        state.advance_step();
        state.record_metrics(NetMetrics {
            step: 1,
            train_loss: 1.5,
            eval_loss: Some(1.6),
            fitness: 0.2,
            informative: vec![],
            frozen: false,
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
                frozen: false,
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
                frozen: false,
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
