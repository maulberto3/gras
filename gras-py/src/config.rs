//! `RaceConfig`: the engine knobs a Python caller turns.
//!
//! The defaults ARE the crate's (`RaceConfig::defaults()`), with one documented
//! exception — `elite_previous_load`, forced off so a Python run never inherits
//! elites from whatever `results/` happens to sit in the caller's cwd. Anything
//! this class does not expose still inherits the crate default, because
//! [`PyRaceConfig::into_inner`] starts from `RaceConfig::defaults()`.

use std::path::PathBuf;

use gras::engine::config::{DEFAULT_GUARDRAIL_MATCHES, SMOOTHING_WINDOW};
use gras::evolution::selection::SelectionMethod;
use gras::graph::topology::FeatureSplitPolicy;
use gras::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Defaults that `RaceConfig::defaults()` sets to a literal, mirrored here so the
// `#[pyo3(signature = …)]` below can name them. `defaults_mirror_the_crate` pins
// every one of them against the crate, so a changed crate default fails a test
// instead of silently re-tuning Python.
const DEFAULT_POP_SIZE: usize = 5;
const DEFAULT_CROSSOVER_ROLLS: usize = 1;
const DEFAULT_MUTATE_ROLLS: usize = 1;
const DEFAULT_CROSSOVER_PROB: f32 = 0.5;
const DEFAULT_MUTATE_PROB: f32 = 0.5;
const DEFAULT_HIDDEN_DIM_STRIDE: usize = 16;
const DEFAULT_TOURNAMENT_SIZE: usize = 2;

/// λ and Ω default to the values that change NOTHING about a ranking: the plain
/// window mean (λ = 0) and the plain step (Ω = 0). The crate's λ default is 1.0
/// — FIT charges a net for its own spread — so that is an opt-in here rather
/// than something a Python caller inherits without asking. Ω is 0.0 in the
/// crate too, so only λ is a deliberate divergence.
const DEFAULT_RISK_AVERSION: f32 = 0.0;
const DEFAULT_TRAIN_MERIT: f32 = 0.0;

/// Engine knobs a Python caller turns. Anything not listed keeps the crate's
/// default, and every field is recorded in the run's `engine.json`.
///
/// Two choices are the binding's own, both to keep a Python run neutral until it
/// asks otherwise: `risk_aversion` is `0.0` (rank on the plain mean, no charge
/// for spread) and `elite_previous_load` is `false` (all-random founders).
#[pyclass(name = "RaceConfig", module = "gras._core")]
#[derive(Clone)]
pub struct PyRaceConfig {
    // ── population + ranking ────────────────────────────────────────────
    #[pyo3(get, set)]
    pub pop_size: usize,
    #[pyo3(get, set)]
    pub elite_count: usize,
    #[pyo3(get, set)]
    pub max_steps: Option<usize>,
    #[pyo3(get, set)]
    pub smoothing_window: usize,
    /// Risk aversion λ: window standard deviations subtracted from a net's
    /// smoothed mean before ranking. **`0.0` here** ranks on the plain mean (the
    /// crate's FIT default is `1.0`).
    #[pyo3(get, set)]
    pub risk_aversion: f32,
    /// Train/eval lottery strength Ω: how much of a clock's budget merit decides;
    /// **`0.0` = off** — the coin is never drawn and every non-crown net runs.
    #[pyo3(get, set)]
    pub train_merit: f32,
    /// Per-step reporting shape, the same vocabulary as the examples'
    /// `--log-level`: `summ` (default), `minimal`, `none`, `debug`, `trace`.
    /// Recorded in `engine.json`, so a run's verbosity is part of its record.
    #[pyo3(get, set)]
    pub log_level: String,

    // ── evolution ───────────────────────────────────────────────────────
    /// Crossover rolls per step (each fires with `crossover_prob`; a spent roll
    /// has no random fallback).
    #[pyo3(get, set)]
    pub crossover_rolls: usize,
    /// Mutation rolls per step (each fires with `mutate_prob` and inserts a
    /// random immigrant by evicting one net).
    #[pyo3(get, set)]
    pub mutate_rolls: usize,
    #[pyo3(get, set)]
    pub crossover_prob: f32,
    #[pyo3(get, set)]
    pub mutate_prob: f32,
    /// The top-k nets act and measure without weight updates.
    #[pyo3(get, set)]
    pub freeze_elites: bool,
    /// Who a surviving crossover child evicts: `worst` (default) or `random`.
    #[pyo3(get, set)]
    pub crossover_cull_policy: String,
    /// Who a firing mutation roll evicts: `worst` (default), `inverse_fitness`,
    /// or `random`.
    #[pyo3(get, set)]
    pub mutation_cull_policy: String,
    /// How a crossover attempt draws parents: `roulette` (default) or
    /// `tournament` (K-way; see `tournament_size`).
    #[pyo3(get, set)]
    pub crossover_selection: String,
    /// K for `crossover_selection="tournament"`; small K is mild pressure.
    #[pyo3(get, set)]
    pub tournament_size: usize,

    // ── search space ────────────────────────────────────────────────────
    /// Hidden-dim sampling range as `(min, max)` inclusive. `None` keeps the
    /// crate's `4..=8`; a one-width range (`(16, 16)`) fixes the hidden width.
    #[pyo3(get, set)]
    pub hidden_dim_pool: Option<(usize, usize)>,
    /// Stride within `hidden_dim_pool` when sampling node dimensions.
    #[pyo3(get, set)]
    pub hidden_dim_stride: usize,
    /// Activation pool; empty means all known activations.
    #[pyo3(get, set)]
    pub activation_pool: Vec<String>,
    /// Combine-op pool; empty means all.
    #[pyo3(get, set)]
    pub combine_op_pool: Vec<String>,
    /// Standardize-op pool; empty means all.
    #[pyo3(get, set)]
    pub standardize_op_pool: Vec<String>,
    /// How the Input node's per-port feature masks are drawn: `None` (default)
    /// for no split at all, `random` (independent per-port coins) or `exclusive`
    /// (a true partition).
    #[pyo3(get, set)]
    pub feature_split: Option<String>,

    // ── artifacts ───────────────────────────────────────────────────────
    /// Write `history.csv` and `attempts.csv`.
    #[pyo3(get, set)]
    pub csv_export: bool,
    /// Write `telemetry.jsonl` (one JSON record per event, full fidelity).
    #[pyo3(get, set)]
    pub trace_file: bool,
    /// Flush `history.csv` after every step instead of at checkpoints.
    #[pyo3(get, set)]
    pub history_flush_each: bool,
    #[pyo3(get, set)]
    pub elite_save_topology: bool,
    #[pyo3(get, set)]
    pub elite_save_safetensors: bool,
    /// Also save the bottom `elite_count` nets (the anti-champion).
    #[pyo3(get, set)]
    pub worst_save_topology: bool,
    #[pyo3(get, set)]
    pub worst_save_safetensors: bool,
    /// Human label recorded in `engine.json` as `"run_name"`.
    #[pyo3(get, set)]
    pub run_name: Option<String>,

    // ── run-to-run inheritance + guardrail ──────────────────────────────
    /// Load the newest compatible previous run's guardrail elites as topologies.
    /// **Off here, unlike the crate** (`true`), so a Python run is all-random
    /// unless the caller asks: the crate's default looks in `results/` relative
    /// to the cwd, which for a notebook is whatever the kernel was started in.
    #[pyo3(get, set)]
    pub elite_previous_load: bool,
    /// Where `elite_previous_load` looks: a run dir or a dir of run dirs.
    #[pyo3(get, set)]
    pub elite_previous_dir: Option<PathBuf>,
    /// Fresh holdout games the post-race guardrail plays per elite; `0` skips it.
    #[pyo3(get, set)]
    pub guardrail_matches: usize,
}

#[pymethods]
impl PyRaceConfig {
    #[new]
    #[pyo3(signature = (
        // population + ranking
        pop_size = DEFAULT_POP_SIZE,
        elite_count = 1,
        max_steps = None,
        smoothing_window = SMOOTHING_WINDOW,
        risk_aversion = DEFAULT_RISK_AVERSION,
        train_merit = DEFAULT_TRAIN_MERIT,
        elite_previous_load = false,
        log_level = String::from("summ"),
        // evolution
        crossover_rolls = DEFAULT_CROSSOVER_ROLLS,
        mutate_rolls = DEFAULT_MUTATE_ROLLS,
        crossover_prob = DEFAULT_CROSSOVER_PROB,
        mutate_prob = DEFAULT_MUTATE_PROB,
        freeze_elites = true,
        crossover_cull_policy = String::from("worst"),
        mutation_cull_policy = String::from("worst"),
        crossover_selection = String::from("roulette"),
        tournament_size = DEFAULT_TOURNAMENT_SIZE,
        // search space
        hidden_dim_pool = None,
        hidden_dim_stride = DEFAULT_HIDDEN_DIM_STRIDE,
        activation_pool = Vec::new(),
        combine_op_pool = Vec::new(),
        standardize_op_pool = Vec::new(),
        feature_split = None,
        // artifacts
        csv_export = true,
        trace_file = false,
        history_flush_each = false,
        elite_save_topology = true,
        elite_save_safetensors = true,
        worst_save_topology = false,
        worst_save_safetensors = false,
        run_name = None,
        // inheritance + guardrail
        elite_previous_dir = None,
        guardrail_matches = DEFAULT_GUARDRAIL_MATCHES,
    ))]
    // A config's kwargs ARE its API: bundling them into an object would buy the
    // lint nothing and cost the caller a line per knob. Explicitly out of scope
    // in TODO's Stage E too.
    #[allow(clippy::too_many_arguments)]
    fn new(
        pop_size: usize,
        elite_count: usize,
        max_steps: Option<usize>,
        smoothing_window: usize,
        risk_aversion: f32,
        train_merit: f32,
        elite_previous_load: bool,
        log_level: String,
        crossover_rolls: usize,
        mutate_rolls: usize,
        crossover_prob: f32,
        mutate_prob: f32,
        freeze_elites: bool,
        crossover_cull_policy: String,
        mutation_cull_policy: String,
        crossover_selection: String,
        tournament_size: usize,
        hidden_dim_pool: Option<(usize, usize)>,
        hidden_dim_stride: usize,
        activation_pool: Vec<String>,
        combine_op_pool: Vec<String>,
        standardize_op_pool: Vec<String>,
        feature_split: Option<String>,
        csv_export: bool,
        trace_file: bool,
        history_flush_each: bool,
        elite_save_topology: bool,
        elite_save_safetensors: bool,
        worst_save_topology: bool,
        worst_save_safetensors: bool,
        run_name: Option<String>,
        elite_previous_dir: Option<PathBuf>,
        guardrail_matches: usize,
    ) -> Self {
        Self {
            pop_size,
            elite_count,
            max_steps,
            smoothing_window,
            risk_aversion,
            train_merit,
            // OFF by default: a Python run must not silently inherit elites from
            // whatever `results/` happens to sit in the caller's cwd.
            elite_previous_load,
            // `summ` matches every Rust example's default: a race reports one
            // line per step instead of running silently.
            log_level,
            crossover_rolls,
            mutate_rolls,
            crossover_prob,
            mutate_prob,
            freeze_elites,
            crossover_cull_policy,
            mutation_cull_policy,
            crossover_selection,
            tournament_size,
            hidden_dim_pool,
            hidden_dim_stride,
            activation_pool,
            combine_op_pool,
            standardize_op_pool,
            feature_split,
            csv_export,
            trace_file,
            history_flush_each,
            elite_save_topology,
            elite_save_safetensors,
            worst_save_topology,
            worst_save_safetensors,
            run_name,
            elite_previous_dir,
            guardrail_matches,
        }
    }

    /// The headline knobs. The remaining fields are reachable by name and land in
    /// `engine.json`; printing 32 of them would bury the ones people change.
    fn __repr__(&self) -> String {
        format!(
            "RaceConfig(pop_size={}, elite_count={}, max_steps={:?}, smoothing_window={}, \
             risk_aversion={}, elite_previous_load={}, log_level={:?})",
            self.pop_size,
            self.elite_count,
            self.max_steps,
            self.smoothing_window,
            self.risk_aversion,
            self.elite_previous_load,
            self.log_level
        )
    }
}

/// `worst` | `random` — the spellings the crate records in `engine.json`.
fn parse_crossover_cull(name: &str) -> Option<CrossCullPolicy> {
    match name {
        "worst" => Some(CrossCullPolicy::Worst),
        "random" => Some(CrossCullPolicy::Random),
        _ => None,
    }
}

/// `worst` | `inverse_fitness` | `random`.
fn parse_mutation_cull(name: &str) -> Option<MutationCullPolicy> {
    match name {
        "worst" => Some(MutationCullPolicy::Worst),
        "inverse_fitness" => Some(MutationCullPolicy::InverseFitness),
        "random" => Some(MutationCullPolicy::Random),
        _ => None,
    }
}

/// `random` | `exclusive`, and the `split_*` spellings `engine.json` records.
fn parse_feature_split(name: &str) -> Option<FeatureSplitPolicy> {
    match name {
        "random" | "split_random" => Some(FeatureSplitPolicy::Random),
        "exclusive" | "split_exclusive" => Some(FeatureSplitPolicy::Exclusive),
        _ => None,
    }
}

impl PyRaceConfig {
    /// The crate config plus the tracing verbosity `logging::init` takes.
    ///
    /// Names are resolved here, before a single net is drawn, so a typo raises
    /// `ValueError` naming the offender and the valid set instead of changing how
    /// a mid-run child is wired.
    pub(crate) fn into_inner(self) -> PyResult<(gras::RaceConfig, Option<&'static str>)> {
        let (log_level, verbosity) = LogLevel::parse_verbose(&self.log_level).ok_or_else(|| {
            PyValueError::new_err(format!(
                "log_level {:?}: expected one of none, summ, minimal, debug, trace",
                self.log_level
            ))
        })?;
        let crossover_cull_policy =
            parse_crossover_cull(&self.crossover_cull_policy).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "crossover_cull_policy {:?}: expected one of worst, random",
                    self.crossover_cull_policy
                ))
            })?;
        let mutation_cull_policy =
            parse_mutation_cull(&self.mutation_cull_policy).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "mutation_cull_policy {:?}: expected one of worst, inverse_fitness, random",
                    self.mutation_cull_policy
                ))
            })?;
        let crossover_selection = match self.crossover_selection.as_str() {
            "roulette" => SelectionMethod::Roulette,
            "tournament" => {
                if self.tournament_size < 2 {
                    return Err(PyValueError::new_err(format!(
                        "tournament_size {}: a tournament needs at least 2 competitors",
                        self.tournament_size
                    )));
                }
                SelectionMethod::Tournament {
                    tournament_size: self.tournament_size,
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "crossover_selection {other:?}: expected one of roulette, tournament"
                )))
            }
        };
        let feature_split = match &self.feature_split {
            Some(name) => Some(parse_feature_split(name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "feature_split {name:?}: expected one of random, exclusive"
                ))
            })?),
            None => None,
        };
        // An empty or inverted range would panic the sampler (`end - start`), so
        // refuse it by name here rather than crash mid-race.
        let hidden_dim_pool = match self.hidden_dim_pool {
            Some((lo, hi)) if lo == 0 || lo > hi => {
                return Err(PyValueError::new_err(format!(
                    "hidden_dim_pool ({lo}, {hi}): expected 1 <= min <= max"
                )))
            }
            Some((lo, hi)) => Some(lo..=hi),
            None => None,
        };

        let mut cfg = gras::RaceConfig::defaults();
        cfg.pop_size = self.pop_size;
        cfg.elite_count = self.elite_count;
        cfg.max_steps = self.max_steps;
        cfg.smoothing_window = self.smoothing_window;
        cfg.risk_aversion = self.risk_aversion;
        cfg.train_merit = self.train_merit;
        cfg.elite_previous_load = self.elite_previous_load;
        cfg.log_level = log_level;
        cfg.crossover_rolls = self.crossover_rolls;
        cfg.mutate_rolls = self.mutate_rolls;
        cfg.crossover_prob = self.crossover_prob;
        cfg.mutate_prob = self.mutate_prob;
        cfg.freeze_elites = self.freeze_elites;
        cfg.crossover_cull_policy = crossover_cull_policy;
        cfg.mutation_cull_policy = mutation_cull_policy;
        cfg.crossover_selection = crossover_selection;
        if let Some(pool) = hidden_dim_pool {
            cfg.hidden_dim_pool = Some(pool);
        }
        cfg.hidden_dim_stride = self.hidden_dim_stride;
        cfg.activation_pool = self.activation_pool;
        cfg.combine_op_pool = self.combine_op_pool;
        cfg.standardize_op_pool = self.standardize_op_pool;
        cfg.feature_split = feature_split;
        cfg.csv_export = self.csv_export;
        cfg.trace_file = self.trace_file;
        cfg.history_flush_each = self.history_flush_each;
        cfg.elite_save_topology = self.elite_save_topology;
        cfg.elite_save_safetensors = self.elite_save_safetensors;
        cfg.worst_save_topology = self.worst_save_topology;
        cfg.worst_save_safetensors = self.worst_save_safetensors;
        cfg.run_name = self.run_name;
        cfg.elite_previous_dir = self.elite_previous_dir;
        cfg.guardrail_matches = self.guardrail_matches;

        // The crate resolves pool names; let IT name a typo, so the message and
        // the accepted set are the ones the engine would have used. (Left to the
        // first seed instead, a bad name silently falls back to "all" when a
        // crossover child rewires the graph.)
        cfg.resolved_activation_pool()
            .map_err(|e| PyValueError::new_err(format!("activation_pool: {e}")))?;
        cfg.resolved_combine_pool()
            .map_err(|e| PyValueError::new_err(format!("combine_op_pool: {e}")))?;
        cfg.resolved_standardize_pool()
            .map_err(|e| PyValueError::new_err(format!("standardize_op_pool: {e}")))?;
        Ok((cfg, verbosity))
    }
}

