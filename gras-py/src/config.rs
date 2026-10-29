//! `RaceConfig`: the engine knobs a Python caller turns.
//!
//! The defaults ARE the crate's (`RaceConfig::defaults()`), with one documented
//! exception — `elite_previous_load`, forced off so a Python run never inherits
//! elites from whatever `results/` happens to sit in the caller's cwd. Anything
//! this class does not expose still inherits the crate default, because
//! [`PyRaceConfig::into_inner`] starts from `RaceConfig::defaults()`.

use std::path::PathBuf;

use gras::engine::config::{
    CrossoverOp, DEFAULT_CHECKPOINT_EVERY, SMOOTHING_WINDOW,
};
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

/// Dropout the crate builds nets with (`TopologyOptions::default().dropout_prob`).
/// `0.0` keeps regularization opt-in per run rather than on by inheritance.
const DEFAULT_DROPOUT_PROB: f32 = 0.0;

/// Cull immunity for every newcomer, in clocks. The crate's shipped value, mirrored
/// here so the signature can name it. Unlike λ and Ω this is not a "neutral at
/// zero" knob: `0` removes a protection rather than switching off an effect on the
/// ranking, so a Python caller inherits the crate's 5 instead of starting bare.
const DEFAULT_PROBATION_STEPS: usize = 5;

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
    /// Which mixing methods a crossover attempt draws from: `"one_point"` and/or
    /// `"uniform"`. Empty (the default) means both, drawn uniformly.
    #[pyo3(get, set)]
    pub crossover_ops_pool: Vec<String>,

    // ── durability ──────────────────────────────────────────────────────
    /// Steps between population checkpoints (the era rotation and the `history.csv`
    /// flush cadence key off this). **`0` = no durability tick** and one eternal era.
    #[pyo3(get, set)]
    pub checkpoint_every: usize,

    // ── probation (cull immunity for newcomers) ─────────────────────────
    /// A random immigrant is cull-immune for its first k clocks; `0` = only the
    /// empty-buffer step-0 immunity.
    #[pyo3(get, set)]
    pub mutation_probation_steps: usize,
    /// A crossover-born net is cull-immune for its first k clocks; `0` = only step-0.
    #[pyo3(get, set)]
    pub crossover_probation_steps: usize,
    /// A net cannot become an elite until it has aged k clocks since arrival.
    #[pyo3(get, set)]
    pub elite_join_probation_steps: usize,
    /// How long a net may stay at peak on the frozen crown before age-elimination;
    /// `0` = permanent until its rank moves.
    #[pyo3(get, set)]
    pub elite_crown_lifetime_steps: usize,

    // ── search space ────────────────────────────────────────────────────
    /// Hidden-dim sampling range as `(min, max)` inclusive. `None` keeps the
    /// crate's `4..=8`; a one-width range (`(16, 16)`) fixes the hidden width.
    #[pyo3(get, set)]
    pub hidden_dim_pool: Option<(usize, usize)>,
    /// Stride within `hidden_dim_pool` when sampling node dimensions.
    #[pyo3(get, set)]
    pub hidden_dim_stride: usize,
    /// How many hidden NODES a sampled net gets, as `(min, max)` inclusive.
    /// `None` keeps the crate's `2..=5`. Distinct from `hidden_dim_pool`, which
    /// is how WIDE each of those nodes is.
    #[pyo3(get, set)]
    pub hidden_nodes_pool: Option<(usize, usize)>,
    /// Wires INTO each hidden node, `(min, max)` inclusive; `None` keeps the
    /// crate's `2..=5` (`set_topology_min/max_inputs_per_node`).
    #[pyo3(get, set)]
    pub hidden_inputs_pool: Option<(usize, usize)>,
    /// Wires OUT of each hidden node, `(min, max)` inclusive; `None` keeps the
    /// crate's `2..=5` (`set_topology_min/max_outputs_per_node`).
    #[pyo3(get, set)]
    pub hidden_outputs_pool: Option<(usize, usize)>,
    /// Dropout baked into each net at build; `0.0` (default) is off. TRAIN path
    /// only, so a replay does not depend on a random mask.
    #[pyo3(get, set)]
    pub dropout_prob: f32,
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
    #[pyo3(get, set)]
    pub elite_save_topology: bool,
    /// Also save the bottom `elite_count` nets' topologies (the anti-champion).
    #[pyo3(get, set)]
    pub worst_save_topology: bool,
    /// Human label recorded in `engine.json` as `"run_name"`.
    #[pyo3(get, set)]
    pub run_name: Option<String>,

    /// Load the newest compatible previous run's final-table elites as topologies.
    /// **Off here, unlike the crate** (`true`), so a Python run is all-random
    /// unless the caller asks: the crate's default looks in `results/` relative
    /// to the cwd, which for a notebook is whatever the kernel was started in.
    #[pyo3(get, set)]
    pub elite_previous_load: bool,
    /// Where `elite_previous_load` looks: a run dir or a dir of run dirs.
    #[pyo3(get, set)]
    pub elite_previous_dir: Option<PathBuf>,
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
        crossover_ops_pool = Vec::new(),
        // durability + probation
        checkpoint_every = DEFAULT_CHECKPOINT_EVERY,
        mutation_probation_steps = DEFAULT_PROBATION_STEPS,
        crossover_probation_steps = DEFAULT_PROBATION_STEPS,
        elite_join_probation_steps = DEFAULT_PROBATION_STEPS,
        elite_crown_lifetime_steps = DEFAULT_PROBATION_STEPS,
        // search space
        hidden_dim_pool = None,
        hidden_dim_stride = DEFAULT_HIDDEN_DIM_STRIDE,
        hidden_nodes_pool = None,
        hidden_inputs_pool = None,
        hidden_outputs_pool = None,
        dropout_prob = DEFAULT_DROPOUT_PROB,
        activation_pool = Vec::new(),
        combine_op_pool = Vec::new(),
        standardize_op_pool = Vec::new(),
        feature_split = None,
        // artifacts
        csv_export = true,
        trace_file = false,
        elite_save_topology = true,
        worst_save_topology = false,
        run_name = None,
        elite_previous_dir = None,
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
        crossover_ops_pool: Vec<String>,
        checkpoint_every: usize,
        mutation_probation_steps: usize,
        crossover_probation_steps: usize,
        elite_join_probation_steps: usize,
        elite_crown_lifetime_steps: usize,
        hidden_dim_pool: Option<(usize, usize)>,
        hidden_dim_stride: usize,
        hidden_nodes_pool: Option<(usize, usize)>,
        hidden_inputs_pool: Option<(usize, usize)>,
        hidden_outputs_pool: Option<(usize, usize)>,
        dropout_prob: f32,
        activation_pool: Vec<String>,
        combine_op_pool: Vec<String>,
        standardize_op_pool: Vec<String>,
        feature_split: Option<String>,
        csv_export: bool,
        trace_file: bool,
        elite_save_topology: bool,
        worst_save_topology: bool,
        run_name: Option<String>,
        elite_previous_dir: Option<PathBuf>,
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
            crossover_ops_pool,
            checkpoint_every,
            mutation_probation_steps,
            crossover_probation_steps,
            elite_join_probation_steps,
            elite_crown_lifetime_steps,
            hidden_dim_pool,
            hidden_dim_stride,
            hidden_nodes_pool,
            hidden_inputs_pool,
            hidden_outputs_pool,
            dropout_prob,
            activation_pool,
            combine_op_pool,
            standardize_op_pool,
            feature_split,
            csv_export,
            trace_file,
            elite_save_topology,
            worst_save_topology,
            run_name,
            elite_previous_dir,
        }
    }

    /// The headline knobs. The remaining fields are reachable by name and land in
    /// `engine.json`; printing 39 of them would bury the ones people change.
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

/// A `(min, max)` inclusive pair → a range, or a `ValueError` naming the field.
///
/// Shared by every sampled range: an empty or inverted pair would panic the
/// sampler (`end - start`), so it is refused before a single net is drawn.
fn parse_range_pool(
    name: &str,
    pair: Option<(usize, usize)>,
) -> PyResult<Option<std::ops::RangeInclusive<usize>>> {
    match pair {
        Some((lo, hi)) if lo == 0 || lo > hi => Err(PyValueError::new_err(format!(
            "{name} ({lo}, {hi}): expected 1 <= min <= max"
        ))),
        Some((lo, hi)) => Ok(Some(lo..=hi)),
        None => Ok(None),
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
                )));
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
        let hidden_dim_pool = parse_range_pool("hidden_dim_pool", self.hidden_dim_pool)?;
        let hidden_nodes_pool = parse_range_pool("hidden_nodes_pool", self.hidden_nodes_pool)?;
        let hidden_inputs_pool = parse_range_pool("hidden_inputs_pool", self.hidden_inputs_pool)?;
        let hidden_outputs_pool =
            parse_range_pool("hidden_outputs_pool", self.hidden_outputs_pool)?;
        // Outside 0..=1 a probability is not a probability; the crate would take
        // it and train on nonsense, so the boundary is drawn here.
        if !(0.0..=1.0).contains(&self.dropout_prob) {
            return Err(PyValueError::new_err(format!(
                "dropout_prob {}: expected 0.0 <= p <= 1.0",
                self.dropout_prob
            )));
        }

        // The crate owns the spelling of an op name too; letting IT reject a typo keeps
        // the message and the accepted set identical between Rust and Python.
        for op in &self.crossover_ops_pool {
            CrossoverOp::try_from(op.as_str())
                .map_err(|e| PyValueError::new_err(format!("crossover_ops_pool: {e}")))?;
        }

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
        cfg.crossover_ops_pool = self.crossover_ops_pool;
        cfg.checkpoint_every = self.checkpoint_every;
        cfg.mutation_probation_steps = self.mutation_probation_steps;
        cfg.crossover_probation_steps = self.crossover_probation_steps;
        cfg.elite_join_probation_steps = self.elite_join_probation_steps;
        cfg.elite_crown_lifetime_steps = self.elite_crown_lifetime_steps;
        if let Some(pool) = hidden_dim_pool {
            cfg.hidden_dim_pool = Some(pool);
        }
        cfg.hidden_dim_stride = self.hidden_dim_stride;
        // The node-count / fan-in / fan-out ranges live on `topology_options`, which
        // the run header already records whole — so setting them here is enough for
        // `engine.json` to describe the search space a run actually sampled.
        if let Some(pool) = hidden_nodes_pool {
            cfg.topology_options.min_hidden_num_nodes = *pool.start();
            cfg.topology_options.max_hidden_num_nodes = *pool.end();
        }
        if let Some(pool) = hidden_inputs_pool {
            cfg.topology_options.min_hidden_inputs_per_node = *pool.start();
            cfg.topology_options.max_hidden_inputs_per_node = *pool.end();
        }
        if let Some(pool) = hidden_outputs_pool {
            cfg.topology_options.min_hidden_outputs_per_node = *pool.start();
            cfg.topology_options.max_hidden_outputs_per_node = *pool.end();
        }
        cfg.topology_options.dropout_prob = self.dropout_prob;
        cfg.activation_pool = self.activation_pool;
        cfg.combine_op_pool = self.combine_op_pool;
        cfg.standardize_op_pool = self.standardize_op_pool;
        cfg.feature_split = feature_split;
        cfg.csv_export = self.csv_export;
        cfg.trace_file = self.trace_file;
        cfg.elite_save_topology = self.elite_save_topology;
        cfg.worst_save_topology = self.worst_save_topology;
        cfg.run_name = self.run_name;
        cfg.elite_previous_dir = self.elite_previous_dir;

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
