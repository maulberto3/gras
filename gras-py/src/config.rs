//! `RaceConfig`: the engine knobs a Python caller turns.

use std::path::PathBuf;

use gras::engine::config::{
    DEFAULT_CHECKPOINT_EVERY, DEFAULT_CROSSOVER_FITTER_BASE, DEFAULT_CROSSOVER_PROBATION_STEPS,
    DEFAULT_CROSSOVER_SHARE, DEFAULT_INIT_PROBATION_STEPS, DEFAULT_MEAN_TYPE,
    DEFAULT_MUTATION_NOVELTY_DRAWS, DEFAULT_MUTATION_NOVELTY_MIN, DEFAULT_MUTATION_PROBATION_STEPS,
    DEFAULT_MUTATION_SHARE, DEFAULT_POLICY_SCOPED_FITNESS, DEFAULT_RECOVERY_RETRIES,
    DEFAULT_RECOVERY_SAME_SEED, DEFAULT_RECOVERY_TOLERANCE, DEFAULT_SAVE_OPTIM,
    DEFAULT_SAVE_WEIGHTS_AT_STOP, DEFAULT_TRAIN_MERIT_P_MAX, DEFAULT_TRAIN_MERIT_P_MIN,
    DEFAULT_TRAIN_TICKET, DEFAULT_WORST_SAVE_TOPOLOGY, SMOOTHING_WINDOW,
};
use gras::evolution::selection::SelectionMethod;
use gras::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Defaults that `RaceConfig::defaults()` sets to a literal, mirrored here so the `#[pyo3(signature = …)]` below can name them. `defaults_mirror_the_crate` pins every one…
const DEFAULT_POP_SIZE: usize = 10;
const DEFAULT_HIDDEN_DIM_STRIDE: usize = 16;
const DEFAULT_TOURNAMENT_SIZE: usize = 2;
/// Step budget a no-arg `RaceConfig()` carries. `None` removes the cap, which
/// leaves the run with no built-in stop at all, so the mirror is the safe default.
const DEFAULT_MAX_STEPS: Option<usize> = Some(10);

/// Dropout the crate builds nets with (`TopologyOptions::default().dropout_prob`);
/// `0.0` turns it off per run.
const DEFAULT_DROPOUT_PROB: f32 = 0.25;

/// Drift on a crossover child's inherited genes (`DEFAULT_TOPOLOGY_DRIFT`).
const DEFAULT_DRIFT_PROB: f32 = 0.1;

/// The share of a SINGLE-POOL dataset held out for eval, mirrored from the crate's `DEFAULT_EVAL_SHARE` (0.2 = 80/20).
const DEFAULT_EVAL_SHARE: f32 = 0.2;

/// Engine knobs a Python caller turns.
#[pyclass(name = "RaceConfig", module = "gras._core")]
#[derive(Clone)]
pub struct PyRaceConfig {
    // ── population + ranking ────────────────────────────────────────────
    #[pyo3(get, set)]
    pub pop_size: usize,
    #[pyo3(get, set)]
    pub champion_count: usize,
    /// Step budget. `None` = no cap (the race then stops only on Ctrl+C or a
    /// custom stop); the crate's default, mirrored here, is 10 clocks.
    #[pyo3(get, set)]
    pub max_steps: Option<usize>,
    /// Optional plateau stop: `(target, patience)`.
    #[pyo3(get, set)]
    pub stop_plateau: Option<(f32, usize)>,
    /// Share of a SINGLE-POOL dataset held out for eval (`0.2` = the last 20% of rows, chosen by a seeded shuffle, are scored and never trained on).
    #[pyo3(get, set)]
    pub eval_share: f32,
    #[pyo3(get, set)]
    pub smoothing_window: usize,
    /// How the smoothing window is averaged into the ranking centre: `"arithmetic"` (default), `"signed_log"` (geometric-like damping that is defined for negative readings) or…
    #[pyo3(get, set)]
    pub mean_type: String,
    /// Build the window statistics only from readings on the net's current weights.
    #[pyo3(get, set)]
    pub policy_scoped_fitness: bool,
    /// Metric labels and weights that also count toward the ranking (`set_selection_metric`).
    #[pyo3(get, set)]
    pub selection_metrics: Vec<(String, f32)>,
    /// FLOOR `p_min` of the ticket band: literally the chance the WORST-graded net gets each
    /// clock (`0.05` default). Every graded net is paid between this and `train_merit_p_max`.
    #[pyo3(get, set)]
    pub train_merit_p_min: f32,
    /// CEILING `p_max` of the ticket band: the chance the BEST-graded net gets each clock (`0.4` default).
    #[pyo3(get, set)]
    pub train_merit_p_max: f32,
    /// The lottery's two legs (`set_train_ticket`), named exactly as the Rust field:
    /// `most_fit_fewest_updates`, `least_fit_most_updates`, etc.
    #[pyo3(get, set)]
    pub train_ticket: String,
    /// A lost ticket costs the weight update, not the clock.
    #[pyo3(get, set)]
    pub train_merit_learn_only: bool,
    /// INIT PROBATION (`set_pop_init_probation_steps`, crate default `0`): the opening clocks the run spends WITH no culls and no births — evolution starts at that clock.
    #[pyo3(get, set)]
    pub init_probation_steps: usize,
    /// Drift on a crossover child's INHERITED genes (`0.0` = exact inheritance), one probability for the whole family: each mask bit flips, and each hidden activation / combine…
    #[pyo3(get, set)]
    pub drift_prob: f32,
    /// Per-step reporting shape, the same vocabulary as the examples' `--log-level`: `summ` (default), `minimal`, `none`, `debug`, `trace`.
    #[pyo3(get, set)]
    pub log_level: String,

    // ── evolution ───────────────────────────────────────────────────────
    /// Mutation NOVELTY floor Δ: how far a random immigrant must sit from the NEAREST blueprint the run has ever made (founders, crossover children and earlier mutants), as a…
    #[pyo3(get, set)]
    pub mutation_novelty_min: f32,
    /// Candidates ONE mutation may spend clearing `mutation_novelty_min` before the most novel refusal is taken anyway, so a culled seat is never left empty.
    #[pyo3(get, set)]
    pub mutation_novelty_draws: usize,
    /// Who a surviving crossover child evicts: `highest_ticket_chance` (default),
    /// `lowest_fitness`, `lowest_fitness_most_updates`, or `random`.
    #[pyo3(get, set)]
    pub crossover_cull_policy: String,
    /// Who a firing mutation roll evicts: `highest_ticket_chance` (default),
    /// `lowest_fitness`, `lowest_fitness_most_updates`, or `random`.
    #[pyo3(get, set)]
    pub mutation_cull_policy: String,
    /// How a crossover attempt draws parents: `roulette` (the default) or
    /// `tournament` (K = `tournament_size`).
    #[pyo3(get, set)]
    pub crossover_selection: String,
    /// K for `crossover_selection="tournament"`; small K is mild pressure.
    #[pyo3(get, set)]
    pub tournament_size: usize,
    /// Which mixing methods a crossover attempt draws from: `"one_point"` and/or
    /// `"uniform"`. Empty (the default) means both, drawn uniformly.
    #[pyo3(get, set)]
    pub crossover_ops_pool: Vec<String>,
    /// Crossover turnover as a share of the live population per clock.
    #[pyo3(get, set)]
    pub crossover_share: f32,
    /// Mutation turnover as a share of the live population per clock.
    #[pyo3(get, set)]
    pub mutation_share: f32,
    /// The fitter drawn parent becomes the crossover base.
    #[pyo3(get, set)]
    pub crossover_fitter_base: bool,

    // ── durability ──────────────────────────────────────────────────────
    /// Steps between population checkpoints (the `history.csv` flush cadence and, on the tabular arm, the eval-permutation era key off this; an RL run's exam is per clock…
    #[pyo3(get, set)]
    pub checkpoint_every: usize,

    // ── probation (cull immunity for newcomers) ─────────────────────────
    /// A random immigrant is cull-immune until it has spent k UPDATES (`U`), never by
    /// mere waiting; `0` = only the empty-buffer step-0 immunity.
    #[pyo3(get, set)]
    pub mutation_probation_steps: usize,
    /// A crossover-born net is cull-immune until it has spent k UPDATES (`U`); `0` = off.
    #[pyo3(get, set)]
    pub crossover_probation_steps: usize,
    /// Extra REAL updates a net may take in one clock after an update leaves it below its standing: the recovery streak.
    #[pyo3(get, set)]
    pub recovery_retries: usize,
    /// How far a reading may drop below the net's standing before a rescue, as a fraction of
    /// that standing (`0.2` = a fifth tolerated). `0.0` = every drop triggers.
    #[pyo3(get, set)]
    pub recovery_tolerance: f32,
    /// `true` (default) = a retry repeats the clock's OWN train draw, so the streak re-tests the update that fell; `false` = each retry walks to the next batch / match block…
    #[pyo3(get, set)]
    pub recovery_same_seed: bool,

    // ── search space ────────────────────────────────────────────────────
    /// Hidden-dim sampling range as `(min, max)` inclusive. `None` keeps the
    /// crate's `4..=8`; a one-width range (`(16, 16)`) fixes the hidden width.
    #[pyo3(get, set)]
    pub hidden_dim_pool: Option<(usize, usize)>,
    /// Stride within `hidden_dim_pool` when sampling node dimensions.
    #[pyo3(get, set)]
    pub hidden_dim_stride: usize,
    /// How many hidden NODES a sampled net gets, as `(min, max)` inclusive. `None` keeps the crate's `2..=5`.
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
    /// Dropout baked into each net at build; `0.25` is the crate's default and `0.0` turns it off.
    #[pyo3(get, set)]
    pub dropout_prob: f32,
    /// Per-net raw-input reach range `(min, max)`; `None` keeps `(1.0, 1.0)`.
    #[pyo3(get, set)]
    pub input_reach: Option<(f32, f32)>,
    /// Keep at most one wire per `(source, target)` pair.
    #[pyo3(get, set)]
    pub unique_source_per_target: bool,
    /// Activation pool; empty means all known activations.
    #[pyo3(get, set)]
    pub activation_pool: Vec<String>,
    /// Combine-op pool; empty means all.
    #[pyo3(get, set)]
    pub combine_op_pool: Vec<String>,
    /// Standardize-op pool; empty means all.
    #[pyo3(get, set)]
    pub standardize_op_pool: Vec<String>,
    /// How the Input node's per-port feature masks are drawn: `None` (default) for no split at all, `random` (independent per-port coins) or `exclusive` (a true partition).
    #[pyo3(get, set)]
    pub feature_split: Option<String>,
    /// The POOL a newborn's split is drawn from: `none`, `random`, `exclusive`.
    #[pyo3(get, set)]
    pub feature_split_pool: Vec<String>,

    // ── artifacts ───────────────────────────────────────────────────────
    /// Write `history.csv` and `attempts.csv`.
    #[pyo3(get, set)]
    pub csv_export: bool,
    /// Write `telemetry.jsonl` (one JSON record per event, full fidelity).
    #[pyo3(get, set)]
    pub trace_file: bool,
    /// Write every live net's weights at the terminal stop.
    #[pyo3(get, set)]
    pub save_weights_at_stop: bool,
    /// Write each net's optimizer state beside its weights.
    #[pyo3(get, set)]
    pub save_optim: bool,
    /// Write the anti-champion topology files at stop.
    #[pyo3(get, set)]
    pub worst_save_topology: bool,
    #[pyo3(get, set)]
    pub champion_save_topology: bool,
    /// Human label recorded in `engine.json` as `"run_name"`.
    #[pyo3(get, set)]
    pub run_name: Option<String>,

    /// Load the newest compatible previous run's final-table elites as topologies. **Off here, unlike the crate** (`true`), so a Python run is all-random unless the caller…
    #[pyo3(get, set)]
    pub champion_previous_load: bool,
    /// Where `champion_previous_load` looks: a run dir or a dir of run dirs.
    #[pyo3(get, set)]
    pub champion_previous_dir: Option<PathBuf>,
}

#[pymethods]
impl PyRaceConfig {
    #[new]
    #[pyo3(signature = (
        // population + ranking
        pop_size = DEFAULT_POP_SIZE,
        champion_count = 1,
        max_steps = DEFAULT_MAX_STEPS,
        eval_share = DEFAULT_EVAL_SHARE,
        smoothing_window = SMOOTHING_WINDOW,
        mean_type = DEFAULT_MEAN_TYPE.to_string(),
        train_merit_p_min = DEFAULT_TRAIN_MERIT_P_MIN,
        train_merit_p_max = DEFAULT_TRAIN_MERIT_P_MAX,
        train_ticket = DEFAULT_TRAIN_TICKET.as_str().to_string(),
        init_probation_steps = DEFAULT_INIT_PROBATION_STEPS,
        drift_prob = DEFAULT_DRIFT_PROB,
        champion_previous_load = false,
        log_level = String::from("summ"),
        // evolution
        mutation_novelty_min = DEFAULT_MUTATION_NOVELTY_MIN,
        mutation_novelty_draws = DEFAULT_MUTATION_NOVELTY_DRAWS,
        crossover_cull_policy = CrossCullPolicy::default().to_string(),
        mutation_cull_policy = MutationCullPolicy::default().to_string(),
        crossover_selection = String::from("roulette"),
        tournament_size = DEFAULT_TOURNAMENT_SIZE,
        crossover_ops_pool = Vec::new(),
        // durability + probation
        checkpoint_every = DEFAULT_CHECKPOINT_EVERY,
        mutation_probation_steps = DEFAULT_MUTATION_PROBATION_STEPS,
        crossover_probation_steps = DEFAULT_CROSSOVER_PROBATION_STEPS,
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
        feature_split_pool = vec![],
        // artifacts
        csv_export = true,
        trace_file = false,
        champion_save_topology = true,
        run_name = None,
        champion_previous_dir = None,
        recovery_retries = DEFAULT_RECOVERY_RETRIES,
        recovery_tolerance = DEFAULT_RECOVERY_TOLERANCE,
        recovery_same_seed = DEFAULT_RECOVERY_SAME_SEED,
        // Rust-builder parity knobs
        crossover_share = DEFAULT_CROSSOVER_SHARE,
        mutation_share = DEFAULT_MUTATION_SHARE,
        crossover_fitter_base = DEFAULT_CROSSOVER_FITTER_BASE,
        policy_scoped_fitness = DEFAULT_POLICY_SCOPED_FITNESS,
        train_merit_learn_only = false,
        save_weights_at_stop = DEFAULT_SAVE_WEIGHTS_AT_STOP,
        save_optim = DEFAULT_SAVE_OPTIM,
        worst_save_topology = DEFAULT_WORST_SAVE_TOPOLOGY,
        input_reach = None,
        unique_source_per_target = true,
        stop_plateau = None,
        selection_metrics = Vec::new(),
    ))]
    // A config's kwargs ARE its API: bundling them into an object would buy the lint nothing and cost the caller a line per knob.
    #[allow(clippy::too_many_arguments)]
    fn new(
        pop_size: usize,
        champion_count: usize,
        max_steps: Option<usize>,
        eval_share: f32,
        smoothing_window: usize,
        mean_type: String,
        train_merit_p_min: f32,
        train_merit_p_max: f32,
        train_ticket: String,
        init_probation_steps: usize,
        drift_prob: f32,
        champion_previous_load: bool,
        log_level: String,
        mutation_novelty_min: f32,
        mutation_novelty_draws: usize,
        crossover_cull_policy: String,
        mutation_cull_policy: String,
        crossover_selection: String,
        tournament_size: usize,
        crossover_ops_pool: Vec<String>,
        checkpoint_every: usize,
        mutation_probation_steps: usize,
        crossover_probation_steps: usize,
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
        feature_split_pool: Vec<String>,
        csv_export: bool,
        trace_file: bool,
        champion_save_topology: bool,
        run_name: Option<String>,
        champion_previous_dir: Option<PathBuf>,
        recovery_retries: usize,
        recovery_tolerance: f32,
        recovery_same_seed: bool,
        crossover_share: f32,
        mutation_share: f32,
        crossover_fitter_base: bool,
        policy_scoped_fitness: bool,
        train_merit_learn_only: bool,
        save_weights_at_stop: bool,
        save_optim: bool,
        worst_save_topology: bool,
        input_reach: Option<(f32, f32)>,
        unique_source_per_target: bool,
        stop_plateau: Option<(f32, usize)>,
        selection_metrics: Vec<(String, f32)>,
    ) -> Self {
        Self {
            pop_size,
            champion_count,
            max_steps,
            eval_share,
            smoothing_window,
            mean_type,
            train_merit_p_min,
            train_merit_p_max,
            train_ticket,
            init_probation_steps,
            drift_prob,
            // OFF by default: a Python run must not silently inherit elites from
            // whatever `results/` happens to sit in the caller's cwd.
            champion_previous_load,
            // `summ` matches every Rust example's default: a race reports one
            // line per step instead of running silently.
            log_level,
            mutation_novelty_min,
            mutation_novelty_draws,
            crossover_cull_policy,
            mutation_cull_policy,
            crossover_selection,
            tournament_size,
            crossover_ops_pool,
            checkpoint_every,
            mutation_probation_steps,
            crossover_probation_steps,
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
            feature_split_pool,
            csv_export,
            trace_file,
            champion_save_topology,
            run_name,
            champion_previous_dir,
            recovery_retries,
            recovery_tolerance,
            recovery_same_seed,
            crossover_share,
            mutation_share,
            crossover_fitter_base,
            policy_scoped_fitness,
            train_merit_learn_only,
            save_weights_at_stop,
            save_optim,
            worst_save_topology,
            input_reach,
            unique_source_per_target,
            stop_plateau,
            selection_metrics,
        }
    }

    /// The headline knobs. The remaining fields are reachable by name and land in
    /// `engine.json`; printing 39 of them would bury the ones people change.
    fn __repr__(&self) -> String {
        format!(
            "RaceConfig(pop_size={}, champion_count={}, max_steps={:?}, smoothing_window={}, \
             champion_previous_load={}, log_level={:?})",
            self.pop_size,
            self.champion_count,
            self.max_steps,
            self.smoothing_window,
            self.champion_previous_load,
            self.log_level
        )
    }
}

/// A `(min, max)` inclusive pair → a range, or a `ValueError` naming the field.
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

/// The four cull spellings the crate records in `engine.json`.
fn parse_crossover_cull(name: &str) -> Option<CrossCullPolicy> {
    match name {
        "lowest_fitness" => Some(CrossCullPolicy::LowestFitness),
        "random" => Some(CrossCullPolicy::Random),
        "highest_ticket_chance" => Some(CrossCullPolicy::HighestTicketChance),
        "lowest_fitness_most_updates" => Some(CrossCullPolicy::LowestFitnessMostUpdates),
        _ => None,
    }
}

fn parse_mutation_cull(name: &str) -> Option<MutationCullPolicy> {
    match name {
        "lowest_fitness" => Some(MutationCullPolicy::LowestFitness),
        "random" => Some(MutationCullPolicy::Random),
        "highest_ticket_chance" => Some(MutationCullPolicy::HighestTicketChance),
        "lowest_fitness_most_updates" => Some(MutationCullPolicy::LowestFitnessMostUpdates),
        _ => None,
    }
}

impl PyRaceConfig {
    /// The crate config plus the tracing verbosity `logging::init` takes.
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
                    "crossover_cull_policy {:?}: expected one of lowest_fitness, random, highest_ticket_chance, lowest_fitness_most_updates",
                    self.crossover_cull_policy
                ))
            })?;
        let mutation_cull_policy =
            parse_mutation_cull(&self.mutation_cull_policy).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "mutation_cull_policy {:?}: expected one of lowest_fitness, random, highest_ticket_chance, lowest_fitness_most_updates",
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
        // `feature_split=` pins the pool to ONE policy; `feature_split_pool=` names the
        // whole draw set; both empty means all four (full, random, exclusive, chunks).
        let feature_split = match &self.feature_split {
            Some(name) => Some(match name.to_ascii_lowercase().as_str() {
                "full" | "none" | "off" | "split_full" | "split_none" => "full".to_string(),
                "random" | "split_random" => "random".to_string(),
                "exclusive" | "split_exclusive" => "exclusive".to_string(),
                "chunks" | "split_chunks" => "chunks".to_string(),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "feature_split {other:?}: expected one of full, random, exclusive, chunks"
                    )));
                }
            }),
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

        let mut cfg = gras::RaceConfig::defaults();
        cfg.population.pop_size = self.pop_size;
        cfg.population.champion_count = self.champion_count;
        cfg.stop.max_steps = self.max_steps;
        // Same bounds the crate's setter enforces: outside them a pool can round
        // to empty (the split then hands the whole dataset to eval).
        if !(0.01..=0.9).contains(&self.eval_share) {
            return Err(PyValueError::new_err(format!(
                "eval_share {}: expected 0.01 <= share <= 0.9 (the fraction of rows held out for eval)",
                self.eval_share
            )));
        }
        cfg.run.eval_share = self.eval_share;
        cfg.ranking.smoothing_window = self.smoothing_window;
        cfg.ranking.mean_type =
            gras::engine::config::MeanType::parse(&self.mean_type).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "mean_type {:?}: expected one of arithmetic, signed_log, median",
                    self.mean_type
                ))
            })?;
        cfg.ranking.policy_scoped_fitness = self.policy_scoped_fitness;
        cfg.ranking.selection_metrics = self
            .selection_metrics
            .iter()
            .map(|(label, weight)| gras::utils::fitness::SelectionMetric {
                label: label.clone(),
                weight: *weight,
            })
            .collect();
        cfg.lottery.train_merit_p_min = if self.train_merit_p_min.is_finite() {
            self.train_merit_p_min.clamp(0.0, 1.0)
        } else {
            DEFAULT_TRAIN_MERIT_P_MIN
        };
        cfg.lottery.train_merit_p_max = if self.train_merit_p_max.is_finite() {
            self.train_merit_p_max.clamp(0.0, 1.0)
        } else {
            DEFAULT_TRAIN_MERIT_P_MAX
        };
        // The LEGS are the rest of the band's arithmetic, so a surface that can set the ends but not the legs still cannot reproduce a run's chances.
        cfg.lottery.train_ticket = gras::engine::config::TrainTicket::parse(&self.train_ticket)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "train_ticket {:?}: expected one of {}",
                    self.train_ticket,
                    gras::engine::config::TrainTicket::ALL_NAMES.join(", ")
                ))
            })?;
        cfg.lottery.train_merit_learn_only = self.train_merit_learn_only;
        cfg.population.init_probation_steps = self.init_probation_steps;
        // Clamped like the Rust setter (a stray 1.5 must not flip every bit twice); the
        // value is recorded in `engine.json` and compared by name on resume.
        cfg.topology.drift_prob = if self.drift_prob.is_finite() {
            self.drift_prob.clamp(0.0, 1.0)
        } else {
            0.0
        };
        cfg.run.champion_previous_load = self.champion_previous_load;
        cfg.run.log_level = log_level;
        // The gate's two numbers take the same bounds the crate's setters enforce, so a typed
        // slip reads as "off" / "one draw" in both languages instead of a mid-race surprise.
        cfg.evolution.mutation_novelty_min = if self.mutation_novelty_min.is_finite() {
            self.mutation_novelty_min.max(0.0)
        } else {
            0.0
        };
        cfg.evolution.mutation_novelty_draws = self.mutation_novelty_draws.max(1);
        cfg.evolution.crossover_cull_policy = crossover_cull_policy;
        cfg.evolution.mutation_cull_policy = mutation_cull_policy;
        cfg.evolution.crossover_selection = crossover_selection;
        cfg.evolution.crossover_ops_pool = self.crossover_ops_pool;
        cfg.evolution.crossover_share = if self.crossover_share.is_finite() {
            self.crossover_share.clamp(0.0, 1.0)
        } else {
            DEFAULT_CROSSOVER_SHARE
        };
        cfg.evolution.mutation_share = if self.mutation_share.is_finite() {
            self.mutation_share.clamp(0.0, 1.0)
        } else {
            DEFAULT_MUTATION_SHARE
        };
        cfg.evolution.crossover_fitter_base = self.crossover_fitter_base;
        cfg.run.checkpoint_every = self.checkpoint_every;
        cfg.population.mutation_probation_steps = self.mutation_probation_steps;
        cfg.population.crossover_probation_steps = self.crossover_probation_steps;
        cfg.population.recovery_retries = self.recovery_retries;
        cfg.population.recovery_tolerance = if self.recovery_tolerance.is_finite() {
            self.recovery_tolerance.max(0.0)
        } else {
            0.0
        };
        cfg.population.recovery_same_seed = self.recovery_same_seed;
        if let Some(pool) = hidden_dim_pool {
            cfg.topology.hidden_dim_pool = Some(pool);
        }
        cfg.topology.hidden_dim_stride = self.hidden_dim_stride;
        // The node-count / fan-in / fan-out ranges live on `topology_options`, which the run header already records whole — so setting them here is enough for `engine.json` to…
        if let Some(pool) = hidden_nodes_pool {
            cfg.topology.options.min_hidden_num_nodes = *pool.start();
            cfg.topology.options.max_hidden_num_nodes = *pool.end();
        }
        if let Some(pool) = hidden_inputs_pool {
            cfg.topology.options.min_hidden_inputs_per_node = *pool.start();
            cfg.topology.options.max_hidden_inputs_per_node = *pool.end();
        }
        if let Some(pool) = hidden_outputs_pool {
            cfg.topology.options.min_hidden_outputs_per_node = *pool.start();
            cfg.topology.options.max_hidden_outputs_per_node = *pool.end();
        }
        cfg.topology.options.dropout_prob = self.dropout_prob;
        if let Some((min, max)) = self.input_reach {
            let min = if min.is_finite() {
                min.clamp(0.0, 1.0)
            } else {
                1.0
            };
            let max = if max.is_finite() {
                max.clamp(0.0, 1.0)
            } else {
                1.0
            };
            cfg.topology.options.input_reach = min;
            cfg.topology.options.min_input_reach = min;
            cfg.topology.options.max_input_reach = max;
        }
        cfg.topology.options.unique_source_per_target = self.unique_source_per_target;
        cfg.topology.activation_pool = self.activation_pool;
        cfg.topology.combine_op_pool = self.combine_op_pool;
        cfg.topology.standardize_op_pool = self.standardize_op_pool;
        // The pool decides what a NEWBORN draws; `feature_split=` pins it to one entry.
        cfg.topology.feature_split_pool = if self.feature_split_pool.is_empty() {
            feature_split.into_iter().collect()
        } else {
            self.feature_split_pool.clone()
        };
        // Name a bad pool entry here, before a single net is drawn — every name-based pool, not just the split: the crate's draw paths fall back to `all` on a typo, so this is…
        cfg.validate_search_pools().map_err(PyValueError::new_err)?;
        cfg.run.csv_export = self.csv_export;
        cfg.run.trace_file = self.trace_file;
        cfg.run.save_weights_at_stop = self.save_weights_at_stop;
        cfg.run.save_optim = self.save_optim;
        cfg.run.worst_save_topology = self.worst_save_topology;
        cfg.stop.plateau = self
            .stop_plateau
            .map(|(target, patience)| gras::engine::clock::stop::Plateau::new(target, patience));
        cfg.run.champion_save_topology = self.champion_save_topology;
        cfg.run.run_name = self.run_name;
        cfg.run.champion_previous_dir = self.champion_previous_dir;

        Ok((cfg, verbosity))
    }
}
