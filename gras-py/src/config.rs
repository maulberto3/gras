//! `RaceConfig`: the engine knobs a Python caller turns.
//!
//! The defaults ARE the crate's (`RaceConfig::defaults()`), with two documented
//! exceptions: λ `0.0` (rank on the plain window mean) and `champion_previous_load`
//! off (a Python run never inherits elites from whatever `results/` sits in the
//! caller's cwd). Anything this class does not expose still inherits the crate
//! default, because [`PyRaceConfig::into_inner`] starts from `RaceConfig::defaults()`.

use std::path::PathBuf;

use gras::engine::config::{
    DEFAULT_CHECKPOINT_EVERY, DEFAULT_INIT_PROBATION_STEPS, DEFAULT_MUTATION_NOVELTY_DRAWS,
    DEFAULT_MUTATION_NOVELTY_MIN, DEFAULT_RECOVERY_RETRIES, DEFAULT_RECOVERY_SAME_SEED,
    DEFAULT_RECOVERY_TOLERANCE, DEFAULT_TICKET_LEGS, DEFAULT_TRAIN_MERIT_P_MAX, SMOOTHING_WINDOW,
};
use gras::evolution::selection::SelectionMethod;
use gras::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Defaults that `RaceConfig::defaults()` sets to a literal, mirrored here so the
// `#[pyo3(signature = …)]` below can name them. `defaults_mirror_the_crate` pins
// every one of them against the crate, so a changed crate default fails a test
// instead of silently re-tuning Python. The RECOVERY knobs are NOT mirrored: their
// crate consts are already `pub`, so they are imported and the number lives once.
const DEFAULT_POP_SIZE: usize = 10;
const DEFAULT_CROSSOVER_ROLLS: usize = 2;
const DEFAULT_MUTATE_ROLLS: usize = 2;
const DEFAULT_CROSSOVER_PROB: f32 = 0.5;
const DEFAULT_MUTATE_PROB: f32 = 0.2;
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

/// Cull immunity for every newcomer, in UPDATES (`U`). The crate's shipped value,
/// mirrored here so the signature can name it. Unlike λ and Ω this is not a
/// "neutral at zero" knob: `0` removes a protection rather than switching off an
/// effect on the ranking, so a Python caller inherits the crate's 5 instead of
/// starting bare.
const DEFAULT_PROBATION_STEPS: usize = 5;
/// λ is the binding's one ranking divergence: `0.0` ranks on the plain window mean,
/// while the crate's FIT default is `0.5` (a net pays for its own spread). Ω and
/// every other knob mirror the crate's tuned reference run.
const DEFAULT_RISK_AVERSION: f32 = 0.0;
/// The retired Ω, at its identity: the band `train_merit_p_min..p_max` is the payout now.
const DEFAULT_TRAIN_MERIT: f32 = 1.0;
/// The lottery's merit floor `p_min`, mirrored from the crate's default. It only has
/// an effect while Ω is above 0, so exposing it changes nothing about the Ω = 0 path
/// a Python run starts on.
const DEFAULT_TRAIN_MERIT_P_MIN: f32 = 0.05;
/// The share of a SINGLE-POOL dataset held out for eval, mirrored from the crate's
/// `DEFAULT_EVAL_SHARE` (0.2 = 80/20). Only consulted when `data_dir` is one pool
/// (`inputs.*` + `targets.*`); a dir with `train/` + `test/` states its own split.
const DEFAULT_EVAL_SHARE: f32 = 0.2;

/// Engine knobs a Python caller turns. Anything not listed keeps the crate's
/// default, and every field is recorded in the run's `engine.json`.
///
/// Two choices are the binding's own, both to keep a Python run neutral until it
/// asks otherwise: `risk_aversion` is `0.0` (rank on the plain mean, no charge
/// for spread) and `champion_previous_load` is `false` (all-random founders).
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
    /// Share of a SINGLE-POOL dataset held out for eval (`0.2` = the last 20% of
    /// rows, chosen by a seeded shuffle, are scored and never trained on). A dir
    /// with `train/` + `test/` subdirs states its own split and this is ignored.
    /// Clamped to `0.01 …= 0.9` on the way into the engine.
    #[pyo3(get, set)]
    pub eval_share: f32,
    #[pyo3(get, set)]
    pub smoothing_window: usize,
    /// Risk aversion λ: window standard deviations subtracted from a net's
    /// smoothed mean before ranking. **`0.0` here** ranks on the plain mean (the
    /// crate's FIT default is `0.5`).
    #[pyo3(get, set)]
    pub risk_aversion: f32,
    /// How the smoothing window is averaged into the ranking centre:
    /// `"arithmetic"` (default), `"signed_log"` (geometric-like damping that is
    /// defined for negative readings) or `"median"` (one fluke cannot move it).
    #[pyo3(get, set)]
    pub mean_type: String,
    /// LEGACY train/eval lottery strength Ω — RETIRED as a second dial: the band
    /// `train_merit_p_min..p_max` is the chance a net gets each clock, so what a run sets is
    /// what it pays. A value passed here is folded into that band once; default `1.0` (identity).
    #[pyo3(get, set)]
    pub train_merit: f32,
    /// FLOOR `p_min` of the ticket band: literally the chance the WORST-graded net gets each
    /// clock (`0.05` default). Every graded net is paid between this and `train_merit_p_max`.
    #[pyo3(get, set)]
    pub train_merit_p_min: f32,
    /// CEILING `p_max` of the ticket band: the chance the BEST-graded net gets each clock
    /// (`0.4` default). With Ω retired the band IS the payout, so this is a literal probability
    /// too — set `p_min == p_max` to pay every graded net the same chance.
    #[pyo3(get, set)]
    pub train_merit_p_max: f32,
    /// The lottery's two LEGS (`set_ticket_legs`, crate default `least_fit_fewest_updates`): which
    /// fit place earns the most tickets and which side of the updates field it multiplies in —
    /// `most_fit_fewest_updates`, `most_fit_most_updates`, `least_fit_fewest_updates`,
    /// `least_fit_most_updates`, or the place alone (`most_fit` / `least_fit`).
    #[pyo3(get, set)]
    pub ticket_legs: String,
    /// INIT PROBATION (`set_pop_init_probation_steps`, crate default `0`): the opening clocks the
    /// run spends WITH no culls and no births — evolution starts at that clock. The recovery
    /// streak is off for the same window.
    #[pyo3(get, set)]
    pub init_probation_steps: usize,
    /// Drift on a crossover child's INHERITED genes (`0.0` = exact inheritance), one
    /// probability for the whole family: each mask bit flips, and each hidden activation /
    /// combine / standardize is re-drawn from its pool (dropout re-drawn in
    /// `0.0..=dropout_prob`). The mutation half a pool draw cannot give — a newborn's genes
    /// are unrelated to any parent, so without drift nothing inherited can ever be edited.
    /// `0.1` (the crate's default) edits about one gene in ten.
    #[pyo3(get, set)]
    pub drift_prob: f32,
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
    /// Mutation NOVELTY floor Δ: how far a random immigrant must sit from the NEAREST
    /// blueprint the run has ever made (founders, crossover children and earlier mutants),
    /// as a Jaccard distance on the gene multiset — the reading the diversity cards print.
    /// **`0.0` = the gate off**: one draw, no archive walk, the pre-gate path exactly. The
    /// crate's default is `0.2` (unrelated nets sit near Δ 0.67; a nearest neighbour 0.17).
    #[pyo3(get, set)]
    pub mutation_novelty_min: f32,
    /// Candidates ONE mutation may spend clearing `mutation_novelty_min` before the most
    /// novel refusal is taken anyway, so a culled seat is never left empty. The gate biases
    /// the draw; it never blocks a birth. Crate default `5`.
    #[pyo3(get, set)]
    pub mutation_novelty_draws: usize,
    /// The top-k nets act and measure without weight updates.
    #[pyo3(get, set)]
    pub freeze_seats: bool,
    /// Who a surviving crossover child evicts: `highest_ticket_chance` (default),
    /// `lowest_fitness`, `lowest_fitness_most_updates`, or `random`.
    #[pyo3(get, set)]
    pub crossover_cull_policy: String,
    /// Who a firing mutation roll evicts: `highest_ticket_chance` (default),
    /// `lowest_fitness`, `lowest_fitness_most_updates`, `inverse_fitness`, or `random`.
    #[pyo3(get, set)]
    pub mutation_cull_policy: String,
    /// How a crossover attempt draws parents: `tournament` (default, K =
    /// `tournament_size`) or `roulette`.
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
    /// Steps between population checkpoints (the `history.csv` flush cadence and, on the
    /// tabular arm, the eval-permutation era key off this; an RL run's exam is per clock
    /// unless `Reinforce::with_per_clock_exam(false)` asks for the era back). **`0` = no
    /// durability tick** and one eternal era.
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
    /// Extra REAL updates a net may take in one clock after an update leaves it below its
    /// standing: the recovery streak. Each retry counts in `U`. `0` = off, and `0` is the default;
    /// any value above it arms the streak for that many retries. The streak never fires inside the
    /// run's INIT PROBATION (`init_probation_steps`), whose clocks are judged on draws every net
    /// shares.
    #[pyo3(get, set)]
    pub recovery_retries: usize,
    /// How far a reading may drop below the net's standing before a rescue, as a fraction of
    /// that standing (`0.2` = a fifth tolerated). `0.0` = every drop triggers.
    #[pyo3(get, set)]
    pub recovery_tolerance: f32,
    /// `true` (default) = a retry repeats the clock's OWN train draw, so the streak re-tests the
    /// update that fell; `false` = each retry walks to the next batch / match block, so it fights
    /// on fresh evidence. The eval exam stays keyed to the clock either way.
    #[pyo3(get, set)]
    pub recovery_same_seed: bool,
    /// A net cannot become an elite until it has aged k clocks since arrival.
    #[pyo3(get, set)]
    pub seat_join_probation_steps: usize,
    /// How long a net may stay at peak on the frozen crown before age-elimination;
    /// `0` = permanent until its rank moves.
    #[pyo3(get, set)]
    pub seat_lifetime_steps: usize,

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
    /// Dropout baked into each net at build; `0.25` is the crate's default and
    /// `0.0` turns it off. TRAIN path only, so a replay does not depend on a
    /// random mask.
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
    /// The POOL a newborn's split is drawn from: `none`, `random`, `exclusive`.
    /// Empty (default) = all three, so a Python run explores the split itself;
    /// `feature_split=` above pins the pool to one entry.
    #[pyo3(get, set)]
    pub feature_split_pool: Vec<String>,

    // ── artifacts ───────────────────────────────────────────────────────
    /// Write `history.csv` and `attempts.csv`.
    #[pyo3(get, set)]
    pub csv_export: bool,
    /// Write `telemetry.jsonl` (one JSON record per event, full fidelity).
    #[pyo3(get, set)]
    pub trace_file: bool,
    #[pyo3(get, set)]
    pub champion_save_topology: bool,
    /// Human label recorded in `engine.json` as `"run_name"`.
    #[pyo3(get, set)]
    pub run_name: Option<String>,

    /// Load the newest compatible previous run's final-table elites as topologies.
    /// **Off here, unlike the crate** (`true`), so a Python run is all-random
    /// unless the caller asks: the crate's default looks in `results/` relative
    /// to the cwd, which for a notebook is whatever the kernel was started in.
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
        risk_aversion = DEFAULT_RISK_AVERSION,
        mean_type = String::from("arithmetic"),
        train_merit = DEFAULT_TRAIN_MERIT,
        train_merit_p_min = DEFAULT_TRAIN_MERIT_P_MIN,
        train_merit_p_max = DEFAULT_TRAIN_MERIT_P_MAX,
        ticket_legs = DEFAULT_TICKET_LEGS.as_str().to_string(),
        init_probation_steps = DEFAULT_INIT_PROBATION_STEPS,
        drift_prob = DEFAULT_DRIFT_PROB,
        champion_previous_load = false,
        log_level = String::from("summ"),
        // evolution
        crossover_rolls = DEFAULT_CROSSOVER_ROLLS,
        mutate_rolls = DEFAULT_MUTATE_ROLLS,
        crossover_prob = DEFAULT_CROSSOVER_PROB,
        mutate_prob = DEFAULT_MUTATE_PROB,
        mutation_novelty_min = DEFAULT_MUTATION_NOVELTY_MIN,
        mutation_novelty_draws = DEFAULT_MUTATION_NOVELTY_DRAWS,
        freeze_seats = false,
        crossover_cull_policy = String::from("unfed"),
        mutation_cull_policy = String::from("unfed"),
        crossover_selection = String::from("tournament"),
        tournament_size = DEFAULT_TOURNAMENT_SIZE,
        crossover_ops_pool = Vec::new(),
        // durability + probation
        checkpoint_every = DEFAULT_CHECKPOINT_EVERY,
        mutation_probation_steps = DEFAULT_PROBATION_STEPS,
        crossover_probation_steps = DEFAULT_PROBATION_STEPS,
        seat_join_probation_steps = DEFAULT_PROBATION_STEPS,
        seat_lifetime_steps = DEFAULT_PROBATION_STEPS,
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
    ))]
    // A config's kwargs ARE its API: bundling them into an object would buy the
    // lint nothing and cost the caller a line per knob. Explicitly out of scope
    // in TODO's Stage E too.
    #[allow(clippy::too_many_arguments)]
    fn new(
        pop_size: usize,
        champion_count: usize,
        max_steps: Option<usize>,
        eval_share: f32,
        smoothing_window: usize,
        risk_aversion: f32,
        mean_type: String,
        train_merit: f32,
        train_merit_p_min: f32,
        train_merit_p_max: f32,
        ticket_legs: String,
        init_probation_steps: usize,
        drift_prob: f32,
        champion_previous_load: bool,
        log_level: String,
        crossover_rolls: usize,
        mutate_rolls: usize,
        crossover_prob: f32,
        mutate_prob: f32,
        mutation_novelty_min: f32,
        mutation_novelty_draws: usize,
        freeze_seats: bool,
        crossover_cull_policy: String,
        mutation_cull_policy: String,
        crossover_selection: String,
        tournament_size: usize,
        crossover_ops_pool: Vec<String>,
        checkpoint_every: usize,
        mutation_probation_steps: usize,
        crossover_probation_steps: usize,
        seat_join_probation_steps: usize,
        seat_lifetime_steps: usize,
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
    ) -> Self {
        Self {
            pop_size,
            champion_count,
            max_steps,
            eval_share,
            smoothing_window,
            risk_aversion,
            mean_type,
            train_merit,
            train_merit_p_min,
            train_merit_p_max,
            ticket_legs,
            init_probation_steps,
            drift_prob,
            // OFF by default: a Python run must not silently inherit elites from
            // whatever `results/` happens to sit in the caller's cwd.
            champion_previous_load,
            // `summ` matches every Rust example's default: a race reports one
            // line per step instead of running silently.
            log_level,
            crossover_rolls,
            mutate_rolls,
            crossover_prob,
            mutate_prob,
            mutation_novelty_min,
            mutation_novelty_draws,
            freeze_seats,
            crossover_cull_policy,
            mutation_cull_policy,
            crossover_selection,
            tournament_size,
            crossover_ops_pool,
            checkpoint_every,
            mutation_probation_steps,
            crossover_probation_steps,
            seat_join_probation_steps,
            seat_lifetime_steps,
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
        }
    }

    /// The headline knobs. The remaining fields are reachable by name and land in
    /// `engine.json`; printing 39 of them would bury the ones people change.
    fn __repr__(&self) -> String {
        format!(
            "RaceConfig(pop_size={}, champion_count={}, max_steps={:?}, smoothing_window={}, \
             risk_aversion={}, champion_previous_load={}, log_level={:?})",
            self.pop_size,
            self.champion_count,
            self.max_steps,
            self.smoothing_window,
            self.risk_aversion,
            self.champion_previous_load,
            self.log_level
        )
    }

    // ── deprecated spellings (one release) ──────────────────────────────
    // The vocabulary split three ways: CHAMPIONS (the saved, inheritable ranking), SEATS (the
    // frozen act-only switch) and the guard. These properties keep a notebook written before
    // the rename working, so the words can change without breaking a caller's cell.
    /// Deprecated: use `champion_count`.
    #[getter(elite_count)]
    fn get_elite_count_legacy(&self) -> usize {
        self.champion_count
    }
    #[setter(elite_count)]
    fn set_elite_count_legacy(&mut self, value: usize) {
        self.champion_count = value;
    }
    /// Deprecated: use `freeze_seats`.
    #[getter(freeze_elites)]
    fn get_freeze_elites_legacy(&self) -> bool {
        self.freeze_seats
    }
    #[setter(freeze_elites)]
    fn set_freeze_elites_legacy(&mut self, value: bool) {
        self.freeze_seats = value;
    }
    /// Deprecated: use `seat_join_probation_steps`.
    #[getter(elite_join_probation_steps)]
    fn get_elite_join_probation_steps_legacy(&self) -> usize {
        self.seat_join_probation_steps
    }
    #[setter(elite_join_probation_steps)]
    fn set_elite_join_probation_steps_legacy(&mut self, value: usize) {
        self.seat_join_probation_steps = value;
    }
    /// Deprecated: use `seat_lifetime_steps`.
    #[getter(elite_crown_lifetime_steps)]
    fn get_elite_crown_lifetime_steps_legacy(&self) -> usize {
        self.seat_lifetime_steps
    }
    #[setter(elite_crown_lifetime_steps)]
    fn set_elite_crown_lifetime_steps_legacy(&mut self, value: usize) {
        self.seat_lifetime_steps = value;
    }
    /// Deprecated: use `champion_save_topology`.
    #[getter(elite_save_topology)]
    fn get_elite_save_topology_legacy(&self) -> bool {
        self.champion_save_topology
    }
    #[setter(elite_save_topology)]
    fn set_elite_save_topology_legacy(&mut self, value: bool) {
        self.champion_save_topology = value;
    }
    /// Deprecated: use `champion_previous_load`.
    #[getter(elite_previous_load)]
    fn get_elite_previous_load_legacy(&self) -> bool {
        self.champion_previous_load
    }
    #[setter(elite_previous_load)]
    fn set_elite_previous_load_legacy(&mut self, value: bool) {
        self.champion_previous_load = value;
    }
    /// Deprecated: use `champion_previous_dir`.
    #[getter(elite_previous_dir)]
    fn get_elite_previous_dir_legacy(&self) -> Option<PathBuf> {
        self.champion_previous_dir.clone()
    }
    #[setter(elite_previous_dir)]
    fn set_elite_previous_dir_legacy(&mut self, value: Option<PathBuf>) {
        self.champion_previous_dir = value;
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

/// `lowest_fitness` | `random` | `highest_ticket_chance` | `lowest_fitness_most_updates` —
/// the spellings the crate records in `engine.json`. The pre-2026-11 names (`worst` /
/// `unfed` / `spent`) still parse, so an old script keeps running.
fn parse_crossover_cull(name: &str) -> Option<CrossCullPolicy> {
    match name {
        "lowest_fitness" | "worst" => Some(CrossCullPolicy::LowestFitness),
        "random" => Some(CrossCullPolicy::Random),
        "highest_ticket_chance" | "unfed" => Some(CrossCullPolicy::HighestTicketChance),
        "lowest_fitness_most_updates" | "spent" => Some(CrossCullPolicy::LowestFitnessMostUpdates),
        _ => None,
    }
}

/// `lowest_fitness` | `inverse_fitness` | `random` | `highest_ticket_chance` |
/// `lowest_fitness_most_updates`, plus the legacy `worst` / `unfed` / `spent`.
fn parse_mutation_cull(name: &str) -> Option<MutationCullPolicy> {
    match name {
        "lowest_fitness" | "worst" => Some(MutationCullPolicy::LowestFitness),
        "inverse_fitness" => Some(MutationCullPolicy::InverseFitness),
        "random" => Some(MutationCullPolicy::Random),
        "highest_ticket_chance" | "unfed" => Some(MutationCullPolicy::HighestTicketChance),
        "lowest_fitness_most_updates" | "spent" => {
            Some(MutationCullPolicy::LowestFitnessMostUpdates)
        }
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
                    "crossover_cull_policy {:?}: expected one of lowest_fitness, random, highest_ticket_chance, lowest_fitness_most_updates",
                    self.crossover_cull_policy
                ))
            })?;
        let mutation_cull_policy =
            parse_mutation_cull(&self.mutation_cull_policy).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "mutation_cull_policy {:?}: expected one of lowest_fitness, inverse_fitness, random, highest_ticket_chance, lowest_fitness_most_updates",
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
        // whole draw set; both empty means all three (none, random, exclusive).
        let feature_split = match &self.feature_split {
            Some(name) => Some(match name.to_ascii_lowercase().as_str() {
                "none" | "off" => "none".to_string(),
                "random" | "split_random" => "random".to_string(),
                "exclusive" | "split_exclusive" => "exclusive".to_string(),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "feature_split {other:?}: expected one of none, random, exclusive"
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
        cfg.pop_size = self.pop_size;
        cfg.champion_count = self.champion_count;
        cfg.max_steps = self.max_steps;
        // Same bounds the crate's setter enforces: outside them a pool can round
        // to empty (`split_indices` then hands the whole dataset to eval).
        if !(0.01..=0.9).contains(&self.eval_share) {
            return Err(PyValueError::new_err(format!(
                "eval_share {}: expected 0.01 <= share <= 0.9 (the fraction of rows held out for eval)",
                self.eval_share
            )));
        }
        cfg.eval_share = self.eval_share;
        cfg.smoothing_window = self.smoothing_window;
        cfg.risk_aversion = self.risk_aversion;
        cfg.mean_type =
            gras::engine::config::MeanType::parse(&self.mean_type).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "mean_type {:?}: expected one of arithmetic, signed_log, median",
                    self.mean_type
                ))
            })?;
        // Ω is RETIRED as a second dial (`fold_omega` is its whole story): a caller that still
        // passes it gets the band it used to buy, folded ONCE, and the header records the identity
        // so a resume never folds twice.
        cfg.train_merit_p_min = self.train_merit_p_min;
        // …and the CEILING is the caller's too: with Ω retired the band is the WHOLE payout, so a
        // surface that can set the floor but not the ceiling cannot reproduce a run's chances.
        // Clamped like the crate's setter, and a non-finite value falls back to the crate default.
        let ceiling = if self.train_merit_p_max.is_finite() {
            self.train_merit_p_max.clamp(0.0, 1.0)
        } else {
            DEFAULT_TRAIN_MERIT_P_MAX
        };
        let (lo, hi) =
            gras::engine::config::fold_omega(self.train_merit, cfg.train_merit_p_min, ceiling);
        cfg.train_merit_p_min = lo;
        cfg.train_merit_p_max = hi;
        cfg.train_merit = 1.0;
        // The LEGS are the rest of the band's arithmetic, so a surface that can set the ends but
        // not the legs still cannot reproduce a run's chances. One spelling names both sides and
        // a resume compares it by name.
        cfg.ticket_legs =
            gras::engine::config::TicketLegs::parse(&self.ticket_legs).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "ticket_legs {:?}: expected one of {}",
                    self.ticket_legs,
                    gras::engine::config::TicketLegs::ALL_NAMES.join(", ")
                ))
            })?;
        cfg.init_probation_steps = self.init_probation_steps;
        // Clamped like the Rust setter (a stray 1.5 must not flip every bit twice); the
        // value is recorded in `engine.json` and compared by name on resume.
        cfg.drift_prob = if self.drift_prob.is_finite() {
            self.drift_prob.clamp(0.0, 1.0)
        } else {
            0.0
        };
        cfg.champion_previous_load = self.champion_previous_load;
        cfg.log_level = log_level;
        cfg.crossover_rolls = self.crossover_rolls;
        cfg.mutate_rolls = self.mutate_rolls;
        cfg.crossover_prob = self.crossover_prob;
        cfg.mutate_prob = self.mutate_prob;
        // The gate's two numbers take the same bounds the crate's setters enforce, so a typed
        // slip reads as "off" / "one draw" in both languages instead of a mid-race surprise.
        cfg.mutation_novelty_min = if self.mutation_novelty_min.is_finite() {
            self.mutation_novelty_min.max(0.0)
        } else {
            0.0
        };
        cfg.mutation_novelty_draws = self.mutation_novelty_draws.max(1);
        cfg.freeze_seats = self.freeze_seats;
        cfg.crossover_cull_policy = crossover_cull_policy;
        cfg.mutation_cull_policy = mutation_cull_policy;
        cfg.crossover_selection = crossover_selection;
        cfg.crossover_ops_pool = self.crossover_ops_pool;
        cfg.checkpoint_every = self.checkpoint_every;
        cfg.mutation_probation_steps = self.mutation_probation_steps;
        cfg.crossover_probation_steps = self.crossover_probation_steps;
        cfg.seat_join_probation_steps = self.seat_join_probation_steps;
        cfg.seat_lifetime_steps = self.seat_lifetime_steps;
        cfg.recovery_retries = self.recovery_retries;
        cfg.recovery_tolerance = if self.recovery_tolerance.is_finite() {
            self.recovery_tolerance.max(0.0)
        } else {
            0.0
        };
        cfg.recovery_same_seed = self.recovery_same_seed;
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
        // The pool decides what a NEWBORN draws; `feature_split=` pins it to one entry.
        cfg.feature_split_pool = if self.feature_split_pool.is_empty() {
            feature_split.into_iter().collect()
        } else {
            self.feature_split_pool.clone()
        };
        // Name a bad pool entry here, before a single net is drawn — every name-based
        // pool, not just the split: the crate's draw paths fall back to `all` on a typo,
        // so this is where Python can turn it into a `ValueError` instead.
        cfg.validate_search_pools().map_err(PyValueError::new_err)?;
        cfg.csv_export = self.csv_export;
        cfg.trace_file = self.trace_file;
        cfg.champion_save_topology = self.champion_save_topology;
        cfg.run_name = self.run_name;
        cfg.champion_previous_dir = self.champion_previous_dir;

        Ok((cfg, verbosity))
    }
}
