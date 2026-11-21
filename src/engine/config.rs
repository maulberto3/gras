//! The aggregate run config: `RaceConfig` + its builder, defaults and snapshot round-trip.
//! Family vocabulary (stop, lottery, smoothing, cull, mode) lives with its family and is
//! re-exported here so every `engine::config::*` path keeps resolving.

use crate::evolution::selection::SelectionMethod;
use crate::graph::topology::FeatureSplitPolicy;
use crate::utils::fitness::Metric;

pub use super::breed::cull::{CrossCullPolicy, MutationCullPolicy};
pub use super::reporting::logging::LogLevel;
pub use super::clock::lottery::{
    DEFAULT_TRAIN_MERIT_LEARN_ONLY, DEFAULT_TRAIN_MERIT_P_MAX, DEFAULT_TRAIN_MERIT_P_MIN,
    DEFAULT_TRAIN_TICKET, TrainTicket,
};
pub use super::diversity::novelty::{DEFAULT_MUTATION_NOVELTY_DRAWS, DEFAULT_MUTATION_NOVELTY_MIN};
pub use super::start::run_spec::{ModeConfig, RunMode};
pub use super::clock::smoothing::{DEFAULT_MEAN_TYPE, MeanType, SMOOTHING_WINDOW};
pub use super::clock::stop::{DEFAULT_MAX_STEPS, RaceSnapshot, StopFn, StopReason};
pub use crate::evolution::crossover::CrossoverOp;
pub use crate::graph::topology::{
    DEFAULT_HIDDEN_DIM_STRIDE, DEFAULT_HIDDEN_POOL, DEFAULT_TOPOLOGY_DRIFT,
};
pub use crate::state::RunMetaCtx;

// ── Consolidated defaults (one place for all engine constants) ────────────────────

/// Default for POLICY-SCOPED FITNESS (`set_fitness_policy_scoped`): `true` = the window averages only readings on the weights the net holds now; `false` = the window averages the last K READINGS, whatever weights produced them (the pre-flag…
pub const DEFAULT_POLICY_SCOPED_FITNESS: bool = true;

/// Default for the ANTI-CHAMPION dump (`set_worst_save_topology`): `false` = no `worst-*.md` files. `true` writes the `champion_count` nets at the BOTTOM of the final table…
pub const DEFAULT_WORST_SAVE_TOPOLOGY: bool = false;

/// Steps between population checkpoints — the cadence the stream's era rotation and the CSV flush
/// key off.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 25;

/// Default for `RaceConfig::run::save_weights_at_stop`: OFF. The stop snapshot is a RECORD of the
/// race; the checkpoint is the resume artifact that actually needs weights.
pub const DEFAULT_SAVE_WEIGHTS_AT_STOP: bool = false;

/// Default population size: `10`, deliberately conservative.
pub const DEFAULT_POP_SIZE: usize = 10;

/// Default turnover per channel: the share of the live population replaced per clock.
pub const DEFAULT_CROSSOVER_SHARE: f32 = 0.05;
pub const DEFAULT_MUTATION_SHARE: f32 = 0.02;

/// Default champion set: the reference's `pop / 10`, read at the conservative default population
/// 10 — one saved net.
pub const DEFAULT_CHAMPION_COUNT: usize = DEFAULT_POP_SIZE / 10;

/// Default cull shield for a newborn immigrant, counted in updates (`U`).
pub const DEFAULT_MUTATION_PROBATION_STEPS: usize = 0;

/// Default cull shield for a crossover child, counted in updates (`U`).
pub const DEFAULT_CROSSOVER_PROBATION_STEPS: usize = 0;

/// Default cap on a post-update RECOVERY streak: after a real update whose reading comes in worse than the net's standing, the engine takes up to this many extra real…
pub const DEFAULT_RECOVERY_RETRIES: usize = 0;

/// Default retry DRAW for a recovery streak (`set_pop_recovery_same_seed`): `false` (default) = every retry draws fresh; `true` = a retry re-runs the SAME train draw the update fell on (same rows, same RNG stream) — a…
pub const DEFAULT_RECOVERY_SAME_SEED: bool = false;

/// Default recovery tolerance: the drop a new reading may take against the net's standing and
/// still count as noise. `0.0` = any drop earns a rescue; `0.2` = a fifth is tolerated.
pub const DEFAULT_RECOVERY_TOLERANCE: f32 = 0.0;

/// Default INIT PROBATION (`set_pop_init_probation_steps`): `0` = nothing is withheld beyond the born shields, which is exactly what every run before the knob did.
pub const DEFAULT_INIT_PROBATION_STEPS: usize = 0;

/// Default width for a `SelectionMethod::Tournament` parent draw — the examples'.
/// `--tournament-size`. NOT the engine default draw: that is `SelectionMethod::Roulette`.
pub const DEFAULT_TOURNAMENT_SIZE: usize = 2;

/// Default: the fitter parent is the crossover base.
pub const DEFAULT_CROSSOVER_FITTER_BASE: bool = true;

/// Default: no `telemetry.jsonl` sidecar.
pub const DEFAULT_TRACE_FILE: bool = false;

/// Default: optimizer moments ride the checkpoint.
pub const DEFAULT_SAVE_OPTIM: bool = true;

/// Default: write the per-step `metrics.csv`.
pub const DEFAULT_CSV_EXPORT: bool = true;

/// Default: write the champions' topology markdown at stop.
pub const DEFAULT_CHAMPION_SAVE_TOPOLOGY: bool = true;

/// Default: inherit the newest compatible previous run's champions.
pub const DEFAULT_CHAMPION_PREVIOUS_LOAD: bool = true;

/// Default run mode: tabular.
pub const DEFAULT_RUN_MODE: RunMode = RunMode::Tabular;

/// Default share of the dataset held out for EVAL (`eval_share`): how much of the rows the race
/// RANKS on and never trains
pub const DEFAULT_EVAL_SHARE: f32 = 0.2;

/// The narrowest and widest `eval_share` a run accepts.
pub const EVAL_SHARE_MIN: f32 = 0.01;
pub const EVAL_SHARE_MAX: f32 = 0.9;

// `DEFAULT_LR` moved to `crate::utils::training`: a training scheme reads it (the reference trainer
// and the example both do), so it belongs below both layers rather than in the engine's config

/// The people in the race: seat count, the champion set, cull shields and recovery.
pub struct PopulationConfig {
    /// Number of live nets in the population.
    pub pop_size: usize,
    /// CHAMPION set: the top-k live nets by smoothed fitness, saved and inheritable at stop (minimum 1).
    pub champion_count: usize,
    /// Mutation probation: a random immigrant is cull-IMMUNE until it has spent
    /// `mutation_probation_steps` UPDATES (`U`, real optimizer steps) — not clocks. `0` = off.
    pub mutation_probation_steps: usize,
    /// Crossover probation: a crossover-born net is cull-IMMUNE until it has spent
    /// `crossover_probation_steps` UPDATES (`U`) — not clocks. `0` = off.
    pub crossover_probation_steps: usize,
    /// INIT PROBATION (`set_pop_init_probation_steps`): clocks the run spends WITHOUT evolution — no culls, no births — however cull-eligible the population already is.
    pub init_probation_steps: usize,
    /// Post-update RECOVERY streak cap: after a real update whose reading comes in WORSE than the net's standing going in, up to `recovery_retries` extra real updates are taken.
    pub recovery_retries: usize,
    /// Recovery tolerance: how far a reading may fall below the net's standing WITHOUT earning a rescue, as a FRACTION of that standing (`0.2` = a fifth).
    pub recovery_tolerance: f32,
    /// Recovery retry DRAW (`set_pop_recovery_same_seed`): `false` (default) = every retry draws fresh; `true` = every retry repeats the clock's OWN train draw.
    pub recovery_same_seed: bool,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            pop_size: DEFAULT_POP_SIZE,
            champion_count: DEFAULT_CHAMPION_COUNT,
            mutation_probation_steps: DEFAULT_MUTATION_PROBATION_STEPS,
            crossover_probation_steps: DEFAULT_CROSSOVER_PROBATION_STEPS,
            init_probation_steps: DEFAULT_INIT_PROBATION_STEPS,
            recovery_retries: DEFAULT_RECOVERY_RETRIES,
            recovery_tolerance: DEFAULT_RECOVERY_TOLERANCE,
            recovery_same_seed: DEFAULT_RECOVERY_SAME_SEED,
        }
    }
}

/// The two turnover channels — each with its own EVICTION policy — plus the novelty gate.
pub struct EvolutionConfig {
    /// CROSSOVER turnover as a share of the live population per clock: `0.1` = 10%, dealt as `round(share · pop)` rolls EVERY clock (at least one).
    pub crossover_share: f32,
    /// MUTATION turnover — the mutation channel's own knob — as a share of the live population per clock. `0.0` switches the channel off.
    pub mutation_share: f32,
    /// Who a surviving crossover child evicts (crossover-only; immigrants follow `mutation_cull_policy`).
    pub crossover_cull_policy: CrossCullPolicy,
    /// Who a firing mutation roll evicts (mutation-only; crossover children follow `crossover_cull_policy`).
    pub mutation_cull_policy: MutationCullPolicy,
    /// Crossover operator pool (`"one_point"` | `"uniform"`), drawn uniformly per attempt. Empty ⇒ both.
    pub crossover_ops_pool: Vec<String>,
    /// How a crossover attempt draws its parents: [`SelectionMethod::Roulette`] (default), a `Tournament { k }`, or [`SelectionMethod::Elite`].
    pub crossover_selection: SelectionMethod,
    /// Fitter parent as the crossover BASE (`set_crossover_fitter_base`).
    pub crossover_fitter_base: bool,
    /// Novelty floor `d_min` on a mutation draw: the minimum Δ to the NEAREST blueprint this run already made. `0.0` = off.
    pub mutation_novelty_min: f32,
    /// Candidates one mutation may spend clearing that floor (`set_mutation_novelty_draws`).
    pub mutation_novelty_draws: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            crossover_share: DEFAULT_CROSSOVER_SHARE,
            mutation_share: DEFAULT_MUTATION_SHARE,
            crossover_cull_policy: CrossCullPolicy::default(),
            mutation_cull_policy: MutationCullPolicy::default(),
            crossover_ops_pool: Vec::new(),
            crossover_selection: SelectionMethod::default(),
            crossover_fitter_base: DEFAULT_CROSSOVER_FITTER_BASE,
            mutation_novelty_min: DEFAULT_MUTATION_NOVELTY_MIN,
            mutation_novelty_draws: DEFAULT_MUTATION_NOVELTY_DRAWS,
        }
    }
}

/// How readings become a ranking: the window, its mean, and the metric terms.
pub struct RankingConfig {
    /// Per-net smoothed-fitness rolling window (K) — how many recent step fitnesses a ranking decision averages over.
    pub smoothing_window: usize,
    /// How the window's readings are averaged into the ranking centre (`set_fitness_mean_type`).
    pub mean_type: MeanType,
    /// POLICY-SCOPED FITNESS (`set_fitness_policy_scoped`): `true` builds the mean/spread/n ONLY from readings taken on the weights the net holds now.
    pub policy_scoped_fitness: bool,
    /// Informative (non-ranking) metrics; their labels determine the extra per-net metric columns a reader may expect.
    pub metrics: Vec<Metric>,
    /// Metrics that ALSO feed the ranking (`set_selection_metric`): each adds `weight × the metric's smoothed window centre`.
    pub selection_metrics: Vec<crate::utils::fitness::SelectionMetric>,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            smoothing_window: SMOOTHING_WINDOW,
            mean_type: DEFAULT_MEAN_TYPE,
            policy_scoped_fitness: DEFAULT_POLICY_SCOPED_FITNESS,
            metrics: Vec::new(),
            selection_metrics: Vec::new(),
        }
    }
}

/// The training lottery: the band, its legs, and the learn-only gate.
pub struct LotteryConfig {
    /// Lottery floor `p_min`: the worst net's chance of a ticket each clock.
    pub train_merit_p_min: f32,
    /// Lottery ceiling `p_max`: the best net's chance of a ticket each clock.
    pub train_merit_p_max: f32,
    /// The lottery's two LEGS (`set_train_ticket`): which fit place earns the ticket and which side of the updates field it multiplies in.
    pub train_ticket: TrainTicket,
    /// LEARN-ONLY lottery (`set_ticket_learn_only`): `true` = a lost ticket costs the weight UPDATE, not the STEP.
    pub train_merit_learn_only: bool,
}

impl Default for LotteryConfig {
    fn default() -> Self {
        Self {
            train_merit_p_min: DEFAULT_TRAIN_MERIT_P_MIN,
            train_merit_p_max: DEFAULT_TRAIN_MERIT_P_MAX,
            train_ticket: DEFAULT_TRAIN_TICKET,
            train_merit_learn_only: DEFAULT_TRAIN_MERIT_LEARN_ONLY,
        }
    }
}

/// The search space every newborn is drawn from.
pub struct TopologyConfig {
    /// Hidden-dim sampling range (`None` ⇒ default 16..=64).
    pub hidden_dim_pool: Option<std::ops::RangeInclusive<usize>>,
    /// Stride within the hidden-dim pool when sampling node dimensions.
    pub hidden_dim_stride: usize,
    /// Combine pools actually used by the run; empty uses known ops (`empty ⇒ all`).
    pub combine_op_pool: Vec<String>,
    /// The feature-split POOL newborns draw from: `none`, `random`, `exclusive`.
    pub feature_split_pool: Vec<String>,
    /// Per-SLOT drift on a crossover child's INHERITED genes (`set_topology_drift`).
    pub drift_prob: f32,
    /// Activation pool (empty ⇒ all known activations).
    pub activation_pool: Vec<String>,
    /// Standardize-op pool (empty ⇒ all known ops).
    pub standardize_op_pool: Vec<String>,
    /// Topology template (input/output dims, hidden node ranges, dropout, etc.) shared by every individual in the run.
    pub options: crate::graph::topology::TopologyOptions,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            hidden_dim_pool: Some(DEFAULT_HIDDEN_POOL),
            hidden_dim_stride: DEFAULT_HIDDEN_DIM_STRIDE,
            combine_op_pool: Vec::new(),
            feature_split_pool: Vec::new(),
            drift_prob: DEFAULT_TOPOLOGY_DRIFT,
            activation_pool: Vec::new(),
            standardize_op_pool: Vec::new(),
            options: crate::graph::topology::TopologyOptions::default(),
        }
    }
}

/// When the race ends.
pub struct StopConfig {
    /// Max steps before stopping (None = no limit).
    pub max_steps: Option<usize>,
    /// Pluggable stop criterion **in addition** to the built-ins. When `None`, only the built-ins apply.
    pub custom_stop: StopFn,
}

impl Default for StopConfig {
    fn default() -> Self {
        Self {
            max_steps: Some(DEFAULT_MAX_STEPS),
            custom_stop: None,
        }
    }
}

/// Run plumbing: durability, reporting, artifacts and inheritance.
pub struct RunSettings {
    /// Steps between population checkpoints (the stream's era rotation and the CSV flush cadence).
    pub checkpoint_every: usize,
    /// Write every live net's WEIGHTS at the terminal stop. **Off by default**: the stop is a reading; the checkpoint is the resume artifact.
    pub save_weights_at_stop: bool,
    /// Write each live net's OPTIMIZER state beside its weights (`nets/<hash>.optim`).
    pub save_optim: bool,
    /// Whether to write `history.csv` and `attempts.csv`.
    pub csv_export: bool,
    /// Write `<run_dir>/telemetry.jsonl`: one JSON record per event at full fidelity.
    pub trace_file: bool,
    /// Per-step log verbosity for the engine.
    pub log_level: LogLevel,
    /// Human label for the experiment, recorded verbatim in `engine.json`.
    pub run_name: Option<String>,
    /// Share of the dataset held out for EVAL: the rows a net is SCORED on and never trained.
    pub eval_share: f32,
    /// At stop, save each champion's topology markdown. Default true.
    pub champion_save_topology: bool,
    /// AUTOMATIC inheritance: load the newest compatible PREVIOUS experiment's champions into first founder slots.
    pub champion_previous_load: bool,
    /// Where `champion_previous_load` looks: a run dir or a dir of run dirs. `None` ⇒ `results/`.
    pub champion_previous_dir: Option<std::path::PathBuf>,
    /// ANTI-CHAMPION dump (`set_worst_save_topology`): `true` writes `worst-*.md` files at stop.
    pub worst_save_topology: bool,
}

impl Default for RunSettings {
    fn default() -> Self {
        Self {
            checkpoint_every: DEFAULT_CHECKPOINT_EVERY,
            save_weights_at_stop: DEFAULT_SAVE_WEIGHTS_AT_STOP,
            save_optim: DEFAULT_SAVE_OPTIM,
            csv_export: DEFAULT_CSV_EXPORT,
            trace_file: DEFAULT_TRACE_FILE,
            log_level: LogLevel::default(),
            run_name: None,
            eval_share: DEFAULT_EVAL_SHARE,
            champion_save_topology: DEFAULT_CHAMPION_SAVE_TOPOLOGY,
            champion_previous_load: DEFAULT_CHAMPION_PREVIOUS_LOAD,
            champion_previous_dir: None,
            worst_save_topology: DEFAULT_WORST_SAVE_TOPOLOGY,
        }
    }
}

/// The aggregate run config. A knob's path says what it governs:
/// `config.lottery.train_merit_p_min`, `config.stop.max_steps`, `config.topology.drift_prob`.
/// Mode is the one split that stays at the top: the arm IS the mode.
pub struct RaceConfig {
    /// Seats, champions, cull shields and recovery.
    pub population: PopulationConfig,
    /// The two turnover channels and the novelty gate.
    pub evolution: EvolutionConfig,
    /// The smoothing window and the metric terms.
    pub ranking: RankingConfig,
    /// The ticket band, its legs and the learn-only gate.
    pub lottery: LotteryConfig,
    /// The search space every newborn is drawn from.
    pub topology: TopologyConfig,
    /// The step budget and the custom stop predicate.
    pub stop: StopConfig,
    /// Durability, reporting, artifacts and inheritance.
    pub run: RunSettings,
    /// The training paradigm / problem space target.
    pub mode: RunMode,
    /// Mode-specific knobs (tabular vs RL) — the arm is carried in the type.
    pub mode_specific: ModeConfig,
}

// `run_progress` and `DEFAULT_LR` live in `crate::utils::training` — the run
// loop and any training scheme both need them, so they sit below both layers.

impl RaceConfig {
    /// Returns the device the engine should run on (CUDA if the cuda feature
    /// enabled, CPU otherwise). Mirrors ``crate::auto_device()``.
    pub fn device(&self) -> flodl::Device {
        #[cfg(feature = "cuda")]
        {
            flodl::Device::CUDA(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            flodl::Device::CPU
        }
    }

    /// The loop the reference run was tuned on (`examples/cartpole.rs`): two rolls a clock on each
    /// channel, tournament-2 crossover parents, a 5-reading window, the inverted lottery
    pub fn defaults() -> Self {
        RaceConfig {
            population: PopulationConfig::default(),
            evolution: EvolutionConfig::default(),
            ranking: RankingConfig::default(),
            lottery: LotteryConfig::default(),
            topology: TopologyConfig::default(),
            stop: StopConfig::default(),
            run: RunSettings::default(),
            mode: DEFAULT_RUN_MODE,
            mode_specific: ModeConfig::Tabular,
        }
    }

    /// Resolve [`Self::selection_metrics`] labels to indices in `metrics`, in ranking order.
    pub fn selection_metric_terms(&self, metrics: &[Metric]) -> Vec<(usize, f32)> {
        self.ranking.selection_metrics
            .iter()
            .filter_map(
                |term| match metrics.iter().position(|m| m.label == term.label) {
                    Some(index) => Some((index, term.weight)),
                    None => {
                        tracing::warn!(
                            "selection_metric {:?}: no metric with that label was registered with \
                         set_run_metrics — the term is ignored",
                            term.label
                        );
                        None
                    }
                },
            )
            .collect()
    }

    /// The pool resolved to drawable policies (`empty ⇒ all three`, in the fixed order none →
    /// random → exclusive so the draw is reproducible).
    pub fn resolved_feature_split_pool(&self) -> Result<Vec<Option<FeatureSplitPolicy>>, String> {
        super::founding::pool::resolve::<super::founding::pool::FeatureSplitPool>(self)
    }
    /// The pool as wire names, in draw order — one spelling for the log, the `engine.json` record
    /// and the resume diff.
    pub fn feature_split_pool_names(&self) -> Vec<String> {
        super::founding::pool::names::<super::founding::pool::FeatureSplitPool>(self)
    }

    /// The inverse of [`crate::state::ConfigSnapshot::from_config`]: rebuild the run-level knobs
    /// header recorded, so a resume needs only its directory.
    pub fn from_snapshot(snap: &crate::state::ConfigSnapshot) -> Self {
        let mut cfg = RaceConfig::defaults();
        cfg.population.pop_size = snap.pop_size;
        cfg.run.checkpoint_every = snap.checkpoint_every;
        cfg.run.save_weights_at_stop = snap.save_weights_at_stop;
        cfg.evolution.crossover_share = snap.crossover_share;
        cfg.evolution.mutation_share = snap.mutation_share;
        // A header written before the gate recorded no floor: `0.0` replays it unchanged.
        cfg.evolution.mutation_novelty_min = snap.mutation_novelty_min;
        cfg.evolution.mutation_novelty_draws = snap.mutation_novelty_draws.max(1);
        cfg.population.champion_count = snap.champion_count;
        cfg.evolution.crossover_ops_pool = snap.crossover_ops_pool.clone();
        cfg.evolution.crossover_fitter_base = snap.crossover_fitter_base;
        cfg.stop.max_steps = snap.max_steps;
        cfg.run.log_level = LogLevel::parse(&snap.log_level).unwrap_or(cfg.run.log_level);
        cfg.run.trace_file = snap.trace_file;
        cfg.ranking.smoothing_window = snap.smoothing_window;
        // `""` = a header written before the knob, which averaged arithmetically.
        cfg.ranking.mean_type = MeanType::parse(&snap.mean_type).unwrap_or(DEFAULT_MEAN_TYPE);
        cfg.lottery.train_ticket = TrainTicket::parse(&snap.train_ticket).unwrap_or(cfg.lottery.train_ticket);
        cfg.run.csv_export = snap.csv_export;
        cfg.run.save_optim = snap.save_optim;
        cfg.run.run_name = snap.run_name.clone();
        cfg.ranking.selection_metrics = snap.selection_metrics.clone();
        cfg.lottery.train_merit_p_min = snap.train_merit_p_min;
        cfg.lottery.train_merit_p_max = snap.train_merit_p_max;
        cfg.lottery.train_merit_learn_only = snap.train_merit_learn_only;
        cfg.ranking.policy_scoped_fitness = snap.policy_scoped_fitness;
        cfg.run.worst_save_topology = snap.worst_save_topology;
        cfg.topology.drift_prob = snap.drift_prob;
        cfg.run.champion_previous_load = snap.champion_previous_load;
        cfg.run.champion_previous_dir = snap
            .champion_previous_dir
            .as_ref()
            .map(std::path::PathBuf::from);
        cfg.run.champion_save_topology = snap.champion_save_topology;
        cfg.population.mutation_probation_steps = snap.mutation_probation_steps;
        cfg.population.crossover_probation_steps = snap.crossover_probation_steps;
        cfg.population.init_probation_steps = snap.init_probation_steps;
        cfg.population.recovery_retries = snap.recovery_retries;
        cfg.population.recovery_tolerance = snap.recovery_tolerance;
        cfg.population.recovery_same_seed = snap.recovery_same_seed;
        // Match the recorded spelling, so a renamed display label still resumes.
        let cull = snap.crossover_cull_policy.to_ascii_lowercase();
        cfg.evolution.crossover_cull_policy = if cull.contains("lowest_fitness_most_updates") {
            CrossCullPolicy::LowestFitnessMostUpdates
        } else if cull.contains("highest_ticket_chance") {
            CrossCullPolicy::HighestTicketChance
        } else if cull.contains("random") {
            CrossCullPolicy::Random
        } else {
            CrossCullPolicy::LowestFitness
        };
        let mcull = snap.mutation_cull_policy.to_ascii_lowercase();
        cfg.evolution.mutation_cull_policy = if mcull.contains("lowest_fitness_most_updates") {
            MutationCullPolicy::LowestFitnessMostUpdates
        } else if mcull.contains("highest_ticket_chance") {
            MutationCullPolicy::HighestTicketChance
        } else if mcull.contains("random") {
            MutationCullPolicy::Random
        } else if mcull.contains("lowest_fitness") {
            MutationCullPolicy::LowestFitness
        } else {
            // Anything else — including a pre-removal `inverse_fitness` — resumes at the deterministic arm.
            MutationCullPolicy::LowestFitness
        };
        let selection = snap.crossover_selection.to_ascii_lowercase();
        cfg.evolution.crossover_selection = if selection.contains("elite") {
            SelectionMethod::Elite
        } else if selection.contains("tournament") {
            let k = selection
                .split(|c: char| !c.is_ascii_digit())
                .filter_map(|n| n.parse::<usize>().ok())
                .next()
                .unwrap_or(2);
            SelectionMethod::Tournament { tournament_size: k }
        } else {
            SelectionMethod::Roulette
        };
        // Pool: a header that recorded one (a post-pool run) wins; an older header recorded
        // single policy (or none), which becomes a one-entry pool so a resume cannot silently start
        cfg.topology.feature_split_pool = if snap.feature_split_pool.is_empty() {
            vec![
                snap.feature_split
                    .clone()
                    .unwrap_or_else(|| "none".to_string()),
            ]
        } else {
            snap.feature_split_pool.clone()
        };
        cfg.mode = if snap.mode.eq_ignore_ascii_case("rl") {
            RunMode::Rl
        } else {
            RunMode::Tabular
        };
        cfg
    }

    /// Fluent builder for the non-CLI config surface, starting from `RaceConfig::defaults`.
    /// Starts TABULAR; for RL prefer [`RlRaceConfig::builder`].
    pub fn builder() -> RaceConfigBuilder {
        RaceConfigBuilder::from_config(RaceConfig::defaults())
    }
}

/// Typed front door for RL runs: `builder()` with the RL arm pre-stamped.
pub type RlRaceConfig = RaceConfig;

/// Typed front door for tabular runs (reads well at the call site).
pub type TabularRaceConfig = RaceConfig;

/// `TabularRaceConfig::builder()` — the tabular front door: a free function so the type
/// alias can't shadow `RaceConfig::builder`.
#[doc(hidden)]
pub fn tabular_race_config_builder() -> RaceConfigBuilder {
    RaceConfigBuilder::from_config(RaceConfig::defaults())
}

/// `RlRaceConfig::builder()` — the RL front door (a free function; an inherent method
/// the alias would collide with `RaceConfig::builder`).
#[doc(hidden)]
pub fn rl_race_config_builder() -> RaceConfigBuilder {
    RaceConfigBuilder::from_config(RaceConfig {
        mode_specific: ModeConfig::Rl,
        ..RaceConfig::defaults()
    })
}

impl Default for RaceConfig {
    fn default() -> Self {
        Self::defaults()
    }
}

mod builder;

pub use builder::RaceConfigBuilder;


impl RaceConfig {
    /// Resolve the crossover operator pool: the configured labels validated, or both when empty.
    pub fn resolved_crossover_ops(&self) -> Result<Vec<CrossoverOp>, String> {
        super::founding::pool::resolve::<super::founding::pool::CrossoverOpsPool>(self)
    }

    /// Resolve the activation pool: the configured labels parsed, or all known activations
    /// when empty. Unknown labels error at run start.
    pub fn resolved_activation_pool(&self) -> Result<Vec<crate::graph::node::Activation>, String> {
        super::founding::pool::resolve::<super::founding::pool::ActivationPool>(self)
    }

    /// Resolve the combine-op sampling pool — see
    /// [`RaceConfig::resolved_activation_pool`].
    pub fn resolved_combine_pool(&self) -> Result<Vec<crate::graph::node::CombineOp>, String> {
        super::founding::pool::resolve::<super::founding::pool::CombineOpPool>(self)
    }

    /// Resolve the standardize-op sampling pool — see
    /// [`RaceConfig::resolved_activation_pool`].
    pub fn resolved_standardize_pool(
        &self,
    ) -> Result<Vec<crate::graph::node::StandardizeOp>, String> {
        super::founding::pool::resolve::<super::founding::pool::StandardizePool>(self)
    }

    /// Check every NAME-based pool once, before a single net is drawn.
    pub fn validate_search_pools(&self) -> Result<(), String> {
        let errors: Vec<String> = [
            super::founding::pool::error::<super::founding::pool::CrossoverOpsPool>(self),
            super::founding::pool::error::<super::founding::pool::ActivationPool>(self),
            super::founding::pool::error::<super::founding::pool::CombineOpPool>(self),
            super::founding::pool::error::<super::founding::pool::StandardizePool>(self),
            super::founding::pool::error::<super::founding::pool::FeatureSplitPool>(self),
        ]
        .into_iter()
        .flatten()
        .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RaceConfig;

    /// A pool NAME is either drawable or a refusal — never silently widened.
    #[test]
    fn an_unparseable_pool_name_is_an_error_not_a_fallback() {
        // Empty pools are the documented "all" and must keep validating.
        assert_eq!(RaceConfig::defaults().validate_search_pools(), Ok(()));

        let cfg = RaceConfig {
            topology: crate::engine::config::TopologyConfig {
        feature_split_pool: vec!["rondom".to_string()],
        ..Default::default()
    },
            ..RaceConfig::defaults()
        };
        // The RECORD no longer invents a pool: the typed spelling is what gets written.
        assert_eq!(cfg.feature_split_pool_names(), vec!["rondom".to_string()]);
        let err = cfg
            .validate_search_pools()
            .expect_err("a typo must not resolve to a pool");
        assert!(
            err.contains("feature_split_pool") && err.contains("rondom"),
            "the error must name the pool and the entry: {err}"
        );

        // Every bad pool is reported at once, not one per run start.
        let many = RaceConfig {
            topology: crate::engine::config::TopologyConfig {
        feature_split_pool: vec!["rondom".to_string()],
        activation_pool: vec!["reluu".to_string()],
        ..Default::default()
    },
            evolution: crate::engine::config::EvolutionConfig {
        crossover_ops_pool: vec!["two_point".to_string()],
        ..Default::default()
    },
            ..RaceConfig::defaults()
        };
        let err = many.validate_search_pools().expect_err("three typos");
        for needle in [
            "feature_split_pool",
            "activation_pool",
            "crossover_ops_pool",
        ] {
            assert!(err.contains(needle), "{needle} missing from: {err}");
        }
    }

    /// Pool entries are matched by WHOLE NAME, like `activation_pool` and `combine_op_pool` —
    /// substring test turns names nobody offers into policies (`"nonexclusive"` → `none`, because
    #[test]
    fn feature_split_pool_entries_match_whole_names() {
        use crate::graph::topology::parse_feature_split_pool_entry as parse;
        // The drawable vocabulary plus the `split_*` spellings `engine.json` records.
        assert_eq!(parse("none"), Some(None));
        assert_eq!(parse("off"), Some(None));
        assert_eq!(
            parse("random"),
            Some(Some(super::FeatureSplitPolicy::Random))
        );
        assert_eq!(
            parse("split_random"),
            Some(Some(super::FeatureSplitPolicy::Random))
        );
        assert_eq!(
            parse("Exclusive"),
            Some(Some(super::FeatureSplitPolicy::Exclusive))
        );
        assert_eq!(
            parse("split_exclusive"),
            Some(Some(super::FeatureSplitPolicy::Exclusive))
        );
        // Near-misses that the substring test used to accept.
        for bad in [
            "nonexclusive",
            "random_only",
            "exclusive_",
            "split_inherited",
            "",
        ] {
            assert_eq!(parse(bad), None, "{bad:?} is not a pool name");
        }
        let cfg = RaceConfig {
            topology: crate::engine::config::TopologyConfig {
        feature_split_pool: vec!["nonexclusive".to_string()],
        ..Default::default()
    },
            ..RaceConfig::defaults()
        };
        assert!(cfg.validate_search_pools().is_err());
    }

    /// The ranking has no risk knob: the window score is the family CENTRE, so two nets
    /// with the same centre tie here regardless of how volatile one of them was.
    #[test]
    fn the_ranking_score_is_the_centre_with_no_spread_charge() {
        let cfg = RaceConfig::defaults();
        let ran = crate::engine::clock::smoothing::rolling_stats_with(
            &crate::engine::clock::smoothing::RollingBuffer::from_values(
                5,
                &[40.0, 500.0, 40.0, 40.0, 40.0],
            ),
            cfg.ranking.mean_type,
        )
        .unwrap();
        let steady = crate::engine::clock::smoothing::rolling_stats_with(
            &crate::engine::clock::smoothing::RollingBuffer::from_values(
                5,
                &[40.0, 40.0, 40.0, 40.0, 40.0],
            ),
            cfg.ranking.mean_type,
        )
        .unwrap();
        assert!(
            steady.score() <= ran.score(),
            "the calm window is not penalised"
        );
        assert_eq!(ran.score(), ran.fam_center, "the score IS the centre");
    }

    /// The parent draw defaults to fitness-proportionate ROULETTE; the tournament and elite arms
    /// must survive the round-trip untouched.
    #[test]
    fn crossover_selection_defaults_to_roulette_and_round_trips() {
        use crate::evolution::selection::SelectionMethod;
        assert_eq!(
            RaceConfig::defaults().evolution.crossover_selection,
            SelectionMethod::Roulette,
            "the knob's default is the fitness-proportionate draw"
        );
        let cfg = RaceConfig::builder()
            .set_crossover_selection(SelectionMethod::Tournament { tournament_size: 3 })
            .build();
        assert_eq!(
            cfg.evolution.crossover_selection,
            SelectionMethod::Tournament { tournament_size: 3 }
        );
        // The recorded label is the Display form the snapshot stores.
        assert_eq!(cfg.evolution.crossover_selection.to_string(), "tournament(size=3)");
        assert_eq!(SelectionMethod::Roulette.to_string(), "roulette");
        assert_eq!(SelectionMethod::Elite.to_string(), "elite");
    }

    /// The stop field setters are mutually exclusive BY CONSTRUCTION: each write clears
    /// sibling, so "const default then flag wins" composes.
    #[test]
    fn stop_setters_are_exclusive_last_writer_wins() {
        // (the old max_target_fitness exclusivity test shrank
        // feature's deletion — only the step budget remains)
        let cfg = RaceConfig::builder().set_stop_max_steps(15).build();
        assert_eq!(cfg.stop.max_steps, Some(15));
    }
}
