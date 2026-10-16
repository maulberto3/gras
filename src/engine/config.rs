//! The step-race knob surface: `RaceConfig`, `StopReason`, the pluggable
//! stop closure alias, the read-only `RaceSnapshot`, and the run-level
//! context stamped into every net's `meta` block (`RunMetaCtx`).

use crate::engine::fitness::Metric;
use crate::graph::topology::FeatureSplitPolicy;

/// What the scheduler checks to decide whether to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The configured step budget was reached.
    MaxSteps,
    /// A user-supplied `custom_stop` closure returned true (Iter 5 pluggable
    /// contract; consulted after all built-ins).
    CustomStop,
        /// Ctrl+C received: the step was abandoned at the loop boundary and the race shut
        /// down through the SAME artifact path. Always on.
    Interrupted,
}

// ── Consolidated defaults (one place for all engine constants) ────────────────────

/// Per-net smoothed-fitness rolling window (K) — the span every ranking
/// decision averages over.
pub const SMOOTHING_WINDOW: usize = 10;

/// Default risk aversion (λ, FIT): how many window standard deviations a net is
/// charged for its own spread. `1.0` = "credit the net only with the value it is
/// still expected to beat after one window of bad luck"; `0.0` = rank on the
/// plain window mean (the pre-FIT behavior).
pub const RISK_AVERSION: f32 = 1.0;

/// Steps between population checkpoints — the cadence the stream's era rotation
/// and the CSV flush key off. `0` is a legal value meaning "no tick" (see
/// [`RaceConfigBuilder::set_run_checkpoint_every`]); this is the default when unset.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 10;

/// Learning rate default for the Adam optimizers.
pub const DEFAULT_LR: f32 = 1e-3;

/// Default number of fresh holdout games the post-race guardrail plays
/// ([`RaceConfigBuilder::set_elite_guardrail_matches`]).
pub const DEFAULT_GUARDRAIL_MATCHES: usize = 16;

/// Per-step log verbosity for the race engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogLevel {
        /// Silent per step: only the run start line and final stop reason print — the
        /// fastest I/O path for users who parse `engine.json`.
    None,
    /// One compact line per step: pop size, loss range+mean, fitness range+mean,
    /// plus a short evolve note on the same line when crossover/mutation fires.
    #[default]
    Summ,
        /// A boxed table per step (from step 2): rollup stats with deltas, evolve counters,
        /// and the current best net. No per-net detail, no separate rollup line.
    Minimal,
}

impl LogLevel {
        /// Parse the engine's own level name (`none`/`summ`/`minimal`, case-insensitive) —
        /// the vocabulary of the examples' `--log-level` flag.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "silent" => Some(LogLevel::None),
            "summ" | "summary" => Some(LogLevel::Summ),
            "minimal" => Some(LogLevel::Minimal),
            _ => None,
        }
    }

        /// Parse a CLI log level into the pair `logging::init` takes: the engine's reporting
        /// SHAPE plus an optional tracing verbosity (`--log-level debug` keeps Summ's detail).
    pub fn parse_verbose(raw: &str) -> Option<(Self, Option<&'static str>)> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "debug" => Some((LogLevel::Summ, Some("debug"))),
            "trace" => Some((LogLevel::Summ, Some("trace"))),
            other => LogLevel::parse(other).map(|level| (level, None)),
        }
    }

        /// The `env_logger` verbosity under which this level's lines are visible: `Summ`
        /// prints through `info!`; `Minimal`/`None` via `println!`.
    pub fn env_filter(self) -> &'static str {
        match self {
            LogLevel::Summ => "info",
            LogLevel::Minimal | LogLevel::None => "warn",
        }
    }
}

    /// Who gets evicted when a crossover child takes a slot.
    /// **Crossover-only** — the mutation channel has its own policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrossCullPolicy {
        /// **Worst** (default) — evict the current worst net by smoothed fitness; strictly
        /// merit-based. Risk: the population can ossify around one lineage.
    #[default]
    Worst,
        /// **Random** — evict a uniformly random live net; keeps slots turning over and
        /// prevents a long-lived leader from starving diversity. Risk: losing a strong net.
    Random,
}

    /// Who gets evicted when a MUTATION roll inserts a fresh random immigrant.
    /// **Mutation-only** — the crossover channel has its own policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MutationCullPolicy {
        /// **InverseFitness** (default) — a fitness-inverse roulette over non-elite nets:
        /// the worst net weighs most, the best is never drawn. Chance is kept.
    #[default]
    InverseFitness,
        /// **Worst** — evict the worst net by smoothed fitness, deterministically (mirrors
        /// [`CrossCullPolicy::Worst`]).
    Worst,
        /// **Random** — evict a uniformly random live net, at the cost of occasionally
        /// losing a good one.
    Random,
}

    /// The two-parent crossover operators, drawn from `crossover_ops_pool` (empty ⇒ both)
    /// per attempt; parent A's post-swap body becomes the child.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossoverOp {
    /// Swap the hidden-node tail after a matching pivot node.
    OnePoint,
    /// Per-node independent swap (requires equal hidden-node counts).
    Uniform,
}

impl CrossoverOp {
        /// How many parent topologies this operator combines. All current operators are
        /// binary, so both return 2 (the engine derives parent count from this).
    pub fn required_parents(&self) -> usize {
        match self {
            CrossoverOp::OnePoint | CrossoverOp::Uniform => 2,
        }
    }
}

    // `TryFrom<&str>` lives with the enum: the parse sits where the type is, gives
    // `?`-compatible conversion, and is reachable for users' own tooling.
impl TryFrom<&str> for CrossoverOp {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "one_point" | "onepoint" => Ok(CrossoverOp::OnePoint),
            "uniform" => Ok(CrossoverOp::Uniform),
            other => Err(format!(
                "unknown crossover op '{other}' (use \"one_point\" or \"uniform\")"
            )),
        }
    }
}

    /// The training paradigm / problem space the run targets. Only `Tabular` and `Rl` are
    /// constructible; the image/NLP arms are reserved future `RunSpec` variants.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub enum RunMode {
    #[default]
    /// Supervised tabular runs.
    Tabular,
    /// Single-channel image runs (reserved).
    OneCImage,
    /// Three-channel image runs (reserved).
    ThreeCImage,
    /// NLP runs (reserved).
    Nlp,
    /// Environment-driven RL runs.
    Rl,
}

    /// Step-race knob surface. Size defaults are small for quick tests; evolution
    /// semantics default to the tuned cartpole reference. Mode knobs live in `mode_specific`.
pub struct RaceConfig {
    /// Number of live nets in the population.
    pub pop_size: usize,
        /// Steps between population checkpoints (the stream's era rotation and the
        /// CSV flush cadence key off this). **`0` = no durability tick**, and no era
        /// rotation either (one eternal era) — see
        /// [`RaceConfigBuilder::set_run_checkpoint_every`].
    pub checkpoint_every: usize,
        /// Independent crossover rolls per step; each fires with `crossover_prob`. A
        /// not-fired or failed roll is SPENT (no random fallback).
    pub crossover_rolls: usize,
        /// Independent mutation rolls per step; each fires with `mutate_prob` and culls one
        /// net (by `mutation_cull_policy`), inserting a random immigrant.
    pub mutate_rolls: usize,
        /// Who a surviving crossover child evicts (crossover-only; immigrants follow
        /// `mutation_cull_policy`).
    pub crossover_cull_policy: CrossCullPolicy,
        /// Who a firing mutation roll evicts (mutation-only; crossover children follow
        /// `crossover_cull_policy`).
    pub mutation_cull_policy: MutationCullPolicy,
        /// Elite guard: the top-k nets by smoothed fitness are immune to ALL culls (minimum
        /// 1). "Elite" is a rank, not an identity.
    pub elite_count: usize,
    /// Anti-devolution A: the top-k nets act-and-measure without weight
    /// updates (see [`RaceConfigBuilder::set_elite_freeze`]). Default `true`.
    pub freeze_elites: bool,
        /// Per-net smoothed-fitness rolling window (K) — how many recent step fitnesses every
        /// ranking decision averages over. Defaults to [`SMOOTHING_WINDOW`].
    pub smoothing_window: usize,
        /// Risk aversion λ (FIT): window standard deviations subtracted from (Maximize) or
        /// added to (Minimize) a net's smoothed mean before ranking. Defaults to
        /// [`RISK_AVERSION`]; `0.0` ranks on the plain mean.
    pub risk_aversion: f32,
        /// Mode-specific knobs (tabular vs RL) — the arm is carried in the type. The engine
        /// core reads only the shared fields; mode branches match on this arm.
    pub mode_specific: ModeConfig,
        /// Crossover operator pool (`"one_point"` | `"uniform"`), drawn uniformly per attempt.
        /// Empty ⇒ both.
    pub crossover_ops_pool: Vec<String>,
    /// Max steps before stopping (None = no limit).
    pub max_steps: Option<usize>,
        /// Hidden-dim sampling range (`None` ⇒ default 4..=8). NOTE: batch geometry lives
        /// on the trainer's `stream_shape()`, not here.
    pub hidden_dim_pool: Option<std::ops::RangeInclusive<usize>>,
    /// Stride within the hidden-dim pool when sampling node dimensions.
    pub hidden_dim_stride: usize,
        /// Combine/activation/standardize pools actually used by the run; empty uses all
        /// known ops (`empty ⇒ all` convention).
    pub combine_op_pool: Vec<String>,
    /// Feature split: how the Input node's per-port feature masks are drawn, or
    /// `None` (**default**) for no split at all — every port keeps cloning the one
    /// mixed input vector. See [`FeatureSplitPolicy`]; set with
    /// `set_topology_feature_split` (there is deliberately no bool).
    pub feature_split: Option<FeatureSplitPolicy>,
    /// Activation pool (empty ⇒ all).
    pub activation_pool: Vec<String>,
    /// Standardize-op pool (empty ⇒ all).
    pub standardize_op_pool: Vec<String>,
    /// Topology template (input/output dims, hidden node ranges, dropout, etc.)
    /// shared by every individual in the run.
    pub topology_options: crate::graph::topology::TopologyOptions,
    /// Chance a crossover roll fires.
    pub crossover_prob: f32,
        /// Per-roll chance a mutation roll FIRES and inserts a random immigrant.
        /// `mutate_rolls` controls rolls per step; this controls how often each fires.
    pub mutate_prob: f32,
    /// Pluggable stop criterion **in addition** to the built-ins (Iter 5
    /// contract). When `None`, only the built-ins apply.
    pub custom_stop: StopFn,
    /// Per-step log verbosity for the engine.
    pub log_level: LogLevel,
        /// Write `<run_dir>/telemetry.jsonl`: one JSON record per event at full fidelity.
        /// Off by default — the console is the view, this is the record.
    pub trace_file: bool,
        /// Informative (non-ranking) metrics; their labels determine the extra per-net
        /// metric columns a reader may expect.
    pub metrics: Vec<Metric>,
    /// The training paradigm / problem space target.
    pub mode: RunMode,
        /// Whether to write `history.csv` (per live net per step) and `attempts.csv` (per
        /// evolution attempt) — separate files, separate headers. No `options.csv`.
    pub csv_export: bool,
        /// Flush `history.csv` after EVERY step instead of at checkpoint/stop boundaries.
        /// Default false; true = zero-loss history at a file write per step.
    pub history_flush_each: bool,
    /// At stop, save the elite's topology markdown (`elite-<hash>.md`).
    /// Default true — the champion's blueprint is the run's headline artifact.
    pub elite_save_topology: bool,
    /// At stop, save the elite's weights as safetensors
    /// (`elite-<hash>.safetensors`). Default true.
    pub elite_save_safetensors: bool,
        /// At stop, additionally save the WORST live net's artifacts. Default false — the
        /// anti-champion is a debugging artifact.
    pub worst_save_topology: bool,
    /// At stop, additionally save the worst net's weights
    /// (`worst-<hash>.safetensors`). Default false.
    pub worst_save_safetensors: bool,
        /// Post-race pruner: when a stop fires and this is `Some`, cull all but the elite and
        /// train the champion solo for `steps` more steps. `None` (default) = stop ends the run.
    pub pop_pruner: Option<PopPruner>,
        /// Human label for the experiment, recorded verbatim in `engine.json` as
        /// `"run_name"`. Informational — does not affect the results folder name.
    pub run_name: Option<String>,
        /// User-supplied run topologies (blueprint-only, fresh weights) seeded BEFORE the
        /// random draws; duplicates are re-rolled. Recorded as `run_topology_count`.
    pub run_topologies: Vec<crate::graph::topology::Topology>,
        /// AUTOMATIC inheritance: load the newest compatible PREVIOUS experiment's
        /// elites into the first founder slots (topology-only). "Compatible" is read
        /// from the header (same mode, dims and trainer); a different experiment, or
        /// no default folder, runs all-random. Default true — turn it off to force a
        /// fresh, all-random founder batch.
    pub elite_previous_load: bool,
        /// Where `elite_previous_load` looks: a run dir or a dir of run dirs (newest run id
        /// wins). `None` ⇒ the default `results/` folder.
    pub elite_previous_dir: Option<std::path::PathBuf>,
        /// Per-checkpoint elite weight snapshot (`checkpoint-elite-<hash>.safetensors`,
        /// overwritten each checkpoint). Default true — a hard kill loses at most one interval.
    pub elite_checkpoint_weights: bool,
        /// Mutation probation: a random immigrant is cull-IMMUNE for its first
        /// `mutation_probation_steps` clocks. `0` (default) = only the empty-buffer step-0 immunity.
    pub mutation_probation_steps: usize,
        /// Crossover probation: a crossover-born net is cull-IMMUNE for its first
        /// `crossover_probation_steps` clocks. `0` (default) = only step-0 immunity.
    pub crossover_probation_steps: usize,
        /// Elite join probation: a net cannot join the elite set until it has aged
        /// `elite_join_probation_steps` clocks since arrival. Default `5`.
    pub elite_join_probation_steps: usize,
        /// Elite crown lifetime: a net on the frozen crown is culled out after
        /// `elite_crown_lifetime_steps` clocks. Default `5`; zero = permanent until rank moves.
    pub elite_crown_lifetime_steps: usize,
        /// Fresh holdout games the post-race guardrail plays. Diagnostic only — never touches
        /// a step's dynamics. **`0` = skip the exam** (no games, no verdict recorded).
        /// Default [`DEFAULT_GUARDRAIL_MATCHES`].
    pub guardrail_matches: usize,
}

    /// How far through its step budget a run is (`0.0` → `1.0`, clamped), or `None` when
    /// unbounded/degenerate. THE one progress definition; pure arithmetic, replay-safe.
pub fn run_progress(step: usize, max_steps: Option<usize>) -> Option<f32> {
    let max = max_steps?;
    if max == 0 {
        return None;
    }
    Some((step as f32 / max as f32).min(1.0))
}

    /// Mode-specific knob set — the arm is carried in the TYPE. Built via
    /// `TabularConfig::builder()`/`RlConfig::builder()`; the engine derives it from `RunSpec`.
#[derive(Clone, Debug, PartialEq)]
pub enum ModeConfig {
    /// Knobs that only exist in dataset-driven (tabular) runs.
    Tabular(TabularConfig),
    /// Knobs that only exist in environment-driven (RL) runs.
    Rl(RlConfig),
}

impl ModeConfig {
    /// The RL arm, when this IS the RL arm.
    pub fn rl(&self) -> Option<&RlConfig> {
        match self {
            ModeConfig::Rl(rl) => Some(rl),
            ModeConfig::Tabular(_) => None,
        }
    }

    /// Whether this is the RL arm (the mode-branch helper the engine core
    /// reads instead of a `RunMode` enum compare).
    pub fn is_rl(&self) -> bool {
        matches!(self, ModeConfig::Rl(_))
    }

}

    /// Tabular-arm knobs (reserved; the tabular mode has no mode-specific knobs today).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TabularConfig {}

impl TabularConfig {
    /// Builder for the tabular-only knob surface.
    pub fn builder() -> TabularConfigBuilder {
        TabularConfigBuilder::default()
    }
}

/// Builder for [`TabularConfig`].
#[derive(Debug, Default)]
pub struct TabularConfigBuilder {
    cfg: TabularConfig,
}

impl TabularConfigBuilder {
    /// Finalize the tabular arm.
    pub fn build(self) -> TabularConfig {
        self.cfg
    }
}

    /// RL-arm knobs (reserved; the RL mode has no mode-specific knobs today).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RlConfig {}

impl RlConfig {
    /// Builder for the RL knob surface.
    pub fn builder() -> RlConfigBuilder {
        RlConfigBuilder::default()
    }
}

/// Builder for [`RlConfig`].
#[derive(Debug, Default)]
pub struct RlConfigBuilder {
    cfg: RlConfig,
}

impl RlConfigBuilder {
    /// Finalize the RL arm.
    pub fn build(self) -> RlConfig {
        self.cfg
    }
}

    /// Type alias for the pluggable stop closure. Plain alias, so `RaceConfig` derives
    /// neither `Clone` nor `Debug`.
pub type StopFn = Option<Box<dyn Fn(&RaceSnapshot) -> bool + Send + Sync + 'static>>;

    /// Post-race pruner strategy: `Hard` = after a stop, cull all but the elite and keep
    /// training the champion alone for fixed extra steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PopPrunerMethod {
    /// On stop: keep only the elite, train it solo for `pop_pruner_steps`.
    #[default]
    Hard,
}

    /// Post-race pruner knob pair. `None` = off; `Some(method)` = after the stop, cull all
    /// but the elite and train them solo for `steps` more steps (no evolution/stop checks).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PopPruner {
    /// Pruner strategy.
    pub method: PopPrunerMethod,
    /// Extra solo-training steps after the stop fires.
    pub steps: usize,
}

    /// The run-level context stamped into every net's `meta` block. Built once at engine
    /// construction; cheap to pass around.
#[derive(Clone, Debug, Default)]
pub struct RunMetaCtx {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Shared-stream batch size; `None` in modes with no shared stream (RL),
    /// where the trainer owns the per-step volume.
    pub batch_size: Option<usize>,
    /// Dropout probability the run used.
    pub dropout_prob: f32,
    /// Ranking fitness label.
    pub fitness_label: String,
    /// Ranking direction, as a string.
    pub direction: String,
    /// Population size.
    pub pop_size: usize,
    /// Run seed.
    pub run_seed: u64,
}

    /// A cheap read-only snapshot of the live race state at decision time, passed to a custom
    /// stop closure. Not a full run dump.
#[derive(Clone)]
pub struct RaceSnapshot {
    /// Live nets at decision time.
    pub live_count: usize,
    /// Current step.
    pub step: usize,
        /// Best smoothed fitness — the RISK-ADJUSTED window score (FIT: `mean ∓ λσ`), which is
        /// what the race actually ranks on. At λ = 0 this is the plain window mean again.
    pub best_smoothed_fitness: f32,
    /// Worst smoothed fitness — the same risk-adjusted score, at the other end.
    pub worst_smoothed_fitness: f32,
        /// Mean of the live risk-adjusted scores (a plain arithmetic mean OF the scores, not a
        /// second smoothing pass).
    pub mean_smoothed_fitness: f32,
    /// Cumulative culls.
    pub culls: usize,
    /// Wall-clock seconds elapsed.
    pub elapsed_seconds: u64,
}

impl RaceConfig {
    /// Returns the device the engine should run on (CUDA if the cuda feature is
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

        /// Conservative defaults for a quick smoke test: stop budgets are inactive by default
        /// (`None` = no limit), so a run stops only when the caller sets a criterion.
    pub fn defaults() -> Self {
        /// Default hidden-dim sampling range.
        const DEFAULT_HIDDEN_POOL: std::ops::RangeInclusive<usize> = 4..=8;

        RaceConfig {
            pop_size: 5,
            checkpoint_every: DEFAULT_CHECKPOINT_EVERY,
            crossover_rolls: 1,
            mutate_rolls: 1,
            crossover_cull_policy: CrossCullPolicy::default(),
            mutation_cull_policy: MutationCullPolicy::Worst,
            elite_count: 1,
            freeze_elites: true,
            elite_join_probation_steps: 5,
            elite_crown_lifetime_steps: 5,
            smoothing_window: SMOOTHING_WINDOW,
            risk_aversion: RISK_AVERSION,
            mode_specific: ModeConfig::Tabular(TabularConfig::default()),
            crossover_ops_pool: Vec::new(),
            max_steps: None,
            hidden_dim_pool: Some(DEFAULT_HIDDEN_POOL),
            hidden_dim_stride: 16,
            combine_op_pool: Vec::new(),
            feature_split: None,
            activation_pool: Vec::new(),
            standardize_op_pool: Vec::new(),
            topology_options: crate::graph::topology::TopologyOptions::default(),
            crossover_prob: 0.5,
            mutate_prob: 0.5,
            custom_stop: None,
            metrics: Vec::new(),
            log_level: LogLevel::default(),
            trace_file: false,
            mode: RunMode::Tabular,
            csv_export: true,
            history_flush_each: false,
            elite_save_topology: true,
            elite_save_safetensors: true,
            worst_save_topology: false,
            worst_save_safetensors: false,
            pop_pruner: None,
            run_name: None,
            run_topologies: Vec::new(),
            elite_previous_load: true,
            elite_previous_dir: None,
            elite_checkpoint_weights: true,
            mutation_probation_steps: 5,
            crossover_probation_steps: 5,
            guardrail_matches: DEFAULT_GUARDRAIL_MATCHES,
        }
    }

        /// Fluent builder for the non-CLI config surface, starting from `RaceConfig::defaults`.
        /// Starts TABULAR; for RL prefer [`RlRaceConfig::builder`].
    pub fn builder() -> RaceConfigBuilder {
        RaceConfigBuilder {
            cfg: RaceConfig::defaults(),
            pending_pruner: None,
        }
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
    RaceConfigBuilder {
        cfg: RaceConfig::defaults(),
        pending_pruner: None,
    }
}

    /// `RlRaceConfig::builder()` — the RL front door (a free function; an inherent method on
    /// the alias would collide with `RaceConfig::builder`).
#[doc(hidden)]
pub fn rl_race_config_builder() -> RaceConfigBuilder {
    RaceConfigBuilder {
        cfg: RaceConfig {
            mode_specific: ModeConfig::Rl(RlConfig::default()),
            ..RaceConfig::defaults()
        },
        pending_pruner: None,
    }
}

impl Default for RaceConfig {
    fn default() -> Self {
        Self::defaults()
    }
}

    /// Fluent builder over [`RaceConfig`], covering options a caller is likely to set;
    /// untouched fields keep conservative defaults.
pub struct RaceConfigBuilder {
    cfg: RaceConfig,
        /// Pruner params stored while the pruner switch is off — replayed by `build()` if
        /// `set_pruner_enabled(true)` comes later. Either call order works.
    pending_pruner: Option<PopPruner>,
}

impl RaceConfigBuilder {
    /// Set the population size.
    pub fn set_run_pop_size(mut self, n: usize) -> Self {
        self.cfg.pop_size = n;
        self
    }
        /// Set steps between checkpoints — the durability tick (frontier snapshot + `elite_
        /// checkpoint_weights` snapshot + CSV flush) and the stream's eval-permutation era.
        /// **`0` = OFF**: no tick and no era rotation (one eternal era), for profiling or a
        /// smoke run that wants no `nets/` churn from checkpoints. The STOP-time writers are
        /// unaffected — `elites.json`, the elite/worst artifacts and the final CSV flush still
        /// land, so a 0-cadence run is still a complete run.
    pub fn set_run_checkpoint_every(mut self, n: usize) -> Self {
        self.cfg.checkpoint_every = n;
        self
    }
    /// Set crossover rolls per step.
    pub fn set_crossover_rolls(mut self, n: usize) -> Self {
        self.cfg.crossover_rolls = n;
        self
    }
    /// Set mutation rolls per step.
    pub fn set_mutate_rolls(mut self, n: usize) -> Self {
        self.cfg.mutate_rolls = n;
        self
    }
        /// Crossover replacement policy (crossover-only). `Worst` (default) or `Random`;
        /// mutation immigrants use `set_mutation_cull_policy`.
    pub fn set_crossover_cull_policy(mut self, policy: CrossCullPolicy) -> Self {
        self.cfg.crossover_cull_policy = policy;
        self
    }
        /// Mutation replacement policy: who a firing mutation roll evicts before inserting its
        /// immigrant. Defaults to `InverseFitness`. Mutation-only.
    pub fn set_mutation_cull_policy(mut self, policy: MutationCullPolicy) -> Self {
        self.cfg.mutation_cull_policy = policy;
        self
    }
        /// Elite guard: top-k nets are immune to ALL culls. Minimum 1; elites hold rank, not
        /// identity.
    pub fn set_elite_count(mut self, n: usize) -> Self {
        self.cfg.elite_count = n.max(1);
        self
    }
        /// ANTI-DEVOLUTION A — elite freeze: the top-`elite_count` nets are exempt from the
        /// trainer call (frozen, but still scored/ranked/parents). Default `true`.
    pub fn set_elite_freeze(mut self, yes: bool) -> Self {
        self.cfg.freeze_elites = yes;
        self
    }
        /// Per-net smoothed-fitness rolling window (K). Default [`SMOOTHING_WINDOW`] = 10.
        /// Replay-relevant: a resumed run must use the same value or construction fails parity.
    pub fn set_run_smoothing_window(mut self, k: usize) -> Self {
        self.cfg.smoothing_window = k.max(1);
        self
    }
        /// Risk aversion λ (FIT) — how many window standard deviations a net pays for its own
        /// spread, so a steady net outranks a volatile one with the same window mean.
        /// Default [`RISK_AVERSION`] = 1.0; `0.0` restores the plain-mean ranking exactly.
        /// Negative values clamp to 0.0 (a risk-SEEKING ranking is not offered) and a
        /// non-finite λ falls back to the default. Replay-relevant: a resumed run must use
        /// the same value or construction fails parity.
    pub fn set_run_risk_aversion(mut self, lambda: f32) -> Self {
        self.cfg.risk_aversion = if lambda.is_finite() {
            lambda.max(0.0)
        } else {
            RISK_AVERSION
        };
        self
    }
        /// A user-supplied stop predicate, consulted every step alongside `max_steps`;
        /// returning `true` ends the race with stop reason `Custom`.
    pub fn set_stop_custom(
        mut self,
        f: impl Fn(&RaceSnapshot) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.cfg.custom_stop = Some(Box::new(f));
        self
    }
        /// Crossover operator pool, drawn uniformly per attempt; empty (default) ⇒ both.
        /// Takes owned or borrowed strings.
    pub fn set_crossover_ops_pool<I, S>(mut self, ops: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.crossover_ops_pool = ops.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }
        /// Enable the feature split and choose how the Input node's per-port masks
        /// are drawn. Unset (`None`, the default) = no split. Off/value rather than a
        /// bool: the policy IS the value, and "off" is simply not calling this.
    pub fn set_topology_feature_split(mut self, policy: FeatureSplitPolicy) -> Self {
        self.cfg.feature_split = Some(policy);
        self
    }
        /// Explicit step budget — the only built-in stop criterion. Accepts a bare value or an
        /// `Option`; `None` (default) = run until externally stopped.
    pub fn set_stop_max_steps(mut self, n: impl Into<Option<usize>>) -> Self {
        self.cfg.max_steps = n.into();
        self
    }
        /// Whole-bundle pruner: the struct form of `set_pruner_enabled`/`_method`/`_steps`.
        /// An `enabled == false` pruner is still stored.
    pub fn set_pruner(mut self, pruner: PopPruner) -> Self {
        self.cfg.pop_pruner = Some(pruner);
        self
    }
        /// Post-race pruner switch: `true` = after a stop, cull all but the elite and keep
        /// training for `steps` more steps. `false` (default) = the stop ends the run.
    pub fn set_pruner_enabled(mut self, enabled: bool) -> Self {
        self.cfg.pop_pruner = enabled.then(|| {
            self.pending_pruner.take().unwrap_or(PopPruner {
                method: PopPrunerMethod::Hard,
                steps: 0,
            })
        });
        self
    }
        /// Pruner strategy. Takes effect only when `set_pruner_enabled(true)` is also called —
        /// either call order works.
    pub fn set_pruner_method(mut self, method: PopPrunerMethod) -> Self {
        match self.cfg.pop_pruner.as_mut() {
            Some(pruner) => pruner.method = method,
            None => {
                let mut p = self.pending_pruner.take().unwrap_or(PopPruner {
                    method: PopPrunerMethod::Hard,
                    steps: 0,
                });
                p.method = method;
                self.pending_pruner = Some(p);
            }
        }
        self
    }
        /// Extra solo-training steps after the stop. Effective only with
        /// `set_pruner_enabled(true)` — either call order works.
    pub fn set_pruner_steps(mut self, steps: usize) -> Self {
        match self.cfg.pop_pruner.as_mut() {
            Some(pruner) => pruner.steps = steps,
            None => {
                let mut p = self.pending_pruner.take().unwrap_or(PopPruner {
                    method: PopPrunerMethod::Hard,
                    steps: 0,
                });
                p.steps = steps;
                self.pending_pruner = Some(p);
            }
        }
        self
    }
    /// Set the hidden-dim sampling range.
    pub fn set_topology_hidden_dim_range(mut self, min: usize, max: usize) -> Self {
        self.cfg.hidden_dim_pool = Some(min..=max);
        self
    }
    /// Set the hidden-dim stride.
    pub fn set_topology_hidden_dim_stride(mut self, n: usize) -> Self {
        self.cfg.hidden_dim_stride = n;
        self
    }
    /// How a node merges its incoming wires (see [`crate::graph::node::CombineOp`]).
    /// Empty ⇒ the full default pool. Accepts owned or borrowed strings.
    pub fn set_topology_combine_op_pool<I, S>(mut self, pool: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.combine_op_pool = pool.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }
    /// Per-node non-linearities (see [`crate::graph::node::Activation`]).
    /// Empty ⇒ the full default pool. Accepts owned or borrowed strings.
    pub fn set_topology_activation_pool<I, S>(mut self, pool: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.activation_pool = pool.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }
    /// Per-node normalization ops (see [`crate::graph::node::StandardizeOp`]).
    /// Empty ⇒ the full default pool. Accepts owned or borrowed strings.
    pub fn set_topology_standardize_op_pool<I, S>(mut self, pool: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.standardize_op_pool = pool.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }
        /// Dropout probability applied when building this run's networks (part of
        /// `TopologyOptions`). TRAIN forwards only.
    pub fn set_topology_dropout_prob(mut self, p: f32) -> Self {
        self.cfg.topology_options.dropout_prob = p;
        self
    }
    // ── individual topology fields (direct, no need to build a full TopologyOptions)
    /// Set the min hidden nodes per individual.
    pub fn set_topology_min_hidden_num_nodes(mut self, n: usize) -> Self {
        self.cfg.topology_options.min_hidden_num_nodes = n;
        self
    }
    /// Set the max hidden nodes per individual.
    pub fn set_topology_max_hidden_num_nodes(mut self, n: usize) -> Self {
        self.cfg.topology_options.max_hidden_num_nodes = n;
        self
    }
    /// Set the min inputs per node.
    pub fn set_topology_min_inputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.min_hidden_inputs_per_node = n;
        self
    }
    /// Set the max inputs per node.
    pub fn set_topology_max_inputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.max_hidden_inputs_per_node = n;
        self
    }
    /// Set the min outputs per node.
    pub fn set_topology_min_outputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.min_hidden_outputs_per_node = n;
        self
    }
    /// Set the max outputs per node.
    pub fn set_topology_max_outputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.max_hidden_outputs_per_node = n;
        self
    }
    /// Set the input dimension.
    pub fn set_topology_input_dim(mut self, n: usize) -> Self {
        self.cfg.topology_options.input_dim = Some(n);
        self
    }
    /// Set the output dimension.
    pub fn set_topology_output_dim(mut self, n: usize) -> Self {
        self.cfg.topology_options.output_dim = Some(n);
        self
    }
    /// Set the crossover fire probability.
    pub fn set_crossover_prob(mut self, v: f32) -> Self {
        self.cfg.crossover_prob = v;
        self
    }
    /// Set the mutation fire probability.
    pub fn set_mutate_prob(mut self, v: f32) -> Self {
        self.cfg.mutate_prob = v;
        self
    }
        /// Extra informative (non-ranking) metrics recorded in history.csv and engine.json.
        /// Never affect selection — the ranking fitness is set separately via `Fitness`.
    pub fn set_run_metrics(mut self, metrics: Vec<Metric>) -> Self {
        self.cfg.metrics = metrics;
        self
    }

        /// Human label for the experiment, written as `"run_name"`. Informational only — the
        /// results folder name is unchanged.
    pub fn set_run_name(mut self, name: impl Into<String>) -> Self {
        self.cfg.run_name = Some(name.into());
        self
    }
    /// Per-step log verbosity: `Summ` (compact one-line rollup) or `Full`
    /// (also dumps every per-net detail line each step).
    pub fn set_run_log_level(mut self, level: LogLevel) -> Self {
        self.cfg.log_level = level;
        self
    }
        /// Also write the event stream to `<run_dir>/telemetry.jsonl` (one JSON record per
        /// event). Off by default.
    pub fn set_run_trace_file(mut self, enabled: bool) -> Self {
        self.cfg.trace_file = enabled;
        self
    }
        /// Set whether to write the lossless event logs (`history.csv` + `attempts.csv`).
        // NOTE: there is no `set_run_mode` — RunMode is DERIVED from the RunSpec variant.
    pub fn set_run_csv_export(mut self, enabled: bool) -> Self {
        self.cfg.csv_export = enabled;
        self
    }
        /// Seed the population with user-supplied run topologies (blueprint-only, fresh
        /// weights), filling the FIRST slots; the rest stay random. Colliders are re-rolled.
    pub fn set_run_topologies(mut self, topos: Vec<crate::graph::topology::Topology>) -> Self {
        self.cfg.run_topologies = topos;
        self
    }
        /// Opt in/out of the automatic inheritance of the newest compatible previous
        /// experiment's elites (topology-only). ON by default; `false` forces a fresh,
        /// all-random founder batch.
    pub fn set_elite_previous_load(mut self, enabled: bool) -> Self {
        self.cfg.elite_previous_load = enabled;
        self
    }
        /// Where the automatic inheritance looks: a run dir or a dir of run dirs
        /// (newest id wins). Unset ⇒ the default `results/`.
    pub fn set_elite_previous_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.cfg.elite_previous_dir = Some(dir.into());
        self
    }
        /// Per-checkpoint elite weight snapshot (overwritten each checkpoint). Default true —
        /// pass false to only write weights at stop.
    pub fn set_elite_checkpoint_weights(mut self, enabled: bool) -> Self {
        self.cfg.elite_checkpoint_weights = enabled;
        self
    }
    /// Mutation probation steps: cull-immunity for a random immigrant's first
    /// `k` clocks (see [`RaceConfig::mutation_probation_steps`]). Default 0.
    pub fn set_mutation_probation_steps(mut self, k: usize) -> Self {
        self.cfg.mutation_probation_steps = k;
        self
    }
    /// Crossover probation steps: cull-immunity for a crossover child's first
    /// `k` clocks (see [`RaceConfig::crossover_probation_steps`]). Default 0.
    pub fn set_crossover_probation_steps(mut self, k: usize) -> Self {
        self.cfg.crossover_probation_steps = k;
        self
    }
        /// Elite join probation steps: a net cannot become an elite until it has aged `k`
        /// clocks since arrival. Default `5`.
    pub fn set_elite_join_probation_steps(mut self, k: usize) -> Self {
        self.cfg.elite_join_probation_steps = k;
        self
    }
        /// Elite crown lifetime: how long a net may stay at peak on the frozen crown before
        /// age-elimination. Default `5`.
    pub fn set_elite_crown_lifetime_steps(mut self, k: usize) -> Self {
        self.cfg.elite_crown_lifetime_steps = k;
        self
    }
        /// Fresh holdout games the post-race guardrail plays ([`DEFAULT_GUARDRAIL_MATCHES`] = 16).
        /// Diagnostic only. (Challenges moved to the TRAINER side — roll `trainer::challenge_fires`.)
        /// **`0` = skip the exam entirely**: no holdout game is played, no verdict is recorded,
        /// and `engine.guardrail(..)` returns `None`. Nothing else changes — the run still ranks,
        /// stops and writes its artifacts, and the next run still inherits via `elites.json`
        /// (which never depended on the verdict).
    pub fn set_elite_guardrail_matches(mut self, n: usize) -> Self {
        self.cfg.guardrail_matches = n;
        self
    }
    /// At stop, save the elite's topology markdown (`elite-<hash>.md`).
    /// Default true.
    pub fn set_elite_save_topology(mut self, enabled: bool) -> Self {
        self.cfg.elite_save_topology = enabled;
        self
    }
    /// At stop, save the elite's weights as safetensors
    /// (`elite-<hash>.safetensors`). Default true.
    pub fn set_elite_save_safetensors(mut self, enabled: bool) -> Self {
        self.cfg.elite_save_safetensors = enabled;
        self
    }
        /// At stop, ALSO save the WORST live net's topology markdown. Useful for diffing what
        /// the search avoided. Default false.
    pub fn set_worst_save_topology(mut self, enabled: bool) -> Self {
        self.cfg.worst_save_topology = enabled;
        self
    }
    /// At stop, ALSO save the worst net's weights as safetensors
    /// (`worst-<hash>.safetensors`). Default false.
    pub fn set_worst_save_safetensors(mut self, enabled: bool) -> Self {
        self.cfg.worst_save_safetensors = enabled;
        self
    }
    /// Finish and return the configured [`RaceConfig`].
    pub fn build(mut self) -> RaceConfig {
        let cfg = self.cfg;
        // Params set by the pruner setters while the switch was still off:
        // surfaced loudly (a silent no-op would hide the misconfig).
        if let Some(p) = self.pending_pruner.take() {
            tracing::warn!(
                "set_pruner_method({:?})/set_pruner_steps({}) were called without set_pruner_enabled(true) — pruner is OFF",
                p.method,
                p.steps
            );
        }
        cfg
    }
}

impl RaceConfig {
    /// Parse an activation label (the `Display` form: "relu", "gelu", …).
    fn parse_activation(s: &str) -> Result<crate::graph::node::Activation, String> {
        use crate::graph::node::Activation::*;
        Ok(match s.to_lowercase().as_str() {
            "identity" => Identity,
            "relu" => ReLU,
            "gelu" => GeLU,
            "silu" => SiLU,
            "selu" => SELU,
            "tanh" => Tanh,
            "sigmoid" => Sigmoid,
            "mish" => Mish,
            "leaky_relu" => LeakyReLU,
            "elu" => ELU,
            "gelu_tanh" => GeluTanh,
            "softplus" => Softplus,
            "hardswish" => HardSwish,
            "hardsigmoid" => HardSigmoid,
            "sin" => Sin,
            "cos" => Cos,
            "softmax" => Softmax,
            "log_softmax" | "logsoftmax" => LogSoftmax,
            other => return Err(format!("unknown activation '{other}'")),
        })
    }

    /// Parse a combine-op label (the `Display` form: "add", "mean", …).
    fn parse_combine(s: &str) -> Result<crate::graph::node::CombineOp, String> {
        use crate::graph::node::CombineOp::*;
        Ok(match s.to_lowercase().as_str() {
            "add" => Add,
            "mean" => Mean,
            "mul" | "multiply" => Multiply,
            "sub" | "subtract" => Subtract,
            "div" | "divide" => Divide,
            "max" => Max,
            "min" => Min,
            other => return Err(format!("unknown combine op '{other}'")),
        })
    }

        /// Resolve the crossover operator pool: the configured labels validated, or both when
        /// empty. Unknown labels error at run start.
    pub fn resolved_crossover_ops(&self) -> Result<Vec<CrossoverOp>, String> {
        if self.crossover_ops_pool.is_empty() {
            return Ok(vec![CrossoverOp::OnePoint, CrossoverOp::Uniform]);
        }
        self.crossover_ops_pool
            .iter()
            .map(|s| CrossoverOp::try_from(s.as_str()))
            .collect()
    }

        /// Resolve the activation pool: the configured labels parsed, or all known activations
        /// when empty. Unknown labels error at run start.
    pub fn resolved_activation_pool(&self) -> Result<Vec<crate::graph::node::Activation>, String> {
        if self.activation_pool.is_empty() {
            return Ok(crate::evolution::pools::all_activations());
        }
        self.activation_pool
            .iter()
            .map(|s| Self::parse_activation(s))
            .collect()
    }

    /// Resolve the combine-op sampling pool — see
    /// [`RaceConfig::resolved_activation_pool`].
    pub fn resolved_combine_pool(&self) -> Result<Vec<crate::graph::node::CombineOp>, String> {
        if self.combine_op_pool.is_empty() {
            return Ok(crate::evolution::pools::all_combine_ops());
        }
        self.combine_op_pool
            .iter()
            .map(|s| Self::parse_combine(s))
            .collect()
    }

    /// Resolve the standardize-op sampling pool — see
    /// [`RaceConfig::resolved_activation_pool`].
    pub fn resolved_standardize_pool(
        &self,
    ) -> Result<Vec<crate::graph::node::StandardizeOp>, String> {
        if self.standardize_op_pool.is_empty() {
            return Ok(crate::evolution::pools::all_standardize_ops());
        }
        self.standardize_op_pool
            .iter()
            .map(|s| crate::graph::node::StandardizeOp::try_from(s.as_str()))
            .collect()
    }

}

#[cfg(test)]
mod tests {
    use super::{LogLevel, RISK_AVERSION, RaceConfig, run_progress};

        /// One progress convention for the whole crate: linear to 1.0 at the budget, clamped,
        /// and flat when unbounded.
    #[test]
    fn run_progress_clamps_and_is_flat_unbounded() {
        assert_eq!(run_progress(0, Some(10)), Some(0.0));
        assert_eq!(run_progress(5, Some(10)), Some(0.5));
        assert_eq!(run_progress(10, Some(10)), Some(1.0));
        // Overshoot clamps: a run resumed past its budget must not go negative.
        assert_eq!(run_progress(99, Some(10)), Some(1.0));
        // No budget (or a degenerate one) = no schedule at all.
        assert_eq!(run_progress(3, None), None);
        assert_eq!(run_progress(3, Some(0)), None);
    }

        /// The examples' `--log-level` vocabulary must parse: handed to env_logger verbatim,
        /// `summ` is read as a module name and mutes the whole run — this mapping is load-bearing.
    #[test]
    fn log_level_names_parse_to_a_visible_verbosity() {
        assert_eq!(LogLevel::parse("summ"), Some(LogLevel::Summ));
        assert_eq!(LogLevel::parse("SUMM"), Some(LogLevel::Summ));
        assert_eq!(LogLevel::parse("minimal"), Some(LogLevel::Minimal));
        assert_eq!(LogLevel::parse("none"), Some(LogLevel::None));
        assert_eq!(
            LogLevel::parse("debug"),
            None,
            "env_logger levels pass through"
        );
        assert_eq!(LogLevel::Summ.env_filter(), "info");
        assert_eq!(LogLevel::Minimal.env_filter(), "warn");
        assert_eq!(LogLevel::None.env_filter(), "warn");
    }

        /// FIT's one knob: the default is the risk-adjusted ranking, `0.0` is the
        /// documented OFF switch (the pre-FIT plain mean), negatives clamp to `0.0`
        /// (no risk-SEEKING ranking exists) and a non-finite λ falls back to the
        /// default rather than poisoning every comparison with NaN.
    #[test]
    fn risk_aversion_defaults_on_and_clamps_at_off() {
        // Default: the risk term is ON.
        assert_eq!(RaceConfig::defaults().risk_aversion, RISK_AVERSION);
        assert!(RISK_AVERSION > 0.0, "the feature ships enabled");
        let set = |lambda: f32| {
            RaceConfig::builder()
                .set_run_risk_aversion(lambda)
                .build()
                .risk_aversion
        };
        // Off is expressible, and it is exactly the plain mean (λ = 0).
        assert_eq!(set(0.0), 0.0);
        // A stronger preference passes through unchanged.
        assert_eq!(set(2.5), 2.5);
        // Negative = risk-seeking: not offered, clamped to off.
        assert_eq!(set(-3.0), 0.0);
        // NaN / ∞ would make every score NaN (or worse, "best" undefined).
        assert_eq!(set(f32::NAN), RISK_AVERSION);
        assert_eq!(set(f32::INFINITY), RISK_AVERSION);
        assert_eq!(set(f32::NEG_INFINITY), RISK_AVERSION);
    }

        /// The stop field setters are mutually exclusive BY CONSTRUCTION: each write clears its
        /// sibling, so "const default then flag wins" composes.
    #[test]
    fn stop_setters_are_exclusive_last_writer_wins() {
        // (the old max_target_fitness exclusivity test shrank with the
        // feature's deletion — only the step budget remains)
        let cfg = RaceConfig::builder().set_stop_max_steps(15).build();
        assert_eq!(cfg.max_steps, Some(15));
    }
}
