//! The step-race knob surface: `RaceConfig`, `StopReason`, the pluggable
//! stop closure alias, the read-only `RaceSnapshot`, and the run-level
//! context stamped into every net's `meta` block (`RunMetaCtx`).

use crate::engine::fitness::Metric;
use crate::graph::topology::TopologyOptions;

/// What the scheduler checks to decide whether to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxSteps,
    TargetScore,
    /// Population smoothed-fitness std fell below `fitness_std_threshold`:
    /// the nets have converged to (near-)identical quality — further steps
    /// are unlikely to differentiate them.
    FitnessStd,
    /// A user-supplied `custom_stop` closure returned true (Iter 5 pluggable
    /// contract; consulted after all built-ins).
    CustomStop,
}

// ── Consolidated defaults (one place for all engine constants) ────────────────────

/// Per-net smoothed-fitness rolling window (K) — the span every ranking
/// decision averages over.
pub const SMOOTHING_WINDOW: usize = 10;

/// Steps between population checkpoints — the cadence every evolution gate
/// hangs off.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 10;

/// Learning rate default for the Adam optimizers.
pub const DEFAULT_LR: f32 = 1e-3;

/// Per-step log verbosity for the race engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogLevel {
    /// Silent per step. Only the run start line and the final stop reason are
    /// printed — nothing per step, no per-net detail, no evolve lines mid-run.
    /// Fastest I/O path for power users who parse `engine.json` + the final
    /// stop line rather than watching the log stream.
    None,
    /// One compact line per step: pop size, loss range+mean, fitness range+mean,
    /// plus a short evolve note on the same line when crossover/mutation fires.
    #[default]
    Summ,
    /// A comfy framed table per step (from step 2 onward — step 1 has no prior
    /// step to diff against, so it is skipped). Reuses the engine's existing
    /// per-step reporting: population rollup stats with deltas vs the last step
    /// (delta omitted when it is exactly 0), the evolve counters (culls, random
    /// inserts, crossover attempted/passed/failed-by-gate, mutation attempted),
    /// and the current best net. No per-net detail lines, no separate rollup
    /// line — just the table.
    Minimal,
    /// One compact line per step plus per-net detail lines and the checkpoint
    /// diagnostic. The readable, sequential evolve block lives here:
    /// what fired, what was checked against gates, what passed/failed, what
    /// entered or left the population, and the net pop change.
    Full,
}

/// How strict the checkpoint gate is for a crossover child.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrossoverGating {
    /// **Hard** — the child's smoothed fitness must beat the population's
    /// mean at **every** checkpoint it passes through. Brutal: forces
    /// children that truly outperform the pop at each historical point.
    #[default]
    Hard,
    /// **Soft** — the child must beat the **mean of the checkpoint means**
    /// (one aggregate bar over the whole replay window), not each individual
    /// checkpoint. A child that dips under one early gate but recovers can
    /// still survive.
    Soft,
}

/// Who gets evicted when a crossover child passes the gate and takes a slot.
/// **Crossover-only** — the mutation/immigrant channel has its own
/// fitness-inverse victim selection and never consults this policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrossCullPolicy {
    /// **Worst** (default) — evict the current worst net by smoothed fitness.
    /// Strictly merit-based: the elite is never touched (short of stagnation),
    /// and every admitted child strictly improves the population's floor.
    /// Risk: the population can ossify around one strong lineage.
    #[default]
    Worst,
    /// **Random** — evict a uniformly random live net (possibly a good one).
    /// Gives every net a finite expected lifetime regardless of rank, which
    /// keeps slots turning over and prevents a long-lived leader from
    /// starving diversity. Risk: a strong net can be lost to bad luck.
    Random,
}

/// The two-parent crossover operators. Drawn from `RaceConfig::crossover_ops_pool`
/// (empty ⇒ both) per attempt; recombines two parent topologies in place and
/// the engine keeps parent A's post-swap body as the child.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossoverOp {
    /// Swap the hidden-node tail after a matching pivot node.
    OnePoint,
    /// Per-node independent swap (requires equal hidden-node counts).
    Uniform,
}

impl CrossoverOp {
    /// How many parent topologies this operator combines. All current
    /// operators are binary — the asexual "clone" path lives in the
    /// `mutate_prob` roll family, not here — so both return 2. The engine
    /// derives parent count from this, keeping the config surface free of a
    /// redundant knob.
    pub fn required_parents(&self) -> usize {
        match self {
            CrossoverOp::OnePoint | CrossoverOp::Uniform => 2,
        }
    }
}

/// The training paradigm / problem space the run targets.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub enum RunMode {
    #[default]
    Tabular,
    OneCImage,
    ThreeCImage,
    Nlp,
    Rl,
}

/// Step-race knob surface. Defaults are conservative and small for quick
/// testing (pop 5, grace 15, threshold 0.20, batch 16, train_eval_split_ratio 0.2,
/// held_out_eval_rows 256, hidden_dim_pool 4..=8) — the user runs bigger or
/// explicitly overrides when it matters.
///
/// No `Clone`/`Debug` derive: the pluggable Iter-5 closure fields
/// (`Option<Box<dyn Fn>>`) support neither, and nothing in the crate needs to
/// clone or pretty-print a config.
pub struct RaceConfig {
    /// Number of live nets in the population.
    pub pop_size: usize,
    /// Steps between population checkpoints. The checkpoint ledger (recorded
    /// population-mean smoothed fitness per checkpoint step) is the bar a
    /// crossover child must clear at every gate to be inserted.
    pub checkpoint_every: usize,
    /// Independent crossover rolls per step. Each roll fires with
    /// `crossover_prob`; a firing roll produces one checkpoint-gated child.
    pub crossover_rolls: usize,
    /// Independent mutation rolls per step. Each roll fires with
    /// `mutate_prob`; a firing roll culls one fitness-inverse-selected net
    /// and inserts a fully-random immigrant (no checkpoint gate).
    pub mutate_rolls: usize,
    /// Checkpoint gate strictness for crossover children (see
    /// [`CrossoverGating`]).
    pub crossover_gating: CrossoverGating,
    /// Extra retries per crossover roll when the gate rejects the child
    /// (`cx_retry_full`). Each retry draws fresh parents and re-runs the full
    /// generate + gate pipeline; the original attempt plus every retry is
    /// recorded in `history.csv`. `0` (default) = one attempt per roll,
    /// gate failure discards the roll (legacy behavior).
    pub crossover_retries: usize,
    /// Who a surviving crossover child evicts (see [`CrossCullPolicy`]).
    /// Crossover-only: mutation immigrants always evict via the
    /// fitness-inverse roulette, regardless of this setting.
    pub crossover_cull_policy: CrossCullPolicy,
    /// Elite guard: the top-k live nets (by smoothed fitness) are immune to
    /// ALL culls — crossover (any policy) and mutation alike. `0` disables
    /// the guard (nothing is protected). The effective guard is clamped to
    /// `live_count − 1` so a cullable victim always exists. Elites still age
    /// out of the set when their smoothed fitness drops out of the top-k —
    /// "elite" is a rank, not an identity.
    pub elite_count: usize,
    /// Crossover operator pool — which recombination operators the two-parent
    /// path may use (`"one_point"` | `"uniform"`). Drawn uniformly per
    /// attempt. Empty ⇒ both operators (the `empty ⇒ all` convention shared
    /// with the activation/combine/standardize pools).
    pub crossover_ops_pool: Vec<String>,
    /// Max steps before stopping (None = no limit).
    pub max_steps: Option<usize>,
    /// Target fitness: stop when the best smoothed fitness reaches this
    /// (None = no target stop). Compared under the fitness direction.
    pub target_score: Option<f32>,
    /// Convergence stop: stop when the population's smoothed-fitness std
    /// falls below this (None = disabled). Low std = the nets agree — they
    /// are all equally good (converged) or equally stuck (stagnated); either
    /// way the race has stopped differentiating them. Guarded by
    /// `fitness_std_min_steps` so an early "everyone equally bad" phase
    /// cannot trigger it.
    pub fitness_std_threshold: Option<f32>,
    /// Minimum step before the fitness-std stop may fire. Default 0 = no
    /// warmup (eligible immediately) — but the threshold defaults to None,
    /// so the criterion is fully off unless explicitly enabled.
    pub fitness_std_min_steps: usize,
    // NOTE: the shared batch stream (batch size, split ratio, eval rows) is
    // NOT a RaceConfig knob. It is engine infrastructure, rebuilt each run
    // from the trainer's optional stream_shape() request + the dataset's
    // seeded split. The trainer owns batch geometry; the engine owns data
    // integrity (split ratio) and can always read the effective shape via
    // stream_info().
    /// Hidden-dim sampling range, same convention as the old generational
    /// engine's `set_hidden_dim_pool(min, max)`. When `None`, the default
    /// 4..=8 range is used.
    pub hidden_dim_pool: Option<std::ops::RangeInclusive<usize>>,
    /// Stride within the hidden-dim pool when sampling node dimensions.
    pub hidden_dim_stride: usize,
    /// Combine/activation/standardize pools actually used by the run.
    /// When empty, the run uses the pool of all known ops for that category
    /// (same `empty ⇒ all` convention the generational engine already uses
    /// in `validate_and_fill_options`). Keeps the race config surface minimal
    /// at the type level while still leaving each pool overridable.
    pub combine_op_pool: Vec<String>,
    pub activation_pool: Vec<String>,
    pub standardize_op_pool: Vec<String>,
    /// Topology template (input/output dims, hidden node ranges, dropout, etc.)
    /// shared by every individual in the run.
    pub topology_options: crate::graph::topology::TopologyOptions,
    pub crossover_prob: f32,
    /// Probability the child gets mutated (yes/no per child). Formerly
    /// `evolve_prob` — renamed to match DEAP-style per-op probabilities.
    pub mutate_prob: f32,
    /// Pluggable stop criterion **in addition** to the built-ins (Iter 5
    /// contract). When `None`, only the built-ins apply.
    pub custom_stop: StopFn,
    /// Per-step log verbosity for the engine.
    pub log_level: LogLevel,
    /// Informative (non-ranking) metrics configured for the run, if any.
    /// Their labels determine the extra columns a reader may expect in a
    /// per-net metrics snapshot.
    pub metrics: Vec<Metric>,
    /// The training paradigm / problem space target.
    pub mode: RunMode,
    /// Whether to write the lossless unified event log (`history.csv`: per-step
    /// live-net metric rows + evolution attempt rows, typed by the `type` column).
    /// All run settings live in `engine.json`; there is no `options.csv`.
    pub csv_export: bool,
}

/// Type alias for the pluggable Iter-5 stop closure.
///
/// Plain alias (not a wrapper struct) means a run that stores a closure owns
/// it inside its single `RaceConfig`; `RaceConfig` therefore derives neither
/// `Clone` nor `Debug`.
pub type StopFn = Option<Box<dyn Fn(&RaceSnapshot) -> bool + Send + Sync + 'static>>;

/// The run-level context stamped into every net's `meta` block (see
/// [`crate::state::NetMeta`]). Built once from the header + config at engine
/// construction; cheap to pass around.
#[derive(Clone, Debug, Default)]
pub struct RunMetaCtx {
    pub input_dim: usize,
    pub output_dim: usize,
    pub batch_size: usize,
    pub dropout_prob: f32,
    pub fitness_label: String,
    pub loss_label: String,
    pub direction: String,
    pub pop_size: usize,
    pub run_seed: u64,
}

/// A cheap read-only snapshot of the live race state at decision time, passed
/// to a custom stop closure (Iter 5 pluggable stop contract). It is **not** a
/// full run dump — only the fields a stop policy would plausibly inspect: live
/// count, current step, best + worst + mean smoothed fitness, cumulative
/// culls, and how long the run has been running.
#[derive(Clone)]
pub struct RaceSnapshot {
    pub live_count: usize,
    pub step: usize,
    pub best_smoothed_fitness: f32,
    pub worst_smoothed_fitness: f32,
    pub mean_smoothed_fitness: f32,
    pub culls: usize,
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

    /// Conservative defaults for a quick smoke test. Stop budgets (max_steps,
    /// wall clock, max culls, target score) are **inactive by default**
    /// (`None` = no limit) — a run only stops when the caller asks it to,
    /// via the builder's `set_*` methods.
    pub fn defaults() -> Self {
        const DEFAULT_HIDDEN_POOL: std::ops::RangeInclusive<usize> = 4..=8;

        RaceConfig {
            pop_size: 5,
            checkpoint_every: DEFAULT_CHECKPOINT_EVERY,
            crossover_rolls: 1,
            mutate_rolls: 1,
            crossover_gating: CrossoverGating::default(),
            crossover_retries: 0,
            crossover_cull_policy: CrossCullPolicy::default(),
            elite_count: 0,
            crossover_ops_pool: Vec::new(),
            max_steps: None,
            target_score: None,
            fitness_std_threshold: None,
            fitness_std_min_steps: 0,
            hidden_dim_pool: Some(DEFAULT_HIDDEN_POOL),
            hidden_dim_stride: 16,
            combine_op_pool: Vec::new(),
            activation_pool: Vec::new(),
            standardize_op_pool: Vec::new(),
            topology_options: crate::graph::topology::TopologyOptions::default(),
            crossover_prob: 0.5,
            mutate_prob: 0.2,
            custom_stop: None,
            metrics: Vec::new(),
            log_level: LogLevel::default(),
            mode: RunMode::Tabular,
            csv_export: true,
        }
    }

    /// Fluent builder for the non-CLI config surface. Starts from
    /// [`RaceConfig::defaults`] and overrides field-by-field; `build()` returns
    /// the finished config.
    pub fn builder() -> RaceConfigBuilder {
        RaceConfigBuilder {
            cfg: RaceConfig::defaults(),
        }
    }
}

impl Default for RaceConfig {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Fluent builder over [`RaceConfig`]. Only covers the options a caller is
/// likely to set explicitly; anything not touched keeps its conservative
/// default. For rarely-used fields (e.g. `custom_stop`), set them directly on
/// the config after `build()`.
pub struct RaceConfigBuilder {
    cfg: RaceConfig,
}

impl RaceConfigBuilder {
    pub fn set_pop_size(mut self, n: usize) -> Self {
        self.cfg.pop_size = n;
        self
    }
    pub fn set_checkpoint_every(mut self, n: usize) -> Self {
        self.cfg.checkpoint_every = n.max(1);
        self
    }
    pub fn set_crossover_rolls(mut self, n: usize) -> Self {
        self.cfg.crossover_rolls = n;
        self
    }
    pub fn set_mutate_rolls(mut self, n: usize) -> Self {
        self.cfg.mutate_rolls = n;
        self
    }
    /// Gate strictness for crossover children: `CrossoverGating::Hard` = beat
    /// every checkpoint mean; `CrossoverGating::Soft` = beat the mean of the
    /// checkpoint means.
    pub fn set_crossover_gating(mut self, mode: CrossoverGating) -> Self {
        self.cfg.crossover_gating = mode;
        self
    }
    /// Set how many extra full retries a crossover roll gets after a gate
    /// rejection (each retry re-selects parents and re-runs generate + gate).
    /// `0` = no retries (legacy: a rejected child discards the roll).
    pub fn set_crossover_retries(mut self, n: usize) -> Self {
        self.cfg.crossover_retries = n;
        self
    }
    /// Crossover replacement policy: `CrossCullPolicy::Worst` = evict the
    /// worst net by smoothed fitness (default); `CrossCullPolicy::Random` =
    /// evict a uniformly random live net (diversity-first, elite can be lost).
    /// Applies ONLY to crossover children — mutation immigrants always evict
    /// via the fitness-inverse roulette.
    pub fn set_crossover_cull_policy(mut self, policy: CrossCullPolicy) -> Self {
        self.cfg.crossover_cull_policy = policy;
        self
    }
    /// Elite guard: top-k nets by smoothed fitness are immune to ALL culls
    /// (crossover and mutation). `0` = no guard (default). Elites hold rank,
    /// not identity — a declining net falls out of the set naturally.
    pub fn set_elite_count(mut self, n: usize) -> Self {
        self.cfg.elite_count = n;
        self
    }
    /// Crossover operator pool (`"one_point"` | `"uniform"`). Drawn uniformly
    /// per crossover attempt. Empty (default) ⇒ both operators.
    pub fn set_crossover_ops_pool(mut self, ops: Vec<String>) -> Self {
        self.cfg.crossover_ops_pool = ops;
        self
    }
    /// Same as [`Self::set_crossover_ops_pool`] with string slices.
    pub fn set_crossover_ops_pool_from_strs(self, ops: &[&str]) -> Self {
        self.set_crossover_ops_pool(ops.iter().map(|s| s.to_string()).collect())
    }
    pub fn set_max_steps(mut self, n: usize) -> Self {
        self.cfg.max_steps = Some(n);
        self
    }
    pub fn set_target_score(mut self, v: f32) -> Self {
        self.cfg.target_score = Some(v);
        self
    }
    /// Convergence stop: fire when the population's smoothed-fitness std
    /// drops below `v`. Only ONE stop criterion may be active.
    pub fn set_fitness_std_threshold(mut self, v: f32) -> Self {
        self.cfg.fitness_std_threshold = Some(v);
        self
    }
    /// Warmup for the fitness-std stop: earliest step it may fire (default
    /// 0 = eligible immediately; the threshold being None keeps it off).
    pub fn set_fitness_std_min_steps(mut self, n: usize) -> Self {
        self.cfg.fitness_std_min_steps = n;
        self
    }
    pub fn set_hidden_range(mut self, min: usize, max: usize) -> Self {
        self.cfg.hidden_dim_pool = Some(min..=max);
        self
    }
    pub fn set_hidden_dim_stride(mut self, n: usize) -> Self {
        self.cfg.hidden_dim_stride = n;
        self
    }
    pub fn set_combine_op_pool(mut self, pool: Vec<String>) -> Self {
        self.cfg.combine_op_pool = pool;
        self
    }
    pub fn set_combine_ops(self, ops: &[&str]) -> Self {
        self.set_combine_op_pool(ops.iter().map(|s| s.to_string()).collect())
    }
    pub fn set_activation_pool(mut self, pool: Vec<String>) -> Self {
        self.cfg.activation_pool = pool;
        self
    }
    pub fn set_activations(self, ops: &[&str]) -> Self {
        self.set_activation_pool(ops.iter().map(|s| s.to_string()).collect())
    }
    pub fn set_standardize_op_pool(mut self, pool: Vec<String>) -> Self {
        self.cfg.standardize_op_pool = pool;
        self
    }
    pub fn set_standardize_ops(self, ops: &[&str]) -> Self {
        self.set_standardize_op_pool(ops.iter().map(|s| s.to_string()).collect())
    }
    pub fn set_topology_options(mut self, opts: TopologyOptions) -> Self {
        self.cfg.topology_options = opts;
        self
    }
    // ── individual topology fields (direct, no need to build a full TopologyOptions)
    pub fn set_min_hidden_num_nodes(mut self, n: usize) -> Self {
        self.cfg.topology_options.min_hidden_num_nodes = n;
        self
    }
    pub fn set_max_hidden_num_nodes(mut self, n: usize) -> Self {
        self.cfg.topology_options.max_hidden_num_nodes = n;
        self
    }
    pub fn set_min_hidden_inputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.min_hidden_inputs_per_node = n;
        self
    }
    pub fn set_max_hidden_inputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.max_hidden_inputs_per_node = n;
        self
    }
    pub fn set_min_hidden_outputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.min_hidden_outputs_per_node = n;
        self
    }
    pub fn set_max_hidden_outputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.max_hidden_outputs_per_node = n;
        self
    }
    pub fn set_input_dim(mut self, n: usize) -> Self {
        self.cfg.topology_options.input_dim = Some(n);
        self
    }
    pub fn set_output_dim(mut self, n: usize) -> Self {
        self.cfg.topology_options.output_dim = Some(n);
        self
    }
    pub fn set_topology_seed(mut self, n: usize) -> Self {
        self.cfg.topology_options.topology_seed = n;
        self
    }
    pub fn set_crossover_prob(mut self, v: f32) -> Self {
        self.cfg.crossover_prob = v;
        self
    }
    pub fn set_mutate_prob(mut self, v: f32) -> Self {
        self.cfg.mutate_prob = v;
        self
    }
    // Propagation strategy when a crossover roll fires but produces no
    pub fn set_metrics(mut self, metrics: Vec<Metric>) -> Self {
        self.cfg.metrics = metrics;
        self
    }
    /// Per-step log verbosity: `Summ` (compact one-line rollup) or `Full`
    /// (also dumps every per-net detail line each step).
    pub fn set_log_level(mut self, level: LogLevel) -> Self {
        self.cfg.log_level = level;
        self
    }
    /// Dropout probability stamped into every new net's topology by the engine
    /// (immigrants + crossover children; the initial population carries its own
    /// `topology_options.dropout_prob` from config construction).
    pub fn set_dropout_prob(mut self, p: f32) -> Self {
        self.cfg.topology_options.dropout_prob = p;
        self
    }
    /// Set the training paradigm / problem space target.
    pub fn set_mode(mut self, mode: RunMode) -> Self {
        self.cfg.mode = mode;
        self
    }
    /// Set whether to write the lossless unified event log (`history.csv`).
    pub fn set_csv_export(mut self, enabled: bool) -> Self {
        self.cfg.csv_export = enabled;
        self
    }
    /// Enforce the one-stop-criterion rule: at most ONE stop budget may be
    /// active (`max_steps`, `wall_clock_seconds`, `max_culls`,
    /// `target_score`, `fitness_std_threshold`). A config with two set is a
    /// bug masquerading as flexibility — the second one to fire would
    /// silently mask the first's meaning in the stop log. Errors here, at
    /// build time, not deep in a step.
    pub(crate) fn validate_single_stop(cfg: &RaceConfig) -> Result<(), String> {
        let mut set: Vec<&str> = Vec::new();
        if cfg.max_steps.is_some() {
            set.push("max_steps");
        }
        if cfg.target_score.is_some() {
            set.push("target_score");
        }
        if cfg.fitness_std_threshold.is_some() {
            set.push("fitness_std_threshold");
        }
        if set.len() > 1 {
            return Err(format!(
                "at most ONE stop criterion may be set, found {}: {}. \
                 Pick the one that means what you intend.",
                set.len(),
                set.join(", "),
            ));
        }
        Ok(())
    }

    pub fn build(self) -> RaceConfig {
        let cfg = self.cfg;
        Self::validate_single_stop(&cfg)
            .unwrap_or_else(|e| panic!("invalid RaceConfig: {e}"));
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

    /// Resolve the crossover operator pool: the configured labels validated,
    /// or **both operators** when empty (the `empty ⇒ all` convention shared
    /// with the activation/combine/standardize pools). Unknown labels error
    /// so a typo surfaces at run start, not deep in a step.
    pub fn resolved_crossover_ops(&self) -> Result<Vec<CrossoverOp>, String> {
        if self.crossover_ops_pool.is_empty() {
            return Ok(vec![CrossoverOp::OnePoint, CrossoverOp::Uniform]);
        }
        self.crossover_ops_pool
            .iter()
            .map(|s| match s.to_lowercase().as_str() {
                "one_point" | "onepoint" => Ok(CrossoverOp::OnePoint),
                "uniform" => Ok(CrossoverOp::Uniform),
                other => Err(format!(
                    "unknown crossover op '{other}' (use \"one_point\" or \"uniform\")"
                )),
            })
            .collect()
    }

    /// Parse a standardize-op label (the `Display` form).
    fn parse_standardize(s: &str) -> Result<crate::graph::node::StandardizeOp, String> {
        use crate::graph::node::StandardizeOp::*;
        Ok(match s.to_lowercase().as_str() {
            "identity" => Identity,
            "layernorm" => LayerNorm,
            other => return Err(format!("unknown standardize op '{other}'")),
        })
    }

    /// Resolve the activation sampling pool: the configured labels parsed to
    /// their enum values, or **all known activations** when the config pool
    /// is empty (the `empty ⇒ all` convention). Unknown labels error so a
    /// typo surfaces at run start, not deep in a step.
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
            .map(|s| Self::parse_standardize(s))
            .collect()
    }
}
