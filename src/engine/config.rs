//! The step-race knob surface: `RaceConfig`, `StopReason`, the pluggable
//! stop closure alias, the read-only `RaceSnapshot`, and the run-level
//! context stamped into every net's `meta` block (`RunMetaCtx`).

use crate::engine::fitness::Metric;
use crate::graph::topology::TopologyOptions;

/// What the scheduler checks to decide whether to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxSteps,
    WallClock,
    MaxCulls,
    TargetScore,
    /// A user-supplied `custom_stop` closure returned true (Iter 5 pluggable
    /// contract; consulted after all built-ins).
    CustomStop,
}

/// Steps between population checkpoints — the cadence every evolution gate
/// hangs off.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 10;

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
    /// One compact line per step plus per-net detail lines and the checkpoint +
    /// divergence diagnostics. The readable, sequential evolve block lives here:
    /// what fired, what was checked against gates, what passed/failed, what
    /// entered or left the population, and the net pop change.
    Full,
}

/// How strict the checkpoint gate is for a crossover child.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CheckMode {
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

/// Learning rate default for the Adam optimizers.
pub const DEFAULT_LR: f32 = 1e-3;

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
    pub cross_rolls: usize,
    /// Independent mutation rolls per step. Each roll fires with
    /// `mutate_prob`; a firing roll culls one fitness-inverse-selected net
    /// and inserts a fully-random immigrant (no checkpoint gate).
    pub mutate_rolls: usize,
    /// Checkpoint gate strictness for crossover children (see [`CheckMode`]).
    pub check: CheckMode,
    /// Max steps before stopping (None = no limit).
    pub max_steps: Option<usize>,
    /// Wall-clock limit in seconds (None = no limit).
    pub wall_clock_seconds: Option<u64>,
    /// Max culls before stopping (None = no limit).
    pub max_culls: Option<usize>,
    /// Target fitness: stop when the best smoothed fitness reaches this
    /// (None = no target stop). Compared under the fitness direction.
    pub target_score: Option<f32>,
    // NOTE: the shared batch stream knobs (batch_size, train_eval_split_ratio,
    // held_out_eval_rows) live on `RunSpec::stream` — engine infrastructure,
    // not training/evolution options. See `run_spec.rs`.
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
    /// How many parents the evolved path may try to combine (1 or 2).
    pub crossover_parents: usize,
    /// Pluggable stop criterion **in addition** to the built-ins (Iter 5
    /// contract). When `None`, only the built-ins apply.
    pub custom_stop: StopFn,
    /// Per-step log verbosity for the engine.
    pub log_level: LogLevel,
    /// Informative (non-ranking) metrics configured for the run, if any.
    /// Their labels determine the extra columns a reader may expect in a
    /// per-net metrics snapshot.
    pub metrics: Vec<Metric>,
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
            cross_rolls: 1,
            mutate_rolls: 1,
            check: CheckMode::default(),
            max_steps: None,
            wall_clock_seconds: None,
            max_culls: None,
            target_score: None,
            hidden_dim_pool: Some(DEFAULT_HIDDEN_POOL),
            hidden_dim_stride: 16,
            combine_op_pool: Vec::new(),
            activation_pool: Vec::new(),
            standardize_op_pool: Vec::new(),
            topology_options: crate::graph::topology::TopologyOptions::default(),
            crossover_prob: 0.5,
            mutate_prob: 0.2,
            crossover_parents: 2,
            custom_stop: None,
            metrics: Vec::new(),
            log_level: LogLevel::default(),
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

/// Fluent builder over [`RaceConfig`]. Only covers the options a caller is
/// likely to set explicitly; anything not touched keeps its conservative
/// default. For rarely-used fields (`divergence_fn`, `custom_stop`), set them
/// directly on the config after `build()`.
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
    pub fn set_cross_rolls(mut self, n: usize) -> Self {
        self.cfg.cross_rolls = n;
        self
    }
    pub fn set_mutate_rolls(mut self, n: usize) -> Self {
        self.cfg.mutate_rolls = n;
        self
    }
    /// Gate strictness: `CheckMode::Hard` = beat every checkpoint mean;
    /// `CheckMode::Soft` = beat the mean of the checkpoint means.
    pub fn set_check(mut self, mode: CheckMode) -> Self {
        self.cfg.check = mode;
        self
    }
    pub fn set_max_steps(mut self, n: usize) -> Self {
        self.cfg.max_steps = Some(n);
        self
    }
    pub fn set_wall_clock_seconds(mut self, s: u64) -> Self {
        self.cfg.wall_clock_seconds = Some(s);
        self
    }
    pub fn set_max_culls(mut self, n: usize) -> Self {
        self.cfg.max_culls = Some(n);
        self
    }
    pub fn set_target_score(mut self, v: f32) -> Self {
        self.cfg.target_score = Some(v);
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
    pub fn set_activation_pool(mut self, pool: Vec<String>) -> Self {
        self.cfg.activation_pool = pool;
        self
    }
    pub fn set_standardize_op_pool(mut self, pool: Vec<String>) -> Self {
        self.cfg.standardize_op_pool = pool;
        self
    }
    pub fn set_topology_options(mut self, opts: TopologyOptions) -> Self {
        self.cfg.topology_options = opts;
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
    pub fn set_crossover_parents(mut self, n: usize) -> Self {
        self.cfg.crossover_parents = n;
        self
    }
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
    pub fn build(self) -> RaceConfig {
        self.cfg
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
        self.activation_pool.iter().map(|s| Self::parse_activation(s)).collect()
    }

    /// Resolve the combine-op sampling pool — see
    /// [`RaceConfig::resolved_activation_pool`].
    pub fn resolved_combine_pool(&self) -> Result<Vec<crate::graph::node::CombineOp>, String> {
        if self.combine_op_pool.is_empty() {
            return Ok(crate::evolution::pools::all_combine_ops());
        }
        self.combine_op_pool.iter().map(|s| Self::parse_combine(s)).collect()
    }

    /// Resolve the standardize-op sampling pool — see
    /// [`RaceConfig::resolved_activation_pool`].
    pub fn resolved_standardize_pool(&self) -> Result<Vec<crate::graph::node::StandardizeOp>, String> {
        if self.standardize_op_pool.is_empty() {
            return Ok(crate::evolution::pools::all_standardize_ops());
        }
        self.standardize_op_pool.iter().map(|s| Self::parse_standardize(s)).collect()
    }
}
