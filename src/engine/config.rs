//! The step-race knob surface: `RaceConfig`, `StopReason`, the pluggable
//! stop closure alias, the read-only `RaceSnapshot`, and the run-level
//! context stamped into every net's `meta` block (`RunMetaCtx`).

use crate::engine::fitness::Metric;

/// What the scheduler checks to decide whether to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxSteps,
    TargetScore,
    /// A user-supplied `custom_stop` closure returned true (Iter 5 pluggable
    /// contract; consulted after all built-ins).
    CustomStop,
    /// Ctrl+C (SIGINT) received: the in-flight step was abandoned at the next
    /// loop boundary and the race shut down through the SAME artifact path as
    /// a natural stop (pruner, frontier, champion export, guardrail). Always
    /// on — a second Ctrl+C during shutdown force-kills.
    Interrupted,
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
}

impl LogLevel {
    /// Parse the engine's own level name: `none` / `summ` / `minimal`
    /// (case-insensitive). This is the vocabulary users type on the examples'
    /// `--log-level` flag, so it must be understood there — passing `summ`
    /// straight to `env_logger` would be read as a *module name* and silence
    /// the whole run.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "silent" => Some(LogLevel::None),
            "summ" | "summary" => Some(LogLevel::Summ),
            "minimal" => Some(LogLevel::Minimal),
            _ => None,
        }
    }

    /// The `env_logger` verbosity under which this level's lines are visible.
    /// `Summ` prints through `log::info!`, so it needs `info`; `Minimal` and
    /// `None` print their own lines through `println!` and only need the
    /// chatter quieted down.
    pub fn env_filter(self) -> &'static str {
        match self {
            LogLevel::Summ => "info",
            LogLevel::Minimal | LogLevel::None => "warn",
        }
    }
}

/// How strict the checkpoint gate is for a crossover child.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrossoverGate {
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
/// **Crossover-only** — the mutation/immigrant channel has its own policy
/// ([`MutationCullPolicy`]) and never consults this one.
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

/// Who gets evicted when a MUTATION roll inserts a fresh random immigrant.
/// **Mutation-only** — the crossover channel has its own policy
/// ([`CrossCullPolicy`]). The two are deliberately separate: a crossover child
/// has just proven itself over the replay window, while an immigrant has
/// proven nothing, so "evict the worst" is not obviously the right rule for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MutationCullPolicy {
    /// **InverseFitness** (default) — a fitness-inverse roulette over the
    /// non-elite nets that have a fitness verdict: the worst net carries the
    /// largest weight, the current best is never drawn (weight 0), and a net
    /// whose fitness collapses up-weights itself naturally. Chance is kept, so
    /// a mediocre net is not condemned deterministically. This is the
    /// historical behavior, now nameable/configurable.
    #[default]
    InverseFitness,
    /// **Worst** — evict the worst net by smoothed fitness, deterministically
    /// (mirrors [`CrossCullPolicy::Worst`]). Strictly merit-based, and always
    /// the same target for a given population.
    Worst,
    /// **Random** — evict a uniformly random live net (mirrors
    /// [`CrossCullPolicy::Random`]): every net keeps a finite expected
    /// lifetime, at the cost of occasionally losing a good one.
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
///
/// Only `Tabular` and `Rl` are constructible today — they are the arms the
/// `RunSpec` variants (`RunSpec::tabular`/`RunSpec::rl`) validate against.
/// The image/NLP arms are RESERVED for future `RunSpec` variants (see the
/// engine's construction error in core.rs); they exist so persisted
/// `engine.json` files can already declare the intended paradigm and so the
/// wire format does not need re-versioning when those specs land.
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
/// testing (pop 5, checkpoint every 10, 1 crossover + 1 mutation roll,
/// elite_count 1, mutate_prob 0.2, smoothing K = 10, hidden_dim_pool 4..=8)
/// — the user runs bigger or explicitly overrides when it matters. Data
/// geometry (batch size, split ratio, held-out rows) is deliberately NOT
/// here: it lives on `RunSpec::stream` (see the batch-stream note below).
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
    /// `crossover_prob`; a firing roll produces one checkpoint-gated recombined
    /// child. A not-fired or failed roll is SPENT — no random fallback (random
    /// whole nets enter only via `mutate_rolls`).
    pub crossover_rolls: usize,
    /// Independent mutation rolls per step. Each roll fires with
    /// `mutate_prob`; a firing roll culls one net (chosen by
    /// `mutation_cull_policy`) and inserts a fully-random immigrant (no
    /// checkpoint gate).
    pub mutate_rolls: usize,
    /// Checkpoint gate strictness for crossover children (see
    /// [`CrossoverGate`]).
    pub crossover_gate: CrossoverGate,
    /// Extra retries per crossover roll when the gate rejects the child
    /// (`cx_retry_full`). Each retry draws fresh parents and re-runs the full
    /// generate + gate pipeline; the original attempt plus every retry is
    /// recorded in `history.csv`. `0` (default) = one attempt per roll,
    /// gate failure discards the roll (legacy behavior).
    pub crossover_retries: usize,
    /// Who a surviving crossover child evicts (see [`CrossCullPolicy`]).
    /// Crossover-only: mutation immigrants follow `mutation_cull_policy`
    /// instead, regardless of this setting.
    pub crossover_cull_policy: CrossCullPolicy,
    /// Who a firing mutation roll evicts before inserting its immigrant (see
    /// [`MutationCullPolicy`]). Mutation-only: crossover children follow
    /// `crossover_cull_policy`.
    pub mutation_cull_policy: MutationCullPolicy,
    /// Elite guard: the top-k live nets (by smoothed fitness) are immune to
    /// ALL culls — crossover (any policy) and mutation alike. Minimum 1 (the
    /// champion is always guarded; the setter clamps lower values up).
    /// The effective guard is clamped to `live_count − 1` so a cullable
    /// victim always exists. Elites still age out of the set when their
    /// smoothed fitness drops out of the top-k — "elite" is a rank, not an
    /// identity.
    pub elite_count: usize,
    /// Anti-devolution A: the top-k nets act-and-measure without weight
    /// updates (see [`RaceConfigBuilder::set_elite_freeze`]). Default `true`.
    pub freeze_elites: bool,
    /// Per-net smoothed-fitness rolling window (K) — how many recent step
    /// fitnesses every ranking decision averages over (see
    /// [`RaceConfigBuilder::set_fitness_smoothing_window`]). Defaults to
    /// [`SMOOTHING_WINDOW`].
    pub smoothing_window: usize,
    /// Mutation/crossover immigrants skip catch-up and start at the current
    /// clock (see [`RaceConfigBuilder::set_mutation_catch_up`]).
    /// RL-effective only: tabular runs always catch up (see the setters).
    pub mutation_catch_up: bool,
    /// Whether a crossover child replays the training stream before insertion
    /// (see [`RaceConfigBuilder::set_crossover_catch_up`]). RL-effective only.
    pub crossover_catch_up: bool,
    /// Whether a population net rejoining the group step replays missed
    /// training (see [`RaceConfigBuilder::set_run_pop_catch_up`]).
    /// RL-effective only. (Reserved: with act-and-measure freeze, dethroned
    /// elites never fall behind the clock, so this has no consumer in the
    /// default flow today.)
    pub run_pop_catch_up: bool,
    /// Reset a dethroned elite's optimizer state (momentum/velocity/step
    /// counters — see [`RaceConfigBuilder::set_dethrone_reset_optimizer_state`]).
    pub dethrone_reset_optimizer_state: bool,
    /// Crossover operator pool — which recombination operators the two-parent
    /// path may use (`"one_point"` | `"uniform"`). Drawn uniformly per
    /// attempt. Empty ⇒ both operators (the `empty ⇒ all` convention shared
    /// with the activation/combine/standardize pools).
    pub crossover_ops_pool: Vec<String>,
    /// Max steps before stopping (None = no limit).
    pub max_steps: Option<usize>,
    /// Max target fitness: stop when the best smoothed fitness reaches this
    /// (None = no target stop). Compared under the fitness direction.
    pub max_target_fitness: Option<f32>,
    // NOTE: the shared batch stream (batch size, split ratio, eval rows) is
    // NOT a RaceConfig knob. It is engine infrastructure, rebuilt each run
    // from the trainer's optional stream_shape() request + the dataset's
    // seeded split. The trainer owns batch geometry; the engine owns data
    // integrity (split ratio) and can always read the effective shape via
    // stream_info().
    /// Hidden-dim sampling range — set with
    /// [`RaceConfig::set_topology_hidden_dim_range`]. When `None`, the
    /// default 4..=8 range is used.
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
    /// DEPRECATED knob kept for API compatibility — part-mutation has been
    /// dropped: crossover children are pure recombination and random
    /// immigrants are never perturbed (crossover exploits, mutation
    /// explores). Setting it has no effect.
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
    /// Flush `history.csv` after EVERY step instead of at checkpoint/stop
    /// boundaries. Default false (checkpoint cadence — a `kill -9` loses at
    /// most `checkpoint_every` steps of history rows, all other artifacts stay
    /// consistent). true = zero-loss history at the cost of a file write per
    /// step; only worth it on flaky infrastructure.
    pub history_flush_each: bool,
    /// At stop, save the elite's topology markdown (`elite-<hash>.md`).
    /// Default true — the champion's blueprint is the run's headline artifact.
    pub elite_save_topology: bool,
    /// At stop, save the elite's weights as safetensors
    /// (`elite-<hash>.safetensors`). Default true.
    pub elite_save_safetensors: bool,
    /// At stop, additionally save the WORST live net's artifacts
    /// (`worst-<hash>.md` + `.safetensors`). Default false — the
    /// anti-champion is a debugging/curiosity artifact.
    pub worst_save_topology: bool,
    /// At stop, additionally save the worst net's weights
    /// (`worst-<hash>.safetensors`). Default false.
    pub worst_save_safetensors: bool,
    /// Post-race pruner. When a stop criterion fires and this is `Some`, the
    /// engine culls every net except the elite and trains the champion solo
    /// for `steps` more steps (evolution off, stop criteria off — they are
    /// evolution-phase concerns). `None` (default) = stop ends the run.
    pub pop_pruner: Option<PopPruner>,
    /// Human label for the experiment, recorded verbatim in `engine.json`
    /// (`"run_name"`). Purely informative — it does NOT affect the results
    /// folder name (that stays `results/<run_id>` unless `RunSpec.run_dir`
    /// is set) — it exists so analysis scripts can group runs by experiment.
    pub run_name: Option<String>,
}

/// Type alias for the pluggable Iter-5 stop closure.
///
/// Plain alias (not a wrapper struct) means a run that stores a closure owns
/// it inside its single `RaceConfig`; `RaceConfig` therefore derives neither
/// `Clone` nor `Debug`.
pub type StopFn = Option<Box<dyn Fn(&RaceSnapshot) -> bool + Send + Sync + 'static>>;

/// Post-race pruner strategy. `Hard` = when a stop criterion fires, cull the
/// whole population except the elite and keep training the champion alone
/// for a fixed number of extra steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PopPrunerMethod {
    /// On stop: keep only the elite, train it solo for `pop_pruner_steps`.
    #[default]
    Hard,
}

/// Post-race pruner knob pair. `None` = the pruner is off (a stop reason
/// ends the run as before). `Some(method)` = after the stop fires, enter the
/// pruner phase: everything but the elite is culled (recorded as normal
/// culls) and the elite trains alone for `steps` more steps — outside the
/// evolution machinery (no crossover, no mutation, no stop checks).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PopPruner {
    pub method: PopPrunerMethod,
    pub steps: usize,
}

/// The run-level context stamped into every net's `meta` block (see
/// [`crate::state::NetMeta`]). Built once from the header + config at engine
/// construction; cheap to pass around.
#[derive(Clone, Debug, Default)]
pub struct RunMetaCtx {
    pub input_dim: usize,
    pub output_dim: usize,
    /// Shared-stream batch size; `None` in modes with no shared stream (RL),
    /// where the trainer owns the per-step volume.
    pub batch_size: Option<usize>,
    pub dropout_prob: f32,
    pub fitness_label: String,
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
            crossover_gate: CrossoverGate::default(),
            crossover_retries: 0,
            crossover_cull_policy: CrossCullPolicy::default(),
            mutation_cull_policy: MutationCullPolicy::default(),
            elite_count: 1,
            freeze_elites: true,
            smoothing_window: SMOOTHING_WINDOW,
            mutation_catch_up: false,
            crossover_catch_up: true,
            run_pop_catch_up: false,
            dethrone_reset_optimizer_state: true,
            crossover_ops_pool: Vec::new(),
            max_steps: None,
            max_target_fitness: None,
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
            history_flush_each: false,
            elite_save_topology: true,
            elite_save_safetensors: true,
            worst_save_topology: false,
            worst_save_safetensors: false,
            pop_pruner: None,
            run_name: None,
        }
    }

    /// Fluent builder for the non-CLI config surface. Starts from
    /// [`RaceConfig::defaults`] and overrides field-by-field; `build()` returns
    /// the finished config.
    pub fn builder() -> RaceConfigBuilder {
        RaceConfigBuilder {
            cfg: RaceConfig::defaults(),
            pending_pruner: None,
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
    /// Pruner params stored by `set_pruner_method`/`set_pruner_steps` while
    /// the pruner switch is still off — replayed by `build()` if
    /// `set_pruner_enabled(true)` comes later. Either call order works.
    pending_pruner: Option<PopPruner>,
}

impl RaceConfigBuilder {
    pub fn set_run_pop_size(mut self, n: usize) -> Self {
        self.cfg.pop_size = n;
        self
    }
    pub fn set_run_checkpoint_every(mut self, n: usize) -> Self {
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
    /// Gate strictness for crossover children: `CrossoverGate::Hard` = beat
    /// every checkpoint mean; `CrossoverGate::Soft` = beat the mean of the
    /// checkpoint means.
    pub fn set_crossover_gate(mut self, mode: CrossoverGate) -> Self {
        self.cfg.crossover_gate = mode;
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
    /// Applies ONLY to crossover children — mutation immigrants evict per
    /// [`Self::set_mutation_cull_policy`] instead.
    pub fn set_crossover_cull_policy(mut self, policy: CrossCullPolicy) -> Self {
        self.cfg.crossover_cull_policy = policy;
        self
    }
    /// Mutation replacement policy (see [`MutationCullPolicy`]): who a firing
    /// mutation roll evicts before inserting its fully-random immigrant.
    /// Defaults to `MutationCullPolicy::InverseFitness` — a fitness-inverse
    /// roulette, the engine's historical behavior, now explicit. Applies ONLY
    /// to the mutation/immigrant channel; crossover children use
    /// [`Self::set_crossover_cull_policy`].
    pub fn set_mutation_cull_policy(mut self, policy: MutationCullPolicy) -> Self {
        self.cfg.mutation_cull_policy = policy;
        self
    }
    /// Elite guard: top-k nets by smoothed fitness are immune to ALL culls
    /// (crossover and mutation). Minimum 1 — the champion is always guarded
    /// (a race with zero protected nets would let a cull evict the best net
    /// at any moment, contradicting the elite concept). Elites hold rank,
    /// not identity — a declining net falls out of the set naturally.
    pub fn set_elite_count(mut self, n: usize) -> Self {
        self.cfg.elite_count = n.max(1);
        self
    }
    /// ANTI-DEVOLUTION A — elite freeze: the top-`elite_count` nets (same
    /// smoothed-fitness ranking the elite guard uses) are exempt from the
    /// trainer call. They are still scored (their frozen skill re-measured
    /// every step), still rank, and still serve as crossover parents — but
    /// their weights never change, so a bad training step cannot erase the
    /// best skill ever found. "Elites hold rank, not identity" still applies
    /// one level deeper: freeze is recomputed each step from the ranking, so
    /// a child that trains past the frozen elite takes the crown next step
    /// and the old elite resumes training. Motivated by on-policy RL, where
    /// a net's own rollouts become its curriculum and one unlucky batch can
    /// collapse it below random (tabular, with its fixed data distribution,
    /// never trips the guard — the feature is dormant there).
    ///
    /// A frozen elite ACTS and MEASURES every step (fresh fitness through a
    /// `NoopOptimizer` — the weight update is discarded), so its standing
    /// stays honest and it never falls behind the clock. Default `true` —
    /// the champion's peak is always worth protecting; pass `false` to let
    /// elites keep training (e.g. when the decision-lag relay needs the
    /// trainer call to fork a shadow).
    pub fn set_elite_freeze(mut self, yes: bool) -> Self {
        self.cfg.freeze_elites = yes;
        self
    }
    /// Catch-up toggles (mutation family): whether a mutation immigrant
    /// replays the training stream before training at the current clock.
    /// Default `false` — no handicap: the immigrant keeps its fresh-init
    /// weights and trains from the current clock on (the old
    /// "fresh-start" behavior, now the default).
    ///
    /// **RL-effective only.** Catch-up exists so a net is *comparable* to the
    /// population: same step count, same weight-update history. In Tabular
    /// that is sacred — everyone shares one data stream, and a net that missed
    /// steps missed DATA, so tabular runs ALWAYS catch up regardless of this
    /// flag. In RL there is no shared stream: every net's matches are
    /// generated fresh from its own seeds, so starting at step k misses no
    /// data, only k updates. With catch-up off the immigrant's rolling
    /// buffers start empty ("no verdict yet") — it cannot be culled or
    /// crowned before its first step, and its first verdict is ~baseline
    /// random (it competes from birth rather than from a replayed adulthood).
    pub fn set_mutation_catch_up(mut self, yes: bool) -> Self {
        self.cfg.mutation_catch_up = yes;
        self
    }
    /// Catch-up toggle (crossover family): whether a crossover child replays
    /// the training stream (and passes the checkpoint gate on the replayed
    /// history) before insertion. Default `true` — exploitation: a child
    /// inherits the population's full update history and is checkpoint-gated
    /// on it, exactly the historical behavior. With `false`, the child skips
    /// catch-up AND the checkpoint gate (a net with no replayed history has
    /// nothing to compare against the historical bars — gating off is the
    /// only sound reading) and trains from the current clock like a mutation
    /// immigrant, competing from birth.
    ///
    /// **RL-effective only** — tabular runs always catch up (shared data
    /// stream; see [`Self::set_mutation_catch_up`]).
    pub fn set_crossover_catch_up(mut self, yes: bool) -> Self {
        self.cfg.crossover_catch_up = yes;
        self
    }
    /// Catch-up toggle (population): whether a population net rejoining the
    /// group step replays the training it missed. Default `false` — no
    /// handicap: every net works on its own from where it stands. Note: with
    /// act-and-measure elite freeze ([`Self::set_elite_freeze`]), a frozen
    /// elite keeps acting and measuring every step, so dethroned elites never
    /// fall behind the clock and nothing re-enters with a gap — this knob is
    /// reserved for future rejoin paths.
    ///
    /// **RL-effective only** — tabular runs always catch up. RESUME IS
    /// EXEMPT: resume always replays (weights are never persisted; replay
    /// parity is the resume contract regardless of this flag).
    pub fn set_run_pop_catch_up(mut self, yes: bool) -> Self {
        self.cfg.run_pop_catch_up = yes;
        self
    }
    /// Dethrone behavior: when a frozen elite loses its crown, reset its
    /// optimizer STATE (Adam momentum/velocity/step counters) while keeping
    /// the weights and all hyperparameters. Why: the state is stale — its
    /// notes describe the gradient landscape of the FROZEN era, and trusting
    /// them makes the first reclaim updates mis-scaled or mis-directed. A
    /// reset gives the comeback a clean, well-scaled warm-up: all the frozen
    /// skill (weights intact), none of the disorientation. Default `true`.
    /// Pass `false` to keep momentum across the freeze (nonstationary envs
    /// where the old trend is still meaningful). Replay-relevant — recorded
    /// in `engine.json` and validated on resume.
    pub fn set_dethrone_reset_optimizer_state(mut self, yes: bool) -> Self {
        self.cfg.dethrone_reset_optimizer_state = yes;
        self
    }
    /// Per-net smoothed-fitness rolling window (K), in steps. Every ranking
    /// decision — elite selection, cull roulette weights,
    /// the checkpoint ledger, stop criteria — averages the last K step
    /// fitnesses instead of reading the latest value, so a single lucky or
    /// unlucky step cannot flip a verdict. Default
    /// [`SMOOTHING_WINDOW`] = 10.
    ///
    /// **Replay-relevant**: like freeze/regression/immigrant knobs, this
    /// changes what a run MEANS — a resumed run must set the same value or
    /// construction fails parity. Larger K = stabler, slower-reacting
    /// rankings (a collapse takes K steps to fully register); smaller K =
    /// jumpier, faster to react.
    pub fn set_run_smoothing_window(mut self, k: usize) -> Self {
        self.cfg.smoothing_window = k.max(1);
        self
    }
    /// A user-supplied stop predicate, consulted every step alongside
    /// `max_steps` / `max_target_fitness`: returning `true` ends the race
    /// with stop reason `Custom`. Receives a [`RaceSnapshot`] of the current
    /// population. Example: stop when the population mean plateaus.
    pub fn set_stop_custom(
        mut self,
        f: impl Fn(&RaceSnapshot) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.cfg.custom_stop = Some(Box::new(f));
        self
    }
    /// Crossover operator pool (`"one_point"` | `"uniform"`). Drawn uniformly
    /// per crossover attempt. Empty (default) ⇒ both operators.
    ///
    /// Takes owned or borrowed strings: `&["uniform"]`, `vec!["uniform".into()]`,
    /// `&existing_vec` all work — one setter, no `_from_strs` twin.
    pub fn set_crossover_ops_pool<I, S>(mut self, ops: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.crossover_ops_pool = ops.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }
    /// Explicit step budget. Mutually exclusive with
    /// [`Self::set_stop_target_fitness`] — exactly one stop criterion.
    /// Accepts a bare value or an `Option` (`impl Into<Option<usize>>`).
    /// Setting this CLEARS any previously-set target fitness: the two are
    /// exclusive by construction, so the last writer wins (a const default
    /// is overridden by a later flag, not combined with it).
    pub fn set_stop_max_steps(mut self, n: impl Into<Option<usize>>) -> Self {
        self.cfg.max_steps = n.into();
        self.cfg.max_target_fitness = None;
        self
    }
    /// Max target fitness: stop when the best smoothed fitness reaches `v`.
    /// Mutually exclusive with [`Self::set_stop_max_steps`] — exactly one stop
    /// criterion. Accepts a bare value or an `Option`.
    /// Setting this CLEARS any previously-set max steps (see above).
    pub fn set_stop_target_fitness(mut self, v: impl Into<Option<f32>>) -> Self {
        self.cfg.max_target_fitness = v.into();
        self.cfg.max_steps = None;
        self
    }
    /// Whole-bundle stop criteria — DELETED: the two field setters
    /// ([`Self::set_stop_max_steps`] / [`Self::set_stop_target_fitness`]) are
    /// the whole surface, and each clears its sibling on write (last writer
    /// wins), which made the struct form redundant.
    /// Whole-bundle pruner — the struct form of [`Self::set_pruner_enabled`] /
    /// [`Self::set_pruner_method`] / [`Self::set_pruner_steps`]:
    ///
    /// ```ignore
    /// .set_pruner(PopPruner { method: PopPrunerMethod::Hard, steps: 50 })
    /// ```
    ///
    /// An `enabled == false` pruner is still stored (the stop simply ends the
    /// run) — pass `method`/`steps` alone instead if you want it off.
    pub fn set_pruner(mut self, pruner: PopPruner) -> Self {
        self.cfg.pop_pruner = Some(pruner);
        self
    }
    /// Post-race pruner switch: `true` = when a stop criterion fires, cull
    /// everything except the top-`elite_count` nets (default 1: the champion)
    /// and keep training them for `steps` more steps (evolution off, stop
    /// criteria off). `false` (default) = the stop reason ends the run.
    pub fn set_pruner_enabled(mut self, enabled: bool) -> Self {
        self.cfg.pop_pruner = enabled.then(|| {
            self.pending_pruner.take().unwrap_or(PopPruner {
                method: PopPrunerMethod::Hard,
                steps: 0,
            })
        });
        self
    }
    /// Pruner strategy (`Hard` = keep the elites, plain solo training).
    /// Takes effect only when [`Self::set_pruner_enabled`](`true`) is also
    /// called — either call order works.
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
    /// Extra solo-training step count after the stop fires (the `50` in
    /// `Hard` + 50). Takes effect only when [`Self::set_pruner_enabled`](`true`)
    /// is also called — either call order works.
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
    pub fn set_topology_hidden_dim_range(mut self, min: usize, max: usize) -> Self {
        self.cfg.hidden_dim_pool = Some(min..=max);
        self
    }
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
    /// Dropout probability applied by the blueprint when building this run's
    /// networks (part of `TopologyOptions` — the blueprint a saved topology
    /// replays from, so it must ride with the run config). TRAIN forwards
    /// only: the trainer flips `net.train()`/`net.eval()` around the loss.
    pub fn set_topology_dropout_prob(mut self, p: f32) -> Self {
        self.cfg.topology_options.dropout_prob = p;
        self
    }
    /// Seed for the topology's own RNG (graph wiring + weight init). The
    /// template seed comes from here; the ENGINE then re-seeds each child's
    /// topology from `(run_seed, clock, child_idx)` at birth, so two runs
    /// with the same run seed build identical children regardless of this
    /// value. Set it only to pin the template's structure deterministically.
    pub fn set_topology_seed(mut self, seed: usize) -> Self {
        self.cfg.topology_options.topology_seed = seed;
        self
    }
    // ── individual topology fields (direct, no need to build a full TopologyOptions)
    pub fn set_topology_min_hidden_num_nodes(mut self, n: usize) -> Self {
        self.cfg.topology_options.min_hidden_num_nodes = n;
        self
    }
    pub fn set_topology_max_hidden_num_nodes(mut self, n: usize) -> Self {
        self.cfg.topology_options.max_hidden_num_nodes = n;
        self
    }
    pub fn set_topology_min_inputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.min_hidden_inputs_per_node = n;
        self
    }
    pub fn set_topology_max_inputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.max_hidden_inputs_per_node = n;
        self
    }
    pub fn set_topology_min_outputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.min_hidden_outputs_per_node = n;
        self
    }
    pub fn set_topology_max_outputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology_options.max_hidden_outputs_per_node = n;
        self
    }
    pub fn set_topology_input_dim(mut self, n: usize) -> Self {
        self.cfg.topology_options.input_dim = Some(n);
        self
    }
    pub fn set_topology_output_dim(mut self, n: usize) -> Self {
        self.cfg.topology_options.output_dim = Some(n);
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
    /// Extra informative (non-ranking) metrics recorded in history.csv.
    /// Accepts anything that converts into a [`Metric`]: a plain label
    /// (`"f1"`), a `Metric::new(label)`, or a `Metric::custom(label, closure)`
    /// with your own scoring function. These NEVER affect selection/culling —
    /// the ranking fitness is set separately via `Fitness`.
    pub fn set_run_metrics<I, M>(mut self, metrics: I) -> Self
    where
        I: IntoIterator<Item = M>,
        M: Into<Metric>,
    {
        self.cfg.metrics = metrics.into_iter().map(|m| m.into()).collect();
        self
    }

    /// Human label for the experiment, written to `engine.json` as
    /// `"run_name"`. Informational only — the results folder name is
    /// unchanged (`results/<run_id>` unless `RunSpec.run_dir` is given).
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
    /// Set the training paradigm / problem space target.
    pub fn set_run_mode(mut self, mode: RunMode) -> Self {
        self.cfg.mode = mode;
        self
    }
    /// Set whether to write the lossless unified event log (`history.csv`).
    pub fn set_run_csv_export(mut self, enabled: bool) -> Self {
        self.cfg.csv_export = enabled;
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
    /// At stop, ALSO save the WORST live net's topology markdown
    /// (`worst-<hash>.md`), right after the elite's. Useful for diffing what
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
    /// Validate the stop-criteria surface: `max_steps` and
    /// `max_target_fitness` are **exclusive** — only one stop criterion at a
    /// time. The field setters already enforce this (each write clears its
    /// sibling), so this is a belt-and-suspenders check for configs built by
    /// hand (struct literal) rather than through the builder — both would
    /// fight for different things (a budget vs. a quality bar), and whichever
    /// fired first would silently mask the other. A config with both panics
    /// at `build()`.
    pub(crate) fn validate_single_stop(cfg: &RaceConfig) -> Result<(), String> {
        if cfg.max_steps.is_some() && cfg.max_target_fitness.is_some() {
            return Err(
                "only one stop criteria can be used at a time: max_steps and max_target_fitness are mutually exclusive".into(),
            );
        }
        Ok(())
    }

    pub fn build(mut self) -> RaceConfig {
        let cfg = self.cfg;
        // Params set by the pruner setters while the switch was still off:
        // surfaced loudly (a silent no-op would hide the misconfig).
        if let Some(p) = self.pending_pruner.take() {
            log::warn!(
                "set_pruner_method({:?})/set_pruner_steps({}) were called without set_pruner_enabled(true) — pruner is OFF",
                p.method,
                p.steps
            );
        }
        Self::validate_single_stop(&cfg).unwrap_or_else(|e| panic!("invalid RaceConfig: {e}"));
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
            "rmsnorm" | "rms_norm" => RmsNorm,
            "instancenorm" | "instance_norm" => InstanceNorm,
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

#[cfg(test)]
mod tests {
    use super::{LogLevel, RaceConfig};

    /// The examples' `--log-level` vocabulary must parse: handed to env_logger
    /// verbatim, `summ` is read as a module name and mutes the whole run, so
    /// this mapping is load-bearing.
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

    /// The stop field setters are mutually exclusive BY CONSTRUCTION: each
    /// write clears its sibling, so "const default then flag wins" composes
    /// without a build error. This is what let every example drop the old
    /// `set_stop`-wipe workaround (regression guard for the deleted
    /// whole-bundle setter).
    #[test]
    fn stop_setters_are_exclusive_last_writer_wins() {
        // const default: steps... then a flag asks for the fitness bar instead.
        let cfg = RaceConfig::builder()
            .set_stop_max_steps(15)
            .set_stop_target_fitness(3050.0)
            .build();
        assert_eq!(cfg.max_steps, None, "fitness flag cleared the step budget");
        assert_eq!(cfg.max_target_fitness, Some(3050.0));

        // and the reverse order: fitness default... then a flag asks for steps.
        let cfg = RaceConfig::builder()
            .set_stop_target_fitness(3050.0)
            .set_stop_max_steps(15)
            .build();
        assert_eq!(cfg.max_steps, Some(15));
        assert_eq!(
            cfg.max_target_fitness, None,
            "step flag cleared the fitness bar"
        );
    }
}
