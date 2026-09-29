//! The step-race knob surface: `RaceConfig`, `StopReason`, the pluggable
//! stop closure alias, the read-only `RaceSnapshot`, and the run-level
//! context stamped into every net's `meta` block (`RunMetaCtx`).

use crate::engine::fitness::Metric;

/// What the scheduler checks to decide whether to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxSteps,
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

/// Default number of fresh holdout games the post-race guardrail plays
/// ([`RaceConfigBuilder::set_guardrail_matches`]).
pub const DEFAULT_GUARDRAIL_MATCHES: usize = 16;

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
    /// A boxed table per step (from step 2 onward — step 1 has no prior
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
    /// `Summ` prints through `tracing::info!`, so it needs `info`; `Minimal` and
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
/// Mode-specific knobs live in [`RaceConfig::mode_specific`] (a
/// [`ModeConfig`] enum arm) — the same split logic the `RunSpec` variants
/// apply: the tabular/RL arm is carried in the TYPE. The engine core reads
/// only the shared fields; mode branches read their own arm.
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
    /// How many RECENT checkpoints the crossover gate reads (see
    /// [`RaceConfigBuilder::set_crossover_gate_window`]). `0` = all of them
    /// (unbounded, the legacy behavior).
    pub crossover_gate_window: usize,
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
    /// Mode-specific knobs (tabular vs RL). See [`ModeConfig`] — the same
    /// type-carrying split the `RunSpec` variants apply. The engine core
    /// reads only the shared fields above; mode branches match on this arm.
    pub mode_specific: ModeConfig,
    /// Crossover operator pool — which recombination operators the two-parent
    /// path may use (`"one_point"` | `"uniform"`). Drawn uniformly per
    /// attempt. Empty ⇒ both operators (the `empty ⇒ all` convention shared
    /// with the activation/combine/standardize pools).
    pub crossover_ops_pool: Vec<String>,
    /// Max steps before stopping (None = no limit).
    pub max_steps: Option<usize>,
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
    /// Per-roll chance a mutation roll FIRES and inserts a fully-random
    /// immigrant (crossover exploits, mutation explores; immigrants are never
    /// perturbed). Applies to both modes — `mutate_rolls` controls how many
    /// rolls are attempted per step, this controls how often each one fires.
    pub mutate_prob: f32,
    /// Pluggable stop criterion **in addition** to the built-ins (Iter 5
    /// contract). When `None`, only the built-ins apply.
    pub custom_stop: StopFn,
    /// Per-step log verbosity for the engine.
    pub log_level: LogLevel,
    /// Write `<run_dir>/telemetry.jsonl`: one JSON record per engine event, at
    /// full fidelity (debug included), next to whatever the console shows.
    /// Off by default — the console is the view, this is the record.
    pub trace_file: bool,
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
    /// User-supplied run topologies (the founding batch) (blueprint-only, fresh weights):
    /// seeded into the population BEFORE the random draws, in order, so a
    /// run can start from proven architectures (e.g. the elite of a prior
    /// run — see `population::run_topologies_from_run_dir`). The rest of
    /// the `pop_size` slots stay random draws; duplicates against the random
    /// batch (or within the run-topology list) are re-rolled like the standard
    /// duplicate gate. Recorded in `engine.json` as `run_topology_count`.
    pub run_topologies: Vec<crate::graph::topology::Topology>,
    /// Per-checkpoint elite weight snapshot: every `checkpoint_every` steps,
    /// export each top-`elite_count` net's weights as
    /// `checkpoint-elite-<hash>.safetensors` (overwritten each checkpoint —
    /// latest wins). Makes a hard kill (`kill -9`, power loss) lose at most
    /// one checkpoint interval of weights; the frontier states already land
    /// every checkpoint. Default true (k is small, the files are small).
    pub elite_checkpoint_weights: bool,
    /// Mutation probation: a net is cull-IMMUNE for its first
    /// `mutation_probation_steps` clocks (while
    /// `clock − entered_at_step < window`). The lower-half analog of the
    /// elite guard: a fresh immigrant with 1–2 bad verdicts gets k steps to
    /// draw its arch-lottery ticket before the inverse-fitness roulette can
    /// claim it. `0` (default) = today's behavior (only the empty-buffer
    /// step-0 immunity). A firing roll ALWAYS yields a slot: if every
    /// non-elite net is on probation, the protection is broken for that
    /// pick (last resort, logged). See `entered_at_step` — no new state,
    /// resume-safe by construction.
    pub mutation_probation_steps: usize,
    /// Anti-plateau challenge probability (`set_run_challenge_prob`):
    /// per-step, per-net chance the engine fires a challenged step and the
    /// trainer's `challenge_step` runs instead of `train_step`. 0.0
    /// (default) = the mechanism is OFF. A single knob by design — the
    /// trainer owns the action draw itself (no engine-side script/ledger).
    pub challenge_prob: f32,
    /// Fresh holdout games the post-race guardrail plays
    /// (`set_guardrail_matches`): the sample size of the honesty check, used
    /// when `CoreEngine::guardrail` is called with `None`. Diagnostic only —
    /// it never touches a step's dynamics. Default
    /// [`DEFAULT_GUARDRAIL_MATCHES`].
    pub guardrail_matches: usize,
}

/// How far through its step budget a run is, `0.0` → `1.0` (clamped), or
/// `None` when the run is unbounded (`max_steps = None`) or the budget is
/// degenerate (`0`).
///
/// THE one definition of "how far along are we": trainers reach it through
/// [`crate::trainer::StepEnv::progress`] and the engine's challenge decay is
/// built on it, so every schedule in the crate agrees on what "halfway"
/// means. Pure arithmetic on persisted values — nothing is stored, so replay,
/// catch-up and resume re-derive the identical number.
pub fn run_progress(step: usize, max_steps: Option<usize>) -> Option<f32> {
    let max = max_steps?;
    if max == 0 {
        return None;
    }
    Some((step as f32 / max as f32).min(1.0))
}

/// The effective (decayed) challenge probability at `step`: the knob scaled
/// linearly to 0 by the end of the step budget, flat when the run is
/// unbounded (same convention as [`run_progress`]).
///
/// Why the decay: challenges fight plateaus, and late-run steps are exactly
/// where the final polish happens — the run ends with a fully challenge-free
/// stretch so the last checkpoints measure the policy clean.
///
/// [`challenge_fires`] DECIDES with this value and
/// `CoreEngine::effective_challenge_prob` LOGS it — one function, so the number
/// shown can never drift from the number used.
pub fn effective_challenge_prob(prob: f32, step: usize, max_steps: Option<usize>) -> f32 {
    let progress = run_progress(step, max_steps).unwrap_or(0.0);
    prob * (1.0 - progress)
}

/// The challenge trigger: a pure function of `(run_seed, net_seed, step,
/// challenge_prob, max_steps)` — deterministic, so replay/catch-up re-fires
/// the exact same challenges the original run saw (no persisted flag needed).
/// `prob <= 0` never fires. It decides with [`effective_challenge_prob`] (the
/// decayed knob) and rolls ONE flip per (net, step): the trainer is handed a
/// bool and owns what "challenged" means (see `RlContext.challenged`).
pub(crate) fn challenge_fires(
    run_seed: u64,
    net_seed: u64,
    step: usize,
    prob: f32,
    max_steps: Option<usize>,
) -> bool {
    let eff = effective_challenge_prob(prob, step, max_steps);
    if eff <= 0.0 {
        return false;
    }
    // Domain-separated seed: independent of every other per-step roll.
    let mut rng =
        fastrand::Rng::with_seed(run_seed ^ net_seed.rotate_left(17) ^ ((step as u64) << 32));
    rng.f32() < eff
}

/// Mode-specific knob set — the arm is carried in the TYPE (same split
/// logic as the `RunSpec` variants). Constructed via
/// [`TabularConfig::builder()`] / [`RlConfig::builder()`]; the engine
/// derives the arm from the `RunSpec` variant at construction, so it can
/// never disagree with the spec.
#[derive(Clone, Debug, PartialEq)]
pub enum ModeConfig {
    /// Knobs that only exist in dataset-driven (tabular) runs.
    Tabular(TabularConfig),
    /// Knobs that only exist in environment-driven (RL) runs.
    Rl(RlConfig),
}

impl ModeConfig {
    /// The recorded `RunMode` this arm stands for (wire-format vocabulary).
    pub fn run_mode(&self) -> RunMode {
        match self {
            ModeConfig::Tabular(_) => RunMode::Tabular,
            ModeConfig::Rl(_) => RunMode::Rl,
        }
    }

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

    /// Mutation catch-up for this arm — `false` means a mutation immigrant
    /// keeps fresh weights and trains from the current clock.
    pub fn mutation_catch_up(&self) -> bool {
        match self {
            ModeConfig::Tabular(t) => t.mutation_catch_up,
            ModeConfig::Rl(r) => r.mutation_catch_up,
        }
    }

    /// Crossover catch-up for this arm — `false` means a crossover child
    /// skips replay AND the checkpoint gate.
    pub fn crossover_catch_up(&self) -> bool {
        match self {
            ModeConfig::Tabular(t) => t.crossover_catch_up,
            ModeConfig::Rl(r) => r.crossover_catch_up,
        }
    }

    /// Population rejoin catch-up for this arm (reserved; see
    /// [`TabularConfig::run_pop_catch_up`]).
    pub fn run_pop_catch_up(&self) -> bool {
        match self {
            ModeConfig::Tabular(t) => t.run_pop_catch_up,
            ModeConfig::Rl(r) => r.run_pop_catch_up,
        }
    }
}

/// Tabular-only knobs: the same catch-up family RL carries, so a tabular run
/// can opt out of replaying a newborn through the shared stream (see
/// [`RaceConfigBuilder::set_mutation_catch_up`]). Defaults keep the
/// historical tabular behavior: crossover children replay + gate (`true`),
/// mutation immigrants start at the current clock (`false`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TabularConfig {
    /// Mutation immigrants skip catch-up and start at the current clock.
    pub mutation_catch_up: bool,
    /// Whether a crossover child replays the training stream (and passes the
    /// checkpoint gate on the replayed history) before insertion.
    pub crossover_catch_up: bool,
    /// Whether a population net rejoining the group step replays missed
    /// training (reserved; no consumer in the default flow today).
    pub run_pop_catch_up: bool,
}

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
    /// Mutation immigrants skip catch-up (see [`TabularConfig::mutation_catch_up`]).
    pub fn set_mutation_catch_up(mut self, yes: bool) -> Self {
        self.cfg.mutation_catch_up = yes;
        self
    }
    /// Crossover children skip catch-up (see [`TabularConfig::crossover_catch_up`]).
    pub fn set_crossover_catch_up(mut self, yes: bool) -> Self {
        self.cfg.crossover_catch_up = yes;
        self
    }
    /// Population rejoin replay toggle (see [`TabularConfig::run_pop_catch_up`]).
    pub fn set_run_pop_catch_up(mut self, yes: bool) -> Self {
        self.cfg.run_pop_catch_up = yes;
        self
    }
    /// Finalize the tabular arm.
    pub fn build(self) -> TabularConfig {
        self.cfg
    }
}

/// RL knobs: the catch-up family. The tabular arm carries an identical copy
/// ([`TabularConfig`]) so both modes expose the same toggles; RL kept its own
/// arm because these started life as RL-only and the wire format already
/// carries them here.
#[derive(Clone, Debug, PartialEq)]
pub struct RlConfig {
    /// Mutation/crossover immigrants skip catch-up and start at the current
    /// clock (see [`RaceConfigBuilder::set_mutation_catch_up`]).
    pub mutation_catch_up: bool,
    /// Whether a crossover child replays the training stream before insertion
    /// (see [`RaceConfigBuilder::set_crossover_catch_up`]). With it OFF the
    /// checkpoint gate is OFF for those children too (nothing replayed to
    /// compare against the bars).
    pub crossover_catch_up: bool,
    /// Whether a population net rejoining the group step replays missed
    /// training (see [`RaceConfigBuilder::set_run_pop_catch_up`]).
    /// (Reserved: with act-and-measure freeze, dethroned elites never fall
    /// behind the clock, so this has no consumer in the default flow today.)
    pub run_pop_catch_up: bool,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            mutation_catch_up: false,
            crossover_catch_up: true,
            run_pop_catch_up: false,
        }
    }
}

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
    /// Mutation immigrants skip catch-up (see [`RlConfig::mutation_catch_up`]).
    pub fn set_mutation_catch_up(mut self, yes: bool) -> Self {
        self.cfg.mutation_catch_up = yes;
        self
    }
    /// Crossover children skip catch-up (see [`RlConfig::crossover_catch_up`]).
    pub fn set_crossover_catch_up(mut self, yes: bool) -> Self {
        self.cfg.crossover_catch_up = yes;
        self
    }
    /// Population rejoin replay toggle (see [`RlConfig::run_pop_catch_up`]).
    pub fn set_run_pop_catch_up(mut self, yes: bool) -> Self {
        self.cfg.run_pop_catch_up = yes;
        self
    }
    /// Finalize the RL arm.
    pub fn build(self) -> RlConfig {
        self.cfg
    }
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
            crossover_gate_window: 0,
            crossover_retries: 0,
            crossover_cull_policy: CrossCullPolicy::default(),
            mutation_cull_policy: MutationCullPolicy::default(),
            elite_count: 1,
            freeze_elites: true,
            smoothing_window: SMOOTHING_WINDOW,
            mode_specific: ModeConfig::Tabular(TabularConfig::default()),
            crossover_ops_pool: Vec::new(),
            max_steps: None,
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
            elite_checkpoint_weights: true,
            mutation_probation_steps: 0,
            challenge_prob: 0.0,
            guardrail_matches: DEFAULT_GUARDRAIL_MATCHES,
        }
    }

    /// Fluent builder for the non-CLI config surface. Starts from
    /// [`RaceConfig::defaults`] and overrides field-by-field; `build()` returns
    /// the finished config.
    ///
    /// The arm starts TABULAR (the `Default` arm). For an RL run prefer the
    /// mode front door [`RlRaceConfig::builder`] — it pre-stamps the RL arm so
    /// the RL-side catch-up copies are the ones a `ModeConfig` read sees.
    /// (Whatever arm the builder carries, the `RunSpec` constructor stamps
    /// its own — the front door just makes the call site honest.)
    pub fn builder() -> RaceConfigBuilder {
        RaceConfigBuilder {
            cfg: RaceConfig::defaults(),
            pending_pruner: None,
        }
    }
}

/// Typed front door for RL runs: `RlRaceConfig::builder()` is
/// [`RaceConfig::builder`] with the RL arm pre-stamped, so the RL-side
/// catch-up copies are the ones the engine reads. The setters are exactly
/// [`RaceConfigBuilder`]'s — same struct underneath, so `RunSpec::rl(..)`
/// consumes the built config unchanged.
pub type RlRaceConfig = RaceConfig;

/// Typed front door for tabular runs (reads well at the call site; the
/// catch-up setters write the tabular arm's own copy).
pub type TabularRaceConfig = RaceConfig;

/// `TabularRaceConfig::builder()` — the tabular front door: a free function
/// so the type alias can't shadow [`RaceConfig::builder`]. Starts from
/// [`RaceConfig::defaults`] with the tabular arm pre-stamped
/// (`ModeConfig::Tabular(TabularConfig::default())`). Functionally identical
/// to [`RaceConfig::builder`] today (the tabular arm IS the default arm);
/// it exists so call sites are mode-honest and a future tabular arm change
/// can't silently leak into RL construction paths.
#[doc(hidden)]
pub fn tabular_race_config_builder() -> RaceConfigBuilder {
    RaceConfigBuilder {
        cfg: RaceConfig::defaults(),
        pending_pruner: None,
    }
}

/// `RlRaceConfig::builder()` — the RL front door: a free function so the
/// type alias can't shadow [`RaceConfig::builder`]. Starts from
/// [`RaceConfig::defaults`] with the RL arm pre-stamped
/// (`ModeConfig::Rl(RlConfig::default())`), so RL-side reads see the RL copy.
// NOTE: a free function, not an inherent impl on the alias — `RlRaceConfig`
// IS `RaceConfig`, so an inherent `builder()` there would collide with the
// one above (two applicable items in scope).
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
    /// every checkpoint bar in the gate window; `CrossoverGate::Soft` = beat
    /// the mean of those bars.
    pub fn set_crossover_gate(mut self, mode: CrossoverGate) -> Self {
        self.cfg.crossover_gate = mode;
        self
    }
    /// Limit the crossover gate to the last `k` RECORDED checkpoints (both
    /// gates). Why: the bars are historical — a run-length ledger averages
    /// over every era the run passed through, so a long run's Soft bar drifts
    /// toward the lifetime average instead of the population's current
    /// standing, and a Hard child is asked to re-beat bars from eras it never
    /// lived in. Windowing keeps the bar local: Soft = mean of the last `k`
    /// checkpoint means; Hard = beat each of the last `k` checkpoint means.
    /// `0` (default) = no window — every recorded checkpoint gates, exactly
    /// as before. Replay-relevant: recorded in `engine.json` and validated on
    /// resume (a resumed run must gate against the same window or verdicts
    /// change).
    pub fn set_crossover_gate_window(mut self, k: usize) -> Self {
        self.cfg.crossover_gate_window = k;
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
    /// **Both modes** (the tabular arm carries its own copy — see
    /// [`TabularConfig`]). Catch-up exists so a net is *comparable* to the
    /// population: same step count, same weight-update history. The
    /// historical tabular answer was "always catch up" (one shared data
    /// stream, and a net should not start having missed DATA); `true` restores
    /// that. With catch-up off the immigrant's rolling buffers start empty
    /// ("no verdict yet") — it cannot be culled or crowned before its first
    /// step, and its first verdict reflects only its fresh weights.
    pub fn set_mutation_catch_up(mut self, yes: bool) -> Self {
        match &mut self.cfg.mode_specific {
            ModeConfig::Rl(rl) => rl.mutation_catch_up = yes,
            ModeConfig::Tabular(t) => t.mutation_catch_up = yes,
        }
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
    /// **Both modes** (the tabular arm carries its own copy; see
    /// [`Self::set_mutation_catch_up`]). Default `true`.
    pub fn set_crossover_catch_up(mut self, yes: bool) -> Self {
        match &mut self.cfg.mode_specific {
            ModeConfig::Rl(rl) => rl.crossover_catch_up = yes,
            ModeConfig::Tabular(t) => t.crossover_catch_up = yes,
        }
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
    /// **Both modes.** RESUME IS EXEMPT: resume always replays (weights are
    /// never persisted; replay parity is the resume contract regardless of
    /// this flag).
    pub fn set_run_pop_catch_up(mut self, yes: bool) -> Self {
        match &mut self.cfg.mode_specific {
            ModeConfig::Rl(rl) => rl.run_pop_catch_up = yes,
            ModeConfig::Tabular(t) => t.run_pop_catch_up = yes,
        }
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
    /// `max_steps`: returning `true` ends the race with stop reason
    /// `Custom`. Receives a [`RaceSnapshot`] of the current
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
    /// Explicit step budget — the only built-in stop criterion
    /// ([`Self::set_stop_custom`] composes an extra one on top of it).
    /// Accepts a bare value or an `Option` (`impl Into<Option<usize>>`);
    /// `None` (default) = run until externally stopped.
    pub fn set_stop_max_steps(mut self, n: impl Into<Option<usize>>) -> Self {
        self.cfg.max_steps = n.into();
        self
    }
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
    /// Extra informative (non-ranking) metrics recorded in history.csv and
    /// engine.json. Each is a `Metric::custom(label, closure)` — there are no
    /// built-in labels, so what gets measured is always explicit. These NEVER
    /// affect selection/culling — the ranking fitness is set separately via
    /// `Fitness`.
    pub fn set_run_metrics(mut self, metrics: Vec<Metric>) -> Self {
        self.cfg.metrics = metrics;
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
    /// Also write the run's event stream to `<run_dir>/telemetry.jsonl` (one
    /// JSON record per event, full fidelity). Off by default: the console
    /// stays the human view, this is the machine-readable record beside it.
    pub fn set_run_trace_file(mut self, enabled: bool) -> Self {
        self.cfg.trace_file = enabled;
        self
    }
    // NOTE: no `set_run_mode` — the RunMode is DERIVED from the RunSpec
    // variant at engine construction (the spec's mechanics are the single
    // source of truth; user-declared mode + cross-check was deleted when the
    // two disagreed no more).
    /// Set whether to write the lossless unified event log (`history.csv`).
    pub fn set_run_csv_export(mut self, enabled: bool) -> Self {
        self.cfg.csv_export = enabled;
        self
    }
    /// Seed the population with user-supplied run topologies (blueprint-only
    /// — fresh weights, same `make_optimizer` the race gives everyone). They
    /// fill slots FIRST (in order), the remaining
    /// `pop_size − run_topologies.len()` slots stay random draws; the duplicate
    /// gate re-rolls colliders. Pair with `population::run_topologies_from_run_dir`
    /// to re-run a prior run's elite architectures.
    pub fn set_run_topologies(mut self, topos: Vec<crate::graph::topology::Topology>) -> Self {
        self.cfg.run_topologies = topos;
        self
    }
    /// Per-checkpoint elite weight snapshot
    /// (`checkpoint-elite-<hash>.safetensors`, overwritten each checkpoint).
    /// Default true — pass false to only write weights at stop.
    pub fn set_elite_checkpoint_weights(mut self, enabled: bool) -> Self {
        self.cfg.elite_checkpoint_weights = enabled;
        self
    }
    /// Mutation probation steps: cull-immunity for a net's first `k`
    /// clocks (see [`RaceConfig::mutation_probation_steps`]). Default 0.
    pub fn set_mutation_probation_steps(mut self, k: usize) -> Self {
        self.cfg.mutation_probation_steps = k;
        self
    }
    /// Per-step, per-net chance an anti-plateau challenge fires. `0.0`
    /// (default) = challenges OFF. On a fired step the trainer is handed
    /// `ctx.challenged = true` and owns what that means: RL forces a drawn
    /// action on the trajectory, tabular jitters its batch — whatever the
    /// scheme decides (a trainer that ignores the flag keeps the knob at 0).
    /// The challenged step's fitness ranks like any other. Replay re-derives
    /// the same trigger from (run_seed, net_seed, step), so resume stays
    /// bit-exact. A future ramp (min_prob → max_prob approaching max_steps)
    /// is tracked in TODO.md.
    pub fn set_run_challenge_prob(mut self, p: f32) -> Self {
        assert!((0.0..=1.0).contains(&p), "challenge_prob must be in [0, 1]");
        self.cfg.challenge_prob = p;
        self
    }
    /// Fresh holdout games the post-race guardrail plays
    /// ([`DEFAULT_GUARDRAIL_MATCHES`] = 16): the tighter the verdict has to
    /// be, the more games — the mean's standard error falls as √N, and each
    /// game costs about one race step. Diagnostic only (never touches a
    /// step's dynamics); `CoreEngine::guardrail`'s explicit count overrides
    /// it for one call.
    pub fn set_guardrail_matches(mut self, n: usize) -> Self {
        self.cfg.guardrail_matches = n.max(1);
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
    /// Validate the stop-criteria surface. Kept as a no-op hook after the
    /// `max_target_fitness` deletion: the exclusivity check lost its subject
    /// (only `max_steps` + `custom_stop` remain, and those compose).
    pub(crate) fn validate_single_stop(_cfg: &RaceConfig) -> Result<(), String> {
        Ok(())
    }

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
    use super::{LogLevel, RaceConfig, challenge_fires, effective_challenge_prob, run_progress};
    use crate::utils::seed::derive_seed;

    /// One decay convention for the whole crate: linear to 0 at the budget,
    /// clamped, and flat when the run is unbounded. The SAME function feeds the
    /// trigger (`challenge_fires`) and the log
    /// (`CoreEngine::effective_challenge_prob`) — a schedule that shows one
    /// number and decides with another is worse than no schedule at all.
    #[test]
    fn run_progress_and_challenge_decay_share_one_convention() {
        assert_eq!(run_progress(0, Some(10)), Some(0.0));
        assert_eq!(run_progress(5, Some(10)), Some(0.5));
        assert_eq!(run_progress(10, Some(10)), Some(1.0));
        // Overshoot clamps: a run resumed past its budget must not go negative.
        assert_eq!(run_progress(99, Some(10)), Some(1.0));
        // No budget (or a degenerate one) = no schedule at all.
        assert_eq!(run_progress(3, None), None);
        assert_eq!(run_progress(3, Some(0)), None);

        // Halfway through the budget the knob is halved; unbounded is flat.
        assert!((effective_challenge_prob(0.2, 5, Some(10)) - 0.1).abs() < 1e-6);
        assert_eq!(effective_challenge_prob(0.2, 5, None), 0.2);
        // The run ends challenge-free by construction.
        assert_eq!(effective_challenge_prob(0.2, 10, Some(10)), 0.0);
        // The trigger's boundary agrees with the logged knob: past the budget the
        // knob is exactly 0 and the roll can never fire, whatever the seed.
        for step in 10..14 {
            for net_seed in 0..50u64 {
                assert!(
                    !challenge_fires(1, net_seed, step, 1.0, Some(10)),
                    "step {step} is past the budget — the knob is 0"
                );
            }
        }
        // prob = 1.0 at step 0 (eff exactly 1.0) → the roll always fires.
        for net_seed in 0..50u64 {
            assert!(challenge_fires(1, net_seed, 0, 1.0, Some(10)));
        }
        // While the knob is alive the trigger is PROBABILISTIC — one flip per
        // (net, step), not per population-step. At step 5 (eff 0.5) a 200-net
        // sample must contain both outcomes; a constant here would mean the roll
        // had stopped depending on the seed.
        let fired = (0..200u64)
            .filter(|net_seed| challenge_fires(1, *net_seed, 5, 1.0, Some(10)))
            .count();
        assert!(
            fired > 0 && fired < 200,
            "eff 0.5 over 200 independent net-steps should mix, got {fired}"
        );
    }

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
        // (the old max_target_fitness exclusivity test shrank with the
        // feature's deletion — only the step budget remains)
        let cfg = RaceConfig::builder().set_stop_max_steps(15).build();
        assert_eq!(cfg.max_steps, Some(15));
    }

    /// The trigger is per-NET, not per-step: at one step with `p = 0.5` a
    /// population of distinct seeds must split roughly in half. A single
    /// shared seed (the founder-seed bug) made it all-or-nothing, so `⚔`
    /// only ever appeared on the steps where that one seed happened to fire.
    #[test]
    fn challenge_trigger_is_per_net_not_per_step() {
        let fired = (0..64usize)
            .filter(|i| challenge_fires(42, derive_seed(42, *i), 3, 0.5, None))
            .count();
        assert!(
            (16..=48).contains(&fired),
            "trigger must track prob across nets, fired {fired}/64"
        );
    }
}
