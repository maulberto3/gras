//! The step-race knob surface: `RaceConfig`, `StopReason`, the pluggable stop closure alias,
//! read-only `RaceSnapshot`, and the run-level context stamped into every net's `meta` block

use crate::evolution::selection::SelectionMethod;
use crate::graph::topology::FeatureSplitPolicy;
use crate::utils::fitness::Metric;

/// What the scheduler checks to decide whether to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The configured step budget was reached.
    MaxSteps,
    /// A user-supplied `custom_stop` closure returned true (Iter 5 pluggable
    /// contract; consulted after all built-ins).
    CustomStop,
    /// Ctrl+C received: the step was abandoned at the loop boundary and the race shut
    /// down through the SAME artifact path. Always
    Interrupted,
}

// ── Consolidated defaults (one place for all engine constants) ────────────────────

/// Per-net smoothed-fitness rolling window (K) — the span every ranking decision averages over.
pub const SMOOTHING_WINDOW: usize = 5;

/// Default risk aversion (λ, FIT): how many window standard deviations a net is charged for its own
/// spread.
pub const RISK_AVERSION: f32 = 0.5;

/// Default recency ramp r for the smoothing window: `0.0` = the plain arithmetic mean every run
/// before this knob used.
pub const DEFAULT_WINDOW_RECENCY: f32 = 0.0;

/// How the smoothing window's readings are averaged into the ranking centre (`set_run_mean_type`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeanType {
    /// The weighted window mean — the plain average, and the default.
    #[default]
    Arithmetic,
    /// `g(x) = sign(x)·ln(1 + |x|)`: centre and spread are taken in that compressed space
    /// score is mapped back, so one spike cannot dominate the window the way it does through
    SignedLog,
    /// The WEIGHTED MEDIAN: no single reading can move the centre at all; the spread is the medians
    /// absolute deviation scaled by `1.4826`, so λ still reads in std-like units.
    Median,
}

impl MeanType {
    /// The spelling `engine.json` records and the log prints.
    pub fn as_str(&self) -> &'static str {
        match self {
            MeanType::Arithmetic => "arithmetic",
            MeanType::SignedLog => "signed_log",
            MeanType::Median => "median",
        }
    }

    /// Parse a recorded spelling. Unknown (and `""` from a header written before
    /// knob) yields `None`, which every caller reads as the arithmetic default.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "arithmetic" | "mean" | "simple" | "average" => Some(MeanType::Arithmetic),
            "signed_log" | "signedlog" | "signed-log" | "log1p" => Some(MeanType::SignedLog),
            "median" => Some(MeanType::Median),
            _ => None,
        }
    }
}

impl std::fmt::Display for MeanType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Default [`MeanType`] — the arithmetic mean, i.e. the ranking every run before
/// knob used.
pub const DEFAULT_MEAN_TYPE: MeanType = MeanType::Arithmetic;

/// LEGACY lottery strength Ω (`set_run_train_merit`, deprecated): retired as a second dial. `1.0`
/// is the identity, so a config built on the defaults pays exactly the band it sets — and every
/// header written since the retirement records `1.0`, which is why a resume never folds twice.
pub const DEFAULT_TRAIN_MERIT: f32 = 1.0;

/// Fold a LEGACY Ω into the band it used to buy: `lo = 1 − Ω(1 − p_min)`,
/// `hi = 1 − Ω(1 − p_max)`. `1.0` is the identity (the band unchanged), `0.0` lifts every net to
/// 1.0 = always train. The ONE application of Ω left in the engine: an old header is folded into
/// its band on load, and the deprecated setter folds at once, so no live decision reads Ω.
pub fn fold_omega(omega: f32, p_min: f32, p_max: f32) -> (f32, f32) {
    let omega = if omega.is_finite() {
        omega.clamp(0.0, 1.0)
    } else {
        1.0
    };
    (1.0 - omega * (1.0 - p_min), 1.0 - omega * (1.0 - p_max))
}

/// Default lottery ORDERING (`set_run_train_merit_inverted`): `true` = the LEAST fit net earns
/// MOST tickets, i.e.
pub const DEFAULT_TRAIN_MERIT_INVERTED: bool = true;

/// Default for the lottery's AGE leg (`set_run_train_merit_age`): `true` = ARMED, so the graded fraction
/// is the PRODUCT `Fm × Ym` (the min-max fitness place times the youth fraction, `1`
pub const DEFAULT_TRAIN_AGE: bool = false;

/// Default for the lottery's LEARN-ONLY switch (`set_run_train_merit_learn_only`): `false` = a lost
/// ticket idles the net (no train, no eval, nothing measured). `true` = every net still ACTS and
/// MEASURES every clock — fresh batch, honest reading — and only the weight UPDATE is gated, so
/// fitness stays expressive while learning stays ticketed. One exception, whichever way the switch
/// reads: a net that has NEVER learned (`U = 0`) has no learned policy to measure, so a denied clock
/// idles it outright rather than paying for a rollout over untouched weights — which is why the
/// born ticket (`set_run_train_merit_born_ticket`) exists: it keeps the coin from denying that net
/// in the first place.
pub const DEFAULT_TRAIN_MERIT_LEARN_ONLY: bool = false;

/// Default for POLICY-SCOPED FITNESS (`set_run_policy_scoped_fitness`): `false` = the window averages
/// the last K READINGS, whatever weights produced them (the pre-flag contract). `true` = it averages
/// only the readings taken on the weights the net holds NOW, so mean/spread/n describe ONE policy
/// and no reading from another weight set enters the statistics.
pub const DEFAULT_POLICY_SCOPED_FITNESS: bool = false;

/// Default for the ANTI-CHAMPION dump (`set_worst_save_topology`): `false` = no `worst-*.md` files.
/// `true` writes the top-`champion_count` nets by TICKET CHANCE at stop — the least fit and, with
/// the age leg armed, the oldest — as topology + mermaid + lineage markdown, so the nets the coin
/// keeps feeding can be read structurally instead of guessed at from a hash.
pub const DEFAULT_WORST_SAVE_TOPOLOGY: bool = false;

/// Steps between population checkpoints — the cadence the stream's era rotation and the CSV flush
/// key off.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 20;

/// Default population size: `10`, deliberately conservative.
pub const DEFAULT_POP_SIZE: usize = 10;

/// Default step budget, read at the same conservative scale as the population: `10` clocks.
pub const DEFAULT_MAX_STEPS: usize = 10;

/// Default rolls offered per step on each channel: the cartpole reference's two draws
/// clock, on the crossover channel and on the mutation channel alike.
pub const DEFAULT_CROSSOVER_ROLLS: usize = 2;
pub const DEFAULT_MUTATE_ROLLS: usize = 2;

/// Default share of an evolution attempt that fires: the cartpole reference's values.
pub const DEFAULT_CROSSOVER_PROB: f32 = 0.5;
pub const DEFAULT_MUTATE_PROB: f32 = 0.2;

/// Default champion guard: the reference's `pop / 10`, read at the conservative default population
/// 10 — one seat.
pub const DEFAULT_CHAMPION_COUNT: usize = 1;

/// Default clocks a net must have lived before it may take an elite seat: `5`, the guarded value.
pub const DEFAULT_SEAT_JOIN_PROBATION: usize = 5;

/// Default cap on a post-update RECOVERY streak: after a real update whose reading comes in worse
/// than the net's standing, the engine takes up to this many extra real updates in the SAME clock,
/// stopping the moment the reading is back where it was. The VALUE IS THE SWITCH: `0` (the
/// default) means off, any value above it arms the mechanic for that many retries.
pub const DEFAULT_RECOVERY_RETRIES: usize = 0;

/// Default retry DRAW for a recovery streak (`set_recovery_same_seed`): `true` = a retry re-runs
/// the SAME train draw the update fell on (same rows, same RNG stream) — a true retry, which asks
/// "was the fall the update's fault?" rather than "would another batch have been kinder?".
pub const DEFAULT_RECOVERY_SAME_SEED: bool = true;

/// Default recovery tolerance: the drop a new reading may take against the net's standing and
/// still count as noise. `0.0` = any drop earns a rescue; `0.2` = a fifth is tolerated.
pub const DEFAULT_RECOVERY_TOLERANCE: f32 = 0.0;

/// Default U leg of the lottery: OFF, so the ticket stays the age-legged product `Fm × Ym` until a
/// run arms it (`set_run_train_merit_updates(true)`).
pub const DEFAULT_TRAIN_MERIT_UPDATES: bool = true;

/// Default BORN TICKET (`set_run_train_merit_born_ticket`): `false` = the pre-flag contract, where
/// the born shield's `1.0` fraction is still folded through Ω to `hi`, so the coin can idle a net
/// that has never trained. `true` = a net with `U = 0` is paid `p = 1.0` and takes its FIRST update
/// on its first live clock, so its shield is served the clock it is born.
pub const DEFAULT_TRAIN_MERIT_BORN_TICKET: bool = false;

/// Default INIT PROBATION (`set_run_init_probation_steps`): `0` = nothing is withheld beyond the
/// born shields, which is exactly what every run before the knob did. Any other value is the
/// number of opening clocks with no culls and no births — evolution starts at that clock.
pub const DEFAULT_INIT_PROBATION_STEPS: usize = 0;

/// Default clocks a crown seat may sit at peak: `5`, the guarded value (a seated net
/// rotates out on its term). The cartpole reference disarms the term (`0`).
pub const DEFAULT_CROWN_LIFETIME: usize = 5;

/// Default hidden-width sampling range: the cartpole reference's `16..=64`,
/// stride [`RaceConfig::defaults`] pins.
pub const DEFAULT_HIDDEN_POOL: std::ops::RangeInclusive<usize> = 16..=64;

/// Default merit floor `p_min`: the smallest share of clocks the WORST net in the lottery's field
/// still gets.
pub const DEFAULT_TRAIN_MERIT_P_MIN: f32 = 0.05;
/// Default lottery CEILING `p_max`: the largest share of clocks the best-graded net may take.
pub const DEFAULT_TRAIN_MERIT_P_MAX: f32 = 0.4;

/// Default share of the dataset held out for EVAL (`eval_share`): how much of the rows the race
/// RANKS on and never trains
pub const DEFAULT_EVAL_SHARE: f32 = 0.2;

/// The narrowest and widest `eval_share` a run accepts.
pub const EVAL_SHARE_MIN: f32 = 0.01;
pub const EVAL_SHARE_MAX: f32 = 0.9;

/// Default chance that ONE inherited drift slot moves on a crossover child (`set_topology_drift`):
/// a mask bit flips, or a hidden node's activation / combine / standardize is re-drawn
pub const DEFAULT_TOPOLOGY_DRIFT: f32 = 0.1;

/// Default novelty floor `d_min` for a mutation draw (`set_mutation_novelty_min`): the clone
/// band — unrelated nets sit near Δ 0.67, while a net's NEAREST neighbour sits near 0.17.
pub const DEFAULT_MUTATION_NOVELTY_MIN: f32 = 0.2;

/// Default candidates ONE mutation may spend before the most novel is taken anyway.
pub const DEFAULT_MUTATION_NOVELTY_DRAWS: usize = 5;

// `DEFAULT_LR` moved to `crate::utils::training`: a training scheme reads it (the reference trainer
// and the example both do), so it belongs below both layers rather than in the engine's config

/// Per-step log verbosity for the race engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogLevel {
    /// Silent per step: only the run start line and final stop reason print —
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

    /// The `tracing` verbosity the console layer is built with: `Summ` prints the engine's `info!`
    /// lines; `Minimal`/`None` mute them (`warn`) — the framed vitals table rides its own target,
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
    /// **Worst** — evict the current worst net by smoothed fitness; strictly
    /// merit-based. Risk: the population can ossify around one lineage.
    Worst,
    /// **Random** — evict a uniformly random live net; keeps slots turning over
    /// prevents a long-lived leader from starving diversity. Risk: losing a strong net.
    Random,
    /// **Unfed** (default) — evict the net the coin pays MOST: the highest ticket chance
    /// `p = p_min + (p_max − p_min) × f` among the graded nets, deterministic with the hash as
    #[default]
    Unfed,
}

impl std::fmt::Display for CrossCullPolicy {
    /// The STABLE spelling recorded in `engine.json` and the run manifest (snake_case,
    /// not the `Debug` casing): a header written today must read the same tomorrow.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Worst => "worst",
            Self::Random => "random",
            Self::Unfed => "unfed",
        })
    }
}

/// Who gets evicted when a MUTATION roll inserts a fresh random immigrant.
/// **Mutation-only** — the crossover channel has its own policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MutationCullPolicy {
    /// **Worst** — evict the worst net by smoothed fitness, deterministically (mirrors
    /// [`CrossCullPolicy::Worst`]).
    Worst,
    /// **InverseFitness** — a fitness-inverse roulette over non-elite nets (an opt-in:
    /// the worst net weighs most, the best is never drawn; chance is kept).
    InverseFitness,
    /// **Random** — evict a uniformly random live net, at the cost of occasionally
    /// losing a good one.
    Random,
    /// **Unfed** (default) — the SAME selector as [`CrossCullPolicy::Unfed`]: evict the net the
    /// train-eval lottery pays the most (the highest `p` among the graded nets).
    #[default]
    Unfed,
}

impl std::fmt::Display for MutationCullPolicy {
    /// The STABLE spelling recorded in `engine.json` and the run manifest.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InverseFitness => "inverse_fitness",
            Self::Worst => "worst",
            Self::Random => "random",
            Self::Unfed => "unfed",
        })
    }
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
    /// How many parent topologies this operator combines. All current operators
    /// binary, so both return 2 (the engine derives parent count
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

/// The training paradigm / problem space the run targets. Only `Tabular` and `Rl`
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
    /// Steps between population checkpoints (the stream's era rotation and the CSV flush cadence
    /// key off
    pub checkpoint_every: usize,
    /// Independent crossover rolls per step; each fires with `crossover_prob`.
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
    /// CHAMPION guard: the top-k nets by smoothed fitness are immune to ALL culls (minimum
    /// 1). A champion is a rank — the saved, inheritable set — not an identity, and the
    /// frozen SEATS below are a separate switch that this k sizes but does not mean.
    pub champion_count: usize,
    /// Anti-devolution A: the top-k nets act-and-measure without weight
    /// updates (see [`RaceConfigBuilder::set_seat_freeze`]). Default `true`.
    pub freeze_seats: bool,
    /// LEGACY lottery strength Ω (`set_run_train_merit`, deprecated): the band below IS the payout
    /// now, so this number is only read when an old header is folded ([`fold_omega`]). A config
    /// built today carries `1.0` (the identity) and no live decision reads it.
    pub train_merit: f32,
    /// FLOOR `p_min` of the ticket band (`set_run_train_merit_p_min`) — literally the chance the
    /// WORST-graded net gets each clock (`1.0` = every clock, `0.0` = never).
    pub train_merit_p_min: f32,
    /// CEILING `p_max` of the ticket band (`set_run_train_merit_p_max`) — literally the chance the
    /// best-graded net gets each clock; every net is paid between the floor and this.
    pub train_merit_p_max: f32,
    /// INVERT the lottery's merit ordering (`set_run_train_merit_inverted`): the LEAST fit net
    /// earns the MOST tickets and the best earns the fewest.
    pub train_merit_inverted: bool,
    /// LEARN-ONLY lottery (`set_run_train_merit_learn_only`): `true` = a lost ticket costs the
    /// weight UPDATE, not the STEP — the net runs its trainer with a no-op optimizer, so it acts,
    /// is measured, and its fitness moves every clock. `false` = the clock idles. Default `false`.
    /// A net that has never learned (`U = 0`) idles under BOTH settings: there is no learned policy
    /// for the rollout to measure.
    pub train_merit_learn_only: bool,
    /// BORN TICKET (`set_run_train_merit_born_ticket`): `true` pays a net that has never trained
    /// (`U = 0`) `p = 1.0`, before the Ω band, so its FIRST update lands on its first live clock
    /// whatever it grades. `false` = the pre-flag contract, where the shield's fraction `1.0` folds
    /// to `hi` and the coin may idle a net that cannot spend its shield without training.
    pub train_merit_born_ticket: bool,
    /// Share of the dataset held out for EVAL (`eval_share`): the rows a net is SCORED on and never
    /// trained
    pub eval_share: f32,
    /// Per-net smoothed-fitness rolling window (K) — how many recent step fitnesses
    /// ranking decision averages over. Defaults to [`SMOOTHING_WINDOW`].
    pub smoothing_window: usize,
    /// Risk aversion λ (FIT): window standard deviations subtracted from (Maximize) or added
    /// (Minimize) a net's smoothed mean before ranking.
    pub risk_aversion: f32,
    /// Recency ramp r for the smoothing window (`set_run_window_recency`): how much
    pub window_recency: f32,
    /// How the window's readings are averaged into the ranking centre (`set_run_mean_type`):
    /// [`MeanType::Arithmetic`] (default, the pre-knob ranking), [`MeanType::SignedLog`]
    pub mean_type: MeanType,
    /// POLICY-SCOPED FITNESS (`set_run_policy_scoped_fitness`): `true` makes the mean/spread/n
    /// describe the CURRENT weights — the readings taken since the net's last real update — so the
    /// statistics are of ONE policy and never a mix of policies. Default `false` (the pre-flag
    /// window, which averages the last K readings across every weight set).
    pub policy_scoped_fitness: bool,
    /// ANTI-CHAMPION dump (`set_worst_save_topology`): `true` writes `worst-<hash>-<fitness>.md` +
    /// `worst.md` at stop for the nets with the HIGHEST ticket chance (least fit, oldest).
    pub worst_save_topology: bool,
    /// Age leg of the lottery (`set_run_train_merit_age`): `true` ARMS the product `Fm × Ym`, `Ym` = `1`
    /// at the youngest graded net.
    pub train_age: bool,
    /// U leg of the lottery (`set_run_train_merit_updates`): `true` ARMS `Um`, the age leg's twin on
    /// UPDATES (`1.0` at the field's fewest, `0.0` at its most), so the ticket is `Fm × Ym × Um` and
    /// a net that spent its updates is paid for out of its own future tickets. `false` = the
    /// pre-knob product, bit for bit.
    pub train_merit_updates: bool,
    /// Mode-specific knobs (tabular vs RL) — the arm is carried in the type. The engine
    /// core reads only the shared fields; mode branches match on this arm.
    pub mode_specific: ModeConfig,
    /// Crossover operator pool (`"one_point"` | `"uniform"`), drawn uniformly per attempt.
    /// Empty ⇒ both.
    pub crossover_ops_pool: Vec<String>,
    /// How a crossover attempt draws its parents: [`SelectionMethod::Roulette`] (default —
    /// engine's historical behavior) or a `Tournament { k }`, which concentrates recombination
    pub crossover_selection: SelectionMethod,
    /// Fitter parent as the crossover BASE (`set_crossover_fitter_base`).
    pub crossover_fitter_base: bool,
    /// Max steps before stopping (None = no limit).
    pub max_steps: Option<usize>,
    /// Hidden-dim sampling range (`None` ⇒ default 4..=8). NOTE: batch geometry lives
    /// on the trainer's `stream_shape()`, not here.
    pub hidden_dim_pool: Option<std::ops::RangeInclusive<usize>>,
    /// Stride within the hidden-dim pool when sampling node dimensions.
    pub hidden_dim_stride: usize,
    /// Combine/activation/standardize pools actually used by the run; empty uses
    /// known ops (`empty ⇒ all` convention).
    pub combine_op_pool: Vec<String>,
    /// The feature-split POOL newborns draw from: `none`, `random`, `exclusive` (the `split_*`
    /// spellings `engine.json` writes are accepted too).
    pub feature_split_pool: Vec<String>,
    /// Per-SLOT drift on a crossover child's INHERITED genes (`set_topology_drift`): the chance,
    /// for each slot below, that it moves
    pub drift_prob: f32,
    /// Activation pool (empty ⇒
    pub activation_pool: Vec<String>,
    /// Standardize-op pool (empty ⇒
    pub standardize_op_pool: Vec<String>,
    /// Topology template (input/output dims, hidden node ranges, dropout, etc.)
    /// shared by every individual in the run.
    pub topology_options: crate::graph::topology::TopologyOptions,
    /// Chance a crossover roll fires.
    pub crossover_prob: f32,
    /// Per-roll chance a mutation roll FIRES and inserts a random immigrant.
    /// `mutate_rolls` controls rolls per step; this controls how often each fires.
    pub mutate_prob: f32,
    /// Novelty floor `d_min` on a mutation draw (`set_mutation_novelty_min`): the minimum Δ to
    /// the NEAREST blueprint this run already made. `0.0` = off.
    pub mutation_novelty_min: f32,
    /// Candidates one mutation may spend clearing that floor (`set_mutation_novelty_draws`).
    pub mutation_novelty_draws: usize,
    /// Pluggable stop criterion **in addition** to the built-ins (Iter 5
    /// contract). When `None`, only the built-ins apply.
    pub custom_stop: StopFn,
    /// Per-step log verbosity for the engine.
    pub log_level: LogLevel,
    /// Write `<run_dir>/telemetry.jsonl`: one JSON record per event at full fidelity.
    /// Off by default — the console is the view, this is the record.
    pub trace_file: bool,
    /// Write each live net's OPTIMIZER state beside its weights (`nets/<hash>.optim`)
    /// checkpoint, so a resume restores Adam's moments instead of restarting them.
    pub save_optim: bool,
    /// Informative (non-ranking) metrics; their labels determine the extra per-net
    /// metric columns a reader may expect.
    pub metrics: Vec<Metric>,
    /// The training paradigm / problem space target.
    pub mode: RunMode,
    /// Whether to write `history.csv` (one row per live net per clock it was TRAINED AND SCORED
    /// — an idle lottery clock has nothing to record) and `attempts.csv` (per evolution attempt) —
    pub csv_export: bool,
    /// At stop, save each champion's topology markdown (`champion-<hash>-<fitness>.md`).
    /// Default true — the ranking's blueprint is the run's headline artifact.
    pub champion_save_topology: bool,
    /// Human label for the experiment, recorded verbatim in `engine.json`
    /// `"run_name"`. Informational — does not affect the results folder name.
    pub run_name: Option<String>,
    /// AUTOMATIC inheritance: load the newest compatible PREVIOUS experiment's CHAMPIONS into
    /// first founder slots (topology-only).
    pub champion_previous_load: bool,
    /// Where `champion_previous_load` looks: a run dir or a dir of run dirs (newest run id
    /// wins). `None` ⇒ the default `results/` folder.
    pub champion_previous_dir: Option<std::path::PathBuf>,
    /// Mutation probation: a random immigrant is cull-IMMUNE until it has spent
    /// `mutation_probation_steps` UPDATES (`U`, real optimizer steps) — not clocks. `0` = off.
    pub mutation_probation_steps: usize,
    /// Crossover probation: a crossover-born net is cull-IMMUNE until it has spent
    /// `crossover_probation_steps` UPDATES (`U`) — not clocks. `0` = off.
    pub crossover_probation_steps: usize,
    /// INIT PROBATION (`set_run_init_probation_steps`): clocks the run spends WITHOUT evolution —
    /// no culls, no births — however cull-eligible the population already is. Like the other
    /// probation steps it is a warm-up, but counted in CLOCKS: it is the RUN's opening, not one
    /// net's shield. `0` (the default) withholds nothing beyond the born shields.
    pub init_probation_steps: usize,
    /// Post-update RECOVERY streak cap: after a real update whose reading comes in WORSE than the
    /// net's standing going in, up to `recovery_retries` extra real updates are taken in the SAME
    /// clock, stopping the moment a reading is back at or above where it was. Every retry is a real
    /// weight update and counts in `U`, so a streak lets `U` outrun the clocks a net lived.
    /// Default `0` = off; any value above it arms the streak for that many retries.
    pub recovery_retries: usize,
    /// Recovery tolerance: how far a reading may fall below the net's standing WITHOUT earning a
    /// rescue, as a FRACTION of that standing (`0.2` = a fifth, so 268 may read down to 214).
    /// `0.0` = any drop triggers the streak; a wide band leaves the ordinary drop rate alone.
    pub recovery_tolerance: f32,
    /// Recovery retry DRAW (`set_recovery_same_seed`): `true` (default) = every retry repeats the
    /// clock's OWN train draw — the same rows and the same RNG stream — so the streak re-tests the
    /// one update that fell instead of shopping for a kinder batch. `false` = each retry walks one
    /// draw further (the next tabular batch / the next block of RL match seeds), so the streak
    /// fights on fresh evidence. The EVAL side is keyed to the clock either way, so the reading
    /// stays comparable to the standing it has to climb back to.
    pub recovery_same_seed: bool,
    /// Seat join probation: a net cannot take a SEAT until it has aged
    /// `seat_join_probation_steps` clocks since arrival. Default `5`.
    pub seat_join_probation_steps: usize,
    /// Seat lifetime: a net on the frozen crown is rotated out after
    /// `seat_lifetime_steps` clocks. Default `5`; zero = permanent until its rank moves.
    pub seat_lifetime_steps: usize,
}

// `run_progress` and `DEFAULT_LR` live in `crate::utils::training` — the run
// loop and any training scheme both need them, so they sit below both layers.

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
    /// Best smoothed fitness — the RISK-ADJUSTED window score (FIT: `mean ∓ λσ`),
    /// what the race actually ranks on. At λ = 0 this is the plain window mean again.
    pub best_smoothed_fitness: f32,
    /// Worst smoothed fitness — the same risk-adjusted score, at the other end.
    pub worst_smoothed_fitness: f32,
    /// Mean of the live risk-adjusted scores (a plain arithmetic mean OF the scores,
    /// second smoothing pass).
    pub mean_smoothed_fitness: f32,
    /// Cumulative culls.
    pub culls: usize,
    /// Wall-clock seconds elapsed.
    pub elapsed_seconds: u64,
}

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
    /// channel, tournament-2 crossover parents, a 5-reading window at λ 0.5, the inverted lottery
    pub fn defaults() -> Self {
        RaceConfig {
            pop_size: DEFAULT_POP_SIZE,
            checkpoint_every: DEFAULT_CHECKPOINT_EVERY,
            crossover_rolls: DEFAULT_CROSSOVER_ROLLS,
            mutate_rolls: DEFAULT_MUTATE_ROLLS,
            crossover_cull_policy: CrossCullPolicy::default(),
            mutation_cull_policy: MutationCullPolicy::default(),
            champion_count: DEFAULT_CHAMPION_COUNT,
            freeze_seats: true,
            train_merit: DEFAULT_TRAIN_MERIT,
            train_merit_p_min: DEFAULT_TRAIN_MERIT_P_MIN,
            train_merit_p_max: DEFAULT_TRAIN_MERIT_P_MAX,
            train_merit_inverted: DEFAULT_TRAIN_MERIT_INVERTED,
            train_merit_learn_only: DEFAULT_TRAIN_MERIT_LEARN_ONLY,
            train_merit_born_ticket: DEFAULT_TRAIN_MERIT_BORN_TICKET,
            eval_share: DEFAULT_EVAL_SHARE,
            seat_join_probation_steps: DEFAULT_SEAT_JOIN_PROBATION,
            seat_lifetime_steps: DEFAULT_CROWN_LIFETIME,
            smoothing_window: SMOOTHING_WINDOW,
            risk_aversion: RISK_AVERSION,
            window_recency: DEFAULT_WINDOW_RECENCY,
            mean_type: DEFAULT_MEAN_TYPE,
            policy_scoped_fitness: DEFAULT_POLICY_SCOPED_FITNESS,
            worst_save_topology: DEFAULT_WORST_SAVE_TOPOLOGY,
            train_age: DEFAULT_TRAIN_AGE,
            train_merit_updates: DEFAULT_TRAIN_MERIT_UPDATES,
            mode_specific: ModeConfig::Tabular(TabularConfig::default()),
            crossover_ops_pool: Vec::new(),
            crossover_selection: SelectionMethod::Tournament { tournament_size: 2 },
            crossover_fitter_base: true,
            max_steps: Some(DEFAULT_MAX_STEPS),
            hidden_dim_pool: Some(DEFAULT_HIDDEN_POOL),
            hidden_dim_stride: 16,
            combine_op_pool: Vec::new(),
            feature_split_pool: Vec::new(),
            drift_prob: DEFAULT_TOPOLOGY_DRIFT,
            activation_pool: Vec::new(),
            standardize_op_pool: Vec::new(),
            topology_options: crate::graph::topology::TopologyOptions::default(),
            crossover_prob: DEFAULT_CROSSOVER_PROB,
            mutate_prob: DEFAULT_MUTATE_PROB,
            mutation_novelty_min: DEFAULT_MUTATION_NOVELTY_MIN,
            mutation_novelty_draws: DEFAULT_MUTATION_NOVELTY_DRAWS,
            custom_stop: None,
            metrics: Vec::new(),
            log_level: LogLevel::default(),
            trace_file: false,
            save_optim: true,
            mode: RunMode::Tabular,
            csv_export: true,
            champion_save_topology: true,
            run_name: None,
            champion_previous_load: true,
            champion_previous_dir: None,
            mutation_probation_steps: 5,
            crossover_probation_steps: 5,
            init_probation_steps: DEFAULT_INIT_PROBATION_STEPS,
            recovery_retries: DEFAULT_RECOVERY_RETRIES,
            recovery_tolerance: DEFAULT_RECOVERY_TOLERANCE,
            recovery_same_seed: DEFAULT_RECOVERY_SAME_SEED,
        }
    }

    /// The pool resolved to drawable policies (`empty ⇒ all three`, in the fixed order none →
    /// random → exclusive so the draw is reproducible).
    pub fn resolved_feature_split_pool(&self) -> Result<Vec<Option<FeatureSplitPolicy>>, String> {
        super::pool::resolve::<super::pool::FeatureSplitPool>(self)
    }
    /// The pool as wire names, in draw order — one spelling for the log, the `engine.json` record
    /// and the resume diff.
    pub fn feature_split_pool_names(&self) -> Vec<String> {
        super::pool::names::<super::pool::FeatureSplitPool>(self)
    }

    /// The inverse of [`crate::state::ConfigSnapshot::from_config`]: rebuild the run-level knobs
    /// header recorded, so a resume needs only its directory.
    pub fn from_snapshot(snap: &crate::state::ConfigSnapshot) -> Self {
        let mut cfg = RaceConfig::defaults();
        cfg.pop_size = snap.pop_size;
        cfg.checkpoint_every = snap.checkpoint_every;
        cfg.crossover_rolls = snap.crossover_rolls;
        cfg.mutate_rolls = snap.mutate_rolls;
        cfg.crossover_prob = snap.crossover_prob;
        cfg.mutate_prob = snap.mutate_prob;
        // A header written before the gate recorded no floor: `0.0` replays it unchanged.
        cfg.mutation_novelty_min = snap.mutation_novelty_min;
        cfg.mutation_novelty_draws = snap.mutation_novelty_draws.max(1);
        cfg.champion_count = snap.champion_count;
        cfg.crossover_ops_pool = snap.crossover_ops_pool.clone();
        cfg.crossover_fitter_base = snap.crossover_fitter_base;
        cfg.max_steps = snap.max_steps;
        cfg.log_level = LogLevel::parse(&snap.log_level).unwrap_or(cfg.log_level);
        cfg.trace_file = snap.trace_file;
        cfg.smoothing_window = snap.smoothing_window;
        cfg.risk_aversion = snap.risk_aversion;
        cfg.window_recency = snap.window_recency;
        // `""` = a header written before the knob, which averaged arithmetically.
        cfg.mean_type = MeanType::parse(&snap.mean_type).unwrap_or(DEFAULT_MEAN_TYPE);
        cfg.train_age = snap.train_age;
        cfg.train_merit_updates = snap.train_merit_updates;
        cfg.csv_export = snap.csv_export;
        cfg.save_optim = snap.save_optim;
        cfg.run_name = snap.run_name.clone();
        cfg.freeze_seats = snap.freeze_seats;
        // LEGACY Ω: folded into the band ONCE, with the same arithmetic the pre-retirement run
        // used — that is what lets an old run resume at the chances it actually paid. New headers
        // record `1.0`, for which this is the identity.
        (cfg.train_merit_p_min, cfg.train_merit_p_max) = fold_omega(
            snap.train_merit,
            snap.train_merit_p_min,
            snap.train_merit_p_max,
        );
        cfg.train_merit = 1.0;
        cfg.train_merit_p_min = snap.train_merit_p_min;
        cfg.train_merit_p_max = snap.train_merit_p_max;
        cfg.train_merit_inverted = snap.train_merit_inverted;
        cfg.train_merit_learn_only = snap.train_merit_learn_only;
        cfg.train_merit_born_ticket = snap.train_merit_born_ticket;
        cfg.policy_scoped_fitness = snap.policy_scoped_fitness;
        cfg.worst_save_topology = snap.worst_save_topology;
        cfg.drift_prob = snap.drift_prob;
        cfg.champion_previous_load = snap.champion_previous_load;
        cfg.champion_previous_dir = snap
            .champion_previous_dir
            .as_ref()
            .map(std::path::PathBuf::from);
        cfg.champion_save_topology = snap.champion_save_topology;
        cfg.mutation_probation_steps = snap.mutation_probation_steps;
        cfg.crossover_probation_steps = snap.crossover_probation_steps;
        cfg.init_probation_steps = snap.init_probation_steps;
        cfg.recovery_retries = snap.recovery_retries;
        cfg.recovery_tolerance = snap.recovery_tolerance;
        cfg.recovery_same_seed = snap.recovery_same_seed;
        cfg.seat_join_probation_steps = snap.seat_join_probation_steps;
        cfg.seat_lifetime_steps = snap.seat_lifetime_steps;
        // The replay-relevant enums, matched on their recorded SPELLING — the snapshots store
        // `Display` output (`"worst"`, `"tournament(size=2)"`, `"split_random"`), so a `contains`
        let cull = snap.crossover_cull_policy.to_ascii_lowercase();
        cfg.crossover_cull_policy = if cull.contains("unfed") {
            CrossCullPolicy::Unfed
        } else if cull.contains("random") {
            CrossCullPolicy::Random
        } else {
            CrossCullPolicy::Worst
        };
        let mcull = snap.mutation_cull_policy.to_ascii_lowercase();
        cfg.mutation_cull_policy = if mcull.contains("unfed") {
            MutationCullPolicy::Unfed
        } else if mcull.contains("random") {
            MutationCullPolicy::Random
        } else if mcull.contains("worst") {
            MutationCullPolicy::Worst
        } else {
            MutationCullPolicy::InverseFitness
        };
        let selection = snap.crossover_selection.to_ascii_lowercase();
        cfg.crossover_selection = if selection.contains("tournament") {
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
        cfg.feature_split_pool = if snap.feature_split_pool.is_empty() {
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
        RaceConfigBuilder {
            cfg: RaceConfig::defaults(),
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
    }
}

/// `RlRaceConfig::builder()` — the RL front door (a free function; an inherent method
/// the alias would collide with `RaceConfig::builder`).
#[doc(hidden)]
pub fn rl_race_config_builder() -> RaceConfigBuilder {
    RaceConfigBuilder {
        cfg: RaceConfig {
            mode_specific: ModeConfig::Rl(RlConfig::default()),
            ..RaceConfig::defaults()
        },
    }
}

/// Parse a recorded (or user-given) feature-split spelling: `"random"`,
/// `"exclusive"`, `"inherited"`, or the `"split_*"` wire form `engine.json` records.
pub(crate) fn parse_feature_split_wire(raw: &str) -> Option<FeatureSplitPolicy> {
    let raw = raw.to_ascii_lowercase();
    if raw.contains("inherited") {
        Some(FeatureSplitPolicy::Inherited)
    } else if raw.contains("exclusive") {
        Some(FeatureSplitPolicy::Exclusive)
    } else if raw.contains("random") {
        Some(FeatureSplitPolicy::Random)
    } else {
        None
    }
}

/// One POOL entry: `Some(None)` is the `none` policy (no split), `Some(Some(_))` a drawable policy,
/// `None` an unknown spelling.
pub(crate) fn parse_feature_split_pool_entry(raw: &str) -> Option<Option<FeatureSplitPolicy>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" | "off" | "split_none" => Some(None),
        "exclusive" | "split_exclusive" => Some(Some(FeatureSplitPolicy::Exclusive)),
        "random" | "split_random" => Some(Some(FeatureSplitPolicy::Random)),
        _ => None,
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
}

impl RaceConfigBuilder {
    /// Set the population size.
    pub fn set_run_pop_size(mut self, n: usize) -> Self {
        self.cfg.pop_size = n;
        self
    }
    /// Set steps between checkpoints — the durability tick (frontier states + EVERY live net's
    /// weights + `ckpt.json` + CSV flush) and the stream's eval-permutation era.
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
    /// Crossover replacement policy (crossover-only).
    pub fn set_crossover_cull_policy(mut self, policy: CrossCullPolicy) -> Self {
        self.cfg.crossover_cull_policy = policy;
        self
    }
    /// Mutation replacement policy: who a firing mutation roll evicts before inserting
    /// immigrant.
    pub fn set_mutation_cull_policy(mut self, policy: MutationCullPolicy) -> Self {
        self.cfg.mutation_cull_policy = policy;
        self
    }
    /// Elite guard: top-k nets are immune to ALL culls. Minimum 1; elites hold rank,
    /// identity.
    pub fn set_champion_count(mut self, n: usize) -> Self {
        self.cfg.champion_count = n.max(1);
        self
    }
    /// ANTI-DEVOLUTION A — elite freeze: the top-`champion_count` nets are exempt
    /// trainer call (frozen, but still scored/ranked/parents). Default `true`.
    pub fn set_seat_freeze(mut self, yes: bool) -> Self {
        self.cfg.freeze_seats = yes;
        self
    }
    /// DEPRECATED — Ω is retired as a second dial: the band `p_min..p_max` IS the payout, so
    /// `set_run_train_merit_p_min` / `set_run_train_merit_p_max` set what a net actually gets.
    /// This setter still WORKS, once: it folds Ω into the band values configured so far
    /// ([`fold_omega`]), which is exactly the chances the old code paid. Being order-sensitive
    /// (`p_min`/`p_max` set AFTER it overwrite the fold), call it last — or, better, drop it.
    /// `0.0` still means "every net trains every clock" (it folds the band to `[1, 1]`).
    #[deprecated(note = "omega is retired: set the band with set_run_train_merit_p_min / _p_max")]
    pub fn set_run_train_merit(mut self, omega: f32) -> Self {
        let (lo, hi) = fold_omega(
            omega,
            self.cfg.train_merit_p_min,
            self.cfg.train_merit_p_max,
        );
        self.cfg.train_merit_p_min = lo;
        self.cfg.train_merit_p_max = hi.max(lo);
        self.cfg.train_merit = 1.0;
        self
    }
    /// The lottery's merit floor `p_min` (default `0.05`, `0.0` = the pre-floor arithmetic).
    pub fn set_run_train_merit_p_min(mut self, p_min: f32) -> Self {
        self.cfg.train_merit_p_min = if p_min.is_finite() {
            p_min.clamp(0.0, 1.0)
        } else {
            DEFAULT_TRAIN_MERIT_P_MIN
        };
        self
    }
    /// CEILING `p_max` on the lottery fraction: the best-graded net's share of clocks (default
    /// `1.0` = no ceiling).
    pub fn set_run_train_merit_p_max(mut self, p_max: f32) -> Self {
        self.cfg.train_merit_p_max = if p_max.is_finite() {
            p_max.clamp(0.0, 1.0)
        } else {
            DEFAULT_TRAIN_MERIT_P_MAX
        };
        self
    }
    /// INVERT the lottery so the LEAST fit net earns the most tickets (default
    /// [`DEFAULT_TRAIN_MERIT_INVERTED`] = `true`; `false` = the merit ordering).
    pub fn set_run_train_merit_inverted(mut self, yes: bool) -> Self {
        self.cfg.train_merit_inverted = yes;
        self
    }
    /// LEARN-ONLY lottery (default `false`): `true` keeps every net ACTING and MEASURING every
    /// clock and gates only the weight update on the ticket, so a net's reading never goes stale.
    /// `false` keeps the pre-flag contract — a lost ticket idles the net outright. `U = 0` is exempt
    /// either way: a never-learned net idles a denied clock, since it has nothing to measure.
    pub fn set_run_train_merit_learn_only(mut self, yes: bool) -> Self {
        self.cfg.train_merit_learn_only = yes;
        self
    }
    /// BORN TICKET (default `false`): `true` pays a net that has never trained (`U = 0`) `p = 1.0`
    /// directly, before the Ω band folds its fraction. The shield already hands such a net a
    /// fraction of `1.0`, but the band turns that into `hi` — so without the born ticket the coin
    /// can idle a net that cannot spend its shield without training, and an idle clock buys it no
    /// update. `false` keeps the pre-flag contract bit for bit.
    pub fn set_run_train_merit_born_ticket(mut self, yes: bool) -> Self {
        self.cfg.train_merit_born_ticket = yes;
        self
    }
    /// POLICY-SCOPED FITNESS (default `false`): `true` builds the mean/spread/n ONLY from readings
    /// taken on the weights the net holds now, so the statistics describe one POLICY and never a
    /// mix of policies. A net that updates every clock reports one datapoint (its value, no
    /// spread); a net that rarely earns a ticket reports many readings of the SAME policy.
    pub fn set_run_policy_scoped_fitness(mut self, yes: bool) -> Self {
        self.cfg.policy_scoped_fitness = yes;
        self
    }
    /// ANTI-CHAMPION dump (default `false`): `true` writes `worst-<hash>-<fitness>.md` for the top-
    /// `champion_count` nets by TICKET CHANCE at stop (least fit × oldest) plus a `worst.md` index,
    /// each with its topology, mermaid graph and lineage — the mirror of `champion_save_topology`.
    pub fn set_worst_save_topology(mut self, enabled: bool) -> Self {
        self.cfg.worst_save_topology = enabled;
        self
    }
    /// DRIFT on a crossover child's inherited genes (default `0.0` = exact inheritance).
    pub fn set_topology_drift(mut self, p: f32) -> Self {
        self.cfg.drift_prob = if p.is_finite() {
            p.clamp(0.0, 1.0)
        } else {
            DEFAULT_TOPOLOGY_DRIFT
        };
        self
    }
    /// Per-net smoothed-fitness rolling window (K). Default [`SMOOTHING_WINDOW`] = 10.
    /// Replay-relevant: a resumed run must use the same value or construction fails parity.
    pub fn set_run_smoothing_window(mut self, k: usize) -> Self {
        self.cfg.smoothing_window = k.max(1);
        self
    }
    /// The lottery's AGE leg (`train_age`): `true` (default) ARMS the product `Fm × Ym`, so a young
    /// net's ticket is lifted; `false` makes the ticket follow the fitness place alone. The leg
    /// multiplies the MERIT fraction, which is what the name says.
    pub fn set_run_train_merit_age(mut self, armed: bool) -> Self {
        self.cfg.train_age = armed;
        self
    }
    /// The lottery's U leg (`train_merit_updates`): `true` ARMS `Um`, the age leg's twin on updates
    /// — `1.0` at the field's fewest updates, `0.0` at its most — so the ticket becomes
    /// `Fm × Ym × Um` (`Fm × Um` with the age leg off). A net that spent its updates on rescue
    /// retries is the OLD one here, and pays for them out of its own future tickets.
    pub fn set_run_train_merit_updates(mut self, armed: bool) -> Self {
        self.cfg.train_merit_updates = armed;
        self
    }
    /// Deprecated spelling of [`Self::set_run_train_merit_age`] — the leg multiplies the lottery's
    /// merit fraction, not the run's age. Kept so existing callers keep compiling.
    #[deprecated(note = "use set_run_train_merit_age")]
    pub fn set_run_train_age(self, armed: bool) -> Self {
        self.set_run_train_merit_age(armed)
    }
    /// Recency ramp r for the smoothing window (`window_recency`).

    pub fn set_run_window_recency(mut self, r: f32) -> Self {
        self.cfg.window_recency = if r.is_finite() {
            r.clamp(0.0, 1.0)
        } else {
            DEFAULT_WINDOW_RECENCY
        };
        self
    }
    /// How the window's readings are averaged into the ranking centre (`mean_type`).
    pub fn set_run_mean_type(mut self, mean_type: MeanType) -> Self {
        self.cfg.mean_type = mean_type;
        self
    }
    /// Risk aversion λ (FIT) — how many window standard deviations a net pays for its own spread,
    /// so a steady net outranks a volatile one with the same window mean.
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
    /// Parent-selection method for the crossover draw.
    pub fn set_crossover_selection(mut self, method: SelectionMethod) -> Self {
        self.cfg.crossover_selection = method;
        self
    }
    /// Make the FITTER of the two drawn parents the crossover BASE
    /// ([`RaceConfig::crossover_fitter_base`]).
    pub fn set_crossover_fitter_base(mut self, on: bool) -> Self {
        self.cfg.crossover_fitter_base = on;
        self
    }
    /// Enable the feature split and choose how the Input node's per-port masks are drawn.
    pub fn set_topology_feature_split(mut self, policy: FeatureSplitPolicy) -> Self {
        self.cfg.feature_split_pool = vec![policy.to_string()];
        self
    }
    /// The pool a NEWBORN's split is drawn from, in draw order — `["none", "random", "exclusive"]`
    /// gives every policy a chance, `["exclusive"]` pins one, `["none"]` turns the split off.
    pub fn set_topology_feature_split_pool<I, S>(mut self, pool: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.feature_split_pool = pool.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }
    /// Explicit step budget — the only built-in stop criterion.
    pub fn set_stop_max_steps(mut self, n: impl Into<Option<usize>>) -> Self {
        self.cfg.max_steps = n.into();
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
    /// Dropout probability applied when building this run's networks (part
    /// `TopologyOptions`). TRAIN forwards
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

    /// Human label for the experiment, written as `"run_name"`. Informational only —
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
    /// Opt in/out of the automatic inheritance of the newest compatible previous experiment's
    /// elites (topology-only).
    pub fn set_champion_previous_load(mut self, enabled: bool) -> Self {
        self.cfg.champion_previous_load = enabled;
        self
    }
    /// Where the automatic inheritance looks: a run dir or a dir of run dirs
    /// (newest id wins). Unset ⇒ the default `results/`.
    pub fn set_champion_previous_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.cfg.champion_previous_dir = Some(dir.into());
        self
    }
    /// Mutation novelty floor `d_min`: the minimum Δ a random immigrant must hold against the
    /// NEAREST blueprint the run already made. `0.0` turns the gate OFF entirely (no parse, no
    /// redraw); the built-in default is [`DEFAULT_MUTATION_NOVELTY_MIN`].
    pub fn set_mutation_novelty_min(mut self, d_min: f32) -> Self {
        self.cfg.mutation_novelty_min = d_min.max(0.0);
        self
    }
    /// Candidates ONE mutation may spend clearing the novelty floor before the most novel draw
    /// is taken anyway. Default [`DEFAULT_MUTATION_NOVELTY_DRAWS`].
    pub fn set_mutation_novelty_draws(mut self, draws: usize) -> Self {
        self.cfg.mutation_novelty_draws = draws.max(1);
        self
    }
    /// Mutation probation steps: cull-immunity for a random immigrant's first
    /// `k` UPDATES — `U`, not clocks, so waiting does not spend it (see
    /// [`RaceConfig::mutation_probation_steps`]). Default 5.
    pub fn set_mutation_probation_steps(mut self, k: usize) -> Self {
        self.cfg.mutation_probation_steps = k;
        self
    }
    /// Crossover probation steps: cull-immunity for a crossover child's first
    /// `k` UPDATES — `U`, not clocks (see [`RaceConfig::crossover_probation_steps`]). Default 5.
    pub fn set_crossover_probation_steps(mut self, k: usize) -> Self {
        self.cfg.crossover_probation_steps = k;
        self
    }
    /// INIT PROBATION (default `0` = off): the opening clocks with no culls and no births —
    /// evolution starts at clock `k`. Before it every evolution roll is forgone and the log names
    /// the probation, so the warm-up is a DECLARED choice rather than a side effect of every
    /// founder sitting inside its cull shield. `0` withholds nothing beyond the born shields; to
    /// let evolution start EARLIER, shorten the shields themselves
    /// (`set_crossover_probation_steps` / `set_mutation_probation_steps`).
    pub fn set_run_init_probation_steps(mut self, k: usize) -> Self {
        self.cfg.init_probation_steps = k;
        self
    }
    /// Recovery updates: how many EXTRA real updates a net may take in the same clock when the one
    /// it just took left its reading worse than it was (see [`RaceConfig::recovery_retries`]).
    /// Default `0` = off; `k > 0` arms the streak.
    ///
    /// The streak never fires inside the INIT PROBATION (`set_run_init_probation_steps`): those
    /// clocks exist to judge what the founders are on draws every net shares, so extra updates
    /// nobody else got are withheld along with the culls and births (same clock arithmetic).
    pub fn set_recovery_retries(mut self, k: usize) -> Self {
        self.cfg.recovery_retries = k;
        self
    }
    /// Recovery tolerance: the FRACTION of the net's standing a reading may lose before the streak
    /// fires (`0.2` = a fifth is tolerated, more triggers). See
    /// [`RaceConfig::recovery_tolerance`]. Default `0.0` = every drop triggers.
    pub fn set_recovery_tolerance(mut self, fraction: f32) -> Self {
        self.cfg.recovery_tolerance = if fraction.is_finite() {
            fraction.max(0.0)
        } else {
            0.0
        };
        self
    }
    /// Recovery retry draw: `true` (default) = a retry repeats the clock's OWN train draw, so the
    /// streak re-tests the update that fell; `false` = each retry walks one draw further, so it
    /// fights on fresh evidence. See [`RaceConfig::recovery_same_seed`].
    pub fn set_recovery_same_seed(mut self, same: bool) -> Self {
        self.cfg.recovery_same_seed = same;
        self
    }
    /// Seat join probation: a net cannot take a seat until it has aged `k`
    /// clocks since arrival. Default `5`.
    pub fn set_seat_join_probation_steps(mut self, k: usize) -> Self {
        self.cfg.seat_join_probation_steps = k;
        self
    }
    /// Seat lifetime: how long a net may stay at peak on the frozen crown before
    /// age-rotation. Default `5`.
    pub fn set_seat_lifetime_steps(mut self, k: usize) -> Self {
        self.cfg.seat_lifetime_steps = k;
        self
    }
    /// At stop, save each champion's topology markdown (`champion-<hash>-<fitness>.md`).
    /// Default true.
    pub fn set_champion_save_topology(mut self, enabled: bool) -> Self {
        self.cfg.champion_save_topology = enabled;
        self
    }
    /// Finish and return the configured [`RaceConfig`].
    pub fn build(self) -> RaceConfig {
        self.cfg
    }
}

impl RaceConfig {
    /// Parse an activation label (the `Display` form: "relu", "gelu", …).
    pub(crate) fn parse_activation(s: &str) -> Result<crate::graph::node::Activation, String> {
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

    /// Parse a combine-op label (the `Display` form: "add", "mean", …) — see
    /// [`RaceConfig::parse_activation`] for why it is `pub(crate)`.
    pub(crate) fn parse_combine(s: &str) -> Result<crate::graph::node::CombineOp, String> {
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

    /// Resolve the crossover operator pool: the configured labels validated, or both when empty.
    pub fn resolved_crossover_ops(&self) -> Result<Vec<CrossoverOp>, String> {
        super::pool::resolve::<super::pool::CrossoverOpsPool>(self)
    }

    /// Resolve the activation pool: the configured labels parsed, or all known activations
    /// when empty. Unknown labels error at run start.
    pub fn resolved_activation_pool(&self) -> Result<Vec<crate::graph::node::Activation>, String> {
        super::pool::resolve::<super::pool::ActivationPool>(self)
    }

    /// Resolve the combine-op sampling pool — see
    /// [`RaceConfig::resolved_activation_pool`].
    pub fn resolved_combine_pool(&self) -> Result<Vec<crate::graph::node::CombineOp>, String> {
        super::pool::resolve::<super::pool::CombineOpPool>(self)
    }

    /// Resolve the standardize-op sampling pool — see
    /// [`RaceConfig::resolved_activation_pool`].
    pub fn resolved_standardize_pool(
        &self,
    ) -> Result<Vec<crate::graph::node::StandardizeOp>, String> {
        super::pool::resolve::<super::pool::StandardizePool>(self)
    }

    /// Check every NAME-based pool once, before a single net is drawn.
    pub fn validate_search_pools(&self) -> Result<(), String> {
        let errors: Vec<String> = [
            super::pool::error::<super::pool::CrossoverOpsPool>(self),
            super::pool::error::<super::pool::ActivationPool>(self),
            super::pool::error::<super::pool::CombineOpPool>(self),
            super::pool::error::<super::pool::StandardizePool>(self),
            super::pool::error::<super::pool::FeatureSplitPool>(self),
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
    use super::{LogLevel, RISK_AVERSION, RaceConfig};

    /// The examples' `--log-level` vocabulary must parse: handed to an `EnvFilter` verbatim, `summ`
    /// would be read as a module name and mute the whole run — this mapping is load-bearing.
    #[test]
    fn log_level_names_parse_to_a_visible_verbosity() {
        assert_eq!(LogLevel::parse("summ"), Some(LogLevel::Summ));
        assert_eq!(LogLevel::parse("SUMM"), Some(LogLevel::Summ));
        assert_eq!(LogLevel::parse("minimal"), Some(LogLevel::Minimal));
        assert_eq!(LogLevel::parse("none"), Some(LogLevel::None));
        assert_eq!(
            LogLevel::parse("debug"),
            None,
            "tracing levels pass through"
        );
        assert_eq!(LogLevel::Summ.env_filter(), "info");
        assert_eq!(LogLevel::Minimal.env_filter(), "warn");
        assert_eq!(LogLevel::None.env_filter(), "warn");
    }

    /// A pool NAME is either drawable or a refusal — never silently widened.
    #[test]
    fn an_unparseable_pool_name_is_an_error_not_a_fallback() {
        // Empty pools are the documented "all" and must keep validating.
        assert_eq!(RaceConfig::defaults().validate_search_pools(), Ok(()));

        let cfg = RaceConfig {
            feature_split_pool: vec!["rondom".to_string()],
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
            feature_split_pool: vec!["rondom".to_string()],
            activation_pool: vec!["reluu".to_string()],
            crossover_ops_pool: vec!["two_point".to_string()],
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
        use super::parse_feature_split_pool_entry as parse;
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
            feature_split_pool: vec!["nonexclusive".to_string()],
            ..RaceConfig::defaults()
        };
        assert!(cfg.validate_search_pools().is_err());
    }

    /// FIT's one knob: the default is the risk-adjusted ranking, `0.0` is the documented OFF switch
    /// (the pre-FIT plain mean), negatives clamp to `0.0` (no risk-SEEKING ranking exists)
    #[test]
    fn risk_aversion_defaults_on_and_clamps_at_off() {
        // Default: the risk term
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

    /// The parent draw defaults to the reference loop's tournament of 2; the tournament arm must
    /// survive the round-trip untouched, and roulette stays expressible for a caller that wants
    #[test]
    fn crossover_selection_defaults_to_tournament_two_and_round_trips() {
        use crate::evolution::selection::SelectionMethod;
        assert_eq!(
            RaceConfig::defaults().crossover_selection,
            SelectionMethod::Tournament { tournament_size: 2 },
            "the knob's default is the reference loop's draw"
        );
        let cfg = RaceConfig::builder()
            .set_crossover_selection(SelectionMethod::Tournament { tournament_size: 3 })
            .build();
        assert_eq!(
            cfg.crossover_selection,
            SelectionMethod::Tournament { tournament_size: 3 }
        );
        // The recorded label is the Display form the snapshot stores.
        assert_eq!(cfg.crossover_selection.to_string(), "tournament(size=3)");
        assert_eq!(SelectionMethod::Roulette.to_string(), "roulette");
    }

    /// The stop field setters are mutually exclusive BY CONSTRUCTION: each write clears
    /// sibling, so "const default then flag wins" composes.
    #[test]
    fn stop_setters_are_exclusive_last_writer_wins() {
        // (the old max_target_fitness exclusivity test shrank
        // feature's deletion — only the step budget remains)
        let cfg = RaceConfig::builder().set_stop_max_steps(15).build();
        assert_eq!(cfg.max_steps, Some(15));
    }
}
