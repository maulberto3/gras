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

/// How the smoothing window's readings are averaged into the ranking centre (`set_fitness_mean_type`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum MeanType {
    /// The weighted window mean — the plain average, the pre-knob ranking.
    Arithmetic,
    /// `g(x) = sign(x)·ln(1 + |x|)`: centre and spread are taken in that compressed space
    /// score is mapped back, so one spike cannot dominate the window the way it does through
    SignedLog,
    /// The WEIGHTED MEDIAN: no single reading can move the centre at all; the spread is the medians absolute deviation scaled by `1.4826`, so a median spread stays comparable…
    #[default]
    Median,
}

impl MeanType {
    /// The spelling `engine.json` records and the log prints.
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }

    /// Parse a recorded spelling. Unknown (and `""` from a header written before the
    /// knob) yields `None`, which every caller reads as [`DEFAULT_MEAN_TYPE`].
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

/// The ranking family a command line names, so a flag can be a typed `MeanType` rather than a
/// string the caller has to parse (the same contract the cull policies use).
impl std::str::FromStr for MeanType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        MeanType::parse(s)
            .ok_or_else(|| format!("{s}: expected one of arithmetic, signed_log, median"))
    }
}

/// Default [`MeanType`] — the WEIGHTED MEDIAN: one spike cannot promote or demote a net on a
/// fluke. `arithmetic` is the older plain-window ranking.
pub const DEFAULT_MEAN_TYPE: MeanType = MeanType::Median;

/// The lottery's two LEGS (`set_train_ticket`): which fit PLACE earns the most tickets, and which side of the UPDATES field the ticket multiplies in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum TrainTicket {
    /// The fittest net earns the most tickets; the updates leg is OFF (`Fm⁺`).
    MostFit,
    /// The fittest net that has trained the LEAST earns the most tickets (`Fm⁺ × Um⁻`) — the engine default ([`DEFAULT_TRAIN_TICKET`]): pay the net that grades best and has had…
    #[default]
    MostFitFewestUpdates,
    /// The fittest net that has trained the MOST earns the most tickets (`Fm⁺ × Um⁺`).
    MostFitMostUpdates,
    /// The LEAST fit net that has trained the LEAST earns the most tickets (`Fm⁻ × Um⁻`) — the
    /// reference run's tuned leg.
    LeastFitFewestUpdates,
    /// The LEAST fit net that has trained the MOST earns the most tickets (`Fm⁻ × Um⁺`).
    LeastFitMostUpdates,
    /// The LEAST fit net earns the most tickets; the updates leg is OFF (`Fm⁻`).
    LeastFit,
}

impl TrainTicket {
    /// Every leg, in the order the docs, the CLI help and the error messages list them — ONE
    /// list, so a surface handed an unknown spelling names the real set instead of a retyped copy.
    pub const ALL: [TrainTicket; 6] = [
        TrainTicket::MostFit,
        TrainTicket::MostFitFewestUpdates,
        TrainTicket::MostFitMostUpdates,
        TrainTicket::LeastFitFewestUpdates,
        TrainTicket::LeastFitMostUpdates,
        TrainTicket::LeastFit,
    ];

    /// [`Self::ALL`] as the spellings [`Self::as_str`] prints, for messages that list the choices.
    pub const ALL_NAMES: [&str; 6] = [
        "most_fit",
        "most_fit_fewest_updates",
        "most_fit_most_updates",
        "least_fit_fewest_updates",
        "least_fit_most_updates",
        "least_fit",
    ];

    /// The spelling `engine.json` records, the CLI accepts and the log prints.
    pub fn as_str(self) -> &'static str {
        self.into()
    }

    /// Parse a recorded or typed spelling: case, spaces and hyphens fold to `_`, so `Least-Fit Fewest Updates` reads as `least_fit_fewest_updates`. `None` = not a leg, which…
    pub fn parse(raw: &str) -> Option<Self> {
        let key = raw.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        match key.as_str() {
            "most_fit" | "best_fit" => Some(TrainTicket::MostFit),
            "most_fit_most_updates" | "best_most" => Some(TrainTicket::MostFitMostUpdates),
            "most_fit_fewest_updates" | "most_fit_least_updates" | "best_fewest" => {
                Some(TrainTicket::MostFitFewestUpdates)
            }
            "least_fit" | "worst_fit" => Some(TrainTicket::LeastFit),
            "least_fit_most_updates" | "worst_most" => Some(TrainTicket::LeastFitMostUpdates),
            "least_fit_fewest_updates" | "least_fit_least_updates" | "worst_fewest" => {
                Some(TrainTicket::LeastFitFewestUpdates)
            }
            _ => None,
        }
    }

    /// Does the ticket pay the LEAST fit net? `true` mirrors the place (`Fm⁻`).
    pub fn fitness_inverted(self) -> bool {
        matches!(
            self,
            TrainTicket::LeastFit
                | TrainTicket::LeastFitFewestUpdates
                | TrainTicket::LeastFitMostUpdates
        )
    }

    /// Does the updates leg multiply the place? `false` = place only.
    pub fn updates_armed(self) -> bool {
        !matches!(self, TrainTicket::MostFit | TrainTicket::LeastFit)
    }

    /// Which side of the updates field the ticket pays: `true` = the LEAST trained net
    /// (`Um⁻`, `1` at the field's fewest updates). Only meaningful while the leg is armed.
    pub fn updates_inverted(self) -> bool {
        matches!(
            self,
            TrainTicket::MostFitFewestUpdates | TrainTicket::LeastFitFewestUpdates
        )
    }

    /// The same legs with the fit PLACE flipped (`most_fit` ↔ `least_fit`), updates side kept.
    pub fn with_fitness_inverted(self, inverted: bool) -> Self {
        match (inverted, self.updates_armed(), self.updates_inverted()) {
            (false, false, _) => TrainTicket::MostFit,
            (true, false, _) => TrainTicket::LeastFit,
            (false, true, true) => TrainTicket::MostFitFewestUpdates,
            (true, true, true) => TrainTicket::LeastFitFewestUpdates,
            (false, true, false) => TrainTicket::MostFitMostUpdates,
            (true, true, false) => TrainTicket::LeastFitMostUpdates,
        }
    }

    /// The same place with the UPDATES leg armed or switched off.
    pub fn with_updates_armed(self, armed: bool) -> Self {
        if !armed {
            return if self.fitness_inverted() {
                TrainTicket::LeastFit
            } else {
                TrainTicket::MostFit
            };
        }
        match (
            self.fitness_inverted(),
            self.updates_armed(),
            self.updates_inverted(),
        ) {
            (false, true, false) => TrainTicket::MostFitMostUpdates,
            (true, true, false) => TrainTicket::LeastFitMostUpdates,
            (false, _, _) => TrainTicket::MostFitFewestUpdates,
            (true, _, _) => TrainTicket::LeastFitFewestUpdates,
        }
    }

    /// The legs the two pre-enum booleans meant: `inverted` = which place, `updates_armed` = U
    /// leg on or off (on means the LEAST trained net, the old `Um⁻`).
    pub fn from_bools(inverted: bool, updates_armed: bool) -> Self {
        // The pair named the LEGACY `Um` leg (fewest updates), not today's default, so start from
        // the leg the booleans actually meant and let the setters move the two axes.
        TrainTicket::MostFitFewestUpdates
            .with_fitness_inverted(inverted)
            .with_updates_armed(updates_armed)
    }
}

impl std::fmt::Display for TrainTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The leg a command line names, so the flag can be a typed `TrainTicket` (the setter's own
/// type) instead of a string parsed at the call site.
impl std::str::FromStr for TrainTicket {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        TrainTicket::parse(s)
            .ok_or_else(|| format!("{s}: expected one of {}", TrainTicket::ALL_NAMES.join(", ")))
    }
}

/// Default ticket LEGS (`set_train_ticket`): the FITTEST net that has trained the FEWEST earns the most tickets (`Fm⁺ × Um⁻`) — the lottery pays the net that grades best…
pub const DEFAULT_TRAIN_TICKET: TrainTicket = TrainTicket::MostFitFewestUpdates;

/// Default for the lottery's LEARN-ONLY switch (`set_ticket_learn_only`): `false` = a lost ticket idles the net (no train, no eval, nothing measured). `true` = every net…
pub const DEFAULT_TRAIN_MERIT_LEARN_ONLY: bool = false;

/// Default for POLICY-SCOPED FITNESS (`set_fitness_policy_scoped`): `false` = the window averages the last K READINGS, whatever weights produced them (the pre-flag…
pub const DEFAULT_POLICY_SCOPED_FITNESS: bool = true;

/// Default for the ANTI-CHAMPION dump (`set_worst_save_topology`): `false` = no `worst-*.md` files. `true` writes the `champion_count` nets at the BOTTOM of the final table…
pub const DEFAULT_WORST_SAVE_TOPOLOGY: bool = false;

/// Steps between population checkpoints — the cadence the stream's era rotation and the CSV flush
/// key off.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 25;

/// Default population size: `10`, deliberately conservative.
pub const DEFAULT_POP_SIZE: usize = 10;

/// Default step budget, read at the same conservative scale as the population: `10` clocks.
pub const DEFAULT_MAX_STEPS: usize = 10;

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

/// Default retry DRAW for a recovery streak (`set_pop_recovery_same_seed`): `true` = a retry re-runs the SAME train draw the update fell on (same rows, same RNG stream) — a…
pub const DEFAULT_RECOVERY_SAME_SEED: bool = false;

/// Default recovery tolerance: the drop a new reading may take against the net's standing and
/// still count as noise. `0.0` = any drop earns a rescue; `0.2` = a fifth is tolerated.
pub const DEFAULT_RECOVERY_TOLERANCE: f32 = 0.0;

/// Default INIT PROBATION (`set_pop_init_probation_steps`): `0` = nothing is withheld beyond the born shields, which is exactly what every run before the knob did.
pub const DEFAULT_INIT_PROBATION_STEPS: usize = 0;

/// Default step between sampled hidden widths.
pub const DEFAULT_HIDDEN_DIM_STRIDE: usize = 16;

/// Default crossover parent tournament size.
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

/// Default hidden-width sampling range: the cartpole reference's `16..=64`,
/// stride [`RaceConfig::defaults`] pins.
pub const DEFAULT_HIDDEN_POOL: std::ops::RangeInclusive<usize> = 16..=64;

/// Default merit floor `p_min`: the smallest share of clocks the WORST net in the lottery's field
/// still gets — a quarter, so even the bottom of the field keeps a ladder back up.
pub const DEFAULT_TRAIN_MERIT_P_MIN: f32 = 0.25;
/// Default lottery CEILING `p_max`: the largest share of clocks the best-graded net may take —
/// three quarters, so the field's best net still leaves clocks for the rest.
pub const DEFAULT_TRAIN_MERIT_P_MAX: f32 = 0.75;

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
pub const DEFAULT_MUTATION_NOVELTY_MIN: f32 = 0.4;

/// Default candidates ONE mutation may spend before the most novel is taken anyway.
pub const DEFAULT_MUTATION_NOVELTY_DRAWS: usize = 10;

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

/// Who gets evicted when a crossover child takes a slot. **Crossover-only** — the mutation channel has its own policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum CrossCullPolicy {
    /// **LowestFitness** — evict the net with the lowest smoothed fitness; strictly
    /// merit-based. Risk: the population can ossify around one lineage.
    LowestFitness,
    /// **Random** — evict a uniformly random live net; keeps slots turning over and
    /// prevents a long-lived leader from starving diversity. Risk: losing a strong net.
    Random,
    /// **HighestTicketChance** — evict the net the coin pays MOST: the highest ticket chance `p = p_min + (p_max - p_min) x f` among the graded nets, deterministic with the…
    HighestTicketChance,
    /// **LowestFitnessMostUpdates** — evict the net with the highest `Fm⁻ × (1 − Um⁻)`: the lowest-fitness net that has ALSO spent its updates.
    #[default]
    LowestFitnessMostUpdates,
}

impl std::fmt::Display for CrossCullPolicy {
    /// The STABLE spelling recorded in `engine.json` and the run manifest (snake_case,
    /// not the `Debug` casing): a header written today must read the same tomorrow.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str((*self).into())
    }
}

/// The reverse of [`Display`](std::fmt::Display): the policy a `--…-cull-policy` flag names.
impl std::str::FromStr for CrossCullPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lowest_fitness" | "worst" => Ok(Self::LowestFitness),
            "random" => Ok(Self::Random),
            "highest_ticket_chance" | "unfed" => Ok(Self::HighestTicketChance),
            "lowest_fitness_most_updates" | "spent" => Ok(Self::LowestFitnessMostUpdates),
            _ => Err(format!(
                "{s}: expected lowest_fitness, random, highest_ticket_chance or lowest_fitness_most_updates"
            )),
        }
    }
}

/// Who gets evicted when a MUTATION roll inserts a fresh random immigrant.
/// **Mutation-only** — the crossover channel has its own policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum MutationCullPolicy {
    /// **LowestFitness** — evict the net with the lowest smoothed fitness, deterministically
    /// (mirrors [`CrossCullPolicy::LowestFitness`]).
    LowestFitness,
    /// **Random** — evict a uniformly random live net, at the cost of occasionally
    /// losing a good one.
    Random,
    /// **HighestTicketChance** — the SAME selector as [`CrossCullPolicy::HighestTicketChance`]: evict the net the train-eval lottery pays the most (the highest `p` among the…
    HighestTicketChance,
    /// **LowestFitnessMostUpdates** — the SAME selector as [`CrossCullPolicy::LowestFitnessMostUpdates`]: evict the net with the highest `Fm⁻ × (1 − Um⁻)`, so the slot goes to…
    #[default]
    LowestFitnessMostUpdates,
}

impl std::fmt::Display for MutationCullPolicy {
    /// The STABLE spelling recorded in `engine.json` and the run manifest.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str((*self).into())
    }
}

/// The reverse of [`Display`](std::fmt::Display) — the same spellings and the same legacy aliases as [`CrossCullPolicy`], so the mutation channel's flag parses exactly like…
impl std::str::FromStr for MutationCullPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lowest_fitness" | "worst" => Ok(Self::LowestFitness),
            "random" => Ok(Self::Random),
            "highest_ticket_chance" | "unfed" => Ok(Self::HighestTicketChance),
            "lowest_fitness_most_updates" | "spent" => Ok(Self::LowestFitnessMostUpdates),
            _ => Err(format!(
                "{s}: expected lowest_fitness, random, highest_ticket_chance or lowest_fitness_most_updates"
            )),
        }
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
    /// CROSSOVER turnover — the channel's ONE knob — as a share of the live population per clock: `0.1` = 10%, dealt as `round(share · pop)` rolls EVERY clock (at least one…
    pub crossover_share: f32,
    /// MUTATION turnover — the mutation channel's own knob — as a share of the live population per
    /// clock. Same contract as [`RaceConfig::crossover_share`]; `0.0` switches the channel off.
    pub mutation_share: f32,
    /// Who a surviving crossover child evicts (crossover-only; immigrants follow
    /// `mutation_cull_policy`).
    pub crossover_cull_policy: CrossCullPolicy,
    /// Who a firing mutation roll evicts (mutation-only; crossover children follow
    /// `crossover_cull_policy`).
    pub mutation_cull_policy: MutationCullPolicy,
    /// CHAMPION set: the top-k live nets by smoothed fitness, saved and inheritable at stop (minimum 1).
    pub champion_count: usize,
    /// Lottery floor `p_min`: the worst net's chance of a ticket each clock.
    pub train_merit_p_min: f32,
    /// Lottery ceiling `p_max`: the best net's chance of a ticket each clock.
    pub train_merit_p_max: f32,
    /// The lottery's two LEGS (`set_train_ticket`): which fit place earns the ticket and which side of the updates field it multiplies in.
    pub train_ticket: TrainTicket,
    /// LEARN-ONLY lottery (`set_ticket_learn_only`): `true` = a lost ticket costs the weight UPDATE, not the STEP — the net runs its trainer with a no-op optimizer, so it acts…
    pub train_merit_learn_only: bool,
    /// Share of the dataset held out for EVAL (`eval_share`): the rows a net is SCORED on and never
    /// trained
    pub eval_share: f32,
    /// Per-net smoothed-fitness rolling window (K) — how many recent step fitnesses
    /// ranking decision averages over. Defaults to [`SMOOTHING_WINDOW`].
    pub smoothing_window: usize,
    /// How the window's readings are averaged into the ranking centre (`set_fitness_mean_type`): [`MeanType::Median`] (the default — no single reading moves the centre)…
    pub mean_type: MeanType,
    /// POLICY-SCOPED FITNESS (`set_fitness_policy_scoped`): `true` makes the mean/spread/n describe the CURRENT weights — the readings taken since the net's last real update —…
    pub policy_scoped_fitness: bool,
    /// ANTI-CHAMPION dump (`set_worst_save_topology`): `true` writes `worst-<hash>-<fitness>.md` + `worst.md` at stop for the nets ranked by `spent` (`Fm⁻ × (1 − Um)`): the bad…
    pub worst_save_topology: bool,
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
    /// Metrics that ALSO feed the ranking (`set_selection_metric`): each adds `weight × the metric's smoothed window centre` to the fitness score, so `score = fitness centre +…
    pub selection_metrics: Vec<crate::utils::fitness::SelectionMetric>,
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
    /// INIT PROBATION (`set_pop_init_probation_steps`): clocks the run spends WITHOUT evolution — no culls, no births — however cull-eligible the population already is.
    pub init_probation_steps: usize,
    /// Post-update RECOVERY streak cap: after a real update whose reading comes in WORSE than the net's standing going in, up to `recovery_retries` extra real updates are taken…
    pub recovery_retries: usize,
    /// Recovery tolerance: how far a reading may fall below the net's standing WITHOUT earning a rescue, as a FRACTION of that standing (`0.2` = a fifth, so 268 may read down…
    pub recovery_tolerance: f32,
    /// Recovery retry DRAW (`set_pop_recovery_same_seed`): `true` (default) = every retry repeats the clock's OWN train draw — the same rows and the same RNG stream — so the…
    pub recovery_same_seed: bool,
}

// `run_progress` and `DEFAULT_LR` live in `crate::utils::training` — the run
// loop and any training scheme both need them, so they sit below both layers.

/// Which mode the run is in; carried in the type, no payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeConfig {
    Tabular,
    Rl,
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
    /// Best RANKING score — the window centre in the run's mean family, plus any
    /// `selection_metrics` terms. This is what the race actually ranks on.
    pub best_smoothed_fitness: f32,
    /// Worst ranking score — the same number, at the other end.
    pub worst_smoothed_fitness: f32,
    /// Mean of the live ranking scores (a plain arithmetic mean OF the scores,
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
    /// channel, tournament-2 crossover parents, a 5-reading window, the inverted lottery
    pub fn defaults() -> Self {
        RaceConfig {
            pop_size: DEFAULT_POP_SIZE,
            checkpoint_every: DEFAULT_CHECKPOINT_EVERY,
            crossover_share: DEFAULT_CROSSOVER_SHARE,
            mutation_share: DEFAULT_MUTATION_SHARE,
            crossover_cull_policy: CrossCullPolicy::default(),
            mutation_cull_policy: MutationCullPolicy::default(),
            champion_count: DEFAULT_CHAMPION_COUNT,
            train_merit_p_min: DEFAULT_TRAIN_MERIT_P_MIN,
            train_merit_p_max: DEFAULT_TRAIN_MERIT_P_MAX,
            train_ticket: DEFAULT_TRAIN_TICKET,
            train_merit_learn_only: DEFAULT_TRAIN_MERIT_LEARN_ONLY,
            eval_share: DEFAULT_EVAL_SHARE,
            smoothing_window: SMOOTHING_WINDOW,
            mean_type: DEFAULT_MEAN_TYPE,
            policy_scoped_fitness: DEFAULT_POLICY_SCOPED_FITNESS,
            worst_save_topology: DEFAULT_WORST_SAVE_TOPOLOGY,
            mode_specific: ModeConfig::Tabular,
            crossover_ops_pool: Vec::new(),
            crossover_selection: SelectionMethod::Tournament {
                tournament_size: DEFAULT_TOURNAMENT_SIZE,
            },
            crossover_fitter_base: DEFAULT_CROSSOVER_FITTER_BASE,
            max_steps: Some(DEFAULT_MAX_STEPS),
            hidden_dim_pool: Some(DEFAULT_HIDDEN_POOL),
            hidden_dim_stride: DEFAULT_HIDDEN_DIM_STRIDE,
            combine_op_pool: Vec::new(),
            feature_split_pool: Vec::new(),
            drift_prob: DEFAULT_TOPOLOGY_DRIFT,
            activation_pool: Vec::new(),
            standardize_op_pool: Vec::new(),
            topology_options: crate::graph::topology::TopologyOptions::default(),
            mutation_novelty_min: DEFAULT_MUTATION_NOVELTY_MIN,
            mutation_novelty_draws: DEFAULT_MUTATION_NOVELTY_DRAWS,
            custom_stop: None,
            metrics: Vec::new(),
            selection_metrics: Vec::new(),
            log_level: LogLevel::default(),
            trace_file: DEFAULT_TRACE_FILE,
            save_optim: DEFAULT_SAVE_OPTIM,
            mode: DEFAULT_RUN_MODE,
            csv_export: DEFAULT_CSV_EXPORT,
            champion_save_topology: DEFAULT_CHAMPION_SAVE_TOPOLOGY,
            run_name: None,
            champion_previous_load: DEFAULT_CHAMPION_PREVIOUS_LOAD,
            champion_previous_dir: None,
            mutation_probation_steps: DEFAULT_MUTATION_PROBATION_STEPS,
            crossover_probation_steps: DEFAULT_CROSSOVER_PROBATION_STEPS,
            init_probation_steps: DEFAULT_INIT_PROBATION_STEPS,
            recovery_retries: DEFAULT_RECOVERY_RETRIES,
            recovery_tolerance: DEFAULT_RECOVERY_TOLERANCE,
            recovery_same_seed: DEFAULT_RECOVERY_SAME_SEED,
        }
    }

    /// Resolve [`Self::selection_metrics`] labels to indices in `metrics`, in ranking order.
    pub fn selection_metric_terms(&self, metrics: &[Metric]) -> Vec<(usize, f32)> {
        self.selection_metrics
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
        cfg.crossover_share = snap.crossover_share;
        cfg.mutation_share = snap.mutation_share;
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
        // `""` = a header written before the knob, which averaged arithmetically.
        cfg.mean_type = MeanType::parse(&snap.mean_type).unwrap_or(DEFAULT_MEAN_TYPE);
        cfg.train_ticket = TrainTicket::parse(&snap.train_ticket).unwrap_or(cfg.train_ticket);
        cfg.csv_export = snap.csv_export;
        cfg.save_optim = snap.save_optim;
        cfg.run_name = snap.run_name.clone();
        cfg.selection_metrics = snap.selection_metrics.clone();
        cfg.train_merit_p_min = snap.train_merit_p_min;
        cfg.train_merit_p_max = snap.train_merit_p_max;
        cfg.train_merit_learn_only = snap.train_merit_learn_only;
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
        // Match the recorded spelling, so a renamed display label still resumes.
        let cull = snap.crossover_cull_policy.to_ascii_lowercase();
        cfg.crossover_cull_policy = if cull.contains("lowest_fitness_most_updates") {
            CrossCullPolicy::LowestFitnessMostUpdates
        } else if cull.contains("highest_ticket_chance") {
            CrossCullPolicy::HighestTicketChance
        } else if cull.contains("random") {
            CrossCullPolicy::Random
        } else {
            CrossCullPolicy::LowestFitness
        };
        let mcull = snap.mutation_cull_policy.to_ascii_lowercase();
        cfg.mutation_cull_policy = if mcull.contains("lowest_fitness_most_updates") {
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
            vec![snap
                .feature_split
                .clone()
                .unwrap_or_else(|| "none".to_string())]
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
            mode_specific: ModeConfig::Rl,
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
    pub fn set_pop_size(mut self, n: usize) -> Self {
        self.cfg.pop_size = n;
        self
    }
    /// Set steps between checkpoints — the durability tick (frontier states + EVERY live net's
    /// weights + `ckpt.json` + CSV flush) and the stream's eval-permutation era.
    pub fn set_run_checkpoint_every(mut self, n: usize) -> Self {
        self.cfg.checkpoint_every = n;
        self
    }
    /// Set whether each live net's optimizer state rides its checkpoint (`nets/<hash>.optim`), so a
    /// resume restores the Adam moments exactly. `false` = weights only. Default true.
    pub fn set_run_save_optim(mut self, enabled: bool) -> Self {
        self.cfg.save_optim = enabled;
        self
    }
    /// Set crossover turnover as a SHARE of the live population per clock (`0.05` = 5%), dealt as `round(share · pop)` rolls each clock. `0.0` = the channel OFF.
    pub fn set_crossover_share(mut self, share: f32) -> Self {
        self.cfg.crossover_share = if share.is_finite() {
            share.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self
    }
    /// Set mutation turnover as a SHARE of the live population per clock, with
    /// `set_crossover_share`'s bounds. `0.0` = the channel OFF.
    pub fn set_mutation_share(mut self, share: f32) -> Self {
        self.cfg.mutation_share = if share.is_finite() {
            share.clamp(0.0, 1.0)
        } else {
            0.0
        };
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
    /// Champion set size: the top-k live nets saved at stop. Minimum 1; a champion is a
    /// rank, not an identity, and it shields nothing from a cull.
    pub fn set_champion_count(mut self, n: usize) -> Self {
        self.cfg.champion_count = n.max(1);
        self
    }
    /// Lottery floor `p_min`: the worst net's chance of a ticket each clock.
    pub fn set_ticket_p_min(mut self, p_min: f32) -> Self {
        self.cfg.train_merit_p_min = if p_min.is_finite() {
            p_min.clamp(0.0, 1.0)
        } else {
            DEFAULT_TRAIN_MERIT_P_MIN
        };
        self
    }
    /// CEILING `p_max` on the lottery fraction: the best-graded net's share of clocks (default
    /// `1.0` = no ceiling).
    pub fn set_ticket_p_max(mut self, p_max: f32) -> Self {
        self.cfg.train_merit_p_max = if p_max.is_finite() {
            p_max.clamp(0.0, 1.0)
        } else {
            DEFAULT_TRAIN_MERIT_P_MAX
        };
        self
    }
    /// The lottery's LEGS (`TrainTicket`): which fit place earns the ticket, and which side of the updates field it multiplies in.
    pub fn set_train_ticket(mut self, legs: TrainTicket) -> Self {
        self.cfg.train_ticket = legs;
        self
    }
    /// LEARN-ONLY lottery (default `false`): `true` keeps every net ACTING and MEASURING every clock and gates only the weight update on the ticket, so a net's reading never…
    pub fn set_ticket_learn_only(mut self, yes: bool) -> Self {
        self.cfg.train_merit_learn_only = yes;
        self
    }
    /// POLICY-SCOPED FITNESS (default `false`): `true` builds the mean/spread/n ONLY from readings taken on the weights the net holds now, so the statistics describe one POLICY…
    pub fn set_fitness_policy_scoped(mut self, yes: bool) -> Self {
        self.cfg.policy_scoped_fitness = yes;
        self
    }
    /// ANTI-CHAMPION dump (default `false`): `true` writes `worst-<hash>-<fitness>.md` for the `champion_count` nets ranked by `spent` (`Fm⁻ × (1 − Um)`, the cull's own…
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
    pub fn set_fitness_smoothing_window(mut self, k: usize) -> Self {
        self.cfg.smoothing_window = k.max(1);
        self
    }
    /// How the window's readings are averaged into the ranking centre (`mean_type`).
    pub fn set_fitness_mean_type(mut self, mean_type: MeanType) -> Self {
        self.cfg.mean_type = mean_type;
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
    /// Sample each newborn's `input_reach` from `min..=max` (`0.0..=1.0`).
    ///
    /// `input_reach` is how much of the raw input the hidden layer may read directly: `1.0` sizes
    /// the Input node at one port per hidden node (every node may read the input, the historical
    /// wiring), lower values hand out fewer ports, so the sockets they cannot cover are fed by
    /// other hidden nodes — the net gets DEEPER instead of wider at its first layer. `1.0, 1.0`
    /// (the default) pins the historical wiring.
    pub fn set_topology_input_reach_range(mut self, min: f32, max: f32) -> Self {
        self.cfg.topology_options.min_input_reach = min.clamp(0.0, 1.0);
        self.cfg.topology_options.max_input_reach = max.clamp(0.0, 1.0);
        // The header's template must not keep a stale sampled value.
        self.cfg.topology_options.input_reach = min.clamp(0.0, 1.0);
        self
    }
    /// Keep at most ONE wire per (source node, target node) pair — `true` by default.
    ///
    /// The rule spans every kind: Input→hidden (where the Input port is Identity, so a second wire
    /// is the same signal), hidden→hidden, and hidden→Output. With it on, `finalize` will not pick
    /// a source that already feeds the target, `validate` rejects a blueprint that does, and a
    /// loaded graph is repaired by one seeded random cull. `false` restores the older wiring.
    pub fn set_topology_unique_source_per_target(mut self, on: bool) -> Self {
        self.cfg.topology_options.unique_source_per_target = on;
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
    /// Extra informative (non-ranking) metrics recorded in history.csv and engine.json.
    /// They affect selection ONLY when [`Self::set_selection_metric`] names them.
    pub fn set_run_metrics(mut self, metrics: Vec<Metric>) -> Self {
        self.cfg.metrics = metrics;
        self
    }

    /// Make an informative metric count toward the RANKING: `score = fitness centre + Σ weight × metric centre`.
    pub fn set_selection_metric(mut self, label: impl Into<String>, weight: f32) -> Self {
        let label = label.into();
        match self
            .cfg
            .selection_metrics
            .iter_mut()
            .find(|m| m.label == label)
        {
            Some(term) => term.weight = weight,
            None => self
                .cfg
                .selection_metrics
                .push(crate::utils::fitness::SelectionMetric { label, weight }),
        }
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
    /// Mutation novelty floor `d_min`: the minimum Δ a random immigrant must hold against the NEAREST blueprint the run already made. `0.0` turns the gate OFF entirely (no…
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
    /// Mutation probation steps: cull-immunity for a random immigrant's first `k` UPDATES — `U`, not clocks, so waiting does not spend it (see…
    pub fn set_mutation_probation_steps(mut self, k: usize) -> Self {
        self.cfg.mutation_probation_steps = k;
        self
    }
    /// Crossover probation steps: cull-immunity for a crossover child's first
    /// `k` UPDATES — `U`, not clocks (see [`RaceConfig::crossover_probation_steps`]). Default 0.
    pub fn set_crossover_probation_steps(mut self, k: usize) -> Self {
        self.cfg.crossover_probation_steps = k;
        self
    }
    /// INIT PROBATION (default `0` = off): the opening clocks with no culls and no births — evolution starts at clock `k`.
    pub fn set_pop_init_probation_steps(mut self, k: usize) -> Self {
        self.cfg.init_probation_steps = k;
        self
    }
    /// Recovery updates: how many EXTRA real updates a net may take in the same clock when the one it just took left its reading worse than it was (see…
    pub fn set_pop_recovery_retries(mut self, k: usize) -> Self {
        self.cfg.recovery_retries = k;
        self
    }
    /// Recovery tolerance: the FRACTION of the net's standing a reading may lose before the streak fires (`0.2` = a fifth is tolerated, more triggers).
    pub fn set_pop_recovery_tol(mut self, fraction: f32) -> Self {
        self.cfg.recovery_tolerance = if fraction.is_finite() {
            fraction.max(0.0)
        } else {
            0.0
        };
        self
    }
    /// Recovery retry draw: `true` (default) = a retry repeats the clock's OWN train draw, so the streak re-tests the update that fell; `false` = each retry walks one draw…
    pub fn set_pop_recovery_same_seed(mut self, same: bool) -> Self {
        self.cfg.recovery_same_seed = same;
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
    use super::{LogLevel, RaceConfig};

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

    /// The ranking has no risk knob: the window score is the family CENTRE, so two nets
    /// with the same centre tie here regardless of how volatile one of them was.
    #[test]
    fn the_ranking_score_is_the_centre_with_no_spread_charge() {
        let cfg = RaceConfig::defaults();
        let ran = crate::engine::smoothing::rolling_stats_with(
            &crate::engine::smoothing::RollingBuffer::from_values(
                5,
                &[40.0, 500.0, 40.0, 40.0, 40.0],
            ),
            cfg.mean_type,
        )
        .unwrap();
        let steady = crate::engine::smoothing::rolling_stats_with(
            &crate::engine::smoothing::RollingBuffer::from_values(
                5,
                &[40.0, 40.0, 40.0, 40.0, 40.0],
            ),
            cfg.mean_type,
        )
        .unwrap();
        assert!(
            steady.score() <= ran.score(),
            "the calm window is not penalised"
        );
        assert_eq!(ran.score(), ran.fam_center, "the score IS the centre");
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

/// The recorded spellings and the alias tables, pinned in ONE place: `strum::IntoStaticStr` derives `as_str`/`Display` from `serialize_all = "snake_case"`, while `parse`…
#[cfg(test)]
mod spelling_pins {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn mean_type_spellings_hold() {
        for (v, s) in [
            (MeanType::Arithmetic, "arithmetic"),
            (MeanType::SignedLog, "signed_log"),
            (MeanType::Median, "median"),
        ] {
            assert_eq!(v.as_str(), s);
            assert_eq!(v.to_string(), s);
            assert_eq!(MeanType::parse(s), Some(v));
            assert_eq!(MeanType::from_str(s).unwrap(), v);
        }
        assert_eq!(MeanType::parse("average"), Some(MeanType::Arithmetic));
        assert_eq!(MeanType::parse("signed-log"), Some(MeanType::SignedLog));
    }

    #[test]
    fn train_ticket_spellings_hold() {
        for (v, s) in TrainTicket::ALL.iter().copied().zip(TrainTicket::ALL_NAMES) {
            assert_eq!(v.as_str(), s);
            assert_eq!(v.to_string(), s);
            assert_eq!(TrainTicket::parse(s), Some(v));
            assert_eq!(TrainTicket::from_str(s).unwrap(), v);
        }
        assert_eq!(
            TrainTicket::parse("best_fewest"),
            Some(TrainTicket::MostFitFewestUpdates)
        );
        assert_eq!(
            TrainTicket::parse("least_fit_least_updates"),
            Some(TrainTicket::LeastFitFewestUpdates)
        );
    }

    #[test]
    fn cull_policy_spellings_hold() {
        for (v, s) in [
            (CrossCullPolicy::LowestFitness, "lowest_fitness"),
            (CrossCullPolicy::Random, "random"),
            (
                CrossCullPolicy::HighestTicketChance,
                "highest_ticket_chance",
            ),
            (
                CrossCullPolicy::LowestFitnessMostUpdates,
                "lowest_fitness_most_updates",
            ),
        ] {
            assert_eq!(v.to_string(), s);
            assert_eq!(CrossCullPolicy::from_str(s).unwrap(), v);
            assert_eq!(MutationCullPolicy::from_str(s).unwrap().to_string(), s);
        }
        assert!(CrossCullPolicy::from_str("nope")
            .unwrap_err()
            .contains("expected lowest_fitness"));
    }
}
