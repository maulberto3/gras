//! The engine-side CLI both examples run with: one `clap::Args` struct, flattened
//! into each example's `Cli` so a shared flag has ONE declaration and ONE home for
//! its default, plus the wiring both runs do identically (log sinks, the seed draw,
//! parent selection, the engine-knob chain). Flags whose default or meaning differ
//! per example stay inline there.
//!
//! Flags are grouped with `help_heading` in the order a run goes through them:
//! what the run IS → who gets paid → who leaves → how reads are ranked → how
//! children are made → what a newborn may be → what lands on disk. The examples
//! reuse the same heading strings, so their own flags land in those sections.

use std::path::PathBuf;

use clap::Args;
use gras::SelectionMethod;
use gras::prelude::{Direction, RaceSnapshot};
use gras::engine::config::{
    CrossCullPolicy, DEFAULT_TOURNAMENT_SIZE, LogLevel, MeanType, MutationCullPolicy,
    RaceConfigBuilder, TrainTicket,
};

/// Live nets in the race (default; `--pop` overrides).
pub const POP: usize = 20;

/// Default step budget (`--max-steps`): 500 clocks — a long race, ended early by the examples'
/// own bar stop and no-change stop long before the budget on a run that converges.
pub const RACE_STEPS: usize = 500;

/// Crossover turnover: the share of the live field replaced per clock, dealt as `round(share × pop)` rolls.
pub const CROSSOVER_TURNOVER: f32 = 0.025;

/// MUTATION turnover default, the mutation channel's own share of the field per clock.
pub const MUTATION_TURNOVER: f32 = 0.005;

/// Steps between population checkpoints — the era rotation and the CSV flush ride this.
pub const CHECKPOINT_EVERY: usize = 50;

/// Every engine knob the two examples agree on — same flag, same `#[arg]`, same default.
///
/// `Cli` derefs to this, so an example body keeps reading `cli.ticket_p_min` and never
/// has to know whether a knob came from here or from its own inline fields.
#[derive(Args, Debug)]
pub struct EngineArgs {
    // ── Run ──────────────────────────────────────────────────────────────────
    /// Live networks in the race (default: the `POP` const).
    #[arg(long, value_name = "N", default_value_t = POP, help_heading = "Run")]
    pub pop: usize,

    /// Stop after this many steps; on `--resume` it EXTENDS the recorded budget.
    #[arg(long, value_name = "N", default_value_t = RACE_STEPS, help_heading = "Run")]
    pub max_steps: usize,

    /// Where a NEW run writes (default: `results/<timestamp>`). Output only —
    /// continuing an existing run is `--resume`.
    #[arg(long, value_name = "DIR", help_heading = "Run")]
    pub run_dir: Option<PathBuf>,

    /// Continue the recorded run here: its seed wins over `--seed`, and `--max-steps` EXTENDS the recorded budget.
    #[arg(long, value_name = "DIR", help_heading = "Run")]
    pub resume: Option<PathBuf>,

    /// Log level (default: the `LOG_LEVEL` const). `none`/`summ`/`minimal` pick
    /// the shape; `debug`/`trace` add the engine's per-roll detail.
    #[arg(
        long,
        value_name = "LEVEL",
        value_parser = ["none", "summ", "minimal", "debug", "trace"],
        default_value = "summ",
        help_heading = "Run"
    )]
    pub log_level: String,

    /// Steps between checkpoints — weights, frontier states and the CSV flush.
    #[arg(long, value_name = "N", default_value_t = CHECKPOINT_EVERY, help_heading = "Run")]
    pub checkpoint_every: usize,

    // ── Tickets (the train-eval lottery) ─────────────────────────────────────
    /// Lottery FLOOR `p_min`: the chance the WORST-graded net is paid each clock (`p_min == p_max` pays all alike).
    #[arg(long, value_name = "P", default_value_t = 0.15, help_heading = "Tickets (lottery)")]
    pub ticket_p_min: f32,

    /// Lottery CEILING `p_max`: the chance the BEST-graded net is paid; everyone between is interpolated.
    #[arg(long, value_name = "P", default_value_t = 0.85, help_heading = "Tickets (lottery)")]
    pub ticket_p_max: f32,

    /// Lottery legs: which fit place pays the most and which updates side it multiplies in; a typo is a CLI error.
    #[arg(
        long,
        value_name = "LEGS",
        default_value_t = gras::engine::config::DEFAULT_TRAIN_TICKET,
        help_heading = "Tickets (lottery)"
    )]
    pub train_ticket: TrainTicket,

    /// LEARN-ONLY: a lost ticket costs the weight UPDATE, not the clock — the net still acts and is measured.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        // Bare `--learn-only` means ON; `--learn-only false` turns it off. A missing value
        // that spells the DEFAULT would make the bare flag a no-op.
        default_missing_value = "true",
        default_value_t = false,
        help_heading = "Tickets (lottery)"
    )]
    pub learn_only: bool,

    // ── Culling, shields and recovery ────────────────────────────────────────
    /// Crossover victim: `highest_ticket_chance` (the coin's favourite), `lowest_fitness`, `lowest_fitness_most_updates` (evidence) or `random`.
    #[arg(
        long,
        value_name = "POLICY",
        default_value_t = CrossCullPolicy::default(),
        help_heading = "Culling & probation"
    )]
    pub crossover_cull_policy: CrossCullPolicy,

    /// Mutation/immigrant victim: the same four arms as the crossover channel.
    #[arg(
        long,
        value_name = "POLICY",
        default_value_t = MutationCullPolicy::default(),
        help_heading = "Culling & probation"
    )]
    pub mutation_cull_policy: MutationCullPolicy,

    /// Opening clocks with NO culls, no births and no recovery streak: evolution starts at clock N.
    #[arg(long, value_name = "N", default_value_t = 0, help_heading = "Culling & probation")]
    pub init_probation: usize,

    /// Mutation probation: cull immunity for an immigrant's first N updates (`U`, not clocks).
    #[arg(long, value_name = "N", default_value_t = 0, help_heading = "Culling & probation")]
    pub mutation_probation: usize,

    /// Crossover probation: the same shield for a crossover child's first N updates.
    #[arg(long, value_name = "N", default_value_t = 0, help_heading = "Culling & probation")]
    pub crossover_probation: usize,

    /// Recovery updates: extra real updates inside one clock after a worse reading (`0` = off).
    #[arg(long, value_name = "N", default_value_t = 0, help_heading = "Culling & probation")]
    pub recovery_retries: usize,

    /// Recovery tolerance: the fraction of its standing a reading may lose before the streak fires.
    #[arg(long, value_name = "F", default_value_t = 0.0, help_heading = "Culling & probation")]
    pub recovery_tol: f32,

    /// Recovery draw: `true` retries the clock's own draw, `false` walks one draw further.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true,
        help_heading = "Culling & probation"
    )]
    pub recovery_same_seed: bool,

    // ── Ranking ──────────────────────────────────────────────────────────────
    /// Window stats scoped to the CURRENT weights: n counts only readings since the last real update.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true,
        help_heading = "Ranking"
    )]
    pub policy_scoped_fitness: bool,

    /// Ranking window K: how many readings are averaged (under `--policy-scoped-fitness`, of ONE policy).
    #[arg(
        long,
        value_name = "K",
        default_value_t = gras::engine::config::SMOOTHING_WINDOW,
        help_heading = "Ranking"
    )]
    pub smoothing_window: usize,

    /// Averaging family: `arithmetic`, `signed_log` (spike-damped) or `median` — the default, where no single reading moves it.
    #[arg(
        long,
        value_name = "FAMILY",
        default_value_t = MeanType::Median,
        help_heading = "Ranking"
    )]
    pub mean_type: MeanType,

    // ── Crossover ────────────────────────────────────────────────────────────
    /// Crossover turnover: the share of the live field replaced per clock (`0` switches the channel off).
    #[arg(
        long,
        value_name = "RATE",
        default_value_t = CROSSOVER_TURNOVER,
        help_heading = "Crossover"
    )]
    pub crossover_turnover: f32,

    /// Parent selection for the crossover draw: `roulette` (the default — fitness-proportionate,
    /// so the fitter a net the likelier it is drawn, with nothing fixed), `tournament`, or
    /// `elite` (deterministic best fitness, ties to the most-updated net).
    #[arg(
        long,
        value_name = "METHOD",
        value_parser = ["roulette", "tournament", "elite"],
        default_value = "roulette",
        help_heading = "Crossover"
    )]
    pub crossover_selection: String,

    /// Tournament width when `--crossover-selection tournament`.
    #[arg(
        long,
        value_name = "K",
        default_value_t = DEFAULT_TOURNAMENT_SIZE,
        help_heading = "Crossover"
    )]
    pub tournament_size: usize,

    /// The FITTER drawn parent becomes the crossover BASE (off: the first drawn parent is).
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true,
        help_heading = "Crossover"
    )]
    pub crossover_fitter_base: bool,

    /// Crossover operator pool, drawn per attempt: `one_point` and/or `uniform`; empty = both.
    #[arg(long, value_name = "LIST", value_delimiter = ',', help_heading = "Crossover")]
    pub crossover_ops_pool: Vec<String>,

    // ── Mutation ─────────────────────────────────────────────────────────────
    /// Mutation turnover: the mutation channel's own share, same contract as `--crossover-turnover`.
    #[arg(
        long,
        value_name = "RATE",
        default_value_t = MUTATION_TURNOVER,
        help_heading = "Mutation"
    )]
    pub mutation_turnover: f32,

    /// Mutation novelty floor Δ: a new immigrant must sit this far (Jaccard) from the nearest blueprint; `0` = off.
    #[arg(long, value_name = "DELTA", default_value_t = 0.3, help_heading = "Mutation")]
    pub mutation_novelty_min: f32,

    // ── Search space (what a newborn may be) ─────────────────────────────────
    // The MIN/MAX pairs of the node counts are declared in each example beside each other:
    // their upper bound differs per example (15 on MNIST, 20 on CartPole), so the pair has to
    // live where that default lives — a `--hidden-min` here and a `--hidden-max` there print
    // a dozen rows apart. What stays shared is what both examples agree on.
    /// Hidden width for a newborn: the lower bound of the sampled range.
    #[arg(long, value_name = "N", default_value_t = 16, help_heading = "Search space")]
    pub hidden_dim_min: usize,

    /// Hidden width for a newborn: the upper bound of the sampled range.
    #[arg(long, value_name = "N", default_value_t = 64, help_heading = "Search space")]
    pub hidden_dim_max: usize,

    /// Raw-input reach for a newborn, lower bound: the share of hidden nodes allowed their own
    /// wire straight from the Input node (`1.0` = one port each). Lower values force the rest of
    /// the net to be fed by other hidden nodes, which is what makes it DEEPER.
    #[arg(long, value_name = "P", default_value_t = 0.05, help_heading = "Search space")]
    pub input_reach_min: f32,

    /// Raw-input reach for a newborn, upper bound (`1.0` = every node may read the input).
    #[arg(long, value_name = "P", default_value_t = 0.95, help_heading = "Search space")]
    pub input_reach_max: f32,

    /// One wire per (source node → target node) pair, for every kind. A second wire from the same
    /// node is the same signal with a different per-port activation, so it inflates fan-in and
    /// params without adding information. `false` restores the older wiring.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true,
        help_heading = "Search space"
    )]
    pub unique_source_per_target: bool,

    /// Feature-split draw order for a newborn: `full`, `split_random`, `split_exclusive` or
    /// `split_chunks`; empty = all four. `full` gives a node every input feature, the other
    /// three hand it a partition (random disjoint masks, exclusive disjoint masks, or
    /// contiguous chunks) — so the split is drawn per net, not per run.
    #[arg(long, value_name = "LIST", value_delimiter = ',', help_heading = "Search space")]
    pub feature_split_pool: Vec<String>,

    /// Combine-op pool: `add`, `mean`, `max`, `min`, `mul`, `sub`, `div`; empty = every combine op.
    #[arg(long, value_name = "LIST", value_delimiter = ',', help_heading = "Search space")]
    pub combine_op_pool: Vec<String>,

    /// Activation pool, from: `identity`, `relu`, `gelu`, `silu`, `selu`, `tanh`, `sigmoid`,
    /// `mish`, `leaky_relu`, `elu`, `gelu_tanh`, `softplus`, `hardswish`, `hardsigmoid`, `sin`,
    /// `cos`, `softmax`, `log_softmax`; empty = every one of them.
    #[arg(long, value_name = "LIST", value_delimiter = ',', help_heading = "Search space")]
    pub activation_pool: Vec<String>,

    /// Standardize-op pool: `identity`, `layernorm`, `rmsnorm`, `instancenorm`, `groupnorm`;
    /// empty = the full default pool. (`instance_norm` / `group_norm` / `rms_norm` are
    /// accepted spellings of the same ops.)
    #[arg(long, value_name = "LIST", value_delimiter = ',', help_heading = "Search space")]
    pub standardize_pool: Vec<String>,

    // ── Elites & artifacts ───────────────────────────────────────────────────
    /// Elite count saved at stop (a rank, not a seat — a champion shields nothing from a cull).
    /// Unset ⇒ one per 10 live nets (`--pop / 10`, at least 1), resolved at runtime.
    // `default_value_t = POP / 10` froze the ratio at the compile-time `POP`, so `--pop 50` still
    // saved 2 elites. The default now comes from the `--pop` actually parsed.
    #[arg(long, value_name = "N", help_heading = "Elites & artifacts")]
    pub champion_count: Option<usize>,

    /// Seed the founders from the newest compatible run's elites.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        // Bare `--champion-previous-load` means ON (this is the flag an earlier hard-coded
        // `false` in `engine_config` made inert); `--champion-previous-load false` turns it off.
        default_missing_value = "true",
        default_value_t = false,
        help_heading = "Elites & artifacts"
    )]
    pub champion_previous_load: bool,

    /// Where that inheritance looks: a run dir or a dir of run dirs (unset: `results/`).
    #[arg(long, value_name = "DIR", help_heading = "Elites & artifacts")]
    pub champion_previous_dir: Option<PathBuf>,

    /// At stop, save each champion's topology markdown (`champion-<hash>-<fitness>.md`).
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true,
        help_heading = "Elites & artifacts"
    )]
    pub champion_save_topology: bool,

    /// At stop, write `worst-<hash>-<fitness>.md` for the `spent`-ranked nets, plus a `worst.md` index.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true,
        help_heading = "Elites & artifacts"
    )]
    pub worst_save_topology: bool,

    /// Write `history.csv` + `attempts.csv` (the lossless event logs).
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true,
        help_heading = "Elites & artifacts"
    )]
    pub csv_export: bool,

    /// Also write `<run_dir>/telemetry.jsonl`, one JSON record per event.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = false,
        help_heading = "Elites & artifacts"
    )]
    pub trace_file: bool,

    /// Write each net's optimizer state (Adam moments) beside its weights, so `--resume` restores them exactly.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true,
        help_heading = "Elites & artifacts"
    )]
    pub save_optim: bool,

    /// Also write every live net's weights at the terminal stop. OFF by default: the stop is the
    /// RECORD, the checkpoint is the resume artifact — and a stop written without weights says so
    /// in `ckpt.json` (`weights: false`), so `--resume` refuses it instead of loading stale state.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = gras::engine::config::DEFAULT_SAVE_WEIGHTS_AT_STOP,
        help_heading = "Elites & artifacts"
    )]
    pub save_weights_at_stop: bool,
}

/// Reporting shape both examples use when `--log-level` is not passed.
pub const LOG_LEVEL: LogLevel = LogLevel::Summ;

/// The examples' own test target: every `EngineArgs` field must reach the `RaceConfig`
/// `engine_config` builds. A flag that parses and is then dropped compiles cleanly, so
/// this is the only layer that can see it.
#[cfg(test)]
#[path = "tests.rs"]
mod tests;

/// Print `msg` as a usage error and leave with code 2 — the examples' one failure path.
pub fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("{msg}");
    std::process::exit(2)
}

/// Parse `--log-level` over [`LOG_LEVEL`] and install the run's log sinks — without
/// the sinks the console stays silent and a live run LOOKS hung.
pub fn init_logging(level_arg: &str) -> LogLevel {
    let (level, verbosity) = LogLevel::parse_verbose(level_arg).unwrap_or((LOG_LEVEL, None));
    gras::engine::reporting::logging::init(level, verbosity);
    level
}

/// The run seed: `--seed` when pinned, else a fresh one DRAWN and printed. Resolved
/// here rather than as a clap `default_value_t`: a random default would rewrite
/// `--help` on every launch and erase the difference between "the user chose this"
/// and "we drew it".
pub fn run_seed(pinned: Option<u64>) -> u64 {
    pinned.unwrap_or_else(|| {
        let drawn = fastrand::u64(..);
        println!("seed: {drawn} (drawn — pass `--seed {drawn}` to replay)");
        drawn
    })
}

/// `--crossover-selection` → the parent draw. The parser already rejected anything
/// outside the three arms, so this only narrows a known set.
fn selection(method: &str, tournament_size: usize) -> SelectionMethod {
    match method {
        "tournament" => SelectionMethod::Tournament { tournament_size },
        "elite" => SelectionMethod::Elite,
        _ => SelectionMethod::Roulette,
    }
}

/// A resume EXTENDS a budget, never shrinks it: `--max-steps` is a floor on a
/// recorded run.
pub fn extended_budget(recorded: Option<usize>, asked: usize) -> usize {
    recorded.unwrap_or(asked).max(asked)
}

/// The BAR stop: has the field's best window-centre reached `bar`?
///
/// Deliberately direction-AGNOSTIC: the comparison goes through the snapshot's own
/// [`Direction`], so the number reads "at least this good" on a maximize metric
/// (accuracy, survival) and "at most this bad" on a minimize one (a loss). A hard-coded
/// `>=` is wrong twice over on a minimizing fitness — it never fires, and a bar that sits
/// below every reading fires on clock 0.
pub fn bar_reached(bar: f32, snap: &RaceSnapshot) -> bool {
    !snap.direction.is_better(bar, snap.best_smoothed_fitness)
}

/// The plateau target that is ALWAYS reached in `direction` — the no-change stop's "no bar".
///
/// `Plateau` only arms once the best score is at least as good as its target, so parking the
/// target on the metric's hopeless end (`-inf` when higher is better, `+inf` when lower is)
/// opens that gate for every clock and leaves the patience window as the only criterion.
/// A fixed `0.0` would do the same for accuracy but silently never arm on a loss.
pub fn no_bar(direction: Direction) -> f32 {
    match direction {
        Direction::Maximize => f32::NEG_INFINITY,
        Direction::Minimize => f32::INFINITY,
    }
}

/// Every engine knob the two examples agree on, wired in ONE chain. The caller passes
/// its own front door (`rl_race_config_builder` / `TabularRaceConfig::builder()`) and
/// chains its own half on top: the flags whose DEFAULT is the example's identity
/// (`--run-name`, the upper search bounds, `--drift`, `--dropout`,
/// `--mutation-novelty-draws`), the stop bar, and the input/output widths.
pub fn engine_config(
    builder: RaceConfigBuilder,
    engine: &EngineArgs,
    log_level: LogLevel,
) -> RaceConfigBuilder {
    let mut builder = builder
        // SETUP
        .set_run_log_level(log_level)
        .set_run_checkpoint_every(engine.checkpoint_every)
        .set_run_save_optim(engine.save_optim)
        .set_run_save_weights_at_stop(engine.save_weights_at_stop)
        .set_run_csv_export(engine.csv_export)
        .set_run_trace_file(engine.trace_file)
        // POP
        .set_pop_size(engine.pop)
        .set_pop_init_probation_steps(engine.init_probation)
        .set_pop_recovery_retries(engine.recovery_retries)
        .set_pop_recovery_tol(engine.recovery_tol)
        .set_pop_recovery_same_seed(engine.recovery_same_seed)
        // FITNESS
        .set_fitness_smoothing_window(engine.smoothing_window)
        .set_fitness_policy_scoped(engine.policy_scoped_fitness)
        .set_fitness_mean_type(engine.mean_type)
        // TRAIN
        .set_ticket_learn_only(engine.learn_only)
        .set_ticket_p_min(engine.ticket_p_min)
        .set_ticket_p_max(engine.ticket_p_max)
        .set_train_ticket(engine.train_ticket)
        // STOP — the budget; the solved bar is the example's own `set_stop_custom`.
        .set_stop_max_steps(engine.max_steps)
        // CROSSOVER
        .set_crossover_share(engine.crossover_turnover)
        .set_crossover_cull_policy(engine.crossover_cull_policy)
        .set_crossover_selection(selection(
            &engine.crossover_selection,
            engine.tournament_size,
        ))
        .set_crossover_fitter_base(engine.crossover_fitter_base)
        .set_crossover_ops_pool(engine.crossover_ops_pool.clone())
        .set_crossover_probation_steps(engine.crossover_probation)
        // MUTATION — its own turnover knob.
        .set_mutation_share(engine.mutation_turnover)
        .set_mutation_cull_policy(engine.mutation_cull_policy)
        .set_mutation_novelty_min(engine.mutation_novelty_min)
        .set_mutation_probation_steps(engine.mutation_probation)
        // TOPOLOGY — the search space; the node-count bounds are the example's own (they sit
        // beside their upper bound in its `Cli`), the widths and pools are shared.
        .set_topology_hidden_dim_range(engine.hidden_dim_min, engine.hidden_dim_max)
        .set_topology_input_reach_range(engine.input_reach_min, engine.input_reach_max)
        .set_topology_unique_source_per_target(engine.unique_source_per_target)
        // The four POOLS are declared here, so they are wired here — one place, both
        // examples, and `tests.rs` can prove every declared flag reaches the config.
        .set_topology_feature_split_pool(engine.feature_split_pool.clone())
        .set_topology_combine_op_pool(engine.combine_op_pool.clone())
        .set_topology_activation_pool(engine.activation_pool.clone())
        .set_topology_standardize_op_pool(engine.standardize_pool.clone())
        // ELITES — inheritance is the example's own flag: `0` keeps the all-random founders.
        .set_champion_previous_load(engine.champion_previous_load)
        .set_champion_count(engine.champion_count.unwrap_or((engine.pop / 10).max(1)))
        .set_champion_save_topology(engine.champion_save_topology)
        .set_worst_save_topology(engine.worst_save_topology);

    if let Some(dir) = &engine.champion_previous_dir {
        builder = builder.set_champion_previous_dir(dir.clone());
    }
    builder
}
