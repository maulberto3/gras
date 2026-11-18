//! The engine-side CLI both examples run with: one `clap::Args` struct, flattened
//! into each example's `Cli` so a shared flag has ONE declaration and ONE home for
//! its default. Flags whose default or meaning differ per example stay inline there.

use std::path::PathBuf;

use clap::Args;
use gras::engine::config::{CrossCullPolicy, MeanType, MutationCullPolicy, TrainTicket};

/// Live nets in the race (default; `--pop` overrides).
pub const POP: usize = 20;

/// Steps a race runs by default — both examples run 20 and pass `--max-steps` to go further.
pub const RACE_STEPS: usize = 20;

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
    /// Live networks in the race (default: the `POP` const).
    #[arg(long, value_name = "N", default_value_t = POP)]
    pub pop: usize,

    /// Stop after this many steps; on `--resume` it EXTENDS the recorded budget.
    #[arg(long, value_name = "N", default_value_t = RACE_STEPS)]
    pub max_steps: usize,

    /// Lottery FLOOR `p_min`: the chance the WORST-graded net is paid each clock (`p_min == p_max` pays all alike).
    #[arg(long, value_name = "P", default_value_t = 0.25)]
    pub ticket_p_min: f32,
    /// Lottery CEILING `p_max`: the chance the BEST-graded net is paid; everyone between is interpolated.
    #[arg(long, value_name = "P", default_value_t = 0.75)]
    pub ticket_p_max: f32,
    /// LEARN-ONLY: a lost ticket costs the weight UPDATE, not the clock — the net still acts and is measured.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = false
    )]
    pub learn_only: bool,
    /// Crossover victim: `highest_ticket_chance` (the coin's favourite), `lowest_fitness`, `lowest_fitness_most_updates` (evidence) or `random`.
    #[arg(long, value_name = "POLICY", default_value_t = CrossCullPolicy::default())]
    pub crossover_cull_policy: CrossCullPolicy,
    /// Mutation/immigrant victim: the same four arms as the crossover channel.
    #[arg(long, value_name = "POLICY", default_value_t = MutationCullPolicy::default())]
    pub mutation_cull_policy: MutationCullPolicy,
    /// Opening clocks with NO culls, no births and no recovery streak: evolution starts at clock N.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub init_probation: usize,
    /// Window stats scoped to the CURRENT weights: n counts only readings since the last real update.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    pub policy_scoped_fitness: bool,
    /// Ranking window K: how many readings are averaged (under `--policy-scoped-fitness`, of ONE policy).
    #[arg(long, value_name = "K", default_value_t = gras::engine::config::SMOOTHING_WINDOW)]
    pub smoothing_window: usize,
    /// Averaging family: `arithmetic`, `signed_log` (spike-damped) or `median` — the default, where no single reading moves it.
    #[arg(long, value_name = "FAMILY", default_value_t = MeanType::Median)]
    pub mean_type: MeanType,
    /// At stop, write `worst-<hash>-<fitness>.md` for the `spent`-ranked nets, plus a `worst.md` index.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    pub worst_save_topology: bool,
    /// Lottery legs: which fit place pays the most and which updates side it multiplies in; a typo is a CLI error.
    #[arg(
        long,
        value_name = "LEGS",
        default_value_t = gras::engine::config::DEFAULT_TRAIN_TICKET
    )]
    pub train_ticket: TrainTicket,
    /// Crossover turnover: the share of the live field replaced per clock (`0` switches the channel off).
    #[arg(long, value_name = "RATE", default_value_t = CROSSOVER_TURNOVER)]
    pub crossover_turnover: f32,
    /// Mutation turnover: the mutation channel's own share, same contract as `--crossover-turnover`.
    #[arg(long, value_name = "RATE", default_value_t = MUTATION_TURNOVER)]
    pub mutation_turnover: f32,
    /// Mutation novelty floor Δ: a new immigrant must sit this far (Jaccard) from the nearest blueprint; `0` = off.
    #[arg(long, value_name = "DELTA", default_value_t = 0.3)]
    pub mutation_novelty_min: f32,
    /// Log level (default: the `LOG_LEVEL` const). `none`/`summ`/`minimal` pick
    /// the shape; `debug`/`trace` add the engine's per-roll detail.
    #[arg(
        long,
        value_name = "LEVEL",
        value_parser = ["none", "summ", "minimal", "debug", "trace"],
        default_value = "summ"
    )]
    pub log_level: String,
    /// Continue the recorded run here: its seed wins over `--seed`, and `--max-steps` EXTENDS the recorded budget.
    #[arg(long, value_name = "DIR")]
    pub resume: Option<PathBuf>,
    /// Steps between checkpoints — weights, frontier states and the CSV flush.
    #[arg(long, value_name = "N", default_value_t = CHECKPOINT_EVERY)]
    pub checkpoint_every: usize,
    /// Write each net's optimizer state (Adam moments) beside its weights, so `--resume` restores them exactly.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    pub save_optim: bool,
    /// Mutation probation: cull immunity for an immigrant's first N updates (`U`, not clocks).
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub mutation_probation: usize,
    /// Crossover probation: the same shield for a crossover child's first N updates.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub crossover_probation: usize,
    /// Recovery updates: extra real updates inside one clock after a worse reading (`0` = off).
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub recovery_retries: usize,
    /// Recovery tolerance: the fraction of its standing a reading may lose before the streak fires.
    #[arg(long, value_name = "F", default_value_t = 0.0)]
    pub recovery_tol: f32,
    /// Recovery draw: `true` retries the clock's own draw, `false` walks one draw further.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    pub recovery_same_seed: bool,
    /// Elite count saved at stop (a rank, not a seat — a champion shields nothing from a cull).
    #[arg(long, value_name = "N", default_value_t = POP / 10)]
    pub champion_count: usize,
    /// At stop, save each champion's topology markdown (`champion-<hash>-<fitness>.md`).
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    pub champion_save_topology: bool,
    /// Seed the founders from the newest compatible run's elites.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    pub champion_previous_load: bool,
    /// Where that inheritance looks: a run dir or a dir of run dirs (unset: `results/`).
    #[arg(long, value_name = "DIR")]
    pub champion_previous_dir: Option<PathBuf>,
    /// Write `history.csv` + `attempts.csv` (the lossless event logs).
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    pub csv_export: bool,
    /// Also write `<run_dir>/telemetry.jsonl`, one JSON record per event.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = false
    )]
    pub trace_file: bool,
    /// Parent selection for the crossover draw: `tournament` or `roulette`.
    #[arg(
        long,
        value_name = "METHOD",
        value_parser = ["tournament", "roulette"],
        default_value = "tournament"
    )]
    pub crossover_selection: String,
    /// Tournament width when `--crossover-selection tournament`.
    #[arg(long, value_name = "K", default_value_t = 2)]
    pub tournament_size: usize,
    /// The FITTER drawn parent becomes the crossover BASE (off: the first drawn parent is).
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    pub crossover_fitter_base: bool,
    /// Crossover operator pool, drawn per attempt; empty = both.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    pub crossover_ops_pool: Vec<String>,
    /// Feature-split draw order for a newborn (`none`,`random`,`exclusive`); empty = all three.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    pub feature_split_pool: Vec<String>,
    /// Combine-op pool (`add`,`mean`,`max`,…); empty = every combine op.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    pub combine_op_pool: Vec<String>,
    /// Activation pool (`relu`,`gelu`,…); empty = the full default pool.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    pub activation_pool: Vec<String>,
    /// Standardize-op pool (`none`,`layer`,…); empty = the full default pool.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    pub standardize_pool: Vec<String>,
    /// Hidden nodes per net: the lower sampling bound.
    #[arg(long, value_name = "N", default_value_t = 2)]
    pub hidden_min: usize,
    /// Inputs per node: the lower sampling bound.
    #[arg(long, value_name = "N", default_value_t = 2)]
    pub inputs_min: usize,
    /// Outputs per node: the lower sampling bound.
    #[arg(long, value_name = "N", default_value_t = 2)]
    pub outputs_min: usize,
    /// Hidden width for a newborn: the lower bound of the sampled range.
    #[arg(long, value_name = "N", default_value_t = 16)]
    pub hidden_dim_min: usize,
    /// Hidden width for a newborn: the upper bound of the sampled range.
    #[arg(long, value_name = "N", default_value_t = 64)]
    pub hidden_dim_max: usize,
    /// Where a NEW run writes (default: `results/<timestamp>`). Output only —
    /// continuing an existing run is `--resume`.
    #[arg(long, value_name = "DIR")]
    pub run_dir: Option<PathBuf>,
}
