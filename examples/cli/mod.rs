//! Shared CLI surface for the two long-running examples.
//!
//! [`EngineArgs`] / [`RlEngineArgs`] carry the fuller engine surface
//! (`--log-level`, `--seed`, `--risk-aversion`, `--train-merit`,
//! `--founders-from`, the evolution knobs, …) for `kaggle_kagiculture` and
//! `mnist`. `cartpole` declares its own `Cli` struct and does not include this
//! module, and `print_topology` takes a run dir + hash only.
//!
//! ```text
//! cargo run --release --example cartpole -- --pop 32 --max-steps 20
//! cargo run --release --example mnist    -- --pop 50 --max-steps 15
//! ```
//!
//! Precedence is simple and explicit: **a flag wins; otherwise the example's
//! own `const`** is used (the const panel at the top of each example stays the
//! documented default). `--help` lists everything the binary accepts.
//!
//! This directory has no `main.rs`, so Cargo does not treat it as an example
//! binary — examples pull it in with `#[path = "cli/mod.rs"] mod cli;`.
//!
//! `dead_code` is allowed module-wide: `mnist` is tabular and `kaggle` is RL, so
//! each includer leaves the other mode's wrapper unused. That is expected.
#![allow(dead_code)]

use std::path::PathBuf;

use clap::Parser;
use gras::prelude::*;

/// The engine-level flags every example shares.
#[derive(Parser, Debug, Clone, Default)]
pub struct EngineArgs {
    /// Live networks in the race (population size).
    #[arg(long, value_name = "N")]
    pub pop: Option<usize>,

    /// Stop after this many steps.
    #[arg(long, value_name = "N")]
    pub max_steps: Option<usize>,

    /// RNG seed. Omit for a random seed (recorded in `engine.json`).
    #[arg(long, value_name = "U64")]
    pub seed: Option<u64>,

    /// Elites kept safe from culls AND exported at stop.
    #[arg(long, value_name = "N")]
    pub elite_count: Option<usize>,

    /// FIT risk aversion λ: window-σs a net pays for its own spread before it is
    /// ranked (0 = rank on the plain window mean). Replay-relevant.
    #[arg(long, value_name = "LAMBDA", visible_alias = "risk-based")]
    pub risk_aversion: Option<f32>,

    /// Train-eval lottery strength Ω: `0` (default) = off, every net trains;
    /// `1` = the full merit tilt; between = that fraction of the withholding.
    /// Replay-relevant — a resume must use the recorded value.
    #[arg(long, value_name = "OMEGA")]
    pub train_merit: Option<f32>,

    /// Seed the founder batch from the newest compatible previous experiment's
    /// elites instead of drawing fresh topologies. Bare flag = look in
    /// `results/`; with a value, that run dir (or a dir of run dirs).
    #[arg(long, value_name = "DIR", num_args = 0..=1, default_missing_value = "results")]
    pub founders_from: Option<String>,

    /// Steps between crossover-gate checkpoint bars.
    #[arg(long, value_name = "N")]
    pub checkpoint_every: Option<usize>,

    /// Probability a crossover roll attempts a child.
    #[arg(long, value_name = "P")]
    pub crossover_prob: Option<f32>,

    /// Probability a mutation roll replaces a net.
    #[arg(long, value_name = "P")]
    pub mutate_prob: Option<f32>,

    /// Crossover rolls per step.
    #[arg(long, value_name = "N")]
    pub crossover_rolls: Option<usize>,

    /// Mutation rolls per step.
    #[arg(long, value_name = "N")]
    pub mutate_rolls: Option<usize>,

    /// Log level: `none`, `summ`, `minimal`, or `debug`/`trace` (which keep the
    /// `summ` shape and additionally let the engine's per-roll detail through).
    /// The same vocabulary `engine.json` records, so a flag and a persisted run
    /// agree on what `summ` means.
    #[arg(long, value_name = "LEVEL")]
    pub log_level: Option<String>,

    /// Run name recorded in `engine.json` (informational).
    #[arg(long, value_name = "NAME")]
    pub run_name: Option<String>,

    /// Where results land (default: `results/<timestamp>`).
    #[arg(long, value_name = "DIR")]
    pub run_dir: Option<PathBuf>,

    /// Who an admitted crossover child evicts: `worst` (default, merit-based)
    /// or `random` (keeps slots turning over). Mutation victims are always
    /// fitness-inverse and ignore this.
    #[arg(long, value_name = "POLICY")]
    pub crossover_cull_policy: Option<String>,

    /// Mutation victim policy for the mutation/immigrant channel: `inverse`
    /// (default, fitness-inverse roulette), `worst` (deterministic worst by
    /// smoothed fitness), or `random` (uniform, keeps slots turning over).
    /// Only fires when a mutation roll fires (see `--mutate-prob`).
    #[arg(long, value_name = "POLICY")]
    pub mutation_cull_policy: Option<String>,

    /// Warm-up before the decision-lag relay engages (RL only): for the
    /// first N engine steps every net just trains — no relay cycles. 0
    /// (default) = relay from step 0. See `DecisionLagTrainer` for the relay
    /// itself, and how to run without it entirely.
    #[arg(long, value_name = "N")]
    pub grace_periods: Option<usize>,

    /// Also write `worst-<hash>.md` + `.safetensors` at stop — the
    /// anti-champions the search avoided. One per elite_count, so both ends of
    /// the ranking keep the same number of nets.
    #[arg(long)]
    pub worst_save: bool,

    /// Skip writing `elite-<hash>.md` + `.safetensors` at stop.
    #[arg(long)]
    pub no_elite_save: bool,

    /// Anti-devolution A: top-elite_count nets act-and-measure without
    /// weight updates (default on). Pass --no-freeze-elites to let elites
    /// keep training.
    #[arg(long = "no-freeze-elites", action = clap::ArgAction::SetFalse)]
    pub freeze_elites: bool,

    /// How often buffered history.csv rows hit disk: `checkpoint` (default,
    /// every `--checkpoint-every` steps and at stop — a `kill -9` loses at
    /// most that many steps of history) or `each` (every step, zero-loss
    /// history at the cost of a file write per step).
    #[arg(long, value_name = "MODE")]
    pub history_flush: Option<String>,
}

/// RL-only engine flags — the mode-specific twin of [`EngineArgs`]. No RL-only
/// flags exist today; the wrapper is kept so the mode split has one home.
#[derive(Parser, Debug, Clone, Default)]
pub struct RlEngineArgs {
    /// The shared engine surface every mode has.
    #[command(flatten)]
    pub engine: EngineArgs,
}

impl std::ops::Deref for RlEngineArgs {
    type Target = EngineArgs;
    fn deref(&self) -> &EngineArgs {
        &self.engine
    }
}

// Each example binary compiles this module with only ITS flags reachable —
// the RL-only methods are dead code in the tabular examples (and vice versa
// would be true of any tabular-only wrapper). Not a smell: the module is the
// union surface of all examples.
#[allow(dead_code)]
impl RlEngineArgs {
    /// Overlay the shared flags (the RL-only set is empty today).
    pub fn apply(&self, b: RaceConfigBuilder) -> RaceConfigBuilder {
        self.engine.apply(b)
    }

    /// Delegate: the seed for `RunSpec` (see [`EngineArgs::seed_or`]).
    pub fn seed_or(&self, default: Option<u64>) -> Option<u64> {
        self.engine.seed_or(default)
    }

    /// Delegate: the run dir for `RunSpec` (see [`EngineArgs::run_dir_or`]).
    pub fn run_dir_or(&self, default: Option<PathBuf>) -> Option<PathBuf> {
        self.engine.run_dir_or(default)
    }

    /// Delegate: logger init (see [`EngineArgs::init_logger`]).
    pub fn init_logger(&self, default_level: LogLevel) -> bool {
        self.engine.init_logger(default_level)
    }
}

impl EngineArgs {
    /// Overlay whatever the user passed onto a `RaceConfig` builder. Anything
    /// left `None` keeps the example's own default (set by its const panel).
    pub fn apply(&self, mut b: RaceConfigBuilder) -> RaceConfigBuilder {
        if let Some(pop) = self.pop {
            b = b.set_run_pop_size(pop);
        }
        if let Some(steps) = self.max_steps {
            b = b.set_stop_max_steps(steps);
        }
        if let Some(k) = self.elite_count {
            b = b.set_elite_count(k);
        }
        if let Some(n) = self.checkpoint_every {
            b = b.set_run_checkpoint_every(n);
        }
        if let Some(lambda) = self.risk_aversion {
            b = b.set_run_risk_aversion(lambda);
        }
        if let Some(omega) = self.train_merit {
            b = b.set_run_train_merit(omega);
        }
        if let Some(dir) = &self.founders_from {
            b = b
                .set_elite_previous_load(true)
                .set_elite_previous_dir(dir.clone());
        }
            // `--history-flush` is a MODE, not a bool: `each` flushes every clock what
            // the engine's flag asks for, `checkpoint` (and anything unset) keeps the
            // cadence the checkpoint tick already gives. Anything else is a typo, not a
            // silent fallback — the run must not quietly pick a durability it did not
            // ask for.
        if let Some(mode) = self.history_flush.as_deref() {
            match mode.trim().to_ascii_lowercase().as_str() {
                "each" => b = b.set_run_history_flush_each(true),
                "checkpoint" => b = b.set_run_history_flush_each(false),
                other => {
                    eprintln!("--history-flush {other}: expected `each` or `checkpoint`");
                    std::process::exit(2);
                }
            }
        }
        if let Some(p) = self.crossover_prob {
            b = b.set_crossover_prob(p);
        }
        if let Some(p) = self.mutate_prob {
            b = b.set_mutate_prob(p);
        }
        if let Some(n) = self.crossover_rolls {
            b = b.set_crossover_rolls(n);
        }
        if let Some(n) = self.mutate_rolls {
            b = b.set_mutate_rolls(n);
        }
        if let Some(raw) = &self.log_level {
            let level = LogLevel::parse(raw)
                .or_else(|| LogLevel::parse_verbose(raw).map(|(l, _)| l))
                .unwrap_or_else(|| {
                    eprintln!(
                        "--log-level {raw}: expected `none`, `summ`, `minimal`, `debug` or `trace`"
                    );
                    std::process::exit(2);
                });
            b = b.set_run_log_level(level);
        }
        if let Some(name) = &self.run_name {
            b = b.set_run_name(name.clone());
        }
        b = b.set_elite_freeze(self.freeze_elites);
        if let Some(policy) = &self.crossover_cull_policy {
            b = b.set_crossover_cull_policy(match policy.trim().to_ascii_lowercase().as_str() {
                "worst" => gras::engine::config::CrossCullPolicy::Worst,
                "random" => gras::engine::config::CrossCullPolicy::Random,
                other => {
                    eprintln!("--crossover-cull-policy {other}: expected `worst` or `random`");
                    std::process::exit(2);
                }
            });
        }
        if let Some(policy) = &self.mutation_cull_policy {
            b = b.set_mutation_cull_policy(match policy.trim().to_ascii_lowercase().as_str() {
                "inverse" | "inverse-fitness" => {
                    gras::engine::config::MutationCullPolicy::InverseFitness
                }
                "worst" => gras::engine::config::MutationCullPolicy::Worst,
                "random" => gras::engine::config::MutationCullPolicy::Random,
                other => {
                    eprintln!(
                        "--mutation-cull-policy {other}: expected `inverse`, `worst`, or `random`"
                    );
                    std::process::exit(2);
                }
            });
        }
        if self.worst_save {
            b = b
                .set_worst_save_topology(true)
                .set_worst_save_safetensors(true);
        }
        if self.no_elite_save {
            b = b
                .set_elite_save_topology(false)
                .set_elite_save_safetensors(false);
        }
        b
    }

    /// The seed to hand to `RunSpec`, falling back to the example's default.
    pub fn seed_or(&self, default: Option<u64>) -> Option<u64> {
        self.seed.or(default)
    }

    /// The run directory to hand to `RunSpec`, falling back to the example's
    /// default (`None` = the engine's `results/<timestamp>`).
    pub fn run_dir_or(&self, default: Option<PathBuf>) -> Option<PathBuf> {
        self.run_dir.clone().or(default)
    }

    /// Install the run's log sinks: the console lines, the `Minimal` frame, and
    /// the telemetry file when the run sets `set_run_trace_file(true)` — the
    /// ENGINE settles that one, since only it knows the run dir.
    ///
    /// `--log-level` overrides `default_level`; RUST_LOG still wins over both,
    /// so power users keep the fine-grained `module=level` filtering.
    pub fn init_logger(&self, default_level: LogLevel) -> bool {
        let (level, verbosity) = self
            .log_level
            .as_deref()
            .and_then(LogLevel::parse_verbose)
            .unwrap_or((default_level, None));
        gras::engine::logging::init(level, verbosity)
    }
}
