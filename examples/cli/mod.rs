//! Shared CLI surface for the examples.
//!
//! Two surfaces live here:
//!
//! - [`SmokeArgs`] — the TWO-FLAG surface the examples flatten (`--pop`,
//!   `--max-steps`): enough for a quick smoke test, nothing to read past.
//!   Every other knob is a `const` at the top of the example, visible in the
//!   source rather than behind a flag. Used by cartpole, categorical,
//!   continuous, custom_trainer and mnist (which adds its own `--data-dir`).
//! - [`EngineArgs`] / [`RlEngineArgs`] — the fuller engine surface
//!   (`--log-level`, `--seed`, `--resume`, the evolution knobs, …), kept for
//!   the one example whose runs are long enough to need it:
//!   `kaggle_kagiculture`.
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
//! `dead_code` is allowed module-wide: every includer uses a different subset
//! of this module, so an unused helper is expected rather than a smell.
#![allow(dead_code)]

use std::path::PathBuf;

use clap::Parser;
use gras::prelude::*;

/// The two-flag smoke surface the examples share: population size and the step
/// budget, nothing else.
///
/// `--pop 8 --max-steps 2` is all a quick smoke test needs. Every other knob
/// lives in the example's own consts and builder chain, where it is visible in
/// the source rather than behind another flag. An example with a genuine extra
/// input (mnist's dataset directory) declares that one flag itself.
#[derive(Parser, Debug)]
pub struct SmokeArgs {
    /// Population size (default: the example's own const).
    #[arg(long, value_name = "N")]
    pub pop: Option<usize>,
    /// Stop after this many engine steps (default: the example's own const).
    #[arg(long = "max-steps", value_name = "N")]
    pub max_steps: Option<usize>,
}

/// Turn on the engine logger at `level`.
///
/// The smoke surface deliberately has no `--log-level` — pass the example's own
/// const, exactly as a reader of the source would expect.
pub fn init_logger(level: LogLevel) -> bool {
    gras::engine::logging::init(level, None)
}

/// Parse an engine log level (`none` | `summ` | `minimal`) on the CLI.
///
/// Reuses the engine's own vocabulary (`LogLevel::parse`) instead of inventing
/// a second set of names: `--log-level summ` is exactly what `engine.json`
/// records, and [`Self::init_logger`] translates it into the console sink's
/// filter — typing `summ` straight into a filter string would be read as a
/// *module* name and mute everything.
fn parse_log_level(raw: &str) -> Result<LogLevel, String> {
    LogLevel::parse(raw).ok_or_else(|| {
        format!("unknown log level \"{raw}\" — expected one of: none, summ, minimal")
    })
}

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

    /// Log level: `none`, `summ`, or `minimal`.
    #[arg(long, value_name = "LEVEL", value_parser = parse_log_level)]
    pub log_level: Option<LogLevel>,

    /// Run name recorded in `engine.json` (informational).
    #[arg(long, value_name = "NAME")]
    pub run_name: Option<String>,

    /// Where results land (default: `results/<timestamp>`).
    #[arg(long, value_name = "DIR")]
    pub run_dir: Option<PathBuf>,

    /// Skip the post-race solo training of the elites.
    #[arg(long)]
    pub no_pruner: bool,

    /// Extra solo steps when the pruner runs.
    #[arg(long, value_name = "N")]
    pub pruner_steps: Option<usize>,

    /// Crossover gate strictness: `hard` (beat the population mean at EVERY
    /// checkpoint the child replays through) or `soft` (beat the mean of the
    /// checkpoint means — one aggregate bar, so an early dip can be survived).
    #[arg(long, value_name = "MODE")]
    pub crossover_gate: Option<String>,

    /// Who an admitted crossover child evicts: `worst` (default, merit-based)
    /// or `random` (keeps slots turning over). Mutation victims are always
    /// fitness-inverse and ignore this. (Crossover gate semantics: §2 of
    /// OPTIONS.md.)
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
    /// (default) = relay from step 0. See OPTIONS.md §8 for the relay
    /// itself, and how to run without it entirely.
    #[arg(long, value_name = "N")]
    pub grace_periods: Option<usize>,

    /// Extra parent-pairing attempts when a crossover draw finds no compatible
    /// parents (0 = give up immediately; a failed roll stays a no-op).
    #[arg(long, value_name = "N")]
    pub crossover_retries: Option<usize>,

    /// Which post-race pruner: `hard` (keep the elite, train it solo for
    /// --pruner-steps).
    #[arg(long, value_name = "METHOD")]
    pub pruner_method: Option<String>,

    /// Also write `worst-<hash>.md` + `.safetensors` at stop — the
    /// anti-champion the search avoided.
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

/// RL-only engine flags — the mode-specific twin of [`EngineArgs`], mirroring
/// the config split: `ModeConfig::Rl` knobs get their own clap struct so a
/// tabular example can't even PARSE `--fresh-immigrants` (clap rejects it at
/// the usage line instead of the old behavior: a runtime panic when the
/// RL-only setter hit the tabular arm).
#[derive(Parser, Debug, Clone, Default)]
pub struct RlEngineArgs {
    /// The shared engine surface every mode has.
    #[command(flatten)]
    pub engine: EngineArgs,

    /// Mutation catch-up toggle (RL only): OFF (default) = no handicap —
    /// the immigrant keeps fresh weights and trains from the current clock.
    /// ON = the immigrant replays the training stream first (comparable
    /// step-count on entry).
    #[arg(long)]
    pub fresh_immigrants: bool,
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
    /// Overlay the shared flags, then the RL-only ones.
    pub fn apply(&self, b: RaceConfigBuilder) -> RaceConfigBuilder {
        let b = self.engine.apply(b);
        if self.fresh_immigrants {
            b.set_mutation_catch_up(true)
        } else {
            b
        }
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
        if let Some(level) = self.log_level {
            b = b.set_run_log_level(level);
        }
        if let Some(name) = &self.run_name {
            b = b.set_run_name(name.clone());
        }
        if self.no_pruner {
            b = b.set_pruner_enabled(false);
        }
        if let Some(steps) = self.pruner_steps {
            b = b.set_pruner_steps(steps);
        }
        b = b.set_elite_freeze(self.freeze_elites);
        if let Some(mode) = &self.crossover_gate {
            b = b.set_crossover_gate(match mode.trim().to_ascii_lowercase().as_str() {
                "hard" => gras::engine::config::CrossoverGate::Hard,
                "soft" => gras::engine::config::CrossoverGate::Soft,
                other => {
                    eprintln!("--crossover-gate {other}: expected `hard` or `soft`");
                    std::process::exit(2);
                }
            });
        }
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
        if let Some(n) = self.crossover_retries {
            b = b.set_crossover_retries(n);
        }
        if let Some(method) = &self.pruner_method {
            b = b.set_pruner_method(match method.trim().to_ascii_lowercase().as_str() {
                "hard" => gras::engine::config::PopPrunerMethod::Hard,
                other => {
                    eprintln!("--pruner-method {other}: expected `hard`");
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
        gras::engine::logging::init(default_level, self.log_level.map(LogLevel::env_filter))
    }
}
