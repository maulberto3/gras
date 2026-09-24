//! Shared CLI surface for the examples — the knobs that every example has.
//!
//! Every example flattens [`EngineArgs`] into its own `clap` parser, so the
//! same engine-level flags work everywhere:
//!
//! ```text
//! cargo run --release --example cartpole -- --pop 32 --max-steps 20 --log-level minimal
//! cargo run --release --example mnist   -- --pop 50 --max-target-fitness 0.9 --seed 7
//! ```
//!
//! Precedence is simple and explicit: **a flag wins; otherwise the example's
//! own `const`** is used (the const panel at the top of each example stays the
//! documented default). `--help` lists everything the binary accepts.
//!
//! This directory has no `main.rs`, so Cargo does not treat it as an example
//! binary — examples pull it in with `#[path = "cli/mod.rs"] mod cli;`.

use std::path::PathBuf;

use clap::Parser;
use gras::engine::config::{LogLevel, RaceConfigBuilder};

/// Parse an engine log level (`none` | `summ` | `minimal`) on the CLI.
///
/// Reuses the engine's own vocabulary (`LogLevel::parse`) instead of inventing
/// a second set of names: `--log-level summ` is exactly what `engine.json`
/// records, and it is translated to an `env_logger` filter by
/// [`Self::init_logger`] — typing `summ` straight into `env_logger` would be
/// read as a *module* name and mute everything.
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

    /// Stop after this many steps. Mutually exclusive with
    /// `--max-target-fitness` (exactly one stop criterion at a time).
    #[arg(long, value_name = "N", conflicts_with = "max_target_fitness")]
    pub max_steps: Option<usize>,

    /// Stop once the best smoothed fitness crosses this value.
    #[arg(long, value_name = "FITNESS")]
    pub max_target_fitness: Option<f32>,

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
    /// fitness-inverse and ignore this.
    #[arg(long, value_name = "POLICY")]
    pub crossover_cull_policy: Option<String>,

    /// Mutation victim policy for the mutation/immigrant channel: `inverse`
    /// (default, fitness-inverse roulette), `worst` (deterministic worst by
    /// smoothed fitness), or `random` (uniform, keeps slots turning over).
    #[arg(long, value_name = "POLICY")]
    pub mutation_cull_policy: Option<String>,

    /// Run-level grace period: the number of engine steps every net trains
    /// normally, with no evolution and no relay (no `DecisionLagTrainer`
    /// wrapper), before the full trainer scheme takes over. `0` (default)
    /// = no grace, the runner uses the plain trainer. Tabular and RL both
    /// honour this; use a runner-level `*Trainer` wrapper (e.g. in a runner
    /// helper) to apply it. See `GracePeriodTrainer`.
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

    /// Anti-devolution A: top-elite_count nets skip the trainer call (their
    /// weights never change — a bad training step can't erase the best skill).
    #[arg(long)]
    pub freeze_elites: bool,

    /// Run-level grace period: the number of engine steps every net trains
    /// normally, with no evolution and no relay (no `DecisionLagTrainer`
    /// wrapper), before the full trainer scheme takes over. `0` (default)
    /// = no grace, the runner uses the plain trainer / whatever wrapper the
    /// Anti-devolution D: demote nets whose smoothed fitness falls below
    /// their entry floor × this tolerance (0.7 = 30% collapse triggers).
    #[arg(long, value_name = "TOL")]
    pub regression_tol: Option<f32>,

    /// RL-only: mutation immigrants skip the catch-up replay and train from
    /// the current clock (a newborn earning its seat from birth). Errors on
    /// a tabular run at engine construction.
    #[arg(long)]
    pub fresh_immigrants: bool,
}

impl EngineArgs {
    /// Overlay whatever the user passed onto a `RaceConfig` builder. Anything
    /// left `None` keeps the example's own default (set by its const panel).
    pub fn apply(&self, mut b: RaceConfigBuilder) -> RaceConfigBuilder {
        if let Some(pop) = self.pop {
            b = b.set_pop_size(pop);
        }
        if let Some(steps) = self.max_steps {
            b = b.set_stop_max_steps(steps);
        }
        if let Some(target) = self.max_target_fitness {
            b = b.set_stop_target_fitness(target);
        }
        if let Some(k) = self.elite_count {
            b = b.set_elite_count(k);
        }
        if let Some(n) = self.checkpoint_every {
            b = b.set_checkpoint_every(n);
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
        if self.freeze_elites {
            b = b.set_elite_freeze(true);
        }
        if let Some(tol) = self.regression_tol {
            b = b.set_fitness_regression_tol(tol);
        }
        if self.fresh_immigrants {
            b = b.set_immigrant_fresh_start(true);
        }
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

    /// Initialize `env_logger` with the engine's vocabulary. `default_level`
    /// is used when `--log-level` is absent (each example passes its own const).
    ///
    /// RUST_LOG still wins when set, so power users keep the fine-grained
    /// `module=level` filtering they expect.
    pub fn init_logger(&self, default_level: LogLevel) {
        use std::io::Write;
        let level = self.log_level.unwrap_or(default_level);
        let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| level.env_filter().to_string());
        let _ = env_logger::Builder::new()
            .parse_filters(&filter)
            .format(|buf, record| writeln!(buf, "{}", record.args()))
            .try_init();
    }
}
