// NOTE: `source env_setup.sh` in the shell first — this example is built and linked against that environment.
//! MNIST — the tabular showcase for the full user-owned surface.
//!
//! The library ships CONTRACTS, not recipes. This file supplies everything
//! concrete, in four sections:
//!   1 DATA      — a gras-format dataset (generated if absent).
//!   2 TRAINER   — a hand-rolled `TabularStep` impl (one trait, one block).
//!   3 CONFIG    — consts → config → spec → run → the final table.
//!
//! Run: `source env_setup.sh && cargo run --release --example mnist`
//! (`data/mnist` already holding real MNIST — the producer is the non-cargo
//! helper `data/mnist/mnist_data.rs` — without that this generates
//! random-label synthetic data, which is
//! a smoke test only: nothing generalizes to random labels).
//! After a run, delete the generated `results/<timestamp>` folder yourself.

mod cli {
    //! Shared CLI surface for the long-running example.
    //!
    //! [`EngineArgs`] / [`RlEngineArgs`] carry the fuller engine surface
    //! (`--log-level`, `--seed`, `--risk-aversion`, the `--merit-p-min` /
    //! `--merit-p-max` band, `--founders-from`, the evolution knobs, …)
    //! for `mnist`. `cartpole`
    //! declares its own `Cli` struct and does not include this module.
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
        pub champion_count: Option<usize>,

        /// FIT risk aversion λ: window-σs a net pays for its own spread before it is
        /// ranked (0 = rank on the plain window mean). Replay-relevant.
        #[arg(long, value_name = "LAMBDA", visible_alias = "risk-based")]
        pub risk_aversion: Option<f32>,

        /// Lottery FLOOR `p_min` (`set_ticket_p_min`, crate default 0.05): the chance
        /// the WORST-graded net is paid each clock. The band `p_min..p_max` IS the payout (Ω
        /// retired), so this is a literal per-clock probability, not a blend: `0` starves the
        /// bottom of the field, and `p_min == p_max` pays every graded net the same.
        #[arg(long, value_name = "P")]
        pub merit_p_min: Option<f32>,

        /// Lottery CEILING `p_max` (`set_ticket_p_max`, crate default 0.4): the chance
        /// the BEST-graded net is paid each clock; every net in between is interpolated by the
        /// graded fraction (`Fm`, times the updates leg `ticket_legs` names). It is the one dial
        /// that LOWERS ticket volume.
        #[arg(long, value_name = "P")]
        pub merit_p_max: Option<f32>,

        /// The lottery's LEGS (`set_ticket_legs`, default `least_fit_fewest_updates`): which fit
        /// place earns the most tickets and which side of the updates field it multiplies in.
        /// `--ticket-legs most_fit_fewest_updates` walks the field best-first instead.
        #[arg(
            long,
            value_name = "LEGS",
            value_parser = [
                "most_fit",
                "most_fit_fewest_updates",
                "most_fit_most_updates",
                "least_fit_fewest_updates",
                "least_fit_most_updates",
                "least_fit"
            ]
        )]
        pub ticket_legs: Option<String>,

        /// INIT PROBATION (`set_pop_init_probation_steps`, default 0): opening clocks with no
        /// culls and no births — evolution starts at clock `N`.
        #[arg(long, value_name = "N")]
        pub init_probation: Option<usize>,

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

        /// Mutation NOVELTY floor Δ: a random immigrant must sit at least this far
        /// (Jaccard distance on the gene multiset) from the nearest blueprint the
        /// run has ever made. `0` turns the gate off — one draw, the pre-gate path.
        #[arg(long, value_name = "DELTA")]
        pub novelty_min: Option<f32>,

        /// Candidates ONE mutation may spend clearing `--novelty-min` before the most
        /// novel refusal is taken anyway (a culled seat is never left empty).
        #[arg(long, value_name = "N")]
        pub novelty_draws: Option<usize>,

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

        /// Who an admitted crossover child evicts: `lowest_fitness` (plain lowest
        /// smoothed fitness), `highest_ticket_chance` (the net the lottery pays MOST —
        /// the highest `p`, so the least fit and least trained), `lowest_fitness_most_updates`
        /// (the lowest-fitness net that has ALREADY spent its updates — evidence, not the
        /// coin's next draw) or `random`. Mutation victims follow
        /// `--mutation-cull-policy` and ignore this.
        #[arg(long, value_name = "POLICY")]
        pub crossover_cull_policy: Option<String>,

        /// Mutation victim policy for the mutation/immigrant channel: `lowest_fitness`
        /// (deterministic lowest smoothed fitness), `inverse_fitness` (fitness-inverse
        /// roulette), `random` (uniform), `highest_ticket_chance` (the net the train-eval
        /// lottery pays most — same selector the crossover channel's
        /// `--crossover-cull-policy highest_ticket_chance` uses) or
        /// `lowest_fitness_most_updates` (the lowest-fitness net that has already spent its
        /// updates, `Fm⁻ × (1 − Um)`).
        /// Only fires when a mutation roll fires (see `--mutate-prob`).
        #[arg(long, value_name = "POLICY")]
        pub mutation_cull_policy: Option<String>,

        /// Warm-up before the decision-lag relay engages (RL only): for the
        /// first N engine steps every net just trains — no relay cycles. 0
        /// (default) = relay from step 0. See `DecisionLagTrainer` for the relay
        /// itself, and how to run without it entirely.
        #[arg(long, value_name = "N")]
        pub grace_periods: Option<usize>,

        /// Skip writing `elite-<hash>.md` + `.safetensors` at stop.
        #[arg(long)]
        pub no_elite_save: bool,

        /// Anti-devolution A: hold the top `champion_count` nets out of the weight update
        /// (they still act, score and sire). The example's default is OFF, exactly like
        /// cartpole's — pass `--freeze-seats` to turn the crown ON.
        #[arg(long)]
        pub freeze_seats: bool,
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
                b = b.set_pop_size(pop);
            }
            if let Some(steps) = self.max_steps {
                b = b.set_stop_max_steps(steps);
            }
            if let Some(k) = self.champion_count {
                b = b.set_champion_count(k);
            }
            if let Some(n) = self.checkpoint_every {
                b = b.set_run_checkpoint_every(n);
            }
            if let Some(lambda) = self.risk_aversion {
                b = b.set_fitness_risk_aversion(lambda);
            }
            // The band IS the payout (Ω retired), so the two ends are ordinary flags and there is
            // no strength to fold: what a run sets here is the chance it pays.
            if let Some(p) = self.merit_p_min {
                b = b.set_ticket_p_min(p);
            }
            if let Some(p) = self.merit_p_max {
                b = b.set_ticket_p_max(p);
            }
            if let Some(legs) = &self.ticket_legs {
                b = b.set_ticket_legs(
                    gras::engine::config::TicketLegs::parse(legs)
                        .unwrap_or(gras::engine::config::DEFAULT_TICKET_LEGS),
                );
            }
            if let Some(k) = self.init_probation {
                b = b.set_pop_init_probation_steps(k);
            }
            if let Some(dir) = &self.founders_from {
                b = b
                    .set_champion_previous_load(true)
                    .set_champion_previous_dir(dir.clone());
            }
            if let Some(p) = self.crossover_prob {
                b = b.set_crossover_prob(p);
            }
            if let Some(p) = self.mutate_prob {
                b = b.set_mutate_prob(p);
            }
            if let Some(d) = self.novelty_min {
                b = b.set_mutation_novelty_min(d);
            }
            if let Some(n) = self.novelty_draws {
                b = b.set_mutation_novelty_draws(n);
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
            // The flag and the builder agree on OFF, so this can only ever turn the crown
            // ON. It used to be a NEGATIVE flag (`--no-freeze-elites`, clap `SetFalse`,
            // i.e. default `true`) applied unconditionally here, which silently overwrote
            // the example's `set_seat_freeze(false)` on every run.
            b = b.set_seat_freeze(self.freeze_seats);
            if let Some(policy) = &self.crossover_cull_policy {
                b = b.set_crossover_cull_policy(
                    match policy.trim().to_ascii_lowercase().as_str() {
                        "lowest_fitness" | "worst" => {
                            gras::engine::config::CrossCullPolicy::LowestFitness
                        }
                        "random" => gras::engine::config::CrossCullPolicy::Random,
                        "highest_ticket_chance" | "unfed" => {
                            gras::engine::config::CrossCullPolicy::HighestTicketChance
                        }
                        "lowest_fitness_most_updates" | "spent" => {
                            gras::engine::config::CrossCullPolicy::LowestFitnessMostUpdates
                        }
                        other => {
                            eprintln!(
                                "--crossover-cull-policy {other}: expected `lowest_fitness`, `random`, `highest_ticket_chance`, or `lowest_fitness_most_updates`"
                            );
                            std::process::exit(2);
                        }
                    },
                );
            }
            if let Some(policy) = &self.mutation_cull_policy {
                b = b.set_mutation_cull_policy(match policy.trim().to_ascii_lowercase().as_str() {
                "inverse" | "inverse-fitness" => {
                    gras::engine::config::MutationCullPolicy::InverseFitness
                }
                "lowest_fitness" | "worst" => {
                    gras::engine::config::MutationCullPolicy::LowestFitness
                }
                "random" => gras::engine::config::MutationCullPolicy::Random,
                "highest_ticket_chance" | "unfed" => {
                    gras::engine::config::MutationCullPolicy::HighestTicketChance
                }
                "lowest_fitness_most_updates" | "spent" => {
                    gras::engine::config::MutationCullPolicy::LowestFitnessMostUpdates
                }
                other => {
                    eprintln!(
                        "--mutation-cull-policy {other}: expected `lowest_fitness`, `inverse`, `random`, `highest_ticket_chance`, or `lowest_fitness_most_updates`"
                    );
                    std::process::exit(2);
                }
            });
            }
            if self.no_elite_save {
                b = b.set_champion_save_topology(false);
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
}

use std::path::{Path, PathBuf};

use clap::Parser;
use gras::flodl::tensor::Result as TensorResult;
use gras::prelude::*;
use gras::utils::race_steps::{deterministic_train_step, eval_one_step};
use gras::utils::{score, tabular_data};

// ═══ 1. DATA ════════════════════════════════════════════════════════════

/// Any gras-format dataset works — dims are auto-peeked. Real MNIST lives
/// here once `data/mnist/mnist_data.rs` has produced it.
///
/// EXPLICIT TRAIN/TEST LAYOUT: point the engine at the PARENT dir
/// (`data/mnist`) and it resolves `train/` + `test/` through
/// `resolve_train_test_datasets` — the race trains ONLY on train rows and
/// evals ONLY on test rows, by construction. A single-pool dataset dir
/// (no train/test subdirs) still works: the engine falls back to its
/// seeded internal split.
const DATA_DIR: &str = "data/mnist";

// ═══ 2. TRAINER ═════════════════════════════════════════════════════════
//
// ONE trait (`TabularStep`), ONE impl block — the library provides no scheme,
// so this is the real code a user writes. REQUIRED: `make_optimizer`, `loss`,
// `train_step`. OPTIONAL: `describe`, `restore_optimizer`, `stream_shape`.
// Compare `examples/cartpole.rs` (same shape, `RlStep`).

const LEARNING_RATE: f32 = 1e-3;
const GRAD_CLIP: f32 = 2.0;
const BATCH_SIZE: usize = 32;
const EVAL_BATCH_SIZE: usize = 64;
const LABEL_SMOOTHING: f32 = 0.1;
/// Peak input jitter std on a challenge step. Scaled by `1 − last_accuracy`,
/// so a weak net explores hard and a strong net is barely disturbed.
const JITTER_SCALE: f32 = 0.3;

/// One train batch + one eval batch per step, Adam, label-smoothed
/// cross-entropy — plus the challenge response the tabular arm enables.
struct MnistTrainer {
    loss_fn: gras::trainer::BoxedLossFn,
    learning_rate: f32,
    grad_clip: f32,
    train_batch: usize,
    eval_batch: usize,
    smoothing: f32,
    /// Last eval accuracy (from the previous step's report) — the metric that
    /// scales challenge jitter.
    last_accuracy: f32,
    /// The trainer's own anti-plateau knob (see `CHALLENGE_PROB`). Tabular
    /// response: every input value rolls independently at the DECAYED rate
    /// (`gras::trainer::effective_challenge_prob`) each step — no per-step
    /// gate, the element-level rolls ARE the challenge. Trainer-owned since
    /// 2026-10-10; the engine never sees it.
    challenge_prob: f32,
}

impl MnistTrainer {
    fn new(smoothing: f32) -> Self {
        Self {
            loss_fn: Box::new(move |pred: &Variable, y: &Variable| {
                score::label_smoothing_cross_entropy_loss(pred, y, smoothing)
            }),
            learning_rate: LEARNING_RATE,
            grad_clip: GRAD_CLIP,
            train_batch: BATCH_SIZE,
            eval_batch: EVAL_BATCH_SIZE,
            smoothing,
            last_accuracy: 0.0,
            challenge_prob: CHALLENGE_PROB,
        }
    }
}

impl TabularStep for MnistTrainer {
    // REQUIRED: the optimizer built for every net at birth (and on resume
    // rebuilds); its state (Adam moments) carries across steps.
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        Box::new(gras::flodl::nn::Adam::new(
            &net.parameters(),
            self.learning_rate as f64,
        ))
    }

    /// OPTIONAL: recorded in `engine.json` under `"trainer"`, verbatim.
    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "mnist-handrolled",
            "optimizer": "adam",
            "loss": format!("label_smoothing_ce_{}", self.smoothing),
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "batch_size": self.train_batch,
            "eval_batch_size": self.eval_batch,
            // The tabular arm's anti-plateau response is element-level jitter
            // rolled inside `train_step` against the decayed rate — TRAINER
            // owned, so this blob is the only record of it (the engine has no
            // challenge surface left).
            "challenge_prob": self.challenge_prob,
            "challenge_decay": "linear",
            "jitter_scale": JITTER_SCALE,
        }))
    }

    /// R9: the challenge jitter is scaled by `1 − last_accuracy`, so a value produced by
    /// the PREVIOUS clocks decides what the next one does — exactly the state the weights
    /// do not hold. It rides each net's `nets/<hash>.json` envelope, so a resume picks it
    /// up before the first continued step instead of restarting the decay at 0.0.
    ///
    /// One run-level scalar stamped into every net: restoring it from any of them is
    /// idempotent, and the last write at a checkpoint is the run's own value.
    fn state(&self, _net_hash: &str) -> Option<serde_json::Value> {
        Some(serde_json::json!({ "last_accuracy": self.last_accuracy }))
    }

    /// R9: read it back on resume (called per loaded net, before its next step).
    fn restore(&mut self, _net_hash: &str, state: &serde_json::Value) {
        if let Some(v) = state.get("last_accuracy").and_then(|v| v.as_f64()) {
            self.last_accuracy = v as f32;
        }
    }

    // REQUIRED: the (pred, target) loss — a supervised scheme's defining choice.
    fn loss(&self) -> gras::trainer::LossFn<'_> {
        &*self.loss_fn
    }

    // OPTIONAL: batch sizes requested from the engine's shared stream (the
    // train/eval split RATIO is engine bookkeeping, not overridable).
    fn stream_shape(&self) -> Option<StreamShape> {
        Some(StreamShape {
            batch_size: self.train_batch,
            eval_batch_size: self.eval_batch,
        })
    }

    // REQUIRED: one clock-step of training + this step's report (`fitness` is
    // the ranking input; `ctx.data` is always present in tabular mode).
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &TabularContext<'_>,
    ) -> TensorResult<TabularStepReport> {
        // Determinism: seed per (net, step) so dropout/jitter replay exactly
        // across a resume.
        gras::seed_step_randomness(ctx.net_seed, step as u64, 0);
        let loss_fn: gras::trainer::LossFn<'_> = &*self.loss_fn;

        let (x, y) = ctx.data.train_batch(step as u64)?;
        // CHALLENGE (trainer-side): `p_eff` is the decayed per-(input,
        // feature) rate — every value rolls independently; strength is
        // scaled by the net's last accuracy — weak nets get pushed harder
        // than strong ones. Linear decay to 0 at the budget (the historical
        // schedule), flat when unbounded.
        let p_eff = gras::trainer::effective_challenge_prob(
            self.challenge_prob,
            step,
            ctx.env.max_steps,
            ChallengeDecay::Linear,
        );
        let x = if p_eff > 0.0 {
            let strength = JITTER_SCALE * (1.0 - self.last_accuracy).max(0.0);
            jitter_inputs(&x, p_eff, strength, ctx.net_seed, step)?.0
        } else {
            x
        };
        let train_loss = deterministic_train_step(
            ctx.net_seed,
            step as u64,
            0,
            net,
            optimizer,
            loss_fn,
            &(x, y),
            self.grad_clip,
        )?;

        let (ex, ey) = ctx.data.eval_batch(step as u64)?;
        let report = eval_one_step(net, loss_fn, ctx.fitness, ctx.metrics, &(ex, ey))?;
        self.last_accuracy = report.fitness;

        Ok(TabularStepReport {
            train_loss,
            eval_loss: report.eval_loss,
            fitness: report.fitness,
            informative: report.metrics,
        })
    }
}

/// Jitter each input value independently with probability `prob`, adding
/// Gaussian noise of std `strength` when it fires. Returns the perturbed
/// batch (the jitter count was bookkeeping for the old engine-side challenge
/// accounting — gone with it, 2026-10-10).
/// The RNG is seeded from `(net_seed, step)` so a replay draws identically.
fn jitter_inputs(
    x: &Tensor,
    prob: f32,
    strength: f32,
    net_seed: u64,
    step: usize,
) -> TensorResult<(Tensor, usize)> {
    let shape = x.shape().to_vec();
    let mut data = x.to_f32_vec()?;
    let mut rng =
        fastrand::Rng::with_seed(net_seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut jittered = 0usize;
    for v in data.iter_mut() {
        if rng.f32() >= prob {
            continue;
        }
        jittered += 1;
        let u1 = rng.f32().max(1e-6);
        let u2 = rng.f32();
        let n = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        *v = (*v + n * strength).clamp(0.0, 1.0);
    }
    Ok((Tensor::from_f32(&data, &shape, x.device())?, jittered))
}

// ═══ 4. RUN ═════════════════════════════════════════════════════════════

const RUN_NAME: &str = "mnist";
const RUN_SEED: Option<u64> = Some(16);
const MAX_STEPS: Option<usize> = Some(50);
const POP_SIZE: usize = 100;
const CHECKPOINT_EVERY: usize = 50;
/// Per-(net, step) chance an anti-plateau challenge fires (see `train_step`).
const CHALLENGE_PROB: f32 = 0.15;
const LOG_LEVEL: LogLevel = LogLevel::Summ;
/// FIT risk aversion λ: window σs an accuracy pays for its own spread, so a net
/// that is steadily 0.90 outranks one alternating 0.80 and 1.00. `0.0` ranks on
/// the plain window mean.
const RISK_AVERSION: f32 = 0.5;

/// The CLI is an OVERLAY on the consts, never a second source of truth: the
/// shared engine surface ([`cli::EngineArgs`] — pop/steps/seed/λ/Ω/founders,
/// the evolution knobs and `--history-flush`), plus the one real input (where
/// the data is). Anything left unset falls back to the const panel above.
#[derive(Parser, Debug)]
#[command(name = "mnist", about = "Hand-rolled tabular trainer on MNIST.")]
struct Cli {
    #[command(flatten)]
    engine: cli::EngineArgs,

    /// Dataset directory (any gras-format dataset works).
    #[arg(long, value_name = "DIR")]
    data_dir: Option<PathBuf>,

    /// Continue an existing run from its directory: everything but the trainer code
    /// comes from that run's `engine.json`. `--max-steps` then EXTENDS the recorded
    /// budget (it is absolute, not per-process).
    #[arg(long, value_name = "DIR")]
    resume: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    cli.engine.init_logger(LOG_LEVEL);

    // ── STEP A — data: generate if missing so the example runs out of the box.
    let data_dir_arg = cli
        .data_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DATA_DIR));
    let data_dir: &Path = &data_dir_arg;
    // The engine accepts BOTH layouts at this path: a parent dir holding
    // `train/` + `test/` (real MNIST, if fetched) — the explicit
    // two-dataset mode — or a single-pool dataset dir (synthetic fallback
    // below) — the classic seeded split.
    let explicit_layout = data_dir.join("train").exists() && data_dir.join("test").exists();
    if !explicit_layout && !data_dir.exists() {
        println!(
            "  {} not found — generating synthetic single-pool data",
            data_dir.display()
        );
    }
    if !explicit_layout && !data_dir.exists() {
        let ds =
            tabular_data::synthetic_classification(1024, 784, 10, 42, gras::auto_device()).unwrap();
        tabular_data::save_dataset(data_dir, &ds).unwrap();
    }
    // Dims peek: train side of the explicit layout, or the single pool.
    let peeked = if explicit_layout {
        let (train, _test) = tabular_data::resolve_train_test_datasets(data_dir).unwrap();
        train
    } else {
        tabular_data::resolve_dataset(data_dir).unwrap()
    };
    let d_in = peeked.inputs.shape()[1] as usize;
    let d_out = peeked.targets.shape()[1] as usize;
    drop(peeked);
    if explicit_layout {
        println!(
            "  explicit train/test layout at {} — eval rows come ONLY from test/ (no train/eval overlap by construction)",
            data_dir.display()
        );
    }

    // ── STEP B — fitness, trainer.
    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
    let trainer = MnistTrainer::new(LABEL_SMOOTHING);

    // ── STEP C — the config surface.
    let pop = cli.engine.pop.unwrap_or(POP_SIZE);
    let max_steps = cli.engine.max_steps.or(MAX_STEPS);
    let metrics = vec![
        // Informative (non-ranking) custom metric: mean gap between the top-1
        // and runner-up logit — one extra history.csv column. There are no
        // built-in labels; a metric is always an explicit closure.
        Metric::custom("top1_margin", |pred: &Variable, y: &Variable| {
            let p = pred.data().to_f32_vec()?;
            let n = y.data().shape()[0] as usize;
            let c = y.data().shape()[1] as usize;
            let mut sum = 0.0f32;
            for r in 0..n {
                let mut row = p[r * c..(r + 1) * c].to_vec();
                row.sort_by(|a, b| b.partial_cmp(a).unwrap());
                sum += row[0] - row[1];
            }
            Ok(sum / n as f32)
        }),
    ];
    let builder = TabularRaceConfig::builder()
        .set_run_name(RUN_NAME)
        .set_run_log_level(LOG_LEVEL)
        .set_run_checkpoint_every(CHECKPOINT_EVERY)
        .set_run_csv_export(true) // history.csv + attempts.csv
        .set_pop_size(pop)
        // --- POP guards (the same knobs cartpole's showcase arms) ---
        // RECOVERY: a reading that collapses by more than `tol` (relative) from the net's own
        // standing is re-run inside the SAME clock, up to `retries` extra updates, so one unlucky
        // batch cannot get a good net culled. `same_seed = false` lets the retry draw a fresh one.
        .set_pop_recovery_retries(5)
        .set_pop_recovery_tol(0.2)
        .set_pop_recovery_same_seed(false)
        // INIT PROBATION: the opening clocks evolve nothing — no culls, no births — so the
        // founders are judged on draws every net shares instead of one lucky clock.
        .set_pop_init_probation_steps(5)
        .set_run_metrics(metrics)
        .set_fitness_smoothing_window(10)
        .set_fitness_risk_aversion(RISK_AVERSION)
        // The averaging FAMILY: `F = centre ∓ λ·spread` in that family's space. `median` charges
        // `λ · 1.4826 · MAD` (one fluke reading cannot move a net); `arithmetic` is the crate
        // default, and the one whose printed legs are literally `mean ∓ λ·σ`.
        .set_fitness_mean_type(gras::engine::config::MeanType::Median)
        // POLICY-SCOPED fitness (crate default `false`): RL-only in practice — a supervised clock
        // that measures IS an update, so scoping here would pin every net to `n = 1` (newest
        // reading, no spread) instead of its K-reading window.
        .set_fitness_policy_scoped(true)
        //
        .set_stop_max_steps(max_steps)
        // STOP (custom): the one stop rule besides the step budget — consulted every clock, and
        // `true` ends the race with reason `Custom`. `best_smoothed_fitness` is F (the
        // risk-adjusted accuracy this run RANKS on, so λ = 0.5 makes it sit below the raw
        // accuracy), which is why the bar is 0.85 rather than 0.85 of raw accuracy. `max_steps`
        // stays the ceiling for a run that never gets there.
        .set_stop_custom(|snap| snap.best_smoothed_fitness > 0.85)
        //
        .set_crossover_rolls(pop / 30)
        .set_crossover_prob(0.5)
        .set_crossover_cull_policy(CrossCullPolicy::LowestFitness)
        .set_crossover_probation_steps(5)
        .set_crossover_selection(SelectionMethod::Tournament { tournament_size: 2 })
        // --- Lottery (the per-clock ticket; Ω retired) ---
        // `p = p_min + (p_max − p_min) × Fm⁻ × Um`: the band IS the payout, and the shipped legs
        // are `least_fit_fewest_updates` (the least fit, fewest updates net pays most). Set at the
        // crate defaults here; `--merit-p-min` / `--merit-p-max` / `--ticket-legs` are the dials.
        .set_ticket_p_min(0.05)
        .set_ticket_p_max(0.95)
        .set_ticket_legs(gras::engine::config::DEFAULT_TICKET_LEGS)
        // LEARN-ONLY (crate default `false`): `true` keeps a denied net ACTING and MEASURING and
        // withholds only the weight write. Supervised has no act-only pass, so a denied ticket
        // idles the net outright — the pre-flag contract.
        .set_ticket_learn_only(true)
        // --- Mutation (both modes) ---
        .set_mutate_rolls(pop / 30)
        .set_mutation_cull_policy(MutationCullPolicy::LowestFitness)
        .set_mutate_prob(0.5) // per-roll chance a random immigrant is inserted
        .set_mutation_probation_steps(5)
        .set_mutation_novelty_min(0.3) // Δ ≥ 0.2 from the nearest blueprint; 0 = the gate off
        .set_mutation_novelty_draws(5) // candidates one mutation may spend before taking the best
        // --- Challenge (anti-plateau): TRAINER-owned since 2026-10-10 —
        // `CHALLENGE_PROB` lives on the trainer and rolls inside `train_step`
        // via `gras::trainer::effective_challenge_prob`. No engine knob.
        // --- Guardrail (post-race honesty check) ---
        // --- Topology & network search boundaries ---
        .set_topology_min_hidden_num_nodes(2)
        .set_topology_max_hidden_num_nodes(15)
        .set_topology_min_inputs_per_node(2)
        .set_topology_max_inputs_per_node(15)
        // A HIDDEN node's own fan-out (2..15, the same bound as its fan-in above), so a node is
        // never born with a single outgoing wire and "width" means what the census column says.
        .set_topology_min_outputs_per_node(2)
        .set_topology_max_outputs_per_node(15)
        // DRIFT: a crossover child inherits its parents' genes EXACTLY at 0.0; 0.05 re-draws a
        // small share of them, so crossover is not a pure cut-and-paste.
        .set_topology_drift(0.1)
        .set_topology_input_dim(d_in)
        .set_topology_output_dim(d_out)
        .set_topology_hidden_dim_range(16, 64)
        .set_topology_hidden_dim_stride(16)
        .set_topology_dropout_prob(0.20)
        // .set_topology_combine_op_pool(COMBINE_OPS)
        // .set_topology_activation_pool(ACTIVATIONS)
        // .set_topology_standardize_op_pool(STANDARDIZE_OPS)
        // .set_topology_feature_split(FEATURE_SPLIT)
        .set_topology_feature_split_pool(["random"])
        // --- Exports ---
        // --- Evolution: the elite/founder rules (mode-agnostic) ---
        .set_seat_freeze(false) // cartpole's choice: a seated elite still scores, ranks and sires
        // (the `--freeze-seats` flag is OFF by default, so this is what a plain run gets)
        .set_seat_join_probation_steps(5)
        .set_seat_lifetime_steps(5)
        .set_champion_count(pop / 10)
        .set_champion_previous_load(true) // on by default
        .set_champion_save_topology(true)
        .set_worst_save_topology(true); // worst-<hash>-<fitness>.md + the worst.md index
    // Every CLI flag is an OVERLAY: a user-passed value wins, anything unset
    // keeps the const above. This is also where `--history-flush`, the `--merit-p-min` /
    // `--merit-p-max` band and `--founders-from` become engine config instead of being
    // silently dropped.
    let config = cli.engine.apply(builder).build();

    // ── STEP D — run. The `Final field` table is the report; there is no exam after it.
    // `--run-dir` stays an OUTPUT path; continuing a run is `--resume <dir>`, which is
    // possible again since R9: `MnistTrainer` now reports `last_accuracy` through
    // `TabularStep::state()` (it rides `nets/<hash>.json`) and reads it back through
    // `restore()`, so the resumed jitter strength is the one the previous clocks
    // produced instead of 0.0. Before R9 this example failed its parity check at step 1.
    let mut engine = match &cli.resume {
        Some(dir) => {
            let mut engine = TabularEngine::resume_from(dir.clone(), fitness, trainer)?;
            // The budget is absolute: on a 20-clock run resumed at 30, 10 clocks run.
            if let Some(steps) = cli.engine.max_steps {
                engine.set_stop_max_steps(steps);
            }
            engine
        }
        None => TabularEngine::from_spec(gras::engine::RunSpec::tabular(
            data_dir,
            config,
            fitness,
            trainer,
            cli.engine.seed_or(RUN_SEED),
            cli.engine.run_dir_or(None), // engine picks results/<timestamp> when unset
        ))?,
    };
    println!("Run Dir: {}", engine.run_dir().display());
    match engine.run() {
        Ok(reason) => println!("Race stopped successfully: {reason:?}"),
        Err(e) => eprintln!("Race encountered an error: {e}"),
    }

    Ok(())
}
