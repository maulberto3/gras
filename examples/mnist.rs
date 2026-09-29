//! MNIST — the tabular showcase for the full user-owned surface.
//!
//! The library ships CONTRACTS, not recipes. This file supplies everything
//! concrete, in four sections:
//!   1 DATA      — a gras-format dataset (generated if absent).
//!   2 TRAINER   — a hand-rolled `StepTrainer` + `TabularStep` impl.
//!   3 GUARDRAIL — the optional holdout scorer on that same trainer.
//!   4 RUN       — consts → config → spec → run → guardrail.
//!
//! Run: `source env_setup.sh && cargo run --release --example mnist`
//! (`cargo run --release --example mnist_data` first to fetch real MNIST —
//! without `data/mnist/` this generates random-label synthetic data, which is
//! a smoke test only: nothing generalizes to random labels).
//! After a run, delete the generated `results/<timestamp>` folder yourself.

#[path = "cli/mod.rs"]
mod cli;

use std::path::{Path, PathBuf};

use clap::Parser;
use gras::engine::config::LogLevel;
use gras::engine::fitness::{Direction, Fitness, Metric};
use gras::engine::{TabularEngine, TabularRaceConfig};
use gras::flodl::Tensor;
use gras::flodl::nn::Module;
use gras::flodl::nn::optim::Optimizer;
use gras::flodl::tensor::{Result as TensorResult, TensorError};
use gras::graph::network::Network;
use gras::trainer::{StepTrainer, StreamShape, TabularContext, TabularStep, TabularStepReport};
use gras::utils::race_steps::{deterministic_train_step, eval_one_step};
use gras::utils::{score, tabular_data};
use gras::{BatchStream, PoolSplit, Variable};

// ═══ 1. DATA ════════════════════════════════════════════════════════════

/// Any gras-format dataset works — dims are auto-peeked. Real MNIST lives
/// here after `cargo run --example mnist_data`.
const DATA_DIR: &str = "data/mnist/train";
/// The GUARDRAIL's held-out split. Real MNIST's own test set — rows no race
/// batch ever trains on. Never point this at `DATA_DIR`.
const HOLDOUT_DIR: &str = "data/mnist/test";

// ═══ 2. TRAINER ═════════════════════════════════════════════════════════
//
// The contract is two traits; the library provides no scheme, so this is the
// real code a user writes. Compare `examples/cartpole.rs` (same shape, `RlStep`).

const LEARNING_RATE: f32 = 1e-3;
const GRAD_CLIP: f32 = 1.0;
const BATCH_SIZE: usize = 64;
const EVAL_BATCH_SIZE: usize = 256;
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
    /// `(test set, deterministic stream over it, draws)` — the guardrail half.
    holdout: Option<(tabular_data::Dataset, BatchStream, usize)>,
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
            holdout: None,
        }
    }

    /// Install the guardrail: score the champion on `rows` rows of `dataset`,
    /// `matches` times. Measurement only — it must never learn.
    fn with_holdout(mut self, dataset: tabular_data::Dataset, rows: usize, matches: usize) -> Self {
        let seed = 0xBEEF; // disjoint from the run seed's pool
        let split = PoolSplit::of(&dataset, 0.1, seed);
        let stream = BatchStream::new(seed, rows, split).with_eval_batch_size(rows);
        self.holdout = Some((dataset, stream, matches));
        self
    }
}

impl StepTrainer for MnistTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        Box::new(gras::flodl::nn::Adam::new(
            &net.parameters(),
            self.learning_rate as f64,
        ))
    }

    /// Recorded in `engine.json` under `"trainer"`, verbatim.
    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "mnist-handrolled",
            "optimizer": "adam",
            "loss": format!("label_smoothing_ce_{}", self.smoothing),
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "batch_size": self.train_batch,
            "eval_batch_size": self.eval_batch,
            "jitter_scale": JITTER_SCALE,
        }))
    }

    /// Score ONE holdout draw — `game_i` seeds which rows.
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> TensorResult<f32> {
        let (dataset, stream, _) = self
            .holdout
            .as_ref()
            .ok_or_else(|| TensorError::new("mnist: no holdout installed"))?;
        let (x, y) = stream.train_batch(dataset, game_i as u64)?;
        net.eval();
        let pred = net.forward(&Variable::new(x, false))?;
        net.train();
        score::accuracy_score(&pred, &Variable::new(y, false))
    }

    fn holdout_matches(&self) -> Option<usize> {
        self.holdout.as_ref().map(|(_, _, m)| *m)
    }
}

impl TabularStep for MnistTrainer {
    fn loss(&self) -> gras::trainer::LossFn<'_> {
        &*self.loss_fn
    }

    fn stream_shape(&self) -> Option<StreamShape> {
        Some(StreamShape {
            batch_size: self.train_batch,
            eval_batch_size: self.eval_batch,
        })
    }

    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &TabularContext<'_>,
    ) -> TensorResult<TabularStepReport> {
        // Determinism: seed per (net, step) so dropout/jitter replay exactly
        // across catch-up and resume.
        gras::seed_step_randomness(ctx.net_seed, step as u64, 0);
        let loss_fn: gras::trainer::LossFn<'_> = &*self.loss_fn;

        let (x, y) = ctx.data.train_batch(step as u64)?;
        // CHALLENGE: `challenge_prob` is the effective per-(input, feature)
        // rate, so every value rolls independently; strength is scaled by the
        // net's last accuracy — weak nets get pushed harder than strong ones.
        // The count we jittered is reported back to the engine's books.
        let (x, challenged_inputs) = if ctx.challenge_prob > 0.0 {
            let strength = JITTER_SCALE * (1.0 - self.last_accuracy).max(0.0);
            jitter_inputs(&x, ctx.challenge_prob, strength, ctx.net_seed, step)?
        } else {
            (x, 0)
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
            challenged_inputs,
        })
    }
}

/// Jitter each input value independently with probability `prob`, adding
/// Gaussian noise of std `strength` when it fires. Returns the perturbed
/// batch and how many values were jittered (the number the engine logs).
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

// ═══ 3. GUARDRAIL ═══════════════════════════════════════════════════════
//
// Measurement only, on rows the race never touched. Exact same estimator as
// the run's fitness (mean accuracy) — anything else hides a collapse.

const HOLDOUT_ROWS: usize = 512;
const HOLDOUT_MATCHES: usize = 5;

// ═══ 4. RUN ═════════════════════════════════════════════════════════════

const RUN_NAME: &str = "mnist";
const RUN_SEED: Option<u64> = Some(16);
const MAX_STEPS: Option<usize> = Some(50);
const POP_SIZE: usize = 100;
const CHECKPOINT_EVERY: usize = 5;
const POP_PRUNER: bool = true;
const PRUNER_STEPS: usize = 25;
/// Per-(net, step) chance an anti-plateau challenge fires (see `train_step`).
const CHALLENGE_PROB: f32 = 0.15;
const MUTATE_PROB: f32 = 0.5;
const LOG_LEVEL: LogLevel = LogLevel::Summ;

/// The CLI is an OVERLAY on the consts, never a second source of truth.
#[derive(Parser, Debug)]
#[command(name = "mnist", about = "Hand-rolled tabular trainer on MNIST.")]
struct Cli {
    #[command(flatten)]
    engine: cli::EngineArgs,

    /// Continue a previous run from its saved frontier (`results/<timestamp>`).
    #[arg(long, value_name = "RUN_DIR")]
    resume: Option<PathBuf>,

    /// Dataset directory (any gras-format dataset works).
    #[arg(long, value_name = "DIR")]
    data_dir: Option<PathBuf>,

    /// Learning rate for the trainer's Adam optimizer.
    #[arg(long, value_name = "LR")]
    learning_rate: Option<f32>,

    /// Training batch size (rows per net per step).
    #[arg(long, value_name = "N")]
    batch_size: Option<usize>,

    /// Label-smoothing epsilon (0.0 = plain cross-entropy).
    #[arg(long, value_name = "EPS")]
    label_smoothing: Option<f32>,
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
    if !data_dir.exists() {
        println!(
            "  {} not found — generating synthetic data",
            data_dir.display()
        );
    }
    if !data_dir.exists() {
        let ds =
            tabular_data::synthetic_classification(1024, 784, 10, 42, gras::auto_device()).unwrap();
        tabular_data::save_dataset(data_dir, &ds).unwrap();
    }
    let peeked = tabular_data::resolve_dataset(data_dir).unwrap();
    let d_in = peeked.inputs.shape()[1] as usize;
    let d_out = peeked.targets.shape()[1] as usize;
    drop(peeked);

    // The guardrail's held-out rows. Real MNIST test split when present;
    // otherwise synthetic noise — which is random-labeled, so a chance-level
    // verdict is the only honest outcome (and is warned about below).
    let holdout = if Path::new(HOLDOUT_DIR).exists() {
        tabular_data::resolve_dataset(Path::new(HOLDOUT_DIR)).unwrap()
    } else {
        tabular_data::synthetic_classification(
            HOLDOUT_ROWS * HOLDOUT_MATCHES,
            d_in,
            d_out,
            0xBEEF,
            gras::auto_device(),
        )
        .unwrap()
    };
    if !Path::new(HOLDOUT_DIR).exists() {
        println!(
            "  ⚠ no {} — the guardrail will score SYNTHETIC random labels (chance by construction)",
            HOLDOUT_DIR
        );
    }

    // ── STEP B — fitness, trainer, guardrail.
    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
    let mut trainer = MnistTrainer::new(cli.label_smoothing.unwrap_or(LABEL_SMOOTHING));
    trainer.learning_rate = cli.learning_rate.unwrap_or(LEARNING_RATE);
    trainer.train_batch = cli.batch_size.unwrap_or(BATCH_SIZE);
    let trainer = trainer.with_holdout(holdout, HOLDOUT_ROWS, HOLDOUT_MATCHES);

    // ── STEP C — the config surface.
    let pop = cli.engine.pop.unwrap_or(POP_SIZE);
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
        .set_run_pop_size(pop)
        .set_run_log_level(LOG_LEVEL)
        .set_run_metrics(metrics)
        .set_stop_max_steps(MAX_STEPS)
        .set_run_checkpoint_every(CHECKPOINT_EVERY)
        .set_elite_count(5)
        .set_elite_freeze(true)
        .set_run_smoothing_window(5)
        .set_crossover_prob(0.5)
        .set_crossover_rolls(pop / 2)
        .set_crossover_gate(gras::engine::config::CrossoverGate::Soft)
        .set_crossover_retries(3)
        // --- Challenge (anti-plateau) ---
        .set_run_challenge_prob(CHALLENGE_PROB) // trainer reads ctx.challenged and jitters
        // --- Mutation & catch-up (both modes; the tabular arm carries its own copy) ---
        .set_mutate_prob(MUTATE_PROB) // per-roll chance a random immigrant is inserted
        .set_mutate_rolls(pop / 5)
        .set_mutation_catch_up(false) // immigrants start at the current clock (no handicap)
        .set_crossover_catch_up(true) // children replay the stream, then pass the gate
        // --- Post-race pruner ---
        .set_pruner_enabled(POP_PRUNER)
        .set_pruner_method(gras::engine::config::PopPrunerMethod::Hard)
        .set_pruner_steps(PRUNER_STEPS)
        // --- Topology & network search boundaries ---
        .set_topology_min_hidden_num_nodes(2)
        .set_topology_max_hidden_num_nodes(15)
        .set_topology_min_inputs_per_node(2)
        .set_topology_max_inputs_per_node(15)
        .set_topology_input_dim(d_in)
        .set_topology_output_dim(d_out)
        .set_topology_hidden_dim_range(16, 128)
        .set_topology_hidden_dim_stride(16)
        .set_topology_dropout_prob(0.25)
        // --- Exports ---
        .set_run_csv_export(true)
        .set_elite_save_topology(true)
        .set_elite_save_safetensors(true)
        .set_worst_save_topology(true)
        .set_worst_save_safetensors(true)
        .set_elite_checkpoint_weights(true);
    let config = cli.engine.apply(builder).build();

    // ── STEP D — run (or resume), then the guardrail.
    let mut engine = match &cli.resume {
        Some(dir) => {
            println!("Resuming from {}", dir.display());
            TabularEngine::resume(
                dir.clone(),
                data_dir.to_path_buf(),
                config,
                fitness,
                trainer,
            )?
        }
        None => TabularEngine::from_spec(gras::engine::RunSpec::tabular(
            data_dir,
            config,
            fitness,
            trainer,
            cli.engine.seed_or(RUN_SEED),
            cli.engine.run_dir_or(None),
        ))?,
    };
    println!("Run Dir: {}", engine.run_dir().display());
    match engine.run() {
        Ok(reason) => println!("Race stopped successfully: {reason:?}"),
        Err(e) => eprintln!("Race encountered an error: {e}"),
    }

    match engine.guardrail(gras::auto_device(), None) {
        Some(v) => {
            let (holdout, std) = (v.mean().unwrap_or(0.0), v.std().unwrap_or(0.0));
            let smoothed = v
                .race_smoothed
                .map(|s| format!(" (race smoothed {s:.3})"))
                .unwrap_or_default();
            // The honest question is not "is 0.8 a good number" but "does the
            // holdout agree with the race signal". A big gap means the pool was
            // memorized; agreement means the ranking is trustworthy.
            let verdict = match v.race_smoothed {
                Some(race) if race - holdout > 0.05 => "weak — the race overfit the eval pool ❌",
                Some(_) => "consistent with the race signal ✅",
                None => "no race-smoothed reference",
            };
            println!("guardrail: holdout accuracy {holdout:.3} ± {std:.3}{smoothed} — {verdict}");
        }
        None => println!("guardrail: no champion exported — skipped"),
    }
    Ok(())
}
