//! MNIST — the tabular showcase for the full user-owned surface.
//!
//! The library ships CONTRACTS, not recipes. This file supplies everything
//! concrete, in four sections:
//!   1 DATA      — a gras-format dataset (generated if absent).
//!   2 TRAINER   — a hand-rolled `TabularStep` impl (one trait, one block).
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
use gras::flodl::tensor::{Result as TensorResult, TensorError};
use gras::prelude::*;
use gras::utils::race_steps::{deterministic_train_step, eval_one_step};
use gras::utils::{score, tabular_data};

// ═══ 1. DATA ════════════════════════════════════════════════════════════

/// Any gras-format dataset works — dims are auto-peeked. Real MNIST lives
/// here after `cargo run --example mnist_data`.
///
/// EXPLICIT TRAIN/TEST LAYOUT: point the engine at the PARENT dir
/// (`data/mnist`) and it resolves `train/` + `test/` through
/// `resolve_train_test_datasets` — the race trains ONLY on train rows and
/// evals ONLY on test rows, by construction. A single-pool dataset dir
/// (no train/test subdirs) still works: the engine falls back to its
/// seeded internal split.
const DATA_DIR: &str = "data/mnist";
/// The GUARDRAIL's held-out split. Real MNIST's own test set — rows no race
/// batch ever trains on. Never point this at `DATA_DIR`.
const HOLDOUT_DIR: &str = "data/mnist/test";

// ═══ 2. TRAINER ═════════════════════════════════════════════════════════
//
// ONE trait (`TabularStep`), ONE impl block — the library provides no scheme,
// so this is the real code a user writes. REQUIRED: `make_optimizer`, `loss`,
// `train_step`. OPTIONAL: `describe`, `holdout_score`, `stream_shape`.
// Compare `examples/cartpole.rs` (same shape, `RlStep`).

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
    /// The trainer's own anti-plateau knob (see `CHALLENGE_PROB`). Tabular
    /// response: every input value rolls independently at the DECAYED rate
    /// (`gras::trainer::effective_challenge_prob`) each step — no per-step
    /// gate, the element-level rolls ARE the challenge. Trainer-owned since
    /// 2026-10-10; the engine never sees it.
    challenge_prob: f32,
    /// `(test set, deterministic stream over it)` — the guardrail half. How
    /// many draws it makes is the RUN's knob (`set_elite_guardrail_matches`),
    /// not this trainer's.
    holdout: Option<(tabular_data::Dataset, BatchStream)>,
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
            holdout: None,
        }
    }

    /// Install the guardrail: score the champion on `rows` rows of `dataset`
    /// per draw. Measurement only — it must never learn.
    fn with_holdout(mut self, dataset: tabular_data::Dataset, rows: usize) -> Self {
        let seed = 0xBEEF; // disjoint from the run seed's pool
        let split = PoolSplit::of(&dataset, 0.1, seed);
        let stream = BatchStream::new(seed, rows, split).with_eval_batch_size(rows);
        self.holdout = Some((dataset, stream));
        self
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

    /// OPTIONAL (guardrail): score ONE holdout draw — `game_i` seeds which rows.
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> TensorResult<f32> {
        let (dataset, stream) = self
            .holdout
            .as_ref()
            .ok_or_else(|| TensorError::new("mnist: no holdout installed"))?;
        let (x, y) = stream.train_batch(dataset, game_i as u64)?;
        net.eval();
        let pred = net.forward(&Variable::new(x, false))?;
        net.train();
        score::accuracy_score(&pred, &Variable::new(y, false))
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

/// The CLI is an OVERLAY on the consts, never a second source of truth: the
/// shared two-flag smoke surface, plus the one real input (where the data is).
#[derive(Parser, Debug)]
#[command(name = "mnist", about = "Hand-rolled tabular trainer on MNIST.")]
struct Cli {
    #[command(flatten)]
    smoke: cli::SmokeArgs,

    /// Dataset directory (any gras-format dataset works).
    #[arg(long, value_name = "DIR")]
    data_dir: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    cli::init_logger(LOG_LEVEL, cli.smoke.log_level.as_deref());

    // ── STEP A — data: generate if missing so the example runs out of the box.
    let data_dir_arg = cli
        .data_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DATA_DIR));
    let data_dir: &Path = &data_dir_arg;
    // The engine accepts BOTH layouts at this path: a parent dir holding
    // `train/` + `test/` (real MNIST after mnist_data) — the explicit
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

    // The guardrail's held-out rows. Real MNIST test split when present;
    // otherwise synthetic noise — which is random-labeled, so a chance-level
    // verdict is the only honest outcome (and is warned about below).
    // EFFICIENCY NOTE (Tier 1 — `impl AsRef<Path>`): the loaders take
    // `impl AsRef<Path>`, so the `&str` const passes directly — no
    // `Path::new(...)` wrapper needed at any call site.
    let holdout_exists = Path::new(HOLDOUT_DIR).exists();
    let holdout = if holdout_exists {
        tabular_data::resolve_dataset(HOLDOUT_DIR).unwrap()
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
    if !holdout_exists {
        println!(
            "  ⚠ no {} — the guardrail will score SYNTHETIC random labels (chance by construction)",
            HOLDOUT_DIR
        );
    }

    // ── STEP B — fitness, trainer, guardrail.
    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
    let trainer = MnistTrainer::new(LABEL_SMOOTHING).with_holdout(holdout, HOLDOUT_ROWS);

    // ── STEP C — the config surface.
    let pop = cli.smoke.pop.unwrap_or(POP_SIZE);
    let max_steps = cli.smoke.max_steps.or(MAX_STEPS);
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
        .set_stop_max_steps(max_steps)
        .set_run_checkpoint_every(CHECKPOINT_EVERY)
        .set_elite_count(5)
        .set_elite_freeze(true)
        .set_run_smoothing_window(5)
        .set_crossover_prob(0.5)
        .set_crossover_rolls(pop / 2)
        // --- Challenge (anti-plateau): TRAINER-owned since 2026-10-10 —
        // `CHALLENGE_PROB` lives on the trainer and rolls inside `train_step`
        // via `gras::trainer::effective_challenge_prob`. No engine knob.
        // --- Mutation (both modes) ---
        .set_mutate_prob(MUTATE_PROB) // per-roll chance a random immigrant is inserted
        .set_mutate_rolls(pop / 5)
        // --- Post-race pruner ---
        .set_pruner_enabled(POP_PRUNER)
        .set_pruner_method(gras::engine::config::PopPrunerMethod::Hard)
        .set_pruner_steps(PRUNER_STEPS)
        // --- Guardrail (post-race honesty check) ---
        .set_elite_guardrail_matches(HOLDOUT_MATCHES) // fresh holdout draws behind the champion verdict
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
    let config = builder.build();

    // ── STEP D — run, then the guardrail.
    let mut engine = TabularEngine::from_spec(gras::engine::RunSpec::tabular(
        data_dir,
        config,
        fitness,
        trainer,
        RUN_SEED,
        None::<&str>, // run dir: the engine picks results/<timestamp>
    ))?;
    println!("Run Dir: {}", engine.run_dir().display());
    match engine.run() {
        Ok(reason) => println!("Race stopped successfully: {reason:?}"),
        Err(e) => eprintln!("Race encountered an error: {e}"),
    }

    match engine.guardrail() {
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
