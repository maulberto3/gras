// NOTE: `source env_setup.sh` in the shell first — this example links against that environment.

//! MNIST — the tabular showcase: a gras-format dataset, one `TabularStep` trainer, config → run.
//! Sections: 1 data, 2 trainer, 3 stop rule, 4 `main()`; **match** = batch, **turn** = net's step.
//! It reads the binary dataset at `data/mnist` (`train/` + `test/`); `--synthetic` opts into
//! random-label smoke data, and without a dataset the run REFUSES instead of faking one.

mod common;

use std::path::{Path, PathBuf};

use common::{die, EngineArgs};
use gras::flodl::tensor::Result as TensorResult;
use gras::prelude::*;
use gras::utils::race_steps::{deterministic_train_step, eval_one_step};
use gras::utils::{score, tabular_data};

// ═══ SECTION 1 — THE DATA (a gras-format dataset on disk) ═══════════════

/// Dataset dir — a `{train,test}` parent (explicit split) or a single pool (seeded internal split).
const DATA_DIR: &str = "data/mnist";

/// Random-label smoke data — nothing to download, and nothing to trust: only
/// `--synthetic` builds it, and its fitness says nothing about MNIST.
fn synthetic_classification(
    n: usize,
    in_dim: usize,
    out_dim: usize,
    seed: u64,
    device: Device,
) -> gras::Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let inputs: Vec<f32> = (0..n * in_dim).map(|_| rng.f32()).collect();
    let inputs = Tensor::from_f32(&inputs, &[n as i64, in_dim as i64], device)
        .unwrap_or_else(|e| die(format!("synthetic inputs: {e}")));
    let mut targets = vec![0.0f32; n * out_dim];
    for row in 0..n {
        targets[row * out_dim + rng.usize(0..out_dim)] = 1.0;
    }
    let targets = Tensor::from_f32(&targets, &[n as i64, out_dim as i64], device)
        .unwrap_or_else(|e| die(format!("synthetic targets: {e}")));
    gras::Dataset::from_tensors(inputs, targets)
        .unwrap_or_else(|e| die(format!("synthetic dataset: {e}")))
}

/// Resolve the dataset and return `(inputs, outputs)` — the widths the run's topologies are built
/// with. The dataset is REQUIRED: a missing dir is an error, and only `--synthetic` fabricates one.
fn dataset_widths(data_dir: &Path, synthetic: bool) -> (usize, usize) {
    // Two layouts: a `{train,test}` parent (explicit split) or one pool (seeded internal split).
    let explicit_layout = data_dir.join("train").exists() && data_dir.join("test").exists();
    let has_pool = data_dir.join("inputs.bin").exists()
        || data_dir.join("inputs.csv").exists()
        || data_dir.join("flodl_data").join("inputs.bin").exists();
    if !explicit_layout && !has_pool {
        // A fabricated set is a SMOKE TEST: its labels are random, so a fitness from it says
        // nothing about MNIST — it must never be produced by accident.
        if !synthetic {
            die(format!(
                "no dataset at {} — expected `train/` + `test/` (each holding inputs.bin|inputs.csv + \
                 targets.bin|targets.csv) or a single pool in the dir; produce the MNIST binaries with \
                 the producer in data/mnist/mnist_data.rs, or pass --synthetic for a random-label \
                 smoke test",
                data_dir.display()
            ));
        }
        println!(
            "  {} not found — generating SYNTHETIC random-label data (smoke test only, not MNIST)",
            data_dir.display()
        );
        let ds = synthetic_classification(1024, 784, 10, 42, gras::auto_device());
        tabular_data::save_dataset(data_dir, &ds)
            .unwrap_or_else(|e| die(format!("saving {}: {e}", data_dir.display())));
    }
    // Dims peek: the train side of the explicit layout, or the single pool.
    let peeked = if explicit_layout {
        let (train, _test) = tabular_data::resolve_train_test_datasets(data_dir)
            .unwrap_or_else(|e| die(format!("{}: {e}", data_dir.display())));
        train
    } else {
        tabular_data::resolve_dataset(data_dir)
            .unwrap_or_else(|e| die(format!("{}: {e}", data_dir.display())))
    };
    let (d_in, d_out) = peeked.dims();
    drop(peeked);
    if explicit_layout {
        println!(
            "  explicit train/test layout at {} — eval rows come ONLY from test/ (no train/eval overlap by construction)",
            data_dir.display()
        );
    } else {
        println!(
            "  single-pool dataset at {} — the run seed splits train/eval internally",
            data_dir.display()
        );
    }
    (d_in, d_out)
}

// ═══ SECTION 2 — THE TRAINER (one `TabularStep` impl) ═══════════════════

/// One train batch + one eval batch per step, Adam, label-smoothed cross-entropy.
const LEARNING_RATE: f32 = 1e-3;
/// Gradient-norm clip before the optimizer step; 0 disables it.
const GRAD_CLIP: f32 = 2.0;
/// Rows per train batch, drawn from the engine's shared stream.
const BATCH_SIZE: usize = 32;
/// Rows per eval batch — the ranking signal; raising it resolves fitness more finely.
const EVAL_BATCH_SIZE: usize = 64;
/// Label smoothing in the cross-entropy target (0 = hard labels).
const LABEL_SMOOTHING: f32 = 0.1;
/// Peak input jitter std on a challenge step, scaled by `1 − last_accuracy`, so a weak net
/// explores hard and a strong net is barely disturbed.
const JITTER_SCALE: f32 = 0.3;

/// The crate's `TabularStep` over MNIST at this example's knobs.
struct MnistTrainer {
    loss_fn: gras::trainer::BoxedLossFn,
    learning_rate: f32,
    grad_clip: f32,
    train_batch: usize,
    eval_batch: usize,
    smoothing: f32,
    /// Last eval accuracy (from the previous step's report) — the metric that scales challenge jitter.
    last_accuracy: f32,
    /// Anti-plateau knob (see `CHALLENGE_PROB`): every input value rolls at the decayed rate.
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
            // The tabular arm's anti-plateau response: element-level jitter rolled in `train_step`.
            "challenge_prob": self.challenge_prob,
            "challenge_decay": "linear",
            "jitter_scale": JITTER_SCALE,
        }))
    }

    /// The jitter strength is scaled by `1 − last_accuracy`, so it must survive a resume.
    fn state(&self, _net_hash: &str) -> Option<serde_json::Value> {
        Some(serde_json::json!({ "last_accuracy": self.last_accuracy }))
    }

    /// Read it back on resume, before the net's next step.
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
        // CHALLENGE: a decayed per-input roll, stronger for weak nets, linear to 0
        // over the NET's OWN step count (`net_step`) — a late-born net is not
        // handed the dregs of the knob the run clock would have left.
        let p_eff = gras::trainer::effective_challenge_prob(
            self.challenge_prob,
            ctx.env.net_step,
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

/// Jitter each input value independently at `prob`, adding Gaussian noise of std `strength`.
/// Seeded from `(net_seed, step)`, so a replay draws identically.
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

// ═══ SECTION 3 — THE STOP RULE (the bar, not a re-score) ════════════════

// `best_smoothed_fitness` is the window CENTRE, so the bar below ends the run on accuracy the
// window actually believes, not on one lucky batch.

// ═══ SECTION 4 — main(): data → config → spec → run ═════════════════════

// `RunSpec::tabular` declares the mode; `Fitness::new` takes the score the report computes.

/// Name recorded in `engine.json` (informational only).
const RUN_NAME: &str = "mnist";
/// Per-(net, step) chance an anti-plateau challenge fires (see `train_step`).
const CHALLENGE_PROB: f32 = 0.15;

/// Every knob the run has; each flag carries its own default, and `--help` lists them all.
#[derive(clap::Parser, Debug)]
#[command(
    name = "mnist",
    about = "MNIST tabular race: every engine knob is a flag, each one carrying the value the run uses."
)]
struct Cli {
    /// Every engine knob shared with the other example (`examples/common/mod.rs`).
    #[command(flatten)]
    engine: EngineArgs,

    /// Pin the run seed. Omitted, a fresh one is DRAWN and printed, so the run is
    /// still replayable; `--resume` uses the recorded seed instead.
    #[arg(long, value_name = "U64")]
    seed: Option<u64>,

    /// Candidates one mutation may spend clearing the floor; the most novel refusal is taken anyway.
    #[arg(long, value_name = "N", default_value_t = 5)]
    mutation_novelty_draws: usize,

    /// Drift on a crossover child's inherited genes — mask bits, activations, combine/standardize,
    /// port activations, hidden width, the `input_reach` wiring gene and dropout (`0` = exact
    /// inheritance).
    #[arg(long, value_name = "P", default_value_t = 0.1)]
    drift: f32,

    /// Dropout probability applied when building the run's networks.
    #[arg(long, value_name = "P", default_value_t = 0.20)]
    dropout: f32,

    /// Name recorded in `engine.json` (informational only).
    #[arg(long, value_name = "NAME", default_value = RUN_NAME)]
    run_name: String,

    /// Hidden nodes per net: the upper sampling bound.
    #[arg(long, value_name = "N", default_value_t = 15)]
    hidden_max: usize,

    /// Inputs per node: the upper sampling bound.
    #[arg(long, value_name = "N", default_value_t = 15)]
    inputs_max: usize,

    /// Outputs per node: the upper sampling bound.
    #[arg(long, value_name = "N", default_value_t = 15)]
    outputs_max: usize,

    /// Counter stride of the hidden-width range (`16` = 16, 32, 48, …).
    #[arg(long, value_name = "N", default_value_t = 16)]
    hidden_dim_stride: usize,

    /// Dataset directory (any gras-format dataset works; unset: the `DATA_DIR` const).
    #[arg(long, value_name = "DIR")]
    data_dir: Option<PathBuf>,

    /// Generate random-label data when the dataset is missing — SMOKE TEST only, never a real race.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = false
    )]
    synthetic: bool,
}

/// The body reads `cli.pop`, `cli.max_steps`, … — the overlay never hides a knob.
impl std::ops::Deref for Cli {
    type Target = EngineArgs;
    fn deref(&self) -> &EngineArgs {
        &self.engine
    }
}

/// Wire the config → run the race; prints the verdict.
fn main() {
    let cli = <Cli as clap::Parser>::parse();
    // `--log-level` wins over the default reporting shape; the sinks are installed here.
    let log_level = common::init_logging(&cli.log_level);

    // STEP A — data: resolve (or generate) it first; the data owns the widths.
    let data_dir = cli
        .data_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DATA_DIR));
    let (d_in, d_out) = dataset_widths(&data_dir, cli.synthetic);

    // STEP B — fitness: the reported accuracy IS the ranking input, maximized.
    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
    let trainer = MnistTrainer::new(LABEL_SMOOTHING);

    // Informative metric: mean top-1/runner-up logit gap — one extra history.csv column.
    // Unused while `.set_run_metrics` is commented out below; uncomment both for the column.
    let _metrics = vec![Metric::custom(
        "top1_margin",
        |pred: &Variable, y: &Variable| {
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
        },
    )];

    let config = common::engine_config(TabularRaceConfig::builder(), &cli.engine, log_level)
        // This example's own half: the flags whose default is its identity.
        .set_run_name(cli.run_name.clone())
        .set_topology_drift(cli.drift)
        .set_topology_max_hidden_num_nodes(cli.hidden_max)
        .set_topology_max_inputs_per_node(cli.inputs_max)
        .set_topology_max_outputs_per_node(cli.outputs_max)
        .set_topology_hidden_dim_stride(cli.hidden_dim_stride)
        .set_topology_dropout_prob(cli.dropout)
        .set_mutation_novelty_draws(cli.mutation_novelty_draws)
        // `best_smoothed_fitness` is the window CENTRE, so the bar is 0.85 on accuracy.
        // .set_stop_custom(|snap| snap.best_smoothed_fitness > 0.85)
        // The data's own widths: 784 pixels in, 10 labels out.
        .set_topology_input_dim(d_in)
        .set_topology_output_dim(d_out)
        .set_topology_feature_split_pool(cli.feature_split_pool.clone())
        .set_topology_combine_op_pool(cli.combine_op_pool.clone())
        .set_topology_activation_pool(cli.activation_pool.clone())
        .set_topology_standardize_op_pool(cli.standardize_pool.clone())
        .build();

    // `--resume` continues a recorded run (its record wins); `--run-dir` names where a fresh one writes.
    let engine = match &cli.resume {
        // The record supplies pop, band and search space; a `--max-steps` here EXTENDS the budget.
        Some(dir) => TabularEngine::resume_from(dir.clone(), fitness, trainer).map(|mut engine| {
            // A resume EXTENDS a budget, never shrinks it: the flag is a floor on a recorded run.
            let budget = common::extended_budget(engine.max_steps(), cli.max_steps);
            engine.set_stop_max_steps(budget);
            engine
        }),
        None => {
            // A fresh run with no `--seed` draws one — and SAYS so, because a
            // random run is only reproducible if the number reaches the reader.
            let run_seed = common::run_seed(cli.seed);
            TabularEngine::from_spec(RunSpec::tabular(
                &data_dir,
                config,
                fitness,
                trainer,
                Some(run_seed),
                cli.run_dir.clone(), // engine picks results/<timestamp> when unset
            ))
        }
    };

    let mut engine = match engine {
        Ok(engine) => engine,
        // Structural impossibilities are refused by name — print the reason as a usage error.
        Err(e) => die(format!("engine construction failed: {e}")),
    };

    println!("Run Dir: {}", engine.run_dir().display());
    match engine.run() {
        Ok(reason) => println!("Race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }
}
