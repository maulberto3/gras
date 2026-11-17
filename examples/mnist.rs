// NOTE: `source env_setup.sh` in the shell first — this example links against that environment.

//! MNIST — the tabular showcase: a gras-format dataset, one `TabularStep` trainer, config → run.
//! Sections: 1 data, 2 trainer, 3 stop rule, 4 `main()`; **match** = batch, **turn** = net's step.
//! It reads the binary dataset at `data/mnist` (`train/` + `test/`); `--synthetic` opts into
//! random-label smoke data, and without a dataset the run REFUSES instead of faking one.

use std::path::{Path, PathBuf};

use gras::engine::config::{MeanType, TrainTicket};
use gras::flodl::tensor::Result as TensorResult;
use gras::prelude::*;
use gras::utils::race_steps::{deterministic_train_step, eval_one_step};
use gras::utils::{score, tabular_data};

// ═══ SECTION 1 — THE DATA (a gras-format dataset on disk) ═══════════════

/// Dataset dir — a `{train,test}` parent (explicit split) or a single pool (seeded internal split).
const DATA_DIR: &str = "data/mnist";

/// Print `msg` as a usage error and leave with code 2 — the examples' one failure path.
fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

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
        // CHALLENGE: a decayed per-input roll, stronger for weak nets, linear to 0 at the budget.
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

/// Live nets in the race (default; `--pop` overrides).
const POP: usize = 100;
/// Crossover turnover: the share of the live field replaced per clock, dealt as `round(share × pop)` rolls.
const CROSSOVER_TURNOVER: f32 = 0.05;
/// MUTATION turnover default, the mutation channel's own share of the field per clock.
const MUTATION_TURNOVER: f32 = 0.01;
/// Step budget for a full run: `--max-steps N` runs exactly N clocks before Stop.
const RACE_STEPS: usize = 50;
/// Steps between population checkpoints — the era rotation and the CSV flush ride this.
const CHECKPOINT_EVERY: usize = 50;
/// Name recorded in `engine.json` (informational only).
const RUN_NAME: &str = "mnist";
/// Seed a fresh run uses when `--seed` is not passed; the record still wins on `--resume`.
const RUN_SEED: Option<u64> = Some(16);
/// Per-(net, step) chance an anti-plateau challenge fires (see `train_step`).
const CHALLENGE_PROB: f32 = 0.15;

/// Every knob the run has; each flag carries its own default, and `--help` lists them all.
#[derive(clap::Parser, Debug)]
#[command(
    name = "mnist",
    about = "MNIST tabular race: every engine knob is a flag, each one carrying the value the run uses."
)]
struct Cli {
    /// Live networks in the race (default: the `POP` const).
    #[arg(long, value_name = "N", default_value_t = POP)]
    pop: usize,
    /// Stop after this many steps; on `--resume` it EXTENDS the recorded budget.
    #[arg(long, value_name = "N", default_value_t = RACE_STEPS)]
    max_steps: usize,
    /// Pin the run seed (otherwise `RUN_SEED`); it is always recorded in `engine.json`.
    #[arg(long, value_name = "U64")]
    seed: Option<u64>,
    /// Lottery FLOOR `p_min`: the chance the WORST-graded net is paid each clock (`p_min == p_max` pays all alike).
    #[arg(long, value_name = "P", default_value_t = 0.25)]
    ticket_p_min: f32,
    /// Lottery CEILING `p_max`: the chance the BEST-graded net is paid; everyone between is interpolated.
    #[arg(long, value_name = "P", default_value_t = 0.75)]
    ticket_p_max: f32,
    /// LEARN-ONLY: a lost ticket costs the weight UPDATE, not the clock — the net still measures.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = false
    )]
    learn_only: bool,
    /// Crossover victim: `highest_ticket_chance` (the coin's favourite), `lowest_fitness`, `lowest_fitness_most_updates` (evidence) or `random`.
    #[arg(long, value_name = "POLICY", default_value_t = CrossCullPolicy::default())]
    crossover_cull_policy: CrossCullPolicy,
    /// Mutation/immigrant victim: the same four arms as the crossover channel.
    #[arg(long, value_name = "POLICY", default_value_t = MutationCullPolicy::default())]
    mutation_cull_policy: MutationCullPolicy,
    /// Opening clocks with NO culls, no births and no recovery streak: evolution starts at clock N.
    #[arg(long, value_name = "N", default_value_t = 0)]
    init_probation: usize,
    /// Window stats scoped to the CURRENT weights: n counts only readings since the last real update.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    policy_scoped_fitness: bool,
    /// Ranking window K: how many readings are averaged (under `--policy-scoped-fitness`, of ONE policy).
    #[arg(long, value_name = "K", default_value_t = gras::engine::config::SMOOTHING_WINDOW)]
    smoothing_window: usize,
    /// Averaging family: `arithmetic`, `signed_log` (spike-damped) or `median` — the default, where no single reading moves it.
    #[arg(long, value_name = "FAMILY", default_value_t = MeanType::Median)]
    mean_type: MeanType,
    /// At stop, write `worst-<hash>-<fitness>.md` for the `spent`-ranked nets, plus a `worst.md` index.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    worst_save_topology: bool,
    /// Lottery legs: which fit place pays the most and which updates side it multiplies in; a typo is a CLI error.
    #[arg(
        long,
        value_name = "LEGS",
        default_value_t = gras::engine::config::DEFAULT_TRAIN_TICKET
    )]
    train_ticket: TrainTicket,
    /// Crossover turnover: the share of the live field replaced per clock (`0` switches the channel off).
    #[arg(long, value_name = "RATE", default_value_t = CROSSOVER_TURNOVER)]
    crossover_turnover: f32,
    /// Mutation turnover: the mutation channel's own share, same contract as `--crossover-turnover`.
    #[arg(long, value_name = "RATE", default_value_t = MUTATION_TURNOVER)]
    mutation_turnover: f32,
    /// Mutation novelty floor Δ: a new immigrant must sit this far (Jaccard) from the nearest blueprint; `0` = off.
    #[arg(long, value_name = "DELTA", default_value_t = 0.3)]
    mutation_novelty_min: f32,
    /// Candidates one mutation may spend clearing the floor; the most novel refusal is taken anyway.
    #[arg(long, value_name = "N", default_value_t = 5)]
    mutation_novelty_draws: usize,
    /// Log level (default: the `LOG_LEVEL` const). `none`/`summ`/`minimal` pick
    /// the shape; `debug`/`trace` add the engine's per-roll detail.
    #[arg(
        long,
        value_name = "LEVEL",
        value_parser = ["none", "summ", "minimal", "debug", "trace"],
        default_value = "summ"
    )]
    log_level: String,
    /// Continue the recorded run here: its seed wins over `--seed`, and `--max-steps` EXTENDS the recorded budget.
    #[arg(long, value_name = "DIR")]
    resume: Option<PathBuf>,
    /// Drift on a crossover child's inherited genes (`0` = exact inheritance).
    #[arg(long, value_name = "P", default_value_t = 0.1)]
    drift: f32,
    /// Dropout probability applied when building the run's networks.
    #[arg(long, value_name = "P", default_value_t = 0.20)]
    dropout: f32,
    /// Name recorded in `engine.json` (informational only).
    #[arg(long, value_name = "NAME", default_value = RUN_NAME)]
    run_name: String,
    /// Steps between checkpoints — weights, frontier states and the CSV flush.
    #[arg(long, value_name = "N", default_value_t = CHECKPOINT_EVERY)]
    checkpoint_every: usize,
    /// Write each net's optimizer state (Adam moments) beside its weights, so `--resume` restores them exactly.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    save_optim: bool,
    /// Mutation probation: cull immunity for an immigrant's first N updates (`U`, not clocks).
    #[arg(long, value_name = "N", default_value_t = 0)]
    mutation_probation: usize,
    /// Crossover probation: the same shield for a crossover child's first N updates.
    #[arg(long, value_name = "N", default_value_t = 0)]
    crossover_probation: usize,
    /// Recovery updates: extra real updates inside one clock after a worse reading (`0` = off).
    #[arg(long, value_name = "N", default_value_t = 0)]
    recovery_retries: usize,
    /// Recovery tolerance: the fraction of its standing a reading may lose before the streak fires.
    #[arg(long, value_name = "F", default_value_t = 0.0)]
    recovery_tol: f32,
    /// Recovery draw: `true` retries the clock's own draw, `false` walks one draw further.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    recovery_same_seed: bool,
    /// Elite count saved at stop (a rank, not a seat — a champion shields nothing from a cull).
    #[arg(long, value_name = "N", default_value_t = POP / 10)]
    champion_count: usize,
    /// At stop, save each champion's topology markdown (`champion-<hash>-<fitness>.md`).
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    champion_save_topology: bool,
    /// Seed the founders from the newest compatible run's elites.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    champion_previous_load: bool,
    /// Where that inheritance looks: a run dir or a dir of run dirs (unset: `results/`).
    #[arg(long, value_name = "DIR")]
    champion_previous_dir: Option<PathBuf>,
    /// Write `history.csv` + `attempts.csv` (the lossless event logs).
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    csv_export: bool,
    /// Also write `<run_dir>/telemetry.jsonl`, one JSON record per event.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = false
    )]
    trace_file: bool,
    /// Parent selection for the crossover draw: `tournament` or `roulette`.
    #[arg(
        long,
        value_name = "METHOD",
        value_parser = ["tournament", "roulette"],
        default_value = "tournament"
    )]
    crossover_selection: String,
    /// Tournament width when `--crossover-selection tournament`.
    #[arg(long, value_name = "K", default_value_t = 2)]
    tournament_size: usize,
    /// The FITTER drawn parent becomes the crossover BASE (off: the first drawn parent is).
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    crossover_fitter_base: bool,
    /// Crossover operator pool, drawn per attempt; empty = both.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    crossover_ops_pool: Vec<String>,
    /// Feature-split draw order for a newborn (`none`,`random`,`exclusive`); `none` = every feature hits every port.
    #[arg(
        long,
        value_name = "LIST",
        value_delimiter = ',',
        default_values_t = [String::from("none")]
    )]
    feature_split_pool: Vec<String>,
    /// Combine-op pool (`add`,`mean`,`max`,…); empty = every combine op.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    combine_op_pool: Vec<String>,
    /// Activation pool (`relu`,`gelu`,…); empty = the full default pool.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    activation_pool: Vec<String>,
    /// Standardize-op pool (`none`,`layer`,…); empty = the full default pool.
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    standardize_pool: Vec<String>,
    /// Hidden nodes per net: the lower sampling bound.
    #[arg(long, value_name = "N", default_value_t = 2)]
    hidden_min: usize,
    /// Hidden nodes per net: the upper sampling bound.
    #[arg(long, value_name = "N", default_value_t = 15)]
    hidden_max: usize,
    /// Inputs per node: the lower sampling bound.
    #[arg(long, value_name = "N", default_value_t = 2)]
    inputs_min: usize,
    /// Inputs per node: the upper sampling bound.
    #[arg(long, value_name = "N", default_value_t = 15)]
    inputs_max: usize,
    /// Outputs per node: the lower sampling bound.
    #[arg(long, value_name = "N", default_value_t = 2)]
    outputs_min: usize,
    /// Outputs per node: the upper sampling bound.
    #[arg(long, value_name = "N", default_value_t = 15)]
    outputs_max: usize,
    /// Hidden width for a newborn: the lower bound of the sampled range.
    #[arg(long, value_name = "N", default_value_t = 16)]
    hidden_dim_min: usize,
    /// Hidden width for a newborn: the upper bound of the sampled range.
    #[arg(long, value_name = "N", default_value_t = 64)]
    hidden_dim_max: usize,
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
    /// Where a NEW run writes (default: `results/<timestamp>`). Output only —
    /// continuing an existing run is `--resume`.
    #[arg(long, value_name = "DIR")]
    run_dir: Option<PathBuf>,
}

/// Wire the config → run the race; prints the verdict.
fn main() {
    // Default reporting shape when `--log-level` is not passed.
    const LOG_LEVEL: LogLevel = LogLevel::Summ;

    let cli = <Cli as clap::Parser>::parse();

    // The CLI level wins over the const when passed; `debug`/`trace` add the
    // engine's per-roll detail on top of the same reporting shape.
    let (log_level, log_verbosity) =
        LogLevel::parse_verbose(&cli.log_level).unwrap_or((LOG_LEVEL, None));
    // Install the run's log sinks — without this the console stays silent and
    // the run LOOKS hung while the race grinds.
    gras::engine::logging::init(log_level, log_verbosity);

    // STEP A — data: resolve (or generate) it first; the data owns the widths.
    let data_dir = cli
        .data_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DATA_DIR));
    let (d_in, d_out) = dataset_widths(&data_dir, cli.synthetic);

    // STEP B — fitness: the reported accuracy IS the ranking input, maximized.
    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
    let trainer = MnistTrainer::new(LABEL_SMOOTHING);

    // Parent selection: a tournament of `--tournament-size` parents, or a roulette draw.
    let selection = match cli.crossover_selection.as_str() {
        "roulette" => SelectionMethod::Roulette,
        _ => SelectionMethod::Tournament {
            tournament_size: cli.tournament_size,
        },
    };

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

    let mut builder = TabularRaceConfig::builder()
        // SETUP
        .set_run_name(cli.run_name.clone())
        .set_run_log_level(log_level)
        .set_run_checkpoint_every(cli.checkpoint_every)
        .set_run_save_optim(cli.save_optim)
        .set_run_csv_export(cli.csv_export)
        .set_run_trace_file(cli.trace_file)
        // .set_run_metrics(_metrics)`.
        // POP
        .set_pop_size(cli.pop)
        .set_pop_init_probation_steps(cli.init_probation)
        .set_pop_recovery_retries(cli.recovery_retries)
        .set_pop_recovery_tol(cli.recovery_tol)
        .set_pop_recovery_same_seed(cli.recovery_same_seed)
        // FITNESS
        .set_fitness_smoothing_window(cli.smoothing_window)
        .set_fitness_policy_scoped(cli.policy_scoped_fitness)
        .set_fitness_mean_type(cli.mean_type)
        // TRAIN
        .set_ticket_learn_only(cli.learn_only)
        .set_ticket_p_min(cli.ticket_p_min)
        .set_ticket_p_max(cli.ticket_p_max)
        .set_train_ticket(cli.train_ticket)
        // STOP
        .set_stop_max_steps(cli.max_steps)
        // `best_smoothed_fitness` is the window CENTRE, so the bar is 0.85 on accuracy.
        .set_stop_custom(|snap| snap.best_smoothed_fitness > 0.85)
        // CROSSOVER
        .set_crossover_share(cli.crossover_turnover)
        .set_crossover_cull_policy(cli.crossover_cull_policy)
        .set_crossover_selection(selection)
        .set_crossover_fitter_base(cli.crossover_fitter_base)
        .set_crossover_ops_pool(cli.crossover_ops_pool.clone())
        .set_crossover_probation_steps(cli.crossover_probation)
        // MUTATION — its own turnover knob.
        .set_mutation_share(cli.mutation_turnover)
        .set_mutation_cull_policy(cli.mutation_cull_policy)
        .set_mutation_novelty_min(cli.mutation_novelty_min)
        .set_mutation_novelty_draws(cli.mutation_novelty_draws)
        .set_mutation_probation_steps(cli.mutation_probation)
        // TOPOLOGY
        .set_topology_drift(cli.drift)
        .set_topology_min_hidden_num_nodes(cli.hidden_min)
        .set_topology_max_hidden_num_nodes(cli.hidden_max)
        .set_topology_min_inputs_per_node(cli.inputs_min)
        .set_topology_max_inputs_per_node(cli.inputs_max)
        .set_topology_min_outputs_per_node(cli.outputs_min)
        .set_topology_max_outputs_per_node(cli.outputs_max)
        .set_topology_hidden_dim_range(cli.hidden_dim_min, cli.hidden_dim_max)
        .set_topology_hidden_dim_stride(cli.hidden_dim_stride)
        .set_topology_dropout_prob(cli.dropout)
        // The data's own widths: 784 pixels in, 10 labels out.
        .set_topology_input_dim(d_in)
        .set_topology_output_dim(d_out)
        .set_topology_feature_split_pool(cli.feature_split_pool.clone())
        .set_topology_combine_op_pool(cli.combine_op_pool.clone())
        .set_topology_activation_pool(cli.activation_pool.clone())
        .set_topology_standardize_op_pool(cli.standardize_pool.clone())
        // ELITES
        .set_champion_previous_load(cli.champion_previous_load)
        .set_champion_count(cli.champion_count)
        .set_champion_save_topology(cli.champion_save_topology)
        .set_worst_save_topology(cli.worst_save_topology);

    if let Some(dir) = &cli.champion_previous_dir {
        builder = builder.set_champion_previous_dir(dir.clone());
    }
    let config = builder.build();

    // `--resume` continues a recorded run (its record wins); `--run-dir` names where a fresh one writes.
    let engine = match &cli.resume {
        // The record supplies pop, band and search space; a `--max-steps` here EXTENDS the budget.
        Some(dir) => TabularEngine::resume_from(dir.clone(), fitness, trainer).map(|mut engine| {
            // A resume EXTENDS a budget, never shrinks it: the flag is a floor on a recorded run.
            let recorded = engine.max_steps().unwrap_or(cli.max_steps);
            engine.set_stop_max_steps(recorded.max(cli.max_steps));
            engine
        }),
        None => TabularEngine::from_spec(RunSpec::tabular(
            &data_dir,
            config,
            fitness,
            trainer,
            cli.seed.or(RUN_SEED),
            cli.run_dir.clone(), // engine picks results/<timestamp> when unset
        )),
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
