//! Bring your own trainer — a custom `Trainer` for the step-race engine.
//!
//! The engine is **training-agnostic**: it orchestrates the population
//! (step clock, smoothed-fitness culling, checkpoint gates, stop criteria) and
//! hands each net to *your* training scheme once per step clock. This
//! example implements `WarmupSgdTrainer`, a scheme that deviates from the
//! default `TabularTrainer` in three ways at once:
//!
//! 1. **Different optimizer** — SGD with momentum instead of Adam
//!    (`make_optimizer` is where optimizer choice + learning rate live).
//! 2. **LR warmup** — the learning rate ramps linearly over the first
//!    `warmup_steps` step clocks (state lives on the trainer; the engine
//!    never sees it).
//! 3. **Skip eval most steps** — eval runs every `eval_every` steps; on
//!    off-steps the report carries `eval_loss: None` and the previous
//!    fitness (the engine stores it as-is and skips missing evals in its
//!    rollups).
//!
//! It also demonstrates the *minimum* contract surface: two methods, and
//! the per-(net, step) seeding pattern (`seed_step_randomness`) that keeps
//! dropout reproducible across catch-up replay and resume. The engine
//! *probes* the per-step clause in debug builds — the report must describe
//! the net's state at the end of this step — so a stale report panics early
//! instead of corrupting the race.
//!
//! Everything else (loss, fitness, topology, config) is the same public
//! surface as any other run: the training scheme is the only seam.
//!
//! Run: `source env_setup.sh && cargo run --example custom_trainer`

use std::path::Path;

use flodl::Tensor;
use flodl::nn::Module;
use flodl::nn::optim::Optimizer;

use gras::Variable;
use gras::engine::fitness::{Direction, Fitness, Metric};
use gras::engine::{RaceConfig, RaceEngine};
use gras::graph::network::Network;
use gras::graph::topology::TopologyOptions;
use gras::trainer::{StepReport, StepTrainer, TabularContext, TabularStep};
use gras::utils::{tabular_data, score};

// ── 1. The custom trainer ────────────────────────────────────────────────────

/// SGD + momentum + linear LR warmup + eval every N steps.
///
/// MINIMUM contract (both methods, ~30 lines total):
/// - `make_optimizer(net)` → the optimizer this scheme trains with. The
///   engine calls it when a net enters the population and on resume; the
///   optimizer state (momentum buffers) carries across steps because the
///   engine persists and re-feeds the same `Box<dyn Optimizer>`.
/// - `train_step(net, optimizer, step, ctx)` → one step-clock of training,
///   returning a `StepReport` describing the net AFTER this step's updates
///   (the per-step clause). Train the net in place — never rebuild it.
///
/// OPTIONAL (all shown here): extra fields on the struct (warmup schedule,
/// eval cadence), internal mutable state (a step counter), skipped evals.
struct WarmupSgdTrainer {
    /// This scheme's loss (supervised paradigm — owned by the trainer).
    loss_fn: Box<dyn Fn(&Variable, &Variable) -> flodl::tensor::Result<Variable> + Send + Sync>,
    /// Peak SGD learning rate, reached after `warmup_steps`.
    peak_lr: f32,
    /// SGD momentum.
    momentum: f32,
    /// Gradient-norm clip per step (0 = off).
    grad_clip: f32,
    /// Linear warmup length in step clocks.
    warmup_steps: usize,
    /// Eval every N step clocks (1 = every step, like TabularTrainer).
    eval_every: usize,
    /// Last reported fitness — reused on skipped-eval steps so the engine's
    /// ranking always has a value.
    last_fitness: f32,
}

impl WarmupSgdTrainer {
    fn new(
        loss_fn: impl Fn(&Variable, &Variable) -> flodl::tensor::Result<Variable>
        + Send
        + Sync
        + 'static,
        peak_lr: f32,
        momentum: f32,
    ) -> Self {
        Self {
            loss_fn: Box::new(loss_fn),
            peak_lr,
            momentum,
            grad_clip: 1.0,
            warmup_steps: 10,
            eval_every: 3,
            last_fitness: 0.0,
        }
    }

    /// Linear warmup: lr(t) = peak * min(1, (t+1)/warmup).
    fn lr_at(&self, step: usize) -> f64 {
        (self.peak_lr * ((step + 1) as f32 / self.warmup_steps as f32).min(1.0)) as f64
    }
}

impl StepTrainer for WarmupSgdTrainer {
    // Called by the engine at every net birth (initial population, crossover
    // child, random immigrant) and on resume rebuilds. The lr baked in here
    // is the *warmup-schedule* lr for the net's birth step — the momentum
    // buffers it starts with are fresh, matching the group's step-0 state.
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        // Birth happens at various clocks; warmup is keyed off the *step*
        // passed to train_step below, so a mid-run entrant just starts at
        // whatever point of the ramp the clock is at (read via the next
        // train_step call). A conservative constant here is fine for the
        // demo; see the note in train_step.
        Box::new(flodl::nn::optim::SGD::new(
            &net.parameters(),
            self.peak_lr as f64 * 0.1, // conservative start-of-ramp lr
            self.momentum as f64,
        ))
    }

    // OPTIONAL: record the scheme's hyperparameters in `engine.json` so the run
    // is reproducible without reading this file. Free-form JSON — the engine
    // persists it verbatim and never interprets it.
    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "warmup-sgd",
            "optimizer": "sgd",
            "peak_lr": self.peak_lr,
            "momentum": self.momentum,
            "grad_clip": self.grad_clip,
            "warmup_steps": self.warmup_steps,
            "eval_every": self.eval_every,
        }))
    }
}

impl TabularStep for WarmupSgdTrainer {
    // REQUIRED: the (pred, target) loss — a supervised scheme's defining
    // choice. (RL schemes implement RlStep instead and have no loss method.)
    fn loss(&self) -> gras::trainer::LossFn<'_> {
        &*self.loss_fn
    }

    // OPTIONAL: shape the engine's shared batch stream. This scheme prefers
    // a smaller train batch (SGD noise) and a bigger eval batch (stabler
    // readings). The split RATIO is not overridable — which rows are held
    // out is engine bookkeeping protecting comparability.
    fn stream_shape(&self) -> Option<gras::trainer::StreamShape> {
        Some(gras::trainer::StreamShape {
            batch_size: 2,
            eval_batch_size: 3,
        })
    }

    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &TabularContext<'_>,
    ) -> flodl::tensor::Result<StepReport> {
        // ── determinism (required pattern) ──
        // Seed per (net, step) so dropout masks are reproducible across
        // catch-up replay and resume. Omit this and resume parity fails
        // loudly — that's the contract enforcement, not a nicety.
        gras::seed_step_randomness(ctx.net_seed, step as u64, 0);

        // ── the training move ──
        // The warmup schedule rewrites the optimizer's lr each step; SGD
        // exposes `set_learning_rate` on the Optimizer trait, so the engine's
        // long-lived optimizer box is updated in place (state preserved).
        let lr = self.lr_at(step);
        optimizer.set_lr(lr);

        // This scheme uses the shared stream + its OWN loss (a supervised
        // scheme keeps its loss internally), and scores eval via the run's
        // fitness — showing which handles are offered vs owned.
        let data = &ctx.data;
        let fitness = ctx.fitness;
        let loss_fn = self.loss();
        let batch = data.train_batch(step as u64)?;
        let train_loss = gras::train_one_step(net, optimizer, loss_fn, &batch, self.grad_clip)?;

        // ── eval cadence (optional) ──
        // Full eval only every `eval_every` steps; on other steps report no
        // eval loss and reuse the last fitness so ranking stays defined.
        let do_eval = step % self.eval_every == 0 || self.last_fitness == 0.0;
        let (eval_loss, fitness, informative) = if do_eval {
            let eval_batch = data.eval_batch(step as u64)?;
            let r = gras::eval_one_step(net, loss_fn, fitness, ctx.metrics, &eval_batch)?;
            self.last_fitness = r.fitness;
            (r.eval_loss, r.fitness, r.metrics)
        } else {
            (None, self.last_fitness, Vec::new())
        };

        Ok(StepReport {
            train_loss,
            eval_loss,
            fitness,
            informative,
            rl: None, // tabular: no environment volume to report
        })
    }
}

// ── 2. The run — plain public surface, like any other example ────────────────

fn main() {
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // Data — XOR: 4 rows, 2 features, 2 one-hot classes. Persisted so the
    // engine's deterministic-split contract holds even for tiny data.
    // Dataset: the repo's `data/` root (generated on first run). Run output:
    // beside this file, `examples/custom_trainer/run/`. Both are anchored to
    // the crate root so the working directory doesn't matter.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = root.join("data/xor");
    let run_dir = root.join("examples/custom_trainer/run");
    if !data_dir.exists() {
        let xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let ys = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]; // XOR one-hot
        let flat_x: Vec<f32> = xs.iter().flat_map(|r| r.iter().copied()).collect();
        let flat_y: Vec<f32> = ys.iter().flat_map(|r| r.iter().copied()).collect();
        let ds = tabular_data::Dataset {
            inputs: Tensor::from_f32(&flat_x, &[4, 2], gras::auto_device()).unwrap(),
            targets: Tensor::from_f32(&flat_y, &[4, 2], gras::auto_device()).unwrap(),
        };
        tabular_data::save_dataset(&data_dir, &ds).unwrap();
    }
    let peeked = tabular_data::resolve_dataset(&data_dir).unwrap();
    let _ = &peeked; // dims only; the engine loads the real copy from data_dir
    drop(peeked);

    // Loss + fitness — the loss is OURS (this trainer's business); fitness
    // is engine business (drives cull/insert ranking).
    let loss_fn = |pred: &Variable, y: &Variable| score::cross_entropy_onehot_loss(pred, y);
    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
    let metrics = vec![Metric::new("accuracy")];

    // Topology — 2 features in, 2 one-hot classes out.
    let mut topo_opts = TopologyOptions::default();
    topo_opts.input_dim = Some(2);
    topo_opts.output_dim = Some(2);
    topo_opts.min_hidden_num_nodes = 2;
    topo_opts.max_hidden_num_nodes = 4;

    // Config — note there are NO training knobs here anymore (no learning
    // rate, no grad clip): those live on the trainer below.
    let config = RaceConfig::builder()
        .set_pop_size(4)
        .set_max_steps(30)
        .set_crossover_gate_checkpoint_every(5)
        .set_topology_options(topo_opts)
        .set_additional_metrics(metrics.clone())
        .build();

    // Run — the ONLY difference from a default run is the trainer argument:
    let run_seed = 42u64;
    let mut engine = RaceEngine::new(gras::engine::RunSpec::tabular(
        data_dir,
        config,
        fitness,
        // Our scheme instead of TabularTrainer — that's the whole swap.
        WarmupSgdTrainer::new(loss_fn, 0.05, 0.9),
        Some(run_seed),
        Some(run_dir),
    ))
    .unwrap();
    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }
}
