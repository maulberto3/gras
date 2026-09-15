//! gras — MNIST Race Example
//!
//! A first-class showcase and guide for the `gras` continuous step-race
//! Neural Architecture Search (NAS) engine. This example demonstrates how to
//! configure, customize, and execute a full evolutionary step-race, showcasing
//! every major evolutionary, topological, and gating control option.
//!
//! Run:
//!   source env_setup.sh && cargo run --example mnist_race --release                          # mnist, CPU
//!   source env_setup.sh cuda && cargo run --example mnist_race --release $GRAS_FEATURES      # mnist, CUDA

use std::io::Write;
use std::path::Path;

use clap::Parser;
use flodl::nn::scheduler::CosineScheduler;
use gras::engine::fitness::{Direction, Fitness, Metric};
use gras::engine::{RaceConfig, RaceEngine};
use gras::utils::{tabular_data, score};
use gras::{RunMode, Variable};

/// CLI arguments for the MNIST race example. Stop-flag names mirror the
/// engine's builder setters 1:1 (`--max-steps` ↔ `.set_max_steps`, etc.) —
/// and the clap ArgGroup over the three stop flags enforces the same
/// one-stop-criterion rule the engine enforces at `build()`, but at parse
/// time so you find out sooner.
#[derive(Parser, Debug)]
#[command(name = "mnist", version, about)]
struct Cli {
    /// Run seed (omit = random, recorded in engine.json for repro)
    #[arg(long)]
    seed: Option<u64>,

    /// Dataset directory holding inputs.csv + targets.csv (any gras-format
    /// dataset works — dims are auto-peeked). See DATA.md for the recipe.
    #[arg(long, default_value = "data/mnist/train")]
    data_dir: String,

    /// Population size (number of live networks in the race)
    #[arg(long, default_value_t = 20)]
    pop_size: usize,

    /// Stop criterion — total global step budget (mirrors .set_max_steps).
    /// Default 10 when no stop flag is given at all.
    #[arg(long)]
    max_steps: Option<usize>,

    /// Stop criterion — stop once best net's smoothed fitness reaches this
    /// (mirrors .set_max_target_fitness). May be combined with the other
    /// stop flags: they race, and the first to fire ends the run.
    #[arg(long)]
    max_target_fitness: Option<f32>,

    /// Record a population checkpoint every N steps
    #[arg(long, default_value_t = 10)]
    checkpoint_every: usize,

    /// Per-step log verbosity: "summ", "minimal", "full", or "none"
    #[arg(long, default_value = "summ")]
    log_level: String,

    /// Post-race pruner: when a stop criterion fires, cull everything except
    /// the top-elite_count nets and keep training them solo for --pruner-steps
    /// more steps (mirrors .set_pop_pruner + .set_pop_pruner_method).
    #[arg(long)]
    pop_pruner: bool,

    /// Extra solo-training steps for the pruner phase (the 50 in Hard, 50)
    #[arg(long, default_value_t = 10)]
    pruner_steps: usize,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();
    let cli = Cli::parse();

    // ── 1. Data Setup ──────────────────────────────────────────────────────
    // The engine loads the dataset itself from data_dir; we generate synthetic
    // data if the local path is missing so the example runs out of the box.
    let data_dir = Path::new(&cli.data_dir);
    if !data_dir.exists() {
        println!("  {} not found — generating synthetic MNIST-shaped data", cli.data_dir);
        let ds = tabular_data::synthetic_classification(1024, 784, 10, 42, gras::auto_device()).unwrap();
        tabular_data::save_dataset(data_dir, &ds).unwrap();
    }

    // We peek at the inputs/targets dimension to properly configure our topology options.
    let peeked = tabular_data::resolve_dataset(data_dir).unwrap();
    let d_in = peeked.inputs.shape()[1] as usize;
    let d_out = peeked.targets.shape()[1] as usize;
    drop(peeked);

    // ── 2. Fitness & Custom Loss (Trainer) ─────────────────────────────────
    // Fitness is the scoring metric that drives selection, ranking, and culling.
    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");

    // The Trainer encapsulates all training mechanics: loss, optimizer choice,
    // learning rate, grad clipping, batch sizes, etc. The engine only interacts with the Trainer contract.
    let trainer = gras::TabularTrainer::new(move |pred: &Variable, y: &Variable| {
        let smooth = 0.1_f32;
        let n_classes = y.data().shape()[1] as f32;
        let scale = flodl::Tensor::from_f32(&[1.0 - smooth], &[1], y.data().device())?;
        let off = flodl::Tensor::from_f32(&[smooth / n_classes], &[1], y.data().device())?;
        let smoothed_y = y.data().mul(&scale)?.add(&off)?;
        score::cross_entropy_onehot_loss(pred, &Variable::new(smoothed_y, false))
    })
    .with_learning_rate(1e-3)
    // .with_grad_clip(1.0)
    .with_batch_size(64) // Showcase new setter: set training batch size
    .with_eval_batch_size(128)
    // LR schedule (flodl::nn::scheduler): pure function of the step clock —
    // cosine annealing from the base LR to 1e-6 over the run budget. Replay-
    // safe by construction: catch-up children get the exact LR the population
    // saw at each historical step. Total = the same budget the stop criterion
    // uses, so LR bottoms out right as the race ends.
    // Alternative (commented): warmup over the smoothing grace period, so LR
    // peaks exactly when the fitness-std stop becomes eligible:
    // .with_lr_schedule(move |s| WarmupScheduler::new(
    //     CosineScheduler::new(1e-3, 1e-6, total), 1e-3, gras::engine::smoothing::SMOOTHING_WINDOW).lr(s))
    // .with_lr_schedule({
    //     let total = cli.max_steps.unwrap_or(10);
    //     move |s| CosineScheduler::new(1e-3, 1e-6, total).lr(s)
    // })
    ;

    // ── 3. Full Config Surface Showcase ──────────────────────────────────
    let log_level = match cli.log_level.as_str() {
        "none" => gras::engine::config::LogLevel::None,
        "minimal" => gras::engine::config::LogLevel::Minimal,
        "full" => gras::engine::config::LogLevel::Full,
        _ => gras::engine::config::LogLevel::Summ,
    };
    
    let builder = RaceConfig::builder()
        // --- Population & Engine options ---
        .set_pop_size(cli.pop_size) // Number of active, live networks in population
        .set_log_level(log_level) // Log level verbosity
        .set_checkpoint_every(cli.checkpoint_every) // Steps between checkpoint gate recordings
        // --- Stop Criteria (any combination may be set — they RACE each other, first to fire wins) ---
        .set_max_steps(cli.max_steps.unwrap_or(10)) // Total global step budget (CLI --max-steps; default 10 when the flag is absent). Always active: a race must terminate even if no other criterion is set.
        .set_max_target_fitness(cli.max_target_fitness) // Stop once best net's smoothed fitness reaches this (CLI --max-target-fitness). None ⇒ not racing
        // --- Evolutionary Probabilities & Rolls ---
        .set_elite_count(1) // Elite guard: top-k nets by smoothed fitness are immune to ALL culls (crossover AND mutation). 0 = no guard. Elites hold rank, not identity — a declining net falls out naturally.
        .set_crossover_prob(0.5) // Probability of a crossover roll firing per step
        .set_crossover_rolls(20) // Number of crossover attempts/rolls per step
        .set_crossover_cull_policy(gras::engine::config::CrossCullPolicy::Worst) // Who a surviving CROSSOVER child evicts: CrossCullPolicy::Worst (worst by smoothed fitness) or CrossCullPolicy::Random (uniform, diversity-first). Mutation immigrants always evict via fitness-inverse roulette — not this knob.
        .set_crossover_gating(gras::engine::config::CrossoverGating::Soft) // Gate strictness for crossover children: CrossoverGating::Hard (beat every checkpoint bar) or CrossoverGating::Soft (beat the mean of the bars)
        .set_crossover_retries(3) // cx_retry_full: on a gate rejection, retry the full attempt (fresh parents + generate + gate) up to 2 extra times. Every attempt (inserted or rejected) is recorded in history.csv
        // .set_crossover_ops_pool(vec!["one_point".into(), "uniform".into()]) // Crossover operators drawn per attempt. Empty (default) ⇒ both. The chosen op is logged per child in its lineage note (crossover-one-point / crossover-uniform).
        .set_mutate_prob(0.5) // Probability of an immigrant roll firing per step
        .set_mutate_rolls(10) // Number of random immigrant substitution rolls per step
        // --- Post-race pruner (runs AFTER any stop criterion fires) ---
        .set_pop_pruner(cli.pop_pruner) // true = on stop, cull all but the top-elite_count nets and train them solo for --pruner-steps more steps (evolution + stop criteria off)
        .set_pop_pruner_method(gras::engine::config::PopPrunerMethod::Hard, cli.pruner_steps) // Strategy + extra step count (Hard = keep the elites, plain solo training)
        .set_metrics(vec![Metric("f1".into()), Metric("precision".into())]) // Informative (non-ranking) metrics: scored on the eval batch each step, one extra history.csv column each. Ranking fitness is set separately via Fitness — these NEVER affect selection/culling.
        // config.custom_stop = Some(Box::new(|snapshot| snapshot.step > 50)); // Rarely-used: custom stop closure joining the stop race (RaceSnapshot: step, live_count, best/worst/mean smoothed fitness, culls, elapsed_seconds)
        // --- Topology Search Boundaries ---
        .set_input_dim(d_in) // Input dimension (features) from the dataset
        .set_output_dim(d_out) // Output dimension (classes) from the dataset
        .set_hidden_range(16, 128) // Hidden dimension sampling pool range (min..=max)
        .set_hidden_dim_stride(16) // Stride step within hidden dimension pool
        .set_min_hidden_num_nodes(2) // Minimum hidden layers/nodes
        .set_max_hidden_num_nodes(10) // Maximum hidden layers/nodes
        .set_min_hidden_inputs_per_node(2) // Minimum input fan-in per node
        .set_max_hidden_inputs_per_node(10) // Maximum input fan-in per node
        .set_min_hidden_outputs_per_node(2) // Minimum output fan-out per node
        .set_max_hidden_outputs_per_node(10) // Maximum output fan-out per node
        .set_dropout_prob(0.1) // Regularization dropout probability applied to layers
        // .set_combine_ops(&["Mean", "Min", "Max"])     // Allowed merge operations for search
        // .set_activations(&["ReLU", "SELU", "GELU"])   // Allowed activations for search
        // .set_standardize_ops(&["Identity"])           // Allowed standardization operations for search
        // --- Tooling, Target Modes & Exports ---
        .set_mode(RunMode::Tabular) // Target paradigm mode (Tabular, OneCImage, NLP, etc.)
        .set_csv_export(true) // Exports the lossless history.csv record (per-step metric rows + evolution attempt rows, typed by the `type` column)
        .build();

    // ── 4. Execution ───────────────────────────────────────────────────────
    // RunSpec::new coerces path-like args (Into<PathBuf>): pass &str/&Path
    // directly — no .to_path_buf() noise. Trainer is auto-boxed by with_trainer.
    let mut engine = RaceEngine::new(
        gras::engine::RunSpec::new(
            data_dir,
            builder,
            fitness,
            trainer,
            cli.seed,
            None::<&str>, // run_dir: None ⇒ results/<timestamp>
        ),
    )
    .unwrap();

    println!("================================================================================");
    println!("MNIST Race Engine Launched successfully!");
    println!("Run Dir: {}", engine.run_dir().display());
    println!("================================================================================");

    match engine.run() {
        Ok(reason) => println!("Race stopped successfully: {reason:?}"),
        Err(e) => eprintln!("Race encountered an error: {e}"),
    }
}
