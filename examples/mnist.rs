//! gras — MNIST Race Example
//!
//! A first-class showcase and guide for the `gras` continuous step-race
//! Neural Architecture Search (NAS) engine. This example demonstrates how to
//! configure, customize, and execute a full evolutionary step-race, showcasing
//! every major evolutionary, topological, and gating control option.
//!
//! Run:
//!   source env_setup.sh && cargo run --release --example mnist
//!   source env_setup.sh cuda && cargo run --release -F cuda --example mnist
//!
//! Every knob lives directly in the builder below — edit it and rerun.
//! After the run, delete the generated `results/<timestamp>` folder yourself
//! (the crate does not clean up runs automatically).

#[path = "cli/mod.rs"]
mod cli;

use std::path::{Path, PathBuf};

use clap::Parser;
use gras::engine::fitness::{Direction, Fitness, Metric};
use gras::engine::{RaceConfig, TabularEngine};
use gras::utils::{score, tabular_data};
use gras::{RunMode, Variable};

// ── Run knobs (the CONST defaults — every one is overridable by a flag) ────
const RUN_SEED: Option<u64> = Some(16); // None = random, recorded in engine.json
const DATA_DIR: &str = "data/mnist/train"; // any gras-format dataset works — dims are auto-peeked
// const MAX_STEPS: usize = 50; // stop criterion — total global step budget
const MAX_TARGET_FITNESS: Option<f32> = Some(0.5); // e.g. Some(0.95): stop once best smoothed fitness crosses it
const POP_SIZE: usize = 100; // live networks in the race
const CHECKPOINT_EVERY: usize = 10; // steps between checkpoint gate recordings
const POP_PRUNER: bool = true; // on stop: keep the elites, train them solo
const PRUNER_STEPS: usize = 50; // extra solo steps for the pruner phase
const LOG_LEVEL: gras::engine::config::LogLevel = gras::engine::config::LogLevel::Summ;
const RUN_NAME: &str = "mnist"; // recorded in engine.json ("run_name") — purely informational
const LEARNING_RATE: f32 = 1e-3;
const BATCH_SIZE: usize = 64;
const EVAL_BATCH_SIZE: usize = 128;
const LABEL_SMOOTHING: f32 = 0.1;

// ── CLI ────────────────────────────────────────────────────────────────────

/// The command line: shared engine flags plus mnist's data/trainer knobs.
#[derive(Parser, Debug)]
#[command(
    name = "mnist",
    about = "Tabular race on MNIST-shaped data: custom loss, metrics, full config surface."
)]
struct Cli {
    #[command(flatten)]
    engine: cli::EngineArgs,

    /// Continue a previous run from its saved frontier (results/<timestamp>).
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

    /// Evaluation batch size (the held-out scoring batch).
    #[arg(long, value_name = "N")]
    eval_batch_size: Option<usize>,

    /// Label-smoothing epsilon (0.0 = plain cross-entropy).
    #[arg(long, value_name = "EPS")]
    label_smoothing: Option<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    cli.engine.init_logger(LOG_LEVEL);
    let lr = cli.learning_rate.unwrap_or(LEARNING_RATE);
    let batch_size = cli.batch_size.unwrap_or(BATCH_SIZE);
    let eval_batch_size = cli.eval_batch_size.unwrap_or(EVAL_BATCH_SIZE);
    let smoothing = cli.label_smoothing.unwrap_or(LABEL_SMOOTHING);

    // ── 1. Data Setup ──────────────────────────────────────────────────────
    // The engine loads the dataset itself from the data dir; we generate
    // synthetic data if the local path is missing so the example runs out of
    // the box.
    let data_dir_arg = cli
        .data_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DATA_DIR));
    let data_dir: &Path = &data_dir_arg;
    if !data_dir.exists() {
        println!(
            "  {} not found — generating synthetic MNIST-shaped data",
            data_dir.display()
        );
        let ds =
            tabular_data::synthetic_classification(1024, 784, 10, 42, gras::auto_device()).unwrap();
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
    // Loss: label-smoothed cross-entropy (ε=0.1), a scored helper from utils/score —
    // same pattern as accuracy_score above (ε=0 reduces to plain cross_entropy_onehot_loss).
    let trainer = gras::TabularTrainer::new(move |pred: &Variable, y: &Variable| {
        score::label_smoothing_cross_entropy_loss(pred, y, smoothing)
    })
    .with_learning_rate(lr)
    // .with_grad_clip(1.0)
    .with_batch_size(batch_size) // Showcase new setter: set training batch size
    .with_eval_batch_size(eval_batch_size)
    // LR schedule showcase — uncomment to decay the LR over the run (pure
    // function of the step clock, so replay/catch-up stays deterministic):
    // .with_lr_schedule({
    //     let total = MAX_STEPS;
    //     move |s| gras::flodl::CosineScheduler::new(1e-3, 1e-6, total).lr(s)
    // })
    ;

    // ── 3. Full Config Surface Showcase ──────────────────────────────────
    // Built from the const panel, then `cli.engine.apply(..)` overlays any
    // flag the user passed (pop, stop criteria, seeds, probs, rolls, log
    // level, pruner, run name/dir) — the showcase list below stays the
    // documented default surface.
    // Rolls scale with the population, so they follow the RESOLVED pop (flag > const).
    let pop = cli.engine.pop.unwrap_or(POP_SIZE);
    let mut builder = RaceConfig::builder()
        // --- Population & Engine options ---
        .set_run_name(RUN_NAME) // Human experiment label → engine.json "run_name" (folder name unchanged)
        .set_pop_size(pop) // Number of active, live networks in population
        .set_run_log_level(LOG_LEVEL) // Log level verbosity
        .set_run_metrics(vec![
            Metric::from("f1"), // built-in label — scored by the score_by_label dispatcher
                                // Metric::from("precision"), // another built-in
                                // Metric::custom("top1_margin", |pred: &Variable, y: &Variable| {
                                //     // Custom closure metric: confidence margin between the top-1
                                //     // logit and the runner-up — one extra history.csv column.
                                //     let p = pred.data().to_f32_vec()?;
                                //     let n = p.len() / y.data().shape()[1] as usize;
                                //     let c = y.data().shape()[1] as usize;
                                //     let mut sum = 0.0f32;
                                //     for r in 0..n {
                                //         let row = &p[r * c..(r + 1) * c];
                                //         let mut s = row.to_vec();
                                //         s.sort_by(|a, b| b.partial_cmp(a).unwrap());
                                //         sum += s[0] - s[1];
                                //     }
                                //     Ok(sum / n as f32)
                                // }),
        ]) // Informative (non-ranking) metrics: scored on the eval batch each step, one extra history.csv column each. Ranking fitness is set separately via Fitness — these NEVER affect selection/culling.
        // --- Stop Criteria (mutually exclusive — set exactly ONE) ---
        // .set_stop_max_steps(MAX_STEPS) // Total global step budget — or pass --max-steps.
        // The const stop target (`MAX_TARGET_FITNESS`) is applied below, only
        // when neither criterion was given on the CLI.
        // --- Evolutionary Probabilities & Rolls ---
        .set_elite_count(1) // Elite guard: top-k nets by smoothed fitness are immune to ALL culls (crossover AND mutation). Minimum 1 — the champion is always guarded. Elites hold rank, not identity — a declining net falls out naturally.
        .set_elite_freeze(true) // Anti-devolution A: the top elite_count nets skip the trainer entirely — their last recorded metrics are carried forward, weights never change
        .set_fitness_smoothing_window(5) // Ranking averages the last K steps of fitness — one lucky eval batch can't flip a verdict
        .set_fitness_regression_tol(0.5) // Anti-devolution D: a net whose smoothed fitness collapses below (1−tol) × its birth fitness is demoted (loses the elite seat, fronts the cull line)
        .set_crossover_prob(0.5) // Probability of a crossover roll firing per step
        .set_crossover_rolls(pop / 2) // Number of crossover attempts/rolls per step
        .set_crossover_cull_policy(gras::engine::config::CrossCullPolicy::Worst) // Who a surviving CROSSOVER child evicts: CrossCullPolicy::Worst (worst by smoothed fitness) or CrossCullPolicy::Random (uniform, diversity-first). Mutation immigrants always evict via fitness-inverse roulette — not this knob.
        .set_checkpoint_every(CHECKPOINT_EVERY) // Steps between checkpoint gate recordings
        .set_crossover_gate(gras::engine::config::CrossoverGate::Soft) // Gate strictness for crossover children: CrossoverGate::Hard (beat every checkpoint bar) or CrossoverGate::Soft (beat the mean of the bars)
        .set_crossover_retries(3) // cx_retry_full: gate-rejected child ⇒ up to 3 TOTAL attempts (fresh parents + generate + gate each), then the roll is spent. Every attempt (inserted or rejected) is recorded in history.csv
        .set_immigrant_fresh_start(false)
        // .set_crossover_ops_pool(["one_point".into(), "uniform".into()]) // Crossover operators drawn per attempt. Empty (default) ⇒ both. The chosen op is logged per child in its lineage note (crossover-one-point / crossover-uniform).
        .set_mutate_prob(0.5) // Probability of an immigrant roll firing per step
        .set_mutate_rolls(pop / 5) // Number of random immigrant substitution rolls per step
        // --- Post-race pruner (runs AFTER any stop criterion fires) ---
        .set_pruner_enabled(POP_PRUNER) // true = on stop, cull all but the top-elite_count nets and train them solo for PRUNER_STEPS more steps (evolution + stop criteria off)
        .set_pruner_method(gras::engine::config::PopPrunerMethod::Hard) // Strategy (Hard = keep the elites, plain solo training)
        .set_pruner_steps(PRUNER_STEPS) // Extra solo-training steps after the stop fires
        // config.custom_stop = Some(Box::new(|snapshot| snapshot.step > 50)); // Rarely-used: custom stop closure joining the stop race (RaceSnapshot: step, live_count, best/worst/mean smoothed fitness, culls, elapsed_seconds)
        // --- Topology Search Boundaries ---
        .set_topology_min_hidden_num_nodes(2) // Minimum hidden layers/nodes
        .set_topology_max_hidden_num_nodes(15) // Maximum hidden layers/nodes
        .set_topology_min_inputs_per_node(2) // Minimum input fan-in per node
        .set_topology_max_inputs_per_node(15) // Maximum input fan-in per node
        .set_topology_min_outputs_per_node(2) // Minimum output fan-out per node
        .set_topology_max_outputs_per_node(15) // Maximum output fan-out per node
        // --- Network Search Boundaries ---
        .set_topology_input_dim(d_in) // Input dimension (features) from the dataset
        .set_topology_output_dim(d_out) // Output dimension (classes) from the dataset
        .set_topology_hidden_dim_range(16, 128) // Hidden dimension sampling pool range (min..=max)
        .set_topology_hidden_dim_stride(16) // Stride step within hidden dimension pool
        // Dropout lives on the blueprint (TopologyOptions), not the engine:
        // this setter writes it there. Replaces the removed
        // set_network_dropout_prob sugar — same 0.1, canonical route.
        .set_topology_dropout_prob(0.1)
        // .set_topology_combine_op_pool(&["Mean", "Min", "Max"])     // Allowed merge operations for search
        // .set_topology_activation_pool(&["ReLU", "SELU", "GELU"])   // Allowed activations for search
        // .set_topology_standardize_op_pool(&["Identity"])           // Allowed standardization operations for search
        // --- Tooling, Target Modes & Exports ---
        .set_run_mode(RunMode::Tabular) // Target paradigm mode (Tabular, OneCImage, NLP, etc.)
        .set_run_csv_export(true) // Exports the lossless history.csv record (per-step metric rows + evolution attempt rows, typed by the `type` column)
        .set_elite_save_topology(true) // At stop, write elite-<hash>.md — the champion's blueprint (default true)
        .set_elite_save_safetensors(true) // At stop, write elite-<hash>.safetensors — the champion's weights (default true)
        .set_worst_save_topology(true) // At stop, also write worst-<hash>.md — the anti-champion's blueprint
        .set_worst_save_safetensors(true); // At stop, also write worst-<hash>.safetensors — the anti-champion's weights
    // The const default stop criterion is applied only when the user asked for
    // neither criterion on the CLI (they are mutually exclusive).
    if cli.engine.max_steps.is_none() && cli.engine.max_target_fitness.is_none() {
        builder = builder.set_stop_target_fitness(MAX_TARGET_FITNESS);
    }
    let builder = cli.engine.apply(builder).build();

    // ── 4. Execution ───────────────────────────────────────────────────────
    // `--resume <run_dir>`: continue a previous run from its saved frontier
    // (`results/<timestamp>`). The trainer, fitness, and config must be the
    // same scheme the run started with (resume validates identity via
    // engine.json and replays each live net with metric parity asserts).
    // RunSpec::tabular coerces path-like args (Into<PathBuf>): pass &Path directly.
    let mut engine = match &cli.resume {
        Some(dir) => {
            println!("Resuming from {}", dir.display());
            TabularEngine::resume(
                dir.clone(),
                data_dir.to_path_buf(),
                builder,
                fitness,
                trainer,
            )?
        }
        None => TabularEngine::from_spec(gras::engine::RunSpec::tabular(
            data_dir,
            builder,
            fitness,
            trainer,
            cli.engine.seed_or(RUN_SEED),
            cli.engine.run_dir_or(None), // None ⇒ results/<timestamp>
        ))?,
    };

    println!("================================================================================");
    println!("MNIST Race Engine Launched successfully!");
    println!("Run Dir: {}", engine.run_dir().display());
    println!("================================================================================");

    match engine.run() {
        Ok(reason) => println!("Race stopped successfully: {reason:?}"),
        Err(e) => eprintln!("Race encountered an error: {e}"),
    }
    Ok(())
}
