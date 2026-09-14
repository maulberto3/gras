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
use gras::engine::fitness::{Direction, Fitness};
use gras::engine::{RaceConfig, RaceEngine};
use gras::utils::{data, score};
use gras::{RunMode, Variable};

/// CLI arguments for the MNIST race example.
#[derive(Parser, Debug)]
#[command(name = "mnist_race", version, about)]
struct Cli {
    /// Run seed (omit = random, recorded in engine.json for repro)
    #[arg(long)]
    seed: Option<u64>,

    /// Population size (number of live networks in the race)
    #[arg(long, default_value_t = 20)]
    pop: usize,

    /// Max steps before stopping
    #[arg(long, default_value_t = 10)]
    steps: usize,

    /// Record a population checkpoint every N steps
    #[arg(long, default_value_t = 10)]
    checkpoint_every: usize,

    /// Per-step log verbosity: "summ", "minimal", "full", or "none"
    #[arg(long, default_value = "summ")]
    log_level: String,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();
    let cli = Cli::parse();

    // ── 1. Data Setup ──────────────────────────────────────────────────────
    // The engine loads the dataset itself from data_dir; we generate synthetic
    // data if the local path is missing so the example runs out of the box.
    let data_dir = Path::new("data/mnist/train");
    if !data_dir.exists() {
        println!("  data/mnist/train not found — generating synthetic MNIST-shaped data");
        let ds = data::synthetic_classification(1024, 784, 10, 42, gras::auto_device()).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    // We peek at the inputs/targets dimension to properly configure our topology options.
    let peeked = data::resolve_dataset(data_dir).unwrap();
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
    .with_grad_clip(1.0)
    .with_batch_size(64) // Showcase new setter: set training batch size
    .with_eval_batch_size(64); // Showcase new setter: set evaluation batch size

    // ── 3. Full Config Surface Showcase ──────────────────────────────────
    let log_level = match cli.log_level.as_str() {
        "none" => gras::engine::config::LogLevel::None,
        "minimal" => gras::engine::config::LogLevel::Minimal,
        "full" => gras::engine::config::LogLevel::Full,
        _ => gras::engine::config::LogLevel::Summ,
    };

    let config = RaceConfig::builder()
        // --- 1. Population & Gating ---
        .set_pop_size(cli.pop) // Number of active, live networks in population
        .set_max_steps(cli.steps) // Max steps budget limit (CLI --steps)
        .set_checkpoint_every(cli.checkpoint_every) // Steps between checkpoint gate recordings
        .set_log_level(log_level) // Log level verbosity
        // --- 2. Evolutionary Probabilities & Rolls ---
        .set_elite_count(1) // Elite guard: top-k nets by smoothed fitness are immune to ALL culls (crossover AND mutation). 0 = no guard. Elites hold rank, not identity — a declining net falls out naturally.
        .set_crossover_prob(0.5) // Probability of a crossover roll firing per step
        .set_crossover_rolls(1) // Number of crossover attempts/rolls per step
        .set_crossover_cull_policy(gras::engine::config::CrossCullPolicy::Worst) // Who a surviving CROSSOVER child evicts: CrossCullPolicy::Worst (worst by smoothed fitness) or CrossCullPolicy::Random (uniform, diversity-first). Mutation immigrants always evict via fitness-inverse roulette — not this knob.
        .set_crossover_gating(gras::engine::config::CrossoverGating::Hard) // Gate strictness for crossover children: CrossoverGating::Hard (beat every checkpoint bar) or CrossoverGating::Soft (beat the mean of the bars)
        .set_crossover_retries(2) // cx_retry_full: on a gate rejection, retry the full attempt (fresh parents + generate + gate) up to 2 extra times. Every attempt (inserted or rejected) is recorded in history.csv
        .set_mutate_prob(0.5) // Probability of an immigrant roll firing per step
        .set_mutate_rolls(1) // Number of random immigrant substitution rolls per step
        // .set_crossover_ops_pool(vec!["one_point".into(), "uniform".into()]) // Crossover operators drawn per attempt. Empty (default) ⇒ both. The chosen op is logged per child in its lineage note (crossover-one-point / crossover-uniform).
        // --- 3. Topology Search Boundaries ---
        .set_input_dim(d_in) // Input dimension (features) from the dataset
        .set_output_dim(d_out) // Output dimension (classes) from the dataset
        .set_hidden_range(8, 72) // Hidden dimension sampling pool range (min..=max)
        .set_hidden_dim_stride(16) // Stride step within hidden dimension pool
        .set_min_hidden_num_nodes(5) // Minimum hidden layers/nodes
        .set_max_hidden_num_nodes(10) // Maximum hidden layers/nodes
        .set_min_hidden_inputs_per_node(4) // Minimum input fan-in per node
        .set_max_hidden_inputs_per_node(10) // Maximum input fan-in per node
        .set_min_hidden_outputs_per_node(4) // Minimum output fan-out per node
        .set_max_hidden_outputs_per_node(10) // Maximum output fan-out per node
        .set_dropout_prob(0.1) // Regularization dropout probability applied to layers
        // --- 4. Operation Pools (Clean String Slice API!) ---
        // .set_combine_ops(&["Mean", "Min", "Max"])     // Allowed merge operations for search
        // .set_activations(&["ReLU", "SELU", "GELU"])   // Allowed activations for search
        // .set_standardize_ops(&["Identity"])           // Allowed standardization operations for search
        // --- 5. Tooling, Target Modes & Exports ---
        .set_mode(RunMode::Tabular) // Target paradigm mode (Tabular, OneCImage, NLP, etc.)
        .set_csv_export(true) // Exports the lossless history.csv record (per-step metric rows + evolution attempt rows, typed by the `type` column)
        // --- 6. Stop Criteria (ONE max — build() errors if two are set) ---
        // .set_target_score(0.99)                     // Stop when a network reaches 99% smoothed fitness
        // .set_fitness_std_threshold(0.005)           // Convergence stop: fire when pop fitness std < 0.005 (all nets equally good/stuck)
        // .set_fitness_std_min_steps(50)              // Warmup for the std stop — earliest step it may fire
        .build();

    // ── 4. Execution ───────────────────────────────────────────────────────
    let mut engine = RaceEngine::new(gras::engine::RunSpec {
        data_dir: data_dir.to_path_buf(),
        config,
        fitness,
        trainer,
        seed: cli.seed,
        run_dir: None, // default: results/<timestamp>
    })
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
