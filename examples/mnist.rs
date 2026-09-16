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

use std::io::Write;
use std::path::Path;

use gras::engine::fitness::{Direction, Fitness, Metric};
use gras::engine::{RaceConfig, RaceEngine};
use gras::utils::{tabular_data, score};
use gras::{RunMode, Variable};

// ── Run knobs (edit these constants instead of CLI flags) ───────────────────
const RUN_SEED: Option<u64> = Some(16); // None = random, recorded in engine.json
const DATA_DIR: &str = "data/mnist/train"; // any gras-format dataset works — dims are auto-peeked
// const MAX_STEPS: usize = 50; // stop criterion — total global step budget
const MAX_TARGET_FITNESS: Option<f32> = Some(0.5); // e.g. Some(0.95): stop once best smoothed fitness crosses it
const POP_SIZE: usize = 20; // live networks in the race
const CHECKPOINT_EVERY: usize = 10; // steps between checkpoint gate recordings
const POP_PRUNER: bool = true; // on stop: keep the elites, train them solo
const PRUNER_STEPS: usize = 10; // extra solo steps for the pruner phase
const LOG_LEVEL: gras::engine::config::LogLevel = gras::engine::config::LogLevel::Summ;
const RUN_NAME: &str = "mnist"; // recorded in engine.json ("run_name") — purely informational

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // ── 1. Data Setup ──────────────────────────────────────────────────────
    // The engine loads the dataset itself from DATA_DIR; we generate synthetic
    // data if the local path is missing so the example runs out of the box.
    let data_dir = Path::new(DATA_DIR);
    if !data_dir.exists() {
        println!("  {DATA_DIR} not found — generating synthetic MNIST-shaped data");
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
    // Loss: label-smoothed cross-entropy (ε=0.1), a scored helper from utils/score —
    // same pattern as accuracy_score above (ε=0 reduces to plain cross_entropy_onehot_loss).
    let trainer = gras::TabularTrainer::new(move |pred: &Variable, y: &Variable| {
        score::label_smoothing_cross_entropy_loss(pred, y, 0.1)
    })
    .with_learning_rate(1e-3)
    // .with_grad_clip(1.0)
    .with_batch_size(64) // Showcase new setter: set training batch size
    .with_eval_batch_size(128)
    // .with_lr_schedule({
    //     let total = MAX_STEPS;
    //     move |s| WarmupScheduler::new(
    //         CosineScheduler::new(1e-3, 1e-6, total),
    //         1e-3,
    //         gras::engine::smoothing::SMOOTHING_WINDOW,
    //     ).lr(s)
    // })
    ;

    // ── 3. Full Config Surface Showcase ──────────────────────────────────
    let builder = RaceConfig::builder()
        // --- Population & Engine options ---
        .set_run_name(RUN_NAME) // Human experiment label → engine.json "run_name" (folder name unchanged)
        .set_pop_size(POP_SIZE) // Number of active, live networks in population
        .set_log_level(LOG_LEVEL) // Log level verbosity
        .set_additional_metrics(vec![
            Metric::from("f1"),        // built-in label — scored by the score_by_label dispatcher
            Metric::from("precision"), // another built-in
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
        // .set_max_steps(MAX_STEPS) // Total global step budget.
        .set_max_target_fitness(MAX_TARGET_FITNESS) // Stop once best net's smoothed fitness reaches this. None ⇒ not racing
        // --- Evolutionary Probabilities & Rolls ---
        .set_elite_count(1) // Elite guard: top-k nets by smoothed fitness are immune to ALL culls (crossover AND mutation). Minimum 1 — the champion is always guarded. Elites hold rank, not identity — a declining net falls out naturally.
        .set_crossover_prob(0.5) // Probability of a crossover roll firing per step
        .set_crossover_rolls(20) // Number of crossover attempts/rolls per step
        .set_crossover_cull_policy(gras::engine::config::CrossCullPolicy::Worst) // Who a surviving CROSSOVER child evicts: CrossCullPolicy::Worst (worst by smoothed fitness) or CrossCullPolicy::Random (uniform, diversity-first). Mutation immigrants always evict via fitness-inverse roulette — not this knob.
        .set_crossover_gate_checkpoint_every(CHECKPOINT_EVERY) // Steps between checkpoint gate recordings
        .set_crossover_gate(gras::engine::config::CrossoverGate::Soft) // Gate strictness for crossover children: CrossoverGate::Hard (beat every checkpoint bar) or CrossoverGate::Soft (beat the mean of the bars)
        .set_crossover_retries(3) // cx_retry_full: on a gate rejection, retry the full attempt (fresh parents + generate + gate) up to 2 extra times. Every attempt (inserted or rejected) is recorded in history.csv
        // .set_crossover_ops_pool(vec!["one_point".into(), "uniform".into()]) // Crossover operators drawn per attempt. Empty (default) ⇒ both. The chosen op is logged per child in its lineage note (crossover-one-point / crossover-uniform).
        .set_mutate_prob(0.5) // Probability of an immigrant roll firing per step
        .set_mutate_rolls(10) // Number of random immigrant substitution rolls per step
        // --- Post-race pruner (runs AFTER any stop criterion fires) ---
        .set_pruner_pop(POP_PRUNER) // true = on stop, cull all but the top-elite_count nets and train them solo for PRUNER_STEPS more steps (evolution + stop criteria off)
        .set_pruner_method(gras::engine::config::PopPrunerMethod::Hard) // Strategy (Hard = keep the elites, plain solo training)
        .set_pruner_steps(PRUNER_STEPS) // Extra solo-training steps after the stop fires
        // config.custom_stop = Some(Box::new(|snapshot| snapshot.step > 50)); // Rarely-used: custom stop closure joining the stop race (RaceSnapshot: step, live_count, best/worst/mean smoothed fitness, culls, elapsed_seconds)
        // --- Topology Search Boundaries ---
        .set_topology_min_hidden_num_nodes(2) // Minimum hidden layers/nodes
        .set_topology_max_hidden_num_nodes(10) // Maximum hidden layers/nodes
        .set_topology_min_inputs_per_node(2) // Minimum input fan-in per node
        .set_topology_max_inputs_per_node(10) // Maximum input fan-in per node
        .set_topology_min_outputs_per_node(2) // Minimum output fan-out per node
        .set_topology_max_outputs_per_node(10) // Maximum output fan-out per node
        // --- Network Search Boundaries ---
        .set_network_input_dim(d_in) // Input dimension (features) from the dataset
        .set_network_output_dim(d_out) // Output dimension (classes) from the dataset
        .set_network_hidden_dim_range(16, 128) // Hidden dimension sampling pool range (min..=max)
        .set_network_hidden_dim_stride(16) // Stride step within hidden dimension pool
        .set_network_dropout_prob(0.1) // Regularization dropout probability applied to layers
        // .set_network_combine_ops(&["Mean", "Min", "Max"])     // Allowed merge operations for search
        // .set_network_activations(&["ReLU", "SELU", "GELU"])   // Allowed activations for search
        // .set_network_standardize_ops(&["Identity"])           // Allowed standardization operations for search
        // --- Tooling, Target Modes & Exports ---
        .set_mode(RunMode::Tabular) // Target paradigm mode (Tabular, OneCImage, NLP, etc.)
        .set_csv_export(true) // Exports the lossless history.csv record (per-step metric rows + evolution attempt rows, typed by the `type` column)
        .set_elite_save_topology(true) // At stop, write elite-<hash>.md — the champion's blueprint (default true)
        .set_elite_save_safetensors(true) // At stop, write elite-<hash>.safetensors — the champion's weights (default true)
        .set_worst_save_topology(true) // At stop, also write worst-<hash>.md — the anti-champion's blueprint
        .set_worst_save_safetensors(true) // At stop, also write worst-<hash>.safetensors — the anti-champion's weights
        .build();

    // ── 4. Execution ───────────────────────────────────────────────────────
    // RunSpec::new coerces path-like args (Into<PathBuf>): pass &Path directly.
    // Trainer is auto-boxed by with_trainer.
    let mut engine = RaceEngine::new(
        gras::engine::RunSpec::new(
            data_dir,
            builder,
            fitness,
            trainer,
            RUN_SEED,
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
