//! gras — user-testing binary (not published as a crate binary).
//!
//! Run: `source env_setup.sh && cargo run`

use std::io::Write;
use std::path::Path;

use gras::data;
use gras::Device;
use gras::trainer::supervised::{OptimizerKind, SupervisedTrainer, TrainingConfig};
use gras::{CrossoverMethod, Direction, Engine, EngineOptions, Fitness, MutationMethod, f1_score};
use gras::selection::SelectionMethod;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    let cli_seed: Option<u64> = std::env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--seed")
        .and_then(|w| w[1].parse().ok());

    let seed = cli_seed.unwrap_or(15);

    // 1. Data — supports .bin (native) or .csv (auto-converts on first run)
    let data_dir = Path::new("data/mnist/train");
    if !data_dir.exists() {
        println!("  data/mnist/train not found — generating synthetic MNIST-shaped data");
        let ds = data::synthetic_classification(1024, 784, 10, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    // 2. Fitness — pure scoring for ranking
    let fitness = Fitness::new(
        |pred, y| f1_score(pred, y),
        Direction::Maximize,
        "f1",
    );

    // 3. Trainer — owns data, loss, split, everything
    let trainer = SupervisedTrainer::new(
        data_dir,
        784,   // input_dim
        10,    // output_dim
        TrainingConfig {
            // >1 epoch: eval runs once per epoch, so the eval curve shows
            // generalization progress and overfitting becomes visible.
            num_epochs: 10,
            learning_rate: 1e-3,
            optimizer: OptimizerKind::Adam,
            grad_clip: 1.0,
            batch_size_train: 16,
            batch_size_eval: 16,
            num_batches_train: 16,
            num_batches_eval: 16,
            train_y_proportional: true,
            test_y_proportional: true,
            eval_ratio: 0.3,
            dropout_prob: 0.1,
            device: gras::auto_device(),
            dtype: gras::DType::Float32,
        },
        |pred, y| gras::cross_entropy_onehot_loss(pred, y),
    )
    .unwrap();

    // 4. Engine options — only topology/evolution settings
    let opts = EngineOptions::builder()
        .set_seed(Some(seed))
        .set_num_threads(1)
        .set_results_dir("results")
        .set_pop_size(20)
        .set_num_generations(5)
        .set_dedup_pop_and_fill(true)
        .set_elite_count(3)
        .set_selection(SelectionMethod::Tournament { tournament_size: 3 })
        .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.25 })
        .set_mutation(MutationMethod::Activation { prob: 0.1 })
        .set_hidden_dim_pool(16..=64)
        .set_hidden_dim_stride(16)
        .set_min_hidden_num_nodes(5)
        .set_max_hidden_num_nodes(20)
        .set_min_hidden_inputs_per_node(5)
        .set_max_hidden_inputs_per_node(20)
        .set_min_hidden_outputs_per_node(5)
        .set_max_hidden_outputs_per_node(20)
        .build()
        .unwrap();

    // 5. Run
    let mut engine = Engine::new(opts, fitness, trainer).unwrap();
    engine.run().unwrap();

    // 6. Robustness analysis
    // engine.show_robustness(10, gras::engine::RobustnessFilter::Both);
}
