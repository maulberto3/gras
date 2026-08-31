//! Categorical (classification) showcase — gras evolves nets for MNIST-style data.
//!
//! Demonstrates: F1 score for ranking, cross-entropy for training.
//!
//! Run: `source env_setup.sh && cargo run --example categorical_showcase`


use flodl::Device;
use gras::data;
use gras::engine::{Engine, EngineOptions};
use gras::fitness::{Direction, Fitness};
use gras::topology::CombineOp;
use gras::trainer::{SupervisedTrainer, TrainingConfig};

fn main() {
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // 1. Data — synthetic classification
    let data_dir = std::env::temp_dir().join(format!("gras_cat_showcase_{}", fastrand::u64(..)));
    let ds = data::synthetic_classification(1024, 16, 4, 42, Device::CPU).unwrap();
    data::save_dataset(&data_dir, &ds).unwrap();

    // 2. Options
    let opts = EngineOptions::builder()
        .set_pop_size(50)
        .set_num_generations(5)
        .set_selection(gras::SelectionMethod::Tournament { tournament_size: 2 })
        .set_crossover(gras::CrossoverMethod::OnePoint { action_prob: 0.5 })
        .set_mutation(gras::MutationMethod::Activation { prob: 0.1 })
        .set_hidden_dim_pool(8..=16)
        .set_combine_op_pool(vec![CombineOp::Add])
        .set_dedup_pop_and_fill(true)
        .set_seed(Some(42))
        .build()
        .unwrap();

    // 3. Fitness — F1 for ranking
    let fitness = Fitness::new(
        |pred, y| gras::f1_score(pred, y),
        Direction::Maximize,
        "f1",
    );

    // 4. Trainer — cross-entropy for training
    let trainer = SupervisedTrainer::new(
        &data_dir,
        16,   // input_dim
        4,    // output_dim (4 classes)
        TrainingConfig {
            num_epochs: 1,
            ..Default::default()
        },
        |pred, y| gras::cross_entropy_onehot_loss(pred, y),
    )
    .unwrap();

    let mut engine = Engine::new(opts, fitness, Box::new(trainer)).unwrap();
    engine.run().unwrap();

    // 5. Inspect robustness.
    engine.show_robustness(5, "best");

    let _ = std::fs::remove_dir_all(&data_dir);
}
