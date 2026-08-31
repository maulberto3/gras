//! Continuous (regression) showcase — gras evolves nets to fit y = sin(2πx).
//!
//! Demonstrates: MSE loss, Minimize direction, single-output target.
//!
//! Run: `source env_setup.sh && cargo run --example continuous_showcase`


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

    // 1. Data — synthetic sine wave
    let data_dir = std::env::temp_dir().join(format!("gras_cont_showcase_{}", fastrand::u64(..)));
    let ds = gras::synthetic::synthetic_sine(256, 42, Device::CPU).unwrap();
    data::save_dataset(&data_dir, &ds).unwrap();

    // 2. Options — the 5 mandatory fields + conservative defaults.
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

    // 3. Fitness — MSE loss, Minimize direction (lower is better).
    let fitness = Fitness::new(
        |p, y| {
            let diff = p.data().sub(&y.data())?;
            let sq = diff.mul(&diff)?;
            Ok(sq.mean()?.item()? as f32)
        },
        Direction::Minimize,
        "mse",
    );

    // 4. Trainer — owns all training config.
    let trainer = SupervisedTrainer::new(
        &data_dir,
        1,    // input_dim (sine: 1 feature)
        1,    // output_dim (sine: 1 target)
        TrainingConfig {
            num_epochs: 1,
            ..Default::default()
        },
        |p, y| flodl::nn::loss::mse_loss(p, y),
    )
    .unwrap();

    let mut engine = Engine::new(opts, fitness, Box::new(trainer)).unwrap();
    engine.run().unwrap();

    // 5. Inspect robustness.
    engine.show_robustness(5, "best");

    // 6. History (always saved).
    if !engine.history.is_empty() {
        println!("\n  generation history:");
        for h in &engine.history {
            println!("    gen {:02}  avg_score={:.4}  topologies={}", h.generation, h.avg_score, h.unique_topos);
        }
    }

    let _ = std::fs::remove_dir_all(&data_dir);
}
