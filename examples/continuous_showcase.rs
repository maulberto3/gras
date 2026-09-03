//! Continuous (regression) showcase — gras evolves nets to fit y = sin(2πx).
//!
//! Demonstrates: MSE loss, Minimize direction, single-output target.
//!
//! Run: `source env_setup.sh && cargo run --example continuous_showcase`

use gras::data;
use gras::engine::{Engine, EngineOptions};
use gras::fitness::{Direction, Fitness};
use gras::flodl::nn::loss::mse_loss;
use gras::flodl::nn::Module;
use gras::flodl::nn::optim::Optimizer;
use gras::flodl::{Adam, Tensor, Variable};
use gras::topology::CombineOp;
use gras::trainer::from_fn;
use gras::utils::data::split_indices;
use gras::{DType, Device};

fn main() {
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // 1. Data — synthetic sine wave (kept in memory — your trainer's job)
    let (inputs, targets) = data::make_sine(256);
    let ds = gras::Dataset { inputs, targets };

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

    // 4. Trainer — build your own from the Trainer contract.
    //    One closure owns the whole loop: split, train, eval, score.
    let trainer = from_fn(1, 1, Device::CPU, DType::Float32, move |net, gen_seed| {
        let n = ds.len();
        let (train_idx, _) = split_indices(n, 0.8, 0.2, gen_seed);
        let (_, eval_idx) = split_indices(n, 0.8, 0.2, gen_seed.wrapping_add(0xFFFF));

        // Train — one epoch of MSE with Adam, 32-row batches
        let params = net.parameters();
        let mut opt = Adam::new(&params, 1e-3);
        net.train();
        for chunk in train_idx.chunks(32) {
            let idx = Tensor::from_i64(chunk, &[chunk.len() as i64], Device::CPU)?;
            let x = Variable::new(ds.inputs.index_select(0, &idx)?, true);
            let y = Variable::new(ds.targets.index_select(0, &idx)?, false);
            let pred = net.forward(&x)?;
            let loss = mse_loss(&pred, &y)?;
            loss.set_requires_grad(true)?;
            opt.zero_grad();
            loss.backward()?;
            opt.step()?;
        }

        // Eval — mean squared error on held-out rows
        net.eval();
        let mut loss_sum = 0.0;
        for chunk in eval_idx.chunks(32) {
            let idx = Tensor::from_i64(chunk, &[chunk.len() as i64], Device::CPU)?;
            let x = Variable::new(ds.inputs.index_select(0, &idx)?, false);
            let y = Variable::new(ds.targets.index_select(0, &idx)?, false);
            let pred = net.forward(&x)?;
            loss_sum += mse_loss(&pred, &y)?.item()? as f32;
        }
        let n_eval = eval_idx.len() as f32;
        let param_count = net
            .parameters()
            .iter()
            .map(|p| p.variable.numel() as usize)
            .sum::<usize>();
        Ok((loss_sum / n_eval, Some(loss_sum / n_eval), param_count))
    });

    let mut engine = Engine::new(opts, fitness, trainer).unwrap();
    engine.run().unwrap();

    // 5. Inspect robustness.
    engine.show_robustness(5, gras::engine::RobustnessFilter::Best);

    // 6. History (always saved).
    if !engine.history.is_empty() {
        println!("\n  generation history:");
        for h in &engine.history {
            println!("    gen {:02}  avg_score={:.4}  topologies={}", h.generation, h.avg_score, h.unique_topos);
        }
    }
}