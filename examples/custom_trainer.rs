//! Custom trainer — full `Trainer` trait implementation.
//!
//! Trains XOR (2 features, 2 classes) with a stateful early-stopping loop:
//! epochs stop when eval loss plateaus, and the best eval loss is tracked
//! across the whole run. Shows what the trait gives you beyond
//! `Trainer::from_fn`: a named reusable type, your own config, interior
//! state, and access to the engine's [`Fitness`] inside `evaluate`.
//!
//! Run: `source env_setup.sh && cargo run --example custom_trainer`

use std::sync::Mutex;

use gras::engine::{Engine, EngineOptions};
use gras::fitness::{Direction, Fitness};
use gras::flodl::nn::Module;
use gras::flodl::nn::optim::Optimizer;
use gras::flodl::tensor::Result;
use gras::flodl::{Adam, Tensor, Variable};
use gras::topology::CombineOp;
use gras::trainer::Trainer;
use gras::utils::data::split_indices;
use gras::{data, DType, Dataset, Device, Network};

/// Stateful trainer: early-stopping supervised loop for XOR.
struct EarlyStoppingTrainer {
    dataset: Dataset,
    input_dim: usize,
    output_dim: usize,
    device: Device,
    dtype: DType,
    /// Epochs without eval-loss improvement before stopping.
    patience: usize,
    /// Best eval loss seen across the whole run (updated by `evaluate`).
    best_eval_loss: Mutex<f64>,
}

impl EarlyStoppingTrainer {
    fn new(dataset: Dataset, input_dim: usize, output_dim: usize) -> Self {
        Self {
            dataset,
            input_dim,
            output_dim,
            device: Device::CPU,
            dtype: DType::Float32,
            patience: 3,
            best_eval_loss: Mutex::new(f64::INFINITY),
        }
    }

    /// Mean cross-entropy over held-out rows.
    fn eval_loss(&self, net: &impl Module, eval_idx: &[i64]) -> Result<f64> {
        let mut total = 0.0;
        for chunk in eval_idx.chunks(32) {
            let idx = Tensor::from_i64(chunk, &[chunk.len() as i64], self.device)?;
            let x = Variable::new(self.dataset.inputs.index_select(0, &idx)?, false);
            let y = Variable::new(self.dataset.targets.index_select(0, &idx)?, false);
            let pred = net.forward(&x)?;
            total += gras::cross_entropy_onehot_loss(&pred, &y)?.item()? as f64;
        }
        Ok(total / eval_idx.len() as f64)
    }
}

impl Trainer for EarlyStoppingTrainer {
    fn input_dim(&self) -> usize {
        self.input_dim
    }
    fn output_dim(&self) -> usize {
        self.output_dim
    }
    fn device(&self) -> Device {
        self.device
    }
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn evaluate(
        &self,
        net: Network,
        fitness: &Fitness,
        gen_seed: u64,
    ) -> Result<gras::EvalOutcome> {
        let n = self.dataset.len();
        let (train_idx, _) = split_indices(n, 0.8, 0.2, gen_seed);
        let (_, eval_idx) = split_indices(n, 0.8, 0.2, gen_seed.wrapping_add(0xFFFF));

        // Train with early stopping on eval loss (up to 50 epochs)
        let params = net.parameters();
        let mut opt = Adam::new(&params, 1e-3);
        net.train();
        let mut best = f64::INFINITY;
        let mut stalled = 0;
        for _epoch in 0..50 {
            for chunk in train_idx.chunks(32) {
                let idx = Tensor::from_i64(chunk, &[chunk.len() as i64], self.device)?;
                let x = Variable::new(self.dataset.inputs.index_select(0, &idx)?, true);
                let y = Variable::new(self.dataset.targets.index_select(0, &idx)?, false);
                let pred = net.forward(&x)?;
                let loss = gras::cross_entropy_onehot_loss(&pred, &y)?;
                loss.set_requires_grad(true)?;
                opt.zero_grad();
                loss.backward()?;
                opt.step()?;
            }
            let eval_loss = self.eval_loss(&net, &eval_idx)?;
            if eval_loss < best - 1e-4 {
                best = eval_loss;
                stalled = 0;
            } else {
                stalled += 1;
                if stalled >= self.patience {
                    break;
                }
            }
        }

        // Final score — the trait receives the engine's Fitness, so use it
        // for ranking (unlike `Trainer::from_fn`, whose closure only sees
        // the network and the seed).
        net.eval();
        let mut score = 0.0;
        let mut n_rows = 0.0;
        for chunk in eval_idx.chunks(32) {
            let idx = Tensor::from_i64(chunk, &[chunk.len() as i64], self.device)?;
            let x = Variable::new(self.dataset.inputs.index_select(0, &idx)?, false);
            let y = Variable::new(self.dataset.targets.index_select(0, &idx)?, false);
            let pred = net.forward(&x)?;
            score += fitness.score(&pred, &y)?;
            n_rows += chunk.len() as f32;
        }

        // Interior state persists across generations (thread-safe: evaluates
        // run in parallel, so the state is behind a Mutex).
        let mut guard = self.best_eval_loss.lock().unwrap();
        if best < *guard {
            *guard = best;
            log::info!("  early-stopping trainer: new best eval loss {best:.4}");
        }

        let param_count = net
            .parameters()
            .iter()
            .map(|p| p.variable.numel() as usize)
            .sum::<usize>();
        Ok(gras::EvalOutcome::new(
            score / n_rows,
            Some(best as f32),
            param_count,
        ))
    }
}

fn main() {
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // 1. Data — XOR: 2 features, 2 classes (one-hot)
    let (inputs, targets) = data::make_xor(256);
    let ds = gras::Dataset { inputs, targets };

    // 2. Options
    let options = EngineOptions::builder()
        .set_pop_size(40)
        .set_num_generations(5)
        .set_selection(gras::SelectionMethod::Tournament { tournament_size: 2 })
        .set_crossover(gras::CrossoverMethod::OnePoint { action_prob: 0.5 })
        .set_mutation(gras::MutationMethod::Activation { prob: 0.1 })
        .set_hidden_dim_pool(4..=8)
        .set_combine_op_pool(vec![CombineOp::Add])
        .set_seed(Some(7))
        .build()
        .unwrap();

    // 3. Fitness — accuracy for ranking
    let fitness = Fitness::new(
        |pred, y| gras::accuracy_score(pred, y),
        Direction::Maximize,
        "accuracy",
    );

    // 4. Trainer — your own struct implementing the Trainer trait
    let trainer = EarlyStoppingTrainer::new(ds, 2, 2);

    // 5. Run
    let mut engine = Engine::new(options, fitness, trainer).unwrap();
    engine.run().unwrap();

    // 6. Inspect robustness.
    engine.show_robustness(5, gras::engine::RobustnessFilter::Best);
}