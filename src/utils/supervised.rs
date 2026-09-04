//! Supervised training loop implementation.
//!
//! Contains [`train_network`] and batch sampling helpers used by
//! [`SupervisedTrainer`](crate::trainer::supervised::SupervisedTrainer).

use flodl::nn::Module;
use flodl::nn::optim::Optimizer;
use flodl::tensor::Result;
use flodl::{Adam, SGD};
use flodl::{Tensor, Variable};
use log::{debug, trace};

use crate::engine::fitness::Fitness;
use crate::graph::network::Network;
use crate::trainer::supervised::{OptimizerKind, TrainingConfig};
use super::data::Dataset;

/// Result of a training run.
pub struct TrainResult {
    /// Mean training loss per epoch, oldest first.
    pub loss_curve: Vec<f32>,
    /// Mean held-out loss per epoch, oldest first — the last entry equals
    /// `eval_loss`. Compare with `loss_curve` to spot overfitting.
    pub eval_loss_curve: Vec<f32>,
    pub score: f32,
    pub eval_loss: Option<f32>,
}

/// Train a network: sample batches, train, score on eval batches.
pub fn train_network(
    net: &mut Network,
    config: &TrainingConfig,
    loss_fn: &dyn Fn(&Variable, &Variable) -> Result<Variable>,
    fitness: &Fitness,
    dataset: &Dataset,
    train_indices: &[i64],
    eval_indices: &[i64],
    gen_seed: u64,
) -> Result<TrainResult> {
    let train_batches = sample_batches_from_indices(
        &dataset.inputs, &dataset.targets, train_indices,
        config.batch_size_train, config.num_batches_train,
        gen_seed, config.train_y_proportional,
    )?;
    let eval_batches = sample_batches_from_indices(
        &dataset.inputs, &dataset.targets, eval_indices,
        config.batch_size_eval, config.num_batches_eval,
        gen_seed.wrapping_add(0xFFFF), config.test_y_proportional,
    )?;

    if config.num_epochs == 0 || train_batches.is_empty() {
        return Ok(TrainResult {
            loss_curve: Vec::new(),
            eval_loss_curve: Vec::new(),
            score: 0.0,
            eval_loss: None,
        });
    }

    // Train
    net.train();
    let params = net.parameters();
    debug!(
        "  train -- {} epochs x {} batches (eval {}), optimizer={:?} lr={}",
        config.num_epochs, train_batches.len(), eval_batches.len(), config.optimizer, config.learning_rate,
    );

    let mut optimizer: Box<dyn Optimizer> = match config.optimizer {
        OptimizerKind::SGD => Box::new(SGD::new(&params, config.learning_rate as f64, 0.9)),
        OptimizerKind::Adam => Box::new(Adam::new(&params, config.learning_rate as f64)),
    };

    // Dropout masks come from libtorch's *global* RNG (not the fastrand chain
    // that seeds weights and batches). Re-seed it from `gen_seed` once per
    // training call so "same seed + same options" reproduces identical masks.
    // NOTE: this is process-global, so it is only deterministic when the
    // engine runs single-threaded (the default). The upstream fix is a
    // per-call torch Generator threaded through flodl's dropout op.
    if net.dropout_layers.iter().any(|d| d.is_some()) {
        flodl::manual_seed(gen_seed);
    }

    let mut loss_curve = Vec::with_capacity(config.num_epochs);
    let mut eval_loss_curve = Vec::with_capacity(config.num_epochs);
    let last_epoch = config.num_epochs - 1;
    for epoch in 0..config.num_epochs {
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0u32;
        for (xb, yb) in &train_batches {
            let x = Variable::new(xb.clone(), true);
            let y = Variable::new(yb.clone(), false);
            let pred = net.forward(&x)?;
            let loss = loss_fn(&pred, &y)?;
            let lv = loss.item().unwrap_or(0.0) as f32;
            epoch_loss += lv;
            n_batches += 1;
            trace!("    batch loss={lv:.6}");
            loss.set_requires_grad(true)?;
            optimizer.zero_grad();
            loss.backward()?;
            if config.grad_clip > 0.0 {
                flodl::clip_grad_norm(&params, config.grad_clip as f64)?;
            }
            optimizer.step()?;
        }
        let epoch_avg = epoch_loss / n_batches as f32;
        loss_curve.push(epoch_avg);
        debug!(
            "    epoch {}/{} avg_loss={:.6}",
            epoch + 1,
            config.num_epochs,
            epoch_avg
        );

        if epoch == last_epoch {
            // Final epoch: score on held-out batches using fitness (ranking
            // only), matching the pre-existing engine semantics exactly.
            let (score, eval_loss) = score_on_eval(
                net,
                fitness,
                loss_fn,
                &eval_batches,
            )?;
            if let Some(l) = eval_loss {
                eval_loss_curve.push(l);
            }
            debug!("  train done -- score={score:.6} eval_loss={eval_loss:?}");
            return Ok(TrainResult {
                loss_curve,
                eval_loss_curve,
                score,
                eval_loss,
            });
        }

        // Intermediate epochs: record held-out loss (eval mode — dropout
        // off — so the test curve is stable) for overfitting tracking.
        let eval_avg = eval_loss_avg(net, loss_fn, &eval_batches)?;
        eval_loss_curve.push(eval_avg);
        net.train();
    }

    unreachable!("last epoch handled above")
}

/// Mean loss over every eval batch, in eval mode. Used between training
/// epochs to build the per-epoch test-loss curve.
fn eval_loss_avg(
    net: &mut Network,
    loss_fn: &dyn Fn(&Variable, &Variable) -> Result<Variable>,
    eval_batches: &[(Tensor, Tensor)],
) -> Result<f32> {
    net.eval();
    let mut total = 0.0f32;
    for (xb, yb) in eval_batches {
        let x = Variable::new(xb.clone(), false);
        let y = Variable::new(yb.clone(), false);
        let pred = net.forward(&x)?;
        total += loss_fn(&pred, &y)?.item()? as f32;
    }
    Ok(total / eval_batches.len().max(1) as f32)
}

/// Score + mean loss over every eval batch — the engine's final metric.
/// Runs in eval mode, so dropout and standardize stats are off.
fn score_on_eval(
    net: &mut Network,
    fitness: &Fitness,
    loss_fn: &dyn Fn(&Variable, &Variable) -> Result<Variable>,
    eval_batches: &[(Tensor, Tensor)],
) -> Result<(f32, Option<f32>)> {
    net.eval();
    let mut score_total = 0.0;
    let mut loss_total = 0.0f32;
    for (xb, yb) in eval_batches {
        let x = Variable::new(xb.clone(), false);
        let y = Variable::new(yb.clone(), false);
        let pred = net.forward(&x)?;
        score_total += fitness.score(&pred, &y)?;
        loss_total += loss_fn(&pred, &y)?.item()? as f32;
    }
    let n_eval = eval_batches.len() as f32;
    let score = if n_eval == 0.0 { 0.0 } else { score_total / n_eval };
    let eval_loss = if n_eval > 0.0 { Some(loss_total / n_eval) } else { None };
    Ok((score, eval_loss))
}

/// Sample random batches from a subset of indices.
/// Silent fallback: if requested more than available, use what's there.
fn sample_batches_from_indices(
    inputs: &Tensor,
    targets: &Tensor,
    indices: &[i64],
    batch_size: usize,
    num_batches: usize,
    seed: u64,
    proportional: bool,
) -> Result<Vec<(Tensor, Tensor)>> {
    let n = indices.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let max_full = n / batch_size;
    let actual = num_batches.min(max_full).max(1);
    if actual != num_batches {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            log::warn!(
                "requested {num_batches} batches but data only supports {max_full} full batches — using {actual} instead"
            );
        });
    }
    let total_samples = actual * batch_size;

    let mut rng = fastrand::Rng::with_seed(seed);

    let use_proportional = proportional && targets.ndim() == 2 && targets.shape()[1] > 1;

    let pool: Vec<i64> = if use_proportional {
        let n_classes = targets.shape()[1] as usize;
        let target_vec = targets.to_f32_vec().unwrap_or_default();
        let mut class_indices: Vec<Vec<i64>> = vec![Vec::new(); n_classes];
        for &idx in indices {
            let row_start = idx as usize * n_classes;
            if row_start + n_classes <= target_vec.len() {
                let row = &target_vec[row_start..row_start + n_classes];
                if let Some(cls) = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                    class_indices[cls.0].push(idx);
                }
            }
        }
        let weights: Vec<f64> = class_indices.iter().map(|c| c.len() as f64 / n as f64).collect();
        let mut sampled = Vec::with_capacity(total_samples);
        for _ in 0..total_samples {
            let r: f64 = rng.f64();
            let mut cum = 0.0;
            let mut chosen = 0;
            for (cls, w) in weights.iter().enumerate() {
                cum += w;
                if r < cum || cls == n_classes - 1 {
                    chosen = cls;
                    break;
                }
            }
            if let Some(pool) = class_indices.get(chosen) {
                if !pool.is_empty() {
                    sampled.push(pool[rng.usize(0..pool.len())]);
                }
            }
        }
        while sampled.len() < total_samples {
            sampled.push(indices[rng.usize(0..n)]);
        }
        sampled
    } else {
        let mut pool = indices.to_vec();
        for i in (1..pool.len()).rev() {
            let j = rng.usize(0..=i);
            pool.swap(i, j);
        }
        pool
    };

    let make_batches = |count: usize| -> Result<Vec<(Tensor, Tensor)>> {
        let mut batches = Vec::with_capacity(count);
        for b in 0..count {
            let s = b * batch_size;
            let e = (s + batch_size).min(pool.len());
            if s >= e { break; }
            let idx: Vec<i64> = pool[s..e].to_vec();
            let idx_t = Tensor::from_i64(&idx, &[idx.len() as i64], inputs.device())?;
            let xb = inputs.index_select(0, &idx_t)?;
            let yb = targets.index_select(0, &idx_t)?;
            batches.push((xb, yb));
        }
        Ok(batches)
    };

    let batches = make_batches(actual)?;
    debug!("  sample_batches_from_indices -- {} batches ({} samples from {} pool)", batches.len(), total_samples, n);
    Ok(batches)
}
