//! Training loop for a single [`Network`].
//!
//! Owns the full pipeline: batch, train, score.
//! Engine loads data once and passes a reference here.

use flodl::nn::Module;
use flodl::nn::optim::Optimizer;
use flodl::tensor::Result;
use flodl::{Adam, SGD};
use flodl::{Tensor, Variable};
use log::{debug, trace};

use crate::network::Network;
use crate::utils::data::Dataset;

// ── Config ──────────────────────────────────────────────────────────────────

/// Optimizer kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OptimizerKind {
    SGD,
    #[default]
    Adam,
}

/// Training hyperparameters + data pipeline settings.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    pub optimizer: OptimizerKind,
    pub learning_rate: f32,
    /// Training epochs (0 = no training, random-init forward pass).
    pub num_epochs: usize,
    /// Gradient clipping max-norm (0.0 = no clipping).
    pub grad_clip: f32,
    /// Rows per training batch.
    pub batch_size_train: usize,
    /// Rows per evaluation batch.
    pub batch_size_eval: usize,
    /// Number of training batches per generation.
    pub num_batches_train: usize,
    /// Number of evaluation batches per generation.
    pub num_batches_eval: usize,
    /// Sample batches proportional to target class frequency.
    pub train_y_proportional: bool,
    pub test_y_proportional: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            optimizer: OptimizerKind::Adam,
            learning_rate: 1e-3,
            num_epochs: 1,
            grad_clip: 0.0,
            batch_size_train: 128,
            batch_size_eval: 128,
            num_batches_train: 16,
            num_batches_eval: 16,
            train_y_proportional: false,
            test_y_proportional: false,
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Result of a training run.
pub struct TrainResult {
    pub loss_curve: Vec<f32>,
    pub score: f32,
    pub eval_loss: Option<f32>,
}

/// Train a network: sample batches, train, score on eval batches.
/// Receives a pre-loaded dataset (no file I/O).
/// `train_indices` — subset for training (resampled each generation).
/// `eval_indices` — fixed subset for evaluation (same every generation).
/// `train_seed` — changes each generation for train batch shuffling.
/// `eval_seed` — fixed across generations for consistent eval batches.
pub fn train_network(
    net: &mut Network,
    config: &TrainingConfig,
    fitness: &crate::fitness::Fitness,
    dataset: &Dataset,
    train_indices: &[i64],
    eval_indices: &[i64],
    train_seed: u64,
    eval_seed: u64,
) -> Result<TrainResult> {
    // Sample train batches from train_indices (resampled each generation)
    let train_batches = sample_batches_from_indices(
        &dataset.inputs,
        &dataset.targets,
        train_indices,
        config.batch_size_train,
        config.num_batches_train,
        train_seed,
        config.train_y_proportional,
    )?;
    // Use eval_indices directly (fixed across generations)
    let eval_batches = sample_batches_from_indices(
        &dataset.inputs,
        &dataset.targets,
        eval_indices,
        config.batch_size_eval,
        config.num_batches_eval,
        eval_seed,  // fixed across generations for consistent eval
        config.test_y_proportional,
    )?;

    if config.num_epochs == 0 || train_batches.is_empty() {
        return Ok(TrainResult {
            loss_curve: Vec::new(),
            score: 0.0,
            eval_loss: None,
        });
    }

    // Train
    net.train();
    let params = net.parameters();
    debug!(
        "  train -- {} epochs x {} batches (eval {}), optimizer={:?} lr={} params={}",
        config.num_epochs,
        train_batches.len(),
        eval_batches.len(),
        config.optimizer,
        config.learning_rate,
        params.len()
    );

    let mut optimizer: Box<dyn Optimizer> = match config.optimizer {
        OptimizerKind::SGD => Box::new(SGD::new(&params, config.learning_rate as f64, 0.9)),
        OptimizerKind::Adam => Box::new(Adam::new(&params, config.learning_rate as f64)),
    };

    let mut loss_curve = Vec::with_capacity(config.num_epochs);

    for epoch in 0..config.num_epochs {
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0u32;
        for (xb, yb) in &train_batches {
            let x = Variable::new(xb.clone(), true);
            let y = Variable::new(yb.clone(), false);

            let pred = net.forward(&x)?;
            let loss = fitness.train_metric(&pred, &y)?;
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
        let avg = epoch_loss / n_batches as f32;
        loss_curve.push(avg);
        debug!(
            "    epoch {}/{} avg_loss={:.6}",
            epoch + 1,
            config.num_epochs,
            avg
        );
    }

    // Eval: score on held-out batches (always uses the trained net)
    net.eval();
    let mut score_total = 0.0;
    let mut loss_total = 0.0f32;
    for (xb, yb) in &eval_batches {
        let x = Variable::new(xb.clone(), false);
        let y = Variable::new(yb.clone(), false);
        let pred = net.forward(&x)?;
        score_total += fitness.score(&pred, &y)?;
        loss_total += fitness.train_metric(&pred, &y)?.item()? as f32;
    }
    let n_eval = eval_batches.len() as f32;
    let score = if n_eval == 0.0 {
        0.0
    } else {
        score_total / n_eval
    };
    let eval_loss = if n_eval > 0.0 {
        Some(loss_total / n_eval)
    } else {
        None
    };

    debug!(
        "  train done -- score={:.6} eval_loss={:?}",
        score, eval_loss
    );

    Ok(TrainResult {
        loss_curve,
        score,
        eval_loss,
    })
}

// ── Private helpers ─────────────────────────────────────────────────────────

/// Sample random batches from a subset of indices.
/// Same as sample_batches but operates on a pre-defined index subset.
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
    let total_samples = actual * batch_size;

    let mut rng = fastrand::Rng::with_seed(seed);

    // Decide whether proportional sampling applies.
    let use_proportional = proportional && targets.ndim() == 2 && targets.shape()[1] > 1;

    let pool: Vec<i64> = if use_proportional {
        // Build per-class index lists from one-hot targets.
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
        // Sample proportional to class frequency.
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
        // Pad if undersampled.
        while sampled.len() < total_samples {
            sampled.push(indices[rng.usize(0..n)]);
        }
        sampled
    } else {
        // Shuffle the index subset.
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
            if s >= e {
                break;
            }
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
