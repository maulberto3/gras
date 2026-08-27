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
    /// Rows per batch.
    pub batch_size: usize,
    /// Number of batches per generation (split into train/eval).
    pub num_batches: usize,
    /// Sample batches proportional to target class frequency.
    pub y_proportional_batches: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            optimizer: OptimizerKind::Adam,
            learning_rate: 1e-3,
            num_epochs: 1,
            grad_clip: 0.0,
            batch_size: 128,
            num_batches: 16,
            y_proportional_batches: false,
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
pub fn train_network(
    net: &mut Network,
    config: &TrainingConfig,
    fitness: &crate::fitness::Fitness,
    dataset: &Dataset,
    seed: u64,
) -> Result<TrainResult> {
    // Sample batches (train/eval split)
    let (train_batches, eval_batches) = sample_batches(
        &dataset.inputs,
        &dataset.targets,
        config.batch_size,
        config.num_batches,
        seed,
        config.y_proportional_batches,
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
        "  train -- {} epochs x {} batches, optimizer={:?} lr={} params={}",
        config.num_epochs,
        train_batches.len(),
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

    // Eval: score on held-out batches
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

/// Shuffle data and split into train/eval batches.
/// When `proportional` is true and targets are categorical (2D one-hot),
/// each batch is sampled proportional to class frequency.
fn sample_batches(
    inputs: &Tensor,
    targets: &Tensor,
    batch_size: usize,
    num_batches: usize,
    seed: u64,
    proportional: bool,
) -> Result<(Vec<(Tensor, Tensor)>, Vec<(Tensor, Tensor)>)> {
    let n = inputs.shape()[0] as usize;
    let max_full = n / batch_size;
    let actual = num_batches.min(max_full).max(1);
    let total_samples = actual * batch_size;

    let mut rng = fastrand::Rng::with_seed(seed);

    // Decide whether proportional sampling applies.
    let use_proportional = proportional && targets.ndim() == 2 && targets.shape()[1] > 1;

    let all_idx: Vec<i64> = if use_proportional {
        // Build per-class index lists from one-hot targets.
        let n_classes = targets.shape()[1] as usize;
        let target_vec = targets.to_f32_vec().unwrap_or_default();
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (i, row) in target_vec.chunks(n_classes).enumerate() {
            if let Some(cls) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            {
                class_indices[cls.0].push(i);
            }
        }
        // Compute class weights (frequency / total).
        let total = n as f64;
        let weights: Vec<f64> = class_indices
            .iter()
            .map(|c| c.len() as f64 / total)
            .collect();
        // Sample `total_samples` indices proportional to class weights.
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
                    sampled.push(pool[rng.usize(0..pool.len())] as i64);
                }
            }
        }
        // Pad if undersampled.
        while sampled.len() < total_samples {
            sampled.push(rng.usize(0..n) as i64);
        }
        sampled
    } else {
        // Uniform random shuffle.
        let mut idx: Vec<i64> = (0..n as i64).collect();
        for i in (1..idx.len()).rev() {
            let j = rng.usize(0..=i);
            idx.swap(i, j);
        }
        idx
    };

    let train_count = (actual / 2).max(1);
    let eval_count = (actual - train_count).max(1);

    let make_batch = |start: usize, count: usize| -> Result<Vec<(Tensor, Tensor)>> {
        let mut batches = Vec::with_capacity(count);
        for b in 0..count {
            let s = start + b * batch_size;
            let e = (s + batch_size).min(all_idx.len());
            if s >= e {
                break;
            }
            let idx: Vec<i64> = all_idx[s..e].to_vec();
            let idx_t = Tensor::from_i64(&idx, &[idx.len() as i64], inputs.device())?;
            let xb = inputs.index_select(0, &idx_t)?;
            let yb = targets.index_select(0, &idx_t)?;
            batches.push((xb, yb));
        }
        Ok(batches)
    };

    let train_batches = make_batch(0, train_count)?;
    let eval_batches = make_batch(train_count * batch_size, eval_count)?;
    let eval_batches = if eval_batches.is_empty() {
        train_batches.clone()
    } else {
        eval_batches
    };

    debug!(
        "  sample_batches -- train={} eval={} (from {} total)",
        train_batches.len(),
        eval_batches.len(),
        actual
    );
    Ok((train_batches, eval_batches))
}
