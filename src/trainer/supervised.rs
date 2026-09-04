//! Optional built-in trainer: standard supervised learning.
//!
//! This is a convenience, not the intended default. For your own training
//! loop, implement [`Trainer`] or wrap a closure with
//! [`Trainer::from_fn`](crate::trainer::Trainer::from_fn).

use flodl::nn::Module;
use flodl::tensor::Result;
use flodl::{DType, Device};
use flodl::Variable;

use crate::engine::fitness::Fitness;
use crate::graph::network::Network;
use crate::trainer::{EvalOutcome, Trainer};

// ── Config ──────────────────────────────────────────────────────────────────

/// Optimizer kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptimizerKind {
    SGD,
    #[default]
    Adam,
}

/// Training hyperparameters + data pipeline settings.
#[derive(Clone, Debug, PartialEq)]
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
    /// Train/eval split ratio (0.2 = 20% eval).
    pub eval_ratio: f32,
    /// Dropout probability on hidden nodes (0.0 = none). A training
    /// hyperparameter — the engine bakes it into every network it builds
    /// and records it in each topology, so saved graphs stay
    /// self-describing.
    pub dropout_prob: f32,
    /// Device for network building and data placement.
    pub device: Device,
    /// Data type for network weights and data.
    pub dtype: DType,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            optimizer: OptimizerKind::Adam,
            learning_rate: 1e-3,
            num_epochs: 1,
            grad_clip: 0.0,
            batch_size_train: 16,
            batch_size_eval: 16,
            num_batches_train: 16,
            num_batches_eval: 16,
            train_y_proportional: false,
            test_y_proportional: false,
            eval_ratio: 0.2,
            dropout_prob: 0.0,
            device: Device::CPU,
            dtype: DType::Float32,
        }
    }
}

// ── SupervisedTrainer ──────────────────────────────────────────────────────

/// Built-in supervised trainer — standard SGD/Adam training loop.
///
/// Owns all training state: dataset, config, loss function.
/// Optional convenience — implement [`Trainer`] or use
/// [`Trainer::from_fn`] for custom training.
pub struct SupervisedTrainer {
    pub(crate) dataset: crate::utils::data::Dataset,
    input_dim: usize,
    output_dim: usize,
    config: TrainingConfig,
    loss_fn: Box<dyn Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync>,
}

impl SupervisedTrainer {
    /// Create a new supervised trainer.
    ///
    /// `input_dim` / `output_dim` — user-provided, must match data.
    /// Both train and eval indices are derived from `gen_seed` at each generation.
    pub fn new(
        data_path: &std::path::Path,
        input_dim: usize,
        output_dim: usize,
        config: TrainingConfig,
        loss_fn: impl Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static,
    ) -> Result<Self> {
        // P1 fix: resolve_dataset loads on CPU — move the data to the
        // trainer's device + dtype up front, so a CUDA `config.device`
        // doesn't device-mismatch on every forward pass (nets are built on
        // `device()`). Also cast to the net's dtype so data and weights
        // always agree.
        let dataset = crate::utils::data::resolve_dataset(data_path)?
            .to_dtype(config.dtype)?
            .to_device(config.device)?;
        Ok(Self {
            dataset,
            input_dim,
            output_dim,
            config,
            loss_fn: Box::new(loss_fn),
        })
    }
}

impl Trainer for SupervisedTrainer {
    fn input_dim(&self) -> usize { self.input_dim }
    fn output_dim(&self) -> usize { self.output_dim }
    fn device(&self) -> Device { self.config.device }
    fn dtype(&self) -> DType { self.config.dtype }

    fn evaluate(&self, mut net: Network, fitness: &Fitness, gen_seed: u64) -> Result<EvalOutcome> {
        let params: usize = net.layers.iter().flat_map(|l| l.parameters()).map(|p| p.variable.numel() as usize).sum();
        let n = self.dataset.len();
        let eval_ratio = self.config.eval_ratio;
        // Train: derived from gen_seed. Eval: derived from gen_seed + the
        // documented eval offset (same contract as eval batch sampling).
        let (train_idx, _) = crate::utils::data::split_indices(
            n, 1.0 - eval_ratio, eval_ratio, gen_seed,
        );
        let (_, eval_idx) = crate::utils::data::split_indices(
            n, 1.0 - eval_ratio, eval_ratio,
            gen_seed.wrapping_add(crate::utils::supervised::EVAL_SEED_OFFSET),
        );
        let result = crate::utils::supervised::train_network(
            &mut net, &self.config, &self.loss_fn, fitness,
            &self.dataset, &train_idx, &eval_idx,
            gen_seed,
        )?;
        // Per-epoch curves ride along so the engine can persist them on this
        // individual in the gen JSONs (overfitting tracking).
        Ok(EvalOutcome {
            score: result.score,
            eval_loss: result.eval_loss,
            param_count: params,
            train_losses: result.loss_curve,
            eval_losses: result.eval_loss_curve,
        })
    }

    fn dropout_prob(&self) -> f32 {
        self.config.dropout_prob
    }
}