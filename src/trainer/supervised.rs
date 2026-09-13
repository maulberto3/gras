//! The reference [`Trainer`]: the classic tabular recipe the race engine
//! used to run inline, extracted verbatim so `TabularTrainer` is a
//! drop-in — identical RNG discipline, identical batches, identical metrics.
//!
//! This is also the template for custom schemes: clone it and change the
//! internals (multi-minibatch steps, curriculum, custom eval cadence, …).
//! The engine only consumes the [`Trainer`] contract — it never trains.
//! (Named "tabular" because the shared-batch, one-step-clock shape fits
//! small structured datasets; image/RL/NLP schemes subclass the same trait.)

use crate::graph::network::Network;
use crate::trainer::{StepContext, StepReport, Trainer};
use crate::utils::race_steps::{eval_one_step, seed_step_randomness, train_one_step};
use flodl::nn::optim::Optimizer;
use flodl::tensor::Result;
use flodl::{Tensor, Variable};

/// Classic tabular scheme: one `train_one_step` on the step's shared train
/// batch, then one `eval_one_step` on the step's shared eval batch. Seeding
/// matches the engine's historical behavior exactly (`seed_step_randomness`
/// with call_index 0 before training, batch drawn from the shared stream) so
/// runs stay deterministic and catch-up replay parity holds.
///
/// Also owns everything about how a net learns: the loss function (a
/// supervised scheme's defining choice), the Adam learning rate (via
/// `make_optimizer`), and the gradient-norm clip. The engine carries none of
/// them — `RaceConfig` has no training knobs.
pub struct TabularTrainer {
    /// The loss this scheme trains against (supervised paradigm).
    pub loss_fn: Box<dyn Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync>,
    /// Adam learning rate for every optimizer this scheme creates.
    pub learning_rate: f32,
    /// Gradient-norm clip applied per optimizer step (0 = off).
    pub grad_clip: f32,
    /// Custom training batch size requested from the engine stream.
    pub batch_size: usize,
    /// Custom evaluation batch size requested from the engine stream.
    pub eval_batch_size: usize,
}

impl TabularTrainer {
    /// A tabular scheme is defined by its loss — constructor takes it.
    pub fn new(
        loss_fn: impl Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static,
    ) -> Self {
        Self {
            loss_fn: Box::new(loss_fn),
            ..Self::default()
        }
    }

    /// Set the Adam learning rate (default: the crate's historical 1e-3).
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the gradient-norm clip (default 1.0; 0 disables clipping).
    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.grad_clip = clip;
        self
    }

    /// Set the training batch size (default 16).
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set the evaluation batch size (default 16).
    pub fn with_eval_batch_size(mut self, size: usize) -> Self {
        self.eval_batch_size = size;
        self
    }
}

impl Default for TabularTrainer {
    fn default() -> Self {
        Self {
            // Fallback loss (plain cross-entropy) so `Default` stays
            // available; real runs construct via `TabularTrainer::new(loss)`.
            loss_fn: Box::new(|pred, y| crate::utils::score::cross_entropy_onehot_loss(pred, y)),
            learning_rate: crate::engine::config::DEFAULT_LR,
            grad_clip: 1.0,
            batch_size: 16,
            eval_batch_size: 16,
        }
    }
}

// ── Placeholder trainers (skeleton for other paradigms) ──────────────────────
// These are stub implementations showing how each paradigm maps to the
// Trainer contract. Each is compiletest-clean but deliberately minimal —
// replace the body with your recipe. The key difference between paradigms
// is what they own vs what they ignore from StepContext.

// Stub RL trainer — no loss function (rewards are internal), its own
// data source (ignores ctx.data), brings its own optimizer.
//
// Pattern: return None from loss(), ignore ctx.data, score via the
// environment's reward signal translated to fitness. For a real RL scheme
// you would bring your own environment/dataloader and seed it per
// (net_seed, step) for catch-up parity.
//
// RL = no loss; reward signal is the trainer's internal business.
pub struct RlTrainer {
    /// Placeholder — replace with your env/dataloader handle.
    _private: (),
}

impl RlTrainer {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for RlTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer for RlTrainer {
    /// RL has no loss function in this sense — rewards are internal.
    fn loss(&self) -> Option<crate::trainer::LossFn<'_>> {
        None
    }

    /// RL brings its own data source — ignore the engine's shared stream.
    fn stream_shape(&self) -> Option<crate::trainer::StreamShape> {
        None // not using the engine stream at all
    }

    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        // TODO: replace with your optimizer (PPO, SAC, …).
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
    }

    fn train_step(
        &mut self,
        _net: &mut Network,
        _optimizer: &mut dyn Optimizer,
        _step: usize,
        _ctx: &StepContext<'_>,
    ) -> Result<StepReport> {
        // TODO: run one env step / policy update. Seed per (net, step) for
        // catch-up parity:
        //   crate::utils::race_steps::seed_step_randomness(ctx.net_seed, step as u64, 0);
        // Score fitness from your reward signal (not from ctx.fitness):
        let fitness = 0.0_f32; // TODO: your reward → fitness mapping
        Ok(StepReport {
            train_loss: 0.0,
            eval_loss: None, // RL usually has no held-out eval
            fitness,
            informative: Vec::new(),
        })
    }
}

/// Stub image trainer — uses the shared stream but with a different batch
/// semantics (e.g. augmentation, multi-crop). Owns its loss + augmentation
/// config; uses `ctx.data` for the base batches.
///
/// Pattern: use `ctx.data` (Option) for the raw batches, apply per-image
/// augmentation inside `train_step`, score with your own loss + `ctx.fitness`.
pub struct ImageTrainer {
    /// Placeholder — replace with your loss + augmentation config.
    _private: (),
}

impl ImageTrainer {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for ImageTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer for ImageTrainer {
    fn loss(&self) -> Option<crate::trainer::LossFn<'_>> {
        None // TODO: return your image loss when implemented
    }

    fn stream_shape(&self) -> Option<crate::trainer::StreamShape> {
        // Image batches are typically larger; override if using the engine
        // stream (or ignore ctx.data and bring your own dataloader).
        Some(crate::trainer::StreamShape {
            batch_size: 32,
            eval_batch_size: 64,
        })
    }

    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
    }

    fn train_step(
        &mut self,
        _net: &mut Network,
        _optimizer: &mut dyn Optimizer,
        _step: usize,
        _ctx: &StepContext<'_>,
    ) -> Result<StepReport> {
        // TODO: draw `ctx.data.train_batch(step)`, augment, train.
        // Seed per (net, step):
        //   crate::utils::race_steps::seed_step_randomness(ctx.net_seed, step as u64, 0);
        let fitness = 0.0_f32; // TODO
        Ok(StepReport {
            train_loss: 0.0,
            eval_loss: None,
            fitness,
            informative: Vec::new(),
        })
    }
}

/// Stub NLP trainer — sequence-shaped batches, per-token or per-sequence
/// loss. Owns its tokenizer/vocab + loss; may use `ctx.data` or bring its
/// own streamer.
///
/// Pattern: same as ImageTrainer — own the sequence machinery, use
/// `ctx.data` if the engine stream fits, otherwise ignore it.
pub struct NlpTrainer {
    /// Placeholder — replace with your tokenizer + language-modeling loss.
    _private: (),
}

impl NlpTrainer {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for NlpTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer for NlpTrainer {
    fn loss(&self) -> Option<crate::trainer::LossFn<'_>> {
        None // TODO: return your LM loss when implemented
    }

    fn stream_shape(&self) -> Option<crate::trainer::StreamShape> {
        Some(crate::trainer::StreamShape {
            batch_size: 16,
            eval_batch_size: 16,
        })
    }

    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
    }

    fn train_step(
        &mut self,
        _net: &mut Network,
        _optimizer: &mut dyn Optimizer,
        _step: usize,
        _ctx: &StepContext<'_>,
    ) -> Result<StepReport> {
        // TODO: draw sequences, train one step, score.
        // Seed per (net, step):
        //   crate::utils::race_steps::seed_step_randomness(ctx.net_seed, step as u64, 0);
        let fitness = 0.0_f32; // TODO
        Ok(StepReport {
            train_loss: 0.0,
            eval_loss: None,
            fitness,
            informative: Vec::new(),
        })
    }
}

// ── Reference trainer (the real implementation) ──────────────────────────────

impl Trainer for TabularTrainer {
    fn loss(&self) -> Option<crate::trainer::LossFn<'_>> {
        Some(&*self.loss_fn)
    }

    fn stream_shape(&self) -> Option<crate::trainer::StreamShape> {
        Some(crate::trainer::StreamShape {
            batch_size: self.batch_size,
            eval_batch_size: self.eval_batch_size,
        })
    }

    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(
            &net.parameters(),
            self.learning_rate as f64,
        ))
    }

    /// Record this scheme's training recipe in `engine.json` (see
    /// [`Trainer::describe`]).
    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "tabular",
            "optimizer": "adam",
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
        }))
    }

    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &StepContext<'_>,
    ) -> Result<StepReport> {
        // The tabular recipe uses every engine-provided handle: shared
        // stream for the step's batches, loss + fitness + metrics for
        // scoring. (A scheme that doesn't fit that mold simply ignores the
        // handles it doesn't need — they're Options.)
        let data = ctx
            .data
            .expect("TabularTrainer requires the run's shared data (stream + dataset)");
        let fitness = ctx
            .fitness
            .expect("TabularTrainer requires a fitness function");
        let loss_fn: crate::trainer::LossFn<'_> = &*self.loss_fn; // our own loss

        // Same seeding the engine did inline: net seed + step clock, hashed
        // with the net's hash so each net sees distinct dropout at one step.
        let h: u64 = ctx
            .net_hash
            .bytes()
            .fold(0u64, |a, b| a.wrapping_add(b as u64));
        seed_step_randomness(ctx.net_seed, step as u64, h);
        let batch: (Tensor, Tensor) = data.train_batch(step as u64)?;
        let train_loss = train_one_step(net, optimizer, loss_fn, &batch, self.grad_clip)?;

        // Eval on the step's shared eval batch (held-out stream), never the
        // train batch — this must mirror catch-up exactly.
        let eval_batch: (Tensor, Tensor) = data.eval_batch(step as u64)?;
        let report = eval_one_step(net, loss_fn, fitness, ctx.metrics, &eval_batch)?;

        Ok(StepReport {
            train_loss,
            eval_loss: report.eval_loss,
            fitness: report.fitness,
            informative: report.metrics,
        })
    }
}
