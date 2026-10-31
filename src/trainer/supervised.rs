//! The reference [`TabularStep`]: the classic tabular recipe the race engine
//! used to run inline, extracted verbatim so `TabularTrainer` is a drop-in —
//! identical RNG discipline, identical batches, identical metrics.
//!
//! This is also the template for custom tabular schemes: clone it and change
//! the internals (multi-minibatch steps, curriculum, custom eval cadence, …).
//! The engine only consumes the [`TabularStep`] contract — it never trains.
//!
//! For the RL counterpart see [`crate::trainer::reinforce`]: `Reinforce<E: Env>`
//! is the shipped [`RlStep`](crate::trainer::RlStep), and a scheme that does not
//! fit it (kagiculture's bridge, decision lag) implements the trait directly.

use crate::graph::network::Network;
use crate::trainer::{TabularContext, TabularStep};
use crate::utils::race_steps::{deterministic_train_step, eval_one_step};
use flodl::nn::optim::Optimizer;
use flodl::tensor::Result;
use flodl::{Tensor, Variable};

/// Classic tabular scheme: one train step + one eval step on the step's shared
/// batches. Also owns the loss, LR (via `make_optimizer`), and grad clip.
pub struct TabularTrainer {
    /// The loss this scheme trains against (supervised paradigm).
    pub loss_fn: crate::trainer::BoxedLossFn,
    /// Adam learning rate for every optimizer this scheme creates.
    pub learning_rate: f32,
    /// Gradient-norm clip applied per optimizer step (0 = off).
    pub grad_clip: f32,
    /// Custom training batch size requested from the engine stream.
    pub batch_size: usize,
    /// Custom evaluation batch size requested from the engine stream.
    pub eval_batch_size: usize,
        /// Optional LR schedule: pure function of the step clock (`None` = fixed LR);
        /// MUST be deterministic in `step` alone — see `TabularStep::scheduled_lr`.
    pub lr_schedule: Option<Box<dyn Fn(usize) -> f64 + Send + Sync>>,
        /// A NAMED schedule (`cosine`, `onecycle`, …), which wins over the closure
        /// above when set. Named, so `engine.json` records what actually ran and a
        /// resume can refuse a different curve instead of silently switching it.
    pub lr_schedule_name: Option<crate::trainer::LrSchedule>,
        /// Which update the optimizer makes (`adam` by default). Recorded, because a
        /// resumed run must load the RIGHT moments — `nets/<hash>.optim` is written
        /// by one optimizer and cannot be read by another.
    pub optimizer: crate::trainer::OptimizerKind,
        /// What objective `loss_fn` optimizes, for the run record. The engine can't
        /// see inside the closure, so replay tools refuse an unlabeled scheme.
    pub loss_label: Option<String>,
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

        /// Name the loss this scheme trains against, for `engine.json` — a plain
        /// name lets replay tools rebuild it; a custom one records the truth.
    pub fn with_loss_label(mut self, label: impl Into<String>) -> Self {
        self.loss_label = Some(label.into());
        self
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

        /// Set an LR schedule: a pure function of the step clock returning the rate
        /// for that step; the closure must be deterministic in `step` alone.
    pub fn with_lr_schedule(
        mut self,
        schedule: impl Fn(usize) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.lr_schedule = Some(Box::new(schedule));
        self
    }

        /// Set a NAMED schedule — the one a config file, a CLI flag or a Python
        /// keyword can spell. It scales [`Self::learning_rate`], so `lr=` and
        /// `lr_schedule=` compose, and it takes precedence over
        /// [`Self::with_lr_schedule`].
    pub fn with_named_lr_schedule(mut self, schedule: crate::trainer::LrSchedule) -> Self {
        self.lr_schedule_name = Some(schedule);
        self
    }

        /// Choose the update by name (`adam` is the default). The choice is recorded
        /// in `engine.json`, so a resume either reloads the same optimizer's moments or
        /// refuses — it cannot continue one run's state under another optimizer.
    pub fn with_optimizer(mut self, optimizer: crate::trainer::OptimizerKind) -> Self {
        self.optimizer = optimizer;
        self
    }

}

impl Default for TabularTrainer {
    fn default() -> Self {
        Self {
            // Fallback loss (plain cross-entropy) so `Default` stays
            // available; real runs construct via `TabularTrainer::new(loss)`.
            loss_fn: Box::new(crate::utils::score::cross_entropy_onehot_loss),
            learning_rate: crate::utils::training::DEFAULT_LR,
            grad_clip: 1.0,
            batch_size: 16,
            eval_batch_size: 16,
            // Unlabeled by default: the engine cannot see inside `loss_fn`, so
            // silence is the honest record. `with_loss_label` names it.
            loss_label: None,
            lr_schedule: None,
            lr_schedule_name: None,
            optimizer: crate::trainer::OptimizerKind::default(),
        }
    }
}

impl TabularStep for TabularTrainer {
    /// Reload `nets/<hash>.optim` into a fresh optimizer of the KIND this scheme
    /// trains with. `Stateful` needs the concrete type — which is exactly why this
    /// lives in the trainer instead of the engine.
    fn restore_optimizer(
        &self,
        net: &Network,
        path: &std::path::Path,
    ) -> Option<Box<dyn Optimizer>> {
        self.optimizer.load(net, self.learning_rate, path)
    }

    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        self.optimizer.make(net, self.learning_rate)
    }

    /// Record this scheme's training recipe in `engine.json` (see
    /// [`TabularStep::describe`]).
    fn describe(&self) -> Option<serde_json::Value> {
        let mut blob = serde_json::json!({
            "trainer": "tabular",
            "optimizer": self.optimizer.label(),
            "loss": self.loss_label,
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            // The BOOLEAN stays for compatibility: runs recorded before named
            // schedules compare this key on resume, so changing it to a string
            // would refuse every one of them.
            "lr_schedule": self.lr_schedule.is_some() || self.lr_schedule_name.is_some(),
        });
        // Only a NAMED schedule gets a label, and only when one is set — so a run
        // with no schedule writes exactly the blob it wrote before this existed.
        if let Some(named) = &self.lr_schedule_name {
            blob["lr_schedule_name"] = serde_json::Value::String(named.label());
        }
        Some(blob)
    }

    fn loss(&self) -> crate::trainer::LossFn<'_> {
        &*self.loss_fn
    }

    /// A NAMED schedule scales the base rate; an anonymous closure is taken at its
    /// word. Named wins, because it is the one that can be recorded and checked.
    fn scheduled_lr(&self, step: usize, max_steps: Option<usize>) -> Option<f64> {
        match (&self.lr_schedule_name, &self.lr_schedule) {
            (Some(named), _) => Some(named.lr(self.learning_rate, step, max_steps) as f64),
            (None, Some(anonymous)) => Some(anonymous(step)),
            (None, None) => None,
        }
    }

    fn stream_shape(&self) -> Option<crate::trainer::StreamShape> {
        Some(crate::trainer::StreamShape {
            batch_size: self.batch_size,
            eval_batch_size: self.eval_batch_size,
        })
    }

    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &TabularContext<'_>,
    ) -> Result<crate::trainer::TabularStepReport> {
            // The tabular recipe uses every engine handle: shared stream for the
            // step's batches, loss + fitness + metrics for scoring.
        let fitness = ctx.fitness;
        let loss_fn: crate::trainer::LossFn<'_> = &*self.loss_fn; // our own loss

        // call_index = hash of the net's name, so each net sees distinct
        // dropout at one step (live nets vs catching-up children too).
        let h: u64 = ctx
            .net_hash
            .bytes()
            .fold(0u64, |a, b| a.wrapping_add(b as u64));
        let batch: (Tensor, Tensor) = ctx.data.train_batch(step as u64)?;
        // LR schedule first (pure function of step — replay-safe), then the
        // deterministic train step (seeds the RNG, then forward/backward).
        if let Some(lr) = self.scheduled_lr(step, ctx.env.max_steps) {
            optimizer.set_lr(lr);
        }
        let train_loss = deterministic_train_step(
            ctx.net_seed,
            step as u64,
            h,
            net,
            optimizer,
            loss_fn,
            &batch,
            self.grad_clip,
        )?;

        // Eval on the step's shared eval batch (held-out stream), never the
        // train batch — this must mirror replay exactly.
        let eval_batch: (Tensor, Tensor) = ctx.data.eval_batch(step as u64)?;
        let report = eval_one_step(net, loss_fn, fitness, ctx.metrics, &eval_batch)?;

        Ok(crate::trainer::TabularStepReport {
            train_loss,
            eval_loss: report.eval_loss,
            fitness: report.fitness,
            informative: report.metrics,
        })
    }
}
