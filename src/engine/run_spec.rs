//! The one bundle a caller hands the engine to start a race — one variant per
//! run mode (see [`RunSpec`]).
//!
//! Everything the engine needs for the chosen mode, nothing else. The
//! training scheme is a **self-contained** struct you build — it owns its
//! loss, its optimizer choice, its learning-rate / grad-clip / schedule, and
//! optionally its own shape for the shared batch stream (Tabular). The engine
//! never touches any of that: it only reads the [`Trainer`] contract from the
//! scheme.
//!
//! The engine derives the rest — run id, run directory, (Tabular only:)
//! dataset load and batch stream — and exposes what it chose via
//! [`crate::RaceEngine::run_dir`] and [`crate::RaceEngine::run_seed`].

use crate::engine::config::RaceConfig;
use crate::engine::fitness::Fitness;
use crate::trainer::{RlStep, TabularStep};

/// Start-a-race spec: the entire `RaceEngine::new` argument surface.
///
/// - `data_dir` — the engine loads the dataset itself (already-resolved
///   on-disk format; generate it beforehand if you need synthetic data).
/// - `config` — the ONE config (population, evolution, budgets,
///   topology, log level, divergence). No training knobs — those live on
///   the trainer you supply.
/// - `fitness` — ranking is engine business (drives cull/insert), so it
///   stays explicit. The *loss* is training business and lives inside the
///   trainer.
/// - `trainer` — your training scheme, self-contained: it owns its loss, its
///   optimizer recipe, its LR / grad-clip / schedule, and (optionally) the
///   shape of the shared batch stream it wants. Accepts any `T: TabularStep`
///   (auto-boxed), or a ready `Box<dyn TabularStep>`.
/// - `seed` — `None` = random (recorded in `engine.json` for repro).
/// - `run_dir` — `None` = `results/<run_id>` with a timestamp id; give a
///   path to place the run anywhere. The chosen dir is readable via
///   `RaceEngine::run_dir()` after construction.
pub struct TabularSpec<T: TabularStep + 'static = Box<dyn TabularStep>> {
    pub data_dir: std::path::PathBuf,
    pub config: RaceConfig,
    pub fitness: Fitness,
    pub trainer: T,
    pub seed: Option<u64>,
    pub run_dir: Option<std::path::PathBuf>,
}

/// RL spec: the run learns from an ENVIRONMENT the trainer drives, not from a
/// dataset. No `data_dir`, no split/stream/eval — the engine's only ranking
/// input is the fitness value the trainer reports in `StepReport.fitness`
/// each step (require `Fitness::reported`). Everything evolution-side
/// (population, rolls, culls, gates, smoothing, exports) is identical to
/// Tabular mode.
pub struct RLSpec<T: RlStep + 'static = Box<dyn RlStep>> {
    pub config: RaceConfig,
    /// MUST be `Fitness::reported(..)` — validated at engine construction.
    pub fitness: Fitness,
    pub trainer: T,
    pub seed: Option<u64>,
    pub run_dir: Option<std::path::PathBuf>,
}

/// Start-a-race spec — one variant per run mode, each self-contained. The
/// compiler enforces that each mode supplies exactly its own requirement set:
///
/// - **[`RunSpec::tabular`]** — supervised tabular style: `data_dir` + engine-
///   computed fitness (see [`Fitness::new`]). This is the historical behavior;
///   `RunSpec::tabular` builds it.
/// - **[`RunSpec::RL`]** — environment/RL style: no dataset at all, a
///   [`Fitness::reported`] ranking signal, and a trainer that drives the
///   environment and reports the reward. `RunSpec::rl` builds it.
pub enum RunSpec<TT: TabularStep + 'static = Box<dyn TabularStep>, TR: RlStep + 'static = Box<dyn RlStep>> {
    Tabular(TabularSpec<TT>),
    RL(RLSpec<TR>),
}

impl RunSpec<Box<dyn TabularStep>, Box<dyn RlStep>> {
    /// Convenience constructor for the (default) Tabular variant — every
    /// path field coerces via `Into<PathBuf>`, so callers can pass `&str`,
    /// `String`, `&Path`, or `PathBuf` directly — no `.to_path_buf()` /
    /// `.clone()` noise:
    ///
    /// ```ignore
    /// RunSpec::tabular("data/mnist/train", config, fitness, trainer, Some(42), run_dir)
    /// ```
    ///
    /// `run_dir` takes `Option<P>` where `P: Into<PathBuf>`; pass
    /// `None::<&str>` for the default `results/<timestamp>` location.
    pub fn tabular<P: Into<std::path::PathBuf>>(
        data_dir: impl Into<std::path::PathBuf>,
        config: RaceConfig,
        fitness: Fitness,
        trainer: impl TabularStep + 'static,
        seed: Option<u64>,
        run_dir: Option<P>,
    ) -> Self {
        Self::Tabular(TabularSpec {
            data_dir: data_dir.into(),
            config,
            fitness,
            trainer: Box::new(trainer),
            seed,
            run_dir: run_dir.map(|p| p.into()),
        })
    }

    /// Convenience constructor for the RL variant: NO data_dir (the trainer
    /// learns from an environment), and `fitness` MUST be
    /// [`Fitness::reported`] — the engine validates and refuses
    /// `Fitness::Computed` here (a (pred, target) scorer has nothing to score
    /// without a dataset).
    pub fn rl(
        config: RaceConfig,
        fitness: Fitness,
        trainer: impl RlStep + 'static,
        seed: Option<u64>,
        run_dir: Option<std::path::PathBuf>,
    ) -> Self {
        Self::RL(RLSpec {
            config,
            fitness,
            trainer: Box::new(trainer),
            seed,
            run_dir,
        })
    }

    /// Convenience: build a spec with the trainer auto-boxed. Accepts any
    /// `U: TabularStep + 'static` so callers can write
    /// `.with_trainer(TabularTrainer::new(loss).with_learning_rate(1e-3))`
    /// instead of `trainer: Box::new(...)`.
    pub fn with_trainer<U: TabularStep + 'static>(
        self,
        trainer: U,
    ) -> RunSpec<Box<dyn TabularStep>, Box<dyn RlStep>> {
        match self {
            RunSpec::Tabular(s) => RunSpec::Tabular(TabularSpec {
                data_dir: s.data_dir,
                config: s.config,
                fitness: s.fitness,
                trainer: Box::new(trainer),
                seed: s.seed,
                run_dir: s.run_dir,
            }),
            RunSpec::RL(s) => RunSpec::RL(RLSpec {
                config: s.config,
                fitness: s.fitness,
                trainer: s.trainer,
                seed: s.seed,
                run_dir: s.run_dir,
            }),
        }
    }
}

/// Ergonomic helper that builds the most common training setup from just a
/// loss closure: a [`TabularTrainer`] (Adam, one train + one eval batch per
/// step) with sensible defaults. Use this when you want the engine's default
/// training recipe but still want it self-contained inside the trainer — no
/// stream shape to wire, no config knobs, no engine responsibility.
///
/// Swap in your own `Trainer` impl any time you need something else (SGD,
/// warmup, multi-minibatch, RL, image, NLP — the engine only sees the
/// [`Trainer`] contract).
pub struct DefaultTrainerBuilder {
    loss: Option<
        Box<
            dyn Fn(&flodl::Variable, &flodl::Variable) -> flodl::tensor::Result<flodl::Variable>
                + Send
                + Sync
                + 'static,
        >,
    >,
    learning_rate: f32,
    grad_clip: f32,
}

impl DefaultTrainerBuilder {
    pub fn new() -> Self {
        Self {
            loss: None,
            learning_rate: 1e-3,
            grad_clip: 1.0,
        }
    }

    pub fn loss(
        mut self,
        loss: impl Fn(&flodl::Variable, &flodl::Variable) -> flodl::tensor::Result<flodl::Variable>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        self.loss = Some(Box::new(loss));
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.grad_clip = clip;
        self
    }

    /// Build the boxed [`TabularStep`] trainer. The loss is required — a run
    /// without a loss has no training signal. The default loss fallback is
    /// cross-entropy (so `Default` stays usable for tests that don't care).
    pub fn build(self) -> Box<dyn crate::trainer::TabularStep> {
        let loss = self.loss.unwrap_or_else(|| {
            Box::new(|pred, y| crate::utils::score::cross_entropy_onehot_loss(pred, y))
        });
        Box::new(
            crate::trainer::TabularTrainer::new(loss)
                .with_learning_rate(self.learning_rate)
                .with_grad_clip(self.grad_clip),
        )
    }
}

/// Stream shape request — what the trainer wants the shared batch stream to
/// look like. Returned from [`crate::trainer::TabularStep::stream_shape`]; the
/// engine uses it to build the `BatchStream` at construction.
///
/// The trainer owns this request; the engine applies it (or the default if
/// the trainer returns `None`). The **split ratio** is NOT in this struct:
/// `train_eval_split_ratio` is engine bookkeeping (which rows are held out
/// protects fitness comparability across all nets, whatever the recipe), and
/// the engine always derives it from the dataset via the seeded split.
///
/// A trainer that doesn't want the engine stream returns `None` from
/// `stream_shape()` and brings its own dataloader — it ignores `ctx.data`.
#[derive(Clone, Copy, Debug)]
pub struct StreamShape {
    /// Rows per shared train batch.
    pub batch_size: usize,
    /// Rows per shared eval batch.
    pub eval_batch_size: usize,
}

impl StreamShape {
    /// No override — use the engine's default stream shape (batch_size from
    /// the trainer's default + eval_batch_size == batch_size).
    pub fn none() -> Self {
        Self {
            batch_size: 16,
            eval_batch_size: 16,
        }
    }

    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    pub fn with_eval_batch_size(mut self, n: usize) -> Self {
        self.eval_batch_size = n;
        self
    }

    /// Convenience for the common case: same size for train + eval.
    pub fn uniform(n: usize) -> Self {
        Self {
            batch_size: n,
            eval_batch_size: n,
        }
    }
}
