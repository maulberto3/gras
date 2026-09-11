//! The one bundle a caller hands the engine to start a race.
//!
//! Everything the engine needs, nothing else: data directory, one config,
//! the ranking fitness, the training scheme (which owns its loss), and an
//! optional seed. Omitting `trainer` is allowed too, when you hand the
//! engine one of the factory-oriented [`TrainerMaker`] structs like
//! [`TabularTrainerMaker`] instead — it boxes and optionally configures
//! the trainer for you.
//!
//! The engine derives the rest — run id, run directory, dataset load, batch
//! stream — and exposes what it chose via [`crate::RaceEngine::run_dir`] and
//! [`crate::RaceEngine::run_seed`].

use crate::engine::config::RaceConfig;
use crate::engine::fitness::Fitness;
use crate::trainer::Trainer;

/// Start-a-race spec: the entire `RaceEngine::new` argument surface.
///
/// - `data_dir` — the engine loads the dataset itself (already-resolved
///   on-disk format; generate it beforehand if you need synthetic data).
/// - `config` — the ONE config (population, evolution, stream, budgets,
///   topology, log level). No training knobs — those live on the trainer.
/// - `fitness` — ranking is engine business (drives cull/insert), so it
///   stays explicit. The *loss* is training business and lives inside the
///   trainer.
/// - `trainer` — your training scheme; supplies its own loss via
///   `Trainer::loss` when its paradigm has one. Accepts any `T: Trainer`
///   (auto-boxed), or a ready `Box<dyn Trainer>`.
/// - `seed` — `None` = random (recorded in `engine.json` for repro).
/// - `run_dir` — `None` = `results/<run_id>` with a timestamp id; give a
///   path to place the run anywhere. The chosen dir is readable via
///   `RaceEngine::run_dir()` after construction.
pub struct RunSpec {
    pub data_dir: std::path::PathBuf,
    pub config: RaceConfig,
    /// Shared batch stream shape — engine infrastructure, grouped here (not
    /// on `RaceConfig`) because it describes the *data plumbing*, not the
    /// evolution or the training. Defaults apply when using
    /// [`RunSpec::with_stream_defaults`]/`RunSpec::default_stream()`.
    pub stream: StreamSpec,
    pub fitness: Fitness,
    pub trainer: Box<dyn Trainer>,
    pub seed: Option<u64>,
    pub run_dir: Option<std::path::PathBuf>,
}

/// Shared-batch-stream shape. The engine builds and owns the deterministic
/// `BatchStream` from this: catch-up replay, resume parity, and fitness
/// comparability across all nets hang off it. The trainer draws from it via
/// `ctx.data` (or ignores it and brings its own data source), and may
/// override the batch sizes at construction via `Trainer::stream_shape` —
/// the split ratio is NOT overridable: which rows are held out protects
/// comparability, whatever the recipe.
#[derive(Clone, Copy, Debug)]
pub struct StreamSpec {
    /// Rows per shared train batch.
    pub batch_size: usize,
    /// Fraction of the dataset held out for evaluation — disjoint from the
    /// train pool, fixed at run start by the seeded split.
    pub train_eval_split_ratio: f32,
    /// Rows per best-net held-out reading (engine bookkeeping for rollups).
    pub held_out_eval_rows: usize,
}

impl Default for StreamSpec {
    fn default() -> Self {
        Self {
            batch_size: 16,
            train_eval_split_ratio: 0.2,
            held_out_eval_rows: 256,
        }
    }
}

impl StreamSpec {
    /// Engine defaults: batch 16, 20% held out, 256 eval rows.
    pub fn defaults() -> Self {
        Self::default()
    }

    /// Set the shared train batch size.
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Set the held-out fraction (0.2 = 20% of rows for eval).
    pub fn with_split_ratio(mut self, v: f32) -> Self {
        self.train_eval_split_ratio = v;
        self
    }

    /// Set the rows per best-net held-out reading.
    pub fn with_held_out_rows(mut self, n: usize) -> Self {
        self.held_out_eval_rows = n;
        self
    }
}

impl RunSpec {
    /// Convenience: build a spec with the trainer auto-boxed. Accepts any
    /// `T: Trainer + 'static` so callers can write
    /// `.with_trainer(TabularTrainer::new(loss).with_learning_rate(1e-3))`
    /// instead of `trainer: Box::new(...)`.
    pub fn with_trainer<T: Trainer + 'static>(self, trainer: T) -> Self {
        Self { trainer: Box::new(trainer), ..self }
    }
}
