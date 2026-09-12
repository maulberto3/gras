//! The deterministic shared batch stream — Iter 1 of the step-race revamp
//! (see RACE_REVAMP.md).
//!
//! Two pieces:
//! - [`PoolSplit`]: the static, seeded train/eval pool split (D12), fixed at
//!   run start so every net is compared on identical data.
//! - [`BatchStream`]: batch `step` is a **pure function of
//!   `(run_seed, step_index)`** — any net can regenerate any past batch for
//!   catch-up or resume (the replay contract). The train pool is drawn
//!   round-robin from a seeded permutation; the eval batch for a step comes
//!   from the eval pool (rotating, so eval coverage grows with steps).
//!
//! There is no epoch concept here (D14): the stream is infinite, and train +
//! eval both happen per step.

use flodl::tensor::Result;
use flodl::Tensor;

use crate::utils::seed::derive_seed;
use crate::utils::data::Dataset;

/// Per-step eval batches are derived from `run_seed + EVAL_STREAM_OFFSET`,
/// never `run_seed` itself — same contract shape as the legacy
/// `EVAL_SEED_OFFSET` (train and eval randomness must not share a seed).
pub const EVAL_STREAM_OFFSET: u64 = 0xFFFF;

// ── PoolSplit (D12) ─────────────────────────────────────────────────────────

/// The static train/eval pool split, fixed once at run start. Every net in
/// the race trains on the train pool and is evaluated on the eval pool —
/// identical conditions by construction.
#[derive(Clone, Debug)]
pub struct PoolSplit {
    /// Row indices nets train on (the stream draws from here).
    pub train: Vec<i64>,
    /// Row indices nets are evaluated on (eval batches draw from here).
    pub eval: Vec<i64>,
}

impl PoolSplit {
    /// Split `dataset.len()` rows into train/eval pools with a seeded,
    /// deterministic shuffle. Ratios must sum to ~1.0 (same contract as
    /// `data::split_indices`). Pure function of `(len, train_eval_split_ratio, seed)`.
    pub fn new(len: usize, train_eval_split_ratio: f32, seed: u64) -> Self {
        let (train, eval) = crate::utils::data::split_indices(len, 1.0 - train_eval_split_ratio, train_eval_split_ratio, seed);
        PoolSplit { train, eval }
    }

    /// Split from an existing dataset — convenience over [`PoolSplit::new`].
    pub fn of(dataset: &Dataset, train_eval_split_ratio: f32, seed: u64) -> Self {
        Self::new(dataset.len(), train_eval_split_ratio, seed)
    }
}

// ── BatchStream ─────────────────────────────────────────────────────────────

/// The shared, deterministic, infinite batch stream.
///
/// Batch for step *s* = `f(run_seed, s)` — no internal cursor, no state. Any
/// net (or tool) can materialize any step's batch independently, which is
/// what makes catch-up and resume replays exact (the replay contract).
#[derive(Clone, Debug)]
pub struct BatchStream {
    run_seed: u64,
    batch_size: usize,
    eval_batch_size: usize,
    held_out_eval_rows: usize,
    train_pool: Vec<i64>,
    eval_pool: Vec<i64>,
}

impl BatchStream {
    /// Build a stream over a run's pools. Holds only indices — dataset rows
    /// are gathered at materialization time, so the stream stays cheap to
    /// clone per net.
    pub fn new(run_seed: u64, batch_size: usize, split: PoolSplit) -> Self {
        BatchStream {
            run_seed,
            batch_size,
            eval_batch_size: batch_size,
            held_out_eval_rows: 256, // default, can be overridden via with_held_out_eval_rows
            train_pool: split.train,
            eval_pool: split.eval,
        }
    }

    /// Override the eval batch size (trainer's `stream_shape` hook). Train
    /// and eval walk *separate* permutations with separate offsets, so the
    /// two sizes are independent — changing one never shifts the other's
    /// window, and `(run_seed, step)` purity holds for both.
    pub fn with_eval_batch_size(mut self, eval_batch_size: usize) -> Self {
        self.eval_batch_size = eval_batch_size.max(1);
        self
    }

    /// Set the rows per best-net held-out reading (engine bookkeeping).
    pub fn with_held_out_eval_rows(mut self, n: usize) -> Self {
        self.held_out_eval_rows = n.max(1);
        self
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn eval_batch_size(&self) -> usize {
        self.eval_batch_size
    }

    /// The train/eval split ratio this stream was built with (from
    /// RunSpec::stream / header). For engine bookkeeping.
    pub fn train_eval_split_ratio(&self) -> f32 {
        self.train_pool.len() as f32 / (self.train_pool.len() + self.eval_pool.len()) as f32
    }

    /// Rows per best-net held-out reading.
    pub fn held_out_eval_rows(&self) -> usize {
        self.held_out_eval_rows
    }

    /// Gather `indices` into one `(inputs, targets)` batch tensor pair.
    fn gather(dataset: &Dataset, indices: &[i64]) -> Result<(Tensor, Tensor)> {
        let idx = Tensor::from_i64(indices, &[indices.len() as i64], dataset.inputs.device())?;
        let x = dataset.inputs.index_select(0, &idx)?;
        let y = dataset.targets.index_select(0, &idx)?;
        Ok((x, y))
    }

    /// A contiguous window of `batch_size` pool rows, starting at `offset`
    /// (wrapping around the pool). One permutation, consumed round-robin:
    /// the "shuffle" happens once (seeded), the walk is positional.
    fn window(pool: &[i64], batch_size: usize, offset: usize) -> Vec<i64> {
        let n = pool.len();
        (0..batch_size).map(|i| pool[(offset + i) % n]).collect()
    }

    /// The **train** batch for `step` — what every live net trains on at
    /// that step. Pure function of `(run_seed, step)`; call it as many
    /// times, from as many places, as you like.
    pub fn train_batch(&self, dataset: &Dataset, step: u64) -> Result<(Tensor, Tensor)> {
        // One seeded permutation of the train pool, derived from the run
        // seed only — steps walk it round-robin via a step-derived offset.
        let perm_seed = derive_seed(self.run_seed, 0);
        let mut perm = self.train_pool.clone();
        let mut rng = fastrand::Rng::with_seed(perm_seed);
        for i in (1..perm.len()).rev() {
            let j = rng.usize(0..=i);
            perm.swap(i, j);
        }
        let offset = (step as usize).wrapping_mul(self.batch_size);
        Self::gather(dataset, &Self::window(&perm, self.batch_size, offset))
    }

    /// The **eval** batch for `step` — the held-out batch every live net is
    /// scored on at that step (eval mode; no gradients). Derived from
    /// `run_seed + EVAL_STREAM_OFFSET` so eval randomness never shares a
    /// seed with training.
    /// First `n` rows of the eval permutation (deterministic subset of the
    /// held-out pool — used by the best-net eval for a stable reading).
    pub fn eval_pool_rows(&self, n: usize) -> Result<Vec<i64>> {
        Ok(Self::window(&self.eval_pool, n.max(1), 0))
    }

    /// Gather arbitrary pool rows into a batch. Public wrapper over the
    /// private `gather` for engine-side ad-hoc evals.
    pub fn gather_rows(&self, dataset: &Dataset, indices: &[i64]) -> Result<(Tensor, Tensor)> {
        Self::gather(dataset, indices)
    }

    pub fn eval_batch(&self, dataset: &Dataset, step: u64) -> Result<(Tensor, Tensor)> {
        let perm_seed = derive_seed(self.run_seed.wrapping_add(EVAL_STREAM_OFFSET), 0);
        let mut perm = self.eval_pool.clone();
        let mut rng = fastrand::Rng::with_seed(perm_seed);
        for i in (1..perm.len()).rev() {
            let j = rng.usize(0..=i);
            perm.swap(i, j);
        }
        let offset = (step as usize).wrapping_mul(self.eval_batch_size);
        Self::gather(dataset, &Self::window(&perm, self.eval_batch_size, offset))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use flodl::Device;

    fn small_dataset() -> Dataset {
        // 64 rows, 2 features, 2 classes — deterministic, tiny.
        crate::utils::data::synthetic_classification(64, 2, 2, 7, Device::CPU).unwrap()
    }

    #[test]
    fn pool_split_is_disjoint_and_covering() {
        let ds = small_dataset();
        let split = PoolSplit::of(&ds, 0.25, 42);
        assert_eq!(split.train.len() + split.eval.len(), ds.len());
        let mut seen = split.train.clone();
        seen.extend(split.eval.clone());
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), ds.len(), "pools must not overlap");
    }

    #[test]
    fn pool_split_is_deterministic() {
        let a = PoolSplit::new(100, 0.2, 123);
        let b = PoolSplit::new(100, 0.2, 123);
        let c = PoolSplit::new(100, 0.2, 124);
        assert_eq!(a.train, b.train);
        assert_eq!(a.eval, b.eval);
        assert_ne!(a.train, c.train, "different seed ⇒ different split");
    }

    #[test]
    fn stream_batch_is_pure_function_of_seed_and_step() {
        let ds = small_dataset();
        let mk = |seed| {
            BatchStream::new(seed, 8, PoolSplit::of(&ds, 0.25, seed))
        };
        let s = mk(99);
        // Same (seed, step) ⇒ identical batch, however many times it's asked.
        let (x1, y1) = s.train_batch(&ds, 5).unwrap();
        let (x2, y2) = s.train_batch(&ds, 5).unwrap();
        assert_eq!(x1.to_f32_vec().unwrap(), x2.to_f32_vec().unwrap());
        assert_eq!(y1.to_f32_vec().unwrap(), y2.to_f32_vec().unwrap());
        // A second stream with the same seed materializes the same batch.
        let (x3, _) = mk(99).train_batch(&ds, 5).unwrap();
        assert_eq!(x1.to_f32_vec().unwrap(), x3.to_f32_vec().unwrap());
        // Different step ⇒ different batch (stream advances).
        let (x4, _) = s.train_batch(&ds, 6).unwrap();
        assert_ne!(x1.to_f32_vec().unwrap(), x4.to_f32_vec().unwrap());
        // Different run_seed ⇒ different batch (runs are isolated).
        let (x5, _) = mk(100).train_batch(&ds, 5).unwrap();
        assert_ne!(x1.to_f32_vec().unwrap(), x5.to_f32_vec().unwrap());
    }

    #[test]
    fn eval_stream_is_independent_of_train_stream() {
        let ds = small_dataset();
        let s = BatchStream::new(99, 8, PoolSplit::of(&ds, 0.25, 99));
        let (tx, _) = s.train_batch(&ds, 0).unwrap();
        let (ex, _) = s.eval_batch(&ds, 0).unwrap();
        // Eval comes from the eval pool — a train batch row set must never
        // appear as the eval batch at the same step.
        assert_ne!(tx.to_f32_vec().unwrap(), ex.to_f32_vec().unwrap());
    }

    #[test]
    fn stream_wraps_around_pool() {
        let ds = small_dataset();
        // Train pool is 64 * 0.75 = 48 rows; batch 8 ⇒ 6 steps per lap.
        let s = BatchStream::new(99, 8, PoolSplit::of(&ds, 0.25, 99));
        let (x0, _) = s.train_batch(&ds, 0).unwrap();
        let (x6, _) = s.train_batch(&ds, 6).unwrap();
        assert_eq!(x0.to_f32_vec().unwrap(), x6.to_f32_vec().unwrap(), "step 6 wraps to the lap-1 start");
    }
}
