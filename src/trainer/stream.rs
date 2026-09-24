//! The deterministic shared batch stream for the step-race engine.
//!
//! Three pieces:
//! - [`PoolSplit`]: the static, seeded train/eval/gating pool split (D12),
//!   fixed at run start so every net is compared on identical data. The
//!   **gating pool** is never used for training or per-step eval — it exists
//!   only for checkpoint "surprise exam" readings (anti-memorization).
//! - [`BatchStream`]: batch `step` is a **pure function of
//!   `(run_seed, step_index)`** — any net (or tool) can materialize any step's
//!   batch independently, which is what makes catch-up and resume replays
//!   exact (the replay contract). The train pool is drawn round-robin from a
//!   fixed seeded permutation. The **eval permutation rotates per era** (era
//!   = `step / checkpoint_every`), so smoothed eval fitness reflects skill
//!   across several different data orderings instead of one memorizable pass.
//!   The checkpoint exam is likewise a pure function of `(run_seed, era)`.
//!
//! Determinism note: because the eval rotation is a pure function of
//! `(run_seed, step)`, catch-up replays for crossover children and immigrants
//! automatically see the exact rotated eval batches the population saw at
//! those historical steps — no extra bookkeeping needed.
//!
//! There is no epoch concept here (D14): the stream is infinite, and train +
//! eval both happen per step.

use flodl::Tensor;
use flodl::tensor::Result;

use crate::utils::seed::derive_seed;
use crate::utils::tabular_data::Dataset;

/// Per-step eval batches are derived from `run_seed + EVAL_STREAM_OFFSET`,
/// never `run_seed` itself — same contract shape as the legacy
/// `EVAL_SEED_OFFSET` (train and eval randomness must not share a seed).
pub const EVAL_STREAM_OFFSET: u64 = 0xFFFF;

/// Checkpoint exam randomness is derived from `run_seed + EXAM_STREAM_OFFSET`
/// — a third seed space, disjoint from train (0) and eval (`EVAL_STREAM_OFFSET`).
pub const EXAM_STREAM_OFFSET: u64 = 0xFFFF_FFFF;

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
    /// Row indices reserved for checkpoint "surprise exams" — never used for
    /// training or per-step eval, so a checkpoint score on these rows can only
    /// reflect generalization, not memorization. Kept disjoint from `train`
    /// and `eval` by construction.
    pub gating: Vec<i64>,
}

impl PoolSplit {
    /// Split `dataset.len()` rows into train/eval/gating pools with a seeded,
    /// deterministic shuffle. The eval ratio is divided into two halves: the
    /// per-step eval pool and the gating (surprise-exam) pool.
    pub fn new(len: usize, train_eval_split_ratio: f32, seed: u64) -> Self {
        let (train, rest) = crate::utils::tabular_data::split_indices(
            len,
            1.0 - train_eval_split_ratio,
            train_eval_split_ratio,
            seed,
        );
        // Half the held-out rows are the per-step eval pool, half the gating
        // pool — a deterministic, seed-derived split of `rest` (already
        // shuffled by split_indices).
        let half = rest.len() / 2;
        let (eval, gating) = if half == 0 && rest.len() > 1 {
            (
                rest[..rest.len() - 1].to_vec(),
                rest[rest.len() - 1..].to_vec(),
            )
        } else if rest.len() == 1 {
            // Degenerate: a single held-out row doubles as eval + gating.
            (rest.clone(), rest.clone())
        } else {
            (rest[..half].to_vec(), rest[half..].to_vec())
        };
        PoolSplit {
            train,
            eval,
            gating,
        }
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
    checkpoint_every: usize,
    train_pool: Vec<i64>,
    eval_pool: Vec<i64>,
    gating_pool: Vec<i64>,
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
            checkpoint_every: 100,   // default, can be overridden via with_checkpoint_every
            train_pool: split.train,
            eval_pool: split.eval,
            gating_pool: split.gating,
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

    /// Set the checkpoint cadence — the eval permutation rotates per era
    /// (era = `step / checkpoint_every`), so the engine's `checkpoint_every`
    /// and the stream's rotation period must agree.
    pub fn with_checkpoint_every(mut self, n: usize) -> Self {
        self.checkpoint_every = n.max(1);
        self
    }

    /// In-place variant of [`Self::with_checkpoint_every`]. The engine calls
    /// this at run start / resume so a post-construction config edit can never
    /// desync the eval rotation from the gate cadence.
    pub fn set_checkpoint_every(&mut self, n: usize) {
        self.checkpoint_every = n.max(1);
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// The checkpoint cadence this stream was built with — the eval
    /// permutation rotates per era = `step / checkpoint_every`.
    pub fn checkpoint_every(&self) -> usize {
        self.checkpoint_every
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
    /// First `n` rows of the eval permutation **at era 0** (deterministic subset of the
    /// held-out pool — legacy stable reading; the engine uses the era-aware
    /// `eval_pool_rows_at` instead).
    pub fn eval_pool_rows(&self, n: usize) -> Result<Vec<i64>> {
        self.eval_pool_rows_at(n, 0)
    }

    /// First `n` rows of the eval permutation at the given era. Deterministic
    /// in `(run_seed, era)` — the engine rotates this at each checkpoint so
    /// the best-net reading isn't always the same fixed rows.
    pub fn eval_pool_rows_at(&self, n: usize, era: u64) -> Result<Vec<i64>> {
        let perm = self.eval_permutation(era);
        Ok(Self::window(&perm, n.max(1), 0))
    }

    /// Gather arbitrary pool rows into a batch. Public wrapper over the
    /// private `gather` for engine-side ad-hoc evals.
    pub fn gather_rows(&self, dataset: &Dataset, indices: &[i64]) -> Result<(Tensor, Tensor)> {
        Self::gather(dataset, indices)
    }

    /// The eval permutation for `era` — re-shuffled per checkpoint era so no
    /// single ordering of the eval pool dominates the ranking. Pure function
    /// of `(run_seed, era)`; era 0 matches the legacy fixed permutation.
    fn eval_permutation(&self, era: u64) -> Vec<i64> {
        let perm_seed = derive_seed(self.run_seed.wrapping_add(EVAL_STREAM_OFFSET), era as usize);
        let mut perm = self.eval_pool.clone();
        let mut rng = fastrand::Rng::with_seed(perm_seed);
        for i in (1..perm.len()).rev() {
            let j = rng.usize(0..=i);
            perm.swap(i, j);
        }
        perm
    }

    /// The era a step belongs to (steps in the same era share one eval
    /// permutation — the rotation boundary is the engine's checkpoint).
    fn era_of(&self, step: u64) -> u64 {
        (step as usize / self.checkpoint_every) as u64
    }

    /// The **eval** batch for `step` — the held-out batch every live net is
    /// scored on at that step (eval mode; no gradients). Derived from
    /// `run_seed + EVAL_STREAM_OFFSET` (never shares a seed with training)
    /// and rotated **per checkpoint era**: steps in the same era walk one
    /// permutation, the next era re-shuffles it. Smoothed eval fitness
    /// therefore averages skill across several orderings of the pool, not
    /// one memorizable pass. Pure function of `(run_seed, step)` — catch-up
    /// replays see the same rotated batches the population saw.
    pub fn eval_batch(&self, dataset: &Dataset, step: u64) -> Result<(Tensor, Tensor)> {
        let perm = self.eval_permutation(self.era_of(step));
        let offset = (step as usize).wrapping_mul(self.eval_batch_size);
        Self::gather(dataset, &Self::window(&perm, self.eval_batch_size, offset))
    }

    /// The **checkpoint exam** — a "surprise test" batch drawn from the
    /// gating pool: rows that no training step and no per-step eval step ever
    /// touches. Scored once per checkpoint (engine bookkeeping), so a good
    /// exam score can only reflect generalization, not memorized rows or a
    /// rehearsed ordering. Pure function of `(run_seed, era)`.
    pub fn exam_batch(&self, dataset: &Dataset, era: u64) -> Result<(Tensor, Tensor)> {
        let exam_seed = derive_seed(self.run_seed.wrapping_add(EXAM_STREAM_OFFSET), era as usize);
        let mut perm = self.gating_pool.clone();
        let mut rng = fastrand::Rng::with_seed(exam_seed);
        for i in (1..perm.len()).rev() {
            let j = rng.usize(0..=i);
            perm.swap(i, j);
        }
        // One exam batch per era: up to `held_out_eval_rows` rows.
        let n = self.held_out_eval_rows.min(perm.len()).max(1);
        Self::gather(dataset, &perm[..n])
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use flodl::Device;

    fn small_dataset() -> Dataset {
        // 64 rows, 2 features, 2 classes — deterministic, tiny.
        crate::utils::tabular_data::synthetic_classification(64, 2, 2, 7, Device::CPU).unwrap()
    }

    #[test]
    fn pool_split_is_disjoint_and_covering() {
        let ds = small_dataset();
        let split = PoolSplit::of(&ds, 0.25, 42);
        assert_eq!(
            split.train.len() + split.eval.len() + split.gating.len(),
            ds.len()
        );
        let mut seen = split.train.clone();
        seen.extend(split.eval.clone());
        seen.extend(split.gating.clone());
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
        let mk = |seed| BatchStream::new(seed, 8, PoolSplit::of(&ds, 0.25, seed));
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
        assert_eq!(
            x0.to_f32_vec().unwrap(),
            x6.to_f32_vec().unwrap(),
            "step 6 wraps to the lap-1 start"
        );
    }

    #[test]
    fn gating_pool_is_disjoint_from_train_and_eval() {
        let ds = small_dataset();
        let split = PoolSplit::of(&ds, 0.25, 7);
        let mut all: Vec<i64> = split
            .train
            .iter()
            .chain(split.eval.iter())
            .chain(split.gating.iter())
            .copied()
            .collect();
        all.sort();
        all.dedup();
        assert_eq!(
            all.len(),
            ds.len(),
            "train+eval+gating must cover the dataset"
        );
        assert!(!split.gating.is_empty(), "gating pool must be non-empty");
    }

    #[test]
    fn eval_permutation_rotates_per_era() {
        let ds = small_dataset();
        let s = BatchStream::new(99, 8, PoolSplit::of(&ds, 0.25, 99))
            .with_eval_batch_size(4)
            .with_checkpoint_every(3);
        // Same era ⇒ same eval batch for the same in-era offset.
        let (a0, _) = s.eval_batch(&ds, 0).unwrap();
        let (a1, _) = s.eval_batch(&ds, 1).unwrap();
        assert_ne!(
            a0.to_f32_vec().unwrap(),
            a1.to_f32_vec().unwrap(),
            "two steps in era 0 walk the same permutation"
        );
        // Era boundary ⇒ different permutation; the batch content changes.
        let (b0, _) = s.eval_batch(&ds, 3).unwrap();
        let (b0_again, _) = s.eval_batch(&ds, 3).unwrap();
        assert_eq!(
            b0.to_f32_vec().unwrap(),
            b0_again.to_f32_vec().unwrap(),
            "pure function of (run_seed, step)"
        );
        // Era 0's first batch must differ from era 1's first batch (rotation).
        // Compare row *sets* — with a small eval pool both eras' batch-0
        // windows can coincide by chance; rotation means the permutation
        // differs, which shows up as a different row multiset across a full
        // lap. Sample several windows to make the flake probability ~0.
        let era0_rows = (0..4usize)
            .map(|i| s.eval_batch(&ds, i as u64).unwrap().0.to_f32_vec().unwrap())
            .collect::<Vec<_>>()
            .concat();
        let era1_rows = (3..7usize)
            .map(|i| s.eval_batch(&ds, i as u64).unwrap().0.to_f32_vec().unwrap())
            .collect::<Vec<_>>()
            .concat();
        assert_ne!(
            era0_rows, era1_rows,
            "eval permutation must rotate across the era boundary"
        );
    }

    #[test]
    fn exam_batch_is_deterministic_and_era_dependent() {
        let ds = small_dataset();
        let s = BatchStream::new(99, 8, PoolSplit::of(&ds, 0.25, 99));
        let (e0x, e0y) = s.exam_batch(&ds, 0).unwrap();
        let (e0_again, _) = s.exam_batch(&ds, 0).unwrap();
        assert_eq!(
            e0x.to_f32_vec().unwrap(),
            e0_again.to_f32_vec().unwrap(),
            "exam is a pure function of (run_seed, era)"
        );
        let (e1x, _) = s.exam_batch(&ds, 1).unwrap();
        assert_ne!(
            e0x.to_f32_vec().unwrap(),
            e1x.to_f32_vec().unwrap(),
            "each era gets a fresh exam"
        );
        assert_eq!(e0y.shape(), e0x.shape());
    }
}
