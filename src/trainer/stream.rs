//! The deterministic shared batch stream for the step-race engine.

use flodl::Tensor;
use flodl::tensor::Result;

use crate::utils::seed::derive_seed;
use crate::utils::tabular_data::Dataset;

/// Per-step eval batches derive from `run_seed + EVAL_STREAM_OFFSET`, never
/// `run_seed` itself — train and eval randomness must not share a seed.
pub const EVAL_STREAM_OFFSET: u64 = 0xFFFF;

/// Deterministic seeded shuffle split: `train_ratio` of the rows train, the rest
/// evaluate. An empty eval side falls back to the whole pool (train/eval overlap).
fn seeded_split(len: usize, train_ratio: f32, eval_ratio: f32, seed: u64) -> (Vec<i64>, Vec<i64>) {
    assert!(
        (train_ratio + eval_ratio - 1.0).abs() < 1e-6,
        "ratios must sum to 1.0"
    );
    let mut indices: Vec<i64> = (0..len as i64).collect();
    let mut rng = fastrand::Rng::with_seed(seed);
    for i in (1..indices.len()).rev() {
        let j = rng.usize(0..=i);
        indices.swap(i, j);
    }
    let split = (len as f32 * train_ratio).round() as usize;
    let train: Vec<i64> = indices[..split].to_vec();
    let eval: Vec<i64> = indices[split..].to_vec();
    if eval.is_empty() {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "eval set is empty (train_ratio={train_ratio}) — falling back to using ALL data for eval (train/eval overlap)"
            );
        });
        (train, indices.clone())
    } else {
        (train, eval)
    }
}

/// The class of every row when `targets` is one-hot (classification), else `None`.
/// Regression and soft-label targets keep the uniform window — the engine casts
/// targets to f32 on load, so this reads the same tensor the loss sees.
fn one_hot_labels(dataset: &Dataset) -> Option<Vec<usize>> {
    let shape = dataset.targets.shape();
    if shape.len() != 2 || shape[1] < 2 {
        return None;
    }
    let classes = shape[1] as usize;
    let values = dataset.targets.to_f32_vec().ok()?;
    let mut labels = Vec::with_capacity(values.len() / classes);
    for row in values.chunks_exact(classes) {
        let mut best = 0usize;
        let mut one = None;
        for (i, &v) in row.iter().enumerate() {
            if v > row[best] {
                best = i;
            }
            if (v - 1.0).abs() <= 1e-4 {
                if one.replace(i).is_some() {
                    return None;
                }
            } else if v.abs() > 1e-4 {
                return None;
            }
        }
        if one != Some(best) {
            return None;
        }
        labels.push(best);
    }
    Some(labels)
}

fn shuffle_rows(rows: &mut [i64], seed: u64) {
    let mut rng = fastrand::Rng::with_seed(seed);
    for i in (1..rows.len()).rev() {
        let j = rng.usize(0..=i);
        rows.swap(i, j);
    }
}

/// Per-class seats in a batch: the floor of the proportional share, then the
/// leftover seats go to the largest remainders (ties to the lower class index).
fn class_quotas(counts: &[usize], batch: usize) -> Vec<usize> {
    let total: usize = counts.iter().sum();
    if total == 0 || batch == 0 {
        return vec![0; counts.len()];
    }
    let mut quotas: Vec<usize> = counts.iter().map(|&c| batch * c / total).collect();
    let mut left = batch - quotas.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] > 0).collect();
    order.sort_by(|&a, &b| {
        let ra = (batch * counts[a]) % total;
        let rb = (batch * counts[b]) % total;
        rb.cmp(&ra).then(a.cmp(&b))
    });
    let mut cursor = 0usize;
    while left > 0 && !order.is_empty() {
        quotas[order[cursor % order.len()]] += 1;
        left -= 1;
        cursor += 1;
    }
    quotas
}

/// One class-proportionate window: every class walks its own seeded permutation,
/// advancing `quota` rows per batch, so batches are stable under replay.
fn stratified_window(groups: &[Vec<i64>], batch: usize, ordinal: usize, seed: u64) -> Vec<i64> {
    let counts: Vec<usize> = groups.iter().map(Vec::len).collect();
    let quotas = class_quotas(&counts, batch);
    let mut out = Vec::with_capacity(batch);
    for (class, &take) in quotas.iter().enumerate() {
        let rows = &groups[class];
        if take == 0 || rows.is_empty() {
            continue;
        }
        let mut perm = rows.clone();
        shuffle_rows(&mut perm, derive_seed(seed, class));
        let start = ordinal.wrapping_mul(take) % perm.len();
        for i in 0..take {
            out.push(perm[(start + i) % perm.len()]);
        }
    }
    out
}

// ── PoolSplit (D12) ─────────────────────────────────────────────────────────

/// The static train/eval pool split, fixed once at run start: every net trains
/// the train pool and is evaluated on the eval pool — identical conditions.
#[derive(Clone, Debug)]
pub struct PoolSplit {
    /// Row indices nets train on (the stream draws from here).
    pub train: Vec<i64>,
    /// Row indices nets are evaluated on (eval batches draw from here) —
    /// held-out rows, so the reading uses every test row the run was given.
    pub eval: Vec<i64>,
}

impl PoolSplit {
    /// Split `dataset.len()` rows into train/eval pools with a seeded, deterministic shuffle:
    /// `train_eval_split_ratio` of the rows train, the rest are the eval pool.
    pub fn new(len: usize, train_eval_split_ratio: f32, seed: u64) -> Self {
        let (train, eval) = seeded_split(
            len,
            1.0 - train_eval_split_ratio,
            train_eval_split_ratio,
            seed,
        );
        PoolSplit { train, eval }
    }

    /// Split from an existing dataset — convenience over [`PoolSplit::new`].
    pub fn of(dataset: &Dataset, train_eval_split_ratio: f32, seed: u64) -> Self {
        Self::new(dataset.len(), train_eval_split_ratio, seed)
    }

    /// Proportionate split: when the targets are one-hot, each class is split
    /// train/eval on its own, so neither pool loses a rare class to the shuffle.
    /// Non-classification datasets fall back to [`Self::of`].
    pub fn stratified(dataset: &Dataset, train_eval_split_ratio: f32, seed: u64) -> Self {
        let Some(labels) = one_hot_labels(dataset) else {
            return Self::of(dataset, train_eval_split_ratio, seed);
        };
        let classes = labels.iter().copied().max().map_or(0, |c| c + 1);
        let mut train = Vec::with_capacity(dataset.len());
        let mut eval = Vec::new();
        let mut every = Vec::with_capacity(dataset.len());
        for class in 0..classes {
            let mut rows: Vec<i64> = labels
                .iter()
                .enumerate()
                .filter(|(_, label)| **label == class)
                .map(|(row, _)| row as i64)
                .collect();
            shuffle_rows(&mut rows, derive_seed(seed, class));
            let held = (rows.len() as f32 * train_eval_split_ratio).round() as usize;
            let held = if train_eval_split_ratio > 0.0 && rows.len() >= 2 {
                held.clamp(1, rows.len() - 1)
            } else {
                held.min(rows.len())
            };
            let cut = rows.len() - held;
            train.extend_from_slice(&rows[..cut]);
            eval.extend_from_slice(&rows[cut..]);
            every.extend(rows);
        }
        if eval.is_empty() {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "stratified split left an empty eval set — using ALL data for eval (train/eval overlap)"
                );
            });
            return PoolSplit { train, eval: every };
        }
        PoolSplit { train, eval }
    }

    /// Split from an EXPLICIT train/test layout: `train` = all train rows;
    /// `eval` = a seeded shuffle of all test rows.
    pub fn explicit(train_len: usize, test_len: usize, seed: u64) -> Self {
        assert!(test_len > 0, "explicit split needs a non-empty test pool");
        let train: Vec<i64> = (0..train_len as i64).collect();
        let mut eval: Vec<i64> = (train_len as i64..(train_len + test_len) as i64).collect();
        // Seeded shuffle of the TEST rows only (the eval permutation re-shuffles per
        // era anyway, but this keeps the raw pool order seed-dependent).
        let mut rng = fastrand::Rng::with_seed(seed);
        for i in (1..eval.len()).rev() {
            let j = rng.usize(0..=i);
            eval.swap(i, j);
        }
        PoolSplit { train, eval }
    }
}

// ── BatchStream ─────────────────────────────────────────────────────────────

/// The shared, deterministic, infinite batch stream: batch for step *s*
/// `f(run_seed, s)` — no cursor, no state — so resume replays exactly.
#[derive(Clone, Debug)]
pub struct BatchStream {
    run_seed: u64,
    batch_size: usize,
    eval_batch_size: usize,
    checkpoint_every: usize,
    train_pool: Vec<i64>,
    eval_pool: Vec<i64>,
    /// Per-class row indices (train), when the targets are one-hot.
    train_classes: Option<Vec<Vec<i64>>>,
    /// Per-class row indices (eval), when the targets are one-hot.
    eval_classes: Option<Vec<Vec<i64>>>,
}

impl BatchStream {
    /// Build a stream over a run's pools. Holds only indices — rows are gathered
    /// at materialization time, so the stream is cheap to clone per net.
    pub fn new(run_seed: u64, batch_size: usize, split: PoolSplit) -> Self {
        BatchStream {
            run_seed,
            batch_size,
            eval_batch_size: batch_size,
            checkpoint_every: 100, // default, can be overridden via with_checkpoint_every
            train_pool: split.train,
            eval_pool: split.eval,
            train_classes: None,
            eval_classes: None,
        }
    }

    /// Make every later batch class-proportionate when `dataset`'s targets are
    /// one-hot. A regression / soft target is a no-op (the uniform window stays).
    pub fn stratified(mut self, dataset: &Dataset) -> Self {
        if let Some(labels) = one_hot_labels(dataset) {
            let classes = labels.iter().copied().max().map_or(0, |c| c + 1);
            let groups = |pool: &[i64]| {
                let mut by_class = vec![Vec::new(); classes];
                for &row in pool {
                    by_class[labels[row as usize]].push(row);
                }
                by_class
            };
            self.train_classes = Some(groups(&self.train_pool));
            self.eval_classes = Some(groups(&self.eval_pool));
        }
        self
    }

    /// Override the eval batch size (trainer's `stream_shape` hook). Train and eval
    /// walk separate permutations, so the two sizes are independent.
    pub fn with_eval_batch_size(mut self, eval_batch_size: usize) -> Self {
        self.eval_batch_size = eval_batch_size.max(1);
        self
    }

    /// Set the checkpoint cadence — the eval permutation rotates per era, so it must agree
    /// engine's `checkpoint_every`.
    pub fn with_checkpoint_every(mut self, n: usize) -> Self {
        self.checkpoint_every = n;
        self
    }

    /// In-place variant of [`Self::with_checkpoint_every`], called by the engine at run start /
    /// resume so a config edit can't desync the eval rotation.
    pub fn set_run_checkpoint_every(&mut self, n: usize) {
        self.checkpoint_every = n;
    }

    /// Train batch size the stream materializes.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// The checkpoint cadence this stream was built with — the eval
    /// permutation rotates per era = `step / checkpoint_every` (`0` = never rotate).
    pub fn checkpoint_every(&self) -> usize {
        self.checkpoint_every
    }

    /// Eval batch size the stream materializes.
    pub fn eval_batch_size(&self) -> usize {
        self.eval_batch_size
    }

    /// The pool index vectors this stream draws from (train, eval) — read-only
    /// view; never mutated after construction.
    pub fn pools(&self) -> (&[i64], &[i64]) {
        (&self.train_pool, &self.eval_pool)
    }

    /// The share of the rows this stream TRAINS on (`train ÷ (train + eval)`), i.e.
    pub fn train_share(&self) -> f32 {
        self.train_pool.len() as f32 / (self.train_pool.len() + self.eval_pool.len()) as f32
    }

    /// Gather `indices` into one `(inputs, targets)` batch tensor pair.
    fn gather(dataset: &Dataset, indices: &[i64]) -> Result<(Tensor, Tensor)> {
        let idx = Tensor::from_i64(indices, &[indices.len() as i64], dataset.inputs.device())?;
        let x = dataset.inputs.index_select(0, &idx)?;
        let y = dataset.targets.index_select(0, &idx)?;
        Ok((x, y))
    }

    /// A contiguous window of `batch_size` pool rows starting at `offset`
    /// (wrapping): one permutation consumed round-robin.
    fn window(pool: &[i64], batch_size: usize, offset: usize) -> Vec<i64> {
        let n = pool.len();
        (0..batch_size).map(|i| pool[(offset + i) % n]).collect()
    }

    /// The **train** batch for `step` — the clock's own pass; see [`Self::train_batch_at`].
    pub fn train_batch(&self, dataset: &Dataset, step: u64) -> Result<(Tensor, Tensor)> {
        self.train_batch_at(dataset, step, 0)
    }

    /// The **train** batch for `step`'s `attempt`: `0` is the clock's own pass, `1..` a recovery retry on `set_pop_recovery_same_seed(false)`, each one batch further along the…
    pub fn train_batch_at(
        &self,
        dataset: &Dataset,
        step: u64,
        attempt: u64,
    ) -> Result<(Tensor, Tensor)> {
        if let Some(groups) = &self.train_classes {
            let ordinal = (step as usize).wrapping_add(attempt as usize);
            let seed = derive_seed(self.run_seed, 0);
            let indices = stratified_window(groups, self.batch_size, ordinal, seed);
            return Self::gather(dataset, &indices);
        }
        // One seeded permutation of the train pool, derived from the run
        // seed only — steps walk it round-robin via a step-derived offset.
        let perm_seed = derive_seed(self.run_seed, 0);
        let mut perm = self.train_pool.clone();
        let mut rng = fastrand::Rng::with_seed(perm_seed);
        for i in (1..perm.len()).rev() {
            let j = rng.usize(0..=i);
            perm.swap(i, j);
        }
        // A retry takes the NEXT window, so it sees rows the clock's own pass did not.
        let offset = (step as usize)
            .wrapping_mul(self.batch_size)
            .wrapping_add((attempt as usize).wrapping_mul(self.batch_size));
        Self::gather(dataset, &Self::window(&perm, self.batch_size, offset))
    }

    /// The eval permutation for `era` — re-shuffled per checkpoint; pure function
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

    /// The era a step belongs to (steps in the same era share one eval permutation — the rotation
    /// boundary is the engine's checkpoint).
    fn era_of(&self, step: u64) -> u64 {
        if self.checkpoint_every == 0 {
            return 0;
        }
        (step as usize / self.checkpoint_every) as u64
    }

    /// The **eval** batch for `step` — the held-out batch every live net is scored
    /// on. Rotated per checkpoint era; pure in `(run_seed, step)`.
    pub fn eval_batch(&self, dataset: &Dataset, step: u64) -> Result<(Tensor, Tensor)> {
        if let Some(groups) = &self.eval_classes {
            let base = self.run_seed.wrapping_add(EVAL_STREAM_OFFSET);
            let seed = derive_seed(base, self.era_of(step) as usize);
            let indices = stratified_window(groups, self.eval_batch_size, step as usize, seed);
            return Self::gather(dataset, &indices);
        }
        let perm = self.eval_permutation(self.era_of(step));
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
        crate::utils::tabular_data::synthetic_classification(64, 2, 2, 7, Device::CPU).unwrap()
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
        // Every held-out row is evaluated — no row is set aside from the reading.
        assert_eq!(split.eval.len(), ds.len() - split.train.len());
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
    fn explicit_split_separates_layouts_by_construction() {
        let split = PoolSplit::explicit(80, 20, 42);
        // Train pool = EXACTLY the train rows, in order, unshuffled.
        assert_eq!(
            split.train,
            (0..80).collect::<Vec<i64>>(),
            "every train row is a training row"
        );
        // Eval = EXACTLY the test rows — disjoint from train by construction,
        // covering all of them.
        let mut held: Vec<i64> = split.eval.clone();
        held.sort();
        assert_eq!(held, (80..100).collect::<Vec<i64>>());
        // Seeded: same (train_len, test_len, seed) ⇒ same split; a different
        // seed re-shuffles the test side
        let again = PoolSplit::explicit(80, 20, 42);
        assert_eq!(split.eval, again.eval);
        let other = PoolSplit::explicit(80, 20, 43);
        assert_ne!(split.eval, other.eval, "seed re-shuffles the test pool");
        // The shuffled side still covers exactly the test rows.
        let mut other_held: Vec<i64> = other.eval.clone();
        other_held.sort();
        assert_eq!(other_held, (80..100).collect::<Vec<i64>>());
    }

    #[test]
    #[should_panic(expected = "non-empty test pool")]
    fn explicit_split_rejects_empty_test_pool() {
        let _ = PoolSplit::explicit(80, 0, 42);
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

    /// A recovery retry walks ONE batch further: attempt 0 is the clock's own draw, so the
    /// default streak re-trains the same rows bit for bit.
    #[test]
    fn retry_attempt_walks_the_next_batch() {
        let ds = small_dataset();
        let s = BatchStream::new(99, 8, PoolSplit::of(&ds, 0.25, 99));
        let (own, _) = s.train_batch(&ds, 5).unwrap();
        let (zero, _) = s.train_batch_at(&ds, 5, 0).unwrap();
        assert_eq!(
            own.to_f32_vec().unwrap(),
            zero.to_f32_vec().unwrap(),
            "attempt 0 IS the clock's own draw"
        );
        let (first, _) = s.train_batch_at(&ds, 5, 1).unwrap();
        assert_ne!(
            own.to_f32_vec().unwrap(),
            first.to_f32_vec().unwrap(),
            "a retry on a fresh draw trains different rows"
        );
        assert_eq!(
            first.to_f32_vec().unwrap(),
            s.train_batch_at(&ds, 5, 1).unwrap().0.to_f32_vec().unwrap(),
            "pure function of (seed, step, attempt) — a replay asks again"
        );
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

    /// `0` is a real cadence, not a clamp: the eval permutation must stay frozen
    /// era 0 forever (and `era_of` must not divide by zero).
    #[test]
    fn zero_cadence_freezes_the_eval_permutation() {
        let ds = small_dataset();
        let s = BatchStream::new(7, 8, PoolSplit::of(&ds, 0.25, 7))
            .with_eval_batch_size(4)
            .with_checkpoint_every(0);
        assert_eq!(s.checkpoint_every(), 0);
        // Every step is era 0 — including steps that would cross many boundaries
        // cadence 1 (where step 5 would be era 5).
        for step in [0u64, 1, 5, 4096] {
            let perm = s.eval_permutation(s.era_of(step));
            let frozen = s.eval_permutation(0);
            assert_eq!(perm, frozen, "step {step} must stay in the eternal era 0");
        }
        // Same batch for the same step (still pure), and the eval batch is unaffected
        // by the cadence knob beyond the permutation choice.
        assert_eq!(
            s.eval_batch(&ds, 3).unwrap().0.to_f32_vec().unwrap(),
            s.eval_batch(&ds, 3).unwrap().0.to_f32_vec().unwrap()
        );
        // A non-zero cadence still rotates (the knob kept its old meaning).
        let rotating = BatchStream::new(7, 8, PoolSplit::of(&ds, 0.25, 7))
            .with_eval_batch_size(4)
            .with_checkpoint_every(1);
        assert_ne!(
            rotating.eval_permutation(rotating.era_of(0)),
            rotating.eval_permutation(rotating.era_of(1))
        );
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
        // Era 0's first batch must differ from era 1's: compare row *sets* across
        // several windows so the flake probability is ~0.
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

    /// Build `[n, 2]` inputs and one-hot targets with the requested class counts.
    fn labelled_dataset(counts: &[usize]) -> Dataset {
        let mut labels = Vec::new();
        for (class, &n) in counts.iter().enumerate() {
            labels.extend(std::iter::repeat(class).take(n));
        }
        let n = labels.len();
        let inputs =
            flodl::Tensor::from_f32(&vec![0.0; n * 2], &[n as i64, 2], flodl::Device::CPU).unwrap();
        let targets =
            crate::utils::tabular_data::one_hot(&labels, counts.len(), flodl::Device::CPU).unwrap();
        Dataset { inputs, targets }
    }

    fn histogram(y: &flodl::Tensor) -> Vec<usize> {
        let cols = y.shape()[1] as usize;
        let vals = y.to_f32_vec().unwrap();
        let mut out = vec![0usize; cols];
        for row in vals.chunks_exact(cols) {
            let mut best = 0usize;
            for (i, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = i;
                }
            }
            out[best] += 1;
        }
        out
    }

    /// The claim: a 70/20/10 dataset gives 7/2/1 seats in every 10-row batch,
    /// on BOTH streams, and the split keeps 80/20 overall.
    #[test]
    fn stratified_split_and_batches_keep_class_proportions() {
        let ds = labelled_dataset(&[70, 20, 10]);
        let split = PoolSplit::stratified(&ds, 0.2, 11);
        assert_eq!((split.train.len(), split.eval.len()), (80, 20));
        let stream = BatchStream::new(11, 10, split).stratified(&ds);
        for step in 0..6u64 {
            let (_, train_y) = stream.train_batch(&ds, step).unwrap();
            assert_eq!(histogram(&train_y), vec![7, 2, 1], "train step {step}");
            let (_, eval_y) = stream.eval_batch(&ds, step).unwrap();
            assert_eq!(histogram(&eval_y), vec![7, 2, 1], "eval step {step}");
        }
    }

    /// A rare class must survive the split: 10 rows at 20% eval keeps 2 held out,
    /// so an eval batch can actually measure it.
    #[test]
    fn stratified_split_keeps_a_rare_class_on_both_sides() {
        let ds = labelled_dataset(&[990, 10]);
        let split = PoolSplit::stratified(&ds, 0.2, 3);
        let targets = ds.targets.to_f32_vec().unwrap();
        let rare = |pool: &[i64]| {
            pool.iter()
                .filter(|&&row| targets[row as usize * 2 + 1] > 0.5)
                .count()
        };
        assert_eq!(rare(&split.train), 8);
        assert_eq!(rare(&split.eval), 2);
    }

    /// Non-one-hot targets are regression / soft labels: the stratified entry
    /// points must be byte-identical to the uniform stream.
    #[test]
    fn non_one_hot_targets_keep_the_uniform_stream() {
        let n = 40usize;
        let inputs =
            flodl::Tensor::from_f32(&vec![0.0; n * 2], &[n as i64, 2], flodl::Device::CPU).unwrap();
        let values: Vec<f32> = (0..n)
            .flat_map(|i| {
                let d = (i % 3) as f32 * 0.1;
                [0.2 + d, 0.8 - d]
            })
            .collect();
        let targets = flodl::Tensor::from_f32(&values, &[n as i64, 2], flodl::Device::CPU).unwrap();
        let ds = Dataset { inputs, targets };
        let split = PoolSplit::stratified(&ds, 0.25, 9);
        let uniform = PoolSplit::of(&ds, 0.25, 9);
        assert_eq!(split.train, uniform.train);
        assert_eq!(split.eval, uniform.eval);
        let stratified = BatchStream::new(9, 5, split).stratified(&ds);
        let plain = BatchStream::new(9, 5, uniform);
        assert_eq!(
            stratified
                .train_batch(&ds, 4)
                .unwrap()
                .0
                .to_f32_vec()
                .unwrap(),
            plain.train_batch(&ds, 4).unwrap().0.to_f32_vec().unwrap()
        );
    }
}
