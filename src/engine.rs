//! The engine -- NAS loop over random topologies: seed, score, evolve.
//!
//! Data contract: flodl-native tensors loaded once at Engine::new,
//! reused per individual per generation. Replicate via
//! Engine::to_json -> Topology::from_json + Network::build.

use std::fs;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use flodl::Device;
use flodl::nn::Module;
use flodl::tensor::Result;
use log::debug;
use rayon::prelude::*;
use serde::Serialize;

pub use crate::crossover::CrossoverMethod;
pub use crate::fitness::{Direction, Fitness, FitnessLabel};
pub use crate::mutation::MutationMethod;
use crate::network::{Network, NetworkOptions};
use crate::node::{Activation, NodeKind};
use crate::selection::SelectionMethod;
use crate::topology::{CombineOp, Topology, TopologyOptions};
use crate::utils::data::Dataset;
use crate::utils::error::EngineError;



// ── EngineOptions -- the experiment configuration ──────────────────────────

/// Run configuration -- serialized to engine.json for reproducibility.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EngineOptions {
    /// Population size (must be >= 2 for crossover).
    pub pop_size: Option<usize>,
    /// Number of evolution cycles (must be >= 1).
    pub num_generations: Option<usize>,
    /// Population base seed. None -> random per run, recorded as run_seed.
    pub seed: Option<u64>,
    /// Shared topology template -- each individual clones and overrides seed.
    pub topology_options: TopologyOptions,
    /// Hidden-dim range sampled per individual. Empty -> fill with defaults.
    pub hidden_dim_pool: RangeInclusive<usize>,
    /// Stride for hidden_dim_pool sampling. Default: 1 (every value allowed).
    /// E.g. pool=32..=64, stride=16 → {32, 48, 64}.
    pub hidden_dim_stride: usize,
    /// Combine-op pool sampled per individual. Empty -> all built-in ops.
    pub combine_op_pool: Vec<CombineOp>,
    /// Activation pool -- per-node at creation, swap source for mutation.
    pub activation_pool: Vec<Activation>,
    /// Standardize-op pool -- per-node normalization. Empty -> all built-in.
    pub standardize_op_pool: Vec<crate::node::StandardizeOp>,
    /// Number of training batches per generation.
    pub train_num_batches: usize,
    /// Number of evaluation batches per generation.
    pub test_num_batches: usize,
    /// Rows per training batch.
    pub train_batch_size: usize,
    /// Rows per evaluation batch.
    pub test_batch_size: usize,
    // 
    pub fitness_label: FitnessLabel,
    // 
    pub train_metric_label: FitnessLabel,
    /// Threads for parallel eval (0 = rayon default).
    pub num_threads: usize,
    pub results_dir: PathBuf,
    /// Training config applied to every individual before scoring.
    pub training: crate::trainer::TrainingConfig,
    /// Selection strategy for the next generation.
    pub selection: Option<SelectionMethod>,
    /// Crossover strategy.
    pub crossover: Option<CrossoverMethod>,
    /// Mutation strategy.
    pub mutation: MutationMethod,
    /// Network execution options (device, dtype, seed).
    pub network: NetworkOptions,
    /// Dropout probability for hidden nodes (0.0 = no dropout).
    pub dropout_prob: f32,


    /// Sample train batches proportional to target class frequency.
    /// Default: false (uniform random sampling).
    pub train_y_proportional: bool,
    /// Sample test batches proportional to target class frequency.
    /// Default: false (uniform random sampling).
    pub test_y_proportional: bool,
    /// Deduplicate population by full topology comparison.
    /// Applied after create_population and after select each generation.
    /// Default: true.
    pub dedup_pop_and_fill: bool,
    /// Number of top individuals to preserve untouched each generation.
    /// Minimum: 1. Default: 2.
    pub elite_count: usize,
    /// Track per-generation metrics in engine.json history.
    /// Default: false.
    pub gens_history: bool,
    /// Paths to prior engine.json or improvement JSON files.
    /// Each best topology is injected into the initial population
    /// at pop[0..N] — a "warm start" from multiple prior runs.
    pub prior_topology_paths: Vec<PathBuf>,
    /// Train/test split ratio (train, test). Must sum to 1.0.
    /// Default: (0.8, 0.2). The eval set is fixed across generations;
    /// the train set is resampled each generation.
    pub train_test_split: (f32, f32),
    /// If true, train batches are resampled each generation (random data).
    /// If false, train batches use a fixed seed across gens.
    /// Default: false.
    pub train_random_data: bool,
    /// If true, eval batches are resampled each generation (random data).
    /// If false, eval batches use a fixed seed across gens (fair comparison).
    /// Default: false.
    pub test_random_data: bool,
    /// If true, train networks use deterministic (seeded) weights.
    /// If false, weights are random — evolution selects for architecture.
    /// Default: false.
    pub train_fixed_coefs: bool,

}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            // ── Required (must be set by user) ─────────────────────────
            pop_size: None,
            num_generations: None,
            mutation: MutationMethod::default(), // prob=0 → build() rejects unless user calls set_mutation()
            // ── Engine ────────────────────────────────────────────────
            seed: None,              // random if not set
            num_threads: 1,
            results_dir: PathBuf::from("results"),
            dedup_pop_and_fill: false,
            elite_count: 2,
            gens_history: false,
            prior_topology_paths: vec![],
            train_test_split: (0.8, 0.2),
            train_random_data: true,
            test_random_data: true,
            train_fixed_coefs: true,

            // ── Topology / GP pools ───────────────────────────────────
            topology_options: TopologyOptions::default(),
            hidden_dim_pool: 4..=8,
            hidden_dim_stride: 1,
            combine_op_pool: vec![],   // empty = all built-ins
            activation_pool: vec![],   // empty = all built-ins
            standardize_op_pool: vec![], // empty = all built-ins
            // ── Evaluation budget ─────────────────────────────────────
            train_num_batches: 16,
            test_num_batches: 16,
            train_batch_size: 32,
            test_batch_size: 32,
            // ── Training ──────────────────────────────────────────────
            training: crate::trainer::TrainingConfig::default(),
            fitness_label: FitnessLabel::default(),
            train_metric_label: FitnessLabel::default(),
            // ── Network ───────────────────────────────────────────────
            network: NetworkOptions::default(), // CPU, Float32
            dropout_prob: 0.1,
            // ── Genetics ──────────────────────────────────────────────
            selection: None,
            crossover: None,
            train_y_proportional: false,
            test_y_proportional: false,
        }
    }
}

impl EngineOptions {
    /// Derive topology options for one individual (clone template + override seed).
    pub(crate) fn derive_topology_options(&self, seed: usize) -> TopologyOptions {
        let mut t = self.topology_options;
        t.seed = seed;
        t
    }

    /// Start a fluent builder. Empty pools auto-fill with all built-ins at build() time.
    pub fn builder() -> EngineOptionsBuilder {
        EngineOptionsBuilder {
            inner: EngineOptions::default(),
        }
    }
}

// ── EngineOptionsBuilder -- fluent builder ─────────────────────────────────

pub struct EngineOptionsBuilder {
    inner: EngineOptions,
}

impl EngineOptionsBuilder {
    /// Validate required fields and return the accumulated options.
    /// Fills empty pools with all built-ins.
    pub fn build(mut self) -> Result<EngineOptions> {
        let o = &mut self.inner;
        // ── Required fields ────────────────────────────────────────────
        let pop_size = o.pop_size.ok_or_else(|| {
            EngineError::InvalidOptions("set_pop_size() is required — must be >= 2".into())
        })?;
        if pop_size < 2 {
            return Err(EngineError::InvalidOptions(
                "pop_size must be >= 2 (required for crossover)".into(),
            )
            .into());
        }
        let num_generations = o.num_generations.ok_or_else(|| {
            EngineError::InvalidOptions("set_num_generations() is required — must be >= 1".into())
        })?;
        if num_generations < 1 {
            return Err(EngineError::InvalidOptions(
                "num_generations must be >= 1 (at least one evolution cycle)".into(),
            )
            .into());
        }
        if o.elite_count < 1 {
            return Err(EngineError::InvalidOptions(
                "elite_count must be >= 1 (at least one elite must survive)".into(),
            )
            .into());
        }
        let split_sum = o.train_test_split.0 + o.train_test_split.1;
        if (split_sum - 1.0).abs() > f32::EPSILON {
            return Err(EngineError::InvalidOptions(
                format!("train_test_split must sum to 1.0, got {}+{}={:.2}",
                    o.train_test_split.0, o.train_test_split.1, split_sum),
            )
            .into());
        }
        if o.selection.is_none() {
            return Err(EngineError::InvalidOptions(
                "set_selection() is required — choose SelectionMethod::Tournament"
                    .into(),
            )
            .into());
        }
        if o.crossover.is_none() {
            return Err(EngineError::InvalidOptions(
                "set_crossover() is required — choose CrossoverMethod::OnePoint or Uniform"
                    .into(),
            )
            .into());
        }
        if o.mutation.prob() <= 0.0 {
            return Err(EngineError::InvalidOptions(
                "set_mutation() is required — choose MutationMethod::Activation, CombineOp, or Standardize"
                    .into(),
            )
            .into());
        }
        // Store validated values back as unwrapped
        o.pop_size = Some(pop_size);
        o.num_generations = Some(num_generations);
        // ── Conservative defaults for non-required fields ──────────────
        if o.train_num_batches > 0 && o.train_batch_size == 0 {
            o.train_batch_size = 32;
        }
        if o.test_num_batches > 0 && o.test_batch_size == 0 {
            o.test_batch_size = 32;
        }
        if o.hidden_dim_pool.is_empty() {
            o.hidden_dim_pool = 4..=8;
        }
        if o.training.num_epochs == 0 {
            o.training.num_epochs = 1;
        }
        // ── Auto-fill pools with all built-ins ─────────────────────────
        if o.combine_op_pool.is_empty() {
            o.combine_op_pool = crate::pools::all_combine_ops();
        }
        if o.activation_pool.is_empty() {
            o.activation_pool = crate::pools::all_activations();
        }
        if o.standardize_op_pool.is_empty() {
            o.standardize_op_pool = crate::pools::all_standardize_ops();
        }
        Ok(self.inner)
    }

    // Engine knobs
    pub fn set_pop_size(mut self, n: usize) -> Self {
        self.inner.pop_size = Some(n);
        self
    }
    pub fn set_num_generations(mut self, n: usize) -> Self {
        self.inner.num_generations = Some(n);
        self
    }
    pub fn set_seed(mut self, s: Option<u64>) -> Self {
        self.inner.seed = s;
        self
    }
    pub fn set_train_num_batches(mut self, n: usize) -> Self {
        self.inner.train_num_batches = n;
        self
    }
    pub fn set_test_num_batches(mut self, n: usize) -> Self {
        self.inner.test_num_batches = n;
        self
    }
    pub fn set_train_batch_size(mut self, n: usize) -> Self {
        self.inner.train_batch_size = n;
        self
    }
    pub fn set_test_batch_size(mut self, n: usize) -> Self {
        self.inner.test_batch_size = n;
        self
    }
    pub fn set_num_threads(mut self, n: usize) -> Self {
        self.inner.num_threads = n;
        self
    }
    pub fn set_results_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.inner.results_dir = p.into();
        self
    }
    pub fn set_selection(mut self, method: SelectionMethod) -> Self {
        self.inner.selection = Some(method);
        self
    }

    // Training knobs
    pub fn set_num_epochs(mut self, n: usize) -> Self {
        self.inner.training.num_epochs = n;
        self
    }
    pub fn set_learning_rate(mut self, lr: f32) -> Self {
        self.inner.training.learning_rate = lr;
        self
    }
    pub fn set_optimizer(mut self, kind: crate::trainer::OptimizerKind) -> Self {
        self.inner.training.optimizer = kind;
        self
    }
    pub fn set_grad_clip(mut self, max_norm: f32) -> Self {
        self.inner.training.grad_clip = max_norm;
        self
    }
    pub fn set_dropout_prob(mut self, p: f32) -> Self {
        self.inner.dropout_prob = p.clamp(0.0, 1.0);
        self
    }


    pub fn set_train_y_proportional(mut self, on: bool) -> Self {
        self.inner.train_y_proportional = on;
        self
    }
    pub fn set_test_y_proportional(mut self, on: bool) -> Self {
        self.inner.test_y_proportional = on;
        self
    }
    pub fn set_crossover(mut self, kind: CrossoverMethod) -> Self {
        self.inner.crossover = Some(kind);
        self
    }
    pub fn set_mutation(mut self, kind: MutationMethod) -> Self {
        self.inner.mutation = kind;
        self
    }
    pub fn set_dedup_pop_and_fill(mut self, on: bool) -> Self {
        self.inner.dedup_pop_and_fill = on;
        self
    }
    pub fn set_elite_count(mut self, n: usize) -> Self {
        self.inner.elite_count = n;
        self
    }
    pub fn set_gens_history(mut self, on: bool) -> Self {
        self.inner.gens_history = on;
        self
    }
    pub fn set_prior_topology(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner.prior_topology_paths.push(path.into());
        self
    }
    pub fn set_prior_topologies(mut self, paths: Vec<impl Into<PathBuf>>) -> Self {
        self.inner.prior_topology_paths.extend(paths.into_iter().map(|p| p.into()));
        self
    }
    pub fn set_train_test_split(mut self, train: f32, test: f32) -> Self {
        self.inner.train_test_split = (train, test);
        self
    }
    pub fn set_train_random_data(mut self, on: bool) -> Self {
        self.inner.train_random_data = on;
        self
    }
    pub fn set_test_random_data(mut self, on: bool) -> Self {
        self.inner.test_random_data = on;
        self
    }
    pub fn set_train_fixed_coefs(mut self, on: bool) -> Self {
        self.inner.train_fixed_coefs = on;
        self
    }


    // Topology template
    pub fn set_topology_options(mut self, t: TopologyOptions) -> Self {
        self.inner.topology_options = t;
        self
    }
    pub fn set_hidden_dim(mut self, n: usize) -> Self {
        self.inner.topology_options.hidden_dim = n;
        self
    }

    // GP search space
    pub fn set_hidden_dim_pool(mut self, r: RangeInclusive<usize>) -> Self {
        self.inner.hidden_dim_pool = r;
        self
    }
    pub fn set_hidden_dim_stride(mut self, s: usize) -> Self {
        self.inner.hidden_dim_stride = s.max(1);
        self
    }
    pub fn set_combine_op_pool(mut self, ops: Vec<CombineOp>) -> Self {
        self.inner.combine_op_pool = ops;
        self
    }
    pub fn set_activation_pool(mut self, acts: Vec<Activation>) -> Self {
        self.inner.activation_pool = acts;
        self
    }
    pub fn set_standardize_op_pool(mut self, ops: Vec<crate::node::StandardizeOp>) -> Self {
        self.inner.standardize_op_pool = ops;
        self
    }

    // Topology knobs (node/wire ranges)
    pub fn set_min_hidden_num_nodes(mut self, n: usize) -> Self {
        self.inner.topology_options.min_hidden_num_nodes = n;
        self
    }
    pub fn set_max_hidden_num_nodes(mut self, n: usize) -> Self {
        self.inner.topology_options.max_hidden_num_nodes = n;
        self
    }
    pub fn set_min_hidden_inputs_per_node(mut self, n: usize) -> Self {
        self.inner.topology_options.min_hidden_inputs_per_node = n;
        self
    }
    pub fn set_max_hidden_inputs_per_node(mut self, n: usize) -> Self {
        self.inner.topology_options.max_hidden_inputs_per_node = n;
        self
    }
    pub fn set_min_hidden_outputs_per_node(mut self, n: usize) -> Self {
        self.inner.topology_options.min_hidden_outputs_per_node = n;
        self
    }
    pub fn set_max_hidden_outputs_per_node(mut self, n: usize) -> Self {
        self.inner.topology_options.max_hidden_outputs_per_node = n;
        self
    }

    // Network options
    pub fn set_network(mut self, n: NetworkOptions) -> Self {
        self.inner.network = n;
        self
    }
    pub fn set_device(mut self, d: Device) -> Self {
        self.inner.network.device = d;
        self
    }
    pub fn set_dtype(mut self, d: flodl::DType) -> Self {
        self.inner.network.dtype = d;
        self
    }
    pub fn set_init_seed(mut self, seed: usize) -> Self {
        self.inner.network.seed = seed;
        self
    }
}

pub use crate::fitness::BestIndividual;

/// Per-generation metrics, accumulated when `gens_history` is enabled.
#[derive(Clone, Debug, Serialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_score: f32,
    pub best_loss: Option<f32>,
    pub worst_score: f32,
    pub worst_loss: Option<f32>,
    pub avg_score: f32,
    pub avg_loss: Option<f32>,
    pub best_params: usize,
    pub avg_params: f32,
    pub unique_topos: usize,
}

// ── Generation statistics (pure computation) ──────────────────────────────

/// Result of per-generation stat computation.
#[derive(Clone, Debug)]
pub(crate) struct GenStats {
    pub best_idx: usize,
    pub _worst_idx: usize,
    pub best_score: f32,
    pub best_loss: Option<f32>,
    pub worst_score: f32,
    pub worst_loss: Option<f32>,
    pub avg_score: f32,
    pub avg_loss: Option<f32>,
    pub best_params: usize,
    pub _worst_params: usize,
    pub avg_params: f32,
    pub unique_topos: usize,
}

/// Pure function: compute per-generation statistics from raw scores.
/// No mutation — safe to test in isolation.
pub(crate) fn compute_gen_stats(
    scores: &[f32],
    eval_losses: &[Option<f32>],
    param_counts: &[usize],
    direction: crate::fitness::Direction,
    pop: &[Topology],
) -> GenStats {
    use std::collections::HashSet;

    let n = scores.len();
    debug_assert!(!scores.is_empty());

    // Find best and worst indices
    let mut best_idx = 0;
    let mut worst_idx = 0;
    for (i, &score) in scores.iter().enumerate() {
        if direction.is_better(score, scores[best_idx]) {
            best_idx = i;
        }
        if direction.is_better(scores[worst_idx], score) {
            worst_idx = i;
        }
    }

    // Averages (filter NaN/Inf)
    let valid_scores: Vec<f32> = scores.iter().copied().filter(|v| v.is_finite()).collect();
    let avg_score = if valid_scores.is_empty() {
        0.0
    } else {
        valid_scores.iter().sum::<f32>() / valid_scores.len() as f32
    };
    let valid_losses: Vec<f32> = eval_losses.iter().filter_map(|&v| v).filter(|v| v.is_finite()).collect();
    let avg_loss = if valid_losses.is_empty() {
        None
    } else {
        Some(valid_losses.iter().sum::<f32>() / valid_losses.len() as f32)
    };

    // Param stats
    let best_params = param_counts[best_idx];
    let worst_params = param_counts[worst_idx];
    let avg_params = param_counts.iter().sum::<usize>() as f32 / n as f32;

    // Unique topologies (JSON-based dedup)
    let unique_topos = {
        let mut seen = HashSet::with_capacity(pop.len());
        for topo in pop {
            if let Ok(j) = topo.to_json() {
                seen.insert(j);
            }
        }
        seen.len()
    };

    GenStats {
        best_idx,
        _worst_idx: worst_idx,
        best_score: scores[best_idx],
        best_loss: eval_losses.get(best_idx).copied().flatten(),
        worst_score: scores[worst_idx],
        worst_loss: eval_losses.get(worst_idx).copied().flatten(),
        avg_score,
        avg_loss,
        best_params,
        _worst_params: worst_params,
        avg_params,
        unique_topos,
    }
}

// ── Robustness tracking ─────────────────────────────────────────────────

/// Tracks how often a topology appears across generations.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct RobustnessEntry {
    pub count: usize,
    pub min_fitness: f32,
    pub max_fitness: f32,
    pub mean: f32,
    pub m2: f32,
    pub param_count: usize,
    pub topology_json: String,
}

impl RobustnessEntry {
    fn new(fitness: f32, param_count: usize, topology_json: String) -> Self {
        Self {
            count: 1,
            min_fitness: fitness,
            max_fitness: fitness,
            mean: fitness,
            m2: 0.0,
            param_count,
            topology_json,
        }
    }

    /// Welford's online update — track mean and variance without storing all values.
    pub fn update(&mut self, fitness: f32) {
        self.count += 1;
        let delta = fitness - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = fitness - self.mean;
        self.m2 += delta * delta2;
    }

    /// Sample standard deviation of fitness across appearances.
    pub fn std_dev(&self) -> f32 {
        if self.count < 2 { 0.0 }
        else { (self.m2 / (self.count - 1) as f32).sqrt() }
    }
}

// ── Engine -- the NAS experiment runner ────────────────────────────────────

pub struct Engine {
    pub options: EngineOptions,
    pub seed: u64,
    pub run_id: String,
    pub run_dir: PathBuf,
    pool: rayon::ThreadPool,
    pub pop: Vec<Topology>,
    pub(crate) fitness: Fitness,
    pub(crate) dataset: Dataset,
    /// Indices into dataset for training (resampled each generation).
    pub(crate) train_indices: Vec<i64>,
    /// Indices into dataset for evaluation (fixed across generations).
    pub(crate) eval_indices: Vec<i64>,
    pub(crate) data_path: PathBuf,
    pub generation: usize,
    pub best: Option<BestIndividual>,
    pub history: Vec<GenerationStats>,

    scores: Vec<f32>,
    eval_losses: Vec<Option<f32>>,
    param_counts: Vec<usize>,
    /// Cached per-gen summary — computed once in save_generation_snapshots.
    gen_best_score: f32,
    gen_best_loss: Option<f32>,
    gen_worst_score: f32,
    gen_worst_loss: Option<f32>,
    gen_avg_score: f32,
    gen_avg_loss: Option<f32>,
    gen_best_params: usize,
    gen_avg_params: f32,
    gen_unique_topos: usize,
    /// Topology robustness tracker — keyed by topology JSON.
    robustness: std::collections::HashMap<String, RobustnessEntry>,
}

// ── Progress tracker — periodic stdout updates during evaluation ─────────

struct ProgressTracker {
    done: std::sync::atomic::AtomicUsize,
}

impl ProgressTracker {
    fn new() -> Self {
        ProgressTracker {
            done: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Workers call this after scoring. Just counts — no printing.
    fn increment(&self) {
        self.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Final count (no-op — logging handled elsewhere).
    fn finish(&self) {}
}

impl Engine {
    // ── Construction ────────────────────────────────────────────────────────

    /// Load dataset, seed population. Auto-detects input_dim/output_dim from data.
    pub fn new(mut options: EngineOptions, data_path: &Path, fitness: Fitness) -> Result<Self> {
        // Step 1: Validate options and fill empty pools
        Self::validate_and_fill_options(&mut options)?;

        // Step 2: Resolve seed
        let seed = Self::resolve_seed(&mut options);

        // Step 3: Load data and bind dims
        let dataset = Self::load_data(&mut options, data_path)?;

        // Step 3b: Split dataset into train/eval indices
        let (train_indices, eval_indices) = Self::split_dataset(&options, &dataset, seed);

        // Step 4: Create population
        let mut pop = Self::create_population(&options, seed)?;

        // Step 4b: Warm start — inject prior topologies if provided
        for (idx, path) in options.prior_topology_paths.iter().enumerate() {
            if idx >= pop.len() {
                break;
            }
            let topo = Self::load_prior_topology(path)?;
            pop[idx] = topo;
            log::info!("  warm start: loaded prior topology into pop[{}] from {}", idx, path.display());
        }

        if options.dedup_pop_and_fill {
            Self::dedup_population(&mut pop);
            Self::refill_population(&options, seed, 0, &mut pop);
        }

        // Step 5: Log initialization
        Self::log_initialization(&options, &dataset, &pop, seed, &fitness, data_path);

        // Step 6: Assemble engine (thread pool, run dir, struct)
        Self::assemble_engine(options, seed, dataset, train_indices, eval_indices, pop, fitness, data_path)
    }

    /// Step 1: Propagate engine-level pools into mutation variant, fill defaults.
    /// All required validation is done in `EngineOptionsBuilder::build()`.
    fn validate_and_fill_options(options: &mut EngineOptions) -> Result<()> {
        // Sanity checks (builder auto-fills these, but direct construction may not).
        if options.train_num_batches > 0 && options.train_batch_size == 0 {
            return Err(EngineError::InvalidOptions(
                "train_num_batches > 0 requires train_batch_size > 0".into(),
            )
            .into());
        }
        if options.test_num_batches > 0 && options.test_batch_size == 0 {
            return Err(EngineError::InvalidOptions(
                "test_num_batches > 0 requires test_batch_size > 0".into(),
            )
            .into());
        }
        if options.train_num_batches == 0 {
            options.train_num_batches = 16;
        }
        if options.test_num_batches == 0 {
            options.test_num_batches = 16;
        }
        if options.train_batch_size == 0 {
            options.train_batch_size = 32;
        }
        if options.test_batch_size == 0 {
            options.test_batch_size = 32;
        }
        if options.training.num_epochs == 0 {
            options.training.num_epochs = 1;
        }
        if options.hidden_dim_pool.is_empty() {
            options.hidden_dim_pool = 4..=8;
        }
        if options.hidden_dim_stride == 0 {
            options.hidden_dim_stride = 1;
        }

        // Auto-fill empty pools with all built-ins.
        if options.combine_op_pool.is_empty() {
            options.combine_op_pool = crate::pools::all_combine_ops();
        }
        if options.activation_pool.is_empty() {
            options.activation_pool = crate::pools::all_activations();
        }
        if options.standardize_op_pool.is_empty() {
            options.standardize_op_pool = crate::pools::all_standardize_ops();
        }
        // Propagate engine-level settings into training config.
        options.training.train_y_proportional = options.train_y_proportional;
        options.training.test_y_proportional = options.test_y_proportional;
        options.training.num_batches_train = options.train_num_batches;
        options.training.num_batches_eval = options.test_num_batches;
        options.training.batch_size_train = options.train_batch_size;
        options.training.batch_size_eval = options.test_batch_size;
        Ok(())
    }

    /// Step 2: Resolve seed -- user-provided or random.
    fn resolve_seed(options: &mut EngineOptions) -> u64 {
        let seed = options.seed.unwrap_or_else(|| {
            let t = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            t ^ fastrand::u64(..)
        });
        options.topology_options.seed = seed as usize;
        options.network.seed = seed as usize;
        seed
    }

    /// Auto-detect data format and load dataset.
    ///
    /// Priority:
    /// 1. `flodl_data/inputs.bin` — cached .bin from previous conversion
    /// 2. `inputs.bin` — user-provided .bin
    /// 3. `inputs.csv` — user-provided CSV → convert to .bin, cache in `flodl_data/`
    fn resolve_dataset(data_path: &Path) -> Result<Dataset> {
        let cache_dir = data_path.join("flodl_data");

        // Priority 1: cached .bin
        if cache_dir.join("inputs.bin").exists() {
            debug!("load_data: using cached .bin from {}", cache_dir.display());
            return crate::utils::data::load_dataset(&cache_dir);
        }

        // Priority 2: .bin in data_path
        if data_path.join("inputs.bin").exists() {
            debug!("load_data: using .bin from {}", data_path.display());
            return crate::utils::data::load_dataset(data_path);
        }

        // Priority 3: CSV → convert → cache
        if data_path.join("inputs.csv").exists() {
            debug!("load_data: converting CSV to .bin in {}", cache_dir.display());
            let dataset = crate::utils::data::load_csv_dataset(data_path)?;
            std::fs::create_dir_all(&cache_dir).map_err(|e| flodl::tensor::TensorError::new(&e.to_string()))?;
            crate::utils::data::save_dataset(&cache_dir, &dataset)?;
            return Ok(dataset);
        }

        Err(EngineError::DataMismatch(format!(
            "no inputs.bin or inputs.csv found in {}", data_path.display()
        )).into())
    }

    /// Step 3: Load dataset and bind input_dim/output_dim to options.
    /// Auto-detects format: .bin (cached or direct) or .csv (converts to .bin).
    fn load_data(options: &mut EngineOptions, data_path: &Path) -> Result<Dataset> {
        let dataset = Self::resolve_dataset(data_path)?
            .to_dtype(options.network.dtype)?
            .to_device(options.network.device)?;
        let data_in = dataset.inputs.shape().get(1).copied().ok_or_else(|| {
            EngineError::DataMismatch("dataset inputs must be 2-D [n, input_dim]".into())
        })?;
        options.topology_options.input_dim = data_in as usize;
        let data_out = dataset.targets.shape().get(1).copied().ok_or_else(|| {
            EngineError::DataMismatch("dataset targets must be 2-D [n, output_dim]".into())
        })?;
        options.topology_options.output_dim = data_out as usize;
        options.fitness_label = crate::fitness::FitnessLabel(options.fitness_label.0.clone());
        options.train_metric_label =
            crate::fitness::FitnessLabel(options.train_metric_label.0.clone());
        debug!(
            "Engine::new -- input_dim={} output_dim={} seed={}",
            options.topology_options.input_dim,
            options.topology_options.output_dim,
            options.seed.unwrap_or(0)
        );
        Ok(dataset)
    }

    /// Step 3b: Split dataset indices into train and eval pools.
    /// Eval indices are fixed across generations; train indices are resampled.
    fn split_dataset(options: &EngineOptions, dataset: &Dataset, seed: u64) -> (Vec<i64>, Vec<i64>) {
        let n = dataset.inputs.shape()[0] as usize;
        let (_train_ratio, eval_ratio) = options.train_test_split;
        let eval_count = (n as f32 * eval_ratio).round() as usize;
        let train_count = n - eval_count;

        // Shuffle all indices deterministically
        let mut indices: Vec<i64> = (0..n as i64).collect();
        let mut rng = fastrand::Rng::with_seed(seed);
        for i in (1..indices.len()).rev() {
            let j = rng.usize(0..=i);
            indices.swap(i, j);
        }

        let eval_indices = indices[..eval_count].to_vec();
        let train_indices = indices[eval_count..eval_count + train_count].to_vec();

        debug!(
            "  split_dataset -- total={} train={} eval={} (split {:.0}%/{:.0}%)",
            n, train_indices.len(), eval_indices.len(),
            options.train_test_split.0 * 100.0, options.train_test_split.1 * 100.0
        );
        (train_indices, eval_indices)
    }

    /// Step 4: Create a population of random topologies, seeded deterministically.
    fn create_population(options: &EngineOptions, seed: u64) -> Result<Vec<Topology>> {
        let pop_size = options.pop_size.unwrap();
        let mut pop = Vec::with_capacity(pop_size);
        for i in 0..pop_size {
            let ind_seed = derive_seed(seed, i);
            let mut rng = fastrand::Rng::with_seed(ind_seed);
            let n_hidden = rng.usize(
                options.topology_options.min_hidden_num_nodes
                    ..=options.topology_options.max_hidden_num_nodes,
            );
            let ind_opts = options.derive_topology_options(ind_seed as usize);

            // Create a random topology with n_hidden nodes, each node randomly assigned hidden_dim, activation, combine_op, and standardize_op from the respective pools.
            let mut graph = Topology::new(i, Some(ind_opts));
            graph.create_random_hidden_nodes(n_hidden);
            let pool_len_a = options.activation_pool.len();
            let pool_len_c = options.combine_op_pool.len();
            let pool_len_s = options.standardize_op_pool.len();
            for node in &mut graph.nodes {
                if node.kind == NodeKind::Hidden {
                    node.hidden_dim = Some(Self::sample_hidden_dim(&options.hidden_dim_pool, options.hidden_dim_stride, &mut rng));
                    node.activation = options.activation_pool[rng.usize(0..pool_len_a)];
                    node.combine_op = Some(options.combine_op_pool[rng.usize(0..pool_len_c)]);
                    node.standardize = Some(options.standardize_op_pool[rng.usize(0..pool_len_s)]);
                }
            }
            graph.refresh_labels();
            graph.finalize();



            debug!(
                "  ind[{i}] seed={} n_hidden={} nodes={} wires={}",
                ind_seed,
                n_hidden,
                graph.nodes.len(),
                graph.connections.len()
            );
            pop.push(graph);
        }
        Ok(pop)
    }

    /// Remove duplicate topologies from the population (full Spec comparison).
    /// Keeps the first occurrence of each unique topology.
    fn dedup_population(pop: &mut Vec<Topology>) {
        use crate::spec::Spec;
        let before = pop.len();
        let mut seen: Vec<Spec> = Vec::new();
        pop.retain(|topo| {
            let spec = Spec::from(topo);
            if seen.iter().any(|s| *s == spec) {
                false
            } else {
                seen.push(spec);
                true
            }
        });
        let removed = before - pop.len();
        if removed > 0 {
            debug!("  dedup: removed {removed} duplicates, {}/{} remain", pop.len(), before);
        }
    }

    /// Load a prior topology from an engine.json or improvement JSON file.
    /// Accepts:
    /// - `best_topology` — engine.json (best of best)
    /// - `file_topology` — improvement JSON (per-gen best/worst)
    /// - `topology` — legacy format
    /// The field can be a nested object or escaped JSON string.
    fn load_prior_topology(path: &Path) -> Result<Topology> {
        let raw = fs::read_to_string(path).map_err(|source| EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| EngineError::Json(format!("prior topology parse: {e}")))?;
        let topo_val = v.get("best_topology")
            .or_else(|| v.get("file_topology"))
            .or_else(|| v.get("topology"))
            .ok_or_else(|| EngineError::Json(
                format!("prior topology file has no 'best_topology', 'file_topology', or 'topology' field: {}", path.display())
            ))?;
        // Handle both formats: nested object vs escaped JSON string
        let topo_str = match topo_val {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other)
                .map_err(|e| EngineError::Json(format!("prior topology serialize: {e}")))?,
        };
        Topology::from_json(&topo_str)
            .map_err(|e| EngineError::Json(format!("prior topology from_json: {e}")).into())
    }

    /// Refill population back to `pop_size` with fresh random individuals.
    /// Called after dedup to keep the population at full strength.
    /// Returns the number of individuals added.
    fn refill_population(
        options: &EngineOptions,
        seed: u64,
        generation: usize,
        pop: &mut Vec<Topology>,
    ) -> usize {
        let target = options.pop_size.unwrap();
        if pop.len() >= target {
            return 0;
        }
        let needed = target - pop.len();
        // Offset into a disjoint seed space (initial pop uses 0..pop_size).
        let base_offset = 1_000_000 + generation * target + pop.len();
        for k in 0..needed {
            let ind_seed = derive_seed(seed, base_offset + k);
            let mut rng = fastrand::Rng::with_seed(ind_seed);
            let n_hidden = rng.usize(
                options.topology_options.min_hidden_num_nodes
                    ..=options.topology_options.max_hidden_num_nodes,
            );
            let ind_opts = options.derive_topology_options(ind_seed as usize);
            let id = pop.len();
            let mut graph = Topology::new(id, Some(ind_opts));
            graph.create_random_hidden_nodes(n_hidden);
            let pool_len_a = options.activation_pool.len();
            let pool_len_c = options.combine_op_pool.len();
            let pool_len_s = options.standardize_op_pool.len();
            for node in &mut graph.nodes {
                if node.kind == NodeKind::Hidden {
                    node.hidden_dim = Some(Self::sample_hidden_dim(&options.hidden_dim_pool, options.hidden_dim_stride, &mut rng));
                    node.activation = options.activation_pool[rng.usize(0..pool_len_a)];
                    node.combine_op = Some(options.combine_op_pool[rng.usize(0..pool_len_c)]);
                    node.standardize = Some(options.standardize_op_pool[rng.usize(0..pool_len_s)]);
                }
            }
            graph.refresh_labels();
            graph.finalize();
            debug!(
                "  refill[{id}] seed={} n_hidden={} nodes={} wires={}",
                ind_seed, n_hidden, graph.nodes.len(), graph.connections.len()
            );
            pop.push(graph);
        }
        debug!("  refill: added {needed} individuals, {}/{} remain", pop.len(), target);
        needed
    }

    /// Step 5: Log all resolved options, dataset, and population.
    fn log_initialization(
        options: &EngineOptions,
        dataset: &Dataset,
        pop: &[Topology],
        seed: u64,
        fitness: &Fitness,
        data_path: &Path,
    ) {
        crate::utils::log_utils::log_initialization(options, dataset, pop, seed, fitness, data_path);
    }

    /// Step 6: Build thread pool, generate run id, assemble Engine struct.
    fn assemble_engine(
        options: EngineOptions,
        seed: u64,
        dataset: Dataset,
        train_indices: Vec<i64>,
        eval_indices: Vec<i64>,
        pop: Vec<Topology>,
        fitness: Fitness,
        data_path: &Path,
    ) -> Result<Engine> {
        let threads = if options.num_threads > 0 {
            options.num_threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| EngineError::Rayon(e.to_string()))?;
        let run_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
            .to_string();
        let run_dir = options.results_dir.join(&run_id);
        Ok(Engine {
            options,
            seed,
            run_id,
            run_dir,
            pool,
            pop,
            fitness,
            dataset,
            train_indices,
            eval_indices,
            data_path: data_path.to_path_buf(),
            generation: 0,
            best: None,
            history: Vec::new(),

            scores: Vec::new(),
            eval_losses: Vec::new(),
            param_counts: Vec::new(),
            gen_best_score: 0.0,
            gen_best_loss: None,
            gen_worst_score: 0.0,
            gen_worst_loss: None,
            gen_avg_score: 0.0,
            gen_avg_loss: None,
            gen_best_params: 0,
            gen_avg_params: 0.0,
            gen_unique_topos: 0,
            robustness: std::collections::HashMap::new(),
        })
    }

    // ── Query ───────────────────────────────────────────────────────────────

    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    // ── Run -- the main loop ─────────────────────────────────────────────────

    pub fn run(&mut self) -> Result<()> {
        let run_start = self.init_run()?;
        self.run_generations();
        self.finalize_run(run_start)
    }

    /// Phase 1: Create dirs, write initial engine.json, start timer.
    fn init_run(&mut self) -> Result<Instant> {
        fs::create_dir_all(&self.run_dir).map_err(|source| EngineError::Io {
            path: self.run_dir.display().to_string(),
            source,
        })?;
        crate::utils::log_utils::log_run_start(&self.run_dir);
        let initial = self.to_json()?;
        fs::write(self.run_dir.join("engine.json"), initial).map_err(|source| EngineError::Io {
            path: self.run_dir.join("engine.json").display().to_string(),
            source,
        })?;
        Ok(Instant::now())
    }

    /// Phase 2: The evolution loop.
    fn run_generations(&mut self) {
        let num_gens = self.options.num_generations.unwrap();
        for g in 0..num_gens {
            let gen_start = Instant::now();
            debug!("== gen {:02}/{:02} ==", g, num_gens);
            log::info!("");
            log::info!("  ── gen {:02} of {:02}  ──", self.generation, num_gens);
            let _improved = self.evaluate_population();
            self.log_generation_summary();
            self.next_generation();
            log::info!("  gen {:02} done in {:.1}s", g, gen_start.elapsed().as_secs_f64());
        }
    }

    /// Phase 3: Write final engine.json, log run summary.
    fn finalize_run(&mut self, run_start: Instant) -> Result<()> {
        let run_elapsed = run_start.elapsed();
        let json = self.to_json()?;
        fs::write(self.run_dir.join("engine.json"), json).map_err(|source| EngineError::Io {
            path: self.run_dir.join("engine.json").display().to_string(),
            source,
        })?;
        crate::utils::log_utils::log_run_summary(
            run_elapsed,
            &self.best,
            &self.fitness,
            &self.run_dir,
        )
        .map_err(|e| EngineError::Json(format!("run summary log: {e}")))?;

        // Log top-N robust topologies
        crate::utils::log_utils::log_repeated_topologies(&self.robustness, 20);

        Ok(())
    }

    // ── Evaluation ──────────────────────────────────────────────────────────

    /// Score every individual. Returns whether the overall best improved.
    fn evaluate_population(&mut self) -> Result<bool> {
        // Step 1: Parallel eval -- build, train, score each individual
        let results = self.eval_all_individuals()?;

        // Step 2: Store scores and losses
        self.update_scores(results);

        // Step 3: Save best + worst for this generation
        self.save_generation_snapshots()
    }

    /// Step 1: Parallel rayon loop -- build, train, score each individual.
    fn eval_all_individuals(&self) -> Result<Vec<(f32, Option<f32>, usize)>> {
        let net_opts = self.options.network;
        let train_cfg = &self.options.training;
        let fitness = &self.fitness;
        let dataset = &self.dataset;
        let train_indices = &self.train_indices;
        let eval_indices = &self.eval_indices;
        let batch_seed = if self.options.train_random_data {
            derive_seed(self.seed, self.generation * 3)  // changes per gen
        } else {
            derive_seed(self.seed, 0)  // fixed across gens
        };
        let test_seed = if self.options.test_random_data {
            derive_seed(self.seed, self.generation * 3 + 1)  // changes per gen
        } else {
            derive_seed(self.seed, usize::MAX)  // fixed across generations
        };
        let tracker = ProgressTracker::new();

        let results = self.pool.install(|| {
            self.pop
                .par_iter()
                .map(|graph| {
                    let mut no = net_opts;
                    no.seed = graph.options.seed;
                    no.dropout_prob = self.options.dropout_prob;
                    // Randomize weights when train_fixed_coefs is false —
                    // isolates architecture selection from weight init luck.
                    if !self.options.train_fixed_coefs {
                        no.seed = fastrand::usize(..);
                    }
                    let mut net = Network::build_with_options(graph, &no)?;
                    let result = crate::trainer::train_network(
                        &mut net, train_cfg, fitness, dataset, train_indices, eval_indices, batch_seed, test_seed,
                    )?;
                    let param_count: usize = net.layers.iter().flat_map(|l| l.parameters()).map(|p| p.variable.numel() as usize).sum();
                    tracker.increment();
                    Ok((result.score, result.eval_loss, param_count))
                })
                .collect::<Result<Vec<_>>>()
        });
        tracker.finish();
        results
    }

    /// Step 2: Store scores, eval_losses, and param_counts from parallel results.
    fn update_scores(&mut self, results: Vec<(f32, Option<f32>, usize)>) {
        self.scores = results.iter().map(|&(s, _, _)| s).collect();
        self.eval_losses = results.iter().map(|&(_, l, _)| l).collect();
        self.param_counts = results.iter().map(|&(_, _, c)| c).collect();
    }

    /// Step 3: Find best + worst in current gen, save both to disk.
    fn save_generation_snapshots(&mut self) -> Result<bool> {
        if self.scores.is_empty() {
            return Ok(false);
        }

        // 1. Compute stats (pure function — no mutation)
        let direction = self.fitness.direction();
        let stats = compute_gen_stats(
            &self.scores,
            &self.eval_losses,
            &self.param_counts,
            direction,
            &self.pop,
        );

        // 2. Cache for logging / history
        self.gen_best_score = stats.best_score;
        self.gen_best_loss = stats.best_loss;
        self.gen_worst_score = stats.worst_score;
        self.gen_worst_loss = stats.worst_loss;
        self.gen_avg_score = stats.avg_score;
        self.gen_avg_loss = stats.avg_loss;
        self.gen_best_params = stats.best_params;
        self.gen_avg_params = stats.avg_params;
        self.gen_unique_topos = stats.unique_topos;

        // 3. Update overall best if this gen's best is better
        let improved = self
            .best
            .as_ref()
            .map(|b| direction.is_better(stats.best_score, b.fitness))
            .unwrap_or(true);
        if improved {
            self.best = Some(BestIndividual {
                fitness: stats.best_score,
                loss: stats.best_loss,
                pop_index: stats.best_idx,
                topology: self.pop[stats.best_idx].clone(),
                param_count: stats.best_params,
            });
        }

        // 4. Save per-gen data (one JSON + one MD)
        self.save_gen_data(&stats)?;

        // 5. Update robustness tracker
        self.update_robustness(&stats);

        Ok(improved)
    }

    // ── Per-gen data persistence ────────────────────────────────────────────

    /// Save one JSON + one MD per generation.
    /// JSON contains all individuals with metrics + topologies.
    /// MD is for the best topology only.
    fn save_gen_data(&self, stats: &GenStats) -> std::result::Result<(), EngineError> {
        let dir = self.run_dir.join("improvements");
        fs::create_dir_all(&dir).map_err(|source| EngineError::Io {
            path: dir.display().to_string(),
            source,
        })?;

        // Build individuals array
        let mut individuals = Vec::with_capacity(self.pop.len());
        for (i, topo) in self.pop.iter().enumerate() {
            let fitness = self.scores[i];
            let loss = self.eval_losses.get(i).copied().flatten();
            let params = self.param_counts[i];
            let topo_json = topo.to_json()
                .map_err(|e| EngineError::Json(format!("individual topo json: {e}")))?;
            // Hash: SHA-like digest of full topology JSON — unique per topology
            let topo_hash = {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                topo_json.hash(&mut hasher);
                let h = hasher.finish();
                format!("{:016x}", h)[..8].to_string()  // first 8 hex chars
            };
            individuals.push(serde_json::json!({
                "idx": i,
                "seed": topo.options.seed,
                "fitness": fitness,
                "loss": loss,
                "params": params,
                "topo_hash": topo_hash,
                "topology": topo_json,
            }));
        }

        // Write gen JSON
        let gen_json = serde_json::json!({
            "generation": self.generation,
            "unique_topos": stats.unique_topos,
            "stats": {
                "best_score": stats.best_score,
                "best_loss": stats.best_loss,
                "best_params": stats.best_params,
                "worst_score": stats.worst_score,
                "worst_loss": stats.worst_loss,
                "avg_score": stats.avg_score,
                "avg_loss": stats.avg_loss,
            },
            "individuals": individuals,
        });
        let json_str = serde_json::to_string_pretty(&gen_json)
            .map_err(|e| EngineError::Json(format!("gen json: {e}")))?;
        let json_path = dir.join(format!("gen_{:02}.json", self.generation));
        fs::write(&json_path, &json_str).map_err(|source| EngineError::Io {
            path: json_path.display().to_string(),
            source,
        })?;

        // Write MD for best topology only
        let best_topo = &self.pop[stats.best_idx];
        let net = Network::build(best_topo, Device::CPU).ok();
        let md = crate::utils::markdown::topology_markdown(
            best_topo,
            Some(stats.best_score),
            net.as_ref(),
        );
        let md_path = dir.join(format!("gen_{:02}.md", self.generation));
        fs::write(&md_path, md).map_err(|source| EngineError::Io {
            path: md_path.display().to_string(),
            source,
        })?;

        debug!("save_gen_data -- gen={} saved", self.generation);
        Ok(())
    }

    /// Update the robustness tracker with this gen's population.
    fn update_robustness(&mut self, _stats: &GenStats) {
        for (i, topo) in self.pop.iter().enumerate() {
            let fitness = self.scores[i];
            let key = match topo.to_json() {
                Ok(j) => j,
                Err(_) => continue,
            };
            use std::collections::hash_map::Entry;
            match self.robustness.entry(key) {
                Entry::Occupied(mut e) => {
                    let entry = e.get_mut();
                    entry.update(fitness);
                    if fitness < entry.min_fitness { entry.min_fitness = fitness; }
                    if fitness > entry.max_fitness { entry.max_fitness = fitness; }
                }
                Entry::Vacant(e) => {
                    let topo_json = e.key().clone();
                    let param_count = self.param_counts.get(i).copied().unwrap_or(0);
                    e.insert(RobustnessEntry::new(fitness, param_count, topo_json));
                }
            }
        }
    }

    // ── Logging ─────────────────────────────────────────────────────────────

    fn log_generation_summary(&self) {
        if self.scores.is_empty() {
            return;
        }
        let global_best = self.best.as_ref().map(|b| b.fitness).unwrap_or(self.gen_best_score);
        let fl = self.fitness.fitness_label();
        log::info!("  {:<16}{:>8}  {:<10}{:>8}  {:<5}{:>8}",
            format!("{fl} global:"), format!("{global_best:.4}"),
            "gen_best:", format!("{:.4}", self.gen_best_score),
            "avg:", format!("{:.4}", self.gen_avg_score));
        if self.gen_best_loss.is_some() || self.gen_avg_loss.is_some() {
            let ll = self.fitness.train_metric_label();
            let global_l = self.best.as_ref().and_then(|b| b.loss)
                .map(|v| format!("{v:.4}")).unwrap_or_else(|| "-".into());
            let best_l = self.gen_best_loss.map(|v| format!("{v:.4}")).unwrap_or_else(|| "-".into());
            let avg_l = self.gen_avg_loss.map(|v| format!("{v:.4}")).unwrap_or_else(|| "-".into());
            log::info!("  {:<16}{:>8}  {:<10}{:>8}  {:<5}{:>8}",
                format!("{ll} global:"), &global_l, "gen_best:", &best_l, "avg:", &avg_l);
        }
        // Params (learnable parameters)
        let global_params = self.best.as_ref().map(|b| b.param_count).unwrap_or(self.gen_best_params);
        log::info!("  {:<16}{:>10}  {:<12}{:>10}  {:<5}{:>10}",
            "params global:", global_params,
            "gen_best:", self.gen_best_params,
            "avg:", format!("{:.0}", self.gen_avg_params));
        // Unique topologies
        log::info!("  {:<16}{:>10}/{:<10}",
            "topologies:",
            self.gen_unique_topos,
            self.pop.len());
    }

    // ── Genetics -- selection, crossover, mutation ────────────────────────────

    /// Sample a hidden_dim from the pool, respecting stride.
    fn sample_hidden_dim(pool: &RangeInclusive<usize>, stride: usize, rng: &mut fastrand::Rng) -> usize {
        let start = *pool.start();
        let end = *pool.end();
        let n = ((end - start) / stride) + 1;
        start + rng.usize(0..n) * stride
    }

    fn next_generation(&mut self) {
        let (unique, sel_label) = self.select();
        let cx_pairs = self.crossover();
        let pre_dedup = self.pop.len();
        if self.options.dedup_pop_and_fill {
            Self::dedup_population(&mut self.pop);
        }
        let dedup_removed = pre_dedup - self.pop.len();
        let refill_added = Self::refill_population(
            &self.options,
            self.seed,
            self.generation,
            &mut self.pop,
        );
        // Post-refill safety: catch any accidental duplicates between refilled
        // individuals and existing crossover survivors.
        let pre_post_dedup = self.pop.len();
        if self.options.dedup_pop_and_fill && refill_added > 0 {
            Self::dedup_population(&mut self.pop);
        }
        let post_refill_dedup = pre_post_dedup - self.pop.len();
        let mut_count = self.mutate();
        // Log in order of operations: select → crossover → dedup → refill → dedup → mutate
        let target = self.options.pop_size.unwrap();
        log::info!("  ── genetics ──");
        log::info!("  {:<14}{:>5}/{:<5}  unique ({})  elite {}", "selection", unique, target, sel_label, self.options.elite_count);
        log::info!("  {:<14}{:>5} pairs  {:>5}/{:<5}", "crossover", cx_pairs, target, target);
        if dedup_removed > 0 {
            log::info!("  {:<14}  -{:<4}  {:>5}/{} → {:>5}/{}", "dedup", dedup_removed, pre_dedup, target, pre_dedup - dedup_removed, target);
        } else {
            log::info!("  {:<14}{:>5}  {:>5}/{}", "dedup", 0, target, target);
        }
        if refill_added > 0 {
            let after_refill = self.pop.len();
            log::info!("  {:<14}{:>5}  {:>5}/{} → {:>5}/{}", "refill", refill_added, after_refill - refill_added, target, after_refill, target);
        } else {
            log::info!("  {:<14}{:>5}  {:>5}/{}", "refill", 0, self.pop.len(), target);
        }
        if post_refill_dedup > 0 {
            log::info!("  {:<14}  -{:<4}  {:>5}/{}", "dedup (post)", post_refill_dedup, self.pop.len(), target);
        }
        log::info!("  {:<14}{:>5} nets  {:>5}/{}", "mutation", mut_count, self.pop.len(), target);
        // Capture history if enabled (uses cached values from save_generation_snapshots)
        if self.options.gens_history && !self.scores.is_empty() {
            self.history.push(GenerationStats {
                generation: self.generation,
                best_score: self.gen_best_score,
                best_loss: self.gen_best_loss,
                worst_score: self.gen_worst_score,
                worst_loss: self.gen_worst_loss,
                avg_score: self.gen_avg_score,
                avg_loss: self.gen_avg_loss,
                best_params: self.gen_best_params,
                avg_params: self.gen_avg_params,
                unique_topos: self.gen_unique_topos,
            });
        }
        self.generation += 1;
    }

    /// Selection -- reorder pop/scores so fittest survive.
    /// Returns (unique_survivors, selection_label).
    pub fn select(&mut self) -> (usize, String) {
        if self.scores.is_empty() {
            return (0, self.options.selection.as_ref().unwrap().label().to_string());
        }
        let dir = self.fitness.direction();
        let mut rng = fastrand::Rng::with_seed(derive_seed(self.seed, self.generation * 3 + 1));
        let selection = self.options.selection.as_ref().unwrap();
        let indices = selection.apply(&self.scores, dir, &mut rng, self.options.elite_count);
        let label = selection.label().to_string();

        // In-place reorder: build new pop from selected indices.
        let new_pop: Vec<Topology> = indices.iter().map(|&i| self.pop[i].clone()).collect();
        let new_scores: Vec<f32> = indices.iter().map(|&i| self.scores[i]).collect();
        self.pop = new_pop;
        self.scores = new_scores;

        let mut counts = vec![0usize; self.pop.len()];
        for &i in &indices {
            counts[i] += 1;
        }
        let unique = counts.iter().filter(|&&c| c > 0).count();
        debug!(
            "  selection [{}] {} unique survivors",
            label, unique
        );
        (unique, label)
    }

    /// Crossover — DEAP-style: clone pop, pair up, apply crossover kind.
    /// Returns number of pairs actually crossed.
    pub fn crossover(&mut self) -> usize {
        let pop_size = self.pop.len();
        if pop_size < 2 {
            return 0;
        }
        let mut rng = fastrand::Rng::with_seed(derive_seed(self.seed, self.generation * 3 + 2));
        let kind = self.options.crossover.as_ref().unwrap();
        let cxpb = kind.action_prob();
        if cxpb <= 0.0 {
            return 0;
        }

        let mut offspring = self.pop.clone();
        let mut cx_count = 0usize;
        let mut i = 0;
        while i + 1 < pop_size {
            if rng.f32() < cxpb {
                let (left, right) = offspring.split_at_mut(i + 1);
                let crossed = match &kind {
                    CrossoverMethod::OnePoint { .. } => {
                        Topology::cx_one_point(&mut left[i], &mut right[0], &mut rng)
                    }
                    CrossoverMethod::Uniform { swap_prob, .. } => {
                        Topology::cx_uniform(&mut left[i], &mut right[0], *swap_prob, &mut rng)
                    }
                };
                if crossed {
                    cx_count += 1;
                }
            }
            i += 2;
        }

        debug!("  crossover {cx_count} pairs ({kind})");
        self.pop = offspring;
        cx_count
    }

    /// Mutation — one roll per individual; if it hits, mutate one random
    /// hidden node according to the configured variant. The mutation pool
    /// is always taken from the engine-level pools. Returns individuals mutated.
    pub fn mutate(&mut self) -> usize {
        let mut rng = fastrand::Rng::with_seed(derive_seed(self.seed, self.generation * 3 + 3));
        let m = &self.options.mutation;
        if m.prob() <= 0.0 {
            return 0;
        }

        let mut mut_count = 0usize;
        for topo in &mut self.pop {
            if rng.f32() >= m.prob() {
                continue;
            }
            // Collect hidden node indices
            let hidden: Vec<usize> = topo
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.kind == NodeKind::Hidden)
                .map(|(i, _)| i)
                .collect();
            if hidden.is_empty() {
                continue;
            }
            let node_idx = hidden[rng.usize(0..hidden.len())];
            let node = &mut topo.nodes[node_idx];
            match m {
                MutationMethod::Activation { .. } => {
                    let pool = &self.options.activation_pool;
                    if pool.is_empty() { continue; }
                    node.activation = pool[rng.usize(0..pool.len())];
                }
                MutationMethod::CombineOp { .. } => {
                    let pool = &self.options.combine_op_pool;
                    if pool.is_empty() { continue; }
                    node.combine_op = Some(pool[rng.usize(0..pool.len())]);
                }
                MutationMethod::Standardize { .. } => {
                    let pool = &self.options.standardize_op_pool;
                    if pool.is_empty() { continue; }
                    node.standardize = Some(pool[rng.usize(0..pool.len())]);
                }
            }
            topo.finalize();
            mut_count += 1;
        }
        debug!("  mutate {mut_count} nets ({})", m);
        mut_count
    }

    // ── Serialization ────────────────────────────────────────────────────────

    /// Build the full JSON envelope for a given best individual.
    /// Shared by `to_json` (run-level) and `record_improvement` (per-snapshot).
    fn build_envelope(
        &self,
        best: Option<&BestIndividual>,
    ) -> std::result::Result<serde_json::Value, EngineError> {
        let best_topology = match best {
            Some(b) => Some(
                b.topology
                    .to_json()
                    .map_err(|e| EngineError::Json(format!("best topology: {e}")))?,
            ),
            None => None,
        };
        let best_net_facts = match best {
            Some(b) => {
                let mut no = self.options.network;
                no.seed = b.topology.options.seed;
                let net = Network::build_with_options(&b.topology, &no)
                    .map_err(|e| EngineError::Json(format!("best net facts build: {e}")))?;
                Some(
                    net.to_json()
                        .map_err(|e| EngineError::Json(format!("best net facts: {e}")))?,
                )
            }
            None => None,
        };
        // Repeated topologies: sort by appearances desc, then mean desc
        let mut robust_entries: Vec<&RobustnessEntry> = self.robustness.values().collect();
        robust_entries.sort_by(|a, b| {
            b.count.cmp(&a.count)
                .then_with(|| b.mean.partial_cmp(&a.mean).unwrap_or(std::cmp::Ordering::Equal))
        });
        let robustness_val = serde_json::to_value(&robust_entries)
            .unwrap_or(serde_json::Value::Null);

        Ok(serde_json::json!({
            "run_id": self.run_id,
            "run_seed": self.seed,
            "data_path": self.data_path.display().to_string(),
            "generation": self.generation,
            "pop_size": self.pop.len(),
            "options": &self.options,
            "topology_options": serde_json::json!({
                "input_dim": self.options.topology_options.input_dim,
                "output_dim": self.options.topology_options.output_dim,
                "hidden_dim_pool": format!("{}..={}", self.options.hidden_dim_pool.start(), self.options.hidden_dim_pool.end()),
                "hidden_dim_stride": self.options.hidden_dim_stride,
                "min_hidden_num_nodes": self.options.topology_options.min_hidden_num_nodes,
                "max_hidden_num_nodes": self.options.topology_options.max_hidden_num_nodes,
                "min_hidden_inputs_per_node": self.options.topology_options.min_hidden_inputs_per_node,
                "max_hidden_inputs_per_node": self.options.topology_options.max_hidden_inputs_per_node,
                "min_hidden_outputs_per_node": self.options.topology_options.min_hidden_outputs_per_node,
                "max_hidden_outputs_per_node": self.options.topology_options.max_hidden_outputs_per_node,
            }),
            "best_fitness": best.map(|b| b.fitness),
            "best_loss": best.and_then(|b| b.loss),
            "best_topology": best_topology,
            "best_net_facts": best_net_facts,
            "history": if self.history.is_empty() { serde_json::Value::Null } else { serde_json::to_value(&self.history).unwrap_or(serde_json::Value::Null) },
            "robustness": robustness_val,
        }))
    }

    pub fn to_json(&self) -> Result<String> {
        let spec = self.build_envelope(self.best.as_ref())
            .map_err(|e| flodl::tensor::TensorError::new(&e.to_string()))?;
        serde_json::to_string_pretty(&spec)
            .map_err(|e| EngineError::Json(format!("to_json: {e}")).into())
    }
}

/// Deterministic child-seed derivation: multiply by golden ratio for spread.
pub(crate) fn derive_seed(base: u64, i: usize) -> u64 {
    base.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flodl::nn::loss::mse_loss;
    use flodl::{DType, Variable};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn temp_data_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gras_engine_test_{}", fastrand::u64(..)));
        let ds = crate::utils::synthetic::synthetic_sine(64, 42, Device::CPU).unwrap();
        crate::utils::data::save_dataset(&dir, &ds).unwrap();
        dir
    }

    fn test_options() -> EngineOptions {
        EngineOptions {
            pop_size: Some(3),
            num_generations: Some(2),
            dedup_pop_and_fill: false,
            topology_options: TopologyOptions {
                hidden_dim: 4,
                ..Default::default()
            },
            hidden_dim_pool: 4..=4,
            selection: Some(SelectionMethod::Tournament { tournament_size: 2 }),
            crossover: Some(CrossoverMethod::OnePoint { action_prob: 0.5 }),
            mutation: crate::mutation::MutationMethod::Activation { prob: 0.1 },
            results_dir: std::env::temp_dir()
                .join(format!("gras_engine_res_{}", fastrand::u64(..))),
            training: crate::trainer::TrainingConfig {
                num_epochs: 1,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_engine_runs_and_checkpoints() {
        let data_dir = temp_data_dir();
        let mut engine = Engine::new(
            test_options(),
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        engine.run().unwrap();
        let imp_dir = engine.run_dir.join("improvements");
        assert!(imp_dir.exists());
        let mut json_files: Vec<_> = std::fs::read_dir(&imp_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|f| f.ends_with(".json"))
            .collect();
        json_files.sort();
        assert!(!json_files.is_empty());
        // One JSON per gen
        let num_gens = engine.options.num_generations.unwrap();
        assert_eq!(json_files.len(), num_gens);
        // Load the last generation's data
        let last_gen = num_gens - 1;
        let gen_file = format!("gen_{:02}.json", last_gen);
        let latest_json =
            std::fs::read_to_string(imp_dir.join(&gen_file)).unwrap();
        let v: serde_json::Value = serde_json::from_str(&latest_json).unwrap();
        // Verify gen JSON structure
        assert_eq!(v["generation"].as_u64().unwrap() as usize, last_gen);
        let individuals = v["individuals"].as_array().unwrap();
        assert_eq!(individuals.len(), engine.options.pop_size.unwrap());
        // Every individual should have valid topology
        for ind in individuals {
            let topo = Topology::from_json(ind["topology"].as_str().unwrap()).unwrap();
            assert_eq!(topo.validate(), Ok(()));
            assert!(ind["fitness"].is_f64());
            assert!(ind["params"].is_u64());
        }
        assert!(engine.run_dir.join("engine.json").exists());
        let fitness = engine.best.as_ref().expect("best must exist").fitness;
        assert!(fitness.is_finite());
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&engine.options.results_dir);
    }

    #[test]
    fn test_engine_to_json_replicates_experiment() {
        let data_dir = temp_data_dir();
        let mut engine = Engine::new(
            test_options(),
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        engine.run().unwrap();
        let json = engine.to_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["run_id"], engine.run_id);
        assert_eq!(v["pop_size"], 3);
        assert!(v["best_fitness"].is_number());
        assert_eq!(v["run_seed"], engine.seed);
        let best_topo = Topology::from_json(v["best_topology"].as_str().unwrap()).unwrap();
        assert_eq!(best_topo.validate(), Ok(()));
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&engine.options.results_dir);
    }

    #[test]
    fn test_engine_custom_fitness_invoked_every_individual() {
        let data_dir = temp_data_dir();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let fitness = Fitness::from_loss(
            move |pred, y| {
                calls2.fetch_add(1, Ordering::SeqCst);
                mse_loss(pred, y)
            },
            Direction::Minimize,
            "mse",
        );
        let opts = test_options();
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        engine.run().unwrap();
        // sample_batches_from_indices clips to min(num_batches, pool_size / batch_size).max(1)
        let n_samples = 64usize; // synthetic_sine default
        let (_train_ratio, eval_ratio) = opts.train_test_split;
        let eval_pool = (n_samples as f32 * eval_ratio).round() as usize;
        let train_pool = n_samples - eval_pool;
        let actual_train = (opts.train_num_batches.min(train_pool / opts.train_batch_size)).max(1);
        let actual_eval = (opts.test_num_batches.min(eval_pool / opts.test_batch_size)).max(1);
        // Train: fitness called once per batch per epoch.
        // Eval: fitness called twice per batch (score + train_metric).
        let expected_per_individual = actual_train * opts.training.num_epochs + actual_eval * 2;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            opts.pop_size.unwrap() * opts.num_generations.unwrap() * expected_per_individual,
        );
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_auto_detects_input_dim() {
        let data_dir = temp_data_dir();
        let opts = EngineOptions {
            topology_options: TopologyOptions {
                input_dim: 999,
                ..Default::default()
            },
            ..test_options()
        };
        let engine = Engine::new(
            opts,
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        assert_eq!(engine.options.topology_options.input_dim, 1);
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[test]
    fn test_engine_batched_evaluation() {
        let data_dir = temp_data_dir();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let fitness = Fitness::from_loss(
            move |pred, y| {
                calls2.fetch_add(1, Ordering::SeqCst);
                mse_loss(pred, y)
            },
            Direction::Minimize,
            "mse",
        );
        let opts = EngineOptions {
            pop_size: Some(3),
            num_generations: Some(2),
            train_num_batches: 3,
            test_num_batches: 3,
            train_batch_size: 8,
            test_batch_size: 8,
            training: crate::trainer::TrainingConfig {
                num_batches_train: 3,
                num_batches_eval: 3,
                batch_size_train: 8,
                batch_size_eval: 8,
                ..crate::trainer::TrainingConfig::default()
            },
            ..test_options()
        };
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        engine.run().unwrap();
        // Account for train_test_split: pool_size = n_samples * ratio
        let n_samples = 64usize;
        let (_train_ratio, eval_ratio) = opts.train_test_split;
        let eval_pool = (n_samples as f32 * eval_ratio).round() as usize;
        let train_pool = n_samples - eval_pool;
        let actual_train = (opts.train_num_batches.min(train_pool / opts.train_batch_size)).max(1);
        let actual_eval = (opts.test_num_batches.min(eval_pool / opts.test_batch_size)).max(1);
        let eval_multiplier = 2;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            opts.pop_size.unwrap()
                * opts.num_generations.unwrap()
                * (actual_train * opts.training.num_epochs + actual_eval * eval_multiplier),
        );
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_rejects_bad_budget() {
        let data_dir = temp_data_dir();
        let bad = EngineOptions {
            train_num_batches: 2,
            test_num_batches: 2,
            train_batch_size: 0,
            test_batch_size: 0,
            ..test_options()
        };
        assert!(
            Engine::new(
                bad,
                &data_dir,
                Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse")
            )
            .is_err()
        );
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[test]
    fn test_engine_maximize_direction() {
        let data_dir = temp_data_dir();
        let make_scorer = |dir: Direction| {
            Fitness::from_loss(
                move |pred, _target| {
                    let vec = pred.data().to_f32_vec().unwrap();
                    let mean = vec.iter().sum::<f32>() / vec.len() as f32;
                    let t = flodl::Tensor::from_f32(&[mean], &[1], Device::CPU).unwrap();
                    Ok(Variable::new(t, false))
                },
                dir,
                "custom",
            )
        };
        let opts = EngineOptions {
            num_generations: Some(1),
            num_threads: 2,
            hidden_dim_pool: 4..=8,
            ..test_options()
        };
        let mut eng =
            Engine::new(opts.clone(), &data_dir, make_scorer(Direction::Maximize)).unwrap();
        eng.run().unwrap();
        let max_best = eng.best.as_ref().unwrap().fitness;
        let mut eng =
            Engine::new(opts.clone(), &data_dir, make_scorer(Direction::Minimize)).unwrap();
        eng.run().unwrap();
        let min_best = eng.best.as_ref().unwrap().fitness;
        assert!(max_best.is_finite());
        assert!(min_best.is_finite());
        assert_ne!(max_best, min_best);
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_builder_chains_and_validates() {
        let opts = EngineOptions::builder()
            .set_pop_size(15)
            .set_num_generations(3)
            .set_seed(Some(42))
            .set_hidden_dim(16)
            .set_hidden_dim_pool(8..=32)
            .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
            .set_activation_pool(vec![Activation::ReLU, Activation::GeLU])
            .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
            .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
            .set_mutation(crate::mutation::MutationMethod::Activation { prob: 0.1 })
            .set_train_num_batches(4)
            .set_test_num_batches(4)
            .set_train_batch_size(32)
            .set_test_batch_size(32)
            .set_num_threads(2)
            .set_dtype(DType::Float32)
            .build()
            .unwrap();
        assert_eq!(opts.pop_size, Some(15));
        assert_eq!(opts.num_generations, Some(3));
        assert_eq!(opts.seed, Some(42));
        assert_eq!(opts.hidden_dim_pool, 8..=32);
        assert_eq!(opts.combine_op_pool, vec![CombineOp::Add, CombineOp::Mean]);
        assert_eq!(opts.train_num_batches, 4);
        assert_eq!(opts.test_num_batches, 4);
        assert_eq!(opts.train_batch_size, 32);
        assert_eq!(opts.test_batch_size, 32);
        assert_eq!(opts.network.dtype, DType::Float32);
        assert!(EngineOptions::builder().set_pop_size(0).build().is_err());
        assert!(
            EngineOptions::builder()
                .set_train_num_batches(2)
                .set_train_batch_size(0)
                .build()
                .is_err()
        );
        // set_mutation() is required — omitting it should error at build time.
        assert!(
            EngineOptions::builder()
                .set_pop_size(4)
                .set_num_generations(1)
                .set_hidden_dim_pool(4..=4)
                .set_train_num_batches(2)
                .set_test_num_batches(2)
                .set_train_batch_size(8)
                .set_test_batch_size(8)
                .set_num_epochs(1)
                .build()
                .is_err()
        );
        let opts = EngineOptions::builder()
            .set_pop_size(4)
            .set_num_generations(1)
            .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
            .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
            .set_mutation(crate::mutation::MutationMethod::Activation { prob: 0.1 })
            .set_combine_op_pool(vec![])
            .build()
            .unwrap();
        assert_eq!(opts.combine_op_pool.len(), 4);
        let opts = EngineOptions::builder()
            .set_pop_size(4)
            .set_num_generations(1)
            .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
            .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
            .set_mutation(crate::mutation::MutationMethod::Activation { prob: 0.1 })
            .set_activation_pool(vec![])
            .build()
            .unwrap();
        assert_eq!(opts.activation_pool.len(), 16);
    }

    #[test]
    fn test_engine_builder_one_shot() {
        let data_dir = temp_data_dir();
        let opts = EngineOptions::builder()
            .set_pop_size(4)
            .set_num_generations(1)
            .set_seed(Some(7))
            .set_hidden_dim_pool(4..=4)
            .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
            .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
            .set_mutation(crate::mutation::MutationMethod::Activation { prob: 0.1 })
            .build()
            .unwrap();
        let mut engine = Engine::new(
            opts,
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        engine.run().unwrap();
        assert!(engine.best.is_some());
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&engine.options.results_dir);
    }

    #[test]
    fn test_engine_gp_sampling_varies_and_reproduces() {
        let data_dir = temp_data_dir();
        let pool = vec![
            Activation::Identity,
            Activation::ReLU,
            Activation::GeLU,
            Activation::SELU,
        ];
        let make_opts = || EngineOptions {
            pop_size: Some(8),
            num_generations: Some(1),
            seed: Some(99),
            hidden_dim_pool: 4..=16,
            combine_op_pool: vec![CombineOp::Add, CombineOp::Mean],
            activation_pool: pool.clone(),
            results_dir: std::env::temp_dir().join(format!("gras_gp_res_{}", fastrand::u64(..))),
            ..test_options()
        };
        let a = Engine::new(
            make_opts(),
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        let b = Engine::new(
            make_opts(),
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        let mut dims: Vec<usize> = Vec::new();
        for g in &a.pop {
            for n in &g.nodes {
                if let Some(d) = n.hidden_dim {
                    if !dims.contains(&d) {
                        dims.push(d);
                    }
                }
            }
        }
        dims.sort_unstable();
        assert!(dims.len() > 1, "hidden dims must vary: {dims:?}");
        for (ga, gb) in a.pop.iter().zip(b.pop.iter()) {
            assert_eq!(ga.options.hidden_dim, gb.options.hidden_dim);
            assert_eq!(crate::spec::Spec::from(ga), crate::spec::Spec::from(gb));
        }
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&a.options.results_dir);
        let _ = std::fs::remove_dir_all(&b.options.results_dir);
    }

    #[test]
    fn test_engine_new_leaves_no_folder() {
        let data_dir = temp_data_dir();
        let opts = test_options();
        let engine = Engine::new(
            opts.clone(),
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        assert!(!engine.run_dir.exists());
        let mut engine = engine;
        engine.run().unwrap();
        assert!(engine.run_dir.exists());
        assert!(engine.run_dir.join("engine.json").exists());
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_random_seed_recorded() {
        let data_dir = temp_data_dir();
        let opts = EngineOptions {
            seed: None,
            num_threads: 4,
            ..test_options()
        };
        let mut engine = Engine::new(
            opts.clone(),
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        engine.run().unwrap();
        let v: serde_json::Value = serde_json::from_str(&engine.to_json().unwrap()).unwrap();
        assert_eq!(v["run_seed"], engine.seed);
        let other = Engine::new(
            opts.clone(),
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        assert_ne!(other.seed, engine.seed);
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_seeded_run_is_reproducible() {
        let data_dir = temp_data_dir();
        let make = || EngineOptions {
            seed: Some(123),
            num_threads: 3,
            dropout_prob: 0.0,
            train_fixed_coefs: true,
            ..test_options()
        };
        let mut a = Engine::new(
            make(),
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        let mut b = Engine::new(
            make(),
            &data_dir,
            Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse"),
        )
        .unwrap();
        a.run().unwrap();
        b.run().unwrap();
        let ba = a.best.as_ref().unwrap();
        let bb = b.best.as_ref().unwrap();
        assert_eq!(ba.fitness, bb.fitness);
        assert_eq!(
            crate::spec::Spec::from(&ba.topology),
            crate::spec::Spec::from(&bb.topology)
        );
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&a.options.results_dir);
        let _ = std::fs::remove_dir_all(&b.options.results_dir);
    }

    #[test]
    fn test_fitness_custom_sees_pred_and_target() {
        let data_dir = temp_data_dir();
        let fitness =
            Fitness::from_loss(|pred, y| flodl::l1_loss(pred, y), Direction::Minimize, "l1");
        let opts = EngineOptions {
            num_generations: Some(1),
            ..test_options()
        };
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        engine.run().unwrap();
        assert!(engine.best.is_some());
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_from_loss_with_diff() {
        let data_dir = temp_data_dir();
        // Train on MSE, evolve on negative MSE (maximize) — different directions
        let fitness = Fitness::from_loss_with_diff(
            |pred, y| {
                let diff = pred.data().sub(&y.data())?;
                let sq = diff.mul(&diff)?;
                Ok(sq.mean()?.item()? as f32)
            },
            Direction::Minimize,
            "mse_score",
            |pred, y| flodl::mse_loss(pred, y),
            Direction::Minimize,
            "mse_train",
        );
        assert!(!fitness.train_metric_is_fitness());
        assert_eq!(fitness.fitness_label(), "mse_score");
        assert_eq!(fitness.train_metric_label(), "mse_train");
        let opts = EngineOptions {
            num_generations: Some(1),
            ..test_options()
        };
        let mut engine = Engine::new(opts, &data_dir, fitness).unwrap();
        engine.run().unwrap();
        assert!(engine.best.is_some());
        let best = engine.best.as_ref().unwrap();
        assert!(best.fitness.is_finite());
        assert!(best.loss.is_some());
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&engine.options.results_dir);
    }

    #[test]
    fn test_hidden_dim_stride() {
        let mut rng = fastrand::Rng::with_seed(42);
        // stride=16, pool 32..=64 → {32, 48, 64}
        let pool = 32..=64usize;
        let stride = 16;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let v = Engine::sample_hidden_dim(&pool, stride, &mut rng);
            seen.insert(v);
            assert!(v >= 32 && v <= 64, "out of range: {v}");
            assert!((v - 32) % 16 == 0, "not on stride: {v}");
        }
        assert_eq!(seen, std::collections::HashSet::from([32, 48, 64]));

        // stride=1, pool 4..=8 → {4,5,6,7,8}
        let pool2 = 4..=8usize;
        let mut seen2 = std::collections::HashSet::new();
        for _ in 0..200 {
            seen2.insert(Engine::sample_hidden_dim(&pool2, 1, &mut rng));
        }
        assert_eq!(seen2, std::collections::HashSet::from([4, 5, 6, 7, 8]));

        // stride=16, pool 50..=100 → {50, 66, 82, 98}
        let pool3 = 50..=100usize;
        let mut seen3 = std::collections::HashSet::new();
        for _ in 0..200 {
            let v = Engine::sample_hidden_dim(&pool3, 16, &mut rng);
            seen3.insert(v);
            assert!((v - 50) % 16 == 0, "not on stride: {v}");
        }
        assert_eq!(seen3, std::collections::HashSet::from([50, 66, 82, 98]));
    }
}
