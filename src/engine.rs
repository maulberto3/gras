//! The engine -- NAS loop over random topologies: seed, score, evolve.
//!
//! Data contract: flodl-native tensors loaded once at Engine::new,
//! reused per individual per generation. Replicate via
//! Engine::to_json -> Topology::from_json + Network::build.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use flodl::Device;
use flodl::tensor::Result;
use log::debug;
use rayon::prelude::*;
use serde::Serialize;

pub use crate::fitness::{Direction, Fitness, FitnessLabel};
use crate::network::{Network, NetworkOptions};
use crate::node::{Activation, NodeKind};
use crate::selection::SelectionMethod;
use crate::topology::{CombineOp, Topology, TopologyOptions};
use crate::utils::data::Dataset;
use crate::utils::error::EngineError;

// ── Crossover operators ───────────────────────────────────────────────────

/// Crossover strategy for combining two parent topologies.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum CrossoverKind {
    TwoPoint,
}

impl std::fmt::Display for CrossoverKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrossoverKind::TwoPoint => write!(f, "two_point"),
        }
    }
}

// ── EngineOptions -- the experiment configuration ──────────────────────────

/// Run configuration -- serialized to engine.json for reproducibility.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EngineOptions {
    pub pop_size: usize,
    pub num_generations: usize,
    /// Population base seed. None -> random per run, recorded as run_seed.
    pub seed: Option<u64>,
    /// Shared topology template -- each individual clones and overrides seed.
    pub topology_options: TopologyOptions,
    /// Hidden-dim range sampled per individual. Empty -> fill with defaults.
    pub hidden_dim_pool: RangeInclusive<usize>,
    /// Combine-op pool sampled per individual. Empty -> all built-in ops.
    pub combine_op_pool: Vec<CombineOp>,
    /// Activation pool -- per-node at creation, swap source for mutation.
    pub activation_pool: Vec<Activation>,
    /// Standardize-op pool -- per-node normalization. Empty -> all built-in.
    pub standardize_op_pool: Vec<crate::node::StandardizeOp>,
    /// Evaluation budget: non-overlapping batches per generation.
    pub num_batches: usize,
    /// Rows per batch.
    pub batch_size: usize,
    pub fitness_label: FitnessLabel,
    pub train_metric_label: FitnessLabel,
    /// Threads for parallel eval (0 = rayon default).
    pub num_threads: usize,
    pub results_dir: PathBuf,
    /// Per-individual activation mutation probability (0.0 = off).
    pub mutate_activ_prob: f32,
    /// Probability of toggling recurrent on eligible hidden nodes (num_in == num_out).
    pub mutate_recurrent_prob: f32,
    /// Probability of mutating hidden_dim per node.
    pub mutate_dim_prob: f32,
    /// Probability of mutating combine_op per node.
    pub mutate_combine_prob: f32,
    /// Probability of mutating standardize_op per node.
    pub mutate_standardize_prob: f32,
    /// Training config applied to every individual before scoring.
    pub training: crate::trainer::TrainingConfig,
    /// Selection strategy for the next generation.
    pub selection: SelectionMethod,
    /// Crossover strategies. Empty -> [TwoPoint].
    pub crossover_pool: Vec<CrossoverKind>,
    /// Network execution options (device, dtype, seed).
    pub network: NetworkOptions,
    /// Dropout probability for hidden nodes (0.0 = no dropout).
    pub dropout_prob: f32,
    /// Enable recurrent hidden nodes (feed output back as additional input).
    /// Default: false.
    pub recurrent: bool,
    /// Probability of toggling recurrent on eligible hidden nodes.
    pub recurrent_prob: f32,
    /// Detach gradients between generations (stop BPTT across gen boundary).
    /// Default: false.
    pub detach: bool,
    /// Sample batches proportional to target class frequency (categorical data).
    /// Default: false (uniform random sampling).
    pub proportional_batches: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            pop_size: 10,
            num_generations: 5,
            seed: None,
            topology_options: TopologyOptions::default(),
            hidden_dim_pool: 4..=8,
            combine_op_pool: vec![],
            activation_pool: vec![],
            standardize_op_pool: vec![],
            num_batches: 16,
            batch_size: 128,
            fitness_label: FitnessLabel::default(),
            train_metric_label: FitnessLabel::default(),
            num_threads: 1,
            results_dir: PathBuf::from("results"),
            mutate_activ_prob: 0.1,
            mutate_recurrent_prob: 0.1,
            mutate_dim_prob: 0.1,
            mutate_combine_prob: 0.1,
            mutate_standardize_prob: 0.1,

            training: crate::trainer::TrainingConfig::default(),
            selection: SelectionMethod::default(),
            crossover_pool: vec![],
            network: NetworkOptions::default(),
            dropout_prob: 0.05,
            recurrent: false,
            recurrent_prob: 0.3f32,
            detach: false,
            proportional_batches: false,
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

    /// The shared topology template, serialized into engine.json.
    fn topology_template(&self) -> TopologyOptions {
        self.topology_options
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
    /// Validate and return the accumulated options. Fills empty pools with all built-ins.
    pub fn build(mut self) -> Result<EngineOptions> {
        let o = &mut self.inner;
        if o.pop_size == 0 {
            return Err(EngineError::InvalidOptions("pop_size must be > 0".into()).into());
        }
        if o.num_batches > 0 && o.batch_size == 0 {
            return Err(EngineError::InvalidOptions(
                "num_batches > 0 requires batch_size > 0".to_string(),
            )
            .into());
        }
        if o.hidden_dim_pool.is_empty() {
            return Err(EngineError::InvalidOptions(
                "hidden_dim_pool must be a non-empty range (start <= end)".to_string(),
            )
            .into());
        }
        if o.combine_op_pool.is_empty() {
            o.combine_op_pool = crate::pools::all_combine_ops();
        }
        if o.activation_pool.is_empty() {
            o.activation_pool = crate::pools::all_activations();
        }
        if o.standardize_op_pool.is_empty() {
            o.standardize_op_pool = crate::pools::all_standardize_ops();
        }
        if o.training.num_epochs == 0 {
            return Err(EngineError::InvalidOptions(
                "num_epochs must be > 0 (set via set_num_epochs)".to_string(),
            )
            .into());
        }
        Ok(self.inner)
    }

    // Engine knobs
    pub fn set_pop_size(mut self, n: usize) -> Self {
        self.inner.pop_size = n;
        self
    }
    pub fn set_num_generations(mut self, n: usize) -> Self {
        self.inner.num_generations = n;
        self
    }
    pub fn set_seed(mut self, s: Option<u64>) -> Self {
        self.inner.seed = s;
        self
    }
    pub fn set_num_batches(mut self, n: usize) -> Self {
        self.inner.num_batches = n;
        self.inner.training.num_batches = n;
        self
    }
    pub fn set_batch_size(mut self, n: usize) -> Self {
        self.inner.batch_size = n;
        self.inner.training.batch_size = n;
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
    pub fn set_mutate_activ_prob(mut self, p: f32) -> Self {
        self.inner.mutate_activ_prob = p.clamp(0.0, 1.0);
        self
    }
    pub fn set_selection(mut self, method: SelectionMethod) -> Self {
        self.inner.selection = method;
        self
    }

    // Training knobs
    pub fn set_num_epochs(mut self, n: usize) -> Self {
        self.inner.training.num_epochs = n;
        self
    }
    pub fn set_learning_rate(mut self, lr: f64) -> Self {
        self.inner.training.learning_rate = lr;
        self
    }
    pub fn set_optimizer(mut self, kind: crate::trainer::OptimizerKind) -> Self {
        self.inner.training.optimizer = kind;
        self
    }
    pub fn set_grad_clip(mut self, max_norm: f64) -> Self {
        self.inner.training.grad_clip = max_norm;
        self
    }
    pub fn set_dropout_prob(mut self, p: f32) -> Self {
        self.inner.dropout_prob = p.clamp(0.0, 1.0);
        self
    }
    pub fn set_recurrent(mut self, on: bool) -> Self {
        self.inner.recurrent = on;
        self
    }
    pub fn set_recurrent_prob(mut self, p: f32) -> Self {
        self.inner.recurrent_prob = p.clamp(0.0, 1.0);
        self
    }
    pub fn set_mutate_recurrent_prob(mut self, p: f32) -> Self {
        self.inner.mutate_recurrent_prob = p.clamp(0.0, 1.0);
        self
    }
    pub fn set_mutate_dim_prob(mut self, p: f32) -> Self {
        self.inner.mutate_dim_prob = p.clamp(0.0, 1.0);
        self
    }
    pub fn set_mutate_combine_prob(mut self, p: f32) -> Self {
        self.inner.mutate_combine_prob = p.clamp(0.0, 1.0);
        self
    }
    pub fn set_mutate_standardize_prob(mut self, p: f32) -> Self {
        self.inner.mutate_standardize_prob = p.clamp(0.0, 1.0);
        self
    }
    pub fn set_crossover_pool(mut self, pool: Vec<CrossoverKind>) -> Self {
        self.inner.crossover_pool = pool;
        self
    }
    pub fn set_detach(mut self, on: bool) -> Self {
        self.inner.detach = on;
        self
    }
    pub fn set_proportional_batches(mut self, on: bool) -> Self {
        self.inner.proportional_batches = on;
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
    pub(crate) data_path: PathBuf,
    pub generation: usize,
    pub best: Option<BestIndividual>,
    pub(crate) improvements: usize,
    last_improvement_hash: Option<u64>,
    last_improvement_prefix: Option<String>,
    scores: Vec<f32>,
    eval_losses: Vec<Option<f32>>,
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

        // Step 4: Create population
        let pop = Self::create_population(&options, seed)?;

        // Step 5: Log initialization
        Self::log_initialization(&options, &dataset, &pop, seed, &fitness);

        // Step 6: Assemble engine (thread pool, run dir, struct)
        Self::assemble_engine(options, seed, dataset, pop, fitness, data_path)
    }

    /// Step 1: Validate options and fill empty pools with all built-ins.
    fn validate_and_fill_options(options: &mut EngineOptions) -> Result<()> {
        if options.pop_size == 0 {
            return Err(EngineError::InvalidOptions("pop_size must be > 0".into()).into());
        }
        if options.num_batches > 0 && options.batch_size == 0 {
            return Err(EngineError::InvalidOptions(
                "num_batches > 0 requires batch_size > 0".to_string(),
            )
            .into());
        }
        if options.hidden_dim_pool.is_empty() {
            return Err(EngineError::InvalidOptions(
                "hidden_dim_pool must be a non-empty range".to_string(),
            )
            .into());
        }
        if options.combine_op_pool.is_empty() {
            options.combine_op_pool = crate::pools::all_combine_ops();
        }
        if options.activation_pool.is_empty() {
            options.activation_pool = crate::pools::all_activations();
        }
        if options.standardize_op_pool.is_empty() {
            options.standardize_op_pool = crate::pools::all_standardize_ops();
        }
        if options.training.num_epochs == 0 {
            return Err(EngineError::InvalidOptions("num_epochs must be > 0".to_string()).into());
        }
        if options.num_batches == 0 {
            options.num_batches = 16;
        }
        if options.crossover_pool.is_empty() {
            options.crossover_pool = vec![CrossoverKind::TwoPoint];
        }
        // Propagate engine-level flags into training config.
        options.training.proportional_batches = options.proportional_batches;
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

    /// Step 3: Load dataset and bind input_dim/output_dim to options.
    fn load_data(options: &mut EngineOptions, data_path: &Path) -> Result<Dataset> {
        let dataset =
            crate::utils::data::load_dataset(data_path)?.to_dtype(options.network.dtype)?;
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

    /// Step 4: Create a population of random topologies, seeded deterministically.
    fn create_population(options: &EngineOptions, seed: u64) -> Result<Vec<Topology>> {
        let mut pop = Vec::with_capacity(options.pop_size);
        for i in 0..options.pop_size {
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
                    node.hidden_dim = Some(rng.usize(options.hidden_dim_pool.clone()));
                    node.activation = options.activation_pool[rng.usize(0..pool_len_a)];
                    node.combine_op = Some(options.combine_op_pool[rng.usize(0..pool_len_c)]);
                    node.standardize = Some(options.standardize_op_pool[rng.usize(0..pool_len_s)]);
                }
            }
            graph.refresh_labels();
            graph.finalize();

            // Randomly toggle recurrent on eligible nodes (num_inputs == num_outs).
            if options.recurrent {
                for node in &mut graph.nodes {
                    if node.recurrent && rng.f64() > options.recurrent_prob as f64 {
                        node.recurrent = false;
                    }
                }
            } else {
                for node in &mut graph.nodes {
                    node.recurrent = false;
                }
            }

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

    /// Step 5: Log all resolved options, dataset, and population.
    fn log_initialization(
        options: &EngineOptions,
        dataset: &Dataset,
        pop: &[Topology],
        seed: u64,
        fitness: &Fitness,
    ) {
        crate::utils::log_utils::log_initialization(options, dataset, pop, seed, fitness);
    }

    /// Step 6: Build thread pool, generate run id, assemble Engine struct.
    fn assemble_engine(
        options: EngineOptions,
        seed: u64,
        dataset: Dataset,
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
            data_path: data_path.to_path_buf(),
            generation: 0,
            best: None,
            improvements: 0,
            last_improvement_hash: None,
            last_improvement_prefix: None,
            scores: Vec::new(),
            eval_losses: Vec::new(),
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
        for g in 0..self.options.num_generations {
            debug!("== gen {:02}/{:02} ==", g, self.options.num_generations);
            let _improved = self.evaluate_population();
            self.log_generation_summary();
            self.next_generation();
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
            self.improvements,
            &self.best,
            &self.fitness,
            &self.options,
            &self.run_dir,
        )
        .map_err(|e| EngineError::Json(format!("run summary log: {e}")))?;
        Ok(())
    }

    // ── Evaluation ──────────────────────────────────────────────────────────

    /// Score every individual. Returns whether the overall best improved.
    fn evaluate_population(&mut self) -> Result<bool> {
        // Step 1: Parallel eval -- build, train, score each individual
        let results = self.eval_all_individuals()?;

        // Step 2: Store scores and losses
        self.update_scores(results);

        // Step 3: Track improvements and record new bests
        self.track_improvements()
    }

    /// Step 1: Parallel rayon loop -- build, train, score each individual.
    fn eval_all_individuals(&self) -> Result<Vec<(f32, Option<f32>)>> {
        let direction = self.fitness.direction();
        let net_opts = self.options.network;
        let train_cfg = &self.options.training;
        let fitness = &self.fitness;
        let dataset = &self.dataset;
        let pop_size = self.pop.len();
        let batch_seed = derive_seed(self.seed, self.generation * 3);
        let done = std::sync::atomic::AtomicUsize::new(0);
        let best_bits =
            std::sync::atomic::AtomicU32::new(f32::to_bits(if direction == Direction::Minimize {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            }));
        let generation = self.generation;
        let progress_lock = std::sync::Mutex::new(());

        self.pool.install(|| {
            self.pop
                .par_iter()
                .map(|graph| {
                    let mut no = net_opts;
                    no.seed = graph.options.seed;
                    no.dropout_prob = self.options.dropout_prob;
                    let mut net = Network::build_with_options(graph, &no)?;
                    let result = crate::trainer::train_network(
                        &mut net, train_cfg, fitness, dataset, batch_seed,
                    )?;
                    let score = result.score;
                    let loss = result.eval_loss;

                    best_bits
                        .fetch_update(
                            std::sync::atomic::Ordering::Relaxed,
                            std::sync::atomic::Ordering::Relaxed,
                            |bits| {
                                if direction.is_better(score, f32::from_bits(bits)) {
                                    Some(score.to_bits())
                                } else {
                                    None
                                }
                            },
                        )
                        .ok();
                    let cur_best =
                        f32::from_bits(best_bits.load(std::sync::atomic::Ordering::Relaxed));
                    let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    {
                        let _guard = progress_lock.lock().unwrap();
                        use std::io::Write;
                        print!(
                            "\r  gen {:02} net {:>2}/{pop_size}  best {cur_best:.4}\x1b[K",
                            generation, n
                        );
                        std::io::stdout().flush().unwrap();
                    }

                    Ok((score, loss))
                })
                .collect::<Result<Vec<_>>>()
        })
    }

    /// Step 2: Store scores and eval_losses from parallel results.
    fn update_scores(&mut self, results: Vec<(f32, Option<f32>)>) {
        println!();
        self.scores = results.iter().map(|&(s, _)| s).collect();
        self.eval_losses = results.iter().map(|&(_, l)| l).collect();
    }

    /// Step 3: Walk scores, record new improvements immediately.
    fn track_improvements(&mut self) -> Result<bool> {
        let direction = self.fitness.direction();
        // Collect indices that beat current best (all, if first gen).
        let mut improved: Vec<usize> = self
            .scores
            .iter()
            .enumerate()
            .filter(|&(_i, &score)| {
                self.best
                    .as_ref()
                    .map(|b| direction.is_better(score, b.fitness))
                    .unwrap_or(true)
            })
            .map(|(i, _)| i)
            .collect();
        if improved.is_empty() {
            return Ok(false);
        }
        // Sort best-first so self.best always tracks the true best.
        improved.sort_by(|&a, &b| direction.cmp(self.scores[a], self.scores[b]));

        let mut any_improved = false;
        for &i in &improved {
            let score = self.scores[i];
            let loss = self.eval_losses.get(i).copied().flatten();
            let topo = self.pop[i].clone();
            // Record to disk + update best (no console logging — progress line shows running best).
            self.best = Some(BestIndividual {
                fitness: score,
                loss,
                pop_index: i,
                topology: topo,
            });
            self.record_improvement()?;
            any_improved = true;
        }
        Ok(any_improved)
    }

    // ── Improvement tracking ────────────────────────────────────────────────

    fn record_improvement(&mut self) -> std::result::Result<(), EngineError> {
        let Some(b) = &self.best else { return Ok(()) };
        let dir = self.run_dir.join("improvements");
        fs::create_dir_all(&dir).map_err(|source| EngineError::Io {
            path: dir.display().to_string(),
            source,
        })?;
        let json_str = self.build_envelope(Some(b))
            .and_then(|v| serde_json::to_string_pretty(&v)
                .map_err(|e| EngineError::Json(format!("improvement json: {e}"))))?;

        // Hash topology for fast comparison.
        let mut hasher = DefaultHasher::new();
        json_str.hash(&mut hasher);
        let hash = hasher.finish();

        let same_topo = self
            .last_improvement_hash
            .map_or(false, |prev| prev == hash);

        if same_topo {
            // Delete old files for this topology, then save with updated fitness.
            if let Some(prefix) = &self.last_improvement_prefix {
                let _ = fs::remove_file(dir.join(format!("{prefix}.json")));
                let _ = fs::remove_file(dir.join(format!("{prefix}.md")));
            }
        } else {
            self.improvements += 1;
        }

        let prefix = format!(
            "{:04}_gen{:02}_fitness{:.4}",
            self.improvements - 1,
            self.generation,
            b.fitness
        );
        let path = dir.join(format!("{prefix}.json"));
        fs::write(&path, &json_str).map_err(|source| EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let md_path = dir.join(format!("{prefix}.md"));
        let net = Network::build(&b.topology, Device::CPU).ok();
        let md = crate::utils::ascii_utils::topology_markdown(
            &b.topology,
            Some(b.fitness),
            net.as_ref(),
        );
        fs::write(&md_path, md).map_err(|source| EngineError::Io {
            path: md_path.display().to_string(),
            source,
        })?;

        self.last_improvement_hash = Some(hash);
        self.last_improvement_prefix = Some(prefix);
        debug!(
            "record_improvement -- saved #{:04} gen={} fitness={:.4}{}",
            self.improvements - 1,
            self.generation,
            b.fitness,
            if same_topo {
                " (same arch, replaced)"
            } else {
                ""
            },
        );
        Ok(())
    }

    // ── Logging ─────────────────────────────────────────────────────────────

    fn log_generation_summary(&self) {
        if let Some(best) = &self.best {
            log::info!("  gen {:02} best {:.4}", self.generation, best.fitness);
        }
    }

    // ── Genetics -- selection, crossover, mutation ────────────────────────────

    fn next_generation(&mut self) {
        debug!(
            "next_generation -- gen {} -> {}",
            self.generation,
            self.generation + 1
        );
        self.select();
        self.crossover();
        self.mutate();
        self.generation += 1;
    }

    /// Selection -- reorder pop/scores so fittest survive.
    pub fn select(&mut self) {
        if self.scores.is_empty() {
            return;
        }
        let dir = self.fitness.direction();
        let mut rng = fastrand::Rng::with_seed(derive_seed(self.seed, self.generation * 3 + 1));
        let indices = self.options.selection.apply(&self.scores, dir, &mut rng);
        // In-place reorder: build new pop from selected indices.
        let new_pop: Vec<Topology> = indices.iter().map(|&i| self.pop[i].clone()).collect();
        let new_scores: Vec<f32> = indices.iter().map(|&i| self.scores[i]).collect();
        self.pop = new_pop;
        self.scores = new_scores;

        let mut counts = vec![0usize; self.pop.len()];
        for &i in &indices {
            counts[i] += 1;
        }
        let selected: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] > 0).collect();
        let highlights: Vec<String> = selected
            .iter()
            .map(|&i| format!("pop[{i}]x{}", counts[i]))
            .collect();
        debug!(
            "  selection [{}] survivors: {}",
            self.options.selection.label(),
            highlights.join(" ")
        );
    }

    pub fn crossover(&mut self) {
        // TODO: topology-level crossover
    }

    pub fn mutate(&mut self) {
        // TODO: topology-level mutation
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
        Ok(serde_json::json!({
            "run_id": self.run_id,
            "run_seed": self.seed,
            "data_path": self.data_path.display().to_string(),
            "generation": self.generation,
            "pop_size": self.pop.len(),
            "options": &self.options,
            "topology_options": self.options.topology_template(),
            "best_fitness": best.map(|b| b.fitness),
            "best_loss": best.and_then(|b| b.loss),
            "best_topology": best_topology,
            "best_net_facts": best_net_facts,
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
            pop_size: 3,
            num_generations: 2,
            topology_options: TopologyOptions {
                hidden_dim: 4,
                ..Default::default()
            },
            hidden_dim_pool: 4..=4,
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
        assert_eq!(json_files.len(), engine.improvements);
        let latest_json =
            std::fs::read_to_string(imp_dir.join(json_files.last().unwrap())).unwrap();
        let v: serde_json::Value = serde_json::from_str(&latest_json).unwrap();
        let latest = Topology::from_json(v["best_topology"].as_str().unwrap()).unwrap();
        let best_topo = engine.best.as_ref().unwrap().topology.clone();
        assert_eq!(
            crate::spec::Spec::from(&latest),
            crate::spec::Spec::from(&best_topo)
        );
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
        let actual_batches = 1usize;
        let expected_per_individual =
            opts.training.num_epochs * actual_batches + 2 * actual_batches;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            opts.pop_size * opts.num_generations * expected_per_individual,
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
            pop_size: 3,
            num_generations: 2,
            num_batches: 3,
            batch_size: 8,
            training: crate::trainer::TrainingConfig {
                num_batches: 3,
                batch_size: 8,
                ..crate::trainer::TrainingConfig::default()
            },
            ..test_options()
        };
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        engine.run().unwrap();
        let train_count = (opts.num_batches / 2).max(1);
        let eval_count = (opts.num_batches - train_count).max(1);
        let eval_multiplier = 2;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            opts.pop_size
                * opts.num_generations
                * (train_count * opts.training.num_epochs + eval_count * eval_multiplier),
        );
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_rejects_bad_budget() {
        let data_dir = temp_data_dir();
        let bad = EngineOptions {
            num_batches: 2,
            batch_size: 0,
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
            num_generations: 1,
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
            .set_num_batches(4)
            .set_batch_size(32)
            .set_num_threads(2)
            .set_dtype(DType::Float32)
            .build()
            .unwrap();
        assert_eq!(opts.pop_size, 15);
        assert_eq!(opts.num_generations, 3);
        assert_eq!(opts.seed, Some(42));
        assert_eq!(opts.hidden_dim_pool, 8..=32);
        assert_eq!(opts.combine_op_pool, vec![CombineOp::Add, CombineOp::Mean]);
        assert_eq!(opts.num_batches, 4);
        assert_eq!(opts.batch_size, 32);
        assert_eq!(opts.network.dtype, DType::Float32);
        assert!(EngineOptions::builder().set_pop_size(0).build().is_err());
        assert!(
            EngineOptions::builder()
                .set_num_batches(2)
                .set_batch_size(0)
                .build()
                .is_err()
        );
        let opts = EngineOptions::builder()
            .set_combine_op_pool(vec![])
            .build()
            .unwrap();
        assert_eq!(opts.combine_op_pool.len(), 4);
        let opts = EngineOptions::builder()
            .set_activation_pool(vec![])
            .build()
            .unwrap();
        assert_eq!(opts.activation_pool.len(), 14);
    }

    #[test]
    fn test_engine_builder_one_shot() {
        let data_dir = temp_data_dir();
        let opts = EngineOptions::builder()
            .set_pop_size(4)
            .set_num_generations(1)
            .set_seed(Some(7))
            .set_hidden_dim_pool(4..=4)
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
            pop_size: 8,
            num_generations: 1,
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
        assert_eq!(v["topology_options"]["seed"], engine.seed);
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
            num_generations: 1,
            ..test_options()
        };
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        engine.run().unwrap();
        assert!(engine.best.is_some());
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }
}
