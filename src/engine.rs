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
    // 
    pub pop_size: usize,
    // 
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


    /// Sample batches proportional to target class frequency (categorical data).
    /// Default: false (uniform random sampling).
    pub y_proportional_batches: bool,
    /// Deduplicate population by full topology comparison.
    /// Applied after create_population and after select each generation.
    /// Default: true.
    pub dedup_pop_and_fill: bool,
    /// Track per-generation metrics in engine.json history.
    /// Default: false.
    pub gens_history: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            // ── Required (must be set by user) ─────────────────────────
            pop_size: 10,            // ≥ 2 required (crossover)
            num_generations: 5,      // > 0 required
            mutation: MutationMethod::default(), // disabled, must call set_mutation()
            // ── Engine ────────────────────────────────────────────────
            seed: None,              // random if not set
            num_threads: 1,
            results_dir: PathBuf::from("results"),
            dedup_pop_and_fill: false,
            gens_history: false,
            // ── Topology / GP pools ───────────────────────────────────
            topology_options: TopologyOptions::default(),
            hidden_dim_pool: 4..=8,
            combine_op_pool: vec![],   // empty = all built-ins
            activation_pool: vec![],   // empty = all built-ins
            standardize_op_pool: vec![], // empty = all built-ins
            // ── Evaluation budget ─────────────────────────────────────
            num_batches: 4,
            batch_size: 32,
            // ── Training ──────────────────────────────────────────────
            training: crate::trainer::TrainingConfig::default(),
            fitness_label: FitnessLabel::default(),
            train_metric_label: FitnessLabel::default(),
            // ── Network ───────────────────────────────────────────────
            network: NetworkOptions::default(), // CPU, Float32
            dropout_prob: 0.05,
            // ── Genetics ──────────────────────────────────────────────
            selection: None,
            crossover: None,
            y_proportional_batches: false,
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
    /// Validate required fields and return the accumulated options.
    /// Fills empty pools with all built-ins.
    pub fn build(mut self) -> Result<EngineOptions> {
        let o = &mut self.inner;
        // ── Required fields ────────────────────────────────────────────
        if o.pop_size < 2 {
            return Err(EngineError::InvalidOptions(
                "pop_size must be >= 2 (required for crossover)".into(),
            )
            .into());
        }
        if o.num_generations == 0 {
            return Err(EngineError::InvalidOptions(
                "num_generations must be > 0".into(),
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
        // ── Conservative defaults for non-required fields ──────────────
        if o.num_batches > 0 && o.batch_size == 0 {
            o.batch_size = 32;
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


    pub fn set_y_proportional_batches(mut self, on: bool) -> Self {
        self.inner.y_proportional_batches = on;
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
    pub fn set_gens_history(mut self, on: bool) -> Self {
        self.inner.gens_history = on;
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

/// Per-generation metrics, accumulated when `gens_history` is enabled.
#[derive(Clone, Debug, Serialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_score: f32,
    pub best_loss: Option<f32>,
    pub worst_score: f32,
    pub worst_loss: Option<f32>,
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
    pub(crate) data_path: PathBuf,
    pub generation: usize,
    pub best: Option<BestIndividual>,
    pub history: Vec<GenerationStats>,

    scores: Vec<f32>,
    eval_losses: Vec<Option<f32>>,
    /// Cached per-gen summary — computed once in save_generation_snapshots.
    gen_best_score: f32,
    gen_best_loss: Option<f32>,
    gen_worst_score: f32,
    gen_worst_loss: Option<f32>,
}

// ── Progress tracker — periodic stdout updates during evaluation ─────────

struct ProgressTracker {
    done: std::sync::atomic::AtomicUsize,
    pop_size: usize,
}

impl ProgressTracker {
    fn new(pop_size: usize) -> Self {
        ProgressTracker {
            done: std::sync::atomic::AtomicUsize::new(0),
            pop_size,
        }
    }

    /// Workers call this after scoring. Just counts — no printing.
    fn increment(&self) {
        self.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Print the final count line.
    fn finish(&self) {
        let n = self.done.load(std::sync::atomic::Ordering::Relaxed);
        println!("  nets scored: {:>3}/{:<3}", n, self.pop_size);
    }
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
        let mut pop = Self::create_population(&options, seed)?;
        if options.dedup_pop_and_fill {
            Self::dedup_population(&mut pop);
            Self::refill_population(&options, seed, 0, &mut pop);
        }

        // Step 5: Log initialization
        Self::log_initialization(&options, &dataset, &pop, seed, &fitness);

        // Step 6: Assemble engine (thread pool, run dir, struct)
        Self::assemble_engine(options, seed, dataset, pop, fitness, data_path)
    }

    /// Step 1: Propagate engine-level pools into mutation variant, fill defaults.
    /// All required validation is done in `EngineOptionsBuilder::build()`.
    fn validate_and_fill_options(options: &mut EngineOptions) -> Result<()> {
        // Sanity checks (builder auto-fills these, but direct construction may not).
        if options.num_batches > 0 && options.batch_size == 0 {
            return Err(EngineError::InvalidOptions(
                "num_batches > 0 requires batch_size > 0".into(),
            )
            .into());
        }
        if options.num_batches == 0 {
            options.num_batches = 4;
        }
        if options.training.num_epochs == 0 {
            options.training.num_epochs = 1;
        }
        if options.hidden_dim_pool.is_empty() {
            options.hidden_dim_pool = 4..=8;
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
        // Propagate engine-level flags into training config.
        options.training.y_proportional_batches = options.y_proportional_batches;
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
            crate::utils::data::load_dataset(data_path)?.to_dtype(options.network.dtype)?.to_device(options.network.device)?;
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

    /// Refill population back to `pop_size` with fresh random individuals.
    /// Called after dedup to keep the population at full strength.
    /// Returns the number of individuals added.
    fn refill_population(
        options: &EngineOptions,
        seed: u64,
        generation: usize,
        pop: &mut Vec<Topology>,
    ) -> usize {
        let target = options.pop_size;
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
                    node.hidden_dim = Some(rng.usize(options.hidden_dim_pool.clone()));
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
            history: Vec::new(),

            scores: Vec::new(),
            eval_losses: Vec::new(),
            gen_best_score: 0.0,
            gen_best_loss: None,
            gen_worst_score: 0.0,
            gen_worst_loss: None,
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
            log::info!("");
            log::info!("  ── gen {:02}  ──", self.generation);
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

        // Step 3: Save best + worst for this generation
        self.save_generation_snapshots()
    }

    /// Step 1: Parallel rayon loop -- build, train, score each individual.
    fn eval_all_individuals(&self) -> Result<Vec<(f32, Option<f32>)>> {
        let net_opts = self.options.network;
        let train_cfg = &self.options.training;
        let fitness = &self.fitness;
        let dataset = &self.dataset;
        let batch_seed = derive_seed(self.seed, self.generation * 3);
        let tracker = ProgressTracker::new(self.pop.len());

        let results = self.pool.install(|| {
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
                    tracker.increment();
                    Ok((result.score, result.eval_loss))
                })
                .collect::<Result<Vec<_>>>()
        });
        tracker.finish();
        results
    }

    /// Step 2: Store scores and eval_losses from parallel results.
    fn update_scores(&mut self, results: Vec<(f32, Option<f32>)>) {
        self.scores = results.iter().map(|&(s, _)| s).collect();
        self.eval_losses = results.iter().map(|&(_, l)| l).collect();
    }

    /// Step 3: Find best + worst in current gen, save both to disk.
    fn save_generation_snapshots(&mut self) -> Result<bool> {
        let direction = self.fitness.direction();
        if self.scores.is_empty() {
            return Ok(false);
        }

        // Find best and worst indices
        let mut best_idx = 0;
        let mut worst_idx = 0;
        for (i, &score) in self.scores.iter().enumerate() {
            if direction.is_better(score, self.scores[best_idx]) {
                best_idx = i;
            }
            if direction.is_better(self.scores[worst_idx], score) {
                worst_idx = i;
            }
        }

        let best_score = self.scores[best_idx];
        let best_loss = self.eval_losses.get(best_idx).copied().flatten();
        let worst_score = self.scores[worst_idx];
        let _worst_loss = self.eval_losses.get(worst_idx).copied().flatten();

        // Cache for log_generation_summary and history (avoid re-iterating)
        self.gen_best_score = best_score;
        self.gen_best_loss = best_loss;
        self.gen_worst_score = worst_score;
        self.gen_worst_loss = self.eval_losses.get(worst_idx).copied().flatten();

        // Update overall best if this gen's best is better
        let improved = self
            .best
            .as_ref()
            .map(|b| direction.is_better(best_score, b.fitness))
            .unwrap_or(true);
        if improved {
            self.best = Some(BestIndividual {
                fitness: best_score,
                loss: best_loss,
                pop_index: best_idx,
                topology: self.pop[best_idx].clone(),
            });
        }

        // Save best + worst snapshots
        let best_topo = self.pop[best_idx].clone();
        let worst_topo = self.pop[worst_idx].clone();
        self.record_snapshot("best", &best_topo, best_score)?;
        self.record_snapshot("worst", &worst_topo, worst_score)?;

        Ok(improved)
    }

    // ── Improvement tracking ────────────────────────────────────────────────

    /// Save a snapshot (best or worst) for the current generation.
    fn record_snapshot(
        &mut self,
        label: &str,
        topo: &Topology,
        fitness: f32,
    ) -> std::result::Result<(), EngineError> {
        let dir = self.run_dir.join("improvements");
        fs::create_dir_all(&dir).map_err(|source| EngineError::Io {
            path: dir.display().to_string(),
            source,
        })?;

        let prefix = format!("gen{:02}_{}_{:.4}", self.generation, label, fitness);
        let env = serde_json::json!({
            "run_id": self.run_id,
            "run_seed": self.seed,
            "data_path": self.data_path.display().to_string(),
            "generation": self.generation,
            "snapshot": label,
            "fitness": fitness,
            "best_topology": topo.to_json()
                .map_err(|e| EngineError::Json(format!("snapshot json: {e}")))?,
        });
        let json_str = serde_json::to_string_pretty(&env)
            .map_err(|e| EngineError::Json(format!("snapshot json: {e}")))?;

        let path = dir.join(format!("{prefix}.json"));
        fs::write(&path, &json_str).map_err(|source| EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;

        let md_path = dir.join(format!("{prefix}.md"));
        let net = Network::build(topo, Device::CPU).ok();
        let md = crate::utils::ascii_utils::topology_markdown(
            topo,
            Some(fitness),
            net.as_ref(),
        );
        fs::write(&md_path, md).map_err(|source| EngineError::Io {
            path: md_path.display().to_string(),
            source,
        })?;

        debug!("record_snapshot -- gen={} {label} fitness={:.4}", self.generation, fitness);
        Ok(())
    }

    // ── Logging ─────────────────────────────────────────────────────────────

    fn log_generation_summary(&self) {
        if self.scores.is_empty() {
            return;
        }
        let all_time = self.best.as_ref().map(|b| b.fitness).unwrap_or(self.gen_best_score);
        let fl = self.fitness.fitness_label();
        log::info!("  best {fl}={:.4}  all {fl}={:.4}", self.gen_best_score, all_time);
        if let Some(loss) = self.gen_best_loss {
            let ll = self.fitness.train_metric_label();
            log::info!("  loss {ll}={loss:.4}");
        }
    }

    // ── Genetics -- selection, crossover, mutation ────────────────────────────

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
        let target = self.options.pop_size;
        log::info!("  ── genetics ──");
        log::info!("  selection    {unique}/{target} unique ({sel_label})");
        log::info!("  crossover    {cx_pairs} pairs  {target}/{target}");
        if dedup_removed > 0 {
            log::info!("  dedup        -{dedup_removed}  {}/{} → {}/{}", pre_dedup, target, pre_dedup - dedup_removed, target);
        } else {
            log::info!("  dedup        -0  {target}/{target}");
        }
        if refill_added > 0 {
            let after_refill = self.pop.len();
            log::info!("  refill       +{refill_added}  {}/{} → {}/{}", after_refill - refill_added, target, after_refill, target);
        } else {
            log::info!("  refill       +0  {}/{}", self.pop.len(), target);
        }
        if post_refill_dedup > 0 {
            log::info!("  dedup        -{post_refill_dedup} (post-refill)  {}/{}", self.pop.len(), target);
        }
        log::info!("  mutation     {mut_count} nets  {}/{}", self.pop.len(), target);
        // Capture history if enabled (uses cached values from save_generation_snapshots)
        if self.options.gens_history && !self.scores.is_empty() {
            self.history.push(GenerationStats {
                generation: self.generation,
                best_score: self.gen_best_score,
                best_loss: self.gen_best_loss,
                worst_score: self.gen_worst_score,
                worst_loss: self.gen_worst_loss,
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
        let indices = selection.apply(&self.scores, dir, &mut rng);
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
            "history": if self.history.is_empty() { serde_json::Value::Null } else { serde_json::to_value(&self.history).unwrap_or(serde_json::Value::Null) },
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
        // 2 files per gen (best + worst)
        assert_eq!(json_files.len(), engine.options.num_generations * 2);
        // Load the best snapshot of the last generation
        let last_gen = engine.options.num_generations - 1;
        let best_prefix = format!("gen{:02}_best_", last_gen);
        let best_file = json_files.iter().find(|f| f.starts_with(&best_prefix)).unwrap();
        let latest_json =
            std::fs::read_to_string(imp_dir.join(best_file)).unwrap();
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
            .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
            .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
            .set_mutation(crate::mutation::MutationMethod::Activation { prob: 0.1 })
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
        // set_mutation() is required — omitting it should error at build time.
        assert!(
            EngineOptions::builder()
                .set_pop_size(4)
                .set_num_generations(1)
                .set_hidden_dim_pool(4..=4)
                .set_num_batches(2)
                .set_batch_size(8)
                .set_num_epochs(1)
                .build()
                .is_err()
        );
        let opts = EngineOptions::builder()
            .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
            .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
            .set_mutation(crate::mutation::MutationMethod::Activation { prob: 0.1 })
            .set_combine_op_pool(vec![])
            .build()
            .unwrap();
        assert_eq!(opts.combine_op_pool.len(), 4);
        let opts = EngineOptions::builder()
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
            num_generations: 1,
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
}
