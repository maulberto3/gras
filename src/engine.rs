//! The engine 🏭 — the flagship public API.
//!
//! A NAS loop over random topologies: seed a population, score every
//! individual with a user-supplied fitness (a built-in like `Mse`, or a
//! drop-in closure), track the current best, record every improvement into
//! `improvements/`, log compactly —
//! and leave room for real genetics (crossover/mutation) later.
//!
//! # Data contract
//!
//! The engine consumes data as a **path to tensors** written by
//! [`crate::utils::data::save_dataset`] — flodl-native data only. The dataset is
//! loaded once at [`Engine::new`] and reused for every individual, every
//! generation.
//!
//! # Replicating an experiment
//!
//! [`Engine::to_json`] dumps everything needed to reproduce a run: the
//! [`EngineOptions`], the data path, the best fitness, and the **best
//! topology as JSON** — feed that back to `Topology::from_json` +
//! `Network::build` and you have the exact best network of the run.

use std::fs;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use flodl::nn::Module;
use flodl::tensor::Result;
use flodl::{DType, Device, Tensor, Variable};
use log::debug;
use rayon::prelude::*;
use serde::Serialize;

use crate::utils::data::Dataset;
use crate::utils::error::EngineError;
pub use crate::fitness::{Direction, Fitness, FitnessLabel};
use crate::network::{Network, NetworkOptions};
use crate::node::{Activation, NodeKind};
use crate::selection::SelectionMethod;
use crate::topology::{CombineOp, Topology, TopologyOptions};

/// Knobs for one engine run. Serialized by [`Engine::to_json`] into the run
/// folder's `engine.json`, so an experiment is fully reproducible.
#[derive(Clone, Debug, PartialEq, Serialize)]
// ── EngineOptions — the experiment configuration ──────────────────────────

pub struct EngineOptions {
    /// Number of individuals in the population.
    pub pop_size: usize,
    /// Number of generations to run.
    pub num_generations: usize,
    /// Population base seed. `None` → a random seed is derived per run and
    /// recorded in `engine.json` (as `run_seed`), so distinct launches
    /// explore fresh topologies while staying re-launchable. Each individual
    /// derives its own seed from this base via a deterministic chain.
    pub seed: Option<u64>,
    /// The shared **topology template** every individual derives from — the
    /// single source of truth for the topology knobs (node/port ranges,
    /// dims, combine op). Each individual clones it and overrides only the
    /// `seed` (derived from the population base seed).
    pub topology_options: TopologyOptions,
    /// **GP search space** — the hidden-dim range sampled **per individual**
    /// at population creation (`8..=8` = every individual uses the
    /// template's `hidden_dim`, the pre-GP behavior).
    pub hidden_dim_pool: RangeInclusive<usize>,
    /// **GP search space** — the combine-op pool sampled **per individual**
    /// at population creation (default `[Add]` = the template's
    /// `combine_op`).
    pub combine_op_pool: Vec<CombineOp>,
    /// **GP search space** — the activation pool. Sampled per-node at
    /// population creation and, later, as the swap source for mutation.
    pub activation_pool: Vec<Activation>,
    /// **GP search space** — the standardize-op pool. Per-node normalization
    /// sampled at population creation (default `[Identity]`).
    pub standardize_op_pool: Vec<crate::node::StandardizeOp>,
    /// **Evaluation budget** — how much of the data each candidate is scored
    /// on. Each generation picks `num_batches` non-overlapping batches of
    /// `batch_size` rows **without replacement**. The same batches are reused
    /// for every individual of a generation (seeded from `run_seed + generation`),
    /// so scores stay comparable and deterministic. If the budget exceeds the
    /// dataset, all rows are exhausted (shuffled) with no duplicates.
    pub num_batches: usize,
    /// Rows per batch (used both for sampled batches and for chunking the
    /// whole-dataset pass; default 128).
    pub batch_size: usize,
    /// Built-in scoring strategy recorded for reproducibility.
    pub fitness_label: FitnessLabel,
    /// Threads for parallel population evaluation (`0` = rayon's default,
    /// i.e. available parallelism; default 3 to stay conservative on shared
    /// machines).
    pub num_threads: usize,
    /// Parent folder for per-run checkpoint folders (`results/<ts>/`).
    pub results_dir: PathBuf,
    /// Probability that an individual's **activation** is mutated each
    /// generation (0.0 = no activation mutation, the default). Each
    /// mutation picks one random hidden node and swaps its activation
    /// to a different one from `activation_pool`.
    pub mutate_activ_prob: f64,
    /// Training hyperparameters applied to every individual before scoring.
    /// `train_epochs = 0` (the default) skips training entirely — a
    /// random-init forward pass, the pre-training behavior.
    pub training: crate::trainer::TrainingConfig,
    /// Selection strategy for picking parents for the next generation.
    /// Default: tournament with size 3.
    pub selection: SelectionMethod,
    /// The **network link** of the option chain (engine → topology →
    /// network): passed to `Network::build_with_options` when materializing
    /// every individual. Today it only carries the device; it's where future
    /// network-level knobs (per-node overrides, …) will land.
    pub network: NetworkOptions,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            pop_size: 10,
            num_generations: 5,
            seed: None, // random per run, recorded in engine.json
            topology_options: TopologyOptions::default(),
            hidden_dim_pool: 4..=8,
            combine_op_pool: vec![],   // empty → all built-in ops at build time
            activation_pool: vec![],   // empty → all built-in activations at build time
            standardize_op_pool: vec![], // empty → all built-in ops at build time
            num_batches: 16,
            batch_size: 128,
            fitness_label: FitnessLabel::default(),
            num_threads: 3,
            results_dir: PathBuf::from("results"),

            mutate_activ_prob: 0.0,
            training: crate::trainer::TrainingConfig::default(),
            selection: SelectionMethod::default(),
            network: NetworkOptions::default(),
        }
    }
}

impl EngineOptions {
    /// Derive topology options for one individual: clone the shared template
    /// and override the seed with the individual's derived seed. Each
    /// population slot gets its own seed via the deterministic chain, so the
    /// whole population is reproducible from the base seed alone.
    pub(crate) fn derive_topology_options(&self, seed: usize) -> TopologyOptions {
        let mut t = self.topology_options;
        t.seed = seed;
        t
    }

    /// The shared topology template (base seed baked in), serialized into
    /// `engine.json` so the file spells out the full option chain
    /// (engine → topology → network).
    fn topology_template(&self) -> TopologyOptions {
        self.topology_options
    }

    /// Start a fluent builder chain — the ergonomic way to configure a run
    /// without a big nested struct literal:
    ///
    /// ```
    /// use gras::engine::{EngineOptions, Fitness};
    /// use gras::node::Activation;
    ///
    /// // Minimal — all pools default to every built-in variant:
    /// let opts = EngineOptions::builder()
    ///     .set_pop_size(15)
    ///     .set_num_generations(3)
    ///     .set_hidden_dim_pool(8..=16)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Narrow a pool to specific options:
    /// let opts = EngineOptions::builder()
    ///     .set_pop_size(15)
    ///     .set_activation_pool(vec![Activation::ReLU, Activation::GeLU])
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// The `set_*` methods are **flat**: each routes into the right layer
    /// (engine knobs, topology template, GP pools, network options), so
    /// callers never touch the nested structs unless they want to.
    ///
    /// **Pool defaults:** empty pools are auto-filled with all built-in
    /// variants at `build()` time. Only call `set_*_pool()` to **narrow**
    /// the search space.
    pub fn builder() -> EngineOptionsBuilder {
        EngineOptionsBuilder {
            inner: EngineOptions::default(),
        }
    }
}

/// Fluent builder for [`EngineOptions`] — a flat `set_*` chain over the
/// nested option structs. Start with [`EngineOptions::builder`], finish with
/// [`build`](EngineOptionsBuilder::build) (validated options), then pass
/// to [`Engine::new`] to start the run.
// ── EngineOptionsBuilder — fluent builder ─────────────────────────────────

pub struct EngineOptionsBuilder {
    inner: EngineOptions,
}

impl EngineOptionsBuilder {
    /// Validate the accumulated options and hand them back. Checks the same
    /// invariants `Engine::new` relies on: non-empty population, valid
    /// evaluation budget, and non-empty GP pools.
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
        // Empty pools → fill with all built-in ops (no set_* = use everything).
        if o.combine_op_pool.is_empty() {
            o.combine_op_pool = vec![
                CombineOp::Add, CombineOp::Mean,
                CombineOp::Max, CombineOp::Min,
            ];
        }
        if o.activation_pool.is_empty() {
            o.activation_pool = vec![
                Activation::Identity, Activation::ReLU, Activation::GeLU,
                Activation::SiLU, Activation::SELU, Activation::Tanh,
                Activation::Sigmoid, Activation::Mish, Activation::LeakyReLU,
                Activation::ELU, Activation::GeluTanh, Activation::Softplus,
                Activation::HardSwish, Activation::HardSigmoid,
            ];
        }
        if o.standardize_op_pool.is_empty() {
            o.standardize_op_pool = vec![
                crate::node::StandardizeOp::Identity,
                crate::node::StandardizeOp::LayerNorm,
            ];
        }
        if o.training.num_epochs == 0 {
            return Err(EngineError::InvalidOptions(
                "num_epochs must be > 0 (set via set_num_epochs)".to_string(),
            )
            .into());
        }
        Ok(self.inner)
    }

    // ── engine knobs ───────────────────────────────────────────────────────
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
        self
    }
    pub fn set_batch_size(mut self, n: usize) -> Self {
        self.inner.batch_size = n;
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

    /// Probability each individual's activation is mutated per generation
    /// (0.0 = no activation mutation, 1.0 = mutate every individual).
    pub fn set_mutate_activ_prob(mut self, p: f64) -> Self {
        self.inner.mutate_activ_prob = p.clamp(0.0, 1.0);
        self
    }
    /// Selection strategy for the genetic loop.
    pub fn set_selection(mut self, method: SelectionMethod) -> Self {
        self.inner.selection = method;
        self
    }

    // ── training knobs ──────────────────────────────────────────────────
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

    // ── topology template (the blueprint's structure knobs) ────────────────
    pub fn set_topology_options(mut self, t: TopologyOptions) -> Self {
        self.inner.topology_options = t;
        self
    }
    pub fn set_hidden_dim(mut self, n: usize) -> Self {
        self.inner.topology_options.hidden_dim = n;
        self
    }
    // ── GP search space: the pools the engine samples per individual ──────
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


    // ── topology knobs (node/wire ranges) ───────────────────────────────
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

    // ── network options (execution knobs) ─────────────────────────────────
    pub fn set_network(mut self, n: NetworkOptions) -> Self {
        self.inner.network = n;
        self
    }
    pub fn set_device(mut self, d: Device) -> Self {
        self.inner.network.device = d;
        self
    }
    pub fn set_dtype(mut self, d: DType) -> Self {
        self.inner.network.dtype = d;
        self
    }
    /// Deterministic weight init: every network built from the same
    /// options gets the exact same weights (same seed ⇒ same built model).
    pub fn set_init_seed(mut self, seed: usize) -> Self {
        self.inner.network.seed = seed;
        self
    }
}

/// The best individual seen so far — re-exported from [`crate::fitness`].
pub use crate::fitness::BestIndividual;

/// A running NAS experiment. Build it with [`Engine::new`], run it with
/// [`Engine::run`].
// ── Engine — the NAS experiment runner ────────────────────────────────────

pub struct Engine {
    pub options: EngineOptions,
        /// The resolved population base seed — `options.seed` when given,
    /// otherwise a fresh random seed derived at construction. Recorded in
    /// `engine.json` so even a randomized run is re-launchable.
    pub seed: u64,
    /// Unix timestamp identifying this run (also the checkpoint folder name).
    pub run_id: String,
    /// `results/<run_id>/` — the checkpoint path for this run. Fixed at
    /// construction; created on disk (with `engine.json` +
    /// `improvements/`) by [`Engine::run`].
    pub run_dir: PathBuf,
    /// Thread pool for parallel population evaluation (`options.num_threads`).
    pool: rayon::ThreadPool,
    /// The population of blueprints (evolved in place by future genetics).
    pub pop: Vec<Topology>,
    /// The scorer (built-in or user closure).
    pub(crate) fitness: Fitness,
    /// The dataset loaded from the user-provided path.
    pub(crate) data: Dataset,
    /// The path the dataset was loaded from (recorded for reproducibility).
    pub(crate) data_path: PathBuf,
    /// Current generation (0-based; incremented by `next_generation`).
    pub generation: usize,
    /// Current best individual (updated whenever a run improves it).
    pub best: Option<BestIndividual>,
    /// How many best-improvements have been recorded into `improvements/`
    /// (also the next filename counter).
    pub(crate) improvements: usize,
    /// Last generation's scores (for compact logging).
    scores: Vec<f32>,
}

impl Engine {
    // ── Construction ────────────────────────────────────────────────────────

    /// Start an experiment: load the dataset from `data_path` and seed a
    /// population of `options.pop_size` random graphs — each individual
    /// derives its seed from the base via a deterministic chain (see
    /// [`Engine::seed`]), so the whole population is reproducible from
    /// that seed alone. The checkpoint path `results/<ts>/` is fixed here,
    /// but the folder is only created on disk by [`Engine::run`].
    ///
    /// Auto-detects `input_dim` from the dataset (the single source of
    /// truth) and propagates it into the topology template.
    pub fn new(mut options: EngineOptions, data_path: &Path, fitness: Fitness) -> Result<Self> {
        // ── validation ──────────────────────────────────────────────────
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
                "hidden_dim_pool must be a non-empty range (start <= end)".to_string(),
            )
            .into());
        }
        // Empty pools → fill with all built-in ops (no set_* = use everything).
        if options.combine_op_pool.is_empty() {
            options.combine_op_pool = vec![
                CombineOp::Add, CombineOp::Mean,
                CombineOp::Max, CombineOp::Min,
            ];
        }
        if options.activation_pool.is_empty() {
            options.activation_pool = vec![
                Activation::Identity, Activation::ReLU, Activation::GeLU,
                Activation::SiLU, Activation::SELU, Activation::Tanh,
                Activation::Sigmoid, Activation::Mish, Activation::LeakyReLU,
                Activation::ELU, Activation::GeluTanh, Activation::Softplus,
                Activation::HardSwish, Activation::HardSigmoid,
            ];
        }
        if options.standardize_op_pool.is_empty() {
            options.standardize_op_pool = vec![
                crate::node::StandardizeOp::Identity,
                crate::node::StandardizeOp::LayerNorm,
            ];
        }
        if options.training.num_epochs == 0 {
            return Err(EngineError::InvalidOptions(
                "num_epochs must be > 0 (set via set_num_epochs)".to_string(),
            )
            .into());
        }
        if options.num_batches == 0 {
            options.num_batches = 16;
        }

        // ── seed resolution ─────────────────────────────────────────────
        let seed = options.seed.unwrap_or_else(|| {
            let t = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            t ^ fastrand::u64(..)
        });
        options.topology_options.seed = seed as usize;
        options.network.seed = seed as usize;

        // ── data loading ────────────────────────────────────────────────
        let data = crate::utils::data::load_dataset(data_path)?.to_dtype(options.network.dtype)?;
        debug!("Engine::new — loaded dataset from {}: inputs {:?} targets {:?}", data_path.display(), data.inputs.shape(), data.targets.shape());
        let data_in = data.inputs.shape().get(1).copied().ok_or_else(|| {
            EngineError::DataMismatch("dataset inputs must be 2-D [n, input_dim]".into())
        })?;
        options.topology_options.input_dim = data_in as usize;
        let data_out = data.targets.shape().get(1).copied().ok_or_else(|| {
            EngineError::DataMismatch("dataset targets must be 2-D [n, output_dim]".into())
        })?;
        options.topology_options.output_dim = data_out as usize;
        options.fitness_label = crate::fitness::FitnessLabel(fitness.label().to_string());
        debug!("Engine::new — input_dim={} output_dim={} seed={} fitness={}", options.topology_options.input_dim, options.topology_options.output_dim, seed, fitness.label());

        debug!("Engine::new — pools: hidden {:?} combine {} acts {} std {}", options.hidden_dim_pool, options.combine_op_pool.len(), options.activation_pool.len(), options.standardize_op_pool.len());

        // ── checkpoint path ─────────────────────────────────────────────
        let run_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
            .to_string();
        let run_dir = options.results_dir.join(&run_id);

        // ── thread pool ─────────────────────────────────────────────────
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

        // ── population ──────────────────────────────────────────────────
        let pop = Self::create_population(&options, seed)?;
        debug!("Engine::new — seeded {} individuals (base seed {})", pop.len(), seed);

        crate::utils::log_utils::log_options(&options);
        crate::utils::log_utils::log_dataset(&data, pop.len(), seed, fitness.direction());
        crate::utils::log_utils::log_population(&pop, &options);

        Ok(Engine {
            options,
            seed,
            run_id,
            run_dir,
            pool,
            pop,
            fitness,
            data,
            data_path: data_path.to_path_buf(),
            generation: 0,
            best: None,
            improvements: 0,
            scores: Vec::new(),
        })
    }

    /// Create a population of random topologies. Each individual derives
    /// its seed from the base via a deterministic chain, so the whole
    /// population is reproducible from `run_seed` alone.
    fn create_population(
        options: &EngineOptions,
        seed: u64,
    ) -> Result<Vec<Topology>> {
        debug!("create_population — pop_size={} min_hidden={}..={} base_seed={}",
            options.pop_size, options.topology_options.min_hidden_num_nodes,
            options.topology_options.max_hidden_num_nodes, seed);
        let mut pop = Vec::with_capacity(options.pop_size);
        for i in 0..options.pop_size {
            let ind_seed = derive_seed(seed, i);
            let mut rng = fastrand::Rng::with_seed(ind_seed);
            let n_hidden =
                rng.usize(options.topology_options.min_hidden_num_nodes..=options.topology_options.max_hidden_num_nodes);
            let ind_opts = options.derive_topology_options(ind_seed as usize);
            let mut graph = Topology::new(i, Some(ind_opts));
            graph.create_random_hidden_nodes(n_hidden);
            for node in &mut graph.nodes {
                if node.kind == NodeKind::Hidden {
                    node.hidden_dim =
                        Some(rng.usize(options.hidden_dim_pool.clone()));
                    node.activation =
                        options.activation_pool[rng.usize(0..options.activation_pool.len())];
                    node.combine_op =
                        Some(options.combine_op_pool[rng.usize(0..options.combine_op_pool.len())]);
                    node.standardize =
                        Some(options.standardize_op_pool[rng.usize(0..options.standardize_op_pool.len())]);
                }
            }
            graph.refresh_labels();
            graph.finalize();
            debug!("  ind[{i}] seed={} n_hidden={} nodes={} wires={} in_ports={}",
                ind_seed, n_hidden, graph.nodes.len(), graph.connections.len(), graph.graph_inputs.len());
            pop.push(graph);
        }
        Ok(pop)
    }

    /// The last generation's fitness scores, index-aligned with `pop`
    /// (`scores[i]` belongs to `pop[i]`). Empty until the first
    /// [`Engine::run`] evaluates the population.
    // ── Query ───────────────────────────────────────────────────────────────

    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Run the full experiment: `num_generations` rounds of
    /// evaluate → log → evolve. Every best-improvement is appended to
    /// `improvements/`, and an `engine.json` snapshot is written at the
    /// start (initial envelope, no best yet) and at the end.
    ///
    /// The per-run checkpoint folder is created **here**, not at
    /// [`Engine::new`] — building an engine (or inspecting its options)
    /// leaves nothing on disk; only an actual run does.
    // ── Run — the main loop ─────────────────────────────────────────────────

    pub fn run(&mut self) -> Result<()> {
        fs::create_dir_all(&self.run_dir).map_err(|source| EngineError::Io {
            path: self.run_dir.display().to_string(),
            source,
        })?;
        debug!("run — starting {} gens with pop={} fitness={:?} run_dir={}",
            self.options.num_generations, self.pop.len(), self.fitness.direction(), self.run_dir.display());
        
        crate::utils::log_utils::log_run_start(&self.options, &self.run_dir, self.fitness.direction());
        
        // Initial experiment envelope (no best yet); the final one is
        // written after the loop.
        let initial = self.to_json()?;
        fs::write(self.run_dir.join("engine.json"), initial).map_err(|source| EngineError::Io {
            path: self.run_dir.join("engine.json").display().to_string(),
            source,
        })?;
        
        let run_start = Instant::now();
        
        // TODO(engine): stop criteria beyond max generations — e.g. a
        // StopCriterion enum with TargetFitness (stop once the best crosses
        // a threshold) and NoImprovement (stop after N stagnant generations).
        for g in 0..self.options.num_generations {
            debug!("══ gen {:02}/{:02} ══", g, self.options.num_generations);
            let _improved = self.evaluate_population()?;
            self.next_generation();
        }
        let run_elapsed = run_start.elapsed();
        let json = self.to_json()?;
        fs::write(self.run_dir.join("engine.json"), json).map_err(|source| EngineError::Io {
            path: self.run_dir.join("engine.json").display().to_string(),
            source,
        })?;

        crate::utils::log_utils::log_best(&self.best, self.generation)?;

        // ══ Log: run summary ══════════════════════════════════════════
        crate::utils::log_utils::log_run_summary(
            run_elapsed,
            self.seed,
            self.generation,
            self.pop.len(),
            &self.options,
            &self.data_path,
            &self.fitness,
            self.improvements,
            &self.best,
            &self.run_dir,
        ).map_err(|e| EngineError::Json(format!("run summary log: {e}")))?;

        Ok(())
    }

    /// Score every individual on the evaluation budget. Returns whether the
    /// overall best improved (and appends it to `improvements/` if so).
    fn evaluate_population(&mut self) -> Result<bool> {
        // One budget per generation: the sampled batches are reused for every
        // individual so scores are comparable (same data, same compute), and
        // re-seeded from `run_seed + generation` so each generation sees
        // fresh data deterministically.
        let mut rng =
            fastrand::Rng::with_seed(self.seed.wrapping_add(self.generation as u64));
        let (train_batches, eval_batches) = self.sample_batches_split(&mut rng)?;
        debug!("  evaluate_population — gen {} train={} eval={}",
            self.generation, train_batches.len(), eval_batches.len());

        // ⚡ Parallel evaluation: one rayon task per individual. The batches
        // are plain `Tensor` pairs (Tensor is Send + Sync); each task wraps
        // its own copies in fresh `Variable`s, because flodl's Variable is
        // Rc-based and can't cross threads.
        let direction = self.fitness.direction();
        let net_opts = self.options.network;
        let train_cfg = &self.options.training;
        let fitness = &self.fitness;
        let pop_size = self.pop.len();
        // Progress counter for tqdm-like logging.
        let done = std::sync::atomic::AtomicUsize::new(0);
        // Best-so-far tracker (score bits, atomic for lock-free updates).
        let best_bits =
            std::sync::atomic::AtomicU32::new(f32::to_bits(if direction == Direction::Minimize {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            }));
        let generation = self.generation;
        let progress_lock = std::sync::Mutex::new(());
        let scores: Vec<f32> = self.pool.install(|| {
            self.pop
                .par_iter()
                .map(|graph| {
                    // 🎲 Deterministic weights: the individual's derived seed
                    // drives its weight init, so a blueprint always scores
                    // identically (same run_seed ⇒ same scores ⇒ same run).
                    let mut no = net_opts;
                    no.seed = graph.options.seed;
                    let node_hidden_dims: Vec<usize> = graph.nodes.iter()
                        .filter_map(|n| n.hidden_dim).collect();
                    debug!("    ind[{}] build: nodes={} seed={} node_dims={:?}",
                        graph.id, graph.nodes.len(), no.seed, node_hidden_dims);
                    let mut net = Network::build_with_options(graph, &no)?;

                    // 🧠 Train (when train_epochs > 0).  Each thread gets
                    // its own Network + optimizer — no cross-thread sharing.
                    // The same fitness function drives backward + scoring.
                    crate::trainer::train_network(&mut net, train_cfg, fitness, &train_batches)?;
                    debug!("    ind[{}] trained: {} params, {} epochs × {} train batches",
                        graph.id, net.parameters().len(), train_cfg.num_epochs, train_batches.len());

                    // Score on held-out eval batches (honest generalization).
                    let mut total = 0.0;
                    for (xb, yb) in &eval_batches {
                        let x = Variable::new(xb.clone(), false);
                        let y = Variable::new(yb.clone(), false);
                        let pred = net.forward(&x)?;
                        total += self.fitness.score(&pred, &y)?;
                    }
                    let score = if eval_batches.is_empty() { 0.0 } else { total / eval_batches.len() as f32 };
                    debug!("    ind[{}] score={:.6}", graph.id, score);

                    // 📊 Progress: update best-so-far and print dynamic progress.
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
                    // Dynamic progress: overwrite the same line with \r.
                    {
                        let _guard = progress_lock.lock().unwrap();
                        use std::io::Write;
                        print!(
                            "\r  gen {:02} net {:>2}/{pop_size}  best {cur_best:.4}\x1b[K",
                            generation, n,
                        );
                        std::io::stdout().flush().unwrap();
                    }

                    Ok(score)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        // Move past the progress line before the generation summary.
        println!();
        self.scores = scores;

        // Walk the scores: any individual that beats the running best
        // is a new improvement, recorded immediately.
        // Copy scores + topologies to avoid borrow conflicts with &mut self.
        let snapshot: Vec<(usize, f32, Topology)> = self.scores.iter().enumerate()
            .map(|(i, &s)| (i, s, self.pop[i].clone()))
            .collect();
        let mut any_improved = false;
        for (i, score, topo) in snapshot {
            let beats_best = self
                .best
                .as_ref()
                .map(|b| direction.is_better(score, b.fitness))
                .unwrap_or(true);
            if beats_best {
                self.best = Some(BestIndividual {
                    fitness: score,
                    pop_index: i,
                    topology: topo,
                });
                log::info!(
                    "🏆 new best: pop[{i}] fitness {score:.4} at gen {}",
                    self.generation
                );
                self.record_improvement()?;
                any_improved = true;
            }
        }
        Ok(any_improved)
    }

    /// Sample the per-generation evaluation budget: `num_batches` batches
    /// of `batch_size` rows **without replacement** from the loaded dataset,
    /// returned as raw `(Tensor, Tensor)` pairs. The caller seeds `rng`
    /// from `run_seed + generation`, so a run reproduces the same batches.
    ///
    /// - Budget ≤ dataset: randomly pick `num_batches` non-overlapping
    ///   batches. Every individual sees the same batches.
    /// - Budget > dataset: exhaust all rows (shuffled), produce as many
    ///   full batches as possible, warn, and continue to the next gen.
    // ── Batch sampling ──────────────────────────────────────────────────────

    /// Split shuffled indices into non-overlapping train + eval batches.
    fn sample_batches_split(
        &self, rng: &mut fastrand::Rng,
    ) -> Result<(Vec<(Tensor, Tensor)>, Vec<(Tensor, Tensor)>)> {
        let n = self.data.inputs.shape()[0] as usize;
        let bs = self.options.batch_size;
        let total = self.options.num_batches;
        let max_full = n / bs;
        let actual = total.min(max_full).max(1);

        let mut all_idx: Vec<i64> = (0..n as i64).collect();
        for i in (1..all_idx.len()).rev() {
            let j = rng.usize(0..=i);
            all_idx.swap(i, j);
        }

        // Split: first half = train, second half = eval (no overlap).
        let train_count = (actual / 2).max(1);
        let eval_count = (actual - train_count).max(1);

        let make_batch = |start: usize, count: usize| -> Result<Vec<(Tensor, Tensor)>> {
            let mut batches = Vec::with_capacity(count);
            for b in 0..count {
                let s = start + b * bs;
                let e = (s + bs).min(n);
                if s >= e { break; }
                let idx: Vec<i64> = all_idx[s..e].to_vec();
                let idx_t = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;
                let xb = self.data.inputs.index_select(0, &idx_t)?;
                let yb = self.data.targets.index_select(0, &idx_t)?;
                batches.push((xb, yb));
            }
            Ok(batches)
        };

        let train_batches = make_batch(0, train_count)?;
        let eval_batches = make_batch(train_count * bs, eval_count)?;
        // When dataset is too small to split, eval falls back to train data.
        let eval_batches = if eval_batches.is_empty() { train_batches.clone() } else { eval_batches };
        debug!("  sample_batches_split — train={} eval={} (from {} total)",
            train_batches.len(), eval_batches.len(), actual);
        Ok((train_batches, eval_batches))
    }

    /// Append the current best to `run_dir/improvements/` — the evolution
    /// trail. Each best-improvement writes the topology **recipe**
    /// (`Topology::from_json` + `Network::build` replicates the net),
    /// so the trail reads top-down and the latest entry is the current best.
    ///   `{counter:04}_gen{gen:02}_fitness{fit:.4}.json`          (recipe)
    // ── Improvement tracking ────────────────────────────────────────────────

    fn record_improvement(&mut self) -> Result<()> {
        let Some(b) = &self.best else { return Ok(()) };
        let dir = self.run_dir.join("improvements");
        fs::create_dir_all(&dir).map_err(|source| EngineError::Io {
            path: dir.display().to_string(),
            source,
        })?;

        // Recipe: the blueprint, exact round-trip.
        let json = b
            .topology
            .to_json()
            .map_err(|e| EngineError::Json(format!("improvement json: {e}")))?;
       
        // Filename: 0000_gen00_fitness0.1234.json
        let name = format!(
            "{:04}_gen{:02}_fitness{:.4}.json",
            self.improvements, self.generation, b.fitness
        );
        
        // Write the recipe to disk.
        let path = dir.join(&name);
        fs::write(&path, json).map_err(|source| EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;
        
        // Markdown visualization: tables + wiring diagram.
        let md_name = name.replace(".json", ".md");
        let md_path = dir.join(&md_name);
        let net = Network::build(&b.topology, Device::CPU).ok();
        let md = crate::utils::ascii_utils::topology_markdown(&b.topology, Some(b.fitness), net.as_ref());
        fs::write(&md_path, md).map_err(|source| EngineError::Io {
            path: md_path.display().to_string(),
            source,
        })?;
        debug!("record_improvement — saved #{:04} gen={} fitness={:.4} → {}",
            self.improvements, self.generation, b.fitness, path.display());
        self.improvements += 1;
        Ok(())
    }

    /// Advance one generation: select → crossover → mutate → generation += 1.
    // ── Genetics — selection, crossover, mutation ────────────────────────────

    fn next_generation(&mut self) {
        debug!("next_generation — gen {} → {}", self.generation, self.generation + 1);
        self.select();
        self.crossover();
        self.mutate();
        self.generation += 1;
    }

    /// 🧬 Selection — elitism + tournament (or future strategies).
    ///
    /// Reorders `self.pop` and `self.scores` in-place so that the fittest
    /// individuals are positioned to survive into the next generation.
    pub fn select(&mut self) {
        if self.scores.is_empty() {
            return;
        }
        let dir = self.fitness.direction();
        let mut rng = fastrand::Rng::with_seed(self.seed + self.generation as u64 + 0xCAFE);
        let indices = match self.options.selection {
            crate::selection::SelectionMethod::Tournament { tournament_size } => {
                crate::selection::select(&self.scores, dir, &mut rng, tournament_size)
            }
        };
        let new_pop: Vec<Topology> = indices.iter().map(|&i| self.pop[i].clone()).collect();
        let new_scores: Vec<f32> = indices.iter().map(|&i| self.scores[i]).collect();
        self.pop = new_pop;
        self.scores = new_scores;
        // Summarize selection: who survived and who was culled.
        let mut counts = vec![0usize; self.pop.len()];
        for &i in &indices { counts[i] += 1; }
        let selected: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] > 0).collect();
        let culled: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] == 0).collect();
        let highlights: Vec<String> = selected.iter()
            .map(|&i| format!("pop[{i}]×{}", counts[i]))
            .collect();
        debug!("  selection  survivors: {} · culled: {:?}", highlights.join(" "), culled);
    }

    /// 🧬 Crossover placeholder — the first genetic operator (planned).
    ///
    /// Future: pick two parents from the population, splice their topologies
    /// (e.g. swap sub-graphs or shuffle node order) to produce offspring, and
    /// replace a slice of the population. **Not implemented yet** — a no-op,
    /// so the population stays static and the engine loop is fully testable.
    pub fn crossover(&mut self) {
        // TODO(engine): topology-level crossover.
    }

    /// 🎲 Uniform mutation — swap one random hidden node's activation.
    ///
    /// For each individual: with probability `mutate_prob`, pick a random
    /// hidden node and swap its activation to a *different* one from
    /// `activation_pool`. Input and Output nodes are never mutated.
    pub fn mutate(&mut self) {
        if self.options.mutate_activ_prob <= 0.0 || self.options.activation_pool.len() < 2 {
            return;
        }
        let mut rng = fastrand::Rng::with_seed(self.seed + self.generation as u64 + 0xBEEF);
        for graph in &mut self.pop {
            if rng.f64() >= self.options.mutate_activ_prob {
                continue;
            }
            // Collect hidden node indices.
            let hiddens: Vec<usize> = graph
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.kind == crate::node::NodeKind::Hidden)
                .map(|(i, _)| i)
                .collect();
            if hiddens.is_empty() {
                continue;
            }
            let idx = hiddens[rng.usize(0..hiddens.len())];
            let cur = graph.nodes[idx].activation;
            // Pick a different activation from the pool.
            let pool = &self.options.activation_pool;
            let new_act = loop {
                let pick = pool[rng.usize(0..pool.len())];
                if pick != cur {
                    break pick;
                }
            };
            graph.nodes[idx].activation = new_act;
            debug!("  ==> A Mutation happened: graph {} node {} activation changed from {} to {}", graph.id, idx, cur, new_act);
        }
    }

    /// Everything needed to replicate this experiment, as JSON: the resolved
    /// `run_seed`, the options (including the topology template every
    /// individual derives from), the data path, the best fitness, the **best
    /// topology** recipe (feed it to `Topology::from_json` +
    /// `Network::build` to recreate the best net) and the **best network
    /// facts** (`Network::to_json` — what that build produced, so the file
    /// is self-describing without running code).
    // ── Serialization ───────────────────────────────────────────────────────

    pub fn to_json(&self) -> Result<String> {
        let best_topology = match &self.best {
            Some(b) => Some(
                b.topology
                    .to_json()
                    .map_err(|e| EngineError::Json(format!("best topology: {e}")))?,
            ),
            None => None,
        };
        let best_net_facts = match &self.best {
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
        let spec = serde_json::json!({
            "run_id": self.run_id,
            "run_seed": self.seed,
            "data_path": self.data_path.display().to_string(),
            "generation": self.generation,
            "pop_size": self.pop.len(),
            "options": &self.options,
            "topology_options": self.options.topology_template(),
            "best_fitness": self.best.as_ref().map(|b| b.fitness),
            "best_topology": best_topology,
            "best_net_facts": best_net_facts,
        });
        serde_json::to_string_pretty(&spec)
            .map_err(|e| EngineError::Json(format!("to_json: {e}")).into())
    }
}

/// Deterministic child-seed derivation: one population base seed chains into
/// every individual's seed, so the whole population is reproducible from
/// `run_seed` alone. Multiply by the golden ratio for spread.
pub(crate) fn derive_seed(base: u64, i: usize) -> u64 {
    base.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flodl::nn::loss::mse_loss;
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
            hidden_dim_pool: 4..=4, // fixed: every individual uses the template dim
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
        let mut engine = Engine::new(test_options(), &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        engine.run().unwrap();

        // Run folder: the improvement history + the final envelope only
        // (options.json / meta.json are deduped into engine.json).
        let imp_dir = engine.run_dir.join("improvements");
        assert!(imp_dir.exists());
        let mut json_files: Vec<_> = std::fs::read_dir(&imp_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|f| f.ends_with(".json"))
            .collect();
        json_files.sort();
        assert!(!json_files.is_empty());
        // Each improvement writes a .json recipe + a .md visualization.
        assert_eq!(json_files.len(), engine.improvements);
        let mut all_files: Vec<_> = std::fs::read_dir(&imp_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        all_files.sort();
        // 2 files per improvement: .json + .md
        assert_eq!(all_files.len(), engine.improvements * 2);
        // Each recipe entry is a valid topology that replicates the best at
        // that point; the latest one matches the final best blueprint.
        let latest_json = std::fs::read_to_string(imp_dir.join(json_files.last().unwrap())).unwrap();
        let latest = Topology::from_json(&latest_json).unwrap();
        let best_topo = engine.best.as_ref().unwrap().topology.clone();
        assert_eq!(
            crate::spec::Spec::from(&latest),
            crate::spec::Spec::from(&best_topo)
        );
        assert_eq!(latest.validate(), Ok(()));
        assert!(engine.run_dir.join("engine.json").exists());
        assert!(!engine.run_dir.join("options.json").exists());
        assert!(!engine.run_dir.join("meta.json").exists());

        // Best fitness is a finite score.
        let fitness = engine.best.as_ref().expect("best must exist").fitness;
        assert!(fitness.is_finite());
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&engine.options.results_dir);
    }

    #[test]
    fn test_engine_to_json_replicates_experiment() {
        let data_dir = temp_data_dir();
        let mut engine = Engine::new(test_options(), &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        engine.run().unwrap();

        let json = engine.to_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["run_id"], engine.run_id);
        assert_eq!(v["pop_size"], 3);
        assert_eq!(v["options"]["pop_size"], 3);
        assert_eq!(v["options"]["topology_options"]["hidden_dim"], 4);
        assert_eq!(v["options"]["fitness_label"], "loss");
        assert_eq!(v["data_path"], data_dir.display().to_string());
        assert!(v["best_fitness"].is_number());
        // The resolved base seed is recorded for re-launchability.
        assert_eq!(v["run_seed"], engine.seed);
        // The topology template + materialized net facts ride along, so the
        // file spells out the full option chain and the best's nutrition.
        assert_eq!(v["topology_options"]["input_dim"], 1);
        assert_eq!(v["topology_options"]["hidden_dim"], 4);
        assert_eq!(v["topology_options"]["seed"], engine.seed);
        // best_net_facts is a nested JSON doc (like best_topology) — parse
        // it and check the materialized-net nutrition label is present.
        let facts: serde_json::Value =
            serde_json::from_str(v["best_net_facts"].as_str().unwrap()).unwrap();
        assert!(facts["num_nodes"].as_u64().unwrap() > 0);
        assert!(facts["param_elements"].as_i64().unwrap() > 0);
        assert!(facts["node_dims"].as_array().unwrap().len() >= 2);
        // best_topology round-trips into a valid Topology
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
        let fitness = Fitness::from_loss(move |pred, y| {
            calls2.fetch_add(1, Ordering::SeqCst);
            // Same math as the built-in Mse, proving the drop-in path.
            mse_loss(pred, y) // Variable — backward + .item() both work
        });
        let opts = test_options();
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        engine.run().unwrap();
        // Each individual calls the fitness: num_batches (scoring) +
        // num_epochs * num_batches (training). With test_options defaults:
        // num_batches=16 on 64-row dataset with batch_size=128 → 1 batch;
        // num_epochs=1 → 1 training + 1 scoring = 2 per individual.
        let actual_batches = 1usize; // 64 rows / 128 batch_size → 1
        // With train/eval split: train=1, eval=1 (fallback to train for tiny datasets)
        let expected_per_individual = opts.training.num_epochs * actual_batches + actual_batches;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            opts.pop_size * opts.num_generations * expected_per_individual,
            "custom fitness must be called for each batch (scoring + training) per individual"
        );
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_auto_detects_input_dim() {
        let data_dir = temp_data_dir();
        // Set input_dim to something different — the engine should
        // override it from the dataset shape.
        let opts = EngineOptions {
            topology_options: TopologyOptions {
                input_dim: 999, // wrong — engine will auto-detect
                ..Default::default()
            },
            ..test_options()
        };
        let engine = Engine::new(opts, &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        assert_eq!(engine.options.topology_options.input_dim, 1); // sine dataset
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[test]
    fn test_fitness_custom_sees_pred_and_target() {
        // Compile-time check that the closure really receives the pieces:
        // a closure that ignores one param would still type-check; this one
        // uses both (prediction and target) to prove the minimal signature
        // is ergonomic — no network, no data plumbing.
        let data_dir = temp_data_dir();
        let fitness = Fitness::from_loss(|pred, y| {
            flodl::l1_loss(pred, y) // MAE as Variable
        });
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
    fn test_engine_batched_evaluation() {
        // num_batches > 0 → each candidate is scored on sampled batches;
        // the custom fitness must be invoked once per batch.
        let data_dir = temp_data_dir();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let fitness = Fitness::from_loss(move |pred, y| {
            calls2.fetch_add(1, Ordering::SeqCst);
            mse_loss(pred, y) // Variable
        });
        let opts = EngineOptions {
            pop_size: 3,
            num_generations: 2,
            num_batches: 3,
            batch_size: 8,
            ..test_options()
        };
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        engine.run().unwrap();
        // With train/eval split: train gets half, eval gets half.
        let train_count = (opts.num_batches / 2).max(1);
        let eval_count = (opts.num_batches - train_count).max(1);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            opts.pop_size * opts.num_generations * (train_count * opts.training.num_epochs + eval_count),
            "fitness must be called for train batches (loss) + eval batches (score) per individual"
        );
        let fitness = engine.best.as_ref().expect("best must exist").fitness;
        assert!(fitness.is_finite());
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_rejects_bad_budget() {
        let data_dir = temp_data_dir();
        // num_batches > 0 with batch_size 0 → rejected.
        let bad = EngineOptions {
            num_batches: 2,
            batch_size: 0,
            ..test_options()
        };
        assert!(Engine::new(bad, &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).is_err());

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[test]
    fn test_engine_budget_serialized() {
        // The evaluation budget must ride along in engine.json for
        // reproducibility.
        let data_dir = temp_data_dir();
        let opts = EngineOptions {
            num_batches: 4,
            batch_size: 8,
            ..test_options()
        };
        let mut engine = Engine::new(opts.clone(), &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        engine.run().unwrap();
        let v: serde_json::Value = serde_json::from_str(&engine.to_json().unwrap()).unwrap();
        assert_eq!(v["options"]["num_batches"], 4);
        assert_eq!(v["options"]["num_batches"], 4);
        assert_eq!(v["options"]["batch_size"], 8);
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_maximize_direction() {
        // Verify direction: Maximize picks the individual with the
        // highest prediction mean; Minimize picks the lowest.  Since
        // different hidden_dim → different architectures → different
        // forward outputs, the scores vary across individuals.
        let data_dir = temp_data_dir();
        let make_scorer = |dir: Direction| {
            Fitness::new(
                move |pred, _target| {
                    let vec = pred.data().to_f32_vec().unwrap();
                    let mean = vec.iter().sum::<f32>() / vec.len() as f32;
                    Ok(mean)
                },
                dir,
                "custom",
            )
        };
        let opts = EngineOptions {
            num_generations: 1,
            num_threads: 2,
            hidden_dim_pool: 4..=8, // vary the width so direction discriminates
            ..test_options()
        };
        // Maximize: best fitness >= all individual fitnesses.
        let mut eng =
            Engine::new(opts.clone(), &data_dir, make_scorer(Direction::Maximize)).unwrap();
        eng.run().unwrap();
        let max_best = eng.best.as_ref().unwrap().fitness;
        // Minimize on the same population picks the smallest value.
        let mut eng =
            Engine::new(opts.clone(), &data_dir, make_scorer(Direction::Minimize)).unwrap();
        eng.run().unwrap();
        let min_best = eng.best.as_ref().unwrap().fitness;
        // The two directions must disagree (or the population is degenerate).
        // At minimum, both runs must produce a valid best.
        assert!(max_best.is_finite(), "maximize best must be finite");
        assert!(min_best.is_finite(), "minimize best must be finite");
        assert_ne!(
            max_best, min_best,
            "maximize and minimize should pick different bests"
        );
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_builder_chains_and_validates() {
        // The flat builder routes each set_* into the right layer.
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
        assert_eq!(opts.topology_options.input_dim, 1);
        assert_eq!(opts.topology_options.hidden_dim, 16);
        assert_eq!(opts.hidden_dim_pool, 8..=32);
        assert_eq!(opts.combine_op_pool, vec![CombineOp::Add, CombineOp::Mean]);
        assert_eq!(opts.num_batches, 4);
        assert_eq!(opts.batch_size, 32);
        assert_eq!(opts.network.dtype, DType::Float32);

        // Validations: empty GP pools, bad budget, zero pop.
        assert!(EngineOptions::builder().set_pop_size(0).build().is_err());
        assert!(
            EngineOptions::builder()
                .set_num_batches(2)
                .set_batch_size(0)
                .build()
                .is_err()
        );
        // Empty pools auto-fill with all built-in ops (no error).
        let opts = EngineOptions::builder().set_combine_op_pool(vec![]).build().unwrap();
        assert_eq!(opts.combine_op_pool.len(), 4); // all CombineOp variants
        let opts = EngineOptions::builder().set_activation_pool(vec![]).build().unwrap();
        assert_eq!(opts.activation_pool.len(), 14); // all Activation variants
        // An empty range (start > end) is rejected too.
        assert!(
            EngineOptions::builder()
                .set_hidden_dim_pool(std::ops::RangeInclusive::new(8, 4))
                .build()
                .is_err()
        );
    }

    #[test]
    fn test_engine_builder_one_shot() {
        // build() + Engine::new in two calls.
        let data_dir = temp_data_dir();
        let opts = EngineOptions::builder()
            .set_pop_size(4)
            .set_num_generations(1)
            .set_seed(Some(7))
            .set_hidden_dim_pool(4..=4)
            .build()
            .unwrap();
        let mut engine = Engine::new(opts, &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        engine.run().unwrap();
        assert!(engine.best.is_some());
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&engine.options.results_dir);
    }

    #[test]
    fn test_engine_gp_sampling_varies_and_reproduces() {
        // With real GP pools, individuals vary along the network axes
        // (hidden dim, combine op, per-node activations) — and the whole
        // population still reproduces from the base seed alone.
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

        let a = Engine::new(make_opts(), &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        let b = Engine::new(make_opts(), &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();

        // 1. The population actually varies: not all nodes share one
        //    hidden dim, one combine op, or one activation profile.
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
        let mut combines: Vec<CombineOp> = Vec::new();
        for g in &a.pop {
            for n in &g.nodes {
                if let Some(op) = n.combine_op {
                    if !combines.contains(&op) {
                        combines.push(op);
                    }
                }
            }
        }
        assert!(combines.len() > 1, "combine ops must vary: {combines:?}");
        let mut acts: Vec<Activation> = Vec::new();
        for g in &a.pop {
            for n in &g.nodes {
                if !acts.contains(&n.activation) {
                    acts.push(n.activation);
                }
            }
        }
        assert!(acts.len() > 1, "activations must vary: {acts:?}");

        // 2. Every hidden node's activation comes from the pool.
        for g in &a.pop {
            for n in &g.nodes {
                if n.kind == crate::node::NodeKind::Hidden {
                    assert!(pool.contains(&n.activation), "activation {n:?} not in pool");
                }
            }
        }

        // 3. Same seed → identical population, blueprint for blueprint.
        for (ga, gb) in a.pop.iter().zip(b.pop.iter()) {
            assert_eq!(ga.options.hidden_dim, gb.options.hidden_dim);
            assert_eq!(
                crate::spec::Spec::from(ga),
                crate::spec::Spec::from(gb),
                "same seed must reproduce the individual"
            );
        }
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&a.options.results_dir);
        let _ = std::fs::remove_dir_all(&b.options.results_dir);
    }

    #[test]
    fn test_engine_new_leaves_no_folder() {
        // Construction is side-effect-free on disk: the checkpoint folder
        // only appears when run() is actually called.
        let data_dir = temp_data_dir();
        let opts = test_options();
        let engine = Engine::new(opts.clone(), &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        assert!(
            !engine.run_dir.exists(),
            "Engine::new must not create the run folder — only run() does"
        );
        let mut engine = engine;
        engine.run().unwrap();
        assert!(engine.run_dir.exists());
        assert!(engine.run_dir.join("engine.json").exists());
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_random_seed_recorded() {
        // seed: None → a fresh random seed is derived per run and recorded
        // in engine.json (and baked into the topology template), so the run
        // is re-launchable despite starting from entropy.
        let data_dir = temp_data_dir();
        let opts = EngineOptions {
            seed: None,
            num_threads: 4,
            ..test_options()
        };
        let mut engine = Engine::new(opts.clone(), &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        engine.run().unwrap();
        let v: serde_json::Value = serde_json::from_str(&engine.to_json().unwrap()).unwrap();
        assert_eq!(v["run_seed"], engine.seed);
        assert_eq!(v["topology_options"]["seed"], engine.seed);
        assert_eq!(v["options"]["seed"], serde_json::Value::Null);
        // And a second randomized launch derives a different base seed.
        let other = Engine::new(opts.clone(), &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        assert_ne!(other.seed, engine.seed);
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_seeded_run_is_reproducible() {
        // Same options ⇒ the WHOLE run reproduces: same population, same
        // deterministic weight init, same scores, same best blueprint. This
        // is the payoff of seeding the network options like the topology.
        let data_dir = temp_data_dir();
        let make = || EngineOptions {
            seed: Some(123),
            num_threads: 3, // determinism holds even in parallel (local RNGs)
            ..test_options()
        };
        let mut a = Engine::new(make(), &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        let mut b = Engine::new(make(), &data_dir, Fitness::from_loss(|p, y| flodl::nn::loss::mse_loss(p, y))).unwrap();
        a.run().unwrap();
        b.run().unwrap();
        let ba = a.best.as_ref().unwrap();
        let bb = b.best.as_ref().unwrap();
        assert_eq!(
            ba.fitness, bb.fitness,
            "same options ⇒ same best fitness (weights must be deterministic)"
        );
        assert_eq!(
            crate::spec::Spec::from(&ba.topology),
            crate::spec::Spec::from(&bb.topology),
            "same options ⇒ same best blueprint"
        );
        // The base init seed is baked into the network options like the
        // topology seed — and recorded in engine.json.
        assert_eq!(a.options.network.seed, 123);
        let v: serde_json::Value = serde_json::from_str(&a.to_json().unwrap()).unwrap();
        assert_eq!(v["options"]["network"]["seed"], 123);
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&a.options.results_dir);
        let _ = std::fs::remove_dir_all(&b.options.results_dir);
    }
}
