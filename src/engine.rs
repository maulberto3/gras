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
//! [`crate::data::save_dataset`] — flodl-native data only. The dataset is
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
use std::time::{SystemTime, UNIX_EPOCH};

use flodl::nn::Module;
use flodl::tensor::Result;
use flodl::{DType, Device, Tensor, Variable};
use rayon::prelude::*;
use serde::Serialize;

use crate::data::Dataset;
use crate::error::EngineError;
pub use crate::fitness::{Direction, Fitness, FitnessKind};
use crate::network::{Network, NetworkOptions};
use crate::node::{Activation, NodeKind};
use crate::topology::{CombineOp, Topology, TopologyOptions};

/// Knobs for one engine run. Serialized by [`Engine::to_json`] into the run
/// folder's `engine.json`, so an experiment is fully reproducible.
#[derive(Clone, Debug, PartialEq, Serialize)]
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
    pub topology: TopologyOptions,
    /// **GP search space** — the hidden-dim range sampled **per individual**
    /// at population creation (`8..=8` = every individual uses the
    /// template's `hidden_dim`, the pre-GP behavior).
    pub hidden_dim_pool: RangeInclusive<usize>,
    /// **GP search space** — the combine-op pool sampled **per individual**
    /// at population creation (default `[Add]` = the template's
    /// `combine_op`).
    pub combine_op_pool: Vec<CombineOp>,
    /// **GP search space** — the activation pool. Used twice: per-node
    /// random activations at population creation
    /// ([`Topology::randomize_activations`]) and, later, as the swap source
    /// for the future `mutate` implementation.
    pub activation_pool: Vec<Activation>,
    /// **Evaluation budget** — how much of the data each candidate is scored
    /// on, so evolution doesn't need the whole dataset every generation
    /// (matters as data grows toward LLM-scale). `num_batches == 0` means
    /// one full pass over the dataset, chunked into `batch_size` slices
    /// (memory-bounded; the original "whole dataset" semantics); otherwise
    /// each epoch samples `num_batches` random batches of `batch_size` rows
    /// and the score is the mean over `num_epochs × num_batches` scores.
    ///
    /// The sampled batches are the **same for every individual of a
    /// generation** (seeded from `run_seed + generation`), so scores stay
    /// comparable, and deterministic across runs with the same options.
    pub num_epochs: usize,
    /// Random batches sampled per epoch when scoring a candidate
    /// (`0` = whole dataset once, the default).
    pub num_batches: usize,
    /// Rows per batch (used both for sampled batches and for chunking the
    /// whole-dataset pass; default 128).
    pub batch_size: usize,
    /// Built-in scoring strategy recorded for reproducibility.
    pub fitness: FitnessKind,
    /// Threads for parallel population evaluation (`0` = rayon's default,
    /// i.e. available parallelism; default 3 to stay conservative on shared
    /// machines).
    pub num_threads: usize,
    /// Parent folder for per-run checkpoint folders (`results/<ts>/`).
    pub results_dir: PathBuf,
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
            topology: TopologyOptions::default(),
            hidden_dim_pool: 8..=8,
            combine_op_pool: vec![CombineOp::Add],
            activation_pool: vec![Activation::Identity, Activation::ReLU, Activation::GeLU],
            num_epochs: 1,
            num_batches: 0, // whole dataset once (chunked), original behavior
            batch_size: 128,
            fitness: FitnessKind::Mse,
            num_threads: 3,
            results_dir: PathBuf::from("results"),
            network: NetworkOptions::default(),
        }
    }
}

impl EngineOptions {
    /// The topology knobs for one individual: the shared template with the
    /// individual's derived seed. Each population slot gets its own seed via
    /// the deterministic chain, so the whole population is reproducible from
    /// the base seed alone.
    fn individual_options(&self, seed: usize) -> TopologyOptions {
        let mut t = self.topology;
        t.seed = seed;
        t
    }

    /// The shared topology template (base seed baked in), serialized into
    /// `engine.json` so the file spells out the full option chain
    /// (engine → topology → network).
    fn topology_template(&self) -> TopologyOptions {
        self.topology
    }

    /// Start a fluent builder chain — the ergonomic way to configure a run
    /// without a big nested struct literal:
    ///
    /// ```
    /// use gras::engine::{EngineOptions, Fitness};
    /// use gras::node::Activation;
    ///
    /// let opts = EngineOptions::builder()
    ///     .set_pop_size(15)
    ///     .set_num_generations(3)
    ///     .set_hidden_dim_pool(8..=16)
    ///     .set_activation_pool(vec![Activation::ReLU, Activation::GeLU])
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// The `set_*` methods are **flat**: each routes into the right layer
    /// (engine knobs, topology template, GP pools, network options), so
    /// callers never touch the nested structs unless they want to.
    pub fn builder() -> EngineOptionsBuilder {
        EngineOptionsBuilder {
            inner: EngineOptions::default(),
        }
    }
}

/// Fluent builder for [`EngineOptions`] — a flat `set_*` chain over the
/// nested option structs. Start with [`EngineOptions::builder`], finish with
/// [`build`](EngineOptionsBuilder::build) (validated options) or
/// [`build_engine`](EngineOptionsBuilder::build_engine) (validated options +
/// a ready [`Engine`]).
pub struct EngineOptionsBuilder {
    inner: EngineOptions,
}

impl EngineOptionsBuilder {
    /// Validate the accumulated options and hand them back. Checks the same
    /// invariants `Engine::new` relies on: non-empty population, valid
    /// evaluation budget, and non-empty GP pools.
    pub fn build(self) -> Result<EngineOptions> {
        let o = &self.inner;
        if o.pop_size == 0 {
            return Err(EngineError::InvalidOptions("pop_size must be > 0".into()).into());
        }
        if o.num_batches > 0 && (o.batch_size == 0 || o.num_epochs == 0) {
            return Err(EngineError::InvalidOptions(
                "num_batches > 0 requires batch_size > 0 and num_epochs > 0".to_string(),
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
            return Err(EngineError::InvalidOptions(
                "combine_op_pool must contain at least one op".to_string(),
            )
            .into());
        }
        Ok(self.inner)
    }

    /// One-shot: validate the options and start an experiment from them
    /// (equivalent to `Engine::new(builder.build()?, data_path, fitness)`).
    pub fn build_engine(self, data_path: &Path, fitness: Fitness) -> Result<Engine> {
        Engine::new(self.build()?, data_path, fitness)
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
    pub fn set_num_epochs(mut self, n: usize) -> Self {
        self.inner.num_epochs = n;
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
    /// Set the whole evaluation budget at once: `num_epochs × num_batches`
    /// batches of `batch_size` rows (0 batches = whole dataset once).
    pub fn set_budget(mut self, num_epochs: usize, num_batches: usize, batch_size: usize) -> Self {
        self.inner.num_epochs = num_epochs;
        self.inner.num_batches = num_batches;
        self.inner.batch_size = batch_size;
        self
    }
    pub fn set_fitness(mut self, kind: FitnessKind) -> Self {
        self.inner.fitness = kind;
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

    // ── topology template (the blueprint's structure knobs) ────────────────
    pub fn set_topology(mut self, t: TopologyOptions) -> Self {
        self.inner.topology = t;
        self
    }
    pub fn set_input_dim(mut self, n: usize) -> Self {
        self.inner.topology.input_dim = n;
        self
    }
    pub fn set_hidden_dim(mut self, n: usize) -> Self {
        self.inner.topology.hidden_dim = n;
        self
    }
    pub fn set_combine_op(mut self, op: CombineOp) -> Self {
        self.inner.topology.combine_op = op;
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
    /// Deterministic weight init: `Some(seed)` → every network built from
    /// the same options gets the exact same weights (same options ⇒ same
    /// built model); `None` (default) → flodl's RNG, fresh weights per build.
    pub fn set_init_seed(mut self, seed: Option<u64>) -> Self {
        self.inner.network.seed = seed;
        self
    }
}

/// The best individual seen so far.
#[derive(Clone, Debug)]
pub struct BestIndividual {
    pub fitness: f64,
    /// The blueprint that scored best — `to_json` it to replicate the net.
    pub topology: Topology,
}

/// A running NAS experiment. Build it with [`Engine::new`], run it with
/// [`Engine::run`].
pub struct Engine {
    pub options: EngineOptions,
    /// The resolved population base seed — `options.seed` when given,
    /// otherwise a fresh random seed derived at construction. Recorded in
    /// `engine.json` so even a randomized run is re-launchable.
    pub run_seed: u64,
    /// Unix timestamp identifying this run (also the checkpoint folder name).
    pub run_id: String,
    /// `results/<run_id>/` — the checkpoint path for this run. Fixed at
    /// construction; created on disk (with `engine.json` + `log.txt` +
    /// `improvements/`) by [`Engine::run`].
    pub run_dir: PathBuf,
    /// Thread pool for parallel population evaluation (`options.num_threads`).
    pool: rayon::ThreadPool,
    /// The population of blueprints (evolved in place by future genetics).
    pub pop: Vec<Topology>,
    /// The scorer (built-in or user closure).
    pub fitness: Fitness,
    /// The dataset loaded from the user-provided path.
    pub data: Dataset,
    /// The path the dataset was loaded from (recorded for reproducibility).
    pub data_path: PathBuf,
    /// Current generation (0-based; incremented by `next_generation`).
    pub generation: usize,
    /// Current best individual (updated whenever a run improves it).
    pub best: Option<BestIndividual>,
    /// How many best-improvements have been recorded into `improvements/`
    /// (also the next filename counter).
    pub improvements: usize,
    /// Last generation's scores (for compact logging).
    scores: Vec<f64>,
}

impl Engine {
    /// Start an experiment: load the dataset from `data_path` and seed a
    /// population of `options.pop_size` random graphs — each individual
    /// derives its seed from the base via a deterministic chain (see
    /// [`Engine::run_seed`]), so the whole population is reproducible from
    /// that seed alone. The checkpoint path `results/<ts>/` is fixed here,
    /// but the folder is only created on disk by [`Engine::run`].
    ///
    /// Fails if the dataset's input dim doesn't match `options.input_dim`.
    pub fn new(mut options: EngineOptions, data_path: &Path, fitness: Fitness) -> Result<Self> {
        if options.pop_size == 0 {
            return Err(EngineError::InvalidOptions("pop_size must be > 0".into()).into());
        }
        if options.num_batches > 0 && (options.batch_size == 0 || options.num_epochs == 0) {
            return Err(EngineError::InvalidOptions(
                "num_batches > 0 requires batch_size > 0 and num_epochs > 0".to_string(),
            )
            .into());
        }
        if options.hidden_dim_pool.is_empty() {
            return Err(EngineError::InvalidOptions(
                "hidden_dim_pool must be a non-empty range (start <= end)".to_string(),
            )
            .into());
        }
        if options.combine_op_pool.is_empty() {
            return Err(EngineError::InvalidOptions(
                "combine_op_pool must contain at least one op".to_string(),
            )
            .into());
        }
        if options.batch_size == 0 && options.num_batches == 0 {
            // The whole-dataset chunker needs a positive slice size.
            options.batch_size = 1;
        }

        // 🎲 Resolve the population base seed: the user's, or a fresh random
        // one (entropy) recorded in engine.json so the run is re-launchable.
        let run_seed = options.seed.unwrap_or_else(|| {
            let t = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            t ^ fastrand::u64(..)
        });
        // Bake the base seed into the topology template (blueprint) AND the
        // network options (weight init), so engine.json spells out the exact
        // chain base. Per individual, the build overrides the init seed with
        // that individual's derived seed — same options ⇒ same built model.
        options.topology.seed = run_seed as usize;
        options.network.seed = Some(run_seed);

        // Data contract: tensors on disk, loaded once, normalized to f32
        // (this crate's default precision).
        let data = crate::data::load_dataset(data_path)?.to_f32()?;
        let data_in = data.inputs.shape().get(1).copied().ok_or_else(|| {
            EngineError::DataMismatch("dataset inputs must be 2-D [n, input_dim]".into())
        })?;
        if data_in != options.topology.input_dim as i64 {
            return Err(EngineError::DataMismatch(format!(
                "dataset input_dim is {data_in} but options.topology.input_dim is {}",
                options.topology.input_dim
            ))
            .into());
        }

        // Per-run checkpoint folder: results/<unix ts>/
        // Milliseconds so two runs started within the same second still get
        // distinct folders (a seconds-resolution id would make them collide).
        let run_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
            .to_string();
        // The checkpoint **path** is fixed here; the folder itself is created
        // on disk by [`Engine::run`] — constructing an engine never leaves
        // artifacts behind.
        let run_dir = options.results_dir.join(&run_id);

        // Thread pool for parallel evaluation (num_threads == 0 → auto).
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

        // Population: random graphs through the standard pipeline (scaffold
        // + wire + auto-de-orphan). Every individual derives its seed from
        // the base via a deterministic chain (`derive_seed`), so the whole
        // population is reproducible from `run_seed` alone — one seed seeds
        // child seeds.
        let mut pop = Vec::with_capacity(options.pop_size);
        for i in 0..options.pop_size {
            let ind_seed = derive_seed(run_seed, i);
            let mut rng = fastrand::Rng::with_seed(ind_seed);
            let n_hidden = rng.usize(2..=6);
            let mut ind_opts = options.individual_options(ind_seed as usize);
            // 🎛️ GP: sample the per-individual architecture values from the
            // pools — hidden dim, combine op, then per-node activations. All
            // draws come from the derived seed chain, so the whole population
            // (dims included) reproduces from `run_seed` alone.
            ind_opts.hidden_dim = rng.usize(options.hidden_dim_pool.clone());
            ind_opts.combine_op =
                options.combine_op_pool[rng.usize(0..options.combine_op_pool.len())];
            let mut graph = Topology::new(i, Some(ind_opts));
            graph.create_random_hidden_nodes(n_hidden);
            graph.randomize_activations(&options.activation_pool, &mut rng);
            // 🎛️ GP per-node values: every hidden node gets its OWN combine
            // op from the pool (combining never changes tensor width, so
            // fan-in stays consistent). Activations are per-node too.
            // Per-node hidden DIMS would break fan-in — a node merging
            // sources of different widths is an invalid graph (the proper
            // fix is per-source projections in `Network::build`, a future
            // feature) — so dims stay per-individual for now.
            for node in &mut graph.nodes {
                if node.kind == NodeKind::Hidden {
                    node.combine_op =
                        Some(options.combine_op_pool[rng.usize(0..options.combine_op_pool.len())]);
                }
            }
            graph.refresh_labels();
            graph.finalize();
            pop.push(graph);
        }

        let engine = Engine {
            options,
            run_seed,
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
        };
        Ok(engine)
    }

    /// The last generation's fitness scores, index-aligned with `pop`
    /// (`scores[i]` belongs to `pop[i]`). Empty until the first
    /// [`Engine::run`] evaluates the population.
    pub fn scores(&self) -> &[f64] {
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
    pub fn run(&mut self) -> Result<()> {
        fs::create_dir_all(&self.run_dir).map_err(|source| EngineError::Io {
            path: self.run_dir.display().to_string(),
            source,
        })?;
        // Initial experiment envelope (no best yet); the final one is
        // written after the loop.
        let initial = self.to_json()?;
        fs::write(self.run_dir.join("engine.json"), initial).map_err(|source| EngineError::Io {
            path: self.run_dir.join("engine.json").display().to_string(),
            source,
        })?;
        // TODO(engine): stop criteria beyond max generations — e.g. a
        // StopCriterion enum with TargetFitness (stop once the best crosses
        // a threshold) and NoImprovement (stop after N stagnant generations).
        for _ in 0..self.options.num_generations {
            let improved = self.evaluate_population()?;
            self.log_generation(improved)?;
            self.next_generation();
        }
        let json = self.to_json()?;
        fs::write(self.run_dir.join("engine.json"), json).map_err(|source| EngineError::Io {
            path: self.run_dir.join("engine.json").display().to_string(),
            source,
        })?;
        Ok(())
    }

    /// Score every individual on the evaluation budget. Returns whether the
    /// overall best improved (and appends it to `improvements/` if so).
    fn evaluate_population(&mut self) -> Result<bool> {
        // One budget per generation: the sampled batches are reused for every
        // individual so scores are comparable (same data, same compute), and
        // re-seeded from `run_seed + generation` so each generation sees
        // fresh data deterministically. `num_batches == 0` → one full pass
        // over the dataset, chunked into `batch_size` slices (memory-bounded
        // — never one giant forward).
        let batches = if self.options.num_batches > 0 {
            let mut rng =
                fastrand::Rng::with_seed(self.run_seed.wrapping_add(self.generation as u64));
            self.sample_batches(&mut rng)?
        } else {
            self.whole_dataset_chunks()?
        };

        // ⚡ Parallel evaluation: one rayon task per individual. The batches
        // are plain `Tensor` pairs (Tensor is Send + Sync); each task wraps
        // its own copies in fresh `Variable`s, because flodl's Variable is
        // Rc-based and can't cross threads.
        let direction = self.fitness.direction();
        let net_opts = self.options.network;
        let scores: Vec<f64> = self.pool.install(|| {
            self.pop
                .par_iter()
                .map(|graph| {
                    // 🎲 Deterministic weights: the individual's derived seed
                    // drives its weight init, so a blueprint always scores
                    // identically (same run_seed ⇒ same scores ⇒ same run).
                    let mut no = net_opts;
                    no.seed = Some(graph.options.seed as u64);
                    let net = Network::build_with_options(graph, &no)?;
                    let mut total = 0.0;
                    for (xb, yb) in &batches {
                        let x = Variable::new(xb.clone(), false);
                        let y = Variable::new(yb.clone(), false);
                        // The engine owns the forward pass; a fitness only
                        // sees prediction vs target (its minimal inputs).
                        let pred = net.forward(&x)?;
                        total += self.fitness.evaluate(&pred, &y)?;
                    }
                    Ok(total / batches.len() as f64)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        self.scores = scores;

        let mut best_idx: Option<(usize, f64)> = None;
        for (i, s) in self.scores.iter().enumerate() {
            if best_idx
                .map(|(_, c)| direction.is_better(*s, c))
                .unwrap_or(true)
            {
                best_idx = Some((i, *s));
            }
        }

        if let Some((i, score)) = best_idx {
            let improved = self
                .best
                .as_ref()
                .map(|b| direction.is_better(score, b.fitness))
                .unwrap_or(true);
            if improved {
                self.best = Some(BestIndividual {
                    fitness: score,
                    topology: self.pop[i].clone(),
                });
                self.record_improvement()?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Sample the per-generation evaluation budget: `num_epochs ×
    /// num_batches` batches of `batch_size` random rows (with replacement)
    /// from the loaded dataset, returned as raw `(Tensor, Tensor)` pairs.
    /// The caller seeds `rng` from `run_seed + generation`, so a run
    /// reproduces the same batches.
    fn sample_batches(&self, rng: &mut fastrand::Rng) -> Result<Vec<(Tensor, Tensor)>> {
        let n = self.data.inputs.shape()[0] as usize;
        let total = self.options.num_epochs * self.options.num_batches;
        let mut batches = Vec::with_capacity(total);
        for _ in 0..total {
            let idx: Vec<i64> = (0..self.options.batch_size)
                .map(|_| rng.usize(0..n) as i64)
                .collect();
            let idx_t = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;
            let xb = self.data.inputs.index_select(0, &idx_t)?;
            let yb = self.data.targets.index_select(0, &idx_t)?;
            batches.push((xb, yb));
        }
        Ok(batches)
    }

    /// The "whole dataset once" pass, split into `batch_size`-row slices so
    /// memory stays bounded no matter how large the dataset grows. The score
    /// is the mean over the chunk scores — same semantics as a single full
    /// forward, without the single giant tensor.
    fn whole_dataset_chunks(&self) -> Result<Vec<(Tensor, Tensor)>> {
        let n = self.data.inputs.shape()[0] as usize;
        let bs = self.options.batch_size.max(1);
        let mut chunks = Vec::with_capacity(n.div_ceil(bs));
        let mut start = 0usize;
        while start < n {
            let end = (start + bs).min(n);
            let idx: Vec<i64> = (start as i64..end as i64).collect();
            let idx_t = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;
            let xb = self.data.inputs.index_select(0, &idx_t)?;
            let yb = self.data.targets.index_select(0, &idx_t)?;
            chunks.push((xb, yb));
            start = end;
        }
        Ok(chunks)
    }

    /// Append the current best to `run_dir/improvements/` — the evolution
    /// trail. Each best-improvement writes a **pair**: the topology **recipe**
    /// (`Topology::from_json` + `Network::build` replicates the net) and the
    /// **nutrition facts** of the materialized net (`Network::to_json` — dims,
    /// wiring stats, real param counts), so the trail reads top-down and the
    /// latest entry is the current best. Files:
    ///   `{counter:04}_gen{gen:02}_fitness{fit:.4}.json`          (recipe)
    ///   `{counter:04}_gen{gen:02}_fitness{fit:.4}.net.json`      (facts)
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
        // Nutrition: build the best once and dump its facts (same shared
        // diagnostics the blueprint uses — no recomputation drift). Built
        // with the best's own init seed so the facts match its weights.
        let mut no = self.options.network;
        no.seed = Some(b.topology.options.seed as u64);
        let net = Network::build_with_options(&b.topology, &no)?;
        let net_json = net
            .to_json()
            .map_err(|e| EngineError::Json(format!("improvement net facts: {e}")))?;
        let name = format!(
            "{:04}_gen{:02}_fitness{:.4}.json",
            self.improvements, self.generation, b.fitness
        );
        let path = dir.join(&name);
        fs::write(&path, json).map_err(|source| EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let net_path = dir.join(format!("{}.net.json", name.trim_end_matches(".json")));
        fs::write(&net_path, net_json).map_err(|source| EngineError::Io {
            path: net_path.display().to_string(),
            source,
        })?;
        self.improvements += 1;
        Ok(())
    }

    /// One compact log line per generation, mirrored into `run_dir/log.txt`.
    /// Emitted through the `log` crate (`log::info!`), so callers control the
    /// sink (examples init a logger; library users plug in their own).
    fn log_generation(&self, improved: bool) -> Result<()> {
        let scores = &self.scores;
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // "best" is direction-aware: the min for losses, the max for metrics.
        let dir = self.fitness.direction();
        let (best_s, worst_s) = match dir {
            Direction::Minimize => (min, max),
            Direction::Maximize => (max, min),
        };
        let flag = if improved { " 🏆 best improved" } else { "" };
        let line = format!(
            "gen {:02} · pop {} · best {best_s:.4}{} · mean {mean:.4} · worst {worst_s:.4}{flag}",
            self.generation,
            self.pop.len(),
            dir.arrow()
        );
        log::info!("{line}");

        // Append-only (never re-read the file each generation).
        let path = self.run_dir.join("log.txt");
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| EngineError::Io {
                path: path.display().to_string(),
                source,
            })?;
        std::io::Write::write_all(&mut f, line.as_bytes())
            .and_then(|_| std::io::Write::write_all(&mut f, b"\n"))
            .map_err(|source| EngineError::Io {
                path: path.display().to_string(),
                source,
            })?;
        Ok(())
    }

    /// Advance one generation: crossover → mutate → generation += 1.
    ///
    /// Crossover and mutation are still documented no-op stubs.
    /// [`crate::genetics::select`] (elitism + tournament) is implemented and
    /// tested in `src/genetics.rs` but **not yet wired into the loop** —
    /// pending design review. Wiring it back is one line:
    /// `self.pop = genetics::select(&self.scores, dir, &mut rng, 3).into_iter().map(|i| self.pop[i].clone()).collect();`
    fn next_generation(&mut self) {
        self.crossover();
        self.mutate();
        self.generation += 1;
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

    /// 🎲 Mutation placeholder — the second genetic operator (planned).
    ///
    /// Future: randomly tweak individuals — rewire, add/remove nodes, swap
    /// activations from `options.activation_pool`, swap combine ops, adjust the
    /// individual's `hidden_dim`. (Per-node `hidden_dim` stays out: a node
    /// merging sources of different widths is an invalid graph — see the
    /// fan-in note in `Engine::new`.) **Not implemented yet** — a no-op, so
    /// the population stays static.
    pub fn mutate(&mut self) {
        // TODO(engine): topology-level mutation.
    }

    /// Everything needed to replicate this experiment, as JSON: the resolved
    /// `run_seed`, the options (including the topology template every
    /// individual derives from), the data path, the best fitness, the **best
    /// topology** recipe (feed it to `Topology::from_json` +
    /// `Network::build` to recreate the best net) and the **best network
    /// facts** (`Network::to_json` — what that build produced, so the file
    /// is self-describing without running code).
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
                no.seed = Some(b.topology.options.seed as u64);
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
            "run_seed": self.run_seed,
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
/// every individual's seed (and their wiring/n_hidden draws), so the whole
/// population is reproducible from `run_seed` alone. SplitMix64 finalizer
/// over `base` mixed with `i` — no extra crate, stable across runs.
fn derive_seed(base: u64, i: usize) -> u64 {
    let mut z = base.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flodl::nn::Module;
    use flodl::nn::loss::mse_loss;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn temp_data_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gras_engine_test_{}", fastrand::u64(..)));
        let ds = crate::fitness::synthetic_sine(64, 42, Device::CPU).unwrap();
        crate::data::save_dataset(&dir, &ds).unwrap();
        dir
    }

    fn test_options() -> EngineOptions {
        EngineOptions {
            pop_size: 3,
            num_generations: 2,
            topology: TopologyOptions {
                hidden_dim: 4,
                ..Default::default()
            },
            hidden_dim_pool: 4..=4, // fixed: every individual uses the template dim
            results_dir: std::env::temp_dir()
                .join(format!("gras_engine_res_{}", fastrand::u64(..))),
            ..Default::default()
        }
    }

    #[test]
    fn test_engine_runs_and_checkpoints() {
        let data_dir = temp_data_dir();
        let mut engine = Engine::new(test_options(), &data_dir, Fitness::mse()).unwrap();
        engine.run().unwrap();

        // Run folder: the improvement history + the final envelope only
        // (options.json / meta.json are deduped into engine.json).
        let imp_dir = engine.run_dir.join("improvements");
        assert!(imp_dir.exists());
        let mut files: Vec<_> = std::fs::read_dir(&imp_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        files.sort();
        assert!(!files.is_empty());
        // Each improvement writes a pair: the topology recipe + the
        // materialized-net facts (`.net.json`).
        let recipes: Vec<_> = files
            .iter()
            .filter(|f| !f.ends_with(".net.json"))
            .cloned()
            .collect();
        assert_eq!(recipes.len(), engine.improvements);
        assert_eq!(files.len(), 2 * engine.improvements);
        // Each recipe entry is a valid topology that replicates the best at
        // that point; the latest one matches the final best blueprint.
        let latest_json = std::fs::read_to_string(imp_dir.join(recipes.last().unwrap())).unwrap();
        let latest = Topology::from_json(&latest_json).unwrap();
        let best_topo = engine.best.as_ref().unwrap().topology.clone();
        assert_eq!(
            crate::spec::Spec::from(&latest),
            crate::spec::Spec::from(&best_topo)
        );
        assert_eq!(latest.validate(), Ok(()));
        // The latest facts file describes the same built net: same node
        // count, same real param elements as a live build.
        let net_json = std::fs::read_to_string(imp_dir.join(format!(
            "{}.net.json",
            recipes.last().unwrap().trim_end_matches(".json")
        )))
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&net_json).unwrap();
        let built = Network::build(&best_topo, engine.options.network.device).unwrap();
        assert_eq!(v["num_nodes"], built.nodes.len() as u64);
        assert_eq!(v["num_wires"], built.connections.len() as u64);
        assert_eq!(
            v["param_elements"].as_i64().unwrap(),
            built
                .parameters()
                .iter()
                .map(|p| p.variable.numel())
                .sum::<i64>()
        );
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
        let mut engine = Engine::new(test_options(), &data_dir, Fitness::mse()).unwrap();
        engine.run().unwrap();

        let json = engine.to_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["run_id"], engine.run_id);
        assert_eq!(v["pop_size"], 3);
        assert_eq!(v["options"]["pop_size"], 3);
        assert_eq!(v["options"]["topology"]["hidden_dim"], 4);
        assert_eq!(v["options"]["fitness"], "Mse");
        assert_eq!(v["data_path"], data_dir.display().to_string());
        assert!(v["best_fitness"].is_number());
        // The resolved base seed is recorded for re-launchability.
        assert_eq!(v["run_seed"], engine.run_seed);
        // The topology template + materialized net facts ride along, so the
        // file spells out the full option chain and the best's nutrition.
        assert_eq!(v["topology_options"]["input_dim"], 1);
        assert_eq!(v["topology_options"]["hidden_dim"], 4);
        assert_eq!(v["topology_options"]["seed"], engine.run_seed);
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
        let fitness = Fitness::custom(move |pred, y| {
            calls2.fetch_add(1, Ordering::SeqCst);
            // Same math as the built-in Mse, proving the drop-in path.
            mse_loss(pred, y)?.item()
        });
        let opts = test_options();
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        engine.run().unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            opts.pop_size * opts.num_generations,
            "custom fitness must run once per individual per generation"
        );
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_rejects_input_dim_mismatch() {
        let data_dir = temp_data_dir();
        let opts = EngineOptions {
            topology: TopologyOptions {
                input_dim: 2, // dataset has input_dim 1
                ..Default::default()
            },
            ..test_options()
        };
        assert!(Engine::new(opts, &data_dir, Fitness::mse()).is_err());
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[test]
    fn test_fitness_custom_sees_pred_and_target() {
        // Compile-time check that the closure really receives the pieces:
        // a closure that ignores one param would still type-check; this one
        // uses both (prediction and target) to prove the minimal signature
        // is ergonomic — no network, no data plumbing.
        let data_dir = temp_data_dir();
        let fitness = Fitness::custom(|pred, y| {
            let diff = pred.data().sub(&y.data())?;
            diff.abs()?.mean()?.item() // MAE
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
        let fitness = Fitness::custom(move |pred, y| {
            calls2.fetch_add(1, Ordering::SeqCst);
            mse_loss(pred, y)?.item()
        });
        let opts = EngineOptions {
            pop_size: 3,
            num_generations: 2,
            num_epochs: 2,
            num_batches: 3,
            batch_size: 8,
            ..test_options()
        };
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        engine.run().unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            opts.pop_size * opts.num_generations * (opts.num_epochs * opts.num_batches),
            "batched evaluation must score one batch at a time"
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
        assert!(Engine::new(bad, &data_dir, Fitness::mse()).is_err());
        // num_batches > 0 with num_epochs 0 → rejected.
        let bad = EngineOptions {
            num_batches: 2,
            num_epochs: 0,
            ..test_options()
        };
        assert!(Engine::new(bad, &data_dir, Fitness::mse()).is_err());
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[test]
    fn test_engine_budget_serialized() {
        // The evaluation budget must ride along in engine.json for
        // reproducibility.
        let data_dir = temp_data_dir();
        let opts = EngineOptions {
            num_epochs: 2,
            num_batches: 4,
            batch_size: 8,
            ..test_options()
        };
        let mut engine = Engine::new(opts.clone(), &data_dir, Fitness::mse()).unwrap();
        engine.run().unwrap();
        let v: serde_json::Value = serde_json::from_str(&engine.to_json().unwrap()).unwrap();
        assert_eq!(v["options"]["num_epochs"], 2);
        assert_eq!(v["options"]["num_batches"], 4);
        assert_eq!(v["options"]["batch_size"], 8);
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&opts.results_dir);
    }

    #[test]
    fn test_engine_maximize_direction() {
        // A higher-is-better scorer: the prediction tensor's element count
        // (shape-only — no data needed). The best after one generation must
        // be the individual with the widest output tensor, proving the
        // engine compares by direction, not min-only.
        let data_dir = temp_data_dir();
        let fitness = Fitness::custom_directed(
            |pred, _target| Ok(pred.data().numel() as f64),
            Direction::Maximize,
        );
        let opts = EngineOptions {
            num_generations: 1,
            num_threads: 2,
            hidden_dim_pool: 4..=8, // vary the width so direction discriminates
            ..test_options()
        };
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        // Output width = the individual's hidden_dim; rows = first chunk.
        let rows = engine.data.inputs.shape()[0].min(opts.batch_size.max(1) as i64) as f64;
        let expected = engine
            .pop
            .iter()
            .map(|g| rows * g.options.hidden_dim as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        engine.run().unwrap();
        let best_fitness = engine.best.as_ref().unwrap().fitness;
        assert!(
            (best_fitness - expected).abs() < 1e-9,
            "maximize best = {best_fitness}, expected {expected}"
        );
        // And a Minimize scorer on the same population picks the minimum.
        let fitness = Fitness::custom_directed(
            |pred, _target| Ok(pred.data().numel() as f64),
            Direction::Minimize,
        );
        let mut engine = Engine::new(opts.clone(), &data_dir, fitness).unwrap();
        let expected = engine
            .pop
            .iter()
            .map(|g| rows * g.options.hidden_dim as f64)
            .fold(f64::INFINITY, f64::min);
        engine.run().unwrap();
        let best_fitness = engine.best.as_ref().unwrap().fitness;
        assert!(
            (best_fitness - expected).abs() < 1e-9,
            "minimize best = {best_fitness}, expected {expected}"
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
            .set_input_dim(1)
            .set_hidden_dim(16)
            .set_hidden_dim_pool(8..=32)
            .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
            .set_activation_pool(vec![Activation::ReLU, Activation::GeLU])
            .set_budget(1, 4, 32)
            .set_num_threads(2)
            .set_dtype(DType::Float32)
            .build()
            .unwrap();
        assert_eq!(opts.pop_size, 15);
        assert_eq!(opts.num_generations, 3);
        assert_eq!(opts.seed, Some(42));
        assert_eq!(opts.topology.input_dim, 1);
        assert_eq!(opts.topology.hidden_dim, 16);
        assert_eq!(opts.hidden_dim_pool, 8..=32);
        assert_eq!(opts.combine_op_pool, vec![CombineOp::Add, CombineOp::Mean]);
        assert_eq!(opts.num_epochs, 1);
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
        assert!(
            EngineOptions::builder()
                .set_num_batches(2)
                .set_num_epochs(0)
                .build()
                .is_err()
        );
        assert!(
            EngineOptions::builder()
                .set_combine_op_pool(vec![])
                .build()
                .is_err()
        );
        // An empty range (start > end) is rejected too.
        assert!(
            EngineOptions::builder()
                .set_hidden_dim_pool(std::ops::RangeInclusive::new(8, 4))
                .build()
                .is_err()
        );
    }

    #[test]
    fn test_engine_builder_build_engine_one_shot() {
        // build_engine = validated options + Engine::new in one call.
        let data_dir = temp_data_dir();
        let mut engine = EngineOptions::builder()
            .set_pop_size(4)
            .set_num_generations(1)
            .set_seed(Some(7))
            .set_input_dim(1)
            .set_hidden_dim_pool(4..=4)
            .build_engine(&data_dir, Fitness::mse())
            .unwrap();
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

        let a = Engine::new(make_opts(), &data_dir, Fitness::mse()).unwrap();
        let b = Engine::new(make_opts(), &data_dir, Fitness::mse()).unwrap();

        // 1. The population actually varies: not all individuals share one
        //    hidden dim, one combine op, or one activation profile.
        let mut dims: Vec<usize> = a.pop.iter().map(|g| g.options.hidden_dim).collect();
        dims.sort_unstable();
        dims.dedup();
        assert!(dims.len() > 1, "hidden dims must vary: {dims:?}");
        let mut combines: Vec<CombineOp> = Vec::new();
        for g in &a.pop {
            if !combines.contains(&g.options.combine_op) {
                combines.push(g.options.combine_op);
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
        let engine = Engine::new(opts.clone(), &data_dir, Fitness::mse()).unwrap();
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
        let mut engine = Engine::new(opts.clone(), &data_dir, Fitness::mse()).unwrap();
        engine.run().unwrap();
        let v: serde_json::Value = serde_json::from_str(&engine.to_json().unwrap()).unwrap();
        assert_eq!(v["run_seed"], engine.run_seed);
        assert_eq!(v["topology_options"]["seed"], engine.run_seed);
        assert_eq!(v["options"]["seed"], serde_json::Value::Null);
        // And a second randomized launch derives a different base seed.
        let other = Engine::new(opts.clone(), &data_dir, Fitness::mse()).unwrap();
        assert_ne!(other.run_seed, engine.run_seed);
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
        let mut a = Engine::new(make(), &data_dir, Fitness::mse()).unwrap();
        let mut b = Engine::new(make(), &data_dir, Fitness::mse()).unwrap();
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
        assert_eq!(a.options.network.seed, Some(123));
        let v: serde_json::Value = serde_json::from_str(&a.to_json().unwrap()).unwrap();
        assert_eq!(v["options"]["network"]["seed"], 123);
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&a.options.results_dir);
        let _ = std::fs::remove_dir_all(&b.options.results_dir);
    }
}
