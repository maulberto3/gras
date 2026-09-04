//! Engine run configuration — [`EngineOptions`] and its fluent builder.

use std::ops::RangeInclusive;
use std::path::PathBuf;

use flodl::tensor::Result;
use serde::Serialize;

use crate::engine::fitness::FitnessLabel;
use crate::evolution::crossover::CrossoverMethod;
use crate::evolution::mutation::MutationMethod;
use crate::evolution::selection::SelectionMethod;
use crate::graph::node::Activation;
use crate::graph::topology::{CombineOp, TopologyOptions};
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
    pub standardize_op_pool: Vec<crate::graph::node::StandardizeOp>,
    /// Fitness name
    pub fitness_label: FitnessLabel,
    /// Threads for parallel eval (0 = rayon default).
    pub num_threads: usize,
    /// Directory for engine.json, per-gen snapshots, and robustness.csv.
    pub results_dir: PathBuf,
    /// Selection strategy for the next generation.
    pub selection: Option<SelectionMethod>,
    /// Crossover strategy.
    pub crossover: Option<CrossoverMethod>,
    /// Mutation strategy.
    pub mutation: MutationMethod,
    /// Deduplicate population by full topology comparison.
    pub dedup_pop_and_fill: bool,
    /// Number of top individuals to preserve untouched each generation.
    pub elite_count: usize,
    /// Paths to prior engine.json or improvement JSON files.
    pub prior_topology_paths: Vec<PathBuf>,
    /// Resume a previous run: path to a run directory containing a
    /// `checkpoint.json`. The checkpoint's population + `run_seed` replace
    /// the fresh random population, the generation counter continues from the
    /// checkpoint, and `num_generations` counts **additional** generations.
    pub resume_from: Option<PathBuf>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            // ── Required (must be set by user) ─────────────────────────
            pop_size: None,
            num_generations: None,
            mutation: MutationMethod::default(), // prob=0 → build() rejects unless user calls set_mutation()
            // ── Engine ────────────────────────────────────────────────
            seed: None, // random if not set
            num_threads: 1,
            results_dir: PathBuf::from("results"),
            dedup_pop_and_fill: false,
            elite_count: 2,
            prior_topology_paths: vec![],
            resume_from: None,
            // ── Topology / GP pools ───────────────────────────────────
            topology_options: TopologyOptions::default(),
            hidden_dim_pool: 4..=8,
            hidden_dim_stride: 1,
            combine_op_pool: vec![],     // empty = all built-ins
            activation_pool: vec![],     // empty = all built-ins
            standardize_op_pool: vec![], // empty = all built-ins
            // ── Labels ─────────────────────────────────────────────
            fitness_label: FitnessLabel::default(),
            // ── Genetics ──────────────────────────────────────────────
            selection: None,
            crossover: None,
        }
    }
}

impl EngineOptions {
    /// Derive topology options for one individual (clone template + override
    /// seed and dropout). `dropout_prob` comes from the *trainer* (a training
    /// hyperparameter, not an evolution setting) and is written into the
    /// blueprint so the saved graph is self-describing —
    /// [`Network::build`](crate::graph::network::Network::build) then
    /// reproduces the exact net the engine built, without needing either
    /// `EngineOptions` or the trainer's config.
    pub(crate) fn derive_topology_options(&self, seed: usize, dropout_prob: f32) -> TopologyOptions {
        let mut t = self.topology_options;
        t.seed = seed;
        t.dropout_prob = dropout_prob;
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
        if o.selection.is_none() {
            return Err(EngineError::InvalidOptions(
                "set_selection() is required — choose SelectionMethod::Tournament".into(),
            )
            .into());
        }
        if o.crossover.is_none() {
            return Err(EngineError::InvalidOptions(
                "set_crossover() is required — choose CrossoverMethod::OnePoint or Uniform".into(),
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
        // Resume replaces the population wholesale — it cannot be combined
        // with warm-starting individual topologies into a fresh population.
        if o.resume_from.is_some() && !o.prior_topology_paths.is_empty() {
            return Err(EngineError::InvalidOptions(
                "set_resume_from() and set_prior_topology() are mutually exclusive — resume replaces the population, warm start injects into a fresh one"
                    .into(),
            )
            .into());
        }
        // Store validated values back as unwrapped
        o.pop_size = Some(pop_size);
        o.num_generations = Some(num_generations);
        // ── Conservative defaults for non-required fields ──────────────
        if o.hidden_dim_pool.is_empty() {
            o.hidden_dim_pool = 4..=8;
        }
        // ── Auto-fill pools with all built-ins ─────────────────────────
        if o.combine_op_pool.is_empty() {
            o.combine_op_pool = crate::evolution::pools::all_combine_ops();
        }
        if o.activation_pool.is_empty() {
            o.activation_pool = crate::evolution::pools::all_activations();
        }
        if o.standardize_op_pool.is_empty() {
            o.standardize_op_pool = crate::evolution::pools::all_standardize_ops();
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
    pub fn set_prior_topology(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner.prior_topology_paths.push(path.into());
        self
    }
    pub fn set_prior_topologies(mut self, paths: Vec<impl Into<PathBuf>>) -> Self {
        self.inner
            .prior_topology_paths
            .extend(paths.into_iter().map(|p| p.into()));
        self
    }
    /// Resume from a previous run's directory (must contain `checkpoint.json`,
    /// written at run start and after every generation). The resumed run
    /// continues with the checkpoint's population and `run_seed`; set
    /// `num_generations` to the number of **additional** generations to run.
    /// Mutually exclusive with `set_prior_topology(ies)`.
    pub fn set_resume_from(mut self, dir: impl Into<PathBuf>) -> Self {
        self.inner.resume_from = Some(dir.into());
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
    pub fn set_standardize_op_pool(mut self, ops: Vec<crate::graph::node::StandardizeOp>) -> Self {
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
}
