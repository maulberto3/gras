//! The engine -- NAS loop over random topologies: seed, score, evolve.
//!
//! Data contract: flodl-native tensors loaded once at Engine::new,
//! reused per individual per generation. Replicate via
//! Engine::to_json -> Topology::from_json + Network::build.

use std::fs;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use flodl::tensor::Result;
use log::debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub use self::fitness::{Direction, Fitness, FitnessLabel};
pub use crate::evolution::crossover::CrossoverMethod;
pub use crate::evolution::mutation::MutationMethod;
pub use crate::evolution::selection::SelectionMethod;
use crate::graph::network::Network;
use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;
use crate::utils::error::EngineError;
use crate::utils::progress::ProgressTracker;
pub(crate) use crate::utils::robustness::RobustnessEntry;

pub mod fitness;
mod options;
mod persistence;
pub use options::{EngineOptions, EngineOptionsBuilder};
#[cfg(test)]
mod tests;

// ── Generation statistics and robustness tracking ─────────────────────────────────

/// Per-generation metrics, always saved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub avg_score: f32,
    pub avg_loss: Option<f32>,
    pub avg_params: f32,
    pub unique_topos: usize,
}

// ── Generation statistics (pure computation) ──────────────────────────────

/// Result of per-generation stat computation.
#[derive(Clone, Debug)]
pub(crate) struct GenStats {
    pub best_score: f32,
    pub best_loss: Option<f32>,
    pub avg_score: f32,
    pub avg_loss: Option<f32>,
    pub avg_params: f32,
    pub unique_topos: usize,
}

/// Digit width for zero-padded generation numbers, derived from the total
/// generation count — so filenames stay fixed-width and sort chronologically
/// at any scale (`gen_00` … `gen_99` … `gen_100` …).
pub(crate) fn gen_width(num_gens: usize) -> usize {
    num_gens.max(1).to_string().len()
}

/// Pure function: compute per-generation statistics from raw scores.
/// No mutation — safe to test in isolation.
pub(crate) fn compute_gen_stats(
    scores: &[f32],
    eval_losses: &[Option<f32>],
    param_counts: &[usize],
    direction: Direction,
    pop: &[Topology],
) -> GenStats {
    use std::collections::HashSet;

    let n = scores.len();
    debug_assert!(!scores.is_empty());

    let mut best_idx = 0;
    for (i, &score) in scores.iter().enumerate() {
        if direction.is_better(score, scores[best_idx]) {
            best_idx = i;
        }
    }

    // Averages (filter NaN/Inf)
    let valid_scores: Vec<f32> = scores.iter().copied().filter(|v| v.is_finite()).collect();
    let avg_score = if valid_scores.is_empty() {
        0.0
    } else {
        valid_scores.iter().sum::<f32>() / valid_scores.len() as f32
    };
    let valid_losses: Vec<f32> = eval_losses
        .iter()
        .filter_map(|&v| v)
        .filter(|v| v.is_finite())
        .collect();
    let avg_loss = if valid_losses.is_empty() {
        None
    } else {
        Some(valid_losses.iter().sum::<f32>() / valid_losses.len() as f32)
    };

    // Param stats
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
        best_score: scores[best_idx],
        best_loss: eval_losses.get(best_idx).copied().flatten(),
        avg_score,
        avg_loss,
        avg_params,
        unique_topos,
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
    pub(crate) trainer: Box<dyn crate::trainer::Trainer>,
    pub generation: usize,
    pub history: Vec<GenerationStats>,

    scores: Vec<f32>,
    eval_losses: Vec<Option<f32>>,
    param_counts: Vec<usize>,
    /// Per-individual per-epoch mean training loss (gen JSON curves).
    train_loss_curves: Vec<Vec<f32>>,
    /// Per-individual per-epoch mean held-out loss (gen JSON curves).
    eval_loss_curves: Vec<Vec<f32>>,
    /// Topology robustness tracker — keyed by topology JSON.
    robustness: std::collections::HashMap<String, RobustnessEntry>,
    /// Per-generation summary — computed once in save_generation_snapshots.
    gen_stats: Option<GenStats>,
    /// Builder-validated core settings — resolved once at construction.
    config: ResolvedConfig,
}

/// Which robustness rows to show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustnessFilter {
    /// Most-repeated topologies with the highest mean fitness.
    Best,
    /// Most-repeated topologies with the lowest mean fitness.
    Worst,
    /// Both.
    Both,
}

/// Result of training + scoring one individual — the ranking metrics plus
/// optional per-epoch train/test loss curves, persisted per individual in
/// the gen JSONs (overfitting tracking).
pub(crate) struct EvalResult {
    pub score: f32,
    pub eval_loss: Option<f32>,
    pub param_count: usize,
    pub train_losses: Vec<f32>,
    pub eval_losses: Vec<f32>,
}

/// Builder-validated core settings, resolved once at construction.
/// Lets the run loop read plain values instead of unwrapping Options.
pub(crate) struct ResolvedConfig {
    pub pop_size: usize,
    pub num_generations: usize,
    pub selection: SelectionMethod,
    pub crossover: CrossoverMethod,
}

impl Engine {
    // ── Construction ────────────────────────────────────────────────────────

    /// Create engine. Pass the trainer directly — no `Box::new` needed.
    pub fn new(
        mut options: EngineOptions,
        fitness: Fitness,
        trainer: impl crate::trainer::Trainer + 'static,
    ) -> Result<Self> {
        // Step 1: Validate options and fill empty pools
        Self::validate_and_fill_options(&mut options)?;

        // Step 2: Resolve seed + builder-validated core settings. The
        // unwraps happen here once, at the construction boundary — never
        // inside the run loop. Both are overridden on resume (the checkpoint
        // carries its own run_seed, and num_generations counts additions).
        let mut seed = Self::resolve_seed(&mut options);
        let mut config = ResolvedConfig {
            pop_size: options.pop_size.expect("validated by build()"),
            num_generations: options.num_generations.expect("validated by build()"),
            selection: options.selection.clone().expect("validated by build()"),
            crossover: options.crossover.clone().expect("validated by build()"),
        };

        // Step 3: Bind dims from trainer
        options.topology_options.input_dim = trainer.input_dim();
        options.topology_options.output_dim = trainer.output_dim();

        // Step 4: Create population
        let mut pop = Self::create_population(&options, config.pop_size, seed)?;

        // Step 4b: Warm start — inject prior topologies if provided
        for (idx, path) in options.prior_topology_paths.iter().enumerate() {
            if idx >= pop.len() {
                break;
            }
            let topo = Self::load_prior_topology(path)?;
            pop[idx] = topo;
            log::info!(
                "  warm start: loaded prior topology into pop[{}] from {}",
                idx,
                path.display()
            );
        }

        // Step 4c: Resume — replace the random population with the exact
        // frontier from a previous run's checkpoint.json. The checkpoint's
        // run_seed keeps the genetics deterministic (selection/crossover/
        // mutation/refill all derive from it), the generation counter
        // continues, and `config.num_generations` counts ADDITIONAL gens.
        let mut resume_state: Option<persistence::LoadedCheckpoint> = None;
        if let Some(dir) = options.resume_from.clone() {
            let state = persistence::load_checkpoint(&dir)?;
            if state.pop.len() != config.pop_size {
                log::warn!(
                    "  resume: checkpoint holds {} individuals but set_pop_size({}) was given — using the checkpoint's size",
                    state.pop.len(),
                    config.pop_size
                );
            }
            config.pop_size = state.pop.len();
            seed = state.run_seed;
            options.seed = Some(seed);
            options.topology_options.seed = seed as usize;
            log::info!(
                "  resume: continuing from generation {} (run_seed {}) with {} individuals — {} additional generations",
                state.generation,
                seed,
                state.pop.len(),
                config.num_generations
            );
            pop = state.pop.clone();
            resume_state = Some(state);
        }

        // Dedup/refill only applies to freshly-created populations — a
        // checkpoint population is already the run's real frontier.
        if resume_state.is_none() && options.dedup_pop_and_fill {
            Self::dedup_population(&mut pop);
            Self::refill_population(&options, config.pop_size, seed, 0, &mut pop);
        }

        // Log initialization
        crate::utils::log_utils::log_initialization(&options, &pop, seed, &fitness);

        // Assemble engine
        let mut engine =
            Self::assemble_engine(options, config, seed, pop, fitness, Box::new(trainer))?;
        if let Some(state) = resume_state {
            engine.generation = state.generation;
            engine.history = state.history;
            engine.robustness = state.robustness;
        }
        Ok(engine)
    }

    /// Step 1: Propagate engine-level pools into mutation variant, fill defaults.
    /// All required validation is done in `EngineOptionsBuilder::build()`.
    fn validate_and_fill_options(options: &mut EngineOptions) -> Result<()> {
        if options.hidden_dim_pool.is_empty() {
            options.hidden_dim_pool = 4..=8;
        }
        if options.hidden_dim_stride == 0 {
            options.hidden_dim_stride = 1;
        }
        if options.combine_op_pool.is_empty() {
            options.combine_op_pool = crate::evolution::pools::all_combine_ops();
        }
        if options.activation_pool.is_empty() {
            options.activation_pool = crate::evolution::pools::all_activations();
        }
        if options.standardize_op_pool.is_empty() {
            options.standardize_op_pool = crate::evolution::pools::all_standardize_ops();
        }
        Ok(())
    }

    /// Step 2: Resolve seed -- user-provided or random.
    /// Writes the resolved seed back into `options.seed` too, so engine.json's
    /// `options` block records the *actual* seed used (not `null` for auto-seeded
    /// runs) — a user can reproduce the experiment from the JSON alone.
    fn resolve_seed(options: &mut EngineOptions) -> u64 {
        let seed = options.seed.unwrap_or_else(|| {
            let t = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            t ^ fastrand::u64(..)
        });
        options.seed = Some(seed);
        options.topology_options.seed = seed as usize;
        seed
    }

    /// Create a population of random topologies, seeded deterministically.
    fn create_population(
        options: &EngineOptions,
        pop_size: usize,
        seed: u64,
    ) -> Result<Vec<Topology>> {
        let mut pop = Vec::with_capacity(pop_size);
        for i in 0..pop_size {
            let ind_seed = derive_seed(seed, i);
            let graph = Self::create_individual(options, i, ind_seed);
            debug!(
                "  ind[{i}] seed={} n_hidden={} nodes={} wires={}",
                ind_seed,
                graph
                    .nodes
                    .iter()
                    .filter(|n| n.kind == NodeKind::Hidden)
                    .count(),
                graph.nodes.len(),
                graph.connections.len()
            );
            pop.push(graph);
        }
        Ok(pop)
    }

    /// Create a single random individual from engine pools.
    /// Shared by `create_population` and `refill_population`.
    fn create_individual(options: &EngineOptions, id: usize, seed: u64) -> Topology {
        let mut rng = fastrand::Rng::with_seed(seed);
        let n_hidden = rng.usize(
            options.topology_options.min_hidden_num_nodes
                ..=options.topology_options.max_hidden_num_nodes,
        );
        let ind_opts = options.derive_topology_options(seed as usize);
        let mut graph = Topology::new(id, Some(ind_opts));
        graph.create_random_hidden_nodes(n_hidden);
        let pool_len_a = options.activation_pool.len();
        let pool_len_c = options.combine_op_pool.len();
        let pool_len_s = options.standardize_op_pool.len();
        for node in &mut graph.nodes {
            if node.kind == NodeKind::Hidden {
                node.hidden_dim = Some(Self::sample_hidden_dim(
                    &options.hidden_dim_pool,
                    options.hidden_dim_stride,
                    &mut rng,
                ));
                node.activation = options.activation_pool[rng.usize(0..pool_len_a)];
                node.combine_op = Some(options.combine_op_pool[rng.usize(0..pool_len_c)]);
                node.standardize = Some(options.standardize_op_pool[rng.usize(0..pool_len_s)]);
            }
        }
        graph.refresh_labels();
        graph.finalize();
        graph
    }

    /// Remove duplicate topologies from the population (full Spec comparison).
    /// Keeps the first occurrence of each unique topology.
    fn dedup_population(pop: &mut Vec<Topology>) {
        use crate::spec::Spec;
        use std::collections::HashSet;
        let before = pop.len();
        let mut seen: HashSet<Spec> = HashSet::new();
        pop.retain(|topo| {
            let spec = Spec::from(topo);
            seen.insert(spec)
        });
        let removed = before - pop.len();
        if removed > 0 {
            debug!(
                "  dedup: removed {removed} duplicates, {}/{} remain",
                pop.len(),
                before
            );
        }
    }

    /// Refill population back to `pop_size` with fresh random individuals.
    /// Called after dedup to keep the population at full strength.
    /// Returns the number of individuals added.
    fn refill_population(
        options: &EngineOptions,
        target: usize,
        seed: u64,
        generation: usize,
        pop: &mut Vec<Topology>,
    ) -> usize {
        if pop.len() >= target {
            return 0;
        }
        let needed = target - pop.len();
        // Offset into a disjoint seed space (initial pop uses 0..pop_size).
        let base_offset = 1_000_000 + generation * target + pop.len();
        for k in 0..needed {
            let ind_seed = derive_seed(seed, base_offset + k);
            let id = pop.len();
            let graph = Self::create_individual(options, id, ind_seed);
            debug!(
                "  refill[{id}] seed={} n_hidden={} nodes={} wires={}",
                ind_seed,
                graph
                    .nodes
                    .iter()
                    .filter(|n| n.kind == NodeKind::Hidden)
                    .count(),
                graph.nodes.len(),
                graph.connections.len()
            );
            pop.push(graph);
        }
        debug!(
            "  refill: added {needed} individuals, {}/{} remain",
            pop.len(),
            target
        );
        needed
    }

    /// Step 5: Log all resolved options, dataset, and population.
    /// Build thread pool, generate run id, assemble Engine struct.
    fn assemble_engine(
        options: EngineOptions,
        config: ResolvedConfig,
        seed: u64,
        pop: Vec<Topology>,
        fitness: Fitness,
        trainer: Box<dyn crate::trainer::Trainer>,
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
            trainer,
            generation: 0,
            history: Vec::new(),

            scores: Vec::new(),
            eval_losses: Vec::new(),
            param_counts: Vec::new(),
            train_loss_curves: Vec::new(),
            eval_loss_curves: Vec::new(),
            gen_stats: None,
            config,
            robustness: std::collections::HashMap::new(),
        })
    }

    // ── Query ───────────────────────────────────────────────────────────────

    /// Returns the scores from the most recent generation.
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// The deterministic training seed the engine passes to every individual
    /// of `generation` (`derive_seed(run_seed, generation)`) — public so
    /// replay tools can reproduce a run's train/eval split and batch
    /// sampling from the gen JSONs alone.
    pub fn gen_seed_for(&self, generation: usize) -> u64 {
        derive_seed(self.seed, generation)
    }

    /// Display topology robustness table (most-repeated topologies).
    /// `which`: "best" (highest fitness), "worst" (lowest fitness), or "both".
    pub fn show_robustness(&self, top_n: usize, which: RobustnessFilter) {
        match which {
            RobustnessFilter::Best => {
                crate::utils::log_utils::log_repeated_topologies(&self.robustness, top_n, true);
            }
            RobustnessFilter::Worst => {
                crate::utils::log_utils::log_repeated_topologies(&self.robustness, top_n, false);
            }
            RobustnessFilter::Both => {
                crate::utils::log_utils::log_repeated_topologies(&self.robustness, top_n, true);
                crate::utils::log_utils::log_repeated_topologies(&self.robustness, top_n, false);
            }
        }
    }

    // ── Run -- the main loop ─────────────────────────────────────────────────

    /// Execute the full evolution loop: init → generations → finalize.
    /// Writes engine.json, per-gen snapshots, and robustness.csv.
    pub fn run(&mut self) -> Result<()> {
        let run_start = self.init_run()?;
        self.run_generations()?;
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
        // Frontier checkpoint — resume target if the run is interrupted
        // before completing its first generation.
        self.save_checkpoint()?;
        Ok(Instant::now())
    }

    /// Phase 2: The evolution loop. On a resumed run, `num_generations` counts
    /// the generations this process runs; `self.generation` already holds the
    /// frontier the loop continues from, so the counter (not the loop index)
    /// is what shows up in logs, gen file names, and gen_seed derivation.
    fn run_generations(&mut self) -> Result<()> {
        let num_gens = self.config.num_generations;
        let total = self.generation + num_gens;
        let width = gen_width(total);
        for _ in 0..num_gens {
            let g = self.generation;
            let gen_start = Instant::now();
            debug!("== gen {g:0width$}/{total:0width$} ==", width = width);
            let _improved = self.evaluate_population()?;
            // One table per generation: scores + evolution, so all of a
            // single generation's data reads as one unit.
            let mut rows = self.generation_summary_rows();
            rows.extend(self.next_generation()?);
            crate::utils::log_utils::log_section_table(
                &format!("gen {g:0width$} of {total:0width$}", width = width),
                &["metric", "best", "avg"],
                &rows,
                crate::utils::log_utils::TABLE_WIDTH,
            );
            log::info!(
                "  gen {g:0width$} done in {elapsed:.1}s",
                g = g,
                elapsed = gen_start.elapsed().as_secs_f64(),
                width = width
            );
        }
        Ok(())
    }

    /// Phase 3: Write final engine.json, log run summary.
    fn finalize_run(&mut self, run_start: Instant) -> Result<()> {
        let run_elapsed = run_start.elapsed();
        let json = self.to_json()?;
        fs::write(self.run_dir.join("engine.json"), json).map_err(|source| EngineError::Io {
            path: self.run_dir.join("engine.json").display().to_string(),
            source,
        })?;
        let num_gens = self.config.num_generations;
        let robustness_csv = self.run_dir.join("robustness.csv");
        self.export_robustness_to(&robustness_csv)?;
        crate::utils::log_utils::log_done(run_elapsed, num_gens, &self.run_dir, &robustness_csv);

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
    fn eval_all_individuals(&self) -> Result<Vec<EvalResult>> {
        let device = self.trainer.device();
        let dtype = self.trainer.dtype();
        let tracker = ProgressTracker::new();
        let results = self.pool.install(|| {
            self.pop
                .par_iter()
                .enumerate()
                .map(|(idx, graph)| {
                    // dropout comes from the individual's own blueprint, which
                    // the engine seeded from EngineOptions at creation — a
                    // saved topology is therefore self-describing.
                    let no = crate::graph::network::NetworkOptions {
                        device,
                        dtype,
                        seed: graph.options.seed,
                        dropout_prob: graph.options.dropout_prob,
                    };
                    let net = Network::build_with_options(graph, &no)?;
                    let gen_seed = derive_seed(self.seed, self.generation);
                    let outcome = self
                        .trainer
                        .evaluate(net, &self.fitness, gen_seed)
                        .map_err(|e| {
                            flodl::tensor::TensorError::new(&format!(
                                "individual {idx} (generation {}) failed to evaluate on {device:?}: {e}",
                                self.generation
                            ))
                        })?;
                    tracker.increment();
                    Ok(EvalResult {
                        score: outcome.score,
                        eval_loss: outcome.eval_loss,
                        param_count: outcome.param_count,
                        train_losses: outcome.train_losses,
                        eval_losses: outcome.eval_losses,
                    })
                })
                .collect::<Result<Vec<_>>>()
        });
        tracker.finish();
        results
    }

    /// Step 2: Store scores, eval_losses, param_counts, and loss curves from
    /// the parallel results — all arrays stay in lockstep with `pop`.
    fn update_scores(&mut self, results: Vec<EvalResult>) {
        self.scores = results.iter().map(|r| r.score).collect();
        self.eval_losses = results.iter().map(|r| r.eval_loss).collect();
        self.param_counts = results.iter().map(|r| r.param_count).collect();
        self.train_loss_curves = results.iter().map(|r| r.train_losses.clone()).collect();
        self.eval_loss_curves = results.iter().map(|r| r.eval_losses.clone()).collect();
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
        self.gen_stats = Some(stats.clone());

        // 3. Save per-gen data (one JSON + one MD)
        self.save_gen_data(&stats)?;

        // 5. Update robustness tracker
        self.update_robustness();

        Ok(true)
    }

    // ── Logging ─────────────────────────────────────────────────────────────

    /// Score rows for the per-generation table: (metric, best, avg).
    fn generation_summary_rows(&self) -> Vec<Vec<String>> {
        let Some(stats) = &self.gen_stats else {
            return Vec::new();
        };
        let fl = self.fitness.label();
        let mut rows: Vec<Vec<String>> = Vec::new();
        rows.push(vec![
            fl.to_string(),
            format!("{:.4}", stats.best_score),
            format!("{:.4}", stats.avg_score),
        ]);
        if stats.best_loss.is_some() || stats.avg_loss.is_some() {
            let best_l = stats
                .best_loss
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "—".into());
            let avg_l = stats
                .avg_loss
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "—".into());
            rows.push(vec!["loss".into(), best_l, avg_l]);
        }
        rows.push(vec![
            "params".into(),
            "—".into(),
            format!("{:.0}", stats.avg_params),
        ]);
        rows.push(vec![
            "topologies".into(),
            format!("{}/{}", stats.unique_topos, self.pop.len()),
            String::new(),
        ]);
        rows
    }

    // ── Genetics -- selection, crossover, mutation ────────────────────────────

    /// Sample a hidden_dim from the pool, respecting stride.
    fn sample_hidden_dim(
        pool: &RangeInclusive<usize>,
        stride: usize,
        rng: &mut fastrand::Rng,
    ) -> usize {
        let start = *pool.start();
        let end = *pool.end();
        let n = ((end - start) / stride) + 1;
        start + rng.usize(0..n) * stride
    }

    /// Returns the evolution rows for the per-generation table.
    fn next_generation(&mut self) -> Result<Vec<Vec<String>>> {
        let (unique, sel_label) = self.select();
        let cx_pairs = self.crossover();
        let pre_dedup = self.pop.len();
        if self.options.dedup_pop_and_fill {
            Self::dedup_population(&mut self.pop);
        }
        let dedup_removed = pre_dedup - self.pop.len();
        let refill_added = Self::refill_population(
            &self.options,
            self.config.pop_size,
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
        let genetics = self.genetics_summary_rows(
            unique,
            &sel_label,
            cx_pairs,
            dedup_removed,
            pre_dedup,
            refill_added,
            post_refill_dedup,
            mut_count,
        );
        // Always capture history
        if let Some(stats) = &self.gen_stats {
            self.history.push(GenerationStats {
                generation: self.generation,
                avg_score: stats.avg_score,
                avg_loss: stats.avg_loss,
                avg_params: stats.avg_params,
                unique_topos: stats.unique_topos,
            });
        }
        self.generation += 1;
        // Persist the frontier: the population about to be evaluated at the
        // next generation, so a run killed mid-evaluation resumes exactly
        // from here (re-evaluating the interrupted generation).
        self.save_checkpoint()?;
        Ok(genetics)
    }

    /// Evolution rows for the per-generation table: (step, count, detail).
    fn genetics_summary_rows(
        &self,
        unique: usize,
        sel_label: &str,
        cx_pairs: usize,
        dedup_removed: usize,
        pre_dedup: usize,
        refill_added: usize,
        post_refill_dedup: usize,
        mut_count: usize,
    ) -> Vec<Vec<String>> {
        let target = self.config.pop_size;
        let mut rows: Vec<Vec<String>> = Vec::new();
        rows.push(vec![
            "selection".into(),
            format!("{unique}/{target}"),
            format!("unique ({sel_label}) · elite {}", self.options.elite_count),
        ]);
        rows.push(vec![
            "crossover".into(),
            format!("{cx_pairs} pairs"),
            format!("{target}/{target}"),
        ]);
        if dedup_removed > 0 {
            rows.push(vec![
                "dedup".into(),
                format!("-{dedup_removed}"),
                format!(
                    "{pre_dedup}/{target} → {}/{target}",
                    pre_dedup - dedup_removed
                ),
            ]);
        } else {
            rows.push(vec![
                "dedup".into(),
                "0".into(),
                format!("{target}/{target}"),
            ]);
        }
        if refill_added > 0 {
            let after_refill = self.pop.len();
            rows.push(vec![
                "refill".into(),
                format!("+{refill_added}"),
                format!(
                    "{}/{target} → {after_refill}/{target}",
                    after_refill - refill_added
                ),
            ]);
        } else {
            rows.push(vec![
                "refill".into(),
                "0".into(),
                format!("{}/{}", self.pop.len(), target),
            ]);
        }
        if post_refill_dedup > 0 {
            rows.push(vec![
                "dedup (post)".into(),
                format!("-{post_refill_dedup}"),
                format!("{}/{}", self.pop.len(), target),
            ]);
        }
        rows.push(vec![
            "mutation".into(),
            format!("{mut_count} nets"),
            format!("{}/{}", self.pop.len(), target),
        ]);
        rows
    }

    /// Selection -- reorder pop/scores so fittest survive.
    /// Returns (unique_survivors, selection_label).
    pub fn select(&mut self) -> (usize, String) {
        if self.scores.is_empty() {
            return (0, self.config.selection.label().to_string());
        }
        let dir = self.fitness.direction();
        let mut rng = fastrand::Rng::with_seed(derive_seed(self.seed, self.generation * 3 + 1));
        let selection = &self.config.selection;
        let indices = selection.apply(&self.scores, dir, &mut rng, self.options.elite_count);
        let label = selection.label().to_string();

        // In-place reorder: build the new pop from selected indices, keeping
        // every parallel metric array in lockstep so `pop[i] ↔ scores[i] ↔
        // eval_losses[i] ↔ param_counts[i]` always refers to one individual.
        // (select() runs right after evaluation, when all three arrays are
        // populated; the length guards keep a caller-visible invariant.)
        let n = self.scores.len();
        let new_pop: Vec<Topology> = indices.iter().map(|&i| self.pop[i].clone()).collect();
        let new_scores: Vec<f32> = indices.iter().map(|&i| self.scores[i]).collect();
        let new_losses: Vec<Option<f32>> = if self.eval_losses.len() == n {
            indices.iter().map(|&i| self.eval_losses[i]).collect()
        } else {
            Vec::new()
        };
        let new_params: Vec<usize> = if self.param_counts.len() == n {
            indices.iter().map(|&i| self.param_counts[i]).collect()
        } else {
            Vec::new()
        };
        let new_train_curves: Vec<Vec<f32>> = if self.train_loss_curves.len() == n {
            indices.iter().map(|&i| self.train_loss_curves[i].clone()).collect()
        } else {
            Vec::new()
        };
        let new_eval_curves: Vec<Vec<f32>> = if self.eval_loss_curves.len() == n {
            indices.iter().map(|&i| self.eval_loss_curves[i].clone()).collect()
        } else {
            Vec::new()
        };
        self.pop = new_pop;
        self.scores = new_scores;
        if new_losses.len() == n {
            self.eval_losses = new_losses;
        }
        if new_params.len() == n {
            self.param_counts = new_params;
        }
        if new_train_curves.len() == n {
            self.train_loss_curves = new_train_curves;
        }
        if new_eval_curves.len() == n {
            self.eval_loss_curves = new_eval_curves;
        }

        let mut counts = vec![0usize; self.pop.len()];
        for &i in &indices {
            counts[i] += 1;
        }
        let unique = counts.iter().filter(|&&c| c > 0).count();
        debug!("  selection [{}] {} unique survivors", label, unique);
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
        let kind = &self.config.crossover;
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
                    if pool.is_empty() {
                        continue;
                    }
                    node.activation = pool[rng.usize(0..pool.len())];
                }
                MutationMethod::CombineOp { .. } => {
                    let pool = &self.options.combine_op_pool;
                    if pool.is_empty() {
                        continue;
                    }
                    node.combine_op = Some(pool[rng.usize(0..pool.len())]);
                }
                MutationMethod::Standardize { .. } => {
                    let pool = &self.options.standardize_op_pool;
                    if pool.is_empty() {
                        continue;
                    }
                    node.standardize = Some(pool[rng.usize(0..pool.len())]);
                }
            }
            topo.finalize();
            mut_count += 1;
        }
        debug!("  mutate {mut_count} nets ({})", m);
        mut_count
    }
}

/// Deterministic child-seed derivation: multiply by golden ratio for spread.
///
/// Every seeded stream in the engine derives from this:
/// - individual `i` of the initial population: `derive_seed(run_seed, i)`
///   (also the topology `options.seed`)
/// - generation `g`'s trainer seed (train/eval split + batch sampling):
///   `derive_seed(run_seed, g)` — see [`Engine::gen_seed_for`]
/// - selection/crossover/mutation streams: `derive_seed(run_seed, g*3 + k)`
pub fn derive_seed(base: u64, i: usize) -> u64 {
    base.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}
