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
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use flodl::nn::Module;
use flodl::nn::loss::mse_loss;
use flodl::tensor::{Result, TensorError};
use flodl::{Device, Variable};
use serde::Serialize;

use crate::data::Dataset;
use crate::network::Network;
use crate::node::Activation;
use crate::topology::{CombineOp, Topology, TopologyOptions};

/// Built-in scoring strategies — the "use this path, go for 'mse'" path.
/// Serializable, so the run's `engine.json` records which one was used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum FitnessKind {
    /// Mean squared error between prediction and target (lower = better).
    #[default]
    Mse,
}

/// A user-supplied scorer: `(net, inputs, targets) -> score`, evaluated
/// identically for every individual, every generation.
pub type FitnessFn = Box<dyn Fn(&Network, &Variable, &Variable) -> Result<f64>>;

/// The scorer the engine actually runs: a built-in, or your drop-in closure.
///
/// - [`Fitness::mse`] — the one-liner built-in.
/// - [`Fitness::custom`] — "use this path, but I want THIS fitness
///   function": a closure `(&Network, &Variable, &Variable) -> Result<f64>`
///   evaluated identically for every individual, every generation.
pub enum Fitness {
    Builtin(FitnessKind),
    Custom(FitnessFn),
}

impl Fitness {
    /// Built-in mean-squared-error scoring (lower = better).
    pub fn mse() -> Self {
        Fitness::Builtin(FitnessKind::Mse)
    }

    /// Drop in your own scorer. `f(net, inputs, targets) -> score`; the
    /// engine tracks the **minimum** as the current best.
    pub fn custom<F>(f: F) -> Self
    where
        F: Fn(&Network, &Variable, &Variable) -> Result<f64> + 'static,
    {
        Fitness::Custom(Box::new(f))
    }

    /// Score one network on the loaded data.
    fn evaluate(&self, net: &Network, x: &Variable, y: &Variable) -> Result<f64> {
        match self {
            Fitness::Builtin(FitnessKind::Mse) => mse_loss(&net.forward(x)?, y)?.item(),
            Fitness::Custom(f) => f(net, x, y),
        }
    }
}

/// Knobs for one engine run. Serialized by [`Engine::to_json`] into the run
/// folder's `engine.json`, so an experiment is fully reproducible.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EngineOptions {
    /// Number of individuals in the population.
    pub pop_size: usize,
    /// Number of generations to run.
    pub num_generations: usize,
    /// Seed for the population generator (same seed → same initial graphs).
    pub seed: u64,
    /// Feature dim of the network input (must match the dataset).
    pub input_dim: usize,
    /// Internal channel width of every node's layer.
    pub hidden_dim: usize,
    /// Activation pool NAS evolution may choose from (reserved for the
    /// future `mutate` implementation).
    pub activations: Vec<Activation>,
    /// Built-in scoring strategy recorded for reproducibility.
    pub fitness: FitnessKind,
    /// Parent folder for per-run checkpoint folders (`results/<ts>/`).
    pub results_dir: PathBuf,
    /// Device to build and evaluate networks on.
    #[serde(skip)]
    pub device: Device,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            pop_size: 10,
            num_generations: 5,
            seed: 16,
            input_dim: 1,
            hidden_dim: 8,
            activations: vec![Activation::Identity, Activation::ReLU, Activation::GeLU],
            fitness: FitnessKind::Mse,
            results_dir: PathBuf::from("results"),
            device: Device::CPU,
        }
    }
}

impl EngineOptions {
    /// The topology knobs shared by every individual of this run.
    fn topology_options(&self, seed: usize) -> TopologyOptions {
        TopologyOptions {
            seed,
            min_num_nodes: 2,
            max_num_nodes: 6,
            min_inputs_per_node: 2,
            max_inputs_per_node: 4,
            min_outputs_per_node: 2,
            max_outputs_per_node: 4,
            num_outputs_net: 1,
            input_dim: self.input_dim,
            hidden_dim: self.hidden_dim,
            combine_op: CombineOp::Add,
        }
    }
}

/// The best individual seen so far.
#[derive(Clone, Debug)]
pub struct Best {
    pub fitness: f64,
    /// The blueprint that scored best — `to_json` it to replicate the net.
    pub topology: Topology,
}

/// A running NAS experiment. Build it with [`Engine::new`], run it with
/// [`Engine::run`].
pub struct Engine {
    pub options: EngineOptions,
    /// Unix timestamp identifying this run (also the checkpoint folder name).
    pub run_id: String,
    /// `results/<run_id>/` — checkpoints and logs for this run.
    pub run_dir: PathBuf,
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
    pub best: Option<Best>,
    /// How many best-improvements have been recorded into `improvements/`
    /// (also the next filename counter).
    pub improvements: usize,
    /// Last generation's scores (for compact logging).
    scores: Vec<f64>,
}

impl Engine {
    /// Start an experiment: load the dataset from `data_path`, create the
    /// per-run checkpoint folder `results/<ts>/`, and seed a population of
    /// `options.pop_size` random graphs.
    ///
    /// Fails if the dataset's input dim doesn't match `options.input_dim`.
    pub fn new(options: EngineOptions, data_path: &Path, fitness: Fitness) -> Result<Self> {
        if options.pop_size == 0 {
            return Err(TensorError::new("engine: pop_size must be > 0"));
        }

        // Data contract: tensors on disk, loaded once.
        let data = crate::data::load_dataset(data_path)?;
        let data_in =
            data.inputs.shape().get(1).copied().ok_or_else(|| {
                TensorError::new("engine: dataset inputs must be 2-D [n, input_dim]")
            })?;
        if data_in != options.input_dim as i64 {
            return Err(TensorError::new(&format!(
                "engine: dataset input_dim is {data_in} but options.input_dim is {}",
                options.input_dim
            )));
        }

        // Per-run checkpoint folder: results/<unix ts>/
        // Milliseconds so two runs started within the same second still get
        // distinct folders (a seconds-resolution id would make them collide).
        let run_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
            .to_string();
        let run_dir = options.results_dir.join(&run_id);
        fs::create_dir_all(&run_dir).map_err(|e| {
            TensorError::new(&format!("engine: cannot create {}: {e}", run_dir.display()))
        })?;

        // Seed the population: pop_size random graphs through the standard
        // pipeline (scaffold + wire + auto-de-orphan).
        let mut rng = fastrand::Rng::with_seed(options.seed);
        let mut pop = Vec::with_capacity(options.pop_size);
        for i in 0..options.pop_size {
            let n_hidden = rng.usize(2..=6);
            let mut graph =
                Topology::new(i, Some(options.topology_options(options.seed as usize + i)));
            graph.create_random_hidden_nodes(n_hidden);
            graph.set_topology();
            graph.set_network();
            pop.push(graph);
        }

        let engine = Engine {
            options,
            run_id,
            run_dir,
            pop,
            fitness,
            data,
            data_path: data_path.to_path_buf(),
            generation: 0,
            best: None,
            improvements: 0,
            scores: Vec::new(),
        };

        // Initial experiment envelope (no best yet); the final one is
        // written at the end of `run()`.
        let initial = engine.to_json()?;
        fs::write(engine.run_dir.join("engine.json"), initial)
            .map_err(|e| TensorError::new(&format!("engine: cannot write engine.json: {e}")))?;
        Ok(engine)
    }

    /// Run the full experiment: `num_generations` rounds of
    /// evaluate → log → evolve. Every best-improvement is appended to
    /// `improvements/`, and a final `engine.json` snapshot is written at the
    /// end.
    pub fn run(&mut self) -> Result<()> {
        for _ in 0..self.options.num_generations {
            let improved = self.evaluate_population()?;
            self.log_generation(improved)?;
            self.next_generation();
        }
        let json = self.to_json()?;
        fs::write(self.run_dir.join("engine.json"), json)
            .map_err(|e| TensorError::new(&format!("engine: cannot write engine.json: {e}")))?;
        Ok(())
    }

    /// Score every individual once on the loaded data. Returns whether the
    /// overall best improved (and appends it to `improvements/` if so).
    fn evaluate_population(&mut self) -> Result<bool> {
        // One Variable pair reused across the whole population — forward is
        // read-only, so no per-individual copies are needed.
        let x = Variable::new(self.data.inputs.clone(), false);
        let y = Variable::new(self.data.targets.clone(), false);

        let mut scores = Vec::with_capacity(self.pop.len());
        let mut best_idx: Option<(usize, f64)> = None;
        for (i, graph) in self.pop.iter().enumerate() {
            let net = Network::build(graph, self.options.device)?;
            let score = self.fitness.evaluate(&net, &x, &y)?;
            scores.push(score);
            if best_idx.map(|(_, s)| score < s).unwrap_or(true) {
                best_idx = Some((i, score));
            }
        }
        self.scores = scores;

        if let Some((i, score)) = best_idx {
            let improved = self
                .best
                .as_ref()
                .map(|b| score < b.fitness)
                .unwrap_or(true);
            if improved {
                self.best = Some(Best {
                    fitness: score,
                    topology: self.pop[i].clone(),
                });
                self.record_improvement()?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Append the current best blueprint to `run_dir/improvements/` — the
    /// evolution trail. One file per best-improvement, named
    /// `{counter:04}_gen{gen:02}_fitness{fit:.4}.json`, so the history reads
    /// top-down and the latest entry is the current best (`Topology::from_json`
    /// + `Network::build` replicates it).
    fn record_improvement(&mut self) -> Result<()> {
        let Some(b) = &self.best else { return Ok(()) };
        let dir = self.run_dir.join("improvements");
        fs::create_dir_all(&dir).map_err(|e| {
            TensorError::new(&format!("engine: cannot create {}: {e}", dir.display()))
        })?;
        let json = b
            .topology
            .to_json()
            .map_err(|e| TensorError::new(&format!("engine: improvement json: {e}")))?;
        let name = format!(
            "{:04}_gen{:02}_fitness{:.4}.json",
            self.improvements, self.generation, b.fitness
        );
        fs::write(dir.join(name), json)
            .map_err(|e| TensorError::new(&format!("engine: cannot write improvement: {e}")))?;
        self.improvements += 1;
        Ok(())
    }

    /// One compact log line per generation, mirrored into `run_dir/log.txt`.
    fn log_generation(&self, improved: bool) -> Result<()> {
        let scores = &self.scores;
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let flag = if improved { " 🏆 best improved" } else { "" };
        let line = format!(
            "gen {:02} · pop {} · best {min:.4} · mean {mean:.4} · worst {max:.4}{flag}",
            self.generation,
            self.pop.len()
        );
        println!("  {line}");

        let mut log = fs::read_to_string(self.run_dir.join("log.txt")).unwrap_or_default();
        log.push_str(&line);
        log.push('\n');
        fs::write(self.run_dir.join("log.txt"), log)
            .map_err(|e| TensorError::new(&format!("engine: cannot write log.txt: {e}")))
    }

    /// Advance one generation: crossover → mutate → generation += 1.
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
    /// activations from `options.activations`, adjust per-node `hidden_dim`.
    /// **Not implemented yet** — a no-op, so the population stays static.
    pub fn mutate(&mut self) {
        // TODO(engine): topology-level mutation.
    }

    /// Everything needed to replicate this experiment, as JSON: the options,
    /// the data path, the best fitness, and the **best topology** (feed it to
    /// `Topology::from_json` + `Network::build` to recreate the best net).
    pub fn to_json(&self) -> Result<String> {
        let best_topology = match &self.best {
            Some(b) => Some(
                b.topology
                    .to_json()
                    .map_err(|e| TensorError::new(&format!("engine: best topology: {e}")))?,
            ),
            None => None,
        };
        let spec = serde_json::json!({
            "run_id": self.run_id,
            "data_path": self.data_path.display().to_string(),
            "generation": self.generation,
            "pop_size": self.pop.len(),
            "options": &self.options,
            "best_fitness": self.best.as_ref().map(|b| b.fitness),
            "best_topology": best_topology,
        });
        serde_json::to_string_pretty(&spec)
            .map_err(|e| TensorError::new(&format!("engine: to_json: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn temp_data_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gras_engine_test_{}", fastrand::u64(..)));
        let ds = crate::data::synthetic_x_squared(64, 42, Device::CPU).unwrap();
        crate::data::save_dataset(&dir, &ds).unwrap();
        dir
    }

    fn test_options() -> EngineOptions {
        EngineOptions {
            pop_size: 3,
            num_generations: 2,
            input_dim: 1,
            hidden_dim: 4,
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
        assert_eq!(files.len(), engine.improvements);
        // Each entry is a valid topology that replicates the best at that
        // point; the latest one matches the final best blueprint.
        let latest_json = std::fs::read_to_string(imp_dir.join(files.last().unwrap())).unwrap();
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
        let mut engine = Engine::new(test_options(), &data_dir, Fitness::mse()).unwrap();
        engine.run().unwrap();

        let json = engine.to_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["run_id"], engine.run_id);
        assert_eq!(v["pop_size"], 3);
        assert_eq!(v["options"]["pop_size"], 3);
        assert_eq!(v["options"]["hidden_dim"], 4);
        assert_eq!(v["options"]["fitness"], "Mse");
        assert_eq!(v["data_path"], data_dir.display().to_string());
        assert!(v["best_fitness"].is_number());
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
        let fitness = Fitness::custom(move |net, x, y| {
            calls2.fetch_add(1, Ordering::SeqCst);
            // Same math as the built-in Mse, proving the drop-in path.
            mse_loss(&net.forward(x)?, y)?.item()
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
            input_dim: 2, // dataset has input_dim 1
            ..test_options()
        };
        assert!(Engine::new(opts, &data_dir, Fitness::mse()).is_err());
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[test]
    fn test_fitness_custom_needs_net_ref() {
        // Compile-time check that the closure really receives the pieces:
        // a closure that ignores the data would still type-check; this one
        // uses all three params to prove the signature is ergonomic.
        let data_dir = temp_data_dir();
        let fitness = Fitness::custom(|net, x, y| {
            let pred = net.forward(x)?;
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
}
