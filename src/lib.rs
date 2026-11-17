//! gras — neural architecture search over random topologies, run
//! continuous step-race.
//!
//! One global step clock: every live net trains and evaluates on the same
//! shared batch each step. Divergence over the population's smoothed fitness
//! culls the worst net and births a caught-up replacement. Blueprints
//! ([`Topology`]) are compiled into executable [`Network`]s; you supply
//! loss and fitness as plain closures.
//!
//! Quick start:
//! ```
//! # use gras::engine::RaceConfig;
//! let config = RaceConfig::builder()
//!     .set_pop_size(6)
//!     .set_stop_max_steps(100)
//!     .build();
//! ```

// ── module tree ──────────────────────────────────────────────────────
pub mod engine;
pub mod evolution;
pub mod graph;
pub mod spec;
pub mod state;
pub mod trainer;
pub mod utils;

// ── flat paths — the folders are an implementation detail ─────────────
pub use evolution::{crossover, mutation, pools, selection};
pub use graph::{network, node, topology};
pub use utils::fitness;
pub use utils::markdown;
pub use utils::tabular_data;

// ── engine — the step-race loop ──────────────────────────────────────
pub use engine::{
    RLSpec, RaceConfig, RaceSnapshot, RlEngine, RlRaceConfig, RunMode, RunSpec, StopReason,
    TabularEngine, TabularRaceConfig, TabularSpec, rl_race_config_builder,
    tabular_race_config_builder,
};
// Scoring vocabulary lives in `utils`, not `engine`: a training scheme must be
// able to rank/report without importing the engine.
pub use utils::fitness::{Direction, Fitness, FitnessLabel};

// ── graph — blueprints + executable networks ─────────────────────────
pub use graph::network::{Network, NetworkOptions};
pub use graph::node::{Activation, CombineOp, Node, NodeKind, StandardizeOp};
pub use graph::topology::{FeatureSplitPolicy, Topology, TopologyOptions};

// ── evolution — the genetic operators ────────────────────────────────
pub use evolution::crossover::CrossoverMethod;
pub use evolution::mutation::MutationMethod;
pub use evolution::selection::SelectionMethod;

// ── trainer — shared deterministic batching ──────────────────────────
pub use trainer::stream::{BatchStream, PoolSplit};
// The library ships the CONTRACT (`TabularStep`/`RlStep`) *and* one reference tabular scheme
// (`TabularTrainer`): the classic recipe every consumer used to copy.
pub use trainer::{
    AnyTrainer, Env, LossFn, RlContext, RlStep, RlStepMeta, RunData, StepEnv, StepReport,
    StreamShape, TabularContext, TabularStep, TabularTrainer,
};

// The name→recipe tables: which LOSS trains, which FITNESS ranks, which METRIC
// is merely informative. One spelling each, shared with the Python binding.
pub use trainer::menu::{
    FITNESS_NAMES, Loss, METRIC_NAMES, TargetKind, fitness_for, metric_for, metrics_for,
};

// ── data ─────────────────────────────────────────────────────────────
pub use utils::tabular_data::{
    DataFormat, Dataset, PoolLayout, detect_pool_layout, exam_dir, load_csv_dataset, load_dataset,
    load_dataset_auto, load_tensor, one_hot, resolve_dataset, resolve_inputs_targets_datasets,
    resolve_train_test_datasets, resolve_train_val_datasets, save_csv_dataset, save_dataset,
    save_dataset_as, save_tensor,
};

// ── scoring helpers ──────────────────────────────────────────────────
pub use utils::score::{
    accuracy_score, argmax_classes, cross_entropy_onehot, cross_entropy_onehot_loss, f1_from_vecs,
    f1_score, l1_loss_score, label_smoothing_cross_entropy_loss, mse_loss_score,
    precision_from_vecs, precision_score, r2_score, rmse_score,
};

// ── step primitives ──────────────────────────────────────────────────
pub use utils::race_steps::{
    deterministic_train_step, deterministic_train_step_fn, eval_one_step, eval_one_step_fn,
    seed_step_randomness, train_one_step, train_one_step_fn, train_one_step_pred_only,
};

// ── flodl — the tensor backend ───────────────────────────────────────
pub use flodl::{DType, Device, Variable};
// The whole crate, so advanced users can reach the full flodl API
// (nn::Module, tensor::Result, ...) as `gras::flodl::...`.
pub use flodl;

// ── prelude — one import for the common surface ──────────────────────

/// One import for the common surface: `use gras::prelude::*;`.
pub mod prelude {
    //! The common surface: engine entry points, fitness/direction types, trainer contracts +
    //! reports, the batch stream, graph types, config enums, and the tensor/device re-exports.
    // Engine config + entry points.
    pub use crate::engine::config::{
        CrossCullPolicy, LogLevel, MutationCullPolicy, RaceConfig, RaceConfigBuilder, RaceSnapshot,
        RlRaceConfig, RunMode, StopReason, TabularRaceConfig, rl_race_config_builder,
        tabular_race_config_builder,
    };
    // Scoring vocabulary is `utils`-resident: shared with any training scheme.
    pub use crate::engine::{RLSpec, RlEngine, RunSpec, TabularEngine, TabularSpec};
    pub use crate::utils::fitness::{Direction, Fitness, FitnessLabel, Metric};

    // Trainer contracts + report types.
    pub use crate::trainer::{
        AnyTrainer, BoxedLossFn, ChallengeDecay, ChallengePolicy, Env, ForcedAction, Loss, LossFn,
        ObsMemory, Reinforce, RlContext, RlStep, RlStepMeta, RlStepReport, RunData, Signal,
        StepEnv, StepReport, StreamShape, TabularContext, TabularStep, TabularStepReport,
        TabularTrainer, TargetKind, challenge_fires, effective_challenge_prob, episode_start_seed,
        fitness_for, forced_action, least_action, metric_for, metrics_for,
    };

    // Shared deterministic batching.
    pub use crate::trainer::stream::{BatchStream, PoolSplit};

    // Genetic operator knobs a builder call needs (e.g. `set_crossover_selection`).
    pub use crate::evolution::selection::SelectionMethod;

    // Graph blueprints + executable networks.
    pub use crate::graph::network::{Network, NetworkOptions};
    pub use crate::graph::node::{Activation, CombineOp, Node, NodeKind, StandardizeOp};
    pub use crate::graph::topology::{FeatureSplitPolicy, Topology, TopologyOptions};

    // Tensors, devices, and the trainer-facing flodl traits.
    pub use crate::flodl::nn::Module;
    pub use crate::flodl::nn::optim::Optimizer;
    pub use crate::flodl::{DType, Device, Tensor, Variable};
}

// ── helpers ──────────────────────────────────────────────────────────
/// CUDA device 0 with the `cuda` feature, otherwise CPU.
pub fn auto_device() -> flodl::Device {
    #[cfg(feature = "cuda")]
    {
        flodl::Device::CUDA(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        flodl::Device::CPU
    }
}

// ── example dependency pin ───────────────────────────────────────────
/// The examples are runnable documentation and must run on what the crate already pulls: a task that seems to need a new crate is a sign the example is reaching past the…
#[cfg(test)]
mod example_dependency_pin {
    const MANIFEST: &str = include_str!("../Cargo.toml");

    /// The crates declared in one `[table]`: the key of each `name = ...` line, sorted.
    fn table(name: &str) -> Vec<String> {
        let header = format!("[{name}]");
        let mut out = Vec::new();
        let mut inside = false;
        for line in MANIFEST.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                inside = trimmed == header;
                continue;
            }
            if !inside || trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some((key, _)) = trimmed.split_once('=') {
                out.push(key.trim().to_string());
            }
        }
        out.sort();
        out
    }

    /// The declared crates, pinned: adding one is a decision, so update this list in the same change.
    const DEPS: &[&str] = &[
        "comfy-table",
        "ctrlc",
        "fastrand",
        "flodl",
        "serde",
        "serde_json",
        "strum",
        "tracing",
        "tracing-subscriber",
        "xxhash-rust",
    ];
    const DEV_DEPS: &[&str] = &["clap", "dhat", "proptest"];

    fn expected(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn examples_add_no_dependencies() {
        assert_eq!(
            table("dependencies"),
            expected(DEPS),
            "[dependencies] changed: an example must not need a new crate (CONVENTIONS.md)"
        );
        assert_eq!(
            table("dev-dependencies"),
            expected(DEV_DEPS),
            "[dev-dependencies] changed: an example must not need a new crate (CONVENTIONS.md)"
        );
    }
}
