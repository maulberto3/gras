//! gras — neural architecture search over random topologies, run as a
//! continuous step-race.
//!
//! One global step clock: every live net trains and evaluates on the same
//! shared batch each step. Divergence over the population's smoothed fitness
//! culls the worst net and births a caught-up replacement. Blueprints
//! ([`Topology`]) are compiled into executable [`Network`]s; you supply the
//! loss and fitness as plain closures.
//!
//! Quick start:
//! ```
//! # use gras::engine::RaceConfig;
//! let config = RaceConfig::builder()
//!     .set_run_pop_size(6)
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
pub use utils::fitness;
pub use evolution::{crossover, mutation, pools, selection};
pub use graph::{network, node, topology};
pub use utils::markdown;
pub use utils::tabular_data;

// ── engine — the step-race loop ──────────────────────────────────────
pub use engine::{
    GuardrailVerdict, RLSpec, RaceConfig, RaceSnapshot, RlConfig, RlEngine, RlRaceConfig, RunMode,
    RunSpec, StopReason, TabularConfig, TabularEngine, TabularRaceConfig, TabularSpec,
    rl_race_config_builder, tabular_race_config_builder,
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
// The library ships the CONTRACT (`TabularStep`/`RlStep`) *and* one reference
// tabular scheme (`TabularTrainer`): the classic recipe every consumer used to
// copy. Anything more specific — an environment, an exotic scheme — stays yours.
pub use trainer::{
    AnyTrainer, Env, LossFn, RlContext, RlStep, RlStepMeta, RunData, StepEnv, StepReport, StreamShape,
    TabularContext, TabularStep, TabularTrainer,
};

// The name→recipe tables: which LOSS trains, which FITNESS ranks, which METRIC
// is merely informative. One spelling each, shared with the Python binding.
pub use trainer::menu::{
    FITNESS_NAMES, Loss, METRIC_NAMES, TargetKind, fitness_for, metric_for, metrics_for,
};

// ── data ─────────────────────────────────────────────────────────────
pub use utils::tabular_data::{
    DataFormat, Dataset, load_csv_dataset, load_dataset, load_dataset_auto, load_tensor, one_hot,
    resolve_dataset, resolve_inputs_targets_datasets,
    resolve_train_test_datasets, save_csv_dataset, save_dataset, save_dataset_as, save_tensor,
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
    //! The common surface: engine entry points, fitness/direction types, trainer
    //! contracts + reports, the batch stream, graph types, config enums, and the
    //! tensor/device re-exports.
    //!
    //! Everything else stays reachable by module path (`gras::engine::config::*`,
    //! `gras::utils::*`), so a glob import here stays legible. Deliberately NOT
    //! re-exported: `flodl::tensor::Result` (a glob-imported `Result` would
    //! collide with `std::result::Result`), the low-level scoring helpers, and
    //! the `DataFormat`/loader family.
    // Engine config + entry points.
    pub use crate::engine::config::{
        CrossCullPolicy, LogLevel, MutationCullPolicy, RaceConfig, RaceConfigBuilder, RaceSnapshot,
        RlConfig, RlRaceConfig, RunMode, StopReason,
        TabularConfig, TabularRaceConfig, rl_race_config_builder, tabular_race_config_builder,
    };
    // Scoring vocabulary is `utils`-resident: shared with any training scheme.
    pub use crate::utils::fitness::{Direction, Fitness, FitnessLabel, Metric};
    pub use crate::engine::guardrail::{AgreementGrade, GuardrailVerdict};
    pub use crate::engine::{RLSpec, RlEngine, RunSpec, TabularEngine, TabularSpec};

    // Trainer contracts + report types.
    pub use crate::trainer::{
        AnyTrainer, BoxedLossFn, ChallengeDecay, ChallengePolicy, Env, ForcedAction, Loss, LossFn,
        Reinforce, RlContext, RlStep, RlStepMeta, RlStepReport, RunData, Signal, StepEnv, StepReport,
        StreamShape, TabularContext, TabularStep, TabularStepReport, TabularTrainer, TargetKind,
        challenge_fires, effective_challenge_prob, episode_start_seed, fitness_for, forced_action,
        least_action, metric_for, metrics_for,
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
