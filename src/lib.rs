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
//!     .set_pop_size(6)
//!     .set_max_steps(100)
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
pub use engine::fitness;
pub use evolution::{crossover, mutation, pools, selection};
pub use graph::{network, node, topology};
pub use utils::tabular_data;
pub use utils::markdown;

// ── engine — the step-race loop ──────────────────────────────────────
pub use engine::{
    Direction, Fitness, FitnessLabel, RaceConfig, RaceEngine, RaceSnapshot, RunMode, RunSpec,
    StopReason,
};

// ── graph — blueprints + executable networks ─────────────────────────
pub use graph::network::{Network, NetworkOptions};
pub use graph::node::{Activation, CombineOp, Node, NodeKind, StandardizeOp};
pub use graph::topology::{Topology, TopologyOptions};

// ── evolution — the genetic operators ────────────────────────────────
pub use evolution::crossover::CrossoverMethod;
pub use evolution::mutation::MutationMethod;
pub use evolution::selection::SelectionMethod;

// ── trainer — shared deterministic batching ──────────────────────────
pub use trainer::stream::{BatchStream, PoolSplit};
pub use trainer::supervised::TabularTrainer;
pub use trainer::{
    IntoBoxedTrainer, LossFn, ModeTrainer, RlContext, RlStep, RunData, StepEnv, StepReport,
    StepTrainer, StreamShape, TabularContext, TabularStep,
};

// ── data ─────────────────────────────────────────────────────────────
pub use utils::tabular_data::{
    DataFormat, Dataset, load_csv_dataset, load_dataset, load_dataset_auto, load_tensor, make_sine,
    make_xor, one_hot, resolve_dataset, resolve_inputs_targets_datasets,
    resolve_train_test_datasets, save_csv_dataset, save_dataset, save_dataset_as, save_tensor,
};

// ── scoring helpers ──────────────────────────────────────────────────
pub use utils::score::{
    accuracy_score, argmax_classes, cross_entropy_onehot, cross_entropy_onehot_loss,
    f1_from_vecs, f1_score, l1_loss_score, label_smoothing_cross_entropy_loss, mse_loss_score,
    precision_from_vecs, precision_score, r2_score, rmse_score,
};

// ── step primitives ──────────────────────────────────────────────────
pub use utils::race_steps::{
    deterministic_train_step, eval_one_step, seed_step_randomness, train_one_step,
    train_one_step_pred_only,
};

// ── flodl — the tensor backend ───────────────────────────────────────
pub use flodl::{DType, Device, Variable};
// The whole crate, so advanced users can reach the full flodl API
// (nn::Module, tensor::Result, ...) as `gras::flodl::...`.
pub use flodl;

// ── helpers ──────────────────────────────────────────────────────────
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
