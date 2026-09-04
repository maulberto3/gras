//! gras — neural architecture search over random topologies.
//!
//! The engine evolves graph blueprints ([`Topology`]) generation by
//! generation — seed, score, select, crossover, mutate — and compiles the
//! winners into executable [`Network`]s. Bring your own [`Trainer`]:
//! wrap a closure with [`trainer::from_fn`], or implement the trait for
//! fully custom training loops.
//!
//! Quick start:
//! ```
//! # use gras::*;
//! let opts = EngineOptions::builder()
//!     .set_pop_size(20)
//!     .set_num_generations(5)
//!     .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
//!     .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.25 })
//!     .set_mutation(MutationMethod::Activation { prob: 0.1 })
//!     .build()
//!     .unwrap();
//! ```

// ── module tree ──────────────────────────────────────────────────────
pub mod engine;
pub mod evolution;
pub mod graph;
pub mod spec;
pub mod trainer;
pub mod utils;

// ── flat paths — the folders are an implementation detail ─────────────
pub use engine::fitness;
pub use evolution::{crossover, mutation, pools, selection};
pub use graph::{network, node, topology};
pub use utils::data;

// ── engine ───────────────────────────────────────────────────────────
pub use engine::{Engine, EngineOptions, GenerationStats, RobustnessFilter};
pub use engine::fitness::{Direction, Fitness, FitnessLabel};

// ── graph — blueprints + executable networks ─────────────────────────
pub use graph::network::{Network, NetworkOptions};
pub use graph::node::{Activation, CombineOp, Node, NodeKind, StandardizeOp};
pub use graph::topology::{Topology, TopologyOptions};

// ── evolution — the genetic operators ────────────────────────────────
pub use evolution::crossover::CrossoverMethod;
pub use evolution::mutation::MutationMethod;
pub use evolution::selection::SelectionMethod;

// ── trainer — the primary extension point ────────────────────────────
// The built-in SupervisedTrainer + TrainingConfig live at
// `trainer::supervised` — an optional convenience, not the default.
pub use trainer::{EvalOutcome, Trainer};
pub use utils::supervised::TrainResult;

// ── data ─────────────────────────────────────────────────────────────
pub use utils::data::{
    DataFormat, Dataset, load_csv_dataset, load_dataset, load_dataset_auto, load_tensor,
    make_sine, make_xor, one_hot, resolve_dataset, save_csv_dataset, save_dataset,
    save_dataset_as, save_tensor,
};

// ── scoring helpers ──────────────────────────────────────────────────
pub use utils::scoring::{
    accuracy_score, argmax_classes, cross_entropy_onehot, cross_entropy_onehot_loss, f1_from_vecs,
    f1_score, l1_loss_score, mse_loss_score, precision_from_vecs, precision_score, r2_score,
    rmse_score,
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