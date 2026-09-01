pub mod crossover;
pub mod engine;
pub mod fitness;
pub mod mutation;
pub mod network;
pub mod node;
pub mod pools;
pub mod selection;
pub mod spec;
pub mod topology;
pub mod trainer;
pub mod utils;

// Re-export utils submodules at crate root for backward compatibility.
pub use utils::data;

// ── re-exports: core types at crate root ──────────────────────────────
pub use crossover::CrossoverMethod;
pub use engine::{Engine, EngineOptions, GenerationStats};
pub use fitness::{Direction, Fitness, FitnessLabel};
pub use mutation::MutationMethod;
pub use network::{Network, NetworkOptions};
pub use node::{Activation, CombineOp, Node, NodeKind, StandardizeOp};
pub use selection::SelectionMethod;
pub use topology::{Topology, TopologyOptions};
pub use trainer::{OptimizerKind, Trainer, TrainingConfig, SupervisedTrainer};
pub use utils::data::{Dataset, make_sine, make_xor, one_hot};
pub use utils::supervised::TrainResult;

// ── re-exports: scoring helpers ──────────────────────────────────────
pub use utils::scoring::{
    accuracy_score, argmax_classes, cross_entropy_onehot, cross_entropy_onehot_loss, f1_from_vecs,
    f1_score, l1_loss_score, mse_loss_score, precision_from_vecs, precision_score, r2_score,
    rmse_score,
};

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
