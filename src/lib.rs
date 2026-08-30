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
pub use utils::{data, synthetic};

// ── re-exports: core types at crate root ──────────────────────────────
pub use crossover::CrossoverMethod;
pub use engine::{Engine, EngineOptions, GenerationStats};
pub use fitness::{BestIndividual, Direction, Fitness, FitnessLabel};
pub use mutation::MutationMethod;
pub use network::{Network, NetworkOptions};
pub use node::{Activation, CombineOp, Node, NodeKind, StandardizeOp};
pub use selection::SelectionMethod;
pub use topology::{Topology, TopologyOptions};
pub use trainer::{OptimizerKind, TrainingConfig};
pub use utils::data::Dataset;

// ── re-exports: scoring helpers ──────────────────────────────────────
pub use fitness::{
    accuracy_score, argmax_classes, cross_entropy_onehot, cross_entropy_onehot_loss, f1_from_vecs,
    f1_score, l1_loss_score, mse_loss_score, precision_from_vecs, precision_score, r2_score,
    rmse_score,
};

// ── device helpers ────────────────────────────────────────────────────

/// Auto-detect the best device based on compiled features.
///
/// - With `cuda` feature: returns `Device::CUDA(0)`
/// - Without `cuda` feature: returns `Device::CPU`
/// Auto-detect the best available device (CUDA if feature enabled, else CPU).
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
