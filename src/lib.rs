pub mod engine;
pub mod fitness;
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
pub use engine::{Engine, EngineOptions, Fitness}; // example: use gras::Engine; (instead of gras::engine::Engine)
pub use fitness::{BestIndividual, Direction, FitnessLabel};
pub use network::Network;
pub use node::{Activation, CombineOp, Node, NodeKind, StandardizeOp};
pub use selection::SelectionMethod;
pub use topology::{Topology, TopologyOptions};
pub use trainer::{OptimizerKind, TrainingConfig};

// ── re-exports: scoring helpers ──────────────────────────────────────
pub use fitness::{
    accuracy_score, argmax_classes, cross_entropy_onehot, cross_entropy_onehot_loss, f1_from_vecs,
    f1_score, l1_loss_score, mse_loss_score, precision_from_vecs, precision_score, r2_score,
    rmse_score,
};
