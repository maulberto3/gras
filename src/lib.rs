pub mod utils;
pub mod engine;
pub mod fitness;
pub mod selection;
pub mod network;
pub mod node;
pub mod spec;
pub mod topology;
pub mod trainer;

// Re-export utils submodules at crate root for backward compatibility.
pub use utils::{data, synthetic};

// ── re-exports: core types at crate root ──────────────────────────────
// Users can write `use gras::Engine` instead of `use gras::engine::Engine`.
pub use engine::{Engine, EngineOptions, Fitness};
pub use fitness::{Direction, FitnessLabel};
pub use network::Network;
pub use node::{Activation, Node};
pub use topology::{CombineOp, Topology, TopologyOptions};
