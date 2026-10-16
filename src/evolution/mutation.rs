//! Mutation operators — strategies for perturbing a single topology.

use serde::Serialize;

/// Mutation strategy for perturbing a single topology. Each variant carries its
/// own probability; the op pools come from the engine, never per-variant.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum MutationMethod {
    /// Swap a random hidden node's activation.
    Activation {
        /// Probability of mutating an individual (0.0 = off).
        prob: f32,
    },
    /// Swap a random hidden node's combine op.
    CombineOp { prob: f32 },
    /// Swap a random hidden node's standardize op.
    Standardize { prob: f32 },
    /// Redraw one non-primary output port's activation (port >= 1) of a random
    /// hidden node; port 0 mirrors the node-level activation by convention.
    PortActivation { prob: f32 },
}

impl Default for MutationMethod {
    /// Disabled by default — the user opts in via `set_mutation()`.
    fn default() -> Self {
        MutationMethod::Activation { prob: 0.0 }
    }
}

impl MutationMethod {
    /// The mutation probability for this variant.
    pub fn prob(&self) -> f32 {
        match self {
            MutationMethod::Activation { prob }
            | MutationMethod::CombineOp { prob }
            | MutationMethod::Standardize { prob }
            | MutationMethod::PortActivation { prob } => *prob,
        }
    }
}
