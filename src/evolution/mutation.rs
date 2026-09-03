//! Mutation operators — strategies for perturbing a single topology.

use serde::Serialize;

/// Mutation strategy for perturbing a single topology.
/// Each variant carries its own probability. The mutation pool (which
/// activations, combine ops, or standardize ops to pick from) is always
/// taken from the engine-level pools — not per-variant.
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
}

impl Default for MutationMethod {
    fn default() -> Self {
        // Disabled by default — user must call set_mutation() explicitly.
        MutationMethod::Activation { prob: 0.0 }
    }
}

impl MutationMethod {
    /// The mutation probability for this variant.
    pub fn prob(&self) -> f32 {
        match self {
            MutationMethod::Activation { prob }
            | MutationMethod::CombineOp { prob }
            | MutationMethod::Standardize { prob } => *prob,
        }
    }
}
