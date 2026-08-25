//! Mutation operators — strategies for perturbing a single topology.

use serde::Serialize;

use crate::node::{Activation, CombineOp, StandardizeOp};

/// Per-individual mutation configuration. One roll per individual;
/// if it hits, a random type is chosen and one random hidden node is mutated.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MutationKind {
    /// Probability of mutating an individual (0.0 = off).
    pub mut_prob: f32,
    pub activation_pool: Vec<Activation>,
    pub combine_pool: Vec<CombineOp>,
    pub standardize_pool: Vec<StandardizeOp>,
    /// Hidden-dim range for dim mutations. Empty -> no dim mutations.
    pub dim_pool: std::ops::RangeInclusive<usize>,
}

impl Default for MutationKind {
    fn default() -> Self {
        MutationKind {
            mut_prob: 0.1,
            activation_pool: vec![],
            combine_pool: vec![],
            standardize_pool: vec![],
            dim_pool: 1..=1,
        }
    }
}
