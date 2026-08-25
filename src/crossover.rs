//! Crossover operators — strategies for combining two parent topologies.

use serde::Serialize;

/// Crossover strategy for combining two parent topologies.
/// Each variant carries its own action probability.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum CrossoverKind {
    /// Swap a segment at a matching-node pivot (requires same num_inputs/num_outputs on the pivot).
    TwoPoint { action_prob: f32 },
    /// Per-node independent swap (requires same-length hidden nodes).
    Uniform { action_prob: f32, swap_prob: f32 },
}

impl Default for CrossoverKind {
    fn default() -> Self {
        CrossoverKind::Uniform {
            action_prob: 0.7,
            swap_prob: 0.5,
        }
    }
}

impl CrossoverKind {
    pub fn action_prob(&self) -> f32 {
        match self {
            CrossoverKind::TwoPoint { action_prob, .. }
            | CrossoverKind::Uniform { action_prob, .. } => *action_prob,
        }
    }
}
