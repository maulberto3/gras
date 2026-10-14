//! Crossover operators — strategies for combining two parent topologies.

use serde::Serialize;

/// Crossover strategy for combining two parent topologies.
/// Each variant carries its own action probability.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum CrossoverMethod {
    /// Swap a segment at a matching-node pivot (requires same num_inputs/num_outputs on the pivot).
    OnePoint { action_prob: f32 },
    /// Per-node independent swap (requires same-length hidden nodes).
    Uniform { action_prob: f32, swap_prob: f32 },
}

impl Default for CrossoverMethod {
    /// Uniform at 0.7 action / 0.5 swap — the engine's default.
    fn default() -> Self {
        CrossoverMethod::Uniform {
            action_prob: 0.7,
            swap_prob: 0.5,
        }
    }
}

impl CrossoverMethod {
    /// The per-roll probability this strategy acts (its own knob).
    pub fn action_prob(&self) -> f32 {
        match self {
            CrossoverMethod::OnePoint { action_prob, .. }
            | CrossoverMethod::Uniform { action_prob, .. } => *action_prob,
        }
    }
}
