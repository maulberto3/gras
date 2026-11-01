//! The engine's small value types (`RollOutcome`, `StepEvolve`, `RlVolume`, the
//! no-op optimizer) — split out of `core/mod.rs` by AGENTS.md rule 19. `use
//! super::*` keeps every name the impls were written against.

use super::*;

/// One evolution roll's outcome, formatted for the single per-roll log line.
/// `survived`: crossover-only — did the child get inserted?
pub(crate) struct RollOutcome {
    /// Crossover-only: did the child get inserted?
    pub survived: bool,
    /// No cull-eligible slot existed (hard probation shield): the roll ended WITHOUT
    /// generating or inserting a child. `run` tells this apart from a spent roll.
    pub no_slot: bool,
    /// Human-readable outcome, rendered as the per-roll log line.
    pub detail: String,
}
impl RollOutcome {
    /// A crossover child that was admitted.
    pub fn survived(detail: impl Into<String>) -> Self {
        Self {
            survived: true,
            no_slot: false,
            detail: detail.into(),
        }
    }
    /// A roll that paid its attempts and produced no child.
    pub fn spent(detail: impl Into<String>) -> Self {
        Self {
            survived: false,
            no_slot: false,
            detail: detail.into(),
        }
    }
    /// A roll skipped: no cull-eligible victim existed this clock.
    pub fn no_slot(detail: impl Into<String>) -> Self {
        Self {
            survived: false,
            no_slot: true,
            detail: detail.into(),
        }
    }
}

/// Per-step evolve counters, reused by the `Minimal` table. Reset each step.
#[derive(Clone, Copy, Default)]
pub(crate) struct StepEvolve {
    /// Nets culled this step.
    pub culls: usize,
    /// Nets inserted this step (crossover survivors + immigrants).
    pub inserts: usize,
    /// Crossover rolls that fired this step.
    pub cross_fired: usize,
    /// Fired crossover rolls whose child was admitted.
    pub cross_survived: usize,
    /// Fired crossover rolls whose child was discarded.
    pub cross_discarded: usize,
    /// Immigrant rolls that fired this step.
    pub mutate_fired: usize,
}

/// A do-nothing optimizer — the freeze pass runs the net's normal trainer step
/// through this: gradients compute, nothing applies, the real optimizer is untouched.
pub(crate) struct NoopOptimizer;

impl flodl::nn::optim::Optimizer for NoopOptimizer {
    fn step(&mut self) -> flodl::tensor::Result<()> {
        Ok(()) // gradients accumulate; nothing applies
    }
    fn zero_grad(&self) {}
    fn lr(&self) -> f64 {
        0.0
    }
    fn set_lr(&mut self, _lr: f64) {}
}

/// Per-step RL volume summed over every net that stepped this clock — the env work
/// actually done. `nets` counts reporters (a silent trainer shows `—`), zero in Tabular.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RlVolume {
    /// Nets that reported `StepReport.rl` this step.
    pub nets: usize,
    /// Matches played by those nets, summed (train + eval).
    pub matches: usize,
    /// Train turns played by those nets, summed.
    pub train_turns: usize,
    /// Eval turns played by those nets, summed.
    pub eval_turns: usize,
}

impl RlVolume {
    /// Total environment turns this step (train + eval) — the `turns/match`
    /// numerator.
    pub(crate) fn turns(&self) -> usize {
        self.train_turns + self.eval_turns
    }

    /// The RL middle column of the rollup: the episode count, the turn identity
    /// `turns = train + eval`, the mean turns per episode, and the per-turn wall
    /// cost — `took` alone cannot answer "did the engine get slower?".
    pub(crate) fn label(&self, step_secs: f32) -> String {
        if self.nets == 0 || self.matches == 0 {
            return "episodes — │ turns — = train — + eval — │ turns/episode — │ ms/turn —"
                .to_string();
        }
        let turns = self.turns();
        let per_turn = if turns == 0 {
            "—".to_string()
        } else {
            fmt_line(step_secs * 1000.0 / turns as f32)
        };
        format!(
            "episodes {} │ turns {} = train {} + eval {} │ turns/episode {} │ {} ms/turn",
            self.matches,
            turns,
            self.train_turns,
            self.eval_turns,
            fmt_line(turns as f32 / self.matches as f32),
            per_turn
        )
    }
}

