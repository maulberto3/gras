//! The engine's small value types (`RollOutcome`, `StepEvolve`, `RlVolume`, the no-op optimizer) —
//! split out of `core/mod.rs` by AGENTS.md rule 19.

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

/// What the mutation gate bought at every birth: how DIFFERENT each immigrant was from every
/// blueprint the run had already archived, and how many candidates the draw had to spend to get
/// there. In memory only — `attempts.csv` keeps the per-row ledger, this is the tally the stop
/// card reads, so a run can say "Δ 0.46 per birth, 1.3 of 5 candidates" without parsing a CSV.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BirthLedger {
    /// Immigrants the mutation channel admitted.
    pub births: usize,
    /// Candidates spent across those births.
    pub draws: usize,
    /// Births whose FIRST candidate was refused, so another had to be drawn.
    pub redrawn: usize,
    /// Most candidates any single birth spent.
    pub draws_max: usize,
    /// Sum of Δ-to-nearest-archive over the births that carried a reading.
    pub delta_sum: f64,
    /// Births that carried a Δ reading (the gate was on and the blueprint parsed).
    pub delta_births: usize,
    /// Smallest Δ of any admitted immigrant.
    pub delta_min: Option<f32>,
    /// Crossover rolls discarded because the child duplicated a live topology.
    pub duplicates: usize,
}

impl BirthLedger {
    /// One admitted immigrant: the gate's reading (`None` when it was never judged) and the
    /// candidates that draw spent. A refused candidate is not a birth — it is a draw.
    pub(crate) fn record_draw(&mut self, delta: Option<f32>, draws: usize) {
        self.births += 1;
        self.draws += draws;
        self.draws_max = self.draws_max.max(draws);
        if draws > 1 {
            self.redrawn += 1;
        }
        if let Some(d) = delta {
            self.delta_sum += d as f64;
            self.delta_births += 1;
            self.delta_min = Some(self.delta_min.map_or(d, |m| m.min(d)));
        }
    }

    /// Δ per birth — the gate's headline. `None` until a judged birth lands.
    pub(crate) fn mean_delta(&self) -> Option<f32> {
        (self.delta_births > 0).then(|| (self.delta_sum / self.delta_births as f64) as f32)
    }

    /// Candidates per birth (`0.0` when nothing was born; `1.0` means the first draw always won).
    pub(crate) fn draws_per_birth(&self) -> f32 {
        if self.births == 0 {
            0.0
        } else {
            self.draws as f32 / self.births as f32
        }
    }
}

#[cfg(test)]
mod birth_ledger_tests {
    use super::BirthLedger;

    #[test]
    fn the_ledger_tallies_candidates_and_delta() {
        let mut b = BirthLedger::default();
        assert_eq!(b.mean_delta(), None, "nothing judged ⇒ no headline");
        assert_eq!(b.draws_per_birth(), 0.0, "nothing born ⇒ no rate");
        b.record_draw(Some(0.6), 1); // the first candidate cleared the floor
        b.record_draw(Some(0.3), 3); // two refusals, then admitted
        b.record_draw(None, 1); // gate off: a birth with no reading
        assert_eq!((b.births, b.draws, b.draws_max), (3, 5, 3));
        assert_eq!(
            b.redrawn, 1,
            "only the draw that spent more than one candidate"
        );
        assert!(
            (b.mean_delta().unwrap() - 0.45).abs() < 1e-6,
            "mean over JUDGED births only"
        );
        assert_eq!(b.delta_min, Some(0.3));
        assert!((b.draws_per_birth() - 5.0 / 3.0).abs() < 1e-6);
    }
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

    /// The RL middle column of the rollup: the episode count, the turn identity `turns = train +
    /// eval`, the mean turns per episode, and the per-turn wall cost — `took` alone cannot answer
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
