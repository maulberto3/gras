//! Stop criteria, checked once per step at the loop boundary.

use crate::engine::core::CoreEngine;
use crate::utils::fitness::Direction;

impl CoreEngine {
    /// Check the stop criteria at this step: `Some(reason)` if one fires, `None`
    /// to continue. `custom_stop` is independent and runs after the built-ins.
    pub(crate) fn check_stop(&mut self, step: usize) -> Option<StopReason> {
        // 1.
        if let Some(max) = self.config.stop.max_steps {
            if step + 1 >= max {
                return Some(StopReason::MaxSteps);
            }
        }
        self.check_stop_outside_the_budget(step)
    }

    /// The same criteria at the TOP of a clock, for a RESUMED race whose clock
    pub(crate) fn check_stop_before_clock(&mut self, next: usize) -> Option<StopReason> {
        if let Some(max) = self.config.stop.max_steps {
            if next >= max {
                return Some(StopReason::MaxSteps);
            }
        }
        self.check_stop_outside_the_budget(next)
    }

    /// Ctrl+C, the built-in plateau, then the user's `custom_stop` — the criteria
    /// that do not count clocks and so read the same from either end of one.
    fn check_stop_outside_the_budget(&mut self, step: usize) -> Option<StopReason> {
        // 0. Ctrl+C first — the user's intent outranks every bar, and abandoning
        // flows through the same artifact path as a natural stop (step included).
        if let Some(flag) = &self.interrupt_flag {
            if flag.load(std::sync::atomic::Ordering::SeqCst) {
                // CFY: the interrupt notice is a table, not a one-liner.
                crate::utils::cfy::emit(&crate::utils::cfy::card(
                    "Race",
                    &crate::utils::cfy::kv(&[(
                        "ctrl+c".to_string(),
                        format!("graceful shutdown after step {step}"),
                    )]),
                    None,
                ));
                return Some(StopReason::Interrupted);
            }
        }

        // The snapshot is built ONCE: both predicates read the same numbers, and
        // the plateau's state is a plain field because `check_stop` has `&mut self`.
        let snap = self.snapshot(step);

        // 1. The built-in plateau: no atomics, no closure, no `to_bits`.
        if let Some(plateau) = self.config.stop.plateau.as_mut() {
            if plateau.observe(&snap) {
                return Some(StopReason::CustomStop);
            }
        }

        // 2. Custom stop: joins the race IN ADDITION to the built-ins,
        // criteria they don't model.
        if let Some(stop) = self.config.stop.custom_stop.as_ref() {
            if stop(&snap) {
                return Some(StopReason::CustomStop);
            }
        }
        None
    }

    /// Build the read-only [`RaceSnapshot`] for a custom stop: `best`/`worst`
    /// direction-aware, `mean` is the plain arithmetic mean.
    pub(crate) fn snapshot(&self, step: usize) -> RaceSnapshot {
        let smoothed = self.smoothed_fitness_values();
        let (best, worst, mean) = if smoothed.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let direction = self.fitness.direction();
            let mut best = smoothed[0];
            let mut worst = smoothed[0];
            for &v in &smoothed[1..] {
                if direction.is_better(v, best) {
                    best = v;
                }
                if direction.is_better(worst, v) {
                    worst = v;
                }
            }
            let mean = smoothed.iter().sum::<f32>() / smoothed.len() as f32;
            (best, worst, mean)
        };
        RaceSnapshot {
            live_count: self.state.live_count(),
            step,
            best_smoothed_fitness: best,
            worst_smoothed_fitness: worst,
            mean_smoothed_fitness: mean,
            culls: self.culls,
            elapsed_seconds: self.elapsed_base_secs + self.started_at_wall.elapsed().as_secs(),
            direction: self.fitness.direction(),
        }
    }
}

/// What the scheduler checks to decide whether to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The configured step budget was reached.
    MaxSteps,
    /// A stop predicate returned true (the built-in plateau or `custom_stop`).
    CustomStop,
    /// Ctrl+C received: the step was abandoned at the loop boundary and the race shut
    /// down through the SAME artifact path. Always
    Interrupted,
}

/// Default step budget, read at the same conservative scale as the population: `10` clocks.
pub const DEFAULT_MAX_STEPS: usize = 10;

/// Type alias for the pluggable stop closure. Plain alias, so `RaceConfig` derives
/// neither `Clone` nor `Debug`.
pub type StopFn = Option<Box<dyn Fn(&RaceSnapshot) -> bool + Send + Sync + 'static>>;

/// A cheap read-only snapshot of the live race state at decision time, passed to a custom
/// stop closure. Not a full run dump.
#[derive(Clone)]
pub struct RaceSnapshot {
    /// Live nets at decision time.
    pub live_count: usize,
    /// Current step.
    pub step: usize,
    /// Best RANKING score — the window centre in the run's mean family, plus any
    /// `selection_metrics` terms. This is what the race actually ranks on.
    pub best_smoothed_fitness: f32,
    /// Worst ranking score — the same number, at the other end.
    pub worst_smoothed_fitness: f32,
    /// Mean of the live ranking scores (a plain arithmetic mean OF the scores,
    /// second smoothing pass).
    pub mean_smoothed_fitness: f32,
    /// Cumulative culls.
    pub culls: usize,
    /// Wall-clock seconds elapsed.
    pub elapsed_seconds: u64,
    /// Which end of the score is better, so a predicate can read `best` with the
    /// run's own sense of "improved".
    pub direction: Direction,
}

/// Default tolerance for "the best score moved": a reading within `0.01` of the
/// record does not restart the patience window.
pub const DEFAULT_PLATEAU_EPSILON: f32 = 1e-2;

/// Built-in "stop when the score stops moving" criterion.
///
/// The engine owns the two numbers a closure used to hide in atomics: the best
/// score seen (`best`) and the clock it was last raised on (`held_at`). Once
/// `best` reaches `target`, the race stops after `patience` clocks without an
/// improvement larger than `epsilon`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plateau {
    target: f32,
    patience: usize,
    epsilon: f32,
    best: f32,
    held_at: usize,
    armed: bool,
}

impl Plateau {
    /// A plateau stop at `target`, tolerating `patience` still clocks after it is reached.
    pub fn new(target: f32, patience: usize) -> Self {
        Self {
            target,
            patience,
            epsilon: DEFAULT_PLATEAU_EPSILON,
            best: 0.0,
            held_at: 0,
            armed: false,
        }
    }

    /// Change the "this counts as movement" tolerance (default [`DEFAULT_PLATEAU_EPSILON`]).
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon.max(0.0);
        self
    }

    /// Forget the high-water mark; the next observation re-arms from its own step.
    pub fn reset(&mut self) {
        self.armed = false;
    }

    /// Feed one clock's snapshot; `true` means the race should stop.
    pub fn observe(&mut self, snap: &RaceSnapshot) -> bool {
        let best = snap.best_smoothed_fitness;
        if !self.armed {
            self.best = best;
            self.held_at = snap.step;
            self.armed = true;
        } else {
            let improved = match snap.direction {
                Direction::Maximize => best > self.best + self.epsilon,
                Direction::Minimize => best < self.best - self.epsilon,
            };
            if improved {
                self.best = best;
                self.held_at = snap.step;
            }
        }

        let reached = match snap.direction {
            Direction::Maximize => self.best >= self.target,
            Direction::Minimize => self.best <= self.target,
        };
        reached && snap.step.saturating_sub(self.held_at) >= self.patience
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(step: usize, best: f32, direction: Direction) -> RaceSnapshot {
        RaceSnapshot {
            live_count: 1,
            step,
            best_smoothed_fitness: best,
            worst_smoothed_fitness: best,
            mean_smoothed_fitness: best,
            culls: 0,
            elapsed_seconds: 0,
            direction,
        }
    }

    #[test]
    fn plateau_waits_for_patience_after_the_target() {
        let mut p = Plateau::new(0.8, 2);
        assert!(!p.observe(&snap(0, 0.5, Direction::Maximize)));
        assert!(!p.observe(&snap(1, 0.81, Direction::Maximize)));
        assert!(!p.observe(&snap(2, 0.81, Direction::Maximize)));
        assert!(p.observe(&snap(3, 0.81, Direction::Maximize)));
    }

    #[test]
    fn plateau_reads_minimize_runs_the_same_way() {
        let mut p = Plateau::new(0.2, 2);
        assert!(!p.observe(&snap(0, 0.5, Direction::Minimize)));
        assert!(!p.observe(&snap(1, 0.19, Direction::Minimize)));
        assert!(!p.observe(&snap(2, 0.19, Direction::Minimize)));
        assert!(p.observe(&snap(3, 0.19, Direction::Minimize)));
    }
}
