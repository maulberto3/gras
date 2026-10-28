//! Stop criteria, checked once per step at the loop boundary.
//!
//! Ctrl+C outranks every bar; then the step budget; then the user's
//! pluggable `custom_stop`, which joins the race in addition to the built-ins
//! and receives a read-only [`RaceSnapshot`].

use super::config::{RaceSnapshot, StopReason};
use super::core::CoreEngine;

impl CoreEngine {
    /// Check the stop criteria at this step: `Some(reason)` if one fires, `None`
    /// to continue. `custom_stop` is independent and runs after the built-ins.
    pub(crate) fn check_stop(&mut self, step: usize) -> Option<StopReason> {
        // 1. Explicit step budget (fires only if set). Budget = clocks RUN:
        // `step` counts clocks COMPLETED (0-based), and the check runs at the
        // END of each clock, so `step + 1 >= max` fires after exactly N clocks
        // for `max_steps = N`. The pre-1.1.8 arithmetic (`step >= max`) ran
        // N+1 clocks — see TODO.md history and DONE.md's stop line.
        if let Some(max) = self.config.max_steps {
            if step + 1 >= max {
                return Some(StopReason::MaxSteps);
            }
        }
        self.check_stop_outside_the_budget(step)
    }

    /// The same criteria at the TOP of a clock, for a RESUMED race whose clock
    /// came from disk: `next` is the clock ABOUT to run, so the budget is spent
    /// when `next >= max` — not `next + 1 >= max`.
    ///
    /// One clock is the whole difference, and it is not cosmetic: reusing
    /// [`Self::check_stop`] here made `max_steps = 4` on a run resumed at clock 2
    /// end at clock 2 (three clocks total) instead of clock 3, because the
    /// pre-flight consumed the last allowed clock without running it. Ctrl+C and
    /// `custom_stop` keep their meaning across the boundary.
    pub(crate) fn check_stop_before_clock(&mut self, next: usize) -> Option<StopReason> {
        if let Some(max) = self.config.max_steps {
            if next >= max {
                return Some(StopReason::MaxSteps);
            }
        }
        self.check_stop_outside_the_budget(next)
    }

    /// Ctrl+C, then the user's `custom_stop` — the criteria that do not count
    /// clocks and so read the same from either end of one.
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
        // 2. Custom stop: joins the race IN ADDITION to the built-ins, for
        // criteria they don't model.
        if let Some(stop) = self.config.custom_stop.as_ref() {
            if stop(&self.snapshot(step)) {
                return Some(StopReason::CustomStop);
            }
        }
        None
    }

    /// Build the read-only [`RaceSnapshot`] for a custom stop: `best`/`worst` are
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
        }
    }
}
