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
        // 1. Explicit step budget (fires only if set).
        if let Some(max) = self.config.max_steps {
            if step >= max {
                return Some(StopReason::MaxSteps);
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

    /// Build the read-only `RaceSnapshot` for a custom stop: `best`/`worst` are
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
