//! Stop criteria, checked once per step at the loop boundary.
//!
//! Ctrl+C outranks every bar; then the step budget; then the user's
//! pluggable `custom_stop`, which joins the race in addition to the built-ins
//! and receives a read-only [`RaceSnapshot`].

use tracing::info;

use super::config::{RaceSnapshot, StopReason};
use super::core::CoreEngine;

impl CoreEngine {
    /// Check stop criteria at the given step. Returns `Some(reason)` if one
    /// fires, `None` if the run should continue.
    ///
    /// `custom_stop` is independent and always evaluated after the built-ins.
    pub(crate) fn check_stop(&mut self, step: usize) -> Option<StopReason> {
        // 0. Ctrl+C (checked FIRST — the user's intent outranks every bar;
        // abandoning flows through the SAME artifact path as a natural stop).
        // The in-flight step IS completed (we are between steps here), then
        // the whole post-race sequence runs.
        if let Some(flag) = &self.interrupt_flag {
            if flag.load(std::sync::atomic::Ordering::SeqCst) {
                info!(
                    "race: Ctrl+C — shutting down gracefully after step {}",
                    step
                );
                return Some(StopReason::Interrupted);
            }
        }
        // 1. Explicit step budget (fires only if set).
        if let Some(max) = self.config.max_steps {
            if step >= max {
                return Some(StopReason::MaxSteps);
            }
        }
        // 2. Custom stop: joins the race **in addition** to the built-ins, so
        // a user policy can stop the run on criteria the built-ins don't
        // model.
        if let Some(stop) = self.config.custom_stop.as_ref() {
            if stop(&self.snapshot(step)) {
                return Some(StopReason::CustomStop);
            }
        }
        None
    }

    /// Build the read-only `RaceSnapshot` handed to a custom stop closure.
    /// `best`/`worst` are direction-aware (the fitness direction decides which
    /// end of the spread is "best"); `mean` is the plain arithmetic mean.
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
