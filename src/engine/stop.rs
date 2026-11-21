//! Stop criteria, checked once per step at the loop boundary.

use super::core::CoreEngine;

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
        // 2. Custom stop: joins the race IN ADDITION to the built-ins,
        // criteria they don't model.
        if let Some(stop) = self.config.stop.custom_stop.as_ref() {
            if stop(&self.snapshot(step)) {
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
        }
    }
}

/// What the scheduler checks to decide whether to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The configured step budget was reached.
    MaxSteps,
    /// A user-supplied `custom_stop` closure returned true (Iter 5 pluggable
    /// contract; consulted after all built-ins).
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
}
