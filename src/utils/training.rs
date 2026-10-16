//! The small training-facing vocabulary the run loop and any scheme **share**:
//! how far a run has progressed, and the default learning rate.
//!
//! These live below both `engine` and `trainer` (see [`crate::utils`]) on
//! purpose — a training scheme, including one a user writes, reaches them
//! without importing the engine.

/// Fraction of the step budget consumed (`0.0` → `1.0`, clamped), or `None` when
/// unbounded/degenerate. Pure arithmetic on the clock, so a replay always agrees.
pub fn run_progress(step: usize, max_steps: Option<usize>) -> Option<f32> {
    let max = max_steps?;
    if max == 0 {
        return None;
    }
    Some((step as f32 / max as f32).min(1.0))
}

/// Learning rate default for the Adam optimizers a scheme builds.
pub const DEFAULT_LR: f32 = 1e-3;

#[cfg(test)]
mod tests {
    use super::{DEFAULT_LR, run_progress};

    /// One progress convention for the whole crate: linear to 1.0 at the budget,
    /// clamped, and flat when unbounded.
    #[test]
    fn run_progress_clamps_and_is_flat_unbounded() {
        assert_eq!(run_progress(0, Some(10)), Some(0.0));
        assert_eq!(run_progress(5, Some(10)), Some(0.5));
        assert_eq!(run_progress(10, Some(10)), Some(1.0));
        // Overshoot clamps: a run resumed past its budget must not go negative.
        assert_eq!(run_progress(99, Some(10)), Some(1.0));
        // No budget (or a degenerate one) = no schedule at all.
        assert_eq!(run_progress(3, None), None);
        assert_eq!(run_progress(3, Some(0)), None);
    }

    #[test]
    fn default_lr_is_the_documented_adam_default() {
        assert_eq!(DEFAULT_LR, 1e-3);
    }
}
