//! Run lifecycle logging — run start + done message.
//!
//! The run log shows three phases only: initialization (see
//! `crate::utils::init_log`), one table per generation, and this done
//! section. Post-run guidance (next steps, examples, how to analyze
//! individuals) lives in the README, not in the log.

use std::path::Path;
use std::time::Duration;

/// Brief header at run() start. Minimal — most info is in initialization.
pub(crate) fn log_run_start(run_dir: &Path) {
    crate::utils::log_utils::log_section_table(
        "run",
        &["setting", "value"],
        &[vec!["run_dir".to_string(), run_dir.display().to_string()]],
        crate::utils::log_utils::TABLE_WIDTH,
    );
}

/// Log done message after evolution completes.
pub(crate) fn log_done(
    run_elapsed: Duration,
    num_gens: usize,
    run_dir: &Path,
    robustness_csv: &Path,
) {
    let secs = run_elapsed.as_secs_f64();
    let h = secs as u64 / 3600;
    let m = (secs as u64 % 3600) / 60;
    let s = secs - (h * 3600 + m * 60) as f64;
    crate::utils::log_utils::log_section_table(
        "done",
        &["setting", "value"],
        &[
            vec!["stopped".to_string(), format!("max generations reached ({num_gens})")],
            vec!["duration".to_string(), format!("{h}h {m}m {s:.3}s")],
            vec!["run_dir".to_string(), run_dir.display().to_string()],
            vec!["robustness".to_string(), robustness_csv.display().to_string()],
        ],
        crate::utils::log_utils::TABLE_WIDTH,
    );
}
