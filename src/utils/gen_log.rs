//! Run lifecycle logging — run start, done message, next steps.

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
    log_next_steps(run_dir);
}

/// Print next steps and examples reference.
fn log_next_steps(run_dir: &Path) {
    crate::utils::log_utils::log_section_table(
        "next steps",
        &["purpose", "command"],
        &[
            vec![
                "see which topologies truly performed well across generations".to_string(),
                "engine.show_robustness(10)".to_string(),
            ],
            vec![
                "top 20 best".to_string(),
                "engine.show_robustness(20, \"best\")".to_string(),
            ],
            vec![
                "bottom 20 worst".to_string(),
                "engine.show_robustness(20, \"worst\")".to_string(),
            ],
        ],
        crate::utils::log_utils::TABLE_WIDTH,
    );
    let run_id = run_dir
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    crate::utils::log_utils::log_section_table(
        "examples",
        &["purpose", "command"],
        &[
            vec![
                "fully train a specific network from a gen_XX.json snapshot".to_string(),
                format!("cargo run --example train_from_gen -- {run_id}/improvements/gen_00.json --best"),
            ],
            vec![
                "custom trainer implementing the Trainer trait (early stopping)".to_string(),
                "cargo run --example custom_trainer".to_string(),
            ],
            vec![
                "quick categorical / continuous showcase".to_string(),
                "cargo run --example categorical_showcase".to_string(),
            ],
        ],
        crate::utils::log_utils::TABLE_WIDTH,
    );
}