//! Run lifecycle logging — run start, done message, next steps.

use std::path::Path;
use std::time::Duration;

/// Brief header at run() start. Minimal — most info is in initialization.
pub(crate) fn log_run_start(run_dir: &Path) {
    log::info!("");
    log::info!("══ run ════════════════════════════════════════════════════════════");
    log::info!("  run_dir    {}", run_dir.display());
}

/// Log done message after evolution completes.
pub(crate) fn log_done(
    run_elapsed: Duration,
    num_gens: usize,
    run_dir: &Path,
    robustness_csv: &Path,
) {
    log::info!("");
    log::info!("══ done ════════════════════════════════════════════════════════════");
    log::info!("  stopped    max generations reached ({num_gens})");
    let secs = run_elapsed.as_secs_f64();
    let h = secs as u64 / 3600;
    let m = (secs as u64 % 3600) / 60;
    let s = secs - (h * 3600 + m * 60) as f64;
    log::info!("  duration   {h}h {m}m {s:.3}s");
    log::info!("  run_dir    {}", run_dir.display());
    log::info!("  robustness {}", robustness_csv.display());
    log_next_steps(run_dir);
}

/// Print next steps and examples reference.
fn log_next_steps(run_dir: &Path) {
    log::info!("");
    log::info!("══ next steps ══════════════════════════════════════════════════════");
    log::info!("  # See which topologies truly performed well across generations");
    log::info!("  engine.show_robustness(10)              # top 10 repeated");
    log::info!("  engine.show_robustness(20, \"best\")     # top 20 best");
    log::info!("  engine.show_robustness(20, \"worst\")    # bottom 20 worst");
    log::info!("");
    log::info!("══ examples ══════════════════════════════════════════════════════");
    log::info!("  # Generate MD for a specific topology by its ID (from robustness CSV)");
    let run_id = run_dir.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    log::info!("  cargo run --example generate_md_from_gen -- {} <topo_id>", run_id);
    log::info!("");
    log::info!("  # Convert .bin data to CSV for inspection");
    log::info!("  cargo run --example bin_to_csv -- <data_dir>");
}
