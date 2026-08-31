//! Engine logging — pretty-printed console output.
//!
//! Every log section maps to one engine phase:
//!   - initialization: options, dataset, population (shown once at new())
//!   - run:            brief header (shown at run() start)
//!   - improvements:   real-time per gen (shown during evaluation)
//!   - done:           stop message + robustness CSV path + next steps

use std::path::Path;
use std::time::Duration;

use crate::engine::EngineOptions;
use crate::fitness::Direction;
use crate::topology::Topology;

// ── Initialization ───────────────────────────────────────────────────────

/// Single section logging all resolved options and population.
/// Shown once at `Engine::new()`. All defaults are visible here.
pub(crate) fn log_initialization(
    options: &EngineOptions,
    pop: &[Topology],
    seed: u64,
    fitness: &crate::fitness::Fitness,
) {
    let fl = fitness.label();
    let score_dir = fitness.direction();
    let score_better = match score_dir {
        Direction::Minimize => "lower = better",
        Direction::Maximize => "higher = better",
    };

    log::info!("");
    log::info!("══ initialization ═══════════════════════════════════════════════════");

    log::info!(
        "  engine    pop {} · {} gens",
        options.pop_size.unwrap_or(0),
        options.num_generations.unwrap_or(0)
    );
    log::info!("  seed      {seed}  (set_seed(Some({seed})) to reproduce)");
    log::info!("  threads   {}", options.num_threads);
    log::info!("  elite     {}", options.elite_count);
    if options.dedup_pop_and_fill {
        log::info!("  dedup     on (full topology comparison)");
    }
    log::info!("  results   {}", options.results_dir.display());

    if !options.prior_topology_paths.is_empty() {
        log::info!("  warm      {} prior topologies", options.prior_topology_paths.len());
        for (i, path) in options.prior_topology_paths.iter().enumerate() {
            log::info!("  warm[{i}]   {}", path.display());
        }
    }

    log::info!("  fitness   {fl} · {score_better}");

    if let Some(ref sel) = options.selection {
        log::info!("  selection {}", sel);
    }
    if let Some(ref cx) = options.crossover {
        log::info!("  crossover {}", cx);
    }
    log::info!("  mutation  {}", options.mutation);

    log::info!("  input     {} · output {}", options.topology_options.input_dim, options.topology_options.output_dim);

    // Topology
    log::info!(
        "  topo      input {} → hidden (pool {}..={} stride {}) → output {}",
        options.topology_options.input_dim,
        options.hidden_dim_pool.start(),
        options.hidden_dim_pool.end(),
        options.hidden_dim_stride,
        options.topology_options.output_dim
    );
    log::info!(
        "  topo      hidden {}..={} nodes · in {}..={} · out {}..={}",
        options.topology_options.min_hidden_num_nodes,
        options.topology_options.max_hidden_num_nodes,
        options.topology_options.min_hidden_inputs_per_node,
        options.topology_options.max_hidden_inputs_per_node,
        options.topology_options.min_hidden_outputs_per_node,
        options.topology_options.max_hidden_outputs_per_node    );

    if options.dropout_prob > 0.0 {
        log::info!("  dropout   {}%", (options.dropout_prob * 100.0) as usize);
    }

    // GP pools
    let act_names: Vec<String> = options
        .activation_pool
        .iter()
        .map(|a| a.to_string())
        .collect();
    let std_names: Vec<String> = options
        .standardize_op_pool
        .iter()
        .map(|s| s.to_string())
        .collect();
    log::info!("  pools     hidden {:?} stride {}", options.hidden_dim_pool, options.hidden_dim_stride);
    log::info!("  pools     combine {:?}", options.combine_op_pool);
    log::info!("  pools     activations [{}]", act_names.join(", "));
    log::info!("  pools     standardize [{}]", std_names.join(", "));

    // Population summary
    let min_nodes = pop.iter().map(|g| g.nodes.len()).min().unwrap_or(0);
    let max_nodes = pop.iter().map(|g| g.nodes.len()).max().unwrap_or(0);
    log::info!(
        "  pop       {} individuals · {}-{} nodes",
        pop.len(),
        min_nodes,
        max_nodes
    );
}

// ── Run start ───────────────────────────────────────────────────────────

/// Brief header at run() start. Minimal — most info is in initialization.
pub(crate) fn log_run_start(run_dir: &Path) {
    log::info!("");
    log::info!("══ run ════════════════════════════════════════════════════════════");
    log::info!("  run_dir    {}", run_dir.display());
}

// ── Run summary ─────────────────────────────────────────────────────────

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
    log::info!("  cargo run --example generate_md_from_gen -- {} <topo_id>", run_dir.display());
    log::info!("");
    log::info!("  # Convert .bin data to CSV for inspection");
    log::info!("  cargo run --example bin_to_csv -- data/mnist/train");
}

/// Print the top-N repeated topologies.
///
/// `best = true`  → top N (appearances desc, std_dev asc, mean desc)
/// `best = false` → bottom N (appearances asc, std_dev desc, mean asc)
pub(crate) fn log_repeated_topologies(
    robustness: &std::collections::HashMap<String, crate::engine::RobustnessEntry>,
    top_n: usize,
    best: bool,
) {
    let mut entries: Vec<&crate::engine::RobustnessEntry> = robustness.values()
        .filter(|e| e.count > 1)
        .collect();

    if best {
        entries.sort_by(|a, b| {
            b.count.cmp(&a.count)
                .then_with(|| a.std_dev().partial_cmp(&b.std_dev()).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| b.mean.partial_cmp(&a.mean).unwrap_or(std::cmp::Ordering::Equal))
        });
    } else {
        entries.sort_by(|a, b| {
            a.count.cmp(&b.count)
                .then_with(|| b.std_dev().partial_cmp(&a.std_dev()).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.mean.partial_cmp(&b.mean).unwrap_or(std::cmp::Ordering::Equal))
        });
    }

    if entries.is_empty() {
        return;
    }

    let n = top_n.min(entries.len());
    let label = if best { "top" } else { "bottom" };
    let has_loss = entries.first().map_or(false, |e| e.has_loss());
    log::info!("");
    log::info!("── repeated topologies ({label} {n}) ──");
    if has_loss {
        log::info!("  {:<6} {:>11} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}  {}",
            "rank", "appearances", "fit_mean", "fit_sd", "fit_min", "fit_max", "loss_mean", "loss_sd", "loss_min", "loss_max", "params", "topo_id");
    } else {
        log::info!("  {:<6} {:>11} {:>9} {:>9} {:>9} {:>9} {:>10}  {}",
            "rank", "appearances", "mean", "std_dev", "min", "max", "params", "topo_id");
    }

    for (rank, entry) in entries.iter().take(n).enumerate() {
        let tid = topo_hash(&entry.topology_json);
        if has_loss {
            log::info!("  {:<6} {:>11} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>10}  {}",
                format!("#{}", rank + 1),
                entry.count,
                entry.mean,
                entry.std_dev(),
                entry.min_fitness,
                entry.max_fitness,
                entry.mean_loss.unwrap_or(0.0),
                entry.std_dev_loss(),
                entry.min_loss.unwrap_or(0.0),
                entry.max_loss.unwrap_or(0.0),
                entry.param_count,
                tid,
            );
        } else {
            log::info!("  {:<6} {:>11} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>10}  {}",
                format!("#{}", rank + 1),
                entry.count,
                entry.mean,
                entry.std_dev(),
                entry.min_fitness,
                entry.max_fitness,
                entry.param_count,
                tid,
            );
        }
    }
}

/// xxh3 hash — 16 hex chars, deterministic, near-zero collisions.
pub(crate) fn topo_hash(topology_json: &str) -> String {
    format!("{:016x}", xxhash_rust::xxh3::xxh3_64(topology_json.as_bytes()))
}
