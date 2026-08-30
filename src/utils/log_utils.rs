//! Engine logging — pretty-printed console output.
//!
//! Every log section maps to one engine phase:
//!   - initialization: options, dataset, population (shown once at new())
//!   - run:            brief header (shown at run() start)
//!   - improvements:   real-time per best (shown during evaluation)
//!   - run summary:    winner + artifacts + rebuild (shown after run())

use std::path::Path;
use std::time::Duration;

use flodl::Device;
use flodl::nn::Module;

use super::error::EngineError;
use crate::engine::{BestIndividual, EngineOptions};
use crate::fitness::Direction;
use crate::network::Network;
use crate::topology::Topology;

/// Hash a topology JSON string — returns first 8 hex chars of DefaultHasher digest.
fn topo_hash(topology_json: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    topology_json.hash(&mut hasher);
    format!("{:016x}", hasher.finish())[..8].to_string()
}
// ── Initialization ───────────────────────────────────────────────────────

/// Single section logging all resolved options, dataset, and population.
/// Shown once at `Engine::new()`. All defaults are visible here.
pub(crate) fn log_initialization(
    options: &EngineOptions,
    dataset: &super::data::Dataset,
    pop: &[Topology],
    seed: u64,
    fitness: &crate::fitness::Fitness,
    data_path: &std::path::Path,
) {
    let fl = fitness.fitness_label();
    let ll = fitness.train_metric_label();
    let score_dir = fitness.direction();
    let train_dir = fitness.train_metric_direction();
    let score_better = match score_dir {
        Direction::Minimize => "lower = better",
        Direction::Maximize => "higher = better",
    };
    let train_better = match train_dir {
        Direction::Minimize => "lower = better",
        Direction::Maximize => "higher = better",
    };

    log::info!("");
    log::info!("══ initialization ═══════════════════════════════════════════════════");

    // Engine core
    log::info!(
        "  engine    pop {} · {} gens",
        options.pop_size.unwrap_or(0),
        options.num_generations.unwrap_or(0)
    );
    log::info!("  seed      {seed}  (set_seed(Some({seed})) to reproduce)");

    // Data split
    log::info!(
        "  split     {:.0}% train / {:.0}% test",
        options.train_test_split.0 * 100.0,
        options.train_test_split.1 * 100.0
    );

    // Batch budget
    log::info!(
        "  budget    train {}bt × {} · test {}bt × {}",
        options.train_num_batches,
        options.train_batch_size,
        options.test_num_batches,
        options.test_batch_size
    );

    // Data & coefs randomization
    log::info!(
        "  random    data train={} test={}  coefs {}",
        if options.train_random_data { "yes" } else { "no" },
        if options.test_random_data { "yes" } else { "no" },
        if options.train_fixed_coefs { "fixed" } else { "rand" }
    );

    // Engine knobs
    log::info!("  threads   {}", options.num_threads);
    log::info!("  elite     {}", options.elite_count);
    if options.dedup_pop_and_fill {
        log::info!("  dedup     on (full topology comparison)");
    }
    log::info!("  results   {}", options.results_dir.display());

    // Warm start
    if !options.prior_topology_paths.is_empty() {
        log::info!("  warm      {} prior topologies", options.prior_topology_paths.len());
        for (i, path) in options.prior_topology_paths.iter().enumerate() {
            let fitness_info = std::fs::read_to_string(path)
                .ok()
                .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
                .and_then(|v| v.get("best_fitness")?.as_f64())
                .map(|f| format!(" fitness={f:.4}"))
                .unwrap_or_default();
            log::info!("  warm[{i}]   {}{}", path.display(), fitness_info);
        }
    }

    // Fitness & direction
    if fl == ll {
        log::info!("  fitness   {fl} · {score_better}");
    } else {
        log::info!("  fitness   {fl} (score) · {score_better}");
        log::info!("            {ll} (train) · {train_better}");
    }

    // Genetics
    if let Some(ref sel) = options.selection {
        log::info!("  selection {}", sel);
    }
    if let Some(ref cx) = options.crossover {
        log::info!("  crossover {}", cx);
    }
    log::info!("  mutation  {}", options.mutation);

    // Dataset
    let n = dataset.inputs.shape()[0];
    let d_in = dataset.inputs.shape().get(1).copied().unwrap_or(0);
    let d_out = dataset.targets.shape().get(1).copied().unwrap_or(0);
    // Detect data format
    let cache_dir = data_path.join("flodl_data");
    let format_str = if cache_dir.join("inputs.bin").exists() {
        "csv → bin (cached)"
    } else if data_path.join("inputs.bin").exists() {
        "bin (native)"
    } else if data_path.join("inputs.csv").exists() {
        "csv → bin (converted)"
    } else {
        "unknown"
    };
    log::info!("  data      [{n}×{d_in}] → [{n}×{d_out}]  {format_str}");

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
        options.topology_options.max_hidden_outputs_per_node
    );

    // Training
    let clip_str = if options.training.grad_clip > 0.0 {
        format!(" · clip {}", options.training.grad_clip)
    } else {
        String::new()
    };
    log::info!(
        "  train     {} epochs · lr {} · {}{}",
        options.training.num_epochs,
        options.training.learning_rate,
        options.training.optimizer,
        clip_str,
    );
    log::info!(
        "  sampling  train={} test={}",
        if options.train_y_proportional { "y_prop" } else { "uniform" },
        if options.test_y_proportional { "y_prop" } else { "uniform" }
    );

    // Network
    log::info!(
        "  net       {:?} · {:?}",
        options.network.device,
        options.network.dtype
    );
    if options.dropout_prob > 0.0 {
        log::info!("  net       dropout {}%", (options.dropout_prob * 100.0) as usize);
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

/// Final summary after run() completes. Winner, artifacts, examples reference.
pub(crate) fn log_run_summary(
    run_elapsed: Duration,
    best: &Option<BestIndividual>,
    fitness: &crate::fitness::Fitness,
    run_dir: &Path,
) -> Result<(), EngineError> {
    log::info!("");
    log::info!("══ run summary ════════════════════════════════════════════════════");

    // Duration (only new info — config was in initialization)
    let secs = run_elapsed.as_secs_f64();
    let h = secs as u64 / 3600;
    let m = (secs as u64 % 3600) / 60;
    let s = secs - (h * 3600 + m * 60) as f64;
    log::info!("  duration  {h}h {m}m {s:.3}s");

    // Winner
    if let Some(best) = best {
        log_winner(best, fitness)?;
    } else {
        log::info!("  winner    none");
    }

    // Artifacts
    log::info!("");
    log::info!("══ artifacts ══════════════════════════════════════════════════════");
    log::info!("  run_dir   {}", run_dir.display());
    log_artifacts(run_dir);

    // Rebuild
    log_examples(run_dir);

    Ok(())
}

// ── Private helpers ─────────────────────────────────────────────────────

/// Log winning individual characteristics.
fn log_winner(best: &BestIndividual, fitness: &crate::fitness::Fitness) -> Result<(), EngineError> {
    let t = &best.topology;
    let dims = t.node_dims();
    let kc = t.kind_counts();
    let acts = t.activation_counts();
    let net = Network::build(t, Device::CPU)
        .map_err(|e| EngineError::Json(format!("winner net build: {e}")))?;
    let total_elements: i64 = net.parameters().iter().map(|p| p.variable.numel()).sum();

    log::info!("");
    let fl = fitness.fitness_label();
    let ll = fitness.train_metric_label();
    if let Some(loss) = best.loss {
        if fl == ll {
            log::info!(
                "  winner    pop[{}] {fl} {:.4}",
                best.pop_index,
                best.fitness
            );
        } else {
            log::info!(
                "  winner    pop[{}] fitness({fl}) {:.4}  train_metric({ll}) {:.4}",
                best.pop_index,
                best.fitness,
                loss
            );
        }
    } else {
        log::info!(
            "  winner    pop[{}] {fl} {:.4}",
            best.pop_index,
            best.fitness
        );
    }
    log::info!(
        "  winner    {} nodes ({} input · {} hidden · {} output) · {} wires",
        t.nodes.len(),
        kc.input,
        kc.hidden,
        kc.output,
        t.connections.len()
    );
    log::info!(
        "  winner    {total_elements} param elements · {} tensors",
        net.parameters().len()
    );
    let act_str: Vec<String> = acts.iter().map(|(a, c)| format!("{a}x{c}")).collect();
    log::info!("  winner    activations [{}]", act_str.join(", "));
    if !dims.is_empty() {
        let dim_str: Vec<String> = dims
            .iter()
            .enumerate()
            .map(|(i, (iin, iout))| format!("n{i}:{iin}->{iout}"))
            .collect();
        log::info!("  winner    dims [{}]", dim_str.join(", "));
    }

    Ok(())
}

/// List artifacts in run directory.
fn log_artifacts(run_dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(run_dir) {
        let mut items: Vec<(String, u64)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let meta = e.metadata().ok()?;
                let name = e.file_name().to_string_lossy().into_owned();
                let size = if meta.is_dir() {
                    std::fs::read_dir(e.path())
                        .ok()
                        .map(|rd| {
                            rd.filter_map(|f| f.ok())
                                .filter_map(|f| f.metadata().ok().map(|m| m.len()))
                                .sum::<u64>()
                        })
                        .unwrap_or(0)
                } else {
                    meta.len()
                };
                Some((name, size))
            })
            .collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, size) in &items {
            let hint = match name.as_str() {
                "engine.json" => "  # full run envelope + best topology",
                "improvements" => "  # topology recipes per improvement",
                _ => "",
            };
            if *size > 0 {
                log::info!("    {name:<20} {size:>8} bytes{hint}");
            } else {
                log::info!("    {name:<20}        (dir){hint}");
            }
        }
    }
}

/// Print examples reference.
fn log_examples(run_dir: &Path) {
    log::info!("");
    log::info!("══ examples ══════════════════════════════════════════════════════");
    log::info!("  # Generate MD for a specific topology from gen_XX.json");
    log::info!("  cargo run --example generate_md_from_gen -- {}/improvements/gen_00.json --best", run_dir.display());
    log::info!("");
    log::info!("  # Fully train a specific network and see it working");
    log::info!("  cargo run --example train_from_gen -- {}/improvements/gen_00.json --best", run_dir.display());
    log::info!("");
    log::info!("  # Convert .bin data to CSV for inspection");
    log::info!("  cargo run --example bin_to_csv -- data/mnist/train");
}

/// Print the top-N repeated topologies sorted by mean fitness.
pub(crate) fn log_repeated_topologies(
    robustness: &std::collections::HashMap<String, crate::engine::RobustnessEntry>,
    top_n: usize,
) {
    // Only show topologies that appeared more than once
    let mut entries: Vec<&crate::engine::RobustnessEntry> = robustness.values()
        .filter(|e| e.count > 1)
        .collect();
    // Best: sort by appearances desc, then std_dev asc, then mean desc
    entries.sort_by(|a, b| {
        b.count.cmp(&a.count)
            .then_with(|| a.std_dev().partial_cmp(&b.std_dev()).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| b.mean.partial_cmp(&a.mean).unwrap_or(std::cmp::Ordering::Equal))
    });

    if entries.is_empty() {
        return;
    }

    let n = top_n.min(entries.len());
    log::info!("");
    log::info!("── repeated topologies (top {n}) ──");
    log::info!("  {:<6} {:>11} {:>9} {:>9} {:>9} {:>9} {:>8}  {}",
        "rank", "appearances", "mean", "std_dev", "min", "max", "params", "topo_id");

    for (rank, entry) in entries.iter().take(n).enumerate() {
        let topo_id = topo_hash(&entry.topology_json);
        log::info!("  {:<6} {:>11} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>8}  {}",
            format!("#{}", rank + 1),
            entry.count,
            entry.mean,
            entry.std_dev(),
            entry.min_fitness,
            entry.max_fitness,
            entry.param_count,
            topo_id,
        );
    }

    // Worst topologies: sort by appearances asc, then std_dev desc, then mean asc
    let mut worst: Vec<&crate::engine::RobustnessEntry> = robustness.values()
        .filter(|e| e.count > 1)
        .collect();
    worst.sort_by(|a, b| {
        a.count.cmp(&b.count)
            .then_with(|| b.std_dev().partial_cmp(&a.std_dev()).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.mean.partial_cmp(&b.mean).unwrap_or(std::cmp::Ordering::Equal))
    });
    let n_worst = top_n.min(worst.len());
    if n_worst > 0 {
        log::info!("");
        log::info!("── worst topologies (bottom {n_worst}) ──");
        log::info!("  {:<6} {:>11} {:>9} {:>9} {:>9} {:>9} {:>8}  {}",
            "rank", "appearances", "mean", "std_dev", "min", "max", "params", "topo_id");
        for (rank, entry) in worst.iter().take(n_worst).enumerate() {
            let topo_id = topo_hash(&entry.topology_json);
            log::info!("  {:<6} {:>11} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>8}  {}",
                format!("#{}", rank + 1),
                entry.count,
                entry.mean,
                entry.std_dev(),
                entry.min_fitness,
                entry.max_fitness,
                entry.param_count,
                topo_id,
            );
        }
    }
}
