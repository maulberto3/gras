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

// ── Initialization ───────────────────────────────────────────────────────

/// Single section logging all resolved options, dataset, and population.
/// Shown once at `Engine::new()`. All defaults are visible here.
pub(crate) fn log_initialization(
    options: &EngineOptions,
    dataset: &super::data::Dataset,
    pop: &[Topology],
    seed: u64,
    fitness: &crate::fitness::Fitness,
) {
    let fl = fitness.fitness_label();
    let ll = fitness.train_metric_label();
    let dir = fitness.direction();
    let better = match dir {
        Direction::Minimize => "lower = better",
        Direction::Maximize => "higher = better",
    };

    log::info!("");
    log::info!("══ initialization ═══════════════════════════════════════════════════");

    // Engine core
    log::info!(
        "  engine    pop {} · {} gens",
        options.pop_size,
        options.num_generations
    );
    log::info!("  seed      {seed}  (set_seed(Some({seed})) to reproduce)");
    log::info!(
        "  engine    budget {}bt x {} · {} threads",
        options.num_batches,
        options.batch_size,
        options.num_threads
    );
    log::info!("  engine    results_dir {}", options.results_dir.display());

    // Fitness & direction
    if fl == ll {
        log::info!("  fitness   {} ({fl}) · {better}", fl);
    } else {
        log::info!("  fitness   {fl} (fitness) · {ll} (train_metric)");
        log::info!("            {better}");
    }
    // Mutation
    log::info!("  mutation  {}", options.mutation);
    // Crossover
    log::info!("  crossover {}", options.crossover);
    // Dedup
    if options.dedup_pop {
        log::info!("  dedup     on (full topology comparison)");
    }

    // Dataset
    let n = dataset.inputs.shape()[0];
    let d_in = dataset.inputs.shape().get(1).copied().unwrap_or(0);
    let d_out = dataset.targets.shape().get(1).copied().unwrap_or(0);
    log::info!("  data      [{n}x{d_in}] -> [{n}x{d_out}]");

    // Topology
    log::info!(
        "  topo      input {} -> hidden (pool {}..={}) -> output {}",
        options.topology_options.input_dim,
        options.hidden_dim_pool.start(),
        options.hidden_dim_pool.end(),
        options.topology_options.output_dim
    );
    log::info!(
        "  topo      hidden nodes {}..={} · inputs/node {}..={} · outputs/node {}..={}",
        options.topology_options.min_hidden_num_nodes,
        options.topology_options.max_hidden_num_nodes,
        options.topology_options.min_hidden_inputs_per_node,
        options.topology_options.max_hidden_inputs_per_node,
        options.topology_options.min_hidden_outputs_per_node,
        options.topology_options.max_hidden_outputs_per_node
    );

    // Network
    log::info!(
        "  net       device {:?} · dtype {:?}",
        options.network.device,
        options.network.dtype
    );
    if options.dropout_prob > 0.0 {
        log::info!("  net       dropout {:.0}%", options.dropout_prob * 100.0);
    }

    // Training
    let clip_str = if options.training.grad_clip > 0.0 {
        format!(" · clip {:.1}", options.training.grad_clip)
    } else {
        String::new()
    };
    let prop_str = if options.y_proportional_batches {
        " · y_proportional"
    } else {
        ""
    };
    log::info!(
        "  train     {} epochs · lr {} · {}{}{}",
        options.training.num_epochs,
        options.training.learning_rate,
        options.training.optimizer,
        clip_str,
        prop_str,
    );

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
    log::info!("  pools     hidden {:?}", options.hidden_dim_pool);
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
    log::info!("  run_dir   {}", run_dir.display());
}

// ── Run summary ─────────────────────────────────────────────────────────

/// Final summary after run() completes. Winner, artifacts, rebuild helper.
/// No GP pool duplication — those are in initialization.
pub(crate) fn log_run_summary(
    run_elapsed: Duration,
    best: &Option<BestIndividual>,
    fitness: &crate::fitness::Fitness,
    options: &EngineOptions,
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
    log_rebuild_helper(run_dir, options);

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

/// Print copy-paste rebuild snippet.
fn log_rebuild_helper(run_dir: &Path, options: &EngineOptions) {
    let dev = options.network.device;
    log::info!("");
    log::info!("══ rebuild ════════════════════════════════════════════════════════");
    log::info!("  copy-paste this to reconstruct any saved topology:");
    log::info!("    use gras::topology::Topology;");
    log::info!("    use gras::network::Network;");
    log::info!("    use flodl::nn::Module;");
    log::info!("");
    log::info!("    // Load from engine.json or any improvement .json (same format)");
    log::info!("    let v: serde_json::Value = serde_json::from_str(");
    log::info!("        &std::fs::read_to_string(\"<path_to_json>\").unwrap()).unwrap();");
    log::info!("    let topo = Topology::from_json(");
    log::info!("        v[\"best_topology\"].as_str().unwrap()).unwrap();");
    log::info!("    let net = Network::build(&topo, {dev:?}).unwrap();");
    log::info!("");
    log::info!("  run_dir: {}", run_dir.display());
    log::info!("  engine.json or improvements/*.json — drop any path into <path_to_json>");
}
