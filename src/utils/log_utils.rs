//! Engine logging 📊 — all pretty-printed console output.
//!
//! Extracted from `Engine` methods to keep the engine focused on the genetic
//! loop. Each section has its own function: `log_options`, `log_dataset`,
//! `log_population`, `log_run_start`, `log_best`, `log_run_summary`.

use std::path::Path;
use std::time::Duration;

use flodl::nn::Module;
use flodl::Device;

use crate::engine::{BestIndividual, EngineOptions};
use super::error::EngineError;
use crate::fitness::Direction;
use crate::network::Network;
use crate::topology::Topology;

// ── Options ────────────────────────────────────────────────────────────────

/// Log the full options cascade: engine knobs, topology template, GP pools,
/// training config, and network device.
pub(crate) fn log_options(options: &EngineOptions) {
    log::info!("");
    log::info!("══ options ═══════════════════════════════════");
    log::info!(
        "  engine  pop {} · {} gens · seed {:?}",
        options.pop_size,
        options.num_generations,
        options.seed
    );
    log::info!(
        "  engine  budget {}bt of {} · {} threads",
        options.num_batches,
        options.batch_size,
        options.num_threads
    );
    if options.fitness_label == options.train_metric_label {
        log::info!(
            "  engine  fitness {} · mut_act {:.0}%",
            options.fitness_label,
            options.mutate_activ_prob * 100.0
        );
    } else {
        log::info!(
            "  engine  fitness {} · train_metric {} · mut_act {:.0}%",
            options.fitness_label, options.train_metric_label,
            options.mutate_activ_prob * 100.0
        );
    }
    log::info!("  engine  GP pool:  hidden {:?}", options.hidden_dim_pool);
    log::info!("  engine  GP pool:  combine {:?}", options.combine_op_pool);
    log::info!("  engine  GP pool:  activations {:?}", options.activation_pool);
    log::info!(
        "  engine  GP pool:  standardize {:?}",
        options.standardize_op_pool
    );
    log::info!(
        "  engine  train {} epochs · lr {} · {}{}",
        options.training.num_epochs,
        options.training.learning_rate,
        options.training.optimizer,
        if options.training.grad_clip > 0.0 {
            format!(" · clip {:.1}", options.training.grad_clip)
        } else {
            String::new()
        }
    );
    log::info!("  engine  results_dir {}", options.results_dir.display());
    log::info!(
        "  topo    input {} → hidden {} (pool {}..={}) → output {}",
        options.topology_options.input_dim,
        options.topology_options.hidden_dim,
        options.hidden_dim_pool.start(),
        options.hidden_dim_pool.end(),
        options.topology_options.output_dim
    );
    log::info!(
        "  topo    input/output dims auto-detected from dataset"
    );
    log::info!(
        "  topo    hidden {}..={} · inputs/node {}..={} · outputs/node {}..={}",
        options.topology_options.min_hidden_num_nodes,
        options.topology_options.max_hidden_num_nodes,
        options.topology_options.min_hidden_inputs_per_node,
        options.topology_options.max_hidden_inputs_per_node,
        options.topology_options.min_hidden_outputs_per_node,
        options.topology_options.max_hidden_outputs_per_node
    );
    log::info!(
        "  net     device {:?} · dtype {:?}",
        options.network.device,
        options.network.dtype
    );
}

// ── Dataset ────────────────────────────────────────────────────────────────

/// Log dataset shape, population size, seed, and fitness direction.
pub(crate) fn log_dataset(
    data: &super::data::Dataset,
    pop_len: usize,
    seed: u64,
    direction: Direction,
) {
    log::info!("");
    log::info!("══ dataset ═══════════════════════════════════");
    log::info!(
        "  [{}×{}] · pop {} · seed {seed} · {:?}",
        data.inputs.shape()[0],
        data.inputs.shape().get(1).copied().unwrap_or(0),
        pop_len,
        direction,
    );
}

// ── Population ─────────────────────────────────────────────────────────────

/// Log population summary: count, node range, hidden dim, device.
pub(crate) fn log_population(pop: &[Topology], options: &EngineOptions) {
    log::info!("");
    log::info!("══ population ═══════════════════════════════");    log::info!("  {} individuals · {}–{} nodes · hidden pool {:?} · device {:?}",
        pop.len(),
        pop.iter().map(|g| g.nodes.len()).min().unwrap_or(0),
        pop.iter().map(|g| g.nodes.len()).max().unwrap_or(0),
        options.hidden_dim_pool,
        options.network.device,
    );
    log::info!("");
}

// ── Run start ──────────────────────────────────────────────────────────────

/// Log the run header when the generation loop begins.
pub(crate) fn log_run_start(options: &EngineOptions, run_dir: &Path, direction: Direction) {
    let better = match direction {
        Direction::Minimize => "lowest",
        Direction::Maximize => "highest",
    };
    log::info!("══ run ══════════════════════════════════════");
    log::info!(
        "  {} gens · budget {}bt of {} · {} threads",
        options.num_generations,
        options.num_batches,
        options.batch_size,
        options.num_threads,
    );
    if options.fitness_label == options.train_metric_label {
        log::info!("  fitness {} ({better} = better)", options.fitness_label);
    } else {
        log::info!("  fitness {} (↑ rank) · train_metric {} (↓ train)", options.fitness_label, options.train_metric_label);
    }
    log::info!("  results_dir {}", run_dir.display());
}

// ── Best ───────────────────────────────────────────────────────────────────

/// Log the current best individual after the generation loop.
pub(crate) fn log_best(best: &Option<BestIndividual>, generation: usize, fitness: &crate::fitness::Fitness) -> Result<(), EngineError> {
    log::info!("");
    log::info!("══ best ═════════════════════════════════════");
    if let Some(best) = best {
        let fl = fitness.fitness_label();
        let ll = fitness.train_metric_label();
        if let Some(loss) = best.loss {
            if fl == ll {
                log::info!("  {} {:.4} after {} gen(s)", fl, best.fitness, generation);
            } else {
                log::info!("  fitness ({}) {:.4}  train_metric ({}) {:.4} after {} gen(s)", fl, best.fitness, ll, loss, generation);
            }
        } else {
            log::info!("  {} {:.4} after {} gen(s)", fl, best.fitness, generation);
        }
        let net = Network::build(&best.topology, Device::CPU)
            .map_err(|e| EngineError::Json(format!("best net build: {e}")))?;
        log::info!(
            "  blueprint: {} nodes · {} wires · {} param tensors",
            best.topology.nodes.len(),
            best.topology.connections.len(),
            net.parameters().len()
        );
        log::info!("  ascii      see improvements/*.md for topology + network diagrams");
    } else {
        log::info!("  no improvements");
    }
    Ok(())
}

// ── Run summary ────────────────────────────────────────────────────────────

/// Log the full run summary after `Engine::run()` completes.
pub(crate) fn log_run_summary(
    run_elapsed: Duration,
    seed: u64,
    generation: usize,
    pop_len: usize,
    options: &EngineOptions,
    data_path: &Path,
    fitness: &crate::fitness::Fitness,
    improvements: usize,
    best: &Option<BestIndividual>,
    run_dir: &Path,
) -> Result<(), EngineError> {
    log::info!("");
    log::info!("══ run summary ══════════════════════════════");
    let secs = run_elapsed.as_secs_f64();
    let h = secs as u64 / 3600;
    let m = (secs as u64 % 3600) / 60;
    let s = secs - (h * 3600 + m * 60) as f64;
    log::info!("  duration    {h}h {m}m {s:.3}s ({secs:.2}s raw)");
    log::info!("  seed        {seed}  (reproduce with .set_seed(Some({seed})))");
    log::info!("  gens        {generation} · pop {pop_len} · budget {}bt of {} · {} threads",
        options.num_batches, options.batch_size, options.num_threads);
    log::info!("  data        {}", data_path.display());
    if options.fitness_label == options.train_metric_label {
        log::info!("  fitness     {} ({})", options.fitness_label,
            match fitness.direction() {
                Direction::Minimize => "lower = better",
                Direction::Maximize => "higher = better",
            });
    } else {
        log::info!("  fitness     {} ({}) · train_metric {} ({})", options.fitness_label,
            match fitness.direction() {
                Direction::Minimize => "lower = better",
                Direction::Maximize => "higher = better",
            },
            options.train_metric_label,
            match fitness.train_metric_direction() {
                Direction::Minimize => "lower = better",
                Direction::Maximize => "higher = better",
            });
    }
    log::info!("  training    {} epochs · lr {} · {}{}",
        options.training.num_epochs, options.training.learning_rate,
        options.training.optimizer,
        if options.training.grad_clip > 0.0 {
            format!(" · clip {:.1}", options.training.grad_clip)
        } else {
            String::new()
        });
    log::info!("  topo opts   input {} → hidden {} → output {}",
        options.topology_options.input_dim, options.topology_options.hidden_dim,
        options.topology_options.output_dim);
    log::info!("  GP pools    hidden {:?} · combine {:?} · std {:?}",
        options.hidden_dim_pool, options.combine_op_pool, options.standardize_op_pool);
    let act_pool: Vec<String> = options.activation_pool.iter().map(|a| a.to_string()).collect();
    log::info!("  GP pool     activations [{}]", act_pool.join(", "));
    log::info!("  improvements {improvements}");

    if let Some(best) = best {
        log_winner(best, fitness)?;
    }

    log::info!("");
    log::info!("  results     {}", run_dir.display());
    log_artifacts(run_dir);

    log_rebuild_helper(run_dir, options);

    Ok(())
}

// ── Private helpers ────────────────────────────────────────────────────────

/// Log the winning individual's characteristics.
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
            log::info!("  winner      pop[{}] {} {:.4}", best.pop_index, fl, best.fitness);
        } else {
            log::info!("  winner      pop[{}] fitness ({}) {:.4}  train_metric ({}) {:.4}", best.pop_index, fl, best.fitness, ll, loss);
        }
    } else {
        log::info!("  winner      pop[{}] {} {:.4}", best.pop_index, fl, best.fitness);
    }
    log::info!("  winner      {} nodes ({} input · {} hidden · {} output) · {} wires",
        t.nodes.len(), kc.input, kc.hidden, kc.output, t.connections.len());
    log::info!("  winner      {total_elements} param elements · {} tensors", net.parameters().len());
    let act_str: Vec<String> = acts.iter().map(|(a, c)| format!("{a}×{c}")).collect();
    log::info!("  winner      activations [{}]", act_str.join(", "));
    if !dims.is_empty() {
        let dim_str: Vec<String> = dims.iter().enumerate()
            .map(|(i, (iin, iout))| format!("n{i}:{iin}→{iout}")).collect();
        log::info!("  winner      dims       [{}]", dim_str.join(", "));
    }

    Ok(())
}

/// List artifact files in the run directory with sizes.
fn log_artifacts(run_dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(run_dir) {
        let mut items: Vec<(String, u64)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let meta = e.metadata().ok()?;
                let name = e.file_name().to_string_lossy().into_owned();
                let size = if meta.is_dir() {
                    std::fs::read_dir(e.path()).ok()
                        .map(|rd| rd.filter_map(|f| f.ok())
                            .filter_map(|f| f.metadata().ok().map(|m| m.len()))
                            .sum::<u64>())
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

/// Print the copy-paste rebuild snippet for the best network.
fn log_rebuild_helper(run_dir: &Path, options: &EngineOptions) {
    let dev = options.network.device;
    log::info!("");
    log::info!("══ rebuild ══════════════════════════════════");
    log::info!("  copy-paste this to reconstruct the best network:");
    log::info!("    use gras::topology::Topology;");
    log::info!("    use gras::network::Network;");
    log::info!("    use flodl::nn::Module;");
    log::info!("");
    log::info!("    let v: serde_json::Value = serde_json::from_str(");
    log::info!("        &std::fs::read_to_string(\"{}/engine.json\").unwrap()).unwrap();", run_dir.display());
    log::info!("    let topo = Topology::from_json(");
    log::info!("        v[\"best_topology\"].as_str().unwrap()).unwrap();");
    log::info!("    let net = Network::build(&topo, {dev:?}).unwrap();");
    log::info!("    // Your best architecture, ready for inference");
}
