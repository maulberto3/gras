//! Initialization logging — options, dataset, population (shown once at new()).

use crate::engine::EngineOptions;
use crate::fitness::Direction;
use crate::topology::Topology;

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
        options.topology_options.max_hidden_outputs_per_node
    );

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
