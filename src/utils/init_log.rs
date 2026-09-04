//! Initialization logging — options, dataset, population (shown once at new()).

use crate::engine::EngineOptions;
use crate::engine::fitness::Direction;
use crate::graph::topology::Topology;

/// Single section logging all resolved options and population.
/// Shown once at `Engine::new()`. All defaults are visible here.
pub(crate) fn log_initialization(
    options: &EngineOptions,
    pop: &[Topology],
    seed: u64,
    fitness: &crate::engine::fitness::Fitness,
) {
    let fl = fitness.label();
    let score_dir = fitness.direction();
    let score_better = match score_dir {
        Direction::Minimize => "lower = better",
        Direction::Maximize => "higher = better",
    };

    let mut rows: Vec<Vec<String>> = Vec::new();
    rows.push(vec![
        "engine".into(),
        format!(
            "pop {} · {} gens",
            options.pop_size.unwrap_or(0),
            options.num_generations.unwrap_or(0)
        ),
    ]);
    rows.push(vec![
        "seed".into(),
        format!("{seed}  (set_seed(Some({seed})) to reproduce)"),
    ]);
    rows.push(vec!["threads".into(), options.num_threads.to_string()]);
    rows.push(vec!["elite".into(), options.elite_count.to_string()]);
    if options.dedup_pop_and_fill {
        rows.push(vec!["dedup".into(), "on (full topology comparison)".into()]);
    }
    rows.push(vec![
        "results".into(),
        options.results_dir.display().to_string(),
    ]);

    if !options.prior_topology_paths.is_empty() {
        rows.push(vec![
            "warm".into(),
            format!("{} prior topologies", options.prior_topology_paths.len()),
        ]);
        for (i, path) in options.prior_topology_paths.iter().enumerate() {
            rows.push(vec![format!("warm[{i}]"), path.display().to_string()]);
        }
    }

    rows.push(vec!["fitness".into(), format!("{fl} · {score_better}")]);
    if let Some(ref sel) = options.selection {
        rows.push(vec!["selection".into(), sel.to_string()]);
    }
    if let Some(ref cx) = options.crossover {
        rows.push(vec!["crossover".into(), cx.to_string()]);
    }
    rows.push(vec!["mutation".into(), options.mutation.to_string()]);

    rows.push(vec![
        "input/output".into(),
        format!(
            "input {} · output {}",
            options.topology_options.input_dim, options.topology_options.output_dim
        ),
    ]);

    rows.push(vec![
        "topology".into(),
        format!(
            "input {} → hidden (pool {}..={} stride {}) → output {}",
            options.topology_options.input_dim,
            options.hidden_dim_pool.start(),
            options.hidden_dim_pool.end(),
            options.hidden_dim_stride,
            options.topology_options.output_dim
        ),
    ]);
    rows.push(vec![
        "nodes".into(),
        format!(
            "hidden {}..={} · in {}..={} · out {}..={}",
            options.topology_options.min_hidden_num_nodes,
            options.topology_options.max_hidden_num_nodes,
            options.topology_options.min_hidden_inputs_per_node,
            options.topology_options.max_hidden_inputs_per_node,
            options.topology_options.min_hidden_outputs_per_node,
            options.topology_options.max_hidden_outputs_per_node
        ),
    ]);

    // Dropout lives on the trainer (a training hyperparameter) and is
    // embedded into every topology — read it back from the population.
    if let Some(dropout) = pop
        .first()
        .map(|t| t.options.dropout_prob)
        .filter(|&d| d > 0.0)
    {
        rows.push(vec![
            "dropout".into(),
            format!("{}%", (dropout * 100.0) as usize),
        ]);
    }

    let act_names: Vec<String> = options.activation_pool.iter().map(|a| a.to_string()).collect();
    let std_names: Vec<String> = options
        .standardize_op_pool
        .iter()
        .map(|s| s.to_string())
        .collect();
    rows.push(vec![
        "pool hidden".into(),
        format!("{:?} stride {}", options.hidden_dim_pool, options.hidden_dim_stride),
    ]);
    rows.push(vec![
        "pool combine".into(),
        format!("{:?}", options.combine_op_pool),
    ]);
    rows.push(vec![
        "pool activations".into(),
        format!("[{}]", act_names.join(", ")),
    ]);
    rows.push(vec![
        "pool standardize".into(),
        format!("[{}]", std_names.join(", ")),
    ]);

    // Population summary
    let min_nodes = pop.iter().map(|g| g.nodes.len()).min().unwrap_or(0);
    let max_nodes = pop.iter().map(|g| g.nodes.len()).max().unwrap_or(0);
    rows.push(vec![
        "population".into(),
        format!("{} individuals · {}-{} nodes", pop.len(), min_nodes, max_nodes),
    ]);

    crate::utils::log_utils::log_section_table(
        "initialization",
        &["setting", "value"],
        &rows,
        crate::utils::log_utils::TABLE_WIDTH,
    );
}