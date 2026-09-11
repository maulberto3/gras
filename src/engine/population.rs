//! Initial population generation — shared by the `race` binary and any
//! embedding (e.g. the mnist `main.rs`).

use super::config::RaceConfig;
use crate::graph::topology::Topology;

pub fn initial_population(config: &RaceConfig, run_seed: u64) -> Vec<Topology> {
    use crate::graph::node::NodeKind;
    // Empty config pool ⇒ all known ops (resolved_* handles the fallback).
    let activation = config.resolved_activation_pool().unwrap_or_else(|_| crate::evolution::pools::all_activations());
    let combine = config.resolved_combine_pool().unwrap_or_else(|_| crate::evolution::pools::all_combine_ops());
    let standardize = config.resolved_standardize_pool().unwrap_or_else(|_| crate::evolution::pools::all_standardize_ops());
    let mut topos = Vec::with_capacity(config.pop_size);
    for i in 0..config.pop_size {
        let seed = crate::utils::seed::derive_seed(run_seed, i) as usize;
        let mut rng = fastrand::Rng::with_seed(seed as u64);
        let opts = config.topology_options;
        let n_hidden = rng.usize(
            opts.min_hidden_num_nodes..=opts.max_hidden_num_nodes,
        );
        let mut graph = Topology::new(seed, Some(opts));
        graph.create_random_hidden_nodes(n_hidden);
        let pool = config.hidden_dim_pool.clone().unwrap_or(4..=8);
        let stride = config.hidden_dim_stride.max(1);
        for node in &mut graph.nodes {
            if node.kind == NodeKind::Hidden {
                let n = ((pool.end() - pool.start()) / stride) + 1;
                node.hidden_dim = Some(pool.start() + rng.usize(0..n) * stride);
                node.activation = activation[rng.usize(0..activation.len())];
                node.combine_op = Some(combine[rng.usize(0..combine.len())]);
                node.standardize = Some(standardize[rng.usize(0..standardize.len())]);
            }
        }
        graph.refresh_labels();
        graph.finalize();
        topos.push(graph);
    }
    topos
}

/// Same shape as [`initial_population`], seeded from the run header's
/// recorded split ratio so a resumed run's new entrants see the same
/// train/eval pools as the run that wrote the checkpoint.
pub fn initial_population_from_header(
    config: &RaceConfig,
    run_seed: u64,
    train_eval_split_ratio: f32,
) -> Vec<Topology> {
    // The initial population topology draw does not depend on the split ratio
    // (that only affects the data stream), so delegate to the shared builder.
    initial_population(config, run_seed)
}
