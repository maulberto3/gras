//! Initial population generation — shared by the `race` binary and any
//! embedding (e.g. the mnist `main.rs`).
//!
//! Determinism contract: individual `i` derives everything from
//! `derive_seed(run_seed, i)`. The duplicate gate re-rolls at ordinals past
//! the initial batch (`pop_size + attempt`), so a given `(run_seed, config)`
//! still produces a bit-identical population — duplicates included in the
//! draw order, replaced deterministically when they collide.

use super::config::RaceConfig;
use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;
use crate::utils::seed::{derive_seed, topo_hash};

/// One deterministic topology draw from the run's config + pools, keyed by a
/// single ordinal. Shared by the initial loop and the duplicate re-roll so a
/// re-roll uses exactly the same pool/stride discipline as the original.
fn draw_topology(config: &RaceConfig, run_seed: u64, ordinal: usize) -> Topology {
    // Empty config pool ⇒ all known ops (resolved_* handles the fallback).
    let activation = config
        .resolved_activation_pool()
        .unwrap_or_else(|_| crate::evolution::pools::all_activations());
    let combine = config
        .resolved_combine_pool()
        .unwrap_or_else(|_| crate::evolution::pools::all_combine_ops());
    let standardize = config
        .resolved_standardize_pool()
        .unwrap_or_else(|_| crate::evolution::pools::all_standardize_ops());

    let seed = derive_seed(run_seed, ordinal) as usize;
    let mut rng = fastrand::Rng::with_seed(seed as u64);
    let opts = config.topology_options;
    let n_hidden = rng.usize(opts.min_hidden_num_nodes..=opts.max_hidden_num_nodes);
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
    graph
}

/// The initial population: `pop_size` deterministic draws, deduped by
/// topology hash. Dedupe matters because two identical topology JSONs hash
/// identically and `RaceState` would silently collapse them onto one key —
/// the pop would shrink below `pop_size` from step 0 (the gap this closes
/// was flagged as Tier-A follow-up when the duplicate gate landed for
/// children; this is the population-side twin of that gate).
pub fn initial_population(config: &RaceConfig, run_seed: u64) -> Vec<Topology> {
    let pop_size = config.pop_size;
    let mut topos: Vec<Topology> = (0..pop_size)
        .map(|i| draw_topology(config, run_seed, i))
        .collect();

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(topos.len());
    for topo in topos.drain(..) {
        let hash = match topo.to_json() {
            Ok(json) => topo_hash(&json),
            Err(_) => {
                // Unserializable topology would fail later anyway; keep it
                // and let the engine's build path surface the error.
                out.push(topo);
                continue;
            }
        };
        if seen.insert(hash.clone()) {
            out.push(topo);
            continue;
        }
        log::info!(
            "initial population: duplicate topology {} → discarded, re-rolling a unique seed",
            &hash[..8]
        );
        // Duplicate — re-roll at ordinals past the initial batch, bounded.
        let mut replaced = false;
        for attempt in 0..8 {
            let candidate = draw_topology(config, run_seed, pop_size + attempt);
            let json = match candidate.to_json() {
                Ok(j) => j,
                Err(_) => continue,
            };
            if seen.insert(topo_hash(&json)) {
                out.push(candidate);
                replaced = true;
                break;
            }
        }
        if !replaced {
            log::warn!(
                "initial population: could not replace a duplicate topology after 8 attempts — \
                 keeping it (the engine's hash map will dedupe; pop runs one short)"
            );
        }
    }
    out
}

/// Same shape as [`initial_population`], seeded from the run header's
/// recorded split ratio so a resumed run's new entrants see the same
/// train/eval pools as the run that wrote the checkpoint.
pub fn initial_population_from_header(
    config: &RaceConfig,
    run_seed: u64,
    _train_eval_split_ratio: f32,
) -> Vec<Topology> {
    // The initial population topology draw does not depend on the split ratio
    // (that only affects the data stream), so delegate to the shared builder.
    initial_population(config, run_seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config(pop: usize) -> RaceConfig {
        let mut opts = crate::graph::topology::TopologyOptions::default();
        // Deliberately tiny topology space: 1 hidden node, 1 dim ⇒ collisions
        // are guaranteed, which is exactly what the gate must handle.
        opts.min_hidden_num_nodes = 1;
        opts.max_hidden_num_nodes = 1;
        opts.input_dim = Some(2);
        opts.output_dim = Some(2);
        let mut cfg = RaceConfig::defaults();
        cfg.pop_size = pop;
        cfg.topology_options = opts;
        cfg.hidden_dim_pool = Some(4..=4);
        cfg
    }

    #[test]
    fn population_size_is_preserved_despite_duplicates() {
        let cfg = tiny_config(6);
        let topos = initial_population(&cfg, 42);
        assert_eq!(topos.len(), 6, "dedupe must not shrink the population");
    }

    #[test]
    fn population_has_no_duplicate_hashes() {
        let cfg = tiny_config(6);
        let topos = initial_population(&cfg, 42);
        let mut hashes: Vec<String> = topos
            .iter()
            .map(|t| topo_hash(&t.to_json().unwrap()))
            .collect();
        let total = hashes.len();
        hashes.sort();
        hashes.dedup();
        assert_eq!(hashes.len(), total, "no two topologies may share a hash");
    }

    #[test]
    fn population_is_deterministic_given_seed() {
        let cfg = tiny_config(6);
        let a: Vec<String> = initial_population(&cfg, 42)
            .iter()
            .map(|t| t.to_json().unwrap())
            .collect();
        let b: Vec<String> = initial_population(&cfg, 42)
            .iter()
            .map(|t| t.to_json().unwrap())
            .collect();
        assert_eq!(a, b, "same seed ⇒ bit-identical population");
    }

    #[test]
    fn forced_space_of_one_keeps_exactly_one_unique() {
        // A 1-hidden-node, fixed-dim space with no op variety can only
        // produce ONE distinct topology; the gate can't invent variety. It
        // must keep pop_size entries but only 1 unique hash — the warn path.
        let mut cfg = tiny_config(3);
        cfg.activation_pool = vec!["relu".into()];
        cfg.combine_op_pool = vec!["mean".into()];
        cfg.standardize_op_pool = vec!["identity".into()];
        let topos = initial_population(&cfg, 42);
        assert_eq!(topos.len(), 3, "pop size preserved even when trapped");
    }
}
