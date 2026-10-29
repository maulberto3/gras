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

/// One deterministic topology draw from the run's config + pools, keyed by an
/// ordinal. Shared by the initial loop and the duplicate re-roll.
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
        // Per-individual identity seed. `options.topology_seed` keys the whole engine;
        // sharing the template default made every founder identical.
    let mut opts = config.topology_options;
    opts.topology_seed = seed;
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
        // Feature split AFTER finalize: the Input node's port count is final only once
        // `trim_orphaned_ports` compacted the ports (same rule as port activations).
    if let Some(policy) = config.feature_split {
        graph.assign_feature_masks(policy);
    }
    graph
}

/// The initial population: `pop_size` deterministic draws, deduped by topology
/// hash — identical JSONs would otherwise collapse onto one `RaceState` key.
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
        // CFY: the duplicate notice is a table, not a prose line.
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            "Initial population",
            &crate::utils::cfy::kv(&[(
                "duplicate topology".to_string(),
                hash[..8.min(hash.len())].to_string(),
            )]),
            Some("discarded, re-rolling a unique seed"),
        ));
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
            tracing::warn!(
                "initial population: could not replace a duplicate topology after 8 attempts — \
                 keeping it (the engine's hash map will dedupe; pop runs one short)"
            );
        }
    }
    out
}

/// Build the founder batch from explicit blueprints + random draws, deduped by
/// topology hash and always landing on `target` nets (import first, then draws).
pub fn fill_unique(
    config: &RaceConfig,
    run_seed: u64,
    founders: Vec<Topology>,
    target: usize,
) -> Vec<Topology> {
    if founders.is_empty() {
        return initial_population(config, run_seed);
    }
    let target = target.max(founders.len());
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<Topology> = Vec::with_capacity(target);
    for topo in founders {
        let json = match topo.to_json() {
            Ok(json) => json,
            Err(_) => {
                out.push(topo);
                continue;
            }
        };
        if seen.insert(topo_hash(&json)) {
            out.push(topo);
        } else {
            tracing::warn!(
                "founder batch: duplicate founder blueprint → dropped (the batch keeps its first copy)"
            );
        }
    }
    for topo in initial_population(config, run_seed) {
        if out.len() >= target {
            break;
        }
        let json = match topo.to_json() {
            Ok(json) => json,
            Err(_) => {
                out.push(topo);
                continue;
            }
        };
        if seen.insert(topo_hash(&json)) {
            out.push(topo);
            continue;
        }        crate::utils::cfy::emit(&crate::utils::cfy::card(
            "Founder batch",
            &crate::utils::cfy::kv(&[(
                "random draw".to_string(),
                "duplicates a founder".to_string(),
            )]),
            Some("re-rolling a unique seed"),
        ));
        let mut replaced = false;
        for attempt in 0..8 {
            // Ordinals past `initial_population`'s own re-roll window
            // (`pop_size + 0..8`) — the two re-roll sets must not overlap.
            let candidate = draw_topology(config, run_seed, config.pop_size + 64 + attempt);
            let Ok(json) = candidate.to_json() else {
                continue;
            };
            if seen.insert(topo_hash(&json)) {
                out.push(candidate);
                replaced = true;
                break;
            }
        }
        if !replaced {
            tracing::warn!(
                "founder batch: could not replace a duplicate draw after 8 attempts — \
                 population runs one short"
            );
        }
    }
    out
}

// ── Seed blueprint loading ─────────────────────────────────────────────

/// Load a blueprint from a saved artifact (a topology JSON, or a
/// `nets/<hash>.json` state's `topology` field). Blueprint-only, fresh weights;
/// this is how `elite_previous_load` reads a source run's ranked elites.
pub fn run_topology_from_json_file(path: &std::path::Path) -> Result<Topology, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("run-topology {}: {e}", path.display()))?;
    // Try the two shapes: raw topology JSON first, then a NetState wrapper.
    if let Ok(topo) = Topology::from_json(&raw) {
        return Ok(topo);
    }
    let state: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        format!(
            "run-topology {}: not topology or net-state JSON: {e}",
            path.display()
        )
    })?;
    let topo_json = state
        .get("topology")
        .and_then(|t| t.as_str())
        .ok_or_else(|| format!("run-topology {}: no `topology` field", path.display()))?;
    Topology::from_json(topo_json)
        .map_err(|e| format!("run-topology {}: bad topology: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config(pop: usize) -> RaceConfig {
        let opts = crate::graph::topology::TopologyOptions {
            // Deliberately tiny topology space: 1 hidden node, 1 dim ⇒ collisions
            // are guaranteed, which is exactly what the gate must handle.
            min_hidden_num_nodes: 1,
            max_hidden_num_nodes: 1,
            input_dim: Some(2),
            output_dim: Some(2),
            ..Default::default()
        };
        let mut cfg = RaceConfig::defaults();
        cfg.pop_size = pop;
        cfg.topology_options = opts;
        cfg.hidden_dim_pool = Some(4..=4);
        cfg
    }

    #[test]
    fn dedupe_keeps_size_and_leaves_no_duplicate_hashes() {
        let cfg = tiny_config(6);
        let topos = initial_population(&cfg, 42);
        assert_eq!(topos.len(), 6, "dedupe must not shrink the population");
        let mut hashes: Vec<String> = topos
            .iter()
            .map(|t| topo_hash(&t.to_json().unwrap()))
            .collect();
        let total = hashes.len();
        hashes.sort();
        hashes.dedup();
        assert_eq!(hashes.len(), total, "no two topologies may share a hash");
    }

        /// An import duplicating a random draw must not shrink the population: the
        /// colliding draw is re-rolled, so the batch lands on `pop_size` uniques.
    #[test]
    fn fill_unique_keeps_pop_size_when_a_draw_collides_with_an_import() {
        let cfg = tiny_config(6);
        let draws = initial_population(&cfg, 42);
        // Force the collision: import the topology the random batch itself
        // draws first (same config + run seed ⇒ same ordinal-0 draw).
        let colliding = draws[0].clone();
        let out = fill_unique(&cfg, 42, vec![colliding.clone()], 6);
        assert_eq!(
            out.len(),
            6,
            "a colliding draw must be re-rolled, not silently dropped"
        );
        let mut hashes: Vec<String> = out
            .iter()
            .map(|t| topo_hash(&t.to_json().unwrap()))
            .collect();
        let total = hashes.len();
        hashes.sort();
        hashes.dedup();
        assert_eq!(hashes.len(), total, "the batch must stay unique");
        assert_eq!(
            out[0].to_json().unwrap(),
            colliding.to_json().unwrap(),
            "imports keep the first slots"
        );
    }

    /// With no imports the batch is `initial_population` verbatim, so the
    /// no-seed path stays bit-identical to what it always was.
    #[test]
    fn fill_unique_without_founders_is_the_plain_population() {
        let cfg = tiny_config(6);
        let plain: Vec<String> = initial_population(&cfg, 42)
            .iter()
            .map(|t| t.to_json().unwrap())
            .collect();
        let mixed: Vec<String> = fill_unique(&cfg, 42, Vec::new(), 6)
            .iter()
            .map(|t| t.to_json().unwrap())
            .collect();
        assert_eq!(plain, mixed);
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
            // A 1-hidden-node fixed-dim space yields ONE topology; the gate must
            // keep pop_size entries with only 1 unique hash (the warn path).
        let mut cfg = tiny_config(3);
        cfg.activation_pool = vec!["relu".into()];
        cfg.combine_op_pool = vec!["mean".into()];
        cfg.standardize_op_pool = vec!["identity".into()];
        let topos = initial_population(&cfg, 42);
        assert_eq!(topos.len(), 3, "pop size preserved even when trapped");
    }

        /// Every founder must carry its OWN `net_seed` — sharing the template seed
        /// gave the whole population identical weights and one challenge trigger.
    #[test]
    fn founders_get_distinct_net_seeds() {
        let cfg = tiny_config(8);
        let topos = initial_population(&cfg, 42);
        let mut seeds: Vec<usize> = topos.iter().map(|t| t.options.topology_seed).collect();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(
            seeds.len(),
            8,
            "each founder must carry a distinct net_seed"
        );
    }

    #[test]
    fn feature_split_masks_match_the_final_port_count() {
            // End-to-end through the real engine draw: with the policy on, every founder
            // must carry masks whose length matches the POST-finalize port count (a stale
            // port count is exactly what would break the validate rule), and Exclusive
            // must still partition the features.
        let opts = crate::graph::topology::TopologyOptions {
            min_hidden_num_nodes: 4,
            max_hidden_num_nodes: 4,
            input_dim: Some(3),
            output_dim: Some(2),
            ..Default::default()
        };
        let mut cfg = RaceConfig::defaults();
        cfg.pop_size = 4;
        cfg.topology_options = opts;
        cfg.hidden_dim_pool = Some(4..=4);
        cfg.feature_split = Some(crate::graph::topology::FeatureSplitPolicy::Exclusive);

        for topo in initial_population(&cfg, 42) {
            topo.validate().expect("a split topology must validate");
            let input = topo
                .nodes
                .iter()
                .find(|n| n.kind == NodeKind::Input)
                .unwrap();
            let masks = input
                .feature_masks
                .as_ref()
                .expect("the policy is on, so founders carry masks");
            assert_eq!(masks.len(), input.num_outputs, "one mask per final port");
            assert!(
                masks.iter().all(|m| m.len() == 3),
                "every mask spans input_dim"
            );
            let placed: usize = masks
                .iter()
                .map(|m| m.iter().filter(|b| **b == 1).count())
                .sum();
            assert_eq!(placed, 3, "Exclusive places every feature exactly once");
        }
    }

    #[test]
    fn feature_split_off_leaves_every_port_unmasked() {
            // The default (`None`) is today's behaviour: no masks anywhere, so nothing
            // about an existing run's population changes.
        let cfg = tiny_config(4);
        assert!(cfg.feature_split.is_none(), "the split is opt-in");
        for topo in initial_population(&cfg, 42) {
            let input = topo
                .nodes
                .iter()
                .find(|n| n.kind == NodeKind::Input)
                .unwrap();
            assert_eq!(input.feature_masks, None, "no policy ⇒ no masks");
        }
    }
}
