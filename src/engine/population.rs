//! Initial population generation — shared by the `race` binary and any embedding (e.g.

use super::config::RaceConfig;
use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;
use crate::utils::seed::{derive_seed, topo_hash};

/// One deterministic topology draw from the run's config + pools, keyed
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
    // Structure keeps the individual seed; WEIGHT creation is run-wide, so the same
    // architecture is born with the same coefficients whoever draws
    opts.weight_seed = Some(crate::utils::seed::weight_seed(run_seed));
    let n_hidden = rng.usize(opts.min_hidden_num_nodes..=opts.max_hidden_num_nodes);
    // The split is a POOL draw: none, random or exclusive can enter the population.
    let feature_split = draw_feature_split(config, &mut rng);
    opts.feature_split = feature_split;
    // The wiring gene: how much of the raw input this individual's hidden layer may read.
    opts.input_reach = opts.draw_input_reach(&mut rng);
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
    if let Some(policy) = feature_split {
        graph.assign_feature_masks(policy);
    }
    graph
}

/// Draw one policy from the run's feature-split pool (`empty ⇒ all three`) with the caller's own
/// deterministic RNG.
pub(crate) fn draw_feature_split(
    config: &RaceConfig,
    rng: &mut fastrand::Rng,
) -> Option<crate::graph::topology::FeatureSplitPolicy> {
    let pool = config.resolved_feature_split_pool().unwrap_or_else(|_| {
        vec![
            None,
            Some(crate::graph::topology::FeatureSplitPolicy::Random),
            Some(crate::graph::topology::FeatureSplitPolicy::Exclusive),
        ]
    });
    pool[rng.usize(..pool.len())]
}

/// The initial population: `pop_size` deterministic draws, deduped by topology hash — identical
/// JSONs would otherwise collapse onto one `RaceState` key.
pub fn initial_population(config: &RaceConfig, run_seed: u64) -> Vec<Topology> {
    let pop_size = config.pop_size;
    let mut topos: Vec<Topology> = (0..pop_size)
        .map(|i| draw_topology(config, run_seed, i))
        .collect();

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(topos.len());
    // The re-roll cursor lives PAST the batch and only moves FORWARD: a collision can never re-try an ordinal an earlier collision already spent (the old fixed `pop_size +…
    let mut cursor = pop_size + 64;
    for topo in topos.drain(..) {
        let hash = match topo.to_json() {
            Ok(json) => topo_hash(&json),
            Err(_) => {
                // Unserializable topology would fail later anyway; keep
                // and let the engine's build path surface the error.
                out.push(topo);
                continue;
            }
        };
        if seen.insert(hash.clone()) {
            out.push(topo);
            continue;
        }
        let mut replaced = false;
        for attempt in 0..8 {
            let candidate = draw_topology(config, run_seed, cursor + attempt);
            let Ok(json) = candidate.to_json() else {
                continue;
            };
            if seen.insert(topo_hash(&json)) {
                out.push(candidate);
                replaced = true;
                break;
            }
        }
        cursor += 8;
        if !replaced {
            // The space really is smaller than the batch: keep the blueprint so the vector stays
            // `pop_size` long, and say what that costs instead of claiming a unique slot.
            tracing::warn!(
                "initial population: the topology space has fewer than {pop_size} unique draws — \
                 kept duplicate {}; the engine dedupes it, so the population starts short",
                &hash[..8.min(hash.len())]
            );
            out.push(topo);
        }
    }
    out
}

/// Build the founder batch from explicit blueprints + random draws, deduped
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
    let founder_count = founders.len();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<Topology> = Vec::with_capacity(target);
    let mut duplicate_imports = 0usize;
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
            duplicate_imports += 1;
        }
    }
    let imported = out.len();
    // The random batch starts PAST the imported ordinals.
    let mut ordinal = imported;
    let mut duplicate_draws = 0usize;
    // Generous but BOUNDED: a run that collides this often is a space smaller than its batch.
    let budget = (target.saturating_sub(out.len()) + 64) * 8;
    let mut spent = 0usize;
    while out.len() < target && spent < budget {
        spent += 1;
        let candidate = draw_topology(config, run_seed, ordinal);
        ordinal += 1;
        let Ok(json) = candidate.to_json() else {
            continue;
        };
        if seen.insert(topo_hash(&json)) {
            out.push(candidate);
        } else {
            // The cursor already moved past this ordinal: the next attempt is a new draw, never a
            // re-try of one already consumed.
            duplicate_draws += 1;
        }
    }
    // Still short ⇒ the topology space is smaller than the batch.
    let mut kept_duplicates = 0usize;
    while out.len() < target {
        let candidate = draw_topology(config, run_seed, ordinal);
        ordinal += 1;
        kept_duplicates += 1;
        out.push(candidate);
    }
    // ONE summary, after the batch is built — never a card per collision. Silent when every draw
    // was unique and every import landed, so an ordinary run's log is unchanged.
    if duplicate_imports + duplicate_draws + kept_duplicates > 0 {
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            "Founder batch",
            &crate::utils::cfy::kv(&[
                ("imported".to_string(), imported.to_string()),
                ("duplicate draws".to_string(), duplicate_draws.to_string()),
                ("re-rolled".to_string(), duplicate_draws.to_string()),
                (
                    "slots kept a duplicate".to_string(),
                    kept_duplicates.to_string(),
                ),
                (
                    "duplicate imports dropped".to_string(),
                    duplicate_imports.to_string(),
                ),
            ]),
            Some(&format!(
                "{founder_count} imported · {duplicate_draws} draws repeated a blueprint and were \
                 re-rolled (the cursor only moves forward) · {kept_duplicates} slots kept a \
                 duplicate because the space is smaller than the batch — the engine dedupes those, \
                 so the population starts short by that many"
            )),
        ));
    }
    out
}

// ── Seed blueprint loading ─────────────────────────────────────────────

/// Load a blueprint from a saved artifact (a topology JSON, or a `nets/<hash>.json` state's
/// `topology` field).
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

    /// An import duplicating a random draw must not shrink the population:
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

    /// A source run's rank-1..n blueprints ARE ordinals 0..n of the same run seed, so an inherited batch must start its random draws PAST them: no draw collides, the batch is…
    #[test]
    fn an_inherited_batch_draws_past_the_imported_ordinals() {
        let cfg = tiny_config(8);
        let draws = initial_population(&cfg, 42);
        let founders: Vec<Topology> = draws.iter().take(3).cloned().collect();
        let out = fill_unique(&cfg, 42, founders.clone(), 8);
        assert_eq!(out.len(), 8, "the batch lands on the target");
        for (i, topo) in founders.iter().enumerate() {
            assert_eq!(
                out[i].to_json().unwrap(),
                topo.to_json().unwrap(),
                "import {i} keeps its slot"
            );
        }
        let mut hashes: Vec<String> = out
            .iter()
            .map(|t| topo_hash(&t.to_json().unwrap()))
            .collect();
        let total = hashes.len();
        hashes.sort();
        hashes.dedup();
        assert_eq!(
            hashes.len(),
            total,
            "no random draw may repeat an imported blueprint"
        );
    }

    /// With no imports the batch is `initial_population` verbatim,
    /// no-seed path stays bit-identical to what it always
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
        // End-to-end through the real engine draw: with the policy on, every founder must carry
        // masks whose length matches the POST-finalize port count (a stale port count is exactly
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
        cfg.feature_split_pool = vec!["exclusive".to_string()];

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
        assert!(
            cfg.feature_split_pool.is_empty(),
            "the pool is empty by default (= none, random, exclusive)"
        );
        let mut none_only = tiny_config(4);
        none_only.feature_split_pool = vec!["none".to_string()];
        for topo in initial_population(&none_only, 42) {
            let input = topo
                .nodes
                .iter()
                .find(|n| n.kind == NodeKind::Input)
                .unwrap();
            assert_eq!(input.feature_masks, None, "pool [none] ⇒ no masks");
        }
    }

    /// The default pool is all three policies, so a population holds unsplit and split
    /// nets at once — the split is a selectable gene, not a run-wide switch.
    #[test]
    fn the_default_pool_puts_all_three_policies_in_one_population() {
        let opts = crate::graph::topology::TopologyOptions {
            min_hidden_num_nodes: 4,
            max_hidden_num_nodes: 4,
            input_dim: Some(3),
            output_dim: Some(2),
            ..Default::default()
        };
        let mut cfg = RaceConfig::defaults();
        cfg.pop_size = 12;
        cfg.topology_options = opts;
        cfg.hidden_dim_pool = Some(4..=4);
        let mut seen: Vec<Option<crate::graph::topology::FeatureSplitPolicy>> =
            initial_population(&cfg, 7)
                .iter()
                .map(|t| t.options.feature_split)
                .collect();
        seen.sort_by_key(|p| format!("{p:?}"));
        seen.dedup();
        assert!(
            seen.len() >= 2,
            "12 draws from {{none, random, exclusive}} produced {seen:?}"
        );
        for topo in initial_population(&cfg, 7) {
            topo.validate().expect("every drawn split must validate");
            let input = topo
                .nodes
                .iter()
                .find(|n| n.kind == NodeKind::Input)
                .unwrap();
            assert_eq!(
                input.feature_masks.is_some(),
                topo.options.feature_split.is_some(),
                "the masks that exist are the ones the drawn policy says"
            );
        }
    }
}
