//! Population diversity — are the live nets still DIFFERENT from each other?
//!
//! A long race can end with every net doing the same thing without the logs ever
//! saying so: the elite table shows the best net, the rollup shows the population's
//! means, and neither notices that the population has collapsed onto one blueprint.
//! This module measures that directly.
//!
//! **The metric is STRUCTURAL and deterministic.** A net becomes a multiset
//! genes — one token per node (`N:<kind>:<combine>:<std>:<activation>:<width>`), one
//! per output port (the activation that port emits), one per Input port's
//! feature-mask width, and one per WIRE (`E:<from_kind>><to_kind>`) — and two nets
//! are compared by multiset Jaccard distance:
//!
//! ```text
//! Δ(a, b) = 1 − (Σ min(a_t, b_t)) / (Σ max(a_t, b_t))      // 0 = identical, 1 = disjoint
//! ```
//!
//! **No matrix is materialised.** Δ is symmetric, so only the upper triangle
//! walked (`i < j`) — one pair at a time, `O(1)` memory beyond the genome set —
//! the pair count is therefore `n(n−1)/2` (a 100-net population is 4,950 pairs,
//! 10,000).
//!
//! `mean pairwise Δ` is the mean of that distance over pairs of live nets,
//! starts high (founders share only the Input/Output skeleton) and falls
//! crossover and selection homogenise the population. `distinct` is the blunter
//! companion number: how many byte-different blueprints are alive
//!
//! **The census is the shape reading.** `nodes`, `wires` and `dim` carry each
//! trait's mean AND its spread across the live nets, because a collapsing
//! population keeps its mean while the spread falls to zero: the mean says what
//! population looks like, the spread says whether it still varies.
//!
//! **What it is not.** A PROXY for behavioural convergence: two nets can share
//! gene and still hold different weights, and two differently-wired nets can compute
//! nearly the same policy. The engine never runs the environment (the trainer owns
//! it), so a behavioural number — action agreement on a probe set, say — belongs
//! beside the trainer's own eval games, not here.

use crate::graph::node::{CombineOp, NodeKind, StandardizeOp};
use crate::graph::topology::Topology;
use crate::state::RaceState;
use std::collections::{HashMap, HashSet};

/// One reading of population diversity (`log_diversity` renders
pub(crate) struct DiversityReport {
    /// Live nets in the population.
    pub live: usize,
    /// How many of them hold a DIFFERENT blueprint (byte-identical topology JSON):
    /// `live` means every net is unique, `1` means one clone army.
    pub distinct: usize,
    /// Mean pairwise multiset-Jaccard distance over the sampled pairs (0..1).
    pub mean_delta: f32,
    /// Pairs actually compared (all pairs up to a cap, then a seeded sample).
    pub pairs: usize,
    /// Per-net node count: mean and spread across the live nets.
    pub nodes: Trait,
    /// Per-net wire count: mean and spread across the live nets.
    pub wires: Trait,
    /// Hidden-node width, pooled over every hidden node in the population (one
    /// value per node, not one mean per net): mean and spread.
    pub dim: Trait,
}

/// One structural trait's population reading: its mean and its spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Trait {
    /// The trait's mean across the live nets.
    pub mean: f32,
    /// The trait's population std — how much the nets still differ
    pub spread: f32,
}

/// Mean and population spread of `values`. An empty slice reads `0.0 ± 0.0`
/// (nothing alive to describe) rather than NaN.
pub(crate) fn trait_of(values: &[f32]) -> Trait {
    if values.is_empty() {
        return Trait {
            mean: 0.0,
            spread: 0.0,
        };
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / values.len() as f32;
    Trait {
        mean,
        spread: var.sqrt(),
    }
}

/// One net's structural reading: the gene multiset the pair metric uses, plus the census legs
/// was read
#[derive(Clone)]
pub(crate) struct NetShape {
    /// The net's genes — exactly what [`delta`] compares.
    pub genome: HashMap<String, usize>,
    /// Nodes in the blueprint.
    pub nodes: usize,
    /// Wires in the blueprint.
    pub wires: usize,
    /// Mean hidden-node width; `0.0` for a net with no hidden node.
    pub dim: f32,
    /// Share of the input each Input port reads on average: `1.0` = no split, `1/ports` = a
    /// perfect partition; `None` = no readable Input node.
    pub mask: Option<f32>,
    /// The wiring gene: the share of hidden nodes allowed a wire straight from the Input node.
    /// `1.0` = one Input port per hidden node (every node may read the raw input).
    pub reach: f32,
    /// The longest wire path from an Input node to any node: `Topology::depths` max. Input = 0,
    /// so `Input → H1 → H2 → Output` reads `3`. Report-only.
    pub depth: usize,
}

/// Read one net's structure: the pair metric's genes plus the census legs.
pub(crate) fn shape_of(topo: &Topology) -> NetShape {
    let hidden: Vec<f32> = topo
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Hidden)
        .filter_map(|n| n.hidden_dim.map(|d| d as f32))
        .collect();
    let mask = match topo.nodes.iter().find(|n| n.kind == NodeKind::Input) {
        None => None,
        Some(input) => match input.feature_masks.as_ref() {
            None => Some(1.0),
            Some(rows) => {
                let ports = rows.len();
                let width = rows.first().map_or(0, Vec::len);
                (ports > 0 && width > 0).then(|| {
                    let ones = rows.iter().flatten().filter(|b| **b != 0).count();
                    ones as f32 / (ports * width) as f32
                })
            }
        },
    };
    NetShape {
        reach: topo.options.input_reach,
        depth: topo.depths().into_iter().max().unwrap_or(0),
        genome: genes(topo),
        nodes: topo.nodes.len(),
        wires: topo.connections.len(),
        dim: trait_of(&hidden).mean,
        mask,
    }
}

/// Mean Δ from ONE net to a set of reference nets: `0.0` means the net is structurally identical
/// every reference, `1.0` that it shares nothing with any of them.
pub(crate) fn mean_delta_to<'a, I>(shape: &NetShape, reference: I) -> Option<f32>
where
    I: IntoIterator<Item = &'a NetShape>,
{
    let mut n = 0usize;
    let mut sum = 0.0f32;
    for other in reference {
        sum += delta(&shape.genome, &other.genome);
        n += 1;
    }
    if n == 0 {
        return None;
    }
    Some(sum / n as f32)
}

/// Below this many live nets every pair is compared exactly (`n(n−1)/2` pairs: 100 nets = 4,950,
/// 512 nets = 130,776).
const ALL_PAIRS_UP_TO: usize = 512;
/// Pairs drawn when sampling (enough to move slowly, cheap enough for every clock).
const SAMPLED_PAIRS: usize = 4096;

/// A net's structural genome: gene token → how many times it occurs.
pub(crate) fn genes(topo: &Topology) -> HashMap<String, usize> {
    let mut out: HashMap<String, usize> = HashMap::new();
    let mut bump = |token: String| *out.entry(token).or_insert(0) += 1;
    for node in &topo.nodes {
        // Effective ops, resolved exactly as the forward pass resolves them (`network.rs`: no
        // per-node override ⇒ `Add` / no standardisation), so two nets that express the same thing
        let combine = node.combine_op.unwrap_or(CombineOp::Add);
        let std = node.standardize.unwrap_or(StandardizeOp::Identity);
        bump(format!(
            "N:{:?}:{combine:?}:{std:?}:{:?}:{}",
            node.kind,
            node.activation,
            node.hidden_dim.unwrap_or(0)
        ));
        // One gene per EMITTED PORT: a port override replaces the node's activation
        // for that wire, so a net whose ports differ must not read as identical.
        for port in 0..node.num_outputs {
            let act = node
                .port_activations
                .as_ref()
                .and_then(|acts| acts.get(port).copied())
                .unwrap_or(node.activation);
            bump(format!("W:{act:?}"));
        }
        // The feature split is genome too: how many features each Input port keeps.
        if let Some(masks) = &node.feature_masks {
            for mask in masks {
                bump(format!(
                    "M:{}",
                    mask.iter().filter(|kept| **kept != 0).count()
                ));
            }
        }
    }
    // WIRES ARE GENOME TOO, one token per connection — keyed by the KINDS it joins, never by node
    // id: crossover and mutation renumber nodes, so the same graph with two ids swapped must read
    let kinds: HashMap<usize, NodeKind> = topo.nodes.iter().map(|n| (n.id, n.kind)).collect();
    for conn in &topo.connections {
        if let (Some(from), Some(to)) = (kinds.get(&conn.from.node), kinds.get(&conn.to.node)) {
            bump(format!("E:{from:?}>{to:?}"));
        }
    }
    out
}

/// Multiset Jaccard distance: `1 − |A ∩ B| / |A ∪ B|`, in `0..=1`.
pub(crate) fn delta(a: &HashMap<String, usize>, b: &HashMap<String, usize>) -> f32 {
    let mut inter = 0usize;
    let mut union = 0usize;
    for (token, count) in a {
        let other = b.get(token).copied().unwrap_or(0);
        inter += (*count).min(other);
        union += (*count).max(other);
    }
    for (token, count) in b {
        if !a.contains_key(token) {
            union += *count;
        }
    }
    if union == 0 {
        return 0.0;
    }
    1.0 - inter as f32 / union as f32
}

/// Measure the live population. `None` when there is nothing to compare (< 2 nets).
pub(crate) fn measure(state: &RaceState, run_seed: u64, clock: usize) -> Option<DiversityReport> {
    let hashes = state.live_hashes();
    let live = hashes.len();
    if live < 2 {
        return None;
    }
    let mut genomes: Vec<HashMap<String, usize>> = Vec::with_capacity(live);
    let mut blueprints: HashSet<&str> = HashSet::new();
    // Census legs: one value per NET (nodes, wires) and one per HIDDEN NODE (dim).
    let mut node_counts: Vec<f32> = Vec::with_capacity(live);
    let mut wire_counts: Vec<f32> = Vec::with_capacity(live);
    let mut widths: Vec<f32> = Vec::new();
    for hash in &hashes {
        let Some(net) = state.net(hash) else { continue };
        blueprints.insert(net.topology.as_str());
        // A blueprint that will not parse is a bug, not a reason to fail a run:
        // net simply does not enter the comparison (`live` still counts
        if let Ok(topo) = net.topology() {
            node_counts.push(topo.nodes.len() as f32);
            wire_counts.push(topo.connections.len() as f32);
            widths.extend(
                topo.nodes
                    .iter()
                    .filter(|n| n.kind == NodeKind::Hidden)
                    .filter_map(|n| n.hidden_dim.map(|d| d as f32)),
            );
            genomes.push(genes(&topo));
        }
    }
    if genomes.len() < 2 {
        return None;
    }
    // Pairs: every pair for a small population, a SEEDED sample above the cap —
    // seed depends on `(run_seed, clock)`, so a replay sees the same pairs.
    let mut rng = fastrand::Rng::with_seed(run_seed.wrapping_add(clock as u64));
    let mut pairs = 0usize;
    let mut sum = 0.0f32;
    if genomes.len() <= ALL_PAIRS_UP_TO {
        for i in 0..genomes.len() {
            for j in (i + 1)..genomes.len() {
                sum += delta(&genomes[i], &genomes[j]);
                pairs += 1;
            }
        }
    } else {
        for _ in 0..SAMPLED_PAIRS {
            let i = rng.usize(..genomes.len());
            let j = rng.usize(..genomes.len());
            if i == j {
                continue;
            }
            // Upper triangle: `Δ(a, b) == Δ(b, a)`, so the drawn order folds away.
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            sum += delta(&genomes[lo], &genomes[hi]);
            pairs += 1;
        }
    }
    if pairs == 0 {
        return None;
    }
    Some(DiversityReport {
        live,
        distinct: blueprints.len(),
        mean_delta: sum / pairs as f32,
        pairs,
        nodes: trait_of(&node_counts),
        wires: trait_of(&wire_counts),
        dim: trait_of(&widths),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn genome(tokens: &[(&str, usize)]) -> HashMap<String, usize> {
        tokens.iter().map(|(t, n)| (t.to_string(), *n)).collect()
    }

    #[test]
    fn identical_genomes_have_zero_distance() {
        let a = genome(&[("N:Hidden:Add:None:Tanh:48", 2), ("W:Tanh", 3)]);
        assert_eq!(delta(&a, &a), 0.0, "a net is not different from itself");
    }

    #[test]
    fn disjoint_genomes_have_full_distance() {
        let a = genome(&[("W:Tanh", 1)]);
        let b = genome(&[("W:SiLU", 1)]);
        assert_eq!(delta(&a, &b), 1.0, "nothing in common ⇒ 1.0");
    }

    #[test]
    fn one_missing_gene_out_of_n_is_one_over_n() {
        let a = genome(&[("W:Tanh", 3), ("W:SiLU", 1)]);
        let b = genome(&[("W:Tanh", 3)]);
        assert!(
            (delta(&a, &b) - 0.25).abs() < 1e-6,
            "Δ = 1/|A|, got {}",
            delta(&a, &b)
        );
    }

    #[test]
    fn delta_is_symmetric_so_the_upper_triangle_suffices() {
        let a = genome(&[("N:Hidden:Add:None:Tanh:48", 2), ("W:Tanh", 3)]);
        let b = genome(&[("N:Hidden:Min:Identity:SiLU:16", 1), ("W:ReLU", 2)]);
        assert_eq!(
            delta(&a, &b),
            delta(&b, &a),
            "Δ is a distance: the pair's order cannot matter"
        );
    }

    #[test]
    fn trait_of_reports_mean_and_population_spread() {
        let t = trait_of(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((t.mean - 5.0).abs() < 1e-6, "mean, got {}", t.mean);
        assert!(
            (t.spread - 2.0).abs() < 1e-6,
            "population std, got {}",
            t.spread
        );
    }

    /// A re-route is a structural change; a RENUMBERING is not. One token per wire,
    /// keyed by kinds, is the only label that satisfies both.
    #[test]
    fn a_reroute_is_a_gene_but_renumbering_is_not() {
        use crate::graph::node::{Activation, Node};
        use crate::graph::topology::{Connection, Port};
        let node = |id: usize, kind: NodeKind, num_inputs: usize, num_outputs: usize| Node {
            id,
            num_inputs,
            num_outputs,
            kind,
            hidden_dim: (kind == NodeKind::Hidden).then_some(8),
            activation: if kind == NodeKind::Hidden {
                Activation::ReLU
            } else {
                Activation::Identity
            },
            port_activations: None,
            combine_op: None,
            standardize: None,
            feature_masks: None,
        };
        let build = |skip: bool| {
            let mut g = Topology::new(0, None);
            g.nodes.push(node(0, NodeKind::Input, 0, 4));
            g.nodes.push(node(1, NodeKind::Hidden, 1, 1));
            g.nodes.push(node(2, NodeKind::Output, 2, 0));
            g.connections.push(Connection {
                from: Port { node: 0, index: 0 },
                to: Port { node: 1, index: 0 },
            });
            g.connections.push(Connection {
                from: Port { node: 1, index: 0 },
                to: Port { node: 2, index: 0 },
            });
            if skip {
                // The SAME nodes, one extra wire: input straight to the output.
                g.connections.push(Connection {
                    from: Port { node: 0, index: 1 },
                    to: Port { node: 2, index: 1 },
                });
            }
            g
        };
        let direct = genes(&build(false));
        let skip = genes(&build(true));
        // Node genes agree; the only difference is the added `E:Input>Output`.
        assert_eq!(direct.get("E:Input>Hidden"), Some(&1));
        assert_eq!(direct.get("E:Input>Output"), None);
        assert_eq!(skip.get("E:Input>Output"), Some(&1));
        assert!(
            delta(&direct, &skip) > 0.0,
            "a re-route must not read as the same architecture"
        );
        // Renumber every node: the same wiring must still read as distance zero.
        let mut moved = build(false);
        for n in &mut moved.nodes {
            n.id += 10;
        }
        for c in &mut moved.connections {
            c.from.node += 10;
            c.to.node += 10;
        }
        assert_eq!(
            delta(&genes(&build(false)), &genes(&moved)),
            0.0,
            "node ids are not a gene"
        );
    }

    #[test]
    fn mean_delta_to_reads_the_distance_to_the_reference_set() {
        let shape = |tokens: &[(&str, usize)]| NetShape {
            reach: 1.0,
            depth: 1,
            genome: genome(tokens),
            nodes: 1,
            wires: 1,
            dim: 4.0,
            mask: None,
        };
        let straggler = shape(&[("W:Tanh", 1)]);
        assert_eq!(
            mean_delta_to(&straggler, &[]),
            None,
            "no reference ⇒ `na`, never a fake 0.0"
        );
        assert_eq!(
            mean_delta_to(&straggler, &[shape(&[("W:Tanh", 1)])]),
            Some(0.0),
            "identical to the only elite"
        );
        assert_eq!(
            mean_delta_to(&straggler, &[shape(&[("W:SiLU", 1)])]),
            Some(1.0),
            "nothing in common"
        );
        // Mean over the reference SET: identical to one elite, disjoint from the other.
        let mixed = mean_delta_to(
            &straggler,
            &[shape(&[("W:Tanh", 1)]), shape(&[("W:SiLU", 1)])],
        );
        assert_eq!(mixed, Some(0.5), "the two legs average, got {mixed:?}");
    }

    /// `None` and `Some(Identity)` are spelled differently and mean the same thing (`network.rs`:
    /// `None` ⇒ no standardisation).
    #[test]
    fn a_missing_standardize_op_is_the_identity_gene() {
        let cfg = crate::engine::config::RaceConfig::defaults();
        let mut spelled_none = crate::engine::population::initial_population(&cfg, 42)
            .into_iter()
            .next()
            .expect("the founders are never empty");
        let mut spelled_identity = spelled_none.clone();
        for node in &mut spelled_none.nodes {
            node.standardize = None;
        }
        for node in &mut spelled_identity.nodes {
            node.standardize = Some(StandardizeOp::Identity);
        }
        assert_eq!(
            genes(&spelled_none),
            genes(&spelled_identity),
            "the identity op and no override are the same gene"
        );
        // A REAL standardisation is a different gene (and so is the combine op).
        for node in &mut spelled_identity.nodes {
            node.standardize = Some(StandardizeOp::LayerNorm);
        }
        assert_ne!(genes(&spelled_none), genes(&spelled_identity));
    }

    #[test]
    fn shape_of_reads_the_same_blueprint_the_pair_metric_does() {
        let cfg = crate::engine::config::RaceConfig::defaults();
        let topos = crate::engine::population::initial_population(&cfg, 42);
        let topo = &topos[0];
        let shape = shape_of(topo);
        assert_eq!(shape.nodes, topo.nodes.len());
        assert_eq!(shape.wires, topo.connections.len());
        assert_eq!(
            shape.depth,
            topo.depths().into_iter().max().unwrap_or(0),
            "depth is the longest wire path in the same blueprint"
        );
        assert_eq!(
            shape.genome,
            genes(topo),
            "the shape carries the pair metric's own genes, not a second spelling"
        );
        let hidden: Vec<f32> = topo
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Hidden)
            .filter_map(|n| n.hidden_dim.map(|d| d as f32))
            .collect();
        assert_eq!(shape.dim, trait_of(&hidden).mean);
    }

    #[test]
    fn shape_of_reads_depth_as_the_longest_wire_path() {
        use crate::graph::node::{Activation, Node};
        use crate::graph::topology::{Connection, Port};
        let node = |id: usize, kind: NodeKind, num_inputs: usize, num_outputs: usize| Node {
            id,
            num_inputs,
            num_outputs,
            kind,
            hidden_dim: (kind == NodeKind::Hidden).then_some(8),
            activation: if kind == NodeKind::Hidden {
                Activation::ReLU
            } else {
                Activation::Identity
            },
            port_activations: None,
            combine_op: None,
            standardize: None,
            feature_masks: None,
        };
        let mut topo = Topology::new(0, None);
        topo.nodes.push(node(0, NodeKind::Input, 0, 2));
        topo.nodes.push(node(1, NodeKind::Hidden, 1, 1));
        topo.nodes.push(node(2, NodeKind::Hidden, 1, 1));
        topo.nodes.push(node(3, NodeKind::Output, 2, 0));
        topo.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        topo.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        topo.connections.push(Connection {
            from: Port { node: 0, index: 1 },
            to: Port { node: 3, index: 0 },
        });
        topo.connections.push(Connection {
            from: Port { node: 2, index: 0 },
            to: Port { node: 3, index: 1 },
        });
        assert_eq!(
            shape_of(&topo).depth,
            3,
            "Input → H1 → H2 → Output is three wires deep"
        );
    }

    #[test]
    fn trait_of_on_nothing_is_zero_not_nan() {
        let t = trait_of(&[]);
        assert_eq!(
            (t.mean, t.spread),
            (0.0, 0.0),
            "no nets to describe ⇒ a readable zero, never NaN"
        );
    }

    #[test]
    fn counts_matter_not_just_presence() {
        // Same token set, different multiplicity: multiset Jaccard still separates them.
        let a = genome(&[("W:Tanh", 1)]);
        let b = genome(&[("W:Tanh", 3)]);
        assert!(
            (delta(&a, &b) - 2.0 / 3.0).abs() < 1e-6,
            "Δ = 1 − 1/3, got {}",
            delta(&a, &b)
        );
    }
}
