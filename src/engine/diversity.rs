//! Population diversity — are the live nets still DIFFERENT from each other?
//!
//! A long race can end with every net doing the same thing without the logs ever
//! saying so: the elite table shows the best net, the rollup shows the population's
//! means, and neither notices that the population has collapsed onto one blueprint.
//! This module measures that directly.
//!
//! **The metric is STRUCTURAL and deterministic.** A net becomes a multiset of
//! genes — one token per node (`N:<kind>:<combine>:<std>:<activation>:<width>`), one
//! per output port (the activation that port emits), and one per Input port's
//! feature-mask width — and two nets are compared by multiset Jaccard distance:
//!
//! ```text
//! Δ(a, b) = 1 − (Σ min(a_t, b_t)) / (Σ max(a_t, b_t))      // 0 = identical, 1 = disjoint
//! ```
//!
//! **No matrix is materialised.** Δ is symmetric, so only the upper triangle is
//! walked (`i < j`) — one pair at a time, `O(1)` memory beyond the genome set — and
//! the pair count is therefore `n(n−1)/2` (a 100-net population is 4,950 pairs, not
//! 10,000).
//!
//! `mean pairwise Δ` is the mean of that distance over pairs of live nets, so it
//! starts high (founders share only the Input/Output skeleton) and falls as
//! crossover and selection homogenise the population. `distinct` is the blunter
//! companion number: how many byte-different blueprints are alive at all.
//!
//! **The census is the shape reading.** `nodes`, `wires` and `dim` carry each
//! trait's mean AND its spread across the live nets, because a collapsing
//! population keeps its mean while the spread falls to zero: the mean says what the
//! population looks like, the spread says whether it still varies.
//!
//! **What it is not.** A PROXY for behavioural convergence: two nets can share every
//! gene and still hold different weights, and two differently-wired nets can compute
//! nearly the same policy. The engine never runs the environment (the trainer owns
//! it), so a behavioural number — action agreement on a probe set, say — belongs
//! beside the trainer's own eval games, not here.

use crate::graph::node::{CombineOp, NodeKind, StandardizeOp};
use crate::graph::topology::Topology;
use crate::state::RaceState;
use std::collections::{HashMap, HashSet};

/// One reading of population diversity (`log_diversity` renders it).
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
///
/// `spread` is the POPULATION std (divide by `n`, not `n − 1`): the live nets are the
/// population itself, not a draw from a larger one. A collapse reads as `spread → 0`
/// while `mean` barely moves, which is why both are printed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Trait {
    /// The trait's mean across the live nets.
    pub mean: f32,
    /// The trait's population std — how much the nets still differ on it.
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

/// One net's structural reading: the gene multiset the pair metric uses, plus the
/// census legs it was read from. Built in ONE pass over the blueprint.
pub(crate) struct NetShape {
    /// The net's genes — exactly what [`delta`] compares.
    pub genome: HashMap<String, usize>,
    /// Nodes in the blueprint.
    pub nodes: usize,
    /// Wires in the blueprint.
    pub wires: usize,
    /// Mean hidden-node width; `0.0` for a net with no hidden node.
    pub dim: f32,
}

/// Read one net's structure: the pair metric's genes plus the census legs.
///
/// Costs one blueprint parse and one token build — the same work the population
/// census does per net, so a caller that already has the `Topology` pays nothing
/// extra for the shape.
pub(crate) fn shape_of(topo: &Topology) -> NetShape {
    let hidden: Vec<f32> = topo
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Hidden)
        .filter_map(|n| n.hidden_dim.map(|d| d as f32))
        .collect();
    NetShape {
        genome: genes(topo),
        nodes: topo.nodes.len(),
        wires: topo.connections.len(),
        dim: trait_of(&hidden).mean,
    }
}

/// Mean Δ from ONE net to a set of reference nets: `0.0` means the net is
/// structurally identical to every reference, `1.0` that it shares nothing with any
/// of them.
///
/// The reference is any iterator of shapes, not a slice, so a caller can measure to
/// a SUBSET without materialising one: the stop-time elite card compares a champion
/// to the OTHER champions (never to itself, which would print a meaningless `0.00`),
/// while the straggler card compares to all of them — one function, one arithmetic.
///
/// `None` when there is no reference to measure against — a run whose elite set is
/// empty (or whose blueprints would not parse) must print `—`, never `0.0`, which
/// would read as "identical to the champions".
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

/// Below this many live nets every pair is compared exactly (`n(n−1)/2` pairs: 100
/// nets = 4,950, 512 nets = 130,776). Above it the pairs are SAMPLED, because the
/// count grows quadratically and the line prints every `checkpoint_every` clocks.
const ALL_PAIRS_UP_TO: usize = 512;
/// Pairs drawn when sampling (enough to move slowly, cheap enough for every clock).
const SAMPLED_PAIRS: usize = 4096;

/// A net's structural genome: gene token → how many times it occurs.
pub(crate) fn genes(topo: &Topology) -> HashMap<String, usize> {
    let mut out: HashMap<String, usize> = HashMap::new();
    let mut bump = |token: String| *out.entry(token).or_insert(0) += 1;
    for node in &topo.nodes {
        // Effective ops, resolved exactly as the forward pass resolves them
        // (`network.rs`: no per-node override ⇒ `Add` / no standardisation), so two
        // nets that express the same thing two ways read as equal. That means the
        // standardize leg too: `None` and `Some(Identity)` are one architecture (both
        // run no standardisation), so they must be one gene — the raw `Option` would
        // have made a spelling-only mutation look like a structural change.
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
                bump(format!("M:{}", mask.iter().filter(|kept| **kept != 0).count()));
            }
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
        // A blueprint that will not parse is a bug, not a reason to fail a run: the
        // net simply does not enter the comparison (`live` still counts it).
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
    // Pairs: every pair for a small population, a SEEDED sample above the cap — the
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
        assert!((delta(&a, &b) - 0.25).abs() < 1e-6, "Δ = 1/|A|, got {}", delta(&a, &b));
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

    #[test]
    fn mean_delta_to_reads_the_distance_to_the_reference_set() {
        let shape = |tokens: &[(&str, usize)]| NetShape {
            genome: genome(tokens),
            nodes: 1,
            wires: 1,
            dim: 4.0,
        };
        let straggler = shape(&[("W:Tanh", 1)]);
        assert_eq!(
            mean_delta_to(&straggler, &[]),
            None,
            "no reference ⇒ `—`, never a fake 0.0"
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

    /// `None` and `Some(Identity)` are spelled differently and mean the same thing
    /// (`network.rs`: `None` ⇒ no standardisation). One architecture must be one
    /// genome, or a spelling-only mutation would read as a structural change.
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
        assert!((delta(&a, &b) - 2.0 / 3.0).abs() < 1e-6, "Δ = 1 − 1/3, got {}", delta(&a, &b));
    }
}
