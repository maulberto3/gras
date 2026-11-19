//! Entrant generation: exactly one child per culled slot (Iter 5 contract).

use flodl::nn::Optimizer;
use flodl::tensor::Result;

/// Parent-pairing attempts before the crossover roll is spent as a no-op — bounds
/// the draw of fresh parent PAIRS (distinct from rolls/retries).
pub(crate) const MAX_PARENT_PAIRINGS: usize = 3;

/// Coin probability for the uniform feature-mask pick — mirrors the topology operator's own
/// per-node swap probability, so the split mixes at the same rate the nodes did.
const UNIFORM_MASK_PICK: f32 = 0.5;

/// Blend the two parents' feature-mask rows into the child's starting split.
///
/// Rows are aligned by PORT INDEX, and each parent is read modulo its OWN row count: a parent with
/// fewer ports than the other is not erased from the tail, so a 3-port split crossed with a 2-port
/// one mixes A's third partition against B's first. Port index is the only alignment two splits
/// ever share — port 2 of one parent feeds a consumer unrelated to port 2 of the other.
fn mix_feature_masks(
    a: Option<Vec<Vec<u8>>>,
    b: Option<Vec<Vec<u8>>>,
    a_policy: Option<FeatureSplitPolicy>,
    b_policy: Option<FeatureSplitPolicy>,
    op: CrossoverOp,
    rng: &mut fastrand::Rng,
) -> (Vec<Vec<u8>>, Option<FeatureSplitPolicy>) {
    let (a, b) = match (a, b) {
        (None, None) => return (Vec::new(), None),
        (Some(rows), None) => return (rows, a_policy),
        (None, Some(rows)) => return (rows, b_policy),
        (Some(a), Some(b)) => (a, b),
    };
    let n = a.len().max(b.len());
    // Mask rows span the raw input, so both parents' rows are the same width; the widest
    // is the safe bound when a donor carries fewer ports than the child needs.
    let width = a.iter().chain(b.iter()).map(|r| r.len()).max().unwrap_or(0);
    if width == 0 {
        return (Vec::new(), None);
    }
    // One BIT from the chosen donor. Each parent is read at `i % its own row count`, so a parent
    // with FEWER ports still has a view for every port the child needs: its split is not erased
    // from the surplus rows just because it had fewer partitions to hand out. A row too short for
    // `f` falls back to the other parent's bit — never a blind `0`.
    let bit = |i: usize, f: usize, take_a: bool| -> u8 {
        let (first, second) = if take_a { (&a, &b) } else { (&b, &a) };
        let at = |rows: &[Vec<u8>]| -> Option<u8> {
            let row = rows.get(i % rows.len().max(1))?;
            row.get(f).copied()
        };
        at(first).or_else(|| at(second)).unwrap_or(1)
    };
    let mixed: Vec<Vec<u8>> = match op {
        CrossoverOp::Uniform => (0..n)
            .map(|i| {
                (0..width)
                    .map(|f| bit(i, f, rng.f32() < UNIFORM_MASK_PICK))
                    .collect()
            })
            .collect(),
        CrossoverOp::OnePoint => {
            // ONE pivot over the whole mask genome, row-major: the earlier ports (and the head
            // the cut row) come from A, the rest from B.
            let total = n * width;
            let pivot = if total > 1 { rng.usize(1..total) } else { 0 };
            (0..n)
                .map(|i| {
                    (0..width)
                        .map(|f| bit(i, f, i * width + f < pivot))
                        .collect()
                })
                .collect()
        }
    };
    // The LABEL follows the CONTENT: rows taken verbatim from one parent inherit that parent's
    // policy, anything actually mixed is `Inherited` — a per-port blend of two partitions
    let policy = if mixed == a {
        a_policy
    } else if mixed == b {
        b_policy
    } else {
        Some(FeatureSplitPolicy::Inherited)
    };
    (mixed, policy)
}

/// MUTATION DRIFT: flip inherited feature-mask bits with probability `prob`.
pub(crate) fn drift_feature_masks(
    rows: &mut [Vec<u8>],
    prob: f32,
    run_seed: u64,
    clock: usize,
    child_idx: usize,
) {
    if !(prob > 0.0) {
        return;
    }
    let p = prob.min(1.0);
    // Salted: the weight seed for the same child is `derive_seed(run_seed, clock*1024+idx)`,
    // so an unsalted draw would make the mask flips follow the weight init stream.
    let seed = derive_seed(run_seed ^ 0x9E37_79B9_7F4A_7C15, clock * 1024 + child_idx);
    let mut rng = fastrand::Rng::with_seed(seed);
    for row in rows.iter_mut() {
        for bit in row.iter_mut() {
            if rng.f32() < p {
                *bit ^= 1;
            }
        }
    }
}

/// Each gene family draws from its OWN salted stream: adding a family later cannot shift the draws
/// of one already in use, so a resumed run keeps reproducing the alleles it already had.
const GENE_SALT_ACTIVATION: u64 = 0x0AC7_0000_0000_0001;
const GENE_SALT_COMBINE: u64 = 0x0AC7_0000_0000_0002;
const GENE_SALT_STANDARDIZE: u64 = 0x0AC7_0000_0000_0003;
const GENE_SALT_PORT: u64 = 0x0AC7_0000_0000_0004;
const GENE_SALT_DROPOUT: u64 = 0x0AC7_0000_0000_0005;

/// GENE DRIFT: the mask flip's coin, on the genes crossover can only DONATE.
pub(crate) fn drift_child_genes(
    topo: &mut Topology,
    prob: f32,
    activation_pool: &[Activation],
    combine_pool: &[CombineOp],
    standardize_pool: &[StandardizeOp],
    dropout_cap: f32,
    run_seed: u64,
    clock: usize,
    child_idx: usize,
) {
    if !(prob > 0.0) {
        return;
    }
    let p = prob.min(1.0);
    let stream = |salt: u64| {
        fastrand::Rng::with_seed(derive_seed(run_seed ^ salt, clock * 1024 + child_idx))
    };

    // Node genes — the same node set the random path draws them on (hidden
    let mut act = stream(GENE_SALT_ACTIVATION);
    let mut comb = stream(GENE_SALT_COMBINE);
    let mut std = stream(GENE_SALT_STANDARDIZE);
    for node in topo.nodes.iter_mut() {
        if node.kind != NodeKind::Hidden {
            continue;
        }
        if !activation_pool.is_empty() && act.f32() < p {
            node.activation = activation_pool[act.usize(0..activation_pool.len())];
        }
        if !combine_pool.is_empty() && comb.f32() < p {
            node.combine_op = Some(combine_pool[comb.usize(0..combine_pool.len())]);
        }
        if !standardize_pool.is_empty() && std.f32() < p {
            node.standardize = Some(standardize_pool[std.usize(0..standardize_pool.len())]);
        }
    }

    if !activation_pool.is_empty() {
        // Fill-only + port 0 restamped from the node gene, so a drifted activation
        // reaches every mirror of it and an untouched port keeps its allele.
        topo.assign_port_activations(activation_pool);
        let mut ports = stream(GENE_SALT_PORT);
        for node in topo.nodes.iter_mut() {
            if node.kind == NodeKind::Input || node.num_outputs <= 1 {
                continue;
            }
            let Some(slots) = node.port_activations.as_mut() else {
                continue;
            };
            for slot in slots.iter_mut().skip(1) {
                if ports.f32() < p {
                    *slot = activation_pool[ports.usize(0..activation_pool.len())];
                }
            }
        }
    }

    // Per-net gene: the cap is the run's `set_topology_dropout_prob`, so drift can
    // lower or zero the dropout but never widen the search past what was asked
    if dropout_cap > 0.0 {
        let mut drop = stream(GENE_SALT_DROPOUT);
        if drop.f32() < p {
            topo.options.dropout_prob = drop.f32() * dropout_cap;
        }
    }
}

#[cfg(test)]
mod drift_tests {
    use super::drift_feature_masks;

    fn rows() -> Vec<Vec<u8>> {
        vec![vec![1, 1, 0, 0], vec![0, 1, 1, 0]]
    }

    #[test]
    fn zero_probability_leaves_every_bit_exactly_as_inherited() {
        let mut r = rows();
        drift_feature_masks(&mut r, 0.0, 7, 3, 1);
        assert_eq!(r, rows());
    }

    #[test]
    fn full_probability_complements_every_bit() {
        let mut r = rows();
        drift_feature_masks(&mut r, 1.0, 7, 3, 1);
        assert_eq!(r, vec![vec![0, 0, 1, 1], vec![1, 0, 0, 1]]);
    }

    #[test]
    fn the_same_child_draws_the_same_bits_and_a_sibling_can_differ() {
        let mut a = rows();
        let mut b = rows();
        drift_feature_masks(&mut a, 0.5, 11, 4, 0);
        drift_feature_masks(&mut b, 0.5, 11, 4, 0);
        assert_eq!(a, b, "replay: same seed, clock and ordinal ⇒ same drift");
        // Some sibling ordinal in 1..=64 must differ, or the ordinal is not in the seed.
        let differs = (1..=64).any(|idx| {
            let mut c = rows();
            drift_feature_masks(&mut c, 0.5, 11, 4, idx);
            c != a
        });
        assert!(differs, "child_idx must reach the drift seed");
    }
}

#[cfg(test)]
mod gene_drift_tests {
    use super::*;
    use crate::graph::topology::TopologyOptions;

    fn acts() -> Vec<Activation> {
        crate::evolution::pools::all_activations()
    }

    fn combs() -> Vec<CombineOp> {
        crate::evolution::pools::all_combine_ops()
    }

    fn stds() -> Vec<StandardizeOp> {
        crate::evolution::pools::all_standardize_ops()
    }

    /// A small finalized net with activations already ported, i.e. the state a crossover
    /// child is in when the drift coins are cast.
    fn topo(seed: usize) -> Topology {
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(2),
            ..TopologyOptions::default()
        };
        let mut t = Topology::new(seed, Some(opts));
        t.create_random_hidden_nodes(3);
        t.finalize();
        t.assign_port_activations(&acts());
        t
    }

    type Genes = (
        Vec<(Activation, Option<CombineOp>, Option<StandardizeOp>)>,
        u32,
    );

    fn genes(t: &Topology) -> Genes {
        (
            t.nodes
                .iter()
                .map(|n| (n.activation, n.combine_op, n.standardize))
                .collect(),
            t.options.dropout_prob.to_bits(),
        )
    }

    #[test]
    fn zero_probability_leaves_every_gene_exactly_as_inherited() {
        let mut t = topo(7);
        let before = genes(&t);
        drift_child_genes(&mut t, 0.0, &acts(), &combs(), &stds(), 0.25, 7, 3, 1);
        assert_eq!(genes(&t), before);
    }

    #[test]
    fn full_probability_reaches_the_genes_without_changing_the_shape() {
        let mut t = topo(7);
        let before = genes(&t);
        drift_child_genes(&mut t, 1.0, &acts(), &combs(), &stds(), 0.25, 7, 3, 1);
        let after = genes(&t);
        assert_eq!(
            after.0.len(),
            before.0.len(),
            "drift is content: no node is added or removed"
        );
        assert!(after.0 != before.0, "p=1 must reach the node genes");
        assert!(
            after.1 != before.1,
            "p=1 must re-draw dropout when a cap is given"
        );
    }

    #[test]
    fn dropout_stays_inside_the_cap_for_every_child_ordinal() {
        for idx in 0..32 {
            let mut t = topo(7);
            drift_child_genes(&mut t, 0.5, &acts(), &combs(), &stds(), 0.25, 7, 3, idx);
            let d = t.options.dropout_prob;
            assert!(
                (0.0..=0.25).contains(&d),
                "child {idx}: dropout {d} escaped 0.0..=cap"
            );
        }
    }

    #[test]
    fn the_same_child_draws_the_same_genes_and_a_sibling_can_differ() {
        let mut a = topo(7);
        let mut b = topo(7);
        drift_child_genes(&mut a, 0.5, &acts(), &combs(), &stds(), 0.25, 11, 4, 0);
        drift_child_genes(&mut b, 0.5, &acts(), &combs(), &stds(), 0.25, 11, 4, 0);
        assert_eq!(genes(&a), genes(&b), "replay: same seed, clock and ordinal");
        let differs = (1..=64).any(|idx| {
            let mut c = topo(7);
            drift_child_genes(&mut c, 0.5, &acts(), &combs(), &stds(), 0.25, 11, 4, idx);
            genes(&c) != genes(&a)
        });
        assert!(differs, "child_idx must reach the drift seeds");
    }
}

/// One hidden node's HERITABLE content — the fields an operator actually moves when it swaps
/// node.
#[derive(Clone, PartialEq)]
struct NodeGene {
    num_inputs: usize,
    num_outputs: usize,
    hidden_dim: Option<usize>,
    activation: Activation,
    combine_op: Option<CombineOp>,
    standardize: Option<StandardizeOp>,
}

/// The hidden-node genes of a topology, in node order.
fn hidden_genes(topo: &Topology) -> Vec<NodeGene> {
    topo.nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Hidden)
        .map(|n| NodeGene {
            num_inputs: n.num_inputs,
            num_outputs: n.num_outputs,
            hidden_dim: n.hidden_dim,
            activation: n.activation,
            combine_op: n.combine_op,
            standardize: n.standardize,
        })
        .collect()
}

/// How much of `fitter` a candidate still carries: `(nodes, hidden_dim mass)` — a MULTISET
/// intersection, so two identical nodes satisfy two different origins and a node that matches
fn genes_from(fitter: &[NodeGene], candidate: &[NodeGene]) -> (usize, usize) {
    let mut pool: Vec<&NodeGene> = fitter.iter().collect();
    let mut nodes = 0usize;
    let mut mass = 0usize;
    for gene in candidate {
        if let Some(pos) = pool.iter().position(|g| **g == *gene) {
            pool.remove(pos);
            nodes += 1;
            mass += gene.hidden_dim.unwrap_or(0);
        }
    }
    (nodes, mass)
}

/// Whether the RECIPROCAL side (`donor`) keeps more of the fitter parent than `base`.
fn keep_donor_side(fitter: &[NodeGene], base: &[NodeGene], donor: &[NodeGene]) -> bool {
    let (base_nodes, base_mass) = genes_from(fitter, base);
    let (donor_nodes, donor_mass) = genes_from(fitter, donor);
    donor_nodes > base_nodes || (donor_nodes == base_nodes && donor_mass > base_mass)
}

/// Whether two built nodes are the same GENE: the fields a cut moves, plus the dims the build
/// derived.
pub(crate) fn same_gene(
    a: &crate::graph::node::Node,
    a_dims: (usize, usize),
    b: &crate::graph::node::Node,
    b_dims: (usize, usize),
) -> bool {
    // The INPUT width is deliberately absent: it follows the wiring, not the gene, and excluding
    // is what lets a true cross keep the shared input columns instead of dropping the whole tensor.
    a_dims.1 == b_dims.1
        && a.kind == b.kind
        && a.num_inputs == b.num_inputs
        && a.num_outputs == b.num_outputs
        && a.hidden_dim == b.hidden_dim
        && a.activation == b.activation
        && a.combine_op == b.combine_op
        && a.standardize == b.standardize
}

/// Pair a freshly built child's nodes with its donors by PROVENANCE: `moved[i]` is the operator's
/// word on the child's i-th hidden node — `Some(node)` names the exact node of `other` that donated
fn provenance_transplant_map(
    child: &Network,
    kept: &Network,
    other: Option<&Network>,
    moved: &[Option<usize>],
) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let hidden_nodes = |net: &Network| -> Vec<usize> {
        net.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.kind == NodeKind::Hidden)
            .map(|(i, _)| i)
            .collect()
    };
    let kept_hidden = hidden_nodes(kept);
    let mut from_kept = Vec::new();
    let mut from_other = Vec::new();
    let mut hidden = 0usize;
    for (child_idx, child_node) in child.nodes.iter().enumerate() {
        let slot = (child_node.kind == NodeKind::Hidden).then_some(hidden);
        if slot.is_some() {
            hidden += 1;
        }
        // `Some(i)` = this gene came from NODE `i` of the other parent; `None` or absent
        // = it stayed with the kept parent at the same hidden ordinal.
        let donor_slot = slot.and_then(|k| moved.get(k).copied().flatten());
        let (donor, donor_idx, to_other) = match donor_slot {
            Some(idx) => match other
                .map(|o| (o, idx))
                .filter(|(o, i)| o.nodes.get(*i).is_some_and(|n| n.kind == NodeKind::Hidden))
            {
                Some((o, i)) => (o, i, true),
                None => continue,
            },
            None => {
                let d = match slot {
                    Some(k) => kept_hidden.get(k).copied(),
                    None => kept.nodes.iter().position(|n| n.kind == child_node.kind),
                };
                match d {
                    Some(d) => (kept, d, false),
                    None => continue,
                }
            }
        };
        // Content is a FILTER, not a search: the index already names the parent,
        // drifted gene (different activation/ports/width) keeps the inventory init.
        if !same_gene(
            child_node,
            child.node_dims[child_idx],
            &donor.nodes[donor_idx],
            donor.node_dims[donor_idx],
        ) {
            continue;
        }
        if to_other {
            from_other.push((child_idx, donor_idx));
        } else {
            from_kept.push((child_idx, donor_idx));
        }
    }
    (from_kept, from_other)
}

#[cfg(test)]
mod inherit_map_tests {
    use super::*;

    fn net(weight_seed: usize, flip_hidden: bool) -> Network {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(crate::graph::node::Node::new_input(0, 2));
        graph
            .nodes
            .push(crate::graph::node::Node::new_hidden(1, 3, 2));
        graph
            .nodes
            .push(crate::graph::node::Node::new_output(2, 2, 1));
        graph.options.topology_seed = 7;
        graph.options.weight_seed = Some(weight_seed);
        if flip_hidden {
            graph.nodes[1].activation = Activation::Tanh;
        }
        graph.refresh_labels();
        graph.finalize();
        Network::build(&graph, flodl::Device::CPU).unwrap()
    }

    /// The map follows PROVENANCE: the gene the cut took from the other parent is paired
    /// there, everything else with the kept parent.
    #[test]
    fn a_moved_gene_inherits_from_the_parent_that_donated_it() {
        let kept = net(1, false);
        let other = net(2, false);
        let child = net(3, false);

        let (from_kept, from_other) =
            provenance_transplant_map(&child, &kept, Some(&other), &[Some(1)]);
        assert_eq!(from_other.len(), 1, "the one hidden gene came from `other`");
        assert_eq!(
            from_kept.len(),
            2,
            "input + output come from the kept parent"
        );
        assert_eq!(child.nodes[from_other[0].0].kind, NodeKind::Hidden);

        let (from_kept, from_other) =
            provenance_transplant_map(&child, &kept, Some(&other), &[None]);
        assert!(
            from_other.is_empty(),
            "no moved genes ⇒ nothing from `other`"
        );
        assert_eq!(from_kept.len(), 3);
    }

    /// A drifted gene matches neither donor's content, so it is left to the inventory
    /// instead of being credited to a parent that no longer owns that gene.
    #[test]
    fn a_drifted_gene_is_left_to_the_inventory() {
        let kept = net(1, false);
        let other = net(2, false);
        let child = net(3, true);
        let (from_kept, from_other) =
            provenance_transplant_map(&child, &kept, Some(&other), &[None]);
        assert_eq!(from_kept.len(), 2, "input + output only");
        assert!(from_other.is_empty());
    }
}

#[cfg(test)]
mod base_choice_tests {
    use super::*;

    fn genes(spec: &[(usize, Activation)]) -> Vec<NodeGene> {
        spec.iter()
            .map(|&(dim, activation)| NodeGene {
                num_inputs: 1,
                num_outputs: 1,
                hidden_dim: Some(dim),
                activation,
                combine_op: None,
                standardize: None,
            })
            .collect()
    }

    /// A cut that keeps one node of the fitter parent leaves the child mostly the donor:
    /// keeping `base` on the strength of "the fitter one is the base" would be wrong.
    #[test]
    fn an_early_cut_keeps_the_donor_side() {
        let fitter = genes(&[(8, Activation::ReLU); 6]);
        let base = genes(&[(8, Activation::ReLU)]);
        let donor = genes(&[(8, Activation::ReLU); 5]);
        assert!(keep_donor_side(&fitter, &base, &donor));
    }

    /// The mirror case: a deep cut leaves the fitter parent's body in `base`.
    #[test]
    fn a_deep_cut_keeps_the_base_side() {
        let fitter = genes(&[(8, Activation::ReLU); 6]);
        let base = genes(&[(8, Activation::ReLU); 5]);
        let donor = genes(&[(8, Activation::ReLU)]);
        assert!(!keep_donor_side(&fitter, &base, &donor));
    }

    /// Equal node counts: the side holding the WIDE gene carries more of the parent.
    #[test]
    fn equal_counts_break_on_the_wider_gene() {
        let fitter = genes(&[(64, Activation::ReLU), (8, Activation::ReLU)]);
        let base = genes(&[(8, Activation::ReLU)]);
        let donor = genes(&[(64, Activation::ReLU)]);
        assert!(keep_donor_side(&fitter, &base, &donor));
    }

    /// A complete tie (and time determinism) keeps the base side.
    #[test]
    fn a_full_tie_keeps_the_base() {
        let fitter = genes(&[(8, Activation::ReLU), (8, Activation::Tanh)]);
        let base = genes(&[(8, Activation::Tanh)]);
        let donor = genes(&[(8, Activation::ReLU)]);
        assert!(!keep_donor_side(&fitter, &base, &donor));
    }
}

use crate::engine::config::CrossoverOp;
use crate::graph::network::Network;
use crate::graph::node::{Activation, CombineOp, NodeKind, StandardizeOp};
use crate::graph::topology::{CxReport, FeatureSplitPolicy, Topology};
use crate::state::NetState;
use crate::utils::seed::derive_seed;

use super::core::CoreEngine;

/// A freshly-built child net being caught up solo before it rejoins the group.
/// Holds its state, live `Network`, and `Optimizer` (Adam state carries forward).
pub struct RaceChild {
    /// The child's persisted state (also written to `nets/<hash>.json`).
    pub state: NetState,
    /// Live network; coefficients mutate in place during training.
    pub net: Network,
    /// Optimizer whose Adam state carries forward across steps.
    pub optimizer: Box<dyn Optimizer>,
    /// What the crossover operator exchanged (`uniform` only), so the run ledger can
    /// total it without reparsing the lineage string.
    pub cx: Option<CxReport>,
}

impl CoreEngine {
    // ── Child generation (Iter 5: roulette → crossover → mutate?) ───────────

    /// Generate exactly one child for a culled slot: deterministic
    /// `derive_seed(run_seed, clock)` — crossover (roulette parents) or a fresh random draw.
    pub(crate) fn generate_child(
        &mut self,
        clock: usize,
        child_idx: usize,
    ) -> Result<Option<RaceChild>> {
        // Same-clock siblings must differ: mix the child ordinal into the derivation,
        // or identical children collapse the pop at dedupe.
        let roll_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx);
        let mut rng = fastrand::Rng::with_seed(roll_seed);

        // Every dealt roll IS a crossover attempt: the channel's turnover decided how many rolls this clock, so there is no probability left to draw.
        match self.evolved_child(clock, child_idx, &mut rng) {
            // Crossover produced nothing viable: the roll does NOTHING (no random
            // fallback). NO log here — the caller logs one line per roll.
            Ok(None) => Ok(None),
            other => other,
        }
    }

    /// The evolved branch: roulette parents → crossover → maybe mutate. Called for EVERY dealt
    /// roll; a failed pairing is a spent roll (no fallback).
    fn evolved_child(
        &mut self,
        clock: usize,
        child_idx: usize,
        rng: &mut fastrand::Rng,
    ) -> Result<Option<RaceChild>> {
        // Ok(None) = the roll produced no child (no compatible pairing); the caller
        // treats it as a spent roll, not an error. 1. Ranking: best-first by smoothed fitness.
        if self.state.live_hashes().is_empty() {
            return Err(flodl::tensor::TensorError::new(
                "race: no live nets to select parents from",
            ));
        }
        let direction = self.fitness.direction();
        // Parent selection prefers nets that HAVE a verdict.
        let ranked = {
            let measured = self.ranked_live();
            if measured.len() >= 2 {
                measured
            } else {
                self.ranked_live_all()
            }
        };
        let scores: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();

        // 3. Crossover: up to 3 attempts; a no-op (incompatible dims) retries
        // FRESH roulette-drawn parents.
        let mut child_topo = None;
        // Inherited feature-mask rows for the child (empty = neither parent carried a split).
        let mut child_masks: Vec<Vec<u8>> = Vec::new();
        // Provenance of `child_masks`, stamped onto the child's blueprint
        // policy it declares is the one its rows actually came
        let mut child_policy: Option<FeatureSplitPolicy> = None;
        // What the operator exchanged, plus the per-slot donor NODES the weight transplant follows.
        let mut cx_report: Option<CxReport> = None;
        let mut cx_note = String::new();
        // Operator pool resolved once per child (not per attempt) so retries vary
        // PARENTS, not the op. Uniform uses its own swap_prob — no magic 0.5.
        let ops = self
            .config
            .resolved_crossover_ops()
            .map_err(|e| flodl::tensor::TensorError::new(&e))?;
        let op = ops[rng.usize(..ops.len())];
        // Parent count is a property of the OPERATOR, not a config knob:
        // CrossoverOp is binary (`op.required_parents()`).
        let n_parents = op.required_parents();
        // Up to 3 attempts, each with FRESH parent draws (the configured selection method) —
        // retrying only the operator would burn all attempts on a doomed pairing.
        let mut parent_hashes: Vec<String> = Vec::new();
        // Which parents' TRAINED coefficients the child inherits, and — for a cut —
        // which genes provably came from the OTHER parent (`CxReport::moved`).
        let mut inherit_kept: Option<String> = None;
        let mut inherit_other: Option<String> = None;
        for _ in 0..MAX_PARENT_PAIRINGS {
            // 2.
            let method = self.config.crossover_selection.clone();
            let mut parent_positions: Vec<usize> = (0..n_parents)
                .filter_map(|_| method.draw_one(&scores, direction, rng))
                .collect();
            if n_parents == 2
                && parent_positions.len() == 2
                && parent_positions[0] == parent_positions[1]
            {
                let others: Vec<usize> = (0..scores.len())
                    .filter(|&p| p != parent_positions[0])
                    .collect();
                if !others.is_empty() {
                    parent_positions[1] = others[rng.usize(..others.len())];
                }
            }
            // FITTER BASE: `ranked` is best-first (`ranked_live_impl` sorts on the direction),
            // the LOWER index is the better net.
            if self.config.crossover_fitter_base
                && parent_positions.len() == 2
                && parent_positions[1] < parent_positions[0]
            {
                parent_positions.swap(0, 1);
            }
            let attempt_parents: Vec<String> = parent_positions
                .iter()
                .map(|&p| ranked[p].0.clone())
                .collect();
            let mut pa = self.load_parent_topology(&attempt_parents[0])?;
            if n_parents == 2 {
                let mut pb = self.load_parent_topology(&attempt_parents[1])?;
                // Snapshot the FITTER parent's genes BEFORE the operator runs: a successful cut
                // mutates BOTH sides into two reciprocal children, so the original is gone
                let fitter_genes = self.config.crossover_fitter_base.then(|| hidden_genes(&pa));
                let cx = match op {
                    CrossoverOp::OnePoint => {
                        let report = Topology::cx_one_point_with_report(&mut pa, &mut pb, rng);
                        let fired = report.is_some();
                        cx_report = report;
                        fired
                    }
                    CrossoverOp::Uniform => {
                        // Per-swap probability for the uniform operator.
                        const UNIFORM_SWAP_PROB: f32 = 0.5;
                        let report = Topology::cx_uniform(&mut pa, &mut pb, UNIFORM_SWAP_PROB, rng);
                        let fired = report.is_some();
                        cx_report = report;
                        fired
                    }
                };
                if cx {
                    // CUT-AWARE KEEPER: the ordering above made `pa` the fitter parent, but the cut
                    // may have landed so early that `pa` is now MOSTLY the donor — keeping it would
                    let mut kept_side = 0usize;
                    if let Some(fitter) = &fitter_genes {
                        if keep_donor_side(fitter, &hidden_genes(&pa), &hidden_genes(&pb)) {
                            std::mem::swap(&mut pa, &mut pb);
                            kept_side = 1;
                        }
                    }
                    // Provenance belongs to the child that was KEPT, not to the operator's
                    // reciprocal's slots are the complement, so hand the report over to the side
                    if kept_side == 1 {
                        if let Some(rep) = cx_report.as_mut() {
                            std::mem::swap(&mut rep.moved, &mut rep.moved_b);
                        }
                    }
                    // Both parents' rows, in their post-finalize port numbering.
                    let (masks, policy) = mix_feature_masks(
                        pa.input_feature_masks(),
                        pb.input_feature_masks(),
                        pa.options.feature_split,
                        pb.options.feature_split,
                        op,
                        rng,
                    );
                    child_masks = masks;
                    child_policy = policy;
                    child_topo = Some(pa);
                    cx_note = match op {
                        CrossoverOp::OnePoint => "crossover-one-point",
                        CrossoverOp::Uniform => "crossover-uniform",
                    }
                    .to_string();
                    // Provenance, not a preference list: the cut records which genes
                    // took from which side, so each one takes that side's coefficients.
                    inherit_kept = Some(attempt_parents[kept_side].clone());
                    inherit_other = Some(attempt_parents[1 - kept_side].clone());
                    parent_hashes = attempt_parents;
                    break;
                }
                // No-op — next loop pass draws a fresh parent pair.
            } else {
                // Single-parent mode: the child is the parent with a new seed (asexual
                // reproduction; crossover needs two).
                child_masks = pa.input_feature_masks().unwrap_or_default();
                child_policy = pa.options.feature_split;
                child_topo = Some(pa);
                cx_note = "clone".to_string();
                inherit_kept = attempt_parents.first().cloned();
                parent_hashes = attempt_parents;
                break;
            }
        }
        let mut child_topo = match child_topo {
            Some(t) => t,
            None => {
                // MAX_PARENT_PAIRINGS exhausted → NO child this roll (no random fallback;
                // the population stays untouched). NO log — the caller's roll line carries
                return Ok(None);
            }
        };

        // NO part-mutation anywhere: crossover exploits, mutation explores.
        // The mutation channel's own turnover deals the immigrant rolls in the race loop.

        // Crossover children get per-port activations — the port pool
        // is part of the search space, not just the immigrant path.
        child_topo.assign_port_activations(
            &self
                .config
                .resolved_activation_pool()
                .unwrap_or_else(|_| crate::evolution::pools::all_activations()),
        );
        // Feature masks: INHERIT what the parents carried (the child's rows are mapped
        // post-finalize port count — extra ports stay unmasked).
        if child_masks.is_empty() {
            if let Some(policy) = child_topo.options.feature_split {
                child_topo.assign_feature_masks(policy);
            }
        } else {
            // Drift BEFORE the install: the rows are the gene, the label is restamped
            // from what they now are (a flipped `exclusive` partition stops being one).
            drift_feature_masks(
                &mut child_masks,
                self.config.drift_prob,
                self.header.run_seed,
                clock,
                child_idx,
            );
            child_topo.install_input_feature_masks(&child_masks, child_policy);
        }
        // GENE DRIFT: the same coin, on the genes crossover can only donate.
        if self.config.drift_prob > 0.0 {
            let activations = self
                .config
                .resolved_activation_pool()
                .unwrap_or_else(|_| crate::evolution::pools::all_activations());
            let combine = self
                .config
                .resolved_combine_pool()
                .unwrap_or_else(|_| crate::evolution::pools::all_combine_ops());
            let standardize = self
                .config
                .resolved_standardize_pool()
                .unwrap_or_else(|_| crate::evolution::pools::all_standardize_ops());
            drift_child_genes(
                &mut child_topo,
                self.config.drift_prob,
                &activations,
                &combine,
                &standardize,
                self.config.topology_options.dropout_prob,
                self.header.run_seed,
                clock,
                child_idx,
            );
        }
        Ok(Some(self.finalize_child(
            child_topo,
            clock,
            child_idx,
            &cx_note,
            cx_report,
            &parent_hashes,
            inherit_kept.as_deref(),
            inherit_other.as_deref(),
            false,
        )?))
    }

    /// Load a live net's topology by hash (parents must still be live —
    /// they were just stepped this loop iteration, so they
    fn load_parent_topology(&self, hash: &str) -> Result<Topology> {
        match self.state.net(hash) {
            Some(state) => state.topology(),
            None => Err(flodl::tensor::TensorError::new(&format!(
                "race: parent {hash} not live"
            ))),
        }
    }

    /// A fresh random child from the run's resolved pools (the `random` branch
    /// the last-resort fallback). Mirrors the generational `create_individual` shape.
    pub(crate) fn random_child(&mut self, clock: usize, child_idx: usize) -> Result<RaceChild> {
        let roll_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx);
        let mut rng = fastrand::Rng::with_seed(roll_seed);
        let topo_opts = self.config.topology_options;
        let n_hidden = rng.usize(topo_opts.min_hidden_num_nodes..=topo_opts.max_hidden_num_nodes);
        // Stamp the run's dataset dims + hidden-dim pool so the child trains
        // same shared stream the rest of the population uses.
        let mut opts = topo_opts;
        opts.input_dim = Some(self.header.input_dim);
        opts.output_dim = Some(self.header.output_dim);
        // Immigrants draw their split from the same pool the founders did,
        // population keeps every policy reachable, not just the ones it started
        opts.feature_split = super::population::draw_feature_split(&self.config, &mut rng);
        // The topology's internal rng must also derive from the run seed,
        // random child with the same n_hidden is structurally identical.
        let topo_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx) as usize;
        let mut opts = opts;
        // The wiring gene, drawn like a hidden width so a population mixes shallow and deep
        opts.input_reach = opts.draw_input_reach(&mut rng);
        let mut graph = Topology::new(topo_seed, Some(opts));
        graph.create_random_hidden_nodes(n_hidden);
        // Children draw from the config's resolved pools — an empty config
        // pool falls back to the full known-op set (empty ⇒
        let activation = self
            .config
            .resolved_activation_pool()
            .unwrap_or_else(|_| crate::evolution::pools::all_activations());
        let combine = self
            .config
            .resolved_combine_pool()
            .unwrap_or_else(|_| crate::evolution::pools::all_combine_ops());
        let standardize = self
            .config
            .resolved_standardize_pool()
            .unwrap_or_else(|_| crate::evolution::pools::all_standardize_ops());
        let hidden_dim_pool = self.config.hidden_dim_pool.clone().unwrap_or(4..=8);
        let stride = self.config.hidden_dim_stride.max(1);
        for node in &mut graph.nodes {
            if node.kind == NodeKind::Hidden {
                let n = ((hidden_dim_pool.end() - hidden_dim_pool.start()) / stride) + 1;
                node.hidden_dim = Some(hidden_dim_pool.start() + rng.usize(0..n) * stride);
                node.activation = activation[rng.usize(0..activation.len())];
                node.combine_op = Some(combine[rng.usize(0..combine.len())]);
                node.standardize = Some(standardize[rng.usize(0..standardize.len())]);
            }
        }
        graph.refresh_labels();
        graph.finalize();
        // Per-port activations AFTER finalize: `num_outputs` is final only once
        // `trim_orphaned_ports` compacted the ports.
        let act_pool = self
            .config
            .resolved_activation_pool()
            .unwrap_or_else(|_| crate::evolution::pools::all_activations());
        graph.assign_port_activations(&act_pool);
        // Feature split for immigrants too — the run's policy owns both paths,
        // masks must be drawn after finalize so the port count is already final.
        if let Some(policy) = graph.options.feature_split {
            graph.assign_feature_masks(policy);
        }
        // NO mutation roll on the immigrant path: a brand-new random topology
        // already maximal exploration; the mutation roll belongs to crossover
        self.finalize_child(
            graph,
            clock,
            child_idx,
            "random",
            None,
            &[],
            None,
            None,
            false,
        )
    }

    /// Shared child finalization: lineage string, `NetState` at step 0, and a built `Network` +
    /// `Optimizer` ready to train.
    fn finalize_child(
        &self,
        mut topo: Topology,
        clock: usize,
        child_idx: usize,
        origin: &str,
        cx_report: Option<CxReport>,
        parents: &[String],
        inherit_kept: Option<&str>,
        inherit_other: Option<&str>,
        mutated: bool,
    ) -> Result<RaceChild> {
        let new_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx) as usize;
        topo.options.topology_seed = new_seed;
        // The STRUCTURE seed above stays per-roll — that is the diversity.
        topo.options.weight_seed = Some(crate::utils::seed::weight_seed(self.header.run_seed));
        // `swapped=3/6` sits BEFORE `:parents=`, so every existing reader keeps working:
        // `lineage_note` splits on `:parents=`, `origin_cell` takes the first `:`-segment
        let origin = match &cx_report {
            Some(rep) => format!("{origin}:swapped={}", rep.label()),
            None => origin.to_string(),
        };
        let lineage = if parents.is_empty() {
            origin
        } else {
            format!("{origin}:parents={}", parents.join(","))
        };
        let lineage = if mutated {
            format!("{lineage}+mut")
        } else {
            lineage
        };
        // The child's step counter starts at 0; `entered_at_step` is patched to the run
        // clock right after. A newborn trains forward from the current clock (no replay).
        let mut state = NetState::new(&topo, 0, Some(lineage))?;
        state.entered_at_step = clock;
        // The LINEAGE is older than the child: the ancestry artifacts read this clock,
        // crossover of two long-lived parents is not credited with `clock` alone.
        state.lineage_born = parents
            .iter()
            .filter_map(|p| self.state.net(p).map(NetState::lineage_birth))
            .min();
        let _net_device = self.config.device();
        // Build from the run's LAYER INVENTORY: a piece's initial values
        // function of its own gene, not of what happened to be built before
        let mut inventory = crate::graph::inventory::LayerInventory::new(self.header.run_seed);
        let mut net = Network::build_with_inventory(&topo, self.config.device(), &mut inventory)?;
        // INHERITANCE: crossover recombines the BLUEPRINT only, so without this the child re-rolls
        // a fresh init and its parents' training is thrown away.
        if let Some(kept_hash) = inherit_kept {
            if let Some(kept) = self.networks.get(kept_hash) {
                let other = inherit_other.and_then(|h| self.networks.get(h));
                let moved: &[Option<usize>] = cx_report
                    .as_ref()
                    .map(|r| r.moved.as_slice())
                    .unwrap_or(&[]);
                let (from_kept, from_other) = provenance_transplant_map(&net, kept, other, moved);
                if !from_kept.is_empty() {
                    net.transplant_weights_from(kept, &from_kept)?;
                }
                if let Some(other) = other {
                    if !from_other.is_empty() {
                        net.transplant_weights_from(other, &from_other)?;
                    }
                }
            }
        }
        state.stamp_meta(&net, &self.meta_ctx);
        let optimizer = self.trainer.make_optimizer(&net);
        Ok(RaceChild {
            state,
            net,
            optimizer,
            cx: cx_report,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic child RNG — the mixer draws from it, so every case below
    /// pins its seed.
    fn rng(seed: u64) -> fastrand::Rng {
        fastrand::Rng::with_seed(seed)
    }

    fn row(bits: [u8; 2]) -> Vec<u8> {
        bits.to_vec()
    }

    /// One parent with a split, one without: the masked parent's rows survive, so a split cannot
    /// fade out of a lineage just because its partner was unmasked — and the child's LABEL
    #[test]
    fn mix_keeps_the_masked_parent_when_only_one_has_a_split() {
        let masked = vec![row([1, 0]), row([0, 1])];
        let (rows, policy) = mix_feature_masks(
            Some(masked.clone()),
            None,
            Some(FeatureSplitPolicy::Random),
            None,
            CrossoverOp::Uniform,
            &mut rng(7),
        );
        assert_eq!(rows, masked, "the masked parent's rows are kept verbatim");
        assert_eq!(
            policy,
            Some(FeatureSplitPolicy::Random),
            "rows from B ⇒ B's policy, not A's None (the old bug)"
        );

        let (rows, policy) = mix_feature_masks(
            None,
            Some(masked.clone()),
            None,
            Some(FeatureSplitPolicy::Random),
            CrossoverOp::OnePoint,
            &mut rng(7),
        );
        assert_eq!(rows, masked);
        assert_eq!(policy, Some(FeatureSplitPolicy::Random));

        // Neither parent carries one: nothing to inherit, so the caller falls back
        // the run's policy (or leaves the child unmasked).
        let (rows, policy) =
            mix_feature_masks(None, None, None, None, CrossoverOp::Uniform, &mut rng(1));
        assert!(rows.is_empty());
        assert_eq!(policy, None);
    }

    /// Both parents masked: the rows are RE-combined, so the honest label
    /// `Inherited` — a per-port blend of two partitions is not itself a partition.
    #[test]
    fn mixing_two_parents_labels_the_child_inherited() {
        let a = vec![row([1, 0]), row([0, 1]), row([1, 0])];
        let b = vec![row([0, 1]), row([1, 0]), row([0, 1])];
        let (rows, policy) = mix_feature_masks(
            Some(a.clone()),
            Some(b.clone()),
            Some(FeatureSplitPolicy::Exclusive),
            Some(FeatureSplitPolicy::Exclusive),
            CrossoverOp::Uniform,
            &mut rng(9),
        );
        assert_eq!(rows.len(), 3);
        assert_eq!(policy, Some(FeatureSplitPolicy::Inherited));

        // Rows that happen to equal ONE parent verbatim inherit THAT parent's label
        // (a one-point cut can land entirely on one side of the donor's rows).
        let (rows, policy) = mix_feature_masks(
            Some(a.clone()),
            Some(vec![row([1, 1]); 3]),
            Some(FeatureSplitPolicy::Exclusive),
            Some(FeatureSplitPolicy::Random),
            CrossoverOp::OnePoint,
            &mut rng(2),
        );
        if rows == a {
            assert_eq!(policy, Some(FeatureSplitPolicy::Exclusive));
        } else {
            assert_eq!(policy, Some(FeatureSplitPolicy::Inherited));
        }
    }

    /// Uniform: one coin per BIT, not per row — a child can inherit half a branch's
    /// receptive field. Same seed reproduces the same child; no bit is invented.
    #[test]
    fn uniform_mix_casts_one_coin_per_bit() {
        let a = vec![row([1, 1]), row([1, 1]), row([1, 1]), row([1, 1])];
        let b = vec![row([0, 0]), row([0, 0]), row([0, 0]), row([0, 0])];
        let (mixed, _) = mix_feature_masks(
            Some(a.clone()),
            Some(b.clone()),
            None,
            None,
            CrossoverOp::Uniform,
            &mut rng(3),
        );
        assert_eq!(mixed.len(), 4);
        for r in &mixed {
            assert_eq!(r.len(), 2, "a row keeps the mask width");
            assert!(r.iter().all(|&bit| bit <= 1), "invented a bit: {r:?}");
        }
        // 16 bits at p = 0.5 must mix WITHIN at least one row; otherwise this is the old
        // row-level copy wearing a new name.
        assert!(
            mixed.iter().any(|r| r != &row([1, 1]) && r != &row([0, 0])),
            "no row is a chimera: {mixed:?}"
        );
        assert_eq!(
            mixed,
            mix_feature_masks(
                Some(a),
                Some(b),
                None,
                None,
                CrossoverOp::Uniform,
                &mut rng(3)
            )
            .0,
            "the same seed must reproduce the same child"
        );
    }

    /// One-point: ONE cut across the flattened `(port × feature)` genome — A's bits before it, B's
    /// after, both sides non-empty.
    #[test]
    fn one_point_mix_cuts_the_flattened_mask_once() {
        let a = vec![row([1, 1]); 4];
        let b = vec![row([0, 0]); 4];
        let (mixed, _) = mix_feature_masks(
            Some(a),
            Some(b),
            None,
            None,
            CrossoverOp::OnePoint,
            &mut rng(5),
        );
        assert_eq!(mixed.len(), 4);
        let flat: Vec<u8> = mixed.iter().flatten().copied().collect();
        assert_eq!(flat.len(), 8, "8 bits = 4 ports × 2 features");
        assert!(
            flat.windows(2).all(|w| w[0] >= w[1]),
            "more than one cut: {flat:?}"
        );
        assert!(
            flat.contains(&1) && flat.contains(&0),
            "the pivot keeps both sides non-empty: {flat:?}"
        );
    }

    /// A donor with fewer ports than the child needs falls back to the OTHER parent's
    /// row for that port — never to an empty row.
    #[test]
    fn mix_maps_donors_by_port_index_with_a_fallback() {
        let a = vec![row([1, 0]), row([0, 1])];
        let b = vec![row([1, 1]); 4];
        let (mixed, _) = mix_feature_masks(
            Some(a),
            Some(b),
            None,
            None,
            CrossoverOp::Uniform,
            &mut rng(11),
        );
        assert_eq!(mixed.len(), 4, "one row per port the donors can cover");
        for (i, r) in mixed.iter().enumerate() {
            assert_ne!(r, &row([0, 0]), "port {i} fell back to an empty row");
        }
    }

    /// Asymmetric splits: a 3-port parent and a 2-port parent must BOTH reach every port. The old
    /// rule copied the longer parent's surplus row verbatim, so the shorter split could never
    /// influence it — port 2 was 100% A however the coins fell.
    #[test]
    fn mix_reads_each_parent_modulo_its_own_row_count() {
        let a = vec![row([1, 0]), row([0, 1]), row([1, 1])];
        let b = vec![row([0, 0]), row([0, 1])];
        let mut saw_b = false;
        for seed in 0..64u64 {
            let (rows, _) = mix_feature_masks(
                Some(a.clone()),
                Some(b.clone()),
                Some(FeatureSplitPolicy::Exclusive),
                Some(FeatureSplitPolicy::Exclusive),
                CrossoverOp::Uniform,
                &mut rng(seed),
            );
            assert_eq!(rows.len(), 3, "the longer split sets the row count");
            // Ports 0/1 pair a0↔b0 and a1↔b1; port 2 pairs a2 with B's FIRST row (2 % 2 = 0).
            for (f, bit) in rows[2].iter().enumerate() {
                assert!(
                    *bit == a[2][f] || *bit == b[0][f],
                    "seed {seed}: port 2 bit {f} came from neither a2 nor b0"
                );
            }
            if rows[2] != a[2] {
                saw_b = true;
            }
        }
        assert!(
            saw_b,
            "port 2 must be able to take B's view — the old rule always copied A's surplus row"
        );
    }
}
