//! Entrant generation: exactly one child per culled slot (Iter 5 contract).
//!
//! `generate_child` rolls `crossover_prob` → roulette-select parents →
//! crossover (≤3 attempts) — a failed crossover makes the roll a NO-OP
//! (no random fallback: random nets enter ONLY via the mutation path).
//! There is NO part-mutation: crossover children are pure recombination
//! (crossover exploits, mutation explores). The child always gets
//! its own `derive_seed(run_seed, clock)` weight seed and a `created_from`
//! lineage string. Catch-up replays the shared stream solo so the child
//! rejoin the group at the current clock.

use flodl::nn::Optimizer;
use flodl::tensor::Result;

/// Parent-pairing attempts before the crossover roll is spent as a no-op — bounds
/// the draw of fresh parent PAIRS (distinct from rolls/retries).
pub(crate) const MAX_PARENT_PAIRINGS: usize = 3;

/// Coin probability for the uniform feature-mask pick — mirrors the topology
/// operator's own per-node swap probability, so the split mixes at the same rate
/// the nodes did.
const UNIFORM_MASK_PICK: f32 = 0.5;

/// Blend the two parents' feature-mask rows into the child's starting split.
///
/// **The blend follows the operator's own granularity, at BIT level** — the split is a
/// genome, and its genes are bits, not rows: `Uniform` casts the same per-gene coin
/// `cx_uniform` casts (each bit from A or B, `UNIFORM_MASK_PICK`), and `OnePoint` makes
/// ONE cut across the flattened `(port × feature)` genome, taking A before the pivot and
/// B after it. Blending whole ROWS instead would hand a child a branch that is entirely
/// one parent's receptive field, which is coarser than the cut the topology received.
///
/// A parent that carries no split does not blank the child's: the masked parent wins, so
/// a split can't fade out of a lineage just because its partner was unmasked. Rows are
/// matched by port index and bits by feature index; a donor missing a cell falls back to
/// the other parent's cell (never to a blind `0` the run never had). Draws come from the
/// shared child RNG in a fixed order, so the same seed reproduces the same child.
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
    // One BIT from the chosen donor, falling back to the other parent's bit for that
    // feature — a missing cell means "the other parent's view", never a blind `0`.
    let bit = |i: usize, f: usize, take_a: bool| -> u8 {
        let (first, second) = if take_a { (&a, &b) } else { (&b, &a) };
        first
            .get(i)
            .and_then(|row| row.get(f).copied())
            .or_else(|| second.get(i).and_then(|row| row.get(f).copied()))
            .unwrap_or(1)
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
            // ONE pivot over the whole mask genome, row-major: the earlier ports (and
            // the head of the cut row) come from A, the rest from B. Both sides stay
            // non-empty whenever there is more than one bit to cut.
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
    // The LABEL follows the CONTENT: rows taken verbatim from one parent inherit that
    // parent's policy, anything actually mixed is `Inherited` — a per-port blend of two
    // partitions is not a partition, and a blend across split/unsplit is neither family.
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
///
/// This is the only heritable change the split can get after recombination: without it a
/// child's rows ARE its parents' rows, bit for bit, so selection can only choose among
/// patterns that already existed at birth (a fresh immigrant's rows are unrelated to any
/// parent, which gives nothing to climb). One bit at a time means a good architecture can
/// gain or drop a single feature and keep everything else.
///
/// **Determinism:** the flips come from a seed DERIVED from `(run_seed, clock, child_idx)`,
/// salted away from the weight seed, so `prob > 0` never consumes the shared stream and a
/// replay draws exactly the same bits. `prob <= 0` returns before building an RNG, which is
/// what keeps `0.0` bit-for-bit identical to the pre-drift code.
///
/// Cost is per ROW WIDTH, not per run: `0.005` flips one bit per ~50 children on a
/// 4-feature row and ~4 bits per ROW on a 784-feature one (× every port, so ~31 per
/// 8-port child). Expectation per row is `input_dim × p`, so `p = 0.5` erases the row
/// into a fresh coin flip rather than editing it — a pool draw, not drift.
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
    let seed = derive_seed(
        run_seed ^ 0x9E37_79B9_7F4A_7C15,
        clock * 1024 + child_idx,
    );
    let mut rng = fastrand::Rng::with_seed(seed);
    for row in rows.iter_mut() {
        for bit in row.iter_mut() {
            if rng.f32() < p {
                *bit ^= 1;
            }
        }
    }
}

/// Each gene family draws from its OWN salted stream: adding a family later cannot
/// shift the draws of one already in use, so a resumed run keeps reproducing the
/// alleles it already had.
const GENE_SALT_ACTIVATION: u64 = 0x0AC7_0000_0000_0001;
const GENE_SALT_COMBINE: u64 = 0x0AC7_0000_0000_0002;
const GENE_SALT_STANDARDIZE: u64 = 0x0AC7_0000_0000_0003;
const GENE_SALT_PORT: u64 = 0x0AC7_0000_0000_0004;
const GENE_SALT_DROPOUT: u64 = 0x0AC7_0000_0000_0005;

/// GENE DRIFT: the mask flip's coin, on the genes crossover can only DONATE.
///
/// `cx_*` swaps WHOLE nodes, so a child inherits its parents' activations, combine ops,
/// standardize ops and dropout — but nothing can ever EDIT one. The only way an allele
/// moves is another parent donating a different one, which means a good net can never
/// shed a single bad choice without being replaced wholesale.
///
/// | gene | coin |
/// |---|---|
/// | hidden `activation` | re-draw from the activation pool |
/// | hidden `combine_op` | re-draw from the combine pool |
/// | hidden `standardize` | re-draw from the standardize pool |
/// | per-port activation (`1..`) | re-draw from the activation pool |
/// | `dropout_prob` | uniform in `0.0..=cap`, never above what the run asked for |
///
/// Every one of them is CONTENT, not shape: no port is added or dropped, so the child
/// keeps its wiring and no rebuild is involved (unlike `hidden_dim` and the edges, which
/// move only by donation). Port 0 mirrors the node's activation by construction
/// (`assign_port_activations`), so it is restamped after the node coin rather than drawn.
///
/// **Determinism:** one salted stream per family, derived from
/// `(run_seed, clock, child_idx)`; `prob <= 0` returns before building any RNG, which is
/// what keeps the default bit-for-bit identical to the pre-drift build.
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

        // Node genes — the same node set the random path draws them on (hidden only).
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
        // lower or zero the dropout but never widen the search past what was asked for.
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

    type Genes = (Vec<(Activation, Option<CombineOp>, Option<StandardizeOp>)>, u32);

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
        assert!(after.1 != before.1, "p=1 must re-draw dropout when a cap is given");
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

/// One hidden node's HERITABLE content — the fields an operator actually moves when it
/// swaps a node. `id` (renumbered after every swap) and `port_activations` (re-assigned
/// by `assign_port_activations` a moment later) are deliberately absent: neither is what
/// crossed over, so counting them would credit the wrong parent.
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

/// How much of `fitter` a candidate still carries: `(nodes, hidden_dim mass)` — a
/// MULTISET intersection, so two identical nodes satisfy two different origins and a
/// node that matches nothing from the fitter is never credited.
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
///
/// A cut hands back two complementary children: `base` = fitter's prefix + other's
/// suffix, `donor` = other's prefix + fitter's suffix. If the pivot landed early in
/// `base`, the kept child is mostly the OTHER parent, and choosing it only because "the
/// fitter one is the base" would preserve the wrong skeleton — so the choice is made on
/// content, after the cut. Ties keep `base`, and a count tie is broken by the wider genes
/// (one `hidden_dim = 64` node carries more of a parent than one `hidden_dim = 8` does).
fn keep_donor_side(fitter: &[NodeGene], base: &[NodeGene], donor: &[NodeGene]) -> bool {
    let (base_nodes, base_mass) = genes_from(fitter, base);
    let (donor_nodes, donor_mass) = genes_from(fitter, donor);
    donor_nodes > base_nodes || (donor_nodes == base_nodes && donor_mass > base_mass)
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

        /// Generate exactly one child for a culled slot: deterministic from
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

            // Every roll is a CROSSOVER roll now: a not-fired or failed roll is simply
            // spent. Random whole nets enter only via mutation.
        let try_crossover = rng.f32() < self.config.crossover_prob;

        if try_crossover {
            match self.evolved_child(clock, child_idx, &mut rng) {
                    // Crossover produced nothing viable: the roll does NOTHING (no random
                    // fallback). NO log here — the caller logs one line per roll.
                Ok(None) => Ok(None),
                other => other,
            }
        } else {
            Ok(None) // probability roll said no — roll spent, no child
        }
    }

        /// The evolved branch: roulette parents → crossover → maybe mutate. Called only
        /// when `crossover_prob` fires; a failed pairing is a spent roll (no fallback).
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
            // Parent selection prefers nets that HAVE a verdict. `ranked_live_all` scores
            // an unmeasured net `0.0`, which reads as worst under Maximize but as BEST
            // under Minimize (a loss) — a newborn with an empty window could win a
            // tournament draw. Fall back to the all-live ranking only when fewer than two
            // nets are measured (clock 0: nothing has a reading yet and a child must still
            // be buildable from whatever is there).
        let ranked = {
            let measured = self.ranked_live();
            if measured.len() >= 2 {
                measured
            } else {
                self.ranked_live_all()
            }
        };
        let scores: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();

            // 3. Crossover: up to 3 attempts; a no-op (incompatible dims) retries with
            // FRESH roulette-drawn parents.
        let mut child_topo = None;
            // Inherited feature-mask rows for the child (empty = neither parent carried a
            // split). Read AFTER the operator: `cx_*` re-finalizes both parents, and
            // `finalize` now carries each parent's masks through its port remap, so the
            // rows line up with the ports the child actually ends up with.
        let mut child_masks: Vec<Vec<u8>> = Vec::new();
            // Provenance of `child_masks`, stamped onto the child's blueprint so the
            // policy it declares is the one its rows actually came from.
        let mut child_policy: Option<FeatureSplitPolicy> = None;
            // What the operator exchanged (`uniform` only — a one-point cut is not a
            // per-gene coin, so it reports nothing rather than a number that would read
            // like one). It rides the lineage string into `nets/<hash>.json`.
        let mut cx_report: Option<CxReport> = None;
        let mut cx_note = String::new();
            // Operator pool resolved once per child (not per attempt) so retries vary the
            // PARENTS, not the op. Uniform uses its own swap_prob — no magic 0.5.
        let ops = self
            .config
            .resolved_crossover_ops()
            .map_err(|e| flodl::tensor::TensorError::new(&e))?;
        let op = ops[rng.usize(..ops.len())];
            // Parent count is a property of the OPERATOR, not a config knob: every
            // CrossoverOp is binary (`op.required_parents()`).
        let n_parents = op.required_parents();
            // Up to 3 attempts, each with FRESH parent draws (the configured selection
            // method) — retrying only the operator would burn all attempts on a doomed
            // pairing.
        let mut parent_hashes: Vec<String> = Vec::new();
        for _ in 0..MAX_PARENT_PAIRINGS {
                // 2. Draw parents INSIDE the attempt loop with the configured method
                // (`set_crossover_selection`; roulette by default). `draw_one` is ONE
                // parent draw — `apply` would build a whole generation per parent.
                // Self-pairing guard: if B duplicated A, redraw B uniformly from the
                // OTHER positions.
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
                // FITTER BASE: `ranked` is best-first (`ranked_live_impl` sorts on the
                // direction), so the LOWER index is the better net. Putting it first makes
                // it `pa` — the parent the operator mutates, i.e. the child's skeleton.
                // ON by default: the base is otherwise just whichever draw came first,
                // which the attempts probe measured as the worse parent ~50% of the time.
                // The draw itself is untouched, only which of the two is the base.
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
                    // Snapshot the FITTER parent's genes BEFORE the operator runs: a
                    // successful cut mutates BOTH sides into two reciprocal children, so
                    // the original is gone by the time the keeper is chosen.
                let fitter_genes = self
                    .config
                    .crossover_fitter_base
                    .then(|| hidden_genes(&pa));
                let cx = match op {
                    CrossoverOp::OnePoint => Topology::cx_one_point(&mut pa, &mut pb, rng),
                    CrossoverOp::Uniform => {
                        // Per-swap probability for the uniform operator.
                        const UNIFORM_SWAP_PROB: f32 = 0.5;
                        let report = Topology::cx_uniform(&mut pa, &mut pb, UNIFORM_SWAP_PROB, rng);
                        cx_report = report;
                        report.is_some()
                    }
                };
                if cx {
                        // CUT-AWARE KEEPER: the ordering above made `pa` the fitter parent,
                        // but the cut may have landed so early that `pa` is now MOSTLY the
                        // donor — keeping it would preserve the wrong skeleton. Both sides
                        // are already built, so pick by CONTENT: keep the reciprocal that
                        // still carries more of the fitter parent, then hand THAT one to the
                        // mask mix so the prefix follows the keeper.
                    if let Some(fitter) = &fitter_genes {
                        if keep_donor_side(fitter, &hidden_genes(&pa), &hidden_genes(&pb)) {
                            std::mem::swap(&mut pa, &mut pb);
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
                    parent_hashes = attempt_parents;
                    break;
                }
                // No-op — next loop pass draws a fresh parent pair.
            } else {
                // Single-parent mode: the child is the parent with a new seed
                // (asexual reproduction; crossover needs two).
                    // Asexual reproduction inherits the parent's own split verbatim — no
                    // re-draw, since no rebuild ran and the rows already fit the ports.
                child_masks = pa.input_feature_masks().unwrap_or_default();
                child_policy = pa.options.feature_split;
                child_topo = Some(pa);
                cx_note = "clone".to_string();
                parent_hashes = attempt_parents;
                break;
            }
        }
        let mut child_topo = match child_topo {
            Some(t) => t,
            None => {
                    // MAX_PARENT_PAIRINGS exhausted → NO child this roll (no random fallback;
                    // the population stays untouched). NO log — the caller's roll line carries it.
                return Ok(None);
            }
        };

            // NO part-mutation anywhere: crossover exploits, mutation explores.
            // `mutate_prob` gates the immigrant rolls in the race loop.

        // Crossover children get per-port activations — the port pool
        // is part of the search space, not just the immigrant path.
        child_topo.assign_port_activations(
            &self
                .config
                .resolved_activation_pool()
                .unwrap_or_else(|_| crate::evolution::pools::all_activations()),
        );
            // Feature masks: INHERIT what the parents carried (the child's rows are
            // mapped to the post-finalize port count — extra ports stay unmasked). Only
            // when there is nothing to inherit does the run's policy draw a fresh split,
            // which is also the founder/immigrant path's behaviour.
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
            // GENE DRIFT: the same coin, on the genes crossover can only donate. Runs
            // after the masks and before the build, so the network is built from the
            // drifted blueprint. Content only — no port is added or dropped.
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
            false,
        )?))
    }

    /// Load a live net's topology by hash (parents must still be live —
    /// they were just stepped this loop iteration, so they are).
    fn load_parent_topology(&self, hash: &str) -> Result<Topology> {
        match self.state.net(hash) {
            Some(state) => state.topology(),
            None => Err(flodl::tensor::TensorError::new(&format!(
                "race: parent {hash} not live"
            ))),
        }
    }

        /// A fresh random child from the run's resolved pools (the `random` branch and
        /// the last-resort fallback). Mirrors the generational `create_individual` shape.
    pub(crate) fn random_child(&mut self, clock: usize, child_idx: usize) -> Result<RaceChild> {
        let roll_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx);
        let mut rng = fastrand::Rng::with_seed(roll_seed);
        let topo_opts = self.config.topology_options;
        let n_hidden = rng.usize(topo_opts.min_hidden_num_nodes..=topo_opts.max_hidden_num_nodes);
            // Stamp the run's dataset dims + hidden-dim pool so the child trains on the
            // same shared stream the rest of the population uses.
        let mut opts = topo_opts;
        opts.input_dim = Some(self.header.input_dim);
        opts.output_dim = Some(self.header.output_dim);
        // Immigrants draw their split from the same pool the founders did, so a
        // population keeps every policy reachable, not just the ones it started with.
        opts.feature_split = super::population::draw_feature_split(&self.config, &mut rng);
            // The topology's internal rng must also derive from the run seed, or every
            // random child with the same n_hidden is structurally identical.
        let topo_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx) as usize;
        let mut graph = Topology::new(topo_seed, Some(opts));
        graph.create_random_hidden_nodes(n_hidden);
        // Children draw from the config's resolved pools — an empty config
        // pool falls back to the full known-op set (empty ⇒ all).
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
            // Feature split for immigrants too — the run's policy owns both paths, and
            // masks must be drawn after finalize so the port count is already final.
        if let Some(policy) = graph.options.feature_split {
            graph.assign_feature_masks(policy);
        }
            // NO mutation roll on the immigrant path: a brand-new random topology is
            // already maximal exploration; the mutation roll belongs to crossover only.
        self.finalize_child(graph, clock, child_idx, "random", None, &[], false)
    }

        /// Shared child finalization: fresh weight seed from `run_seed + clock`, lineage
        /// string, `NetState` at step 0, and built `Network` + `Optimizer` ready to train.
    fn finalize_child(
        &self,
        mut topo: Topology,
        clock: usize,
        child_idx: usize,
        origin: &str,
        cx_report: Option<CxReport>,
        parents: &[String],
        mutated: bool,
    ) -> Result<RaceChild> {
        let new_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx) as usize;
        topo.options.topology_seed = new_seed;
            // `swapped=3/6` sits BEFORE `:parents=`, so every existing reader keeps
            // working: `lineage_note` splits on `:parents=`, `origin_cell` takes the
            // first `:`-segment as the kind, and `created_from.rsplit(",")` for a
            // parent list still sees only hashes. Old runs' strings simply lack it.
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
        let _net_device = self.config.device();
        let net = Network::build(&topo, self.config.device())?;
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

/// How many clocks sit inside `frozen_spans` — the crown's act-and-measure passes.
/// Shared with the stop-time table's `act` column, so log and ledger cannot disagree.
pub(crate) fn crown_act_count(state: &NetState) -> usize {
    state.frozen_clocks()
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

    /// One parent with a split, one without: the masked parent's rows survive, so a
    /// split cannot fade out of a lineage just because its partner was unmasked — and
    /// the child's LABEL is the masked parent's policy, never the maskless one's `None`.
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

        // Neither parent carries one: nothing to inherit, so the caller falls back to
        // the run's policy (or leaves the child unmasked).
        let (rows, policy) = mix_feature_masks(None, None, None, None, CrossoverOp::Uniform, &mut rng(1));
        assert!(rows.is_empty());
        assert_eq!(policy, None);
    }

    /// Both parents masked: the rows are RE-combined, so the honest label is
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
            mix_feature_masks(Some(a), Some(b), None, None, CrossoverOp::Uniform, &mut rng(3)).0,
            "the same seed must reproduce the same child"
        );
    }

    /// One-point: ONE cut across the flattened `(port × feature)` genome — A's bits
    /// before it, B's after, both sides non-empty. A cut can land inside a row, which is
    /// exactly what row-level blending could not do.
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
        let (mixed, _) = mix_feature_masks(Some(a), Some(b), None, None, CrossoverOp::Uniform, &mut rng(11));
        assert_eq!(mixed.len(), 4, "one row per port the donors can cover");
        for (i, r) in mixed.iter().enumerate() {
            assert_ne!(r, &row([0, 0]), "port {i} fell back to an empty row");
        }
    }
}
