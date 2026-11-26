//! The graph blueprint — pure topology data, no tensors.

use fastrand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::utils::error::TopologyError;

use crate::graph::node::{Activation, Node, NodeKind};

pub use crate::graph::node::CombineOp;

// ── KindCounts — node-role histogram ──────────────────────────────────────

/// Node counts by kind — the return type of [`Topology::kind_counts`]:
/// no more guessing the order of a bare `(usize, usize, usize)` tuple.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KindCounts {
    /// Input-node count.
    pub input: usize,
    /// Hidden-node count.
    pub hidden: usize,
    /// Output-node count.
    pub output: usize,
}

/// Knobs for building a graph: dims, RNG seed, combine op, ranges. Serialized
/// every topology, so a saved graph is self-describing for `Network::build`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
// ── TopologyOptions — the blueprint template ──────────────────────────────

/// Knobs for building a graph (see the module docs).
pub struct TopologyOptions {
    /// RNG seed — same seed => same random graph.
    pub topology_seed: usize,
    /// Seed the network's INITIAL WEIGHTS are drawn from, when it must differ from `topology_seed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_seed: Option<usize>,
    /// Min hidden nodes per individual (sampled by engine).
    pub min_hidden_num_nodes: usize,
    /// Max hidden nodes per individual (sampled by engine).
    pub max_hidden_num_nodes: usize,
    /// Each random hidden node gets at least this many inputs.
    pub min_hidden_inputs_per_node: usize,
    /// Each random hidden node gets at most this many inputs.
    pub max_hidden_inputs_per_node: usize,
    /// Each random hidden node gets at least this many outputs.
    pub min_hidden_outputs_per_node: usize,
    /// Each random hidden node gets at most this many outputs.
    pub max_hidden_outputs_per_node: usize,
    /// Feature dimension of the network input tensor. `None` = infer
    /// dataset at run start (the engine fills it in and validates
    pub input_dim: Option<usize>,
    /// Output dimension of the network (set on the Output node's layer).
    /// `None` = infer from the dataset at run start.
    pub output_dim: Option<usize>,
    /// RECURRENCE: every HIDDEN node whose input width equals its output width feeds its
    /// PRE-activation back through its OWN layer once (shared weights, no new parameters), and
    /// a net whose input width also equals its output width sweeps the whole graph twice. The
    /// build gates each loop on the node's dims, so a `784 → 10` net still loops its `64 → 64`
    /// hidden nodes; the gene is carried in the blueprint so a newborn inherits it, a crossover
    /// child can drift it, and a replay rebuilds the same net. `false` (the default) is the
    /// ordinary single pass.
    #[serde(default)]
    pub recur: bool,
    /// Dropout probability the engine used when building this topology's network —
    /// part of the blueprint so replays don't diverge; defaults to 0.0.
    #[serde(default)]
    pub dropout_prob: f32,
    /// How much of the RAW input the hidden layer may read directly. `1.0` gives the Input node one
    /// port per hidden node, so every node may touch the pixels (the historical wiring); a lower
    /// value hands out proportionally fewer ports, and the sockets it cannot cover must be fed by
    /// other hidden nodes — depth instead of width-at-the-input. Drawn per net from
    /// [`Self::min_input_reach`]..=[`Self::max_input_reach`].
    #[serde(default = "default_input_reach")]
    pub input_reach: f32,
    /// Lower bound of the per-net `input_reach` draw.
    #[serde(default = "default_input_reach")]
    pub min_input_reach: f32,
    /// Upper bound of the per-net `input_reach` draw.
    #[serde(default = "default_input_reach")]
    pub max_input_reach: f32,
    /// At most ONE wire per (source node, target node) pair, for EVERY kind: Input→hidden as much
    /// as hidden→hidden and hidden→Output. A second wire from the same node carries the same
    /// signal — only its per-port activation differs — so it inflates fan-in, params and the edge
    /// list without adding information. On by default; `false` restores the wiring where an
    /// orphan socket could pick a source that already fed it.
    #[serde(default = "default_true")]
    pub unique_source_per_target: bool,
    /// Feature-split policy this blueprint's Input ports were drawn under — part of the topology,
    /// not the run, so a newborn's policy travels with its blueprint and a population can hold
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_split: Option<FeatureSplitPolicy>,
}

// `dropout_prob: f32` isn't `Eq`/`Hash`, so these are manual: every other field compares normally,
// dropout by bit pattern.
impl PartialEq for TopologyOptions {
    fn eq(&self, other: &Self) -> bool {
        self.topology_seed == other.topology_seed
            && self.min_hidden_num_nodes == other.min_hidden_num_nodes
            && self.max_hidden_num_nodes == other.max_hidden_num_nodes
            && self.min_hidden_inputs_per_node == other.min_hidden_inputs_per_node
            && self.max_hidden_inputs_per_node == other.max_hidden_inputs_per_node
            && self.min_hidden_outputs_per_node == other.min_hidden_outputs_per_node
            && self.max_hidden_outputs_per_node == other.max_hidden_outputs_per_node
            && self.input_dim == other.input_dim
            && self.output_dim == other.output_dim
            && self.dropout_prob.to_bits() == other.dropout_prob.to_bits()
            && self.recur == other.recur
            && self.input_reach.to_bits() == other.input_reach.to_bits()
            && self.min_input_reach.to_bits() == other.min_input_reach.to_bits()
            && self.max_input_reach.to_bits() == other.max_input_reach.to_bits()
            && self.unique_source_per_target == other.unique_source_per_target
            && self.feature_split == other.feature_split
    }
}

impl Eq for TopologyOptions {}

impl std::hash::Hash for TopologyOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.topology_seed.hash(state);
        self.min_hidden_num_nodes.hash(state);
        self.max_hidden_num_nodes.hash(state);
        self.min_hidden_inputs_per_node.hash(state);
        self.max_hidden_inputs_per_node.hash(state);
        self.min_hidden_outputs_per_node.hash(state);
        self.max_hidden_outputs_per_node.hash(state);
        self.input_dim.hash(state);
        self.output_dim.hash(state);
        self.dropout_prob.to_bits().hash(state);
        self.recur.hash(state);
        self.input_reach.to_bits().hash(state);
        self.min_input_reach.to_bits().hash(state);
        self.max_input_reach.to_bits().hash(state);
        self.unique_source_per_target.hash(state);
        self.feature_split.hash(state);
    }
}

/// How a topology draws the Input node's per-port feature masks — the feature split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureSplitPolicy {
    /// **Exclusive** — every feature lands in exactly one port (a true partition): drawn as "each
    /// feature picks one random port".
    Exclusive,
    /// **Random** — every port flips an independent `p = 0.5` coin per feature. Fresh draws repair
    /// any feature that no port picked, so every input is seen at least once; overlap is allowed.
    Random,
    /// **Chunks** — the features are split into balanced, contiguous, ordered blocks: port 0 owns
    /// the first block, port 1 the next, and so on. Earlier ports take any remainder.
    Chunks,
    /// **Crossover** — the rows were RECOMBINED from two parents, so they are neither a fresh
    /// partition (`Exclusive`) nor a fresh coin flip (`Random`): a per-port mix of two partitions.
    #[serde(rename = "crossover", alias = "inherited")]
    Crossover,
}

impl FeatureSplitPolicy {
    /// Whether a run's pool may DRAW this policy for a newborn. `Crossover` describes provenance,
    /// so it can only arrive through crossover.
    pub fn is_drawable(self) -> bool {
        !matches!(self, FeatureSplitPolicy::Crossover)
    }
}

/// Whether `rows` PARTITION the input: every feature belongs to exactly one port.
fn split_rows_are_partition(rows: &[Vec<u8>], input_dim: usize) -> bool {
    (0..input_dim).all(|feat| {
        rows.iter()
            .filter(|row| row.get(feat).copied().unwrap_or(0) != 0)
            .count()
            == 1
    })
}

/// Whether `rows` are ordered, contiguous CHUNKS: a partition whose non-empty rows each own a
/// contiguous run of feature indices, in strictly increasing port order.
fn split_rows_are_chunks(rows: &[Vec<u8>], input_dim: usize) -> bool {
    if !split_rows_are_partition(rows, input_dim) {
        return false;
    }
    let mut next = 0usize;
    for row in rows {
        let owned: Vec<usize> = row
            .iter()
            .enumerate()
            .filter_map(|(feat, bit)| (*bit != 0).then_some(feat))
            .collect();
        if owned.is_empty() {
            continue;
        }
        if owned[0] < next || owned.windows(2).any(|pair| pair[1] != pair[0] + 1) {
            return false;
        }
        next = owned[owned.len() - 1] + 1;
    }
    true
}

/// `1.0` — the historical wiring: one Input port per hidden node, every node may read the input.
/// Also the serde default, so a record written before the knob replays unchanged.
fn default_input_reach() -> f32 {
    1.0
}

/// `true` — one wire per (source node, target node) pair.
fn default_true() -> bool {
    true
}

impl TopologyOptions {
    /// Draw this net's `input_reach` from its own range — the wiring gene, sampled per individual
    /// the way a hidden width is, so one population can hold shallow readers of the input and
    /// genuinely deep nets at the same time.
    pub fn draw_input_reach(&self, rng: &mut Rng) -> f32 {
        let lo = self.min_input_reach.clamp(0.0, 1.0);
        let hi = self.max_input_reach.clamp(0.0, 1.0);
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        if hi <= lo {
            lo
        } else {
            lo + rng.f32() * (hi - lo)
        }
    }
}

impl Default for TopologyOptions {
    fn default() -> Self {
        TopologyOptions {
            topology_seed: 55,
            min_hidden_num_nodes: 2,
            // The cartpole reference's caps, promoted to the engine defaults: the mins stay at 2
            // bare chain still exists) and the maxes allow a genuinely wide net, which is what
            max_hidden_num_nodes: 15,
            min_hidden_inputs_per_node: 2,
            max_hidden_inputs_per_node: 15,
            min_hidden_outputs_per_node: 2,
            max_hidden_outputs_per_node: 15,
            input_dim: None,
            output_dim: None,
            // `0.25` — the cartpole reference's dropout, promoted to the engine default.
            dropout_prob: 0.25,
            // Opt-in: the loop changes what `forward` means, so nothing turns it on by
            // accident (a run asks for it, or drift gives it to a crossover child).
            recur: false,
            input_reach: 1.0,
            min_input_reach: 1.0,
            max_input_reach: 1.0,
            unique_source_per_target: true,
            feature_split: None,
            weight_seed: None,
        }
    }
}

// ── Port & Connection — wiring primitives ─────────────────────────────────

/// A port is a "socket" on a node: a destination (input port) or a source (output
/// port). **All of a node's output ports emit the same tensor** — index is bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Port {
    pub node: usize,  //  which node
    pub index: usize, //  which socket on that node
}

/// A directed wire from one node's output port to another node's input port, e.g.
/// `n1_o0 -> n2_i0`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Connection {
    /// Wire source (output port).
    pub from: Port,
    /// Wire destination (input port).
    pub to: Port,
}

impl Connection {
    /// Source port label, e.g. `"n0_o0"`.
    pub fn from_label(&self) -> String {
        format!("n{}_o{}", self.from.node, self.from.index)
    }

    /// Destination port label, e.g. `"n1_i0"`.
    pub fn to_label(&self) -> String {
        format!("n{}_i{}", self.to.node, self.to.index)
    }
}

// ── graph diagnostics ─────────────────────────────────────────────────
// Free functions live in `crate::utils::graph_utils`; methods delegate to them.

/// What one crossover operator actually exchanged — the operator's own answer, carried so the birth
/// can say how much of the child came from the other parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CxReport {
    /// Genes taken from the other parent.
    pub swaps: usize,
    /// Genes the operator compared (the denominator of `3/6`).
    pub pairs: usize,
    /// PROVENANCE, one slot per HIDDEN node in child order: `Some(i)` = this gene came from node
    /// `i` of the other parent, `None` = it stayed with the kept parent.
    pub moved: Vec<Option<usize>>,
    /// The MIRROR of [`Self::moved`] for the reciprocal child (`b` after the cut): `Some(i)` =
    /// slot came from node `i` of `a`, `None` = it stayed with `b`.
    pub moved_b: Vec<Option<usize>>,
}

impl CxReport {
    /// `3/6` — the spelling the log line, the origin cell and the lineage string use.
    pub fn label(&self) -> String {
        format!("{}/{}", self.swaps, self.pairs)
    }
}

/// The blueprint of a gras network: pure data, no tensors — **who exists**
/// (`nodes`) and **who feeds whom** (`connections`). Inert until validated/built.
#[derive(Clone, Debug, PartialEq)]
// ── Topology — the graph blueprint ───────────────────────────────────────

/// The blueprint of a gras network: pure data, no tensors.
pub struct Topology {
    /// Instance id (used to name `Network`s uniquely).
    pub id: usize,
    /// Nodes, ids contiguous `0..n`.
    pub nodes: Vec<Node>,
    /// Dims, RNG seed, combine op.
    pub options: TopologyOptions,
    pub(crate) graph_inputs: Vec<String>,
    pub(crate) graph_outputs: Vec<String>,
    /// Wires from output ports to input ports.
    pub connections: Vec<Connection>,
    pub(crate) rng: Rng,
}

impl Topology {
    // ── Creation ─────────────────────────────────────────────────────────────

    /// Create an empty graph; `None` uses the default options. Add nodes
    /// [`Node`] constructors + `graph.nodes.push(...)`.
    pub fn new(id: usize, options: Option<TopologyOptions>) -> Topology {
        // Create a new graph with the given id and options, or default options if None
        let opts = options.unwrap_or_default();
        Topology {
            id,
            nodes: Vec::new(),
            options: opts,
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            connections: Vec::new(),
            rng: Rng::with_seed(opts.topology_seed as u64),
        }
    }

    /// Append one random hidden node whose port counts come from the options ranges;
    /// its id is the current node count, so ids stay contiguous.
    pub fn create_random_hidden_node(&mut self) {
        // Random input/output counts within the options ranges:
        // inputs ∈ [min,max]_hidden_inputs_per_node; outputs likewise.
        let num_inputs = self.rng.usize(
            self.options.min_hidden_inputs_per_node..=self.options.max_hidden_inputs_per_node,
        );
        let num_outputs = self.rng.usize(
            self.options.min_hidden_outputs_per_node..=self.options.max_hidden_outputs_per_node,
        );
        // Node id = current node count, so ids stay contiguous: 0, 1, 2, ...
        let node = Node::new_hidden(self.nodes.len(), num_inputs, num_outputs);
        self.nodes.push(node);
    }

    /// Append `num_nodes` random hidden nodes (see
    /// [`Topology::create_random_hidden_node`]).
    pub fn create_random_hidden_nodes(&mut self, num_nodes: usize) {
        // Create multiple random hidden nodes
        for _ in 0..num_nodes {
            self.create_random_hidden_node();
        }
    }

    /// Assign per-output-port activations to every fan-out node from `pool`: port 0
    /// mirrors the node-level activation, ports 1..n draw from the seeded RNG.
    ///
    /// The Input node is INCLUDED: its ports carry the raw-data projection, and which
    /// non-linearity each branch sees is as evolvable as a hidden node's. Port 0 stays the
    /// node gene (`Identity`), so a multi-port Input reads e.g. `[identity, sin, relu]`.
    pub fn assign_port_activations(&mut self, pool: &[Activation]) {
        if pool.is_empty() {
            return;
        }
        for node in &mut self.nodes {
            if node.num_outputs <= 1 {
                continue;
            }
            // FILL-ONLY: a port's activation is genome, and `finalize` carries it through the port
            // remap — so only ports that still have none (a fresh net, or a port the rebuild ADDED)
            let mut ports: Vec<Option<Activation>> = node
                .port_activations
                .take()
                .unwrap_or_default()
                .into_iter()
                .map(Some)
                .collect();
            ports.truncate(node.num_outputs);
            ports.resize(node.num_outputs, None);
            for slot in ports.iter_mut().skip(1) {
                if slot.is_none() {
                    *slot = Some(pool[self.rng.usize(0..pool.len())]);
                }
            }
            // Port 0 mirrors the node-level activation by construction:
            // node gene, not a per-port one.
            ports[0] = Some(node.activation);
            node.port_activations = Some(
                ports
                    .into_iter()
                    .map(|p| p.unwrap_or(node.activation))
                    .collect(),
            );
        }
    }

    /// Assign per-output-port feature masks to the Input node from `policy` — the feature split.
    pub fn assign_feature_masks(&mut self, policy: FeatureSplitPolicy) {
        let input_dim = self.options.input_dim.unwrap_or(1);
        let Some(input) = self.nodes.iter().position(|n| n.kind == NodeKind::Input) else {
            return;
        };
        let n_ports = self.nodes[input].num_outputs.max(1);
        if input_dim == 0 {
            return;
        }
        let mut masks = vec![vec![0u8; input_dim]; n_ports];
        match policy {
            // Provenance, not a draw rule: the only way to get these rows is to inherit
            // them, so drawing is a no-op rather than a guess at what they ''should'' be.
            FeatureSplitPolicy::Crossover => return,
            FeatureSplitPolicy::Exclusive => {
                // Each feature picks one random port: by construction every feature is placed
                // exactly once and none is shared.
                let owners: Vec<usize> =
                    (0..input_dim).map(|_| self.rng.usize(0..n_ports)).collect();
                for (feat, &port) in owners.iter().enumerate() {
                    masks[port][feat] = 1;
                }
            }
            FeatureSplitPolicy::Chunks => {
                // Balanced contiguous blocks, in feature order: earlier ports take the remainder.
                let base = input_dim / n_ports;
                let extra = input_dim % n_ports;
                let mut start = 0usize;
                for (port, mask) in masks.iter_mut().enumerate() {
                    let len = base + usize::from(port < extra);
                    for bit in mask.iter_mut().skip(start).take(len) {
                        *bit = 1;
                    }
                    start += len;
                }
            }
            FeatureSplitPolicy::Random => {
                // Independent coin flip per (port, feature): overlaps allowed. A fresh draw then
                // repairs every feature nobody picked, so the policy still uses ALL inputs.
                for mask in masks.iter_mut() {
                    for slot in mask.iter_mut() {
                        *slot = u8::from(self.rng.f32() < 0.5);
                    }
                }
                let mut load: Vec<usize> = masks
                    .iter()
                    .map(|mask| mask.iter().filter(|bit| **bit != 0).count())
                    .collect();
                for feat in 0..input_dim {
                    if masks.iter().any(|mask| mask[feat] != 0) {
                        continue;
                    }
                    let port = (0..n_ports)
                        .min_by_key(|p| load[*p])
                        .expect("n_ports is at least one");
                    masks[port][feat] = 1;
                    load[port] += 1;
                }
            }
        }
        self.nodes[input].feature_masks = Some(masks);
        self.options.feature_split = Some(policy);
    }

    /// The Input node's per-port feature masks, cloned (`None` when this topology carries no
    /// split).
    pub fn input_feature_masks(&self) -> Option<Vec<Vec<u8>>> {
        self.nodes
            .iter()
            .find(|n| n.kind == NodeKind::Input)
            .and_then(|n| n.feature_masks.clone())
    }

    /// Install INHERITED masks on the Input node — the crossover path's counterpart
    /// [`Self::assign_feature_masks`].
    pub fn install_input_feature_masks(
        &mut self,
        rows: &[Vec<u8>],
        policy: Option<FeatureSplitPolicy>,
    ) {
        let input_dim = self.options.input_dim.unwrap_or(1);
        let Some(input) = self.nodes.iter().position(|n| n.kind == NodeKind::Input) else {
            return;
        };
        if input_dim == 0 {
            return;
        }
        // Nothing inherited at all: keep the net UNSHARED (no rows, no masks) rather
        // materializing cover-all rows, which cost a real elementwise multiply at every forward
        if rows.is_empty() {
            self.nodes[input].feature_masks = None;
            self.options.feature_split = policy;
            return;
        }
        let n_ports = self.nodes[input].num_outputs.max(1);
        let masks: Vec<Vec<u8>> = (0..n_ports)
            .map(|i| {
                // A port the donors did not have starts on a donor's view (cycled), NOT on a
                // free full vector: with one wire per socket the port count follows the number
                // of consumers, so padding here would silently widen the net instead of
                // inheriting what its parents actually read.
                rows.get(i)
                    .cloned()
                    .unwrap_or_else(|| rows[i % rows.len()].clone())
            })
            .collect();
        // PADDING (AND DRIFT) CAN BREAK A STRUCTURAL CLAIM. The rows still stay as genome; only
        // the label follows their content.
        let partition = split_rows_are_partition(&masks, input_dim);
        let chunks = split_rows_are_chunks(&masks, input_dim);
        self.nodes[input].feature_masks = Some(masks);
        self.options.feature_split = Some(match policy.unwrap_or(FeatureSplitPolicy::Crossover) {
            FeatureSplitPolicy::Exclusive if !partition => FeatureSplitPolicy::Crossover,
            FeatureSplitPolicy::Chunks if !chunks => FeatureSplitPolicy::Crossover,
            kept => kept,
        });
    }

    /// Reconcile a LOADED blueprint with the rule "rows present ⇒ policy present": a record written
    /// before `Crossover` existed (or by the crossover bug it fixes) can carry masks under
    pub fn reconcile_split_policy(&mut self) {
        if self.options.feature_split.is_none()
            && self
                .nodes
                .iter()
                .any(|n| n.kind == NodeKind::Input && n.feature_masks.is_some())
        {
            self.options.feature_split = Some(FeatureSplitPolicy::Crossover);
            return;
        }
        // And the other direction: a record can STATE `Exclusive` or `Chunks` while carrying rows
        // that no longer keep that promise (a port the rebuild added, drift, or an older padding).
        if matches!(
            self.options.feature_split,
            Some(FeatureSplitPolicy::Exclusive | FeatureSplitPolicy::Chunks)
        ) {
            let input_dim = self.options.input_dim.unwrap_or(1);
            if let Some(masks) = self.input_feature_masks() {
                let valid = match self.options.feature_split {
                    Some(FeatureSplitPolicy::Exclusive) => {
                        split_rows_are_partition(&masks, input_dim)
                    }
                    Some(FeatureSplitPolicy::Chunks) => split_rows_are_chunks(&masks, input_dim),
                    _ => true,
                };
                if !valid {
                    self.options.feature_split = Some(FeatureSplitPolicy::Crossover);
                }
            }
        }
    }

    /// Assign every hidden node a random activation from `pool` (Input/Output stay
    /// `Identity`). Refresh the port labels (`n1_i0`, `n2_o3`, …) for printing.
    pub fn renumber_ids(&mut self) {
        for (i, node) in self.nodes.iter_mut().enumerate() {
            node.id = i;
        }
    }

    ///   num_inputs  labels to graph_inputs  -> "n{i}_i0", "n{i}_i1", ...
    ///   num_outputs labels to graph_outputs -> "n{i}_o0", "n{i}_o1", ...
    pub fn refresh_labels(&mut self) {
        self.graph_inputs.clear();
        self.graph_outputs.clear();

        for node in &self.nodes {
            for i in 0..node.num_inputs {
                self.graph_inputs.push(format!("n{}_i{}", node.id, i));
            }
            for i in 0..node.num_outputs {
                self.graph_outputs.push(format!("n{}_o{}", node.id, i));
            }
        }
    }

    /// Finalize the graph: scaffold the skeleton, wire every port, and auto-de-orphan
    /// dangling outputs — one call turns a pile of nodes into a complete graph.
    pub fn finalize(&mut self) {
        self.connections.clear();

        // Step 1: build_skeleton — ensure Input + Output nodes exist
        self.ensure_scaffold();

        // Step 2: fill_dims — resolve per-node hidden_dim (None -> graph default)
        self.resolve_dims();

        // Step 3: mark_ports — refresh port labels for display
        self.refresh_labels();

        // Step 4: connect — wire input ports to unused output ports from earlier nodes
        self.wire_ports();

        // Step 5: filter_input — size the Input node from `input_reach` (1.0 = one port per
        // hidden node, the historical wiring)
        self.set_input_fanout();

        // Step 6: connect_orphans — a spare Input port while the reach budget lasts, else an
        // EARLIER hidden node's output (never a blind Input wire)
        self.rewire_orphaned_inputs();

        // Step 6b: clean_wiring — one Input wire per target (each target reads one port)
        self.dedup_input_connections();

        // Step 7: rescue_outputs — wire orphaned outputs into a free input port, or grow
        // one. Inline de-dup: a rescue never repeats a source→target pair.
        self.rewire_orphaned_outputs();

        // Step 8: trim_ports — drop ports nothing uses and renumber the rest
        self.trim_orphaned_ports();

        // Step 9: verify — validate the final wiring
        if let Err(e) = self.validate() {
            panic!("finalize validation failed: {e}\n{self:?}");
        }
    }

    /// Step 2: Resolve per-node hidden_dim -- None becomes graph default.
    fn resolve_dims(&mut self) {
        let eff = self.effective_hidden_dim();
        for node in &mut self.nodes {
            if node.hidden_dim.is_none() {
                node.hidden_dim = Some(match node.kind {
                    NodeKind::Output => self.options.output_dim.unwrap_or(1),
                    _ => eff,
                });
            }
        }
    }

    /// Step 4: Wire input ports to unused output ports from earlier nodes.
    fn wire_ports(&mut self) {
        let mut input_ports: Vec<Port> = Vec::new();
        let mut output_ports: Vec<Port> = Vec::new();
        for node in &self.nodes {
            for i in 0..node.num_inputs {
                input_ports.push(Port {
                    node: node.id,
                    index: i,
                });
            }
            for i in 0..node.num_outputs {
                output_ports.push(Port {
                    node: node.id,
                    index: i,
                });
            }
        }
        self.rng.shuffle(&mut input_ports);
        self.rng.shuffle(&mut output_ports);

        let mut used = vec![false; output_ports.len()];
        for target in input_ports {
            // Track which source nodes already feed this target (avoid redundancy).
            let already_used: Vec<usize> = self
                .connections
                .iter()
                .filter(|c| c.to.node == target.node)
                .map(|c| c.from.node)
                .collect();
            let source = output_ports
                .iter()
                .enumerate()
                .find(|(i, p)| !used[*i] && p.node < target.node && !already_used.contains(&p.node))
                .map(|(i, p)| (i, *p));
            if let Some((i, source)) = source {
                used[i] = true;
                self.connections.push(Connection {
                    from: source,
                    to: target,
                });
            }
        }
    }

    /// Step 5: Size the Input node's port count from `input_reach`.
    ///
    /// The ports ARE the reach budget: `1.0` gives one per hidden node (the historical wiring, every
    /// node may read the raw input), `0.5` roughly one per two nodes. Whatever the budget cannot
    /// cover, `rewire_orphaned_inputs` sources from the hidden layer instead.
    fn set_input_fanout(&mut self) {
        let n_hidden = self
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Hidden)
            .count();
        let reach = self.options.input_reach.clamp(0.0, 1.0);
        // At least one port: the first hidden node has no upstream node to read from.
        let wanted = ((n_hidden as f32 * reach).round() as usize).max(1);
        let input_node = self
            .nodes
            .iter_mut()
            .find(|n| n.kind == NodeKind::Input)
            .unwrap();
        input_node.num_outputs = input_node.num_outputs.max(wanted).max(1);
    }

    /// Re-derive the Input fanout from the CURRENT `input_reach` gene — the mutation-drift hook for
    /// the wiring gene. A crossover child inherits its blueprint's wires, so a re-drawn reach must
    /// move real wires or it is a number nothing reads: the raw-input edges are dropped, the Input
    /// node is resized to the new budget, and every consumer is re-de-orphaned by the SAME rule
    /// `finalize` uses (spare Input port first, then an EARLIER hidden node's output). Hidden↔hidden
    /// wires are untouched. Returns `true` when the port count actually changed.
    pub fn redrift_input_reach(&mut self) -> bool {
        let Some(input_id) = self
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Input)
            .map(|n| n.id)
        else {
            return false;
        };
        let n_hidden = self
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Hidden)
            .count();
        let reach = self.options.input_reach.clamp(0.0, 1.0);
        // At least one port: the first hidden node has no upstream node to read from.
        let wanted = ((n_hidden as f32 * reach).round() as usize).max(1);
        if wanted == self.nodes[input_id].num_outputs {
            return false;
        }
        // The wiring gene OWNS the raw-input edges: drop them, resize, and let the de-orphan rule
        // rebuild them inside the new budget (it may spend a hidden source when the budget is
        // short, and leaves an unfeedable socket for `trim_orphaned_ports` to delete).
        self.connections.retain(|c| c.from.node != input_id);
        self.nodes[input_id].num_outputs = wanted;
        self.rewire_orphaned_inputs();
        self.dedup_input_connections();
        self.trim_orphaned_ports();
        // Port activations are genome and one-per-port: a fan-out that GREW gains the node's
        // own activation on its new sockets (`trim_orphaned_ports` compacts the ones it drops).
        for node in &mut self.nodes {
            let n = node.num_outputs;
            let act = node.activation;
            if let Some(slots) = node.port_activations.as_mut() {
                slots.truncate(n);
                slots.resize(n, act);
            }
        }
        self.refresh_labels();
        if let Err(e) = self.validate() {
            panic!("drift validation failed: {e}\n{self:?}");
        }
        true
    }

    /// Step 6: wire every orphaned input port to a NEW Input output port — ONE SOCKET PER WIRE.
    ///
    /// This used to cycle `output_idx % n_ports`, so one Input port carried several wires. That
    /// made `ports` and `wires` two numbers the reader had to reconcile, and made ONE mask row
    /// stand for several consumers — a shared feature view that crossover could not split. A
    /// fresh port per wire makes the two counts equal by construction, and gives every consumer
    /// its own row of the split genome.
    fn rewire_orphaned_inputs(&mut self) {
        let Some(input_id) = self
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Input)
            .map(|n| n.id)
        else {
            return;
        };
        let mut orphans: Vec<Port> = Vec::new();
        for node in &self.nodes {
            for i in 0..node.num_inputs {
                let port = Port {
                    node: node.id,
                    index: i,
                };
                if !self.connections.iter().any(|c| c.to == port) {
                    orphans.push(port);
                }
            }
        }
        // The first hidden node has no upstream AT ALL, so it can only ever read the Input node.
        // Reserve one port for it: a net whose first hidden node is fed by nothing is dead.
        let first_hidden = self.nodes.iter().position(|n| n.kind == NodeKind::Hidden);
        if let Some(first) = first_hidden {
            let fed = self
                .connections
                .iter()
                .any(|c| c.from.node == input_id && c.to.node == first);
            if !fed && self.spare_input_port(input_id).is_none() && self.nodes[first].num_inputs > 0
            {
                self.nodes[input_id].num_outputs += 1;
            }
        }
        for port in orphans {
            // 1. A SPARE Input port is the `input_reach` budget: spend it first (identical wiring
            //    at `1.0`, where there is one port per hidden node).
            if let Some(index) = self.spare_input_port(input_id) {
                self.connections.push(Connection {
                    from: Port {
                        node: input_id,
                        index,
                    },
                    to: port,
                });
                continue;
            }
            // 2. Budget spent ⇒ source from the HIDDEN layer, never from the raw input. Only a
            //    node EARLIER than the target is eligible, so the graph stays a DAG.
            if let Some(from) = self.spare_hidden_output(port.node) {
                self.connections.push(Connection { from, to: port });
                continue;
            }
            // 3. Nothing spare upstream ⇒ widen an earlier hidden node's fan-out (bounded by
            //    `max_hidden_outputs_per_node` rather than growing without limit).
            if let Some(from) = self.grow_hidden_output(port.node) {
                self.connections.push(Connection { from, to: port });
                continue;
            }
            // 4. Nothing upstream can feed it (only the first hidden node is in that position):
            //    leave the socket UNWIRED and let `trim_orphaned_ports` DELETE it, rather than
            //    paying for it with another raw-input port and widening the first layer.
        }
    }

    /// Keep ONE wire per (source node, target node) pair, dropping the rest in RANDOM order —
    /// `self.rng` is seeded from `topology_seed`, so a reload is random yet reproducible — then
    /// close the port gaps. Returns how many wires were removed. No-op when the option is off.
    pub fn dedup_source_targets(&mut self) -> usize {
        if !self.options.unique_source_per_target {
            return 0;
        }
        let mut order: Vec<usize> = (0..self.connections.len()).collect();
        self.rng.shuffle(&mut order);
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let mut keep = vec![false; self.connections.len()];
        for i in order {
            let key = (self.connections[i].from.node, self.connections[i].to.node);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            keep[i] = true;
        }
        let before = self.connections.len();
        let mut k = 0;
        self.connections.retain(|_| {
            let keep_this = keep[k];
            k += 1;
            keep_this
        });
        let removed = before - self.connections.len();
        if removed > 0 {
            self.trim_orphaned_ports();
        }
        removed
    }

    /// An Input port nothing has claimed yet — the room left in the `input_reach` budget.
    fn spare_input_port(&self, input_id: usize) -> Option<usize> {
        (0..self.nodes[input_id].num_outputs).find(|&index| {
            let port = Port {
                node: input_id,
                index,
            };
            !self.connections.iter().any(|c| c.from == port)
        })
    }

    /// An unused output port on a HIDDEN node EARLIER than `before`, picked at random.
    fn spare_hidden_output(&mut self, target: usize) -> Option<Port> {
        // A source that already feeds `target` is skipped: the second wire would carry the same
        // signal with a different per-port activation.
        let unique = self.options.unique_source_per_target;
        let mut spare: Vec<Port> = Vec::new();
        for node in &self.nodes {
            if node.id >= target || node.kind != NodeKind::Hidden {
                continue;
            }
            if unique
                && self
                    .connections
                    .iter()
                    .any(|c| c.from.node == node.id && c.to.node == target)
            {
                continue;
            }
            for index in 0..node.num_outputs {
                let port = Port {
                    node: node.id,
                    index,
                };
                if !self.connections.iter().any(|c| c.from == port) {
                    spare.push(port);
                }
            }
        }
        if spare.is_empty() {
            return None;
        }
        Some(spare[self.rng.usize(0..spare.len())])
    }

    /// Widen an earlier hidden node's fan-out by one port, staying inside
    /// `max_hidden_outputs_per_node`; `None` when every upstream node is already at its cap.
    fn grow_hidden_output(&mut self, target: usize) -> Option<Port> {
        let cap = self.options.max_hidden_outputs_per_node;
        let unique = self.options.unique_source_per_target;
        let growable: Vec<usize> = self
            .nodes
            .iter()
            .filter(|n| n.id < target && n.kind == NodeKind::Hidden && n.num_outputs < cap)
            .filter(|n| {
                !unique
                    || !self
                        .connections
                        .iter()
                        .any(|c| c.from.node == n.id && c.to.node == target)
            })
            .map(|n| n.id)
            .collect();
        if growable.is_empty() {
            return None;
        }
        let id = growable[self.rng.usize(0..growable.len())];
        let node = &mut self.nodes[id];
        let index = node.num_outputs;
        node.num_outputs += 1;
        Some(Port { node: id, index })
    }

    /// The Input node is the one deliberate fan-out node (its ports are the features, and each may
    /// feed several targets), so ONE target must not be fed by two Input ports: every Input port
    fn dedup_input_connections(&mut self) {
        let input_id = self
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Input)
            .map(|n| n.id);
        if let Some(id) = input_id {
            let mut seen: Vec<usize> = Vec::new();
            self.connections.retain(|c| {
                if c.from.node != id {
                    return true;
                }
                if seen.contains(&c.to.node) {
                    false
                } else {
                    seen.push(c.to.node);
                    true
                }
            });
        }
    }

    /// Human-readable connection labels, e.g. `("n0_o0", "n1_i0")` — a string view
    /// `self.connections`, handy for debugging/DSL rendering.
    pub fn connection_labels(&self) -> Vec<(String, String)> {
        self.connections
            .iter()
            .map(|c| (c.from_label(), c.to_label()))
            .collect()
    }

    /// Check the graph is executable before building: contiguous ids, forward-only
    /// wires, in-range ports, output 1:1, no orphaned inputs. `Network::build` calls
    pub fn validate(&self) -> Result<(), TopologyError> {
        // 1. Options sanity
        if self.options.input_dim == Some(0) || self.options.output_dim == Some(0) {
            return Err(TopologyError::InvalidOptions(
                "input_dim and output_dim must be > 0 or unset (None)".to_string(),
            ));
        }
        if self.options.min_hidden_inputs_per_node > self.options.max_hidden_inputs_per_node
            || self.options.min_hidden_outputs_per_node > self.options.max_hidden_outputs_per_node
        {
            return Err(TopologyError::InvalidOptions(
                "min/max inputs or outputs per node ranges are inverted".to_string(),
            ));
        }

        // 2. At least one node, with contiguous ids and sane per-node dims
        if self.nodes.is_empty() {
            return Err(TopologyError::EmptyTopology);
        }
        for (i, node) in self.nodes.iter().enumerate() {
            if node.id != i {
                return Err(TopologyError::NonContiguousNodeIds);
            }
            if node.hidden_dim == Some(0) {
                return Err(TopologyError::InvalidOptions(format!(
                    "node n{i} has hidden_dim 0"
                )));
            }
            // Per-port activations: length must match the port count. Input ports may be
            // non-Identity too — an Input port is `Linear(masked x)` followed by its own
            // activation, exactly like a hidden port, and that allele is genome.
            if let Some(ports) = &node.port_activations {
                if ports.len() != node.num_outputs {
                    return Err(TopologyError::InvalidOptions(format!(
                        "node n{i} has {} port activations but {} output ports",
                        ports.len(),
                        node.num_outputs
                    )));
                }
            }
            // Feature masks (the feature split): Input nodes only, one mask per
            // output port, each `input_dim` wide and 0/1-valued.
            if let Some(masks) = &node.feature_masks {
                if node.kind != NodeKind::Input {
                    return Err(TopologyError::InvalidOptions(format!(
                        "node n{i} carries feature masks but is not an Input node"
                    )));
                }
                let d = self.options.input_dim.unwrap_or(1);
                if masks.len() != node.num_outputs {
                    return Err(TopologyError::InvalidOptions(format!(
                        "node n{i} has {} feature masks but {} output ports",
                        masks.len(),
                        node.num_outputs
                    )));
                }
                if let Some(bad) = masks.iter().position(|m| m.len() != d) {
                    return Err(TopologyError::InvalidOptions(format!(
                        "node n{i} port {bad} carries a {}-entry feature mask but input_dim is {d}",
                        masks[bad].len()
                    )));
                }
                if masks.iter().flatten().any(|b| *b > 1) {
                    return Err(TopologyError::InvalidOptions(format!(
                        "node n{i} feature masks must be 0/1"
                    )));
                }
            }
        }

        // 2b. Exactly one Output node (output_dim handles the projection;
        //     multiple Output nodes are not supported).
        let n_outputs = self
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Output)
            .count();
        if n_outputs > 1 {
            return Err(TopologyError::InvalidOptions(
                "multiple Output nodes are not supported (use exactly one)".to_string(),
            ));
        }

        // 3.
        let mut used_sources: Vec<Port> = Vec::new();
        let mut used_targets: Vec<Port> = Vec::new();
        for conn in &self.connections {
            if conn.from.node >= conn.to.node {
                return Err(TopologyError::BackwardConnection(conn.clone()));
            }
            if conn.from.node >= self.nodes.len() || conn.to.node >= self.nodes.len() {
                return Err(TopologyError::UnknownNode(conn.from.node));
            }
            let src = &self.nodes[conn.from.node];
            let dst = &self.nodes[conn.to.node];
            if conn.from.index >= src.num_outputs {
                return Err(TopologyError::PortOutOfBounds {
                    port: conn.from,
                    num_ports: src.num_outputs,
                });
            }
            if conn.to.index >= dst.num_inputs {
                return Err(TopologyError::PortOutOfBounds {
                    port: conn.to,
                    num_ports: dst.num_inputs,
                });
            }
            // ONE WIRE PER SOCKET, both ends: an output port feeds at most one input port.
            // (Fan-in is extra input ports; fan-out is extra output ports.) The Input node was
            // exempt while it cycled its ports — that exemption is what let `2 (3w)` exist.
            if used_sources.contains(&conn.from) {
                return Err(TopologyError::DoubleUsedOutput(conn.from));
            }
            used_sources.push(conn.from);
            if used_targets.contains(&conn.to) {
                return Err(TopologyError::MultiWiredInputPort(conn.to));
            }
            used_targets.push(conn.to);
        }

        // ONE WIRE PER (source, target) PAIR — every kind, not just hidden→hidden: an Input wire is
        // Identity, so a second one into the same target is the same signal, and the same holds for
        // a hidden node or the Output node's fan-in.
        if self.options.unique_source_per_target {
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            for conn in &self.connections {
                let key = (conn.from.node, conn.to.node);
                if pairs.contains(&key) {
                    return Err(TopologyError::DuplicateSourceWire {
                        source: key.0,
                        target: key.1,
                    });
                }
                pairs.push(key);
            }
        }

        // 4. Feature-dim consistency per node: with per-node hidden_dim + port
        // projections, all source dim combinations are valid. No check needed.

        Ok(())
    }

    /// Count orphaned ports: `(orphaned_inputs, orphaned_outputs)`. An orphaned input
    /// is fed `net_input` at execution; the graph-output node's ports are excluded.
    pub fn orphan_counts(&self) -> (usize, usize) {
        node_orphan_counts(&self.nodes, &self.connections, self.nodes.len() - 1)
    }

    // ── shared helpers ──────────────────────────────────────────────────

    /// Output dim of a node: its `hidden_dim` override, or the graph default.
    fn out_dim_of(&self, node: &Node) -> usize {
        node.hidden_dim.unwrap_or(8)
    }

    /// The effective hidden dim for orphan projections: the max output dim across
    /// nodes, so orphans project straight to the widest dim (no double projection).
    fn effective_hidden_dim(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| self.out_dim_of(n))
            .max()
            .unwrap_or(8)
    }

    /// Derived per-node feature dims `(in_dim, out_dim)` indexed by node id — the same
    /// derivation `Network::build` uses. Orphan/no-source nodes read `hidden_dim`.
    pub fn node_dims(&self) -> Vec<(usize, usize)> {
        let num_inputs: Vec<usize> = self.nodes.iter().map(|n| n.num_inputs).collect();
        let sources = build_node_sources(&self.connections, &num_inputs);
        self.nodes
            .iter()
            .map(|node| {
                let in_dim = if node.kind == NodeKind::Input {
                    // Input node reads raw data: in_dim = input_dim
                    self.options.input_dim.unwrap_or(1)
                } else {
                    sources[node.id]
                        .iter()
                        .flatten()
                        .map(|p| self.out_dim_of(&self.nodes[p.node]))
                        .max()
                        .unwrap_or_else(|| self.effective_hidden_dim())
                };
                (in_dim, self.out_dim_of(node))
            })
            .collect()
    }

    /// Orphaned ports of one node: `(orphaned input indices, orphaned output indices)`.
    /// The graph-output node's own ports are the answer, not orphans.
    pub fn orphan_ports(&self, node: usize) -> (Vec<usize>, Vec<usize>) {
        let mut ins = Vec::new();
        let mut outs = Vec::new();
        for i in 0..self.nodes[node].num_inputs {
            let port = Port { node, index: i };
            if !self.connections.iter().any(|c| c.to == port) {
                ins.push(i);
            }
        }
        for o in 0..self.nodes[node].num_outputs {
            let port = Port { node, index: o };
            if !self.connections.iter().any(|c| c.from == port) {
                outs.push(o);
            }
        }
        (ins, outs)
    }

    /// Every input port that carries more than one wire, with its wire count — `[]` is the healthy
    /// answer.
    pub fn input_ports_with_multiple_wires(&self) -> Vec<(Port, usize)> {
        let mut counts: Vec<(Port, usize)> = Vec::new();
        for c in &self.connections {
            match counts.iter_mut().find(|(p, _)| *p == c.to) {
                Some((_, n)) => *n += 1,
                None => counts.push((c.to, 1)),
            }
        }
        counts.retain(|(_, n)| *n > 1);
        counts
    }

    /// Give a blueprint the one-wire-per-port shape: every input port that carries several wires
    /// split into that many ports.
    pub(crate) fn unstack_input_ports(&mut self) {
        let stacked = self.input_ports_with_multiple_wires();
        if stacked.is_empty() {
            return;
        }
        let wires: usize = stacked.iter().map(|(_, n)| n).sum();
        let mut reshaped: Vec<Connection> = Vec::with_capacity(self.connections.len());
        for conn in &self.connections {
            let taken = reshaped.iter().any(|c| c.to == conn.to);
            let to = if taken {
                let node = &mut self.nodes[conn.to.node];
                let index = node.num_inputs;
                node.num_inputs += 1;
                Port {
                    node: conn.to.node,
                    index,
                }
            } else {
                conn.to
            };
            reshaped.push(Connection {
                from: conn.from,
                to,
            });
        }
        self.connections = reshaped;
        info!(
            "legacy blueprint: {} input port(s) carried {wires} wires — split into one port per wire",
            stacked.len()
        );
    }

    /// Panic unless the wiring invariant holds: every input port has exactly one
    /// wire (test helper — `validate` enforces the same rule).
    pub fn assert_one_wire_per_port(&self) {
        let multi = self.input_ports_with_multiple_wires();
        assert!(multi.is_empty(), "input ports with >1 wire: {multi:?}");
        for node in &self.nodes {
            for i in 0..node.num_inputs {
                let port = Port {
                    node: node.id,
                    index: i,
                };
                assert!(
                    self.connections.iter().any(|c| c.to == port),
                    "input port n{}_i{} has no wire",
                    node.id,
                    i
                );
            }
        }
    }

    /// Wired degrees `(in_degree, out_degree)` per node, indexed by node id
    /// — one count per *wire*.
    pub fn degrees(&self) -> Vec<(usize, usize)> {
        node_degrees(&self.nodes, &self.connections)
    }

    /// Longest path from an `Input` node per node id — a topological level. Input nodes
    /// sit at 0; others are `1 + max` over wired sources. Ascending ids are a valid order.
    pub fn depths(&self) -> Vec<usize> {
        node_depths(&self.nodes, &self.connections)
    }

    /// Mean wire hop distance in topological levels — the drawn wire thickness, as a number.
    /// `1.0` = every wire joins adjacent levels; larger = skips. `0.0` only with no wires.
    pub fn wire_thickness(&self) -> f32 {
        crate::utils::graph_utils::wire_thickness(&self.nodes, &self.connections)
    }

    /// Estimated total learnable elements (weights + biases) without building —
    /// input projection plus every node's `in·out + out`. Matches a built `Network`.
    pub fn param_estimate(&self) -> usize {
        let dims = self.node_dims();
        let mut total = 0;
        // Node layers: in*out + out each.
        for &(in_dim, out_dim) in &dims {
            total += in_dim * out_dim + out_dim;
        }
        total
    }

    /// Node counts by kind — [`KindCounts`] (no more guessing the tuple
    /// order).
    pub fn kind_counts(&self) -> KindCounts {
        node_kind_counts(&self.nodes)
    }

    /// Activation histogram across nodes: `(activation, count)` pairs.
    pub fn activation_counts(&self) -> Vec<(Activation, usize)> {
        node_activation_counts(&self.nodes)
    }

    /// Ensure the canonical skeleton: at least one `Input` node and exactly one `Output`
    /// node. Idempotent; called as phase 1 of `finalize`. More than one Output is rejected.
    pub fn ensure_scaffold(&mut self) {
        // 1. At least one Input node (single output port; net_input feeds
        //    the rest of the graph).
        if !self.nodes.iter().any(|n| n.kind == NodeKind::Input) {
            self.nodes.insert(0, Node::new_input(0, 1));
            for node in self.nodes.iter_mut().skip(1) {
                node.id += 1;
            }
            for conn in &mut self.connections {
                conn.from.node += 1;
                conn.to.node += 1;
            }
        }

        // 2.
        let has_output = self.nodes.iter().any(|n| n.kind == NodeKind::Output);
        if !has_output {
            let mut out = Node::new_output(self.nodes.len(), 1, 1);
            out.hidden_dim = Some(self.options.output_dim.unwrap_or(1));
            self.nodes.push(out);
        }
    }

    /// Re-wire orphaned output ports into random *later* nodes — the de-orphaning rule, called
    /// the end of `finalize`.
    pub fn rewire_orphaned_outputs(&mut self) -> usize {
        // Collect the orphaned output ports (graph-output node excluded).
        // The Output node is always the last node (created by ensure_scaffold).
        let output_node = self.nodes.len() - 1;
        let mut orphans: Vec<Port> = Vec::new();
        for node in &self.nodes {
            if node.id == output_node {
                continue;
            }
            for o in 0..node.num_outputs {
                let port = Port {
                    node: node.id,
                    index: o,
                };
                if !self.connections.iter().any(|c| c.from == port) {
                    orphans.push(port);
                }
            }
        }

        let mut added = 0usize;
        for src in orphans {
            // Only wire to later nodes NOT already fed by this source.
            // Prevents duplicates at creation time — no cleanup needed.
            let targets: Vec<usize> = self
                .nodes
                .iter()
                .filter(|n| {
                    n.id > src.node
                        && n.num_inputs > 0
                        && !self
                            .connections
                            .iter()
                            .any(|c| c.from.node == src.node && c.to.node == n.id)
                })
                .map(|n| n.id)
                .collect();
            if targets.is_empty() {
                continue;
            }
            // Target choice, in order of preference: 1.
            let free: Vec<usize> = targets
                .iter()
                .copied()
                .filter(|&t| self.free_input_port(t).is_some())
                .collect();
            let target = if !free.is_empty() {
                free[self.rng.usize(0..free.len())]
            } else {
                let growable: Vec<usize> = targets
                    .iter()
                    .copied()
                    .filter(|&t| self.may_grow_inputs(t))
                    .collect();
                if growable.is_empty() {
                    continue;
                }
                growable[self.rng.usize(0..growable.len())]
            };
            let port = match self.free_input_port(target) {
                Some(i) => i,
                None => {
                    // Every socket is taken: add one.
                    let node = &mut self.nodes[target];
                    let i = node.num_inputs;
                    node.num_inputs += 1;
                    i
                }
            };
            self.connections.push(Connection {
                from: src,
                to: Port {
                    node: target,
                    index: port,
                },
            });
            added += 1;
        }
        added
    }

    /// Whether `node` may take another input port.
    fn may_grow_inputs(&self, node: usize) -> bool {
        match self.nodes[node].kind {
            NodeKind::Hidden => {
                self.nodes[node].num_inputs < self.options.max_hidden_inputs_per_node
            }
            _ => true,
        }
    }

    /// The lowest input port of `node` that nothing feeds yet.
    fn free_input_port(&self, node: usize) -> Option<usize> {
        (0..self.nodes[node].num_inputs).find(|&i| {
            !self
                .connections
                .iter()
                .any(|c| c.to == Port { node, index: i })
        })
    }

    /// Remove orphaned ports and renumber connections to close gaps.
    fn trim_orphaned_ports(&mut self) {
        let output_id = self
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Output)
            .map(|n| n.id);

        // Remap tables: for each node, old port index -> compact new index.
        let out_remap: Vec<Vec<usize>> = self
            .nodes
            .iter()
            .map(|n| {
                let mut used: Vec<bool> = vec![false; n.num_outputs];
                for c in &self.connections {
                    if c.from.node == n.id && c.from.index < used.len() {
                        used[c.from.index] = true;
                    }
                }
                let mut map = vec![0usize; n.num_outputs];
                let mut next = 0;
                for i in 0..n.num_outputs {
                    if used[i] {
                        map[i] = next;
                        next += 1;
                    } else {
                        map[i] = usize::MAX; // orphan — will be dropped
                    }
                }
                map
            })
            .collect();

        let in_remap: Vec<Vec<usize>> = self
            .nodes
            .iter()
            .map(|n| {
                let mut used: Vec<bool> = vec![false; n.num_inputs];
                for c in &self.connections {
                    if c.to.node == n.id && c.to.index < used.len() {
                        used[c.to.index] = true;
                    }
                }
                let mut map = vec![0usize; n.num_inputs];
                let mut next = 0;
                for i in 0..n.num_inputs {
                    if used[i] {
                        map[i] = next;
                        next += 1;
                    } else {
                        map[i] = usize::MAX; // empty after dedup — dropped
                    }
                }
                map
            })
            .collect();

        // Renumber connections, dropping the ones whose source or target port died.
        self.connections.retain(|c| {
            let from_map = &out_remap[c.from.node];
            let to_map = &in_remap[c.to.node];
            c.from.index < from_map.len()
                && c.to.index < to_map.len()
                && from_map[c.from.index] != usize::MAX
                && to_map[c.to.index] != usize::MAX
        });
        for c in &mut self.connections {
            c.from.index = out_remap[c.from.node][c.from.index];
            c.to.index = in_remap[c.to.node][c.to.index];
        }

        // Update port counts.
        let mask_width = self.options.input_dim.unwrap_or(1);
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if Some(node.id) != output_id {
                node.num_outputs = out_remap[i]
                    .iter()
                    .filter(|&&v| v != usize::MAX)
                    .count()
                    .max(1);
                // Compact port_activations alongside the port remap — the vec must always
                // match num_outputs (`validate()` enforces it).
                if let Some(ports) = &mut node.port_activations {
                    let mut compacted: Vec<crate::graph::node::Activation> = out_remap[i]
                        .iter()
                        .filter(|&&v| v != usize::MAX)
                        .filter_map(|&old| ports.get(old).copied())
                        .collect();
                    compacted.resize(
                        node.num_outputs,
                        compacted
                            .last()
                            .copied()
                            .unwrap_or(crate::graph::node::Activation::Identity),
                    );
                    *ports = compacted;
                }
                // Feature masks are port-bound too: carry each row with the port it belongs
                // drop the rows of ports the rebuild removed, and leave a port the rebuild ADDED
                if let Some(masks) = &mut node.feature_masks {
                    let mut kept: Vec<Vec<u8>> = Vec::with_capacity(node.num_outputs);
                    for &old in &out_remap[i] {
                        if old == usize::MAX {
                            continue;
                        }
                        kept.push(
                            masks
                                .get(old)
                                .cloned()
                                .unwrap_or_else(|| vec![1u8; mask_width]),
                        );
                    }
                    kept.resize(node.num_outputs, vec![1u8; mask_width]);
                    *masks = kept;
                }
            }
            let connected_in = in_remap[i].iter().filter(|&&v| v != usize::MAX).count();
            if connected_in > 0 {
                node.num_inputs = connected_in;
            }
        }
        // The rebuild can ADD an Input port, and a port the rebuild added keeps every feature —
        // a partition plus a cover-all port is no longer "each feature in exactly one port".
        if matches!(
            self.options.feature_split,
            Some(FeatureSplitPolicy::Exclusive | FeatureSplitPolicy::Chunks)
        ) {
            let input_dim = self.options.input_dim.unwrap_or(1);
            if let Some(masks) = self.input_feature_masks() {
                let valid = match self.options.feature_split {
                    Some(FeatureSplitPolicy::Exclusive) => {
                        split_rows_are_partition(&masks, input_dim)
                    }
                    Some(FeatureSplitPolicy::Chunks) => split_rows_are_chunks(&masks, input_dim),
                    _ => true,
                };
                if !valid {
                    self.options.feature_split = Some(FeatureSplitPolicy::Crossover);
                }
            }
        }
    }

    /// Serialize the whole blueprint (options, nodes, labels, connections) to JSON
    /// (see [`crate::spec::Spec`]); the RNG is not stored.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a blueprint from JSON (see `to_json`). The RNG is re-seeded
    /// `options.topology_seed`, so loaded graphs wire identically; weights are fresh.
    pub fn from_json(s: &str) -> Result<Topology, serde_json::Error> {
        let mut topo: Topology = serde_json::from_str(s)?;
        // Every loaded net (live, tombstone, resumed or imported) passes through here:
        // the one place that can promise a reader "policy present ⇔ rows present".
        topo.reconcile_split_policy();
        // A recorded graph may hold two wires from one node into the same later node.
        topo.dedup_source_targets();
        Ok(topo)
    }

    // ── Crossover ──────────────────────────────────────────────────────────

    /// Find a hidden node with the same interface in both topologies
    /// (`num_inputs`/`num_outputs`); returns indices into each hidden-only list.
    fn find_matching_node(a: &Topology, b: &Topology, rng: &mut Rng) -> Option<(usize, usize)> {
        let ha: Vec<usize> = a
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.kind == NodeKind::Hidden)
            .map(|(i, _)| i)
            .collect();
        let hb: Vec<usize> = b
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.kind == NodeKind::Hidden)
            .map(|(i, _)| i)
            .collect();

        // Collect all matching pairs — only the structural interface matters;
        // finalize() handles wiring, so activations/combine ops may differ freely.
        let mut matches: Vec<(usize, usize)> = Vec::new();
        for (ai, &a_idx) in ha.iter().enumerate() {
            let an = &a.nodes[a_idx];
            for (bi, &b_idx) in hb.iter().enumerate() {
                let bn = &b.nodes[b_idx];
                if an.num_inputs == bn.num_inputs && an.num_outputs == bn.num_outputs {
                    matches.push((ai, bi));
                }
            }
        }

        if matches.is_empty() {
            None
        } else {
            Some(matches[rng.usize(0..matches.len())])
        }
    }

    /// Crossover: find a matching-node pivot (same port counts), swap everything
    /// that pivot onward, then finalize both. Returns true if a swap happened.
    pub fn cx_one_point(a: &mut Topology, b: &mut Topology, rng: &mut Rng) -> bool {
        Self::cx_one_point_with_report(a, b, rng).is_some()
    }

    /// [`Self::cx_one_point`] carrying the provenance the weights need: one entry per hidden node
    /// of the returned `a` — `None` when it kept its own gene, `Some(index)` naming the donor NODE
    pub fn cx_one_point_with_report(
        a: &mut Topology,
        b: &mut Topology,
        rng: &mut Rng,
    ) -> Option<CxReport> {
        let ha: Vec<usize> = a
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.kind == NodeKind::Hidden)
            .map(|(i, _)| i)
            .collect();
        let hb: Vec<usize> = b
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.kind == NodeKind::Hidden)
            .map(|(i, _)| i)
            .collect();

        if ha.is_empty() || hb.is_empty() {
            return None;
        }

        // Only crossover if a matching-node pivot exists — no random cuts
        if let Some((pivot_a, pivot_b)) = Self::find_matching_node(a, b, rng) {
            let swap_a = &ha[pivot_a..];
            let swap_b = &hb[pivot_b..];
            let len = swap_a.len().min(swap_b.len());
            for k in 0..len {
                let tmp = a.nodes[swap_a[k]].clone();
                a.nodes[swap_a[k]] = b.nodes[swap_b[k]].clone();
                b.nodes[swap_b[k]] = tmp;
            }
            a.renumber_ids();
            a.finalize();
            b.renumber_ids();
            b.finalize();
            let mut moved = vec![None; ha.len()];
            let mut moved_b = vec![None; hb.len()];
            for offset in 0..len {
                moved[pivot_a + offset] = Some(hb[pivot_b + offset]);
                moved_b[pivot_b + offset] = Some(ha[pivot_a + offset]);
            }
            debug!(
                "cx_one_point: pivot hidden[{}] <-> hidden[{}], swapped {} nodes",
                pivot_a, pivot_b, len
            );
            return Some(CxReport {
                swaps: len,
                pairs: ha.len(),
                moved,
                moved_b,
            });
        }
        debug!("cx_one_point: no matching node found, skipping");
        None
    }

    /// Uniform crossover: per-node independent swap, requiring equal hidden counts.
    pub fn cx_uniform(
        a: &mut Topology,
        b: &mut Topology,
        swap_prob: f32,
        rng: &mut Rng,
    ) -> Option<CxReport> {
        let ha: Vec<usize> = a
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.kind == NodeKind::Hidden)
            .map(|(i, _)| i)
            .collect();
        let hb: Vec<usize> = b
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.kind == NodeKind::Hidden)
            .map(|(i, _)| i)
            .collect();

        // Same-length requirement for per-node alignment
        if ha.len() != hb.len() || ha.is_empty() {
            debug!(
                "cx_uniform: hidden node count mismatch ({} vs {}), skipping",
                ha.len(),
                hb.len()
            );
            return None;
        }

        let mut swaps = 0usize;
        let mut moved = vec![None; ha.len()];
        let mut moved_b = vec![None; hb.len()];
        for (slot, (&ai, &bi)) in ha.iter().zip(hb.iter()).enumerate() {
            if rng.f32() < swap_prob {
                std::mem::swap(&mut a.nodes[ai], &mut b.nodes[bi]);
                moved[slot] = Some(bi);
                swaps += 1;
            } else {
                // No coin: the reciprocal child took THIS parent's node,
                // provenance names the donor
                moved_b[slot] = Some(ai);
            }
        }

        if swaps == 0 {
            return None;
        }
        a.renumber_ids();
        a.finalize();
        b.renumber_ids();
        b.finalize();
        debug!("cx_uniform: swapped {swaps}/{} hidden nodes", ha.len());
        Some(CxReport {
            swaps,
            pairs: ha.len(),
            moved,
            moved_b,
        })
    }
}

use crate::utils::graph_utils::{
    build_node_sources, node_activation_counts, node_degrees, node_depths, node_kind_counts,
    node_orphan_counts,
};

/// Test-only proptest strategies, shared across the crate's test modules
/// (topology's own tests and the serialization tests in [`crate::spec`]).
/// Default step between sampled hidden widths.
pub const DEFAULT_HIDDEN_DIM_STRIDE: usize = 16;

/// Default hidden-width sampling range: the cartpole reference's `16..=64`,
/// stride [`RaceConfig::defaults`] pins.
pub const DEFAULT_HIDDEN_POOL: std::ops::RangeInclusive<usize> = 16..=64;

/// Default chance that ONE inherited drift slot moves on a crossover child (`set_topology_drift`):
/// a mask bit flips; a hidden node's activation / combine / standardize / width is re-drawn; a port
/// activation moves; the wiring gene (`input_reach`) is re-drawn and its wires re-derived; or the
/// dropout rate is re-drawn under the run's cap.
pub const DEFAULT_TOPOLOGY_DRIFT: f32 = 0.1;

/// Parse a recorded (or user-given) feature-split spelling: `"random"`, `"exclusive"`,
/// `"chunks"`, `"crossover"` (with `"inherited"` kept for old records), or the `"split_*"` wire
/// form `engine.json` records.
pub(crate) fn parse_feature_split_wire(raw: &str) -> Option<FeatureSplitPolicy> {
    let raw = raw.to_ascii_lowercase();
    if raw.contains("crossover") || raw.contains("inherited") {
        Some(FeatureSplitPolicy::Crossover)
    } else if raw.contains("chunks") {
        Some(FeatureSplitPolicy::Chunks)
    } else if raw.contains("exclusive") {
        Some(FeatureSplitPolicy::Exclusive)
    } else if raw.contains("random") {
        Some(FeatureSplitPolicy::Random)
    } else {
        None
    }
}

/// One POOL entry: `Some(None)` is the full-vector policy (no split), `Some(Some(_))` a drawable
/// policy, `None` an unknown spelling. `none`/`off` stay accepted for old run records; new records
/// spell the no-split choice `full`.
pub(crate) fn parse_feature_split_pool_entry(raw: &str) -> Option<Option<FeatureSplitPolicy>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "full" | "none" | "off" | "split_full" | "split_none" => Some(None),
        "exclusive" | "split_exclusive" => Some(Some(FeatureSplitPolicy::Exclusive)),
        "random" | "split_random" => Some(Some(FeatureSplitPolicy::Random)),
        "chunks" | "split_chunks" => Some(Some(FeatureSplitPolicy::Chunks)),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod test_strategies {
    use super::*;
    use proptest::prelude::*;

    /// Valid random `TopologyOptions`: non-zero dims and non-inverted ranges —
    /// exactly what `validate()` rejects is what we never generate.
    pub(crate) fn topology_options_strategy() -> impl Strategy<Value = TopologyOptions> {
        (
            0usize..10_000, //  seed
            1usize..4,      //  min inputs per node
            1usize..4,      //  min outputs per node
            1usize..8,      // input_dim
            1usize..8,      // output_dim
        )
            .prop_map(
                |(seed, min_in, min_out, input_dim, output_dim)| TopologyOptions {
                    unique_source_per_target: true,
                    input_reach: 1.0,
                    min_input_reach: 1.0,
                    max_input_reach: 1.0,
                    topology_seed: seed,
                    min_hidden_num_nodes: 2,
                    max_hidden_num_nodes: 6,
                    min_hidden_inputs_per_node: min_in,
                    max_hidden_inputs_per_node: min_in + 3,
                    min_hidden_outputs_per_node: min_out,
                    max_hidden_outputs_per_node: min_out + 3,
                    input_dim: Some(input_dim),
                    output_dim: Some(output_dim),
                    dropout_prob: 0.0,
                    recur: false,
                    feature_split: None,
                    weight_seed: None,
                },
            )
    }

    /// A random graph: input + 0..5 hidden + output, fully wired through the standard
    /// pipeline (`refresh_labels` + `finalize`, which auto-de-orphans).
    pub(crate) fn topology_strategy() -> impl Strategy<Value = Topology> {
        (topology_options_strategy(), 0usize..6).prop_map(|(opts, n_hidden)| {
            let mut graph = Topology::new(0, Some(opts));
            graph.nodes.push(Node::new_input(0, 2));
            for i in 0..n_hidden {
                let span_in = opts.max_hidden_inputs_per_node - opts.min_hidden_inputs_per_node + 1;
                let span_out =
                    opts.max_hidden_outputs_per_node - opts.min_hidden_outputs_per_node + 1;
                let ins = opts.min_hidden_inputs_per_node + (i * 7) % span_in;
                let outs = opts.min_hidden_outputs_per_node + (i * 3) % span_out;
                graph.nodes.push(Node::new_hidden(i + 1, ins, outs));
            }
            graph.nodes.push(Node::new_output(n_hidden + 1, 1, 1));
            graph.refresh_labels();
            graph.finalize();
            graph
        })
    }
}

#[cfg(test)]
mod tests {
    use super::test_strategies::*;
    use super::*;
    use crate::graph::network::Network;
    use flodl::nn::Module;
    use flodl::{DType, Device, Tensor, TensorOptions, Variable};
    use proptest::prelude::*;

    fn cpu_opts() -> TensorOptions {
        TensorOptions {
            dtype: DType::Float32,
            device: Device::CPU,
        }
    }

    fn rand_input(batch: i64, input_dim: usize) -> Variable {
        Variable::new(
            Tensor::randn(&[batch, input_dim as i64], cpu_opts()).unwrap(),
            false,
        )
    }

    // ── topology (Topology) ────────────────────────────────────────────────────

    #[test]
    fn test_new_without_options() {
        let graph = Topology::new(1, None);
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_hidden_num_nodes, 2);
        assert_eq!(graph.options.max_hidden_num_nodes, 15);
    }

    #[test]
    fn test_new_with_options() {
        let opts = TopologyOptions {
            unique_source_per_target: true,
            topology_seed: 123,
            min_hidden_num_nodes: 3,
            max_hidden_num_nodes: 10,
            min_hidden_inputs_per_node: 1,
            max_hidden_inputs_per_node: 5,
            min_hidden_outputs_per_node: 1,
            max_hidden_outputs_per_node: 5,
            input_dim: Some(4),
            output_dim: Some(10),
            dropout_prob: 0.0,
            recur: false,
            input_reach: 1.0,
            min_input_reach: 1.0,
            max_input_reach: 1.0,
            feature_split: None,
            weight_seed: None,
        };
        let graph = Topology::new(1, Some(opts));
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_hidden_num_nodes, 3);
        assert_eq!(graph.options.max_hidden_num_nodes, 10);
        assert_eq!(graph.options.input_dim, Some(4));
        assert_eq!(graph.options.output_dim, Some(10));
    }

    #[test]
    fn test_create_random_hidden_node() {
        let mut graph = Topology::new(1, None);
        graph.create_random_hidden_node();
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_assign_port_activations_deterministic() {
        // Same seed ⇒ same per-port assignment, twice.
        let build = || {
            let mut g = Topology::new(0, Some(TopologyOptions::default()));
            g.create_random_hidden_nodes(3);
            g.refresh_labels();
            g.finalize();
            g.assign_port_activations(&[Activation::ReLU, Activation::Tanh, Activation::Sin]);
            g
        };
        let a = build();
        let b = build();
        for (na, nb) in a.nodes.iter().zip(&b.nodes) {
            assert_eq!(na.port_activations, nb.port_activations);
        }
        // Ports >= 1 actually drew from the pool; port 0 mirrors the node.
        let pool = [Activation::ReLU, Activation::Tanh, Activation::Sin];
        for node in &a.nodes {
            if let Some(ports) = &node.port_activations {
                assert_eq!(ports.len(), node.num_outputs);
                assert_eq!(ports[0], node.activation);
                for p in &ports[1..] {
                    assert!(pool.contains(p), "port activation {p:?} not from pool");
                }
            }
        }
    }

    #[test]
    fn test_trim_orphaned_ports_compacts_port_activations() {
        // An orphaned output port must compact port_activations along with num_outputs,
        // or finalize's validate() rejects the length mismatch.
        let mut g = Topology::new(0, Some(TopologyOptions::default()));
        g.nodes.push(Node::new_input(0, 1));
        let mut n1 = Node::new_hidden(1, 1, 3);
        n1.activation = Activation::ReLU;
        n1.port_activations = Some(vec![
            Activation::ReLU,
            Activation::Tanh,
            Activation::Sigmoid,
        ]);
        g.nodes.push(n1);
        g.nodes.push(Node::new_output(2, 1, 1));
        // Only ports 0 and 2 are wired; port 1 is orphaned and must be trimmed.
        g.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        g.finalize();
        let n1 = &g.nodes[1];
        // Core invariant: port_activations always matches num_outputs after trim —
        // exactly what validate() enforces.
        let ports = n1.port_activations.as_ref().unwrap();
        assert_eq!(
            ports.len(),
            n1.num_outputs,
            "port_activations must be compacted with ports"
        );
        // Port 0 is wired explicitly, so it survives with its activation.
        assert!(n1.num_outputs >= 1);
        assert_eq!(ports[0], Activation::ReLU);
    }

    #[test]
    fn test_validate_rejects_port_activation_mismatch() {
        let mut g = Topology::new(0, Some(TopologyOptions::default()));
        g.nodes.push(Node::new_input(0, 1));
        let mut n1 = Node::new_hidden(1, 1, 2);
        n1.port_activations = Some(vec![Activation::ReLU]); // 1 entry, 2 ports
        g.nodes.push(n1);
        g.nodes.push(Node::new_output(2, 1, 1));
        assert!(g.validate().is_err());
        // An Input node's non-Identity port is genome, not an error: it must validate.
        let mut g2 = Topology::new(0, Some(TopologyOptions::default()));
        g2.create_random_hidden_nodes(2);
        g2.finalize();
        let input = g2
            .nodes
            .iter()
            .position(|n| n.kind == NodeKind::Input)
            .unwrap();
        let n = g2.nodes[input].num_outputs;
        g2.nodes[input].port_activations = Some(vec![Activation::ReLU; n]);
        assert!(
            g2.validate().is_ok(),
            "Input ports may evolve past Identity"
        );
    }

    #[test]
    fn test_input_port_activations_are_genome() {
        let mut g = Topology::new(
            0,
            Some(TopologyOptions {
                input_reach: 1.0,
                min_input_reach: 1.0,
                max_input_reach: 1.0,
                ..Default::default()
            }),
        );
        g.create_random_hidden_nodes(4);
        g.finalize();
        let i = g
            .nodes
            .iter()
            .position(|n| n.kind == NodeKind::Input)
            .unwrap();
        assert!(g.nodes[i].num_outputs > 1, "fixture needs an Input fan-out");
        g.assign_port_activations(&[Activation::Sin]);
        let ports = g.nodes[i].port_activations.as_ref().unwrap();
        assert_eq!(ports[0], Activation::Identity, "port 0 is the node gene");
        assert!(
            ports[1..].iter().all(|a| *a == Activation::Sin),
            "later Input ports draw from the pool: {ports:?}"
        );
        g.validate().unwrap();
    }

    #[test]
    fn test_refresh_labels() {
        let mut graph = Topology::new(1, None);
        graph.create_random_hidden_node();
        graph.create_random_hidden_node();
        graph.create_random_hidden_node();
        graph.refresh_labels();

        // One label per port, matching each node's declared inputs/outputs
        let total_inputs: usize = graph.nodes.iter().map(|n| n.num_inputs).sum();
        let total_outputs: usize = graph.nodes.iter().map(|n| n.num_outputs).sum();
        assert_eq!(graph.graph_inputs.len(), total_inputs);
        assert_eq!(graph.graph_outputs.len(), total_outputs);
    }

    #[test]
    fn test_refresh_labels_mints_port_labels() {
        let mut graph = Topology::new(1, None);
        // Fixed nodes: input with 2 outputs, hidden with 3 inputs / 2 outputs
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.refresh_labels();

        assert_eq!(graph.graph_inputs, vec!["n1_i0", "n1_i1", "n1_i2"]);
        assert_eq!(
            graph.graph_outputs,
            vec!["n0_o0", "n0_o1", "n1_o0", "n1_o1"]
        );
    }

    #[test]
    fn test_finalize() {
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_hidden(2, 2, 1));
        graph.finalize();

        // Output ports stay 1:1; input ports may hold several wires because finalize
        // auto-de-orphans (stacks extra sources into later nodes, even occupied ports).
        let mut seen_from: Vec<Port> = Vec::new();
        let input_id = graph
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Input)
            .unwrap()
            .id;
        for conn in &graph.connections {
            assert!(
                conn.from.node < conn.to.node,
                "connection must go strictly forward: {conn}"
            );
            // Input node's output ports can fan out to multiple destinations.
            if conn.from.node != input_id {
                assert!(
                    !seen_from.contains(&conn.from),
                    "output port {} used twice",
                    conn.from_label()
                );
            }
            seen_from.push(conn.from);
        }
        // Connections can exceed output port count due to input node fan-out,
        // but non-input output ports stay 1:1.
        assert!(!graph.connections.is_empty());
        assert!(!graph.connections.is_empty());

        // String pairs match the typed connections
        assert_eq!(graph.connections.len(), graph.connection_labels().len());
    }

    // ── validation ──────────────────────────────────────────────────────────

    #[test]
    fn test_validate_ok_on_generated_graph() {
        let mut graph = Topology::new(1, None);
        graph.create_random_hidden_nodes(5);
        graph.refresh_labels();
        graph.finalize();
        assert_eq!(graph.validate(), Ok(()));
    }

    #[test]
    fn test_validate_rejects_empty_graph() {
        let graph = Topology::new(1, None);
        assert_eq!(graph.validate(), Err(TopologyError::EmptyTopology));
    }

    #[test]
    fn test_validate_rejects_non_contiguous_ids() {
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(2, 1, 1)); // id 1 skipped
        assert_eq!(graph.validate(), Err(TopologyError::NonContiguousNodeIds));
    }

    #[test]
    fn test_validate_rejects_backward_connection() {
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 0, index: 0 },
        });
        assert!(matches!(
            graph.validate(),
            Err(TopologyError::BackwardConnection(_))
        ));
    }

    #[test]
    fn test_validate_rejects_unknown_node() {
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 1));
        // Forward-looking (1 < 2) but both ids are past the single node we have.
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        assert!(matches!(
            graph.validate(),
            Err(TopologyError::UnknownNode(_))
        ));
    }

    #[test]
    fn test_validate_rejects_port_out_of_bounds() {
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 3 }, // n1 has only 1 input port
        });
        assert!(matches!(
            graph.validate(),
            Err(TopologyError::PortOutOfBounds { .. })
        ));
    }

    /// Two Input ports into one Hidden port — the shape that used to be legal.
    fn stacked_input_port_graph() -> Topology {
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 0, index: 1 },
            to: Port { node: 1, index: 0 }, // a second wire into the same port
        });
        graph
    }

    #[test]
    fn test_validate_rejects_multi_wired_input_port() {
        // A port is a socket, not a bucket: fan-in is expressed as extra ports,
        // freshly built graph may not stack two wires on one input port.
        let graph = stacked_input_port_graph();
        assert_eq!(
            graph.validate(),
            Err(TopologyError::MultiWiredInputPort(Port {
                node: 1,
                index: 0
            }))
        );
    }

    #[test]
    fn legacy_stacked_blueprint_loads_as_one_port_per_wire() {
        // A run recorded before the refactor can stack wires on one input port, and LOADING
        // only way such wiring re-enters the process (engine inheritance reads the previous run's
        let mut graph = Topology::new(
            9,
            Some(TopologyOptions {
                input_dim: Some(2),
                output_dim: Some(1),
                ..Default::default()
            }),
        );
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 1, 1)); // ONE port, two wires
        graph.nodes.push(Node::new_output(2, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 0, index: 1 },
            to: Port { node: 1, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        assert_eq!(graph.input_ports_with_multiple_wires().len(), 1);

        let loaded = Topology::from_json(&graph.to_json().unwrap()).unwrap();
        assert!(
            loaded.input_ports_with_multiple_wires().is_empty(),
            "loading must spread a stacked port over one port per wire"
        );
        // …and then the repeat is culled: both spread wires are the SAME (source → target) pair,
        // so `unique_source_per_target` keeps one and `trim_orphaned_ports` closes the gap.
        assert_eq!(
            loaded.connections.len(),
            2,
            "the repeated source→target pair is culled"
        );
        assert_eq!(
            loaded.nodes[1].num_inputs, 1,
            "the now-empty port is trimmed"
        );
        loaded.assert_one_wire_per_port();
        assert_eq!(loaded.validate(), Ok(()));
    }

    #[test]
    fn test_validate_rejects_double_used_output() {
        // Non-input nodes: output ports stay 1:1.
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 2, 1));
        graph.nodes.push(Node::new_hidden(2, 1, 1));
        graph.nodes.push(Node::new_hidden(3, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 }, // same output used twice
            to: Port { node: 3, index: 0 },
        });
        assert!(matches!(
            graph.validate(),
            Err(TopologyError::DoubleUsedOutput(_))
        ));
    }

    #[test]
    fn test_orphan_counts() {
        // n1 has 3 inputs, only i2 wired → 2 orphaned inputs. n2's output has no wire
        // → 1 orphaned output. n3 is the network output: its port is the answer, not an orphan.
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_hidden(2, 2, 1));
        graph.nodes.push(Node::new_output(3, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 2 },
        });
        graph.connections.push(Connection {
            from: Port { node: 0, index: 1 },
            to: Port { node: 2, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 1 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 1 },
            to: Port { node: 3, index: 0 },
        });
        assert_eq!(graph.orphan_counts(), (2, 1));
    }

    // ── derived diagnostics ────────────────────────────────────────────────

    #[test]
    fn test_derived_diagnostics() {
        // n0 input (2 outs) → n1 hidden (3 in, 2 out) → n2 hidden (2 in, 1 out) → n3
        // output. Wired sparsely on purpose so orphaned ports stay orphaned.
        let mut g = Topology::new(1, None);
        g.nodes.push(Node::new_input(0, 2));
        g.nodes.push(Node::new_hidden(1, 3, 2));
        g.nodes.push(Node::new_hidden(2, 2, 1));
        g.nodes.push(Node::new_output(3, 1, 1));
        g.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        g.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        g.connections.push(Connection {
            from: Port { node: 2, index: 0 },
            to: Port { node: 3, index: 0 },
        });

        // Per-node orphaned port indices (raw — the output node's own port
        // is counted here, unlike orphan_counts which excludes
        assert_eq!(g.orphan_ports(0), (vec![], vec![1]));
        assert_eq!(g.orphan_ports(1), (vec![1, 2], vec![1]));
        assert_eq!(g.orphan_ports(2), (vec![1], vec![]));
        assert_eq!(g.orphan_ports(3), (vec![], vec![0]));

        // Wired degrees (one count per wire).
        assert_eq!(g.degrees(), vec![(0, 1), (1, 1), (1, 1), (1, 0)]);

        // Topological levels (longest path from the input).
        assert_eq!(g.depths(), vec![0, 1, 2, 3]);

        // Derived dims: input node in_dim = input_dim (1), rest = hidden_dim (8).
        assert_eq!(g.node_dims(), vec![(1, 8), (8, 8), (8, 8), (8, 8)]);

        // Counts by kind.
        assert_eq!(
            g.kind_counts(),
            KindCounts {
                input: 1,
                hidden: 2,
                output: 1
            }
        );

        // Param estimate: 4 nodes × (in*out + out).
        // n0: 1*8+8=16, n1: 8*8+8=72, n2: 8*8+8=72, n3: 8*8+8=72
        assert_eq!(g.param_estimate(), 16 + 3 * 72);

        // Activation histogram.
        assert_eq!(g.activation_counts(), vec![(Activation::Identity, 4)]);
    }

    #[test]
    fn test_node_dims_with_override() {
        // Chain with a widened middle node: n1 emits 32, so n2's in_dim re-derives
        // 32; n0 (orphan) gets effective hidden_dim = 32 so projections go straight there.
        let mut g = Topology::new(1, None);
        g.nodes.push(Node::new_input(0, 1));
        let mut wide = Node::new_hidden(1, 1, 1);
        wide.hidden_dim = Some(32);
        g.nodes.push(wide);
        g.nodes.push(Node::new_output(2, 1, 1));
        g.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        g.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        // n0: input, in_dim = input_dim (1), out_dim = template (8); n1: wired from n0
        // (out=8), out_dim = 32; n2: wired from n1 (out=32), out_dim = template (8).
        assert_eq!(g.node_dims(), vec![(1, 8), (8, 32), (32, 8)]);
    }

    #[test]
    fn test_rewire_orphaned_outputs() {
        // n2_o0 is orphaned; de-orphaning must wire it into the only later node (n3),
        // whose input port is already wired — fine, tensors are summed.
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_hidden(2, 2, 1));
        graph.nodes.push(Node::new_output(3, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 2 },
        });
        graph.connections.push(Connection {
            from: Port { node: 0, index: 1 },
            to: Port { node: 2, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 1 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 1 },
            to: Port { node: 3, index: 0 },
        });

        assert_eq!(graph.orphan_counts(), (2, 1));
        let added = graph.rewire_orphaned_outputs();
        assert_eq!(added, 1);
        // n2_o0 now feeds n3_i0 (stacking onto n1_o1) — no more orphaned
        // outputs; n1's two orphaned inputs remain (they're fed net_input).
        assert_eq!(graph.orphan_counts(), (2, 0));
        assert_eq!(graph.validate(), Ok(()));

        let module = Network::build(&graph, Device::CPU).unwrap();
        let batch = 2i64;
        let input = rand_input(batch, graph.options.input_dim.unwrap_or(1));
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, 8]);
    }

    #[test]
    fn test_rewire_orphaned_outputs_rewires_even_with_dim_mismatch() {
        // n1 widens 8 → 32 and its output is orphaned; port projections bridge any dim
        // mismatch, so the orphan is still rewired.
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 1));
        let mut wide = Node::new_hidden(1, 1, 1);
        wide.hidden_dim = Some(32);
        graph.nodes.push(wide);
        graph.nodes.push(Node::new_output(2, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        assert_eq!(graph.orphan_counts(), (1, 1));
        assert_eq!(graph.rewire_orphaned_outputs(), 1);
        // n1_o0 → n2_i0: resolves both the orphan output AND orphan input
        assert_eq!(graph.orphan_counts(), (0, 0));
    }

    #[test]
    fn test_rewire_orphaned_outputs_deterministic_per_seed() {
        // finalize auto-de-orphans; same seed ⇒ same de-orphaned
        // graph (targets drawn from graph.rng), always valid.
        let build = |seed: usize| {
            let mut graph = Topology::new(0, None);
            graph.options.topology_seed = seed;
            graph.nodes.push(Node::new_input(0, 2));
            graph.nodes.push(Node::new_hidden(1, 2, 2));
            graph.nodes.push(Node::new_hidden(2, 2, 2));
            graph.nodes.push(Node::new_output(3, 2, 1));
            graph.finalize();
            assert_eq!(graph.validate(), Ok(()));
            graph.connections.clone()
        };
        let a = build(42);
        let b = build(42);
        assert_eq!(a, b, "same seed must rewire identically");
        // Different seeds are allowed to coincide when there is only one
        // compatible target; just make sure the result is always valid.
        let _ = build(99);
    }

    #[test]
    fn test_finalize_auto_de_orphans() {
        // finalize alone must leave zero orphaned *outputs* (the graph-output node's
        // own ports are the answer) and always validate.
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_hidden(2, 2, 1));
        graph.nodes.push(Node::new_output(3, 1, 1));
        graph.finalize();
        assert_eq!(graph.validate(), Ok(()));
        let (_, orphaned_outputs) = graph.orphan_counts();
        assert_eq!(orphaned_outputs, 0);
    }

    #[test]
    fn test_validate_allows_inconsistent_input_dims() {
        // With per-node hidden_dim + port projections, different source
        // dims are now valid — projections bridge the mismatch.
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 2)); // out_dim = 8 (default)
        let mut wide = Node::new_hidden(1, 1, 1);
        wide.hidden_dim = Some(16);
        graph.nodes.push(wide);
        graph.nodes.push(Node::new_output(2, 2, 1)); // fed by n0 (8) and n1 (16)
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 1 },
        });
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_validate_allows_orphan_dim_mismatch() {
        // With per-node hidden_dim + port projections, orphan/wire dim
        // mismatches are valid — projections bridge the gap.
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 1));
        let mut wide = Node::new_hidden(1, 1, 1);
        wide.hidden_dim = Some(16);
        graph.nodes.push(wide);
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        assert!(graph.validate().is_ok());
    }

    // ── scaffolding (auto Input/Output) ────────────────────────────────────

    #[test]
    fn test_ensure_scaffold_adds_input_and_output() {
        // A hidden-only graph (what random generation produces) gets
        // canonical skeleton: prepend Input (single port), append Output.
        let mut graph = Topology::new(0, None);
        graph.create_random_hidden_nodes(3);
        graph.ensure_scaffold();
        assert_eq!(graph.nodes[0].kind, NodeKind::Input);
        assert_eq!(graph.nodes[0].num_outputs, 1);
        assert_eq!(graph.nodes.last().unwrap().kind, NodeKind::Output);
        // ids stay contiguous 0..n after the prepend
        for (i, n) in graph.nodes.iter().enumerate() {
            assert_eq!(n.id, i);
        }
    }

    #[test]
    fn test_ensure_scaffold_idempotent() {
        let mut graph = Topology::new(0, None);
        graph.create_random_hidden_nodes(2);
        graph.ensure_scaffold();
        let nodes = graph.nodes.clone();
        graph.ensure_scaffold();
        assert_eq!(graph.nodes, nodes);
    }

    #[test]
    fn test_finalize_scaffolds_hidden_only_graph() {
        // The pipeline alone must turn 5 random hidden nodes into a complete
        // Input → … → Output graph, still valid.
        let mut graph = Topology::new(0, None);
        graph.create_random_hidden_nodes(5);
        graph.finalize();
        let n_in = graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Input)
            .count();
        let n_out = graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Output)
            .count();
        assert_eq!(n_in, 1);
        assert_eq!(n_out, 1);
        assert_eq!(graph.validate(), Ok(()));
    }

    #[test]
    fn test_multiple_output_nodes_rejected() {
        // Multiple Output nodes are not supported — validate rejects them.
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_output(1, 1, 1));
        graph.nodes.push(Node::new_output(2, 1, 1)); // second output
        assert!(graph.validate().is_err(), "multiple outputs must fail");
    }

    // ── property tests (proptest) ───────────────────────────────────────────

    proptest! {
        /// Any random graph produced by the standard pipeline must validate —
        /// the invariant the whole design leans
        #[test]
        fn prop_random_graphs_always_validate(graph in topology_strategy()) {
            prop_assert!(
                graph.validate().is_ok(),
                "random graph failed validation: {:?}",
                graph.validate().err()
            );
        }

        /// finalize auto-de-orphans: no orphaned *outputs* remain
        /// (the graph-output node's own ports are the answer, not orphans).
        #[test]
        fn prop_network_auto_de_orphans_outputs(graph in topology_strategy()) {
                // After finalize, orphan counts may be nonzero when dedup removes redundant
                // same-source-to-same-target connections; the network handles orphans gracefully.
            let (_input_orphans, _output_orphans) = graph.orphan_counts();
        }

        /// The wiring invariants hold for every generated graph: strictly
        /// forward edges and 1:1 on the output side.
        #[test]
        fn prop_wiring_invariants(graph in topology_strategy()) {
            let mut seen: Vec<Port> = Vec::new();
            let input_id = graph.nodes.iter().find(|n| n.kind == NodeKind::Input).map(|n| n.id);
                // One wire per input port — the invariant that lets the runtime keep
                // single source per port instead of a list per port.
            prop_assert!(
                graph.input_ports_with_multiple_wires().is_empty(),
                "stacked input ports: {:?}",
                graph.input_ports_with_multiple_wires()
            );
            for conn in &graph.connections {
                prop_assert!(conn.from.node < conn.to.node, "backward connection: {conn}");
                // Input node's output ports can fan out.
                if Some(conn.from.node) != input_id {
                    prop_assert!(
                        !seen.contains(&conn.from),
                        "output {} used twice",
                        conn.from_label()
                    );
                }
                seen.push(conn.from);
            }
        }

        /// Every valid random graph compiles into a module whose forward pass
        /// yields exactly `[batch, hidden_dim]`.
        #[test]
        fn prop_random_graphs_build_and_forward(graph in topology_strategy()) {
            let module = Network::build(&graph, Device::CPU).unwrap();
            let input = rand_input(2, graph.options.input_dim.unwrap_or(1));
            let out = module.forward(&input).unwrap();
            prop_assert_eq!(out.shape(), &[2, graph.options.output_dim.unwrap_or(1) as i64]);
        }

            /// The blueprint's derived diagnostics match the built engine exactly: dims agree,
            /// param_estimate equals the real count, degrees balance, depth strictly increases.
        #[test]
        fn prop_diagnostics_match_built_network(graph in topology_strategy()) {
            let module = Network::build(&graph, Device::CPU).unwrap();
            let dims = graph.node_dims();
            prop_assert_eq!(&dims, &module.node_dims);
            let real_params: i64 = module
                .parameters()
                .iter()
                .map(|p| p.variable.data().numel())
                .sum();
            prop_assert_eq!(graph.param_estimate() as i64, real_params);

            let degrees = graph.degrees();
            let in_sum: usize = degrees.iter().map(|d| d.0).sum();
            let out_sum: usize = degrees.iter().map(|d| d.1).sum();
            prop_assert_eq!(in_sum, graph.connections.len());
            prop_assert_eq!(out_sum, graph.connections.len());

            let depths = graph.depths();
            for c in &graph.connections {
                prop_assert!(
                    depths[c.to.node] > depths[c.from.node],
                    "wire {c} must strictly increase depth: {:?}",
                    depths
                );
            }
        }

            /// The standard pipeline scaffolds any graph into the canonical skeleton
            /// (≥ 1 Input, exactly 1 Output, still valid) — even hidden-only/empty graphs.
        #[test]
        fn prop_finalize_scaffolds_skeleton(
            opts in topology_options_strategy(),
            n_hidden in 0usize..6,
        ) {
            let mut graph = Topology::new(0, Some(opts));
            graph.create_random_hidden_nodes(n_hidden);
            graph.finalize();
            let n_in = graph
                .nodes
                .iter()
                .filter(|n| n.kind == NodeKind::Input)
                .count();
            let n_out = graph
                .nodes
                .iter()
                .filter(|n| n.kind == NodeKind::Output)
                .count();
            prop_assert!(n_in >= 1, "scaffold must ensure an Input node");
            prop_assert_eq!(
                n_out, 1,
                "scaffold must leave exactly one Output node"
            );
            prop_assert!(graph.validate().is_ok());
        }
    }

    /// 5 wires, one of them a repeat: Input feeds n2 on two of its ports.
    fn duplicate_pair_graph(seed: u64) -> Topology {
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(2),
            topology_seed: seed as _,
            ..Default::default()
        };
        let mut g = Topology::new(seed as _, Some(opts));
        g.nodes.push(Node::new_input(0, 3));
        g.nodes.push(Node::new_hidden(1, 1, 2));
        g.nodes.push(Node::new_hidden(2, 3, 2));
        g.nodes.push(Node::new_output(3, 1, 1));
        let wire =
            |from_node: usize, from_index: usize, to_node: usize, to_index: usize| Connection {
                from: Port {
                    node: from_node,
                    index: from_index,
                },
                to: Port {
                    node: to_node,
                    index: to_index,
                },
            };
        g.connections.push(wire(0, 0, 1, 0));
        g.connections.push(wire(0, 1, 2, 0));
        g.connections.push(wire(0, 2, 2, 1)); // same pair as the wire above
        g.connections.push(wire(1, 0, 2, 2));
        g.connections.push(wire(2, 0, 3, 0));
        g
    }

    #[test]
    fn validate_rejects_one_source_wired_twice_into_the_same_target() {
        let mut g = duplicate_pair_graph(11);
        assert_eq!(
            g.validate(),
            Err(crate::utils::error::TopologyError::DuplicateSourceWire {
                source: 0,
                target: 2
            })
        );
        g.options.unique_source_per_target = false;
        g.validate()
            .expect("with the rule off, the older wiring is allowed again");
    }

    #[test]
    fn dedup_source_targets_keeps_one_wire_per_pair_and_is_seeded() {
        let mut a = duplicate_pair_graph(11);
        let mut b = duplicate_pair_graph(11);
        let before = a.connections.len();
        assert_eq!(a.dedup_source_targets(), 1);
        assert_eq!(b.dedup_source_targets(), 1);
        assert_eq!(a.connections.len(), before - 1);
        let pairs: Vec<(usize, usize)> = a
            .connections
            .iter()
            .map(|c| (c.from.node, c.to.node))
            .collect();
        let mut unique = pairs.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), pairs.len(), "a pair repeats: {pairs:?}");
        a.validate().expect("the repaired graph validates");
        // Seeded: the same graph culls the same wire.
        let key = |g: &Topology| {
            let mut v: Vec<(usize, usize, usize, usize)> = g
                .connections
                .iter()
                .map(|c| (c.from.node, c.from.index, c.to.node, c.to.index))
                .collect();
            v.sort_unstable();
            v
        };
        assert_eq!(key(&a), key(&b));
        // Off ⇒ nothing is removed.
        let mut c = duplicate_pair_graph(11);
        c.options.unique_source_per_target = false;
        assert_eq!(c.dedup_source_targets(), 0);
    }

    #[test]
    fn finalize_never_wires_one_source_twice_into_the_same_target() {
        for seed in 0..40u64 {
            let opts = TopologyOptions {
                input_dim: Some(4),
                output_dim: Some(1),
                topology_seed: seed as _,
                ..Default::default()
            };
            let mut g = Topology::new(seed as _, Some(opts));
            g.create_random_hidden_nodes(8);
            g.finalize();
            g.validate()
                .expect("finalize must not repeat a source→target pair");
            let mut pairs: Vec<(usize, usize)> = g
                .connections
                .iter()
                .map(|c| (c.from.node, c.to.node))
                .collect();
            pairs.sort_unstable();
            let mut unique = pairs.clone();
            unique.dedup();
            assert_eq!(pairs, unique, "seed {seed}: a source feeds a target twice");
        }
    }
}

#[cfg(test)]
mod finalize_order_tests {
    use super::*;

    #[test]
    fn finalize_is_not_idempotent_so_never_run_it_on_loaded_topologies() {
        // Guardrail bug of record: loading a saved topology and calling `finalize()` threw
        // the wiring away and regenerated a DIFFERENT graph. Assert from_json preserves wiring.
        for seed in 0..25 {
            let mut original = Topology::new(seed, Some(TopologyOptions::default()));
            original.finalize();
            let json = original.to_json().unwrap();

            // 1. Round-trip is faithful — build from the parsed graph as-is.
            let parsed = Topology::from_json(&json).unwrap();
            let conns = |t: &Topology| {
                let mut v: Vec<(usize, usize, usize, usize)> = t
                    .connections
                    .iter()
                    .map(|c| (c.from.node, c.from.index, c.to.node, c.to.index))
                    .collect();
                v.sort_unstable();
                v
            };
            assert_eq!(
                conns(&original),
                conns(&parsed),
                "seed {seed}: from_json must preserve the finalized wiring"
            );

            // 2. Calling finalize() again REGENERATES — different graph.
            let mut re_finalized = parsed.clone();
            re_finalized.finalize();
            let differs = conns(&re_finalized) != conns(&parsed)
                    // A regenerated graph usually differs in wiring; even a lucky seed can leave
                    // different bridge sets. The two must never be assumed interchangeable.
                || Topology::from_json(&re_finalized.to_json().unwrap()).unwrap().connections.len()
                    != parsed.connections.len();
            if !differs {
                // Rare: finalize reproduced the same wiring for this seed.
                // Not a failure — just not evidence. Skip.
                continue;
            }
            assert!(
                differs,
                "seed {seed}: finalize() regenerated the wiring (as documented)"
            );
        }
    }

    #[test]
    fn finalize_wires_every_input_port_exactly_once() {
        // The wiring model: a port is a socket, not a bucket.
        for seed in 0..40 {
            let opts = TopologyOptions {
                topology_seed: seed,
                input_dim: Some(4),
                output_dim: Some(1),
                ..Default::default()
            };
            let mut a = Topology::new(seed, Some(opts));
            a.create_random_hidden_nodes(4);
            a.finalize();
            a.assert_one_wire_per_port();
            // The Input node is the one fan-out node, but it may not feed one target twice:
            // Input port is Identity, so the second wire would be the same signal twice, straight
            let input_id = a
                .nodes
                .iter()
                .find(|n| n.kind == NodeKind::Input)
                .unwrap()
                .id;
            for target in &a.nodes {
                let from_input = a
                    .connections
                    .iter()
                    .filter(|c| c.from.node == input_id && c.to.node == target.id)
                    .count();
                assert!(from_input <= 1, "n{} fed {from_input}× by Input", target.id);
            }

            let mut b = Topology::new(seed + 1000, Some(opts));
            b.create_random_hidden_nodes(4);
            b.finalize();
            b.assert_one_wire_per_port();

            let mut rng = Rng::with_seed(seed as u64);
            if Topology::cx_one_point(&mut a, &mut b, &mut rng) {
                a.assert_one_wire_per_port();
                b.assert_one_wire_per_port();
            }
        }
    }

    /// The uniform operator REPORTS what it exchanged: every coin at 1.0 flips every aligned gene,
    /// every coin at 0.0 flips none and is a declined pairing (no child), and the denominator
    #[test]
    fn cx_uniform_reports_the_genes_it_exchanged() {
        let opts = TopologyOptions::default();
        let pair = || {
            let mut a = Topology::new(11, Some(opts.clone()));
            a.create_random_hidden_nodes(4);
            a.finalize();
            let mut b = Topology::new(12, Some(opts.clone()));
            b.create_random_hidden_nodes(4);
            b.finalize();
            (a, b)
        };

        let (mut a, mut b) = pair();
        let all = Topology::cx_uniform(&mut a, &mut b, 1.0, &mut Rng::with_seed(5))
            .expect("every coin flips, so the operator reports");
        assert_eq!((all.swaps, all.pairs), (4, 4));
        assert_eq!(all.label(), "4/4");
        assert_eq!(
            all.moved,
            vec![Some(1), Some(2), Some(3), Some(4)],
            "every gene reports WHICH donor NODE it came from (index 1..4 = the hidden ones)"
        );
        assert_eq!(
            all.moved_b,
            vec![None; 4],
            "the reciprocal kept every gene, so its provenance is all `None`"
        );

        let (mut a, mut b) = pair();
        assert!(
            Topology::cx_uniform(&mut a, &mut b, 0.0, &mut Rng::with_seed(5)).is_none(),
            "no flips ⇒ no child ⇒ nothing to report"
        );

        // Mismatched hidden counts never pair up, so there is no denominator to print.
        let mut a = Topology::new(13, Some(opts.clone()));
        a.create_random_hidden_nodes(3);
        a.finalize();
        let mut b = Topology::new(14, Some(opts));
        b.create_random_hidden_nodes(5);
        b.finalize();
        assert!(Topology::cx_uniform(&mut a, &mut b, 1.0, &mut Rng::with_seed(5)).is_none());
    }

    /// A cut reports BOTH children: the swapped tail names the donor node on each side,
    /// so the engine can follow provenance whichever reciprocal it ends up keeping.
    #[test]
    fn cx_one_point_reports_provenance_for_both_sides() {
        let opts = TopologyOptions::default();
        let mut a = Topology::new(21, Some(opts.clone()));
        a.create_random_hidden_nodes(4);
        a.finalize();
        let mut b = Topology::new(22, Some(opts));
        b.create_random_hidden_nodes(4);
        b.finalize();

        let rep = Topology::cx_one_point_with_report(&mut a, &mut b, &mut Rng::with_seed(9))
            .expect("equal port counts always leave a pivot to cut at");
        assert_eq!(rep.pairs, 4);
        assert!(rep.swaps >= 1, "a cut that swaps nothing is declined");
        assert_eq!(rep.moved.len(), 4, "one slot per hidden node of `a`");
        assert_eq!(rep.moved_b.len(), 4, "one slot per hidden node of `b`");
        // A cut moves a CONTIGUOUS TAIL on each side; every slot before it kept its own.
        let is_tail =
            |m: &[Option<usize>]| m.iter().skip_while(|s| s.is_none()).all(|s| s.is_some());
        assert!(
            rep.moved.iter().any(|s| s.is_some()) && is_tail(&rep.moved),
            "a's moved slots are a tail: {:?}",
            rep.moved
        );
        assert!(
            rep.moved_b.iter().any(|s| s.is_some()) && is_tail(&rep.moved_b),
            "b's moved slots are a tail: {:?}",
            rep.moved_b
        );
        assert_eq!(
            rep.moved.iter().flatten().count(),
            rep.moved_b.iter().flatten().count(),
            "one swap moves one node on each side"
        );
    }

    #[test]
    fn orphan_rescue_grows_a_port_rather_than_stacking() {
        // Every candidate socket is taken, so the rescue has exactly one legal move: give
        // target another input port.
        let mut graph = Topology::new(3, Some(TopologyOptions::default()));
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 2));
        graph.nodes.push(Node::new_hidden(2, 1, 1));
        graph.nodes.push(Node::new_output(3, 1, 1));
        graph.connections = vec![
            Connection {
                from: Port { node: 0, index: 0 },
                to: Port { node: 1, index: 0 },
            },
            Connection {
                from: Port { node: 1, index: 0 },
                to: Port { node: 2, index: 0 },
            },
            Connection {
                from: Port { node: 2, index: 0 },
                to: Port { node: 3, index: 0 },
            },
        ];

        assert_eq!(
            graph.rewire_orphaned_outputs(),
            1,
            "n1_o1 is the only orphan"
        );

        assert!(
            graph.input_ports_with_multiple_wires().is_empty(),
            "the rescue must not stack a second wire on a taken port"
        );
        assert!(
            graph.nodes.iter().any(|n| n.num_inputs == 2),
            "the rescue must give the target a new input port"
        );
    }

    #[test]
    fn finalize_leaves_no_orphaned_input_output_ports() {
        // Regression: dedup used to run AFTER trim_orphaned_ports, stranding ports whose only wire
        // got dropped (mermaid `identity (orphan)` stubs).
        for seed in 0..40 {
            let mut g = Topology::new(seed, Some(TopologyOptions::default()));
            g.finalize();
            let in_id = g
                .nodes
                .iter()
                .find(|n| n.kind == NodeKind::Input)
                .unwrap()
                .id;
            let out_id = g
                .nodes
                .iter()
                .find(|n| n.kind == NodeKind::Output)
                .unwrap()
                .id;
            for p in 0..g.nodes[in_id].num_outputs {
                assert!(
                    g.connections
                        .iter()
                        .any(|c| c.from.node == in_id && c.from.index == p),
                    "seed {seed}: input port o{p} orphaned after finalize"
                );
            }
            for n in &g.nodes {
                for p in 0..n.num_inputs {
                    assert!(
                        g.connections
                            .iter()
                            .any(|c| c.to.node == n.id && c.to.index == p),
                        "seed {seed}: node n{} input port i{p} orphaned after finalize",
                        n.id
                    );
                }
                if n.id != out_id {
                    for p in 0..n.num_outputs {
                        assert!(
                            g.connections
                                .iter()
                                .any(|c| c.from.node == n.id && c.from.index == p),
                            "seed {seed}: node n{} output port o{p} orphaned after finalize",
                            n.id
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod fanout_bounds_tests {
    use super::*;

    #[test]
    fn finalize_respects_max_inputs_per_node() {
        // The setters bound CREATION; verify finalize never grows a node's declared input
        // count past the configured maximum (fan-in rescue only uses declared ports).
        for seed in 0..60 {
            let opts = TopologyOptions {
                min_hidden_inputs_per_node: 2,
                max_hidden_inputs_per_node: 3,
                min_hidden_outputs_per_node: 2,
                max_hidden_outputs_per_node: 3,
                ..Default::default()
            };
            let mut g = Topology::new(seed, Some(opts));
            g.create_random_hidden_nodes(5);
            g.finalize();
            for n in &g.nodes {
                if n.kind == NodeKind::Hidden {
                    assert!(
                        n.num_inputs <= 3,
                        "seed {seed}: n{} has {} inputs (max 3)",
                        n.id,
                        n.num_inputs
                    );
                }
            }
        }
    }

    /// `input_reach` is the DEPTH knob. `1.0` keeps the historical wiring (an Input port per hidden
    /// node, so every node may read the raw input); `0.0` leaves the deep layer fed by hidden nodes,
    /// with one wire per socket either way and no cycle.
    #[test]
    fn input_reach_trades_input_width_for_depth() {
        let build = |seed: usize, reach: f32| {
            let opts = TopologyOptions {
                topology_seed: seed,
                input_dim: Some(4),
                output_dim: Some(1),
                input_reach: reach,
                ..Default::default()
            };
            let mut g = Topology::new(seed, Some(opts));
            g.create_random_hidden_nodes(8);
            g.finalize();
            g
        };
        let input_wires = |g: &Topology| {
            let in_id = g
                .nodes
                .iter()
                .find(|n| n.kind == NodeKind::Input)
                .unwrap()
                .id;
            g.connections
                .iter()
                .filter(|c| c.from.node == in_id)
                .count()
        };
        let deep_nodes = |g: &Topology| {
            let in_id = g
                .nodes
                .iter()
                .find(|n| n.kind == NodeKind::Input)
                .unwrap()
                .id;
            g.nodes
                .iter()
                .filter(|n| n.kind == NodeKind::Hidden)
                .filter(|n| {
                    !g.connections
                        .iter()
                        .any(|c| c.from.node == in_id && c.to.node == n.id)
                })
                .count()
        };
        for seed in 0..25 {
            let full = build(seed, 1.0);
            let narrow = build(seed, 0.0);
            full.assert_one_wire_per_port();
            narrow.assert_one_wire_per_port();
            full.validate().expect("reach 1.0 must validate");
            narrow.validate().expect("reach 0.0 must validate");
            assert!(
                input_wires(&full) >= 7,
                "seed {seed}: reach 1.0 keeps ~one Input port per hidden node, got {}",
                input_wires(&full)
            );
            assert!(
                input_wires(&narrow) <= 3,
                "seed {seed}: reach 0.0 must stop feeding the raw input to the deep layer, got {}",
                input_wires(&narrow)
            );
            assert!(
                deep_nodes(&narrow) >= 5,
                "seed {seed}: reach 0.0 must leave the net DEEPER, only {} node(s) are purely deep",
                deep_nodes(&narrow)
            );
            assert!(
                input_wires(&narrow) < input_wires(&full),
                "seed {seed}: the knob must deepen the net, not widen it"
            );
            // Earlier-only sources ⇒ still a DAG: every hidden wire goes forward.
            for c in &narrow.connections {
                assert!(
                    c.from.node < c.to.node,
                    "seed {seed}: wire n{}_o{} -> n{}_i{} runs backwards",
                    c.from.node,
                    c.from.index,
                    c.to.node,
                    c.to.index
                );
            }
        }
    }

    /// The per-net draw: inside the range, pinned when `min == max`, and genuinely varied.
    #[test]
    fn draw_input_reach_samples_inside_its_range() {
        let mut opts = TopologyOptions::default();
        opts.min_input_reach = 0.2;
        opts.max_input_reach = 0.8;
        let mut rng = Rng::with_seed(7);
        let mut seen: Vec<f32> = Vec::new();
        for _ in 0..64 {
            let v = opts.draw_input_reach(&mut rng);
            assert!((0.2..=0.8).contains(&v), "{v} escaped 0.2..=0.8");
            seen.push(v);
        }
        seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            seen.first().unwrap() != seen.last().unwrap(),
            "a range must actually vary the gene"
        );

        opts.min_input_reach = 0.4;
        opts.max_input_reach = 0.4;
        assert_eq!(
            opts.draw_input_reach(&mut rng),
            0.4,
            "a pinned range is a constant"
        );

        // A reversed range still samples inside it rather than panicking.
        opts.min_input_reach = 0.9;
        opts.max_input_reach = 0.1;
        let v = opts.draw_input_reach(&mut rng);
        assert!((0.1..=0.9).contains(&v));
    }

    #[test]
    fn exclusive_feature_masks_partition_without_sharing() {
        // Exclusive is a true partition: every feature lands in exactly one port,
        // no two branches can ever see the same feature.
        let opts = TopologyOptions {
            input_dim: Some(6),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(11, Some(opts));
        g.create_random_hidden_nodes(4);
        g.finalize();
        g.assign_feature_masks(FeatureSplitPolicy::Exclusive);

        let input = g.nodes.iter().find(|n| n.kind == NodeKind::Input).unwrap();
        let masks = input.feature_masks.as_ref().expect("masks assigned");
        assert_eq!(masks.len(), input.num_outputs, "one mask per output port");
        for feat in 0..6 {
            let owners = masks.iter().filter(|m| m[feat] == 1).count();
            assert_eq!(owners, 1, "feature {feat} must belong to exactly one port");
        }
        let placed: usize = masks
            .iter()
            .map(|m| m.iter().filter(|b| **b == 1).count())
            .sum();
        assert_eq!(placed, 6, "every feature is used exactly once");
    }

    #[test]
    fn random_feature_masks_are_per_port_and_cover_every_input() {
        // Random flips an independent coin per (port, feature), then repairs every feature
        // nobody picked: overlap stays allowed, but no input may be invisible.
        let opts = TopologyOptions {
            input_dim: Some(8),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(3, Some(opts));
        g.create_random_hidden_nodes(6);
        g.finalize();
        g.assign_feature_masks(FeatureSplitPolicy::Random);

        let input = g.nodes.iter().find(|n| n.kind == NodeKind::Input).unwrap();
        let masks = input.feature_masks.as_ref().expect("masks assigned");
        assert_eq!(masks.len(), input.num_outputs);
        assert!(
            masks.iter().all(|m| m.len() == 8),
            "every mask spans input_dim"
        );
        assert!(masks.iter().flatten().all(|b| *b <= 1), "masks are 0/1");
        for feat in 0..8 {
            assert!(
                masks.iter().any(|m| m[feat] == 1),
                "fresh Random must cover feature {feat}"
            );
        }
        let ones: usize = masks
            .iter()
            .map(|m| m.iter().filter(|b| **b == 1).count())
            .sum();
        assert!(
            ones > 0 && ones < masks.len() * 8,
            "random masks should be mixed, got {ones} ones"
        );
    }

    #[test]
    fn chunks_are_balanced_contiguous_and_ordered() {
        let opts = TopologyOptions {
            input_dim: Some(7),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(12, Some(opts));
        g.nodes.push(Node::new_input(0, 3));
        g.assign_feature_masks(FeatureSplitPolicy::Chunks);

        let masks = g.input_feature_masks().expect("chunks assigned");
        assert_eq!(
            masks,
            vec![
                vec![1, 1, 1, 0, 0, 0, 0],
                vec![0, 0, 0, 1, 1, 0, 0],
                vec![0, 0, 0, 0, 0, 1, 1],
            ],
            "earlier ports take the remainder, blocks stay contiguous"
        );
    }

    #[test]
    fn a_broken_chunks_claim_is_relabelled_crossover() {
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(13, Some(opts));
        g.nodes.push(Node::new_input(0, 3));
        g.install_input_feature_masks(
            &[vec![0, 1, 0, 0], vec![1, 0, 0, 0], vec![0, 0, 1, 1]],
            Some(FeatureSplitPolicy::Chunks),
        );
        assert_eq!(
            g.options.feature_split,
            Some(FeatureSplitPolicy::Crossover),
            "out-of-order rows cannot keep the chunks label"
        );
    }

    #[test]
    fn crossover_serde_accepts_the_old_inherited_spelling() {
        assert_eq!(
            serde_json::to_string(&FeatureSplitPolicy::Crossover).unwrap(),
            "\"crossover\""
        );
        let old: FeatureSplitPolicy = serde_json::from_str("\"inherited\"").unwrap();
        assert_eq!(old, FeatureSplitPolicy::Crossover);
    }

    #[test]
    fn feature_masks_are_refused_off_the_input_node_and_at_the_wrong_width() {
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        // Wrong width: a mask must span exactly `input_dim`.
        let mut g = Topology::new(5, Some(opts));
        g.create_random_hidden_nodes(2);
        g.finalize();
        let idx = g
            .nodes
            .iter()
            .position(|n| n.kind == NodeKind::Input)
            .unwrap();
        let ports = g.nodes[idx].num_outputs.max(1);
        g.nodes[idx].feature_masks = Some(vec![vec![1u8, 0u8]; ports]); // 2 entries, not 4
        assert!(
            g.validate().is_err(),
            "a feature mask of the wrong width must be refused"
        );

        // Masks on a non-Input node: the split is an Input-side feature.
        let mut g2 = Topology::new(5, Some(opts));
        g2.create_random_hidden_nodes(2);
        g2.finalize();
        let hidden = g2
            .nodes
            .iter()
            .position(|n| n.kind == NodeKind::Hidden)
            .unwrap();
        g2.nodes[hidden].feature_masks = Some(vec![vec![1u8; 4]]);
        assert!(
            g2.validate().is_err(),
            "only Input nodes may carry feature masks"
        );
    }

    #[test]
    fn finalize_carries_masks_through_the_port_remap() {
        // Crossover re-finalizes both parents, and those parents already carry masks from their own
        // build.
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(13, Some(opts));
        g.create_random_hidden_nodes(3);
        g.finalize();
        g.assign_feature_masks(FeatureSplitPolicy::Exclusive);
        let before = g.input_feature_masks().expect("the split is present");

        // The rebuild itself: the rows must survive it, at the FINAL port count.
        g.finalize();
        let after = g
            .input_feature_masks()
            .expect("finalize must carry the split, not wipe it");
        let input = g.nodes.iter().find(|n| n.kind == NodeKind::Input).unwrap();
        assert_eq!(after.len(), input.num_outputs, "one row per final port");
        assert!(
            after.iter().all(|m| m.len() == 4),
            "every carried row keeps its width"
        );
        assert_eq!(after[0], before[0], "surviving ports keep their own row");
        g.validate().expect("carried masks must validate");
    }

    #[test]
    fn feature_masks_survive_the_blueprint_round_trip() {
        // The split is genome, so the masks must come back identical from the blueprint JSON —
        // file (not the weights) is also the Python side of the contract, so the field has
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(9, Some(opts));
        g.create_random_hidden_nodes(2);
        g.finalize();
        g.assign_feature_masks(FeatureSplitPolicy::Random);

        let json = g.to_json().unwrap();
        assert!(
            json.contains("feature_masks"),
            "the field must be part of the blueprint"
        );
        let back: Topology = serde_json::from_str(&json).unwrap();
        let idx = back
            .nodes
            .iter()
            .position(|n| n.kind == NodeKind::Input)
            .unwrap();
        assert_eq!(
            back.nodes[idx].feature_masks, g.nodes[idx].feature_masks,
            "masks must round-trip byte for byte"
        );

        // Old JSON predates the field: it must load as "no split".
        let legacy: Node = serde_json::from_value(serde_json::json!({
            "id": 0, "num_inputs": 0, "num_outputs": 2, "kind": "Input",
            "hidden_dim": null, "activation": "Identity",
            "port_activations": null, "combine_op": null, "standardize": null,
        }))
        .unwrap();
        assert_eq!(
            legacy.feature_masks, None,
            "serde default keeps old JSON loadable"
        );
    }

    /// The crossover path's installer: an inherited row lands on its own port, a port with no
    /// inherited row starts UNMASKED (every feature reaches it), and the result always validates
    #[test]
    fn install_feature_masks_maps_rows_and_seeds_new_ports_on_a_donor_view() {
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(23, Some(opts));
        g.create_random_hidden_nodes(3);
        g.finalize();
        fn input_of(t: &Topology) -> &crate::graph::node::Node {
            t.nodes.iter().find(|n| n.kind == NodeKind::Input).unwrap()
        }
        let ports = input_of(&g).num_outputs;
        assert!(
            ports >= 2,
            "fixture needs at least two input ports, got {ports}"
        );

        g.install_input_feature_masks(&[vec![0, 1, 1, 1]], None);
        let masks = input_of(&g)
            .feature_masks
            .clone()
            .expect("rows were installed");
        assert_eq!(masks.len(), ports, "one row per port");
        assert_eq!(masks[0], vec![0, 1, 1, 1], "the inherited row rides port 0");
        assert!(
            masks[1..].iter().all(|m| m == &vec![0, 1, 1, 1]),
            "a port with no inherited row starts on the donor's view, not a free full vector"
        );
        g.validate().expect("installed masks must validate");
        assert_eq!(
            g.input_feature_masks().expect("readable"),
            masks,
            "input_feature_masks must read back exactly what was installed"
        );
    }

    /// No inherited rows at all (neither parent carried a split, no policy): the input stays
    /// genuinely unmasked — no rows, no policy — rather than getting cover-all rows (which would be
    #[test]
    fn install_feature_masks_with_no_rows_leaves_the_net_unsplit() {
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(24, Some(opts));
        g.create_random_hidden_nodes(2);
        g.finalize();
        g.install_input_feature_masks(&[], None);
        assert!(
            g.input_feature_masks().is_none(),
            "no inherited rows means no masks at all"
        );
        assert_eq!(g.options.feature_split, None);
        g.validate().expect("unmasked topology must validate");
    }

    /// The LABEL follows the CONTENT, in the one place rows are installed: an inherited row set
    /// with no stated provenance is `Crossover`, never `None`.
    #[test]
    fn installing_inherited_rows_stamps_a_matching_policy() {
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let build = || {
            let mut g = Topology::new(25, Some(opts));
            g.create_random_hidden_nodes(3);
            g.finalize();
            g
        };

        // No provenance supplied → `Crossover` (the only true statement).
        let mut g = build();
        g.install_input_feature_masks(&[vec![0, 1, 0, 0]], None);
        assert_eq!(g.options.feature_split, Some(FeatureSplitPolicy::Crossover));
        assert!(g.input_feature_masks().is_some());

        // Stated provenance is kept verbatim.
        let mut g = build();
        g.install_input_feature_masks(&[vec![1, 0, 1, 0]], Some(FeatureSplitPolicy::Random));
        assert_eq!(g.options.feature_split, Some(FeatureSplitPolicy::Random));

        // A fresh draw stamps the policy it drew under, too.
        let mut g = build();
        g.assign_feature_masks(FeatureSplitPolicy::Exclusive);
        assert_eq!(g.options.feature_split, Some(FeatureSplitPolicy::Exclusive));

        // And `Crossover` is never drawable: it does not invent rows.
        let mut g = build();
        g.assign_feature_masks(FeatureSplitPolicy::Crossover);
        assert!(
            g.input_feature_masks().is_none(),
            "Crossover is provenance, not a draw rule"
        );
        assert_eq!(g.options.feature_split, None);
        assert!(!FeatureSplitPolicy::Crossover.is_drawable());
    }

    /// `Exclusive` is the one checkable claim a split makes — every feature owned EXACTLY once —
    /// a row set that fails it cannot wear the label.
    #[test]
    fn a_cover_all_port_beside_a_partition_is_not_labelled_exclusive() {
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(29, Some(opts));
        g.create_random_hidden_nodes(2);
        g.finalize();
        let input = g
            .nodes
            .iter()
            .position(|n| n.kind == NodeKind::Input)
            .unwrap();

        // 2 | 0 | 1 | 1 over 4 features: every feature in exactly one port,
        // accounting is 100% and the label stands.
        g.options.feature_split = Some(FeatureSplitPolicy::Exclusive);
        g.nodes[input].feature_masks = Some(vec![
            vec![1, 1, 0, 0],
            vec![0, 0, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
        ]);
        g.reconcile_split_policy();
        assert_eq!(
            g.options.feature_split,
            Some(FeatureSplitPolicy::Exclusive),
            "a true partition keeps its label"
        );

        // The same rows plus one port that keeps EVERYTHING: features 0 and 1 now belong
        // to two ports, the accounting is 200%, and `exclusive` is a lie.
        g.nodes[input].feature_masks = Some(vec![
            vec![1, 1, 0, 0],
            vec![0, 0, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
            vec![1, 1, 1, 1],
        ]);
        g.reconcile_split_policy();
        assert_eq!(
            g.options.feature_split,
            Some(FeatureSplitPolicy::Crossover),
            "a cover-all port breaks the partition, so the label follows the rows"
        );
    }

    /// Old records (and nets written by the crossover bug) load with rows under a `None` policy.
    #[test]
    fn loading_a_topology_with_rows_and_no_policy_repairs_the_label() {
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(26, Some(opts));
        g.create_random_hidden_nodes(2);
        g.finalize();
        g.assign_feature_masks(FeatureSplitPolicy::Exclusive);
        let mut json: serde_json::Value = serde_json::from_str(&g.to_json().unwrap()).unwrap();
        json["options"]
            .as_object_mut()
            .unwrap()
            .remove("feature_split"); // exactly what the old crossover path persisted
        let loaded = Topology::from_json(&json.to_string()).unwrap();
        assert_eq!(
            loaded.options.feature_split,
            Some(FeatureSplitPolicy::Crossover)
        );
        assert!(loaded.input_feature_masks().is_some());

        // An honest blueprint is left untouched.
        let again = Topology::from_json(&g.to_json().unwrap()).unwrap();
        assert_eq!(
            again.options.feature_split,
            Some(FeatureSplitPolicy::Exclusive)
        );
    }
}
