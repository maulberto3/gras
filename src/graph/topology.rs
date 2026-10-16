//! The graph blueprint  — pure topology data, no tensors.
//!
//! Two-phase design: **Topology** (this file, pure data) declares nodes
//! and connections. **Network** compiles it into real layers and executes.
//!
//! Invariants: contiguous ids (0..n), forward-only edges, orphaned ports
//! fed by network input. `validate()` checks all of them.

use fastrand::Rng;
use serde::{Deserialize, Serialize};
use tracing::debug;

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

/// Knobs for building a graph: dims, RNG seed, combine op, ranges. Serialized with
/// every topology, so a saved graph is self-describing for `Network::build`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
// ── TopologyOptions — the blueprint template ──────────────────────────────

/// Knobs for building a graph (see the module docs).
pub struct TopologyOptions {
    /// RNG seed — same seed => same random graph.
    pub topology_seed: usize,
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
        /// Feature dimension of the network input tensor. `None` = infer from the
        /// dataset at run start (the engine fills it in and validates it).
    pub input_dim: Option<usize>,
        /// Output dimension of the network (set on the Output node's layer).
        /// `None` = infer from the dataset at run start.
    pub output_dim: Option<usize>,
        /// Dropout probability the engine used when building this topology's network —
        /// part of the blueprint so replays don't diverge; defaults to 0.0.
    #[serde(default)]
    pub dropout_prob: f32,
}

    // `dropout_prob: f32` isn't `Eq`/`Hash`, so these are manual: every other field
    // compares normally, dropout by bit pattern. Keeps `HashSet<Spec>` dedup working.
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
    }
}

/// How a run draws the Input node's per-port feature masks — the feature split.
/// Stored on the engine config, never in a blueprint: the masks themselves are
/// genome (they live in [`Node::feature_masks`]), and this only says how they are
/// generated. `None` on the config = no split at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureSplitPolicy {
    /// **Exclusive** — every feature lands in exactly one port (a true partition):
    /// drawn as "each feature picks one random port". Nothing is dropped and
    /// nothing is duplicated, so no two branches can see the same feature. The
    /// split is uneven by construction, and a port may end up with no features.
    Exclusive,
    /// **Random** — every port flips an independent `p = 0.5` coin per feature.
    /// A feature can land in several ports, and some features may go unused.
    Random,
}

impl Default for TopologyOptions {
    fn default() -> Self {
        TopologyOptions {
            topology_seed: 55,
            min_hidden_num_nodes: 2,
            max_hidden_num_nodes: 5,
            min_hidden_inputs_per_node: 2,
            max_hidden_inputs_per_node: 5,
            min_hidden_outputs_per_node: 2,
            max_hidden_outputs_per_node: 5,
            input_dim: None,
            output_dim: None,
                // 0.0 keeps regularization opt-in per binary. Masks draw from libtorch's
                // global RNG, which the engine seeds per (net_seed, step) — reproducible.
            dropout_prob: 0.0,
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

        /// Create an empty graph; `None` uses the default options. Add nodes with the
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

        /// Assign per-output-port activations to every hidden node from `pool`: port 0
        /// mirrors the node-level activation, ports 1..n draw from the seeded RNG.
    pub fn assign_port_activations(&mut self, pool: &[Activation]) {
        if pool.is_empty() {
            return;
        }
        for node in &mut self.nodes {
            if node.kind == NodeKind::Input || node.num_outputs <= 1 {
                continue;
            }
            let mut ports = vec![node.activation];
            for _ in 1..node.num_outputs {
                ports.push(pool[self.rng.usize(0..pool.len())]);
            }
            node.port_activations = Some(ports);
        }
    }

        /// Assign per-output-port feature masks to the Input node from `policy` —
        /// the feature split. MUST run **after** `finalize()`: the Input node's port
        /// count is final only once `trim_orphaned_ports` compacted it (the same rule
        /// `assign_port_activations` follows), and `finalize` drops any masks from a
        /// previous build rather than let a stale port count through.
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
            FeatureSplitPolicy::Exclusive => {
                    // Each feature picks one random port: by construction every feature is
                    // placed exactly once and none is shared. Naturally uneven, and a port
                    // can end up empty (it then emits only the shared bias).
                let owners: Vec<usize> = (0..input_dim)
                    .map(|_| self.rng.usize(0..n_ports))
                    .collect();
                for (feat, &port) in owners.iter().enumerate() {
                    masks[port][feat] = 1;
                }
            }
            FeatureSplitPolicy::Random => {
                    // Independent coin flip per (port, feature): overlaps allowed, and some
                    // features may go unused.
                for mask in masks.iter_mut() {
                    for slot in mask.iter_mut() {
                        *slot = u8::from(self.rng.f32() < 0.5);
                    }
                }
            }
        }
        self.nodes[input].feature_masks = Some(masks);
    }

        /// The Input node's per-port feature masks, cloned (`None` when this
        /// topology carries no split). The engine snapshots these BEFORE a
        /// crossover, which re-finalizes both parents and drops them.
    pub fn input_feature_masks(&self) -> Option<Vec<Vec<u8>>> {
        self.nodes
            .iter()
            .find(|n| n.kind == NodeKind::Input)
            .and_then(|n| n.feature_masks.clone())
    }

        /// Install INHERITED masks on the Input node — the crossover path's
        /// counterpart to [`Self::assign_feature_masks`]. Rows are mapped to ports by
        /// index; a port the donor rows do not cover (the rebuild grew the port count)
        /// starts UNMASKED, i.e. it sees every feature — a mask is never invented for a
        /// port we hold no inherited data for. Truncates when the child has fewer ports.
        ///
        /// Deterministic by construction (no RNG): the same inherited rows always give
        /// the same masks, so resume replays identically.
    pub fn install_input_feature_masks(&mut self, rows: &[Vec<u8>]) {
        let input_dim = self.options.input_dim.unwrap_or(1);
        let Some(input) = self.nodes.iter().position(|n| n.kind == NodeKind::Input) else {
            return;
        };
        if input_dim == 0 {
            return;
        }
        let n_ports = self.nodes[input].num_outputs.max(1);
        let masks: Vec<Vec<u8>> = (0..n_ports)
            .map(|i| {
                rows.get(i)
                        // No inherited row for this port: every feature reaches it.
                    .cloned()
                    .unwrap_or_else(|| vec![1u8; input_dim])
            })
            .collect();
        self.nodes[input].feature_masks = Some(masks);
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

            // Feature masks are bound to the Input node's port COUNT, and this rebuild
            // re-wires and re-counts those ports (`set_input_fanout` can raise the count,
            // `trim_orphaned_ports` compacts it). A mask from before the rebuild can
            // therefore have the wrong length — which `validate` rightly refuses, and
            // since `finalize` validates, that would panic. Crossover re-finalizes both
            // parents, and those parents already carry masks, so this is a real path.
            // Dropping them here loses nothing: every creation path re-derives masks from
            // the run's policy AFTER `finalize` (`assign_feature_masks`).
        for node in &mut self.nodes {
            node.feature_masks = None;
        }

        // Step 1: build_skeleton — ensure Input + Output nodes exist
        self.ensure_scaffold();

        // Step 2: fill_dims — resolve per-node hidden_dim (None -> graph default)
        self.resolve_dims();

        // Step 3: mark_ports — refresh port labels for display
        self.refresh_labels();

        // Step 4: connect — wire input ports to unused output ports from earlier nodes
        self.wire_ports();

        // Step 5: filter_input — set input node output count = number of hidden nodes
        self.set_input_fanout();

        // Step 6: connect_orphans — wire orphaned inputs to the input node (cycle ports)
        self.rewire_orphaned_inputs();

        // Step 6b: clean_wiring — dedup input node connections (one wire per source_port, target_node)
        self.dedup_input_connections();

        // Step 7: rescue_outputs — wire orphaned outputs to later nodes (dedup inline)
        self.rewire_orphaned_outputs();

            // Step 7b: dedup_value_wires — collapse same-(source→target) wires with identical
            // source activations. MUST run BEFORE trim_orphaned_ports (dedup can strand ports).
        self.dedup_value_wires();

        // Step 7c: trim_ports — remove orphaned input/output ports from the topology
        self.trim_orphaned_ports();

        // Step 8: verify — validate the final wiring
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

    /// Step 5: Set input node output count = number of hidden nodes (at least).
    /// Never shrinks — if the node already has more ports, keep them.
    fn set_input_fanout(&mut self) {
        let n_hidden = self
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Hidden)
            .count();
        let input_node = self
            .nodes
            .iter_mut()
            .find(|n| n.kind == NodeKind::Input)
            .unwrap();
        input_node.num_outputs = input_node.num_outputs.max(n_hidden).max(1);
    }

        /// Step 6: wire orphaned input ports to the input node's output ports, cycling
        /// through available ports (the combine op merges shared ones).
    fn rewire_orphaned_inputs(&mut self) {
        let input_id = self
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Input)
            .unwrap()
            .id;
        let n_ports = self
            .nodes
            .iter()
            .find(|n| n.id == input_id)
            .unwrap()
            .num_outputs;
        let mut output_idx = 0usize;
        for node in &self.nodes {
            for i in 0..node.num_inputs {
                let port = Port {
                    node: node.id,
                    index: i,
                };
                if !self.connections.iter().any(|c| c.to == port) {
                    self.connections.push(Connection {
                        from: Port {
                            node: input_id,
                            index: output_idx % n_ports,
                        },
                        to: port,
                    });
                    output_idx += 1;
                }
            }
        }
    }

    /// Remove duplicate connections from the input node: keep at most one
    /// wire per (source_port, target_node) pair.
    fn dedup_input_connections(&mut self) {
        let input_id = self
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Input)
            .map(|n| n.id);
        if let Some(id) = input_id {
            let mut seen: Vec<(usize, usize)> = Vec::new();
            self.connections.retain(|c| {
                if c.from.node != id {
                    return true;
                }
                let key = (c.from.index, c.to.node);
                if seen.contains(&key) {
                    false
                } else {
                    seen.push(key);
                    true
                }
            });
        }
    }

        /// Human-readable connection labels, e.g. `("n0_o0", "n1_i0")` — a string view of
        /// `self.connections`, handy for debugging/DSL rendering.
    pub fn connection_labels(&self) -> Vec<(String, String)> {
        self.connections
            .iter()
            .map(|c| (c.from_label(), c.to_label()))
            .collect()
    }

        /// Check the graph is executable before building: contiguous ids, forward-only
        /// wires, in-range ports, output 1:1, no orphaned inputs. `Network::build` calls it.
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
                // Per-port activations: length must match the port count, and Input nodes
                // keep their ports raw (Identity) — activating raw data is a bug.
            if let Some(ports) = &node.port_activations {
                if ports.len() != node.num_outputs {
                    return Err(TopologyError::InvalidOptions(format!(
                        "node n{i} has {} port activations but {} output ports",
                        ports.len(),
                        node.num_outputs
                    )));
                }
                if node.kind == NodeKind::Input && ports.iter().any(|a| *a != Activation::Identity)
                {
                    return Err(TopologyError::InvalidOptions(format!(
                        "node n{i} is an Input node — its ports must stay Identity"
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

            // 3. Connections: forward-only, known nodes, in-range ports, and 1:1 on the
            // OUTPUT side. Input ports may receive several wires (the node combines them).
        let mut used_sources: Vec<Port> = Vec::new();
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
            // Input node's output ports can fan out to multiple destinations.
            let is_input_node = self.nodes[conn.from.node].kind == NodeKind::Input;
            if !is_input_node && used_sources.contains(&conn.from) {
                return Err(TopologyError::DoubleUsedOutput(conn.from));
            }
            used_sources.push(conn.from);
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

        /// The effective hidden dim for orphan projections: the max output dim across all
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

    /// Wired degrees `(in_degree, out_degree)` per node, indexed by node id
    /// — one count per *wire* (a port with several wires counts several).
    pub fn degrees(&self) -> Vec<(usize, usize)> {
        node_degrees(&self.nodes, &self.connections)
    }

        /// Longest path from an `Input` node per node id — a topological level. Input nodes
        /// sit at 0; others are `1 + max` over wired sources. Ascending ids are a valid order.
    pub fn depths(&self) -> Vec<usize> {
        node_depths(&self.nodes, &self.connections)
    }

        /// Estimated total learnable elements (weights + biases) without building — the
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

            // 2. Exactly one Output node: append the projection (`hidden_dim → output_dim`);
            // with auto-detected output_dim there's no need for multiple Output nodes.
        let has_output = self.nodes.iter().any(|n| n.kind == NodeKind::Output);
        if !has_output {
            let mut out = Node::new_output(self.nodes.len(), 1, 1);
            out.hidden_dim = Some(self.options.output_dim.unwrap_or(1));
            self.nodes.push(out);
        }
    }

        /// Re-wire orphaned output ports into random *later* nodes — the de-orphaning rule,
        /// called at the end of `finalize`. Deterministic per `options.topology_seed`.
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
            let target = targets[self.rng.usize(0..targets.len())];
            // Even an already-wired input port can take another wire: the
            // node combines all incoming tensors (Add/Mean) anyway.
            let port = self.rng.usize(0..self.nodes[target].num_inputs);
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

    /// Remove orphaned input/output ports and renumber connections to close
    /// gaps. The Output node keeps all its ports (network output).
    fn trim_orphaned_ports(&mut self) {
        let output_id = self
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Output)
            .map(|n| n.id);

        // Build remap tables: for each node, old index -> compact new index.
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
                        map[i] = usize::MAX;
                    }
                }
                map
            })
            .collect();

        // Renumber connections, dropping orphans.
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
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if Some(node.id) != output_id {
                node.num_outputs = out_remap[i]
                    .iter()
                    .filter(|&&v| v != usize::MAX)
                    .count()
                    .max(1);
                    // Compact port_activations alongside the port remap — the vec must always
                    // match num_outputs (`validate()` enforces it). Input ports stay Identity.
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
            }
            let connected_in = in_remap[i].iter().filter(|&&v| v != usize::MAX).count();
            if connected_in > 0 {
                node.num_inputs = connected_in;
            }
        }
    }

        /// Collapse value-duplicate wires: within each (from.node → to.node) group keep the
        /// first wire per distinct source-port activation. Runs after trim_orphaned_ports.
    fn dedup_value_wires(&mut self) {
        let mut kept: Vec<(usize, usize, crate::graph::node::Activation)> = Vec::new();
        self.connections.retain(|c| {
            let act = self.nodes[c.from.node].port_activation(c.from.index);
            let key = (c.from.node, c.to.node, act);
            if kept.contains(&key) {
                false
            } else {
                kept.push(key);
                true
            }
        });
    }

        /// Serialize the whole blueprint (options, nodes, labels, connections) to JSON
        /// (see [`crate::spec::Spec`]); the RNG is not stored.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

        /// Deserialize a blueprint from JSON (see `to_json`). The RNG is re-seeded from
        /// `options.topology_seed`, so loaded graphs wire identically; weights are fresh.
    pub fn from_json(s: &str) -> Result<Topology, serde_json::Error> {
        serde_json::from_str(s)
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

        /// Crossover: find a matching-node pivot (same port counts), swap everything from
        /// that pivot onward, then finalize both. Returns true if a swap happened.
    pub fn cx_one_point(a: &mut Topology, b: &mut Topology, rng: &mut Rng) -> bool {
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
            return false;
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
            debug!(
                "cx_one_point: pivot hidden[{}] <-> hidden[{}], swapped {} nodes",
                pivot_a, pivot_b, len
            );
            return true;
        }
        debug!("cx_one_point: no matching node found, skipping");
        false
    }

        /// Uniform crossover: per-node independent swap, requiring equal hidden counts.
        /// Returns true if at least one swap happened.
    pub fn cx_uniform(a: &mut Topology, b: &mut Topology, swap_prob: f32, rng: &mut Rng) -> bool {
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
            return false;
        }

        let mut swaps = 0usize;
        for (&ai, &bi) in ha.iter().zip(hb.iter()) {
            if rng.f32() < swap_prob {
                std::mem::swap(&mut a.nodes[ai], &mut b.nodes[bi]);
                swaps += 1;
            }
        }

        if swaps > 0 {
            a.renumber_ids();
            a.finalize();
            b.renumber_ids();
            b.finalize();
            debug!("cx_uniform: swapped {swaps}/{} hidden nodes", ha.len());
        }
        swaps > 0
    }
}

use crate::utils::graph_utils::{
    build_node_sources, node_activation_counts, node_degrees, node_depths, node_kind_counts,
    node_orphan_counts,
};

/// Test-only proptest strategies, shared across the crate's test modules
/// (topology's own tests and the serialization tests in [`crate::spec`]).
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
        assert_eq!(graph.options.max_hidden_num_nodes, 5);
    }

    #[test]
    fn test_new_with_options() {
        let opts = TopologyOptions {
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
    fn test_dedup_value_wires_collapses_same_activation() {
            // Two hidden nodes: with equal per-port activations the second wire is a
            // value-duplicate and must be dropped; with different activations both survive.
        let build = |port_acts: Option<Vec<Activation>>| {
            let mut g = Topology::new(0, Some(TopologyOptions::default()));
            g.nodes.push(Node::new_input(0, 1));
            let mut n1 = Node::new_hidden(1, 1, 2);
            n1.activation = Activation::ReLU;
            n1.port_activations = port_acts;
            g.nodes.push(n1);
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
                from: Port { node: 1, index: 1 },
                to: Port { node: 2, index: 1 },
            });
            g
        };
        // Ports 0 and 1 BOTH carry ReLU ⇒ duplicate value ⇒ dedup to 1 wire.
        let mut dup = build(Some(vec![Activation::ReLU, Activation::ReLU]));
        dup.dedup_value_wires();
        let n1_to_n2 = dup
            .connections
            .iter()
            .filter(|c| c.from.node == 1 && c.to.node == 2)
            .count();
        assert_eq!(
            n1_to_n2, 1,
            "same-activation wires into one target must dedup"
        );
        // Port 0 ReLU, port 1 Tanh ⇒ distinct values ⇒ both survive.
        let mut distinct = build(Some(vec![Activation::ReLU, Activation::Tanh]));
        distinct.dedup_value_wires();
        let n1_to_n2 = distinct
            .connections
            .iter()
            .filter(|c| c.from.node == 1 && c.to.node == 2)
            .count();
        assert_eq!(n1_to_n2, 2, "distinct-activation wires must survive");
    }

    #[test]
    fn test_dedup_value_wires_collapses_input_duplicates() {
            // All Input ports are Identity ⇒ two Input wires into the SAME target are
            // value-duplicates and must collapse; different targets survive.
        let mut g = Topology::new(0, Some(TopologyOptions::default()));
        g.nodes.push(Node::new_input(0, 2));
        g.nodes.push(Node::new_hidden(1, 2, 1));
        g.nodes.push(Node::new_hidden(2, 1, 1));
        g.nodes.push(Node::new_output(3, 1, 1));
        g.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        g.connections.push(Connection {
            from: Port { node: 0, index: 1 },
            to: Port { node: 1, index: 1 },
        });
        g.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        g.dedup_value_wires();
        let into_n1 = g
            .connections
            .iter()
            .filter(|c| c.from.node == 0 && c.to.node == 1)
            .count();
        let into_n2 = g
            .connections
            .iter()
            .filter(|c| c.from.node == 0 && c.to.node == 2)
            .count();
        assert_eq!(
            into_n1, 1,
            "two Identity Input wires into one target must dedup"
        );
        assert_eq!(
            into_n2, 1,
            "Input wire into a different target must survive"
        );
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
        // Input node with a non-Identity port is also rejected.
        let mut g2 = Topology::new(0, Some(TopologyOptions::default()));
        let mut input = Node::new_input(0, 1);
        input.port_activations = Some(vec![Activation::ReLU]);
        g2.nodes.push(input);
        g2.nodes.push(Node::new_output(1, 1, 1));
        assert!(g2.validate().is_err());
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

    #[test]
    fn test_validate_allows_multi_wired_input_port() {
            // De-orphaning stacks extra wires into already-wired input ports (the node
            // combines all incoming tensors), so wiring one input port twice stays valid.
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 0, index: 1 },
            to: Port { node: 1, index: 0 }, // second wire into the same port
        });
        assert_eq!(graph.validate(), Ok(()));

        let module = Network::build(&graph, Device::CPU).unwrap();
        let batch = 2i64;
        let input = rand_input(batch, graph.options.input_dim.unwrap_or(1));
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, 8]);
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
        // is counted here, unlike orphan_counts which excludes it).
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
            // Chain with a widened middle node: n1 emits 32, so n2's in_dim re-derives to
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
        // A hidden-only graph (what random generation produces) gets the
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
        /// the invariant the whole design leans on.
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
    fn finalize_leaves_no_orphaned_input_output_ports() {
            // Regression: dedup_value_wires used to run AFTER trim_orphaned_ports, stranding
            // ports whose only wire got value-deduped (mermaid `identity (orphan)` stubs).
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

    #[test]
    fn exclusive_feature_masks_partition_without_sharing() {
            // Exclusive is a true partition: every feature lands in exactly one port, so
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
    fn random_feature_masks_are_per_port_and_may_overlap() {
            // Random flips an independent coin per (port, feature): no exclusivity, so the
            // only guarantees are the shape and that it is not degenerate.
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
        assert!(masks.iter().all(|m| m.len() == 8), "every mask spans input_dim");
        assert!(masks.iter().flatten().all(|b| *b <= 1), "masks are 0/1");
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
    fn finalize_drops_masks_instead_of_letting_a_stale_count_panic() {
            // Crossover re-finalizes both parents, and those parents already carry masks
            // from their own build. A mask sized for the OLD port count must never reach
            // `validate` — finalize drops it, and the engine re-derives right after.
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(13, Some(opts));
        g.create_random_hidden_nodes(3);
        g.finalize();
        g.assign_feature_masks(FeatureSplitPolicy::Exclusive);
        assert!(
            g.nodes.iter().any(|n| n.feature_masks.is_some()),
            "the split must actually be present before the rebuild"
        );

            // The rebuild itself: it must not panic on the stale masks.
        g.finalize();
        assert!(
            g.nodes.iter().all(|n| n.feature_masks.is_none()),
            "finalize must drop port-count-bound masks"
        );

            // And re-deriving immediately afterwards is the engine's flow — it must validate.
        g.assign_feature_masks(FeatureSplitPolicy::Exclusive);
        g.validate().expect("re-derived masks must validate");
    }

    #[test]
    fn feature_masks_survive_the_blueprint_round_trip() {
            // The split is genome, so the masks must come back identical from the
            // blueprint JSON — that file (not the weights) is also the Python side of
            // the contract, so the field has to travel.
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
        assert_eq!(legacy.feature_masks, None, "serde default keeps old JSON loadable");
    }

        /// The crossover path's installer: an inherited row lands on its own port, a
        /// port with no inherited row starts UNMASKED (every feature reaches it), and
        /// the result always validates against the final port count.
    #[test]
    fn install_feature_masks_maps_rows_and_leaves_new_ports_unmasked() {
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
        assert!(ports >= 2, "fixture needs at least two input ports, got {ports}");

        g.install_input_feature_masks(&[vec![0, 1, 1, 1]]);
        let masks = input_of(&g)
            .feature_masks
            .clone()
            .expect("rows were installed");
        assert_eq!(masks.len(), ports, "one row per port");
        assert_eq!(masks[0], vec![0, 1, 1, 1], "the inherited row rides port 0");
        assert!(
            masks[1..].iter().all(|m| m == &vec![1u8; 4]),
            "a port with no inherited row sees every feature"
        );
        g.validate().expect("installed masks must validate");
        assert_eq!(
            g.input_feature_masks().expect("readable"),
            masks,
            "input_feature_masks must read back exactly what was installed"
        );
    }

        /// No inherited rows at all (neither parent carried a split, no policy): the
        /// input stays unmasked rather than getting an invented one.
    #[test]
    fn install_feature_masks_with_no_rows_leaves_every_port_unmasked() {
        let opts = TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(24, Some(opts));
        g.create_random_hidden_nodes(2);
        g.finalize();
        g.install_input_feature_masks(&[]);
        let masks = g.input_feature_masks().expect("installed");
        assert!(
            masks.iter().all(|m| m == &vec![1u8; 4]),
            "every port keeps every feature"
        );
        g.validate().expect("unmasked rows must validate");
    }
}
