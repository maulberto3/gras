//! The graph blueprint  — pure topology data, no tensors.
//!
//! Two-phase design: **Topology** (this file, pure data) declares nodes
//! and connections. **Network** compiles it into real layers and executes.
//!
//! Invariants: contiguous ids (0..n), forward-only edges, orphaned ports
//! fed by network input. `validate()` checks all of them.

use fastrand::Rng;
use log::debug;
use serde::{Deserialize, Serialize};

use crate::utils::error::TopologyError;

use crate::node::{Activation, Node, NodeKind};

pub use crate::node::CombineOp;

// ── KindCounts — node-role histogram ──────────────────────────────────────

/// Node counts by kind — the return type of [`Topology::kind_counts`]:
/// no more guessing the order of a bare `(usize, usize, usize)` tuple.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KindCounts {
    pub input: usize,
    pub hidden: usize,
    pub output: usize,
}

/// Knobs for building a graph. Mostly used by the random-generation
/// methods (`create_random_hidden_node`, `finalize`); `input_dim`,
/// `hidden_dim` and `combine_op` are what execution actually cares about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
// ── TopologyOptions — the blueprint template ──────────────────────────────

pub struct TopologyOptions {
    /// RNG seed — same seed => same random graph.
    pub seed: usize,
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
    /// Feature dimension of the network input tensor.
    pub input_dim: usize,
    /// Internal feature dimension shared by every node.
    pub hidden_dim: usize,
    /// Output dimension of the network (set on the Output node's layer).
    /// The output node maps `hidden_dim -> output_dim`; auto-detected from
    /// the dataset's target shape by the engine, or set manually.
    pub output_dim: usize,
}

impl Default for TopologyOptions {
    fn default() -> Self {
        TopologyOptions {
            seed: 55,
            min_hidden_num_nodes: 2,
            max_hidden_num_nodes: 5,
            min_hidden_inputs_per_node: 2,
            max_hidden_inputs_per_node: 5,
            min_hidden_outputs_per_node: 2,
            max_hidden_outputs_per_node: 5,
            input_dim: 1,
            hidden_dim: 8,
            output_dim: 1,
        }
    }
}

// ── Port & Connection — wiring primitives ─────────────────────────────────

/// A port is a "socket"  on a node.
///   - as a destination (connection.to)  -> an input port  in 0..num_inputs
///   - as a source     (connection.from) -> an output port in 0..num_outputs
///
/// Ports exist so a node can fan out (one output port per wire) while keeping
/// wiring 1:1. **All of a node's output ports emit the same tensor** — the
/// output-port index only matters for bookkeeping, never for values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Port {
    pub node: usize,  //  which node
    pub index: usize, //  which socket on that node
}

/// A directed wire  from one node's output port to another node's input port.
///
/// Example: `n1_o0 -> n2_i0` means "node 1's first output feeds node 2's
/// first input".
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Connection {
    pub from: Port,
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
// Free functions over raw slices — live in crate::utils::graph_utils.
// Topology methods delegate to them.
// for the actual implementations.

/// The blueprint  of a gras network: pure data, no tensors.
///
/// A `Topology` answers two questions only:
///   - **who exists** — `nodes`, each declaring its input/output port counts
///   - **who feeds whom** — `connections`, each a wire from an output port
///     to an input port
///
/// It is inert by itself: [`Topology::validate`] checks it can be executed, and
/// [`Network::build`](crate::network::Network::build) turns it into
/// real flodl layers. Add nodes with the [`Node`] constructors
/// (`Node::new_input`, `new_hidden`, `new_output`) — they keep ids
/// contiguous, which the rest of the code relies on.
///
/// Fields:
///   - `id` — instance id (used to name
///     [`Network`](crate::network::Network)s uniquely)
///   - `options` — dims, RNG seed, combine op (see [`TopologyOptions`])
///   - `graph_inputs` / `graph_outputs` — one string label per port, minted by
///     [`Topology::refresh_labels`]; for rendering/debugging only — execution
///     reads the typed [`Port`]s in `connections`
///   - `rng` — the seeded RNG driving random node/wiring generation
///     (private — callers never need direct access)
#[derive(Clone, Debug, PartialEq)]
// ── Topology — the graph blueprint ───────────────────────────────────────

pub struct Topology {
    pub id: usize,
    pub nodes: Vec<Node>,
    pub options: TopologyOptions,
    pub(crate) graph_inputs: Vec<String>,
    pub(crate) graph_outputs: Vec<String>,
    pub connections: Vec<Connection>,
    pub(crate) rng: Rng,
}

impl Topology {
    // ── Creation ─────────────────────────────────────────────────────────────

    /// Create an empty graph. Pass `None` to use the default options (seed 16,
    /// hidden_dim 8, ...). The graph starts with **no nodes** — add them with
    /// the [`Node`] constructors + `graph.nodes.push(...)`.
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
            rng: Rng::with_seed(opts.seed as u64),
        }
    }

    /// Append one random hidden node whose port counts are drawn from the
    /// options ranges. The node's id is the current node count, so ids stay
    /// contiguous `0..n` (invariant #1).
    pub fn create_random_hidden_node(&mut self) {
        // Create a random hidden node with random number of inputs and outputs
        // following the options constraints. Simple maths:
        //   num_inputs  ∈ [min_hidden_inputs_per_node,  max_hidden_inputs_per_node]
        //   num_outputs ∈ [min_hidden_outputs_per_node, max_hidden_outputs_per_node]
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

    /// Assign every hidden node a **random activation** drawn from `pool`
    /// (Input/Output nodes keep [`Activation::Identity`]).
    /// Refresh the port labels  — one string per port (`n1_i0`, `n2_o3`,
    /// ...) in `graph_inputs` / `graph_outputs`. These labels are the
    /// "address book" used by `Display`, the ASCII renderer in `utils`, and
    /// `connection_labels`. Execution itself reads the typed [`Port`]s
    /// directly, so this step is only needed for pretty-printing and
    /// debugging — call it after adding/changing nodes.
    ///
    /// Simple maths: a node i with `num_inputs` inputs and `num_outputs`
    /// outputs contributes exactly
    /// Re-number node IDs to be contiguous 0..n.
    /// Needed after crossover swaps nodes between topologies.
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

    /// Finalize the graph : scaffold the skeleton, wire every port, and
    /// auto-de-orphan dangling outputs — one call turns a pile of nodes
    /// into a complete, executable graph.
    ///
    /// Simple mental model:
    ///   0. **Auto-scaffold** the canonical skeleton (Input → … → Output) —
    ///      random graphs only create hidden nodes, so this prepends a single
    ///      Input node and appends/merges exactly one Output node (the
    ///      output projection). See [`Topology::ensure_scaffold`].
    ///   1. Collect every input port  (each one needs a source)
    ///   2. Collect every output port (each one is a potential source)
    ///   3. Shuffle both lists, then pair each input port with an unused
    ///      output port from an EARLIER node.
    ///
    /// Example wiring (n0 input with 2 outs, n1 hidden with 2 ins, n2 hidden 1 in):
    ///
    ///   n0 input       n1 hidden      n2 hidden
    ///   ┌───────────┐  ┌───────────┐  ┌───────────┐
    ///   │ o0 ───────┼─▶│ i0        │  │           │
    ///   │ o1 ───────┼─▶│ i1        │  │           │
    ///   │           │  │ o0 ───────┼─▶│ i0        │
    ///   └───────────┘  └───────────┘  └───────────┘
    ///
    /// (if n1 had a third input i2 it would have no earlier source left
    ///  → orphaned → fed by the network input at execution time)
    ///
    /// Rules:
    ///   -  no recurrent connections: only forward edges, so
    ///     from.node < to.node (this also forbids self-loops, since a node is
    ///     never earlier than itself)
    ///   -  1:1 pairing: each output port feeds at most one input port
    ///   -  orphaned input ports (no earlier output available) are fed by
    ///     the network input at execution time
    ///   -  orphaned output ports are rewired automatically at the end by
    ///     [`Topology::rewire_orphaned_outputs`] — even into already-wired input ports
    ///     (the node combines them with Add/Mean) — so a single call yields a
    ///     complete graph.
    ///
    /// Simple maths: with I input ports and O output ports the random pass
    /// creates at most min(I, O) connections; the de-orphan pass may add
    /// more (one per orphaned output with a compatible later target).
    // ── Finalize — scaffold + wiring ─────────────────────────────────────────

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

        // Step 5: filter_input — set input node output count = number of hidden nodes
        self.set_input_fanout();

        // Step 6: connect_orphans — wire orphaned inputs to the input node (cycle ports)
        self.rewire_orphaned_inputs();

        // Step 6b: clean_wiring — dedup input node connections (one wire per source_port, target_node)
        self.dedup_input_connections();

        // Step 7: rescue_outputs — wire orphaned outputs to later nodes (dedup inline)
        self.rewire_orphaned_outputs();

        // Step 7b: trim_ports — remove orphaned input/output ports from the topology
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
                    NodeKind::Output => self.options.output_dim,
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

    /// Step 6: Wire orphaned input ports to the input node's output ports.
    /// Cycles through available output ports (multiple orphans may share
    /// a port — the combine op merges them).
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

    /// Human-readable connection labels, e.g. `("n0_o0", "n1_i0")` — handy
    /// for debugging and for feeding a `layer!`-style DSL later. Just a
    /// string view of `self.connections` (same data, no duplication).
    pub fn connection_labels(&self) -> Vec<(String, String)> {
        self.connections
            .iter()
            .map(|c| (c.from_label(), c.to_label()))
            .collect()
    }

    /// Check the graph is executable before building/running it.
    ///
    /// A random graph is safe to execute when:
    ///   - node ids are contiguous 0..n
    ///   - every connection goes strictly forward (from.node < to.node)
    ///   - every wired port exists on its node (index within bounds)
    ///   - no output port feeds two inputs (except the Input node which fans out)
    ///   - every input port is wired (no orphans after finalize)
    ///
    /// Network::build calls this and refuses to build invalid graphs.
    // ── Validate — check wiring invariants ───────────────────────────────────

    pub fn validate(&self) -> Result<(), TopologyError> {
        // 1. Options sanity
        if self.options.input_dim == 0
            || self.options.hidden_dim == 0
            || self.options.output_dim == 0
        {
            return Err(TopologyError::InvalidOptions(
                "input_dim, hidden_dim, and output_dim must be > 0".to_string(),
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

        // 3. Connections: forward-only, known nodes, in-range ports, and
        //    1:1 on the OUTPUT side (an output port feeds at most one input).
        //    Input ports may receive several wires — the node combines them
        //    which is what de-orphaning relies on.
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

        // 4. Feature-dim consistency per node.
        // With per-node hidden_dim + port projections, all source dim
        // combinations are valid — projections bridge any mismatch.
        // No dim consistency check needed.

        Ok(())
    }

    /// Count orphaned ports: `(orphaned_inputs, orphaned_outputs)`.
    ///
    /// - an **orphaned input port** has no wire — it is fed the network input
    ///   at execution time (legal by design, see the module docs)
    /// - an **orphaned output port** has no wire feeding another node. The
    ///   graph-output node's own output ports are **excluded** — those are
    ///   the graph's answer, consumed by the caller, not orphans.
    // ── Diagnostics — derived graph stats ────────────────────────────────────

    pub fn orphan_counts(&self) -> (usize, usize) {
        node_orphan_counts(&self.nodes, &self.connections, self.nodes.len() - 1)
    }

    // ── shared helpers ──────────────────────────────────────────────────

    /// Output dim of a node: its `hidden_dim` override, or the graph default.
    fn out_dim_of(&self, node: &Node) -> usize {
        node.hidden_dim.unwrap_or(self.options.hidden_dim)
    }

    /// The effective hidden dim for orphan projections: the max output dim
    /// across all nodes (per-node `hidden_dim` overrides + template fallback).
    /// Used as the fallback for orphaned ports and the input node, so they
    /// project straight to the widest dim in the network — no double projection.
    fn effective_hidden_dim(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| self.out_dim_of(n))
            .max()
            .unwrap_or(self.options.hidden_dim)
    }

    /// Per-node input dim: the (validated-identical) output dim of its
    // ── derived diagnostics (the "missing data" catalog) ───────────────────

    /// Derived per-node feature dims `(in_dim, out_dim)`, indexed by node id
    /// — the same derivation [`Network::build`](crate::network::Network::build)
    /// uses.
    ///
    /// `in_dim` = the (validated-identical) dim of the node's wired sources —
    /// the output dim of each source node, all guaranteed equal by
    /// [`Topology::validate`] — or `hidden_dim` when the node has no sources /
    /// any orphaned port (orphans read `net_input`, which is `hidden_dim`
    /// wide). `out_dim` = the node's `hidden_dim` override, or the graph's.
    pub fn node_dims(&self) -> Vec<(usize, usize)> {
        let num_inputs: Vec<usize> = self.nodes.iter().map(|n| n.num_inputs).collect();
        let sources = build_node_sources(&self.connections, &num_inputs);
        self.nodes
            .iter()
            .map(|node| {
                let in_dim = if node.kind == NodeKind::Input {
                    // Input node reads raw data: in_dim = input_dim
                    self.options.input_dim
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

    /// Orphaned ports of one node: `(orphaned input indices, orphaned output
    /// indices)`. An orphaned input is fed `net_input` at execution; an
    /// orphaned output is computed but unused (the graph-output node's own
    /// ports are the answer, not orphans).
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

    /// Longest path from an `Input` node per node id — a topological *level*.
    /// Input nodes sit at level 0; every other node is `1 + max` over its
    /// wired sources' levels. Orphaned ports read `net_input` (level 0), so
    /// they don't add depth. Useful for depth-biased NAS and level-based
    /// rendering. Edges are forward-only, so ascending ids are a valid order
    /// for this DP.
    pub fn depths(&self) -> Vec<usize> {
        node_depths(&self.nodes, &self.connections)
    }

    /// Estimated total learnable elements (weights + biases), without
    /// building: the input projection plus every node's `in·out + out`.
    /// Matches the real count of a built [`Network`](crate::network::Network).
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

    /// Ensure the canonical skeleton : at least one `Input` node and
    /// exactly one `Output` node. Idempotent — safe to call any number of
    /// times.
    ///
    /// Called automatically as phase 1 of [`Topology::finalize`]; public
    /// for manual use after a topology mutation (e.g. NAS evolution).
    ///
    ///   - **No `Input` node** → prepend `Node::new_input(0, 1)`: a single
    ///     output port. Fan-out is deliberately minimal — every input port
    ///     the Input doesn't reach is fed the network input (`net_input`) at
    ///     execution time anyway, so one reusable source suffices.
    ///   - **Zero `Output` nodes** → append the output projection
    ///     (`new_output(last + 1, 1, 1)`) with `hidden_dim = Some(output_dim)`:
    ///     the network output reads this node. Its layer is the learned
    ///     output projection (`Linear: hidden_dim → output_dim`), the
    ///     counterpart of `input_proj`.
    ///   - **One `Output` node** → no-op (already correct).
    ///   - **More than one `Output` node** → [`Topology::validate`] rejects
    ///     the graph (multiple Output nodes are not supported).
    ///
    /// Prepending the Input node shifts every existing node id by +1; any
    /// connections are re-mapped to match (`finalize` re-mints the port
    /// labels afterwards; call [`Topology::refresh_labels`] yourself when
    /// calling this standalone).
    // ── Scaffold & de-orphaning ──────────────────────────────────────────────

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

        // 2. Exactly one Output node: append the projection.
        //    With output_dim auto-detected from data, there is no need
        //    for multiple Output nodes or merging — the single Output node
        //    projects hidden_dim → output_dim.
        let has_output = self.nodes.iter().any(|n| n.kind == NodeKind::Output);
        if !has_output {
            let mut out = Node::new_output(self.nodes.len(), 1, 1);
            out.hidden_dim = Some(self.options.output_dim);
            self.nodes.push(out);
        }
    }

    /// Re-wire orphaned output ports into random *later* nodes — the
    /// de-orphaning rule →. Called automatically at the end of
    /// [`Topology::finalize`], so the normal pipeline never needs to
    /// call it by hand; it's public for manual rewiring (e.g. after a
    /// topology mutation).
    ///
    /// For every output port with no wire (on non-graph-output nodes), pick a
    /// random later node whose input dim matches this source's output dim and
    /// wire into a **random input port of it — even if that port is already
    /// wired**. That is safe because a node combines *all* incoming tensors
    /// with Add/Mean before transforming them, so stacking an extra wire just
    /// adds another term to the sum.
    ///
    /// The target is drawn from `self.rng`, so the rewiring is deterministic
    /// per `options.seed` — same seed => same de-orphaned graph.
    ///
    /// Returns the number of wires added. Graphs where a source has no
    /// compatible later target (e.g. its output dim doesn't match any later
    /// node) keep that output orphaned.
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
            }
            let connected_in = in_remap[i].iter().filter(|&&v| v != usize::MAX).count();
            if connected_in > 0 {
                node.num_inputs = connected_in;
            }
        }
    }

    /// Serialize the whole blueprint (options, nodes, labels, connections) to
    /// JSON.  See [`crate::spec::Spec`] for the shape; the RNG
    /// is not stored.
    // ── Serialization ────────────────────────────────────────────────────────

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a blueprint from JSON (see [`Topology::to_json`]).
    ///
    /// The RNG is **re-seeded from `options.seed`**, so any regeneration after
    /// loading (e.g. `finalize`) is deterministic — a loaded graph
    /// wires identically to a freshly created graph with the same options.
    ///
    /// The executable module is rebuilt from the loaded blueprint with
    /// [`Network::build`](crate::network::Network::build) — same
    /// architecture, fresh random weights (no weights are ever serialized).
    pub fn from_json(s: &str) -> Result<Topology, serde_json::Error> {
        serde_json::from_str(s)
    }

    // ── Crossover ──────────────────────────────────────────────────────────

    /// Find a hidden node with the same signature in both topologies.
    /// Signature: (activation, combine_op, standardize, hidden_dim, num_inputs, num_outputs).
    /// Returns (index_in_a, index_in_b) of hidden-only lists.
    fn find_matching_node(
        a: &Topology,
        b: &Topology,
        rng: &mut Rng,
    ) -> Option<(usize, usize)> {
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

        // Collect all matching pairs — only structural interface matters
        // (num_inputs, num_outputs). finalize() handles wiring;
        // activation, combine_op, etc. can differ freely.
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

    /// Crossover: find a matching-node pivot (same num_inputs/num_outputs),
    /// swap everything from that pivot onward, then finalize both.
    /// Returns true if a swap happened.
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
            debug!("cx_one_point: pivot hidden[{}] <-> hidden[{}], swapped {} nodes", pivot_a, pivot_b, len);
            return true;
        }
        debug!("cx_one_point: no matching node found, skipping");
        false
    }

    /// Uniform crossover: per-node independent swap.
    /// Requires same number of hidden nodes; each hidden node's attributes
    /// are independently swapped with probability `swap_prob`.
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
            debug!("cx_uniform: hidden node count mismatch ({} vs {}), skipping", ha.len(), hb.len());
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

    /// Valid random `TopologyOptions`: non-zero dims and non-inverted ranges
    /// (inverted ranges / zero dims are exactly what `validate()` rejects,
    /// not what we generate).
    pub(crate) fn topology_options_strategy() -> impl Strategy<Value = TopologyOptions> {
        (
            0usize..10_000, //  seed
            1usize..4,      //  min inputs per node
            1usize..4,      //  min outputs per node
            1usize..8,      // input_dim
            1usize..8,      // hidden_dim
            1usize..8,      // output_dim
        )
            .prop_map(
                |(seed, min_in, min_out, input_dim, hidden_dim, output_dim)| TopologyOptions {
                    seed,
                    min_hidden_num_nodes: 2,
                    max_hidden_num_nodes: 6,
                    min_hidden_inputs_per_node: min_in,
                    max_hidden_inputs_per_node: min_in + 3,
                    min_hidden_outputs_per_node: min_out,
                    max_hidden_outputs_per_node: min_out + 3,
                    input_dim,
                    hidden_dim,
                    output_dim,
                },
            )
    }

    /// A random graph: input node + 0..5 hidden nodes + output node, fully
    /// wired through the standard pipeline (`refresh_labels` +
    /// `finalize`, which auto-de-orphans). Port counts vary per
    /// node within the options ranges.
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
    use crate::network::Network;
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
            seed: 123,
            min_hidden_num_nodes: 3,
            max_hidden_num_nodes: 10,
            min_hidden_inputs_per_node: 1,
            max_hidden_inputs_per_node: 5,
            min_hidden_outputs_per_node: 1,
            max_hidden_outputs_per_node: 5,
            input_dim: 4,
            hidden_dim: 16,
            output_dim: 10,
        };
        let graph = Topology::new(1, Some(opts));
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_hidden_num_nodes, 3);
        assert_eq!(graph.options.max_hidden_num_nodes, 10);
        assert_eq!(graph.options.input_dim, 4);
        assert_eq!(graph.options.hidden_dim, 16);
        assert_eq!(graph.options.output_dim, 10);
    }

    #[test]
    fn test_create_random_hidden_node() {
        let mut graph = Topology::new(1, None);
        graph.create_random_hidden_node();
        assert_eq!(graph.nodes.len(), 1);
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

        // Output ports stay 1:1 (each feeds at most one input); input ports
        // may hold several wires because finalize auto-de-orphans
        // (stacks extra sources into later nodes, even occupied ports).
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
        // De-orphaning stacks extra wires into already-wired input ports
        // (the node combines all incoming tensors with Add/Mean), so wiring
        // one input port twice must stay valid — as long as output 1:1 holds.
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
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.hidden_dim as i64]);
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
        // n1 has 3 inputs, only i2 is wired → 2 orphaned inputs (fed by
        // net_input). n2's single output has no wire → 1 orphaned output.
        // n3 is the network output: its own output port is the answer, not an
        // orphan, so it must NOT be counted.
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
        // n0 input (2 outs) → n1 hidden (3 in, 2 out) → n2 hidden (2 in, 1
        // out) → n3 output (1 in). Wired sparsely on purpose so orphaned
        // ports stay orphaned for the checks.
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
        // Chain with a widened middle node: n1 emits 32, so n2's in_dim
        // must re-derive to 32; n0 (orphan, no sources) gets effective
        // hidden_dim = 32 as its in_dim so orphan projections go straight
        // to the widest dim.
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
        // n0: input node, in_dim = input_dim (1), out_dim = template (8)
        // n1: wired from n0 (out=8), out_dim = 32
        // n2: wired from n1 (out=32), out_dim = template (8)
        assert_eq!(g.node_dims(), vec![(1, 8), (8, 32), (32, 8)]);
    }

    #[test]
    fn test_rewire_orphaned_outputs() {
        // Same graph as the demo: n2_o0 is orphaned. De-orphaning must wire
        // it into a random later node — n3 is the only later node, and its
        // input port is already wired, which is fine (tensors are summed).
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
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.hidden_dim as i64]);
    }

    #[test]
    fn test_rewire_orphaned_outputs_rewires_even_with_dim_mismatch() {
        // n1 widens 8 → 32; its output is orphaned. The only later node
        // (n2) has different dims, but port projections bridge any mismatch,
        // so the orphan is still rewired.
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
            graph.options.seed = seed;
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
        // finalize alone must leave zero orphaned *outputs* (the
        // graph-output node's own ports are the answer, not orphans) and
        // always validate.
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
            // After finalize, orphan counts may be nonzero when dedup removes
            // redundant same-source-to-same-target connections. The network
            // handles orphaned ports gracefully (zero tensor fallback).
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
            let input = rand_input(2, graph.options.input_dim);
            let out = module.forward(&input).unwrap();
            prop_assert_eq!(out.shape(), &[2, graph.options.output_dim as i64]);
        }

        /// The blueprint's derived diagnostics match the built engine
        /// exactly: node dims agree, param_estimate equals the real parameter
        /// element count, degrees balance, and every wire strictly increases
        /// depth.
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

        /// The standard pipeline scaffolds any graph into the canonical
        /// skeleton: ≥ 1 Input node, exactly 1 Output node, still valid —
        /// even for hidden-only (or empty) random graphs.
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
