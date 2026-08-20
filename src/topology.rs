//! The graph blueprint 🧬 — pure topology data, no tensors.
//!
//! # The two-phase design 🪜
//!
//! A gras network is described in two layers:
//!
//! 1. **Blueprint — [`Topology`] (this file, pure data).** Declares only *who
//!    exists* (nodes with input/output port counts) and *who feeds whom*
//!    (connections). It is inert: you can print it, mutate it, and validate it
//!    without any tensor backend.
//! 2. **Engine — [`Network`](crate::network::Network) (a flodl
//!    [`Module`](flodl::nn::Module)).**
//!    [`Network::build`](crate::network::Network::build) compiles a
//!    validated `Topology` into real `Linear` layers, and
//!    [`Network::forward`](crate::network::Network) executes it
//!    tensor by tensor.
//!
//! Serialization (the JSON round-trip) lives in [`crate::spec`]; the
//! JSON methods on [`Topology`] (`to_json` / `from_json`) delegate to it.
//!
//! # Why execution stays simple
//!
//! Everything that makes the forward pass trivial is an **invariant the code
//! enforces rather than computes**:
//!
//! - **Node ids are contiguous `0..n` and double as array indices** — a node's
//!   id is also its execution order and its index into `Network.layers`.
//! - **Edges only go forward** (`from.node < to.node`), so ascending node id
//!   *is* a topological order — no cycle detection at runtime.
//! - **Output ports feed at most one input; input ports may hold several
//!   wires** — a node combines *every* tensor that arrives (Add/Mean), so
//!   de-orphaning can stack an extra wire into an already-wired input port.
//! - **All output ports of a node emit the same tensor** — a node's output
//!   depends on its inputs, not on *which* output port you read, so fan-out
//!   is free.
//! - **Orphaned input ports** (nothing wired) are fed the network input — a
//!   random graph never needs to be fully connected to be executable.
//!
//! [`Topology::validate`] checks all of the above;
//! [`Network::build`](crate::network::Network::build) refuses to
//! compile a graph that fails.
//!
//! # Suggested reading order 👀
//!
//! [`Port`] + [`Connection`] → [`Topology`] → [`Topology::finalize`] →
//! [`Topology::validate`] →
//! [`Network::build`](crate::network::Network::build) →
//! [`Network::forward`](crate::network::Network).

use std::collections::HashMap;

use fastrand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::TopologyError;
use crate::node::{Activation, Node, NodeKind};

/// How multiple incoming tensors into a node are combined before the node
/// transforms them.
///
/// Simple maths: a node receiving tensors [a, b, c]
///   - Add  -> a + b + c          (sum)
///   - Mean -> (a + b + c) / 3    (average)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombineOp {
    /// Sum the incoming tensors.
    Add,
    /// Average the incoming tensors.
    Mean,
}

/// Node counts by kind — the return type of [`Topology::kind_counts`]:
/// no more guessing the order of a bare `(usize, usize, usize)` tuple.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KindCounts {
    pub input: usize,
    pub hidden: usize,
    pub output: usize,
}

/// Knobs for building a graph 🎛️. Mostly used by the random-generation    /// methods (`create_random_hidden_node`, `finalize`); `input_dim`,
/// `hidden_dim` and `combine_op` are what execution actually cares about.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopologyOptions {
    pub seed: usize,                 // 🎲 RNG seed: same seed => same random graph
    pub min_num_nodes: usize,        // min hidden nodes per individual (sampled by engine)
    pub max_num_nodes: usize,        // max hidden nodes per individual (sampled by engine)
    pub min_inputs_per_node: usize,  // 🔽 each random hidden node gets at least this many inputs
    pub max_inputs_per_node: usize,  // 🔽 ... and at most this many
    pub min_outputs_per_node: usize, // 🔼 each random hidden node gets at least this many outputs
    pub max_outputs_per_node: usize, // 🔼 ... and at most this many
    /// Feature dimension of the network input tensor.
    pub input_dim: usize,
    /// Internal feature dimension shared by every node.
    pub hidden_dim: usize,
    /// Output dimension of the network (set on the Output node's layer).
    /// The output node maps `hidden_dim → output_dim`; auto-detected from
    /// the dataset's target shape by the engine, or set manually.
    pub output_dim: usize,
    /// How a node combines multiple incoming tensors.
    pub combine_op: CombineOp,
}

impl Default for TopologyOptions {
    fn default() -> Self {
        TopologyOptions {
            seed: 55,
            min_num_nodes: 2,
            max_num_nodes: 5,
            min_inputs_per_node: 2,
            max_inputs_per_node: 5,
            min_outputs_per_node: 2,
            max_outputs_per_node: 5,
            input_dim: 1,
            hidden_dim: 8,
            output_dim: 1,
            combine_op: CombineOp::Add,
        }
    }
}

/// A port is a "socket" 🔌 on a node.
///   - as a destination (connection.to)  -> an input port  in 0..num_inputs
///   - as a source     (connection.from) -> an output port in 0..num_outputs
///
/// Ports exist so a node can fan out (one output port per wire) while keeping
/// wiring 1:1. **All of a node's output ports emit the same tensor** — the
/// output-port index only matters for bookkeeping, never for values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Port {
    pub node: usize,  // 🏷️ which node
    pub index: usize, // 🔢 which socket on that node
}

/// A directed wire 🔗 from one node's output port to another node's input port.
///
/// Example: `n1_o0 -> n2_i0` means "node 1's first output feeds node 2's
/// first input".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

/// Precompute the wiring table: for each node (by id), one entry per input
/// port — a *list* of source ports feeding it (empty = orphaned, fed by
/// net_input). A port can hold several wires because de-orphaning stacks
/// extra sources into already-wired ports; the node combines them all.
///
/// Built once at compile/build time so the forward pass resolves each port
/// in O(1) instead of scanning the connection list — and so the blueprint
/// diagnostics ([`Topology::node_dims`], [`Topology::param_estimate`]) share
/// the exact table the engine uses.
pub(crate) fn build_node_sources(
    connections: &[Connection],
    num_inputs: &[usize],
) -> Vec<Vec<Vec<Port>>> {
    // (to → [from, ...]) lookup table
    let mut input_map: HashMap<Port, Vec<Port>> = HashMap::new();

    // Build the reverse map: for each input port, which source ports feed it.
    for c in connections {
        input_map.entry(c.to).or_default().push(c.from);
    }

    // For each node, for each input port, look up the list of sources (or
    // empty if orphaned). In simpler words, "for each node, for each input port, which source ports feed it?"
    num_inputs
        .iter()
        .enumerate()
        .map(|(node_id, &n)| {
            (0..n)
                .map(|i| {
                    input_map
                        .get(&Port {
                            node: node_id,
                            index: i,
                        })
                        .cloned()
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect()
}

// ── shared derived diagnostics ───────────────────────────────────────────
//
// The "nutrition facts" of a graph. They live here as **free functions over
// raw node/connection data** so both the blueprint ([`Topology`] methods
// below) and the materialized module
// ([`Network::to_json`](crate::network::Network::to_json)) report the *same*
// numbers from the *same* code — no drift between what the topology says and
// what the built net says. The `Topology` methods are thin delegations.

/// Orphaned-port counts `(inputs, outputs)` over raw graph data — the same
/// numbers [`Topology::orphan_counts`] reports. `output_node`'s own output
/// ports are the graph's answer, never orphans.
/// Without this, we wouldn't know how many tensors to feed from the network
/// input at execution time.
pub(crate) fn node_orphan_counts(
    nodes: &[Node],
    connections: &[Connection],
    output_node: usize,
) -> (usize, usize) {
    let mut orphaned_inputs = 0usize;
    let mut orphaned_outputs = 0usize;
    for (id, node) in nodes.iter().enumerate() {
        for i in 0..node.num_inputs {
            let port = Port { node: id, index: i };
            if !connections.iter().any(|c| c.to == port) {
                orphaned_inputs += 1;
            }
        }
        if id == output_node {
            continue; // graph-output ports are the answer, not orphans
        }
        for o in 0..node.num_outputs {
            let port = Port { node: id, index: o };
            if !connections.iter().any(|c| c.from == port) {
                orphaned_outputs += 1;
            }
        }
    }
    (orphaned_inputs, orphaned_outputs)
}

/// Wired degrees `(in_degree, out_degree)` per node (one count per *wire*).
/// Without this, we wouldn't know how many tensors to combine at each node.
/// See [`Topology::degrees`].
pub(crate) fn node_degrees(nodes: &[Node], connections: &[Connection]) -> Vec<(usize, usize)> {
    let mut deg = vec![(0usize, 0usize); nodes.len()];
    for c in connections {
        deg[c.to.node].0 += 1;
        deg[c.from.node].1 += 1;
    }
    deg
}

/// Longest path from an `Input` node per node id — a topological *level*.
/// Without this, we wouldn't know how many sequential layers to build in
/// the forward pass.
/// See [`Topology::depths`].
pub(crate) fn node_depths(nodes: &[Node], connections: &[Connection]) -> Vec<usize> {
    let mut depth = vec![0usize; nodes.len()];
    for node in nodes {
        if node.kind == NodeKind::Input {
            continue;
        }
        let mut d = 0usize;
        for i in 0..node.num_inputs {
            let target = Port {
                node: node.id,
                index: i,
            };
            for c in connections.iter().filter(|c| c.to == target) {
                d = d.max(depth[c.from.node] + 1);
            }
        }
        depth[node.id] = d;
    }
    depth
}

/// Node counts by kind. See [`Topology::kind_counts`] and [`KindCounts`].
/// Without this, we wouldn't know how many input/hidden/output nodes to
/// build in the forward pass.
pub(crate) fn node_kind_counts(nodes: &[Node]) -> KindCounts {
    let mut counts = KindCounts::default();
    for n in nodes {
        match n.kind {
            NodeKind::Input => counts.input += 1,
            NodeKind::Hidden => counts.hidden += 1,
            NodeKind::Output => counts.output += 1,
        }
    }
    counts
}

/// Activation histogram across nodes: `(activation, count)` pairs.
/// Without this, we wouldn't know how many nodes of each activation
/// type to build in the forward pass.
/// See [`Topology::activation_counts`].
pub(crate) fn node_activation_counts(nodes: &[Node]) -> Vec<(Activation, usize)> {
    let mut counts: Vec<(Activation, usize)> = Vec::new();
    for n in nodes {
        if let Some(entry) = counts.iter_mut().find(|(a, _)| *a == n.activation) {
            entry.1 += 1;
        } else {
            counts.push((n.activation, 1));
        }
    }
    counts
}

/// The blueprint 🧬 of a gras network: pure data, no tensors.
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
#[derive(Clone, Debug, PartialEq)]
pub struct Topology {
    pub id: usize,
    pub nodes: Vec<Node>,
    pub options: TopologyOptions,
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<String>,
    pub connections: Vec<Connection>,
    pub rng: Rng,
}

impl Topology {
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
        //   num_inputs  ∈ [min_inputs_per_node,  max_inputs_per_node]
        //   num_outputs ∈ [min_outputs_per_node, max_outputs_per_node]
        let num_inputs = self
            .rng
            .usize(self.options.min_inputs_per_node..=self.options.max_inputs_per_node);
        let num_outputs = self
            .rng
            .usize(self.options.min_outputs_per_node..=self.options.max_outputs_per_node);
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
    /// (Input/Output nodes keep [`Activation::Identity`]). 🧠
    ///
    /// This is the GP hook for the activation axis: the engine calls it
    /// after `create_random_hidden_nodes` with the individual's derived RNG,
    /// so per-node activations vary across the population (and across runs)
    /// while staying reproducible from the seed chain. `pool` must be
    /// non-empty.
    pub fn randomize_activations(&mut self, pool: &[Activation], rng: &mut fastrand::Rng) {
        if pool.is_empty() {
            return;
        }
        for node in &mut self.nodes {
            if node.kind == NodeKind::Hidden {
                node.activation = pool[rng.usize(0..pool.len())];
            }
        }
    }

    /// Refresh the port labels 📇 — one string per port (`n1_i0`, `n2_o3`,
    /// ...) in `graph_inputs` / `graph_outputs`. These labels are the
    /// "address book" used by `Display`, the ASCII renderer in `utils`, and
    /// `connection_labels`. Execution itself reads the typed [`Port`]s
    /// directly, so this step is only needed for pretty-printing and
    /// debugging — call it after adding/changing nodes.
    ///
    /// Simple maths: a node i with `num_inputs` inputs and `num_outputs`
    /// outputs contributes exactly
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

    /// Finalize the graph 🔗: scaffold the skeleton, wire every port, and
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
    ///   - ⛔ no recurrent connections: only forward edges, so
    ///     from.node < to.node (this also forbids self-loops, since a node is
    ///     never earlier than itself)
    ///   - 🔁 1:1 pairing: each output port feeds at most one input port
    ///   - 🕳️ orphaned input ports (no earlier output available) are fed by
    ///     the network input at execution time
    ///   - 🗑️ orphaned output ports are rewired automatically at the end by
    ///     [`Topology::rewire_orphaned_outputs`] — even into already-wired input ports
    ///     (the node combines them with Add/Mean) — so a single call yields a
    ///     complete graph.
    ///
    /// Simple maths: with I input ports and O output ports the random pass
    /// creates at most min(I, O) connections; the de-orphan pass may add
    /// more (one per orphaned output with a compatible later target).
    pub fn finalize(&mut self) {
        self.connections.clear();

        // 🏗️ Phase 1 — ensure the canonical skeleton: ≥ 1 Input node and
        // exactly 1 Output node (auto-scaffold, idempotent). Random graphs
        // only create hidden nodes, so this turns them into a complete
        // Input → … → Output graph.
        self.ensure_scaffold();

        // Resolve per-node hidden_dim: None → graph default.
        // This makes the topology JSON self-describing — every node shows
        // the actual dim the network will use, not a latent "inherit" marker.
        for node in &mut self.nodes {
            if node.hidden_dim.is_none() {
                node.hidden_dim = Some(match node.kind {
                    NodeKind::Output => self.options.output_dim,
                    _ => self.options.hidden_dim,
                });
            }
        }

        // Labels must reflect any nodes the scaffold just added.
        self.refresh_labels();

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

        // Randomize pairing order
        self.rng.shuffle(&mut input_ports);
        self.rng.shuffle(&mut output_ports);

        // Wires already laid down before this pass (e.g. de-orphaning)
        // keep their sources reserved: an output port stays 1:1.
        let mut used = vec![false; output_ports.len()];
        for conn in &self.connections {
            if let Some((i, _)) = output_ports
                .iter()
                .enumerate()
                .find(|(_, p)| **p == conn.from)
            {
                used[i] = true;
            }
        }
        for target in input_ports {
            let source = output_ports
                .iter()
                .enumerate()
                .find(|(i, p)| !used[*i] && p.node < target.node)
                .map(|(i, p)| (i, *p));
            if let Some((i, source)) = source {
                used[i] = true;
                self.connections.push(Connection {
                    from: source,
                    to: target,
                });
            }
        }

        // 🕳️ Automatic de-orphan: rewire output ports nobody consumed into
        // random later nodes (even occupied input ports — the node combines
        // all incoming tensors). One call ⇒ a complete graph.
        self.rewire_orphaned_outputs();
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

    /// Check the graph is executable before building/running it. 🛡️
    ///
    /// A random graph is only safe to execute when:
    ///   - node ids are contiguous `0..n` (they double as array indices)
    ///   - every connection goes strictly forward (`from.node < to.node`), so
    ///     ascending id order is a valid topological order (also forbids
    ///     self-loops and recurrent edges)
    ///   - every wired port exists on its node (index within bounds)
    ///   - no output port feeds two inputs (1:1 on the output side; input
    ///     ports may hold several wires — they are summed/averaged)
    ///   - feature dims are consistent: all tensors entering a node share one
    ///     dim, and orphaned ports (fed by the network input, dim =
    ///     `hidden_dim`) force every wired source of that node to `hidden_dim`
    ///
    /// `Network::build` calls this and refuses to build invalid graphs.
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
        if self.options.min_inputs_per_node > self.options.max_inputs_per_node
            || self.options.min_outputs_per_node > self.options.max_outputs_per_node
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
        //    (Add/Mean), which is what de-orphaning relies on.
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
            if used_sources.contains(&conn.from) {
                return Err(TopologyError::DoubleUsedOutput(conn.from));
            }
            used_sources.push(conn.from);
        }

        // 4. Feature-dim consistency per node.
        //    Every tensor entering a node must share one dim; orphaned ports
        //    are fed by net_input (dim = hidden_dim), so they pin the node's
        //    input dim to hidden_dim.
        let hidden_dim = self.options.hidden_dim;
        let out_dim = |n: &Node| n.hidden_dim.unwrap_or(hidden_dim);
        for node in &self.nodes {
            let mut source_dims: Vec<usize> = Vec::new();
            let mut has_orphan = false;
            for port in 0..node.num_inputs {
                let target = Port {
                    node: node.id,
                    index: port,
                };
                // A port may now receive several wires; collect them all.
                let mut found = false;
                for conn in &self.connections {
                    if conn.to == target {
                        source_dims.push(out_dim(&self.nodes[conn.from.node]));
                        found = true;
                    }
                }
                if !found {
                    has_orphan = true;
                }
            }
            source_dims.sort_unstable();
            source_dims.dedup();
            if has_orphan
                && let Some(&d) = source_dims.first()
                && d != hidden_dim
            {
                return Err(TopologyError::OrphanDimMismatch {
                    node: node.id,
                    hidden_dim,
                    source_dims,
                });
            } else if source_dims.len() > 1 {
                return Err(TopologyError::InconsistentInputDims {
                    node: node.id,
                    dims: source_dims,
                });
            }
        }

        Ok(())
    }

    /// Count orphaned ports: `(orphaned_inputs, orphaned_outputs)`.
    ///
    /// - an **orphaned input port** has no wire — it is fed the network input
    ///   at execution time (legal by design, see the module docs)
    /// - an **orphaned output port** has no wire feeding another node. The
    ///   graph-output node's own output ports are **excluded** — those are
    ///   the graph's answer, consumed by the caller, not orphans.
    pub fn orphan_counts(&self) -> (usize, usize) {
        // The Output node is always the last one (created by ensure_scaffold).
        node_orphan_counts(&self.nodes, &self.connections, self.nodes.len() - 1)
    }

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
        let hidden_dim = self.options.hidden_dim;
        let out_dim = |n: &Node| n.hidden_dim.unwrap_or(hidden_dim);
        let num_inputs: Vec<usize> = self.nodes.iter().map(|n| n.num_inputs).collect();
        let sources = build_node_sources(&self.connections, &num_inputs);
        self.nodes
            .iter()
            .map(|node| {
                let in_dim = sources[node.id]
                    .iter()
                    .flatten()
                    .map(|p| out_dim(&self.nodes[p.node]))
                    .max()
                    .unwrap_or(hidden_dim);
                (in_dim, out_dim(node))
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
        let mut total = self.options.input_dim * self.options.hidden_dim + self.options.hidden_dim;
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

    /// Ensure the canonical skeleton 🏗️: at least one `Input` node and
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
    /// de-orphaning rule 🕳️→🔗. Called automatically at the end of
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
    /// per `options.seed` — same seed ⇒ same de-orphaned graph.
    ///
    /// Returns the number of wires added. Graphs where a source has no
    /// compatible later target (e.g. its output dim doesn't match any later
    /// node) keep that output orphaned.
    pub fn rewire_orphaned_outputs(&mut self) -> usize {
        let hidden_dim = self.options.hidden_dim;
        let out_dim = |n: &Node| n.hidden_dim.unwrap_or(hidden_dim);

        // Input dim of each node: every tensor entering a node must share one
        // dim. A node with any orphaned input port is pinned to hidden_dim
        // (orphans read net_input); otherwise it's the common dim of its
        // wired sources (or hidden_dim if it has no sources at all).
        let node_in_dim: Vec<usize> = self
            .nodes
            .iter()
            .map(|node| {
                let mut dims: Vec<usize> = Vec::new();
                let mut has_orphan = false;
                for i in 0..node.num_inputs {
                    let target = Port {
                        node: node.id,
                        index: i,
                    };
                    let mut found = false;
                    for conn in &self.connections {
                        if conn.to == target {
                            dims.push(out_dim(&self.nodes[conn.from.node]));
                            found = true;
                        }
                    }
                    if !found {
                        has_orphan = true;
                    }
                }
                dims.sort_unstable();
                dims.dedup();
                if has_orphan || dims.is_empty() {
                    hidden_dim
                } else {
                    dims[0]
                }
            })
            .collect();

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
            let src_dim = out_dim(&self.nodes[src.node]);
            // Later nodes whose input dim matches this source's output dim.
            let targets: Vec<usize> = self
                .nodes
                .iter()
                .filter(|n| n.id > src.node && n.num_inputs > 0 && node_in_dim[n.id] == src_dim)
                .map(|n| n.id)
                .collect();
            if targets.is_empty() {
                continue; // no compatible later node — keep it orphaned
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

    /// Serialize the whole blueprint (options, nodes, labels, connections) to
    /// JSON. 🗂️ See [`crate::spec::Spec`] for the shape; the RNG
    /// is not stored.
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
}

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
            0usize..10_000, // 🎲 seed
            1usize..4,      // 🔽 min inputs per node
            1usize..4,      // 🔼 min outputs per node
            1usize..8,      // input_dim
            1usize..8,      // hidden_dim
            1usize..8,      // output_dim
            any::<bool>(),  // combine op
        )
            .prop_map(
                |(seed, min_in, min_out, input_dim, hidden_dim, output_dim, mean)| {
                    TopologyOptions {
                        seed,
                        min_num_nodes: 2,
                        max_num_nodes: 6,
                        min_inputs_per_node: min_in,
                        max_inputs_per_node: min_in + 3,
                        min_outputs_per_node: min_out,
                        max_outputs_per_node: min_out + 3,
                        input_dim,
                        hidden_dim,
                        output_dim,
                        combine_op: if mean {
                            CombineOp::Mean
                        } else {
                            CombineOp::Add
                        },
                    }
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
                let span_in = opts.max_inputs_per_node - opts.min_inputs_per_node + 1;
                let span_out = opts.max_outputs_per_node - opts.min_outputs_per_node + 1;
                let ins = opts.min_inputs_per_node + (i * 7) % span_in;
                let outs = opts.min_outputs_per_node + (i * 3) % span_out;
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
        assert_eq!(graph.options.min_num_nodes, 2);
        assert_eq!(graph.options.max_num_nodes, 5);
    }

    #[test]
    fn test_new_with_options() {
        let opts = TopologyOptions {
            seed: 123,
            min_num_nodes: 3,
            max_num_nodes: 10,
            min_inputs_per_node: 1,
            max_inputs_per_node: 5,
            min_outputs_per_node: 1,
            max_outputs_per_node: 5,
            input_dim: 4,
            hidden_dim: 16,
            output_dim: 10,
            combine_op: CombineOp::Mean,
        };
        let graph = Topology::new(1, Some(opts));
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_num_nodes, 3);
        assert_eq!(graph.options.max_num_nodes, 10);
        assert_eq!(graph.options.input_dim, 4);
        assert_eq!(graph.options.hidden_dim, 16);
        assert_eq!(graph.options.output_dim, 10);
        assert_eq!(graph.options.combine_op, CombineOp::Mean);
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

        let num_output_ports: usize = graph.nodes.iter().map(|n| n.num_outputs).sum();

        // Output ports stay 1:1 (each feeds at most one input); input ports
        // may hold several wires because finalize auto-de-orphans
        // (stacks extra sources into later nodes, even occupied ports). So
        // the wire count can exceed the input-port count but never the
        // output-port count.
        let mut seen_from: Vec<Port> = Vec::new();
        for conn in &graph.connections {
            assert!(
                conn.from.node < conn.to.node,
                "connection must go strictly forward: {conn}"
            );
            assert!(
                !seen_from.contains(&conn.from),
                "output port {} used twice",
                conn.from_label()
            );
            seen_from.push(conn.from);
        }
        assert!(graph.connections.len() <= num_output_ports);
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
        // Output ports stay 1:1 — a single output port feeds at most one input.
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 2, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 }, // same output used twice
            to: Port { node: 1, index: 1 },
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

        // Derived dims: default hidden_dim 8 everywhere on this chain.
        assert_eq!(g.node_dims(), vec![(8, 8), (8, 8), (8, 8), (8, 8)]);

        // Counts by kind.
        assert_eq!(
            g.kind_counts(),
            KindCounts {
                input: 1,
                hidden: 2,
                output: 1
            }
        );

        // Param estimate: input_proj (1·8+8) + 4 nodes × (8·8+8).
        assert_eq!(g.param_estimate(), 16 + 4 * 72);

        // Activation histogram.
        assert_eq!(g.activation_counts(), vec![(Activation::Identity, 4)]);
    }

    #[test]
    fn test_node_dims_with_override() {
        // Chain with a widened middle node: n1 emits 32, so n2's in_dim
        // must re-derive to 32 while n0 stays 8.
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
        assert_eq!(g.node_dims(), vec![(8, 8), (8, 32), (32, 8)]);
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
    fn test_rewire_orphaned_outputs_skips_incompatible_dims() {
        // n1 widens 8 → 32; its output is orphaned, but the only later node
        // (n2) has an orphaned input pinning it to hidden_dim (8). No wire
        // can be added without breaking dim consistency → stays orphaned.
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
        // n2_i0 is orphaned → n2's input dim is pinned to hidden_dim (8).
        assert_eq!(graph.orphan_counts(), (1, 1));
        assert_eq!(graph.rewire_orphaned_outputs(), 0);
        assert_eq!(graph.orphan_counts(), (1, 1));
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
    fn test_validate_rejects_inconsistent_input_dims() {
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
        assert!(matches!(
            graph.validate(),
            Err(TopologyError::InconsistentInputDims { .. })
        ));
    }

    #[test]
    fn test_validate_rejects_orphan_dim_mismatch() {
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 1));
        let mut wide = Node::new_hidden(1, 1, 1);
        wide.hidden_dim = Some(16);
        graph.nodes.push(wide);
        // n2 has 2 inputs: one wired from n1 (dim 16), one orphaned (fed by
        // net_input, dim 8) -> mismatch.
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        assert!(matches!(
            graph.validate(),
            Err(TopologyError::OrphanDimMismatch { .. })
        ));
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
            prop_assert_eq!(graph.orphan_counts().1, 0);
        }

        /// The wiring invariants hold for every generated graph: strictly
        /// forward edges and 1:1 on the output side.
        #[test]
        fn prop_wiring_invariants(graph in topology_strategy()) {
            let mut seen: Vec<Port> = Vec::new();
            for conn in &graph.connections {
                prop_assert!(conn.from.node < conn.to.node, "backward connection: {conn}");
                prop_assert!(
                    !seen.contains(&conn.from),
                    "output {} used twice",
                    conn.from_label()
                );
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
