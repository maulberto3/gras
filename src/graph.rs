//! Topology + execution of a gras graph, in one place.
//!
//! # The two-phase design 🪜
//!
//! A gras network is described in two layers:
//!
//! 1. **Blueprint — [`Graph`] (pure data, no tensors).** Declares only *who
//!    exists* (nodes with input/output port counts) and *who feeds whom*
//!    (connections). It is inert: you can print it, mutate it, and validate it
//!    without any tensor backend.
//! 2. **Engine — [`GrasGraph`] (a flodl [`Module`]).** [`GrasGraph::build`]
//!    compiles a validated `Graph` into real `Linear` layers, and
//!    [`GrasGraph::forward`] executes it tensor by tensor.
//!
//! Typical pipeline:
//!
//! ```text
//!   1. Graph::new                  empty graph + options (dims, RNG seed)
//!   2. graph.nodes.push(Node::…)   add input / hidden / output nodes
//!   3. set_graph_topology()        one string label per port (rendering)
//!   4. set_graph_network()         wire output ports → input ports (random)
//!   5. validate()                  check the random wiring is executable
//!   6. GrasGraph::build()          one flodl Linear per node + input proj
//!   7. forward()                   run it
//!   8. to_json() / from_json()     save & reload (reproducibility)
//! ```
//!
//! ```text
//!   net ──▶ input_proj ──▶ n0 ──▶ n1 ──▶ n2 ──▶ n3 ──▶ y
//!                             │         ▲
//!                             └─────────┘   (extra wire: n1 feeds n2 directly)
//! ```
//!
//! # Why execution stays simple
//!
//! Everything that makes the forward pass trivial is an **invariant the code
//! enforces rather than computes**:
//!
//! - **Node ids are contiguous `0..n` and double as array indices** — a node's
//!   id is also its execution order and its index into `GrasGraph.layers`.
//! - **Edges only go forward** (`from.node < to.node`), so ascending node id
//!   *is* a topological order — no cycle detection at runtime.
//! - **Wiring is 1:1** — one output port feeds at most one input port, and
//!   an input port has at most one wire.
//! - **All output ports of a node emit the same tensor** — a node's output
//!   depends on its inputs, not on *which* output port you read, so fan-out
//!   is free.
//! - **Orphaned input ports** (nothing wired) are fed the network input — a
//!   random graph never needs to be fully connected to be executable.
//!
//! [`Graph::validate`] checks all of the above; [`GrasGraph::build`] refuses
//! to compile a graph that fails.
//!
//! # Why the forward pass is a loop, not generated code
//!
//! Every node is a linear layer over its combined inputs, so a runtime loop
//! *is* the "unrolling" — a compile-time `layers!` macro could only follow
//! connections written literally in source, never a graph the RNG produced at
//! runtime (that would be like `println!` printing a random string). The loop
//! stays readable because [`GrasGraph::build`] precomputes the wiring once
//! (per input port: which source feeds it, or "orphan") and `forward` just
//! walks that table.
//!
//! # Suggested reading order 👀
//!
//! [`Port`] + [`Connection`] → [`Graph`] → [`Graph::set_graph_network`] →
//! [`Graph::validate`] → [`GrasGraph::build`] → [`GrasGraph::forward`].

use std::collections::HashMap;

use fastrand::Rng;
use flodl::nn::{Linear, Module, Parameter};
use flodl::tensor::TensorError;
use flodl::{Device, Variable};
use serde::de::Deserializer;
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

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

/// Knobs for building a graph 🎛️. Mostly used by the random-generation
/// methods (`create_random_hidden_node`, `set_graph_network`); `input_dim`,
/// `hidden_dim` and `combine_op` are what execution actually cares about.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphOptions {
    pub seed: usize,                 // 🎲 RNG seed: same seed => same random graph
    pub min_num_nodes: usize,        // min nodes in a generated graph (unused for now)
    pub max_num_nodes: usize,        // max nodes in a generated graph (unused for now)
    pub min_inputs_per_node: usize,  // 🔽 each random hidden node gets at least this many inputs
    pub max_inputs_per_node: usize,  // 🔽 ... and at most this many
    pub min_outputs_per_node: usize, // 🔼 each random hidden node gets at least this many outputs
    pub max_outputs_per_node: usize, // 🔼 ... and at most this many
    pub num_outputs_net: usize,      // desired graph outputs (unused for now)
    /// Feature dimension of the network input tensor.
    pub input_dim: usize,
    /// Internal feature dimension shared by every node.
    pub hidden_dim: usize,
    /// How a node combines multiple incoming tensors.
    pub combine_op: CombineOp,
}

impl GraphOptions {
    fn new() -> Self {
        GraphOptions {
            seed: 16,
            min_num_nodes: 2,
            max_num_nodes: 5,
            min_inputs_per_node: 2,
            max_inputs_per_node: 5,
            min_outputs_per_node: 2,
            max_outputs_per_node: 5,
            num_outputs_net: 1,
            input_dim: 1,
            hidden_dim: 8,
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

impl std::fmt::Display for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.from_label(), self.to_label())
    }
}

/// Why a [`Graph`] failed [`Graph::validate`]. 🛡️
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// Node ids aren't 0, 1, 2, ... — they double as array indices, so gaps
    /// or duplicates make indexing unsafe.
    NonContiguousNodeIds,
    /// No nodes at all — nothing to execute.
    EmptyGraph,
    /// Options are internally inconsistent (inverted ranges, zero dims, ...).
    InvalidOptions(String),
    /// A connection references a node id that doesn't exist.
    UnknownNode(usize),
    /// A connection doesn't go strictly forward (`from.node < to.node`);
    /// this also forbids self-loops and recurrent edges.
    BackwardConnection(Connection),
    /// A wired port index is outside the node's declared port count.
    PortOutOfBounds { port: Port, num_ports: usize },
    /// An input port is wired twice (violates 1:1 pairing).
    DoubleWiredPort(Port),
    /// An output port feeds two inputs (violates 1:1 pairing).
    DoubleUsedOutput(Port),
    /// A node's input tensors have different feature dims, so they can't be
    /// combined.
    InconsistentInputDims { node: usize, dims: Vec<usize> },
    /// A node with orphaned ports (fed by net_input, dim = hidden_dim) is
    /// also fed by wired sources of a different dim.
    OrphanDimMismatch {
        node: usize,
        hidden_dim: usize,
        source_dims: Vec<usize>,
    },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::NonContiguousNodeIds => {
                write!(
                    f,
                    "node ids must be contiguous 0..n (they double as array indices)"
                )
            }
            GraphError::EmptyGraph => write!(f, "graph has no nodes"),
            GraphError::InvalidOptions(msg) => write!(f, "invalid options: {msg}"),
            GraphError::UnknownNode(id) => write!(f, "connection references unknown node {id}"),
            GraphError::BackwardConnection(c) => {
                write!(
                    f,
                    "connection is not forward-only (from.node must be < to.node): {c}"
                )
            }
            GraphError::PortOutOfBounds { port, num_ports } => write!(
                f,
                "port n{}[{}] is out of bounds (node has {num_ports} ports)",
                port.node, port.index
            ),
            GraphError::DoubleWiredPort(p) => {
                write!(f, "input port n{}_i{} is wired twice", p.node, p.index)
            }
            GraphError::DoubleUsedOutput(p) => {
                write!(f, "output port n{}_o{} is used twice", p.node, p.index)
            }
            GraphError::InconsistentInputDims { node, dims } => write!(
                f,
                "node n{node} receives inputs with inconsistent dims {dims:?}"
            ),
            GraphError::OrphanDimMismatch {
                node,
                hidden_dim,
                source_dims,
            } => write!(
                f,
                "node n{node} has orphaned ports (fed by net_input, dim {hidden_dim}) but wired sources have dims {source_dims:?}"
            ),
        }
    }
}

/// The blueprint 🧬 of a gras network: pure data, no tensors.
///
/// A `Graph` answers two questions only:
///   - **who exists** — `nodes`, each declaring its input/output port counts
///   - **who feeds whom** — `connections`, each a wire from an output port
///     to an input port
///
/// It is inert by itself: [`Graph::validate`] checks it can be executed, and
/// [`GrasGraph::build`] turns it into real flodl layers. Add nodes with the
/// [`Node`] constructors (`Node::new_input`, `new_hidden`, `new_output`) —
/// they keep ids contiguous, which the rest of the code relies on.
///
/// Fields:
///   - `id` — instance id (used to name [`GrasGraph`]s uniquely)
///   - `options` — dims, RNG seed, combine op (see [`GraphOptions`])
///   - `graph_inputs` / `graph_outputs` — one string label per port, minted by
///     [`Graph::set_graph_topology`]; for rendering/debugging only — execution
///     reads the typed [`Port`]s in `connections`
///   - `rng` — the seeded RNG driving random node/wiring generation
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub id: usize,
    pub nodes: Vec<Node>,
    pub options: GraphOptions,
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<String>,
    pub connections: Vec<Connection>,
    pub rng: Rng,
}

impl Graph {
    /// Create an empty graph. Pass `None` to use the default options (seed 16,
    /// hidden_dim 8, ...). The graph starts with **no nodes** — add them with
    /// the [`Node`] constructors + `graph.nodes.push(...)`.
    pub fn new(id: usize, options: Option<GraphOptions>) -> Graph {
        // Create a new graph with the given id and options, or default options if None
        let opts = match options {
            Some(options) => options,
            None => GraphOptions::new(),
        };
        Graph {
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
    /// [`Graph::create_random_hidden_node`]).
    pub fn create_random_hidden_nodes(&mut self, num_nodes: usize) {
        // Create multiple random hidden nodes
        for _ in 0..num_nodes {
            self.create_random_hidden_node();
        }
    }

    /// Set the graph topology: mint one label per port, kept only at the
    /// graph level. These labels are the "address book" 📇 — a human-readable
    /// view of the ports (`n1_i0`, `n2_o3`, ...) used by `Display`, the
    /// ASCII renderer in `utils`, and `connection_pairs`. Execution itself
    /// reads the typed [`Port`]s directly, so this step is only needed for
    /// pretty-printing and debugging.
    ///
    /// Simple maths: a node i with `num_inputs` inputs and `num_outputs`
    /// outputs contributes exactly
    ///   num_inputs  labels to graph_inputs  -> "n{i}_i0", "n{i}_i1", ...
    ///   num_outputs labels to graph_outputs -> "n{i}_o0", "n{i}_o1", ...
    pub fn set_graph_topology(&mut self) {
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

    /// Wire the connections 🔗 between nodes based on the topology.
    ///
    /// Simple mental model:
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
    ///   - 🗑️ output ports nobody consumes are simply dropped
    ///
    /// Simple maths: with I input ports and O output ports we create at most
    /// min(I, O) connections.
    pub fn set_graph_network(&mut self) {
        self.connections.clear();

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

        let mut used = vec![false; output_ports.len()];
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
    }

    /// Human-readable connection pairs, e.g. `("n0_o0", "n1_i0")` — handy for
    /// debugging and for feeding a `layer!`-style DSL later. Just a string view
    /// of `self.connections` (same data, no duplication).
    pub fn connection_pairs(&self) -> Vec<(String, String)> {
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
    ///   - no input port is wired twice and no output port feeds two inputs
    ///     (1:1 pairing)
    ///   - feature dims are consistent: all tensors entering a node share one
    ///     dim, and orphaned ports (fed by the network input, dim =
    ///     `hidden_dim`) force every wired source of that node to `hidden_dim`
    ///
    /// `GrasGraph::build` calls this and refuses to build invalid graphs.
    pub fn validate(&self) -> Result<(), GraphError> {
        // 1. Options sanity
        if self.options.input_dim == 0 || self.options.hidden_dim == 0 {
            return Err(GraphError::InvalidOptions(
                "input_dim and hidden_dim must be > 0".to_string(),
            ));
        }
        if self.options.min_inputs_per_node > self.options.max_inputs_per_node
            || self.options.min_outputs_per_node > self.options.max_outputs_per_node
        {
            return Err(GraphError::InvalidOptions(
                "min/max inputs or outputs per node ranges are inverted".to_string(),
            ));
        }

        // 2. At least one node, with contiguous ids and sane per-node dims
        if self.nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }
        for (i, node) in self.nodes.iter().enumerate() {
            if node.id != i {
                return Err(GraphError::NonContiguousNodeIds);
            }
            if node.hidden_dim == Some(0) {
                return Err(GraphError::InvalidOptions(format!(
                    "node n{i} has hidden_dim 0"
                )));
            }
        }

        // 3. Connections: forward-only, known nodes, in-range ports, 1:1
        let mut wired_targets: Vec<Port> = Vec::new();
        let mut used_sources: Vec<Port> = Vec::new();
        for conn in &self.connections {
            if conn.from.node >= conn.to.node {
                return Err(GraphError::BackwardConnection(conn.clone()));
            }
            if conn.from.node >= self.nodes.len() || conn.to.node >= self.nodes.len() {
                return Err(GraphError::UnknownNode(conn.from.node));
            }
            let src = &self.nodes[conn.from.node];
            let dst = &self.nodes[conn.to.node];
            if conn.from.index >= src.num_outputs {
                return Err(GraphError::PortOutOfBounds {
                    port: conn.from,
                    num_ports: src.num_outputs,
                });
            }
            if conn.to.index >= dst.num_inputs {
                return Err(GraphError::PortOutOfBounds {
                    port: conn.to,
                    num_ports: dst.num_inputs,
                });
            }
            if wired_targets.contains(&conn.to) {
                return Err(GraphError::DoubleWiredPort(conn.to));
            }
            if used_sources.contains(&conn.from) {
                return Err(GraphError::DoubleUsedOutput(conn.from));
            }
            wired_targets.push(conn.to);
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
                match self.connections.iter().find(|c| c.to == target) {
                    Some(conn) => source_dims.push(out_dim(&self.nodes[conn.from.node])),
                    None => has_orphan = true,
                }
            }
            source_dims.sort_unstable();
            source_dims.dedup();
            if has_orphan
                && let Some(&d) = source_dims.first()
                && d != hidden_dim
            {
                return Err(GraphError::OrphanDimMismatch {
                    node: node.id,
                    hidden_dim,
                    source_dims,
                });
            } else if source_dims.len() > 1 {
                return Err(GraphError::InconsistentInputDims {
                    node: node.id,
                    dims: source_dims,
                });
            }
        }

        Ok(())
    }

    /// Serialize the whole blueprint (options, nodes, labels, connections) to
    /// JSON. 🗂️ See [`GraphSpec`] for the shape; the RNG is not stored.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a blueprint from JSON (see [`Graph::to_json`]).
    ///
    /// The RNG is **re-seeded from `options.seed`**, so any regeneration after
    /// loading (e.g. `set_graph_network`) is deterministic — a loaded graph
    /// wires identically to a freshly created graph with the same options.
    ///
    /// The executable module is rebuilt from the loaded blueprint with
    /// [`GrasGraph::build`] — same architecture, fresh random weights (no
    /// weights are ever serialized).
    pub fn from_json(s: &str) -> Result<Graph, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// JSON round-trip representation of a [`Graph`] — the blueprint minus the
/// RNG. `options.seed` is what makes regeneration reproducible, so the RNG is
/// rebuilt from it on load rather than stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphSpec {
    pub id: usize,
    pub options: GraphOptions,
    pub nodes: Vec<Node>,
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<String>,
    pub connections: Vec<Connection>,
}

impl From<&Graph> for GraphSpec {
    fn from(g: &Graph) -> Self {
        GraphSpec {
            id: g.id,
            options: g.options,
            nodes: g.nodes.clone(),
            graph_inputs: g.graph_inputs.clone(),
            graph_outputs: g.graph_outputs.clone(),
            connections: g.connections.clone(),
        }
    }
}

impl Serialize for Graph {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        GraphSpec::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Graph {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let spec = GraphSpec::deserialize(deserializer)?;
        Ok(Graph {
            id: spec.id,
            nodes: spec.nodes,
            options: spec.options,
            graph_inputs: spec.graph_inputs,
            graph_outputs: spec.graph_outputs,
            connections: spec.connections,
            // 🎲 Reproducibility: the RNG is re-seeded from options.seed, so
            // a loaded graph regenerates wiring identically to a fresh one.
            rng: Rng::with_seed(spec.options.seed as u64),
        })
    }
}

impl std::fmt::Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The rendering logic lives in utils (presentation concern, not core).
        write!(f, "{}", crate::utils::graph_ascii_topology(self))
    }
}

/// Compact per-node metadata captured at build time, indexed by node id.
/// Used by utils for rendering.
#[derive(Clone, Copy)]
pub(crate) struct NodeInfo {
    pub(crate) kind: NodeKind,
    pub(crate) num_inputs: usize,
    pub(crate) num_outputs: usize,
    /// Feature dim this node's layer consumes (from its sources).
    pub(crate) in_dim: usize,
    /// Feature dim this node's layer emits (its `hidden_dim` or the graph's).
    pub(crate) out_dim: usize,
    /// Activation applied after the node's linear transform.
    pub(crate) activation: Activation,
}

/// Precompute the wiring table: for each node (by id), one entry per input
/// port — the source port if wired, `None` if orphaned (fed by net_input).
/// Built once at compile/build time so the forward pass resolves each port
/// in O(1) instead of scanning the connection list.
fn build_node_sources(connections: &[Connection], num_inputs: &[usize]) -> Vec<Vec<Option<Port>>> {
    // (to → from) lookup table
    let input_map: HashMap<Port, Port> = connections.iter().map(|c| (c.to, c.from)).collect();
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
                        .copied()
                })
                .collect()
        })
        .collect()
}

/// A self-contained flodl module that executes a gras graph.
pub struct GrasGraph {
    /// Projects the raw network input (input_dim → hidden_dim) once; feeds
    /// every orphaned input port. 🚪
    input_proj: Linear,
    /// Unique instance name 🏷️ — flodl uses `Module::name` as a node-id
    /// prefix when a module is embedded in a bigger graph, so every GrasGraph
    /// in a population must have a distinct one. Built from the graph id plus
    /// a fastrand suffix (no extra crates needed).
    name: String,
    /// Input feature dimension (kept for pretty printing in utils).
    pub(crate) input_dim: usize,
    /// Graph-level hidden dimension (kept for pretty printing in utils).
    pub(crate) hidden_dim: usize,
    /// One linear layer per node, indexed by node id. Each node's layer maps
    /// its combined input dim → its own output dim. This is the actual
    /// "compute" of each node. 🧮
    pub(crate) layers: Vec<Linear>,
    /// The wires between nodes, copied from the Graph. 🔗
    pub(crate) connections: Vec<Connection>,
    /// Per-node metadata (kind, port counts, dims, activation), indexed by
    /// node id.
    pub(crate) node_info: Vec<NodeInfo>,
    /// Precomputed wiring: for each node, one entry per input port — the
    /// source port if wired, `None` if orphaned (fed by net_input). Built
    /// once here so the forward pass never scans the connection list.
    node_sources: Vec<Vec<Option<Port>>>,
    /// Which node's output is the graph output. 🏁
    pub(crate) output_node: usize,
    /// How multiple incoming tensors into a node are combined.
    combine_op: CombineOp,
}

impl GrasGraph {
    /// Compile a validated blueprint into an executable flodl module. 🏭
    ///
    /// What happens, step by step:
    ///   1. [`Graph::validate`] — refuse broken graphs before spending any work
    ///   2. `input_proj` — one shared `Linear(input_dim → hidden_dim)`: the raw
    ///      network input passes through it once, and every orphaned input
    ///      port reads this tensor
    ///   3. per node — build the wiring table (`node_sources`: which source
    ///      feeds each input port, or orphan), derive the node's `in_dim` from
    ///      its sources, and create one `Linear(in_dim → out_dim)` where
    ///      `out_dim` is the node's own `hidden_dim` (or the graph's)
    ///   4. pick the output node — highest-id `Output` node, else the last node
    ///
    /// Result: `num_nodes + 1` linears (weight + bias each), so
    /// `2 * (num_nodes + 1)` learnable parameters total.
    pub fn build(graph: &Graph, device: Device) -> flodl::tensor::Result<Self> {
        // 🛡️ Random graphs must validate before execution.
        graph
            .validate()
            .map_err(|e| TensorError::new(&format!("invalid graph: {e}")))?;

        let opts = &graph.options;
        let input_dim = opts.input_dim;
        let hidden_dim = opts.hidden_dim;

        // 🚪 Network input projection: input_dim → hidden_dim
        let input_proj = Linear::on_device(input_dim as i64, hidden_dim as i64, device)?;

        // Node ids are contiguous (0, 1, 2, ...) — validated above — so we
        // can index everything by id.
        let num_ids = graph.nodes.len();
        let out_dim = |n: &Node| n.hidden_dim.unwrap_or(hidden_dim);

        // Precompute the wiring table once (per input port: which source, or
        // orphan) so the forward pass never scans the connection list.
        let node_inputs: Vec<usize> = graph.nodes.iter().map(|n| n.num_inputs).collect();
        let node_sources = build_node_sources(&graph.connections, &node_inputs);

        let mut node_info = vec![
            NodeInfo {
                kind: NodeKind::Hidden,
                num_inputs: 0,
                num_outputs: 0,
                in_dim: hidden_dim,
                out_dim: hidden_dim,
                activation: Activation::Identity,
            };
            num_ids
        ];
        let mut layers = Vec::with_capacity(num_ids);

        for node in &graph.nodes {
            // Which tensor feeds each input port: Some(source port) if wired,
            // None if orphaned (fed by net_input).
            let sources = &node_sources[node.id];

            // Input dim = the (validated-identical) dim of wired sources, or
            // hidden_dim when ports are absent/orphaned (net_input). `.max()`
            // is safe: validation guarantees all source dims are equal.
            let in_dim = sources
                .iter()
                .flatten()
                .map(|p| out_dim(&graph.nodes[p.node]))
                .max()
                .unwrap_or(hidden_dim);
            let out_dim = out_dim(node);

            node_info[node.id] = NodeInfo {
                kind: node.kind,
                num_inputs: node.num_inputs,
                num_outputs: node.num_outputs,
                in_dim,
                out_dim,
                activation: node.activation,
            };
            // 🧮 One Linear per node: in_dim → out_dim
            layers.push(Linear::on_device(in_dim as i64, out_dim as i64, device)?);
        }

        // 🏷️ Unique instance name: graph id + fastrand suffix (global RNG is
        // auto-seeded, so distinct instances get distinct names).
        let name = format!("gras_graph_{}_{}", graph.id, fastrand::u64(..));

        // 🏁 Graph output = the highest-id Output node if any, otherwise the
        // highest-id node overall (a graph with only hidden nodes ends at the
        // last one).
        let output_node = graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Output)
            .map(|n| n.id)
            .max()
            .or_else(|| graph.nodes.iter().map(|n| n.id).max())
            .unwrap_or(0);

        Ok(GrasGraph {
            input_proj,
            name,
            input_dim,
            hidden_dim,
            layers,
            connections: graph.connections.clone(),
            node_info,
            node_sources,
            output_node,
            combine_op: opts.combine_op,
        })
    }
}

impl std::fmt::Display for GrasGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The rendering logic lives in utils (presentation concern, not core).
        write!(f, "{}", crate::utils::gras_graph_ascii_net(self))
    }
}

impl Module for GrasGraph {
    /// Execute the graph on an input tensor. 📤
    ///
    /// Worked example for a tiny graph:
    ///   nodes:       n0 (input,  0 in / 2 out), n1 (hidden, 2 in / 1 out)
    ///   connections: n0_o0 -> n1_i0,  n0_o1 -> n1_i1
    ///   combine_op:  Add
    ///
    /// ```text
    ///   x ──▶ input_proj ──▶ n0_out ──┐
    ///                                 ▼
    ///                            combined ──▶ layers[1] ──▶ act ──▶ n1_out = y
    ///
    ///   1. net_input = input_proj(x)              // [batch, hidden_dim]
    ///   2. n0 has no input ports
    ///      => n0_out = act(layers[0](net_input))  // [batch, hidden_dim]
    ///   3. n1's inputs: n1_i0 <- n0_out, n1_i1 <- n0_out
    ///      combined = n0_out + n0_out             // Add: a + b
    ///      n1_out = act(layers[1](combined))      // [batch, hidden_dim]
    ///   4. return n1_out                          // 🏁 output_node = n1
    /// ```
    ///
    /// The loop below is the "unrolling": every node is a linear layer over
    /// its combined inputs, so ascending node id (a valid topological order,
    /// since edges only go forward) is all we need.
    fn forward(&self, input: &Variable) -> flodl::tensor::Result<Variable> {
        // 🚪 1. Project the network input once; it feeds every orphaned input
        //    port (and every input node).
        let net_input = self.input_proj.forward(input)?;

        // Output tensor per node — all of a node's output ports emit the same
        // tensor, so we store one entry per node id. This map is the runtime
        // analogue of the wiring table: it's how an arbitrary DAG "flows" —
        // each node only ever reads sources that were already computed.
        let mut node_outputs: HashMap<usize, Variable> = HashMap::new();

        // Connections only go forward (from.node < to.node), so ascending
        // node id order is a valid topological execution order ✅ — every
        // source is already computed when we read it.
        for node_id in 0..self.layers.len() {
            // 2. Gather: resolve each input port to its tensor — a wired
            //    source computed earlier, or the network input if orphaned.
            let mut combined: Option<Variable> = None;
            let mut num_sources = 0usize;
            for source in &self.node_sources[node_id] {
                let t = match source {
                    Some(p) => &node_outputs[&p.node],
                    None => &net_input,
                };
                combined = Some(match combined {
                    None => t.clone(),
                    Some(prev) => prev.add(t)?, // ➕ accumulate (sum)
                });
                num_sources += 1;
            }

            // 3. Combine the gathered tensors per the graph's CombineOp.
            let combined = match combined {
                // Node with no input ports (e.g. an input node): feed it the
                // network input directly.
                None => net_input.clone(),
                Some(c) if self.combine_op == CombineOp::Mean && num_sources > 1 => {
                    c.mul_scalar(1.0 / num_sources as f64)? // ➗ average: (a+b+c)/3
                }
                Some(c) => c,
            };

            // 4. Transform + activate: run the node's layer, apply its
            //    activation, store the output tensor.
            let out = self.layers[node_id].forward(&combined)?;
            let out = self.node_info[node_id].activation.apply(&out)?;
            node_outputs.insert(node_id, out);
        }

        // 🏁 Return the output node's tensor (net_input as a fallback for an
        // empty graph).
        Ok(node_outputs
            .get(&self.output_node)
            .cloned()
            .unwrap_or(net_input))
    }

    /// All learnable parameters: the input projection plus every node layer.
    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.input_proj.parameters();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    /// Unique per-instance name, e.g. `"gras_graph_0_12345"` — never the
    /// shared constant, so multiple GrasGraphs can coexist in one flodl graph.
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flodl::{DType, Tensor, TensorOptions};

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

    // ── topology (Graph) ────────────────────────────────────────────────────

    #[test]
    fn test_new_without_options() {
        let graph = Graph::new(1, None);
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_num_nodes, 2);
        assert_eq!(graph.options.max_num_nodes, 5);
        // assert_eq!(graph.graph_topology.node_topologies.len(), 0);
    }

    #[test]
    fn test_new_with_options() {
        let opts = GraphOptions {
            seed: 123,
            min_num_nodes: 3,
            max_num_nodes: 10,
            min_inputs_per_node: 1,
            max_inputs_per_node: 5,
            min_outputs_per_node: 1,
            max_outputs_per_node: 5,
            num_outputs_net: 1,
            input_dim: 4,
            hidden_dim: 16,
            combine_op: CombineOp::Mean,
        };
        let graph = Graph::new(1, Some(opts));
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_num_nodes, 3);
        assert_eq!(graph.options.max_num_nodes, 10);
        assert_eq!(graph.options.input_dim, 4);
        assert_eq!(graph.options.hidden_dim, 16);
        assert_eq!(graph.options.combine_op, CombineOp::Mean);
        // assert_eq!(graph.graph_topology.node_topologies.len(), 0);
    }

    #[test]
    fn test_create_random_hidden_node() {
        let mut graph = Graph::new(1, None);
        graph.create_random_hidden_node();
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_set_graph_topology() {
        let mut graph = Graph::new(1, None);
        graph.create_random_hidden_node();
        graph.create_random_hidden_node();
        graph.create_random_hidden_node();
        graph.set_graph_topology();

        // One label per port, matching each node's declared inputs/outputs
        let total_inputs: usize = graph.nodes.iter().map(|n| n.num_inputs).sum();
        let total_outputs: usize = graph.nodes.iter().map(|n| n.num_outputs).sum();
        assert_eq!(graph.graph_inputs.len(), total_inputs);
        assert_eq!(graph.graph_outputs.len(), total_outputs);
    }

    #[test]
    fn test_set_graph_topology_labels() {
        let mut graph = Graph::new(1, None);
        // Fixed nodes: input with 2 outputs, hidden with 3 inputs / 2 outputs
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.set_graph_topology();

        assert_eq!(graph.graph_inputs, vec!["n1_i0", "n1_i1", "n1_i2"]);
        assert_eq!(
            graph.graph_outputs,
            vec!["n0_o0", "n0_o1", "n1_o0", "n1_o1"]
        );
    }

    #[test]
    fn test_set_graph_network() {
        let mut graph = Graph::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_hidden(2, 2, 1));
        graph.set_graph_network();

        let num_input_ports: usize = graph.nodes.iter().map(|n| n.num_inputs).sum();
        let num_output_ports: usize = graph.nodes.iter().map(|n| n.num_outputs).sum();

        // 1:1 pairing: no port may be used more than once, and the total
        // number of connections is bounded by the smaller port count
        let mut seen_to: Vec<Port> = Vec::new();
        let mut seen_from: Vec<Port> = Vec::new();
        for conn in &graph.connections {
            assert!(
                conn.from.node < conn.to.node,
                "connection must go strictly forward: {conn}"
            );
            assert!(
                !seen_to.contains(&conn.to),
                "input port {} wired twice",
                conn.to_label()
            );
            assert!(
                !seen_from.contains(&conn.from),
                "output port {} used twice",
                conn.from_label()
            );
            seen_to.push(conn.to);
            seen_from.push(conn.from);
        }
        assert!(graph.connections.len() <= num_input_ports.min(num_output_ports));
        assert!(!graph.connections.is_empty());

        // String pairs match the typed connections
        assert_eq!(graph.connections.len(), graph.connection_pairs().len());
    }

    // ── validation ──────────────────────────────────────────────────────────

    #[test]
    fn test_validate_ok_on_generated_graph() {
        let mut graph = Graph::new(1, None);
        graph.create_random_hidden_nodes(5);
        graph.set_graph_topology();
        graph.set_graph_network();
        assert_eq!(graph.validate(), Ok(()));
    }

    #[test]
    fn test_validate_rejects_empty_graph() {
        let graph = Graph::new(1, None);
        assert_eq!(graph.validate(), Err(GraphError::EmptyGraph));
    }

    #[test]
    fn test_validate_rejects_non_contiguous_ids() {
        let mut graph = Graph::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(2, 1, 1)); // id 1 skipped
        assert_eq!(graph.validate(), Err(GraphError::NonContiguousNodeIds));
    }

    #[test]
    fn test_validate_rejects_backward_connection() {
        let mut graph = Graph::new(1, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 0, index: 0 },
        });
        assert!(matches!(
            graph.validate(),
            Err(GraphError::BackwardConnection(_))
        ));
    }

    #[test]
    fn test_validate_rejects_unknown_node() {
        let mut graph = Graph::new(1, None);
        graph.nodes.push(Node::new_input(0, 1));
        // Forward-looking (1 < 2) but both ids are past the single node we have.
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        assert!(matches!(graph.validate(), Err(GraphError::UnknownNode(_))));
    }

    #[test]
    fn test_validate_rejects_port_out_of_bounds() {
        let mut graph = Graph::new(1, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 3 }, // n1 has only 1 input port
        });
        assert!(matches!(
            graph.validate(),
            Err(GraphError::PortOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_validate_rejects_double_wired_port() {
        let mut graph = Graph::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 0, index: 1 },
            to: Port { node: 1, index: 0 }, // same input port wired again
        });
        assert!(matches!(
            graph.validate(),
            Err(GraphError::DoubleWiredPort(_))
        ));
    }

    #[test]
    fn test_validate_rejects_double_used_output() {
        let mut graph = Graph::new(1, None);
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
            Err(GraphError::DoubleUsedOutput(_))
        ));
    }

    #[test]
    fn test_validate_rejects_inconsistent_input_dims() {
        let mut graph = Graph::new(1, None);
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
            Err(GraphError::InconsistentInputDims { .. })
        ));
    }

    #[test]
    fn test_validate_rejects_orphan_dim_mismatch() {
        let mut graph = Graph::new(1, None);
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
            Err(GraphError::OrphanDimMismatch { .. })
        ));
    }

    // ── execution (GrasGraph) ───────────────────────────────────────────────

    #[test]
    fn test_gras_graph_build_and_forward() {
        let mut graph = Graph::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.set_graph_topology();
        graph.set_graph_network();
        assert!(!graph.connections.is_empty());

        let module = GrasGraph::build(&graph, Device::CPU).unwrap();

        let batch = 4i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.hidden_dim as i64]);

        // One Linear (weight + bias) per node, plus the input projection
        assert_eq!(module.parameters().len(), (graph.nodes.len() + 1) * 2);
    }

    #[test]
    fn test_gras_graph_forward_mean() {
        let mut graph = Graph::new(0, None);
        graph.options.combine_op = CombineOp::Mean;
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.set_graph_network();

        let module = GrasGraph::build(&graph, Device::CPU).unwrap();
        let batch = 2i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.hidden_dim as i64]);
    }

    #[test]
    fn test_gras_graph_forward_orphans() {
        // n0 feeds n2's first input; n2's second input is orphaned and must
        // be fed by net_input.
        let mut graph = Graph::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 1 },
        });
        // n2's first input (index 0) has no wire -> orphaned.
        graph.connections.remove(0);

        let module = GrasGraph::build(&graph, Device::CPU).unwrap();
        let batch = 2i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.hidden_dim as i64]);
    }

    #[test]
    fn test_gras_graph_forward_activation_and_node_dim() {
        // n1 widens 8 -> 32 and applies ReLU; n2 narrows back 32 -> 8.
        let mut graph = Graph::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        let mut wide = Node::new_hidden(1, 1, 1);
        wide.hidden_dim = Some(32);
        wide.activation = Activation::ReLU;
        graph.nodes.push(wide);
        graph.nodes.push(Node::new_output(2, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });

        let module = GrasGraph::build(&graph, Device::CPU).unwrap();
        // Per-node dims are captured for rendering
        assert_eq!(module.node_info[0].out_dim, 8);
        assert_eq!(module.node_info[1].in_dim, 8);
        assert_eq!(module.node_info[1].out_dim, 32);
        assert_eq!(module.node_info[1].activation, Activation::ReLU);
        assert_eq!(module.node_info[2].in_dim, 32);
        assert_eq!(module.node_info[2].out_dim, 8);

        let batch = 3i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, 8]);
    }

    #[test]
    fn test_gras_graph_build_rejects_invalid_graph() {
        let mut graph = Graph::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 0, index: 0 }, // backward!
        });
        assert!(GrasGraph::build(&graph, Device::CPU).is_err());
    }

    #[test]
    fn test_unique_names() {
        // Two modules built from the same graph must have distinct names, so
        // flodl node-id prefixes never collide.
        let mut graph = Graph::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_output(1, 1, 1));
        graph.set_graph_network();

        let a = GrasGraph::build(&graph, Device::CPU).unwrap();
        let b = GrasGraph::build(&graph, Device::CPU).unwrap();
        assert_ne!(a.name(), b.name());
        assert!(a.name().starts_with("gras_graph_0_"));
    }

    // ── serialization ───────────────────────────────────────────────────────

    #[test]
    fn test_graph_json_roundtrip() {
        let mut graph = Graph::new(7, None);
        graph.create_random_hidden_nodes(4);
        graph.set_graph_topology();
        graph.set_graph_network();
        let json = graph.to_json().unwrap();
        let loaded: Graph = Graph::from_json(&json).unwrap();
        assert_eq!(loaded.id, graph.id);
        assert_eq!(loaded.options, graph.options);
        assert_eq!(loaded.nodes, graph.nodes);
        assert_eq!(loaded.graph_inputs, graph.graph_inputs);
        assert_eq!(loaded.graph_outputs, graph.graph_outputs);
        assert_eq!(loaded.connections, graph.connections);
    }

    #[test]
    fn test_graph_json_rewiring_is_deterministic() {
        // The RNG is re-seeded from options.seed on load, so a loaded graph
        // wires identically to a fresh graph with the same options.
        let mut original = Graph::new(3, None);
        original.nodes.push(Node::new_input(0, 2));
        original.nodes.push(Node::new_hidden(1, 3, 2));
        original.nodes.push(Node::new_output(2, 2, 1));

        let json = original.to_json().unwrap();
        let mut loaded = Graph::from_json(&json).unwrap();
        let mut fresh = Graph::new(3, None);
        fresh.nodes = original.nodes.clone();

        original.set_graph_network();
        loaded.set_graph_network();
        fresh.set_graph_network();
        assert_eq!(loaded.connections, fresh.connections);
        assert_eq!(loaded.connections, original.connections);
    }
    #[test]
    fn test_graph_json_rebuilds_same_architecture() {
        // The blueprint is the single source of truth: saving/loading it and
        // re-building yields the same architecture (fresh random weights —
        // weights are never serialized).
        let mut graph = Graph::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        let mut wide = Node::new_hidden(1, 3, 2);
        wide.hidden_dim = Some(32);
        wide.activation = Activation::GeLU;
        graph.nodes.push(wide);
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.set_graph_network();

        let module = GrasGraph::build(&graph, Device::CPU).unwrap();
        let json = graph.to_json().unwrap();
        let reloaded_graph = Graph::from_json(&json).unwrap();
        let rebuilt = GrasGraph::build(&reloaded_graph, Device::CPU).unwrap();

        // Same output node, same per-node dims/activations, same param count.
        assert_eq!(rebuilt.output_node, module.output_node);
        for (a, b) in rebuilt.node_info.iter().zip(module.node_info.iter()) {
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.num_inputs, b.num_inputs);
            assert_eq!(a.num_outputs, b.num_outputs);
            assert_eq!(a.in_dim, b.in_dim);
            assert_eq!(a.out_dim, b.out_dim);
            assert_eq!(a.activation, b.activation);
        }
        assert_eq!(rebuilt.parameters().len(), module.parameters().len());
        let input = rand_input(2, graph.options.input_dim);
        assert_eq!(
            rebuilt.forward(&input).unwrap().shape(),
            module.forward(&input).unwrap().shape()
        );
    }
}

// //     fn add_node(mut self, node: Node) -> Self {
// //         node.validate_node();
// //         self.nodes.push(node);
// //         self
// //     }

// //     fn create_from_nodes(nodes: Vec<Node>) -> Self {
// //         for node in &nodes {
// //             node.validate_node();
// //         }
// //         Graph {
// //             nodes,
// //             options: GraphOptions::new(),
// //         }
// //     }

// //     fn create_random_graph(options: Option<GraphOptions>) -> () {
// //         let graph = Graph::new(options);
// //         let mut rng = Rng::with_seed(graph.options.seed as u64);
// //         let num_nodes = rng.choice(
// //             graph.options.min_num_nodes .. graph.options.max_num_nodes
// //         ).unwrap();

// //         // Create one required input node
// //         let mut nodes = vec![];
// //         nodes.push(Node::new(0, NodeKind::Input, None));

// //         for index in 0..num_nodes {
// //             let node = Node::new(
// //                 index,
// //                 NodeKind::Hidden,
// //                 Some(vec![index - 1]),
// //             )
// //         }

// //         (())
// //     }

// //     // fn validate_graph(mut self) -> Self {
// //     // }
// // }

//     //     // proptest! {
//     //     //     // Test that any valid options (min_num_nodes >= 3, max_num_nodes >= min_num_nodes) successfully create a Graph
//     //     //     #[test]
//     //     //     fn test_prop_valid_graph_options(
//     //     //         min_num_nodes in 3..100_usize,
//     //     //         extra in 0..50_usize
//     //     //     ) {
//     //     //         let max_num_nodes = min_num_nodes + extra;
//     //     //         println!{"Testing with min_num_nodes = {}, max_num_nodes = {}", min_num_nodes, max_num_nodes}; // <-- Add this
//     //     //         let options = GraphOptions { min_num_nodes, max_num_nodes };

//     //     //         let graph = Graph::new(Some(options.clone()));

//     //     //         assert_eq!(graph.options.min_num_nodes, min_num_nodes);
//     //     //         assert_eq!(graph.options.max_num_nodes, max_num_nodes);
//     //     //         assert_eq!(graph.nodes.len(), 0);
//     //     //     }
//     //     // }

//     //     #[test]
//     //     fn test_new_with_options() {
//     //         let options = GraphOptions {
//     //             min_num_nodes: 4,
//     //             max_num_nodes: 6,
//     //         };
//     //         let graph = Graph::new(Some(options.clone()));
//     //         assert_eq!(graph.nodes.len(), 0);
//     //         assert_eq!(graph.options.min_num_nodes, options.min_num_nodes);
//     //         assert_eq!(graph.options.max_num_nodes, options.max_num_nodes);
//     //     }

//     //     #[test]
//     //     fn test_add_node() {
//     //         let graph = Graph::new(None);
//     //         let node = Node {
//     //             id: 0,
//     //             kind: NodeKind::Input,
//     //             inputs: None,
//     //         };
//     //         let graph = graph.add_node(node.clone());
//     //         assert_eq!(graph.nodes.len(), 1);
//     //         assert_eq!(graph.nodes[0], node);
//     //     }

//     //     #[test]
//     //     fn test_create_from_nodes() {
//     //         let nodes = vec![
//     //             Node {
//     //                 id: 0,
//     //                 kind: NodeKind::Input,
//     //                 inputs: None,
//     //             },
//     //             Node {
//     //                 id: 1,
//     //                 kind: NodeKind::Hidden,
//     //                 inputs: Some(vec![0]),
//     //             },
//     //             Node {
//     //                 id: 2,
//     //                 kind: NodeKind::Output,
//     //                 inputs: Some(vec![1]),
//     //             },
//     //         ];
//     //         let graph = Graph::create_from_nodes(nodes.clone());
//     //         assert_eq!(graph.nodes.len(), 3);
//     //         assert_eq!(graph.nodes[0].id, 0);
//     //         assert_eq!(graph.nodes[1].kind, NodeKind::Hidden);
//     //         assert_eq!(graph.nodes[2].inputs, Some(vec![1]));
//     //     }
