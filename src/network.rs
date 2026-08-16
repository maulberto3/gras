//! Execution engine of a gras graph — the flodl [`Module`]. 🏭
//!
//! The blueprint lives in [`crate::topology`] (pure data, no tensors); this
//! file compiles it into real flodl layers ([`Network::build`]) and
//! executes it ([`Network::forward`]).
//!
//! # Why the forward pass is a loop, not generated code
//!
//! Every node is a linear layer over its combined inputs, so a runtime loop
//! *is* the "unrolling" — a compile-time `layers!` macro could only follow
//! connections written literally in source, never a graph the RNG produced at
//! runtime (that would be like `println!` printing a random string). The loop
//! stays readable because [`Network::build`] precomputes the wiring once
//! (per input port: which sources feed it, or "orphan") and `forward` just
//! walks that table.
//!
//! # Why execution stays simple
//!
//! Everything that makes the forward pass trivial is an **invariant the code
//! enforces rather than computes** (all checked by
//! [`Topology::validate`](crate::topology::Topology::validate)):
//!
//! - node ids are contiguous `0..n` and double as array indices into
//!   `Network.layers`
//! - edges only go forward, so ascending node id *is* a topological order —
//!   no cycle detection at runtime
//! - output ports feed at most one input; input ports may hold several wires
//!   (de-orphaning stacks extra sources; the node combines them with
//!   Add/Mean)
//! - all output ports of a node emit the same tensor, so fan-out is free
//! - orphaned input ports are fed the network input (via `input_proj`)

use std::collections::HashMap;

use flodl::nn::{Linear, Module, Parameter};
use flodl::tensor::TensorError;
use flodl::{Device, Variable};

use crate::node::{Activation, Node, NodeKind};
use crate::topology::{CombineOp, Connection, Port, Topology};

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
/// port — a *list* of source ports feeding it (empty = orphaned, fed by
/// net_input). A port can hold several wires because de-orphaning stacks
/// extra sources into already-wired ports; the node combines them all.
/// Built once at compile/build time so the forward pass resolves each port
/// in O(1) instead of scanning the connection list.
fn build_node_sources(connections: &[Connection], num_inputs: &[usize]) -> Vec<Vec<Vec<Port>>> {
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

/// A self-contained flodl module that executes a gras graph.
pub struct Network {
    /// Projects the raw network input (input_dim → hidden_dim) once; feeds
    /// every orphaned input port. 🚪
    input_proj: Linear,
    /// Unique instance name 🏷️ — flodl uses `Module::name` as a node-id
    /// prefix when a module is embedded in a bigger graph, so every Network
    /// in a population must have a distinct one. Built from the graph id plus
    /// a fastrand suffix (no extra crates needed).
    name: String,
    /// Input feature dimension (kept for pretty printing in utils).
    pub(crate) input_dim: usize,
    /// Topology-level hidden dimension (kept for pretty printing in utils).
    pub(crate) hidden_dim: usize,
    /// One linear layer per node, indexed by node id. Each node's layer maps
    /// its combined input dim → its own output dim. This is the actual
    /// "compute" of each node. 🧮
    pub(crate) layers: Vec<Linear>,
    /// The wires between nodes, copied from the Topology. 🔗
    pub(crate) connections: Vec<Connection>,
    /// Per-node metadata (kind, port counts, dims, activation), indexed by
    /// node id.
    pub(crate) node_info: Vec<NodeInfo>,
    /// Precomputed wiring: for each node, one entry per input port — the
    /// *list* of source ports feeding it (empty = orphaned, fed by
    /// net_input). A port can hold several wires (de-orphaning stacks extra
    /// sources); the node combines them all. Built once here so the forward
    /// pass never scans the connection list.
    node_sources: Vec<Vec<Vec<Port>>>,
    /// Which node's output is the network output. 🏁
    pub(crate) output_node: usize,
    /// How multiple incoming tensors into a node are combined.
    combine_op: CombineOp,
}

impl Network {
    /// Compile a validated blueprint into an executable flodl module. 🏭
    ///
    /// What happens, step by step:
    ///   1. [`Topology::validate`](crate::topology::Topology::validate) — refuse
    ///      broken graphs before spending any work
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
    pub fn build(graph: &Topology, device: Device) -> flodl::tensor::Result<Self> {
        // 🛡️ Random graphs must validate before execution.
        graph
            .validate()
            .map_err(|e| TensorError::new(&format!("invalid graph: {e}")))?;

        // Some options are needed to build the network, so we capture them here.
        let opts = &graph.options;
        let input_dim = opts.input_dim;
        let hidden_dim = opts.hidden_dim;

        // 🚪 Network input projection: input_dim → hidden_dim
        let input_proj = Linear::on_device(input_dim as i64, hidden_dim as i64, device)?;

        // Node ids are contiguous (0, 1, 2, ...) — validated above — so we
        // can index everything by id.
        let num_ids = graph.nodes.len();

        // In simple words, "given a node, what is the feature dimension it emits?"
        let out_dim = |n: &Node| n.hidden_dim.unwrap_or(hidden_dim);

        // Precompute the wiring table once (per input port: which sources,
        // or orphan → empty list) so the forward pass never scans the
        // connection list.
        let node_inputs: Vec<usize> = graph.nodes.iter().map(|n| n.num_inputs).collect();

        // This just helps with rendering: for each node, for each input port, which source ports feed it (or empty if orphaned).
        // A port can hold several wires (de-orphaning stacks extra sources); the node combines them all.
        let node_sources = build_node_sources(&graph.connections, &node_inputs);

        // Useful for rendering: per-node metadata (kind, port counts, dims, activation),
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

        // Build the per-node layers: one Linear per node.
        let mut layers = Vec::with_capacity(num_ids);

        // For example, if topology says "node 1 has 2 inputs", we need to look at the
        // sources of those inputs to determine the in_dim for node 1.
        // The in_dim is derived from the sources of the node's input ports.
        // If a node has multiple input ports, we need to ensure that all sources
        // feeding those ports have the same output dimension.
        // If they don't, it would be an invalid graph. The in_dim for a node
        // is determined by the output dimensions of its source nodes.
        for node in &graph.nodes {
            // Which tensors feed each input port: a list of source ports, or
            // empty if orphaned (fed by net_input). A port may hold several
            // wires (de-orphaning); the node combines them all.
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

            // Why overwrite the node_info entry instead of building it in one go? Because
            // the node's in_dim is derived from its sources, which we only know after
            // building the wiring table. So we fill in the rest of the info first,
            // then overwrite the in_dim once we compute it.
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
        let name = format!("network_{}_{}", graph.id, fastrand::u64(..));

        // 🏁 Topology output: the highest-id Output node if any, otherwise the
        // last node overall.
        let output_node = graph.output_node_id();

        Ok(Network {
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

impl Module for Network {
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
            // 2. Gather: resolve every incoming tensor — each input port
            //    contributes its list of sources (computed earlier), or the
            //    network input if the port is orphaned (empty list). A port
            //    can hold several wires (de-orphaning), so we sum them all.
            let mut combined: Option<Variable> = None;
            let mut num_sources = 0usize;
            for sources in &self.node_sources[node_id] {
                if sources.is_empty() {
                    // Orphaned port → fed by the network input.
                    combined = Some(match combined {
                        None => net_input.clone(),
                        Some(prev) => prev.add(&net_input)?, // ➕ accumulate
                    });
                    num_sources += 1;
                } else {
                    for p in sources {
                        let t = &node_outputs[&p.node];
                        combined = Some(match combined {
                            None => t.clone(),
                            Some(prev) => prev.add(t)?, // ➕ accumulate (sum)
                        });
                        num_sources += 1;
                    }
                }
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

    /// Unique per-instance name, e.g. `"network_0_12345"` — never the
    /// shared constant, so multiple Networks can coexist in one flodl graph.
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

    // ── execution (Network) ───────────────────────────────────────────────

    #[test]
    fn test_network_build_and_forward() {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.set_topology();
        graph.set_network();
        assert!(!graph.connections.is_empty());

        let module = Network::build(&graph, Device::CPU).unwrap();

        let batch = 4i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.hidden_dim as i64]);

        // One Linear (weight + bias) per node, plus the input projection
        assert_eq!(module.parameters().len(), (graph.nodes.len() + 1) * 2);
    }

    #[test]
    fn test_network_forward_mean() {
        let mut graph = Topology::new(0, None);
        graph.options.combine_op = CombineOp::Mean;
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.set_network();

        let module = Network::build(&graph, Device::CPU).unwrap();
        let batch = 2i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.hidden_dim as i64]);
    }

    #[test]
    fn test_network_forward_orphans() {
        // n0 feeds n2's first input; n2's second input is orphaned and must
        // be fed by net_input.
        let mut graph = Topology::new(0, None);
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

        let module = Network::build(&graph, Device::CPU).unwrap();
        let batch = 2i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.hidden_dim as i64]);
    }

    #[test]
    fn test_network_forward_activation_and_node_dim() {
        // n1 widens 8 -> 32 and applies ReLU; n2 narrows back 32 -> 8.
        let mut graph = Topology::new(0, None);
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

        let module = Network::build(&graph, Device::CPU).unwrap();
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
    fn test_network_build_rejects_invalid_graph() {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 0, index: 0 }, // backward!
        });
        assert!(Network::build(&graph, Device::CPU).is_err());
    }

    #[test]
    fn test_unique_names() {
        // Two modules built from the same graph must have distinct names, so
        // flodl node-id prefixes never collide.
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_output(1, 1, 1));
        graph.set_network();

        let a = Network::build(&graph, Device::CPU).unwrap();
        let b = Network::build(&graph, Device::CPU).unwrap();
        assert_ne!(a.name(), b.name());
        assert!(a.name().starts_with("network_0_"));
    }
}
