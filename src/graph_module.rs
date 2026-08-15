//! Maps a gras [`Graph`] topology/network onto a self-contained [`flodl`]
//! module with an explicit forward pass.
//!
//! Pipeline overview 🪜:
//!   1. [`Graph`] defines nodes + their ports (topology)
//!   2. [`Graph::set_graph_network`] wires output ports → input ports
//!   3. [`GrasGraph::build`] turns that into one flodl `Linear` per node
//!   4. [`GrasGraph::forward`] executes the graph tensor by tensor
//!
//! ```text
//!   net ──▶ input_proj ──▶ n0 ──▶ n1 ──▶ n2 ──▶ n3 ──▶ y
//!                             │         ▲
//!                             └─────────┘   (extra wire: n1 feeds n2 directly)
//! ```
//!
//! Because the forward pass is explicit, arbitrary DAG wiring (e.g. a
//! first-layer output feeding the last layer) resolves naturally: an extra
//! input to a node is just another source that gets combined.

use std::collections::HashMap;

use flodl::nn::{Linear, Module, Parameter};
use flodl::{Device, Variable};

use crate::graph::{CombineOp, Connection, Graph, Port};
use crate::node::NodeKind;

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
    /// Internal feature dimension shared by every node (kept for pretty printing).
    pub(crate) hidden_dim: usize,
    /// One linear layer per node (hidden_dim → hidden_dim), indexed by node
    /// id. This is the actual "compute" of each node. 🧮
    pub(crate) layers: Vec<Linear>,
    /// The wires between nodes, copied from the Graph. 🔗
    pub(crate) connections: Vec<Connection>,
    /// Per-node metadata (kind, port counts), indexed by node id.
    pub(crate) node_info: Vec<NodeInfo>,
    /// Which node's output is the graph output. 🏁
    pub(crate) output_node: usize,
    /// How multiple incoming tensors into a node are combined.
    combine_op: CombineOp,
}

/// Compact per-node metadata captured at build time, indexed by node id.
/// Visible to the crate so utils can render it.
#[derive(Clone, Copy)]
pub(crate) struct NodeInfo {
    pub(crate) kind: NodeKind,
    pub(crate) num_inputs: usize,
    pub(crate) num_outputs: usize,
}

impl GrasGraph {
    /// Build a flodl module from a gras graph.
    ///
    /// One `Linear` is created per node plus one shared input projection.
    /// Simple maths: `num_nodes + 1` linears, each with weight + bias, so
    /// `2 * (num_nodes + 1)` parameters total.
    pub fn build(graph: &Graph, device: Device) -> flodl::tensor::Result<Self> {
        let opts = &graph.options;
        let input_dim = opts.input_dim;
        let hidden_dim = opts.hidden_dim;

        // 🚪 Network input projection: input_dim → hidden_dim
        let input_proj = Linear::on_device(input_dim as i64, hidden_dim as i64, device)?;

        // Node ids are contiguous (0, 1, 2, ...), so we can index everything
        // by id. num_ids = (max id) + 1.
        let num_ids = graph
            .nodes
            .iter()
            .map(|n| n.id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let mut node_info: Vec<NodeInfo> = vec![
            NodeInfo {
                kind: NodeKind::Hidden,
                num_inputs: 0,
                num_outputs: 0,
            };
            num_ids
        ];
        for node in &graph.nodes {
            node_info[node.id] = NodeInfo {
                kind: node.kind,
                num_inputs: node.num_inputs,
                num_outputs: node.num_outputs,
            };
        }
        // 🧮 One Linear per node, all hidden_dim → hidden_dim
        let mut layers = Vec::with_capacity(num_ids);
        for _ in 0..num_ids {
            layers.push(Linear::on_device(hidden_dim as i64, hidden_dim as i64, device)?);
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
    ///                            combined ──▶ layers[1] ──▶ n1_out = y
    /// ```
    ///
    ///   1. net_input = input_proj(x)              // [batch, hidden_dim]
    ///   2. n0 has no input ports
    ///      => n0_out = layers[0](net_input)       // [batch, hidden_dim]
    ///   3. n1's inputs: n1_i0 <- n0_out, n1_i1 <- n0_out
    ///      combined = n0_out + n0_out             // Add: a + b
    ///      n1_out = layers[1](combined)           // [batch, hidden_dim]
    ///   4. return n1_out                          // 🏁 output_node = n1
    fn forward(&self, input: &Variable) -> flodl::tensor::Result<Variable> {
        // 🚪 Project the network input once; it feeds every orphaned input port.
        let net_input = self.input_proj.forward(input)?;

        // Output tensor per node, shared across all of the node's output ports
        // (all its output ports emit the same tensor for now).
        let mut node_outputs: HashMap<usize, Variable> = HashMap::new();

        // Connections only go forward (from.node < to.node), so ascending node
        // id order is a valid topological execution order ✅ — every source is
        // already computed when we read it.
        for node_id in 0..self.layers.len() {
            let num_inputs = self.node_info[node_id].num_inputs;

            // Gather this node's input tensors: either from a wired connection
            // or (if orphaned) from the network input.
            let mut combined: Option<Variable> = None;
            let mut num_sources = 0usize;
            for port in 0..num_inputs {
                let target = Port {
                    node: node_id,
                    index: port,
                };
                let source = self
                    .connections
                    .iter()
                    .find(|c| c.to == target)              // 🔗 is this port wired?
                    .and_then(|c| node_outputs.get(&c.from.node))  // already-computed source
                    .unwrap_or(&net_input);                // 🕳️ orphan -> network input
                combined = Some(match combined {
                    None => source.clone(),
                    Some(prev) => prev.add(source)?,       // ➕ accumulate (sum)
                });
                num_sources += 1;
            }

            // Combine the gathered tensors per the graph's CombineOp.
            let combined = match combined {
                // Node with no input ports (e.g. an input node): feed it the
                // network input directly.
                None => net_input.clone(),
                Some(c) if self.combine_op == CombineOp::Mean && num_sources > 1 => {
                    c.mul_scalar(1.0 / num_sources as f64)?   // ➗ average: (a+b+c)/3
                }
                Some(c) => c,
            };

            // 🧮 Transform: run the node's layer, store its output tensor.
            let out = self.layers[node_id].forward(&combined)?;
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

    use crate::graph::Graph;
    use crate::node::Node;

    fn cpu_opts() -> TensorOptions {
        TensorOptions {
            dtype: DType::Float32,
            device: Device::CPU,
        }
    }

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
        let input = Variable::new(
            Tensor::randn(
                &[batch, graph.options.input_dim as i64],
                cpu_opts(),
            )
            .unwrap(),
            false,
        );
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
        let input = Variable::new(
            Tensor::randn(&[batch, graph.options.input_dim as i64], cpu_opts()).unwrap(),
            false,
        );
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.hidden_dim as i64]);
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
}
