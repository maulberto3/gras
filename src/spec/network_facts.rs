//! Materialized-network diagnostics — the network's "nutrition label".
//!
//! Mirrors what [`Network::to_json`] used to produce; lives here alongside
//! [`Spec`](super::Spec) so all serialization stays in one module.

use flodl::nn::Module;

use crate::utils::graph_utils;

/// Materialized-network diagnostics — the network's "nutrition label".
/// Mirrors what [`Network::to_json`] used to produce; lives here alongside
/// [`Spec`](super::Spec) so all serialization stays in one module.
pub struct NetworkFacts {
    pub name: String,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_node: usize,
    pub num_nodes: usize,
    pub num_wires: usize,
    pub param_tensors: usize,
    pub param_elements: i64,
    pub port_projections: usize,
    pub node_dims: Vec<(usize, usize)>,
    pub degrees: Vec<(usize, usize)>,
    pub depths: Vec<usize>,
    pub orphan_counts: (usize, usize),
    pub kind_counts: (usize, usize, usize),
    pub activation_counts: Vec<(crate::node::Activation, usize)>,
    pub standardize_counts: Vec<(crate::node::StandardizeOp, usize)>,
}

impl NetworkFacts {
    /// Compute facts from a built network (pure read-only diagnostics).
    pub fn from_network(net: &crate::network::Network) -> Self {
        let output_node = net.output_node;
        let (orphan_in, orphan_out) =
            graph_utils::node_orphan_counts(&net.nodes, &net.connections, output_node);
        let kind_counts = graph_utils::node_kind_counts(&net.nodes);
        let params = net.parameters();
        let param_elements: i64 = params.iter().map(|p| p.variable.numel()).sum();
        NetworkFacts {
            name: net.name.clone(),
            input_dim: net.input_dim,
            hidden_dim: net.hidden_dim,
            output_node,
            num_nodes: net.nodes.len(),
            num_wires: net.connections.len(),
            param_tensors: params.len(),
            param_elements,
            port_projections: net
                .port_projections
                .iter()
                .flatten()
                .flatten()
                .filter(|p| p.is_some())
                .count(),
            node_dims: net.node_dims.clone(),
            degrees: graph_utils::node_degrees(&net.nodes, &net.connections),
            depths: graph_utils::node_depths(&net.nodes, &net.connections),
            orphan_counts: (orphan_in, orphan_out),
            kind_counts: (kind_counts.input, kind_counts.hidden, kind_counts.output),
            activation_counts: graph_utils::node_activation_counts(&net.nodes),
            standardize_counts: graph_utils::node_standardize_counts(&net.nodes),
        }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> flodl::tensor::Result<String> {
        let spec = serde_json::json!({
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_node": self.output_node,
            "num_nodes": self.num_nodes,
            "num_wires": self.num_wires,
            "param_tensors": self.param_tensors,
            "param_elements": self.param_elements,
            "port_projections": self.port_projections,
            "node_dims": self.node_dims,
            "degrees": self.degrees,
            "depths": self.depths,
            "orphan_counts": [self.orphan_counts.0, self.orphan_counts.1],
            "kind_counts": [self.kind_counts.0, self.kind_counts.1, self.kind_counts.2],
            "activation_counts": self.activation_counts,
            "standardize_counts": self.standardize_counts,
        });
        serde_json::to_string_pretty(&spec).map_err(|e| {
            crate::utils::error::NetworkError::Json(format!("network facts: {e}")).into()
        })
    }
}
