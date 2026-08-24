//! JSON serialization for the graph blueprint.
//!
//! [`Topology`] can't derive `Serialize` directly — it holds an RNG
//! (`rng: fastrand::Rng`) that isn't serializable — so this module defines:
//!
//! - [`Spec`] — a plain, fully-serializable mirror of the blueprint:
//!   everything except the RNG
//! - `Serialize` / `Deserialize` impls for [`Topology`] that convert through
//!   [`Spec`]
//!
//! # Reproducibility
//!
//! The RNG is **re-seeded from `options.seed`** on load, so a loaded graph
//! regenerates wiring identically to a fresh graph with the same options
//! (`Topology::to_json` / `Topology::from_json` delegate here).
//!
//! Weights are **never** serialized — the blueprint is the single source of
//! truth, and the executable module is rebuilt with fresh random weights via
//! [`Network::build`](crate::network::Network::build). Saving a
//! found architecture is `graph.to_json()`, reloading it is
//! `Topology::from_json(&json)` then `Network::build(&graph, device)`.

use fastrand::Rng;
use serde::de::Deserializer;
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

use flodl::nn::Module;

use crate::node::Node;
use crate::topology::{Connection, Topology, TopologyOptions};
use crate::utils::graph_utils;

/// JSON round-trip representation of a [`Topology`] — the blueprint minus the
/// RNG. `options.seed` is what makes regeneration reproducible, so the RNG is
/// rebuilt from it on load rather than stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spec {
    pub id: usize,
    pub options: TopologyOptions,
    pub nodes: Vec<Node>,
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<String>,
    pub connections: Vec<Connection>,
}

impl From<&Topology> for Spec {
    fn from(g: &Topology) -> Self {
        Spec {
            id: g.id,
            options: g.options,
            nodes: g.nodes.clone(),
            graph_inputs: g.graph_inputs.clone(),
            graph_outputs: g.graph_outputs.clone(),
            connections: g.connections.clone(),
        }
    }
}

impl Serialize for Topology {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Spec::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Topology {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let spec = Spec::deserialize(deserializer)?;
        Ok(Topology {
            id: spec.id,
            nodes: spec.nodes,
            options: spec.options,
            graph_inputs: spec.graph_inputs,
            graph_outputs: spec.graph_outputs,
            connections: spec.connections,
            // Reproducibility: the RNG is re-seeded from options.seed, so
            // a loaded graph regenerates wiring identically to a fresh one.
            rng: Rng::with_seed(spec.options.seed as u64),
        })
    }
}

// ════════════════════════════════════════════════════════════════════════
// Network facts — the materialized-network companion to `Spec`.
// ════════════════════════════════════════════════════════════════════════

/// Materialized-network diagnostics — the network's "nutrition label".
/// Mirrors what [`Network::to_json`] used to produce; lives here alongside
/// [`Spec`] so all serialization stays in one module.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Network;
    use crate::node::{Activation, Node};
    use crate::topology::test_strategies::topology_strategy;
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

    // ── serialization ───────────────────────────────────────────────────────

    #[test]
    fn test_topology_json_roundtrip() {
        let mut graph = Topology::new(7, None);
        graph.create_random_hidden_nodes(4);
        graph.refresh_labels();
        graph.finalize();
        let json = graph.to_json().unwrap();
        let loaded: Topology = Topology::from_json(&json).unwrap();
        assert_eq!(loaded.id, graph.id);
        assert_eq!(loaded.options, graph.options);
        assert_eq!(loaded.nodes, graph.nodes);
        assert_eq!(loaded.graph_inputs, graph.graph_inputs);
        assert_eq!(loaded.graph_outputs, graph.graph_outputs);
        assert_eq!(loaded.connections, graph.connections);
    }

    #[test]
    fn test_topology_json_rewiring_is_deterministic() {
        // The RNG is re-seeded from options.seed on load, so a loaded graph
        // wires identically to a fresh graph with the same options.
        let mut original = Topology::new(3, None);
        original.nodes.push(Node::new_input(0, 2));
        original.nodes.push(Node::new_hidden(1, 3, 2));
        original.nodes.push(Node::new_output(2, 2, 1));

        let json = original.to_json().unwrap();
        let mut loaded = Topology::from_json(&json).unwrap();
        let mut fresh = Topology::new(3, None);
        fresh.nodes = original.nodes.clone();

        original.finalize();
        loaded.finalize();
        fresh.finalize();
        assert_eq!(loaded.connections, fresh.connections);
        assert_eq!(loaded.connections, original.connections);
    }

    #[test]
    fn test_topology_json_rebuilds_same_architecture() {
        // The blueprint is the single source of truth: saving/loading it and
        // re-building yields the same architecture (fresh random weights —
        // weights are never serialized).
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        let mut wide = Node::new_hidden(1, 3, 2);
        wide.hidden_dim = Some(32);
        wide.activation = Activation::GeLU;
        graph.nodes.push(wide);
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.finalize();

        let module = Network::build(&graph, Device::CPU).unwrap();
        let json = graph.to_json().unwrap();
        let reloaded_graph = Topology::from_json(&json).unwrap();
        let rebuilt = Network::build(&reloaded_graph, Device::CPU).unwrap();

        // Same output node, same nodes + derived dims, same param count.
        assert_eq!(rebuilt.output_node, module.output_node);
        assert_eq!(rebuilt.nodes, module.nodes);
        assert_eq!(rebuilt.node_dims, module.node_dims);
        assert_eq!(rebuilt.parameters().len(), module.parameters().len());
        let input = rand_input(2, graph.options.input_dim);
        assert_eq!(
            rebuilt.forward(&input).unwrap().shape(),
            module.forward(&input).unwrap().shape()
        );
    }

    // ── property tests (proptest) ───────────────────────────────────────────

    proptest! {
        /// Any random blueprint round-trips through JSON exactly: the loaded
        /// spec matches the original field-for-field.
        #[test]
        fn prop_topology_json_roundtrip(graph in topology_strategy()) {
            let json = graph.to_json().unwrap();
            let loaded: Topology = Topology::from_json(&json).unwrap();
            prop_assert_eq!(Spec::from(&loaded), Spec::from(&graph));

            // Reproducibility: with both RNGs back at the seed, a reloaded
            // graph rewires identically (same connections, same order) to a
            // fresh graph with the same topology and options.
            let mut loaded = loaded;
            let mut fresh = graph;
            loaded.rng = fastrand::Rng::with_seed(loaded.options.seed as u64);
            fresh.rng = fastrand::Rng::with_seed(fresh.options.seed as u64);
            loaded.finalize();
            fresh.finalize();
            prop_assert_eq!(loaded.connections, fresh.connections);
        }
    }
}
