//! Graph diagnostic utilities — shared helpers over raw node/connection data.

use std::collections::HashMap;

use crate::node::{Activation, Node, NodeKind};
use crate::topology::{Connection, KindCounts, Port};

/// Precompute wiring table: per node, per port, list of source ports.
pub(crate) fn build_node_sources(
    connections: &[Connection],
    num_inputs: &[usize],
) -> Vec<Vec<Vec<Port>>> {
    let mut input_map: HashMap<Port, Vec<Port>> = HashMap::new();
    for c in connections {
        input_map.entry(c.to).or_default().push(c.from);
    }
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

/// Orphaned-port counts `(inputs, outputs)` over raw graph data.
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
            continue;
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

/// Wired degrees `(in_degree, out_degree)` per node.
pub(crate) fn node_degrees(nodes: &[Node], connections: &[Connection]) -> Vec<(usize, usize)> {
    let mut deg = vec![(0usize, 0usize); nodes.len()];
    for c in connections {
        deg[c.to.node].0 += 1;
        deg[c.from.node].1 += 1;
    }
    deg
}

/// Longest path from an Input node per node id.
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

/// Node counts by kind.
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

/// Activation histogram across nodes.
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

/// Standardize-op histogram across nodes.
pub(crate) fn node_standardize_counts(nodes: &[Node]) -> Vec<(crate::node::StandardizeOp, usize)> {
    let mut counts: Vec<(crate::node::StandardizeOp, usize)> = Vec::new();
    for n in nodes {
        let op = n.standardize.unwrap_or_default();
        if let Some(entry) = counts.iter_mut().find(|(s, _)| *s == op) {
            entry.1 += 1;
        } else {
            counts.push((op, 1));
        }
    }
    counts
}
