//! Custom error types. 🛡️
//!
//! [`TopologyError`] is the blueprint's validation error — produced by
//! [`Topology::validate`](crate::topology::Topology::validate) and wrapped
//! by [`Network::build`](crate::network::Network::build) into a flodl
//! tensor error when a graph fails to compile.

use std::fmt::Display;

use crate::topology::{Connection, Port};

/// Why a [`Topology`](crate::topology::Topology) failed
/// [`Topology::validate`](crate::topology::Topology::validate). 🛡️
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// Node ids aren't 0, 1, 2, ... — they double as array indices, so gaps
    /// or duplicates make indexing unsafe.
    NonContiguousNodeIds,
    /// No nodes at all — nothing to execute.
    EmptyTopology,
    /// Options are internally inconsistent (inverted ranges, zero dims, ...).
    InvalidOptions(String),
    /// A connection references a node id that doesn't exist.
    UnknownNode(usize),
    /// A connection doesn't go strictly forward (`from.node < to.node`);
    /// this also forbids self-loops and recurrent edges.
    BackwardConnection(Connection),
    /// A wired port index is outside the node's declared port count.
    PortOutOfBounds { port: Port, num_ports: usize },
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

impl Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopologyError::NonContiguousNodeIds => {
                write!(
                    f,
                    "node ids must be contiguous 0..n (they double as array indices)"
                )
            }
            TopologyError::EmptyTopology => write!(f, "graph has no nodes"),
            TopologyError::InvalidOptions(msg) => write!(f, "invalid options: {msg}"),
            TopologyError::UnknownNode(id) => write!(f, "connection references unknown node {id}"),
            TopologyError::BackwardConnection(c) => {
                write!(
                    f,
                    "connection is not forward-only (from.node must be < to.node): {c}"
                )
            }
            TopologyError::PortOutOfBounds { port, num_ports } => write!(
                f,
                "port n{}[{}] is out of bounds (node has {num_ports} ports)",
                port.node, port.index
            ),
            TopologyError::DoubleUsedOutput(p) => {
                write!(f, "output port n{}_o{} is used twice", p.node, p.index)
            }
            TopologyError::InconsistentInputDims { node, dims } => write!(
                f,
                "node n{node} receives inputs with inconsistent dims {dims:?}"
            ),
            TopologyError::OrphanDimMismatch {
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
