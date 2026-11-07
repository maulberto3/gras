//! Custom error types — one enum per domain, all in one place.

use std::fmt::Display;
use std::io;

use flodl::tensor::TensorError;

use crate::graph::topology::{Connection, Port};

// ── Topology ─────────────────────────────────────────────────────────────

/// Why a [`Topology`](crate::graph::topology::Topology) failed
/// [`Topology::validate`](crate::graph::topology::Topology::validate).
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
    /// An input port receives more than one wire.
    MultiWiredInputPort(Port),
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
            TopologyError::MultiWiredInputPort(p) => write!(
                f,
                "input port n{}_i{} carries more than one wire (one wire per port: \
                 give the node another input port instead)",
                p.node, p.index
            ),
        }
    }
}

// ── Node ─────────────────────────────────────────────────────────────────

// ── Network ──────────────────────────────────────────────────────────────

/// Compiling a blueprint into an executable [`Network`](crate::graph::network::Network).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkError {
    /// The blueprint failed [`Topology::validate`](crate::graph::topology::Topology::validate) —
    /// refused before any tensor work is spent.
    InvalidTopology(TopologyError),
    /// A build-time inconsistency not covered by validation.
    Build(String),
    /// The materialized-net facts failed to serialize.
    Json(String),
}

impl Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkError::InvalidTopology(e) => write!(f, "invalid graph: {e}"),
            NetworkError::Build(msg) => write!(f, "cannot build network: {msg}"),
            NetworkError::Json(msg) => write!(f, "network: JSON: {msg}"),
        }
    }
}

// ── Engine ───────────────────────────────────────────────────────────────

/// The NAS loop: options, data contract, checkpoints.
#[derive(Debug)]
pub enum EngineError {
    /// Options that can't produce a valid run (e.g. `pop_size == 0`).
    InvalidOptions(String),
    /// The dataset or a seeded topology doesn't match the engine options
    /// (e.g. wrong input dim).
    DataMismatch(String),
    /// Filesystem failure (run dir, checkpoints, logs).
    Io { path: String, source: io::Error },
    /// A JSON document failed to serialize or parse.
    Json(String),
    /// The rayon thread pool used for parallel evaluation failed to build.
    Rayon(String),
}

impl Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::InvalidOptions(msg) => write!(f, "engine: invalid options: {msg}"),
            EngineError::DataMismatch(msg) => write!(f, "engine: {msg}"),
            EngineError::Io { path, source } => {
                write!(f, "engine: cannot access {path}: {source}")
            }
            EngineError::Json(msg) => write!(f, "engine: JSON: {msg}"),
            EngineError::Rayon(msg) => write!(f, "engine: parallel evaluation pool: {msg}"),
        }
    }
}

// ── Data ─────────────────────────────────────────────────────────────────

/// The tensor file / dataset format.
#[derive(Debug)]
pub enum DataError {
    /// Filesystem failure while reading or writing a tensor file or dataset
    /// directory.
    Io { path: String, source: io::Error },
    /// The file doesn't start with the gras magic bytes — not one of ours.
    BadMagic { path: String },
    /// The header or body is shorter than the declared shape requires.
    Truncated(String),
    /// The blob size doesn't match the declared shape × dtype.
    SizeMismatch {
        path: String,
        expected: usize,
        found: usize,
    },
    /// `inputs` and `targets` disagree on how many samples they hold. Raised
    /// an IN-MEMORY dataset constructor, so there is no file to name.
    RowCountMismatch { inputs: usize, targets: usize },
    /// A dtype we can't write to disk (only Float32/Float64/Int64).
    UnsupportedDtype(String),
    /// An unknown dtype tag in a file header.
    UnknownDtypeTag(u8),
    /// JSON metadata (`meta.json`) serialization failure.
    Json(String),
    /// CSV parsing error.
    Csv(String),
}

impl Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataError::Io { path, source } => {
                write!(f, "gras data: cannot access {path}: {source}")
            }
            DataError::BadMagic { path } => {
                write!(f, "gras data: {path} is not a gras tensor file (bad magic)")
            }
            DataError::Truncated(msg) => write!(f, "gras data: truncated tensor file: {msg}"),
            DataError::SizeMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "gras data: {path} has {found} data bytes, expected {expected}"
            ),
            DataError::RowCountMismatch { inputs, targets } => write!(
                f,
                "gras data: inputs and targets disagree on row count: {inputs} vs {targets}"
            ),
            DataError::UnsupportedDtype(msg) => {
                write!(f, "gras data: unsupported dtype: {msg}")
            }
            DataError::UnknownDtypeTag(tag) => {
                write!(f, "gras data: unknown dtype tag {tag} in tensor file")
            }
            DataError::Json(msg) => write!(f, "gras data: JSON: {msg}"),
            DataError::Csv(msg) => write!(f, "gras data: CSV: {msg}"),
        }
    }
}

// ── Conversion at the API boundary ───────────────────────────────────────

impl From<DataError> for TensorError {
    fn from(e: DataError) -> Self {
        TensorError::new(&e.to_string())
    }
}

impl From<NetworkError> for TensorError {
    fn from(e: NetworkError) -> Self {
        TensorError::new(&e.to_string())
    }
}

impl From<EngineError> for TensorError {
    fn from(e: EngineError) -> Self {
        TensorError::new(&e.to_string())
    }
}
