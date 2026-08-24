//! Custom error types  — one enum per domain, all in one place.
//!
//! Every module's fallible paths build its **typed** error here, then convert
//! into flodl's [`TensorError`] at the API boundary (the public signatures
//! stay `flodl::tensor::Result`). The conversion is a single `From` impl per
//! enum, so call sites never format strings for errors.
//!
//! - [`TopologyError`] — blueprint validation ([`Topology::validate`]).
//! - [`NodeError`] — per-node invariants. Forward-looking: node constructors
//!   are infallible today and `Activation::apply` delegates straight to
//!   flodl, so nothing constructs these yet — they exist so future node-level
//!   rules (port sanity, dim checks) have a typed home.
//! - [`NetworkError`] — compiling a blueprint into a [`Network`] module.
//! - [`EngineError`] — the NAS loop (options, data contract, checkpoints).
//! - [`DataError`] — the tensor file/dataset format.
//!
//! [`Topology::validate`]: crate::topology::Topology::validate
//! [`Network`]: crate::network::Network

use std::fmt::Display;
use std::io;

use flodl::tensor::TensorError;

use crate::topology::{Connection, Port};

// ── Topology ─────────────────────────────────────────────────────────────

/// Why a [`Topology`](crate::topology::Topology) failed
/// [`Topology::validate`](crate::topology::Topology::validate).
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
        }
    }
}

// ── Node ─────────────────────────────────────────────────────────────────

/// Per-node invariant violations.
///
/// Forward-looking: node constructors and builders are infallible, and
/// `Activation::apply` delegates to flodl directly — so nothing constructs
/// these yet. They give future node-level rules (port sanity, dim checks) a
/// typed home in this module.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum NodeError {
    /// A node with zero input **and** zero output ports contributes nothing.
    ZeroPorts(usize),
    /// A `hidden_dim` override of 0 is invalid (also rejected by
    /// `Topology::validate`).
    ZeroHiddenDim(usize),
}

impl Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeError::ZeroPorts(id) => {
                write!(f, "node n{id} has neither input nor output ports")
            }
            NodeError::ZeroHiddenDim(id) => write!(f, "node n{id} has hidden_dim 0"),
        }
    }
}

// ── Network ──────────────────────────────────────────────────────────────

/// Compiling a blueprint into an executable [`Network`](crate::network::Network).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkError {
    /// The blueprint failed
    /// [`Topology::validate`](crate::topology::Topology::validate) — refused
    /// before any tensor work is spent.
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
    /// A dtype we can't write to disk (only Float32/Float64/Int64).
    UnsupportedDtype(String),
    /// An unknown dtype tag in a file header.
    UnknownDtypeTag(u8),
    /// JSON metadata (`meta.json`) serialization failure.
    Json(String),
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
            DataError::UnsupportedDtype(msg) => {
                write!(f, "gras data: unsupported dtype: {msg}")
            }
            DataError::UnknownDtypeTag(tag) => {
                write!(f, "gras data: unknown dtype tag {tag} in tensor file")
            }
            DataError::Json(msg) => write!(f, "gras data: JSON: {msg}"),
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
