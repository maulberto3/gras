//! The computational graph — nodes, topologies (blueprints), and networks.
//!
//! - [`node`] — the NAS knobs: activations, combine ops, standardize ops.
//! - [`topology`] — the graph blueprint (DNA) being evolved.
//! - [`network`] — compiles a topology into an executable flodl module.

pub mod network;
pub mod node;
pub mod topology;