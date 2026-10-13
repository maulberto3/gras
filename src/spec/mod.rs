//! JSON serialization for the graph blueprint and network diagnostics.
//!
//! This module provides:
//!
//! - [`Spec`] — a plain, fully-serializable mirror of [`Topology`](crate::graph::topology::Topology)
//! - [`NetworkFacts`] — materialized-network diagnostics ("nutrition label")

mod blueprint;
mod network_facts;

pub use blueprint::Spec;
pub use network_facts::NetworkFacts;
