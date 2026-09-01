//! JSON serialization for the graph blueprint and network diagnostics.
//!
//! This module provides:
//!
//! - [`Spec`] — a plain, fully-serializable mirror of [`Topology`](crate::topology::Topology)
//! - [`NetworkFacts`] — materialized-network diagnostics ("nutrition label")

mod network_facts;
mod spec;

pub use network_facts::NetworkFacts;
pub use spec::Spec;
