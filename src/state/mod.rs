//! State serialization and persistence for the race engine.
//!
//! Sibling file `persistence.rs` holds the core serialization structures and logic.

pub mod persistence;

pub use persistence::{
    ConfigSnapshot, NetMeta, NetMetrics, NetState, RaceState, RunConfig, RunHeader,
    TopologyLifetime, load_engine_json, load_net_state, write_engine_json, write_net_state,
};
