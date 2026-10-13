//! State serialization and persistence for the race engine.
//!
//! Sibling file `persistence.rs` holds the core serialization structures and logic.

pub mod persistence;

pub use persistence::{
    ConfigSnapshot, GuardrailRecord, NetMeta, NetMetrics, NetState, RaceState, RunAccounting,
    RunConfig, RunHeader, SeedSourceRecord, TopologyLifetime, load_engine_json, load_net_state,
    write_engine_json, write_guardrail_record, write_net_state,
};
