//! State serialization and persistence for the race engine.
//!
//! Sibling file `persistence.rs` holds the core serialization structures and logic.

pub mod persistence;

pub use persistence::{
    ConfigSnapshot, FRONTIER_MANIFEST_SCHEMA, FrontierManifest, FrontierNet, GuardrailElite,
    GuardrailRecord, NetMeta, NetMetrics, NetState, RaceState, RunAccounting, RunConfig, RunHeader,
    SeedSourceRecord, TopologyLifetime, load_engine_json, load_frontier_manifest, load_net_state,
    truncate_csv_to_clock, write_engine_json, write_frontier_manifest, write_guardrail_record,
    write_net_state,
};
