//! State serialization and persistence for the race engine.

pub mod persistence;

pub use persistence::{
    load_engine_json, load_frontier_manifest, load_net_state, truncate_csv_to_clock,
    write_engine_json, write_frontier_manifest, write_net_state, ConfigSnapshot, EvalBench,
    FrontierManifest, FrontierNet, NetMeta, NetMetrics, NetState, RaceState, RunAccounting,
    RunConfig, RunHeader, SeedSourceRecord, TopologyLife, TopologyLifetime,
    FRONTIER_MANIFEST_SCHEMA,
};
