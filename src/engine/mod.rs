//! The engine — the step-race NAS loop.

pub mod breed;
pub mod clock;
pub mod config;
pub mod core;
pub mod diversity;
pub mod founding;
pub mod reporting;
pub mod start;

pub use config::{
    CrossCullPolicy, ModeConfig, MutationCullPolicy, RaceConfig, RaceSnapshot, RlRaceConfig,
    RunMode, StopReason, TabularRaceConfig, rl_race_config_builder, tabular_race_config_builder,
};
pub use core::CoreEngine;
pub use founding::seed::{SeedOutcome, SeedPick, resolve as resolve_seed_source};
pub use start::rl_engine::RlEngine;
pub use start::run_spec::{RLSpec, RunSpec, StreamShape, TabularSpec};
pub use start::tabular_engine::TabularEngine;
