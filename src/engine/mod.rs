//! The engine — the step-race NAS loop.

pub mod child;
pub mod clock;
pub mod config;
pub mod core;
pub mod cull;
pub mod diversity;
pub mod evolve;
pub mod exam;
pub mod format;
pub mod logging;
pub mod novelty;
pub mod pool;
pub mod population;
pub mod reporting;
pub mod reports;
pub mod rl_engine;
pub mod run_spec;
pub mod seed;
pub mod tabular_engine;

pub use config::{
    CrossCullPolicy, ModeConfig, MutationCullPolicy, RaceConfig, RaceSnapshot, RlRaceConfig,
    RunMode, StopReason, TabularRaceConfig, rl_race_config_builder, tabular_race_config_builder,
};
pub use core::CoreEngine;
pub use rl_engine::RlEngine;
pub use run_spec::{RLSpec, RunSpec, StreamShape, TabularSpec};
pub use seed::{SeedOutcome, SeedPick, resolve as resolve_seed_source};
pub use tabular_engine::TabularEngine;
