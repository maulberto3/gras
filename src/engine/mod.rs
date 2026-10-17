//! The engine — the step-race NAS loop.
//!
//! One global step clock; every live net trains and evals on the same shared
//! batch each step. Divergence over the population's smoothed fitness culls
//! the worst net and births a caught-up replacement. The old generational
//! engine was removed for 1.0 — [`RaceEngine`] is the only engine.
//!
//! Module layout (one file per concern):
//! - [`config`] — `RaceConfig` knob surface, `StopReason`, divergence/stop
//!   closure types, `RaceSnapshot`, `RunMetaCtx`
//! - [`population`] — initial population generation from the config's pools
//! - [`child`] — child generation (roulette → crossover → mutate) + resume replay
//! - [`smoothing`] — the rolling fitness buffer + the window's mean/spread and the
//!   risk-adjusted score the rankings read
//! - [`core`] — the `CoreEngine` itself: construction, the run loop,
//!   per-net stepping, culling, persistence hooks
//! - [`logging`] — the `tracing` subscriber + its three sinks (console,
//!   `Minimal` frame, opt-in `telemetry.jsonl`)
//! - [`reporting`] — the user-facing log lines (per-step rollup, `Minimal`
//!   frame, stop summary)
//! - [`format`] — display/CSV formatting + the build identity stamp
//! - [`artifacts`] — champion/worst dumps + their ranking helpers
//! - [`evolve`] — the crossover/immigrant rolls, cull/insert, victim selection
//! - [`step`] — one net's step path (trainer dispatch, freeze pass, RNG seeding)
//! - [`stop`] — stop criteria + `RaceSnapshot`
//! - [`tabular_engine`] — tabular-mode constructors (`resume`)
//! - [`rl_engine`] — RL-mode constructors (`resume_rl`)
//! - [`seed`] — founder-batch seeding from a previous run's guardrail champion
//! - `fitness` / `cfy` — **MOVED to [`crate::utils`]**: `Fitness`, `Direction`,
//!   `Metric`, `FitnessLabel` and the CFY log tables are vocabulary the engine
//!   *shares* with any training scheme, so they live below both layers and a
//!   trainer can reach them without importing the engine.

pub mod artifacts;
pub mod child;
pub mod config;
pub mod core;
pub mod evolve;
pub mod format;
pub mod guardrail;
pub mod logging;
pub mod population;
pub mod reporting;
pub mod rl_engine;
pub mod run_spec;
pub mod seed;
pub mod smoothing;
pub mod step;
pub mod stop;
pub mod tabular_engine;

pub use config::{
    CrossCullPolicy, ModeConfig, MutationCullPolicy, RaceConfig, RaceSnapshot, RlConfig,
    RlRaceConfig, RunMode, StopReason, TabularConfig, TabularRaceConfig, rl_race_config_builder,
    tabular_race_config_builder,
};
pub use core::CoreEngine;
pub use guardrail::GuardrailVerdict;
pub use rl_engine::RlEngine;
pub use run_spec::{RLSpec, RunSpec, StreamShape, TabularSpec};
pub use seed::{SeedOutcome, SeedPick, resolve as resolve_seed_source};
pub use tabular_engine::TabularEngine;
