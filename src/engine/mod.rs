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
//! - [`child`] — child generation (roulette → crossover → mutate) + catch-up
//! - [`smoothing`] — the rolling fitness buffer + its mean (ranking inputs)
//! - [`core`] — the `RaceEngine` itself: construction, the run loop,
//!   per-net stepping, culling, persistence hooks
//! - [`tabular_engine`] — tabular-mode constructors (`resume`)
//! - [`rl_engine`] — RL-mode constructors (`resume_rl`)
//! - [`fitness`] — `Fitness`, `Direction`, `Metric`, `FitnessLabel`

pub mod child;
pub mod config;
pub mod core;
pub mod fitness;
pub mod population;
pub mod rl_engine;
pub mod run_spec;
pub mod smoothing;
pub mod tabular_engine;

pub use config::{
    CrossCullPolicy, MutationCullPolicy, RaceConfig, RaceSnapshot, RunMode, StopReason,
};
pub use core::CoreEngine;
pub use fitness::{Direction, Fitness, FitnessLabel};
pub use rl_engine::RlEngine;
pub use run_spec::{RunSpec, StreamShape};
pub use tabular_engine::TabularEngine;
