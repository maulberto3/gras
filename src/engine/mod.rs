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
//! - [`race_engine`] — the `RaceEngine` itself: construction, the run loop,
//!   per-net stepping, culling, persistence hooks
//! - [`fitness`] — `Fitness`, `Direction`, `Metric`, `FitnessLabel`

pub mod child;
pub mod config;
pub mod smoothing;
pub mod fitness;
pub mod population;
pub mod race_engine;
pub mod run_spec;

pub use config::{RaceConfig, RaceSnapshot, RunMode, StopReason};
pub use fitness::{Direction, Fitness, FitnessLabel};
pub use race_engine::RaceEngine;
pub use run_spec::{RunSpec, StreamShape};
