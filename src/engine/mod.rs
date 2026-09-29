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
//! - [`core`] — the `CoreEngine` itself: construction, the run loop,
//!   per-net stepping, culling, persistence hooks
//! - [`logging`] — the `tracing` subscriber + its three sinks (console,
//!   `Minimal` frame, opt-in `telemetry.jsonl`)
//! - [`reporting`] — the user-facing log lines (per-step rollup, `Minimal`
//!   frame, stop summary)
//! - [`format`] — display/CSV formatting + the build identity stamp
//! - [`artifacts`] — champion/worst dumps + their ranking helpers
//! - [`checkpoint`] — the checkpoint ledger, gate bar, and surprise exam
//! - [`evolve`] — the crossover/immigrant rolls, cull/insert, victim selection
//! - [`pruner`] — the post-race solo-training phase
//! - [`step`] — one net's step path (trainer dispatch, freeze pass, RNG seeding)
//! - [`stop`] — stop criteria + `RaceSnapshot`
//! - [`tabular_engine`] — tabular-mode constructors (`resume`)
//! - [`rl_engine`] — RL-mode constructors (`resume_rl`)
//! - [`fitness`] — `Fitness`, `Direction`, `Metric`, `FitnessLabel`

pub mod artifacts;
pub mod checkpoint;
pub mod child;
pub mod config;
pub mod core;
pub mod evolve;
pub mod fitness;
pub mod format;
pub mod logging;
pub mod population;
pub mod guardrail;
pub mod pruner;
pub mod reporting;
pub mod step;
pub mod stop;
pub mod rl_engine;
pub mod run_spec;
pub mod smoothing;
pub mod tabular_engine;

pub use config::{
    rl_race_config_builder, tabular_race_config_builder, CrossCullPolicy, ModeConfig,
    MutationCullPolicy, RaceConfig, RaceSnapshot, RlConfig, RlRaceConfig, RunMode, StopReason,
    TabularConfig, TabularRaceConfig,
};
pub use guardrail::{check_champion, ChampionScorer, GuardrailVerdict};
pub use core::CoreEngine;
pub use fitness::{Direction, Fitness, FitnessLabel};
pub use rl_engine::RlEngine;
pub use run_spec::{RLSpec, RunSpec, StreamShape, TabularSpec};
pub use tabular_engine::TabularEngine;
