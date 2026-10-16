//! Utilities — data I/O, graph helpers, scoring, rendering, and the
//! step-race training primitives.
//!
//! This layer sits BELOW `engine` and `trainer`, and holds the vocabulary the
//! two must share. That is deliberate: it is what lets a training scheme be
//! written without importing the engine — see [`fitness`], [`cfy`] and
//! [`training`].

pub mod ascii;
pub mod cfy;
pub mod display;
pub(crate) mod error;
pub mod fitness;
pub mod graph_utils;
pub mod markdown;
pub mod mermaid;
pub mod training;
pub mod race_steps;
pub mod safetensors;
pub mod score;
pub mod seed;
pub mod tabular_data;
