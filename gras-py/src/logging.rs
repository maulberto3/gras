//! The process-wide console sink ceiling.
//!
//! The engine's console sink is installed exactly once per process: the FIRST
//! `RaceEngine` sets the level, and later engines can only ask for a quieter
//! reporting shape, never a louder one.

/// The engine's console sink is process-wide, so it can be installed exactly
/// once: the FIRST `RaceEngine` in a process sets the ceiling. Later engines can
/// still ask for a quieter reporting shape, never a louder one.
pub(crate) static LOGGING: std::sync::Once = std::sync::Once::new();
