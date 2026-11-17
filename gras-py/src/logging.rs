//! The process-wide console sink ceiling.

/// The engine's console sink is process-wide, so it can be installed exactly once: the FIRST `RaceEngine` in a process sets the ceiling.
pub(crate) static LOGGING: std::sync::Once = std::sync::Once::new();
