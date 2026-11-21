//! The fluent builder: struct and `build()`, with the setters split by family so each
//! file answers one question ("where is the lottery knob?").

mod evolution;
mod lottery;
mod population;
mod ranking;
mod run;
mod topology;

use super::RaceConfig;

/// Fluent builder over [`RaceConfig`], covering options a caller is likely to set;
/// untouched fields keep conservative defaults.
pub struct RaceConfigBuilder {
    cfg: RaceConfig,
}

impl RaceConfigBuilder {
    /// Start from a base config — the constructor the mode front doors use.
    pub(crate) fn from_config(cfg: RaceConfig) -> Self {
        Self { cfg }
    }

    /// Finish and return the configured [`RaceConfig`].
    pub fn build(self) -> RaceConfig {
        self.cfg
    }
}
