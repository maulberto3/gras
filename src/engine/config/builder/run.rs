//! Builder setters for run plumbing, stop criteria and the artifact switches.

use super::RaceConfigBuilder;
use crate::engine::config::{LogLevel, RaceSnapshot};

impl RaceConfigBuilder {
    /// Set steps between checkpoints — the durability tick (frontier states + EVERY live net's
    /// weights + `ckpt.json` + CSV flush) and the stream's eval-permutation era.
    pub fn set_run_checkpoint_every(mut self, n: usize) -> Self {
        self.cfg.checkpoint_every = n;
        self
    }

    /// Set whether the TERMINAL stop writes every live net's weights. Default false: the stop
    /// is a reading of the race, the checkpoint is the artifact a resume loads.
    pub fn set_run_save_weights_at_stop(mut self, enabled: bool) -> Self {
        self.cfg.save_weights_at_stop = enabled;
        self
    }

    /// Set whether each live net's optimizer state rides its checkpoint (`nets/<hash>.optim`), so a
    /// resume restores the Adam moments exactly. `false` = weights only. Default true.
    pub fn set_run_save_optim(mut self, enabled: bool) -> Self {
        self.cfg.save_optim = enabled;
        self
    }

    /// Per-step log verbosity: `Summ` (compact one-line rollup) or `Full`
    /// (also dumps every per-net detail line each step).
    pub fn set_run_log_level(mut self, level: LogLevel) -> Self {
        self.cfg.log_level = level;
        self
    }

    /// Also write the event stream to `<run_dir>/telemetry.jsonl` (one JSON record per
    /// event). Off by default.
    pub fn set_run_trace_file(mut self, enabled: bool) -> Self {
        self.cfg.trace_file = enabled;
        self
    }

    /// Set whether to write the lossless event logs (`history.csv` + `attempts.csv`).
    // NOTE: there is no `set_run_mode` — RunMode is DERIVED from the RunSpec variant.
    pub fn set_run_csv_export(mut self, enabled: bool) -> Self {
        self.cfg.csv_export = enabled;
        self
    }

    /// Human label for the experiment, written as `"run_name"`. Informational only —
    /// results folder name is unchanged.
    pub fn set_run_name(mut self, name: impl Into<String>) -> Self {
        self.cfg.run_name = Some(name.into());
        self
    }

    /// Explicit step budget — the only built-in stop criterion.
    pub fn set_stop_max_steps(mut self, n: impl Into<Option<usize>>) -> Self {
        self.cfg.max_steps = n.into();
        self
    }

    /// A user-supplied stop predicate, consulted every step alongside `max_steps`;
    /// returning `true` ends the race with stop reason `Custom`.
    pub fn set_stop_custom(
        mut self,
        f: impl Fn(&RaceSnapshot) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.cfg.custom_stop = Some(Box::new(f));
        self
    }

    /// Opt in/out of the automatic inheritance of the newest compatible previous experiment's
    /// elites (topology-only).
    pub fn set_champion_previous_load(mut self, enabled: bool) -> Self {
        self.cfg.champion_previous_load = enabled;
        self
    }

    /// Where the automatic inheritance looks: a run dir or a dir of run dirs
    /// (newest id wins). Unset ⇒ the default `results/`.
    pub fn set_champion_previous_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.cfg.champion_previous_dir = Some(dir.into());
        self
    }

    /// At stop, save each champion's topology markdown (`champion-<hash>-<fitness>.md`).
    /// Default true.
    pub fn set_champion_save_topology(mut self, enabled: bool) -> Self {
        self.cfg.champion_save_topology = enabled;
        self
    }
}
