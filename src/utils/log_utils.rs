//! Engine logging — re-exports from split modules.
//!
//! - [`init_log`] — options, dataset, population (shown once at new())
//! - [`gen_log`] — run start, done message, next steps
//! - [`summary_log`] — robustness table display

pub(crate) use crate::utils::init_log::log_initialization;
pub(crate) use crate::utils::gen_log::{log_done, log_run_start};
pub(crate) use crate::utils::summary_log::{log_repeated_topologies, topo_hash};
