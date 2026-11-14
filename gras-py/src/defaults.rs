//! `gras.defaults()` — the numbers a fresh `RaceConfig()` actually holds.
//!
//! Rust callers get this for free (`RaceConfig::defaults()`, `TopologyOptions::
//! default()`); a notebook has no way to read a `TopologyOptions` at all, and the
//! Python surface hides a dozen knobs behind `None` (= "use the crate default") or
//! an empty pool (= "all"). So the source of truth here is the CRATE's
//! `RaceConfig::defaults()`, with the two deliberate Python divergences applied and
//! re-stated in `binding_overrides`: a Python run ranks on the plain mean
//! (`risk_aversion = 0.0`, not the crate's FIT 1.0) and starts all-random
//! (`champion_previous_load = false`).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use gras::graph::topology::TopologyOptions;

/// `f32` → `f64` without the binary-tail noise: `0.05f32` must print as `0.05`, not
/// `0.05000000074505806`. Six decimals is far finer than any knob here.
fn round32(v: f32) -> f64 {
    ((v as f64) * 1e6).round() / 1e6
}

/// Everything a default run holds, as one flat dict plus the nested blueprint + the
/// binding's own override list.
#[pyfunction]
pub fn defaults(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let cfg = gras::RaceConfig::defaults();
    let opts: TopologyOptions = cfg.topology_options;
    let d = PyDict::new(py);

    // ── run / ranking ───────────────────────────────────────────────────
    d.set_item("pop_size", cfg.pop_size)?;
    d.set_item("champion_count", cfg.champion_count)?;
    d.set_item("max_steps", cfg.max_steps)?;
    d.set_item("checkpoint_every", cfg.checkpoint_every)?;
    // The EVAL share of a single-pool dataset (0.2 = 80/20). A `train/`+`test/`
    // dir states its own split, so this is the "gras decides" default.
    d.set_item("eval_share", round32(cfg.eval_share))?;
    d.set_item("freeze_seats", cfg.freeze_seats)?;
    d.set_item("smoothing_window", cfg.smoothing_window)?;
    d.set_item("mean_type", cfg.mean_type.to_string())?;
    // The BINDING's values: the two knobs a Python run does not inherit from the crate.
    d.set_item("risk_aversion", 0.0)?;
    d.set_item("train_merit", round32(cfg.train_merit))?;
    d.set_item("train_merit_p_min", round32(cfg.train_merit_p_min))?;
    d.set_item("train_merit_p_max", round32(cfg.train_merit_p_max))?;
    // The ticket's LEGS: the shipped default is `least_fit_fewest_updates` (`Fm⁻ × Um`), no init
    // probation. Emitted so a notebook can read the shipped product instead of inferring it from
    // the column names.
    d.set_item("ticket_legs", cfg.ticket_legs.to_string())?;
    d.set_item("init_probation_steps", cfg.init_probation_steps)?;
    d.set_item("drift_prob", round32(cfg.drift_prob))?;
    d.set_item("feature_split_pool", cfg.feature_split_pool_names())?;
    d.set_item("log_level", format!("{:?}", cfg.log_level).to_lowercase())?;
    d.set_item("mode", format!("{:?}", cfg.mode).to_lowercase())?;

    // ── evolution ───────────────────────────────────────────────────────
    d.set_item("crossover_rolls", cfg.crossover_rolls)?;
    d.set_item("mutate_rolls", cfg.mutate_rolls)?;
    d.set_item("crossover_prob", round32(cfg.crossover_prob))?;
    d.set_item("mutate_prob", round32(cfg.mutate_prob))?;
    // The mutation novelty gate: `0.0` is OFF (one draw), so the default 0.2 is a real
    // setting, not a neutral zero — it is the crate's measured clone band.
    d.set_item("mutation_novelty_min", round32(cfg.mutation_novelty_min))?;
    d.set_item("mutation_novelty_draws", cfg.mutation_novelty_draws)?;
    d.set_item("crossover_selection", cfg.crossover_selection.to_string())?;
    d.set_item(
        "crossover_cull_policy",
        cfg.crossover_cull_policy.to_string(),
    )?;
    d.set_item("mutation_cull_policy", cfg.mutation_cull_policy.to_string())?;
    d.set_item("crossover_ops_pool", cfg.crossover_ops_pool.clone())?;
    d.set_item("mutation_probation_steps", cfg.mutation_probation_steps)?;
    d.set_item("crossover_probation_steps", cfg.crossover_probation_steps)?;
    d.set_item("seat_join_probation_steps", cfg.seat_join_probation_steps)?;
    d.set_item("seat_lifetime_steps", cfg.seat_lifetime_steps)?;
    d.set_item("recovery_retries", cfg.recovery_retries)?;
    d.set_item("recovery_tolerance", round32(cfg.recovery_tolerance))?;
    d.set_item("recovery_same_seed", cfg.recovery_same_seed)?;

    // ── search space (the blueprint the Python surface only shows as `None`) ──
    d.set_item(
        "hidden_dim_pool",
        cfg.hidden_dim_pool
            .map(|r| format!("{}..={}", r.start(), r.end())),
    )?;
    d.set_item("hidden_dim_stride", cfg.hidden_dim_stride)?;
    d.set_item("activation_pool", cfg.activation_pool.clone())?;
    d.set_item("combine_op_pool", cfg.combine_op_pool.clone())?;
    d.set_item("standardize_op_pool", cfg.standardize_op_pool.clone())?;
    let topo = PyDict::new(py);
    topo.set_item("min_hidden_num_nodes", opts.min_hidden_num_nodes)?;
    topo.set_item("max_hidden_num_nodes", opts.max_hidden_num_nodes)?;
    topo.set_item(
        "min_hidden_inputs_per_node",
        opts.min_hidden_inputs_per_node,
    )?;
    topo.set_item(
        "max_hidden_inputs_per_node",
        opts.max_hidden_inputs_per_node,
    )?;
    topo.set_item(
        "min_hidden_outputs_per_node",
        opts.min_hidden_outputs_per_node,
    )?;
    topo.set_item(
        "max_hidden_outputs_per_node",
        opts.max_hidden_outputs_per_node,
    )?;
    topo.set_item("dropout_prob", round32(opts.dropout_prob))?;
    // `None` = every port sees the whole input vector (no feature split).
    topo.set_item("feature_split", opts.feature_split.map(|p| p.to_string()))?;
    d.set_item("topology_options", topo)?;

    // ── artifacts ───────────────────────────────────────────────────────
    d.set_item("csv_export", cfg.csv_export)?;
    d.set_item("trace_file", cfg.trace_file)?;
    d.set_item("save_optim", cfg.save_optim)?;
    d.set_item("champion_save_topology", cfg.champion_save_topology)?;
    d.set_item("run_name", cfg.run_name.clone())?;
    d.set_item("champion_previous_load", false)?;
    d.set_item(
        "champion_previous_dir",
        cfg.champion_previous_dir
            .as_ref()
            .map(|p| p.display().to_string()),
    )?;

    // What `RaceConfig()` deliberately does NOT take from the crate. Printed rather
    // than silently folded in, so a reader can see the two sets differ on purpose.
    let overrides = PyDict::new(py);
    overrides.set_item("risk_aversion", {
        let kv = PyDict::new(py);
        kv.set_item("binding", 0.0)?;
        kv.set_item("crate", round32(gras::engine::config::RISK_AVERSION))?;
        kv
    })?;
    overrides.set_item("champion_previous_load", {
        let kv = PyDict::new(py);
        kv.set_item("binding", false)?;
        kv.set_item("crate", cfg.champion_previous_load)?;
        kv
    })?;
    d.set_item("binding_overrides", overrides)?;

    Ok(d.unbind())
}
