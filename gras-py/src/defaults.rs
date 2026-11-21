//! `gras.defaults()` — the numbers a fresh `RaceConfig()` actually holds.

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
    let opts: TopologyOptions = cfg.topology.options;
    let d = PyDict::new(py);

    // ── run / ranking ───────────────────────────────────────────────────
    d.set_item("pop_size", cfg.population.pop_size)?;
    d.set_item("champion_count", cfg.population.champion_count)?;
    d.set_item("max_steps", cfg.stop.max_steps)?;
    d.set_item("checkpoint_every", cfg.run.checkpoint_every)?;
    // The EVAL share of a single-pool dataset (0.2 = 80/20). A `train/`+`test/`
    // dir states its own split, so this is the "gras decides" default.
    d.set_item("eval_share", round32(cfg.run.eval_share))?;
    d.set_item("smoothing_window", cfg.ranking.smoothing_window)?;
    d.set_item("mean_type", cfg.ranking.mean_type.to_string())?;
    // The BINDING's values: the two knobs a Python run does not inherit from the crate.
    d.set_item("train_merit_p_min", round32(cfg.lottery.train_merit_p_min))?;
    d.set_item("train_merit_p_max", round32(cfg.lottery.train_merit_p_max))?;
    // The ticket's LEGS: the shipped default is `least_fit_fewest_updates` (`Fm⁻ × Um`), no init probation.
    d.set_item("ticket_legs", cfg.lottery.train_ticket.to_string())?;
    d.set_item("init_probation_steps", cfg.population.init_probation_steps)?;
    d.set_item("drift_prob", round32(cfg.topology.drift_prob))?;
    d.set_item("feature_split_pool", cfg.feature_split_pool_names())?;
    d.set_item("log_level", format!("{:?}", cfg.run.log_level).to_lowercase())?;
    d.set_item("mode", format!("{:?}", cfg.mode).to_lowercase())?;

    // ── evolution ───────────────────────────────────────────────────────
    // The mutation novelty gate: `0.0` is OFF (one draw), so the default 0.2 is a real
    // setting, not a neutral zero — it is the crate's measured clone band.
    d.set_item("mutation_novelty_min", round32(cfg.evolution.mutation_novelty_min))?;
    d.set_item("mutation_novelty_draws", cfg.evolution.mutation_novelty_draws)?;
    d.set_item("crossover_selection", cfg.evolution.crossover_selection.to_string())?;
    d.set_item(
        "crossover_cull_policy",
        cfg.evolution.crossover_cull_policy.to_string(),
    )?;
    d.set_item("mutation_cull_policy", cfg.evolution.mutation_cull_policy.to_string())?;
    d.set_item("crossover_ops_pool", cfg.evolution.crossover_ops_pool.clone())?;
    d.set_item("mutation_probation_steps", cfg.population.mutation_probation_steps)?;
    d.set_item("crossover_probation_steps", cfg.population.crossover_probation_steps)?;
    d.set_item("recovery_retries", cfg.population.recovery_retries)?;
    d.set_item("recovery_tolerance", round32(cfg.population.recovery_tolerance))?;
    d.set_item("recovery_same_seed", cfg.population.recovery_same_seed)?;

    // ── search space (the blueprint the Python surface only shows as `None`) ──
    d.set_item(
        "hidden_dim_pool",
        cfg.topology.hidden_dim_pool
            .map(|r| format!("{}..={}", r.start(), r.end())),
    )?;
    d.set_item("hidden_dim_stride", cfg.topology.hidden_dim_stride)?;
    d.set_item("activation_pool", cfg.topology.activation_pool.clone())?;
    d.set_item("combine_op_pool", cfg.topology.combine_op_pool.clone())?;
    d.set_item("standardize_op_pool", cfg.topology.standardize_op_pool.clone())?;
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
    d.set_item("csv_export", cfg.run.csv_export)?;
    d.set_item("trace_file", cfg.run.trace_file)?;
    d.set_item("save_optim", cfg.run.save_optim)?;
    d.set_item("champion_save_topology", cfg.run.champion_save_topology)?;
    d.set_item("run_name", cfg.run.run_name.clone())?;
    d.set_item("champion_previous_load", false)?;
    d.set_item(
        "champion_previous_dir",
        cfg.run.champion_previous_dir
            .as_ref()
            .map(|p| p.display().to_string()),
    )?;

    // What `RaceConfig()` deliberately does NOT take from the crate. Printed rather
    // than silently folded in, so a reader can see the two sets differ on purpose.
    let overrides = PyDict::new(py);
    overrides.set_item("champion_previous_load", {
        let kv = PyDict::new(py);
        kv.set_item("binding", false)?;
        kv.set_item("crate", cfg.run.champion_previous_load)?;
        kv
    })?;
    d.set_item("binding_overrides", overrides)?;

    Ok(d.unbind())
}
