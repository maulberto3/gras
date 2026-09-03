//! Engine persistence — JSON/MD/CSV output and prior-topology loading.
//!
//! Everything here writes run artifacts (`engine.json`, per-gen snapshots,
//! `robustness.csv`) or reads them back (`load_prior_topology`). Kept out
//! of the evolution core in `engine/mod.rs`.

use std::fs;
use std::io::Write;
use std::path::Path;

use flodl::tensor::Result;
use log::debug;

use crate::graph::topology::Topology;
use crate::utils::error::EngineError;
use crate::utils::robustness::RobustnessEntry;

use super::{Engine, GenStats};

impl Engine {
    /// Load a prior topology from an engine.json or improvement JSON file.
    /// Accepts:
    /// - `best_topology` — engine.json (best of best)
    /// - `file_topology` — improvement JSON (per-gen best/worst)
    /// - `topology` — legacy format
    /// The field can be a nested object or escaped JSON string.
    pub(crate) fn load_prior_topology(path: &Path) -> Result<Topology> {
        let raw = fs::read_to_string(path).map_err(|source| EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| EngineError::Json(format!("prior topology parse: {e}")))?;
        let topo_val = v.get("best_topology")
            .or_else(|| v.get("file_topology"))
            .or_else(|| v.get("topology"))
            .ok_or_else(|| EngineError::Json(
                format!("prior topology file has no 'best_topology', 'file_topology', or 'topology' field: {}", path.display())
            ))?;
        // Handle both formats: nested object vs escaped JSON string
        let topo_str = match topo_val {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other)
                .map_err(|e| EngineError::Json(format!("prior topology serialize: {e}")))?,
        };
        Topology::from_json(&topo_str)
            .map_err(|e| EngineError::Json(format!("prior topology from_json: {e}")).into())
    }

    /// Export robustness data to `robustness.csv` in the run directory.
    pub fn export_robustness(&self) -> Result<()> {
        let path = self.run_dir.join("robustness.csv");
        self.export_robustness_to(&path)
    }

    pub(crate) fn export_robustness_to(&self, path: &Path) -> Result<()> {
        let has_loss = self.robustness.values().any(|e| e.has_loss());
        let mut file = fs::File::create(path).map_err(|source| EngineError::Io { path: path.display().to_string(), source })?;
        if has_loss {
            writeln!(file, "topo_id,appearances,fit_mean,fit_sd,fit_min,fit_max,loss_mean,loss_sd,loss_min,loss_max,params").map_err(|source| EngineError::Io { path: path.display().to_string(), source })?;
        } else {
            writeln!(file, "topo_id,appearances,fit_mean,fit_sd,fit_min,fit_max,params").map_err(|source| EngineError::Io { path: path.display().to_string(), source })?;
        }
        let mut entries: Vec<&RobustnessEntry> = self.robustness.values().collect();
        entries.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| b.mean.partial_cmp(&a.mean).unwrap_or(std::cmp::Ordering::Equal)));
        for entry in &entries {
            let tid = crate::utils::log_utils::topo_hash(&entry.topology_json);
            if has_loss {
                writeln!(file, "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
                    tid, entry.count, entry.mean, entry.std_dev(), entry.min_fitness, entry.max_fitness,
                    entry.mean_loss.unwrap_or(0.0), entry.std_dev_loss(),
                    entry.min_loss.unwrap_or(0.0), entry.max_loss.unwrap_or(0.0), entry.param_count)
                    .map_err(|source| EngineError::Io { path: path.display().to_string(), source })?;
            } else {
                writeln!(file, "{},{},{:.6},{:.6},{:.6},{:.6},{}",
                    tid, entry.count, entry.mean, entry.std_dev(), entry.min_fitness, entry.max_fitness, entry.param_count)
                    .map_err(|source| EngineError::Io { path: path.display().to_string(), source })?;
            }
        }
        Ok(())
    }

    /// Save one JSON + one MD per generation.
    /// JSON contains all individuals with metrics + topologies.
    /// MD is for the best topology only.
    pub(crate) fn save_gen_data(&self, stats: &GenStats) -> std::result::Result<(), EngineError> {
        let dir = self.run_dir.join("improvements");
        fs::create_dir_all(&dir).map_err(|source| EngineError::Io {
            path: dir.display().to_string(),
            source,
        })?;

        // Build individuals array
        let mut individuals = Vec::with_capacity(self.pop.len());
        for (i, topo) in self.pop.iter().enumerate() {
            let fitness = self.scores[i];
            let loss = self.eval_losses.get(i).copied().flatten();
            let params = self.param_counts[i];
            let topo_json = topo.to_json()
                .map_err(|e| EngineError::Json(format!("individual topo json: {e}")))?;
            // Hash: SHA-like digest of full topology JSON — unique per topology
            let topo_hash = {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                topo_json.hash(&mut hasher);
                let h = hasher.finish();
                format!("{:016x}", h)[..8].to_string()  // first 8 hex chars
            };
            individuals.push(serde_json::json!({
                "idx": i,
                "seed": topo.options.seed,
                "fitness": fitness,
                "loss": loss,
                "params": params,
                "topo_hash": topo_hash,
                "topology": topo_json,
            }));
        }

        // Write gen JSON
        let gen_json = serde_json::json!({
            "generation": self.generation,
            "unique_topos": stats.unique_topos,
            "stats": {
                "best_score": stats.best_score,
                "best_loss": stats.best_loss,
                "avg_score": stats.avg_score,
                "avg_loss": stats.avg_loss,
                "avg_params": stats.avg_params,
            },
            "individuals": individuals,
        });
        let json_str = serde_json::to_string_pretty(&gen_json)
            .map_err(|e| EngineError::Json(format!("gen json: {e}")))?;
        let width = super::gen_width(self.config.num_generations);
        let json_path = dir.join(format!("gen_{:0width$}.json", self.generation, width = width));
        fs::write(&json_path, &json_str).map_err(|source| EngineError::Io {
            path: json_path.display().to_string(),
            source,
        })?;

        let best_topo = &self.pop[0];
        let md = crate::utils::markdown::topology_markdown(best_topo, Some(stats.best_score), None);
        let md_path = dir.join(format!("gen_{:0width$}.md", self.generation, width = width));
        fs::write(&md_path, md).map_err(|source| EngineError::Io {
            path: md_path.display().to_string(),
            source,
        })?;

        debug!("save_gen_data -- gen={} saved", self.generation);
        Ok(())
    }

    /// Update the robustness tracker with this gen's population.
    pub(crate) fn update_robustness(&mut self) {
        for (i, topo) in self.pop.iter().enumerate() {
            let fitness = self.scores[i];
            let loss = self.eval_losses.get(i).and_then(|l| *l);
            let key = match topo.to_json() {
                Ok(j) => j,
                Err(_) => continue,
            };
            use std::collections::hash_map::Entry;
            match self.robustness.entry(key) {
                Entry::Occupied(mut e) => {
                    let entry = e.get_mut();
                    entry.update(fitness, loss);
                }
                Entry::Vacant(e) => {
                    let topo_json = e.key().clone();
                    let param_count = self.param_counts.get(i).copied().unwrap_or(0);
                    e.insert(RobustnessEntry::new(fitness, loss, param_count, topo_json));
                }
            }
        }
    }

    /// Build the full JSON envelope for a given best individual.
    /// Shared by `to_json` (run-level) and `record_improvement` (per-snapshot).
    fn build_envelope(&self) -> std::result::Result<serde_json::Value, EngineError> {
        // Repeated topologies: sort by appearances desc, then mean desc
        let mut robust_entries: Vec<&RobustnessEntry> = self.robustness.values().collect();
        robust_entries.sort_by(|a, b| {
            b.count.cmp(&a.count)
                .then_with(|| b.mean.partial_cmp(&a.mean).unwrap_or(std::cmp::Ordering::Equal))
        });
        let robustness_val = serde_json::to_value(&robust_entries)
            .unwrap_or(serde_json::Value::Null);

        Ok(serde_json::json!({
            "run_id": self.run_id,
            "run_seed": self.seed,
            "generation": self.generation,
            "pop_size": self.pop.len(),
            "fitness_label": self.fitness.label(),
            "fitness_direction": format!("{:?}", self.fitness.direction()),
            "input_dim": self.options.topology_options.input_dim,
            "output_dim": self.options.topology_options.output_dim,
            "options": &self.options,
            "topology_options": serde_json::json!({
                "input_dim": self.options.topology_options.input_dim,
                "output_dim": self.options.topology_options.output_dim,
                "hidden_dim_pool": format!("{}..={}", self.options.hidden_dim_pool.start(), self.options.hidden_dim_pool.end()),
                "hidden_dim_stride": self.options.hidden_dim_stride,
                "min_hidden_num_nodes": self.options.topology_options.min_hidden_num_nodes,
                "max_hidden_num_nodes": self.options.topology_options.max_hidden_num_nodes,
                "min_hidden_inputs_per_node": self.options.topology_options.min_hidden_inputs_per_node,
                "max_hidden_inputs_per_node": self.options.topology_options.max_hidden_inputs_per_node,
                "min_hidden_outputs_per_node": self.options.topology_options.min_hidden_outputs_per_node,
                "max_hidden_outputs_per_node": self.options.topology_options.max_hidden_outputs_per_node,
            }),
            "history": if self.history.is_empty() { serde_json::Value::Null } else { serde_json::to_value(&self.history).unwrap_or(serde_json::Value::Null) },
            "robustness": robustness_val,
        }))
    }

    pub fn to_json(&self) -> Result<String> {
        let spec = self.build_envelope().map_err(|e| flodl::tensor::TensorError::new(&e.to_string()))?;
        serde_json::to_string_pretty(&spec)
            .map_err(|e| EngineError::Json(format!("to_json: {e}")).into())
    }
}