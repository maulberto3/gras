//! Run artifacts: the champion (elite) and anti-champion (worst) dumps.

use super::core::CoreEngine;
use super::format::{ReadingStyle, fmt_reading};
use flodl::tensor::Result;

impl CoreEngine {
    /// Write the MACHINE-READABLE elite ranking — `elites.json`, best first,
    /// k = `elite_count`. UNCONDITIONAL: an inheritance receipt, not a log line.
    pub(crate) fn write_elite_index(&mut self) -> Result<()> {
        self.record_champions();
        // `fitness` stays the RANKING value (the key the next run and this file's readers already
        // agree on); the reading behind it is written out in full so "ranked 1st because
        let entries: Vec<serde_json::Value> = self
            .champion_ranked()
            .enumerate()
            .map(|(rank, (hash, fitness))| {
                let state = self.state.net(&hash);
                let reading = self.race_reading_of(&hash);
                serde_json::json!({
                    "rank": rank + 1,
                    "hash": hash,
                    "fitness": fitness,
                    "fitness_mean": reading.map(|r| r.mean),
                    "fitness_std": reading.map(|r| r.std),
                    "fitness_samples": reading.map(|r| r.n),
                    "risk_aversion": self.config.risk_aversion,
                    "mean_type": self.config.mean_type.to_string(),
                    "params": state.and_then(|s| s.meta.params),
                    "born": state.map(|s| s.entered_at_step),
                })
            })
            .collect();
        let path = self.run_dir.join("elites.json");
        let body = serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string());
        if let Err(source) = std::fs::write(&path, body) {
            tracing::warn!(
                "elite index write failed: {}",
                crate::utils::error::EngineError::Io {
                    path: path.display().to_string(),
                    source,
                }
            );
        } else if self.verbose_detail() {
            crate::utils::cfy::emit(&crate::utils::cfy::card(
                "Elite index",
                &crate::utils::cfy::kv(&[
                    ("file".to_string(), path.display().to_string()),
                    ("ranked".to_string(), entries.len().to_string()),
                ]),
                Some("inheritable by the next run"),
            ));
        }
        Ok(())
    }

    /// Write the top-k elites' topology markdown, one file per elite
    /// (`elite-<hash>.md`). UNCONDITIONAL at every log level.
    pub(crate) fn write_champion_markdown(&mut self) -> Result<()> {
        if !self.config.elite_save_topology {
            return Ok(());
        }
        self.record_champions();
        // The champion count is known without consuming the iterator — read
        // it from the config (the same k the iterator caps
        let k = self.config.elite_count.max(1);
        // CFY: rows of the written files, emitted as one table after the loop.
        let mut written: Vec<Vec<String>> = Vec::new();
        for (rank, (hash, fitness)) in self.champion_ranked().enumerate() {
            let Some(state) = self.state.net(&hash) else {
                continue;
            };
            let Ok(topo) = state.topology() else {
                continue;
            };
            let short = &hash[..8.min(hash.len())];
            let path = self.run_dir.join(format!("elite-{short}.md"));
            let label = if k <= 1 {
                "Elite".to_string()
            } else {
                format!("Elite #{}", rank + 1)
            };
            // The header names both legs: the score it ranked by, and the reading it was charged
            // from — `μ ±σ (SE, n)` in the same spelling as the logs, so a reader never has
            let reading = self.race_reading_of(&hash).map_or_else(
                || "—".to_string(),
                |r| {
                    fmt_reading(
                        Some(r.mean),
                        (r.n > 1).then_some(r.std),
                        r.n,
                        ReadingStyle::Doc,
                    )
                },
            );
            // The run seed rides the header: this file outlives the run dir, and the seed
            // one input a later replay cannot reconstruct from the net.
            let md = format!(
                "**{label} · fitness {} (risk-adjusted) = {:.4} · window {} · λ={:.2} · run seed {} · net seed {}**\n\n{}\n\n{}",
                self.fitness.direction().arrow(),
                fitness,
                reading,
                self.config.risk_aversion,
                self.header.run_seed,
                state.net_seed,
                crate::utils::markdown::topology_markdown(&topo, None),
                // The net's own birth, as a mermaid graph: a founder is one box,
                // crossover draws its parents above and converges down to this net.
                self.lineage_markdown(&hash, "ELITE"),
            );
            match std::fs::write(&path, &md) {
                Ok(()) => {
                    written.push(vec![
                        (rank + 1).to_string(),
                        short.to_string(),
                        path.display().to_string(),
                    ]);
                }
                Err(source) => tracing::warn!(
                    "elite markdown write failed: {}",
                    crate::utils::error::EngineError::Io {
                        path: path.display().to_string(),
                        source,
                    }
                ),
            }
        }
        if self.verbose_detail() && !written.is_empty() {
            crate::utils::cfy::emit(&crate::utils::cfy::card(
                "Elite topology (markdown)",
                &crate::utils::cfy::table(&["#", "hash", "file"], &written),
                Some("ready to view"),
            ));
        }
        Ok(())
    }

    /// Write EVERY live net's weights to `nets/<hash>.safetensors`.
    pub(crate) fn write_live_net_weights(&self) -> Result<()> {
        let dir = self.run_dir.join("nets");
        std::fs::create_dir_all(&dir).map_err(|source| crate::utils::error::EngineError::Io {
            path: dir.display().to_string(),
            source,
        })?;
        for hash in self.state.live_hashes() {
            let Some(net) = self.networks.get(hash.as_str()) else {
                tracing::warn!("net weights: live hash {hash} has no network in memory");
                continue;
            };
            let path = dir.join(format!("{hash}.safetensors"));
            if let Err(e) = crate::utils::safetensors::export_safetensors(net, &path) {
                tracing::warn!("net weights export failed for {hash}: {e}");
            }
            // The optimizer's own state, when asked
            if self.config.save_optim {
                let optim = dir.join(format!("{hash}.optim"));
                match self.optimizers.get(hash.as_str()) {
                    Some(o) => {
                        if let Err(e) = o.save_state_to(&optim.to_string_lossy()) {
                            tracing::warn!("net optimizer state write failed for {hash}: {e}");
                        }
                    }
                    None => {
                        tracing::warn!("net optimizer state: live hash {hash} has no optimizer")
                    }
                }
            }
        }
        Ok(())
    }

    /// Delete `nets/*.safetensors` whose hash is not live.
    pub(crate) fn sweep_stale_net_weights(&self) {
        let live: std::collections::HashSet<String> =
            self.state.live_hashes().into_iter().collect();
        let dir = self.run_dir.join("nets");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str());
            if ext != Some("safetensors") && ext != Some("optim") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if !live.contains(stem) {
                if let Err(source) = std::fs::remove_file(&path) {
                    tracing::warn!("net weights sweep failed for {}: {source}", path.display());
                }
            }
        }
    }

    /// Write the stop-time `Final field` table as `final-field.csv`: the SAME header and the SAME
    /// cells the card printed, one row per individual, in rank order.
    pub(crate) fn write_final_field_csv(&self, header: &[String], rows: &[Vec<String>]) {
        let path = self.run_dir.join("final-field.csv");
        let mut body = csv_row(header);
        for row in rows {
            body.push('\n');
            body.push_str(&csv_row(row));
        }
        body.push('\n');
        if let Err(source) = std::fs::write(&path, body) {
            tracing::warn!(
                "final field csv write failed: {}",
                crate::utils::error::EngineError::Io {
                    path: path.display().to_string(),
                    source,
                }
            );
        }
    }

    /// Full hashes of the elite set exported at stop — the single source
    /// truth for "which nets are the champions" (never file mtimes/glob order).
    pub fn champion_hashes(&self) -> &[String] {
        &self.champions
    }

    /// The champion set: top-`elite_count` live hashes by smoothed fitness.
    /// ONE ranking all elite writers consume, so they can never disagree.
    fn champion_ranked(&self) -> impl Iterator<Item = (String, f32)> {
        self.champion_slice(&self.ranked_live()).into_iter()
    }

    /// [`Self::champion_ranked`] over a ranking the caller already holds.
    pub(crate) fn champion_slice(&self, ranked: &[(String, f32)]) -> Vec<(String, f32)> {
        let k = self.config.elite_count.max(1);
        ranked.iter().take(k).cloned().collect()
    }

    /// Snapshot the champion set into `self.champions` before any writing,
    /// `champion_hashes()` reflects exactly what the writers iterate.
    pub(crate) fn record_champions(&mut self) {
        self.champions = self.champion_ranked().map(|(h, _)| h).collect();
    }
}

/// One CSV line from already-formatted cells, RFC4180-quoted by [`csv_field`].
fn csv_row(cells: &[String]) -> String {
    cells
        .iter()
        .map(|c| crate::engine::format::csv_field(c))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::csv_row;

    /// The export is a CSV, not a `join` with a comma: cells carrying a comma are quoted,
    /// cells carrying a quote are escaped, and no row grows or loses a column.
    #[test]
    fn csv_row_quotes_only_the_cells_that_need_it() {
        let row = csv_row(&[
            "7".to_string(),
            "abc12345".to_string(),
            "0.6348".to_string(),
            "—".to_string(),
            "cx one-point@14, +mut".to_string(),
        ]);
        assert_eq!(row, "7,abc12345,0.6348,—,\"cx one-point@14, +mut\"");
        // The quoted comma is FIELD TEXT: a naive split sees 6 pieces for 5 columns,
        // which is exactly why the quoting is here.
        assert_eq!(row.split(',').count(), 6);
        assert_eq!(crate::engine::format::csv_split_line(&row).len(), 5);
        assert_eq!(csv_row(&["a\"b".to_string()]), "\"a\"\"b\"");
        // An empty grid cell is an empty CSV field, never a missing column.
        assert_eq!(
            csv_row(&["a".to_string(), String::new(), "b".to_string()]),
            "a,,b"
        );
    }
}
