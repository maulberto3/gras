//! Run artifacts: the CHAMPION ranking and the anti-champion (worst) dumps.

use super::core::CoreEngine;
use super::diversity::{shape_of, NetShape};
use super::format::{fmt_cell, fmt_cfg, fmt_family_reading, fmt_micros, fmt_units, ReadingStyle};
use flodl::tensor::Result;

/// Why the list is not simply the bottom of `fit (F)`, stamped into every `worst-*.md`: the file
/// outlives the run, and the criterion reads engine knobs a later reader cannot see.
const WORST_PICK_NOTE: &str = "**Why this net.** The pick is the `spent` criterion the cross/mutation cull evicts on: `spent = Fm⁻ × (1 − Um⁻)` — how badly the net grades (`Fm⁻`: 0.99 = the field's worst, 0.01 = its best) times how much of its updates it has already spent (`1 − Um`: 0.01 at the field's least-updated net, 0.99 at its most). So a weak net the lottery has NOT paid yet ranks BELOW a weak net that trained and stayed weak — the net worth reading, and the reason this list is not just the bottom of `fit (F)`. The place is UN-TILTED: the fit side of `train_ticket` (worst-first / best-first) is undone as `1 − p` before the product, so that leg decides who the band PAYS and never which end counts as bad, while the `U` leg is read whether or not `train_ticket` arms it for tickets. A net the lottery cannot grade (shielded, unread, flat field, lottery off) prints `na` and falls to the tail of this index ordered by `fit (F)` — the pre-criterion dump.";

impl CoreEngine {
    /// Write the champion ranking (`champions.json`, best first).
    pub(crate) fn write_champion_index(&mut self) -> Result<()> {
        self.record_champions();
        // `fitness` stays the RANKING value (the key the next run and this file's readers already
        // agree on); the reading behind it is written out in full so "ranked 1st because
        let entries: Vec<serde_json::Value> = self
            .champion_ranked()
            .enumerate()
            .map(|(rank, (hash, fitness))| {
                let state = self.state.net(&hash);
                let reading = self.race_reading_of(&hash);
                let trend = self.rolling_trend.get(&hash);
                serde_json::json!({
                    "rank": rank + 1,
                    "hash": hash,
                    "fitness": fitness,
                    "fitness_mean": reading.map(|r| r.mean),
                    "fitness_std": reading.map(|r| r.std),
                    // The arithmetic pair keeps its keys (readers already key on them); the pair the SCORE was charged from is APPENDED beside it, so a `median` / `signed_log` run's JSON can…
                    "fitness_centre": reading.map(|r| r.fam_center),
                    "fitness_spread": reading.map(|r| r.fam_spread),
                    "fitness_samples": reading.map(|r| r.n),
                    // The WHOLE-RUN line beside the window's numbers: `fitness_samples` counts the K-reading window, `fitness_readings` every reading the net ever took, and `fitness_slope` is…
                    "fitness_readings": trend.map(|t| t.len()),
                    "fitness_slope": trend.and_then(|t| t.slope()),
                    "mean_type": self.config.ranking.mean_type.to_string(),
                    "params": state.and_then(|s| s.meta.params),
                    "born": state.map(|s| s.entered_at_step),
                })
            })
            .collect();
        let path = self.run_dir.join("champions.json");
        let body = serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string());
        if let Err(source) = std::fs::write(&path, body) {
            tracing::warn!(
                "champion index write failed: {}",
                crate::utils::error::EngineError::Io {
                    path: path.display().to_string(),
                    source,
                }
            );
        } else if self.verbose_detail() {
            crate::utils::cfy::emit(&crate::utils::cfy::card(
                "Champion index",
                &crate::utils::cfy::kv(&[
                    ("file".to_string(), path.display().to_string()),
                    ("ranked".to_string(), entries.len().to_string()),
                ]),
                Some("inheritable by the next run"),
            ));
        }
        Ok(())
    }

    /// Write the top-k champions' topology markdown, one file per champion (`champion-<hash>-<fitness>.md`, the suffix being the risk-adjusted score the file ranked by, rounded…
    pub(crate) fn write_champion_markdown(&mut self, clock: usize) -> Result<()> {
        if !self.config.run.champion_save_topology {
            return Ok(());
        }
        self.record_champions();
        // The champion count is known without consuming the iterator — read
        // it from the config (the same k the iterator caps
        let k = self.config.population.champion_count.max(1);
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
            // `champion-<rank>-<hash>-<F>`: the RANK orders a directory listing (the one thing a reader scrolls these files for), the hash is the identity, and the two decimals are…
            let path = self
                .run_dir
                .join(format!("champion-{:02}-{short}-{fitness:.2}.md", rank + 1));
            let label = if k <= 1 {
                "Champion".to_string()
            } else {
                format!("Champion #{}", rank + 1)
            };
            // The header names both legs: the score it ranked by, and the reading it was charged
            // from — `μ ±σ (SE, n)` in the same spelling as the logs, so a reader never has
            let reading = self.race_reading_of(&hash).map_or_else(
                || crate::utils::cfy::NO_VALUE.to_string(),
                |r| {
                    // The pair the SCORE was charged from (`fam_center` / `fam_spread`), under the family's own nouns — a median run's card redoes its own header instead of printing an…
                    fmt_family_reading(
                        Some(r.fam_center),
                        (r.n > 1).then_some(r.fam_spread),
                        r.n,
                        ReadingStyle::Doc,
                        self.config.ranking.mean_type,
                    )
                },
            );
            // The run seed rides the header: this file outlives the run dir, and the seed
            // one input a later replay cannot reconstruct from the net.
            // ── Facts: every number the stop card prints for this net, in ONE table ── the file
            // outlives the run, so it carries the census, the eval price and the read window the
            // console showed, not only the blueprint. Prices are per SAMPLE (tabular) or per TURN
            // (RL), the same unit the `eval time (µs)` column states in its header.
            let shape = shape_of(&topo);
            let facts = self.net_facts(&hash, &shape, clock, &reading);
            let facts_md = facts_markdown(&facts);
            let md = format!(
                // `updates` sits beside the score, not only in the lineage box at the foot of the
                // file: two nets can share a fitness and differ by whether it cost 2 updates or
                // 200, and the header is what a reader sees first.
                "**{label} · fitness {} = {:.4} · updates {} · window {} · run seed {} · net seed {}**\n\n{facts_md}{}\n\n{}",
                self.fitness.direction().arrow(),
                fitness,
                state.step,
                reading,
                self.header.run_seed,
                state.net_seed,
                crate::utils::markdown::topology_markdown(&topo, None),
                // The net's own birth, as a mermaid graph: a founder is one box,
                // crossover draws its parents above and converges down to this net.
                self.lineage_markdown(&hash, "CHAMPION"),
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
                    "champion markdown write failed: {}",
                    crate::utils::error::EngineError::Io {
                        path: path.display().to_string(),
                        source,
                    }
                ),
            }
        }
        if self.verbose_detail() && !written.is_empty() {
            crate::utils::cfy::emit(&crate::utils::cfy::card(
                "Champion topology (markdown)",
                &crate::utils::cfy::table(&["#", "hash", "file"], &written),
                Some("ready to view"),
            ));
        }
        Ok(())
    }

    /// The `## Facts` rows: every number the stop card prints for one net — census,
    /// eval price, read window. Shared by the CHAMPION and ANTI-CHAMPION markdown so the
    /// two ends of one ranking leave the same record on disk.
    fn net_facts(
        &self,
        hash: &str,
        shape: &NetShape,
        clock: usize,
        reading: &str,
    ) -> Vec<(&'static str, String)> {
        let na = crate::utils::cfy::NO_VALUE.to_string();
        let Some(state) = self.state.net(hash) else {
            return Vec::new();
        };
        // Prices are per SAMPLE (tabular) or per TURN (RL) — the same unit the stop
        // column states in its header.
        let eval = &state.meta.eval;
        let eval_cell = match eval.secs_per_unit() {
            Some(secs) => format!(
                "{} µs / {} ({} timed passes)",
                fmt_micros(secs),
                if eval.unit.is_empty() {
                    "pass"
                } else {
                    eval.unit.as_str()
                },
                eval.passes
            ),
            None => na.clone(),
        };
        let trend = self.rolling_trend.get(hash).and_then(|t| t.slope());
        vec![
            ("hash", hash.to_string()),
            (
                "origin",
                state
                    .created_from
                    .clone()
                    .unwrap_or_else(|| "founder".to_string()),
            ),
            ("born (clock)", state.entered_at_step.to_string()),
            (
                "age (A)",
                (clock.saturating_sub(state.entered_at_step) + 1).to_string(),
            ),
            (
                "params",
                state
                    .meta
                    .params
                    .map(fmt_units)
                    .unwrap_or_else(|| na.clone()),
            ),
            ("nodes", shape.nodes.to_string()),
            ("wires", shape.wires.to_string()),
            ("width", fmt_cell(shape.dim)),
            ("reach", format!("{:.2}", shape.reach)),
            ("depth (D)", shape.depth.to_string()),
            (
                "feature split",
                shape
                    .mask
                    .map_or_else(|| na.clone(), |m| format!("{}%", fmt_cell(m * 100.0))),
            ),
            ("eval time (t)", eval_cell),
            ("updates (U)", state.step.to_string()),
            ("evals (E)", state.measured_clocks().to_string()),
            ("fit (F)", reading.to_string()),
            (
                "slope (S)",
                trend.map_or_else(|| na.clone(), |v| format!("{v:.4}")),
            ),
            ("net seed", state.net_seed.to_string()),
            ("run seed", self.header.run_seed.to_string()),
            ("mean type", self.config.ranking.mean_type.to_string()),
        ]
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
            if self.config.run.save_optim {
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

    /// The anti-champion dump: the nets worth READING, not merely the lowest `fit (F)` — one file per net (`worst-<hash>-<fit>.md`: topology + mermaid graph + lineage) plus a…
    pub(crate) fn write_worst_markdown(&mut self, clock: usize) -> Result<()> {
        if !self.config.run.worst_save_topology {
            return Ok(());
        }
        // The SAME frame the coin dealt from: the ticket chance printed here is the number the
        // next clock would compare against, not a second opinion.
        let frame = self.lottery_frame();
        // The SAME evidence the `spent` cull evicts on: `Fm⁻ × (1 − Um)`, bad AND already paid for. `None` = the lottery cannot grade the net, which is why the tail below still…
        let rows: Vec<(String, f32, usize, Option<f32>)> = self
            .ranked_live()
            .into_iter()
            .map(|(hash, score)| {
                let age = self
                    .state
                    .net(&hash)
                    .map(|s| clock.saturating_sub(s.entered_at_step) + 1)
                    .unwrap_or(0);
                let spent = self.train_lottery_spent_fraction(&hash, frame, self.merit(&hash));
                (hash, score, age, spent)
            })
            .collect();
        // The coin's own favourite, named beside the list: the two questions differ, and hiding
        // that is what let a champion be filed as an anti-champion.
        let favourite = rows
            .iter()
            .filter_map(|(h, _, _, _)| {
                self.train_lottery_parts(h, frame, self.merit(h))
                    .map(|_| (h.clone(), self.train_lottery_p_eff(h, frame)))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)));
        let mut rows = worst_order(rows);
        let k = self.config.population.champion_count.max(1).min(rows.len());
        rows.truncate(k);
        let mut written: Vec<Vec<String>> = Vec::new();
        let mut index = String::from(
            "| # | hash | spent | fit (F) | ticket p | merit | age | depth (D) | file |\n|---|---|---|---|---|---|---|---|---|\n",
        );
        for (rank, (hash, _fraction, age, spent)) in rows.iter().enumerate() {
            let Some(state) = self.state.net(hash) else {
                continue;
            };
            let Ok(topo) = state.topology() else {
                continue;
            };
            let short = &hash[..8.min(hash.len())];
            // `na` when the lottery cannot grade the net (no reading, flat field, shield): the table
            // prints the same dash, because a neutral fraction is not a chance anybody is paid.
            let p = self
                .train_lottery_parts(hash, frame, self.merit(hash))
                .map(|_| self.train_lottery_p_eff(hash, frame));
            let merit = self.merit(hash);
            let reading = self.race_reading_of(hash).map_or_else(
                || crate::utils::cfy::NO_VALUE.to_string(),
                |r| {
                    // The pair the SCORE was charged from (`fam_center` / `fam_spread`), under the family's own nouns — a median run's card redoes its own header instead of printing an…
                    fmt_family_reading(
                        Some(r.fam_center),
                        (r.n > 1).then_some(r.fam_spread),
                        r.n,
                        ReadingStyle::Doc,
                        self.config.ranking.mean_type,
                    )
                },
            );
            // The same shape as the champion files — `worst-<rank>-<hash>-<F>` — so the two ends of one ranking are named alike and both listings sort by place.
            let fit = self.smoothed_fitness_of(hash);
            let score = fit.map_or_else(|| "na".to_string(), |s| format!("{s:.2}"));
            let file = format!("worst-{:02}-{short}-{score}.md", rank + 1);
            let fit_cell = fit.map_or_else(|| crate::utils::cfy::NO_VALUE.to_string(), fmt_cfg);
            let ticket_cell = p.map_or_else(|| crate::utils::cfy::NO_VALUE.to_string(), fmt_cfg);
            let spent_cell = spent.map_or_else(|| crate::utils::cfy::NO_VALUE.to_string(), fmt_cfg);
            // The same `## Facts` table the champion files carry: this file outlives the
            // run too, and an anti-champion is read for exactly the same numbers.
            let shape = shape_of(&topo);
            let facts = self.net_facts(hash, &shape, clock, &reading);
            let facts_md = facts_markdown(&facts);
            let md = format!(
                // `updates` rides the header beside `age`: `spent` says how the net ranks, the
                // COUNT says what the ranking is made of, and the lineage box at the foot of the
                // file already prints it — the two must agree.
                "**Worst #{} · spent {} · fit (F) {} · ticket p {} · merit {} · updates {} · age {} clocks · window {} · run seed {} · net seed {}**\n\n{facts_md}{}\n\n{}\n\n{}",
                rank + 1,
                spent_cell,
                fit_cell,
                ticket_cell,
                merit.map_or_else(|| crate::utils::cfy::NO_VALUE.to_string(), fmt_cfg),
                state.step,
                age,
                reading,
                self.header.run_seed,
                state.net_seed,
                WORST_PICK_NOTE,
                crate::utils::markdown::topology_markdown(&topo, None),
                self.lineage_markdown(hash, "WORST"),
            );
            match std::fs::write(self.run_dir.join(&file), &md) {
                Ok(()) => {
                    index.push_str(&format!(
                        "| {} | {short} | {spent_cell} | {fit_cell} | {ticket_cell} | {} | {age} | {} | [{file}]({file}) |\n",
                        rank + 1,
                        merit.map_or_else(|| crate::utils::cfy::NO_VALUE.to_string(), fmt_cfg),
                        shape.depth,
                    ));
                    written.push(vec![
                        (rank + 1).to_string(),
                        short.to_string(),
                        fit_cell.clone(),
                        file,
                    ]);
                }
                Err(source) => tracing::warn!(
                    "worst markdown write failed: {}",
                    crate::utils::error::EngineError::Io {
                        path: self.run_dir.join(&file).display().to_string(),
                        source,
                    }
                ),
            }
        }
        if !written.is_empty() {
            // The distinction the old dump hid, said out loud once: `fit (F)` picks the list, the
            // ticket prices each row, and the coin's favourite is usually NOT the worst net.
            let favourite_line = favourite.map_or_else(String::new, |(h, p)| {
                format!(
                    "The coin's own favourite is `{}` at ticket p = {} — the ticket prices merit × updates and follows the tilt, while this list ranks badness × updates already spent, so the net with the fewest updates can top the chance and still stay out of the list.\n\n",
                    &h[..8.min(h.len())],
                    fmt_cfg(p),
                )
            });
            let body = format!(
                "# Anti-champion field at clock {clock}\n\nThese are the `champion_count` nets that already BURNED their updates and stayed bad — ranked by the `spent` criterion `Fm⁻ × (1 − Um⁻)`, the same number the `spent` cross/mutation cull evicts on, not by `fit (F)` alone: a weak net the lottery has not paid yet ranks BELOW a weak net that trained and stayed weak. Every row is one net's topology + mermaid graph + lineage; `spent` is the ranking value, `ticket p` is the chance the train-eval lottery gives that net on the NEXT clock (dash = not graded, so it has no chance to name), and `fit (F)` rides along for comparison, and `depth (D)` is the net's longest wire path from the Input node (`Input → H1 → H2 → Output` = `3`), report-only. A row with `na` spent was not graded and is listed after the graded ones, by `fit (F)` ascending.\n\n{favourite_line}{index}"
            );
            let path = self.run_dir.join("worst.md");
            if let Err(source) = std::fs::write(&path, body) {
                tracing::warn!(
                    "worst index write failed: {}",
                    crate::utils::error::EngineError::Io {
                        path: path.display().to_string(),
                        source,
                    }
                );
            }
            if self.verbose_detail() {
                crate::utils::cfy::emit(&crate::utils::cfy::card(
                    "Anti-champion topology (markdown)",
                    &crate::utils::cfy::table(&["#", "hash", "fit (F)", "file"], &written),
                    Some(
                        "the bad nets that already spent their updates (`Fm⁻ × (1 − Um⁻)`); ungraded nets fall back to the lowest `fit (F)`",
                    ),
                ));
            }
        }
        Ok(())
    }

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
        // The DISPLAY header breaks the ticket equation over lines so the console table stays
        // narrow; a CSV header is one line, so the same cells are flattened here.
        let flat: Vec<String> = header.iter().map(|h| h.replace('\n', " ")).collect();
        let mut body = csv_row(&flat);
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

    /// Full hashes of the champion set exported at stop — the single source
    /// truth for "which nets are the champions" (never file mtimes/glob order).
    pub fn champion_hashes(&self) -> &[String] {
        &self.champions
    }

    /// The champion set: top-`champion_count` live hashes by smoothed fitness.
    /// ONE ranking all champion writers consume, so they can never disagree.
    fn champion_ranked(&self) -> impl Iterator<Item = (String, f32)> {
        self.champion_slice(&self.ranked_live()).into_iter()
    }

    /// [`Self::champion_ranked`] over a ranking the caller already holds.
    pub(crate) fn champion_slice(&self, ranked: &[(String, f32)]) -> Vec<(String, f32)> {
        let k = self.config.population.champion_count.max(1);
        ranked.iter().take(k).cloned().collect()
    }

    /// Snapshot the champion set into `self.champions` before any writing,
    /// `champion_hashes()` reflects exactly what the writers iterate.
    pub(crate) fn record_champions(&mut self) {
        self.champions = self.champion_ranked().map(|(h, _)| h).collect();
    }
}

/// Anti-champion order: the HIGHEST `spent` (`Fm⁻ × (1 − Um⁻)`) first — the bad nets the field has already paid for — then the lowest `fit (F)`, the older net, the hash.
/// Field/value pairs printed per `## Facts` row. Two keeps the card short without dragging the
/// long cells (`origin`, `fit (F)`) across the page.
const FACTS_PAIRS_PER_ROW: usize = 2;

/// Render a `## Facts` table. One row list, one spelling, for both ends of the ranking.
///
/// The list is laid out as a GRID — `FACTS_PAIRS_PER_ROW` pairs per row — because the card is read
/// beside the mermaid graph, and eighteen single-pair lines pushed the numbers off the screen.
/// A short final row is padded with blank cells so the table stays a rectangle.
fn facts_markdown(facts: &[(&str, String)]) -> String {
    let cols = FACTS_PAIRS_PER_ROW * 2;
    let mut out = String::from("## Facts\n\n|");
    for c in 0..cols {
        out.push_str(if c % 2 == 0 { " field |" } else { " value |" });
    }
    out.push('\n');
    out.push('|');
    for _ in 0..cols {
        out.push_str("---|");
    }
    out.push('\n');
    for row in facts.chunks(FACTS_PAIRS_PER_ROW) {
        out.push('|');
        for (key, value) in row {
            out.push_str(&format!(" {key} | {value} |"));
        }
        for _ in row.len()..FACTS_PAIRS_PER_ROW {
            out.push_str("  |  |");
        }
        out.push('\n');
    }
    out.push('\n');
    out
}

pub(crate) fn worst_order(
    mut rows: Vec<(String, f32, usize, Option<f32>)>,
) -> Vec<(String, f32, usize, Option<f32>)> {
    rows.sort_by(|a, b| match (a.3, b.3) {
        (Some(x), Some(y)) => y
            .total_cmp(&x)
            .then_with(|| a.1.total_cmp(&b.1))
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| a.0.cmp(&b.0)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => {
            a.1.total_cmp(&b.1)
                .then_with(|| b.2.cmp(&a.2))
                .then_with(|| a.0.cmp(&b.0))
        }
    });
    rows
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
    use super::{csv_row, worst_order};

    /// The export is a CSV, not a `join` with a comma: cells carrying a comma are quoted,
    /// cells carrying a quote are escaped, and no row grows or loses a column.
    #[test]
    fn the_anti_champion_order_is_spent_then_fit_then_age_then_hash() {
        let rows = vec![
            ("b".to_string(), -1.0, 3, Some(0.5)),
            ("a".to_string(), -1.0, 3, Some(0.5)),
            ("c".to_string(), -1.0, 9, Some(0.5)),
            ("d".to_string(), 9.0, 99, Some(0.9)),
            ("e".to_string(), 0.0, 0, None),
            ("f".to_string(), 2.0, 7, None),
        ];
        let ordered = worst_order(rows);
        let got: Vec<&str> = ordered.iter().map(|r| r.0.as_str()).collect();
        assert_eq!(
            got,
            vec!["d", "c", "a", "b", "e", "f"],
            "highest spent first, then lowest fit, the older net, the hash; ungraded nets last"
        );
    }

    /// The fallback is not a tie: a net the lottery cannot grade never outranks a graded one, and
    /// the ungraded tail keeps the pre-criterion order (`fit (F)` ascending).
    #[test]
    fn an_ungraded_net_never_outranks_a_graded_one() {
        let rows = vec![
            ("graded_bad".to_string(), 900.0, 1, Some(0.01)),
            ("ungraded_worse_fit".to_string(), -50.0, 5, None),
        ];
        let ordered = worst_order(rows);
        assert_eq!(ordered[0].0, "graded_bad");
    }

    #[test]
    fn csv_row_quotes_only_the_cells_that_need_it() {
        let row = csv_row(&[
            "7".to_string(),
            "abc12345".to_string(),
            "0.6348".to_string(),
            crate::utils::cfy::NO_VALUE.to_string(),
            "cx one-point@14, +mut".to_string(),
        ]);
        assert_eq!(row, "7,abc12345,0.6348,na,\"cx one-point@14, +mut\"");
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
