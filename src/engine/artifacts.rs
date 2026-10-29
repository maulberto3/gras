//! Run artifacts: the champion (elite) and anti-champion (worst) dumps.
//!
//! These write files to the run dir — they are artifacts, not log lines, so
//! they land at every log level. The ranking helpers (`champion_ranked`,
//! `record_champions`) live here because they exist only to feed the writers;
//! both read the shared `ranked_live()` primitive from `core`. The public
//! `champion_hashes()` read-back means post-race tooling never has to guess
//! champions from file mtimes or glob order.

use super::core::CoreEngine;
use super::format::{ReadingStyle, fmt_cfg, fmt_reading};
use flodl::tensor::Result;

impl CoreEngine {
    /// Write the MACHINE-READABLE elite ranking — `elites.json`, best first,
    /// k = `elite_count`. UNCONDITIONAL: an inheritance receipt, not a log line.
    pub(crate) fn write_elite_index(&mut self) -> Result<()> {
        self.record_champions();
            // `fitness` stays the RANKING value (the key the next run and this file's
            // readers already agree on); the reading behind it is written out in full so
            // "ranked 1st because it is steady, not because it is high" is visible.
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
        // it from the config (the same k the iterator caps at).
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
                // The header names both legs: the score it ranked by, and the reading
                // it was charged from — `μ ±σ (SE, n)` in the same spelling as the logs,
                // so a reader never has to guess whether `±` meant σ or σ/√n.
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
            let md = format!(
                "**{label} · fitness {} (risk-adjusted) = {:.4} · window {} · λ={:.2}**\n\n{}",
                self.fitness.direction().arrow(),
                fitness,
                reading,
                self.config.risk_aversion,
                crate::utils::markdown::topology_markdown(&topo, None),
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
    ///
    /// The weights half of a checkpoint (R4): `nets/<hash>.json` carries the genome, the
    /// counters and the ledger; this carries the coefficients — so `nets/` alone describes
    /// the frontier and a resume can LOAD instead of replaying. Called at the durability
    /// tick BEFORE the manifest, so a manifest never names a net whose weights are absent,
    /// and again at stop. Unconditional: there is no flag, because a checkpoint without
    /// weights is exactly the gap this replaced.
    ///
    /// Cost is bounded by the POPULATION, not the run: one file per live net, overwritten
    /// in place (measured 2.83 MB/net ⇒ pop 8 = 22.6 MB, constant).
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
        }
        Ok(())
    }

    /// Delete `nets/*.safetensors` whose hash is not live.
    ///
    /// Deliberately NOT "delete on cull": a cull can land AFTER a tick, and that tick's
    /// file is the snapshot a resume needs — removing it then would recreate the hole
    /// `ckpt.json` closed for the STATES (`notes/RESUME.md` §4, where 8 live nets became
    /// 2 after a kill). Sweeping at the tick instead keeps disk bounded by the population,
    /// and a crash between a cull and the next tick can only leave files the manifest
    /// ignores.
    pub(crate) fn sweep_stale_net_weights(&self) {
        let live: std::collections::HashSet<String> = self.state.live_hashes().into_iter().collect();
        let dir = self.run_dir.join("nets");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
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

    /// Write the stop-time `Final field` table as `final-field.csv`: the SAME header and
    /// the SAME cells the card printed, one row per individual, in rank order.
    ///
    /// The console table is the report; this is the identical report for tooling (pandas,
    /// a diff between two seeds), so it is written from the PRINTED cells rather than from
    /// a second query over the population — a second query is exactly how a table and its
    /// export start disagreeing about who was worst. Missing values stay `—`, the same
    /// spelling as the card, so the file needs no second legend.
    ///
    /// UNCONDITIONAL at every log level (an artifact, not a log line), and re-written at
    /// every stop — a resumed run replaces it with the field of the run it actually
    /// finished, never leaving a stale table from the pre-kill stop behind.
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

    /// Full hashes of the elite set exported at stop — the single source of
    /// truth for "which nets are the champions" (never file mtimes/glob order).
    pub fn champion_hashes(&self) -> &[String] {
        &self.champions
    }

    /// The champion set: top-`elite_count` live hashes by smoothed fitness. The
    /// ONE ranking all elite writers consume, so they can never disagree.
    fn champion_ranked(&self) -> impl Iterator<Item = (String, f32)> {
        let k = self.config.elite_count.max(1);
        self.ranked_live().into_iter().take(k)
    }

    /// Snapshot the champion set into `self.champions` before any writing, so
    /// `champion_hashes()` reflects exactly what the writers iterate.
    pub(crate) fn record_champions(&mut self) {
        self.champions = self.champion_ranked().map(|(h, _)| h).collect();
    }

    /// When any `worst_save_*` flag is set: save the `elite_count` WORST live nets'
    /// artifacts (`worst-<hash>.md` / `.safetensors`) — the SAME count the elite writers
    /// save, so both ends of the ranking keep k nets and `elite_count` is the one number
    /// that answers "how many of each end does a run keep?". MUST run BEFORE any cull
    /// shrinking.
    ///
    /// The card is emitted when ANY of the two write paths is on: a topology-only run
    /// used to write `worst-<hash>.md` and say nothing at all (the old card lived inside
    /// the safetensors branch), which left the file unexplained in the log.
    pub(crate) fn write_worst_artifacts(&self) -> Result<()> {
        if !self.config.worst_save_topology {
            return Ok(());
        }
        // The "truly worst" set — see `worst_ranked`: spent × gap, not just the lowest
        // score, so a newborn's first bad clock cannot win the title.
        let picks = self.worst_ranked();
        if picks.is_empty() {
            return Ok(()); // nothing ever scored — nothing to dump
        }
        // CFY: one table for the whole bottom-k; each row names the files that landed.
        let mut written: Vec<Vec<String>> = Vec::new();
        for (rank, pick) in picks.iter().enumerate() {
            let short = &pick.hash[..8.min(pick.hash.len())];
            let mut files: Vec<String> = Vec::new();
            if self.config.worst_save_topology {
                if let Some(state) = self.state.net(&pick.hash) {
                    if let Ok(topo) = state.topology() {
                        let path = self.run_dir.join(format!("worst-{short}.md"));
                        let reading = self.race_reading_of(&pick.hash).map_or_else(
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
                        // A single worst keeps the old bare label; a bottom-k numbers
                        // the files, so a reader can tell which of the k they opened.
                        let label = if picks.len() <= 1 {
                            "Worst".to_string()
                        } else {
                            format!("Worst #{} of {}", rank + 1, picks.len())
                        };
                        // The selection arithmetic rides in the header: a reader must be
                        // able to see WHY this net is one of the worst, not just which.
                        let md = format!(
                            "**{label} · fitness {} (risk-adjusted) = {:.4} · window {} · \
                             (D) prop {:.2} (trained {} of the field's {}) × (A) rescaled {:.2} \
                             (field best {:.4} … worst {:.4}) = {:.4}**\n\n{}",
                            self.fitness.direction().arrow(),
                            pick.score,
                            reading,
                            pick.spent,
                            pick.trained,
                            pick.max_trained,
                            pick.gap,
                            pick.best_score,
                            pick.worst_score,
                            pick.spent * pick.gap,
                            crate::utils::markdown::topology_markdown(&topo, None),
                        );
                        match std::fs::write(&path, &md) {
                            Ok(()) => files.push(format!("worst-{short}.md")),
                            Err(source) => tracing::warn!(
                                "worst markdown write failed: {}",
                                crate::utils::error::EngineError::Io {
                                    path: path.display().to_string(),
                                    source,
                                }
                            ),
                        }
                    }
                }
            }
            written.push(vec![
                (rank + 1).to_string(),
                short.to_string(),
                fmt_cfg(pick.score),
                fmt_cfg(pick.spent * pick.gap),
                if files.is_empty() {
                    "—".to_string()
                } else {
                    files.join(" + ")
                },
            ]);
        }
        if self.verbose_detail() {
            crate::utils::cfy::emit(&crate::utils::cfy::card(
                &format!("Worst-net artifacts (bottom-{})", picks.len()),
                    // The columns carry the SAME letters the stop-time `Final field`
                    // table uses (`A` = fit, `E` = inefficiency), so an artifact row and its table
                    // row name the same two numbers, computed by the same
                    // `inefficiency_field()`.
                &crate::utils::cfy::table(
                    &["#", "hash", "A", "E", "files"],
                    &written,
                ),
                Some(&format!(
                    "the top-{} of the inefficiency column (`E = (D) prop × (A) rescaled`): (D) prop = trained / the field's {} (its most-trained net), (A) rescaled = |best − score| / |best − worst|, so a newborn cannot win either leg · the count is `elite_count` ({}), the same k the elite writers save — both ends of the ranking keep the same number of nets (set_worst_save_topology decides WHETHER, never how many) · a net saved as an elite is never also saved as a worst · `—` = not written (no readable blueprint, or a failed export)",
                    picks.len(),
                    picks.first().map(|p| p.max_trained).unwrap_or(0),
                    self.config.elite_count.max(1),
                )),
            ));
        }
        Ok(())
    }

    /// The `elite_count` "truly worst" nets, worst first: the ones that TRAINED the most
    /// and still scored the worst.
    ///
    /// `write_worst_artifacts` used to dump the single lowest score outright, which is
    /// usually a newborn still on its first (unlucky) reading — a bad DRAW, not a bad
    /// architecture. The ranking orders the measured live field by the product of two
    /// normalised numbers:
    ///
    /// ```text
    /// spent = trained / the field's max trained      (0..1 — clocks it actually spent learning)
    /// gap   = |best − score| / |best − worst|        (0..1 — 1 = the field's worst score)
    /// ```
    ///
    /// A newborn has `spent ≈ 1/max_trained`, so it cannot win even at `gap = 1`;
    /// a net that trained most clocks and still sits at the bottom wins on both. Ties
    /// break on the hash, so the saved set never depends on map or ranking order. No new
    /// state: `state.step` already counts real optimizer updates (crown passes and
    /// lottery idles excluded).
    ///
    /// **The count comes from `elite_count`.** The elite writers take the top k of
    /// `ranked_live()`; this takes the bottom k of the probe ranking, so both ends of a
    /// run keep the same number of nets and there is no second "worst count" to keep in
    /// sync. A net saved as an elite is EXCLUDED here (a run must never write both
    /// `elite-<hash>` and `worst-<hash>` for one net), which is why a population smaller
    /// than `2k` yields fewer picks — the same rule the stop-time `✖` mark applies when it
    /// refuses to mark a net that is already marked `★`.
    pub(crate) fn worst_ranked(&self) -> Vec<WorstPick> {
        let Some(field) = self.inefficiency_field() else {
            return Vec::new();
        };
        let champions: Vec<String> = self.champion_ranked().map(|(h, _)| h).collect();
        let mut picks: Vec<WorstPick> = field
            .nets
            .iter()
            .filter(|(hash, _)| !champions.contains(hash))
            .map(|(hash, (score, spent, gap))| WorstPick {
                hash: hash.clone(),
                score: *score,
                spent: *spent,
                gap: *gap,
                trained: self.state.net(hash).map(|s| s.step).unwrap_or(0),
                max_trained: field.max_trained,
                best_score: field.best,
                worst_score: field.worst,
            })
            .collect();
        picks.sort_by(|a, b| {
            (b.spent * b.gap)
                .partial_cmp(&(a.spent * a.gap))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.hash.cmp(&b.hash))
        });
        picks.truncate(self.config.elite_count.max(1));
        picks
    }

    /// The hashes `write_worst_artifacts` would save — the set the stop-time table marks
    /// `✖`. Same call, same order, so the mark and the file cannot disagree.
    pub(crate) fn worst_set(&self) -> Vec<String> {
        self.worst_ranked().into_iter().map(|p| p.hash).collect()
    }

    /// The MEASURED live field under the inefficiency metric
    /// (`E = spent × gap`, see [`Self::worst_ranked`] for the legs).
    ///
    /// ONE computation serves both readers — the `worst-*` artifact pick and the
    /// stop-time `inefficiency` column — so the table can never rank a net by a different
    /// arithmetic than the file it points at. `None` when nothing is ranked yet.
    pub(crate) fn inefficiency_field(&self) -> Option<InefficiencyField> {
        // Best-first, measured nets only — the same ranking every other decision reads.
        let ranked = self.ranked_live();
        let best = ranked.first().map(|(_, s)| *s)?;
        let worst = ranked.last().map(|(_, s)| *s).unwrap_or(best);
        let span = (best - worst).abs();
        let max_trained = self
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| self.state.net(h).map(|s| s.step))
            .max()
            .unwrap_or(0);
        let nets: std::collections::HashMap<String, (f32, f32, f32)> = ranked
            .iter()
            .map(|(hash, score)| {
                let trained = self.state.net(hash).map(|s| s.step).unwrap_or(0);
                // No net trained yet (clock 0-ish): spent is uniform, so the ranking
                // degrades to the lowest score rather than to a random one.
                let spent = if max_trained == 0 {
                    1.0
                } else {
                    trained as f32 / max_trained as f32
                };
                let gap = if span > f32::EPSILON {
                    ((best - score).abs() / span).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                (hash.clone(), (*score, spent, gap))
            })
            .collect();
        Some(InefficiencyField {
            best,
            worst,
            max_trained,
            nets,
        })
    }
}

/// The inefficiency field of the measured live population.
///
/// `E = (D ÷ Dmax) × G`, with `G = |A_best − A| ÷ |A_best − A_worst|`: the numbers the
/// stop-time `inefficiency` column prints and the `worst-*` artifacts are chosen by.
pub(crate) struct InefficiencyField {
    /// Field best risk-adjusted score (`A_best`, the `G` numerator's anchor).
    pub best: f32,
    /// Field worst risk-adjusted score (`A_worst`, the `G` denominator).
    pub worst: f32,
    /// The field's most optimizer steps (`Dmax`).
    pub max_trained: usize,
    /// hash → (score `A`, spent `D ÷ Dmax`, gap `G`), for every MEASURED live net.
    pub nets: std::collections::HashMap<String, (f32, f32, f32)>,
}

impl InefficiencyField {
    /// `E` for one net; `None` when it carries no reading (it is not in the field).
    pub(crate) fn inefficiency(&self, hash: &str) -> Option<f32> {
        self.nets.get(hash).map(|(_, spent, gap)| spent * gap)
    }

    /// The `(D) prop` leg alone (`D ÷ Dmax`) — the half of `E` that is the net's OWN
    /// updates, printed beside `(A) rescaled` so the product is read off the row.
    pub(crate) fn updates_prop(&self, hash: &str) -> Option<f32> {
        self.nets.get(hash).map(|(_, spent, _)| *spent)
    }

    /// The `(A) rescaled` leg alone (`|best − score| ÷ |best − worst|`) — `A` mapped onto
    /// the field's span, i.e. the half of `E` that is NOT the net's own updates, so the
    /// stop-time table can print it and let a reader redo `E = (D) prop × (A) rescaled`
    /// from the row instead of trusting the product.
    pub(crate) fn fit_rescaled(&self, hash: &str) -> Option<f32> {
        self.nets.get(hash).map(|(_, _, gap)| *gap)
    }
}

/// One CSV line from already-formatted cells, RFC4180-quoted by [`csv_field`].
///
/// The cells a table prints ARE the cells the file carries: a `—` is a `—` in both, and a
/// cell holding a comma (a metric label, an origin string) is quoted rather than silently
/// splitting the row into two columns.
///
/// [`csv_field`]: crate::engine::format::csv_field
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
        assert_eq!(csv_row(&["a".to_string(), String::new(), "b".to_string()]), "a,,b");
    }
}

/// One `worst_*` pick with its arithmetic, so each artifact can explain itself.
pub(crate) struct WorstPick {
    pub hash: String,
    /// Risk-adjusted score the pick was measured at.
    pub score: f32,
    /// `trained / max_trained` over the measured live field.
    pub spent: f32,
    /// `|best − score| / |best − worst|` over the same field (1 = the worst score).
    pub gap: f32,
    /// Optimizer updates the pick actually spent.
    pub trained: usize,
    /// The most updates any live net spent (the `spent` denominator).
    pub max_trained: usize,
    pub best_score: f32,
    pub worst_score: f32,
}
