//! Split out of `core/mod.rs` by AGENTS.md rule 19 (a MOVE: no behaviour, no public surface, no
//! renamed method).

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct KnobDrift {
    /// How the log names the knob: `smoothing_window (K)`, `risk_aversion (λ)`, …
    pub(crate) label: &'static str,
    /// The value the run recorded, formatted as the reader will see
    pub(crate) recorded: String,
    /// The value the resume asked
    pub(crate) current: String,
    /// The setter call that reproduces the record, e.g. `set_run_risk_aversion(0.00)`.
    pub(crate) undo: String,
}

impl CoreEngine {
    /// Every BEHAVIOURAL knob whose resume value differs from the record.
    pub(crate) fn behavioural_drift(&self) -> Vec<KnobDrift> {
        let recorded = &self.header.config;
        let current = &self.config;
        let mut drift: Vec<KnobDrift> = Vec::new();
        let mut push = |label: &'static str, rec: String, cur: String, undo: String| {
            if rec != cur {
                drift.push(KnobDrift {
                    label,
                    recorded: rec,
                    current: cur,
                    undo,
                });
            }
        };
        push(
            "smoothing_window (K)",
            recorded.smoothing_window.to_string(),
            current.smoothing_window.to_string(),
            format!("set_run_smoothing_window({})", recorded.smoothing_window),
        );
        push(
            "risk_aversion (λ)",
            fmt_cfg(recorded.risk_aversion),
            fmt_cfg(current.risk_aversion),
            format!("set_run_risk_aversion({})", fmt_cfg(recorded.risk_aversion)),
        );
        push(
            "window_recency (r)",
            fmt_cfg(recorded.window_recency),
            fmt_cfg(current.window_recency),
            format!(
                "set_run_window_recency({})",
                fmt_cfg(recorded.window_recency)
            ),
        );
        // `""` = a header written before the knob: those runs averaged arithmetically,
        // which is what the default spells, so the two compare equal.
        let rec_mean = if recorded.mean_type.is_empty() {
            crate::engine::config::DEFAULT_MEAN_TYPE.to_string()
        } else {
            recorded.mean_type.clone()
        };
        push(
            "mean_type",
            rec_mean.clone(),
            current.mean_type.to_string(),
            format!("set_run_mean_type({rec_mean})",),
        );
        push(
            "train_age",
            if recorded.train_age { "armed" } else { "off" }.to_string(),
            if current.train_age { "armed" } else { "off" }.to_string(),
            format!("set_run_train_age({})", recorded.train_age),
        );
        // `""` = a header written before the knob drew roulette, the default.
        let rec_selection = if recorded.crossover_selection.is_empty() {
            SelectionMethod::Roulette.to_string()
        } else {
            recorded.crossover_selection.clone()
        };
        push(
            "crossover_selection",
            rec_selection.clone(),
            current.crossover_selection.to_string(),
            format!("set_crossover_selection({rec_selection})"),
        );
        push(
            "train_merit (Ω)",
            fmt_cfg(recorded.train_merit),
            fmt_cfg(current.train_merit),
            format!("set_run_train_merit({})", fmt_cfg(recorded.train_merit)),
        );
        push(
            "train_merit_p_min (p_min)",
            fmt_cfg(recorded.train_merit_p_min),
            fmt_cfg(current.train_merit_p_min),
            format!(
                "set_run_train_merit_p_min({})",
                fmt_cfg(recorded.train_merit_p_min)
            ),
        );
        push(
            "train_merit_p_max (p_max)",
            fmt_cfg(recorded.train_merit_p_max),
            fmt_cfg(current.train_merit_p_max),
            format!(
                "set_run_train_merit_p_max({})",
                fmt_cfg(recorded.train_merit_p_max)
            ),
        );
        // Which END of the field the tickets favour. The same idle ledger means
        // different search under the other order, so it is named like Ω.
        push(
            "train_merit inverted",
            recorded.train_merit_inverted.to_string(),
            current.train_merit_inverted.to_string(),
            format!(
                "set_run_train_merit_inverted({})",
                recorded.train_merit_inverted
            ),
        );
        // Crossover's base choice decides WHICH child a roll yields (same draw, different
        // recipient), so adopting the header's value silently would change every birth from here
        push(
            "crossover_fitter_base",
            recorded.crossover_fitter_base.to_string(),
            current.crossover_fitter_base.to_string(),
            format!(
                "set_crossover_fitter_base({})",
                recorded.crossover_fitter_base
            ),
        );
        // Pool drift changes which masks the NEXT newborn gets (the nets already live keep theirs),
        // so it is a behavioural warning like Ω.
        let rec_pool = if recorded.feature_split_pool.is_empty() {
            vec![
                recorded
                    .feature_split
                    .clone()
                    .unwrap_or_else(|| "none".to_string()),
            ]
        } else {
            recorded.feature_split_pool.clone()
        };
        // Drift changes the rows the NEXT crossover child inherits (the nets already
        // live keep theirs), so it is a behavioural warning like the pool above.
        push(
            "drift_prob",
            fmt_cfg(recorded.drift_prob),
            fmt_cfg(current.drift_prob),
            format!("set_topology_drift({})", fmt_cfg(recorded.drift_prob)),
        );
        push(
            "feature_split_pool",
            rec_pool.join(","),
            current.feature_split_pool_names().join(","),
            format!(
                "set_topology_feature_split_pool([{}])",
                rec_pool
                    .iter()
                    .map(|p| format!("\"{p}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
        for (label, setter, rec, cur) in [
            (
                "crossover_cull_policy",
                "set_crossover_cull_policy",
                recorded.crossover_cull_policy.clone(),
                current.crossover_cull_policy.to_string(),
            ),
            (
                "mutation_cull_policy",
                "set_mutation_cull_policy",
                recorded.mutation_cull_policy.clone(),
                current.mutation_cull_policy.to_string(),
            ),
        ] {
            // `""` = a header written before the field: those runs culled `worst`.
            let rec = if rec.is_empty() {
                "worst".to_string()
            } else {
                rec
            };
            push(label, rec.clone(), cur, format!("{setter}({rec})"));
        }
        for (label, setter, rec, cur) in [
            (
                "mutation_probation_steps",
                "set_mutation_probation_steps",
                recorded.mutation_probation_steps,
                current.mutation_probation_steps,
            ),
            (
                "crossover_probation_steps",
                "set_crossover_probation_steps",
                recorded.crossover_probation_steps,
                current.crossover_probation_steps,
            ),
            (
                "elite_join_probation_steps",
                "set_elite_join_probation_steps",
                recorded.elite_join_probation_steps,
                current.elite_join_probation_steps,
            ),
            (
                "elite_crown_lifetime_steps",
                "set_elite_crown_lifetime_steps",
                recorded.elite_crown_lifetime_steps,
                current.elite_crown_lifetime_steps,
            ),
        ] {
            push(
                label,
                rec.to_string(),
                cur.to_string(),
                format!("{setter}({rec})"),
            );
        }
        drift
    }

    /// Load every live net from `nets/` and replay it to its recorded step.
    /// reloads the checkpoint ledger so the history is the SAME.
    pub(crate) fn load_live_frontier(&mut self) -> Result<()> {
        // R6 — refuse only the impossible.
        for drift in self.behavioural_drift() {
            tracing::warn!(
                "race resume: {} differs from the record — the run recorded {}, this resume sets {} \
                 — the search from here is NOT the recorded one; pass {} to replay the recorded race",
                drift.label,
                drift.recorded,
                drift.current,
                drift.undo
            );
        }
        let run_dir = self.run_dir.clone();
        let nets_dir = run_dir.join("nets");
        let mut loaded = 0usize;
        // The frontier as a SET, when the run recorded one (`ckpt.json`). Handled
        // returned first, so the scan below stays exactly the pre-manifest path.
        if let Some(manifest) = crate::state::load_frontier_manifest(&run_dir)? {
            let history = Self::load_history_windows(&run_dir, self.config.smoothing_window);
            let mut replayed: Vec<Vec<String>> = Vec::new();
            let loaded = self.replay_manifest_frontier(&manifest, &history, &mut replayed)?;
            self.report_resume(loaded, &replayed, &nets_dir, &run_dir)?;
            return Ok(());
        }
        let entries = std::fs::read_dir(&nets_dir).map_err(|source| {
            crate::utils::error::EngineError::Io {
                path: nets_dir.display().to_string(),
                source,
            }
        })?;
        // CFY: per-net replay rows collected here, emitted as one table after the load.
        let mut replayed: Vec<Vec<String>> = Vec::new();
        // The smoothing windows themselves are not persisted per net (only its LAST reading
        // but `history.csv` holds every reading the run took.
        let history = Self::load_history_windows(&run_dir, self.config.smoothing_window);
        for entry in entries {
            let path = entry
                .map_err(|source| crate::utils::error::EngineError::Io {
                    path: nets_dir.display().to_string(),
                    source,
                })?
                .path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let raw = std::fs::read_to_string(&path).map_err(|source| {
                crate::utils::error::EngineError::Io {
                    path: path.display().to_string(),
                    source,
                }
            })?;
            let state = NetState::from_json(&raw)?;
            if !state.is_alive {
                tracing::debug!(
                    "race resume: skipping tombstone net {} (culled)",
                    state.hash
                );
                continue;
            }
            // Ghost guard (defensive): a rejected child once leaked as a live state; a net
            // born at step N cannot have trained clocks < N — warn and skip, don't fail.
            let entered = state.entered_at_step;
            if entered > 0 {
                if let Some(lm) = state.last_metrics.as_ref() {
                    if entered > lm.step + 1 {
                        tracing::warn!(
                            "race resume: skipping GHOST net {} (entered_at_step {} but last trained clock {} — a rejected child that leaked into nets/)",
                            state.hash,
                            entered,
                            lm.step
                        );
                        continue;
                    }
                }
            }
            let clock = state
                .last_metrics
                .as_ref()
                .map(|m| m.step)
                .unwrap_or(state.step);
            if let Some(trainer_state) = state.trainer_state.clone() {
                self.trainer
                    .restore_trainer_state(&state.hash, &trainer_state);
            }
            let child = self.load_child_from_weights(state, clock)?;
            let (hash, step) = self.insert_replayed(child, &history);
            loaded += 1;
            replayed.push(vec![hash, step.to_string()]);
        }
        self.report_resume(loaded, &replayed, &nets_dir, &run_dir)?;
        Ok(())
    }

    /// The frontier exactly as `ckpt.json` recorded it: those hashes, at the clocks they had
    /// reached, whatever the directory has said since.
    pub(crate) fn replay_manifest_frontier(
        &mut self,
        manifest: &crate::state::FrontierManifest,
        history: &HashMap<String, Vec<(f32, f32, Option<f32>)>>,
        replayed: &mut Vec<Vec<String>>,
    ) -> Result<usize> {
        if manifest.run_seed != self.header.run_seed {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: ckpt.json belongs to run seed {} but engine.json records {} — the run \
                 dir mixes two runs",
                manifest.run_seed, self.header.run_seed
            ))
            .into());
        }
        let nets_dir = self.run_dir.join("nets");
        // Count first: a pop mismatch is a config error and must be reported as one,
        // not as the weight load's "incomplete checkpoint" on the first net.
        if manifest.nets.len() != self.config.pop_size {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: expected {} live nets in population (per config.pop_size), but found \
                 {} in ckpt.json — pass the pop size this run used, or start a fresh race",
                self.config.pop_size,
                manifest.nets.len()
            ))
            .into());
        }
        let mut loaded = 0usize;
        for entry in &manifest.nets {
            let path = nets_dir.join(format!("{}.json", entry.hash));
            let raw = std::fs::read_to_string(&path).map_err(|source| {
                crate::utils::error::EngineError::InvalidOptions(format!(
                    "resume: ckpt.json lists net {} at clock {} but {} is unreadable ({source}) — \
                     the frontier snapshot is incomplete",
                    entry.hash,
                    entry.step,
                    path.display()
                ))
            })?;
            let mut state = NetState::from_json(&raw)?;
            if state.hash != entry.hash {
                return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                    "resume: {} holds net {} but ckpt.json lists it as {} — the files and the \
                     manifest disagree",
                    path.display(),
                    state.hash,
                    entry.hash
                ))
                .into());
            }
            // How far the file has moved since the tick.
            let file_clock = state
                .last_metrics
                .as_ref()
                .map(|m| m.step)
                .unwrap_or(state.step);
            if file_clock < entry.step {
                return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                    "resume: net {} is recorded at clock {file_clock} but ckpt.json says the \
                     frontier was at {} — the net files predate the manifest (a checkpoint \
                     write failed?); this run cannot be resumed exactly",
                    entry.hash, entry.step
                ))
                .into());
            }
            if file_clock > entry.step {
                // A resume loads weights instead of replaying them, and a loaded weight cannot be
                // rewound: the file on disk is from a LATER clock than the snapshot this checkpoint
                return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                    "resume: net {} on disk is at clock {file_clock}, past the checkpoint's \
                     {} — its weights are newer than the snapshot and cannot be rewound; \
                     resume from a later checkpoint (the stop snapshot) instead",
                    entry.hash, entry.step
                ))
                .into());
            }
            state.is_alive = true;
            // The checkpoint carries the weights, so a resume LOADS instead
            // replaying. The manifest's clock is a consistency check, not work.
            if let Some(trainer_state) = state.trainer_state.clone() {
                self.trainer
                    .restore_trainer_state(&state.hash, &trainer_state);
            }
            let child = self.load_child_from_weights(state, entry.step)?;
            let (hash, step) = self.insert_replayed(child, history);
            replayed.push(vec![hash, format!("{step} (manifest {})", entry.step)]);
            loaded += 1;
        }
        Ok(loaded)
    }

    /// Build a live net from the checkpoint — blueprint + `nets/<hash>.safetensors`, plus
    /// `nets/<hash>.optim` when the run saved optimizer state.
    pub(crate) fn load_child_from_weights(
        &self,
        state: NetState,
        _clock: usize,
    ) -> Result<crate::engine::child::RaceChild> {
        let topo = state.topology()?;
        let mut net = crate::graph::network::Network::build(&topo, self.config.device())?;
        let nets = self.run_dir.join("nets");
        let weights = nets.join(format!("{}.safetensors", state.hash));
        crate::utils::safetensors::load_safetensors(&mut net, &weights).map_err(|e| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: net {} has no loadable weights ({}): {e} — the checkpoint is \
                 incomplete (a failed tick write?)",
                state.hash,
                weights.display()
            ))
        })?;
        let optim = nets.join(format!("{}.optim", state.hash));
        let optimizer = if self.config.save_optim && optim.exists() {
            match self.trainer.restore_optimizer(&net, &optim) {
                Some(o) => o,
                None => {
                    tracing::warn!(
                        "resume: {} exists but this trainer cannot restore optimizer state \
                         — the moments restart (option A)",
                        optim.display()
                    );
                    self.trainer.make_optimizer(&net)
                }
            }
        } else {
            self.trainer.make_optimizer(&net)
        };
        Ok(crate::engine::child::RaceChild {
            state,
            net,
            optimizer,
            // A replayed net's exchange count is in its lineage string, not here:
            // path rebuilds a resumed run, and re-reporting it would double-count.
            cx: None,
        })
    }

    /// Insert a replayed net into the live maps, with its rolling windows seeded from the run's own
    /// recorded readings so the smoothed stats are warm on resume, not cold.
    pub(crate) fn insert_replayed(
        &mut self,
        child: crate::engine::child::RaceChild,
        history: &HashMap<String, Vec<(f32, f32, Option<f32>)>>,
    ) -> (String, usize) {
        let hash = child.state.hash.clone();
        let step = child
            .state
            .last_metrics
            .as_ref()
            .map(|m| m.step)
            .unwrap_or(child.state.step);
        let mut state = child.state.clone();
        state.elite_probation_age = child.state.elite_probation_age;
        state.elite_seat_age = child.state.elite_seat_age;
        self.state.insert(state, None);
        self.networks.insert(hash.clone(), child.net);
        self.optimizers.insert(hash.clone(), child.optimizer);
        let mut buf = RollingBuffer::new(self.config.smoothing_window);
        let mut train_buf = RollingBuffer::new(self.config.smoothing_window);
        let mut eval_buf = RollingBuffer::new(self.config.smoothing_window);
        match history.get(&hash) {
            // The recorded runway, oldest → newest: the same window the live run held.
            Some(rows) if !rows.is_empty() => {
                for (fitness, train_loss, eval_loss) in rows {
                    buf.push(*fitness);
                    train_buf.push(*train_loss);
                    if let Some(eval) = eval_loss {
                        eval_buf.push(*eval);
                    }
                }
            }
            // No `history.csv` (csv_export off, or a run recorded before it): fall back
            // to the single last reading, which is all the net state carries.
            _ => {
                if let Some(m) = self.state.net(&hash).and_then(|s| s.last_metrics.as_ref()) {
                    buf.push(m.fitness);
                    train_buf.push(m.train_loss);
                    if let Some(e) = m.eval_loss {
                        eval_buf.push(e);
                    }
                }
            }
        }
        self.rolling_fitness.insert(hash.clone(), buf);
        self.rolling_train.insert(hash.clone(), train_buf);
        self.rolling_eval.insert(hash.clone(), eval_buf);
        (hash, step)
    }

    /// The one pop guard + resume card both frontiers end
    pub(crate) fn report_resume(
        &self,
        loaded: usize,
        replayed: &[Vec<String>],
        _nets_dir: &std::path::Path,
        run_dir: &std::path::Path,
    ) -> Result<()> {
        if loaded != self.config.pop_size {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: expected {} live nets in population (per config.pop_size), but found \
                 {} live nets in {}",
                self.config.pop_size,
                loaded,
                run_dir.display()
            ))
            .into());
        }
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            "Race resume",
            &crate::utils::cfy::table(&["hash", "replayed step"], replayed),
            Some(&format!("loaded {loaded} nets from {}", run_dir.display())),
        ));
        Ok(())
    }

    /// The last `window` readings per net, oldest → newest, read back from `<run_dir>/history.csv`
    /// (`fitness`, `train_loss`, `eval_loss` columns, located by the header so appended
    pub(crate) fn load_history_windows(
        run_dir: &std::path::Path,
        window: usize,
    ) -> HashMap<String, Vec<(f32, f32, Option<f32>)>> {
        let Ok(raw) = std::fs::read_to_string(run_dir.join("history.csv")) else {
            return HashMap::new();
        };
        let mut lines = raw.lines();
        let header = lines.next().unwrap_or_default();
        let cols = crate::engine::format::csv_split_line(header);
        let find = |name: &str| cols.iter().position(|c| *c == name);
        let (Some(hash_i), Some(train_i), Some(eval_i), Some(fit_i)) = (
            find("hash"),
            find("train_loss"),
            find("eval_loss"),
            find("fitness"),
        ) else {
            return HashMap::new();
        };
        let mut out: HashMap<String, std::collections::VecDeque<(f32, f32, Option<f32>)>> =
            HashMap::new();
        for line in lines {
            let fields = crate::engine::format::csv_split_line(line);
            let needed = hash_i.max(train_i).max(eval_i).max(fit_i);
            if fields.len() <= needed {
                continue;
            }
            let Ok(train_loss) = fields[train_i].parse::<f32>() else {
                continue;
            };
            let Ok(fitness) = fields[fit_i].parse::<f32>() else {
                continue;
            };
            let eval_loss = fields[eval_i].parse::<f32>().ok();
            let q = out.entry(fields[hash_i].to_string()).or_default();
            q.push_back((fitness, train_loss, eval_loss));
            if q.len() > window {
                q.pop_front();
            }
        }
        out.into_iter()
            .map(|(hash, queue)| (hash, queue.into_iter().collect()))
            .collect()
    }

    // ── Population setup ────────────────────────────────────────────
}
