//! The engine's user-facing output: the per-step rollup (`Summ`), the framed in-place table
//! (`Minimal`), and the stop-time summary.

use tracing::info;

use super::core::CoreEngine;
use crate::engine::child::crown_act_count;
use crate::engine::diversity::{NetShape, mean_delta_to, shape_of};
use crate::engine::format::{
    ReadingStyle, fmt_cell, fmt_cell_opt, fmt_census, fmt_cfg, fmt_line, fmt_units,
};
use crate::engine::smoothing::rolling_mean;

mod lineage;

impl CoreEngine {
    /// One rollup line per step: pop size, train-loss mean ± std, the mode's middle
    /// column, and the fitness column. Printed LAST, so it lands at the terminal bottom.
    pub(crate) fn log_step_rollup(&mut self, clock: usize) {
        // Smoothed (K-step rolling mean) population stats — raw per-step
        // values bounce with batch difficulty; the trend is what matters.
        let mut trains = Vec::new();
        let mut evals = Vec::new();
        for h in self.state.live_hashes() {
            if let Some(buf) = self.rolling_train.get(h.as_str()) {
                trains.push(rolling_mean(buf));
            }
            if let Some(buf) = self.rolling_eval.get(h.as_str()) {
                if !buf.is_empty() {
                    evals.push(rolling_mean(buf));
                }
            }
        }
        // Population mean ± population std via the census spelling: every net in the population
        // measured, so this is not a sample and carries no `SE` (see `format::fmt_census`).
        let stats = |v: &[f32]| {
            if v.is_empty() {
                "—".to_string()
            } else {
                let n = v.len() as f32;
                let mean = v.iter().sum::<f32>() / n;
                let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
                fmt_census(mean, var.sqrt(), ReadingStyle::Line)
            }
        };
        // Per-step wall time: the step's own cost (train + evolve), not time
        // since run start. Matters most in RL (one step = many full matches).
        let step_secs = self.step_started_at_wall.elapsed().as_secs_f32();
        // Fitness column: same mean ± std shape so all three read alike.
        let readings: Vec<_> = self
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| self.race_reading_of(h))
            .collect();
        let direction = self.fitness.direction();
        let lambda = self.config.risk_aversion;
        let fits: Vec<f32> = readings
            .iter()
            .map(|r| r.score(direction, lambda))
            .collect();
        let windows: Vec<f32> = readings.iter().map(|r| r.mean).collect();
        let spreads: Vec<f32> = readings.iter().map(|r| r.std).collect();
        // A window with n = 1 has no observable spread — `WindowStats::std` is 0 by construction
        // there, and printing that 0 as `window spread 0.0000` would claim a steadiness nobody
        let spread_readable = readings.iter().any(|r| r.n > 1);
        // A plain average (not `stats`): these are single population aggregates,
        // distribution being described.
        let avg = |v: &[f32]| {
            if v.is_empty() {
                f32::NAN
            } else {
                v.iter().sum::<f32>() / v.len() as f32
            }
        };
        // The mode's middle column: tabular reports the held-out eval loss,
        // RL reports the environment volume it actually played.
        let mid = if !self.trainer.is_rl() {
            format!("eval_loss↓ {}", stats(&evals))
        } else {
            self.step_rl.label(step_secs)
        };
        // Per-step rollup: Summ = compact one-liner; Minimal = framed table (step 2+).
        // The ★ badge names the current elite set in rank order, each with its seat state.
        let freeze_badge = {
            let elites = self.elite_hashes();
            if elites.is_empty() {
                String::new()
            } else {
                let j = self.config.seat_lifetime_steps;
                let names: Vec<String> = elites
                    .iter()
                    .map(|h| {
                        let short = &h[..8.min(h.len())];
                        let seat_age = self.state.net(h).map(|s| s.elite_seat_age).unwrap_or(0);
                        elite_badge_entry(short, seat_age, j)
                    })
                    .collect();
                format!(" │ ★ {}", names.join(" "))
            }
        };
        // The score identity, printed with its own arithmetic: the mean of the per-net `avg ∓
        // λ·spread` equals `window avg ∓ λ·window spread` because λ is a constant.
        let window_avg = avg(&windows);
        let window_spread = if spread_readable {
            avg(&spreads)
        } else {
            f32::NAN
        };
        let sign = match direction {
            crate::utils::fitness::Direction::Maximize => "−",
            crate::utils::fitness::Direction::Minimize => "+",
        };
        // The spread leg of both spellings, `—` until at least one net has read twice.
        let spread = if window_spread.is_finite() {
            fmt_line(window_spread)
        } else {
            "—".to_string()
        };
        // The spread ACROSS the per-net scores: how wide the field is in ranking terms.
        let fits_spread = {
            let n = fits.len();
            if n < 2 {
                "—".to_string()
            } else {
                let mean = avg(&fits);
                let var = fits.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
                fmt_line(var.sqrt())
            }
        };
        let score_cell = if readings.is_empty() {
            format!(
                "score{} avg — (no reading yet) │ window avg —",
                direction.arrow()
            )
        } else {
            format!(
                "score{} avg {} ± {} (σ over nets) = window avg {} {} {} × window spread {} (λ={}) │ window avg {} (spread {})",
                direction.arrow(),
                fmt_line(avg(&fits)),
                fits_spread,
                fmt_line(window_avg),
                sign,
                fmt_cfg(lambda),
                spread,
                fmt_cfg(lambda),
                fmt_line(window_avg),
                spread,
            )
        };
        if self.log_level == crate::engine::config::LogLevel::Minimal {
            self.log_minimal_table(&trains, &evals, &fits, window_avg, window_spread, step_secs);
        } else {
            info!(
                "step {} │ nets {} │ loss↓ {} │ {} │ {} │ {}s{}{}",
                clock,
                self.state.live_count(),
                stats(&trains),
                mid,
                score_cell,
                fmt_line(step_secs),
                freeze_badge,
                self.informative_rollup(),
            );
        }
    }

    /// The configured informative metrics as a rollup segment: the POPULATION MEAN of each,
    /// config order (the order `history.csv` writes its columns in), `—` until somebody reports
    fn informative_rollup(&self) -> String {
        if self.metrics.is_empty() {
            return String::new();
        }
        let parts: Vec<String> = (0..self.metrics.len())
            .map(|i| {
                let values: Vec<f32> = self
                    .state
                    .live_hashes()
                    .iter()
                    .filter_map(|h| self.state.net(h))
                    .filter_map(|s| s.last_metrics.as_ref())
                    .filter_map(|m| m.informative.get(i).copied())
                    .filter(|v| v.is_finite())
                    .collect();
                if values.is_empty() {
                    format!("{} —", self.metrics[i].label())
                } else {
                    let mean = values.iter().sum::<f32>() / values.len() as f32;
                    format!(
                        "{} avg {} ({} net(s))",
                        self.metrics[i].label(),
                        fmt_line(mean),
                        values.len()
                    )
                }
            })
            .collect();
        format!(" │ {}", parts.join(" │ "))
    }

    /// `Minimal` mode: one boxed vitals table per step (step 2+), redrawn in place
    /// terminal. Row 1's tail is mode-dependent (eval_loss / env volume).
    pub(crate) fn log_minimal_table(
        &mut self,
        trains: &[f32],
        evals: &[f32],
        fits: &[f32],
        window_mean: f32,
        window_std: f32,
        step_secs: f32,
    ) {
        let mean = |v: &[f32]| {
            if v.is_empty() {
                f32::NAN
            } else {
                v.iter().sum::<f32>() / v.len() as f32
            }
        };
        let (train_m, eval_m, fit_m) = (mean(trains), mean(evals), mean(fits));
        // Delta vs the previous step's mean; omitted when 0 or when it dies at 2-decimal rounding
        // (the first render sets the baseline).
        let delta = |cur: f32, prev: Option<f32>| match prev {
            Some(p) => {
                let d = cur - p;
                let s = fmt_line(d);
                if !d.is_finite() || s.parse::<f32>() == Ok(0.0) {
                    String::new()
                } else if d > 0.0 {
                    format!(" (∆+{s})")
                } else {
                    format!(" (∆{s})")
                }
            }
            None => String::new(),
        };
        let (pt, pe, pf) = self
            .minimal_prev_means
            .unwrap_or((f32::NAN, f32::NAN, f32::NAN));
        let pop = self.state.live_count();
        let e = self.step_evolve;
        // Rows first, THEN size the box; padding counts chars, not bytes (↓/↑/∆
        // are 3 bytes but one column wide).
        let mut rows: Vec<String> = vec![
            if !self.trainer.is_rl() {
                format!(
                    "pop {:>3} │ train_loss↓ {}{} │ eval_loss↓ {}{}",
                    pop,
                    fmt_line(train_m),
                    delta(train_m, Some(pt)),
                    fmt_line(eval_m),
                    delta(eval_m, Some(pe)),
                )
            } else {
                format!(
                    "pop {:>3} │ train_loss↓ {}{} │ {}",
                    pop,
                    fmt_line(train_m),
                    delta(train_m, Some(pt)),
                    self.step_rl.label(step_secs),
                )
            },
            // The population MEAN of smoothed per-net scores, printed with the arithmetic the Summ
            // line uses — the two sinks must not disagree.
            format!(
                "{} │ culls {} │ inserts {}",
                minimal_score_cell(
                    fit_m,
                    &delta(fit_m, Some(pf)),
                    window_mean,
                    window_std,
                    self.fitness.direction(),
                    self.config.risk_aversion,
                ),
                e.culls,
                e.inserts,
            ),
            format!(
                "crossover fired {} ({} born, {} spent) │ mutation fired {} ({} born)",
                e.cross_fired, e.cross_survived, e.cross_discarded, e.mutate_fired, e.mutate_fired,
            ),
            format!("step took {}s", fmt_line(step_secs)),
        ];
        // Anti-devolution row — only when a knob is on (zero noise by default):
        // elite set and/or the current demotion count.
        let mut guard_cells: Vec<String> = Vec::new();
        if self.config.freeze_seats || self.config.champion_count > 0 {
            let elites = self.elite_hashes();
            let mut names: Vec<String> = elites
                .iter()
                .map(|h| h[..8.min(h.len())].to_string())
                .collect();
            names.sort();
            // An armed crown with no members (pop too small for a seat) must not print
            // a labelled empty row.
            if !names.is_empty() {
                guard_cells.push(format!("champion: {}", names.join(" ")));
            }
        }
        if !guard_cells.is_empty() {
            rows.push(guard_cells.join(" │ "));
        }
        // Display width: count chars, not bytes (↓/↑/∆ are 3 bytes, 1 char).
        let inner = rows
            .iter()
            .map(|r| r.chars().count())
            .max()
            .unwrap_or(0)
            .max(20);
        // `&str` over `&String`: the closure only READS the text, so `&str` suffices
        // and accepts literals too — one deref less.
        let pad = |s: &str| {
            let visible = s.chars().count();
            format!("│ {}{} │", s, " ".repeat(inner.saturating_sub(visible)))
        };
        let rule = |left: char, right: char| format!("{left}{}{right}", "─".repeat(inner + 2));

        // One boxed block handed to the frame sink whole, so it can be redrawn
        // place: the divider splits population state from engine activity.
        let mut frame: Vec<String> = vec![rule('┌', '┐')];
        for (i, row) in rows.iter().enumerate() {
            if i == 2 {
                frame.push(rule('├', '┤'));
            }
            frame.push(pad(row));
        }
        frame.push(rule('└', '┘'));
        super::logging::frame(&frame.join("\n"));
        // Baseline for the next step's deltas.
        self.minimal_prev_means = Some((train_m, eval_m, fit_m));
    }

    /// Stop-time summary: which JSONs were written as the live frontier
    /// the resume command to continue from here.
    pub(crate) fn log_stop_summary(&self, clock: usize) -> flodl::tensor::Result<()> {
        let live = self.state.live_hashes();
        // CFY: the stop block is comfy tables — an artifact/resume card, the accounting matrix,
        // then the elite card.
        let artifact_rows = vec![
            (
                "frontier snapshots".to_string(),
                format!("{} → nets/<hash>.json (one per live net)", live.len()),
            ),
            ("live nets at stop".to_string(), live.len().to_string()),
            ("stop step".to_string(), clock.to_string()),
            ("resume dir".to_string(), self.run_dir.display().to_string()),
            ("resume pop_size".to_string(), live.len().to_string()),
            (
                "smoothing_window (K)".to_string(),
                self.config.smoothing_window.to_string(),
            ),
            (
                "checkpoint_every".to_string(),
                self.config.checkpoint_every.to_string(),
            ),
        ];
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            "Artifacts & resume",
            &crate::utils::cfy::kv(&artifact_rows),
            Some(
                "resume parity: the one-argument resume takes every knob from engine.json — a STRUCTURAL mismatch (mode, pop_size, dataset, trainer blob) is refused, a BEHAVIOURAL one (K, λ, Ω, parent draw, cull policies, probation) resumes and warns with BOTH values, so a caller who wants a different race passes its own config (TabularEngine::resume(dir, data_dir, config, …)) and reads the warning",
            ),
        ));

        // What the run DID comes before what it produced: a population tally
        // rolls matrix, both read from the engine's own persisted counters.
        let (rolls_header, rolls_rows) = accounting_rolls(&self.accounting);
        let accounting_body = format!(
            "{}\n{}",
            crate::utils::cfy::kv(&accounting_population(&self.accounting, live.len())),
            crate::utils::cfy::table(&rolls_header, &rolls_rows),
        );
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            "Run accounting",
            &accounting_body,
            Some(
                "built = nets admitted; discarded = crossover rolls that produced no viable child; blocked = mutation rolls stopped by the cull shield",
            ),
        ));

        // What the DRAW did, not just what it cost: the novelty gate's own reading at birth.
        if self.births.births > 0 || self.births.duplicates > 0 {
            let b = &self.births;
            let dash = || "—".to_string();
            let rows = vec![
                ("immigrants born".to_string(), b.births.to_string()),
                (
                    "candidates spent".to_string(),
                    format!(
                        "{} = {:.2} per birth · max {}",
                        b.draws,
                        b.draws_per_birth(),
                        b.draws_max
                    ),
                ),
                (
                    "re-drawn (first refused)".to_string(),
                    b.redrawn.to_string(),
                ),
                (
                    "Δ to nearest archive".to_string(),
                    match (b.mean_delta(), b.delta_min) {
                        (Some(mean), Some(min)) => format!(
                            "mean {} · min {} over {} judged",
                            fmt_cfg(mean),
                            fmt_cfg(min),
                            b.delta_births
                        ),
                        _ => dash(),
                    },
                ),
                ("novelty floor (Δ)".to_string(), fmt_cfg(self.novelty.min())),
                (
                    "archive (blueprints seen)".to_string(),
                    self.novelty.len().to_string(),
                ),
                (
                    "crossover duplicate(s) discarded".to_string(),
                    b.duplicates.to_string(),
                ),
            ];
            crate::utils::cfy::emit(&crate::utils::cfy::card(
                "Births & novelty",
                &crate::utils::cfy::kv(&rows),
                Some(
                    "Δ = multiset-Jaccard distance from the child's blueprint to the CLOSEST one the run has ever seen (0 = identical, 1 = nothing in common); a mutation draw spends up to `mutation_novelty_draws` candidates to clear the floor and takes the most novel refusal when it cannot",
                ),
            ));
        }

        // NOT here: the final-field TABLE is an artifact first — it is written at every log level,
        // `None` included — so `finish_race` renders it after this block, gating only the card
        Ok(())
    }

    /// The stop-time summary of the WHOLE measured field: one table, every live net, ranked by FIT,
    /// with the saved elite set marked in place.
    pub(crate) fn log_final_field(&self, clock: usize, verbose: bool) {
        // ONE ranking for the whole card.
        let ranked = self.ranked_live();
        let mut rows: Vec<(String, Option<f32>)> = ranked
            .iter()
            .map(|(hash, fit)| (hash.clone(), Some(*fit)))
            .collect();
        // EVERY individual, not only the measured ones: a live net with no reading yet still
        // exists, and dropping it would make the table's count disagree with the pop size
        let measured = rows.len();
        let seen: std::collections::HashSet<String> =
            rows.iter().map(|(hash, _)| hash.clone()).collect();
        let mut unmeasured: Vec<String> = self
            .state
            .live_hashes()
            .into_iter()
            .filter(|hash| !seen.contains(hash))
            .collect();
        unmeasured.sort();
        rows.extend(unmeasured.into_iter().map(|hash| (hash, None)));
        // No live net at all ⇒ nothing to print: an empty table says less than nothing.
        if rows.is_empty() {
            return;
        }
        let k = self.config.champion_count.max(1); // report at least the champion
        let top: Vec<(String, f32)> = self.champion_slice(&ranked);
        let mut marks: std::collections::HashMap<String, &'static str> =
            std::collections::HashMap::new();
        // ONE glyph answers "what did the run do with this net", and the crown is part
        // answer: `★` = saved as a CHAMPION AND seated in the acting crown (it acted), `☆` = saved,
        let mut barred = 0usize;
        for (hash, _) in &top {
            let seated = !self.config.freeze_seats || self.frozen_crown.contains(hash);
            if !seated {
                barred += 1;
            }
            marks.insert(hash.clone(), if seated { "★" } else { "☆" });
        }
        let arrow = self.fitness.direction().arrow();
        // The lottery field the LAST clock drew from (`dealt_clock` = the clock the nets actually
        // stepped

        // ONE exam pass, here, for the reported nets: the same scorer the race ranked with, over
        // rows no ranking clock ever read.
        let exam_fits: Vec<f32> = top.iter().map(|(_, fit)| *fit).collect();
        let exam_report = self.exam_report(
            &top.iter().map(|(h, _)| h.clone()).collect::<Vec<_>>(),
            &exam_fits,
        );
        let exam_scores = exam_report
            .as_ref()
            .map(|r| r.scores.clone())
            .unwrap_or_default();
        if let Some(report) = &exam_report {
            self.write_exam_json(report, &top);
        }
        // The plan comes AFTER the exam: whether `exam (X)` is a column is a fact the exam
        // pass decides, and the header, the rows and the legend all read it here.
        let plan = self.stop_plan(&top, exam_report.is_some());
        // The lottery field the LAST clock drew from (`dealt_clock` = the clock the nets actually
        // stepped
        let lottery = self.lottery_frame(plan.dealt_clock);
        let (table, stats, grid) = self.stop_card(&rows, &plan, &marks, &exam_scores, lottery);
        let sign = match self.fitness.direction() {
            crate::utils::fitness::Direction::Maximize => "−",
            crate::utils::fitness::Direction::Minimize => "+",
        };
        let (eligible, probation, age_blocked) = self.elite_gate_counts(plan.dealt_clock);
        let baseline = self.stop_baseline(clock);
        // The legend is a TABLE that assigns each column its LETTER and states the relations
        // those letters: `A = mean ∓ λ × spread` is checked against the `fit`, `mean` and `spread`
        let legend_ctx = LegendCtx {
            sign,
            lambda: self.config.risk_aversion,
            window: self.config.smoothing_window,
            recency: self.config.window_recency,
            mean_type: self.config.mean_type,
            inverted: self.config.train_merit_inverted,
            age_leg: self.train_age(),
            omega: self.train_merit_omega(),
            p_min: self.train_merit_p_min(),
            p_max: self.train_merit_p_max(),
            tilt: self.lottery_tilt(),
            direction: self.fitness.direction(),
            lottery,
            k,
            dealt_clock: plan.dealt_clock,
            crossover_probation: self.config.crossover_probation_steps,
            mutation_probation: self.config.mutation_probation_steps,
            join_probation: self.config.seat_join_probation_steps,
            crown_lifetime: self.config.seat_lifetime_steps,
        };
        // A column that is not printed has no legend row: an explanation for a cell the reader
        // cannot find is noise.
        let legend_rows: Vec<(String, String)> = stop_legend_rows(&legend_ctx)
            .into_iter()
            .filter(|(key, _)| {
                !STOP_HEADER_FIXED.contains(&key.as_str())
                    || plan.columns.iter().any(|c| *c == key.as_str())
            })
            .collect();
        let legend = crate::utils::cfy::table_wrapped(&["abbrev", "description"], &legend_rows);
        // A barred elite is only named when there is one: `☆ 0` on a healthy run is noise.
        let seat_note = if barred > 0 {
            format!(" · ☆ {barred}")
        } else {
            String::new()
        };
        // The run's seed rides the card: it is the one input a replay cannot recover
        // once the run dir is gone, and the console output is what a reader keeps.
        let run_seed = self.header.run_seed;
        // The exam line states its size and its AGREEMENT with the score the table ranks
        let exam_note = match &exam_report {
            Some(r) => match r.correlation {
                Some(corr) => format!(
                    "exam: {} row(s) no clock read · r(fit, exam) = {} over {} net(s) · report only",
                    r.rows,
                    format!("{corr:.3}"),
                    r.scores.values().filter(|v| v.is_some()).count()
                ),
                None => format!(
                    "exam: {} row(s) no clock read · r not reported (fewer than 3 scored nets, or no spread) · report only",
                    r.rows
                ),
            },
            None => "exam: none supplied (`exam/` next to the dataset turns this column on — or a `test/` beside a ranking `val/`, which is the row set no clock read)".to_string(),
        };
        // The FILE first, and unconditionally: `final-field.csv` is run data, written
        // from the very cells the card would print (one `StopPlan`, no second query).
        self.write_final_field_csv(&plan.header, &grid);
        let field_card = crate::utils::cfy::card(
            &format!(
                "Final field ({} nets — {} measured, ranked by F{arrow} · ★ {}{seat_note})",
                rows.len(),
                measured,
                top.len() - barred,
            ),
            &format!("{legend}\n{table}"),
            Some(&format!(
                "Δcrown = mean multiset-Jaccard distance to the {} reported champion(s): 0.00 = structurally identical to one of them, 1.00 = shares no gene with any · Δpop = mean distance to every other live net: 0.00 = a clone of the crowd, 1.00 = unlike everything alive · read the two AGAINST the baseline: Δcrown ≈ Δpop means the champion is structurally ordinary (selection is ranking numbers, not designs), Δcrown below it means the champions are a structural cluster · {} · {baseline} · crown {}/{} seats · {} eligible · {} gated (join window or cull shield, {} clocks) · {} age-blocked ({} clocks) · stopped at clock {clock} · run seed {run_seed} (`--seed {run_seed}` replays this run bit-for-bit) · the same table is written to `final-field.csv` ({} row(s) under this header) · {exam_note}",
                plan.crown.len(),
                stats.line(),
                self.frozen_crown.len(),
                self.config
                    .champion_count
                    .min(self.state.live_count().saturating_sub(1)),
                eligible,
                probation,
                self.config.seat_join_probation_steps,
                age_blocked,
                self.config.seat_lifetime_steps,
                grid.len(),
            )),
        );
        if !verbose {
            return;
        }
        // The table is the LAST thing the console prints.
        crate::utils::cfy::emit(&field_card);
    }

    /// The plan the ONE stop-time table renders from: one header, one crown, one live
    /// population, one clock.
    fn stop_plan(&self, top: &[(String, f32)], has_exam: bool) -> StopPlan {
        // The crown `Δcrown` measures to, in elite rank order; `crown_idx` maps a hash to its index
        // in it, so a champion's own row can skip itself.
        let mut shapes: std::collections::HashMap<String, NetShape> =
            std::collections::HashMap::new();
        for hash in self.state.live_hashes() {
            if let Some(shape) = self.stop_shape(&hash) {
                shapes.insert(hash, shape);
            }
        }
        let mut crown: Vec<NetShape> = Vec::new();
        let mut crown_idx: std::collections::HashMap<String, Option<usize>> =
            std::collections::HashMap::new();
        for (hash, _) in top {
            match shapes.get(hash) {
                Some(shape) => {
                    crown_idx.insert(hash.clone(), Some(crown.len()));
                    crown.push(shape.clone());
                }
                None => {
                    crown_idx.insert(hash.clone(), None);
                }
            }
        }
        // The `Δpop` reference set: the same parsed shapes, taken in the state's live
        // order so the pair sampling stays the one a replay expects.
        let mut population: Vec<(String, NetShape)> = Vec::new();
        for hash in self.state.live_hashes() {
            if let Some(shape) = shapes.remove(&hash) {
                population.push((hash, shape));
            }
        }
        // The `eval loss` column is a column only when a net in the field carries one.
        let has_eval = top.iter().any(|(hash, _)| {
            self.state
                .net(hash)
                .and_then(|s| s.last_metrics.as_ref())
                .and_then(|m| m.eval_loss)
                .is_some()
        });
        // The columns this run can actually fill: the two cells only an armed age leg reads,
        // the exam, are columns only when the run supplies them.
        let mut columns = printed_columns(self.train_age(), has_exam);
        let fit = fit_column(self.config.mean_type);
        for c in columns.iter_mut() {
            if *c == COL_FIT {
                *c = fit;
            }
        }
        let header = self.stop_header(&columns, has_eval);
        StopPlan {
            header,
            columns,
            has_eval,
            has_exam,
            dealt_clock: self.step_clock().saturating_sub(1),
            crown,
            crown_idx,
            population,
        }
    }

    /// The table's header, in NARRATIVE order: what the net IS (`#`/`hash`/`origin`, age, census) →
    /// how it is RANKED (fit and its min-max place) → what the coin paid it (`age`, the youth leg,
    fn stop_header(&self, columns: &[&'static str], has_eval: bool) -> Vec<String> {
        let mut header: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        // TWO setters shape these cells, so BOTH are read here: the tilt mirrors the place, and
        // the age leg decides whether `Ym` is a factor at all. One helper per cell, so the header
        // cannot name a symbol the row does not print.
        let (place, ticket) = (
            fit_minmax_column(self.config.train_merit_inverted),
            ticket_column(self.config.train_merit_inverted, self.train_age()),
        );
        for cell in header.iter_mut() {
            if cell == COL_FIT_MINMAX {
                *cell = place.to_string();
            } else if cell == COL_TICKET {
                *cell = ticket.to_string();
            }
        }
        if has_eval {
            header.push("eval loss".to_string());
        }
        for m in &self.metrics {
            // `m:` marks the informative band: measured on the same eval batch as `fit (F)`,
            // printed, and never ranked.
            header.push(format!("m:{}", m.label()));
        }
        header
    }

    /// Render the stop-time table from the shared plan: one row per ranked net, and the field's own
    /// Δ means.
    fn stop_card(
        &self,
        rows: &[(String, Option<f32>)],
        plan: &StopPlan,
        marks: &std::collections::HashMap<String, &'static str>,
        exam: &std::collections::HashMap<String, Option<f32>>,
        lottery: Option<super::step::LotteryFrame>,
    ) -> (comfy_table::Table, CardStats, Vec<Vec<String>>) {
        let mut table = comfy_table::Table::new();
        table.load_preset(comfy_table::presets::UTF8_FULL_CONDENSED);
        table.set_header(plan.header.clone());
        // The same cells, kept: the CSV export is written from THIS grid, so the printed
        // table and the file can never describe two different fields.
        let mut grid: Vec<Vec<String>> = Vec::with_capacity(rows.len());
        let mut crowns: Vec<Option<f32>> = Vec::with_capacity(rows.len());
        let mut pops: Vec<Option<f32>> = Vec::with_capacity(rows.len());
        // The RANK counts the ranked nets only: an unmeasured net prints `—` there (it has
        // no FIT to be ranked by), so the numbers keep meaning "rank in the race".
        let mut rank = 0usize;
        // The plan already parsed every live net's blueprint for `Δpop`.
        let shapes: std::collections::HashMap<&str, &NetShape> = plan
            .population
            .iter()
            .map(|(h, s)| (h.as_str(), s))
            .collect();
        for (hash, fit) in rows {
            if fit.is_some() {
                rank += 1;
            }
            let self_idx = plan.crown_idx.get(hash).copied().flatten();
            // `None` on an unreadable blueprint, which the cells render `—` rather than pretending
            // the net is structureless.
            let shape = shapes.get(hash.as_str()).copied();
            crowns.push(delta_crown(shape, plan, self_idx));
            pops.push(delta_pop(shape, hash, plan));
            let cells = self.stop_row(StopRow {
                rank: fit.is_some().then_some(rank),
                mark: marks.get(hash).copied().unwrap_or("·"),
                hash,
                fit: *fit,
                shape,
                plan,
                exam: exam.get(hash).copied().flatten(),
                lottery,
            });
            table.add_row(cells.inlined);
            // `final-field.csv` carries no verdict cell: every column stays the plain
            // value (or `—`), and the marks live in the console card
            grid.push(cells.plain);
        }
        (
            table,
            CardStats {
                delta_crown: mean_of(&crowns),
                delta_pop: mean_of(&pops),
            },
            grid,
        )
    }

    /// ONE row of the stop-time table: the cells, in header order, with the single `—`
    /// rule for "no value".
    fn stop_row(&self, row: StopRow<'_>) -> StopCells {
        let state = self.state.net(row.hash);
        // ONE window read for the whole row: the `fit (F)` cell, the `A/B` ratio derived
        // from it and the three window legs printed beside
        let reading = self.race_reading_of(row.hash);
        let score = reading
            .map(|r| r.score(self.fitness.direction(), self.config.risk_aversion))
            .or(row.fit);
        // The lottery's three grading numbers, derived ONCE for the three cells that print them;
        // `None` = the lottery did not grade this net (off, exempt, no reading, flat field),
        let parts = row.lottery.and_then(|frame| {
            self.train_lottery_parts(row.hash, row.plan.dealt_clock, Some(frame), score)
        });
        // TWO different counts, deliberately: `M` is what the table PRINTS (clocks measured, `U +
        // O`), `U` (`state.step`) is the learning budget a ticket bought.
        let measured = state.map(|s| s.measured_clocks());
        let metrics = state.and_then(|s| s.last_metrics.as_ref());
        let born = state.map(|s| s.entered_at_step).unwrap_or(0);
        // Clocks lived INCLUSIVE of the birth clock and the last run — the convention
        // freeze lines use (`clock − entered_at_step`).
        let age = row.plan.dealt_clock.saturating_sub(born) + 1;
        let shape = row.shape;
        // The saved-set mark plus the net's GATE LETTERS, in one cell: the mark says what the run
        // DID with the net, the letters say what the net is still subject
        let shielded = self.on_probation(row.hash, row.plan.dealt_clock);
        let gate_born = if shielded { "b" } else { "" };
        let mut gate_seat = String::new();
        if self.config.freeze_seats {
            let join_k = self.config.seat_join_probation_steps;
            let term_j = self.config.seat_lifetime_steps;
            if let Some(s) = state {
                // `j` is suppressed while the born shield is on: the two windows run on the same
                // clock, so `bj` was two letters for one bar — and it read as "born AND already
                if s.elite_probation_age < join_k {
                    if !shielded {
                        gate_seat.push('j');
                    }
                } else if term_j > 0 && s.elite_seat_age >= term_j {
                    gate_seat.push('t');
                }
            }
        }
        let mut cells = vec![
            row.rank.map(|r| r.to_string()).unwrap_or_else(dash),
            row.hash[..8.min(row.hash.len())].to_string(),
            // ── what this net IS ─────────────────────────────────────── `origin` beside the hash:
            // where it came from, before the numbers it would otherwise be blamed
            origin_cell(state.and_then(|s| s.created_from.as_deref()), born),
            // `40.2K`, not `40242`: a parameter count is read at a glance
            // (the exact value stays in the net's JSON and `engine.json`).
            state
                .and_then(|s| s.meta.params)
                .map(fmt_units)
                .unwrap_or_else(|| "?".to_string()),
            shape.map(|s| s.nodes.to_string()).unwrap_or_else(dash),
            shape.map(|s| s.wires.to_string()).unwrap_or_else(dash),
            // Mean hidden width is a MEASURED number like every other cell, so it takes the band
            // rule (`fmt_cell`) instead of a hand-picked `{:.1}` — a 3.5-wide net used to print
            shape.map(|s| fmt_cell(s.dim)).unwrap_or_else(dash),
            // ── how it is RANKED ─────────────────────────────────────── The window's THREE legs,
            // one column each: `Wm ∓ λ·Ws` is redoable from the row.
            reading.map(|r| fmt_cell(r.mean)).unwrap_or_else(dash),
            // `0.0000` at `n = 1`, never `—`: one reading has no observable spread
            reading
                .map(|r| fmt_cell(if r.n > 1 { r.std } else { 0.0 }))
                .unwrap_or_else(dash),
            // The ranking value the two legs make, printed after them:
            score.map(fmt_cell).unwrap_or_else(dash),
            // The coin's three grading numbers, from the SAME frame and score as the ticket cell
            // below — one read, so the row cannot show a fraction that disagrees
            parts.map(|(f, _, _)| fmt_cell(f)).unwrap_or_else(dash),
            // `A` (clocks lived) and `U` (optimizer steps run) are separate columns: `M` spends
            // `U`, and `U ≠ A` whenever the crown or the lottery withheld a clock.
            age.to_string(),
        ];
        // `youth minmax (Ym)` is a column only when the age leg is ARMED: with it OFF the youth
        // 1, so it could only repeat a neighbour's value.
        if self.train_age() {
            cells.push(parts.map(|(_, y, _)| fmt_cell(y)).unwrap_or_else(dash));
        }
        // The chance the coin gave this net on the LAST dealt clock — the same `p_eff` the debug
        // card prints.
        cells.push(self.stop_ticket_p(row.hash, row.plan.dealt_clock, row.lottery, score));
        // ── what its learning COST ─────────────────────────────────────
        // The two ADDENDS, then the sum: `B = U + O` reads across three adjacent cells.
        cells.push(state.map(|s| s.step.to_string()).unwrap_or_else(dash));
        // Act-only clocks: the crown trained and was scored, its update was discarded.
        cells.push(
            state
                .map(|s| crown_act_count(s).to_string())
                .unwrap_or_else(dash),
        );
        cells.push(measured.map(|t| t.to_string()).unwrap_or_else(dash));
        // Clocks the lottery left IDLE (no train, no eval) — after the sum, so the payment
        // and the withholding are not interleaved.
        cells.push(
            state
                .map(|s| s.idle_clocks().to_string())
                .unwrap_or_else(dash),
        );
        // The exam: selection's number's honest twin, a column only when the run supplied one
        // still `—` for a net outside the reported set — the exam costs a forward pass per net,
        if row.plan.has_exam {
            cells.push(row.exam.map(fmt_cell).unwrap_or_else(dash));
        }
        debug_assert_eq!(
            cells.len(),
            row.plan.columns.len(),
            "the row must be as wide as the columns the header prints"
        );
        // ── the verdict glyphs, each INSIDE the column that decided it ── Appended to the value,
        // never substituted, and the numeric prefix stays parseable for a reader: `494.7 ★j` says
        let col = |name: &str| {
            row.plan
                .columns
                .iter()
                .position(|c| *c == name)
                .expect("every glyph target is a printed column")
        };
        // A glyph is a WORD beside the number, never part of it: one space keeps `5b` from reading
        // as a unit (`5 b`), and keeps the value the thing the eye lands on when a row is scanned.
        let mut inlined = cells.clone();
        let mut seat_marks = String::new();
        if let Some(seat) = match row.mark {
            "★" => Some('★'),
            "☆" => Some('☆'),
            _ => None,
        } {
            seat_marks.push(seat);
        }
        seat_marks.push_str(&gate_seat);
        if !seat_marks.is_empty() {
            let fit = col(fit_column(self.config.mean_type));
            inlined[fit].push(' ');
            inlined[fit].push_str(&seat_marks);
        }
        if !gate_born.is_empty() {
            inlined[col(COL_AGE)].push(' ');
            inlined[col(COL_AGE)].push('b');
        }
        let mut tail: Vec<String> = Vec::new();
        if row.plan.has_eval {
            tail.push(fmt_cell_opt(&metrics.and_then(|m| m.eval_loss)));
        }
        // One cell per configured informative metric, in config order — the same order
        // `history.csv` writes its columns
        for i in 0..self.metrics.len() {
            tail.push(
                metrics
                    .and_then(|m| m.informative.get(i).copied())
                    .map(fmt_cell)
                    .unwrap_or_else(dash),
            );
        }
        cells.extend(tail.iter().cloned());
        inlined.extend(tail);
        // The CSV's own verdict column: the export keeps every numeric cell numeric, so the whole
        // verdict rides ONE trailing cell there (the table spells it across the columns instead).
        StopCells {
            plain: cells,
            inlined,
        }
    }

    /// The ticket cell: the chance the coin gave `hash` on the LAST dealt clock — `T = lo + (hi −
    /// lo) × Fm × Ym`, the same `p_eff` the per-clock debug card prints, computed from the same
    fn stop_ticket_p(
        &self,
        hash: &str,
        clock: usize,
        lottery: Option<super::step::LotteryFrame>,
        score: Option<f32>,
    ) -> String {
        let Some(frame) = lottery else {
            return dash();
        };
        if self
            .lottery_abstain(hash, clock, Some(frame), score)
            .is_some()
        {
            return dash();
        }
        fmt_cell(self.train_lottery_p_eff_from(hash, clock, Some(frame), score))
    }

    /// One blueprint parse → the census legs plus the gene multiset the Δ columns compare.
    fn stop_shape(&self, hash: &str) -> Option<NetShape> {
        self.state
            .net(hash)
            .and_then(|s| s.topology().ok())
            .map(|t| shape_of(&t))
    }

    /// The population's own diversity reading, for the stop-time footnotes: without it a card's
    /// `Δpop` has only the other card to be read against.
    fn stop_baseline(&self, clock: usize) -> String {
        match crate::engine::diversity::measure(&self.state, self.header.run_seed, clock) {
            Some(rep) => format!(
                "population baseline: mean pairwise Δ {} over {} pair(s) · {} of {} blueprint(s) distinct",
                fmt_cfg(rep.mean_delta),
                rep.pairs,
                rep.distinct,
                rep.live
            ),
            None => "population baseline: — (fewer than two nets to compare)".to_string(),
        }
    }

    /// The training lottery's per-net card for ONE clock — the DEBUG view (the caller gates
    /// the debug filter), printed at the START of the clock's step, not once at stop,
    pub(crate) fn log_lottery_card(&self, clock: usize, frame: super::step::LotteryFrame) {
        // (exempt?, p, row): the nets the lottery actually grades come first, so the exempt rows
        // (which all grade 1.00 by construction) cannot bury the graded field at the top
        let mut rows: Vec<(bool, f32, Vec<String>)> = self
            .state
            .live_hashes()
            .into_iter()
            .map(|h| {
                let state = self.state.net(&h);
                // ONE window read per net, per clock: the score, its divisor, the ratio,
                // probability mapped from that ratio and the coin that probability implies all come
                let smt = self.smoothed_fitness_of(&h);
                // The merit is the score ITSELF; `F/U` rides beside it as the efficiency
                // clocks the net got (`U`, not `M`: a crown act pass costs no learning budget).
                let measured = state.map(|s| s.measured_clocks()).unwrap_or(0);
                let merit = Self::merit_from(smt);
                let ratio = state
                    .filter(|s| s.step > 0)
                    .and_then(|s| smt.map(|a| a / s.step as f32))
                    .filter(|v| v.is_finite());
                // The BLENDED probability — the one the coin compares against —
                // so `p` and the `ticket?` column beside it cannot disagree.
                let p = self.train_lottery_p_eff_from(&h, clock, Some(frame), merit);
                // Why a net is outside the lottery's reach, in one word:
                // crown and the born shield always train (p = 1.00).
                let exempt = if self.frozen_crown.contains(&h) {
                    "crown"
                } else if self.on_probation(&h, clock) {
                    "shield"
                } else {
                    "—"
                };
                // `smt / D` is ONE quantity, so it prints as its two legs and the ratio between
                // them: the numerator here, the divisor (`updates`) and the quotient beside each
                let trains =
                    !self.frozen_crown.contains(&h) && self.train_lottery_draws_at(&h, clock, p);
                let row = vec![
                    h[..8.min(h.len())].to_string(),
                    exempt.to_string(),
                    smt.map(fmt_cell).unwrap_or_else(|| "—".to_string()),
                    if state.is_some() {
                        measured.to_string()
                    } else {
                        "—".to_string()
                    },
                    ratio.map(fmt_cfg).unwrap_or_else(|| "—".to_string()),
                    fmt_cfg(p),
                    if trains { "yes" } else { "no" }.to_string(),
                    state
                        .map(|s| s.idle_clocks().to_string())
                        .unwrap_or_else(|| "—".to_string()),
                ];
                (exempt != "—", p, row)
            })
            .collect();
        rows.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        let idled: usize = self
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| self.state.net(h))
            .map(|s| s.idle_clocks())
            .sum();
        // The field's own bounds, converted back into the COLUMN's units (the frame is stored
        // direction-adjusted): `best` is the value `p = 1` earns in either direction,
        let (best, worst) = match self.fitness.direction() {
            crate::utils::fitness::Direction::Maximize => (frame.best, frame.worst),
            crate::utils::fitness::Direction::Minimize => (-frame.best, -frame.worst),
        };
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            &format!(
                "Training lottery (experimental, Ω {}, {}, band [{}, {}]) — step {clock}",
                fmt_cfg(self.train_merit_omega()),
                self.lottery_tilt(),
                fmt_cfg(self.train_merit_p_min()),
                fmt_cfg(self.train_merit_p_max())
            ),
            &crate::utils::cfy::table(
                &["net", "exempt", "F", "M", "F ÷ U", "T", "ticket?", "idle"],
                &rows.into_iter().map(|(_, _, row)| row).collect::<Vec<_>>(),
            ),
            Some(&format!(
                "F = the net's smoothed risk-adjusted score (K-reading mean of window avg ∓ λ × window spread, the ranking's number) · M = clocks the net was measured on: real updates + act-only passes (the crown's, and every ticket the coin denied while `train_merit_learn_only` is on) · `F ÷ U` = F ÷ its update count (this debug view only — the final table carries no efficiency column): informational — what one optimizer step bought, so a cheap net reads from an expensive one · field THIS clock drew from: best {} worst {} · T = where the net's SCORE sits between those two bounds as the graded fraction f = the product above (Fm × Ym, or Fm with the age leg off; Fm = 1 at the BEST end, 0 at the worst — unless the tilt says otherwise, and this run is {}), then paid between the two ends Ω leaves — `lo = 1 − Ω·(1 − p_min {})`, `hi = 1 − Ω·(1 − p_max {})`, `p = lo + (hi − lo) × f` — so Ω < 1 lifts every net toward 1 = always train · the coin is a seeded `fastrand` draw `d ∈ [0,1)` compared to `p`: `d < p` sells the ticket, otherwise the coin withholds the weight UPDATE — with `train_merit_learn_only` on the net still acts and measures on a fresh batch, with it off the clock idles (no train, no eval, nothing measured) · ticket? = yes if THIS clock sells it the weight update; `no` means it did not learn, which with the flag off means it did nothing at all · exempt nets (crown, born shield) always run and grade 1.00 — but a crown reads no, because the crown's act pass outranks the lottery · idle = clocks the lottery left it idle BEFORE this clock ({} across the run); the step's lottery line names this clock's count",
                fmt_cfg(best),
                fmt_cfg(worst),
                self.lottery_tilt(),
                fmt_cfg(self.train_merit_p_min()),
                fmt_cfg(self.train_merit_p_max()),
                idled,
            )),
        ));
    }
}

impl CoreEngine {
    /// The clock's ONE-LINE lottery report, printed after the group loop: what the coin did
    /// aggregate, never who it happened to (the per-net list is DEBUG).
    pub(crate) fn log_diversity(&self, clock: usize) {
        let every = self.config.checkpoint_every;
        if every == 0
            || self.log_level == crate::engine::config::LogLevel::None
            || clock % every != 0
        {
            return;
        }
        let Some(rep) = crate::engine::diversity::measure(&self.state, self.header.run_seed, clock)
        else {
            return;
        };
        info!(
            "step {} │ diversity │ {} of {} blueprint(s) distinct │ mean pairwise Δ {} over {} pair(s) │ nodes {:.1}±{:.1} · wires {:.1}±{:.1} · dim {:.1}±{:.1}",
            clock,
            rep.distinct,
            rep.live,
            fmt_cfg(rep.mean_delta),
            rep.pairs,
            rep.nodes.mean,
            rep.nodes.spread,
            rep.wires.mean,
            rep.wires.spread,
            rep.dim.mean,
            rep.dim.spread
        );
    }

    /// The uniform operator's MIXING RATE, cumulative: of every gene it compared, how many
    /// actually took from the other parent.
    pub(crate) fn log_crossover_mix(&self, clock: usize) {
        let every = self.config.checkpoint_every;
        if every == 0
            || self.log_level == crate::engine::config::LogLevel::None
            || clock % every != 0
        {
            return;
        }
        let a = &self.accounting;
        if a.uniform_cx_births == 0 || a.uniform_cx_pairs == 0 {
            return;
        }
        let births = a.uniform_cx_births as f32;
        info!(
            "step {} │ crossover mix │ uniform {} birth(s) · {} gene(s) swapped of {} compared ({:.0}%) · {:.1} swapped per birth over {:.1} aligned · a pairing that swapped nothing is a spent roll, never a child",
            clock,
            a.uniform_cx_births,
            a.uniform_cx_swaps,
            a.uniform_cx_pairs,
            100.0 * a.uniform_cx_swaps as f32 / a.uniform_cx_pairs as f32,
            a.uniform_cx_swaps as f32 / births,
            a.uniform_cx_pairs as f32 / births,
        );
    }

    /// The mutation gate's reading, cumulative: how DIFFERENT the immigrants that were born turned
    /// out to be (Δ to the nearest archived blueprint) and what each birth cost in candidates. The
    /// same numbers the stop card's `Births & novelty` table prints, so a watched run and a finished
    /// one cannot disagree.
    pub(crate) fn log_birth_draws(&self, clock: usize) {
        let every = self.config.checkpoint_every;
        if every == 0
            || self.log_level == crate::engine::config::LogLevel::None
            || clock % every != 0
        {
            return;
        }
        let b = &self.births;
        if b.births == 0 && b.duplicates == 0 {
            return;
        }
        let delta = match (b.mean_delta(), b.delta_min) {
            (Some(mean), Some(min)) => format!(
                "mean Δ {} · min {} over {} judged birth(s) · floor {}",
                fmt_cfg(mean),
                fmt_cfg(min),
                b.delta_births,
                fmt_cfg(self.novelty.min())
            ),
            // Gate off (or every blueprint unreadable): say WHICH, never a fake Δ.
            _ if self.novelty.enabled() => "no Δ reading (blueprint unreadable)".to_string(),
            _ => "no Δ reading (novelty gate off)".to_string(),
        };
        info!(
            "step {} │ births │ {} immigrant(s) │ {} candidate(s) spent ({:.2} per birth, max {}) · re-drawn {} │ {} │ {} seen │ {} crossover duplicate(s) discarded",
            clock,
            b.births,
            b.draws,
            b.draws_per_birth(),
            b.draws_max,
            b.redrawn,
            delta,
            self.novelty.len(),
            b.duplicates,
        );
    }

    pub(crate) fn log_lottery_summary(
        &self,
        clock: usize,
        frame: super::step::LotteryFrame,
        ticketed: usize,
        idled: usize,
    ) {
        // The field the coin graded: live, non-crown, out of the born shield,
        // reading — exactly `lottery_frame`'s filter, re-asked through the same helpers.
        let mut graded = 0usize;
        let mut p_sum = 0.0f32;
        for h in self.state.live_hashes() {
            if self.frozen_crown.contains(&h) || self.on_probation(&h, clock) {
                continue;
            }
            // ONE read per graded net: the score, the ratio from it and the blended probability
            // the same three quantities the pre-step card prints, so they are derived here rather
            let score = self.smoothed_fitness_of(&h);
            let merit = Self::merit_from(score);
            if merit.is_some() {
                graded += 1;
                // The BLENDED probability, not the raw fraction: this line's `won` is a draw
                // against `p_eff`, so printing the raw p would make `won ≈ graded · p̄` look
                p_sum += self.last_ticket_p.get(&h).copied().unwrap_or_else(|| {
                    self.train_lottery_p_eff_from(&h, clock, Some(frame), merit)
                });
            }
        }
        // Budget spent: means over the nets that have ever had a ticket (a newborn's
        // zeros would otherwise drag the averages).
        let (mut upd_sum, mut tkt_sum, mut budgeted) = (0.0f32, 0.0f32, 0usize);
        for h in self.state.live_hashes() {
            let Some(s) = self.state.net(&h) else {
                continue;
            };
            if s.measured_clocks() == 0 {
                continue;
            }
            budgeted += 1;
            upd_sum += s.step as f32;
            tkt_sum += s.measured_clocks() as f32;
        }
        let mean = |sum: f32, n: usize| {
            if n == 0 {
                "—".to_string()
            } else {
                fmt_line(sum / n as f32)
            }
        };
        // The frame is stored on the direction-adjusted (higher-is-better) scale;
        // print it back in the row's own units so ↑ and ↓ read alike.
        let (best, worst) = match self.fitness.direction() {
            crate::utils::fitness::Direction::Maximize => (frame.best, frame.worst),
            crate::utils::fitness::Direction::Minimize => (-frame.best, -frame.worst),
        };
        let (floor, ceiling) = (self.train_merit_p_min(), self.train_merit_p_max());
        let won = graded.saturating_sub(idled);
        // Runners the lottery did not grade: the born shield and the no-reading yet nets always
        // take the ticket, so the rollup's `tickets` is this much larger than `won`.
        let exempt = ticketed.saturating_sub(won);
        let p_bar = (graded > 0).then(|| p_sum / graded as f32);
        // The band rule, not 2dp: `expected` is computed from the same value, so a reader
        // multiplying the PRINTED chance by `graded` must land on the printed expectation (2dp
        let p_txt = p_bar.map_or_else(|| "—".to_string(), fmt_line);
        // Expected vs drawn, side by side: `p̄ × graded` is what the coin was worth,
        // `won` is what it actually produced. The gap is the coin, not a bug.
        let expected = p_bar.map_or_else(|| "—".to_string(), |p| fmt_line(p * graded as f32));
        let omega = self.config.train_merit;
        // Under learn-only the denied nets RAN and were measured: the same count is named
        // "act-only" — it is the coin's loss, not a missing row.
        let lost = if self.config.train_merit_learn_only {
            "act-only"
        } else {
            "idled"
        };
        let body = format!(
            "graded {} = won {} + {} {} │ ticketed {} = won {} + exempt {} │ avg ticket chance {} over the {} graded → expected {}, drawn {} │ field: best {} worst {} │ avg updates {} and avg tickets {} over the {} measured nets │ exempt = crown or born shield (always run)",
            graded,
            won,
            lost,
            idled,
            ticketed,
            won,
            exempt,
            p_txt,
            graded,
            expected,
            won,
            fmt_cfg(best),
            fmt_cfg(worst),
            mean(upd_sum, budgeted),
            mean(tkt_sum, budgeted),
            budgeted,
        );
        info!(
            "step {} │ lottery (Ω {}, {}, band [{}, {}], so ticket T = lo + (hi − lo) × f with lo {} and hi {}) │ {}",
            clock,
            fmt_cfg(omega),
            self.lottery_tilt(),
            fmt_cfg(floor),
            fmt_cfg(ceiling),
            fmt_cfg(1.0 - omega * (1.0 - floor)),
            fmt_cfg(1.0 - omega * (1.0 - ceiling)),
            body,
        );
    }
}

/// The printed column names, spelled ONCE: the header cell, the legend key, the glyph placement
/// the CSV header all read these strings, so a rename cannot leave one of them on an old spelling.
const COL_FIT: &str = "fit (F = mean ∓ λ·spread)";
/// The same column when the run averages in a SIGNED-LOG space (`MeanType::SignedLog`).
const COL_FIT_SIGNED_LOG: &str = "fit (F = g⁻¹(ḡ ∓ λ·σ̄), g = sign(x)·ln(1+|x|))";
/// The same column for the order-statistic family (`MeanType::Median`).
const COL_FIT_MEDIAN: &str = "fit (F = median ∓ λ·1.4826·MAD)";

/// The header cell for `fit (F)`, in the equation of the family this run averages
/// ONE function, so the header, the glyph target and the legend key are the same string.
fn fit_column(mean_type: crate::engine::config::MeanType) -> &'static str {
    match mean_type {
        crate::engine::config::MeanType::Arithmetic => COL_FIT,
        crate::engine::config::MeanType::SignedLog => COL_FIT_SIGNED_LOG,
        crate::engine::config::MeanType::Median => COL_FIT_MEDIAN,
    }
}
const COL_FIT_MINMAX: &str = "fit minmax (Fm)";
/// The same column under the INVERTED ordering: the tilt mirrors the place, so the equation
/// header states has to mirror with it (`1 −` the best-first place).
const COL_FIT_MINMAX_TILTED: &str = "fit minmax (Fm⁻)";
const COL_AGE: &str = "age (A)";
const COL_YOUTH: &str = "youth minmax (Ym)";
const COL_TICKET: &str = "ticket minmax (T = lo + (hi − lo) × Fm × Ym)";
/// The ticket under the INVERTED ordering: it multiplies the MIRRORED place, so the equation must
/// name the cell actually on its left (`Fm⁻`), not the best-first `Fm` it no longer reads.
const COL_TICKET_TILTED: &str = "ticket minmax (T = lo + (hi − lo) × Fm⁻ × Ym)";
/// The ticket with the AGE leg OFF (`set_run_train_merit_age(false)`): nothing multiplies the place, so
/// the equation must not name a `Ym` the table does not print.
const COL_TICKET_NO_AGE: &str = "ticket minmax (T = lo + (hi − lo) × Fm)";
/// The no-age ticket under the inverted ordering.
const COL_TICKET_NO_AGE_TILTED: &str = "ticket minmax (T = lo + (hi − lo) × Fm⁻)";

/// The place cell this run prints — ONE function, so the header and its legend key cannot drift.
fn fit_minmax_column(inverted: bool) -> &'static str {
    if inverted {
        COL_FIT_MINMAX_TILTED
    } else {
        COL_FIT_MINMAX
    }
}

/// The ticket cell this run prints — named after the place it actually multiplies and the legs
/// the setters actually armed (`set_run_train_merit_inverted`, `set_run_train_merit_age`).
fn ticket_column(inverted: bool, age_leg: bool) -> &'static str {
    match (inverted, age_leg) {
        (false, true) => COL_TICKET,
        (true, true) => COL_TICKET_TILTED,
        (false, false) => COL_TICKET_NO_AGE,
        (true, false) => COL_TICKET_NO_AGE_TILTED,
    }
}
const COL_UPDATES: &str = "updates (U)";
const COL_ACT: &str = "act (O)";
const COL_MEASURED: &str = "measured (M = U + O)";
const COL_IDLE: &str = "idle (I)";
const COL_EXAM: &str = "exam (X)";

/// The columns the card actually PRINTS, in reading order — `STOP_HEADER_FIXED` minus the cells
/// this run cannot fill.
fn printed_columns(age_leg: bool, has_exam: bool) -> Vec<&'static str> {
    STOP_HEADER_FIXED
        .iter()
        .copied()
        .filter(|c| match *c {
            COL_YOUTH => age_leg,
            COL_EXAM => has_exam,
            _ => true,
        })
        .collect()
}

/// The fixed left-hand columns of the stop-time table, in NARRATIVE order: what the net IS (`#`,
/// `hash`, `origin`, census) → how it is RANKED (`fit (F)` and its min-max place) → what it cost
const STOP_HEADER_FIXED: &[&str] = &[
    "#",
    "hash",
    "origin",
    "params",
    "nodes",
    "wires",
    "width",
    // The word AND its letter, so a legend equation can be read straight off the row; the DERIVED
    // column is named by the value it produces (`ticket`).
    "mean",
    "spread",
    // The value the two legs MAKE comes after them, so the row reads left to right.
    COL_FIT,
    // ── the coin's own arithmetic, in the order it runs ── `fit minmax (Fm)` = this net's `fit
    // (F)` as a 0..1 place in the last field (the tilt already applied: `1` = the end that earns),
    COL_FIT_MINMAX,
    COL_AGE,
    COL_YOUTH,
    COL_TICKET,
    // The two ADDENDS first, then their sum: `B = U + O` reads across three adjacent cells,
    // sum is only a total of things already printed on its left.
    "updates (U)",
    // Act-only clocks immediately after `U`. `act`, not `frozen`: the crown seat DOES
    // train and evaluate — calling that net frozen claimed it did nothing.
    "act (O)",
    COL_MEASURED,
    // Idle comes AFTER the two counters it is not part of, then the age identity `A = U + O + I +
    // 1` reads left to right; the +1 is the BIRTH clock (it ages the net, it is neither a ticket
    "idle (I)",
    // The EXAM: rows no clock ever read, scored ONCE at stop on the reported nets.
    COL_EXAM,
    // The Δ columns are NOT here: a raw multiset-Jaccard distance sits at 0.7–0.9 between any two
    // unrelated nets, so a per-row value had no reference scale.
];

/// Everything the stop-time legend reads, bundled so its rows can be built (and checked
/// against the header) without a live engine.
struct LegendCtx<'a> {
    sign: &'a str,
    lambda: f32,
    window: usize,
    /// Recency ramp r of the window (`set_run_window_recency`), stated wherever
    /// window mean is, so the `mean` column's arithmetic is complete.
    recency: f32,
    /// The family the window is averaged in — its header equation and its legend row
    /// picked by this, so the two cannot disagree (`set_run_mean_type`).
    mean_type: crate::engine::config::MeanType,
    /// Worst-first: the place is mirrored, so the place cell and the ticket equation name `Fm⁻`.
    inverted: bool,
    /// The age leg (`set_run_train_merit_age`): ARMED, `Ym` is a column and a factor; OFF, the ticket
    /// equation names the place alone, because a factor the table does not print cannot be read.
    age_leg: bool,

    /// The last clock the lottery dealt — which clock the ticket column is read
    /// against.
    dealt_clock: usize,
    omega: f32,
    /// The lottery's band and ordering: the stop table's field row states them, because
    /// the two ends it names are not the two ends that earn the tickets when tilted.
    p_min: f32,
    p_max: f32,
    tilt: &'a str,
    direction: crate::utils::fitness::Direction,
    lottery: Option<super::step::LotteryFrame>,
    k: usize,
    crossover_probation: usize,
    mutation_probation: usize,
    join_probation: usize,
    crown_lifetime: usize,
}

/// The stop-time legend: `abbrev | description`, one row per column, keyed
fn stop_legend_rows(ctx: &LegendCtx<'_>) -> Vec<(String, String)> {
    // The symbol the printed table uses for the place, and the header that cell carries.
    let place = if ctx.inverted { "Fm⁻" } else { "Fm" };
    let minmax = fit_minmax_column(ctx.inverted);
    let mut rows = vec![
        (
            "#".to_string(),
            "rank by `fit (F)`, best first; `—` for a live net with no reading yet (it is listed last, unranked)"
                .to_string(),
        ),
        (
            "hash".to_string(),
            "the first 8 hex of the blueprint's xxh3 — the net's identity in `nets/<hash>.json`, `champions.json` and this table"
                .to_string(),
        ),
        // ── what this net IS ────────────────────────────────────────────
        (
            "origin".to_string(),
            "where the net came from: a founder slot, or the crossover/mutation op and the clock it was born at (a newborn's first ticket is the NEXT clock)"
                .to_string(),
        ),
        (
            COL_AGE.to_string(),
            "`A = clock − born + 1` — clocks lived, inclusive of the birth clock".to_string(),
        ),
        (
            "params".to_string(),
            "trainable parameters, in K/M/B units (`40.2K` = 40 200); the exact count is in `nets/<hash>.json`"
                .to_string(),
        ),
        (
            "nodes".to_string(),
            "nodes in the blueprint: input + hidden + output".to_string(),
        ),
        (
            "wires".to_string(),
            "connections in the blueprint, one per output-port → input-port wire".to_string(),
        ),
        (
            "width".to_string(),
            "mean hidden-node width — the mean `hidden_dim` across the hidden nodes".to_string(),
        ),
        // ── how it is RANKED ────────────────────────────────────────────
        (
            fit_column(ctx.mean_type).to_string(),
            format!(
                "`F` = {} — the ranking value. This run charges `{} λ × spread` with λ = {} and averages in `{}`. `mean` and `spread` are ALWAYS the plain arithmetic pair, so with `arithmetic` the row redoes `F` directly; the other families state the transform that turns that pair into the centre the λ charge applies to. Geometric/harmonic averaging is not offered: their domain excludes a non-positive reading, and a reward or centred fitness can cross zero — `signed_log` is the no-cliff form of the same damping",
                match ctx.mean_type {
                    crate::engine::config::MeanType::Arithmetic => {
                        "the window mean, charged for its own spread (`F = mean ∓ λ × spread`)"
                    }
                    crate::engine::config::MeanType::SignedLog => {
                        "the window average taken in `g(x) = sign(x)·ln(1+|x|)`, charged `λ` of the spread IN THAT SPACE, then mapped back with `g⁻¹`"
                    }
                    crate::engine::config::MeanType::Median => {
                        "the window's weighted MEDIAN, charged `λ × 1.4826 × MAD` (the median absolute deviation, scaled so λ still reads in std-like units) — one fluke cannot move it at all"
                    }
                },
                ctx.sign,
                fmt_cfg(ctx.lambda),
                ctx.mean_type.as_str(),
            ),
        ),
        (
            "mean".to_string(),
            format!(                    "the window mean of the last `n = min(B, K)` results, K = {}, weighted by the recency ramp r = {} (weights run 1 on the OLDEST reading to 1 + r·(n−1) on the newest, so r = 0 is the plain mean) — `n` gets no column of its own: it only falls short of K for a net born within K clocks of the stop, so it is read off `measured (M)`",
                ctx.window,
                fmt_cfg(ctx.recency)
            ),
        ),
        (
            "spread".to_string(),
            "the spread across those `n` results — `0.0000` at `n = 1`, because one result shows no spread and the ranking CHARGES that zero (`F = mean ∓ λ × spread` with spread 0). A dash here would make the row's own arithmetic impossible to redo".to_string(),
        ),
        (
            minmax.to_string(),
            format!(
                "this net's `fit (F)` as a 0..1 position between the field's two ends on the LAST dealt clock (clock {}): `Fm = (F − F_worst) ÷ (F_best − F_worst)` over the nets the lottery GRADES (live, non-crown, out of its born shield, with a reading). The tilt is ALREADY applied, so `1` is the end that earns the tickets ({} this run) and `0` the end that earns fewest — under worst-first the place is mirrored, which is what lets the coin read one ordering, and the HEADER spells that mirror out (`Fm⁻ = 1 − …`) because that factor is the one the tilt reverses. The two ends themselves are printed with their numbers in the `F field · Ω` row",
                ctx.dealt_clock, ctx.tilt
            ),
        ),
        (
            ticket_column(ctx.inverted, ctx.age_leg).to_string(),
            match ctx.lottery.as_ref() {
                Some(_) => {
                    // The equation names EXACTLY the cells the table prints: `Ym` is a column
                    // only while the age leg is armed, so its factor cannot be stated blind.
                    let eq = if ctx.age_leg {
                        format!("`T = lo + (hi − lo) × {place} × Ym`")
                    } else {
                        format!("`T = lo + (hi − lo) × {place}`")
                    };
                    let legs = if ctx.age_leg {
                        format!("the product of the two cells on its LEFT (`{place}` × `Ym`)")
                    } else {
                        format!("the single leg on its LEFT (`{place}`)")
                    };
                    let arm = if ctx.age_leg {
                        "The age leg is ARMED (`set_run_train_merit_age(true)`), so `Ym` multiplies the place and an older net's cell falls"
                    } else {
                        "The age leg is OFF (`set_run_train_merit_age(false)`), so nothing multiplies the place and the cell is one step from it"
                    };
                    format!(
                        "the chance the coin gave this net on the LAST clock it dealt (clock {}): {eq}, {legs}, between the two ends Ω leaves (`lo`, `hi` stated with their numbers in the `F field · Ω` row) — and `T` is the probability ITSELF, on the coin's own line: the draw is `d = fastrand ∈ [0,1)` and the net LEARNS iff `d < T` — a lost draw still acts and measures when `train_merit_learn_only` is on, so `lo` and `hi` are the share of the unit interval the field's chances cover (`T = 1` trains every clock, `T = 0` never). {arm}. `—` when the coin was not thrown for this net: the lottery is off or was never armed, it holds a crown seat (the act pass outranks the coin), it is inside its born shield, it has no reading yet, or the graded field had no spread to place it in. Those nets always run, but a number here would claim a draw they never took part in. Under tilt {} this column FALLS as `fit (F)` rises",
                        ctx.dealt_clock,
                        ctx.tilt
                    )
                }
                None => "the chance the coin gave this net on the last clock: `—` on every row, because there was no field to grade (Ω = 0, or the lottery was never armed)".to_string(),
            },
        ),
        (
            COL_YOUTH.to_string(),
                "the age leg: `Ym = 1 − (A − A_min) ÷ (A_max − A_min)` over the graded field, so `1` at the YOUNGEST graded net and `0` at the oldest — NOT mirrored by the tilt, because \"young earns a look\" points the same way in both orderings. Printed only when the leg is ARMED: with it OFF nothing multiplies it, so it is not a column at all".to_string(),
        ),
        // ── what its learning COST ──────────────────────────────────────
        (
            COL_MEASURED.to_string(),
            "`M = U + O` — clocks the net was MEASURED: it ran the trainer and was scored (`U`), or it held the crown and was scored with its update discarded (`O`). It is the measuring cost, not the learning cost: the lottery hands out `U` only, so a crown pass lifts `M` without ever being drawn for. An IDLE clock is in neither — nothing ran, nothing was measured"
                .to_string(),
        ),
        (
            "F field · Ω".to_string(),
            match ctx.lottery.as_ref() {
                Some(f) => {
                    // The frame is stored direction-adjusted (`best` = the value `p = 1` earns,
                    // either direction), so the printed pair is converted back into the COLUMN's
                    let (best, worst) = match ctx.direction {
                        crate::utils::fitness::Direction::Maximize => (f.best, f.worst),
                        crate::utils::fitness::Direction::Minimize => (-f.best, -f.worst),
                    };
                    // The two ends Ω folds the band into: the interval the ticket is paid
                    // printed with their numbers so the last cell of the row is redoable
                    let lo = 1.0 - ctx.omega * (1.0 - ctx.p_min);
                    let hi = 1.0 - ctx.omega * (1.0 - ctx.p_max);
                    format!(
                        "the lottery field the last clock drew from: best `fit (F) = {}` and worst `fit (F) = {}` over the nets it grades (crown and born-shield nets are exempt, so they never set the scale) · Ω = {} = how much of that spread it withholds, and it is ONE map with the band — folded into the two ends the chance is paid between: `lo = 1 − Ω·(1 − p_min) = {}`, `hi = 1 − Ω·(1 − p_max) = {}` · band [p_min {}, p_max {}] = the wall-to-wall payout before Ω folds it · tilt {} = which END earns the tickets (worst-first: the LEAST fit net takes most of them, best-first: the fitter net does) · the column prints `fit (F)` at cell precision so the two ends these bounds name are findable in it",
                        fmt_cell(best),
                        fmt_cell(worst),
                        fmt_cfg(ctx.omega),
                        fmt_cfg(lo),
                        fmt_cfg(hi),
                        fmt_cfg(ctx.p_min),
                        fmt_cfg(ctx.p_max),
                        ctx.tilt,
                    )
                }
                None => format!(
                    "no lottery field at the last clock: Ω = {} (0 = off, every net trains) or the field was not armed yet — every graded net took its ticket (tilt {} · band [p_min {}, p_max {}] from the manifest)",
                    fmt_cfg(ctx.omega),
                    ctx.tilt,
                    fmt_cfg(ctx.p_min),
                    fmt_cfg(ctx.p_max)
                ),
            },
        ),
        (
            COL_UPDATES.to_string(),
            "`U` — clocks it actually applied an optimizer step on".to_string(),
        ),
        // `O` sits beside `U` in the header, so its legend row states the sum it closes.
        (
            COL_ACT.to_string(),
            "`O` — clocks that held an act-only pass: the crown's seat (it trained, evaluated and was scored, but its update was discarded). The weights held still, the net did not. It sits next to `updates (U)` so `measured (M) = U + O` reads across two adjacent cells"
                .to_string(),
        ),
        (
            COL_IDLE.to_string(),
            "`I` — clocks the lottery gave no ticket: nothing ran, so it is not a ticket and not an update. It comes after the `U + O` pair, and `A = U + O + I + 1` reads left to right"
                .to_string(),
        ),
        // ── the age identity ───────────────────────────────────
        (
            "A = U + O + I (+1 if born mid-run)".to_string(),
            "every clock a net lived is an update, an act-only clock, or an idle clock — PLUS the birth clock a mid-run child sits out (a founder runs clock 0; a child's first ticket is the NEXT clock), which is why the sum can land one short of `A`"
                .to_string(),
        ),
        (
            "B = U + O".to_string(),
            "every ticket is an update or an act-only clock".to_string(),
        ),
        // ── what this net IS (the verdict) ────────────────────────────
        (
            COL_EXAM.to_string(),
            "the EXAM: the same ranking scorer over rows the race NEVER read — a separate `exam/` store, not the eval pool, and not reachable from any clock. Scored once at stop for the reported nets only and written to `exam.json`; `—` = no exam supplied, or a net outside the reported set. It ranks nothing: read it against `fit (F)`, because `F` is what selection saw and `X` is what nothing saw"
                .to_string(),
        ),
        (
            "mark".to_string(),
            format!(
                "the verdict glyph, INLINE in the console card — `★` saved as CHAMPION (top-{} by `fit (F)`) and holding a crown SEAT · `☆` saved as champion but barred from the seat (the `j` or `t` beside it names the gate) · `·` neither; with the seats off nobody sits, so every champion reads `★`. Champion and seat are different questions: the first is the saved, inheritable ranking, the second the frozen act-only switch. It sits inside the column that decided it (`★`/`☆`/`j`/`t` on `fit (F)`, `b` on `age (A)`); `final-field.csv` carries no mark cell, its columns stay plain values",
                ctx.k
            ),
        ),
        (
            "mark b · j · t".to_string(),
            format!(
                "gate flags, and they can co-occur (bt): b = inside its born cull shield (crossover {} / mutation {} clock(s)) — it cannot be culled AND cannot take a seat · j = its seat window is not served yet ({} clocks needed; a seat exit restarts the count) — printed only when the born shield is not already the reason, so `bj` never appears · t = past the crown term ({}, age-blocked)",
                ctx.crossover_probation,
                ctx.mutation_probation,
                ctx.join_probation,
                // A lifetime of 0 means the term is DISARMED — printing `0 clocks`
                // read as a rule that blocks every net.
                match ctx.crown_lifetime {
                    0 => "off".to_string(),
                    j => format!("{j} clocks"),
                },
            ),
        ),
        // ── the informative band (appended after the last fixed column) ──── ONE row stands
        // the whole band: its cells are as dynamic as the caller's `set_run_metrics(...)` list,
        (
            "m:<name>".to_string(),
            "an informative METRIC you registered with `set_run_metrics` — `m:` marks it as measured but NEVER ranked: computed on the same eval batch as `fit (F)`, recorded in `history.csv`, and read by no ranking, champion, seat, lottery draw or cull. `<name>` is the label you gave `Metric::custom(...)`, verbatim, and the columns are appended after `exam (X)` in config order"
                .to_string(),
        ),
    ];
    // The legend follows the HEADER: a reader with the table and the legend side by side finds
    // same column in the same relative place.
    rows.sort_by_key(|(key, _)| {
        STOP_HEADER_FIXED
            .iter()
            .position(|c| *c == key)
            .unwrap_or(STOP_HEADER_FIXED.len())
    });
    rows
}

#[cfg(test)]
mod stop_legend_tests {
    use super::*;

    fn ctx() -> LegendCtx<'static> {
        LegendCtx {
            sign: "−",
            lambda: 0.5,
            window: 10,
            recency: 0.0,
            mean_type: crate::engine::config::MeanType::Arithmetic,
            inverted: false,
            age_leg: true,
            dealt_clock: 0,
            omega: 0.0,
            p_min: 0.0,
            p_max: 1.0,
            tilt: "best-first",
            direction: crate::utils::fitness::Direction::Maximize,
            lottery: None,
            k: 1,
            crossover_probation: 5,
            mutation_probation: 5,
            join_probation: 5,
            crown_lifetime: 5,
        }
    }

    /// The legend and the table are ONE contract: every column the header prints has a row
    /// explains
    #[test]
    fn every_column_has_a_legend_row() {
        let rows = stop_legend_rows(&ctx());
        for cell in STOP_HEADER_FIXED {
            assert!(
                rows.iter().any(|(key, _)| key == cell),
                "column `{cell}` has no legend row — keys: {:?}",
                rows.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>()
            );
        }
        // The conditional columns (`eval loss`, the informative metrics) are the only ones outside
        // the 1:1 rule: they exist only when a trainer reports them.
        assert!(
            rows.iter().any(|(k, _)| k == "m:<name>"),
            "the `m:` metric prefix needs a legend row"
        );
    }

    /// Under tilt BOTH equation cells name the mirror: the ticket multiplies `Fm⁻`, so it must not
    /// print a `Fm` its neighbour no longer holds.
    #[test]
    fn tilted_ticket_names_the_mirrored_place() {
        let mut c = ctx();
        c.inverted = true;
        let rows = stop_legend_rows(&c);
        let keys: Vec<&str> = rows.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&COL_FIT_MINMAX_TILTED), "{keys:?}");
        assert!(keys.contains(&COL_TICKET_TILTED), "{keys:?}");
        assert!(!keys.contains(&COL_FIT_MINMAX), "{keys:?}");
        assert!(!keys.contains(&COL_TICKET), "{keys:?}");
        assert!(COL_TICKET_TILTED.contains("× Fm⁻ × Ym"));
        assert!(COL_TICKET.contains("× Fm × Ym"));
        // The header rewrite and the legend key read the SAME helper, so the cell and its row agree.
        assert_eq!(fit_minmax_column(true), COL_FIT_MINMAX_TILTED);
        assert_eq!(ticket_column(false, true), COL_TICKET);
    }

    /// TWO setters shape the ticket cell: the tilt swaps `Fm` for `Fm⁻`, the age leg decides
    /// whether `Ym` is a factor at all — so each of the four combinations has its own equation,
    /// and the header a reader sees is the key its legend row carries.
    #[test]
    fn the_ticket_equation_follows_the_tilt_and_the_age_leg() {
        let mut headers: Vec<&'static str> = Vec::new();
        for inverted in [false, true] {
            for age_leg in [false, true] {
                let mut c = ctx();
                c.inverted = inverted;
                c.age_leg = age_leg;
                let header = ticket_column(inverted, age_leg);
                let rows = stop_legend_rows(&c);
                assert!(
                    rows.iter().any(|(k, _)| k == header),
                    "`{header}` has no legend row — keys: {:?}",
                    rows.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>()
                );
                assert_eq!(header.contains("Ym"), age_leg, "{header}");
                assert_eq!(header.contains("Fm⁻"), inverted, "{header}");
                headers.push(header);
            }
        }
        headers.sort_unstable();
        headers.dedup();
        assert_eq!(
            headers.len(),
            4,
            "the four setter combinations must print four different equations: {headers:?}"
        );
    }

    /// Each mean family prints ITS OWN equation, and that exact string is the legend key:
    /// a reader of a `median` run never sees the arithmetic row, and vice versa.
    #[test]
    fn each_mean_family_has_its_own_header_and_legend_row() {
        use crate::engine::config::MeanType;
        let mut headers: Vec<&'static str> = Vec::new();
        for family in [MeanType::Arithmetic, MeanType::SignedLog, MeanType::Median] {
            let mut c = ctx();
            c.mean_type = family;
            let rows = stop_legend_rows(&c);
            let header = fit_column(family);
            assert!(
                rows.iter().any(|(k, _)| k == header),
                "{family} prints `{header}` but has no legend row keyed the same"
            );
            headers.push(header);
        }
        headers.sort_unstable();
        headers.dedup();
        assert_eq!(
            headers.len(),
            3,
            "the three families must print three different equations"
        );
        // The recorded spelling round-trips, and `""` (a pre-knob header)
        // spelling but a LEGACY value the caller resolves to the arithmetic default.
        for family in [MeanType::Arithmetic, MeanType::SignedLog, MeanType::Median] {
            assert_eq!(MeanType::parse(family.as_str()), Some(family));
        }
        assert_eq!(MeanType::parse(""), None);
        assert_eq!(MeanType::default(), MeanType::Arithmetic);
    }

    /// One key, one explanation — two rows for a column can disagree with each other.
    #[test]
    fn legend_keys_are_unique() {
        let rows = stop_legend_rows(&ctx());
        let mut seen = std::collections::HashSet::new();
        for (key, _) in &rows {
            assert!(seen.insert(key.clone()), "duplicate legend key `{key}`");
        }
    }
}

/// The plan BOTH stop-time cards render from: one column list, one crown, one live population, one
/// clock.
struct StopPlan {
    /// The shared header — `rank … Δcrown · Δpop · origin`, plus the conditional
    /// `eval loss` and informative-metric columns.
    header: Vec<String>,
    /// The columns the header, the rows and the `final-field.csv` export share, in reading
    /// order — `STOP_HEADER_FIXED` minus the cells this run cannot fill.
    columns: Vec<&'static str>,
    /// Whether the run supplied an exam, i.e. whether `exam (X)` is a column
    has_exam: bool,
    /// Whether the `eval loss` column is in the plan (some reported net carries one).
    has_eval: bool,
    /// The last clock the lottery dealt: the `seat` and `age` columns count
    dealt_clock: usize,
    /// The reported champions' shapes, in elite rank order — what `Δcrown` measures
    crown: Vec<NetShape>,
    /// By hash, that champion's index in `crown` (`None`: unreadable blueprint; absent:
    /// the row is not an elite).
    crown_idx: std::collections::HashMap<String, Option<usize>>,
    /// Every live net's shape, by hash — what `Δpop` measures
    population: Vec<(String, NetShape)>,
}

/// One row of a stop-time card.
struct StopCells {
    inlined: Vec<String>,
    plain: Vec<String>,
}

struct StopRow<'a> {
    /// 1-based rank in the race, best FIT first; `None` for an UNMEASURED net, which has
    /// no FIT to be ranked by (it prints `—` there and rides at the bottom of the table).
    rank: Option<usize>,
    /// The saved-set mark: `★` saved elite AND seated, `☆` saved but barred, `·` neither.
    mark: &'a str,
    /// The net, by full hash (the `hash` cell prints the first eight).
    hash: &'a str,
    /// The ranking value (FIT) the `fit (F)` column prints; `None` when never measured.
    fit: Option<f32>,
    /// This net's structure, parsed once by the caller (`None`: unreadable blueprint).
    /// Borrowed, not owned: the card's shapes live in its `StopPlan` for its whole life.
    shape: Option<&'a NetShape>,
    /// The plan the whole card prints
    plan: &'a StopPlan,
    /// The EXAM reading `exam (X)`: the same ranking scorer over rows the race never
    /// read. `None` = no exam supplied, or this net could not be scored.
    exam: Option<f32>,
    /// The field the LAST dealt clock drew from, shared by every row of the card
    /// ticket cell costs no extra walk. `None` = the lottery was off / never armed.
    lottery: Option<super::step::LotteryFrame>,
}

/// One card's own structural summary: the mean `Δcrown` and `Δpop` over the rows it printed.
struct CardStats {
    delta_crown: Option<f32>,
    delta_pop: Option<f32>,
}

impl CardStats {
    /// `card mean: Δcrown 0.31 · Δpop 0.55` — `—` when no row carried the number.
    fn line(&self) -> String {
        format!(
            "card mean: Δcrown {} · Δpop {}",
            opt_cfg(self.delta_crown),
            opt_cfg(self.delta_pop)
        )
    }
}

/// The one spelling of "no value here" every stop-time cell uses.
fn dash() -> String {
    "—".to_string()
}

/// A formatted scalar, or `—` when there is none.
fn opt_cfg(v: Option<f32>) -> String {
    v.map(fmt_cfg).unwrap_or_else(dash)
}

/// Mean of the values that EXIST (`None` legs are unreadable blueprints, not zeros),
/// `None` when no row carried one.
fn mean_of(values: &[Option<f32>]) -> Option<f32> {
    let kept: Vec<f32> = values.iter().flatten().copied().collect();
    if kept.is_empty() {
        return None;
    }
    Some(kept.iter().sum::<f32>() / kept.len() as f32)
}

/// `Δcrown` for one row: the mean multiset-Jaccard distance to the reported champions.
fn delta_crown(shape: Option<&NetShape>, plan: &StopPlan, self_idx: Option<usize>) -> Option<f32> {
    let shape = shape?;
    match self_idx {
        Some(i) => mean_delta_to(
            shape,
            plan.crown
                .iter()
                .take(i)
                .chain(plan.crown.iter().skip(i + 1)),
        ),
        None => mean_delta_to(shape, plan.crown.iter()),
    }
}

/// `Δpop` for one row: the mean distance to every OTHER live net — "does this net look like
/// crowd?".
fn delta_pop(shape: Option<&NetShape>, hash: &str, plan: &StopPlan) -> Option<f32> {
    let shape = shape?;
    mean_delta_to(
        shape,
        plan.population
            .iter()
            .filter(|(h, _)| h != hash)
            .map(|(_, s)| s),
    )
}

/// One net's origin cell for the final elite table: WHO made it, without parent-hash
/// ids, with the birth clock when non-zero. Stored lineage is unchanged.
fn origin_cell(created_from: Option<&str>, born: usize) -> String {
    let Some(raw) = created_from else {
        return format!("unknown@{born}");
    };
    let short8 = |hash: &str| hash[..8.min(hash.len())].to_string();
    // `crossover-one-point:parents=h1,h2[+mut]` → `cx one-point@<born>[+mut]` —
    // crossover TYPE is useful; the parent ids are not (widest column by far).
    if let Some(rest) = raw.strip_prefix("crossover-") {
        let (head, mutated) = rest
            .strip_suffix("+mut")
            .map(|p| (p, true))
            .unwrap_or((rest, false));
        let mut segs = head.split(':');
        let kind = segs.next().unwrap_or("one-point");
        // How much of the child the other parent contributed (`uniform` only, since a one-point cut
        // exchanges no per-gene coins): `3/6` is the honest cousin of `fit` — two children
        let swapped = segs
            .find_map(|s| s.strip_prefix("swapped="))
            .map(|n| format!(" · {n}"))
            .unwrap_or_default();
        return format!(
            "cx {kind}@{born}{swapped}{}",
            if mutated { "+mut" } else { "" }
        );
    }
    // `clone:parents=h1` — asexual single-parent child: type + step, no id.
    if raw == "clone" || raw.starts_with("clone:") {
        return format!("clone@{born}");
    }
    if let Some(rest) = raw.strip_prefix("seed:") {
        // `seed:<run_id>:<champion_hash>` → `seed:<run_id>:<hash8>` —
        // run id is the provenance that matters for inheritance, keep
        let mut parts = rest.splitn(2, ':');
        let run_id = parts.next().unwrap_or("?");
        return match parts.next() {
            Some(hash) => format!("seed:{run_id}:{}", short8(hash)),
            None => format!("seed:{run_id}"),
        };
    }
    if raw == "random" {
        // A bare `random` tells nothing about WHEN — tag the birth clock.
        return format!("random@{born}");
    }
    // `founder-N` and anything else: verbatim — a founder's birth clock is 0
    // by definition, so a `@0` suffix would be pure noise.
    raw.to_string()
}

/// Run-accounting population tally as `field | count` rows for the CFY stop card.
/// Every counter the old prose carried is one cell; `alive` is passed in at stop.
pub(crate) fn accounting_population(
    a: &crate::state::RunAccounting,
    alive: usize,
) -> Vec<(String, String)> {
    vec![
        ("born".to_string(), a.born.to_string()),
        ("  founders (random)".to_string(), a.founders.to_string()),
        (
            "  founders (inherited)".to_string(),
            a.inherited_founders.to_string(),
        ),
        ("  mutations (random)".to_string(), a.immigrants.to_string()),
        (
            "  crossover children".to_string(),
            a.crossover_children.to_string(),
        ),
        (
            "    of them uniform (swaps/birth)".to_string(),
            if a.uniform_cx_births == 0 {
                "—".to_string()
            } else {
                format!(
                    "{:.1} of {:.1} genes",
                    a.uniform_cx_swaps as f32 / a.uniform_cx_births as f32,
                    a.uniform_cx_pairs as f32 / a.uniform_cx_births as f32,
                )
            },
        ),
        ("culled".to_string(), a.culls.to_string()),
        (
            "  culled (crossover roll)".to_string(),
            a.culls_crossover.to_string(),
        ),
        (
            "  culled (mutation roll)".to_string(),
            a.culls_immigrant.to_string(),
        ),
        ("alive at stop".to_string(), alive.to_string()),
        (
            "evolution-deferred clocks".to_string(),
            a.deferred_clocks.to_string(),
        ),
    ]
}

/// Run-accounting rolls MATRIX (CFY-C): one row per roll kind, one column per outcome.
/// Returns `(header, rows)` ready for [`crate::utils::cfy::table`].
pub(crate) fn accounting_rolls(
    a: &crate::state::RunAccounting,
) -> (Vec<&'static str>, Vec<Vec<String>>) {
    let dash = || "—".to_string();
    let header = vec![
        "event",
        "rolls",
        "fired",
        "attempts",
        "built",
        "discarded",
        "blocked",
    ];
    let rows = vec![
        vec![
            "crossover".to_string(),
            a.crossover_rolls.to_string(),
            a.crossover_fired.to_string(),
            a.crossover_attempts.to_string(),
            a.crossover_children.to_string(),
            a.crossover_discarded.to_string(),
            dash(),
        ],
        vec![
            "mutation".to_string(),
            a.mutation_rolls.to_string(),
            a.mutation_fired.to_string(),
            dash(),
            a.immigrants.to_string(),
            dash(),
            a.mutation_blocked.to_string(),
        ],
    ];
    (header, rows)
}

/// Flatten a trainer's `describe()` blob into `field | value` ROWS for a CFY table (`nested.key`
/// for nested objects; arrays join into one cell).
pub(crate) fn summarize_rows(v: &serde_json::Value) -> Vec<(String, String)> {
    /// Max facts rendered before an explicit `… +n more` row.
    const MAX_FIELDS: usize = 12;

    /// Flatten a nested JSON object into `(key, value)` fragments.
    fn flat(out: &mut Vec<(String, String)>, prefix: &str, v: &serde_json::Value) {
        let key = |k: &str| {
            if prefix.is_empty() {
                k.to_string()
            } else {
                format!("{prefix}.{k}")
            }
        };
        match v {
            serde_json::Value::Object(map) => {
                for (k, val) in map {
                    flat(out, &key(k), val);
                }
            }
            serde_json::Value::String(s) => out.push((prefix.to_string(), s.clone())),
            // Display-only: a float shows 4 decimals (an f32 round-trips as f64
            // `0.10000000149011612`); integers stay integers.
            serde_json::Value::Number(n) => {
                let text = if n.is_f64() {
                    format!("{:.4}", n.as_f64().unwrap_or(0.0))
                } else {
                    n.to_string()
                };
                out.push((prefix.to_string(), text));
            }
            serde_json::Value::Bool(b) => out.push((prefix.to_string(), b.to_string())),
            // Arrays: the elements are already short scalars in practice
            // (`["linear", "relu"]`), so one comma-joined cell beats N lines.
            serde_json::Value::Array(items) => {
                let cells: Vec<String> = items.iter().map(|x| x.to_string()).collect();
                out.push((prefix.to_string(), format!("[{}]", cells.join(","))));
            }
            // Null carries no information; `{}` (an empty object) flattens
            // nothing, which is how an empty `describe()` reads as "no table".
            serde_json::Value::Null => {}
        }
    }

    let mut parts = Vec::new();
    flat(&mut parts, "", v);
    if parts.len() > MAX_FIELDS {
        let extra = parts.len() - MAX_FIELDS;
        parts.truncate(MAX_FIELDS);
        parts.push(("…".to_string(), format!("+{extra} more")));
    }
    parts
}

/// One ★-badge entry: the elite net's short hash plus its seat state. Seated ⇒ tenure
/// over the crown lifetime `j` (`j = 0` disarms); not seated ⇒ `no seat`.
pub(crate) fn elite_badge_entry(short: &str, seat_age: usize, j: usize) -> String {
    if seat_age == 0 {
        format!("{short}(no seat)")
    } else if j == 0 {
        format!("{short}(term {seat_age})")
    } else {
        format!("{short}(term {seat_age}/{j})")
    }
}

/// The clocks a net LIVED but did not update (`age − trainings`), or `None` when it trained
/// one.
pub(crate) fn no_update_gap(trainings: usize, age: usize) -> Option<usize> {
    (age > trainings).then(|| age - trainings)
}

/// One seat TAKEN, as the deal line spells it: `+ b2dc1462 #1 47.5 (n6)` — the seat it won
/// (`elite_rank`, 1 = the best SEATED net), the value it won it on, and the readings behind
pub(crate) fn seat_entry_cell(
    short: &str,
    elite_rank: usize,
    fit: Option<f32>,
    n: usize,
) -> String {
    format!(
        "+ {short} #{elite_rank} {} (n{n})",
        fit.map(fmt_line).unwrap_or_else(dash)
    )
}

/// One seat LOST: `− 37652b84 term 5/5 r3/20 · 1 no-update`.
pub(crate) fn seat_exit_cell(
    short: &str,
    reason: &str,
    held: &str,
    standing: Option<(usize, usize)>,
    gap: Option<usize>,
) -> String {
    let standing = standing
        .map(|(r, live)| format!(" r{r}/{live}"))
        .unwrap_or_default();
    let gap = gap.map(|g| format!(" · {g} no-update")).unwrap_or_default();
    format!("− {short} {reason} {held}{standing}{gap}")
}

/// The deal's ONE seat line: the occupancy AFTER the deal, then one `+`/`−` segment per seat
/// changed, in rank order.
pub(crate) fn seat_news_line(
    clock: usize,
    seat: usize,
    capacity: usize,
    news: &[String],
) -> String {
    format!(
        "step {clock} │ seats {seat}/{capacity} │ {}",
        news.join(" │ ")
    )
}

/// The `Minimal` frame's score cell: the population's mean score with the arithmetic that produced
/// it (`score = window avg ∓ λ × window spread`).
fn minimal_score_cell(
    fit_m: f32,
    fit_delta: &str,
    window_mean: f32,
    window_std: f32,
    direction: crate::utils::fitness::Direction,
    lambda: f32,
) -> String {
    // Maximize charges the spread with a minus, minimize with a plus — the same
    // operator `WindowStats::risk_adjusted` applies.
    let sign = match direction {
        crate::utils::fitness::Direction::Maximize => "−",
        crate::utils::fitness::Direction::Minimize => "+",
    };
    // `n = 1` has no observable spread: the leg prints `—`, never a `0.0000` that would
    // read as "perfectly steady" off a single reading (the same rule as the log line).
    let spread = if window_std.is_finite() {
        fmt_line(window_std)
    } else {
        "—".to_string()
    };
    format!(
        "score{} {} = window avg {} {} {} × window spread {}{}",
        direction.arrow(),
        fmt_line(fit_m),
        fmt_line(window_mean),
        sign,
        fmt_cfg(lambda),
        spread,
        fit_delta,
    )
}

#[cfg(test)]
mod tests {
    use super::lineage::*;
    use super::{
        CardStats, NetShape, StopPlan, delta_crown, delta_pop, elite_badge_entry, mean_of,
        minimal_score_cell, no_update_gap, origin_cell, seat_entry_cell, seat_exit_cell,
        seat_news_line,
    };
    use crate::utils::fitness::Direction;

    #[test]
    fn elite_badge_entry_says_whether_the_net_holds_a_seat() {
        // Seated: tenure out of the armed lifetime.
        assert_eq!(elite_badge_entry("abc12345", 3, 5), "abc12345(term 3/5)");
        // Elite but NOT seated (barred by the gate, or aged out) — the case
        // that used to print `(0/5)`, indistinguishable from a fresh seat.
        assert_eq!(elite_badge_entry("abc12345", 0, 5), "abc12345(no seat)");
        // Lifetime disarmed: no denominator invented.
        assert_eq!(elite_badge_entry("abc12345", 4, 0), "abc12345(term 4)");
    }
    #[test]
    fn lineage_parents_reads_the_ids_a_tree_walks() {
        // Crossover: two full hashes, in recorded order, `+mut` is a child flag not a parent.
        assert_eq!(
            lineage_parents(Some(
                "crossover-one-point:parents=aaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbb+mut"
            )),
            vec![
                "aaaaaaaaaaaaaaaa".to_string(),
                "bbbbbbbbbbbbbbbb".to_string()
            ]
        );
        // Asexual clone: one parent still walks like a tree edge.
        assert_eq!(
            lineage_parents(Some("clone:parents=cccccccccccccccc")),
            vec!["cccccccccccccccc".to_string()]
        );
        // Roots name no in-run parent: a founder, an inherited seed, or no record
        assert!(lineage_parents(Some("random")).is_empty());
        assert!(lineage_parents(Some("founder-2")).is_empty());
        assert!(lineage_parents(Some("seed:1793334914211:de1f105d")).is_empty());
        assert!(lineage_parents(None).is_empty());
    }

    #[test]
    fn the_no_update_gap_is_named_only_when_there_is_one() {
        // A net born at step 7 that trained 5 of its 12 lived clocks: the other 7
        // named, not left for the reader to subtract.
        assert_eq!(no_update_gap(5, 12), Some(7));
        // The two counters agree when nothing was withheld — and the seat line then adds
        // no cell at all, rather than a `· 0 no-update` that says nothing.
        assert_eq!(no_update_gap(0, 0), None);
        assert_eq!(no_update_gap(12, 12), None);
    }

    /// The seat deal is ONE line of news, and its segments say only what changed: the golden string
    /// is the shape agreed on after the old two-line pair was read back and found to repeat six
    #[test]
    fn the_seat_line_is_one_line_of_news() {
        let news = vec![
            seat_entry_cell("b2dc1462", 1, Some(47.5), 6),
            seat_exit_cell("37652b84", "term", "5/5", Some((3, 20)), Some(1)),
        ];
        assert_eq!(
            seat_news_line(6, 2, 2, &news),
            "step 6 │ seats 2/2 │ + b2dc1462 #1 47.5 (n6) │ − 37652b84 term 5/5 r3/20 · 1 no-update"
        );
        // A clean record has no gap cell, and a net with no standing omits the rank.
        assert_eq!(
            seat_exit_cell("aa11bb22", "outranked", "2/5", None, None),
            "− aa11bb22 outranked 2/5"
        );
        // A net the field never measured still took a seat: `n0`, never a missing cell.
        assert_eq!(
            seat_entry_cell("cc33dd44", 2, None, 0),
            "+ cc33dd44 #2 — (n0)"
        );
    }

    #[test]
    fn minimal_score_cell_prints_the_identity_it_was_built_from() {
        // The worked example from LOG_CLARITY_PLAN.md 3.1: λ = 0.50, population
        // score 37.4 out of a window level 48.9 with spread 23.0.
        let cell = minimal_score_cell(37.4, "", 48.9, 23.0, Direction::Maximize, 0.5);
        assert_eq!(
            cell,
            "score↑ 37.4 = window avg 48.9 − 0.50 × window spread 23.0"
        );
        // The printed legs close the identity for this run's λ — the reason they
        // are printed
        assert!((48.9f32 - 0.5 * 23.0 - 37.4).abs() < 0.05);
        // A step-over-step delta rides the score it belongs
        assert!(cell_with_delta().ends_with("× window spread 23.0 (∆+4.50)"));
        // Minimize charges the spread with a plus, exactly as the ranking does.
        let minimized = minimal_score_cell(37.4, "", 48.9, 23.0, Direction::Minimize, 0.5);
        assert!(
            minimized.contains("window avg 48.9 + 0.50 × window spread 23.0"),
            "{minimized}"
        );
    }

    /// The delta case for the test above, kept separate so the primary assertion
    /// reads as one golden string.
    fn cell_with_delta() -> String {
        minimal_score_cell(37.4, " (∆+4.50)", 48.9, 23.0, Direction::Maximize, 0.5)
    }

    use super::{accounting_population, accounting_rolls, summarize_rows};
    use crate::state::RunAccounting;

    #[test]
    fn summarize_rows_flattens_a_trainer_description() {
        let blob = serde_json::json!({
            "trainer": "cartpole",
            "lr": 0.01,
            "challenge_prob": 0.25,
            "optimizer": { "kind": "Adam", "weight_decay": 0.0 },
            "activations": ["relu", "tanh"],
            "schedules": null
        });
        let rows = summarize_rows(&blob);
        let get = |k: &str| {
            rows.iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(get("trainer").as_deref(), Some("cartpole"));
        // Floats display with 4 decimals (an f32 round-trips as a noisy f64).
        assert_eq!(get("lr").as_deref(), Some("0.0100"));
        assert_eq!(get("challenge_prob").as_deref(), Some("0.2500"));
        assert_eq!(get("optimizer.weight_decay").as_deref(), Some("0.0000"));
        assert_eq!(
            get("optimizer.kind").as_deref(),
            Some("Adam"),
            "nested keys flatten: {rows:?}"
        );
        assert_eq!(get("activations").as_deref(), Some("[\"relu\",\"tanh\"]"));
        // Nulls carry nothing and empty objects produce no rows
        assert!(get("schedules").is_none(), "{rows:?}");
        assert!(summarize_rows(&serde_json::json!({})).is_empty());
        assert!(summarize_rows(&serde_json::json!(null)).is_empty());
        // Bounded: a dump-heavy describe() gets an explicit marker row, not a screenful.
        let big: serde_json::Map<String, serde_json::Value> = (0..30)
            .map(|i| (format!("k{i}"), serde_json::json!(i)))
            .collect();
        let rows = summarize_rows(&serde_json::Value::Object(big));
        assert!(
            rows.iter().any(|(k, v)| k == "…" && v == "+18 more"),
            "{rows:?}"
        );
    }

    #[test]
    fn accounting_tables_tally_births_deaths_and_wasted_rolls() {
        let a = RunAccounting {
            born: 214,
            founders: 45,
            inherited_founders: 5,
            crossover_children: 65,
            // The uniform operator's exchange tally: 30 of its 65 births reported
            // count, 78 genes swapped of 180 compared.
            uniform_cx_births: 30,
            uniform_cx_swaps: 78,
            uniform_cx_pairs: 180,
            immigrants: 99,
            culls: 164,
            culls_crossover: 65,
            culls_immigrant: 99,
            crossover_rolls: 250,
            crossover_fired: 62,
            crossover_attempts: 171,
            crossover_discarded: 7,
            mutation_rolls: 250,
            mutation_fired: 49,
            mutation_blocked: 0,
            deferred_clocks: 12,
        };
        // Population tally: every counter the old prose carried, one cell each.
        let pop = accounting_population(&a, 50);
        let get = |k: &str| pop.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
        assert_eq!(get("born").as_deref(), Some("214"));
        assert_eq!(get("  founders (random)").as_deref(), Some("45"));
        assert_eq!(get("  founders (inherited)").as_deref(), Some("5"));
        assert_eq!(get("  mutations (random)").as_deref(), Some("99"));
        assert_eq!(get("  crossover children").as_deref(), Some("65"));
        assert_eq!(get("culled").as_deref(), Some("164"));
        assert_eq!(get("  culled (crossover roll)").as_deref(), Some("65"));
        assert_eq!(get("  culled (mutation roll)").as_deref(), Some("99"));
        assert_eq!(get("alive at stop").as_deref(), Some("50"));
        assert_eq!(get("evolution-deferred clocks").as_deref(), Some("12"));
        // Rolls matrix: one row per roll kind, one column per outcome. A roll
        // bought nothing is reported, not hidden.
        let (header, rows) = accounting_rolls(&a);
        assert_eq!(
            header,
            vec![
                "event",
                "rolls",
                "fired",
                "attempts",
                "built",
                "discarded",
                "blocked"
            ]
        );
        assert_eq!(
            rows[0],
            vec!["crossover", "250", "62", "171", "65", "7", "—"]
        );
        assert_eq!(rows[1], vec!["mutation", "250", "49", "—", "99", "—", "0"]);
    }

    /// The origin cell: crossover → TYPE + birth clock (no parent ids),
    /// random → birth clock, seeds keep the source run id + champion hash.
    #[test]
    fn origin_cell_names_type_and_birth_without_parent_ids() {
        assert_eq!(origin_cell(None, 7), "unknown@7");
        // A bare `random` says WHEN it entered.
        assert_eq!(origin_cell(Some("random"), 14), "random@14");
        // A founder's birth clock is 0 by definition; the label stands.
        assert_eq!(origin_cell(Some("founder-2"), 0), "founder-2");
        // Seeds keep the source run id + the champion hash (short8) —
        // is the inheritance provenance, not a parent id.
        assert_eq!(
            origin_cell(Some("seed:1791759444988:257486c8abcdef"), 0),
            "seed:1791759444988:257486c8"
        );
        // Crossovers: the OP (not the parents) + the birth clock;
        // `+mut` flag survives.
        assert_eq!(
            origin_cell(
                Some("crossover-one-point:parents=aaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbb"),
                14
            ),
            "cx one-point@14"
        );
        assert_eq!(
            origin_cell(
                Some("crossover-uniform:parents=aaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbb+mut"),
                9
            ),
            "cx uniform@9+mut"
        );
        // A uniform birth that recorded its exchange says HOW MUCH of the child came
        // from the other parent; the `+mut` flag keeps its place at the end.
        assert_eq!(
            origin_cell(
                Some("crossover-uniform:swapped=3/6:parents=aaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbb"),
                40
            ),
            "cx uniform@40 · 3/6"
        );
        // One-point reports its cut now, and old runs' strings have no `swapped=`
        // segment at all — both keep the exact old spelling either way.
        assert_eq!(
            origin_cell(
                Some(
                    "crossover-one-point:swapped=2/9:parents=aaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbb+mut"
                ),
                7
            ),
            "cx one-point@7 · 2/9+mut"
        );
        // Single-parent asexual child: same treatment.
        assert_eq!(
            origin_cell(Some("clone:parents=aaaaaaaaaaaaaaaa"), 5),
            "clone@5"
        );
        // Short hashes are not chopped further.
        assert_eq!(origin_cell(Some("seed:42:abc"), 0), "seed:42:abc");
    }

    // ── the shared structural legs of the two stop-time cards ─────────────

    /// A shape with one gene token — enough for `delta` to tell the cases below apart
    /// without building a blueprint.
    fn shape(token: &str) -> NetShape {
        NetShape {
            genome: [(token.to_string(), 1usize)].into_iter().collect(),
            nodes: 1,
            wires: 1,
            dim: 4.0,
        }
    }

    /// A shape with two gene tokens, ONE of them `W:Tanh` — so its Δ to [`shape`]
    /// lands on 0.5 (one shared gene out of two) rather than 0 or 1.
    fn shape_pair() -> NetShape {
        NetShape {
            genome: [("W:Tanh".to_string(), 1usize), ("W:SiLU".to_string(), 1)]
                .into_iter()
                .collect(),
            nodes: 1,
            wires: 2,
            dim: 8.0,
        }
    }

    /// A plan whose crown is `W:Tanh`, `W:Tanh+W:SiLU`, `W:ReLU` by hash `a/b/c` — the reference
    /// set both Δ legs are measured against, WITH the hash→crown-slot map the row builder actually
    fn plan_of_three() -> StopPlan {
        let crown_idx = crown_idx_of(&["a", "b", "c"]);
        StopPlan {
            header: Vec::new(),

            columns: Vec::new(),
            has_exam: false,
            has_eval: false,
            dealt_clock: 7,
            crown: vec![shape("W:Tanh"), shape_pair(), shape("W:ReLU")],
            crown_idx,
            population: vec![
                ("a".to_string(), shape("W:Tanh")),
                ("b".to_string(), shape_pair()),
                ("c".to_string(), shape("W:ReLU")),
            ],
        }
    }

    /// The crown slot of each hash, in rank order — the same map `stop_plan` builds.
    fn crown_idx_of(hashes: &[&str]) -> std::collections::HashMap<String, Option<usize>> {
        hashes
            .iter()
            .enumerate()
            .map(|(i, h)| (h.to_string(), Some(i)))
            .collect()
    }

    /// The two Δ legs must compare against the RIGHT set — the whole point of putting them on both
    /// cards: a champion skips its own row (against itself every champion would read a meaningless
    #[test]
    fn delta_crown_skips_the_champions_own_row() {
        let plan = plan_of_three();
        let me = shape("W:Tanh");
        // The row looks its OWN slot up by hash (what `stop_card` does) —
        // position in the printed table.
        let slot = |hash: &str| plan.crown_idx.get(hash).copied().flatten();
        // Hash `a` IS champion 0: 0.5 to `W:Tanh+W:SiLU`, 1.0 to `W:ReLU` ⇒ 0.75.
        assert_eq!(slot("a"), Some(0));
        assert_eq!(delta_crown(Some(&me), &plan, slot("a")), Some(0.75));
        // Champion 1 skips ITS own slot the same way: 0.5 to Tanh, 1.0 to ReLU ⇒ 0.75.
        assert_eq!(
            delta_crown(Some(&shape_pair()), &plan, slot("b")),
            Some(0.75)
        );
        // A net with NO crown slot (any other hash) is measured against all three,
        // coincidence with champion 0 counts: (0 + 0.5 + 1.0) / 3.
        assert_eq!(slot("zz"), None);
        assert_eq!(delta_crown(Some(&me), &plan, slot("zz")), Some(0.5));
        // An unreadable blueprint prints `—`, never a number.
        assert_eq!(delta_crown(None, &plan, slot("a")), None);
        // `k = 1`: a champion has no OTHER champion to be measured against ⇒ `—`.
        let solo = StopPlan {
            header: Vec::new(),

            columns: Vec::new(),
            has_exam: false,
            has_eval: false,
            dealt_clock: 7,
            crown: vec![shape("W:Tanh")],
            crown_idx: crown_idx_of(&["a"]),
            population: Vec::new(),
        };
        assert_eq!(delta_crown(Some(&me), &solo, solo.crown_idx["a"]), None);
    }

    /// `Δpop` is the crowd, minus the row itself — and `—` (never `0.00`) when there
    /// nothing to compare against.
    #[test]
    fn delta_pop_skips_the_row_itself() {
        let plan = plan_of_three();
        let me = shape("W:Tanh");
        // Live at `a`: the other two are 0.5 and 1.0 away ⇒ 0.75.
        assert_eq!(delta_pop(Some(&me), "a", &plan), Some(0.75));
        // A hash that is not in the population (or not live) is compared to all three.
        assert_eq!(delta_pop(Some(&me), "zz", &plan), Some(0.5));
        assert_eq!(delta_pop(None, "a", &plan), None);
        // A lone net has no other live net to be unlike ⇒ `—`.
        let lone = StopPlan {
            population: vec![("a".to_string(), shape("W:Tanh"))],
            ..plan
        };
        assert_eq!(delta_pop(Some(&me), "a", &lone), None);
    }

    /// The card's own means: `—` legs are unreadable blueprints, not zeros,
    /// empty card says `—` rather than inventing a `0.00`.
    #[test]
    fn card_stats_average_only_the_rows_that_have_a_number() {
        assert_eq!(mean_of(&[Some(1.0), None, Some(0.5)]), Some(0.75));
        assert_eq!(mean_of(&[None, None]), None, "nothing to average");
        assert_eq!(mean_of(&[]), None);
        let stats = CardStats {
            delta_crown: Some(0.5),
            delta_pop: None,
        };
        assert_eq!(stats.line(), "card mean: Δcrown 0.50 · Δpop —");
    }
}
