//! The engine's user-facing output: the per-step rollup (`Summ`), the framed
//! in-place table (`Minimal`), and the stop-time summary.
//!
//! Reporting is a pure consumer of engine state — it never mutates the
//! population, the clock, or the ledger (the one exception is the `Minimal`
//! delta baseline, `minimal_prev_means`). Everything else in the engine stays
//! free to read these numbers; nothing here feeds back into the search.

use tracing::info;

use super::core::CoreEngine;
use crate::engine::smoothing::rolling_mean;

impl CoreEngine {
    /// One rollup line per step: pop size, the train-loss mean ± std, the
    /// mode's middle column, and the fitness column in the same shape. Two
    /// decimals throughout — the spread is the signal worth reading, the
    /// extremes are noise. Printed LAST for the step (see the run loop), so
    /// it lands at the bottom of the terminal.
    ///
    /// Middle column by mode: Tabular = held-out `eval_loss`; RL = the step's
    /// environment volume (`matches`/`turns`, from [`RlVolume`]) — an RL net
    /// has no held-out batch, and the volume is what explains the wall time.
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
        // Population mean ± population std, 2 decimals.
        let stats = |v: &[f32]| {
            if v.is_empty() {
                "—".to_string()
            } else {
                let n = v.len() as f32;
                let mean = v.iter().sum::<f32>() / n;
                let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
                format!("{mean:.2} ± {:.2}", var.sqrt())
            }
        };
        // Per-step wall time: the step's own cost (train + evolve), not time
        // since run start. Matters most in RL (one step = many full matches).
        let step_secs = self.step_started_at_wall.elapsed().as_secs_f32();
        // Fitness column: same mean ± std shape so all three read alike.
        let fits: Vec<f32> = self
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| {
                self.rolling_fitness
                    .get(h.as_str())
                    .filter(|b| !b.is_empty())
            })
            .map(rolling_mean)
            .collect();
        let fit_stats = stats(&fits);
        // The mode's middle column: tabular reports the held-out eval loss,
        // RL reports the environment volume it actually played.
        let mid = if !self.trainer.is_rl() {
            format!("eval_loss↓ {}", stats(&evals))
        } else {
            self.step_rl.label()
        };
        // Per-step rollup: Summ = compact one-liner; Minimal = framed table
        // (from step 2 — deltas need a prior step). The ★ badge names the
        // CURRENT elite set (top-`elite_count` by smoothed fitness this
        // step) — it explains who is immune to culls right now. With
        // `freeze_elites` on, every elite seat is frozen each step, so the
        // set needs no per-net marker: the badge lists the names, each with
        // its SEAT AGE — `hash(seat/j)` — because the freeze lines only ever
        // show an entrant's own age (always 0 at entry, printed before the
        // tick), so without this the reader can never see how long a seated
        // net has been sitting, i.e. how close it is to the lifetime age-out.
        let freeze_badge = {
            let elites = self.elite_hashes();
            if elites.is_empty() {
                String::new()
            } else {
                let j = self.config.elite_crown_lifetime_steps;
                let seat_age = |h: &String| {
                    self.state
                        .net(h)
                        .map(|s| s.elite_seat_age)
                        .unwrap_or(0)
                };
                // `(seat/j)` only when the lifetime is armed (`j > 0`) — with
                // j = 0 there is no age-out, so the number would mean nothing.
                let mut names: Vec<String> = elites
                    .iter()
                    .map(|h| {
                        let short = &h[..8.min(h.len())];
                        if j == 0 {
                            short.to_string()
                        } else {
                            format!("{short}({}/{j})", seat_age(h))
                        }
                    })
                    .collect();
                names.sort();
                format!(" │ ★ {}", names.join(" "))
            }
        };
        if self.log_level == crate::engine::config::LogLevel::Minimal {
            self.log_minimal_table(&trains, &evals, &fits, step_secs);
        } else {
            info!(
                "step {} │ pop {} │ train_loss↓ {} │ {} │ fitness{} {} │ took {:.1}s{}",
                clock,
                self.state.live_count(),
                stats(&trains),
                mid,
                self.fitness.direction().arrow(),
                fit_stats,
                step_secs,
                freeze_badge,
            );
        }
    }

    /// `Minimal` mode: one boxed vitals table per step (starting at step 2 —
    /// step 1 has no prior step to diff against). The frame sink redraws the
    /// block in place on a terminal, so only the numbers move; off a terminal
    /// it prints plainly, with no escape codes in the file. Deltas reuse
    /// what the engine already reports (population-mean smoothed values vs the
    /// last step); a zero delta drops the parenthetical. The evolve counters
    /// and the best-net footer are plain current values — no delta on the
    /// footer. Nothing else is printed per step in this mode.
    ///
    /// Row 1's tail is mode-dependent, exactly like the `Summ` line: tabular
    /// shows the held-out `eval_loss`, RL shows the step's environment volume
    /// (`matches`/`turns`). Rendered at the END of the step, so the evolve
    /// counters in the frame are this step's own.
    pub(crate) fn log_minimal_table(
        &mut self,
        trains: &[f32],
        evals: &[f32],
        fits: &[f32],
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
        // Delta vs the previous step's mean; omitted entirely when 0. The
        // first table render (step 2) sets the baseline without deltas.
        // Only surface a delta that survives 2-decimal rounding, so the table
        // never shows a meaningless `(∆+0.00)`.
        let delta = |cur: f32, prev: Option<f32>| match prev {
            Some(p) if (cur - p).abs() >= 5e-3 => format!(" (∆{:+.2})", cur - p),
            _ => String::new(),
        };
        let (pt, pe, pf) = self
            .minimal_prev_means
            .unwrap_or((f32::NAN, f32::NAN, f32::NAN));
        let pop = self.state.live_count();
        let e = self.step_evolve;
        // Rows first, THEN size the box: a fixed width overflowed on long
        // delta lines and swallowed the right border. Padding counts chars,
        // not bytes — ↓/↑/∆ are 3 bytes each but one column wide.
        let mut rows: Vec<String> = vec![
            if !self.trainer.is_rl() {
                format!(
                    "pop {:>3} │ train_loss↓ {:.2}{} │ eval_loss↓ {:.2}{}",
                    pop,
                    train_m,
                    delta(train_m, Some(pt)),
                    eval_m,
                    delta(eval_m, Some(pe)),
                )
            } else {
                format!(
                    "pop {:>3} │ train_loss↓ {:.2}{} │ {}",
                    pop,
                    train_m,
                    delta(train_m, Some(pt)),
                    self.step_rl.label(),
                )
            },
            // `smt` tag: this rollup cell is the population MEAN of smoothed
            // per-net fitness — tagged so the rollup and the per-net raw
            // values on other lines are never confused.
            format!(
                "fitness{} smt{:.2}{} │ culls {} │ inserts {}",
                self.fitness.direction().arrow(),
                fit_m,
                delta(fit_m, Some(pf)),
                e.culls,
                e.inserts,
            ),
            format!(
                "crossover fired {} ({} inserted, {} spent) │ mutation rolled {} ({} immigrant(s))",
                e.cross_fired, e.cross_survived, e.cross_discarded, e.mutate_fired, e.mutate_fired,
            ),
            format!("step took {:.1}s", step_secs),
        ];
        // Anti-devolution row — only when a knob is on (zero noise by
        // default): the elite set and/or the current demotion count. Same
        // annotation style as the Summ line's badge.
        let mut guard_cells: Vec<String> = Vec::new();
        if self.config.freeze_elites || self.config.elite_count > 0 {
            let elites = self.elite_hashes();
            let mut names: Vec<String> = elites
                .iter()
                .map(|h| h[..8.min(h.len())].to_string())
                .collect();
            names.sort();
            guard_cells.push(format!("elite: {}", names.join(" ")));
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
        // EFFICIENCY NOTE (Tier 1 — `&str` over `&String`): a `&String` is a
        // double pointer (ptr + len + capacity); the closure only READS the
        // text, so `&str` suffices and accepts string literals too. Perk:
        // one deref less and the borrower's `String` needn't exist — borrow
        // the `&str` view directly.
        let pad = |s: &str| {
            let visible = s.chars().count();
            format!("│ {}{} │", s, " ".repeat(inner.saturating_sub(visible)))
        };
        let rule = |left: char, right: char| format!("{left}{}{right}", "─".repeat(inner + 2));

        // One boxed block, handed to the frame sink as a whole so it can be
        // redrawn in place: the divider splits what the population IS (state
        // rows) from what the engine DID to it (rolls, timing, elite seats).
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

    /// Stop-time summary: which JSONs were written as the live frontier and
    /// the resume command to continue from here.
    pub(crate) fn log_stop_summary(&self, clock: usize) -> flodl::tensor::Result<()> {
        let live = self.state.live_hashes();
        info!(
            "  wrote {} frontier snapshot(s) → nets/<hash>.json (one per live net)",
            live.len(),
        );
        // The population size is part of the resume call (the engine refuses a
        // mismatch instead of silently patching it), so print the number the
        // caller must pass — not just the directory. Phrased in ENGINE terms,
        // not CLI flags: only the long-run example keeps a `--resume` flag, so
        // a flag spelling here would be wrong for most runs.
        info!(
            "  {} live net(s) at step {} → resume from {} with pop_size {}",
            live.len(),
            clock,
            self.run_dir.display(),
            live.len(),
        );

        // Final elite report: the top-k nets by smoothed fitness, with their
        // metadata — the run's champions at stop time. Elite = rank, not
        // identity, so this is "who holds the top-k right now".
        let k = self.config.elite_count.max(1); // report at least the champion
                                                // (hash, smoothed, step, eval_loss)
        let ranked: Vec<(String, f32, usize, Option<f32>)> = self
            .ranked_live()
            .into_iter()
            .map(|(h, fit)| {
                let metrics = self.state.net(&h).and_then(|s| s.last_metrics.as_ref());
                let step = metrics.map(|m| m.step).unwrap_or(0);
                let eval_loss = metrics.and_then(|m| m.eval_loss);
                (h, fit, step, eval_loss)
            })
            .collect();
        let direction = self.fitness.direction();
        let arrow = direction.arrow();
        // The list below is RANK-top-k, not the crown: a net can rank top-k
        // while still barred by the elite gate (it never held a seat). The
        // gate numbers here say how the population actually split, and each
        // line names that net's own seat state.
        //
        // Counted at the LAST DEALT clock — the report runs after the final
        // group step, so `step_clock()` is one past it and would grade every
        // net's born-window `k` clocks late (see `elite_gate_counts`).
        let (eligible, probation, age_blocked) =
            self.elite_gate_counts(self.step_clock().saturating_sub(1));
        // Compact one-line-per-elite table, built with `comfy-table` so the
        // column widths are computed from the actual cell content. The old
        // hand-padded `format!` table had three defects: the separator's
        // `┬` characters landed in the wrong columns, the footer header row
        // was printed twice (once above the bottom border, once below it),
        // and the `step` column actually printed `entered_at_step` (the net's
        // birth clock) while the real last-metric step was never shown. The
        // table below prints the true recorded step, the seat age only for
        // nets that actually hold a seat, and one header row at the top.
        let mut table = comfy_table::Table::new();
        table.load_preset(comfy_table::presets::UTF8_FULL_CONDENSED);
        table.set_header(vec![
            "#",
            "hash",
            "smt",
            "step",
            "seat",
            "born",
            "params",
            "origin",
            "eval",
        ]);
        for (rank, (h, fit, step, eval_loss)) in ranked.iter().take(k).enumerate() {
            // ± std over the smoothing window, same framing as the holdout
            // guardrail: the mean alone hides the mix of raw values behind it
            // (two elites at 500 can be 500-always vs 500-by-luck). A wide ±
            // means the smoothed value rests on noisy raw steps — and at race
            // end, that the ORDERING of this list itself is luck-sensitive.
            // eval_loss: `—` when the mode doesn't produce one (RL trainers
            // report fitness only). `None` is Rust-speak; a dash reads as
            // "not applicable", which is the truth.
            // `smt` tag: the ranked figure is the smoothed fitness (decided on
            // it); the ± note is the spread of RAW steps behind it.
            let smt = crate::engine::format::fmt2_smt(*fit);
            let short = &h[..8.min(h.len())];
            let state = self.state.net(h);
            let on_crown = self.config.freeze_elites && self.frozen_crown.contains(h);
            let seat = if on_crown {
                state
                    .map(|ns| format!("{}", ns.elite_seat_age))
                    .unwrap_or_else(|| "?".to_string())
            } else {
                "—".to_string()
            };
            let born = state
                .map(|ns| ns.entered_at_step.to_string())
                .unwrap_or_else(|| "?".to_string());
            let origin_label = state
                .as_ref()
                .and_then(|s| s.created_from.as_deref())
                .unwrap_or("unknown")
                .split(' ')
                .next()
                .unwrap_or("unknown")
                .chars()
                .take(8)
                .collect::<String>();
            let params_label = state
                .and_then(|s| s.meta.params)
                .map(|p| p.to_string())
                .unwrap_or_else(|| "?".to_string());
            let eval = eval_loss
                .map(|l| format!("{l:.4}"))
                .unwrap_or_else(|| "—".to_string());
            table.add_row(vec![
                (rank + 1).to_string(),
                short.to_string(),
                smt,
                step.to_string(),
                seat,
                born,
                params_label,
                origin_label,
                eval,
            ]);
        }
        // comfy-table computes the widths; indent each line so the block
        // sits inside the summary's two-space margin.
        for line in table.to_string().lines() {
            info!("  {line}");
        }
        info!(
            "  ── final elites (top-{k}, fitness{arrow} smoothed) ── [crown {}/{} seats: {} eligible, {} on probation (k={}), {} age-blocked (j={})]",
            self.frozen_crown.len(),
            self.config
                .elite_count
                .min(self.state.live_count().saturating_sub(1)),
            eligible,
            probation,
            self.config.elite_join_probation_steps,
            age_blocked,
            self.config.elite_crown_lifetime_steps,
        );
        Ok(())
    }
}
