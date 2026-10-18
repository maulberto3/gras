//! The engine's user-facing output: the per-step rollup (`Summ`), the framed
//! in-place table (`Minimal`), and the stop-time summary.
//!
//! Reporting is a pure consumer of engine state — it never mutates the
//! population, the clock, or the ledger (the one exception is the `Minimal`
//! delta baseline, `minimal_prev_means`). Everything else in the engine stays
//! free to read these numbers; nothing here feeds back into the search.

use tracing::info;

use super::core::CoreEngine;
use crate::engine::child::frozen_act_count;
use crate::engine::format::{
    ReadingStyle, fmt_cell, fmt_cell_opt, fmt_cell_smt, fmt_census, fmt_cfg, fmt_line, fmt_reading,
};
use crate::engine::smoothing::rolling_mean;

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
        // Population mean ± population std via the census spelling: every net in the
        // population is measured, so this is not a sample and carries no `SE` (see
        // `format::fmt_census`). Log-line precision, floored so a small loss never
        // reads as zero.
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
            // Fitness column: same mean ± std shape so all three read alike. Since FIT the
            // value is the per-net RISK-ADJUSTED score (what the race decides on), with the
            // readable window level and spread printed right beside it: `μ` is "how much does
            // the population's net actually score", `σ` the spread it is charged for.
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
            .map(|r| r.risk_adjusted(direction, lambda))
            .collect();
        let windows: Vec<f32> = readings.iter().map(|r| r.mean).collect();
        let spreads: Vec<f32> = readings.iter().map(|r| r.std).collect();
        // A plain average (not `stats`): these are single population aggregates, not a
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
                let j = self.config.elite_crown_lifetime_steps;
                let names: Vec<String> = elites
                    .iter()
                    .map(|h| {
                        let short = &h[..8.min(h.len())];
                        let seat_age =
                            self.state.net(h).map(|s| s.elite_seat_age).unwrap_or(0);
                        elite_badge_entry(short, seat_age, j)
                    })
                    .collect();
                format!(" │ ★ {}", names.join(" "))
            }
        };
        // The score identity, printed with its own arithmetic: the mean of the
        // per-net `avg ∓ λ·spread` equals `window avg ∓ λ·window spread` because λ
        // is a constant. Without the two legs and the operator, `score 37.4` beside
        // `window avg 48.9` reads as a contradiction instead of a charge for spread.
        let window_avg = avg(&windows);
        let window_spread = avg(&spreads);
        let sign = match direction {
            crate::utils::fitness::Direction::Maximize => "−",
            crate::utils::fitness::Direction::Minimize => "+",
        };
        let score_cell = if readings.is_empty() {
            format!(
                "score{} avg — (no reading yet) │ window avg —",
                direction.arrow()
            )
        } else {
            format!(
                "score{} avg {} = window avg {} {} {} × window spread {} (λ={}) │ window avg {} (spread {})",
                direction.arrow(),
                fmt_line(avg(&fits)),
                fmt_line(window_avg),
                sign,
                fmt_cfg(lambda),
                fmt_line(window_spread),
                fmt_cfg(lambda),
                fmt_line(window_avg),
                fmt_line(window_spread),
            )
        };
        if self.log_level == crate::engine::config::LogLevel::Minimal {
            self.log_minimal_table(&trains, &evals, &fits, window_avg, window_spread, step_secs);
        } else {
            info!(
                "step {} │ nets {} │ loss↓ {} │ {} │ {} │ {}s{}",
                clock,
                self.state.live_count(),
                stats(&trains),
                mid,
                score_cell,
                fmt_line(step_secs),
                freeze_badge,
            );
        }
    }

        /// `Minimal` mode: one boxed vitals table per step (step 2+), redrawn in place on a
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
            // Delta vs the previous step's mean; omitted when 0 or when it dies at
            // 2-decimal rounding (the first render sets the baseline).
        // Gate the delta on the PRINTED form, not a magic epsilon: a change is
        // shown exactly when the line's own precision can represent it. The first
        // render carries a NaN baseline, which is omitted (`!is_finite`).
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
                // The population MEAN of smoothed per-net scores, printed with the
                // arithmetic the Summ line uses — the two sinks must not disagree.
                // The window census is not repeated: it is the identity's own legs.
            format!(
                "score{} {} = window avg {} {} {} × window spread {}{} │ culls {} │ inserts {}",
                self.fitness.direction().arrow(),
                fmt_line(fit_m),
                fmt_line(window_mean),
                match self.fitness.direction() {
                    crate::utils::fitness::Direction::Maximize => "−",
                    crate::utils::fitness::Direction::Minimize => "+",
                },
                fmt_cfg(self.config.risk_aversion),
                fmt_line(window_std),
                delta(fit_m, Some(pf)),
                e.culls,
                e.inserts,
            ),
            format!(
                "crossover fired {} ({} inserted, {} spent) │ immigrant rolled {} ({} inserted)",
                e.cross_fired, e.cross_survived, e.cross_discarded, e.mutate_fired, e.mutate_fired,
            ),
            format!("step took {}s", fmt_line(step_secs)),
        ];
            // Anti-devolution row — only when a knob is on (zero noise by default): the
            // elite set and/or the current demotion count.
        let mut guard_cells: Vec<String> = Vec::new();
        if self.config.freeze_elites || self.config.elite_count > 0 {
            let elites = self.elite_hashes();
            let mut names: Vec<String> = elites
                .iter()
                .map(|h| h[..8.min(h.len())].to_string())
                .collect();
            names.sort();
            // An armed crown with no members (pop too small for a seat) must not print
            // a labelled empty row.
            if !names.is_empty() {
                guard_cells.push(format!("elite: {}", names.join(" ")));
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

            // One boxed block handed to the frame sink whole, so it can be redrawn in
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

    /// Stop-time summary: which JSONs were written as the live frontier and
    /// the resume command to continue from here.
    pub(crate) fn log_stop_summary(&self, clock: usize) -> flodl::tensor::Result<()> {
        let live = self.state.live_hashes();
        // CFY: the stop block is comfy tables — an artifact/resume card, the accounting
        // matrix, then the elite card. None of these lines starts with `step`, so none
        // stays a one-liner; the prose collapses into a card footnote.
        let artifact_rows = vec![
            (
                "frontier snapshots".to_string(),
                format!("{} → nets/<hash>.json (one per live net)", live.len()),
            ),
            ("live nets at stop".to_string(), live.len().to_string()),
            ("stop step".to_string(), clock.to_string()),
            (
                "resume dir".to_string(),
                self.run_dir.display().to_string(),
            ),
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
            Some("resume parity: replay-relevant knobs come from engine.json — a mismatch is refused, not patched"),
        ));

            // What the run DID comes before what it produced: a population tally and a
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
            Some("built = nets admitted; discarded = crossover rolls that bought no viable child; blocked = immigrant rolls stopped by the cull shield"),
        ));

            // Final elite report: the top-k nets by smoothed fitness — the run's
            // champions. Elite = rank, not identity.
        let k = self.config.elite_count.max(1); // report at least the champion
            // (hash, risk-adjusted score, window mean, window σ, readings n, step,
            // eval_loss). The score is the ranking value (FIT); the mean/σ/n legs are the
            // reading it was charged from, rendered by the ONE reading formatter.
        let ranked: Vec<(String, f32, Option<f32>, Option<f32>, usize, usize, Option<f32>)> = self
            .ranked_live()
            .into_iter()
            .map(|(h, fit)| {
                let reading = self.race_reading_of(&h);
                let metrics = self.state.net(&h).and_then(|s| s.last_metrics.as_ref());
                let step = metrics.map(|m| m.step).unwrap_or(0);
                let eval_loss = metrics.and_then(|m| m.eval_loss);
                (
                    h,
                    fit,
                    reading.map(|r| r.mean),
                    // A single-reading window has no observable spread — `None`, not 0.0,
                    // which would read as "perfectly steady".
                    reading.and_then(|r| (r.n > 1).then_some(r.std)),
                    reading.map(|r| r.n).unwrap_or(0),
                    step,
                    eval_loss,
                )
            })
            .collect();
        let direction = self.fitness.direction();
        let arrow = direction.arrow();
            // The list is RANK-top-k, not the crown: a net can rank top-k while barred by
            // the elite gate. Counted at the LAST DEALT clock (not `step_clock()`).
        let dealt_clock = self.step_clock().saturating_sub(1);
        let (eligible, probation, age_blocked) = self.elite_gate_counts(dealt_clock);
            // Compact one-line-per-elite table via `comfy-table`, so column widths come
            // from cell content; prints the true recorded step and one header row.
        let mut table = comfy_table::Table::new();
        table.load_preset(comfy_table::presets::UTF8_FULL_CONDENSED);
            // RL trainers report fitness only — show the `eval_loss` column only when at
            // least one reported net carries one.
        let has_eval = ranked.iter().take(k).any(|(_, _, _, _, _, _, e)| e.is_some());
            // Every column recycles state the engine already carries (`trained`,
            // `frozen`, each informative metric labeled as in `history.csv`).
            // `reading` is ONE column, not a `mean` + `spread` pair: the spread is only
            // meaningful with the n it came from, so `μ ±σ (SE, n)` is rendered whole
            // from `format::fmt_reading` — the same spelling the logs use.
        let mut header = vec![
            "#", "hash", "score", "reading", "score/tkt", "step", "seat age", "trained",
            "frozen", "idle", "params", "origin",
        ];
        if has_eval {
            header.push("eval");
        }
        for m in &self.metrics {
            header.push(m.label());
        }
        table.set_header(header);
        for (rank, (h, fit, mean, std, n, step, eval_loss)) in ranked.iter().take(k).enumerate() {
                // Two legs of ONE reading: `smt` = the risk-adjusted score the rank was
                // decided on (FIT); `reading` = `μ ±σ (SE, n)` behind it. A wide σ means the
                // ordering is luck-sensitive; a small n is what makes SE (not σ) the right
                // bar for comparing it against something else. eval_loss `—` when N/A.
            let smt = fmt_cell_smt(*fit);
            let reading_cell = fmt_reading(*mean, *std, *n, ReadingStyle::Cell);
            let short = &h[..8.min(h.len())];
            let state = self.state.net(h);
                // The ticket lottery's number, readable: window level per TICKET — the
                // clocks that actually ran the trainer and scored it. An idle clock is
                // not a ticket, so it cannot dilute the divisor, and sitting out leaves
                // both legs untouched. `—` for a net with no reading yet.
            let born_num = state.map(|ns| ns.entered_at_step).unwrap_or(0);
            let age = dealt_clock.saturating_sub(born_num) + 1;
            let per_ticket = self
                .merit_per_ticket(h)
                .map(fmt_cfg)
                .unwrap_or_else(|| "—".to_string());
                // Clocks the lottery left IDLE (no train, no eval). `frozen` stays the
                // crown's own count, so the two causes never hide inside one number.
            let idle = state
                .map(|s| s.idle_clocks().to_string())
                .unwrap_or_else(|| "—".to_string());
            let on_crown = self.config.freeze_elites && self.frozen_crown.contains(h);
            let seat = if on_crown {
                state
                    .map(|ns| format!("{}", ns.elite_seat_age))
                    .unwrap_or_else(|| "?".to_string())
            } else {
                "—".to_string()
            };
            let origin_label = origin_cell(
                state.as_ref().and_then(|s| s.created_from.as_deref()),
                born_num,
            );
            let params_label = state
                .and_then(|s| s.meta.params)
                .map(|p| p.to_string())
                .unwrap_or_else(|| "?".to_string());
                // Clocks lived INCLUSIVE of the birth clock and the last run — same
                // convention the freeze lines use (`clock − entered_at_step`); `age` is
                // computed above, where the `smt/tkt` divisor needs it too.
            // `trained n/m` as two cells: optimizer updates over clocks lived.
            let trained = state
                .map(|ns| format!("{}/{age}", ns.step))
                .unwrap_or_else(|| "—".to_string());
            let frozen = state
                .map(|ns| frozen_act_count(ns).to_string())
                .unwrap_or_else(|| "—".to_string());
            let mut row = vec![
                (rank + 1).to_string(),
                short.to_string(),
                smt,
                reading_cell,
                per_ticket,
                step.to_string(),
                seat,
                trained,
                frozen,
                idle,
                params_label,
                origin_label,
            ];
            if has_eval {
                row.push(
                    fmt_cell_opt(eval_loss),
                );
            }
            // One cell per configured informative metric, in config order —
            // the same order `history.csv` writes its columns in.
            let metrics = state.and_then(|ns| ns.last_metrics.as_ref());
            for i in 0..self.metrics.len() {
                row.push(
                    metrics
                        .and_then(|m| m.informative.get(i).copied())
                        .map(fmt_cell)
                        .unwrap_or_else(|| "—".to_string()),
                );
            }
            table.add_row(row);
        }
        // CFY: the elite report is a card — the same comfy table, now with the gate
        // tally and the score/reading legend as its footnote (they were a trailing prose line).
        let sign = match self.fitness.direction() {
            crate::utils::fitness::Direction::Maximize => "−",
            crate::utils::fitness::Direction::Minimize => "+",
        };
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            &format!("Final elites (top-{k}, ranked by fitness{arrow})"),
            &table.to_string(),
            Some(&format!(
                "score = risk-adjusted score (window avg {} λ × window spread, λ={}) · reading = the window level and its step-to-step spread (SE = spread/√n, the uncertainty of that mean, over n readings) · rank charges spread, comparisons need SE · score/tkt = risk-adjusted score per TICKET (one clock trained and scored; what the ticket lottery grades — until the first ticket) · idle = clocks the lottery gave no ticket (no train, no eval, so nothing above moved) · crown {}/{} seats · {} eligible · {} gated (join window or cull shield, {} clocks) · {} age-blocked ({} clocks)",
                sign,
                fmt_cfg(self.config.risk_aversion),
                self.frozen_crown.len(),
                self.config
                    .elite_count
                    .min(self.state.live_count().saturating_sub(1)),
                eligible,
                probation,
                self.config.elite_join_probation_steps,
                age_blocked,
                self.config.elite_crown_lifetime_steps,
            )),
        ));

        Ok(())
    }

        /// The training lottery's per-net card for ONE clock — the DEBUG view (the caller
        /// gates it on the debug filter), printed at the START of the clock's step, not
        /// once at stop, so the numbers are the coin's own INPUTS and `p(ticket)` can be
        /// checked against the field it was mapped from. The `Summ` view is the single
        /// aggregate line from `log_lottery_summary`.
        ///
        /// `frame` is the field `lottery_frame(clock)` built for this clock, passed in
        /// by the caller that also hands it to `train_lottery_draws` — never a
        /// recomputation: after the nets step, their readings have already moved on, so
        /// a field rebuilt here would grade a clock that never happened.
        ///
        /// `idle` is therefore the count BEFORE this clock draws; the `Summ` lottery line
        /// after the loop states what this clock itself idled (`won`/`idled`).
        ///
        /// Silent when the knob is off or the field is not armed yet (the caller checks
        /// both: an unarmed clock grades nobody, so there is no card to print).
        ///
        /// Lists EVERY live net (the lottery grades the whole field, not the top-k),
        /// the graded rows first by `p` descending, then the exempt ones.
    pub(crate) fn log_lottery_card(&self, clock: usize, frame: super::step::LotteryFrame) {
            // (exempt?, p, row): the nets the lottery actually grades come first,
            // so the exempt rows (which all grade 1.00 by construction) cannot
            // bury the graded field at the top of the card.
        let mut rows: Vec<(bool, f32, Vec<String>)> = self
            .state
            .live_hashes()
            .into_iter()
            .map(|h| {
                let state = self.state.net(&h);
                // The BLENDED probability — the one the coin compares against —
                // so `p` and the `ticket?` column beside it cannot disagree.
                let p = self.train_lottery_p_eff(&h, clock, Some(frame));
                    // Why a net is outside the lottery's reach, in one word: the
                    // crown and the born shield always train (p = 1.00).
                let exempt = if self.frozen_crown.contains(&h) {
                    "crown"
                } else if self.on_probation(&h, clock) {
                    "shield"
                } else {
                    "—"
                };
                    // `smt / tkt` is ONE quantity, so it prints as its two legs and the
                    // ratio between them: the numerator here, the divisor (`measured
                    // clocks` — real updates plus crown act passes) and the quotient
                    // beside each other, left to right. A reader can redo the division.
                    // The numerator takes the CELL precision (4dp, widened below 5e-5) so
                    // a small loss-based smt never rounds to `0.00` while the ratio
                    // beside it is plainly non-zero.
                let smt = self.smoothed_fitness_of(&h);
                    // This clock's coin, asked the same way the step will ask it:
                    // `train_lottery_draws` is a pure function of (run seed, clock, hash,
                    // frame) and consumes nothing, so calling it here returns the answer
                    // the step is about to act on — not a second draw that could differ.
                    // The crown is forced `no`: the elite freeze takes the trainer step
                    // for its OWN reason, so its `p = 1.00` means "the lottery is not what
                    // denied it", never "it misses this clock's ticket".
                let trains = !self.frozen_crown.contains(&h)
                    && self.train_lottery_draws(&h, clock, Some(frame));
                let row = vec![
                    h[..8.min(h.len())].to_string(),
                    exempt.to_string(),
                    smt.map(fmt_cell).unwrap_or_else(|| "—".to_string()),
                    state
                        .map(|s| s.measured_clocks().to_string())
                        .unwrap_or_else(|| "—".to_string()),
                    self.merit_per_ticket(&h)
                        .map(fmt_cfg)
                        .unwrap_or_else(|| "—".to_string()),
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
            // The field's own bounds, converted back into the COLUMN's units (the frame
            // is stored direction-adjusted): `best` is the value `p = 1` earns in either
            // direction, so the printed pair reads the same for ↑ and ↓.
        let (best, worst) = match self.fitness.direction() {
            crate::utils::fitness::Direction::Maximize => (frame.best, frame.worst),
            crate::utils::fitness::Direction::Minimize => (-frame.best, -frame.worst),
        };
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            &format!(
                "Training lottery (experimental, Ω {}) — step {clock}",
                fmt_cfg(self.train_merit_omega())
            ),
            &crate::utils::cfy::table(                    &[
                    "net",
                    "exempt",
                    "score",
                    "tickets",
                    "score/ticket",
                    "p(ticket)",
                    "ticket?",
                    "idle",
                ],
                &rows.into_iter().map(|(_, _, row)| row).collect::<Vec<_>>(),
            ),
            Some(&format!(
                "score = the net's smoothed risk-adjusted score (K-reading mean of window avg ∓ λ × window spread, the ranking's number) · tickets = clocks that got a TRAIN-EVAL TICKET — it ran the trainer and was scored (real updates + crown passes); score is the numerator and tickets the divisor · score/ticket = score ÷ tickets (2dp, so a redo from the two legs can miss in the last digit): the window level per ticket spent, so idling leaves it untouched · field THIS clock drew from: best {} worst {} · p(ticket) = where the net's score/ticket sits between those two bounds (1 = best, 0 = worst), then blended by Ω: `1 − Ω·(1 − p)`, so Ω < 1 lifts every net toward 1 = always train · ticket? = yes if THIS clock sells it the step (train + eval), no if it IDLES instead (no train, no eval, nothing measured) · exempt nets (crown, born shield) always run and grade 1.00 — but a crown reads no, because the elite freeze outranks the lottery · idle = clocks the lottery left it idle BEFORE this clock ({} across the run); the step's lottery line names this clock's count",
                fmt_cfg(best),
                fmt_cfg(worst),
                idled,
            )),
        ));
    }
}

impl CoreEngine {
        /// The clock's ONE-LINE lottery report, printed after the group loop: what the
        /// coin did in aggregate, never who it happened to (the per-net list is DEBUG).
        ///
        /// Two tenses on one line, deliberately: `graded`, `p̄` and `field` are this
        /// clock's INPUTS (the pre-clock frame the draws used, so `p̄` can be checked
        /// against `won`), while `won`, `exempt`, `upd̄`, `tkt̄` and `idled` are the
        /// OUTCOME read after the nets stepped. `p̄` is the mean of the BLENDED
        /// probability the coin actually compared against — `p_eff = 1 − Ω·(1 − p)`,
        /// not the raw fraction — so `won ≈ graded · p̄` holds at every Ω.
        ///
        /// Two identities close on this line, and printing `exempt` is what makes the
        /// second one visible:
        ///
        /// - `graded = won + idled` — every net the field graded either got the ticket
        ///   or idled.
        /// - `ticketed = won + exempt` — the rollup's ticket count also covers the nets
        ///   that always run without being graded (the born shield, and a net with no
        ///   reading yet). Without `exempt` a reader compares `won` against the rollup's
        ///   `tickets` and sees two numbers that cannot both be right.
        ///
        /// `idled`/`ticketed` come from the idle-ledger delta and the group split.
    pub(crate) fn log_lottery_summary(
        &self,
        clock: usize,
        frame: super::step::LotteryFrame,
        ticketed: usize,
        idled: usize,
    ) {
        // The field the coin graded: live, non-crown, out of the born shield, with a
        // reading — exactly `lottery_frame`'s filter, re-asked through the same helpers.
        let mut graded = 0usize;
        let mut p_sum = 0.0f32;
        for h in self.state.live_hashes() {
            if self.frozen_crown.contains(&h) || self.on_probation(&h, clock) {
                continue;
            }
            if self.merit_per_ticket(&h).is_some() {
                graded += 1;
                    // The BLENDED probability, not the raw fraction: this line's
                    // `won` is a draw against `p_eff`, so printing the raw p would
                    // make `won ≈ graded · p̄` look violated whenever Ω < 1.
                p_sum += self.train_lottery_p_eff(&h, clock, Some(frame));
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
        let won = graded.saturating_sub(idled);
            // Runners the lottery did not grade: the born shield and the no-reading yet
            // nets always take the ticket, so the rollup's `tickets` is this much larger
            // than `won`.
        let exempt = ticketed.saturating_sub(won);
        let p_bar = (graded > 0).then(|| p_sum / graded as f32);
        // The band rule, not 2dp: `expected` is computed from the same value, so a
        // reader multiplying the PRINTED chance by `graded` must land on the
        // printed expectation (2dp could miss by a tenth).
        let p_txt = p_bar.map_or_else(|| "—".to_string(), fmt_line);
        // Expected vs drawn, side by side: `p̄ × graded` is what the coin was worth,
        // `won` is what it actually produced. The gap is the coin, not a bug.
        let expected = p_bar.map_or_else(|| "—".to_string(), |p| fmt_line(p * graded as f32));
        let omega = self.config.train_merit;
        let body = format!(
            "graded {} = won {} + idled {} │ ticketed {} = won {} + exempt {} │ avg ticket chance {} over the {} graded → expected {}, drawn {} │ field: best {} worst {} │ avg updates {} and avg tickets {} over the {} measured nets │ exempt = crown or born shield (always run)",
            graded,
            won,
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
            "step {} │ lottery (Ω {}, so p(ticket) = 1 − {} × (1 − p)) │ {}",
            clock,
            fmt_cfg(omega),
            fmt_cfg(omega),
            body,
        );
    }
}

/// One net's origin cell for the final elite table: WHO made it, without parent-hash
/// ids, with the birth clock when non-zero. Stored lineage is unchanged.
fn origin_cell(created_from: Option<&str>, born: usize) -> String {
    let Some(raw) = created_from else {
        return format!("unknown@{born}");
    };
    let short8 = |hash: &str| hash[..8.min(hash.len())].to_string();
        // `crossover-one-point:parents=h1,h2[+mut]` → `cx one-point@<born>[+mut]` — the
        // crossover TYPE is useful; the parent ids are not (widest column by far).
    if let Some(rest) = raw.strip_prefix("crossover-") {
        let (head, mutated) = rest
            .strip_suffix("+mut")
            .map(|p| (p, true))
            .unwrap_or((rest, false));
        let kind = head.split(':').next().unwrap_or("one-point");
        return format!("cx {kind}@{born}{}", if mutated { "+mut" } else { "" });
    }
    // `clone:parents=h1` — asexual single-parent child: type + step, no id.
    if raw == "clone" || raw.starts_with("clone:") {
        return format!("clone@{born}");
    }
    if let Some(rest) = raw.strip_prefix("seed:") {
        // `seed:<run_id>:<champion_hash>` → `seed:<run_id>:<hash8>` — the
        // run id is the provenance that matters for inheritance, keep it.
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
        ("  immigrants".to_string(), a.immigrants.to_string()),
        (
            "  crossover children".to_string(),
            a.crossover_children.to_string(),
        ),
        ("culled".to_string(), a.culls.to_string()),
        (
            "  culled (crossover slot)".to_string(),
            a.culls_crossover.to_string(),
        ),
        (
            "  culled (immigrant slot)".to_string(),
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

/// Flatten a trainer's `describe()` blob into `field | value` ROWS for a CFY table
/// (`nested.key` for nested objects; arrays join into one cell). Bounded at
/// `MAX_FIELDS` with an explicit `… | +n more` row. Nulls flatten to nothing, so an
/// empty `describe()` yields no rows.
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
            // Display-only: a float shows 4 decimals (an f32 round-trips as f64 as
            // `0.10000000149011612`); integers stay integers. The stored value is
            // never touched — only this table cell is rounded.
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
            // Null carries no information; `{}` (an empty object) flattens to
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

/// `trained n of m clocks` for the seat lines: real optimizer updates over the
/// clocks the net has LIVED (`entered_at_step` → now) — not the old absolute-clock
/// delta. The remainder is named too, so `20 of 45` does not leave the reader to
/// work out where the other 25 went.
pub(crate) fn trained_note(trainings: usize, age: usize) -> String {
    if age > trainings {
        format!(
            "trained {trainings} of {age} clocks ({} no update)",
            age - trainings
        )
    } else {
        format!("trained {trainings} of {age} clocks")
    }
}

#[cfg(test)]
mod tests {
    use super::{elite_badge_entry, origin_cell, trained_note};

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
    fn trained_note_is_trainings_over_clocks_lived() {
        // A net born at step 7 that trained 5 of its 12 lived clocks: the other
        // 7 are named, not left for the reader to subtract.
        assert_eq!(trained_note(5, 12), "trained 5 of 12 clocks (7 no update)");
        // The two counters can only agree when nothing was withheld, and
        // `trainings ≤ age` always holds.
        assert_eq!(trained_note(0, 0), "trained 0 of 0 clocks");
        assert_eq!(trained_note(12, 12), "trained 12 of 12 clocks");
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
        let get = |k: &str| rows.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
        assert_eq!(get("trainer").as_deref(), Some("cartpole"));
        // Floats display with 4 decimals (an f32 round-trips as a noisy f64).
        assert_eq!(get("lr").as_deref(), Some("0.0100"));
        assert_eq!(get("challenge_prob").as_deref(), Some("0.2500"));
        assert_eq!(get("optimizer.weight_decay").as_deref(), Some("0.0000"));
        assert_eq!(get("optimizer.kind").as_deref(), Some("Adam"), "nested keys flatten: {rows:?}");
        assert_eq!(get("activations").as_deref(), Some("[\"relu\",\"tanh\"]"));
        // Nulls carry nothing and empty objects produce no rows at all.
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
        assert_eq!(get("  immigrants").as_deref(), Some("99"));
        assert_eq!(get("  crossover children").as_deref(), Some("65"));
        assert_eq!(get("culled").as_deref(), Some("164"));
        assert_eq!(get("  culled (crossover slot)").as_deref(), Some("65"));
        assert_eq!(get("  culled (immigrant slot)").as_deref(), Some("99"));
        assert_eq!(get("alive at stop").as_deref(), Some("50"));
        assert_eq!(get("evolution-deferred clocks").as_deref(), Some("12"));
        // Rolls matrix: one row per roll kind, one column per outcome. A roll that
        // bought nothing is reported, not hidden.
        let (header, rows) = accounting_rolls(&a);
        assert_eq!(
            header,
            vec!["event", "rolls", "fired", "attempts", "built", "discarded", "blocked"]
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
        // Seeds keep the source run id + the champion hash (short8) — that
        // is the inheritance provenance, not a parent id.
        assert_eq!(
            origin_cell(Some("seed:1791759444988:257486c8abcdef"), 0),
            "seed:1791759444988:257486c8"
        );
        // Crossovers: the OP (not the parents) + the birth clock; the
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
        // Single-parent asexual child: same treatment.
        assert_eq!(
            origin_cell(Some("clone:parents=aaaaaaaaaaaaaaaa"), 5),
            "clone@5"
        );
        // Short hashes are not chopped further.
        assert_eq!(origin_cell(Some("seed:42:abc"), 0), "seed:42:abc");
    }
}
