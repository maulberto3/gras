//! The engine's user-facing output: the per-step rollup (`Summ`), the framed
//! in-place table (`Minimal`), and the stop-time summary.
//!
//! Reporting is a pure consumer of engine state — it never mutates the
//! population, the clock, or the ledger (the one exception is the `Minimal`
//! delta baseline, `minimal_prev_means`). Everything else in the engine stays
//! free to read these numbers; nothing here feeds back into the search.

use tracing::info;

use super::core::CoreEngine;
use crate::engine::child::crown_act_count;
use crate::engine::diversity::{NetShape, mean_delta_to, shape_of};
use crate::engine::format::{
    ReadingStyle, fmt_cell, fmt_cell_opt, fmt_census, fmt_cfg, fmt_line, fmt_units,
};
use crate::engine::smoothing::{WindowStats, rolling_mean};

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
        // A window with n = 1 has no observable spread — `WindowStats::std` is 0 by
        // construction there, and printing that 0 as `window spread 0.0000` would claim a
        // steadiness nobody measured. Both sinks print `—` instead, the same rule the stop
        // cards' `Ws` column follows.
        let spread_readable = readings.iter().any(|r| r.n > 1);
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
        let window_spread = if spread_readable { avg(&spreads) } else { f32::NAN };
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
        // It is NOT the window spread — that is the `λ × window spread` leg of the
        // identity beside it, a charge paid by each net — so a field of identical nets
        // keeps its mean and drives this one to 0. `—` below two scores: one score has
        // no observable population spread.
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

    /// The configured informative metrics as a rollup segment: the POPULATION MEAN of
    /// each, in config order (the order `history.csv` writes its columns in), `—` until
    /// somebody reports one.
    ///
    /// They are the only numbers on the line the engine does not itself define, so they
    /// also ride the per-net DEBUG line (`step.rs`). Before this they were written to
    /// `history.csv` and readable in the stop cards' right-hand columns — i.e. only by
    /// someone who looked for them, and never while the run was live.
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
                    format!("{} avg {} ({} net(s))", self.metrics[i].label(), fmt_line(mean), values.len())
                }
            })
            .collect();
        format!(" │ {}", parts.join(" │ "))
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
            Some("built = nets admitted; discarded = crossover rolls that produced no viable child; blocked = mutation rolls stopped by the cull shield"),
        ));

        // NOT here: the final-field TABLE is an artifact first — it is written at every
        // log level, `None` included — so `finish_race` renders it after this block,
        // gating only the card on `verbose_detail`. One table for the whole measured
        // field: the elite and the worst are MARKS on the same ranking, not two cards.
        Ok(())
    }

    /// The stop-time summary of the WHOLE measured field: one table, every live net,
    /// ranked by FIT, with the two saved sets marked in place.
    ///
    /// Why one table and not "elites" + "stragglers": the net worth looking at is the one
    /// that spent the most clocks and still scores at the bottom — and that net is NOT the
    /// last row of the fitness ranking. Cutting two cards from the two ends of
    /// `ranked_live()` could only show `A`'s extremes; the `D * A` column puts
    /// the reading on every row, and `✖` marks the nets `worst-<hash>.*` actually saved,
    /// so both ends and everything between them are read in one place.
    ///
    /// The marks are taken from the SAME calls the artifact writers make (`champion_ranked`
    /// / `worst_set`), so a mark and a file can never disagree, and a net saved as an elite
    /// is never marked `✖` (a population under `2k` therefore yields fewer than k worst
    /// picks).
    ///
    /// Stop-time only, and it cannot perturb a replay (no forward pass, no environment, no
    /// RNG draw beyond the read-only diversity baseline).
    ///
    /// `verbose` gates the CARD only: the table itself is an ARTIFACT, and
    /// `final-field.csv` is written at every log level (`None` included), because a
    /// scripted run is exactly the caller who wants the table as a file.
    pub(crate) fn log_final_field(&self, clock: usize, verbose: bool) {
        let mut rows: Vec<(String, Option<f32>)> = self
            .ranked_live()
            .into_iter()
            .map(|(hash, fit)| (hash, Some(fit)))
            .collect();
        // EVERY individual, not only the measured ones: a live net with no reading yet
        // still exists, and dropping it would make the table's count disagree with the pop
        // size in the accounting card above. It rides last, unranked (`F = —`), because a
        // net nobody measured has no FIT to rank — its `·` mark says the artifact writers
        // did not choose it either.
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
        let k = self.config.elite_count.max(1); // report at least the champion
        let top: Vec<(String, f32)> = self.ranked_live().into_iter().take(k).collect();
        let worst = self.worst_set();
        let inefficiency = self.inefficiency_field();
        let mut marks: std::collections::HashMap<String, &'static str> =
            std::collections::HashMap::new();
        // ONE glyph answers "what did the run do with this net", and the crown is part of
        // that answer: `★` = saved as an elite AND seated in the acting crown (it acted),
        // `☆` = saved, but barred from the seat — the `j`/`t` printed beside it names the
        // gate. A net holding a seat is always saved, so no row can be seated-and-★-less.
        // With the crown off nobody sits, so every saved elite reads `★`.
        let mut barred = 0usize;
        for (hash, _) in &top {
            let seated = !self.config.freeze_elites || self.frozen_crown.contains(hash);
            if !seated {
                barred += 1;
            }
            marks.insert(hash.clone(), if seated { "★" } else { "☆" });
        }
        for hash in &worst {
            marks.entry(hash.clone()).or_insert("✖");
        }
        let plan = self.stop_plan(&top);
        let arrow = self.fitness.direction().arrow();
        // The lottery field the LAST clock drew from (`dealt_clock` = the clock the nets
        // actually stepped on). `None` when the lottery is off (Ω = 0) or was never armed;
        // the `p(ticket)` column then reads `—` instead of inventing a probability.
        let lottery = self.lottery_frame(plan.dealt_clock);
        let (table, stats, grid) =
            self.stop_card(&rows, &plan, &marks, inefficiency.as_ref(), lottery);
        let sign = match self.fitness.direction() {
            crate::utils::fitness::Direction::Maximize => "−",
            crate::utils::fitness::Direction::Minimize => "+",
        };
        let (eligible, probation, age_blocked) = self.elite_gate_counts(plan.dealt_clock);
        let baseline = self.stop_baseline(clock);
        // The legend is a TABLE that assigns each column its LETTER and states the
        // relations in those letters: `A/B` is checked against the `fit` and `tickets`
        // cells in the same row rather than trusted from prose, which is what a reader
        // comparing `6.58` with `fit 243.1` and `age 41` actually needs. The words live in
        // the header, the arithmetic lives here — one place each.
        // `table_wrapped`, not `kv`: the block is NAMED — `abbrev | description` says which
        // column is the symbol to look up and which is the explanation — and a table is as
        // wide as its widest cell, so the sentences are wrapped. An unwrapped legend
        // stretched the whole card to ~350 columns.
        // ONE idea per row, the EQUATION as the key and plain English as the value: a
        // legend that packs three clauses into a cell is a paragraph with a table drawn
        // around it. Every value stays inside `PROSE_WIDTH`, so the card is readable in
        // any terminal.
        let legend = crate::utils::cfy::table_wrapped(
            &["abbrev", "description"],
            &stop_legend_rows(&LegendCtx {
                sign,
                lambda: self.config.risk_aversion,
                window: self.config.smoothing_window,
                omega: self.train_merit_omega(),
                direction: self.fitness.direction(),
                lottery,
                inefficiency: inefficiency.as_ref(),
                worst_len: worst.len(),
                k,
                crossover_probation: self.config.crossover_probation_steps,
                mutation_probation: self.config.mutation_probation_steps,
                join_probation: self.config.elite_join_probation_steps,
                crown_lifetime: self.config.elite_crown_lifetime_steps,
            }),
        );
        // A barred elite is only named when there is one: `☆ 0` on a healthy run is noise.
        let seat_note = if barred > 0 {
            format!(" · ☆ {barred}")
        } else {
            String::new()
        };
        // The run's seed rides the card: it is the one input a replay cannot recover
        // once the run dir is gone, and the console output is what a reader keeps.
        let run_seed = self.header.run_seed;
        // The FILE first, and unconditionally: `final-field.csv` is run data, written
        // from the very cells the card would print (one `StopPlan`, no second query).
        self.write_final_field_csv(&plan.header, &grid);
        let field_card = crate::utils::cfy::card(
            &format!(
                "Final field ({} nets — {} measured, ranked by fit{arrow} (A) · ★ {}{seat_note} · ✖ {})",
                rows.len(),
                measured,
                top.len() - barred,
                worst.len()
            ),
            &format!("{legend}\n{table}"),
            Some(&format!(
                "Δcrown = mean multiset-Jaccard distance to the {} reported champion(s): 0.00 = structurally identical to one of them, 1.00 = shares no gene with any · Δpop = mean distance to every other live net: 0.00 = a clone of the crowd, 1.00 = unlike everything alive · read the two AGAINST the baseline: Δcrown ≈ Δpop means the elite is structurally ordinary (selection is ranking numbers, not designs), Δcrown below it means the elite is a structural cluster · {} · {baseline} · crown {}/{} seats · {} eligible · {} gated (join window or cull shield, {} clocks) · {} age-blocked ({} clocks) · stopped at clock {clock} · run seed {run_seed} (`--seed {run_seed}` replays this run bit-for-bit) · the same table is written to `final-field.csv` ({} row(s) under this header)",
                plan.crown.len(),
                stats.line(),
                self.frozen_crown.len(),
                self.config
                    .elite_count
                    .min(self.state.live_count().saturating_sub(1)),
                eligible,
                probation,
                self.config.elite_join_probation_steps,
                age_blocked,
                self.config.elite_crown_lifetime_steps,
                grid.len(),
            )),
        );
        if !verbose {
            return;
        }
        crate::utils::cfy::emit(&field_card);
        // And immediately after the table, HOW those nets were born: the same reported
        // set, drawn as ancestry. The table says what a net IS; without this card the
        // reader has to open `nets/*.json` and decode `created_from` by hand to learn
        // which two nets crossed to make it.
        self.log_lineage(&top, &marks);
    }

    /// The stop-time ANCESTRY card: every reported net with its parents below it.
    ///
    /// The edges already exist in the run's own data — a child's `created_from` names its
    /// parent hashes, and a culled parent leaves a full `nets/<hash>.json` tombstone — so
    /// this card adds no bookkeeping; it only draws what is recorded. It is emitted right
    /// after the final-field table because that is the question the table raises: `#1` is
    /// `cx one-point@50`, and the reader wants to know WHICH nets crossed and how they
    /// scored. Stop-time only, read-only (a few tombstone file reads, no RNG, no forward
    /// pass), so it cannot perturb a replay.
    fn log_lineage(
        &self,
        top: &[(String, f32)],
        marks: &std::collections::HashMap<String, &'static str>,
    ) {
        if top.is_empty() {
            return;
        }
        let lines = self.lineage_lines(top, marks);
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            &format!(
                "Lineage — how the {} reported net(s) were born",
                top.len()
            ),
            &lines.join("\n"),
            Some(&format!(
                "read downward, oldest first: each bracketed group sits directly above the net it made (a crossover group is `p1`/`p2`), so the reported net closes its own ancestry at the bottom · `Δfit` = the child's fit minus this parent's, signed so `+` means the child scored better (the ranking's own direction); `Δparams` = child − parent · `†<clock> <channel>` = culled at that clock, `live` = still in the population at stop · `↺` = this ancestor's own ancestry is already drawn above · `…` = ancestry stops at the depth cap of {LINEAGE_MAX_DEPTH} generations (the parent's own `nets/<hash>.json` still carries the rest) · edges are each net's recorded `created_from`, and a dead parent's numbers come from its tombstone, so the tree outlives the cull · a reported net with no recorded parents prints as ONE row (`· root`) — the group above the next `#N` belongs to that net, never to the row above it"
            )),
        ));
    }

    /// The ancestry card's LINES — split from [`Self::log_lineage`] so the drawing is one
    /// function and the emission is another.
    fn lineage_lines(
        &self,
        top: &[(String, f32)],
        marks: &std::collections::HashMap<String, &'static str>,
    ) -> Vec<String> {
        let mut tree = LineageTree::new(self);
        let mut out: Vec<String> = Vec::new();
        for (rank, (hash, fit)) in top.iter().enumerate() {
            // The root line mirrors the table's identity columns (rank, hash, origin) and
            // repeats its mark, so a tree is matched to its row by eye. `fit` falls back to
            // the table's own value when the net carries no stored reading.
            tree.root_fit = Some(*fit);
            tree.root_mark = marks.get(hash).copied().unwrap_or("·");
            tree.render_node(hash, &format!("#{}", rank + 1), None, 0, "", &mut out);
            // A VISIBLE divider between reported nets, not a blank line: the card body is a
            // single reflowed block, so an empty line is swallowed and the next net's parent
            // group then reads as the previous net's ancestry (measured: a `random` champion
            // looked like it had four generations of parents). The divider cannot be eaten.
            if rank + 1 < top.len() {
                out.push("   ⋯".to_string());
            }
        }
        if tree.truncated {
            out.push(format!(
                "… ancestry stops at the {LINEAGE_MAX_DEPTH}-generation / {LINEAGE_NODE_BUDGET}-line cap"
            ));
        }
        out
    }

    /// The `## Ancestry` section of ONE elite's markdown file (`elite-<hash>.md`): the
    /// same recorded lineage the stop card draws, as a mermaid `graph TB` a doc viewer
    /// renders. The card is the console's answer; this is the per-net artifact's, so the
    /// elite file explains its own birth long after the run's log is gone.
    ///
    /// A founder (random draw / inherited `seed:` blueprint) has no recorded parents and
    /// comes out as the single box the reader expects; a crossover converges parent →
    /// child, with the parents at the top and this net at the bottom.
    pub(crate) fn lineage_markdown(&self, root: &str) -> String {
        let mut tree = LineageTree::new(self);
        let has_parents = tree
            .facts(root)
            .map(|f| !f.parents.is_empty())
            .unwrap_or(false);
        let (diagram, cut) = tree.mermaid(root);
        let mut prose = if has_parents {
            "Read downward: an arrow goes from a parent to the child it made (`p1`/`p2` are the crossover slots), so this net sits at the bottom. `Δfit` on an arrow is the child's fit minus that parent's, signed so `+` means the child scored better. Edges are each net's recorded `created_from`; a culled parent's numbers come from its `nets/<hash>.json` tombstone."
                .to_string()
        } else {
            "No recorded parents: this net entered as a founder — a random draw, an inherited `seed:` blueprint, or the initial batch — so its ancestry is the single box below."
                .to_string()
        };
        if cut {
            prose.push_str(&format!(
                " The walk stops at {LINEAGE_MD_MAX_DEPTH} generations / {LINEAGE_MD_NODE_BUDGET} nets; each ancestor's own `nets/<hash>.json` still carries the next step back."
            ));
        }
        format!("## Ancestry\n\n{prose}\n\n{diagram}")
    }

    /// The plan the ONE stop-time table renders from: one header, one crown, one live
    /// population, one clock.
    fn stop_plan(&self, top: &[(String, f32)]) -> StopPlan {
        // The crown `Δcrown` measures to, in elite rank order; `crown_idx` maps a hash to
        // its index in it, so a champion's own row can skip itself. A blueprint that
        // will not parse simply does not enter (`None` → the cells read `—`).
        let mut crown: Vec<NetShape> = Vec::new();
        let mut crown_idx: std::collections::HashMap<String, Option<usize>> =
            std::collections::HashMap::new();
        for (hash, _) in top {
            match self.stop_shape(hash) {
                Some(shape) => {
                    crown_idx.insert(hash.clone(), Some(crown.len()));
                    crown.push(shape);
                }
                None => {
                    crown_idx.insert(hash.clone(), None);
                }
            }
        }
        // The `Δpop` reference set: every live net's shape, one parse each. Same work the
        // diversity line already does on its own cadence, and stop-time only.
        let mut population: Vec<(String, NetShape)> = Vec::new();
        for hash in self.state.live_hashes() {
            if let Some(shape) = self.stop_shape(&hash) {
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
        StopPlan {
            header: self.stop_header(has_eval),
            has_eval,
            dealt_clock: self.step_clock().saturating_sub(1),
            crown,
            crown_idx,
            population,
        }
    }

    /// The table's header, in NARRATIVE order: what the net IS (`mark`/`#`/`hash`/`origin`,
    /// age, census) → how it is RANKED (fit and the window behind it) → what its
    /// LEARNING cost (`tickets`, `A/B`, `updates`, `idle`) → what that effort BOUGHT
    /// (`prop (D)`, `rescaled (A)`, `D * A`) → what the ELITE rules did with it (`only_act`,
    /// the two Δ columns), then the conditional `eval loss` / informative-metric
    /// columns.
    ///
    /// Each measured quantity carries its WORD and its capital LETTER (`fit (A)`,
    /// `tickets (B)`, `updates (D)`), a DERIVED column is named by its own equation
    /// (`A/B`, `D * A`), and the card's legend states the relations in those
    /// same letters (`D * A = prop (D) × rescaled (A)`). The word is what a reader says, the letter is what
    /// the arithmetic is written in, and a row is checkable by hand — `0.06` is `A ÷ B`
    /// (fit per ticket), not `A ÷ C` (per clock lived).
    fn stop_header(&self, has_eval: bool) -> Vec<String> {
        let mut header: Vec<String> = STOP_HEADER_FIXED.iter().map(|s| s.to_string()).collect();
        if has_eval {
            header.push("eval loss".to_string());
        }
        for m in &self.metrics {
            header.push(m.label().to_string());
        }
        header
    }

    /// Render the stop-time table from the shared plan: one row per ranked net, and the
    /// field's own Δ means.
    ///
    /// `marks` is the saved-set marking (`★`/`☆`/`✖`/`·`). `lottery` is the field the
    /// LAST clock drew from: with it, each row's `p(ticket)` is the arithmetic the coin
    /// actually used, not a re-derivation against a field that no longer exists.
    fn stop_card(
        &self,
        rows: &[(String, Option<f32>)],
        plan: &StopPlan,
        marks: &std::collections::HashMap<String, &'static str>,
        inefficiency: Option<&super::artifacts::InefficiencyField>,
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
        for (hash, fit) in rows {
            if fit.is_some() {
                rank += 1;
            }
            let self_idx = plan.crown_idx.get(hash).copied().flatten();
            // One parse per reported net; `None` on an unreadable blueprint, which the
            // cells render `—` rather than pretending the net is structureless.
            let shape = self.stop_shape(hash);
            crowns.push(delta_crown(shape.as_ref(), plan, self_idx));
            pops.push(delta_pop(shape.as_ref(), hash, plan));
            let cells = self.stop_row(StopRow {
                rank: fit.is_some().then_some(rank),
                mark: marks.get(hash).copied().unwrap_or("·"),
                hash,
                fit: *fit,
                shape,
                plan,
                d_prop: inefficiency.and_then(|f| f.updates_prop(hash)),
                a_rescaled: inefficiency.and_then(|f| f.fit_rescaled(hash)),
                inefficiency: inefficiency.and_then(|f| f.inefficiency(hash)),
                // The coin's own number: the BLENDED probability, asked exactly as the
                // step asks it (same clock, same frame), so the column cannot disagree
                // with `idle`/`tickets` beside it.
                ticket_p: lottery.map(|f| {
                    self.train_lottery_p_eff(hash, plan.dealt_clock, Some(f))
                }),
            });
            table.add_row(cells.clone());
            grid.push(cells);
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
    fn stop_row(&self, row: StopRow<'_>) -> Vec<String> {
        let state = self.state.net(row.hash);
        let reading = self.race_reading_of(row.hash);
        let metrics = state.and_then(|s| s.last_metrics.as_ref());
        let born = state.map(|s| s.entered_at_step).unwrap_or(0);
        // Clocks lived INCLUSIVE of the birth clock and the last run — the convention the
        // freeze lines use (`clock − entered_at_step`).
        let age = row.plan.dealt_clock.saturating_sub(born) + 1;
        let shape = row.shape.as_ref();
        // The saved-set mark plus the net's GATE LETTERS, in one cell: the mark says what
        // the run DID with the net, the letters say what the net is still subject to.
        //   `b` = inside its born CULL SHIELD (the per-channel window that keeps a newborn
        //         from being culled — crossover uses `crossover_probation_steps`, every
        //         other origin `mutation_probation_steps`)
        //   `j` = its seat WINDOW is not served yet (`elite_join_probation_steps` clocks;
        //         any seat exit resets it) — SUPPRESSED while `b` applies, because both
        //         windows run off the birth clock: `bj` was one bar spelled twice and read
        //         as "born AND already an elite"
        //   `t` = past the crown TERM (`elite_crown_lifetime_steps`), i.e. age-blocked
        // `b` and `t` can co-occur (`bt`); the footnote's gate tally is the COARSER
        // partition (born-shield-or-join-window counted as one "gated").
        let mut gate = String::new();
        let shielded = self.on_probation(row.hash, row.plan.dealt_clock);
        if shielded {
            gate.push('b');
        }
        if self.config.freeze_elites {
            let join_k = self.config.elite_join_probation_steps;
            let term_j = self.config.elite_crown_lifetime_steps;
            if let Some(s) = state {
                // `j` is suppressed while the born shield is on: the two windows run on
                // the same clock, so `bj` was two letters for one bar — and it read as
                // "born AND already an elite". `j` alone then means the seat window is
                // still unserved (the usual case: a seat exit restarted it).
                if s.elite_probation_age < join_k {
                    if !shielded {
                        gate.push('j');
                    }
                } else if term_j > 0 && s.elite_seat_age >= term_j {
                    gate.push('t');
                }
            }
        }
        let mut cells = vec![
            row.rank.map(|r| r.to_string()).unwrap_or_else(dash),
            row.hash[..8.min(row.hash.len())].to_string(),
            // ── what this net IS ───────────────────────────────────────
            // `origin` beside the hash: where it came from, before the numbers it would
            // otherwise be blamed for.
            origin_cell(state.and_then(|s| s.created_from.as_deref()), born),
            // `A` (clocks lived) and `S` (optimizer steps run) are separate columns: `B`
            // spends `S`, and `S ≠ A` whenever the crown or the lottery withheld a clock.
            age.to_string(),
            // `40.2K`, not `40242`: a parameter count is read at a glance or not at all
            // (the exact value stays in the net's JSON and `engine.json`).
            state
                .and_then(|s| s.meta.params)
                .map(fmt_units)
                .unwrap_or_else(|| "?".to_string()),
            shape
                .map(|s| s.nodes.to_string())
                .unwrap_or_else(dash),
            shape
                .map(|s| s.wires.to_string())
                .unwrap_or_else(dash),
            // Mean hidden width is a MEASURED number like every other cell, so it takes
            // the band rule (`fmt_cell`) instead of a hand-picked `{:.1}` — a 3.5-wide
            // net used to print `3.5` where 3.50 was the honest spelling.
            shape.map(|s| fmt_cell(s.dim)).unwrap_or_else(dash),
            // ── how it is RANKED ───────────────────────────────────────
            row.fit.map(fmt_cell).unwrap_or_else(dash),
            // The window's THREE legs, one column each: `Wm ∓ λ·Ws` is redoable from the
            // row. `Ws = —` at n = 1 — one reading has no observable spread, and a
            // printed 0.0000 would claim a steadiness nobody measured.
            reading.map(|r| fmt_cell(r.mean)).unwrap_or_else(dash),
            reading
                .and_then(|r| (r.n > 1).then_some(r.std))
                .map(fmt_cell)
                .unwrap_or_else(dash),
            // ── what its learning COST ─────────────────────────────────
            state
                .map(|s| s.measured_clocks().to_string())
                .unwrap_or_else(dash),
            // `A/B` takes the CELL precision, not the config one: the lottery's field is
            // often narrow (0.044…0.064 on a loss-like metric), and at 2dp two nets that
            // earn `p = 1.00` and `p = 0.50` would both print `0.06` beside each other.
            self.merit_per_ticket(row.hash)
                .map(fmt_cell)
                .unwrap_or_else(dash),
            row.ticket_p.map(fmt_cfg).unwrap_or_else(dash),
            state.map(|s| s.step.to_string()).unwrap_or_else(dash),
            // Clocks the lottery left IDLE (no train, no eval). `only_act` stays the
            // crown's own count, so the two causes never hide inside one number.
            state
                .map(|s| s.idle_clocks().to_string())
                .unwrap_or_else(dash),
            // ── what the ELITE rules did with it ───────────────────────
            // Act-only clocks: the crown trained and was scored, its update was discarded.
            // The SEAT itself is not a column — it rides the `mark` cell as `★` vs `☆`,
            // so one glyph says what the run did with the net instead of two symbols
            // meaning almost the same thing.
            state
                .map(|s| crown_act_count(s).to_string())
                .unwrap_or_else(dash),
            // ── what that effort BOUGHT ────────────────────────────────
            // `prop (D)`, `rescaled (A)` and `D * A` side by side, so the product is read
            // straight across the row: no division, no legend lookup in the middle.
            row.d_prop.map(fmt_cell).unwrap_or_else(dash),
            row.a_rescaled.map(fmt_cell).unwrap_or_else(dash),
            row.inefficiency.map(fmt_cell).unwrap_or_else(dash),
            // The VERDICT, last: the mark says what the run did with the net, and the
            // gate letters say why it is not sitting.
            format!("{}{gate}", row.mark),
        ];
        if row.plan.has_eval {
            cells.push(fmt_cell_opt(&metrics.and_then(|m| m.eval_loss)));
        }
        // One cell per configured informative metric, in config order — the same order
        // `history.csv` writes its columns in.
        for i in 0..self.metrics.len() {
            cells.push(
                metrics
                    .and_then(|m| m.informative.get(i).copied())
                    .map(fmt_cell)
                    .unwrap_or_else(dash),
            );
        }
        cells
    }

    /// One blueprint parse → the census legs plus the gene multiset the Δ columns
    /// compare. `None` when the blueprint will not parse: the cells read `—` rather than
    /// pretending the net is structureless.
    fn stop_shape(&self, hash: &str) -> Option<NetShape> {
        self.state
            .net(hash)
            .and_then(|s| s.topology().ok())
            .map(|t| shape_of(&t))
    }

    /// The population's own diversity reading, for the stop-time footnotes: without it a
    /// card's `Δpop` has only the other card to be read against. `—` when the population
    /// is too small to compare (the diversity module returns `None` below two nets).
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
                "score = the net's smoothed risk-adjusted score (K-reading mean of window avg ∓ λ × window spread, the ranking's number) · tickets = clocks that got a TRAIN-EVAL TICKET — it ran the trainer and was scored (real updates + crown passes); score is the numerator and tickets the divisor · score/ticket = score ÷ tickets (2dp, so a redo from the two legs can miss in the last digit): the window level per ticket spent, so idling leaves it untouched · field THIS clock drew from: best {} worst {} · p(ticket) = where the net's score/ticket sits between those two bounds (1 = best, 0 = worst), then blended by Ω: `1 − Ω·(1 − p)`, so Ω < 1 lifts every net toward 1 = always train · ticket? = yes if THIS clock sells it the step (train + eval), no if it IDLES instead (no train, no eval, nothing measured) · exempt nets (crown, born shield) always run and grade 1.00 — but a crown reads no, because the crown's act pass outranks the lottery · idle = clocks the lottery left it idle BEFORE this clock ({} across the run); the step's lottery line names this clock's count",
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
    /// The anti-collapse reading: how structurally different the live nets are RIGHT NOW.
    ///
    /// Four readings in one line: `distinct` blueprints, `mean pairwise Δ` over the
    /// population's upper triangle, and the CENSUS — `nodes`, `wires` and `dim` as
    /// mean ± population spread. The mean says what the population looks like, the
    /// spread says whether it still varies; a collapse keeps the mean and drives the
    /// spread to zero.
    ///
    /// Rides the DURABILITY cadence — it prints on the same clocks the checkpoint tick
    /// fires (`checkpoint_every`; `0` = the tick is off, so the line is off), at any
    /// level but `None`, and ABOVE the step rollup so the rollup stays the clock's last
    /// line. Reads the blueprints only — no forward pass, no environment, no RNG draw.
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

/// The fixed left-hand columns of the stop-time table, in NARRATIVE order: what the net
/// IS (`#`, `hash`, `origin`, age, census) → how it is RANKED (`fit (A)` and the two
/// window legs) → what its learning COST (`tickets (B)`, `A/B`, `p(ticket)`, `updates
/// (D)`, `idle (I)`, `act (O)`) → what that effort BOUGHT (`prop (D)`, `rescaled (A)`,
/// `D * A`) → the verdict (`mark`). One list: the header prints it and the legend must
/// explain every entry (`every_column_has_a_legend_row`).
const STOP_HEADER_FIXED: &[&str] = &[
    "#",
    "hash",
    "origin",
    "age (C)",
    "params",
    "nodes",
    "wires",
    "width",
    // The word AND its letter, so a legend equation can be read straight off the row;
    // the DERIVED columns are named by their own operation (`A/B`, `D * A`).
    "fit (A)",
    // The window is TWO columns, not one `μ ±σ (SE, n)` cell: `A = mean ∓ λ × spread` can
    // only be redone when its legs are readable on their own. `n` gets no column — it is
    // `min(B, K)`, read off `tickets (B)`.
    "mean",
    "spread",
    "tickets (B)",
    // The quotient beside its own denominator, so `fit ÷ tickets` is checked against the
    // row instead of guessed (it is NOT `fit ÷ age`).
    "A/B",
    // The lottery's OUTPUT for the last clock: what that `A/B` earned the net in draw
    // probability, beside its two inputs, so it does not look like another score.
    "p(ticket)",
    "updates (D)",
    // The two counters that close the age identity get letters too, so `B = D + O` and
    // `C = D + O + I` read in the same alphabet as the columns that hold them.
    "idle (I)",
    // Act-only clocks: the crown trained and was scored, its update discarded. `act`, not
    // `frozen`: the crown seat DOES train and evaluate — calling that net frozen claimed
    // it did nothing.
    "act (O)",
    // The two legs of `D * A` beside each other, then the product: no division and no
    // legend lookup in the middle of the row.
    "prop (D)",
    "rescaled (A)",
    "D * A",
    // The Δ columns are NOT here: a raw multiset-Jaccard distance sits at 0.7–0.9 between
    // any two unrelated nets, so a per-row value had no reference scale. What the field CAN
    // support is stated ONCE in the footnote (the card mean against the population's own
    // baseline), not repeated per row. LAST: everything before it is evidence, this is the
    // verdict.
    "mark",
];

/// Everything the stop-time legend reads, bundled so its rows can be built (and checked
/// against the header) without a live engine.
struct LegendCtx<'a> {
    sign: &'a str,
    lambda: f32,
    window: usize,
    omega: f32,
    direction: crate::utils::fitness::Direction,
    lottery: Option<super::step::LotteryFrame>,
    inefficiency: Option<&'a super::artifacts::InefficiencyField>,
    worst_len: usize,
    k: usize,
    crossover_probation: usize,
    mutation_probation: usize,
    join_probation: usize,
    crown_lifetime: usize,
}

/// The stop-time legend: `abbrev | description`, one row per column, keyed by the
/// column's OWN spelling — plus the identity rows that state how the columns add up
/// (`B = D + O`, `C = D + O + I`, the field bounds). Every header cell must appear here
/// as a key; `every_column_has_a_legend_row` enforces it.
fn stop_legend_rows(ctx: &LegendCtx<'_>) -> Vec<(String, String)> {
    let sign = ctx.sign;
    vec![
        (
            "#".to_string(),
            "rank by `fit (A)`, best first; `—` for a live net with no reading yet (it is listed last, unranked)"
                .to_string(),
        ),
        (
            "hash".to_string(),
            "the first 8 hex of the blueprint's xxh3 — the net's identity in `nets/<hash>.json`, `elites.json` and this table"
                .to_string(),
        ),
        // ── what this net IS ────────────────────────────────────────────
        (
            "origin".to_string(),
            "where the net came from: a founder slot, or the crossover/mutation op and the clock it was born at (a newborn's first ticket is the NEXT clock)"
                .to_string(),
        ),
        (
            "age (C)".to_string(),
            "`C = clock − born + 1` — clocks lived, inclusive of the birth clock".to_string(),
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
            "fit (A)".to_string(),
            format!(
                "`A = mean ∓ λ × spread` — the ranking value: the window mean, charged for its own spread. This run charges `{sign} λ × spread` with λ = {}",
                fmt_cfg(ctx.lambda)
            ),
        ),
        (
            "mean".to_string(),
            format!(
                "the window mean of the last `n = min(B, K)` results, K = {} — `n` gets no column of its own: it only falls short of K for a net born within K clocks of the stop, so it is read off `tickets (B)`",
                ctx.window
            ),
        ),
        (
            "spread".to_string(),
            "the spread across those `n` results (`—` at `n = 1`: one result shows no spread)"
                .to_string(),
        ),
        // ── what its learning COST ──────────────────────────────────────
        (
            "tickets (B)".to_string(),
            "`B = D + O` — clocks the net ran the trainer on AND was scored for; an idle clock is not one"
                .to_string(),
        ),
        (
            "A/B".to_string(),
            "fit per ticket — the LOTTERY's merit: what one clock of train+eval bought from this net. It divides by `B` (tickets), NOT by `D` (updates): a crown act pass spent a ticket and produced a fresh reading, so charging it as if the net had not run would hand the crown a free ratio. `B = D + O`, which is why the two columns differ on a net that held a seat"
                .to_string(),
        ),
        (
            "p(ticket)".to_string(),
            "the chance this net had of getting the LAST clock's train+eval ticket: its `A/B` as a position in the split field below, then blended by Ω — `1 − Ω × (1 − p)`; 1.00 = always trains (the worst net keeps a standing `1 − Ω`), 0.00 = never, `—` = the lottery is off (Ω = 0) or the field was never armed"
                .to_string(),
        ),
        (
            "A/B field · Ω".to_string(),
            match ctx.lottery.as_ref() {
                Some(f) => {
                    // The frame is stored direction-adjusted (`best` = the value `p = 1`
                    // earns, in either direction), so the printed pair is converted back
                    // into the COLUMN's own units.
                    let (best, worst) = match ctx.direction {
                        crate::utils::fitness::Direction::Maximize => (f.best, f.worst),
                        crate::utils::fitness::Direction::Minimize => (-f.best, -f.worst),
                    };
                    format!(
                        "the lottery field the last clock drew from: best `A/B = {}` and worst `A/B = {}` over the nets it grades (crown and born-shield nets are exempt, so they never set the scale) · Ω = {} = how much of that spread it withholds; the column prints `A/B` at cell precision so the two ends these bounds name are findable in it",
                        fmt_cell(best),
                        fmt_cell(worst),
                        fmt_cfg(ctx.omega),
                    )
                }
                None => format!(
                    "no lottery field at the last clock: Ω = {} (0 = off, every net trains) or the field was not armed yet — so `p(ticket)` reads `—`",
                    fmt_cfg(ctx.omega)
                ),
            },
        ),
        (
            "updates (D)".to_string(),
            "`D` — clocks it actually applied an optimizer step on".to_string(),
        ),
        (
            "idle (I)".to_string(),
            "`I` — clocks the lottery gave no ticket: nothing ran, so it is not a ticket and not an update"
                .to_string(),
        ),
        // ── what the ELITE rules did with it ──────────────────────────
        (
            "act (O)".to_string(),
            "`O` — clocks that held the crown's act and nothing else: it trained, evaluated and was scored, but its update was discarded — the weights held still, the net did not"
                .to_string(),
        ),
        // ── what that effort BOUGHT ───────────────────────────────────
        (
            "prop (D)".to_string(),
            format!(
                "`prop (D) = D ÷ Dmax` — updates proportional: this net's `D` over the field's busiest net's (`Dmax = {}` this run), i.e. how much of the field's training it took. 1.00 = the most trained net alive, near 0 = barely trained. Not a time (`C` is age) and not a share of the mean: 1.00 is one net, the busiest",
                ctx.inefficiency.map(|f| f.max_trained).unwrap_or(0),
            ),
        ),
        (
            "rescaled (A)".to_string(),
            "`rescaled (A) = |A_best − A| ÷ |A_best − A_worst|` — how far below the field's best this net scored, as a fraction of the field's span: 0 = the best, 1 = the worst".to_string(),
        ),
        (
            "Dmax · A_best · A_worst".to_string(),
            format!(
                "this run's field: `Dmax = {}` (the most updates any live net took) · `A_best = {}` and `A_worst = {}` (the two ends of the span `rescaled (A)` measures between)",
                ctx.inefficiency.map(|f| f.max_trained).unwrap_or(0),
                ctx.inefficiency.map(|f| fmt_cell(f.best)).unwrap_or_else(dash),
                ctx.inefficiency.map(|f| fmt_cell(f.worst)).unwrap_or_else(dash),
            ),
        ),
        (
            "D * A".to_string(),
            format!(
                "`D * A = prop (D) × rescaled (A)` — inefficiency, the product of the two legs beside it, EACH ALREADY NORMALISED to 0..1 (raw `D × A` is never meant). 1.00 = the field's most training on the field's worst score, near 0 = little effort or a score near the best. REPORT ONLY: nothing in the loop reads it — it picks the `✖` worst artifacts (top-{} of this column, elites excluded) and nothing else. Ticket allocation uses `A/B`, not this",
                ctx.worst_len
            ),
        ),
        (
            "C = D + O + I (+1 if born mid-run)".to_string(),
            "every clock a net lived is an update, an act-only clock, or an idle clock — PLUS the birth clock a mid-run child sits out (a founder runs clock 0; a child's first ticket is the NEXT clock), which is why the sum can land one short of `C`"
                .to_string(),
        ),
        (
            "B = D + O".to_string(),
            "every ticket is an update or an act-only clock".to_string(),
        ),
        // ── what this net IS (the verdict) ────────────────────────────
        (
            "mark".to_string(),
            format!(
                "`★` saved as elite (top-{} by A) and holding a crown seat · `☆` saved as elite but barred from the seat (the `j` or `t` beside it names the gate) · `✖` saved as worst (top-{} by `D * A`) · `·` neither; with the crown off nobody sits, so every saved elite reads `★`",
                ctx.k,
                ctx.worst_len
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
    ]
}

#[cfg(test)]
mod stop_legend_tests {
    use super::*;

    fn ctx() -> LegendCtx<'static> {
        LegendCtx {
            sign: "−",
            lambda: 0.5,
            window: 10,
            omega: 0.0,
            direction: crate::utils::fitness::Direction::Maximize,
            lottery: None,
            inefficiency: None,
            worst_len: 1,
            k: 1,
            crossover_probation: 5,
            mutation_probation: 5,
            join_probation: 5,
            crown_lifetime: 5,
        }
    }

    /// The legend and the table are ONE contract: every column the header prints has a
    /// row that explains it. Without this, a rename lands in one place and the legend
    /// keeps teaching the old name.
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
        // The conditional columns (`eval loss`, the informative metrics) are the only ones
        // outside the 1:1 rule: they exist only when a trainer reports them.
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

/// The plan BOTH stop-time cards render from: one column list, one crown, one live
/// population, one clock. Built once at stop, so the elite and straggler tables are
/// column-identical by construction instead of by two lists kept in sync by hand.
struct StopPlan {
    /// The shared header — `rank … Δcrown · Δpop · origin`, plus the conditional
    /// `eval loss` and informative-metric columns.
    header: Vec<String>,
    /// Whether the `eval loss` column is in the plan (some reported net carries one).
    has_eval: bool,
    /// The last clock the lottery dealt: the `seat` and `age` columns count to it.
    dealt_clock: usize,
    /// The reported champions' shapes, in elite rank order — what `Δcrown` measures to.
    crown: Vec<NetShape>,
    /// By hash, that champion's index in `crown` (`None`: unreadable blueprint; absent:
    /// the row is not an elite).
    crown_idx: std::collections::HashMap<String, Option<usize>>,
    /// Every live net's shape, by hash — what `Δpop` measures to.
    population: Vec<(String, NetShape)>,
}

/// One row of a stop-time card. BOTH cards come through the same row builder, which —
/// with the shared header — is what keeps them column-identical: same cells, in the same
/// order, with the same `—` rules.
struct StopRow<'a> {
    /// 1-based rank in the race, best FIT first; `None` for an UNMEASURED net, which has
    /// no FIT to be ranked by (it prints `—` there and rides at the bottom of the table).
    rank: Option<usize>,
    /// The saved-set mark: `★` saved elite AND seated, `☆` saved but barred, `✖` worst,
    /// `·` neither.
    mark: &'a str,
    /// The net, by full hash (the `hash` cell prints the first eight).
    hash: &'a str,
    /// The ranking value (FIT) the `fit (A)` column prints; `None` when never measured.
    fit: Option<f32>,
    /// This net's structure, parsed once by the caller (`None`: unreadable blueprint).
    shape: Option<NetShape>,
    /// The plan the whole card prints from.
    plan: &'a StopPlan,
    /// `prop (D) = D ÷ Dmax` — the updates leg of `D * A` alone, printed so the product
    /// below can be read straight off the row.
    d_prop: Option<f32>,
    /// `rescaled (A) = |A_best − A| ÷ |A_best − A_worst|` — the fit leg alone, printed
    /// so the product below can be redone from the row.
    a_rescaled: Option<f32>,
    /// `D * A = prop (D) × rescaled (A)` — the net's inefficiency reading
    /// (`None`: no ranking, so it is not in the field the artifact writer picks from).
    inefficiency: Option<f32>,
    /// `p(ticket)` — the BLENDED probability the LAST clock's coin used, computed by the
    /// same `train_lottery_p_eff` the step called (`None`: the lottery is off at Ω = 0, or
    /// the field was never armed — no probability to print).
    ticket_p: Option<f32>,
}

/// One card's own structural summary: the mean `Δcrown` and `Δpop` over the rows it
/// printed. This is the pair to read ACROSS the two cards — a champion set whose `Δpop`
/// runs above the discarded set's is selection keeping what the population had not
/// converged on — so each card states its own rather than making the reader average a
/// column by eye.
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

/// Mean of the values that EXIST (`None` legs are unreadable blueprints, not zeros), or
/// `None` when no row carried one.
fn mean_of(values: &[Option<f32>]) -> Option<f32> {
    let kept: Vec<f32> = values.iter().flatten().copied().collect();
    if kept.is_empty() {
        return None;
    }
    Some(kept.iter().sum::<f32>() / kept.len() as f32)
}

/// `Δcrown` for one row: the mean multiset-Jaccard distance to the reported champions.
/// A champion (`self_idx = Some(i)`) is measured against the OTHER champions — against
/// itself it would print a fake `0.00` — while a straggler is measured against all of
/// them. `None` (→ `—`) when the row has no readable blueprint, or when there is no
/// other champion to compare against.
fn delta_crown(shape: Option<&NetShape>, plan: &StopPlan, self_idx: Option<usize>) -> Option<f32> {
    let shape = shape?;
    match self_idx {
        Some(i) => mean_delta_to(
            shape,
            plan.crown.iter().take(i).chain(plan.crown.iter().skip(i + 1)),
        ),
        None => mean_delta_to(shape, plan.crown.iter()),
    }
}

/// `Δpop` for one row: the mean distance to every OTHER live net — "does this net look
/// like the crowd?". Read beside `Δcrown`: a champion that is unlike the crowd (high
/// `Δpop`) is doing something the population has not converged on.
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

/// How many generations above a reported net the lineage card walks. A search that
/// crossed for 100 clocks has a deep DAG; 4 generations answers "who made this and how
/// did they score" without turning one card into a screenful, and every ancestor's own
/// `nets/<hash>.json` still carries the next step back.
const LINEAGE_MAX_DEPTH: usize = 4;
/// How many tree lines ONE card may print, across all reported nets.
const LINEAGE_NODE_BUDGET: usize = 64;
/// How many generations the elite files' mermaid diagram walks. A DOC can afford more
/// than a terminal card: mermaid lays the graph out itself, so depth costs nodes, not
/// screen width.
const LINEAGE_MD_MAX_DEPTH: usize = 6;
/// The most nodes one elite diagram may name. Beyond this the walk stops and the
/// markdown says so, rather than shipping a 500-node graph in a per-net file.
const LINEAGE_MD_NODE_BUDGET: usize = 150;

/// The hashes named by a lineage string — `crossover-one-point:parents=<h1>,<h2>[+mut]`
/// or `clone:parents=<h1>` — as the parent list, oldest first. Founders (`random`,
/// `founder-N`) and inherited nets (`seed:<run>:<hash>`) name no in-run parents and come
/// back empty, which is what makes them the roots of a tree.
fn lineage_parents(created_from: Option<&str>) -> Vec<String> {
    let Some(raw) = created_from else {
        return Vec::new();
    };
    // `+mut` is a flag ON the child (crossover, then mutated), not part of a parent id.
    let head = raw.strip_suffix("+mut").unwrap_or(raw);
    let Some((_, parents)) = head.split_once(":parents=") else {
        return Vec::new();
    };
    parents
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect()
}

/// The first 8 hex of a hash — the width every other stop-time identity prints at.
fn short_hash(hash: &str) -> String {
    hash[..8.min(hash.len())].to_string()
}

/// One node's mermaid label: identity, origin, fit, and what happened to it (the root
/// says `ELITE`; an ancestor says `live` or the clock it was culled at).
fn mermaid_node_label(hash: &str, facts: Option<&LineageFacts>, root: bool) -> String {
    let Some(facts) = facts else {
        return short_hash(hash);
    };
    let mut parts = vec![
        short_hash(hash),
        origin_cell(facts.origin.as_deref(), facts.born),
    ];
    if let Some(fit) = facts.fit {
        parts.push(format!("fit {}", fmt_cell(fit)));
    }
    if root {
        parts.push("ELITE".to_string());
    } else if facts.alive {
        parts.push("live".to_string());
    } else if let Some(clock) = facts.culled_at {
        parts.push(format!("†{clock}"));
    }
    parts.join("<br/>")
}

/// One net's ancestry facts, whether it is LIVE or a culled tombstone. One shape for
/// both is what lets the tree cross a cull: the numbers a dead parent needs
/// (`final_risk_score` / `final_fitness_mean` / `final_fitness_std` / `final_fitness_samples`)
/// are the same three legs a live net reads from its window.
#[derive(Clone, Debug)]
struct LineageFacts {
    /// The recorded lineage string (`created_from`), kept verbatim for [`origin_cell`].
    origin: Option<String>,
    /// Clock it entered the population.
    born: usize,
    /// The ranking value (risk-adjusted fit); `None` when it never measured.
    fit: Option<f32>,
    mean: Option<f32>,
    std: Option<f32>,
    n: usize,
    params: Option<i64>,
    alive: bool,
    culled_at: Option<usize>,
    cull_reason: Option<String>,
    /// Parent hashes named by `created_from` (empty = a root).
    parents: Vec<String>,
}

/// The ancestry card's one walker: resolve a hash (live memory first, then the tombstone
/// on disk), draw it, then recurse into its parents.
struct LineageTree<'a> {
    engine: &'a CoreEngine,
    /// Resolved facts per hash — a shared ancestor is read from disk at most once.
    cache: std::collections::HashMap<String, Option<LineageFacts>>,
    /// Hashes already drawn in THIS card: a repeated ancestor is marked `↺` and not
    /// expanded again, so a DAG shared by two reported nets stays one screen.
    shown: std::collections::HashSet<String>,
    budget: usize,
    truncated: bool,
    /// The mark (`★`/`☆`) and table fit of the ROOT currently being drawn.
    root_mark: &'static str,
    root_fit: Option<f32>,
}

impl<'a> LineageTree<'a> {
    fn new(engine: &'a CoreEngine) -> Self {
        Self {
            engine,
            cache: std::collections::HashMap::new(),
            shown: std::collections::HashSet::new(),
            budget: LINEAGE_NODE_BUDGET,
            truncated: false,
            root_mark: "·",
            root_fit: None,
        }
    }

    /// One net's facts: the live state first (a parent may still be alive), then the
    /// `nets/<hash>.json` tombstone a cull leaves behind. `None` when neither exists —
    /// e.g. a parent inherited from another run's directory.
    fn facts(&mut self, hash: &str) -> Option<LineageFacts> {
        if let Some(hit) = self.cache.get(hash) {
            return hit.clone();
        }
        let engine = self.engine;
        let facts = if let Some(state) = engine.state.net(hash) {
            let reading = engine.race_reading_of(hash);
            Some(LineageFacts {
                origin: state.created_from.clone(),
                born: state.entered_at_step,
                fit: engine.smoothed_fitness_of(hash),
                mean: reading.map(|r| r.mean),
                // `n = 1` has no observable spread, the same rule the table's `spread` cell
                // follows — a printed `0` would claim a steadiness nobody measured.
                std: reading.and_then(|r| (r.n > 1).then_some(r.std)),
                n: reading.map(|r| r.n).unwrap_or(0),
                params: state.meta.params,
                alive: true,
                culled_at: None,
                cull_reason: None,
                parents: lineage_parents(state.created_from.as_deref()),
            })
        } else {
            match crate::state::load_net_state(&engine.run_dir, hash) {
                Ok(state) => Some(LineageFacts {
                    origin: state.created_from.clone(),
                    born: state.entered_at_step,
                    // A legacy tombstone may predate `final_risk_score`; the window mean is
                    // then the only level it recorded, so it is read as the fit.
                    fit: state
                        .final_risk_score
                        .filter(|v| v.is_finite())
                        .or_else(|| state.final_fitness_mean.filter(|v| v.is_finite())),
                    mean: state.final_fitness_mean.filter(|v| v.is_finite()),
                    std: state
                        .final_fitness_std
                        .filter(|v| v.is_finite())
                        .filter(|_| state.final_fitness_samples > 1),
                    n: state.final_fitness_samples,
                    params: state.meta.params,
                    alive: state.is_alive,
                    culled_at: state.culled_at_step,
                    cull_reason: state.cull_reason.clone(),
                    parents: lineage_parents(state.created_from.as_deref()),
                }),
                Err(_) => None,
            }
        };
        self.cache.insert(hash.to_string(), facts.clone());
        facts
    }

    /// This net's ancestry as a mermaid `graph TB`: parents at the TOP, the net at the
    /// BOTTOM, one arrow per parent → child edge. Shared ancestors are ONE node (mermaid
    /// keys by id), so a DAG converges instead of being repeated the way the console
    /// tree's `↺` avoids; a founder comes out as a single box.
    ///
    /// Returns the fenced diagram and whether the walk was cut short (depth or node
    /// budget), which the caller states in the section's prose.
    fn mermaid(&mut self, root: &str) -> (String, bool) {
        let mut nodes: Vec<String> = Vec::new();
        let mut edges: Vec<(String, String, usize, usize)> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut stack: Vec<(String, usize)> = vec![(root.to_string(), 0)];
        let mut cut = false;
        while let Some((hash, depth)) = stack.pop() {
            if !seen.insert(hash.clone()) {
                continue;
            }
            if nodes.len() >= LINEAGE_MD_NODE_BUDGET {
                cut = true;
                break;
            }
            nodes.push(hash.clone());
            let Some(facts) = self.facts(&hash) else {
                continue;
            };
            if facts.parents.is_empty() {
                continue;
            }
            if depth >= LINEAGE_MD_MAX_DEPTH {
                cut = true;
                continue;
            }
            let n = facts.parents.len();
            for (i, parent) in facts.parents.iter().enumerate() {
                edges.push((parent.clone(), hash.clone(), i + 1, n));
                stack.push((parent.clone(), depth + 1));
            }
        }
        let known: std::collections::HashSet<String> = nodes.iter().cloned().collect();
        let direction = self.engine.fitness.direction();
        let adjusted = |v: f32| match direction {
            crate::utils::fitness::Direction::Maximize => v,
            crate::utils::fitness::Direction::Minimize => -v,
        };
        let mut out = String::from("```mermaid\ngraph TB\n");
        // Classes first, so a viewer never renders an unstyled node while reading down.
        out.push_str("    classDef elite fill:#ffe9a8,stroke:#a87b00,stroke-width:2px\n");
        out.push_str("    classDef culled fill:#f2f2f2,stroke:#9a9a9a,color:#555\n");
        out.push_str("    classDef ancestor fill:#eaf4ff,stroke:#5a8fbf\n");
        for hash in &nodes {
            let facts = self.facts(hash);
            out.push_str(&format!(
                "    n_{}[\"{}\"]\n",
                short_hash(hash),
                mermaid_node_label(hash, facts.as_ref(), hash == root)
            ));
        }
        for (parent, child, slot, n) in &edges {
            // A parent beyond the node budget was never declared: drop the edge with the
            // cut instead of letting mermaid invent a blank node for it.
            if !known.contains(parent) || !known.contains(child) {
                cut = true;
                continue;
            }
            let parent_facts = self.facts(parent);
            let child_facts = self.facts(child);
            let mut label = if *n == 1 {
                "p".to_string()
            } else {
                format!("p{slot}")
            };
            if let (Some(parent_facts), Some(child_facts)) = (parent_facts, child_facts) {
                if let (Some(parent_fit), Some(child_fit)) = (parent_facts.fit, child_facts.fit) {
                    let gain = adjusted(child_fit) - adjusted(parent_fit);
                    label.push_str(&format!(
                        " · Δfit {}{}",
                        if gain >= 0.0 { "+" } else { "−" },
                        fmt_cell(gain.abs())
                    ));
                }
            }
            out.push_str(&format!(
                "    n_{} -->|\"{}\"| n_{}\n",
                short_hash(parent),
                label,
                short_hash(child)
            ));
        }
        for hash in &nodes {
            let class = if hash == root {
                "elite"
            } else if self.facts(hash).map(|f| !f.alive).unwrap_or(false) {
                "culled"
            } else {
                "ancestor"
            };
            out.push_str(&format!("    class n_{} {class}\n", short_hash(hash)));
        }
        out.push_str("```\n");
        (out, cut)
    }

        /// Draw one node and, depth permitting, its parents ABOVE it. `label` is the
        /// root's `#<rank>` or a parent slot `p1`/`p2`; `child` is the net this node
        /// parented, which is what the `Δ` columns are measured against. Ancestors print
        /// first, so the card reads downward in birth order: parents, then the net they
        /// made, ending at the reported net.
    fn render_node(
        &mut self,
        hash: &str,
        label: &str,
        child: Option<&LineageFacts>,
        depth: usize,
        connector: &str,
        out: &mut Vec<String>,
    ) {
        let indent = "   ".repeat(depth);
        if self.budget == 0 {
            self.truncated = true;
            out.push(format!("{indent}…"));
            return;
        }
        self.budget -= 1;
        let Some(facts) = self.facts(hash) else {
            out.push(format!(
                "{indent}{connector}{label} {} · no `nets/{}.json` — ancestry unknown",
                short_hash(hash),
                short_hash(hash),
            ));
            return;
        };
        let repeated = child.is_some() && self.shown.contains(hash);
        // Parents FIRST: each generation sits one indent deeper, and the bracketed group
        // directly above a net is that net's parent set. That is the natural reading order
        // (oldest ancestor at the top, the reported net at the bottom).
        if !repeated && depth + 1 < LINEAGE_MAX_DEPTH {
            let n = facts.parents.len();
            for (i, parent) in facts.parents.iter().enumerate() {
                let next_label = if n == 1 {
                    "p".to_string()
                } else {
                    format!("p{}", i + 1)
                };
                // A single parent is the group's only line and closes it.
                let next_connector = if n == 1 || i + 1 == n {
                    "└─ "
                } else if i == 0 {
                    "┌─ "
                } else {
                    "├─ "
                };
                self.render_node(
                    parent,
                    &next_label,
                    Some(&facts),
                    depth + 1,
                    next_connector,
                    out,
                );
            }
        }
        self.shown.insert(hash.to_string());
        let mut line = format!(
            "{indent}{connector}{label} {} · {}",
            short_hash(hash),
            origin_cell(facts.origin.as_deref(), facts.born),
        );
        // The ranking value. A root falls back to the table's own cell when the net has no
        // stored reading; an ancestor prints only what its record carries.
        let fit = if child.is_none() {
            facts.fit.or(self.root_fit)
        } else {
            facts.fit
        };
        if let Some(v) = fit {
            line.push_str(&format!(" · fit {}", fmt_cell(v)));
        }
        if let (Some(mean), true) = (facts.mean, facts.n > 0) {
            line.push_str(&format!(" · μ {}", fmt_cell(mean)));
            if let Some(std) = facts.std {
                line.push_str(&format!(" σ {}", fmt_cell(std)));
            }
            line.push_str(&format!(" n {}", facts.n));
        }
        if let Some(params) = facts.params {
            line.push_str(&format!(" · {}", fmt_units(params)));
        }
        if !facts.alive {
            match (facts.culled_at, facts.cull_reason.as_deref()) {
                (Some(clock), Some(reason)) => line.push_str(&format!(" · †{clock} {reason}")),
                (Some(clock), None) => line.push_str(&format!(" · †{clock}")),
                _ => line.push_str(" · dead"),
            }
        } else if child.is_some() {
            line.push_str(" · live");
        }
        // The `Δ` leg, read parent-against-child in the RANKING's own direction, so `+`
        // always means "the child scored better than this parent" — printing a raw `fit`
        // difference would flip meaning on a minimize run.
        if let Some(child) = child {
            let direction = self.engine.fitness.direction();
            let adjusted = |v: f32| match direction {
                crate::utils::fitness::Direction::Maximize => v,
                crate::utils::fitness::Direction::Minimize => -v,
            };
            if let (Some(child_fit), Some(parent_fit)) = (child.fit, facts.fit) {
                let gain = adjusted(child_fit) - adjusted(parent_fit);
                line.push_str(&format!(
                    " · Δfit {}{}",
                    if gain >= 0.0 { "+" } else { "−" },
                    fmt_cell(gain.abs())
                ));
            }
            if let (Some(child_params), Some(parent_params)) = (child.params, facts.params) {
                let delta = child_params - parent_params;
                line.push_str(&format!(
                    " · Δparams {}{}",
                    if delta >= 0 { "+" } else { "−" },
                    fmt_units(delta.abs())
                ));
            }
        }
        if child.is_none() {
            line.push_str(&format!(" · {}", self.root_mark));
            // A reported net with NO group above it is a founder (random draw, inherited
            // blueprint, initial batch). Saying so on the row is what stops the NEXT net's
            // ancestry from being read as this one's.
            if facts.parents.is_empty() {
                line.push_str(" · root (no recorded parents)");
            }
        }
        if repeated {
            line.push_str(" · ↺ shown above");
        }
        // Depth cap: the parents could not be drawn above, so the cut is named on the net
        // they belong to rather than inventing a line with no net behind it.
        if !repeated && depth + 1 >= LINEAGE_MAX_DEPTH && !facts.parents.is_empty() {
            line.push_str(&format!(
                " · … {} parent(s), depth cap {LINEAGE_MAX_DEPTH}",
                facts.parents.len()
            ));
            self.truncated = true;
        }
        out.push(line);
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
        ("  mutations (random)".to_string(), a.immigrants.to_string()),
        (
            "  crossover children".to_string(),
            a.crossover_children.to_string(),
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

/// The `Minimal` frame's score cell: the population's mean score with the
/// arithmetic that produced it (`score = window avg ∓ λ × window spread`). The
/// caller appends the evolve tallies, so this stays a pure, small function the
/// identity can be tested through; the `Summ` rollup prints the same identity
/// in prose.
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

/// The seat line's score cell: the value a seat was dealt on, with the
/// arithmetic that produced it (`score = avg ∓ λ·spread`). The run legend
/// defines λ, so the operator's multiplier is named rather than repeated.
pub(crate) fn seat_score_cell(
    fit: Option<f32>,
    reading: Option<&WindowStats>,
    direction: crate::utils::fitness::Direction,
) -> String {
    match (fit, reading) {
        (Some(s), Some(r)) => format!(
            "score {} = {} {} λ·{}",
            fmt_line(s),
            fmt_line(r.mean),
            match direction {
                crate::utils::fitness::Direction::Maximize => "−",
                crate::utils::fitness::Direction::Minimize => "+",
            },
            fmt_line(r.std)
        ),
        (Some(s), None) => format!("score {}", fmt_line(s)),
        _ => "score —".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CardStats, NetShape, StopPlan, delta_crown, delta_pop, elite_badge_entry, lineage_parents,
        mean_of, minimal_score_cell, origin_cell, seat_score_cell, trained_note,
    };
    use crate::engine::smoothing::WindowStats;
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
            vec!["aaaaaaaaaaaaaaaa".to_string(), "bbbbbbbbbbbbbbbb".to_string()]
        );
        // Asexual clone: one parent still walks like a tree edge.
        assert_eq!(
            lineage_parents(Some("clone:parents=cccccccccccccccc")),
            vec!["cccccccccccccccc".to_string()]
        );
        // Roots name no in-run parent: a founder, an inherited seed, or no record at all.
        assert!(lineage_parents(Some("random")).is_empty());
        assert!(lineage_parents(Some("founder-2")).is_empty());
        assert!(lineage_parents(Some("seed:1793334914211:de1f105d")).is_empty());
        assert!(lineage_parents(None).is_empty());
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
        // are printed at all.
        assert!((48.9f32 - 0.5 * 23.0 - 37.4).abs() < 0.05);
        // A step-over-step delta rides the score it belongs to.
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

    #[test]
    fn seat_score_cell_prints_the_arithmetic_behind_the_seat_score() {
        // Legs that close EXACTLY at λ = 0.50: 161.3 − 0.50 × 91.0 = 115.8. (The
        // plan's worked example used σ = 90.8, which is a tenth off — the legs are
        // printed at 1dp, so a redo from them can miss in the last digit.)
        let reading = WindowStats {
            n: 10,
            mean: 161.3,
            std: 91.0,
            min: 100.0,
            max: 300.0,
            latest: 200.0,
        };
        let cell = seat_score_cell(Some(115.8), Some(&reading), Direction::Maximize);
        assert_eq!(cell, "score 115.8 = 161.3 − λ·91.0");
        assert!((161.3f32 - 0.5 * 91.0 - 115.8).abs() < 0.05);
        // Minimize flips the operator; a net with no window yet prints the score alone.
        let min = seat_score_cell(Some(115.8), Some(&reading), Direction::Minimize);
        assert!(min.contains("= 161.3 + λ·91.0"), "{min}");
        assert_eq!(
            seat_score_cell(Some(9.5), None, Direction::Maximize),
            "score 9.50"
        );
        assert_eq!(seat_score_cell(None, None, Direction::Maximize), "score —");
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
        assert_eq!(get("  mutations (random)").as_deref(), Some("99"));
        assert_eq!(get("  crossover children").as_deref(), Some("65"));
        assert_eq!(get("culled").as_deref(), Some("164"));
        assert_eq!(get("  culled (crossover roll)").as_deref(), Some("65"));
        assert_eq!(get("  culled (mutation roll)").as_deref(), Some("99"));
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

    /// A plan whose crown is `W:Tanh`, `W:Tanh+W:SiLU`, `W:ReLU` by hash `a/b/c` — the
    /// reference set both Δ legs are measured against, WITH the hash→crown-slot map the
    /// row builder actually uses (so a test can look a row up the way `stop_card` does).
    fn plan_of_three() -> StopPlan {
        let crown_idx = crown_idx_of(&["a", "b", "c"]);
        StopPlan {
            header: Vec::new(),
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

    /// The two Δ legs must compare against the RIGHT set — the whole point of putting
    /// them on both cards: a champion skips its own row (against itself every champion
    /// would read a meaningless `0.00`), a straggler does not.
    #[test]
    fn delta_crown_skips_the_champions_own_row() {
        let plan = plan_of_three();
        let me = shape("W:Tanh");
        // The row looks its OWN slot up by hash (what `stop_card` does) — not by its
        // position in the printed table.
        let slot = |hash: &str| plan.crown_idx.get(hash).copied().flatten();
        // Hash `a` IS champion 0: 0.5 to `W:Tanh+W:SiLU`, 1.0 to `W:ReLU` ⇒ 0.75.
        assert_eq!(slot("a"), Some(0));
        assert_eq!(delta_crown(Some(&me), &plan, slot("a")), Some(0.75));
        // Champion 1 skips ITS own slot the same way: 0.5 to Tanh, 1.0 to ReLU ⇒ 0.75.
        assert_eq!(delta_crown(Some(&shape_pair()), &plan, slot("b")), Some(0.75));
        // A net with NO crown slot (any other hash) is measured against all three, so its
        // coincidence with champion 0 counts: (0 + 0.5 + 1.0) / 3.
        assert_eq!(slot("zz"), None);
        assert_eq!(delta_crown(Some(&me), &plan, slot("zz")), Some(0.5));
        // An unreadable blueprint prints `—`, never a number.
        assert_eq!(delta_crown(None, &plan, slot("a")), None);
        // `k = 1`: a champion has no OTHER champion to be measured against ⇒ `—`.
        let solo = StopPlan {
            header: Vec::new(),
            has_eval: false,
            dealt_clock: 7,
            crown: vec![shape("W:Tanh")],
            crown_idx: crown_idx_of(&["a"]),
            population: Vec::new(),
        };
        assert_eq!(delta_crown(Some(&me), &solo, solo.crown_idx["a"]), None);
    }

    /// `Δpop` is the crowd, minus the row itself — and `—` (never `0.00`) when there is
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

    /// The card's own means: `—` legs are unreadable blueprints, not zeros, and an
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
