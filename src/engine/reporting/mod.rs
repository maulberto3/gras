//! The engine's user-facing output: the per-step rollup (`Summ`), the framed in-place table
//! (`Minimal`), and the stop-time summary.

pub mod format;
pub mod logging;
pub mod reports;

use tracing::info;

use self::format::{ReadingStyle, fmt_cell, fmt_census, fmt_cfg, fmt_line, fmt_micros, fmt_units};
use super::core::CoreEngine;
use crate::engine::clock::smoothing::rolling_mean;
use crate::engine::clock::step::LotteryFrame;
use crate::engine::diversity::{NetShape, mean_delta_to, shape_of};

mod lineage;

/// Untimed eval passes before the bench's clock starts — the first forward pays for allocator
/// and kernel selection, which is not what the cell should report.
const EVAL_BENCH_WARMUP: usize = 3;
/// Timed eval passes. The reading is a PRICE PER ROW, and the sweep pays it for every live net,
/// so the sweep is `pop × (warm-up + reps)` forwards: 8 is enough to average out scheduler noise
/// without turning the stop into the run's most expensive phase (64 reps cost 115 s at pop 500).
const EVAL_BENCH_REPS: usize = 8;
/// Rows in the benchmark input: the run's eval batch, capped so one huge batch cannot turn the
/// stop sweep into a long job. The cell prints cost per ROW, so the cap changes no reading.
const EVAL_BENCH_MAX_ROWS: usize = 256;

impl CoreEngine {
    /// Price every live net's eval forward ONCE, at the terminal stop, and stamp it into
    /// `meta.eval` — the number the `eval time (µs)` column and `nets/<hash>.json` carry.
    ///
    /// Deliberately NOT a per-step timer: timing a forward inside the loop would cost the race
    /// more than the reading is worth, and the price is a property of the blueprint + device +
    /// batch, not of the weights, so one measurement at the end is the number the loop would
    /// have averaged. RL has no shared batch: one pass = one TURN, a single observation.
    pub(crate) fn stamp_eval_bench(&mut self) {
        let hashes = self.state.live_hashes();
        if hashes.is_empty() {
            return;
        }
        let (unit, rows) = if self.trainer.is_rl() {
            ("turn", 1usize)
        } else {
            (
                "sample",
                self.stream
                    .as_ref()
                    .map(|s| s.eval_batch_size())
                    .unwrap_or(1)
                    .clamp(1, EVAL_BENCH_MAX_ROWS),
            )
        };
        let dim = self.meta_ctx.input_dim.max(1);
        // Values never change a forward's cost, so zeros keep the sweep off the RNG and off the
        // dataset: a timing run must not perturb the race's own streams.
        let data = vec![0.0f32; rows * dim];
        let shape = [rows as i64, dim as i64];
        let input = match flodl::Tensor::from_f32(&data, &shape, self.config.device()) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("eval bench input build failed: {e}");
                return;
            }
        };
        for hash in hashes {
            let Some(net) = self.networks.get_mut(hash.as_str()) else {
                continue;
            };
            match crate::utils::race_steps::bench_eval_forward(
                net,
                &input,
                EVAL_BENCH_WARMUP,
                EVAL_BENCH_REPS,
            ) {
                Ok(secs) => {
                    if let Some(state) = self.state.net_mut(&hash) {
                        state.meta.eval = crate::state::EvalBench {
                            secs_per_pass: Some(secs),
                            units_per_pass: Some(rows),
                            unit: unit.to_string(),
                            passes: EVAL_BENCH_REPS,
                        };
                    }
                }
                Err(e) => tracing::warn!("eval bench failed for {hash}: {e}"),
            }
        }
    }

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
                crate::utils::cfy::NO_VALUE.to_string()
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
        let fits: Vec<f32> = readings.iter().map(|r| r.score()).collect();
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
        let centers: Vec<f32> = readings.iter().map(|r| r.fam_center).collect();
        let fam_spreads: Vec<f32> = readings.iter().map(|r| r.fam_spread).collect();
        let leg_center = avg(&centers);
        let leg_spread = if spread_readable {
            avg(&fam_spreads)
        } else {
            f32::NAN
        };
        // The spread ACROSS the per-net scores: how wide the field is in ranking terms.
        let fits_spread = {
            let n = fits.len();
            if n < 2 {
                crate::utils::cfy::NO_VALUE.to_string()
            } else {
                let mean = avg(&fits);
                let var = fits.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
                fmt_line(var.sqrt())
            }
        };
        let score_cell = if readings.is_empty() {
            let (_, leg, _, _, _) = window_legs(self.config.ranking.mean_type);
            format!(
                "score{} avg — (no reading yet) │ {leg} —",
                direction.arrow()
            )
        } else if !self.config.ranking.selection_metrics.is_empty() {
            // A COMPOSITE ranking: the fitness centre plus weighted metric centres. Printing the
            // `= window avg …` identity here would be false, so the terms are named instead.
            let terms: Vec<String> = self
                .config
                .ranking
                .selection_metrics
                .iter()
                .map(|t| format!("{}·{}", t.label, fmt_cfg(t.weight)))
                .collect();
            format!(
                "score{} avg {} ± {} (σ over nets) = fitness centre + {} (COMPOSITE)",
                direction.arrow(),
                fmt_line(avg(&fits)),
                &fits_spread,
                terms.join(" + ")
            )
        } else {
            step_score_cell(
                avg(&fits),
                &fits_spread,
                leg_center,
                leg_spread,
                direction,
                self.config.ranking.mean_type,
            )
        };
        if self.log_level == crate::engine::config::LogLevel::Minimal {
            self.log_minimal_table(&trains, &evals, &fits, step_secs);
        } else {
            info!(
                "step {} │ nets {} │ loss↓ {} │ {} │ {} │ {}s{}",
                clock,
                self.state.live_count(),
                stats(&trains),
                mid,
                score_cell,
                fmt_line(step_secs),
                self.informative_rollup(),
            );
        }
    }

    /// The configured informative metrics as a rollup segment: the POPULATION MEAN of each,
    /// config order (the order `history.csv` writes its columns in), `na` until somebody reports
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
        let rows: Vec<String> = vec![
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
                minimal_score_cell(fit_m, &delta(fit_m, Some(pf)), self.fitness.direction()),
                e.culls,
                e.inserts,
            ),
            format!(
                "crossover fired {} ({} born, {} spent) │ mutation fired {} ({} born)",
                e.cross_fired, e.cross_survived, e.cross_discarded, e.mutate_fired, e.mutate_fired,
            ),
            format!("step took {}s", fmt_line(step_secs)),
        ];
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
        logging::frame(&frame.join("\n"));
        // Baseline for the next step's deltas.
        self.minimal_prev_means = Some((train_m, eval_m, fit_m));
    }

    /// Stop-time summary: which JSONs were written as the live frontier
    /// the resume command to continue from here.
    pub(crate) fn log_stop_summary(&self) -> flodl::tensor::Result<()> {
        let live = self.state.live_hashes();
        // CFY: the stop block is comfy tables — an artifact/resume card, the accounting matrix, then the elite card.
        let artifact_rows = vec![
            (
                "frontier snapshots".to_string(),
                format!(
                    "{} → nets/<hash>.json (one per live net) · pass pop_size {} to resume",
                    live.len(),
                    live.len()
                ),
            ),
            ("resume dir".to_string(), self.run_dir.display().to_string()),
            (
                "smoothing_window (K)".to_string(),
                self.config.ranking.smoothing_window.to_string(),
            ),
        ];
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            "Artifacts & resume",
            &crate::utils::cfy::kv(&artifact_rows),
            Some(
                "resume parity: the one-argument resume takes every knob from engine.json — a STRUCTURAL mismatch (mode, pop_size, dataset, trainer blob) is refused, a BEHAVIOURAL one (K, the mean family, any selection terms, the ticket band, parent draw, cull policies, probation) resumes and warns with BOTH values, so a caller who wants a different race passes its own config (TabularEngine::resume(dir, data_dir, config, …)) and reads the warning",
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
            Some(&format!(
                "built = nets admitted; discarded = crossover rolls whose attempts ran out with no viable child; blocked = mutation rolls stopped by the cull shield; {} crossover child(ren) were separately rejected as duplicates of a live topology — the roll is spent, the attempt budget is not, which is why they are named here instead of `discarded` (the old `Births & novelty` row, merged)",
                self.births.duplicates
            )),
        ));

        // What the DRAW did, not just what it cost: the novelty gate's own reading at birth.
        if self.births.births > 0 || self.births.duplicates > 0 {
            let b = &self.births;
            let dash = || crate::utils::cfy::NO_VALUE.to_string();
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

    /// The summary of the WHOLE measured field: one table, every live net, ranked by FIT, with the
    /// saved elite set marked in place. `final_table` says which MOMENT this is: the stop (artifacts
    /// written, footer says "stopped") or a checkpoint tick of a long race (a reading only — the
    /// table's cells are the same either way, but `final-field.csv`/`exam.json` stay stop-time).
    pub(crate) fn log_final_field(
        &self,
        clock: usize,
        verbose: bool,
        final_table: bool,
    ) -> CardStats {
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
            return CardStats::default();
        }
        let k = self.config.population.champion_count.max(1); // report at least the champion
        let top: Vec<(String, f32)> = self.champion_slice(&ranked);
        let mut marks: std::collections::HashMap<String, &'static str> =
            std::collections::HashMap::new();
        // ONE glyph answers "what did the run do with this net": `★` = saved as a CHAMPION —
        // the inheritable top-k. Nothing else can mark a champion, because nothing else exists.
        for (hash, _) in &top {
            marks.insert(hash.clone(), "★");
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
        if final_table {
            if let Some(report) = &exam_report {
                self.write_exam_json(report, &top);
            }
        }
        // The plan comes AFTER the exam: whether `exam (X)` is a column is a fact the exam
        // pass decides, and the header, the rows and the legend all read it here.
        let plan = self.stop_plan(&top, exam_report.is_some());
        // The lottery field the LAST clock drew from (`dealt_clock` = the clock the nets actually
        // stepped
        let lottery = self.lottery_frame();
        let (table, stats, grid) = self.stop_card(&rows, &plan, &marks, &exam_scores, lottery);
        let baseline = self.stop_baseline(clock);
        // The legend is a TABLE that assigns each column its LETTER and states the relations
        // those letters: `A` is checked against the `fit`, the centre leg and `spread`
        let legend_ctx = LegendCtx {
            window: self.config.ranking.smoothing_window,
            mean_type: self.config.ranking.mean_type,
            legs: self.config.lottery.train_ticket,
            p_min: self.train_merit_p_min(),
            p_max: self.train_merit_p_max(),
            tilt: self.lottery_tilt(),
            direction: self.fitness.direction(),
            lottery,
            k,
            dealt_clock: plan.dealt_clock,
            crossover_probation: self.config.population.crossover_probation_steps,
            mutation_probation: self.config.population.mutation_probation_steps,
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
        // The FILE first, and unconditionally at the STOP: `final-field.csv` is run data, written
        // from the very cells the card would print (one `StopPlan`, no second query). A checkpoint
        // card leaves it alone — `final-field.csv` names the end of the race, not one tick of it.
        if final_table {
            self.write_final_field_csv(&plan.header, &grid);
        }
        // `Summ` gets one compact digest of every column; the card below is the human table.
        // Built from the SAME header and grid the CSV just used, so it costs no second pass.
        if self.log_level == crate::engine::config::LogLevel::Summ {
            info!(
                "final-field summary │ {}",
                final_field_summary(&plan.header, &grid)
            );
        }
        // The title and the tail name the MOMENT: a checkpoint card must not claim the race
        // stopped, nor point at a file it did not write.
        let title = if final_table {
            format!(
                "Final field ({} nets — {} measured, ranked by F{arrow} · ★ {})",
                rows.len(),
                measured,
                top.len(),
            )
        } else {
            format!(
                "Field at checkpoint {clock} ({} nets — {} measured, ranked by F{arrow} · ★ {})",
                rows.len(),
                measured,
                top.len(),
            )
        };
        let moment = if final_table {
            format!(
                "stopped at clock {clock} · run seed {run_seed} (`--seed {run_seed}` replays this run bit-for-bit) · the same table is written to `final-field.csv` under this header"
            )
        } else {
            format!(
                "still running at clock {clock} · run seed {run_seed} · a reading of the field as it stands; `final-field.csv` is written when the race stops"
            )
        };
        let field_card = crate::utils::cfy::card(
            &title,
            &format!("{legend}\n{table}"),
            Some(&format!(
                "Δcrown = mean multiset-Jaccard distance to the {} reported champion(s): 0.00 = structurally identical to one of them, 1.00 = shares no gene with any · Δpop = mean distance to every other live net: 0.00 = a clone of the crowd, 1.00 = unlike everything alive · read the two AGAINST the baseline: Δcrown ≈ Δpop means the champion is structurally ordinary (selection is ranking numbers, not designs), Δcrown below it means the champions are a structural cluster · {} · {baseline} · {moment} · {exam_note}",
                plan.crown.len(),
                stats.line(),
            )),
        );
        if verbose {
            // The table is the LAST thing the console prints.
            crate::utils::cfy::emit(&field_card);
        }
        stats
    }

    /// The plan the ONE stop-time table renders from: one header, one champion set, one live
    /// population, one clock.
    fn stop_plan(&self, top: &[(String, f32)], has_exam: bool) -> StopPlan {
        // The champion set `Δcrown` measures to, in rank order; `crown_idx` maps a hash to its index
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
        // The columns this run can actually fill: the U leg's cell and the exam are columns
        // only when the run supplies them.
        let mut columns =
            printed_columns(self.config.lottery.train_ticket.updates_armed(), has_exam);
        let fit = fit_column(self.config.ranking.mean_type);
        // The window legs are GONE from the card (2026-11-14): with the spread charge removed, `F` IS the window centre, so a `mean`/`median` column beside it printed the same…
        for c in columns.iter_mut() {
            if *c == COL_FIT {
                *c = fit;
            }
        }
        let header = self.stop_header(&columns);
        StopPlan {
            header,
            columns,
            has_exam,
            dealt_clock: self.step_clock().saturating_sub(1),
            crown,
            crown_idx,
            population,
        }
    }

    /// The table's header, in NARRATIVE order: what the net IS (`#`/`hash`/`origin`, age, census) →
    /// how it is RANKED (fit and its min-max place) → what the coin paid it (the U leg,
    fn stop_header(&self, columns: &[&'static str]) -> Vec<String> {
        let mut header: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        // TWO setters shape these cells, so BOTH are read here: the tilt mirrors the place, and the U leg decides whether `Um` is a factor at all.
        let legs = self.config.lottery.train_ticket;
        let (place, u_cell, ticket) = (
            fit_minmax_column(legs.fitness_inverted()),
            updates_minmax_column(legs),
            ticket_column(legs),
        );
        for cell in header.iter_mut() {
            if cell == COL_FIT_MINMAX {
                *cell = place.to_string();
            } else if cell == COL_UPMINMAX {
                *cell = u_cell.to_string();
            } else if cell == COL_TICKET {
                *cell = ticket.to_string();
            }
            // BREAK the label at every joint: the header is the widest thing in the table, so its
            // height is what buys the rows their width.
            let wrapped = wrap_header(cell);
            *cell = wrapped;
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
        lottery: Option<LotteryFrame>,
    ) -> (comfy_table::Table, CardStats, Vec<Vec<String>>) {
        let mut table = comfy_table::Table::new();
        table.load_preset(comfy_table::presets::UTF8_FULL_CONDENSED);
        table.set_header(plan.header.clone());
        // The same cells, kept: the CSV export is written from THIS grid, so the printed
        // table and the file can never describe two different fields.
        let mut grid: Vec<Vec<String>> = Vec::with_capacity(rows.len());
        let mut crowns: Vec<Option<f32>> = Vec::with_capacity(rows.len());
        let mut pops: Vec<Option<f32>> = Vec::with_capacity(rows.len());
        // The RANK counts the ranked nets only: an unmeasured net prints `na` there (it has
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
            // `None` on an unreadable blueprint, which the cells render `na` rather than pretending
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
            // value (or `na`), and the marks live in the console card
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

    /// ONE row of the stop-time table: the cells, in header order, with the single `na`
    /// rule for "no value".
    fn stop_row(&self, row: StopRow<'_>) -> StopCells {
        let state = self.state.net(row.hash);
        // The RANKING value, not just the fitness half: with selection terms the two differ, and
        // `fit (F)` must print the number the run actually sorted on.
        let score = self.ranking_score_of(row.hash).or(row.fit);
        // The lottery's three grading numbers, derived ONCE for the three cells that print them;
        // `None` = the lottery did not grade this net (off, exempt, no reading, flat field),
        let parts = row
            .lottery
            .and_then(|frame| self.train_lottery_parts(row.hash, Some(frame), score));
        // TWO different counts, deliberately: `E` is what the table PRINTS (every pass that
        // scored the net: `U + O`), `U` (`state.step`) is the learning budget a ticket bought.
        let evals = state.map(|s| s.measured_clocks());
        let metrics = state.and_then(|s| s.last_metrics.as_ref());
        let born = state.map(|s| s.entered_at_step).unwrap_or(0);
        // Clocks lived INCLUSIVE of the birth clock and the last run — the convention
        // freeze lines use (`clock − entered_at_step`).
        let age = row.plan.dealt_clock.saturating_sub(born) + 1;
        let shape = row.shape;
        // The saved-set mark plus the net's GATE LETTERS, in one cell: the mark says what the run
        // DID with the net, the letters say what the net is still subject
        let shielded = self.on_probation(row.hash);
        let gate_born = if shielded { "b" } else { "" };
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
            // The wiring GENE, read beside the census it shaped: `1.00` = one Input port per
            // hidden node (the historical wiring), lower = fewer raw-input wires, deeper net.
            shape
                .map(|s| format!("{:.2}", s.reach))
                .unwrap_or_else(dash),
            // `H` — mean hop distance over the wires, right beside the gene that shapes them.
            shape.map(|s| fmt_cell(s.thickness)).unwrap_or_else(dash),
            // `D` — the longest wire path in this net's live blueprint. Report-only.
            shape.map(|s| s.depth.to_string()).unwrap_or_else(dash),
            // `t` — what ONE eval pass costs on this net's live blueprint, measured ONCE at
            // stop (`meta.eval`): seconds per SAMPLE (tabular) or per TURN (RL). Report-only.
            state
                .and_then(|s| s.meta.eval.secs_per_unit())
                .map(fmt_micros)
                .unwrap_or_else(dash),
            shape.map(|s| s.nodes.to_string()).unwrap_or_else(dash),
            // The split's POLICY and mean kept share: the label alone cannot tell a fresh
            // `Random` draw from `Exclusive`, and the number alone cannot tell either.
            shape.and_then(|s| s.split_cell()).unwrap_or_else(dash),
            shape.map(|s| s.wires.to_string()).unwrap_or_else(dash),
            // Mean hidden width is a MEASURED number like every other cell, so it takes the band
            // rule (`fmt_cell`) instead of a hand-picked `{:.1}` — a 3.5-wide net used to print
            shape.map(|s| fmt_cell(s.dim)).unwrap_or_else(dash),
            // ── how it is RANKED ─────────────────────────────────────── ONE number: `F`, the
            // window's centre — the value `WindowStats::score` returns, plus any
            // `selection_metrics` terms (the Summ line names those). The centre column and the
            // diagnostic spread column that used to sit to its left were dropped 2026-11-14: with
            // nothing charging a spread, the centre printed `F`'s own number twice and the spread
            // (`σ`, `1.4826·MAD`) was a statistic no cell reads.
            score.map(fmt_cell).unwrap_or_else(dash),
            // ── the whole-run trend, immediately right of the score ── `F` is where the net is
            // NOW (its last `n = min(B, K)` readings); `S` is the least-squares line through EVERY
            // reading it ever took, per READING and not per clock, so a starved net's gaps cost it
            // nothing and a net measured 5 times is comparable with one measured 80. Report-only:
            // no ranking, champion, seat, ticket or cull reads it. `na` = fewer than two readings
            // (a newborn), because one point has no line to fit.
            self.rolling_trend
                .get(row.hash)
                .and_then(|t| t.slope())
                .map(fmt_slope)
                .unwrap_or_else(dash),
            // The coin's three grading numbers, from the SAME frame and score as the ticket cell
            // below — one read, so the row cannot show a fraction that disagrees
            parts.map(|(f, _, _)| fmt_cell(f)).unwrap_or_else(dash),
        ];
        // `Um` is derived off `updates (U)`: `1` at the end the LEG pays and `0` at the other, so a `most_updates` leg prints the mirror of a `fewest_updates` one.
        if self.config.lottery.train_ticket.updates_armed() {
            let legs = self.config.lottery.train_ticket;
            let um = match (parts.is_some(), row.lottery, state) {
                (true, Some(frame), Some(s)) => {
                    let fresh =
                        Self::inverted_minmax(s.step, frame.fewest_updates, frame.most_updates);
                    Some(
                        if frame.most_updates <= frame.fewest_updates || legs.updates_inverted() {
                            fresh
                        } else {
                            1.0 - fresh
                        },
                    )
                }
                _ => None,
            };
            cells.push(um.map(fmt_cell).unwrap_or_else(dash));
        }
        // The chance the coin gave this net on the LAST dealt clock — the same `p_eff` the debug
        // card prints.
        cells.push(self.stop_ticket_p(row.hash, row.lottery, score));
        // `A` — clocks lived, inclusive of the birth clock — closes the ranking run beside the ticket: `T` is the chance per clock, `A` how many clocks this net was there to be…
        cells.push(age.to_string());
        // ── what its learning COST ─────────────────────────────────────
        // The budget it spent learning, then what it cost to MEASURE it: `U`, then `E` (every
        // pass that scored the net, act-only included). TWO cells: `E` already holds the act-only
        // passes, so there is nothing left for a reader to add up.
        cells.push(state.map(|s| s.step.to_string()).unwrap_or_else(dash));
        cells.push(evals.map(|t| t.to_string()).unwrap_or_else(dash));
        // NO `idle (I)` column: the withholding is visible in `E` versus `A`, and the debug card still prints the count for a clock that needs it.
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
            _ => None,
        } {
            seat_marks.push(seat);
        }
        if !seat_marks.is_empty() {
            let fit = col(fit_column(self.config.ranking.mean_type));
            inlined[fit].push(' ');
            inlined[fit].push_str(&seat_marks);
        }
        if !gate_born.is_empty() {
            inlined[col(COL_AGE)].push(' ');
            inlined[col(COL_AGE)].push('b');
        }
        // Two marks on the column that owns what they describe — `updates (U)`, the learning budget: `r<n>` = the recovery retries this net spent inside single clocks (each one a…
        if let Some(s) = state {
            let u = col(COL_UPDATES);
            let retries = s.recovery_retries();
            if retries > 0 {
                inlined[u].push(' ');
                inlined[u].push('r');
                inlined[u].push_str(&retries.to_string());
            }
            if s.step == 0 {
                inlined[u].push_str(" n");
            }
        }
        // NO `eval loss` column: it is the net's LAST clock's eval loss, i.e. the closing row that net already has in `history.csv` — a copy that costs a column and a forward pass…
        let mut tail: Vec<String> = Vec::new();
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

    /// The ticket cell: the chance the coin gave `hash` on the LAST dealt clock — `T = p_min +
    /// (p_max − p_min) × f`, the same `p_eff` the per-clock debug card prints, from the same band
    fn stop_ticket_p(
        &self,
        hash: &str,
        lottery: Option<LotteryFrame>,
        score: Option<f32>,
    ) -> String {
        let Some(frame) = lottery else {
            return dash();
        };
        if self.lottery_abstain(hash, Some(frame), score).is_some() {
            return dash();
        }
        fmt_cell(self.train_lottery_p_eff_from(hash, Some(frame), score))
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
    pub(crate) fn log_lottery_card(&self, clock: usize, frame: LotteryFrame) {
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
                // clocks the net got (`U`, not `M`: an act-only pass costs no learning budget).
                let measured = state.map(|s| s.measured_clocks()).unwrap_or(0);
                let merit = Self::merit_from(smt);
                let ratio = state
                    .filter(|s| s.step > 0)
                    .and_then(|s| smt.map(|a| a / s.step as f32))
                    .filter(|v| v.is_finite());
                // The BLENDED probability — the one the coin compares against —
                // so `p` and the `ticket?` column beside it cannot disagree.
                let p = self.train_lottery_p_eff_from(&h, Some(frame), merit);
                // Why a net is outside the lottery's reach, in one word:
                // the born shield always trains (p = 1.00).
                let exempt = if self.on_probation(&h) {
                    "shield"
                } else {
                    crate::utils::cfy::NO_VALUE
                };
                // `smt / D` is ONE quantity, so it prints as its two legs and the ratio between them: the numerator here, the divisor (`updates`) and the quotient beside each A…
                let born = state.is_some_and(|s| s.step == 0);
                let trains = born || self.train_lottery_draws_at(&h, clock, p);
                let row = vec![
                    h[..8.min(h.len())].to_string(),
                    exempt.to_string(),
                    smt.map(fmt_cell)
                        .unwrap_or_else(|| crate::utils::cfy::NO_VALUE.to_string()),
                    if state.is_some() {
                        measured.to_string()
                    } else {
                        crate::utils::cfy::NO_VALUE.to_string()
                    },
                    ratio
                        .map(fmt_cfg)
                        .unwrap_or_else(|| crate::utils::cfy::NO_VALUE.to_string()),
                    fmt_cfg(p),
                    if trains { "yes" } else { "no" }.to_string(),
                    state
                        .map(|s| s.idle_clocks().to_string())
                        .unwrap_or_else(|| crate::utils::cfy::NO_VALUE.to_string()),
                ];
                (exempt != crate::utils::cfy::NO_VALUE, p, row)
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
                "Training lottery (experimental, {}, band [{}, {}] = the chance each clock) — step {clock}",
                self.lottery_tilt(),
                fmt_cfg(self.train_merit_p_min()),
                fmt_cfg(self.train_merit_p_max())
            ),
            &crate::utils::cfy::table(
                &["net", "exempt", "F", "M", "F ÷ U", "T", "ticket?", "idle"],
                &rows.into_iter().map(|(_, _, row)| row).collect::<Vec<_>>(),
            ),
            Some(&format!(
                "F = the net's smoothed score (the K-reading window CENTRE, the ranking's number; the window spread is printed but never charged) · M = clocks the net was measured on: real updates + act-only passes (every ticket the coin denied while `train_merit_learn_only` is on) · `F ÷ U` = F ÷ its update count (this debug view only — the final table carries no efficiency column): informational — what one optimizer step bought, so a cheap net reads from an expensive one · field THIS clock drew from: best {} worst {} · T = where the net's SCORE sits between those two bounds as the graded fraction f = the product above (Fm⁺ × Um⁻ while the U leg is armed, otherwise Fm⁺; Fm = 1 at the BEST end, 0 at the worst — unless the tilt says otherwise, and this run is {}), then paid INSIDE the band — `p = p_min + (p_max − p_min) × f` over [p_min {}, p_max {}] — so the floor is literally the worst-graded net's chance per clock and the ceiling the best-graded one's (Ω retired: the band IS the payout) · the coin is a seeded `fastrand` draw `d ∈ [0,1)` compared to `p`: `d < p` sells the ticket, otherwise the coin withholds the weight UPDATE — with `train_merit_learn_only` on the net still acts and measures on a fresh batch, with it off the clock idles (no train, no eval, nothing measured) · ticket? = yes if THIS clock sells it the weight update; `no` means it did not learn, which with the flag off means it did nothing at all · exempt nets (the born shield) always run and grade 1.00 · idle = clocks the lottery left it idle BEFORE this clock ({} across the run); the step's lottery line names this clock's count",
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
        let every = self.config.run.checkpoint_every;
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
        let every = self.config.run.checkpoint_every;
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

    /// The mutation gate's reading, cumulative: how DIFFERENT the immigrants that were born turned out to be (Δ to the nearest archived blueprint) and what each birth cost in…
    pub(crate) fn log_birth_draws(&self, clock: usize) {
        let every = self.config.run.checkpoint_every;
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
        frame: LotteryFrame,
        ticketed: usize,
        idled: usize,
    ) {
        // The field the coin graded: live, out of the born shield,
        // reading — exactly `lottery_frame`'s filter, re-asked through the same helpers.
        let mut graded = 0usize;
        let mut p_sum = 0.0f32;
        for h in self.state.live_hashes() {
            if self.on_probation(&h) {
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
                p_sum += self
                    .last_ticket_p
                    .get(&h)
                    .copied()
                    .unwrap_or_else(|| self.train_lottery_p_eff_from(&h, Some(frame), merit));
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
                crate::utils::cfy::NO_VALUE.to_string()
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
        let p_txt = p_bar.map_or_else(|| crate::utils::cfy::NO_VALUE.to_string(), fmt_line);
        // Expected vs drawn, side by side: `p̄ × graded` is what the coin was worth,
        // `won` is what it actually produced. The gap is the coin, not a bug.
        let expected = p_bar.map_or_else(
            || crate::utils::cfy::NO_VALUE.to_string(),
            |p| fmt_line(p * graded as f32),
        );
        // Under learn-only the denied nets RAN and were measured: the same count is named
        // "act-only" — it is the coin's loss, not a missing row.
        let lost = if self.config.lottery.train_merit_learn_only {
            "act-only"
        } else {
            "idled"
        };
        let body = format!(
            "graded {} = won {} + {} {} │ ticketed {} = won {} + exempt {} │ avg ticket chance {} over the {} graded → expected {}, drawn {} │ field: best {} worst {} │ avg updates {} and avg tickets {} over the {} measured nets │ exempt = born shield (always run)",
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
            "step {} │ lottery ({}, band [{}, {}] = the chance each clock, so ticket T = p_min + (p_max − p_min) × f) │ {}",
            clock,
            self.lottery_tilt(),
            fmt_cfg(floor),
            fmt_cfg(ceiling),
            body,
        );
    }
}

/// The printed column names, spelled ONCE: the header cell, the legend key, the glyph placement
/// the CSV header all read these strings, so a rename cannot leave one of them on an old spelling.
const COL_FIT: &str = "fit (F = mean)";
/// The same column when the run averages in a SIGNED-LOG space (`MeanType::SignedLog`).
const COL_FIT_SIGNED_LOG: &str = "fit (F = g⁻¹(ḡ), g = sign(x)·ln(1+|x|))";
/// The same column for the order-statistic family (`MeanType::Median`).
const COL_FIT_MEDIAN: &str = "fit (F = median)";

/// The header cell for `fit (F)`, in the equation of the family this run averages
/// ONE function, so the header, the glyph target and the legend key are the same string.
fn fit_column(mean_type: crate::engine::config::MeanType) -> &'static str {
    match mean_type {
        crate::engine::config::MeanType::Arithmetic => COL_FIT,
        crate::engine::config::MeanType::SignedLog => COL_FIT_SIGNED_LOG,
        crate::engine::config::MeanType::Median => COL_FIT_MEDIAN,
    }
}
/// The whole-run TREND, printed immediately right of `fit (F)`: `F` says where the net is now, `S`
/// says which way it has been moving.
const COL_SLOPE: &str = "slope (S)";

/// The `slope (S)` cell: `fmt_cell`'s precision, EXCEPT a nonzero trend too fine to show in four
/// decimals — `0.0000` beside `F` would claim the net is flat when its own readings say otherwise.
fn fmt_slope(v: f32) -> String {
    if v != 0.0 && v.abs() < 0.0001 {
        format!("{v:+.1e}")
    } else {
        fmt_cell(v)
    }
}

const COL_FIT_MINMAX: &str = "fit minmax (Fm⁺)";
/// The same column under the INVERTED ordering: the tilt mirrors the place, so the equation
/// header states has to mirror with it (`1 −` the best-first place).
const COL_FIT_MINMAX_TILTED: &str = "fit minmax (Fm⁻)";
const COL_AGE: &str = "age (A)";
/// The U leg's cell (`set_train_ticket`): `0.99` at the field's fewest updates, `0.01` at its
/// most, so the ticket falls for a net that has spent its budget. Every leg keeps its SIGN.
const COL_UPMINMAX: &str = "updates minmax (Um⁻)";
/// The same cell under a `most_updates` leg: the updates are graded the other way round, so the
/// header names the mirror it now prints (`Um⁺ = (U − U_min) ÷ (U_max − U_min)`).
const COL_UPMINMAX_PLUS: &str = "updates minmax (Um⁺)";
const COL_TICKET: &str = "ticket minmax (T = p_min + (p_max − p_min) × Fm⁺)";
/// The ticket under the INVERTED ordering: it multiplies the MIRRORED place, so the equation must
/// name the cell actually on its left (`Fm⁻`), not the best-first `Fm` it no longer reads.
const COL_TICKET_TILTED: &str = "ticket minmax (T = p_min + (p_max − p_min) × Fm⁻)";
/// The ticket with a `fewest_updates` leg (`train_ticket`): the product is the place times `Um`.
const COL_TICKET_UP: &str = "ticket minmax (T = p_min + (p_max − p_min) × Fm⁺ × Um⁻)";
const COL_TICKET_UP_TILTED: &str = "ticket minmax (T = p_min + (p_max − p_min) × Fm⁻ × Um⁻)";
/// The ticket with a `most_updates` leg: the updates factor is read forward, so the equation names
/// `Um⁺` and the cell FALLS for the net that has trained least.
const COL_TICKET_UP_PLUS: &str = "ticket minmax (T = p_min + (p_max − p_min) × Fm⁺ × Um⁺)";
const COL_TICKET_UP_PLUS_TILTED: &str = "ticket minmax (T = p_min + (p_max − p_min) × Fm⁻ × Um⁺)";

/// The place cell this run prints — ONE function, so the header and its legend key cannot drift.
fn fit_minmax_column(inverted: bool) -> &'static str {
    if inverted {
        COL_FIT_MINMAX_TILTED
    } else {
        COL_FIT_MINMAX
    }
}

/// The ticket cell this run prints — named after the legs `set_train_ticket` actually armed, so the
/// equation names exactly the factors the row multiplies.
fn ticket_column(legs: crate::engine::config::TrainTicket) -> &'static str {
    use crate::engine::config::TrainTicket as L;
    match legs {
        L::MostFit => COL_TICKET,
        L::LeastFit => COL_TICKET_TILTED,
        L::MostFitFewestUpdates => COL_TICKET_UP,
        L::LeastFitFewestUpdates => COL_TICKET_UP_TILTED,
        L::MostFitMostUpdates => COL_TICKET_UP_PLUS,
        L::LeastFitMostUpdates => COL_TICKET_UP_PLUS_TILTED,
    }
}

/// BREAK a header over lines at its joints — after `=`, before `minmax`, before a `(`, between
/// `split`/`keep`/`%`. A column is as wide as its longest LINE, not as wide as the label was on
/// one line, so the header's height is what buys every ROW its width.
fn wrap_header(cell: &str) -> String {
    cell.replace("split (wires) · keep", "split\n(wires)\nkeep")
        .replace(" = ", " =\n")
        .replace(" minmax", "\nminmax")
        .replace(" + (", " +\n(")
        .replace(" × ", "\n× ")
        .replace(" (", "\n(")
        .replace("split · keep", "split\nkeep")
}

/// The U leg's cell under the leg's own polarity — `Um⁻` pays the fewest updates, `Um⁺` the most.
/// A disarmed leg has no polarity to name, so the bare cell stands.
fn updates_minmax_column(legs: crate::engine::config::TrainTicket) -> &'static str {
    if legs.updates_armed() && !legs.updates_inverted() {
        COL_UPMINMAX_PLUS
    } else {
        COL_UPMINMAX
    }
}
const COL_UPDATES: &str = "updates (U)";
/// `E`: every time the net ran the trainer and was SCORED — `U` (the updates that stuck) plus the act-only passes the lottery denied under `train_merit_learn_only`.
const COL_EVALS: &str = "evals (E)";
/// `H`: mean wire hop distance in topological levels — the drawn thickness as a number. `1.0` =
/// every wire joins adjacent levels; larger = skips. Report-only.
const COL_THICKNESS: &str = "thickness (H)";
/// `D`: the longest wire path from an Input node to any node; Input = 0, so
/// `Input → H1 → H2 → Output` reads `3`. Report-only: no ranking, seat, ticket or cull reads it.
const COL_DEPTH: &str = "depth (D)";
/// `t`: the wall-clock cost of ONE eval-mode forward pass, measured once at stop — per SAMPLE
/// in tabular mode, per TURN in RL. The unit (`µs`) is stated ONCE here, in the header, so the
/// cells hold bare numbers; the legend's row states what one unit was priced over. The header
/// says `time` because the cell is a DURATION: `evals (E)` right of it is a count of passes, and
/// two adjacent columns both called "eval" read as the same quantity.
const COL_EVAL: &str = "eval time (µs)";
const COL_EXAM: &str = "exam (X)";
/// `split (wires) · keep %`: the policy, how many wires leave the Input node, then the mean share
/// of the input each Input port reads. `Full` carries no masks (`100.0`), `Exclusive`/`Chunks` are
/// partitions (`1/P × 100`), `Random` is a repaired coin flip, and `Crossover` is a blended split.
/// The `%` is stated once, in the header; the parenthetical names the middle value, because the
/// share alone only names a partition's wire count.
const COL_MASK: &str = "split (wires) · keep %";

/// The columns the card actually PRINTS, in reading order — `STOP_HEADER_FIXED` minus the cells
/// this run cannot fill.
fn printed_columns(updates_leg: bool, has_exam: bool) -> Vec<&'static str> {
    STOP_HEADER_FIXED
        .iter()
        .copied()
        .filter(|c| match *c {
            COL_UPMINMAX => updates_leg,
            COL_EXAM => has_exam,
            _ => true,
        })
        .collect()
}

/// The fixed left-hand columns of the stop-time table, in NARRATIVE order: what the net IS (`#`, `hash`, `origin`, census) → how it is RANKED (`fit (F)`, its whole-run…
const STOP_HEADER_FIXED: &[&str] = &[
    "#",
    "hash",
    "origin",
    "params",
    "reach",
    // `H` and `D` sit right of the wiring gene: how thick its wires are, how deep the net is,
    // then what one eval pass of it COSTS — all properties of the blueprint, all read before
    // the census.
    COL_THICKNESS,
    COL_DEPTH,
    COL_EVAL,
    "nodes",
    COL_MASK,
    "wires",
    "width",
    // The word AND its letter, so a legend equation can be read straight off the row; the DERIVED column is named by the value it produces (`ticket`).
    COL_FIT,
    // The trend sits IMMEDIATELY right of the score: `F` is the level the field ranked on, `S` the direction its own readings have been going — the second thing a reader wants…
    COL_SLOPE,
    // ── the coin's own arithmetic, in the order it runs ── `fit minmax (Fm⁺)` = this net's `fit
    // (F)` as a 0..1 place in the last field (the tilt already applied: `1` = the end that earns),
    COL_FIT_MINMAX,
    COL_UPMINMAX,
    COL_TICKET,
    // `A` (clocks lived) closes the ranking run, right of the ticket: `T` is the chance the coin paid per clock and `A` is how many clocks this net was there to be paid for —…
    COL_AGE,
    // The learning budget, then the eval cost: `E` already holds the act-only passes, so these
    // are the only two cost columns — `A` stays the only count of clocks the net lived.
    COL_UPDATES,
    COL_EVALS,
    // The EXAM: rows no clock ever read, scored ONCE at stop on the reported nets.
    COL_EXAM,
    // The Δ columns are NOT here: a raw multiset-Jaccard distance sits at 0.7–0.9 between any two
    // unrelated nets, so a per-row value had no reference scale.
];

/// Everything the stop-time legend reads, bundled so its rows can be built (and checked
/// against the header) without a live engine.
struct LegendCtx<'a> {
    window: usize,
    /// The family the window is averaged in — its header equation and its legend row
    /// picked by this, so the two cannot disagree (`set_fitness_mean_type`).
    mean_type: crate::engine::config::MeanType,
    /// The lottery's two LEGS (`set_train_ticket`): the place cell, the U cell and the ticket
    /// equation all name the side this one value pays, so they cannot disagree.
    legs: crate::engine::config::TrainTicket,

    /// The last clock the lottery dealt — which clock the ticket column is read
    /// against.
    dealt_clock: usize,
    /// The lottery's band and ordering: the stop table's field row states them, because
    /// the two ends it names are not the two ends that earn the tickets when tilted.
    p_min: f32,
    p_max: f32,
    tilt: &'a str,
    direction: crate::utils::fitness::Direction,
    lottery: Option<LotteryFrame>,
    k: usize,
    crossover_probation: usize,
    mutation_probation: usize,
}

/// The stop-time legend: `abbrev | description`, one row per column, keyed
fn stop_legend_rows(ctx: &LegendCtx<'_>) -> Vec<(String, String)> {
    // The symbol the printed table uses for the place, and the header that cell carries.
    let place = if ctx.legs.fitness_inverted() {
        "Fm⁻"
    } else {
        "Fm⁺"
    };
    let minmax = fit_minmax_column(ctx.legs.fitness_inverted());
    let family = match ctx.mean_type {
        crate::engine::config::MeanType::Arithmetic => "the window mean",
        crate::engine::config::MeanType::SignedLog => {
            "`g⁻¹` of the mean of `g(x) = sign(x)·ln(1+|x|)`"
        }
        crate::engine::config::MeanType::Median => "the window's weighted median",
    };
    let ticket = match ctx.lottery.as_ref() {
        Some(_) => format!(
            "`T`: the chance the coin gave this net on the last dealt clock (clock {}), paid inside the band — it trains iff `d < T`, `d` uniform on [0,1). `na` when it was not graded (born shield, no reading, lottery off)",
            ctx.dealt_clock
        ),
        None => "`na` on every row: no lottery field was graded at the last clock".to_string(),
    };
    let band = match ctx.lottery.as_ref() {
        // The frame is stored direction-adjusted, so convert back into the COLUMN's direction.
        Some(f) => {
            let (best, worst) = match ctx.direction {
                crate::utils::fitness::Direction::Maximize => (f.best, f.worst),
                crate::utils::fitness::Direction::Minimize => (-f.best, -f.worst),
            };
            format!(
                "the last clock's lottery field: best `F = {}`, worst `F = {}` over the graded nets; band [p_min {}, p_max {}] = each clock's chance (the worst-graded net is paid `p_min`, the best `p_max`); tilt {} picks which end earns",
                fmt_cell(best),
                fmt_cell(worst),
                fmt_cfg(ctx.p_min),
                fmt_cfg(ctx.p_max),
                ctx.tilt,
            )
        }
        None => format!(
            "no lottery field at the last clock (floor 1.0, everyone trains, or the field was never armed): band [p_min {}, p_max {}], tilt {}",
            fmt_cfg(ctx.p_min),
            fmt_cfg(ctx.p_max),
            ctx.tilt,
        ),
    };
    let mut rows = vec![
        (
            "#".to_string(),
            "rank by `fit (F)`, best first; `na` = a live net with no reading yet (listed last, unranked)"
                .to_string(),
        ),
        (
            "hash".to_string(),
            "first 8 hex of the blueprint's xxh3 — its key in `nets/<hash>.json`, `champions.json` and the CSV"
                .to_string(),
        ),
        // ── what this net IS ────────────────────────────────────────────
        (
            "origin".to_string(),
            "a founder slot, or the crossover/mutation op and birth clock (a child's first ticket is the NEXT clock)"
                .to_string(),
        ),
        (
            "params".to_string(),
            "trainable parameters in K/M/B units (`40.2K` = 40 200); the exact count is in `nets/<hash>.json`"
                .to_string(),
        ),
        (
            "reach".to_string(),
            "the `input_reach` gene: share of hidden nodes wired straight from the Input node; lower = fewer Input ports, deeper net"
                .to_string(),
        ),
        (
            COL_THICKNESS.to_string(),
            "`H`: mean wire hop distance in topological levels — `1.0` = every wire joins adjacent levels, larger = skips. Report-only: no ranking, seat, ticket or cull reads it"
                .to_string(),
        ),
        (
            COL_DEPTH.to_string(),
            "`D`: the longest wire path from an Input node to any node (`Input` = 0). A chain `Input → H1 → H2 → Output` reads `3`; hidden-layer count is `D − 1`. Report-only: no ranking, seat, ticket or cull reads it"
                .to_string(),
        ),
        (
            COL_EVAL.to_string(),
            format!(
                "`t`: wall-clock cost of ONE eval-mode forward, per sample (tabular) or per turn (RL) — {EVAL_BENCH_WARMUP} warm-up + {EVAL_BENCH_REPS} timed passes, run once at stop and kept in `meta.eval`. Report-only: no ranking, seat, ticket or cull reads it"
            ),
        ),
        (
            "nodes".to_string(),
            "blueprint size: input + hidden + output".to_string(),
        ),
        (
            COL_MASK.to_string(),
            "the split policy, the wires leaving the Input node and the mean share of the input each Input port reads, in percent: `Full · 1 · 100.0` = no masks, `Exclusive · P · 1/P×100` = a partition (the middle number is the wire count), `Chunks · 100/P` = an ordered partition, `Random`/`Crossover` = a coin-flipped or blended split, where the share alone would not name the wire count"
                .to_string(),
        ),
        (
            "wires".to_string(),
            "blueprint connections, one per output-port → input-port wire".to_string(),
        ),
        (
            "width".to_string(),
            "mean `hidden_dim` across the hidden nodes".to_string(),
        ),
        (
            COL_AGE.to_string(),
            "`A = clock − born + 1`: clocks lived, birth clock included — the clocks there were to take a ticket in"
                .to_string(),
        ),
        // ── how it is RANKED ────────────────────────────────────────────
        (
            fit_column(ctx.mean_type).to_string(),
            format!(
                "`F` = {family} over the last `n = min(B, K)` readings, K = {}; `selection_metrics` terms are added. No spread charge",
                ctx.window
            ),
        ),
        (
            COL_SLOPE.to_string(),
            "`S`: least-squares slope of the net's own readings, per READING (not per clock); sign follows `direction`. Report-only; `na` = fewer than two readings"
                .to_string(),
        ),
        (
            minmax.to_string(),
            format!(
                "`{place} = (F − F_worst) ÷ (F_best − F_worst)`: `F` as a 0..1 place in the last field (clock {}), tilt {} already applied — `0.99` is the end that earns. Both ends sit inside `[0.01, 0.99]`; their numbers are in the `F field · band` row",
                ctx.dealt_clock, ctx.tilt
            ),
        ),
        (ticket_column(ctx.legs).to_string(), ticket),
        (
            updates_minmax_column(ctx.legs).to_string(),
            "the U leg: `U` as a 0..1 place in the graded field (`Um⁻` pays the fewest updates, `Um⁺` the most), NOT mirrored by the tilt. Printed only while the leg is armed and the net graded"
                .to_string(),
        ),
        // ── what its learning COST ──────────────────────────────────────
        (
            COL_EVALS.to_string(),
            "`E`: eval passes — `U` plus the act-only passes a withheld ticket still ran; an idle clock is in neither, so `E` can trail `A` (and a recovery streak can push it ahead)"
                .to_string(),
        ),
        ("F field · band".to_string(), band),
        (
            COL_UPDATES.to_string(),
            "`U`: real optimizer updates applied (normally one per clock, more inside a recovery streak); `r<n>` = retries taken; `n` = never learned yet"
                .to_string(),
        ),
        // ── the age identity, as far as the row can show it now that `idle` is not printed ──
        (
            "A = E + withheld (+1 if born mid-run)".to_string(),
            "each clock lived is an eval pass or a clock the lottery withheld; a mid-run child also sits out its birth clock"
                .to_string(),
        ),
        (
            "B = E".to_string(),
            "every ticket is an eval pass".to_string(),
        ),
        // ── what this net IS (the verdict) ────────────────────────────
        (
            COL_EXAM.to_string(),
            "`X`: the same scorer over rows no clock ever read, scored once at stop into `exam.json`; ranks nothing — read it against `F`"
                .to_string(),
        ),
        (
            "mark".to_string(),
            format!(
                "`★` = saved champion (top-{} by `F`), `·` = not saved; inline in the deciding column, and not carried by the CSV",
                ctx.k
            ),
        ),
        (
            "mark b · j · t".to_string(),
            format!(
                "`b` = inside its born cull shield (crossover {} / mutation {} updates of `U`): cannot be culled; serving it takes training",
                ctx.crossover_probation, ctx.mutation_probation,
            ),
        ),
        // ── the informative band (appended after the last fixed column) ────
        (
            "m:<name>".to_string(),
            "an informative metric from `set_run_metrics`: measured on the eval batch, in `history.csv`, ranked by nothing — appended after `exam (X)` in config order"
                .to_string(),
        ),
    ];
    // The legend follows the HEADER: a reader with the table and the legend side by side finds
    // same column in the same relative place.
    let slot = |key: &str| -> usize {
        STOP_HEADER_FIXED
            .iter()
            .position(|c| *c == key)
            .unwrap_or(STOP_HEADER_FIXED.len())
    };
    rows.sort_by_key(|(key, _)| slot(key));
    rows
}

#[cfg(test)]
mod stop_legend_tests {
    use super::*;

    fn ctx() -> LegendCtx<'static> {
        LegendCtx {
            window: 10,
            mean_type: crate::engine::config::MeanType::Arithmetic,
            legs: crate::engine::config::TrainTicket::MostFit,
            dealt_clock: 0,
            p_min: 0.0,
            p_max: 1.0,
            tilt: "best-first",
            direction: crate::utils::fitness::Direction::Maximize,
            lottery: None,
            k: 1,
            crossover_probation: 5,
            mutation_probation: 5,
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
        // The conditional columns (`m:<name>`, the informative metrics) are the only ones outside
        // the 1:1 rule: they exist only when a trainer reports them.
        assert!(
            rows.iter().any(|(k, _)| k == "m:<name>"),
            "the `m:` metric prefix needs a legend row"
        );
    }

    /// Thickness and depth are read beside the wiring gene they follow, each with its own
    /// explanation.
    #[test]
    fn thickness_and_depth_sit_right_of_reach_and_have_legend_rows() {
        let cols = printed_columns(true, true);
        let reach = cols
            .iter()
            .position(|c| *c == "reach")
            .expect("reach is a printed column");
        assert_eq!(cols[reach + 1], COL_THICKNESS, "columns: {cols:?}");
        assert_eq!(cols[reach + 2], COL_DEPTH, "columns: {cols:?}");
        let rows = stop_legend_rows(&ctx());
        for col in [COL_THICKNESS, COL_DEPTH] {
            assert!(
                rows.iter().any(|(k, _)| k == col),
                "{col} has no legend row: {:?}",
                rows.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>()
            );
        }
    }

    /// The card prints NO window leg beside `F`: with the spread charge gone `F` IS the window centre, so a `mean`/`median` column held the same number twice, and the…
    #[test]
    fn the_card_prints_no_window_leg_beside_the_score() {
        use crate::engine::clock::smoothing::{RollingBuffer, rolling_stats_with};
        use crate::engine::config::MeanType;
        // The window that started it: two good readings and one low one.
        let buf = RollingBuffer::from_values(10, &[0.828125, 0.859375, 0.859375]);
        let stats = rolling_stats_with(&buf, MeanType::Median).unwrap();
        // `F` is the centre, so the dropped column carried exactly the cell `fit (F)` carries.
        assert_eq!(stats.score(), stats.fam_center, "F is the centre");
        // Nothing named like a window leg survives in the header the rows, the CSV export and the
        // legend all share.
        let cols = printed_columns(true, true);
        for gone in [
            "mean",
            "spread",
            "window median",
            "1.4826·MAD",
            "window g-avg",
            "window g-spread",
        ] {
            assert!(
                !cols.contains(&gone),
                "`{gone}` is still a column: {cols:?}"
            );
        }
        // The family still rides the `fit` header, so the row says which window the number is.
        assert_eq!(fit_column(MeanType::Median), COL_FIT_MEDIAN);
        assert_eq!(fit_column(MeanType::Arithmetic), COL_FIT);
        assert_eq!(fit_column(MeanType::SignedLog), COL_FIT_SIGNED_LOG);
        assert!(COL_FIT_MEDIAN.contains("median") && COL_FIT.contains("mean"));
    }

    /// The clock columns are `age (A)` (beside the ticket) and `updates (U) | evals (E)` — `E` holds the act-only passes, so a reader has no third number to add up (and none to…
    #[test]
    fn the_clock_columns_are_age_updates_and_evals() {
        let rows = stop_legend_rows(&ctx());
        let keys: Vec<&str> = rows.iter().map(|(k, _)| k.as_str()).collect();
        assert!(
            keys.contains(&COL_EVALS),
            "the evals column has no legend row: {keys:?}"
        );
        assert!(
            keys.iter().all(|k| !k.contains("act (O)")),
            "the act column has no legend row of its own: {keys:?}"
        );
        // The plan's column list drives the header, the rows and the export, so dropping a column
        // somewhere it is still listed would break all three at once.
        let cols = printed_columns(true, true);
        assert!(cols.contains(&COL_UPDATES), "{cols:?}");
        assert!(cols.contains(&COL_EVALS), "{cols:?}");
        assert!(
            STOP_HEADER_FIXED.iter().all(|c| !c.contains("act (O)")),
            "the act column is still in the header: {STOP_HEADER_FIXED:?}"
        );
    }

    /// `S` sits immediately right of `F`, and `A` immediately right of the ticket: the two
    /// placements the row's left-to-right reading depends on.
    #[test]
    fn the_slope_sits_right_of_fit_and_age_right_of_the_ticket() {
        let cols = printed_columns(true, true);
        let at = |c: &str| cols.iter().position(|x| *x == c).expect(c);
        assert_eq!(at(COL_SLOPE), at(COL_FIT) + 1, "{cols:?}");
        assert_eq!(at(COL_FIT_MINMAX), at(COL_SLOPE) + 1, "{cols:?}");
        assert_eq!(at(COL_AGE), at(COL_TICKET) + 1, "{cols:?}");
        // The legend orders itself off the same header, so the two cannot drift apart.
        let rows = stop_legend_rows(&ctx());
        let keys: Vec<&str> = rows.iter().map(|(k, _)| k.as_str()).collect();
        let k = |c: &str| keys.iter().position(|x| *x == c).expect(c);
        assert_eq!(k(COL_SLOPE), k(fit_column(ctx().mean_type)) + 1, "{keys:?}");
        assert_eq!(k(COL_AGE), k(ticket_column(ctx().legs)) + 1, "{keys:?}");
    }

    /// A trend too fine for four decimals is not printed as `0.0000`.
    #[test]
    fn a_tiny_slope_is_printed_in_exponent_form() {
        assert_eq!(fmt_slope(0.0), fmt_cell(0.0));
        assert_eq!(fmt_slope(0.0123), fmt_cell(0.0123));
        assert!(fmt_slope(0.00003).contains('e'), "{}", fmt_slope(0.00003));
        assert!(
            fmt_slope(-0.00003).contains("e-"),
            "{}",
            fmt_slope(-0.00003)
        );
    }

    /// Under tilt BOTH equation cells name the mirror: the ticket multiplies `Fm⁻`, so it must not
    /// print a `Fm` its neighbour no longer holds.
    #[test]
    fn tilted_ticket_names_the_mirrored_place() {
        let mut c = ctx();
        c.legs = crate::engine::config::TrainTicket::LeastFit;
        let rows = stop_legend_rows(&c);
        let keys: Vec<&str> = rows.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&COL_FIT_MINMAX_TILTED), "{keys:?}");
        assert!(keys.contains(&COL_TICKET_TILTED), "{keys:?}");
        assert!(!keys.contains(&COL_FIT_MINMAX), "{keys:?}");
        assert!(!keys.contains(&COL_TICKET), "{keys:?}");
        assert!(COL_TICKET_TILTED.contains("× Fm⁻"));
        assert!(COL_TICKET.contains("× Fm⁺"));
        assert!(
            COL_TICKET_UP.contains("× Fm⁺ × Um⁻"),
            "a leg without a sign is a leg a reader cannot place"
        );
        assert!(COL_UPMINMAX.contains("Um⁻"));
        // The header rewrite and the legend key read the SAME helper, so the cell and its row agree.
        assert_eq!(fit_minmax_column(true), COL_FIT_MINMAX_TILTED);
        assert_eq!(
            ticket_column(crate::engine::config::TrainTicket::MostFit),
            COL_TICKET
        );
    }

    /// ONE value names both legs: every `TrainTicket` spelling has its own equation, and the header
    /// a reader sees is the key its legend row carries.
    #[test]
    fn the_ticket_equation_follows_the_legs() {
        use crate::engine::config::TrainTicket;
        let mut headers: Vec<&'static str> = Vec::new();
        for legs in [
            TrainTicket::MostFit,
            TrainTicket::MostFitFewestUpdates,
            TrainTicket::MostFitMostUpdates,
            TrainTicket::LeastFit,
            TrainTicket::LeastFitFewestUpdates,
            TrainTicket::LeastFitMostUpdates,
        ] {
            let mut c = ctx();
            c.legs = legs;
            let header = ticket_column(legs);
            let rows = stop_legend_rows(&c);
            assert!(
                rows.iter().any(|(k, _)| k == header),
                "`{header}` has no legend row — keys: {:?}",
                rows.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>()
            );
            assert_eq!(header.contains("Fm⁻"), legs.fitness_inverted(), "{header}");
            assert_eq!(header.contains("× Um"), legs.updates_armed(), "{header}");
            assert_eq!(
                header.contains("Um⁺"),
                legs.updates_armed() && !legs.updates_inverted(),
                "{header}"
            );
            headers.push(header);
        }
        headers.sort_unstable();
        headers.dedup();
        assert_eq!(
            headers.len(),
            6,
            "every leg must print its own equation: {headers:?}"
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
        assert_eq!(MeanType::default(), MeanType::Median);
    }

    /// The `Summ` digest names every column and reduces it to a range, a distinct count, or a
    /// mean ± spread — the same shape as the table, in one line.
    #[test]
    fn the_final_field_summary_reduces_every_column() {
        let header = vec![
            "#".to_string(),
            "hash".to_string(),
            "origin".to_string(),
            "params".to_string(),
            "thickness (H)".to_string(),
            "depth (D)".to_string(),
            "split (wires) · keep %".to_string(),
            "fit (F = mean)".to_string(),
            "slope (S)".to_string(),
        ];
        let rows = vec![
            vec![
                "1",
                "aaaa",
                "founder@0",
                "1.0K",
                "1.0",
                "2",
                "Exclusive · 4 · 50.0",
                "0.50",
                "0.01",
            ],
            vec![
                "2",
                "bbbb",
                "founder@0",
                "2.0K",
                "1.5",
                "3",
                "Random · 2 · 75.0",
                "0.70",
                "na",
            ],
        ]
        .into_iter()
        .map(|row| row.into_iter().map(str::to_string).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let line = final_field_summary(&header, &rows);
        assert!(line.starts_with("2 net(s) · # 1..2"), "{line}");
        for part in [
            "hash 2 uniq",
            "origin 1 uniq",
            "params 1.5K",
            "H 1.25",
            "D 2.50",
            "split 2 policy · keep 62.5",
            "F 0.6000",
            "S 0.0100 (na 1)",
        ] {
            assert!(line.contains(part), "missing `{part}` in {line}");
        }
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
    /// The shared header — `rank … Δcrown · Δpop · origin`, plus the informative-metric columns.
    header: Vec<String>,
    /// The columns the header, the rows and the `final-field.csv` export share, in reading
    /// order — `STOP_HEADER_FIXED` minus the cells this run cannot fill.
    columns: Vec<&'static str>,
    /// Whether the run supplied an exam, i.e. whether `exam (X)` is a column
    has_exam: bool,
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

/// How one final-field column reduces to a single summary cell.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SummaryKind {
    /// `#`: report the range, not a mean rank.
    Rank,
    /// `hash` / `origin`: report how many distinct values there are.
    Text,
    /// `params`: values carry `K`/`M`/`B`, so parse to a base count.
    Params,
    /// `split`: policy label + numeric keep share, so summarize both halves.
    Split,
    /// Everything else numeric: mean ± population spread.
    Number,
}

fn summary_kind(header: &str) -> SummaryKind {
    let h = header.replace('\n', " ");
    let h = h.trim();
    if h == "#" {
        SummaryKind::Rank
    } else if h == "hash" || h == "origin" {
        SummaryKind::Text
    } else if h == "params" {
        SummaryKind::Params
    } else if h.starts_with("split") {
        SummaryKind::Split
    } else {
        SummaryKind::Number
    }
}

/// The short label a `Summ` digest uses for a final-field column: `D`/`H`/`F` stay the letters
/// the card already teaches, while the long equation headers collapse to their first noun.
fn short_column(header: &str) -> String {
    let h = header.replace('\n', " ");
    let h = h.trim();
    let short = if h == "#" {
        "#"
    } else if h.starts_with("thickness") {
        "H"
    } else if h.starts_with("depth") {
        "D"
    } else if h.starts_with("eval time") {
        "t"
    } else if h.starts_with("split") {
        "split"
    } else if h.starts_with("fit minmax") {
        "Fm"
    } else if h.starts_with("updates minmax") {
        "Um"
    } else if h.starts_with("ticket minmax") {
        "T"
    } else if h.starts_with("fit (") {
        "F"
    } else if h.starts_with("slope") {
        "S"
    } else if h.starts_with("age") {
        "A"
    } else if h.starts_with("updates (") {
        "U"
    } else if h.starts_with("evals (") {
        "E"
    } else if h.starts_with("exam") {
        "X"
    } else if h.starts_with("m:") {
        h
    } else {
        h.split([' ', '(']).next().unwrap_or(h)
    };
    short.to_string()
}

fn summary_number(cell: &str, kind: SummaryKind) -> Option<f64> {
    let cell = cell.trim();
    if kind == SummaryKind::Params {
        for (suffix, scale) in [("B", 1e9_f64), ("M", 1e6), ("K", 1e3)] {
            if let Some(number) = cell.strip_suffix(suffix) {
                return number.parse::<f64>().ok().map(|v| v * scale);
            }
        }
    }
    cell.parse::<f64>().ok()
}

/// One compact `Summ` digest of the final field: every column, reduced to a range, a distinct
/// count, or mean ± spread. Built from the same header and plain cells as `final-field.csv`.
fn final_field_summary(header: &[String], rows: &[Vec<String>]) -> String {
    let mut parts = vec![format!("{} net(s)", rows.len())];
    for (col, head) in header.iter().enumerate() {
        let label = short_column(head);
        let kind = summary_kind(head);
        let values: Vec<&str> = rows
            .iter()
            .filter_map(|row| row.get(col))
            .map(|cell| cell.trim())
            .filter(|cell| {
                !cell.is_empty()
                    && *cell != crate::utils::cfy::NO_VALUE
                    && *cell != "—"
                    && *cell != "?"
            })
            .collect();
        if values.is_empty() {
            parts.push(format!("{label} na"));
            continue;
        }
        if kind == SummaryKind::Text {
            let unique: std::collections::HashSet<&str> = values.iter().copied().collect();
            parts.push(format!("{label} {} uniq", unique.len()));
            continue;
        }
        let numbers: Vec<f64> = match kind {
            SummaryKind::Split => values
                .iter()
                .filter_map(|cell| {
                    cell.rsplit_once(" · ")
                        .and_then(|(_, keep)| keep.parse().ok())
                })
                .collect(),
            _ => values
                .iter()
                .filter_map(|cell| summary_number(cell, kind))
                .collect(),
        };
        if numbers.len() != values.len() {
            // A mixed column is still a value distribution; count its distinct spellings rather
            // than inventing a mean from the subset that happened to parse.
            let unique: std::collections::HashSet<&str> = values.iter().copied().collect();
            parts.push(format!("{label} {} uniq", unique.len()));
            continue;
        }
        let missing = rows.len().saturating_sub(values.len());
        let mut text = match kind {
            SummaryKind::Rank => {
                let min = numbers.iter().copied().fold(f64::INFINITY, f64::min);
                let max = numbers.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                format!("{label} {min}..{max}")
            }
            SummaryKind::Params => {
                let mean = numbers.iter().sum::<f64>() / numbers.len() as f64;
                format!("{label} {}", fmt_units(mean.round() as i64))
            }
            SummaryKind::Split => {
                let policies: std::collections::HashSet<&str> = values
                    .iter()
                    .filter_map(|cell| cell.split_once(" · ").map(|(policy, _)| policy))
                    .collect();
                let mean = numbers.iter().sum::<f64>() / numbers.len() as f64;
                let var =
                    numbers.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / numbers.len() as f64;
                format!(
                    "{label} {} policy · keep {}±{}",
                    policies.len(),
                    fmt_line(mean as f32),
                    fmt_line(var.sqrt() as f32)
                )
            }
            SummaryKind::Text => unreachable!("text columns return above"),
            SummaryKind::Number => {
                let mean = numbers.iter().sum::<f64>() / numbers.len() as f64;
                if numbers.len() > 1 {
                    let var = numbers.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / numbers.len() as f64;
                    format!(
                        "{label} {}±{}",
                        fmt_line(mean as f32),
                        fmt_line(var.sqrt() as f32)
                    )
                } else {
                    format!("{label} {}", fmt_line(mean as f32))
                }
            }
        };
        if missing > 0 {
            text.push_str(&format!(" (na {missing})"));
        }
        parts.push(text);
    }
    parts.join(" · ")
}

/// One row of a stop-time card.
struct StopCells {
    inlined: Vec<String>,
    plain: Vec<String>,
}

struct StopRow<'a> {
    /// 1-based rank in the race, best FIT first; `None` for an UNMEASURED net, which has
    /// no FIT to be ranked by (it prints `na` there and rides at the bottom of the table).
    rank: Option<usize>,
    /// The saved-set mark: `★` saved as a champion, `·` not saved.
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
    lottery: Option<LotteryFrame>,
}

/// One card's own structural summary: the mean `Δcrown` and `Δpop` over the rows it printed.
/// Returned by `log_final_field` so the stop record can persist the two numbers the card showed.
#[derive(Clone, Debug, Default)]
pub(crate) struct CardStats {
    pub(crate) delta_crown: Option<f32>,
    pub(crate) delta_pop: Option<f32>,
}

impl CardStats {
    /// `card mean: Δcrown 0.31 · Δpop 0.55` — `na` when no row carried the number.
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
    crate::utils::cfy::NO_VALUE.to_string()
}

/// A formatted scalar, or `na` when there is none.
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
        // How much of the child the other parent contributed: under one-point the cut's hidden nodes,
        // under uniform the per-gene coins — `3/6` is the honest cousin of `fit`.
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
                crate::utils::cfy::NO_VALUE.to_string()
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
    let dash = || crate::utils::cfy::NO_VALUE.to_string();
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
            "crossover-lowest-fitness".to_string(),
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

/// How the `Summ` line spells its window legs, per family: `(separator, centre noun, spread noun, short spread noun, note)`.
fn window_legs(
    family: crate::engine::config::MeanType,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    use crate::engine::config::MeanType;
    match family {
        MeanType::Arithmetic => ("=", "window avg", "window spread", "spread", ""),
        // The header's spelling (`COL_FIT_MEDIAN`): `F = median`.
        MeanType::Median => ("=", "window median", "1.4826·MAD", "1.4826·MAD", ""),
        MeanType::SignedLog => (
            "≈",
            "window g-avg",
            "window g-spread",
            "g-spread",
            " · F = g⁻¹(legs)",
        ),
    }
}

/// The `Summ` line's `score↑` cell: the population's mean score, the spread ACROSS nets, and the window legs that produced it — in the FAMILY the ranking used, so the…
fn step_score_cell(
    fit_m: f32,
    fit_spread: &str,
    leg_center: f32,
    leg_spread: f32,
    direction: crate::utils::fitness::Direction,
    family: crate::engine::config::MeanType,
) -> String {
    let (sep, leg, spread_noun, _spread_short, note) = window_legs(family);
    // No net read twice ⇒ no observable spread: `na`, never a `0.0000` that reads as steadiness
    // (the same rule the arithmetic pair applies).
    let spread = if leg_spread.is_finite() {
        fmt_line(leg_spread)
    } else {
        crate::utils::cfy::NO_VALUE.to_string()
    };
    format!(
        "score{} avg {} ± {} (σ over nets) {} {} {} │ {} {}{}",
        direction.arrow(),
        fmt_line(fit_m),
        fit_spread,
        sep,
        leg,
        fmt_line(leg_center),
        spread_noun,
        spread,
        note,
    )
}

/// The `Minimal` frame's score cell: the population's mean score with the arithmetic that produced
/// it (`score = the window centre`).
fn minimal_score_cell(
    fit_m: f32,
    fit_delta: &str,
    direction: crate::utils::fitness::Direction,
) -> String {
    // The score is the window centre: there is no charge to spell out, so the cell is the number plus its delta (the `Minimal` frame has no room for the diagnostic legs, and…
    format!(
        "score{} {}{}",
        direction.arrow(),
        fmt_line(fit_m),
        fit_delta
    )
}

#[cfg(test)]
mod tests {
    use super::lineage::*;
    use super::{
        CardStats, NetShape, StopPlan, delta_crown, delta_pop, mean_of, minimal_score_cell,
        origin_cell, step_score_cell,
    };
    use crate::utils::fitness::Direction;

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

    /// A for the `Summ` line: the legs it prints are the FAMILY pair the score came from, so the
    /// row stays readable — but no charge is printed, because the score IS the centre.
    #[test]
    fn the_step_score_cell_prints_the_legs_that_made_the_score() {
        // Arithmetic: the centre and its σ, with no `λ × spread` charge between them.
        assert_eq!(
            step_score_cell(
                17.4,
                "18.9",
                18.8,
                1.05,
                Direction::Maximize,
                crate::engine::config::MeanType::Arithmetic
            ),
            "score↑ avg 17.4 ± 18.9 (σ over nets) = window avg 18.8 │ window spread 1.05"
        );
        // Direction no longer changes the arithmetic: there is no charge to sign.
        let minimized = step_score_cell(
            17.4,
            "18.9",
            18.8,
            1.05,
            Direction::Minimize,
            crate::engine::config::MeanType::Arithmetic,
        );
        assert!(minimized.contains("= window avg 18.8 │"), "{minimized}");
        assert!(!minimized.contains('λ'), "no charge is named: {minimized}");
        // No net read twice (`leg_spread` = NaN): `na`, never a confident `0.0000`.
        let once = step_score_cell(
            17.4,
            crate::utils::cfy::NO_VALUE,
            18.8,
            f32::NAN,
            Direction::Maximize,
            crate::engine::config::MeanType::Arithmetic,
        );
        assert!(once.contains("window spread na"), "{once}");
        assert!(once.ends_with("│ window spread na"), "{once}");
    }

    /// A median run ranks on the median: the cell must print THAT leg, not the arithmetic pair,
    /// and the spread it prints beside it must be the MAD (diagnostic, never charged).
    #[test]
    fn the_median_step_cell_closes_on_the_median_not_the_mean() {
        use crate::engine::clock::smoothing::{RollingBuffer, rolling_stats_with};
        use crate::engine::config::MeanType;
        // One net, four readings: mean 25.75 / σ 14.77, but median 12 and MAD 2.
        let stats = rolling_stats_with(
            &RollingBuffer::from_values(4, &[10.0, 12.0, 40.0, 41.0]),
            MeanType::Median,
        )
        .unwrap();
        let fit = stats.score();
        assert!((fit - 12.0).abs() < 1e-3, "F is the centre: {fit}");
        let cell = step_score_cell(
            fit,
            crate::utils::cfy::NO_VALUE,
            stats.fam_center,
            stats.fam_spread,
            Direction::Maximize,
            MeanType::Median,
        );
        assert!(
            cell.contains("= window median 12.0 │ 1.4826·MAD 2.97"),
            "{cell}"
        );
        assert!(!cell.contains('λ'), "no charge is named: {cell}");
        // The leg the cell prints is the score itself, and the MAD is the diagnostic beside it.
        assert!((stats.fam_center - fit).abs() < 1e-6, "{cell}");
        assert!(
            (stats.mean - fit).abs() > 9.0,
            "the arithmetic mean cannot be the printed centre"
        );
    }

    /// `signed_log` averages in `g`, where `g⁻¹` is nonlinear: the cell reads `≈` and names `g`
    /// rather than claiming an identity it cannot have.
    #[test]
    fn the_signed_log_step_cell_names_the_space_it_averaged_in() {
        let cell = step_score_cell(
            41.2,
            "18.9",
            3.12,
            0.41,
            Direction::Maximize,
            crate::engine::config::MeanType::SignedLog,
        );
        assert!(cell.contains("≈ window g-avg 3.12"), "{cell}");
        assert!(cell.contains("│ window g-spread 0.4100"), "{cell}");
        assert!(cell.ends_with("· F = g⁻¹(legs)"), "{cell}");
        assert!(!cell.contains('λ'), "no charge is named: {cell}");
    }

    #[test]
    fn minimal_score_cell_prints_the_centre_it_ranked() {
        // The Minimal frame has no room for the diagnostic legs, and there is no charge left to
        // spell out: the cell is the score itself plus its delta.
        let cell = minimal_score_cell(37.4, "", Direction::Maximize);
        assert_eq!(cell, "score↑ 37.4");
        // A step-over-step delta rides the score it belongs.
        assert_eq!(cell_with_delta(), "score↑ 37.4 (∆+4.50)");
        assert_eq!(
            minimal_score_cell(37.4, "", Direction::Minimize),
            "score↓ 37.4",
            "the arrow is the only direction-dependent part left"
        );
    }

    /// The delta case for the test above, kept separate so the primary assertion reads as one
    /// golden string.
    fn cell_with_delta() -> String {
        minimal_score_cell(37.4, " (∆+4.50)", Direction::Maximize)
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
            vec![
                "crossover-lowest-fitness",
                "250",
                "62",
                "171",
                "65",
                "7",
                crate::utils::cfy::NO_VALUE
            ]
        );
        assert_eq!(
            rows[1],
            vec![
                "mutation",
                "250",
                "49",
                crate::utils::cfy::NO_VALUE,
                "99",
                crate::utils::cfy::NO_VALUE,
                "0"
            ]
        );
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
            reach: 1.0,
            depth: 1,
            thickness: 1.0,
            split: None,
            genome: [(token.to_string(), 1usize)].into_iter().collect(),
            nodes: 1,
            wires: 1,
            input_wires: 1,
            dim: 4.0,
            mask: Some(1.0),
        }
    }

    #[test]
    fn split_cell_carries_both_the_policy_and_the_share() {
        use crate::graph::topology::FeatureSplitPolicy;
        let mut shape = shape("W:Tanh");
        shape.input_wires = 4;
        shape.split = Some(FeatureSplitPolicy::Random);
        shape.mask = Some(0.75);
        assert_eq!(shape.split_cell().as_deref(), Some("Random · 4 · 75.0"));
        shape.split = Some(FeatureSplitPolicy::Crossover);
        assert_eq!(shape.split_cell().as_deref(), Some("Crossover · 4 · 75.0"));
        shape.split = None;
        shape.mask = Some(1.0);
        assert_eq!(shape.split_cell().as_deref(), Some("Full · 4 · 100.0"));
    }

    /// A shape with two gene tokens, ONE of them `W:Tanh` — so its Δ to [`shape`]
    /// lands on 0.5 (one shared gene out of two) rather than 0 or 1.
    fn shape_pair() -> NetShape {
        NetShape {
            reach: 1.0,
            depth: 2,
            thickness: 1.5,
            split: None,
            genome: [("W:Tanh".to_string(), 1usize), ("W:SiLU".to_string(), 1)]
                .into_iter()
                .collect(),
            nodes: 1,
            wires: 2,
            input_wires: 2,
            dim: 8.0,
            mask: Some(0.5),
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
        // An unreadable blueprint prints `na`, never a number.
        assert_eq!(delta_crown(None, &plan, slot("a")), None);
        // `k = 1`: a champion has no OTHER champion to be measured against ⇒ `na`.
        let solo = StopPlan {
            header: Vec::new(),

            columns: Vec::new(),
            has_exam: false,
            dealt_clock: 7,
            crown: vec![shape("W:Tanh")],
            crown_idx: crown_idx_of(&["a"]),
            population: Vec::new(),
        };
        assert_eq!(delta_crown(Some(&me), &solo, solo.crown_idx["a"]), None);
    }

    /// `Δpop` is the crowd, minus the row itself — and `na` (never `0.00`) when there
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
        // A lone net has no other live net to be unlike ⇒ `na`.
        let lone = StopPlan {
            population: vec![("a".to_string(), shape("W:Tanh"))],
            ..plan
        };
        assert_eq!(delta_pop(Some(&me), "a", &lone), None);
    }

    /// The card's own means: `na` legs are unreadable blueprints, not zeros,
    /// empty card says `na` rather than inventing a `0.00`.
    #[test]
    fn card_stats_average_only_the_rows_that_have_a_number() {
        assert_eq!(mean_of(&[Some(1.0), None, Some(0.5)]), Some(0.75));
        assert_eq!(mean_of(&[None, None]), None, "nothing to average");
        assert_eq!(mean_of(&[]), None);
        let stats = CardStats {
            delta_crown: Some(0.5),
            delta_pop: None,
        };
        assert_eq!(stats.line(), "card mean: Δcrown 0.50 · Δpop na");
    }
}
