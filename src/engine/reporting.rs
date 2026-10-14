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

        /// `Minimal` mode: one boxed vitals table per step (step 2+), redrawn in place on a
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
            // Delta vs the previous step's mean; omitted when 0 or when it dies at
            // 2-decimal rounding (the first render sets the baseline).
        let delta = |cur: f32, prev: Option<f32>| match prev {
            Some(p) if (cur - p).abs() >= 5e-3 => format!(" (∆{:+.2})", cur - p),
            _ => String::new(),
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
                    self.step_rl.label(step_secs),
                )
            },
                // `smt` tag: this cell is the population MEAN of smoothed per-net fitness,
                // never confused with the per-net raw values.
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
        info!(
            "  wrote {} frontier snapshot(s) → nets/<hash>.json (one per live net)",
            live.len(),
        );
            // Print the pop size the caller must pass (the engine refuses a mismatch) —
            // phrased in ENGINE terms, not CLI flags.
        info!(
            "  {} live net(s) at step {} → resume from {} with pop_size {}",
            live.len(),
            clock,
            self.run_dir.display(),
            live.len(),
        );
            // Parity is a property of the CONFIG: the engine re-reads the replay-relevant
            // knobs and REFUSES a mismatch, so the resume call has to agree.
        info!(
            "  resume parity: replay-relevant knobs come from engine.json (K={}, checkpoint_every={}, probation windows) — a mismatch is refused, not patched",
            self.config.smoothing_window,
            self.config.checkpoint_every,
        );

            // What the run DID comes before what it produced: births, deaths, and what
            // the roll budget bought. Read from the engine's own persisted counters.
        info!("  ── run accounting ──");
        for line in accounting_lines(&self.accounting, live.len()) {
            info!("{line}");
        }

            // Final elite report: the top-k nets by smoothed fitness — the run's
            // champions. Elite = rank, not identity.
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
        let has_eval = ranked.iter().take(k).any(|(_, _, _, e)| e.is_some());
            // Every column recycles state the engine already carries (`spread`, `trained`,
            // `frozen`, each informative metric labeled as in `history.csv`).
        let mut header = vec![
            "#", "hash", "smt", "spread", "step", "seat age", "trained", "frozen", "params",
            "origin",
        ];
        if has_eval {
            header.push("eval");
        }
        for m in &self.metrics {
            header.push(m.label());
        }
        table.set_header(header);
        for (rank, (h, fit, step, eval_loss)) in ranked.iter().take(k).enumerate() {
                // ± std over the smoothing window (same framing as the guardrail): a wide ±
                // means the ordering is luck-sensitive. eval_loss `—` when not applicable.
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
            let born_num = state.map(|ns| ns.entered_at_step).unwrap_or(0);
            let origin_label = origin_cell(
                state.as_ref().and_then(|s| s.created_from.as_deref()),
                born_num,
            );
            let params_label = state
                .and_then(|s| s.meta.params)
                .map(|p| p.to_string())
                .unwrap_or_else(|| "?".to_string());
                // The raw spread behind `smt`: two nets at the same smoothed value can be
                // "always that good" or "one lucky step".
            let spread = self
                .rolling_fitness
                .get(h.as_str())
                .filter(|b| b.len() > 1)
                .map(|b| {
                    let n = b.len() as f32;
                    let mean = b.iter().sum::<f32>() / n;
                    let sd = (b.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt();
                    format!("{sd:.1}")
                })
                .unwrap_or_else(|| "—".to_string());
                // Clocks lived INCLUSIVE of the birth clock and the last run — same
                // convention the freeze lines use (`clock − entered_at_step`).
            let age = dealt_clock.saturating_sub(born_num) + 1;
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
                spread,
                step.to_string(),
                seat,
                trained,
                frozen,
                params_label,
                origin_label,
            ];
            if has_eval {
                row.push(
                    eval_loss
                        .map(|l| format!("{l:.4}"))
                        .unwrap_or_else(|| "—".to_string()),
                );
            }
            // One cell per configured informative metric, in config order —
            // the same order `history.csv` writes its columns in.
            let metrics = state.and_then(|ns| ns.last_metrics.as_ref());
            for i in 0..self.metrics.len() {
                row.push(
                    metrics
                        .and_then(|m| m.informative.get(i).copied())
                        .map(|v| format!("{v:.2}"))
                        .unwrap_or_else(|| "—".to_string()),
                );
            }
            table.add_row(row);
        }
        // comfy-table computes the widths; indent each line so the block
        // sits inside the summary's two-space margin.
        for line in table.to_string().lines() {
            info!("  {line}");
        }
            // One line: the header names the list, then the gate tally follows inline,
            // separated by ` · ` (it was the longest of the twelve fields).
        info!(
            "  ── final elites (top-{k}, fitness{arrow} smoothed) ──{}",
            detail_inline(&format!(
                "crown {}/{} seats · {} eligible · {} on probation (k={}) · {} age-blocked (j={})",
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
        );
        Ok(())
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

/// The run-accounting block for the stop summary: what the evolution did, one labeled
/// line per fact (born → died → what the rolls bought). `alive` is passed in at stop.
pub(crate) fn accounting_lines(a: &crate::state::RunAccounting, alive: usize) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!(
        "  born {} = {} founders ({} random + {} inherited) + {} immigrants + {} crossover children",
        a.born,
        a.founders + a.inherited_founders,
        a.founders,
        a.inherited_founders,
        a.immigrants,
        a.crossover_children
    ));
    lines.push(format!(
        "  culled {} ({} crossover slot(s), {} immigrant slot(s)) │ {} alive at stop",
        a.culls, a.culls_crossover, a.culls_immigrant, alive
    ));
    lines.push(format!(
        "  crossover rolls {}: {} fired → {} attempt(s), {} child(ren) built, {} discarded (no viable child)",
        a.crossover_rolls, a.crossover_fired, a.crossover_attempts, a.crossover_children,
        a.crossover_discarded
    ));
    lines.push(format!(
        "  mutation rolls {}: {} fired → {} immigrant(s), {} blocked by the cull shield",
        a.mutation_rolls, a.mutation_fired, a.immigrants, a.mutation_blocked
    ));
    if a.deferred_clocks > 0 {
        lines.push(format!(
            "  evolution deferred on {} clock(s): no cull-eligible net (every non-elite net was fresh or elite)",
            a.deferred_clocks
        ));
    }
    lines
}

/// Flatten a trainer's `describe()` blob into one readable line
/// (`key=value │ nested.key=value`). Bounded at `MAX_FIELDS` with an explicit `… +n more`.
pub(crate) fn summarize_json(v: &serde_json::Value) -> String {
    /// Max facts rendered before an explicit `… +n more` marker.
    const MAX_FIELDS: usize = 12;

    /// Flatten a nested JSON object into `key=value` fragments.
    fn flat(out: &mut Vec<String>, prefix: &str, v: &serde_json::Value) {
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
            serde_json::Value::String(s) => out.push(format!("{}={s}", prefix)),
            serde_json::Value::Number(n) => out.push(format!("{}={n}", prefix)),
            serde_json::Value::Bool(b) => out.push(format!("{}={b}", prefix)),
            // Arrays: the elements are already short scalars in practice
            // (`["linear", "relu"]`), so one comma-joined cell beats N lines.
            serde_json::Value::Array(items) => {
                let cells: Vec<String> = items.iter().map(|x| x.to_string()).collect();
                out.push(format!("{}=[{}]", prefix, cells.join(",")));
            }
            // Null carries no information; `{}` (an empty object) flattens to
            // nothing, which is how an empty `describe()` reads as "no line".
            serde_json::Value::Null => {}
        }
    }

    let mut parts = Vec::new();
    flat(&mut parts, "", v);
    if parts.len() > MAX_FIELDS {
        let extra = parts.len() - MAX_FIELDS;
        parts.truncate(MAX_FIELDS);
        parts.push(format!("… +{extra} more"));
    }
    parts.join(" │ ")
}

/// One ★-badge entry: the elite net's short hash plus its seat state. Seated ⇒ tenure
/// over the crown lifetime `j` (`j = 0` disarms); not seated ⇒ `no seat`.
pub(crate) fn elite_badge_entry(short: &str, seat_age: usize, j: usize) -> String {
    if seat_age == 0 {
        format!("{short}(no seat)")
    } else if j == 0 {
        format!("{short}(tenure {seat_age})")
    } else {
        format!("{short}(tenure {seat_age}/{j})")
    }
}

/// `trained n/m` for the freeze lines: real optimizer updates over the clocks the net
/// has LIVED (`entered_at_step` → now) — not the old absolute-clock delta.
pub(crate) fn trained_note(trainings: usize, age: usize) -> String {
    format!("trained {trainings}/{age} clocks")
}

/// Append an event's detail on the SAME line, separated by ` · ` so an event is always
/// one log line. Empty detail returns "" (no dangling separator).
pub(crate) fn detail_inline(detail: &str) -> String {
    if detail.is_empty() {
        String::new()
    } else {
        format!(" · {detail}")
    }
}

/// The elite gate in words for the freeze lines: `probation` is the window a fresh net
/// must serve before a seat; `tenure` is the seat-age countdown against `j`.
pub(crate) fn elite_gate_note(probation_age: usize, seat_age: usize, k: usize, j: usize) -> String {
    let probation = if probation_age >= k {
        format!("probation served ({probation_age}≥{k})")
    } else {
        format!("probation {probation_age}/{k}")
    };
    let tenure = if j == 0 {
        format!("tenure {seat_age}")
    } else {
        format!("tenure {seat_age}/{j}")
    };
    format!(", {probation}, {tenure}")
}

#[cfg(test)]
mod tests {
    use super::{detail_inline, elite_badge_entry, elite_gate_note, origin_cell, trained_note};

    #[test]
    fn elite_badge_entry_says_whether_the_net_holds_a_seat() {
        // Seated: tenure out of the armed lifetime.
        assert_eq!(elite_badge_entry("abc12345", 3, 5), "abc12345(tenure 3/5)");
        // Elite but NOT seated (barred by the gate, or aged out) — the case
        // that used to print `(0/5)`, indistinguishable from a fresh seat.
        assert_eq!(elite_badge_entry("abc12345", 0, 5), "abc12345(no seat)");
        // Lifetime disarmed: no denominator invented.
        assert_eq!(elite_badge_entry("abc12345", 4, 0), "abc12345(tenure 4)");
    }
    #[test]
    fn trained_note_is_trainings_over_clocks_lived() {
        // A net born at step 7 that trained 5 of its 12 lived clocks.
        assert_eq!(trained_note(5, 12), "trained 5/12 clocks");
            // The two counters can only agree when nothing was frozen, and
            // `trainings ≤ age` always holds.
        assert_eq!(trained_note(0, 0), "trained 0/0 clocks");
    }

    #[test]
    fn detail_inline_keeps_an_event_on_one_line() {
        // The seat line's bookkeeping: inline joins mid-line, never a wrap.
        let tail = detail_inline(
            "age 5 · trained 5/5 clocks · 10850 params, probation served (5≥5), tenure 0/5",
        );
        assert_eq!(
            tail,
            " · age 5 · trained 5/5 clocks · 10850 params, probation served (5≥5), tenure 0/5"
        );
        assert!(!tail.contains('\n'), "the seat line stays one log line: {tail}");
        // Empty detail adds nothing — no dangling separator.
        assert_eq!(detail_inline(""), "");
    }

    #[test]
    fn elite_gate_note_never_prints_a_bar_that_was_overshot() {
        // Still serving: numerator < k, plain fraction.
        assert_eq!(elite_gate_note(3, 0, 5, 5), ", probation 3/5, tenure 0/5");
        // Served and ticking on: reported as served, never `16/5`.
        assert_eq!(
            elite_gate_note(16, 2, 5, 5),
            ", probation served (16≥5), tenure 2/5"
        );
        // Exactly at the bar is served, not "5/5 of the window remaining".
        assert_eq!(
            elite_gate_note(5, 5, 5, 5),
            ", probation served (5≥5), tenure 5/5"
        );
        // k = 0: every net is served at once (the shield is off).
        assert_eq!(elite_gate_note(0, 0, 0, 5), ", probation served (0≥0), tenure 0/5");
        // j = 0: no lifetime armed, so no denominator is invented.
        assert_eq!(elite_gate_note(7, 0, 5, 0), ", probation served (7≥5), tenure 0");
    }

    use super::{accounting_lines, summarize_json};
    use crate::state::RunAccounting;

    #[test]
    fn summarize_json_flattens_a_trainer_description() {
        let blob = serde_json::json!({
            "trainer": "cartpole",
            "lr": 0.01,
            "challenge_prob": 0.25,
            "optimizer": { "kind": "Adam", "weight_decay": 0.0 },
            "activations": ["relu", "tanh"],
            "schedules": null
        });
        let line = summarize_json(&blob);
        assert!(line.contains("trainer=cartpole"), "{line}");
        assert!(line.contains("lr=0.01"), "{line}");
        assert!(line.contains("challenge_prob=0.25"), "{line}");
        assert!(line.contains("optimizer.kind=Adam"), "nested keys flatten: {line}");
        assert!(line.contains("activations=[\"relu\",\"tanh\"]"), "{line}");
        // Nulls carry nothing and empty objects produce no line at all.
        assert!(!line.contains("schedules"), "{line}");
        assert_eq!(summarize_json(&serde_json::json!({})), "");
        assert_eq!(summarize_json(&serde_json::json!(null)), "");
        // Bounded: a dump-heavy describe() gets an explicit marker, not a
        // screenful.
        let big: serde_json::Map<String, serde_json::Value> = (0..30)
            .map(|i| (format!("k{i}"), serde_json::json!(i)))
            .collect();
        let line = summarize_json(&serde_json::Value::Object(big));
        assert!(line.contains("… +18 more"), "{line}");
    }

    #[test]
    fn accounting_lines_report_births_deaths_and_wasted_rolls() {
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
        let lines = accounting_lines(&a, 50);
        assert_eq!(
            lines[0],
            "  born 214 = 50 founders (45 random + 5 inherited) + 99 immigrants + 65 crossover children"
        );
        assert_eq!(
            lines[1],
            "  culled 164 (65 crossover slot(s), 99 immigrant slot(s)) │ 50 alive at stop"
        );
            // A roll that bought nothing is reported, not hidden: "no viable child".
        assert_eq!(
            lines[2],
            "  crossover rolls 250: 62 fired → 171 attempt(s), 65 child(ren) built, 7 discarded (no viable child)"
        );
        assert_eq!(
            lines[3],
            "  mutation rolls 250: 49 fired → 99 immigrant(s), 0 blocked by the cull shield"
        );
        assert_eq!(
            lines[4],
            "  evolution deferred on 12 clock(s): no cull-eligible net (every non-elite net was fresh or elite)"
        );
        // Nothing deferred (the usual steady-state run): that line is omitted
        // rather than printed as a zero.
        let mut b = a;
        b.deferred_clocks = 0;
        assert_eq!(accounting_lines(&b, 50).len(), 4);
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
