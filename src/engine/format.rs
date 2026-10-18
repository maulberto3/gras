//! Console + CSV formatting helpers for the engine.
//!
//! Owns the display-only float formatting, RFC4180 CSV quoting, and the
//! one-line build identity stamp. Nothing here is persisted through the
//! rounded formatters — artifacts always serialize the raw `f32`.
//!
//! **One rule for every measured number — its own magnitude picks the
//! precision** ([`fmt_band`], shared by [`fmt_line`] and [`fmt_cell`]):
//!
//! | value | decimals | example |
//! |---|---|---|
//! | `\|v\| > 10` | 1 | `117.6` |
//! | `1 ≤ \|v\| ≤ 10` | 2 | `3.25` |
//! | `\|v\| < 1` | 4 | `0.6348` |
//!
//! The sink no longer changes the precision: a `0.6348` accuracy and a `117.6`
//! score print the same in a log line and in a table cell, so a reader never has
//! to learn which writer produced the digits. Knob echoes stay flat 2dp
//! ([`fmt_cfg`]) — a setting is not a measurement.
//!
//! Widening is not decoration: at 4dp a `0.00004` still prints as `0.0000`, and
//! a `train_loss` that reads as zero is worse than a wider column. The floor is
//! capped, so a value small enough to vanish even at the cap prints honestly.

/// One-line description of the RUNNING binary: crate version, profile, and the
/// exe's build mtime — so "stale process, not stale logic" is one glance.
///
/// The exe PATH is deliberately NOT printed: it is long enough to dominate the
/// manifest row it sits in, and the mtime already answers the staleness
/// question. The stamp is recorded in `engine.json` too, but never compared on
/// resume (the gate is the trainer blob), so its shape is free to change.
pub(crate) fn build_stamp() -> String {
    let built = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| {
            let secs = d.as_secs();
            // Compact, timezone-free UTC stamp (YYYY-MM-DD HH:MM:SS).
            let (days, rem) = (secs / 86_400, secs % 86_400);
            let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
            let (y, mo, d) = civil_from_days(days as i64);
            format!("{y:04}-{mo:02}-{d:02} {h:02}:{m:02}:{s:02}Z")
        })
        .unwrap_or_else(|| "unknown".into());
    format!(
        "gras {} [{}] built {built}",
        env!("CARGO_PKG_VERSION"),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    )
}

/// Days-since-epoch → (year, month, day), proleptic Gregorian (Howard Hinnant's
/// `civil_from_days`). Self-contained so the engine needs no date dependency.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ── display-only float formatting ──────────────────────────────────────────

/// Above this magnitude decimal notation stops paying for itself: `{:.1}` on
/// `1.5e38` emits 41 characters, and on `1.23e9` it exposes f32 drift the
/// shortest-round-trip `{}` hides. Measured: decimal caps at 11 chars by 1e8,
/// scientific holds at 7-8 chars at any scale.
const SCI_AT: f32 = 1.0e9;

/// The band boundary between 1dp and 2dp: values from here up print 1 decimal.
/// The boundary value itself belongs to the wide band, so `10.0` prints `10.0`
/// and not `10.00` — a column of tens never sprouts a second decimal.
const BAND_WIDE: f32 = 10.0;
/// The band boundary between 2dp and 4dp: values below this print 4 decimals.
const BAND_SMALL: f32 = 1.0;

/// The shared core: at least `min` decimals, widening up to `max` while the
/// result still READS as zero for a non-zero value.
///
/// Testing the rendered string by *parsing it back* (rather than comparing it
/// to `"0.0"`) is what makes every signed-zero spelling collapse correctly —
/// `-0.0`, `-0.00` and friends all parse to a value equal to `0.0`. Non-finite
/// values need no special case: `"NaN".parse()` is `Ok(NaN)`, which is never
/// equal to `0.0`, so the loop cannot fire on them.
fn fmt_span(v: f32, min: usize, max: usize) -> String {
    if v.is_finite() && v.abs() >= SCI_AT {
        return format!("{v:.3e}");
    }
    let mut s = format!("{v:.min$}");
    let mut p = min;
    while p < max && v != 0.0 && s.parse::<f32>() == Ok(0.0) {
        p += 1;
        s = format!("{v:.p$}");
    }
    s
}

/// **The scalar display rule**: the number's own magnitude picks its precision.
///
/// `|v| > 10` → 1 decimal, `1 ≤ |v| ≤ 10` → 2, `|v| < 1` → 4. Every measured
/// value in the output goes through here, so the same quantity is spelled the
/// same way wherever it appears, and the reader can compare two numbers by
/// their digit count alone. Above the decimal bands the scientific threshold
/// still wins ([`SCI_AT`]).
///
/// The widening floor survives at every band: a real value never prints as
/// `0`, which is what keeps a small `train_loss` from reading as exactly zero.
fn fmt_band(v: f32) -> String {
    if v.is_finite() && v.abs() >= SCI_AT {
        return format!("{v:.3e}");
    }
    let a = v.abs();
    if a >= BAND_WIDE {
        fmt_span(v, 1, 2)
    } else if a >= BAND_SMALL {
        fmt_span(v, 2, 4)
    } else {
        fmt_span(v, 4, 6)
    }
}

/// A console log LINE. Same band rule as a table cell — see [`fmt_band`].
pub(crate) fn fmt_line(v: f32) -> String {
    fmt_band(v)
}

/// A CFY TABLE cell. Same band rule as a log line — see [`fmt_band`].
pub(crate) fn fmt_cell(v: f32) -> String {
    fmt_band(v)
}

/// A config / knob echo: flat 2 decimals. These are settings rather than
/// measurements, so a fourth decimal would claim precision that isn't there
/// (`λ=0.7000` says nothing `λ=0.70` doesn't).
pub(crate) fn fmt_cfg(v: f32) -> String {
    format!("{v:.2}")
}

/// Value-unit tag for a ranking score in a TABLE cell: `score` = the
/// risk-adjusted value the race decides on. The CSV column keeps `smt`.
pub(crate) fn fmt_cell_smt(v: f32) -> String {
    format!("score {}", fmt_cell(v))
}

/// Optional float, log-line form — plain `—` when unset (never `Some(...)` in
/// user-facing logs).
pub(crate) fn fmt_line_opt(v: &Option<f32>) -> String {
    v.as_ref().map_or_else(|| "—".into(), |x| fmt_line(*x))
}

/// Optional float, table-cell form — plain `—` when unset.
pub(crate) fn fmt_cell_opt(v: &Option<f32>) -> String {
    v.as_ref().map_or_else(|| "—".into(), |x| fmt_cell(*x))
}

/// Which sink a reading is rendered for — a console line and a table cell share
/// one rule (the value's magnitude, [`fmt_band`]); only a document header still
/// differs, at a flat 2dp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadingStyle {
    /// A console/log LINE ([`fmt_line`]).
    Line,
    /// A CFY TABLE cell ([`fmt_cell`] — the same band rule as [`ReadingStyle::Line`]).
    Cell,
    /// A markdown/document header ([`fmt_cfg`] precision, flat 2dp).
    Doc,
}

/// The scalar rule for a style — the one thing [`fmt_reading`] and
/// [`fmt_census`] let vary; the SYMBOLS never vary.
fn scalar(style: ReadingStyle) -> fn(f32) -> String {
    match style {
        ReadingStyle::Line => fmt_line,
        ReadingStyle::Cell => fmt_cell,
        ReadingStyle::Doc => fmt_cfg,
    }
}

/// **The ONE way a sampled mean is spelled** —
/// `avg 327.8 (spread 107.8; SE 34.1 = spread/√10, n=10)`.
///
/// The two quantities answer different questions and are therefore ALWAYS both
/// printed, each labelled with a word rather than a Greek letter:
/// - `spread` is the spread of the samples behind the mean — how variable the
///   thing being measured is. This is what the ranking charges
///   (`score = avg − λ·spread`).
/// - `SE = spread/√n` is the uncertainty OF that mean — how well we know it.
///   This is what a comparison between two means has to use (the guardrail's
///   band), and the equation is printed so the reader can check the division.
///
/// `n` rides along because it is what turns one into the other. A bare `±` used
/// to mean either of the two depending on the line, which is exactly the
/// confusion this function removes — callers should not hand-roll one.
///
/// A population CENSUS (every net, not a sample) is the one sanctioned shorter
/// form — [`fmt_census`]: `avg … (spread …)` with no `SE`, since there is no
/// sampling error to state.
pub fn fmt_reading(
    mean: Option<f32>,
    sigma: Option<f32>,
    n: usize,
    style: ReadingStyle,
) -> String {
    let f = scalar(style);
    let Some(mean) = mean else {
        return "—".to_string();
    };
    match sigma {
        Some(s) if n > 0 => format!(
            "avg {} (spread {}; SE {} = spread/√{n}, n={n})",
            f(mean),
            f(s),
            f(s / (n as f32).sqrt())
        ),
            // No observable spread yet (one reading) or none recorded: say so rather
            // than print `spread 0`, which would read as "perfectly steady".
        _ => format!("avg {} (n={n}, no spread yet)", f(mean)),
    }
}

/// **The census spelling**: `avg … (spread …)` — a mean over the whole
/// population of nets, not a sample of readings.
///
/// [`fmt_reading`] states `SE = spread/√n` because a sample's mean is itself an
/// estimate; a census *is* the population, so there is no sampling error to
/// state and printing one would invent a claim. `spread` is the population's
/// own spread and names itself the same way it does in `fmt_reading` — that
/// shared word is the whole point: one meaning, spread of what is being
/// averaged, everywhere in the output.
pub fn fmt_census(mean: f32, sigma: f32, style: ReadingStyle) -> String {
    let f = scalar(style);
    format!("avg {} (spread {})", f(mean), f(sigma))
}

/// RFC4180-quote a CSV field holding a delimiter, quote, or newline — lineage
/// strings contain commas, so this is not optional.
pub(crate) fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ReadingStyle, csv_field, fmt_cell, fmt_cell_smt, fmt_census, fmt_cfg, fmt_line,
        fmt_line_opt, fmt_reading,
    };

    // ── the band rule: magnitude picks the precision ──────────────────────

    #[test]
    fn band_above_ten_is_one_decimal() {
        assert_eq!(fmt_line(117.55), "117.6");
        assert_eq!(fmt_line(97.1), "97.1");
        assert_eq!(fmt_line(20.5), "20.5");
        assert_eq!(fmt_line(214.6), "214.6");
        // f32 is exact here, and `{:.1}` rounds half to even: 42.25 -> 42.2.
        assert_eq!(fmt_line(-42.25), "-42.2");
        assert_eq!(fmt_line(999_999.0), "999999.0");
        // The boundary value belongs to the wide band.
        assert_eq!(fmt_line(10.0), "10.0");
    }

    #[test]
    fn band_one_up_to_ten_is_two_decimals() {
        assert_eq!(fmt_line(1.0), "1.00");
        assert_eq!(fmt_line(3.25), "3.25");
        assert_eq!(fmt_line(9.99), "9.99");
        assert_eq!(fmt_line(-5.5), "-5.50");
    }

    #[test]
    fn band_below_one_is_four_decimals() {
        // The case the band exists for: a sub-1 reading keeps its precision
        // (1dp would print this accuracy as `0.6`).
        assert_eq!(fmt_line(0.6348), "0.6348");
        assert_eq!(fmt_line(0.02), "0.0200");
        assert_eq!(fmt_line(-0.02), "-0.0200");
        assert_eq!(fmt_line(0.5), "0.5000");
    }

    #[test]
    fn band_floor_is_capped_and_true_zero_is_not_widened() {
        assert_eq!(fmt_line(0.0), "0.0000");
        assert_eq!(fmt_line(-0.0), "-0.0000");
        // Too small to show even at the 6dp cap: prints honestly, not "0.0000".
        assert_eq!(fmt_line(0.000_04), "0.00004");
        assert_eq!(fmt_line(1e-8), "0.000000");
    }

    #[test]
    fn band_uses_scientific_notation_only_at_scale() {
        assert_eq!(fmt_line(1e9), "1.000e9");
        assert_eq!(fmt_line(1.5e38), "1.500e38");
        assert_eq!(fmt_line(-1.5e38), "-1.500e38");
        // Just under the threshold stays decimal. Note the boundary is exact:
        // `f32` cannot represent 999_999_999.0 (it rounds up to 1e9), so the
        // largest value that still prints in decimal is 990_000_000.0.
        assert_eq!(fmt_line(990_000_000.0), "990000000.0");
        assert_eq!(fmt_line(999_999_999.0), "1.000e9", "f32 rounds this up to 1e9");
    }

    #[test]
    fn non_finite_passes_through_without_widening() {
        // "NaN".parse() is Ok(NaN) which is != 0.0, so the widen loop cannot fire.
        assert_eq!(fmt_line(f32::NAN), "NaN");
        assert_eq!(fmt_line(f32::INFINITY), "inf");
        assert_eq!(fmt_line(f32::NEG_INFINITY), "-inf");
    }

    // ── the same rule in a table cell ─────────────────────────────────────

    #[test]
    fn cell_and_line_agree_by_construction() {
        for v in [117.55, 97.1, 3.25, 1.0, 0.6348, 0.5, 0.000_04, 1e-8, 0.0, -42.25] {
            assert_eq!(fmt_cell(v), fmt_line(v), "{v} spelled differently in a cell");
        }
    }

    #[test]
    fn cell_widens_below_its_floor_like_the_line_rule() {
        assert_eq!(fmt_cell(0.5), "0.5000");
        assert_eq!(fmt_cell(0.000_04), "0.00004");
        assert_eq!(fmt_cell(1e-8), "0.000000");
    }

    #[test]
    fn cell_uses_scientific_notation_at_scale() {
        assert_eq!(fmt_cell(1e9), "1.000e9");
        assert_eq!(fmt_cell(1.5e38), "1.500e38");
    }

    // ── the knob rule: flat 2dp ────────────────────────────────────────────

    #[test]
    fn cfg_is_flat_two_decimals() {
        assert_eq!(fmt_cfg(0.7), "0.70");
        assert_eq!(fmt_cfg(0.0), "0.00");
        assert_eq!(fmt_cfg(1.0), "1.00");
    }

    // ── tagged + optional forms ───────────────────────────────────────────

    #[test]
    fn score_cell_carries_the_unit_tag() {
        assert_eq!(fmt_cell_smt(117.55), "score 117.6");
    }


    #[test]
    fn optional_forms_render_unset_as_dash() {
        assert_eq!(fmt_line_opt(&None), "—");
        assert_eq!(fmt_line_opt(&Some(0.02)), "0.0200");
    }

    // ── the one reading spelling: spread and SE, always labelled ──────────

    #[test]
    fn reading_states_spread_and_standard_error_together() {
        // n = 4 so SE = spread/2 exactly — no rounding ambiguity in the
        // assertion. Each number takes its OWN band: spread is 10.0 (1dp)
        // while SE = 5.0 lands in the 1..10 band (2dp). Rule working, not a slip.
        assert_eq!(
            fmt_reading(Some(100.0), Some(10.0), 4, ReadingStyle::Line),
            "avg 100.0 (spread 10.0; SE 5.00 = spread/√4, n=4)"
        );
        // A cell now spells a reading exactly as a line does.
        assert_eq!(
            fmt_reading(Some(100.0), Some(10.0), 4, ReadingStyle::Cell),
            "avg 100.0 (spread 10.0; SE 5.00 = spread/√4, n=4)"
        );
        assert_eq!(
            fmt_reading(Some(100.0), Some(10.0), 4, ReadingStyle::Doc),
            "avg 100.00 (spread 10.00; SE 5.00 = spread/√4, n=4)"
        );
    }

    #[test]
    fn reading_does_not_invent_a_spread_for_one_sample() {
        assert_eq!(
            fmt_reading(Some(327.8), None, 1, ReadingStyle::Line),
            "avg 327.8 (n=1, no spread yet)"
        );
        assert_eq!(fmt_reading(None, None, 0, ReadingStyle::Line), "—");
    }

    #[test]
    fn census_omits_se_because_a_population_has_none() {
        assert_eq!(
            fmt_census(100.0, 10.0, ReadingStyle::Line),
            "avg 100.0 (spread 10.0)"
        );
        assert_eq!(
            fmt_census(100.0, 10.0, ReadingStyle::Doc),
            "avg 100.00 (spread 10.00)"
        );
    }

    // ── untouched neighbours ───────────────────────────────────────────────

    #[test]
    fn csv_field_quotes_only_when_needed() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    }
}
