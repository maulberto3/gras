//! Console + CSV formatting helpers for the engine.

/// One-line description of the RUNNING binary: crate version, profile, and the exe's build mtime —
/// so "stale process, not stale logic" is one glance.
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

/// Above this magnitude decimal notation stops paying for itself: `{:.1}` on `1.5e38` emits 41
/// characters, and on `1.23e9` it exposes f32 drift the shortest-round-trip `{}` hides.
const SCI_AT: f32 = 1.0e9;

/// The band boundary between 1dp and 2dp: values from here up print 1 decimal.
const BAND_WIDE: f32 = 10.0;
/// The band boundary between 2dp and 4dp: values below this print 4 decimals.
const BAND_SMALL: f32 = 1.0;

/// The shared core: at least `min` decimals, widening up to `max` while the result still READS
/// zero for a non-zero value.
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

/// A COUNT in human units: `402`, `40.2K`, `1.5M`, `2.0B`.
pub(crate) fn fmt_units(n: i64) -> String {
    const UNITS: [&str; 3] = ["K", "M", "B"];
    let mut value = n as f64;
    let mut idx = 0usize;
    while value.abs() >= 1000.0 && idx < UNITS.len() {
        value /= 1000.0;
        idx += 1;
    }
    if idx == 0 {
        return n.to_string();
    }
    // `{:.1}` on `999.95` would print `1000.0`: take the next unit instead.
    if (value * 10.0).round().abs() >= 10_000.0 && idx < UNITS.len() {
        value /= 1000.0;
        idx += 1;
    }
    format!("{value:.1}{}", UNITS[idx - 1])
}

/// A config / knob echo: flat 2 decimals.
pub(crate) fn fmt_cfg(v: f32) -> String {
    format!("{v:.2}")
}

/// Optional float, log-line form — plain `na` when unset (never `Some(...)`
/// user-facing logs).
pub(crate) fn fmt_line_opt(v: &Option<f32>) -> String {
    v.as_ref()
        .map_or_else(|| crate::utils::cfy::NO_VALUE.into(), |x| fmt_line(*x))
}

/// Which sink a reading is rendered for — a console line and a table cell share one rule for
/// a value's magnitude (`fmt_band`); only a document header still differs, at a flat 2dp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadingStyle {
    /// A console/log LINE (`fmt_line`).
    Line,
    /// A CFY TABLE cell (`fmt_cell` — the same band rule as [`ReadingStyle::Line`]).
    Cell,
    /// A markdown/document header (`fmt_cfg` precision, flat 2dp).
    Doc,
}

/// The scalar rule for a style — the one thing [`fmt_reading`]
/// [`fmt_census`] let vary; the SYMBOLS never vary.
fn scalar(style: ReadingStyle) -> fn(f32) -> String {
    match style {
        ReadingStyle::Line => fmt_line,
        ReadingStyle::Cell => fmt_cell,
        ReadingStyle::Doc => fmt_cfg,
    }
}

/// **The ONE way a sampled mean is spelled** — `avg 327.8 (spread 107.8; SE 34.1 = spread/√10,
/// n=10)`.
pub fn fmt_reading(mean: Option<f32>, sigma: Option<f32>, n: usize, style: ReadingStyle) -> String {
    let f = scalar(style);
    let Some(mean) = mean else {
        return crate::utils::cfy::NO_VALUE.to_string();
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

/// The FAMILY spelling of a sampled reading — the pair the score was actually charged from
/// (`WindowStats::fam_center` / `fam_spread`), named as that family's centre and spread.
///
/// `arithmetic` delegates to [`fmt_reading`], so its bytes are unchanged
/// (`avg 0.86 (spread 0.02; SE 0.01 = spread/√4, n=4)`). The other families have no `SE` to
/// report: `1.4826·MAD` over a handful of readings is not a standard error, and printing
/// `SE = spread/√n` beside it would invent a confidence nobody computed. A card that showed
/// `avg`/`spread` beside a median score was a header a reader could not redo — the score charges
/// `median` and its scaled MAD, not `mean` and `σ`.
pub(crate) fn fmt_family_reading(
    centre: Option<f32>,
    spread: Option<f32>,
    n: usize,
    style: ReadingStyle,
    family: crate::engine::config::MeanType,
) -> String {
    use crate::engine::config::MeanType;
    // Arithmetic is the ONLY family whose spread has a standard error, and its spelling is the
    // pre-family one, byte for byte.
    if family == MeanType::Arithmetic {
        return fmt_reading(centre, spread, n, style);
    }
    let f = scalar(style);
    let Some(centre) = centre else {
        return crate::utils::cfy::NO_VALUE.to_string();
    };
    let (centre_noun, spread_noun) = match family {
        MeanType::Arithmetic => unreachable!("delegated above"),
        // The header's spelling (`COL_FIT_MEDIAN`): `F = median`.
        MeanType::Median => ("window median", "1.4826·MAD"),
        MeanType::SignedLog => ("window g-avg", "g-spread"),
    };
    match spread {
        Some(s) if n > 0 => format!(
            "{centre_noun} {} ({spread_noun} {}, n={n})",
            f(centre),
            f(s)
        ),
        // No observable spread yet (one reading): say so, never a `0` that reads as steadiness.
        _ => format!("{centre_noun} {} (n={n}, no spread yet)", f(centre)),
    }
}

/// **The census spelling**: `avg … (spread …)` — a mean over the whole population of nets,
/// sample of readings.
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

/// Split one CSV row written by [`csv_field`]: commas inside a quoted field are field text,
/// `""` is an escaped quote (it toggles twice, so it is harmless here).
pub(crate) fn csv_split_line(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut start = 0usize;
    let mut in_quotes = false;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' => in_quotes = !in_quotes,
            b',' if !in_quotes => {
                out.push(&line[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&line[start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::{
        ReadingStyle, csv_field, fmt_cell, fmt_census, fmt_cfg, fmt_line, fmt_line_opt,
        fmt_reading, fmt_units,
    };
    use crate::engine::format::fmt_family_reading;

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
        // Just under the threshold stays decimal.
        assert_eq!(fmt_line(990_000_000.0), "990000000.0");
        assert_eq!(
            fmt_line(999_999_999.0),
            "1.000e9",
            "f32 rounds this up to 1e9"
        );
    }

    #[test]
    fn non_finite_passes_through_without_widening() {
        // "NaN".parse() is Ok(NaN) which is != 0.0, so the widen loop cannot fire.
        assert_eq!(fmt_line(f32::NAN), "NaN");
        assert_eq!(fmt_line(f32::INFINITY), "inf");
        assert_eq!(fmt_line(f32::NEG_INFINITY), "-inf");
    }

    // ── the same rule in a table cell ─────────────────────────────────────

    /// `fmt_cell` and `fmt_line` are ONE rule (`fmt_band`), so this test is the tripwire
    /// being true and the band's edges being the ones asserted above — a cell that starts spelling
    #[test]
    fn cell_shares_the_line_band_rule_at_every_edge() {
        for v in [
            117.55, 97.1, 3.25, 1.0, 0.6348, 0.5, 0.000_04, 1e-8, 0.0, -42.25,
        ] {
            assert_eq!(
                fmt_cell(v),
                fmt_line(v),
                "{v} spelled differently in a cell"
            );
        }
        // The widening floor and the scientific threshold, through the CELL form.
        assert_eq!(fmt_cell(0.5), "0.5000");
        assert_eq!(fmt_cell(0.000_04), "0.00004");
        assert_eq!(fmt_cell(1e-8), "0.000000");
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

    // ── counts in human units ─────────────────────────────────────────────

    #[test]
    fn units_suffix_thousands_and_promote_on_rounding() {
        // Hundreds stay bare: the suffix would cost a column and add nothing.
        assert_eq!(fmt_units(0), "0");
        assert_eq!(fmt_units(402), "402");
        assert_eq!(fmt_units(999), "999");
        // Thousands and up take K/M/B with one decimal.
        assert_eq!(fmt_units(1_000), "1.0K");
        assert_eq!(fmt_units(6_218), "6.2K");
        assert_eq!(fmt_units(40_242), "40.2K");
        assert_eq!(fmt_units(142_946), "142.9K");
        assert_eq!(fmt_units(1_500_000), "1.5M");
        assert_eq!(fmt_units(2_000_000_000), "2.0B");
        // Rounding must not print `1000.0K`: the unit is promoted instead.
        assert_eq!(fmt_units(999_950), "1.0M");
        assert_eq!(fmt_units(999_999_999), "1.0B");
        // Beyond B there is no unit left to promote to; the number stays honest.
        assert_eq!(fmt_units(5_000_000_000_000), "5000.0B");
        // Signed counts (a deficit is still a count) keep their sign.
        assert_eq!(fmt_units(-40_242), "-40.2K");
    }

    // ── optional forms ────────────────────────────────────────────────────

    #[test]
    fn optional_forms_render_unset_as_dash() {
        assert_eq!(fmt_line_opt(&None), crate::utils::cfy::NO_VALUE);
        assert_eq!(fmt_line_opt(&Some(0.02)), "0.0200");
    }

    // ── the one reading spelling: spread and SE, always labelled ──────────

    #[test]
    fn reading_states_spread_and_standard_error_together() {
        // n = 4 so SE = spread/2 exactly — no rounding ambiguity in the assertion.
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
        assert_eq!(
            fmt_reading(None, None, 0, ReadingStyle::Line),
            crate::utils::cfy::NO_VALUE
        );
    }

    /// A non-arithmetic run's card names ITS pair and refuses to invent an SE: the score charges
    /// the median and its scaled MAD, so a header saying `avg … (spread …; SE …)` could not be redone.
    #[test]
    fn a_family_reading_names_its_own_pair() {
        use crate::engine::config::MeanType;
        // Arithmetic stays byte-identical to the pre-family spelling.
        assert_eq!(
            fmt_family_reading(
                Some(100.0),
                Some(10.0),
                4,
                ReadingStyle::Doc,
                MeanType::Arithmetic
            ),
            fmt_reading(Some(100.0), Some(10.0), 4, ReadingStyle::Doc)
        );
        let median = fmt_family_reading(
            Some(0.84375),
            Some(0.023166),
            4,
            ReadingStyle::Doc,
            MeanType::Median,
        );
        assert_eq!(median, "window median 0.84 (1.4826·MAD 0.02, n=4)");
        assert!(
            !median.contains("SE"),
            "a scaled MAD has no standard error: {median}"
        );
        assert_eq!(
            fmt_family_reading(Some(0.5), None, 1, ReadingStyle::Doc, MeanType::SignedLog),
            "window g-avg 0.50 (n=1, no spread yet)"
        );
        assert_eq!(
            fmt_family_reading(None, None, 0, ReadingStyle::Line, MeanType::Median),
            crate::utils::cfy::NO_VALUE
        );
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
