//! Console + CSV formatting helpers for the engine.
//!
//! Owns the display-only float formatting, RFC4180 CSV quoting, and the
//! one-line build identity stamp. Nothing here is persisted through the
//! rounded formatters — artifacts always serialize the raw `f32`.
//!
//! Three display contexts, three precision rules, one core:
//!
//! | context | rule | helper |
//! |---|---|---|
//! | a console log LINE | 1 decimal, widened only when 1dp would erase the value | [`fmt_line`] |
//! | a CFY TABLE cell | 4 decimals, widened only below 5e-5 | [`fmt_cell`] |
//! | a config/knob echo | flat 2 decimals | [`fmt_cfg`] |
//!
//! Widening is not decoration: `0.02` at 1dp prints `0.0`, and a `train_loss`
//! that reads as zero is worse than a wider column. The floor is capped, so a
//! value small enough to vanish even at the cap still prints honestly.

/// One-line description of the RUNNING binary: crate version, profile, and
/// the exe path + mtime — so "stale process, not stale logic" is one glance.
pub(crate) fn build_stamp() -> String {
    let exe = std::env::current_exe().ok();
    let built = exe
        .as_ref()
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
    let path = exe
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "-".into());
    format!(
        "gras {} [{}] built {built} — exe {path}",
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

/// A console log LINE: 1 decimal, widened only when 1dp would erase the value.
///
/// `0.02` stays `0.02` rather than collapsing to `0.0`, and `0.5` stays `0.5`
/// rather than being padded to `0.50` — the widening only fires when needed.
pub(crate) fn fmt_line(v: f32) -> String {
    fmt_span(v, 1, 4)
}

/// A CFY TABLE cell: 4 decimals, widened only below `5e-5`.
///
/// Flat 4dp would render `0.00004` as `0.0000` — the same erasure the line
/// rule exists to prevent, at the cell's threshold. Normal values are exactly
/// 4 decimals; only values too small to show are given more room.
pub(crate) fn fmt_cell(v: f32) -> String {
    fmt_span(v, 4, 6)
}

/// A config / knob echo: flat 2 decimals. These are settings rather than
/// measurements, so a fourth decimal would claim precision that isn't there
/// (`λ=0.7000` says nothing `λ=0.70` doesn't).
pub(crate) fn fmt_cfg(v: f32) -> String {
    format!("{v:.2}")
}

/// Value-unit tag for a ranking score: `smt` = the risk-adjusted score the
/// race decides on. The log-line form.
pub(crate) fn fmt_line_smt(v: f32) -> String {
    format!("smt{}", fmt_line(v))
}

/// Value-unit tag for a ranking score: `smt` = the risk-adjusted score the
/// race decides on. The table-cell form.
pub(crate) fn fmt_cell_smt(v: f32) -> String {
    format!("smt{}", fmt_cell(v))
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
    use super::{csv_field, fmt_cell, fmt_cell_smt, fmt_cfg, fmt_line, fmt_line_opt, fmt_line_smt};

    // ── the line rule: 1dp, floored so a real value never reads as zero ─────

    #[test]
    fn line_is_one_decimal_at_normal_magnitudes() {
        assert_eq!(fmt_line(117.55), "117.6");
        assert_eq!(fmt_line(97.1), "97.1");
        assert_eq!(fmt_line(20.5), "20.5");
        assert_eq!(fmt_line(214.6), "214.6");
        assert_eq!(fmt_line(1.0), "1.0");
        assert_eq!(fmt_line(999_999.0), "999999.0");
    }

    #[test]
    fn line_floor_keeps_small_values_visible() {
        // The case the floor exists for: 1dp would print 0.0.
        assert_eq!(fmt_line(0.02), "0.02");
        assert_eq!(fmt_line(0.004), "0.004");
        assert_eq!(fmt_line(-0.02), "-0.02");
        // 1dp is already enough, so no widening — and no "0.50" padding.
        assert_eq!(fmt_line(0.5), "0.5");
    }

    #[test]
    fn line_floor_is_capped_and_true_zero_is_not_widened() {
        assert_eq!(fmt_line(0.0), "0.0");
        assert_eq!(fmt_line(-0.0), "-0.0");
        // Too small to show even at the 4dp cap: prints honestly, not "0.0".
        assert_eq!(fmt_line(0.000_04), "0.0000");
        assert_eq!(fmt_line(1e-8), "0.0000");
    }

    #[test]
    fn line_uses_scientific_notation_only_at_scale() {
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
    fn line_passes_non_finite_through_without_widening() {
        // "NaN".parse() is Ok(NaN) which is != 0.0, so the widen loop cannot fire.
        assert_eq!(fmt_line(f32::NAN), "NaN");
        assert_eq!(fmt_line(f32::INFINITY), "inf");
        assert_eq!(fmt_line(f32::NEG_INFINITY), "-inf");
    }

    // ── the cell rule: 4dp, same floor at the cell's threshold ──────────────

    #[test]
    fn cell_is_four_decimals_at_normal_magnitudes() {
        assert_eq!(fmt_cell(117.55), "117.5500");
        assert_eq!(fmt_cell(97.1), "97.1000");
        assert_eq!(fmt_cell(0.0), "0.0000");
        assert_eq!(fmt_cell(999_999.0), "999999.0000");
    }

    #[test]
    fn cell_widens_only_below_its_floor() {
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

    // ── tagged + optional forms ────────────────────────────────────────────

    #[test]
    fn smt_variants_carry_the_unit_tag() {
        assert_eq!(fmt_line_smt(117.55), "smt117.6");
        assert_eq!(fmt_cell_smt(117.55), "smt117.5500");
    }

    #[test]
    fn optional_forms_render_unset_as_dash() {
        assert_eq!(fmt_line_opt(&None), "—");
        assert_eq!(fmt_line_opt(&Some(0.02)), "0.02");
    }

    // ── untouched neighbours ───────────────────────────────────────────────

    #[test]
    fn csv_field_quotes_only_when_needed() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    }
}
