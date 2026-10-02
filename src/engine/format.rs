//! Console + CSV formatting helpers for the engine.
//!
//! Owns the display-only float formatting (`fmt2`, `fmt_opt2`), RFC4180 CSV
//! quoting, and the one-line build identity stamp. Nothing here is persisted
//! through the rounded formatters — artifacts always serialize the raw `f32`.

/// One-line description of the RUNNING binary: crate version, profile, and
/// the executable path + its mtime. Printed at start and recorded in
/// `engine.json`.
///
/// Why the mtime matters: an artifact error (or a "parity failed") is often a
/// STALE PROCESS, not stale logic — a long run started before a fix keeps the
/// old behavior for hours, and its artifacts look wrong for no visible
/// reason. Stamping the exe + build time makes "was this the new code?" a
/// one-glance question instead of an investigation.
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

/// Format a float to 2 decimals for the **console log only**. Artifacts
/// (`engine.json`, `nets/<hash>.json`, `history.csv`, `checkpoints.json`) are
/// always written from the raw `f32` — serde and `Display` emit the shortest
/// decimal that round-trips, i.e. full precision. Never route anything
/// persisted through this function.
pub(crate) fn fmt2(v: f32) -> String {
    format!("{v:.2}")
}

/// Format an optional float to 2 decimals — plain `—` when unset (never
/// `Some(...)` in user-facing logs).
pub(crate) fn fmt_opt2(v: &Option<f32>) -> String {
    v.as_ref()
        .map(|x| format!("{x:.2}"))
        .unwrap_or_else(|| "—".into())
}

/// Value-unit tags — every fitness number on a log line names its KIND, so a
/// smoothed mean is never misread as a raw last-step value (or vice versa).
/// Three kinds appear on engine lines:
/// - `smt` — smoothed (rolling mean of the recent raw steps); what ranking,
///   freeze/dethrone and the gates DECIDE on.
/// - `avg` — average of checkpoint means (gate bars, population ledger
///   rollups); a historical aggregate, not this step's signal.
/// - bare — raw last-step value (a single measurement, no tag).
/// Tags are written inline in the log format string (`smt{:.2}`, `avg{:.4}`,
/// …) so each site picks its own precision; `fmt2_smt` is for sites that
/// build the tagged value as a `String` before splicing it in.
pub(crate) fn fmt2_smt(v: f32) -> String {
    format!("smt{:.2}", v)
}

/// RFC4180-quote a CSV field when it holds a delimiter, quote, or newline.
/// Lineage strings (`crossover:parents=h1,h2`) contain commas, so this is not
/// optional — an unquoted lineage would shift every later column.
pub(crate) fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}
