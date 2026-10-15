//! CFY — "Comfy Format for You": the engine's NON-step log lines rendered as
//! compact `comfy-table` blocks.
//!
//! Rule (the split the feature is named for): a line that starts with
//! `step <n>` keeps its familiar one-liner form; every other engine line is a
//! titled table, with any remaining prose collapsed into a single-line
//! footnote under its table. Raw metadata is never altered — a value is the
//! same string the old prose carried, only laid out in a cell.
//!
//! The layout mixes the five CFY revamps:
//! - **B** run manifest — one vertical `field | value` table at run start
//! - **A** sections — search space / trainer as their own labeled tables
//! - **C** matrix — run accounting as `event × counter` rows
//! - **D** cards — a banner + table + footnote wherever prose used to sit
//! - **E** legend footer — the log legend rides under the manifest

use comfy_table::presets::{UTF8_FULL, UTF8_FULL_CONDENSED};
use comfy_table::{Cell, Table};
use tracing::info;

/// Render a `header` + `rows` table in the engine's compact UTF-8 style.
pub(crate) fn table(header: &[&str], rows: &[Vec<String>]) -> String {
    let mut t = Table::new();
    t.load_preset(UTF8_FULL_CONDENSED);
    t.set_header(header.iter().map(Cell::new).collect::<Vec<_>>());
    for row in rows {
        t.add_row(row.iter().map(Cell::new).collect::<Vec<_>>());
    }
    t.to_string()
}

/// A vertical two-column `field | value` table — the workhorse for parameter
/// blocks (the run manifest and each section).
pub(crate) fn kv(rows: &[(String, String)]) -> String {
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|(k, v)| vec![k.clone(), v.clone()])
        .collect();
    table(&["field", "value"], &cells)
}

/// A one-cell banner naming a section — the card header.
pub(crate) fn banner(title: &str) -> String {
    let mut t = Table::new();
    t.load_preset(UTF8_FULL);
    t.add_row(vec![Cell::new(format!("▸ {title}"))]);
    t.to_string()
}

/// A titled card: banner, table body, then an optional one-line footnote
/// (where prose is summarized rather than left as a paragraph).
pub(crate) fn card(title: &str, body: &str, note: Option<&str>) -> String {
    let mut s = banner(title);
    s.push('\n');
    s.push_str(body);
    if let Some(n) = note {
        if !n.is_empty() {
            s.push('\n');
            s.push_str(&format!("╰─ {n}"));
        }
    }
    s
}

/// Emit a CFY block to the console, one line per rendered line — the block
/// itself is passed whole so `Minimal`-style redraw tooling could adopt it.
pub(crate) fn emit(text: &str) {
    for line in text.lines() {
        info!("{line}");
    }
}

/// Emit a CFY block that must show at EVERY log level (the inheritance receipt):
/// same table shape, but through `println!` rather than the level-filtered
/// `tracing` console.
pub(crate) fn emit_always(text: &str) {
    for line in text.lines() {
        println!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::{banner, card, kv, table};

    #[test]
    fn table_renders_header_and_rows() {
        let out = table(
            &["a", "b"],
            &[vec!["1".to_string(), "2".to_string()]],
        );
        assert!(out.contains("a"), "{out}");
        assert!(out.contains("b"), "{out}");
        assert!(out.contains('1'), "{out}");
        assert!(out.contains('2'), "{out}");
    }

    #[test]
    fn kv_uses_field_value_header_and_preserves_cells() {
        let out = kv(&[
            ("seed".to_string(), "42".to_string()),
            ("K".to_string(), "10".to_string()),
        ]);
        assert!(out.contains("field"), "{out}");
        assert!(out.contains("value"), "{out}");
        assert!(out.contains("seed"), "{out}");
        assert!(out.contains("42"), "{out}");
    }

    #[test]
    fn card_carries_banner_body_and_footnote() {
        let out = card("Run", &kv(&[("pop".to_string(), "50".to_string())]), Some("note"));
        assert!(out.contains("Run"), "{out}");
        assert!(out.contains("pop"), "{out}");
        assert!(out.contains("╰─ note"), "{out}");
        // No note ⇒ no dangling footnote line.
        let bare = card("Run", "│ x │", None);
        assert!(!bare.contains("╰─"), "{bare}");
    }

    #[test]
    fn banner_names_the_section() {
        assert!(banner("Search Space").contains("Search Space"));
    }
}
