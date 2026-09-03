//! Engine logging — re-exports from split modules.
//!
//! - [`init_log`] — options, dataset, population (shown once at new())
//! - [`gen_log`] — run start, done message, next steps
//! - [`summary_log`] — robustness table display

pub(crate) use crate::utils::init_log::log_initialization;
pub(crate) use crate::utils::gen_log::{log_done, log_run_start};
pub(crate) use crate::utils::summary_log::{log_repeated_topologies, topo_hash};

use comfy_table::{ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};

/// Single width for every section table — one consistent look across all logs.
pub(crate) const TABLE_WIDTH: u16 = 160;

/// Master switch: `false` = wide, airy tables (default); `true` = compact.
/// Flip this one const and every table in every section changes at once.
pub(crate) const COMPACT_TABLES: bool = false;

/// Render a `══ section ══` heading plus a comfy-table through `log::info`.
///
/// Wide mode: forced `width` with (2, 2) column padding.
/// Compact mode: content-sized columns with (0, 1) padding — `width` ignored.
pub(crate) fn log_section_table(heading: &str, header: &[&str], rows: &[Vec<String>], width: u16) {
    log::info!("");
    log::info!("══ {heading} ══{}", "═".repeat(60usize.saturating_sub(heading.len() + 4)));

    let mut table = Table::new();
    if COMPACT_TABLES {
        // Disabled = columns sized to content, no terminal-width stretching.
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Disabled);
        table.column_iter_mut().for_each(|c| {
            c.set_padding((0, 1));
        });
    } else {
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::DynamicFullWidth)
            .set_width(width);
        table.column_iter_mut().for_each(|c| {
            c.set_padding((2, 2));
        });
    }
    table.set_header(header);
    for row in rows {
        table.add_row(row.clone());
    }
    for line in table.to_string().lines() {
        log::info!("  {line}");
    }
}