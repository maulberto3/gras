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
//! - **B** run manifest — one vertical key/value table at run start
//! - **A** sections — search space / trainer as their own labeled tables
//! - **C** matrix — run accounting as `event × counter` rows
//! - **D** cards — a banner + table + footnote wherever prose used to sit
//! - **E** legend footer — the log legend rides under the manifest

use comfy_table::presets::{UTF8_FULL, UTF8_FULL_CONDENSED};
use comfy_table::{Cell, Table};
use tracing::info;

/// The shared renderer: a condensed table with an OPTIONAL column-header row.
///
/// `None` is for a key/value block. Such a block needs no header because its
/// card's banner already names it — a literal `field | value` row only repeats
/// the title on every parameter card. (A real table keeps its header: `hash`,
/// `smt` and friends carry information the title does not.)
fn render(header: Option<&[&str]>, rows: &[Vec<String>]) -> String {
    let mut t = Table::new();
    t.load_preset(UTF8_FULL_CONDENSED);
    if let Some(header) = header {
        t.set_header(header.iter().map(Cell::new).collect::<Vec<_>>());
    }
    for row in rows {
        t.add_row(row.iter().map(Cell::new).collect::<Vec<_>>());
    }
    t.to_string()
}

/// Render a `header` + `rows` table in the engine's compact UTF-8 style.
pub(crate) fn table(header: &[&str], rows: &[Vec<String>]) -> String {
    render(Some(header), rows)
}

/// A vertical two-column key/value table — the workhorse for parameter blocks
/// (the run manifest and each section). Headerless on purpose (see [`render`]).
pub(crate) fn kv(rows: &[(String, String)]) -> String {
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|(k, v)| vec![k.clone(), v.clone()])
        .collect();
    render(None, &cells)
}

/// Max width of PROSE inside a card — a legend value or a footnote.
///
/// A table is as wide as its widest cell, so ONE 400-character legend value used
/// to stretch the whole block to 400 columns: readable in a maximized terminal,
/// unreadable everywhere else, and unquotable in a bug report. Prose is wrapped to
/// this width instead, wrapped deterministically so a saved log stays diffable.
pub(crate) const PROSE_WIDTH: usize = 110;

/// Hard-wrap `text` at `width` COLUMNS (chars, not bytes: `·`, `Δ` and `—` are one
/// column but several bytes).
///
/// Word boundaries are preferred; a single token longer than the width (a hash, a
/// path) is broken rather than allowed to stretch the block it sits in.
pub(crate) fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut len = 0usize;
    for word in text.split(' ') {
        let word_len = word.chars().count();
        if len > 0 && len + 1 + word_len > width {
            lines.push(std::mem::take(&mut current));
            len = 0;
        }
        if len > 0 {
            current.push(' ');
            len += 1;
        }
        if word_len > width {
            // An unbreakable token: emit it in `width`-column chunks so the block's
            // width stays bounded whatever the input carries.
            for (i, ch) in word.chars().enumerate() {
                if i > 0 && len == width {
                    lines.push(std::mem::take(&mut current));
                    len = 0;
                }
                current.push(ch);
                len += 1;
            }
            continue;
        }
        current.push_str(word);
        len += word_len;
    }
    if !current.is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}

/// The [`table`] whose SECOND column is prose: a real header row, values wrapped at
/// [`PROSE_WIDTH`]. This is the legend's shape — `abbrev | description`.
///
/// comfy-table renders the `\n` inside a cell as extra lines of the SAME row (and
/// trims their leading space), so a wrapped entry stays one entry: the key cell is
/// empty beside it and the value column lines up under itself.
///
/// The header is what makes a legend a table rather than a list of unrelated pairs:
/// the left column is a symbol to look UP (`E`), the right is what it means, and the
/// reader knows which is which without working it out from the content.
pub(crate) fn table_wrapped(header: &[&str], rows: &[(String, String)]) -> String {
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|(k, v)| vec![k.clone(), wrap(v, PROSE_WIDTH).join("\n")])
        .collect();
    render(Some(header), &cells)
}

/// A one-cell banner naming a section — the card header.
pub(crate) fn banner(title: &str) -> String {
    let mut t = Table::new();
    t.load_preset(UTF8_FULL);
    t.add_row(vec![Cell::new(format!("▸ {title}"))]);
    t.to_string()
}

/// A titled card: banner, table body, then an optional footnote (where prose is
/// summarized rather than left as a paragraph).
///
/// The footnote is wrapped at [`PROSE_WIDTH`] like a legend value, and its
/// continuation lines are indented under the text — not under the `╰─` — so the
/// marker still names exactly one footnote however many lines it takes.
pub(crate) fn card(title: &str, body: &str, note: Option<&str>) -> String {
    let mut s = banner(title);
    s.push('\n');
    s.push_str(body);
    if let Some(n) = note {
        if !n.is_empty() {
            for (i, line) in wrap(n, PROSE_WIDTH).iter().enumerate() {
                s.push('\n');
                if i == 0 {
                    s.push_str(&format!("╰─ {line}"));
                } else {
                    s.push_str(&format!("   {line}"));
                }
            }
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
    use super::{PROSE_WIDTH, card, kv, table, table_wrapped, wrap};

    /// The ONE decision in these wrappers worth pinning: a key/value block has no
    /// header row (the banner names it), while a named table keeps the header that
    /// says what it holds.
    #[test]
    fn kv_is_headerless_and_a_named_table_is_not() {
        let block = kv(&[
            ("seed".to_string(), "42".to_string()),
            ("K".to_string(), "10".to_string()),
        ]);
        assert!(!block.contains("field"), "{block}");
        assert!(!block.contains("value"), "{block}");
        assert!(block.contains("seed"), "{block}");
        assert!(block.contains("42"), "{block}");
        let named = table(
            &["hash", "file"],
            &[vec!["abc".to_string(), "x.json".to_string()]],
        );
        assert!(named.contains("hash"), "{named}");
        assert!(named.contains("file"), "{named}");
    }

    /// `wrap` is the whole fix for the 400-column legend: words are never split,
    /// nothing is dropped, and no line exceeds the requested width — not even for a
    /// single token longer than it.
    #[test]
    fn wrap_breaks_on_words_and_respects_the_width() {
        let text = "★ saved as elite (top-5 by A) · ✖ saved as worst (bottom-5 by E) · · neither · b = inside its born cull shield (crossover 5 / mutation 5 clock(s)) · j = inside the elite join window (5 served) · t = past the crown term (5 clocks, age-blocked)";
        let lines = wrap(text, 60);
        assert!(lines.len() > 2, "a long sentence wraps: {lines:?}");
        for line in &lines {
            assert!(
                line.chars().count() <= 60,
                "{} columns in {line:?}",
                line.chars().count()
            );
        }
        // Wrapping is lossless: the words, in order.
        assert_eq!(lines.join(" "), text);
        // A single unbreakable token is chunked rather than allowed to stretch a block.
        let long = wrap(&"x".repeat(50), 20);
        assert_eq!(long.len(), 3, "{long:?}");
        assert!(long.iter().all(|l| l.chars().count() <= 20));
        assert_eq!(long.concat(), "x".repeat(50));
        // Empty input still yields one (empty) line, so callers can join "\n".
        assert_eq!(wrap("", 40), vec![String::new()]);
    }

    /// The prose width through BOTH sinks, and the card's own shape.
    ///
    /// One test owns the card contract: banner, body, exactly one `╰─` marker per
    /// footnote, continuations under the TEXT — plus the two blocks that used to
    /// blow the width out (a sentence-long legend value, a paragraph footnote).
    #[test]
    fn prose_is_wrapped_in_a_legend_and_a_footnote() {
        // ── the card's shape ────────────────────────────────────────────
        let short = card(
            "Run",
            &kv(&[("pop".to_string(), "50".to_string())]),
            Some("note"),
        );
        assert!(short.contains("Run"), "{short}");
        assert!(short.contains("pop"), "{short}");
        assert!(short.contains("╰─ note"), "{short}");
        // No note ⇒ no dangling footnote line.
        let bare = card("Run", "│ x │", None);
        assert!(!bare.contains("╰─"), "{bare}");

        // ── a legend value wraps INSIDE its own row ─────────────────────
        let long = "window mean · step-to-step spread · how many readings; `—` = no reading yet, or the spread at one reading, which has no observable spread at all";
        let legend = table_wrapped(&["abbrev", "description"], &[("A".to_string(), long.to_string())]);
        let widest = legend.lines().map(|l| l.chars().count()).max().unwrap_or(0);
        // The HEADER widens the key column (`description` is 11 columns), so the bound is
        // the prose width plus that key cell — still one screen, never the old 350.
        assert!(
            widest <= PROSE_WIDTH + 24,
            "the block stays near the prose width, got {widest} columns"
        );
        // The key appears ONCE, and the wrapped tail rides the value column.
        assert_eq!(legend.matches("A").count(), 1, "{legend}");
        assert!(
            legend.contains("┆ which has no observable") || legend.contains("┆ no observable"),
            "continuation stays in the value column: {legend}"
        );

        // ── a paragraph footnote wraps UNDER the text, one marker ───────
        let note = "Δcrown = mean multiset-Jaccard distance to the reported champions: 0.00 = structurally identical to one of them · Δpop = mean distance to every other live net: 0.00 = a clone of the crowd, 1.00 = unlike everything alive";
        let wrapped = card("Final field", "│ x │", Some(note));
        assert!(wrapped.contains("╰─ Δcrown"), "{wrapped}");
        assert_eq!(
            wrapped.matches("╰─").count(),
            1,
            "one marker, one footnote: {wrapped}"
        );
        assert!(wrapped.contains("\n   "), "continuations sit under the text: {wrapped}");
        for line in wrapped
            .lines()
            .filter(|l| !l.starts_with('┌') && !l.starts_with('│'))
        {
            assert!(
                line.chars().count() <= PROSE_WIDTH + 3,
                "{} columns: {line:?}",
                line.chars().count()
            );
        }
        // The wrap is lossless: the tail words survive.
        let flat = wrapped.split_whitespace().collect::<Vec<_>>().join(" ");
        for word in ["structurally", "unlike", "everything", "alive"] {
            assert!(flat.contains(word), "{word} lost in {wrapped}");
        }
    }

    /// A legend is a NAMED table: the header says which column is the symbol and which
    /// is the explanation, and the wrapped value is still ONE entry.
    #[test]
    fn table_wrapped_keeps_its_header_and_wraps_the_value() {
        let out = table_wrapped(
            &["abbrev", "description"],
            &[(
                "E = (D ÷ Dmax) × G".to_string(),
                "inefficiency: the effort a net spent against the score it earned, so a net that trained the whole field's budget and still landed at the bottom reads 1.00 while one that barely trained reads near 0.00".to_string(),
            )],
        );
        assert!(out.contains("abbrev"), "{out}");
        assert!(out.contains("description"), "{out}");
        // One entry, several lines: the key is printed once, beside its first line.
        assert_eq!(out.matches("abbrev").count(), 1, "{out}");
        assert!(out.lines().count() >= 5, "the value wraps: {out}");
        // Every rendered line stays inside the prose width plus the key cell.
        let widest = out.lines().map(|l| l.chars().count()).max().unwrap_or(0);
        assert!(widest <= PROSE_WIDTH + 24, "{widest} columns: {out}");
    }
}
