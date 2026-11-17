//! CFY — "Comfy Format for You": the engine's NON-step log lines rendered as compact `comfy-table`
//! blocks.

use comfy_table::presets::{UTF8_FULL, UTF8_FULL_CONDENSED};
use comfy_table::{Cell, Table};
use tracing::info;

/// The shared renderer: a condensed table with an OPTIONAL column-header row.
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

/// Max width of a card's prose BLOCK — a legend row (key AND value) or a footnote.
pub(crate) const PROSE_WIDTH: usize = 110;

/// The table's "no value" cell: `na` (not available).
pub(crate) const NO_VALUE: &str = "na";

/// The floor the key column may squeeze the prose down to. Without it a very long key
/// (`fit (F = g⁻¹(ḡ), g = sign(x)·ln(1+|x|))`) would leave one word per line.
const MIN_PROSE_WIDTH: usize = 40;

/// Hard-wrap `text` at `width` COLUMNS (chars, not bytes: `·`, `Δ` and `na` are one column
/// several bytes).
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

/// The [`table`] whose SECOND column is prose: a real header row, values wrapped so the WHOLE block stays near [`PROSE_WIDTH`].
pub(crate) fn table_wrapped(header: &[&str], rows: &[(String, String)]) -> String {
    let key_width = rows
        .iter()
        .map(|(k, _)| k.chars().count())
        .chain(header.iter().map(|h| h.chars().count()))
        .max()
        .unwrap_or(0);
    let prose_width = PROSE_WIDTH.saturating_sub(key_width).max(MIN_PROSE_WIDTH);
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|(k, v)| vec![k.clone(), wrap(v, prose_width).join("\n")])
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

/// A titled card: banner, table body, then an optional footnote (where prose is summarized rather
/// than left as a paragraph).
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
/// itself is passed whole so `Minimal`-style redraw tooling could adopt
pub(crate) fn emit(text: &str) {
    for line in text.lines() {
        info!("{line}");
    }
}

/// Emit a CFY block that must show at EVERY log level (the inheritance receipt): same table shape,
/// but through `println!` rather than the level-filtered `tracing` console.
pub(crate) fn emit_always(text: &str) {
    for line in text.lines() {
        println!("{line}");
    }
}
#[cfg(test)]
mod tests {
    use super::{PROSE_WIDTH, card, kv, table, table_wrapped, wrap};

    /// The ONE decision in these wrappers worth pinning: a key/value block has no header row
    /// banner names it), while a named table keeps the header that says what it holds.
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

    /// `wrap` is the whole fix for the 400-column legend: words are never split, nothing
    /// dropped, and no line exceeds the requested width — not even for a single token longer
    #[test]
    fn wrap_breaks_on_words_and_respects_the_width() {
        let text = "★ saved as elite (top-5 by A) · x saved as worst (bottom-5 by E) · · neither · b = inside its born cull shield (crossover 5 / mutation 5 update(s)) · j = inside the elite join window (5 served) · t = past the crown term (5 clocks, age-blocked)";
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
        let long = "window mean · step-to-step spread · how many readings; `na` = no reading yet, or the spread at one reading, which has no observable spread at all";
        let legend = table_wrapped(
            &["abbrev", "description"],
            &[("A".to_string(), long.to_string())],
        );
        let widest = legend.lines().map(|l| l.chars().count()).max().unwrap_or(0);
        // The key column (`description` is 11 columns) comes OUT of the prose budget, so the
        // block stays at the prose width plus its borders — one screen, never the old 350.
        assert!(
            widest <= PROSE_WIDTH + 24,
            "the block stays near the prose width, got {widest} columns"
        );
        // The key appears ONCE, and the wrapped tail rides the value column.
        assert_eq!(legend.matches("A").count(), 1, "{legend}");
        // The wrapped tail continues in the VALUE column. Which word breaks is not fixed (the
        // prose budget is `PROSE_WIDTH` minus the key column), but where the text goes is.
        let tail = legend
            .lines()
            .find(|l| l.contains("observable"))
            .expect("the value's tail is printed");
        assert!(
            tail.contains('┆'),
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
        assert!(
            wrapped.contains("\n   "),
            "continuations sit under the text: {wrapped}"
        );
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

    /// A legend is a NAMED table: the header says which column is the symbol
    /// is the explanation, and the wrapped value is still ONE entry.
    #[test]
    fn table_wrapped_keeps_its_header_and_wraps_the_value() {
        let out = table_wrapped(
            &["abbrev", "description"],
            &[(
                "E = (U ÷ Umax) × G".to_string(),
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
