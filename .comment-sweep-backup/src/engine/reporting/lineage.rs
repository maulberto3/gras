//! The ancestry renderer: the per-elite mermaid diagram, drawn from ONE
//! `LineageFacts` walker (split out of `reporting` by AGENTS.md rule 19). The
//! stop-time ASCII card was REMOVED on 2026-11-01: the final field table is the
//! console's last word, and the ancestry it drew is already in `history.csv`
//! (`origin` carries `parents=`) and in every `nets/<hash>.json` (`created_from`).
//! The mermaid section inside `elite-<hash>.md` stays — a doc outlives the run dir.

use super::*;

impl CoreEngine {
    /// The `## Ancestry` section of ONE elite's markdown file (`elite-<hash>.md`): the
    /// same recorded lineage the stop card draws, as a mermaid `graph TB` a doc viewer
    /// renders. The card is the console's answer; this is the per-net artifact's, so the
    /// elite file explains its own birth long after the run's log is gone.
    ///
    /// A founder (random draw / inherited `seed:` blueprint) has no recorded parents and
    /// comes out as the single box the reader expects; a crossover converges parent →
    /// child, with the parents at the top and this net at the bottom.
    pub(crate) fn lineage_markdown(&self, root: &str, root_role: &str) -> String {
        let mut tree = LineageTree::new(self);
        let has_parents = tree
            .facts(root)
            .map(|f| !f.parents.is_empty())
            .unwrap_or(false);
        let (diagram, cut) = tree.mermaid(root, root_role);
        let mut prose = if has_parents {
            "Read downward: an arrow goes from a parent to the child it made (`p1`/`p2` are the crossover slots), so this net sits at the bottom. **Gold** is this net, **blue** an ancestor still live at stop, **grey** one that was culled (the number after `†` is the clock it was culled at). `Δfit` on an arrow is the child's fit minus that parent's, signed so `+` means the child scored better. Edges are each net's recorded `created_from`; a culled parent's numbers come from its `nets/<hash>.json` tombstone."
                .to_string()
        } else {
            "No recorded parents: this net entered as a founder — a random draw, an inherited `seed:` blueprint, or the initial batch — so its ancestry is the single box below."
                .to_string()
        };
        if cut {
            prose.push_str(&format!(
                " The walk stops at {LINEAGE_MD_MAX_DEPTH} generations / {LINEAGE_MD_NODE_BUDGET} nets; each ancestor's own `nets/<hash>.json` still carries the next step back."
            ));
        }
        format!("## Ancestry\n\n{prose}\n\n{diagram}")
    }
}

/// The three paints a lineage box can wear. Kept as consts (not `classDef` strings) so
/// the inline `style` line, the label's `<span>` and the class all read the SAME values
/// — three channels that cannot drift apart.
#[derive(Clone, Copy)]
struct LineagePaint {
    fill: &'static str,
    stroke: &'static str,
    ink: &'static str,
    width: &'static str,
}

/// The net the file is about (`elite-<hash>.md`).
const FOCUS: LineagePaint = LineagePaint {
    fill: "#fff3cf",
    stroke: "#8a5a00",
    ink: "#2b1d00",
    width: "2.5px",
};

/// An ancestor still live when the run stopped.
const LIVE: LineagePaint = LineagePaint {
    fill: "#e6f1fb",
    stroke: "#2f6fa8",
    ink: "#0d2538",
    width: "1.5px",
};

/// An ancestor that was culled (its `†clock` says when).
const CULLED: LineagePaint = LineagePaint {
    fill: "#ededed",
    stroke: "#5f5f5f",
    ink: "#1f1f1f",
    width: "1.5px",
};

/// How many generations the elite files' mermaid diagram walks. A DOC can afford more
/// than a terminal card: mermaid lays the graph out itself, so depth costs nodes, not
/// screen width.
const LINEAGE_MD_MAX_DEPTH: usize = 6;
/// The most nodes one elite diagram may name. Beyond this the walk stops and the
/// markdown says so, rather than shipping a 500-node graph in a per-net file.
const LINEAGE_MD_NODE_BUDGET: usize = 150;

/// The hashes named by a lineage string — `crossover-one-point:parents=<h1>,<h2>[+mut]`
/// or `clone:parents=<h1>` — as the parent list, oldest first. Founders (`random`,
/// `founder-N`) and inherited nets (`seed:<run>:<hash>`) name no in-run parents and come
/// back empty, which is what makes them the roots of a tree.
pub(super) fn lineage_parents(created_from: Option<&str>) -> Vec<String> {
    let Some(raw) = created_from else {
        return Vec::new();
    };
    // `+mut` is a flag ON the child (crossover, then mutated), not part of a parent id.
    let head = raw.strip_suffix("+mut").unwrap_or(raw);
    let Some((_, parents)) = head.split_once(":parents=") else {
        return Vec::new();
    };
    parents
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect()
}

/// The first 8 hex of a hash — the width every other stop-time identity prints at.
pub(super) fn short_hash(hash: &str) -> String {
    hash[..8.min(hash.len())].to_string()
}

/// One node's mermaid label: identity, origin, fit, and what happened to it (the root
/// says `ELITE`; an ancestor says `live` or the clock it was culled at).
pub(super) fn mermaid_node_label(
    hash: &str,
    facts: Option<&LineageFacts>,
    root_role: Option<&str>,
) -> String {
    let Some(facts) = facts else {
        return short_hash(hash);
    };
    let mut parts = vec![
        short_hash(hash),
        origin_cell(facts.origin.as_deref(), facts.born),
    ];
    if let Some(fit) = facts.fit {
        parts.push(format!("fit {}", fmt_cell(fit)));
    }
    // The root says what the FILE says it is (`ELITE` in `elite-<hash>.md`) — one
    // renderer, so the role stays the caller's.
    if let Some(role) = root_role {
        parts.push(role.to_string());
    } else if facts.alive {
        parts.push("live".to_string());
    } else if let Some(clock) = facts.culled_at {
        parts.push(format!("†{clock}"));
    }
    parts.join("<br/>")
}

/// One net's ancestry facts, whether it is LIVE or a culled tombstone. One shape for
/// both is what lets the tree cross a cull (`final_risk_score` for a tombstone is the
/// same ranking value a live net reads from its window).
///
/// Only what a mermaid node says: identity, origin, ranking fit, and how the net left
/// the race. The window mean/spread stay in `nets/<hash>.json` — they are the TABLE's
/// business.
#[derive(Clone, Debug)]
pub(super) struct LineageFacts {
    /// The recorded lineage string (`created_from`), kept verbatim for [`origin_cell`].
    origin: Option<String>,
    /// Clock it entered the population.
    born: usize,
    /// The ranking value (risk-adjusted fit); `None` when it never measured.
    fit: Option<f32>,
    alive: bool,
    culled_at: Option<usize>,
    /// Parent hashes named by `created_from` (empty = a root).
    parents: Vec<String>,
}

/// The ancestry card's one walker: resolve a hash (live memory first, then the tombstone
/// on disk), draw it, then recurse into its parents.
pub(super) struct LineageTree<'a> {
    engine: &'a CoreEngine,
    /// Resolved facts per hash — a shared ancestor is read from disk at most once.
    cache: std::collections::HashMap<String, Option<LineageFacts>>,
}

impl<'a> LineageTree<'a> {
    fn new(engine: &'a CoreEngine) -> Self {
        Self {
            engine,
            cache: std::collections::HashMap::new(),
        }
    }

    /// One net's facts: the live state first (a parent may still be alive), then the
    /// `nets/<hash>.json` tombstone a cull leaves behind. `None` when neither exists —
    /// e.g. a parent inherited from another run's directory.
    fn facts(&mut self, hash: &str) -> Option<LineageFacts> {
        if let Some(hit) = self.cache.get(hash) {
            return hit.clone();
        }
        let engine = self.engine;
        let facts = if let Some(state) = engine.state.net(hash) {
            Some(LineageFacts {
                origin: state.created_from.clone(),
                born: state.entered_at_step,
                fit: engine.smoothed_fitness_of(hash),
                alive: true,
                culled_at: None,
                parents: lineage_parents(state.created_from.as_deref()),
            })
        } else {
            match crate::state::load_net_state(&engine.run_dir, hash) {
                Ok(state) => Some(LineageFacts {
                    origin: state.created_from.clone(),
                    born: state.entered_at_step,
                    // A legacy tombstone may predate `final_risk_score`; the window mean is
                    // then the only level it recorded, so it is read as the fit.
                    fit: state
                        .final_risk_score
                        .filter(|v| v.is_finite())
                        .or_else(|| state.final_fitness_mean.filter(|v| v.is_finite())),
                    alive: state.is_alive,
                    culled_at: state.culled_at_step,
                    parents: lineage_parents(state.created_from.as_deref()),
                }),
                Err(_) => None,
            }
        };
        self.cache.insert(hash.to_string(), facts.clone());
        facts
    }

    /// This net's ancestry as a mermaid `graph TB`: parents at the TOP, the net at the
    /// BOTTOM, one arrow per parent → child edge. Shared ancestors are ONE node (mermaid
    /// keys by id), so a DAG converges instead of being repeated the way the console
    /// tree's `↺` avoids; a founder comes out as a single box.
    ///
    /// Returns the fenced diagram and whether the walk was cut short (depth or node
    /// budget), which the caller states in the section's prose.
    fn mermaid(&mut self, root: &str, root_role: &str) -> (String, bool) {
        let mut nodes: Vec<String> = Vec::new();
        let mut edges: Vec<(String, String, usize, usize)> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut stack: Vec<(String, usize)> = vec![(root.to_string(), 0)];
        let mut cut = false;
        while let Some((hash, depth)) = stack.pop() {
            if !seen.insert(hash.clone()) {
                continue;
            }
            if nodes.len() >= LINEAGE_MD_NODE_BUDGET {
                cut = true;
                break;
            }
            nodes.push(hash.clone());
            let Some(facts) = self.facts(&hash) else {
                continue;
            };
            if facts.parents.is_empty() {
                continue;
            }
            if depth >= LINEAGE_MD_MAX_DEPTH {
                cut = true;
                continue;
            }
            let n = facts.parents.len();
            for (i, parent) in facts.parents.iter().enumerate() {
                edges.push((parent.clone(), hash.clone(), i + 1, n));
                stack.push((parent.clone(), depth + 1));
            }
        }
        let known: std::collections::HashSet<String> = nodes.iter().cloned().collect();
        let direction = self.engine.fitness.direction();
        let adjusted = |v: f32| match direction {
            crate::utils::fitness::Direction::Maximize => v,
            crate::utils::fitness::Direction::Minimize => -v,
        };
        // ── Contrast: every box is LIGHT, every label states its own near-black ink ──
        //
        // This diagram shipped twice before with a readable-on-paper palette that a real
        // viewer disagreed with, so the ink is now stated on THREE independent channels:
        // an inline `style … color:` (an attribute, so it outranks a theme's CSS rule), a
        // `<span style="color:…">` inside the label (survives a viewer that repaints label
        // text wholesale — the exact failure that ate the first two palettes), and the
        // `classDef`'s own `color:`. A LIGHT fill is what makes the fallback safe: if all
        // three are ignored, the theme's own text colour still lands on a light box, which
        // is dark-on-light in every theme. The previous palette used dark fills with white
        // labels, so a theme that kept its near-white text produced white words on a white
        // page — dark boxes only read if the label colour WINS, and that is the one thing a
        // viewer is free to override.
        //
        // `htmlLabels` and the rest of the theme are pinned in `init` so the graph does not
        // half-inherit a dark theme (canvas, edges, edge-label background).
        let mut out = String::from("```mermaid\n");
        out.push_str(
            "%%{init: {\"theme\": \"base\", \"flowchart\": {\"htmlLabels\": true, \"curve\": \"basis\"}, \"themeVariables\": {\"background\": \"#ffffff\", \"primaryColor\": \"#e6f1fb\", \"primaryTextColor\": \"#0d2538\", \"primaryBorderColor\": \"#2f6fa8\", \"lineColor\": \"#3d6b96\", \"textColor\": \"#0d2538\", \"edgeLabelBackground\": \"#ffffff\", \"fontSize\": \"14px\"}}}%%\n",
        );
        out.push_str("graph TB\n");
        // Classes first, so a viewer that only reads classes never sees an unstyled box.
        for (name, paint) in [("focus", FOCUS), ("live", LIVE), ("culled", CULLED)] {
            out.push_str(&format!(
                "    classDef {name} fill:{},stroke:{},stroke-width:{},color:{}\n",
                paint.fill, paint.stroke, paint.width, paint.ink
            ));
        }
        for hash in &nodes {
            let facts = self.facts(hash);
            let is_root = hash == root;
            // A tombstone is grey; everything else on the walk is a live ancestor.
            let (paint, class) = if is_root {
                (FOCUS, "focus")
            } else if facts.as_ref().map(|f| !f.alive).unwrap_or(false) {
                (CULLED, "culled")
            } else {
                (LIVE, "live")
            };
            let label = mermaid_node_label(hash, facts.as_ref(), is_root.then_some(root_role));
            let id = short_hash(hash);
            // Single quotes inside the span: the label itself is double-quoted mermaid.
            out.push_str(&format!(
                "    n_{id}[\"<span style='color:{}'>{label}</span>\"]\n",
                paint.ink
            ));
            out.push_str(&format!(
                "    style n_{id} fill:{},stroke:{},stroke-width:{},color:{}\n",
                paint.fill, paint.stroke, paint.width, paint.ink
            ));
            out.push_str(&format!("    class n_{id} {class}\n"));
        }
        for (parent, child, slot, n) in &edges {
            // A parent beyond the node budget was never declared: drop the edge with the
            // cut instead of letting mermaid invent a blank node for it.
            if !known.contains(parent) || !known.contains(child) {
                cut = true;
                continue;
            }
            let parent_facts = self.facts(parent);
            let child_facts = self.facts(child);
            let mut label = if *n == 1 {
                "p".to_string()
            } else {
                format!("p{slot}")
            };
            if let (Some(parent_facts), Some(child_facts)) = (parent_facts, child_facts) {
                if let (Some(parent_fit), Some(child_fit)) = (parent_facts.fit, child_facts.fit) {
                    let gain = adjusted(child_fit) - adjusted(parent_fit);
                    label.push_str(&format!(
                        " · Δfit {}{}",
                        if gain >= 0.0 { "+" } else { "−" },
                        fmt_cell(gain.abs())
                    ));
                }
            }
            out.push_str(&format!(
                "    n_{} -->|\"{}\"| n_{}\n",
                short_hash(parent),
                label,
                short_hash(child)
            ));
        }
        out.push_str("```\n");
        (out, cut)
    }
}
