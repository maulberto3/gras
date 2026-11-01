//! The ancestry renderer: the stop-time ASCII tree and the per-elite mermaid
//! diagram, drawn from ONE `LineageFacts` walker (split out of `reporting` by
//! AGENTS.md rule 19 — a move, not a rewrite). `LineageTree` resolves facts and
//! reads nothing else; both renderers consume what it resolved, and top-level
//! items are `pub(super)` because the tests live beside this module.

use super::*;

impl CoreEngine {
    /// The stop-time ANCESTRY card: every reported net with its parents below it.
    ///
    /// The edges already exist in the run's own data — a child's `created_from` names its
    /// parent hashes, and a culled parent leaves a full `nets/<hash>.json` tombstone — so
    /// this card adds no bookkeeping; it only draws what is recorded. It is emitted right
    /// after the final-field table because that is the question the table raises: `#1` is
    /// `cx one-point@50`, and the reader wants to know WHICH nets crossed and how they
    /// scored. Stop-time only, read-only (a few tombstone file reads, no RNG, no forward
    /// pass), so it cannot perturb a replay.
    pub(crate) fn log_lineage(
        &self,
        top: &[(String, f32)],
        marks: &std::collections::HashMap<String, &'static str>,
    ) {
        if top.is_empty() {
            return;
        }
        let lines = self.lineage_lines(top, marks);
        crate::utils::cfy::emit(&crate::utils::cfy::card(
            &format!(
                "Lineage — how the {} reported net(s) were born",
                top.len()
            ),
            &lines.join("\n"),
            Some(&format!(
                "read downward in birth order: oldest ancestors at the top, the reported net on the `▼` line at the bottom · `g<N>` = generation above the net (`g1` = its own parents), so depth is a printed number instead of indentation · `g<N> ┌ parents of <hash>` introduces ONE child's parent set: the `│`/`└` lines under it are the nets that made that child, and `p1`/`p2` are the crossover slots (`p` when it has a single parent) · `fit` = the risk-adjusted score the net ranked on · `Δfit` = the child's fit minus this parent's, signed so `+` means the child scored better (the ranking's own direction) · `Δparams` = child − parent · `†<clock> <channel>` = culled at that clock, `live` = still in the population at stop · `↺ shown above` = this hash is already drawn as a deeper ancestor, i.e. the two branches share it · `…` = the chain continues past the {LINEAGE_MAX_DEPTH}-generation cap · edges are each net's recorded `created_from`, and a dead parent's numbers come from its tombstone · `▼ … root (no recorded parents)` = this net entered as a founder — a random draw, an inherited `seed:` blueprint, or the initial batch — so it has no ancestry to draw"
            )),
        ));
    }

    /// The ancestry card's LINES — split from [`Self::log_lineage`] so the drawing is one
    /// function and the emission is another.
    fn lineage_lines(
        &self,
        top: &[(String, f32)],
        marks: &std::collections::HashMap<String, &'static str>,
    ) -> Vec<String> {
        let mut tree = LineageTree::new(self);
        let mut out: Vec<String> = Vec::new();
        for (rank, (hash, fit)) in top.iter().enumerate() {
            // The root line mirrors the table's identity columns (rank, hash, origin) and
            // repeats its mark, so a tree is matched to its row by eye. `fit` falls back to
            // the table's own value when the net carries no stored reading.
            tree.root_fit = Some(*fit);
            tree.root_mark = marks.get(hash).copied().unwrap_or("·");
            // `↺ shown above` is per NOMINEE: each reported net gets its own chain, so an
            // ancestor shared with the previous net is drawn again rather than pointing at a
            // card the reader has already scrolled past.
            tree.shown.clear();
            tree.render_chain(hash, rank + 1, &mut out);
            // A VISIBLE divider between reported nets, not a blank line: the card body is a
            // single reflowed block, so an empty line is swallowed and the next net's parent
            // group then reads as the previous net's ancestry (measured: a `random` champion
            // looked like it had four generations of parents). The divider cannot be eaten.
            if rank + 1 < top.len() {
                out.push("   ⋯".to_string());
            }
        }
        if tree.truncated {
            out.push(format!(
                "… ancestry stops at the {LINEAGE_MAX_DEPTH}-generation / {LINEAGE_NODE_BUDGET}-line cap"
            ));
        }
        out
    }

    /// The `## Ancestry` section of ONE elite's markdown file (`elite-<hash>.md`): the
    /// same recorded lineage the stop card draws, as a mermaid `graph TB` a doc viewer
    /// renders. The card is the console's answer; this is the per-net artifact's, so the
    /// elite file explains its own birth long after the run's log is gone.
    ///
    /// A founder (random draw / inherited `seed:` blueprint) has no recorded parents and
    /// comes out as the single box the reader expects; a crossover converges parent →
    /// child, with the parents at the top and this net at the bottom.
    pub(crate) fn lineage_markdown(&self, root: &str) -> String {
        let mut tree = LineageTree::new(self);
        let has_parents = tree
            .facts(root)
            .map(|f| !f.parents.is_empty())
            .unwrap_or(false);
        let (diagram, cut) = tree.mermaid(root);
        let mut prose = if has_parents {
            "Read downward: an arrow goes from a parent to the child it made (`p1`/`p2` are the crossover slots), so this net sits at the bottom. `Δfit` on an arrow is the child's fit minus that parent's, signed so `+` means the child scored better. Edges are each net's recorded `created_from`; a culled parent's numbers come from its `nets/<hash>.json` tombstone."
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

/// How many generations above a reported net the lineage card walks. A search that
/// crossed for 100 clocks has a deep DAG; 4 generations answers "who made this and how
/// did they score" without turning one card into a screenful, and every ancestor's own
/// `nets/<hash>.json` still carries the next step back.
const LINEAGE_MAX_DEPTH: usize = 4;
/// How many tree lines ONE card may print, across all reported nets.
const LINEAGE_NODE_BUDGET: usize = 64;
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
pub(super) fn mermaid_node_label(hash: &str, facts: Option<&LineageFacts>, root: bool) -> String {
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
    if root {
        parts.push("ELITE".to_string());
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
/// Only what a lineage question needs: identity, ranking fit, the param count the `Δparams`
/// leg measures against, and how the net left the race. The window mean/spread stay in
/// `nets/<hash>.json` — they are the TABLE's business, and carrying them here made every
/// ancestry line twice as long as it needed to be.
#[derive(Clone, Debug)]
pub(super) struct LineageFacts {
    /// The recorded lineage string (`created_from`), kept verbatim for [`origin_cell`].
    origin: Option<String>,
    /// Clock it entered the population.
    born: usize,
    /// The ranking value (risk-adjusted fit); `None` when it never measured.
    fit: Option<f32>,
    params: Option<i64>,
    alive: bool,
    culled_at: Option<usize>,
    cull_reason: Option<String>,
    /// Parent hashes named by `created_from` (empty = a root).
    parents: Vec<String>,
}

/// The ancestry card's one walker: resolve a hash (live memory first, then the tombstone
/// on disk), draw it, then recurse into its parents.
pub(super) struct LineageTree<'a> {
    engine: &'a CoreEngine,
    /// Resolved facts per hash — a shared ancestor is read from disk at most once.
    cache: std::collections::HashMap<String, Option<LineageFacts>>,
    /// Hashes already drawn in THIS card: a repeated ancestor is marked `↺` and not
    /// expanded again, so a DAG shared by two reported nets stays one screen.
    shown: std::collections::HashSet<String>,
    budget: usize,
    truncated: bool,
    /// The mark (`★`/`☆`) and table fit of the ROOT currently being drawn.
    root_mark: &'static str,
    root_fit: Option<f32>,
}

impl<'a> LineageTree<'a> {
    fn new(engine: &'a CoreEngine) -> Self {
        Self {
            engine,
            cache: std::collections::HashMap::new(),
            shown: std::collections::HashSet::new(),
            budget: LINEAGE_NODE_BUDGET,
            truncated: false,
            root_mark: "·",
            root_fit: None,
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
                params: state.meta.params,
                alive: true,
                culled_at: None,
                cull_reason: None,
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
                    params: state.meta.params,
                    alive: state.is_alive,
                    culled_at: state.culled_at_step,
                    cull_reason: state.cull_reason.clone(),
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
    fn mermaid(&mut self, root: &str) -> (String, bool) {
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
        // The diagram carries its OWN palette AND a base theme. Measured on a rendered
        // `elite-<hash>.md` in a dark-themed viewer: light fills plus a `color:` in
        // `classDef` produced near-white labels on a near-white box — the label colour was
        // ignored, the theme's own text colour won, and the prose was unreadable.
        //
        // So every fill is DARK and every class states a white label: a renderer that honours
        // `color` gets white-on-dark, and one that overrides it with the theme's near-white
        // text gets the same readable result by construction. The `init` directive pins the
        // rest (canvas, edges, edge-label background) so the graph does not half-inherit a
        // dark theme.
        let mut out = String::from("```mermaid\n");
        out.push_str(
            "%%{init: {\"theme\": \"base\", \"themeVariables\": {\"background\": \"#ffffff\", \"primaryColor\": \"#1f4f7a\", \"primaryTextColor\": \"#ffffff\", \"primaryBorderColor\": \"#7fb3d5\", \"lineColor\": \"#5a8fbf\", \"textColor\": \"#1b1b1b\", \"edgeLabelBackground\": \"#ffffff\", \"fontSize\": \"14px\"}}}%%\n",
        );
        out.push_str("graph TB\n");
        // Classes first, so a viewer never renders an unstyled node while reading down.
        out.push_str(
            "    classDef elite fill:#8a5a00,stroke:#ffd166,stroke-width:2px,color:#ffffff\n",
        );
        out.push_str("    classDef culled fill:#4a4a4a,stroke:#9a9a9a,color:#ffffff\n");
        out.push_str("    classDef ancestor fill:#1f4f7a,stroke:#7fb3d5,color:#ffffff\n");
        for hash in &nodes {
            let facts = self.facts(hash);
            out.push_str(&format!(
                "    n_{}[\"{}\"]\n",
                short_hash(hash),
                mermaid_node_label(hash, facts.as_ref(), hash == root)
            ));
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
        for hash in &nodes {
            let class = if hash == root {
                "elite"
            } else if self.facts(hash).map(|f| !f.alive).unwrap_or(false) {
                "culled"
            } else {
                "ancestor"
            };
            out.push_str(&format!("    class n_{} {class}\n", short_hash(hash)));
        }
        out.push_str("```\n");
        (out, cut)
    }

    /// ONE reported net's ancestry, generation by generation: oldest ancestors at the
    /// TOP and the reported net on the LAST line, so the card reads downward in the order
    /// the nets were born.
    ///
    /// Depth is a printed `g<N>` tag (1 = the net's own parents), NOT indentation. The old
    /// drawing deepened the `│` rail for every generation, which put the OLDEST ancestors
    /// in the widest column and the reported net back at the margin — the eye had to walk a
    /// staircase inward to find the net, and a parent group could only be tied to its child
    /// by counting indents. Here every group is introduced by the child it explains, so no
    /// line has to be matched against a line further down.
    ///
    /// A shared ancestor therefore prints ONCE per generation (each level is de-duplicated,
    /// and `↺ shown above` marks a hash already drawn deeper) instead of the same block
    /// appearing twice under two different parents.
    fn render_chain(&mut self, root: &str, rank: usize, out: &mut Vec<String>) {
        /// One child's parent set — what a `g<N> ┌ parents of …` header introduces.
        struct Group {
            child: String,
            /// `(parent hash, crossover slot, how many parents the child has)`
            parents: Vec<(String, usize, usize)>,
        }
        // `levels[0]` = the reported net's parents, `levels[1]` = THEIR parents, ...
        let mut levels: Vec<Vec<Group>> = Vec::new();
        let mut frontier: Vec<String> = vec![root.to_string()];
        for _ in 1..LINEAGE_MAX_DEPTH {
            let mut groups: Vec<Group> = Vec::new();
            for child in &frontier {
                let Some(facts) = self.facts(child) else {
                    continue;
                };
                if facts.parents.is_empty() {
                    continue;
                }
                let n = facts.parents.len();
                groups.push(Group {
                    child: child.clone(),
                    parents: facts
                        .parents
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (p.clone(), i + 1, n))
                        .collect(),
                });
            }
            if groups.is_empty() {
                break;
            }
            let mut next: Vec<String> = groups
                .iter()
                .flat_map(|g| g.parents.iter().map(|p| p.0.clone()))
                .collect();
            next.sort();
            next.dedup();
            frontier = next;
            levels.push(groups);
        }
        // Deepest level FIRST: the oldest ancestors lead, the reported net closes the card.
        for (idx, groups) in levels.iter().enumerate().rev() {
            let generation = idx + 1;
            let deepest = idx + 1 == levels.len();
            for group in groups {
                let child = if group.child == root {
                    format!("#{rank} {}", short_hash(&group.child))
                } else {
                    short_hash(&group.child)
                };
                if self.budget == 0 {
                    self.truncated = true;
                    out.push("   …".to_string());
                    return;
                }
                self.budget -= 1;
                out.push(format!("   g{generation} ┌ parents of {child}"));
                let n = group.parents.len();
                for (i, (parent, slot, total)) in group.parents.iter().enumerate() {
                    if self.budget == 0 {
                        self.truncated = true;
                        out.push(format!("      └ … {} more parent(s)", n - i));
                        break;
                    }
                    self.budget -= 1;
                    let mut body = short_hash(parent);
                    match self.facts(parent) {
                        None => body.push_str(" · no `nets/<hash>.json` — ancestry unknown"),
                        Some(f) => {
                            body.push_str(&format!(
                                " · {}",
                                origin_cell(f.origin.as_deref(), f.born)
                            ));
                            if let Some(v) = f.fit {
                                body.push_str(&format!(" · fit {}", fmt_cell(v)));
                            }
                            if !f.alive {
                                match (f.culled_at, f.cull_reason.as_deref()) {
                                    (Some(clock), Some(reason)) => {
                                        body.push_str(&format!(" · †{clock} {reason}"))
                                    }
                                    (Some(clock), None) => body.push_str(&format!(" · †{clock}")),
                                    _ => body.push_str(" · dead"),
                                }
                            } else {
                                body.push_str(" · live");
                            }
                            // The `Δ` legs measure THIS parent against the child the group
                            // explains, in the ranking's own direction.
                            if let Some(child_facts) = self.facts(&group.child) {
                                let direction = self.engine.fitness.direction();
                                let adjusted = |v: f32| match direction {
                                    crate::utils::fitness::Direction::Maximize => v,
                                    crate::utils::fitness::Direction::Minimize => -v,
                                };
                                if let (Some(child_fit), Some(parent_fit)) =
                                    (child_facts.fit, f.fit)
                                {
                                    let gain = adjusted(child_fit) - adjusted(parent_fit);
                                    body.push_str(&format!(
                                        " · Δfit {}{}",
                                        if gain >= 0.0 { "+" } else { "−" },
                                        fmt_cell(gain.abs())
                                    ));
                                }
                                if let (Some(child_params), Some(parent_params)) =
                                    (child_facts.params, f.params)
                                {
                                    let delta = child_params - parent_params;
                                    body.push_str(&format!(
                                        " · Δparams {}{}",
                                        if delta >= 0 { "+" } else { "−" },
                                        fmt_units(delta.abs())
                                    ));
                                }
                            }
                            if self.shown.contains(parent) {
                                body.push_str(" · ↺ shown above");
                            }
                        }
                    }
                    self.shown.insert(parent.clone());
                    let branch = if i + 1 == n { "└" } else { "│" };
                    let slot = if *total == 1 {
                        "p".to_string()
                    } else {
                        format!("p{slot}")
                    };
                    out.push(format!("      {branch} {slot} {body}"));
                }
                // The cap is named where the chain is cut: these parents may have parents of
                // their own, and without this line they would read as roots.
                if deepest && generation + 1 >= LINEAGE_MAX_DEPTH {
                    let deeper = group
                        .parents
                        .iter()
                        .any(|(p, _, _)| self.facts(p).map(|f| !f.parents.is_empty()).unwrap_or(false));
                    if deeper {
                        out.push(format!(
                            "      └ … beyond the {LINEAGE_MAX_DEPTH}-generation cap"
                        ));
                        self.truncated = true;
                    }
                }
            }
        }
        // The reported net closes its own chain, at the margin so the eye lands on it.
        if self.budget == 0 {
            self.truncated = true;
            return;
        }
        self.budget -= 1;
        let mut line = format!("▼ #{rank} {}", short_hash(root));
        match self.facts(root) {
            None => line.push_str(" · no `nets/<hash>.json` — ancestry unknown"),
            Some(f) => {
                line.push_str(&format!(" · {}", origin_cell(f.origin.as_deref(), f.born)));
                if let Some(v) = f.fit.or(self.root_fit) {
                    line.push_str(&format!(" · fit {}", fmt_cell(v)));
                }
                line.push_str(&format!(" · {}", self.root_mark));
                if f.parents.is_empty() {
                    line.push_str(" · root (no recorded parents)");
                }
            }
        }
        out.push(line);
    }
}
