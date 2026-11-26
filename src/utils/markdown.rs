//! Markdown rendering for topologies — includes nodes table, edge list,
//! ASCII wiring diagram, and Mermaid flowchart.

use super::ascii::{AsciiNode, edge_list, render_wire_diagram};
use super::mermaid::{split_policy_name, topology_mermaid};
use crate::graph::network::Network;
use crate::graph::node::NodeKind;
use crate::graph::topology::{FeatureSplitPolicy, Topology};

/// Markdown for a [`Topology`]: header, nodes table, edge list, wiring diagram
/// and Mermaid. Stamp-free by design — scores live in the CSV, not here.
pub fn topology_markdown(graph: &Topology, net: Option<&Network>) -> String {
    let mut out = String::new();

    // Header
    out.push_str(&format!("# Topology #{}\n\n", graph.id));

    // Summary
    let hidden_range: Vec<usize> = graph.nodes.iter().filter_map(|n| n.hidden_dim).collect();
    let hidden_info = if hidden_range.is_empty() {
        "hidden 8".to_string()
    } else {
        let min = *hidden_range.iter().min().unwrap();
        let max = *hidden_range.iter().max().unwrap();
        if min == max {
            format!("hidden {min}")
        } else {
            format!("hidden {min}–{max}")
        }
    };
    out.push_str(&format!(
        "**{} nodes** · **{} wires** · input {} → {} → output {}\n\n",
        graph.nodes.len(),
        graph.connections.len(),
        graph
            .options
            .input_dim
            .map(|x| x.to_string())
            .unwrap_or_else(|| "?".into()),
        hidden_info,
        graph
            .options
            .output_dim
            .map(|x| x.to_string())
            .unwrap_or_else(|| "?".into()),
    ));

    // ── The two blueprint knobs that never appear in a box ── A reader diffing two `champion-*-*-*.md`
    // files needs the split and the seed: the split decides WHICH features each branch can see,
    let mut knobs: Vec<String> = Vec::new();
    match graph.input_feature_masks() {
        Some(masks) => knobs.push(format!(
            "feature split `{}` over {} Input port(s)",
            split_policy_name(graph.options.feature_split),
            masks.len()
        )),
        None => knobs.push("no feature split — every Input port sees every feature".to_string()),
    }
    // ALWAYS printed, even at 0.00: a gene that vanishes at its own default cannot be diffed
    // between two files, and dropout is a real gene on every net (an elite at 0.00 and one at 0.25
    knobs.push(format!("dropout {:.2}", graph.options.dropout_prob));
    // Same rule for the loop gene: `on`/`off` is a gene, so a champion file that hides it
    // at its default cannot be diffed against a sibling that recurs.
    knobs.push(format!(
        "recurrence {}",
        if graph.options.recur { "on" } else { "off" }
    ));
    // Seed PLUS the bounds it was drawn from: a seed alone reproduces a blueprint
    // ranges that constrained the draw travel with it, and this file is meant to outlive the run
    let o = &graph.options;
    out.push_str(&format!(
        "Blueprint: {} · topology seed {} · sampled {}–{} nodes · {}–{} in · {}–{} out · reach {:.2}–{:.2}\n\n",
        knobs.join(" · "),
        o.topology_seed,
        o.min_hidden_num_nodes,
        o.max_hidden_num_nodes,
        o.min_hidden_inputs_per_node,
        o.max_hidden_inputs_per_node,
        o.min_hidden_outputs_per_node,
        o.max_hidden_outputs_per_node,
        o.min_input_reach,
        o.max_input_reach,
    ));

    // ── Recurrence — the loop gene, ALWAYS printed so on / off / inert are diffable ──
    let in_dim = graph.options.input_dim;
    let out_dim = graph.options.output_dim;
    let qualifies = matches!((in_dim, out_dim), (Some(i), Some(o)) if i == o);
    // Hidden nodes whose OWN widths match: each hands its pre-activation back through its own
    // layer once. This is the loop that can fire on a 784 → 10 net.
    let node_loops = super::graph_utils::looping_nodes(graph);
    let dims = format!(
        "input {} · output {}",
        in_dim.map(|d| d.to_string()).unwrap_or_else(|| "?".into()),
        out_dim.map(|d| d.to_string()).unwrap_or_else(|| "?".into()),
    );
    out.push_str("## Recurrence\n\n");
    if !graph.options.recur {
        out.push_str(
            "**Off — one sweep.** No feedback edge: the output is the graph's own answer. The gene is inherited like any other, and a crossover child may drift it on.\n\n",
        );
    } else if !node_loops.is_empty() {
        let n = node_loops.len();
        let list = node_loops
            .iter()
            .map(|(i, d)| format!("n{i} {d} → {d}"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "**On — {n} hidden node loop{}** ({dims}): {list}. Each hands its PRE-activation back through the SAME layer once — shared weights, no extra parameters, and the gradients unroll over both passes. The gate is `in_dim == out_dim` on the node itself, so this fires inside nets whose outer widths could never match.\n\n",
            if n == 1 { "" } else { "s" }
        ));
        if qualifies {
            out.push_str("The network-level loop runs too: the input and output widths match, so the Output node's pre-activation also comes back in as the network input, doubling the whole sweep.\n\n");
        } else {
            out.push_str("The network-level loop stays off — the input and output widths differ — so the graph itself runs ONE sweep; the loops live inside the hidden nodes above.\n\n");
        }
    } else if qualifies {
        out.push_str(&format!(
            "**On — one loop, two sweeps** ({dims}). The Output node's PRE-activation is fed back as the network input, so the output takes the same hidden path twice through the SAME layers — shared weights, no extra parameters, and the gradients unroll over both sweeps. The widths match, so the feedback fits the first layer.\n\n"
        ));
    } else {
        out.push_str(&format!(
            "**On, but inert** ({dims}). The gene is inherited, but no width qualifies: the feedback neither fits the first layer nor any hidden node's own input, so this net runs ONE sweep. Drift can flip the gene either way.\n\n"
        ));
    }
    // The NETWORK-level loop's curve — only a LIVE net can measure it; the champion files render
    // the blueprint alone (weights are a separate artifact), so they get the gene above.
    if let Some(net) = net.filter(|n| n.recur_outer) {
        if let Ok(curve) = net.recurrence_curve() {
            out.push_str("Zero probe, batch 1 — the loop's own trajectory:\n\n");
            out.push_str(
                "| sweep | mean \\|y\\| | max \\|y\\| | mean \\|Δ\\| |\n|---:|---:|---:|---:|\n",
            );
            for (i, (mean, max, delta)) in curve.iter().enumerate() {
                out.push_str(&format!(
                    "| {} | {mean:.4} | {max:.4} | {delta:.4} |\n",
                    i + 1
                ));
            }
            if curve.len() > 1 {
                let (before, after) = (curve[0].0, curve[curve.len() - 1].0);
                let trend = if after > before {
                    "grows"
                } else if after < before {
                    "shrinks"
                } else {
                    "holds"
                };
                out.push_str(&format!(
                    "\nThe loop {trend} the output's magnitude between sweeps ({before:.4} → {after:.4}) and moved it by **{:.4}** on average.\n\n",
                    curve[curve.len() - 1].2
                ));
            }
        }
    }

    // ── Nodes table (merged topology + network layer info) ──
    out.push_str("## Nodes\n\n");
    if let Some(net) = net {
        // Full table: ID, Kind, In, Out, Linear, Activation, Combine, Std, Sources
        out.push_str(
            "| ID | Kind | In | Out | Recur | Linear | Activation | Combine | Std | Sources |\n",
        );
        out.push_str(
            "|----|------|----|-----|-------|--------|------------|---------|-----|---------|\n",
        );
        let looping: Vec<usize> = super::graph_utils::looping_nodes(graph)
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        let node_sources = super::graph_utils::build_node_sources(
            &net.connections,
            &net.nodes.iter().map(|n| n.num_inputs).collect::<Vec<_>>(),
        );
        for (i, node) in graph.nodes.iter().enumerate() {
            let kind = match node.kind {
                NodeKind::Input => "Input",
                NodeKind::Hidden => "Hidden",
                NodeKind::Output => "Output",
            };
            // Activation column: node-level, plus `[a|b|…]` when ports
            // diverge — mirrors the mermaid per-port sig.
            let act = match &node.port_activations {
                Some(ports) if ports.iter().any(|a| *a != node.activation) => {
                    let list: Vec<String> = ports.iter().map(|a| format!("{a}")).collect();
                    format!("{} [{}]", node.activation, list.join("|"))
                }
                _ => format!("{}", node.activation),
            };
            let combine = node
                .combine_op
                .map_or(crate::utils::cfy::NO_VALUE.into(), |op| format!("{op}"));
            let std = node
                .standardize
                .map_or(crate::utils::cfy::NO_VALUE.into(), |s| format!("{s}"));
            let (in_dim, out_dim) = graph.node_dims().get(i).copied().unwrap_or((0, 0));
            let recur = if looping.contains(&i) { "yes" } else { "no" };
            let marker = if i == net.output_node { " " } else { "" };
            let sources: Vec<String> = node_sources[i]
                .iter()
                .flatten()
                .map(|p| format!("n{}_o{}", p.node, p.index))
                .collect();
            let src_str = if sources.is_empty() {
                "*".into()
            } else {
                sources.join(", ")
            };
            let in_wires = graph
                .connections
                .iter()
                .filter(|c| c.to.node == node.id)
                .count();
            let out_wires = graph
                .connections
                .iter()
                .filter(|c| c.from.node == node.id)
                .count();
            out.push_str(&format!(
                "| {id} {kind}{marker} | {kind} | {ni} | {no} | {recur} | {in_dim} → {out_dim} | {act} | {combine} | {std} | {src_str} |\n",
                id = i,
                ni = ports_cell(node.num_inputs, in_wires),
                no = ports_cell(node.num_outputs, out_wires),
            ));
        }
    } else {
        // Topology-only table: no linear dims or sources
        out.push_str("| ID | Kind | In | Out | Recur | Activation | Combine | Std | Dims |\n");
        out.push_str("|----|------|----|-----|-------|------------|---------|-----|------|\n");
        let looping: Vec<usize> = super::graph_utils::looping_nodes(graph)
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        for (node_id, node) in graph.nodes.iter().enumerate() {
            let kind = match node.kind {
                NodeKind::Input => "Input",
                NodeKind::Hidden => "Hidden",
                NodeKind::Output => "Output",
            };
            let act = node.activation;
            let combine = node
                .combine_op
                .map_or(crate::utils::cfy::NO_VALUE.into(), |op| format!("{op}"));
            let std = node
                .standardize
                .map_or(crate::utils::cfy::NO_VALUE.into(), |s| format!("{s}"));
            let (in_dim, out_dim) = graph.node_dims().get(node_id).copied().unwrap_or((0, 0));
            let in_wires = graph
                .connections
                .iter()
                .filter(|c| c.to.node == node.id)
                .count();
            let out_wires = graph
                .connections
                .iter()
                .filter(|c| c.from.node == node.id)
                .count();
            let recur = if looping.contains(&node_id) {
                "yes"
            } else {
                "no"
            };
            out.push_str(&format!(
                "| {id} | {kind} | {ni} | {no} | {recur} | {act} | {combine} | {std} | {in_dim}→{out_dim} |\n",
                id = node.id,
                ni = ports_cell(node.num_inputs, in_wires),
                no = ports_cell(node.num_outputs, out_wires),
            ));
        }
    }
    out.push('\n');

    // ── Feature split — the Input node's per-port genome ── The rows are genome (`M:<width>`
    // the diversity tokens) but a table cannot show them: they live per PORT, not per node.
    if let Some(masks) = graph.input_feature_masks() {
        let input_dim = graph.options.input_dim.unwrap_or(1).max(1);
        let seen = (0..input_dim)
            .filter(|f| masks.iter().any(|m| m.get(*f).copied().unwrap_or(0) != 0))
            .count();
        let total: usize = masks
            .iter()
            .map(|m| m.iter().filter(|b| **b != 0).count())
            .sum();
        let mean = total as f32 / masks.len().max(1) as f32;
        // The accounting is the check a reader can do at a glance: `exclusive` is a partition,
        // the rows must add up to exactly one full vector (100%).
        let pct = 100.0 * total as f32 / input_dim as f32;
        // A partition is a PER-FEATURE claim, so the total cannot prove it: rows `[1,1,0] [0,1,0]` add up to one full vector yet own feature 1 twice and feature 2 never.
        let owned_by: Vec<usize> = (0..input_dim)
            .map(|f| {
                masks
                    .iter()
                    .filter(|m| m.get(f).copied().unwrap_or(0) != 0)
                    .count()
            })
            .collect();
        let shared: Vec<usize> = (0..input_dim).filter(|f| owned_by[*f] > 1).collect();
        let unowned: Vec<usize> = (0..input_dim).filter(|f| owned_by[*f] == 0).collect();
        // The accounting means something DIFFERENT per policy, so the sentence says what this net's
        // own label makes of it: 100% is a partition, >100% is sharing, <100% is dropped features —
        // Capped so a 784-wide vector cannot turn one sentence into a wall of indices.
        let name_features = |feats: &[usize]| -> String {
            const CAP: usize = 8;
            let mut shown: Vec<String> = feats.iter().take(CAP).map(|f| format!("`{f}`")).collect();
            if feats.len() > CAP {
                shown.push(format!("… +{}", feats.len() - CAP));
            }
            shown.join(", ")
        };
        let accounting = match graph.options.feature_split {
            Some(FeatureSplitPolicy::Exclusive) => {
                if shared.is_empty() && unowned.is_empty() {
                    "`exclusive` is a partition: every feature is owned by exactly one port."
                        .to_string()
                } else {
                    let shared_note = if shared.is_empty() {
                        String::new()
                    } else {
                        format!(" (first {})", name_features(&shared))
                    };
                    let unowned_note = if unowned.is_empty() {
                        String::new()
                    } else {
                        format!(" (first {})", name_features(&unowned))
                    };
                    format!(
                        "`exclusive` is a partition, but {} feature(s) are shared by two or more ports{} and {} feature(s) reach no port{}.",
                        shared.len(),
                        shared_note,
                        unowned.len(),
                        unowned_note
                    )
                }
            }
            Some(FeatureSplitPolicy::Random) => format!(
                "`random` flips a coin per (port, feature), so sharing is normal — {seen} of {input_dim} features are seen by some port."
            ),
            Some(FeatureSplitPolicy::Chunks) => {
                if shared.is_empty() && unowned.is_empty() {
                    "`chunks` is an ordered, contiguous partition: every feature is owned by exactly one block."
                        .to_string()
                } else {
                    format!(
                        "`chunks` claims an ordered contiguous partition, but {} feature(s) are shared and {} feature(s) reach no port.",
                        shared.len(),
                        unowned.len()
                    )
                }
            }
            Some(FeatureSplitPolicy::Crossover) => "`crossover` rows are a per-port crossover blend, not the run's draw policy: a blend of partitions need not be one."
                .to_string(),
            None => "No policy recorded — read the rows.".to_string(),
        };
        out.push_str("## Feature split\n\n");
        out.push_str(&format!(
            "`{}` over **{}** Input port(s): **{seen} of {input_dim}** features reach a port, **{total} keeps = {pct:.0}%** of the vector, mean **{mean:.1}** per port. {accounting} Rows are genome (`1` keeps a feature, `0` hides it); `features` writes the row out — a bit string at ≤{MASK_BIT_MAX_DIM} wide, else the kept indices (capped at {MASK_INDEX_CAP}).\n\n",
            split_policy_name(graph.options.feature_split),
            masks.len()
        ));
        // Which wires each port feeds: the Input node is the one node that FANS OUT, so its port
        // count and its wire count can differ. A diagram draws one slot per WIRE, so a row has to
        // name the consumers its single masked slice is read by — else three slots look like a
        // third mask row the table forgot.
        let input_node = graph
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Input)
            .map(|n| n.id);
        let feeds: Vec<String> = (0..masks.len())
            .map(|port| {
                let consumers: Vec<String> = graph
                    .connections
                    .iter()
                    .filter(|c| Some(c.from.node) == input_node && c.from.index == port)
                    .map(|c| c.to_label())
                    .collect();
                if consumers.is_empty() {
                    crate::utils::cfy::NO_VALUE.to_string()
                } else {
                    consumers.join(", ")
                }
            })
            .collect();
        out.push_str("| port | keeps | share | features | wires |\n|---:|---:|---:|---|---|\n");
        for (port, mask) in masks.iter().enumerate() {
            let keeps = mask.iter().filter(|b| **b != 0).count();
            let note = if keeps == 0 { " (bias only)" } else { "" };
            out.push_str(&format!(
                "| {port} | {keeps}{note} | {:.1}% | {} | {} |\n",
                100.0 * keeps as f32 / input_dim as f32,
                kept_features(mask, input_dim),
                feeds[port]
            ));
        }
        // Rows are PORTS, the diagram draws WIRES: say so where the two disagree.
        let wires: usize = graph
            .connections
            .iter()
            .filter(|c| Some(c.from.node) == input_node)
            .count();
        if wires != masks.len() {
            out.push_str(&format!(
                "\n**{} port(s) carry {wires} wire(s)** — a LEGACY blueprint, saved before one-wire-per-socket: the Input node fans out — {}, and a port's masked slice is computed ONCE and read by every wire it feeds. This table is one row per PORT; the diagram draws one slot per WIRE (its `Out` cell counts both). A graph built now never prints this line.\n\n",
                masks.len(),
                (0..masks.len())
                    .map(|p| format!("port {p} → {}", feeds[p]))
                    .collect::<Vec<_>>()
                    .join(" · ")
            ));
        }
        out.push('\n');
    } else {
        // The other half of the consistency rule: a net WITHOUT the gene says so where a net WITH it lists its rows.
        let input_dim = graph.options.input_dim.unwrap_or(1).max(1);
        let ports = graph
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Input)
            .map(|n| n.num_outputs.max(1))
            .unwrap_or(1);
        out.push_str("## Feature split\n\n");
        out.push_str(&format!(
            "`{}` over **{ports}** Input port(s) — no mask rows are recorded: every port projects the FULL **{input_dim}**-feature vector (`keeps` = {input_dim}, 100% each). This is the ABSENCE of the split gene, not a partition of it.\n\n",
            split_policy_name(graph.options.feature_split)
        ));
    }

    // ── Edge list with distance markers ──
    out.push_str("## Edges\n\n");
    out.push_str("```text\n");
    out.push_str(&edge_list(graph));
    out.push_str("```\n\n");

    // ── Wiring diagram (with out dims) ──
    out.push_str("## Wiring diagram\n\n");
    out.push_str("```text\n");
    let node_dims = graph.node_dims();
    let looping: Vec<usize> = super::graph_utils::looping_nodes(graph)
        .into_iter()
        .map(|(i, _)| i)
        .collect();
    let nodes: Vec<AsciiNode> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let in_wires = graph
                .connections
                .iter()
                .filter(|c| c.to.node == n.id)
                .count();
            let out_wires = graph
                .connections
                .iter()
                .filter(|c| c.from.node == n.id)
                .count();
            AsciiNode {
                id: n.id,
                kind: n.kind,
                num_inputs: n.num_inputs,
                num_outputs: n.num_outputs,
                out_dim: node_dims.get(i).map(|&(_, out)| out),
                loops: looping.contains(&i),
                in_wires,
                out_wires,
            }
        })
        .collect();
    out.push_str(&render_wire_diagram(&nodes, &graph.connections));
    out.push_str("```\n");
    out.push_str("\n> **Legend**:\n");
    out.push_str("> - I=input  H=hidden  O=output\n");
    out.push_str("> - `2w→2i/1o→5w` = wires→input ports / output ports→wires. A port is a SOCKET: on a graph built now every port carries exactly ONE wire, so `w == i` and `o == w`. A label repeats (`o1 o1`) only in a legacy blueprint saved before one-wire-per-socket. Every marker maps to exactly one wire\n");
    out.push_str("> - ->dim = output dimension\n");
    out.push_str("> - `x2` = recurrence: this hidden node runs its own layer a SECOND time (its pre-activation goes back through it). A label fact, not a wire — no wire count or hop marker includes it\n");
    out.push_str("> - ▶ connected output  ◀ connected input  * orphaned port\n");

    // Mermaid flowchart
    out.push_str("\n## Graph\n\n");
    out.push_str(&topology_mermaid(graph));

    // ── Legend — one entry per visual channel, so a reader never has to guess
    // which structural fact a border, wire or label encodes. ──
    out.push_str("\n## How to read this\n\n");
    out.push_str("| Visual | Meaning |\n");
    out.push_str("|---|---|");
    out.push_str("\n| **Box border thickness (Graph)** | node width (`hidden_dim`, scaled min→max within this net). |");
    out.push_str("\n| **Wire thickness (Graph)** | hop distance — thick = long-range skip, thin = neighbor. |");
    out.push_str("\n| **Box height (Wiring diagram)** | the node's port count (`max(i, o)`, scaled min→max within this net) — a tall box is a wide fan-in/fan-out node. Wire thickness is a Graph-only channel by design; in the text diagram, hop distance is the `N hops <<<<` marker in the edge list. |");
    out.push_str("\n| **Box label (Graph)** | `H<id> <combine>→lin→<std> <dim>` — the node's internal ops in EXECUTION order: gather (a port whose source width differs is bridged) → `combine` → `lin` → `std` → dropout → per-port activation. Activation is NOT here: it lives on each outgoing arrow. |");
    out.push_str("\n| **Arrow label (Graph)** | the wire's own activation — per-port, so one node can emit `mish` to one neighbor and `tanh` to another. Equal labels from the same node = same signal; duplicates are impossible (dedup collapses them). |");
    out.push_str("\n| **Dashed stub `selu (input)` / `mish (orphan)`** | orphaned port — an input nothing feeds / an output nothing reads (mermaid's counterpart of the ASCII `*`). |");
    out.push_str("\n| **`in <n>` / `out <n>`** | dataset shape (fixed by the problem). |");
    out.push_str("\n| **Input box → second line** | the Input node's fan-out and projection width — `<ports>p → <out dim>`, the same numbers the wiring diagram prints as `4o ->64` and every hidden box carries as its trailing width. The dataset width stays on the first line, so `in 784` is the raw vector, not the node's width. |");
    out.push_str("\n| **Input box → third line** | the feature split, when the net has one: `split <policy> · <ports>p · <seen>/<n> seen`. Present for ANY policy that carries rows (`exclusive`, `random`, `chunks`, `crossover`); absent for `full`, where every port already carries the full mixed vector — that is the common case, and why the box is usually two lines. Per-port rows are in *Feature split* above. |");
    out.push_str("\n| **`na` in `Combine` / `Std`** | no per-node override, so the forward pass uses `Add` / no standardisation. `na` is not a missing gene: it is the gene spelled as its default (the diversity metric folds `None` and `Identity` into one token for the same reason). |");
    out.push_str("\n| **Nodes table → `Sources`** | which output port feeds each input port (`n1_o2` = output 2 of node 1), one entry per port in port order — the `k`-th entry feeds `i<k>`; `*` = an unwired port (never left behind by `finalize`). |");
    out.push_str("\n| **Nodes table → `In` / `Out` with `(Nw)`** | PORT counts; a parenthesised wire count appears only where the two differ (`Out 2 (3w)` = 2 ports, 3 wires). A graph built now is one-wire-per-socket, so a `(Nw)` cell can only come from a legacy blueprint — the wiring diagram draws one slot per WIRE, which is why its `Out` slots can outnumber its ports. |");
    out.push_str("\n| **Nodes table → `In` / `Out`** | PORT counts. Only a legacy blueprint can read `2 (3w)` — two ports carrying three wires, the same numbers the wiring diagram labels `2o→3w` and the edge list shows as a repeated port (`n0_o0` twice). A hidden/output row reads plain numbers because an input port carries exactly one wire. |");
    out.push_str("\n| **Nodes table → `Combine` / `Std`** | how multiple inputs are merged / whether they're normalized first. |");
    out.push_str("\n| **Nodes table → `Recur`** | `yes` = this hidden node runs its own layer a SECOND time (its pre-activation goes back through it, shared weights); `no` = one pass — the gene is off, the node is Input/Output, or its widths differ (`64 → 48`). |");
    out.push_str("\n| **`x2` / `↻` (Wiring diagram, Graph)** | recurrence: the node loops once. `x2` is the ASCII label suffix; `↻` the mermaid label suffix and the dashed self-loop `n2 -.- n2`. Not a wire — wire counts and hop markers do not include it. |");
    out.push_str("\n| **Edge list & wiring diagram** | one line per wire, exact ports (`n0_o1 → n2_i0`); `<<<<` = long jump; `▶ ◀ *` = wired out / wired in / orphaned; a `loops (N):` header names the hidden nodes that loop (`x2`), and `[x2 source]` marks a wire leaving one. |");
    out.push('\n');

    out
}

/// A node's `In` / `Out` cell: the PORT count, annotated with the wire count only when the node carries MORE wires than ports.
fn ports_cell(ports: usize, wires: usize) -> String {
    if wires > ports {
        format!("{ports} ({wires}w)")
    } else {
        ports.to_string()
    }
}

/// Above this `input_dim`, a mask row is printed as kept INDICES instead of a bit
/// string: 32 bits is still one glance, 784
const MASK_BIT_MAX_DIM: usize = 32;

/// How many indices the index form may list before it says `… +n`.
const MASK_INDEX_CAP: usize = 32;

/// One mask row, spelled so the reader can check WHICH features a port keeps, not just how many:
/// bit string while the vector is short (`1010`), else the kept indices (`0,2,… +41`).
fn kept_features(mask: &[u8], input_dim: usize) -> String {
    if input_dim <= MASK_BIT_MAX_DIM {
        return (0..input_dim)
            .map(|f| {
                if mask.get(f).copied().unwrap_or(0) != 0 {
                    '1'
                } else {
                    '0'
                }
            })
            .collect();
    }
    let kept: Vec<usize> = (0..input_dim)
        .filter(|f| mask.get(*f).copied().unwrap_or(0) != 0)
        .collect();
    if kept.is_empty() {
        return crate::utils::cfy::NO_VALUE.to_string();
    }
    let shown: Vec<String> = kept
        .iter()
        .take(MASK_INDEX_CAP)
        .map(|i| i.to_string())
        .collect();
    if kept.len() > MASK_INDEX_CAP {
        format!("{} … +{}", shown.join(","), kept.len() - MASK_INDEX_CAP)
    } else {
        shown.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::node::Node;
    use crate::graph::topology::{Connection, Port, TopologyOptions};

    /// The Input box says what the ASCII node says: how many ports fan out of it and how wide
    /// each projection is, so `in 3` alone is never read as a single-wire input.
    #[test]
    fn the_topology_mermaid_names_the_input_fanout() {
        let mut g = Topology::new(0, None);
        g.options.input_dim = Some(3);
        g.options.output_dim = Some(2);
        g.nodes.push(Node::new_input(0, 2));
        let mut hidden = Node::new_hidden(1, 2, 1);
        hidden.hidden_dim = Some(8);
        g.nodes.push(hidden);
        g.nodes.push(Node::new_output(2, 1, 1));
        g.finalize();
        let m = crate::utils::mermaid::topology_mermaid(&g);
        assert!(m.contains("n0[\"in 3<br/>2p → 8\"]"), "{m}");
    }

    /// The loop gene is about HIDDEN nodes: a 784 → 10 net cannot sweep itself, but its 64 → 64
    /// hidden nodes each loop, and the section says so instead of calling the gene inert.
    #[test]
    fn hidden_node_loops_are_named_when_the_net_cannot_sweep() {
        let mut g = Topology::new(0, None);
        g.options.input_dim = Some(784);
        g.options.output_dim = Some(10);
        g.options.recur = true;
        let mut input = Node::new_input(0, 2);
        input.hidden_dim = Some(64);
        g.nodes.push(input);
        let mut hidden = Node::new_hidden(1, 2, 2);
        hidden.hidden_dim = Some(64);
        g.nodes.push(hidden);
        let mut out = Node::new_output(2, 2, 1);
        out.hidden_dim = Some(10);
        g.nodes.push(out);
        g.finalize();
        let md = topology_markdown(&g, None);
        assert!(md.contains("1 hidden node loop"), "{md}");
        assert!(md.contains("n1 64 → 64"), "{md}");
        assert!(md.contains("network-level loop stays off"), "{md}");
        assert!(!md.contains("on, inert"), "{md}");
    }

    /// The loop is a GENE in the file: on / off / inert all print, and a live net's zero-probe
    /// curve shows how far the second sweep moved the output.
    #[test]
    fn recurrence_is_printed_and_measured() {
        let mut g = Topology::new(0, None);
        g.options.input_dim = Some(4);
        g.options.output_dim = Some(4);
        g.options.recur = true;
        g.nodes.push(Node::new_input(0, 2));
        let mut hidden = Node::new_hidden(1, 2, 2);
        hidden.hidden_dim = Some(4);
        g.nodes.push(hidden);
        let mut out = Node::new_output(2, 1, 1);
        out.hidden_dim = Some(4);
        g.nodes.push(out);
        g.refresh_labels();
        g.finalize();
        let net = Network::build(&g, flodl::Device::CPU).unwrap();
        let md = topology_markdown(&g, Some(&net));
        assert!(md.contains("recurrence on"), "{md}");
        assert!(
            md.contains("## Recurrence") && md.contains("one loop, two sweeps"),
            "{md}"
        );
        assert!(md.contains("| sweep | mean \\|y\\|"), "{md}");
        assert_eq!(net.recurrence_curve().unwrap().len(), 2);

        // Off: the same section says one sweep.
        let mut off = g.clone();
        off.options.recur = false;
        let md = topology_markdown(&off, None);
        assert!(
            md.contains("recurrence off") && md.contains("one sweep"),
            "{md}"
        );

        // Inert: on with mismatched widths names the gate.
        let mut narrow = g.clone();
        narrow.options.output_dim = Some(2);
        narrow.nodes[2].hidden_dim = Some(2);
        let md = topology_markdown(&narrow, None);
        assert!(md.contains("On, but inert"), "{md}");
    }

    /// The topology graph curves its wires like the lineage graph does.
    #[test]
    fn the_topology_mermaid_curves_its_wires_like_the_lineage_graph() {
        let m = super::topology_mermaid(&sample());
        // FRONT MATTER, not an `%%{init}%%` directive: directives are deprecated since 10.5 and are ignored at the default `securityLevel: "strict"`, so a pinned renderer written…
        assert!(m.starts_with("```mermaid\n---\nconfig:\n"), "{m}");
        // …and the LAYOUT is named, not inherited: Mermaid 12 defaults to ELK, whose wires are
        // right-angle segments, and `curve` can only round their corners.
        assert!(
            m.contains("\n  layout: dagre\n"),
            "the layout is pinned: {m}"
        );
        assert!(m.contains("\n    curve: basis\n"), "{m}");
        // The block has to close BEFORE the diagram keyword, or `---` is parsed as diagram text.
        let config = m.find("  layout: dagre").expect("config present");
        let diagram = m.find("flowchart LR").expect("diagram keyword present");
        assert!(
            config < diagram,
            "front matter must precede the diagram: {m}"
        );
        assert!(m.contains("---\nflowchart LR\n"), "{m}");
        assert!(
            !m.contains("%%{init"),
            "directives are deprecated, and ignored at strict: {m}"
        );
        assert!(!m.contains("graph LR"), "the legacy keyword is gone: {m}");
    }

    fn sample() -> Topology {
        let mut g = Topology::new(
            7,
            Some(TopologyOptions {
                input_dim: Some(3),
                output_dim: Some(1),
                ..Default::default()
            }),
        );
        g.create_random_hidden_nodes(3);
        g.finalize();
        g
    }

    /// The consistency rule: a net WITHOUT the gene names it in the same section a net WITH it lists rows, instead of the section vanishing and leaving "no split"…
    #[test]
    fn a_net_with_no_feature_split_still_gets_the_section() {
        let md = topology_markdown(&sample(), None);
        assert!(
            md.contains("## Feature split"),
            "the heading is unconditional: {md}"
        );
        assert!(
            md.contains("no mask rows are recorded"),
            "and it says why there are no rows: {md}"
        );
        assert!(
            !md.contains("| port | keeps"),
            "the rows table stays exclusive to nets that carry rows"
        );
    }

    /// The allele the counts used to hide: a split row now names WHICH features it keeps,
    /// the difference `[1,0,1,0]` makes against `[0,1,0,0]` — both are "keeps 2 of 4"-ish rows
    #[test]
    fn a_split_row_names_its_features_instead_of_only_counting_them() {
        use crate::graph::node::Node;
        use crate::graph::topology::{Connection, Port};
        let mut g = Topology::new(
            9,
            Some(TopologyOptions {
                input_dim: Some(4),
                output_dim: Some(1),
                dropout_prob: 0.0,
                ..Default::default()
            }),
        );
        g.nodes.push(Node::new_input(0, 2));
        g.nodes.push(Node::new_hidden(1, 2, 1));
        g.nodes.push(Node::new_output(2, 1, 1));
        g.finalize();
        // `finalize` picks its OWN port mapping, so the fixture installs the wires it wants to
        // read AFTER the build — exactly where the engine installs masks.
        g.connections = vec![
            Connection {
                from: Port { node: 0, index: 0 },
                to: Port { node: 1, index: 0 },
            },
            Connection {
                from: Port { node: 0, index: 1 },
                to: Port { node: 2, index: 0 },
            },
            Connection {
                from: Port { node: 1, index: 0 },
                to: Port { node: 2, index: 1 },
            },
        ];
        // Masks are genome but `finalize` drops them (a rebuild may renumber ports),
        // the fixture installs them exactly where the engine does: after the build.
        g.nodes[0].feature_masks = Some(vec![vec![1, 0, 1, 0], vec![0, 1, 0, 0]]);
        g.options.feature_split = Some(FeatureSplitPolicy::Exclusive);

        let md = topology_markdown(&g, None);
        assert!(
            md.contains("| port | keeps | share | features | wires |"),
            "{md}"
        );
        assert!(
            md.contains("| 0 | 2 | 50.0% | 1010 | n1_i0 |"),
            "port 0 keeps features 0 and 2:\n{md}"
        );
        assert!(
            md.contains("| 1 | 1 | 25.0% | 0100 | n2_i0 |"),
            "port 1 keeps feature 1, which the count alone could not say:\n{md}"
        );
        // The blueprint line is complete: dropout stated even at 0, and the seed travels
        // with the bounds it was drawn from, so the draw is reproducible from the file.
        assert!(md.contains("dropout 0.00"), "{md}");
        assert!(
            md.contains("sampled 2–15 nodes · 2–15 in · 2–15 out · reach 1.00–1.00"),
            "the topology seed needs its bounds beside it:\n{md}"
        );
    }

    /// The Input node is the only node that FANS OUT: two ports can carry three wires, so a row
    /// has to name the consumers its masked slice feeds — otherwise the diagram's three input
    /// slots read like a third mask row the table forgot.
    #[test]
    fn a_fanning_input_names_the_wires_each_port_feeds() {
        use crate::graph::node::Node;
        use crate::graph::topology::{Connection, Port};
        let mut g = Topology::new(
            12,
            Some(TopologyOptions {
                input_dim: Some(4),
                output_dim: Some(1),
                dropout_prob: 0.0,
                ..Default::default()
            }),
        );
        g.nodes.push(Node::new_input(0, 2));
        g.nodes.push(Node::new_hidden(1, 1, 1));
        g.nodes.push(Node::new_output(2, 3, 1));
        // port 0 feeds TWO consumers, port 1 one: 2 ports, 3 wires out of the Input node.
        g.finalize();
        // Port 0 feeds TWO consumers, port 1 one: 2 ports, 3 wires — the shape the real
        // champion `2 (3w)` cell reports, installed after the build like the engine does.
        g.connections = vec![
            Connection {
                from: Port { node: 0, index: 0 },
                to: Port { node: 1, index: 0 },
            },
            Connection {
                from: Port { node: 0, index: 0 },
                to: Port { node: 2, index: 1 },
            },
            Connection {
                from: Port { node: 0, index: 1 },
                to: Port { node: 2, index: 0 },
            },
        ];
        g.nodes[0].feature_masks = Some(vec![vec![1, 1, 0, 0], vec![0, 0, 1, 1]]);
        g.options.feature_split = Some(FeatureSplitPolicy::Exclusive);

        let md = topology_markdown(&g, None);
        // The row names BOTH wires port 0 feeds, and the single wire port 1 feeds.
        assert!(
            md.contains("| 0 | 2 | 50.0% | 1100 | n1_i0, n2_i1 |"),
            "port 0 feeds two wires:\n{md}"
        );
        assert!(
            md.contains("| 1 | 2 | 50.0% | 0011 | n2_i0 |"),
            "port 1 feeds one:\n{md}"
        );
        // And where ports and wires disagree, the table says so in words.
        assert!(
            md.contains("**2 port(s) carry 3 wire(s)**"),
            "the fan-out is stated next to the rows:\n{md}"
        );
    }

    /// A 100% total is NOT proof of a partition: rows `[1,1,0]` and `[0,1,0]` add up to a full 3-wide vector while owning feature 1 twice and feature 2 never.
    #[test]
    fn a_hundred_percent_total_does_not_make_an_exclusive_split_a_partition() {
        use crate::graph::node::Node;
        use crate::graph::topology::{Connection, Port};
        let mut g = Topology::new(
            11,
            Some(TopologyOptions {
                input_dim: Some(3),
                output_dim: Some(1),
                dropout_prob: 0.0,
                ..Default::default()
            }),
        );
        g.nodes.push(Node::new_input(0, 2));
        g.nodes.push(Node::new_hidden(1, 2, 1));
        g.nodes.push(Node::new_output(2, 1, 1));
        for (from, to) in [(0, 0), (1, 1)] {
            g.connections.push(Connection {
                from: Port {
                    node: 0,
                    index: from,
                },
                to: Port { node: 1, index: to },
            });
        }
        g.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        g.finalize();
        g.nodes[0].feature_masks = Some(vec![vec![1, 1, 0], vec![0, 1, 0]]);
        g.options.feature_split = Some(FeatureSplitPolicy::Exclusive);

        let md = topology_markdown(&g, None);
        assert!(
            md.contains("3 keeps = 100%"),
            "the fixture is the trap: the totals DO read 100%:\n{md}"
        );
        assert!(
            !md.contains("it does: every feature is owned by exactly one port"),
            "a 100% total must not be reported as a partition:\n{md}"
        );
        assert!(
            md.contains("1 feature(s) are shared by two or more ports (first `1`)"),
            "the shared feature is named:\n{md}"
        );
        assert!(
            md.contains("1 feature(s) reach no port (first `2`)"),
            "and so is the one left unowned:\n{md}"
        );
    }

    /// A BLENDED child names itself `crossover` — the provenance of rows the run's draw pool
    /// never contains.
    #[test]
    fn a_blended_split_is_labelled_crossover() {
        let mut g = Topology::new(
            0,
            Some(TopologyOptions {
                input_dim: Some(3),
                output_dim: Some(1),
                ..Default::default()
            }),
        );
        g.nodes.push(Node::new_input(0, 2));
        g.nodes.push(Node::new_hidden(1, 2, 1));
        g.nodes.push(Node::new_output(2, 1, 1));
        for (from, to) in [(0, 0), (1, 1)] {
            g.connections.push(Connection {
                from: Port {
                    node: 0,
                    index: from,
                },
                to: Port { node: 1, index: to },
            });
        }
        g.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        g.finalize();
        // Two rows that share feature 1 and leave feature 2 unowned: exactly what a per-port
        // blend of two partitions looks like, and why its total can pass 100%.
        g.nodes[0].feature_masks = Some(vec![vec![1, 1, 0], vec![0, 1, 0]]);
        g.options.feature_split = Some(FeatureSplitPolicy::Crossover);
        let md = topology_markdown(&g, None);
        assert!(md.contains("`crossover` over"), "{md}");
        assert!(
            !md.contains("`inherited` over"),
            "the old provenance spelling is gone from new records:\n{md}"
        );
        assert!(
            md.contains("crossover blend"),
            "the mechanism is still named so the reader knows what produced the rows:\n{md}"
        );
    }

    /// Wide vectors switch to indices, and a mask past the cap says how many it trimmed
    /// rather than silently truncating into a lie about the row.
    #[test]
    fn kept_features_caps_a_wide_row() {
        assert_eq!(kept_features(&[1, 0, 1, 0], 4), "1010");
        assert_eq!(kept_features(&[0, 0, 0, 0], 4), "0000");
        assert_eq!(kept_features(&[0u8; 40], 40), crate::utils::cfy::NO_VALUE);
        let every: Vec<u8> = vec![1; 40];
        let spelled = kept_features(&every, 40);
        assert!(spelled.starts_with("0,1,2,"), "{spelled}");
        assert!(spelled.ends_with("… +8"), "32 shown, 8 trimmed: {spelled}");
    }

    #[test]
    fn markdown_legend_describes_one_wire_per_port() {
        // The old caveat ("a port with several wires is drawn once per wire") described
        // the wiring model that no longer exists; the legend must say what replaced
        let md = topology_markdown(&sample(), None);
        assert!(
            md.contains("every port carries exactly ONE wire"),
            "legend must state the one-wire rule:\n{md}"
        );
        assert!(
            !md.contains("only the Input node fans out"),
            "the Input node is no longer the one fan-out:\n{md}"
        );
        assert!(
            !md.contains("a port with several wires is drawn once per wire"),
            "stale wire-per-port caveat still in the legend:\n{md}"
        );
    }

    /// The Input node is the ONE fan-out, so its `Out` cell must carry both numbers: the port count the topology declares and the wire count the diagram draws.
    #[test]
    fn a_fanning_input_names_its_wire_count_beside_its_ports() {
        use crate::graph::node::Node;
        use crate::graph::topology::{Connection, Port};
        let mut g = Topology::new(
            11,
            Some(TopologyOptions {
                input_dim: Some(4),
                output_dim: Some(2),
                ..Default::default()
            }),
        );
        g.nodes.push(Node::new_input(0, 2));
        g.nodes.push(Node::new_hidden(1, 2, 2));
        g.nodes.push(Node::new_output(2, 1, 1));
        // Two ports, three wires: port 0 feeds both the hidden and the output node.
        for (from, to_node, to_index) in [(0usize, 1usize, 0usize), (0, 2, 0), (1, 1, 1)] {
            g.connections.push(Connection {
                from: Port {
                    node: 0,
                    index: from,
                },
                to: Port {
                    node: to_node,
                    index: to_index,
                },
            });
        }

        let md = topology_markdown(&g, None);
        assert!(
            md.contains("| 0 | Input | 0 | 2 (3w) |"),
            "the fan-out is named where the numbers differ:\n{md}"
        );
        assert!(
            md.contains("PORT counts"),
            "the legend must explain what the cells are:\n{md}"
        );
        assert!(
            !md.contains("(1w)") && !md.contains("(0w)"),
            "no annotation where wires do not exceed ports:\n{md}"
        );
    }

    /// The loop is a PER-NODE fact, so every surface has to carry it: the Nodes table's `Recur`
    /// column, the wiring label (`x2`), the edge-list header, and the mermaid self-loop.
    #[test]
    fn recurrence_is_visible_in_the_tables_and_the_diagrams() {
        use crate::graph::node::Node;
        let mut g = Topology::new(
            21,
            Some(TopologyOptions {
                input_dim: Some(4),
                output_dim: Some(1),
                dropout_prob: 0.0,
                recur: true,
                ..Default::default()
            }),
        );
        g.nodes.push(Node::new_input(0, 1));
        g.nodes.push(Node::new_hidden(1, 1, 1));
        g.nodes.push(Node::new_hidden(2, 1, 1));
        g.nodes.push(Node::new_output(3, 1, 1));
        g.nodes[0].hidden_dim = Some(4); // Input projects 4 → 4
        g.nodes[1].hidden_dim = Some(4); // in 4, out 4: QUALIFIES
        g.nodes[2].hidden_dim = Some(2); // in 4, out 2: does not
        g.finalize();

        let md = topology_markdown(&g, None);
        assert!(md.contains("| ID | Kind | In | Out | Recur |"), "{md}");
        assert!(
            md.contains("| 1 | Hidden | 1 | 1 | yes |"),
            "the looping node says yes:\n{md}"
        );
        assert!(
            md.contains("| 2 | Hidden | 1 | 1 | no |"),
            "a mismatched node says no:\n{md}"
        );
        assert!(
            md.contains("| 0 | Input | 0 | 2 | no |"),
            "Input never loops, even with matching widths:\n{md}"
        );
        assert!(md.contains("->4 x2"), "the wiring label marks it:\n{md}");
        assert!(
            md.contains("loops (1): n1 x2 (4→4)"),
            "the edge list names it:\n{md}"
        );
        assert!(
            md.contains("[x2 source]"),
            "wires out of a looping node say so:\n{md}"
        );
        assert!(
            md.contains("n1 -.- n1"),
            "the graph draws the self-loop:\n{md}"
        );

        // Gene off: no cell says yes and no diagram carries a marker.
        let mut off = g.clone();
        off.options.recur = false;
        let md = topology_markdown(&off, None);
        assert!(!md.contains("| yes |"), "{md}");
        assert!(!md.contains("->4 x2"), "no wiring suffix:\n{md}");
        assert!(!md.contains("  loops ("), "no edge-list header:\n{md}");
        assert!(!md.contains("n1 -.- n1"), "no self-loop:\n{md}");
    }
}
