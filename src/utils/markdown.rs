//! Markdown rendering for topologies — includes nodes table, edge list,
//! ASCII wiring diagram, and Mermaid flowchart.

use super::ascii::{AsciiNode, edge_list, render_wire_diagram};
use super::mermaid::{split_policy_name, topology_mermaid};
use crate::graph::network::Network;
use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;

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

    // ── The two blueprint knobs that never appear in a box ──
    // A reader diffing two `elite-*.md` files needs the split and the seed: the split
    // decides WHICH features each branch can see, and the seed is what makes this exact
    // blueprint reproducible. Load a file without them and two nets can read identically
    // while computing different things (the masks are genome, see `Node::feature_masks`).
    let mut knobs: Vec<String> = Vec::new();
    match graph.input_feature_masks() {
        Some(masks) => knobs.push(format!(
            "feature split `{}` over {} Input port(s)",
            split_policy_name(graph.options.feature_split),
            masks.len()
        )),
        None => knobs.push(
            "no feature split — every Input port sees every feature".to_string(),
        ),
    }
    if graph.options.dropout_prob > 0.0 {
        knobs.push(format!("dropout {:.2}", graph.options.dropout_prob));
    }
    out.push_str(&format!(
        "Blueprint: {} · topology seed {}\n\n",
        knobs.join(" · "),
        graph.options.topology_seed
    ));

    // ── Nodes table (merged topology + network layer info) ──
    out.push_str("## Nodes\n\n");
    if let Some(net) = net {
        // Full table: ID, Kind, In, Out, Linear, Activation, Combine, Std, Sources
        out.push_str("| ID | Kind | In | Out | Linear | Activation | Combine | Std | Sources |\n");
        out.push_str("|----|------|----|-----|--------|------------|---------|-----|---------|\n");
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
            let combine = node.combine_op.map_or("—".into(), |op| format!("{op}"));
            let std = node.standardize.map_or("—".into(), |s| format!("{s}"));
            let (in_dim, out_dim) = graph.node_dims().get(i).copied().unwrap_or((0, 0));
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
            out.push_str(&format!(
                "| {id} {kind}{marker} | {kind} | {ni} | {no} | {in_dim} → {out_dim} | {act} | {combine} | {std} | {src_str} |\n",
                id = i,
                ni = node.num_inputs,
                no = node.num_outputs,
            ));
        }
    } else {
        // Topology-only table: no linear dims or sources
        out.push_str("| ID | Kind | In | Out | Activation | Combine | Std | Dims |\n");
        out.push_str("|----|------|----|-----|------------|---------|-----|------|\n");
        for (node_id, node) in graph.nodes.iter().enumerate() {
            let kind = match node.kind {
                NodeKind::Input => "Input",
                NodeKind::Hidden => "Hidden",
                NodeKind::Output => "Output",
            };
            let act = node.activation;
            let combine = node.combine_op.map_or("—".into(), |op| format!("{op}"));
            let std = node.standardize.map_or("—".into(), |s| format!("{s}"));
            let (in_dim, out_dim) = graph.node_dims().get(node_id).copied().unwrap_or((0, 0));
            out.push_str(&format!(
                "| {id} | {kind} | {ni} | {no} | {act} | {combine} | {std} | {in_dim}→{out_dim} |\n",
                id = node.id,
                ni = node.num_inputs,
                no = node.num_outputs,
            ));
        }
    }
    out.push('\n');

    // ── Feature split — the Input node's per-port genome ──
    // The rows are genome (`M:<width>` in the diversity tokens) but a table cannot show
    // them: they live per PORT, not per node. One row per port is the whole of it, and
    // the coverage line is the number that matters — `exclusive` covers every feature by
    // construction, `random` can drop one entirely (a feature no branch ever sees).
    if let Some(masks) = graph.input_feature_masks() {
        let input_dim = graph.options.input_dim.unwrap_or(1).max(1);
        let seen = (0..input_dim)
            .filter(|f| {
                masks
                    .iter()
                    .any(|m| m.get(*f).copied().unwrap_or(0) != 0)
            })
            .count();
        let mean = masks
            .iter()
            .map(|m| m.iter().filter(|b| **b != 0).count())
            .sum::<usize>() as f32
            / masks.len().max(1) as f32;
        out.push_str("## Feature split\n\n");
        out.push_str(&format!(
            "`{}` over **{}** Input port(s): **{seen} of {input_dim}** features reach at least one port, mean **{mean:.1}** per port. Rows are genome — a port keeps a feature where its row reads `1` and hides it where it reads `0`, and the rows cross over with the ports that carry them.\n\n",
            split_policy_name(graph.options.feature_split),
            masks.len()
        ));
        out.push_str("| port | keeps | share |\n|---:|---:|---:|\n");
        for (port, mask) in masks.iter().enumerate() {
            let keeps = mask.iter().filter(|b| **b != 0).count();
            let note = if keeps == 0 {
                " (bias only)"
            } else {
                ""
            };
            out.push_str(&format!(
                "| {port} | {keeps}{note} | {:.1}% |\n",
                100.0 * keeps as f32 / input_dim as f32
            ));
        }
        out.push('\n');
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
                in_wires,
                out_wires,
            }
        })
        .collect();
    out.push_str(&render_wire_diagram(&nodes, &graph.connections));
    out.push_str("```\n");
    out.push_str("\n> **Legend**:\n");
    out.push_str("> - I=input  H=hidden  O=output\n");
    out.push_str("> - `2w→2i/1o→5w` = wires→input ports / output ports→wires. An input port carries exactly ONE wire (fan-in is extra ports), so `w == i`; only the Input node fans out, which is why its labels can repeat (`o1 o1`). Every marker maps to exactly one wire\n");
    out.push_str("> - ->dim = output dimension\n");
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
    out.push_str("\n| **Box label (Graph)** | `H<id> <combine>·<std> <dim>` — the node's internal processing. Activation is NOT here: it lives on each outgoing arrow. |");
    out.push_str("\n| **Arrow label (Graph)** | the wire's own activation — per-port, so one node can emit `mish` to one neighbor and `tanh` to another. Equal labels from the same node = same signal; duplicates are impossible (dedup collapses them). |");
    out.push_str("\n| **Dashed stub `selu (input)` / `mish (orphan)`** | orphaned port — an input nothing feeds / an output nothing reads (mermaid's counterpart of the ASCII `*`). |");
    out.push_str("\n| **`in <n>` / `out <n>`** | dataset shape (fixed by the problem). |");
    out.push_str("\n| **Input box → second line** | the feature split when the net has one: `split <policy> · <ports>p · <seen>/<n> seen`. Absent = no split (every port carries the full mixed vector), which is the common case and why the box is usually one line. Per-port rows are in *Feature split* above. |");
    out.push_str("\n| **`—` in `Combine` / `Std`** | no per-node override, so the forward pass uses `Add` / no standardisation. `—` is not a missing gene: it is the gene spelled as its default (the diversity metric folds `None` and `Identity` into one token for the same reason). |");
    out.push_str("\n| **Nodes table → `Sources`** | which output port feeds each input port (`n1_o2` = output 2 of node 1), one entry per port in port order — the `k`-th entry feeds `i<k>`; `*` = an unwired port (never left behind by `finalize`). |");
    out.push_str("\n| **Nodes table → `Combine` / `Std`** | how multiple inputs are merged / whether they're normalized first. |");
    out.push_str("\n| **Edge list & wiring diagram** | one line per wire, exact ports (`n0_o1 → n2_i0`); `<<<<` = long jump; `▶ ◀ *` = wired out / wired in / orphaned. |");
    out.push('\n');

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::topology::TopologyOptions;

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

    #[test]
    fn markdown_legend_describes_one_wire_per_port() {
        // The old caveat ("a port with several wires is drawn once per wire") described
        // the wiring model that no longer exists; the legend must say what replaced it.
        let md = topology_markdown(&sample(), None);
        assert!(
            md.contains("An input port carries exactly ONE wire"),
            "legend must state the one-wire rule:\n{md}"
        );
        assert!(
            !md.contains("a port with several wires is drawn once per wire"),
            "stale wire-per-port caveat still in the legend:\n{md}"
        );
    }

    #[test]
    fn wiring_diagram_wire_count_equals_port_count() {
        // `Nw→Mi` on the input side of every box: with one wire per port the two numbers
        // are the same, which is the visual statement of the invariant.
        let g = sample();
        let md = topology_markdown(&g, None);
        for node in &g.nodes {
            let in_wires = g.connections.iter().filter(|c| c.to.node == node.id).count();
            assert_eq!(in_wires, node.num_inputs, "n{}", node.id);
            assert!(
                md.contains(&format!("{}w→{}i", node.num_inputs, node.num_inputs)),
                "n{} should be labelled {}w→{}i:\n{md}",
                node.id,
                node.num_inputs,
                node.num_inputs
            );
        }
    }
}
