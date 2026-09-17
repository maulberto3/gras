//! Markdown rendering for topologies — includes nodes table, edge list,
//! ASCII wiring diagram, and Mermaid flowchart.

use super::ascii::{AsciiNode, edge_list, render_wire_diagram};
use super::mermaid::topology_mermaid;
use crate::graph::network::Network;
use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;

/// Markdown rendering of a [`Topology`]: header, merged nodes/network table,
/// edge list, and the visual wiring diagram in a code block.
/// When a built [`Network`] is provided, the nodes table includes linear dims
/// and source wiring (merged topology + network view).
///
/// Deliberately stamp-free: the header only identifies the topology — it does
/// not claim a fitness or rank. Scores live in the CSV; this is a structural
/// view you generate for any individual you choose.
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
    out.push_str("> - `4w→2i/1o→5w` = wires→input ports / output ports→wires (a port may carry several wires; the combine op merges them)\n");
    out.push_str("> - ->dim = output dimension\n");
    out.push_str("> - ▶ connected output  ◀ connected input  * orphaned port\n");

    // Mermaid flowchart
    out.push_str("\n## Graph\n\n");
    out.push_str(&topology_mermaid(graph));

    // ── Legend — plain-words reasoning guide for every visual above ──
    // Each visual channel (border, wire, label) encodes one structural
    // fact; the legend says which, so a reader never has to guess.
    out.push_str("\n## How to read this\n\n");
    out.push_str("| Visual | Meaning |\n");
    out.push_str("|---|---|");
    out.push_str("\n| **Box border thickness** | node width (`hidden_dim`, scaled min→max within this net). |");
    out.push_str("\n| **Wire thickness** | hop distance — thick = long-range skip, thin = neighbor. |");
    out.push_str("\n| **Box label (Graph)** | `H<id> <combine>·<std> <dim>` — the node's internal processing. Activation is NOT here: it lives on each outgoing arrow. |");
    out.push_str("\n| **Arrow label (Graph)** | the wire's own activation — per-port, so one node can emit `mish` to one neighbor and `tanh` to another. Equal labels from the same node = same signal; duplicates are impossible (dedup collapses them). |");
    out.push_str("\n| **Dashed stub `selu (input)` / `mish (orphan)`** | orphaned port — an input nothing feeds / an output nothing reads (mermaid's counterpart of the ASCII `*`). |");
    out.push_str("\n| **`in <n>` / `out <n>`** | dataset shape (fixed by the problem). |");
    out.push_str("\n| **Nodes table → `Sources`** | which output ports feed this node (`n1_o2` = output 2 of node 1); `*` = orphaned. |");
    out.push_str("\n| **Nodes table → `Combine` / `Std`** | how multiple inputs are merged / whether they're normalized first. |");
    out.push_str("\n| **Edge list & wiring diagram** | one line per wire, exact ports (`n0_o1 → n2_i0`); `<<<<` = long jump; `▶ ◀ *` = wired out / wired in / orphaned. |");
    out.push_str("\n");

    out
}
