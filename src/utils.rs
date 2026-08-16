//! ASCII pretty-printing helpers 🎨 for graphs and flodl nets.
//!
//! Rendering is a presentation concern, so it lives outside the core graph
//! types: `graph_ascii_topology` draws a [`Graph`] and `gras_graph_ascii_net`
//! draws a [`GrasGraph`], both via the shared Manhattan-wiring diagram
//! `render_manhattan`. The types only keep their `Display` impls, which
//! delegate here.

use flodl::nn::Module;

use crate::graph::{Connection, Graph, GrasGraph, Port};
use crate::node::NodeKind;

/// Per-node description consumed by [`render_manhattan`].
#[derive(Clone, Copy)]
pub(crate) struct AsciiNode {
    pub id: usize,
    pub kind: NodeKind,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

/// Render nodes + connections as an ASCII diagram with Manhattan (right-angle)
/// arrows 🔗, like a circuit schematic:
///
/// ```text
///  n0 input (in 0 · out 2)
///                        o0 ▶───────────────┐
///                        o1 ▶───────┐       │
///                                   │       │
///  n1 hidden (in 3 · out 1)         │       │
///    i0 ◀───────────────────────────┘       │
///    i1 ◀───────────────────────────────────┘
///    i2*                                  (orphan → network input)
/// ```
///
/// Layout: one node per 4 rows (label / inputs / outputs / blank); each wire
/// gets its own vertical lane on the right, so a long jump (n1 → n7) shows as
/// a tall `│` run. Assumes contiguous node ids (row = id × 4), which our
/// constructors guarantee. Ports beyond index 9 are not rendered faithfully.
///
/// Edges are strictly forward (from.node < to.node) — like the schematic style
/// from GP papers, but WITHOUT the backward edges that figure shows. So the
/// source is always above the target and every arrow points DOWN ⬇; long
/// jumps are fine, backward jumps are not.
pub(crate) fn render_manhattan(nodes: &[AsciiNode], connections: &[Connection]) -> String {
    if nodes.is_empty() {
        return "(empty graph)".to_string();
    }

    let kind_name = |kind: NodeKind| match kind {
        NodeKind::Input => "input",
        NodeKind::Hidden => "hidden",
        NodeKind::Output => "output",
    };

    // ── layout constants (character columns) ──
    let labels: Vec<String> = nodes
        .iter()
        .map(|n| {
            format!(
                "n{} {} (in {} · out {})",
                n.id,
                kind_name(n.kind),
                n.num_inputs,
                n.num_outputs
            )
        })
        .collect();
    let label_w = labels.iter().map(|l| l.len()).max().unwrap_or(0) + 2;
    let max_in = nodes.iter().map(|n| n.num_inputs).max().unwrap_or(0);
    let max_out = nodes.iter().map(|n| n.num_outputs).max().unwrap_or(0);

    let indent = 2usize;
    let in_x0 = indent + label_w + 2; // first input-marker column
    let in_step = 4usize; // input markers every 4 cols: i0  i1  i2
    let out_x0 = in_x0 + max_in * in_step + 2; // first output-marker column
    let out_step = 4usize; // output markers every 4 cols: o0  o1
    let lane_x0 = out_x0 + max_out * out_step + 2; // first wire-lane column
    let lane_step = 2usize; // one lane per wire, 2 cols apart
    let width = lane_x0 + connections.len() * lane_step + 2;

    let in_col = |q: usize| in_x0 + q * in_step; // column of the "i{q}" marker
    let out_col = |p: usize| out_x0 + p * out_step; // column of the "o{p}" marker

    // ── canvas ──
    let rows_per_node = 4;
    let n_rows = nodes.len() * rows_per_node + 1;
    let mut canvas = vec![vec![' '; width]; n_rows];

    // Only write into empty cells, so arrows "pass behind" labels instead of
    // overwriting them.
    fn put(canvas: &mut [Vec<char>], r: usize, c: usize, ch: char) {
        if r < canvas.len() && c < canvas[r].len() && canvas[r][c] == ' ' {
            canvas[r][c] = ch;
        }
    }

    // node rows: label / inputs / outputs / blank
    for (i, node) in nodes.iter().enumerate() {
        let r0 = i * rows_per_node;
        for (k, ch) in labels[i].chars().enumerate() {
            canvas[r0][indent + k] = ch;
        }
        for q in 0..node.num_inputs {
            let c = in_col(q);
            canvas[r0 + 1][c] = 'i';
            canvas[r0 + 1][c + 1] = char::from_digit(q as u32, 10).unwrap();
        }
        for p in 0..node.num_outputs {
            let c = out_col(p);
            canvas[r0 + 2][c] = 'o';
            canvas[r0 + 2][c + 1] = char::from_digit(p as u32, 10).unwrap();
        }
    }

    // orphan markers: '*' next to input ports with no incoming wire
    for (i, node) in nodes.iter().enumerate() {
        let r1 = i * rows_per_node + 1;
        for q in 0..node.num_inputs {
            let target = Port {
                node: node.id,
                index: q,
            };
            if !connections.iter().any(|c| c.to == target) {
                put(&mut canvas, r1, in_col(q) + 2, '*');
            }
        }
    }

    // ── wires: draw in 3 phases so later strokes never erase earlier ones ──
    // (arrowheads first, then corners + verticals, then horizontals).

    // phase 1: arrowheads ▶ at each source, ◀ at each target
    for conn in connections {
        // Edges are strictly forward (from.node < to.node), so the source is
        // always above the target and arrows point DOWN — no backward edges.
        debug_assert!(
            conn.from.node < conn.to.node,
            "render_manhattan assumes forward-only edges, got: {conn}"
        );
        let src_row = conn.from.node * rows_per_node + 2;
        let tgt_row = conn.to.node * rows_per_node + 1;
        put(&mut canvas, src_row, out_col(conn.from.index) + 2, '▶');
        put(&mut canvas, tgt_row, in_col(conn.to.index) - 1, '◀');
    }

    // phase 2: one vertical lane per wire, with ┐/┘ corners at its ends
    for (j, conn) in connections.iter().enumerate() {
        let src_row = conn.from.node * rows_per_node + 2;
        let tgt_row = conn.to.node * rows_per_node + 1;
        let lane = lane_x0 + j * lane_step;
        put(&mut canvas, src_row, lane, '┐');
        for r in src_row + 1..tgt_row {
            put(&mut canvas, r, lane, '│');
        }
        put(&mut canvas, tgt_row, lane, '┘');
    }

    // phase 3: horizontal runs from source ▶ to lane, and lane to target ◀
    for (j, conn) in connections.iter().enumerate() {
        let src_row = conn.from.node * rows_per_node + 2;
        let tgt_row = conn.to.node * rows_per_node + 1;
        let src = out_col(conn.from.index) + 2;
        let tgt = in_col(conn.to.index);
        let lane = lane_x0 + j * lane_step;
        for c in src + 1..lane {
            put(&mut canvas, src_row, c, '─');
        }
        for c in (tgt + 1..lane).rev() {
            put(&mut canvas, tgt_row, c, '─');
        }
    }

    let mut out = String::new();
    for row in &canvas {
        out.push_str(row.iter().collect::<String>().trim_end());
        out.push('\n');
    }
    out
}

/// ASCII topology view of a [`Graph`] 🎨: a header box plus the Manhattan-wired
/// node diagram.
pub(crate) fn graph_ascii_topology(graph: &Graph) -> String {
    let header = format!(
        "🌐 Graph #{} · {} nodes · {} input ports · {} output ports · {} wires · combine: {:?}",
        graph.id,
        graph.nodes.len(),
        graph.graph_inputs.len(),
        graph.graph_outputs.len(),
        graph.connections.len(),
        graph.options.combine_op,
    );
    let mut out = String::new();
    out.push_str(&format!("┌{}┐\n", "─".repeat(header.len() + 4)));
    out.push_str(&format!("│  {}  │\n", header));
    out.push_str(&format!("└{}┘\n", "─".repeat(header.len() + 4)));
    out.push('\n');

    let nodes: Vec<AsciiNode> = graph
        .nodes
        .iter()
        .map(|n| AsciiNode {
            id: n.id,
            kind: n.kind,
            num_inputs: n.num_inputs,
            num_outputs: n.num_outputs,
        })
        .collect();
    out.push_str(&render_manhattan(&nodes, &graph.connections));

    out.push_str("\n▶ output port · ◀ input port · * orphaned input (fed by network input)\n");
    out
}

/// ASCII net view of a [`GrasGraph`] 🧠: the Manhattan-wired diagram plus the
/// input projection and one Linear per node.
pub(crate) fn gras_graph_ascii_net(g: &GrasGraph) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "🧠 flodl net (GrasGraph) · {} nodes · {} wires · {} params\n\n",
        g.layers.len(),
        g.connections.len(),
        g.parameters().len(),
    ));

    // ── Manhattan-wired diagram of the nodes ──
    let nodes: Vec<AsciiNode> = (0..g.layers.len())
        .map(|id| AsciiNode {
            id,
            kind: g.node_info[id].kind,
            num_inputs: g.node_info[id].num_inputs,
            num_outputs: g.node_info[id].num_outputs,
        })
        .collect();
    out.push_str(&render_manhattan(&nodes, &g.connections));

    // ── layer table 🧮 ──
    out.push('\n');
    out.push_str(&format!(
        "   🚪 input_proj : Linear({} → {})\n",
        g.input_dim, g.hidden_dim
    ));
    for node_id in 0..g.layers.len() {
        let info = g.node_info[node_id];
        let kind = match info.kind {
            NodeKind::Input => "input",
            NodeKind::Hidden => "hidden",
            NodeKind::Output => "output",
        };
        let marker = if node_id == g.output_node {
            "   ← 🏁 graph output"
        } else {
            ""
        };
        out.push_str(&format!(
            "   🧮 n{} {:<7} : Linear({} → {}) · act: {} · in:{} out:{}{}\n",
            node_id,
            kind,
            info.in_dim,
            info.out_dim,
            info.activation,
            info.num_inputs,
            info.num_outputs,
            marker
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{graph_ascii_topology, gras_graph_ascii_net};
    use crate::graph::{Graph, GrasGraph};
    use crate::node::Node;
    use flodl::Device;

    #[test]
    fn test_render_graph_topology() {
        let mut graph = Graph::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 1));
        graph.set_graph_topology();
        graph.set_graph_network();

        let s = graph_ascii_topology(&graph);
        // Header box with the graph summary
        assert!(s.contains("Graph #1"));
        // One label per node with its port counts
        assert!(s.contains("n0 input (in 0 · out 2)"));
        assert!(s.contains("n1 hidden (in 3 · out 1)"));
        // Manhattan corners + box border present
        assert!(s.contains('┌'));
        assert!(s.contains('┐'));
        assert!(s.contains('┘'));
        // Every wire is drawn with an arrowhead at each end (the legend line
        // contributes one extra ▶ and ◀)
        assert_eq!(s.matches('▶').count(), graph.connections.len() + 1);
        assert_eq!(s.matches('◀').count(), graph.connections.len() + 1);
        // Display impl delegates to the utils function
        assert_eq!(format!("{graph}"), s);
    }

    #[test]
    fn test_render_gras_graph_net() {
        let mut graph = Graph::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_output(1, 2, 1));
        graph.set_graph_network();

        let module = GrasGraph::build(&graph, Device::CPU).unwrap();
        let s = gras_graph_ascii_net(&module);

        // Manhattan diagram: node labels + one arrowhead per wire end
        assert!(s.contains("n0 input (in 0 · out 2)"));
        assert!(s.contains("n1 output (in 2 · out 1)"));
        assert_eq!(s.matches('▶').count(), module.connections.len());
        assert_eq!(s.matches('◀').count(), module.connections.len());
        // Input projection + one Linear per node, with dims
        assert!(s.contains("input_proj : Linear(1 → 8)"));
        assert!(s.contains("Linear(8 → 8)"));
        // Output node is marked 🏁
        assert!(s.contains("🏁 graph output"));
        // Display impl delegates to the utils function
        assert_eq!(format!("{module}"), s);
    }
}
