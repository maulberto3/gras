//! ASCII pretty-printing helpers 🎨 for graphs and flodl nets.
//!
//! Rendering is a presentation concern, so it lives outside the core graph
//! types: `topology_ascii` draws a [`Topology`] and `network_ascii`
//! draws a [`Network`], both via the shared Manhattan-wiring diagram
//! `render_manhattan`. The types only keep their `Display` impls, which
//! delegate here.

use flodl::nn::Module;

use crate::network::Network;
use crate::node::NodeKind;
use crate::topology::{Connection, Port, Topology};

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

    // ── row layout: node blocks + per-wire track rows ──
    // Each node block is 4 rows (label / inputs / outputs / blank). Between
    // consecutive blocks sits a track zone whose height is the number of
    // horizontal runs that must live there: wires leaving the upper node
    // (one source-track row each) plus wires entering the lower node (one
    // target-track row each). Every horizontal run gets its own row, so two
    // paths never overlap horizontally — you can always trace which ▶ feeds
    // which ┐ (dedicated rows ⇒ taller diagram, never ambiguous).
    let valid: Vec<bool> = connections
        .iter()
        .map(|c| c.from.node < c.to.node && c.to.node < nodes.len() && c.to.node > 0)
        .collect();
    let out_deg = |i: usize| {
        connections
            .iter()
            .zip(&valid)
            .filter(|(c, ok)| **ok && c.from.node == i)
            .count()
    };
    let in_deg = |i: usize| {
        connections
            .iter()
            .zip(&valid)
            .filter(|(c, ok)| **ok && c.to.node == i)
            .count()
    };

    let rows_per_node = 4usize;
    let mut block_row = vec![0usize; nodes.len()]; // label row of each block
    let mut gap_row = vec![0usize; nodes.len().saturating_sub(1)]; // first track row after block i
    let mut row = 0usize;
    for i in 0..nodes.len() {
        block_row[i] = row;
        row += rows_per_node;
        if i + 1 < nodes.len() {
            gap_row[i] = row;
            row += out_deg(i) + in_deg(i + 1);
        }
    }
    let n_rows = row + 1;

    // Per-wire horizontal track rows: the source track sits in the gap below
    // the source node (wires leaving it, in order), the target track in the
    // gap above the target node (wires entering it, in order).
    let mut src_rank = vec![0usize; nodes.len()];
    let mut tgt_rank = vec![0usize; nodes.len()];
    let mut src_track = vec![0usize; connections.len()];
    let mut tgt_track = vec![0usize; connections.len()];
    for (j, c) in connections.iter().enumerate() {
        if !valid[j] {
            continue; // invalid wire — not drawable (see debug_assert below)
        }
        let s = src_rank[c.from.node];
        src_rank[c.from.node] += 1;
        src_track[j] = gap_row[c.from.node] + s;

        let t = tgt_rank[c.to.node];
        tgt_rank[c.to.node] += 1;
        let gap = c.to.node - 1; // the gap above the target node
        tgt_track[j] = gap_row[gap] + out_deg(gap) + t;
    }

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
        let r0 = block_row[i];
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

    // orphan markers: '*' on input ports with no wire (fed by net_input) and
    // on output ports with no wire — except the graph-output node's own
    // output ports, which are the graph's answer, not orphans.
    let output_node = nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Output)
        .map(|n| n.id)
        .max()
        .or_else(|| nodes.iter().map(|n| n.id).max());
    for (i, node) in nodes.iter().enumerate() {
        let r1 = block_row[i] + 1;
        for q in 0..node.num_inputs {
            let target = Port {
                node: node.id,
                index: q,
            };
            if !connections.iter().any(|c| c.to == target) {
                put(&mut canvas, r1, in_col(q) + 2, '*');
            }
        }
        if Some(node.id) != output_node {
            let r2 = block_row[i] + 2;
            for p in 0..node.num_outputs {
                let source = Port {
                    node: node.id,
                    index: p,
                };
                if !connections.iter().any(|c| c.from == source) {
                    put(&mut canvas, r2, out_col(p) + 2, '*');
                }
            }
        }
    }

    // ── wires: draw in 3 phases so later strokes never erase earlier ones ──
    // (arrowheads first, then corners + verticals, then horizontals).

    // phase 1: arrowheads ▶ at each source, ◀ at each target
    for (j, conn) in connections.iter().enumerate() {
        if !valid[j] {
            continue;
        }
        // Edges are strictly forward (from.node < to.node), so the source is
        // always above the target and arrows point DOWN — no backward edges.
        debug_assert!(
            conn.from.node < conn.to.node,
            "render_manhattan assumes forward-only edges, got: {conn}"
        );
        let src_row = block_row[conn.from.node] + 2;
        let tgt_row = block_row[conn.to.node] + 1;
        put(&mut canvas, src_row, out_col(conn.from.index) + 2, '▶');
        put(&mut canvas, tgt_row, in_col(conn.to.index) - 1, '◀');
    }

    // phase 2: corners + verticals — the ▶ drops to its source track row,
    // runs to its lane, down the lane to the target track row, then drops to
    // the ◀. Each wire owns its rows, so nothing overlaps.
    for (j, conn) in connections.iter().enumerate() {
        if !valid[j] {
            continue;
        }
        let src_row = block_row[conn.from.node] + 2;
        let tgt_row = block_row[conn.to.node] + 1;
        let src = out_col(conn.from.index) + 2;
        let tgt = in_col(conn.to.index) - 1;
        let lane = lane_x0 + j * lane_step;
        let st = src_track[j];
        let tt = tgt_track[j];

        // source: drop from ▶ down to the source track row, then east
        for r in src_row + 1..st {
            put(&mut canvas, r, src, '│');
        }
        put(&mut canvas, st, src, '└'); // turn east onto the track
        put(&mut canvas, st, lane, '┐'); // turn south into the lane
        // lane: straight down to the target track row
        for r in st + 1..tt {
            put(&mut canvas, r, lane, '│');
        }
        put(&mut canvas, tt, lane, '┘'); // turn west off the lane
        // target: run west on the target track row, then drop to the ◀
        put(&mut canvas, tt, tgt, '┌'); // turn south to the port
        for r in tt + 1..tgt_row {
            put(&mut canvas, r, tgt, '│');
        }
    }

    // phase 3: horizontal runs on each wire's own track rows
    for (j, conn) in connections.iter().enumerate() {
        if !valid[j] {
            continue;
        }
        let src = out_col(conn.from.index) + 2;
        let tgt = in_col(conn.to.index) - 1;
        let lane = lane_x0 + j * lane_step;
        for c in src + 1..lane {
            put(&mut canvas, src_track[j], c, '─');
        }
        for c in (tgt + 1..lane).rev() {
            put(&mut canvas, tgt_track[j], c, '─');
        }
    }

    let mut out = String::new();
    for row in &canvas {
        out.push_str(row.iter().collect::<String>().trim_end());
        out.push('\n');
    }
    out
}

/// ASCII topology view of a [`Topology`] 🎨: a header box plus the Manhattan-wired
/// node diagram.
pub(crate) fn topology_ascii(graph: &Topology) -> String {
    let header = format!(
        "🌐 Topology #{} · {} nodes · {} input ports · {} output ports · {} wires · combine: {:?}",
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

    out.push_str(
        "\n▶ output port · ◀ input port · * orphaned port (input: fed by network input · output: unused)\n",
    );
    out
}

/// Compact net view of a [`Network`] 🧠 — derived from the blueprint,
/// focused on what execution actually cares about:
///
/// ```text
///  🧠 flodl net (Network) · 7 nodes · 12 wires · 16 param tensors (weight+bias per layer) · combine: Add
///     🚪 input_proj : Linear(1 → 8)
///     🧮 n0 input   : Linear(8 → 8) · act: identity · out 1
///     🧮 n1 hidden  : Linear(8 → 8) · act: gelu     · in 3 · i0←n0_o0 · i1←n0_o1 · i2←n2_o0
///     🧮 n2 hidden  : Linear(8 → 8) · act: identity · in 2 · i0←n1_o1 · i1←n1_o2
///     🧮 n3 output  : Linear(8 → 8) · act: identity · in 1 · i0←n1_o0   ← 🏁 network output
/// ```
///
/// Per node: layer dims (`in → out`, i.e. the hidden dims), activation, and
/// the **incoming-input operations** — one entry per input port listing its
/// source(s), or `i{k}*` when orphaned (fed `net_input`). The wire *diagram*
/// lives in the topology view; this table is the dense summary.
pub(crate) fn network_ascii(g: &Network) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "🧠 flodl net (Network) · {} nodes · {} wires · {} param tensors (weight+bias per layer) · combine: {:?}\n",
        g.layers.len(),
        g.connections.len(),
        g.parameters().len(),
        g.combine_op,
    ));
    out.push_str(&format!(
        "   🚪 input_proj : Linear({} → {})\n",
        g.input_dim, g.hidden_dim
    ));

    let kind_name = |kind: NodeKind| match kind {
        NodeKind::Input => "input",
        NodeKind::Hidden => "hidden",
        NodeKind::Output => "output",
    };

    for node_id in 0..g.layers.len() {
        let node = &g.nodes[node_id];
        let (in_dim, out_dim) = g.node_dims[node_id];
        let marker = if node_id == g.output_node {
            "   ← 🏁 network output"
        } else {
            ""
        };

        // Incoming ops: one entry per input port — its sources (`i0←n1_o0`),
        // or `*` when orphaned (fed net_input). Nodes with no input ports
        // (input nodes) show their output count instead.
        let mut ports: Vec<String> = Vec::new();
        for i in 0..node.num_inputs {
            let target = Port {
                node: node_id,
                index: i,
            };
            let sources: Vec<String> = g
                .connections
                .iter()
                .filter(|c| c.to == target)
                .map(|c| c.from_label())
                .collect();
            ports.push(if sources.is_empty() {
                format!("i{i}*")
            } else {
                format!("i{i}←{}", sources.join("+"))
            });
        }
        let port_str = if ports.is_empty() {
            format!("out {}", node.num_outputs)
        } else {
            format!("in {} · {}", node.num_inputs, ports.join(" · "))
        };

        out.push_str(&format!(
            "   🧮 n{} {:<7} : Linear({} → {}) · act: {:<8} · {}{}\n",
            node_id,
            kind_name(node.kind),
            in_dim,
            out_dim,
            node.activation,
            port_str,
            marker,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{network_ascii, topology_ascii};
    use crate::network::Network;
    use crate::node::Node;
    use crate::topology::Topology;
    use flodl::Device;

    #[test]
    fn test_render_graph_topology() {
        let mut graph = Topology::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 1));
        graph.set_topology();
        graph.set_network();

        let s = topology_ascii(&graph);
        // Header box with the graph summary
        assert!(s.contains("Topology #1"));
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
    fn test_render_network_net() {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_output(1, 2, 1));
        graph.set_network();

        let module = Network::build(&graph, Device::CPU).unwrap();
        let s = network_ascii(&module);

        // Compact table, no Manhattan diagram in the net view
        assert!(s.contains("combine: Add"));
        assert!(!s.contains('▶'));
        assert!(!s.contains('┌'));
        // Input projection + one Linear per node, with dims + activation
        assert!(s.contains("input_proj : Linear(1 → 8)"));
        assert!(s.contains("Linear(8 → 8)"));
        assert!(s.contains("act: identity"));
        // Incoming-input ops: per-port source labels
        assert!(s.contains("n0 input"));
        assert!(s.contains("i0←n0_o0"));
        assert!(s.contains("i1←n0_o1"));
        // Output node is marked 🏁
        assert!(s.contains("🏁 network output"));
        // Display impl delegates to the utils function
        assert_eq!(format!("{module}"), s);
    }

    #[test]
    fn test_render_network_orphan_marker() {
        // n1 has 2 input ports but only 1 source → the second is orphaned
        // and must show `i1*` (fed by net_input).
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_output(1, 2, 1));
        graph.set_network();

        let module = Network::build(&graph, Device::CPU).unwrap();
        let s = network_ascii(&module);
        assert!(s.contains("i1*"), "orphaned port must be marked: {s}");
    }
}
