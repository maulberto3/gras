//! ASCII pretty-printing helpers 🎨 for graphs and flodl nets.
//!
//! Rendering is a presentation concern, so it lives outside the core graph
//! types: `topology_ascii` draws a [`Topology`] and `network_ascii`
//! draws a [`Network`], both via the shared Manhattan-wiring diagram
//! `render_wire_diagram`. The types only keep their `Display` impls, which
//! delegate here.

use flodl::nn::Module;

use crate::network::Network;
use crate::node::NodeKind;
use crate::topology::{Connection, Port, Topology};

/// Per-node description consumed by [`render_wire_diagram`].
#[derive(Clone, Copy)]
pub(crate) struct AsciiNode {
    pub id: usize,
    pub kind: NodeKind,
    pub num_inputs: usize,
    pub num_outputs: usize,
    /// Output dimension (from node_dims), shown in parentheses.
    pub out_dim: Option<usize>,
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
pub(crate) fn render_wire_diagram(nodes: &[AsciiNode], connections: &[Connection]) -> String {
    if nodes.is_empty() {
        return "(empty graph)".to_string();
    }

    let kind_name = |kind: NodeKind| match kind {
        NodeKind::Input => "input",
        NodeKind::Hidden => "hidden",
        NodeKind::Output => "output",
    };

    // ── 1. Clean Label Strings (Pure ASCII) ──
    let labels: Vec<String> = nodes
        .iter()
        .map(|n| match n.out_dim {
            Some(dim) => format!(
                "n{} {} (in {} . out {} -> {})",
                n.id, kind_name(n.kind), n.num_inputs, n.num_outputs, dim
            ),
            None => format!(
                "n{} {} (in {} . out {})",
                n.id, kind_name(n.kind), n.num_inputs, n.num_outputs
            ),
        })
        .collect();

    let label_w = labels.iter().map(|l| l.chars().count()).max().unwrap_or(0) + 2;
    let max_in = nodes.iter().map(|n| n.num_inputs).max().unwrap_or(0);
    let max_out = nodes.iter().map(|n| n.num_outputs).max().unwrap_or(0);

    let indent = 2usize;
    let in_step = 4usize;
    let out_step = 4usize;

    let in_x0 = indent + label_w + 2;
    let out_x0 = (in_x0 + max_in * in_step + 2).max(indent + label_w + max_in * in_step + out_step + 4);
    let lane_x0 = out_x0 + max_out * out_step + 2;

    let n_wires = connections.len();
    let lane_step = if n_wires <= 10 {
        3usize
    } else if n_wires <= 25 {
        2usize
    } else {
        1usize
    };

    let max_width = 200usize;
    let raw_width = lane_x0 + n_wires * lane_step + 2;
    let width = raw_width.min(max_width);

    let in_col = |q: usize| in_x0 + q * in_step;
    let out_col = |p: usize| out_x0 + p * out_step;

    // ── 2. Track & Row Allocation ──
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
    let mut block_row = vec![0usize; nodes.len()];
    let mut gap_row = vec![0usize; nodes.len().saturating_sub(1)];
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

    let mut src_rank = vec![0usize; nodes.len()];
    let mut tgt_rank = vec![0usize; nodes.len()];
    let mut src_track = vec![0usize; connections.len()];
    let mut tgt_track = vec![0usize; connections.len()];

    for (j, c) in connections.iter().enumerate() {
        if !valid[j] {
            continue;
        }
        let s = src_rank[c.from.node];
        src_rank[c.from.node] += 1;
        src_track[j] = gap_row[c.from.node] + s;

        let t = tgt_rank[c.to.node];
        tgt_rank[c.to.node] += 1;
        let gap = c.to.node - 1;
        tgt_track[j] = gap_row[gap] + out_deg(gap) + t;
    }

    let mut canvas = vec![vec![' '; width]; n_rows];

    let put = |cv: &mut Vec<Vec<char>>, r: usize, c: usize, ch: char| {
        if r < cv.len() && c < cv[r].len() && cv[r][c] == ' ' {
            cv[r][c] = ch;
        }
    };

    // Render Labels & Ports
    for (i, node) in nodes.iter().enumerate() {
        let r0 = block_row[i];
        for (k, ch) in labels[i].chars().enumerate() {
            canvas[r0][indent + k] = ch;
        }
        for q in 0..node.num_inputs {
            let c = in_col(q);
            canvas[r0 + 1][c] = 'i';
            canvas[r0 + 1][c + 1] = char::from_digit(q as u32, 10).unwrap_or('?');
        }
        for p in 0..node.num_outputs {
            let c = out_col(p);
            canvas[r0 + 2][c] = 'o';
            canvas[r0 + 2][c + 1] = char::from_digit(p as u32, 10).unwrap_or('?');
        }
    }

    // Render Orphans
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

    // ── Phase 1: Arrowheads ──
    for (j, conn) in connections.iter().enumerate() {
        if !valid[j] { continue; }
        let src_row = block_row[conn.from.node] + 2;
        let tgt_row = block_row[conn.to.node] + 1;
        put(&mut canvas, src_row, out_col(conn.from.index) + 2, '>');
        put(&mut canvas, tgt_row, in_col(conn.to.index) - 1, '<');
    }

    // ── Phase 2: Complete Port Drops, Vertical Lanes & Corners ──
    for (j, conn) in connections.iter().enumerate() {
        if !valid[j] { continue; }

        let src_row = block_row[conn.from.node] + 2;
        let tgt_row = block_row[conn.to.node] + 1;
        let src = out_col(conn.from.index) + 2;
        let tgt = in_col(conn.to.index) - 1;
        let lane = lane_x0 + j * lane_step;
        let st = src_track[j];
        let tt = tgt_track[j];

        // 1. Source drop: from '>' down to source track row, then turn east
        for r in (src_row + 1)..st {
            put(&mut canvas, r, src, '│');
        }
        put(&mut canvas, st, src, '└'); // Turn east onto horizontal track
        put(&mut canvas, st, lane, '┐'); // Turn south down vertical lane

        // 2. Vertical Lane: down to target track row
        for r in (st + 1)..tt {
            put(&mut canvas, r, lane, '│');
        }
        put(&mut canvas, tt, lane, '┘'); // Turn west off vertical lane

        // 3. Target drop: west on track row, turn south to '<'
        put(&mut canvas, tt, tgt, '┌'); // Turn south to port
        for r in (tt + 1)..tgt_row {
            put(&mut canvas, r, tgt, '│');
        }
    }

    // ── Phase 3: Horizontals with Collision Crossovers (┼) ──
    for (j, conn) in connections.iter().enumerate() {
        if !valid[j] { continue; }

        let src = out_col(conn.from.index) + 2;
        let tgt = in_col(conn.to.index) - 1;
        let lane = lane_x0 + j * lane_step;

        // East run from source port to lane
        for c in (src + 1)..lane {
            let ch = canvas[src_track[j]][c];
            canvas[src_track[j]][c] = match ch {
                '│' => '┼',
                ' ' => '─',
                other => other,
            };
        }

        // West run from lane to target port
        for c in (tgt + 1..lane).rev() {
            let ch = canvas[tgt_track[j]][c];
            canvas[tgt_track[j]][c] = match ch {
                '│' => '┼',
                ' ' => '─',
                other => other,
            };
        }
    }

    // Assembly Output
    let mut out = String::new();
    for row in &canvas {
        out.push_str(row.iter().collect::<String>().trim_end());
        out.push('\n');
    }
    out
}

/// Compact edge list with distance markers for a [`Topology`].
///
/// Sorts connections by source node, shows the distance (in hops) between
/// source and target, and highlights long-range jumps with `>>>` markers.
///
/// ```text
/// edges (7 wires):
///   n0_o0 → n1_i0   ··  (1)
///   n0_o1 → n2_i0   >>> (2)     long jump
///   n1_o0 → n3_i0   >>>>>> (4)  long jump
/// ```
pub(crate) fn edge_list(graph: &Topology) -> String {
    let mut out = String::new();
    let n = graph.connections.len();
    out.push_str(&format!("edges ({n} wires):\n"));

    // Sort by (from.node, from.index, to.node, to.index) for scannability.
    let mut sorted: Vec<&Connection> = graph.connections.iter().collect();
    sorted.sort_by_key(|c| (c.from.node, c.from.index, c.to.node, c.to.index));

    for conn in sorted {
        let dist = conn.to.node.saturating_sub(conn.from.node);
        let label = format!("n{}_o{} → n{}_i{}", conn.from.node, conn.from.index, conn.to.node, conn.to.index);
        let marker = match dist {
            0 => unreachable!("forward-only wiring"),
            1 => "  ·· ".to_string(),
            2 => "  >  ".to_string(),
            _ => format!("  {}>", ">".repeat(dist.saturating_sub(1))),
        };
        let tag = if dist >= 2 {
            format!("  {dist} hops  <<<< long jump")
        } else {
            String::new()
        };
        out.push_str(&format!("  {label}{marker}({dist}){tag}\n"));
    }
    out
}

/// ASCII topology view of a [`Topology`] 🎨: a header box plus the Manhattan-wired
/// node diagram.
pub(crate) fn topology_ascii(graph: &Topology) -> String {
    let header = format!(
        "🌐 Topology #{} · {} nodes · {} input ports · {} output ports · {} wires",
        graph.id,
        graph.nodes.len(),
        graph.graph_inputs.len(),
        graph.graph_outputs.len(),
        graph.connections.len(),
    );
    let mut out = String::new();
    out.push_str(&format!("┌{}┐\n", "─".repeat(header.len() + 4)));
    out.push_str(&format!("│  {}  │\n", header));
    out.push_str(&format!("└{}┘\n", "─".repeat(header.len() + 4)));
    out.push('\n');

    let node_dims = graph.node_dims();
    let nodes: Vec<AsciiNode> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| AsciiNode {
            id: n.id,
            kind: n.kind,
            num_inputs: n.num_inputs,
            num_outputs: n.num_outputs,
            out_dim: node_dims.get(i).map(|&(_, out)| out),
        })
        .collect();
    out.push_str(&render_wire_diagram(&nodes, &graph.connections));

    out.push_str(
        "\n▶ output port · ◀ input port · * orphaned port (input: fed by network input · output: unused)\n",
    );
    out.push('\n');
    out.push_str(&edge_list(graph));
    out
}

/// Compact net view of a [`Network`] 🧠 — derived from the blueprint,
/// focused on what execution actually cares about:
///
/// ```text
/// 🧠 flodl net (Network) · 7 nodes · 12 wires · 16 param tensors
///     🧮 n0 input   : orphan_proj(1→8) · Linear(8 → 8) · act: identity
///     🧮 n1 hidden  : Linear(8 → 8) · act: gelu · in 3 · i0←n0_o0 · i1←n0_o1
///     🧮 n2 hidden  : orphan_proj(1→16) · Linear(16 → 16) · act: identity · i1* 
///     🧮 n3 output  : Linear(16 → 10) · act: identity ← 🏁 network output
/// ```
///
/// Per node: orphan projection (if any), layer dims, activation, and
/// source wiring. `i{k}*` = orphaned port fed via orphan projection.
pub(crate) fn network_ascii(g: &Network) -> String {
    let mut out = String::new();
    let n_orphan_params: i64 = g.orphan_projections.iter()
        .filter_map(|p| p.as_ref())
        .map(|p| p.parameters().iter().map(|pp| pp.variable.numel()).sum::<i64>())
        .sum();
    out.push_str(&format!(
        "🧠 flodl net (Network) · {} nodes · {} wires · {} param tensors",
        g.layers.len(), g.connections.len(), g.parameters().len(),
    ));
    if n_orphan_params > 0 {
        out.push_str(&format!(" ({} orphan proj)", n_orphan_params));
    }
    out.push('\n');

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

        let std_str = node.standardize.map_or(String::new(), |s| format!(" · std: {s}"));
        let orphan_str = if g.orphan_projections[node_id].is_some() {
            format!("orphan_proj({}→{}) · ", g.input_dim, in_dim)
        } else {
            String::new()
        };
        out.push_str(&format!(
            "   🧮 n{} {:<7} : {}Linear({} → {}) · act: {:<8} · {}{}{}\n",
            node_id,
            kind_name(node.kind),
            orphan_str,
            in_dim,
            out_dim,
            node.activation,
            port_str,
            marker,
            std_str,
        ));
    }
    out
}

// ── Markdown output ────────────────────────────────────────────────────

/// Markdown rendering of a [`Topology`]: header, merged nodes/network table,
/// edge list, and the visual wiring diagram in a code block.
/// When a built [`Network`] is provided, the nodes table includes linear dims
/// and source wiring (merged topology + network view).
pub(crate) fn topology_markdown(graph: &Topology, fitness: Option<f32>, net: Option<&Network>) -> String {
    let mut out = String::new();

    // Header
    if let Some(f) = fitness {
        out.push_str(&format!(
            "# Topology #{} · fitness {f:.4}\n\n", graph.id
        ));
    } else {
        out.push_str(&format!(
            "# Topology #{}\n\n", graph.id
        ));
    }

    // Summary
    let hidden_range: Vec<usize> = graph.nodes.iter()
        .filter_map(|n| n.hidden_dim)
        .collect();
    let hidden_info = if hidden_range.is_empty() {
        format!("hidden {}", graph.options.hidden_dim)
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
        graph.options.input_dim,
        hidden_info,
        graph.options.output_dim,
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
                crate::node::NodeKind::Input => "Input",
                crate::node::NodeKind::Hidden => "Hidden",
                crate::node::NodeKind::Output => "Output",
            };
            let act = node.activation;
            let combine = node.combine_op.map_or("—".into(), |op| format!("{op}"));
            let std = node.standardize.map_or("—".into(), |s| format!("{s}"));
            let (in_dim, out_dim) = graph.node_dims().get(i).copied().unwrap_or((0, 0));
            // Orphan projection row (before the node)
            if net.orphan_projections[i].is_some() {
                out.push_str(&format!(
                    "| orphan_proj(n{}) | — | — | — | {} → {} | — | — | — | raw_input |\n",
                    i, net.input_dim, in_dim,
                ));
            }
            let marker = if i == net.output_node { " 🏁" } else { "" };
            let sources: Vec<String> = node_sources[i]
                .iter()
                .flatten()
                .map(|p| format!("n{}_o{}", p.node, p.index))
                .collect();
            let src_str = if sources.is_empty() { "*".into() } else { sources.join(", ") };
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
                crate::node::NodeKind::Input => "Input",
                crate::node::NodeKind::Hidden => "Hidden",
                crate::node::NodeKind::Output => "Output",
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
        .map(|(i, n)| AsciiNode {
            id: n.id,
            kind: n.kind,
            num_inputs: n.num_inputs,
            num_outputs: n.num_outputs,
            out_dim: node_dims.get(i).map(|&(_, out)| out),
        })
        .collect();
    out.push_str(&render_wire_diagram(&nodes, &graph.connections));
    out.push_str("```\n");
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
        graph.refresh_labels();
        graph.finalize();

        let s = topology_ascii(&graph);
        // Header box with the graph summary
        assert!(s.contains("Topology #1"));
        // One label per node with its port counts
        // Labels now include the output dimension: n{id} {kind} (in {ports} · out {ports} → {dim})
        assert!(s.contains("n0 input"), "label: {s}");
        assert!(s.contains("n1 hidden"), "label: {s}");
        assert!(s.contains("out 2 -> 8"), "label: {s}");
        // Manhattan corners + box border present
        assert!(s.contains('┌'));
        assert!(s.contains('┐'));
        assert!(s.contains('┘'));
        // Every wire is drawn with arrowheads on the canvas.
        // Count only standalone '>' and '<' (not '->' from labels).
        let has_arrows = s.lines().any(|line| line.contains('>') && !line.contains("->"));
        assert!(has_arrows, "wiring diagram must contain arrowheads");
        // Display impl delegates to the utils function
        assert_eq!(format!("{graph}"), s);
    }

    #[test]
    fn test_render_network_net() {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_output(1, 2, 1));
        graph.finalize();

        let module = Network::build(&graph, Device::CPU).unwrap();
        let s = network_ascii(&module);

        // Compact table, no Manhattan diagram in the net view
        assert!(!s.contains('▶'));
        assert!(!s.contains('┌'));
        // Orphan projection + one Linear per node, with dims + activation
        assert!(s.contains("orphan_proj"));
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
        graph.finalize();

        let module = Network::build(&graph, Device::CPU).unwrap();
        let s = network_ascii(&module);
        assert!(s.contains("i1*"), "orphaned port must be marked: {s}");
    }
}
