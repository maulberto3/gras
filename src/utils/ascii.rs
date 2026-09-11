//! ASCII diagram primitives used by [`super::markdown`] — the Manhattan-wire
//! rendering and compact edge list. Extracted from the old ascii module;
//! only the pieces markdown needs survive.

use crate::graph::node::NodeKind;
use crate::graph::topology::{Connection, Port, Topology};

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

pub(crate) fn render_wire_diagram(nodes: &[AsciiNode], connections: &[Connection]) -> String {
    if nodes.is_empty() {
        return "(empty graph)".to_string();
    }

    let kind_name = |kind: NodeKind| match kind {
        NodeKind::Input => "I",
        NodeKind::Hidden => "H",
        NodeKind::Output => "O",
    };

    // ── 1. Clean Label Strings (Pure ASCII) ──
    let labels: Vec<String> = nodes
        .iter()
        .map(|n| match n.out_dim {
            Some(dim) => format!(
                "n{} {} {}i/{}o ->{}",
                n.id,
                kind_name(n.kind),
                n.num_inputs,
                n.num_outputs,
                dim
            ),
            None => format!(
                "n{} {} {}i/{}o",
                n.id,
                kind_name(n.kind),
                n.num_inputs,
                n.num_outputs
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
    let out_x0 =
        (in_x0 + max_in * in_step + 2).max(indent + label_w + max_in * in_step + out_step + 4);
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
        if r < cv.len() && c < cv[r].len() {
            let existing = cv[r][c];
            if existing == ' ' {
                cv[r][c] = ch;
            } else if existing == '│' && ch == '│' {
                // vertical meets vertical — keep vertical
            } else if existing == '│' && ch == '─' {
                cv[r][c] = '┼'; // crossing
            } else if existing == '│' && ch == '└' {
                cv[r][c] = '├'; // vertical meets turn-down = T-junction
            }
        }
    };

    // Render Labels & Ports
    for (i, node) in nodes.iter().enumerate() {
        let r0 = block_row[i];
        for (k, ch) in labels[i].chars().enumerate() {
            put(&mut canvas, r0, indent + k, ch);
        }
        for q in 0..node.num_inputs {
            let c = in_col(q);
            put(&mut canvas, r0 + 1, c, 'i');
            if q < 10 {
                put(
                    &mut canvas,
                    r0 + 1,
                    c + 1,
                    char::from_digit(q as u32, 10).unwrap_or('?'),
                );
            } else {
                let dig = format!("{}", q);
                for (k, ch) in dig.chars().enumerate() {
                    put(&mut canvas, r0 + 1, c + 1 + k, ch);
                }
            }
        }
        for p in 0..node.num_outputs {
            let c = out_col(p);
            put(&mut canvas, r0 + 2, c, 'o');
            if p < 10 {
                put(
                    &mut canvas,
                    r0 + 2,
                    c + 1,
                    char::from_digit(p as u32, 10).unwrap_or('?'),
                );
            } else {
                let dig = format!("{}", p);
                for (k, ch) in dig.chars().enumerate() {
                    put(&mut canvas, r0 + 2, c + 1 + k, ch);
                }
            }
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
                    let marker_col = out_col(p) + if p < 10 { 2 } else { 3 };
                    put(&mut canvas, r2, marker_col, '*');
                }
            }
        }
    }

    // ── Phase 1: Arrowheads ──
    for (j, conn) in connections.iter().enumerate() {
        if !valid[j] {
            continue;
        }
        let src_row = block_row[conn.from.node] + 2;
        let tgt_row = block_row[conn.to.node] + 1;
        let arrow_col = out_col(conn.from.index) + if conn.from.index < 10 { 2 } else { 3 };
        put(&mut canvas, src_row, arrow_col, '>');
        put(&mut canvas, tgt_row, in_col(conn.to.index) - 1, '<');
    }

    // ── Phase 2: Complete Port Drops, Vertical Lanes & Corners ──
    for (j, conn) in connections.iter().enumerate() {
        if !valid[j] {
            continue;
        }

        let src_row = block_row[conn.from.node] + 2;
        let tgt_row = block_row[conn.to.node] + 1;
        let src = out_col(conn.from.index) + 2;
        let tgt = in_col(conn.to.index) - 1;
        let lane = (lane_x0 + j * lane_step).min(width - 1);
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
        if !valid[j] {
            continue;
        }

        let src = out_col(conn.from.index) + 2;
        let tgt = in_col(conn.to.index) - 1;
        let lane = (lane_x0 + j * lane_step).min(width - 1);

        // East run from source port to lane
        for c in (src + 1)..lane.min(width) {
            if c < width {
                let ch = canvas[src_track[j]][c];
                canvas[src_track[j]][c] = match ch {
                    '│' => '┼',
                    ' ' => '─',
                    other => other,
                };
            }
        }

        // West run from lane to target port
        for c in (tgt + 1..lane).rev() {
            if c < width {
                let ch = canvas[tgt_track[j]][c];
                canvas[tgt_track[j]][c] = match ch {
                    '│' => '┼',
                    ' ' => '─',
                    other => other,
                };
            }
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

/// ASCII topology view of a [`Topology`]: a header box plus the Manhattan-wired
/// node diagram.

pub(crate) fn edge_list(graph: &Topology) -> String {
    let mut out = String::new();
    let n = graph.connections.len();
    out.push_str(&format!("edges ({n} wires):\n"));

    // Sort by (from.node, from.index, to.node, to.index) for scannability.
    let mut sorted: Vec<&Connection> = graph.connections.iter().collect();
    sorted.sort_by_key(|c| (c.from.node, c.from.index, c.to.node, c.to.index));

    for conn in sorted {
        let dist = conn.to.node.saturating_sub(conn.from.node);
        let label = format!(
            "n{}_o{} → n{}_i{}",
            conn.from.node, conn.from.index, conn.to.node, conn.to.index
        );
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

