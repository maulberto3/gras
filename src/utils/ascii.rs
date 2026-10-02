//! ASCII diagram primitives used by [`super::markdown`] — the Manhattan-wire
//! rendering and compact edge list. Extracted from the old ascii module;
//! only the pieces markdown needs survive.

use crate::graph::node::NodeKind;
use crate::graph::topology::{Connection, Topology};

/// Per-node description consumed by [`render_wire_diagram`].
#[derive(Clone, Copy)]
pub(crate) struct AsciiNode {
    pub id: usize,
    pub kind: NodeKind,
    pub num_inputs: usize,
    pub num_outputs: usize,
    /// Output dimension (from node_dims), shown in parentheses.
    pub out_dim: Option<usize>,
    /// Wires ARRIVING at this node (may exceed num_inputs: several wires can
    /// share one input port — the combine op merges them).
    pub in_wires: usize,
    /// Wires LEAVING this node (may exceed num_outputs: the Input node's
    /// single port fans out to many targets by design).
    pub out_wires: usize,
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
        .map(|n| {
            // Wires vs ports, stated explicitly on both sides:
            // `in: 2 wires → 1 port · out: 1 port → 3 wires`. A bare port
            // count read as "one input" while 4 arrows arrived (or hid a
            // fan-out) — the wire counts make the box match the arrows.
            let in_part = format!("{}w→{}i", n.in_wires.max(n.num_inputs), n.num_inputs);
            let out_part = format!("{}o→{}w", n.num_outputs, n.out_wires.max(n.num_outputs));
            match n.out_dim {
                Some(dim) => format!(
                    "n{} {} {}/{} ->{}",
                    n.id,
                    kind_name(n.kind),
                    in_part,
                    out_part,
                    dim
                ),
                None => format!("n{} {} {}/{}", n.id, kind_name(n.kind), in_part, out_part),
            }
        })
        .collect();

    let label_w = labels.iter().map(|l| l.chars().count()).max().unwrap_or(0) + 2;

    // ── 2. Track & Row Allocation ──
    let valid: Vec<bool> = connections
        .iter()
        .map(|c| c.from.node < c.to.node && c.to.node < nodes.len() && c.to.node > 0)
        .collect();

    // ── Port slots: ONE PER WIRE, so every wire is drawn one-to-one. ──
    // A port can carry several wires (the Input node fans out; any node's
    // fan-in is merged by the combine op). Drawing the port once made those
    // wires share a column and split ambiguously — a `> ` with three wires
    // leaving it read as one signal, not three. Instead each wire gets its
    // OWN slot, labelled with its port index plus a suffix letter when the
    // port is shared (`o1a o1b` = two wires off output 1), so every
    // `>`/`<` marker maps to exactly one wire and the text diagram matches
    // the mermaid edge list visually. A port with no wires still gets one
    // slot (an orphan `*`).
    let mut out_slots: Vec<Vec<(usize, Option<usize>)>> = Vec::with_capacity(nodes.len());
    let mut in_slots: Vec<Vec<(usize, Option<usize>)>> = Vec::with_capacity(nodes.len());
    let mut src_slot = vec![0usize; connections.len()];
    let mut tgt_slot = vec![0usize; connections.len()];
    for (i, node) in nodes.iter().enumerate() {
        let mut slots: Vec<(usize, Option<usize>)> = Vec::new();
        for p in 0..node.num_outputs {
            let mut wired = false;
            for (j, c) in connections.iter().enumerate() {
                if valid[j] && c.from.node == i && c.from.index == p {
                    src_slot[j] = slots.len();
                    slots.push((p, Some(j)));
                    wired = true;
                }
            }
            if !wired {
                slots.push((p, None));
            }
        }
        out_slots.push(slots);
    }
    for (i, node) in nodes.iter().enumerate() {
        let mut slots: Vec<(usize, Option<usize>)> = Vec::new();
        for p in 0..node.num_inputs {
            let mut wired = false;
            for (j, c) in connections.iter().enumerate() {
                if valid[j] && c.to.node == i && c.to.index == p {
                    tgt_slot[j] = slots.len();
                    slots.push((p, Some(j)));
                    wired = true;
                }
            }
            if !wired {
                slots.push((p, None));
            }
        }
        in_slots.push(slots);
    }
    // Slot labels: `o<port>` for a lone wire, `o<port><letter>` when the
    // port carries several (`o1a o1b`) — every wire's marker pair is then
    // uniquely named. Suffix runs a, b, c… (numeric fallback past 26).
    let slot_labels = |prefix: char, ports: &[usize]| -> Vec<String> {
        let mut labels = Vec::with_capacity(ports.len());
        for (q, &p) in ports.iter().enumerate() {
            let total = ports.iter().filter(|&&x| x == p).count();
            if total <= 1 {
                labels.push(format!("{prefix}{p}"));
            } else {
                let idx = ports[..q].iter().filter(|&&x| x == p).count();
                let suffix = if idx < 26 {
                    ((b'a' + idx as u8) as char).to_string()
                } else {
                    format!("{idx}")
                };
                labels.push(format!("{prefix}{p}{suffix}"));
            }
        }
        labels
    };
    let out_labels: Vec<Vec<String>> = out_slots
        .iter()
        .map(|slots| {
            let ports: Vec<usize> = slots.iter().map(|(p, _)| *p).collect();
            slot_labels('o', &ports)
        })
        .collect();
    let in_labels: Vec<Vec<String>> = in_slots
        .iter()
        .map(|slots| {
            let ports: Vec<usize> = slots.iter().map(|(p, _)| *p).collect();
            slot_labels('i', &ports)
        })
        .collect();
    // Offset from a slot's column to its marker: the label's own width.
    let out_off = |i: usize, q: usize| out_labels[i][q].chars().count();
    let in_off = |i: usize, q: usize| in_labels[i][q].chars().count();

    let max_in_slots = in_slots.iter().map(Vec::len).max().unwrap_or(0);
    let max_out_slots = out_slots.iter().map(Vec::len).max().unwrap_or(0);

    let indent = 2usize;
    // A slot's pitch must clear its label plus the `>`/`<` marker.
    let max_in_label = in_labels
        .iter()
        .flatten()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(1);
    let max_out_label = out_labels
        .iter()
        .flatten()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(1);
    let in_step = 4usize.max(max_in_label + 2);
    let out_step = 4usize.max(max_out_label + 2);

    let in_x0 = indent + label_w + 2;
    let out_x0 = (in_x0 + max_in_slots * in_step + 2)
        .max(indent + label_w + max_in_slots * in_step + out_step + 4);
    let lane_x0 = out_x0 + max_out_slots * out_step + 2;

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
    let out_col = |q: usize| out_x0 + q * out_step;

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

    // Per-node box height, driven by port count and normalized within the
    // net (the same relative-sizing spirit as the wire-lane tiers below):
    // the node with the fewest ports gets the base height, the node with the
    // most ports gets base + MAX_EXTRA, everything in between scales
    // linearly. Box size thus carries information — a tall box is a wide
    // fan-in/fan-out node. Rows 0..3 are structural (label / inputs /
    // outputs / padding) and must not move: wire drops anchor at +1 and +2.
    let base_rows = 4usize;
    const MAX_EXTRA: usize = 4;
    let ports = |n: &AsciiNode| n.num_inputs.max(n.num_outputs);
    let min_ports = nodes.iter().map(ports).min().unwrap_or(0);
    let max_ports = nodes.iter().map(ports).max().unwrap_or(0);
    let extra_rows = |n: &AsciiNode| -> usize {
        if max_ports == min_ports {
            0
        } else {
            (ports(n) - min_ports) * MAX_EXTRA / (max_ports - min_ports)
        }
    };
    let node_rows: Vec<usize> = nodes.iter().map(|n| base_rows + extra_rows(n)).collect();

    let mut block_row = vec![0usize; nodes.len()];
    let mut gap_row = vec![0usize; nodes.len().saturating_sub(1)];
    let mut row = 0usize;

    for i in 0..nodes.len() {
        block_row[i] = row;
        row += node_rows[i];
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
    for i in 0..nodes.len() {
        let r0 = block_row[i];
        for (k, ch) in labels[i].chars().enumerate() {
            put(&mut canvas, r0, indent + k, ch);
        }
        // One label per incoming WIRE (`i0`, or `i0a i0b` when shared).
        for (q, label) in in_labels[i].iter().enumerate() {
            let c = in_col(q);
            for (k, ch) in label.chars().enumerate() {
                put(&mut canvas, r0 + 1, c + k, ch);
            }
        }
        // One label per outgoing WIRE (`o0`, or `o1a o1b` when shared).
        for (q, label) in out_labels[i].iter().enumerate() {
            let c = out_col(q);
            for (k, ch) in label.chars().enumerate() {
                put(&mut canvas, r0 + 2, c + k, ch);
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
        for (q, &(_, conn)) in in_slots[i].iter().enumerate() {
            if conn.is_none() {
                put(&mut canvas, r1, in_col(q) + in_off(i, q), '*');
            }
        }
        if Some(node.id) != output_node {
            let r2 = block_row[i] + 2;
            for (q, &(_, conn)) in out_slots[i].iter().enumerate() {
                if conn.is_none() {
                    put(&mut canvas, r2, out_col(q) + out_off(i, q), '*');
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
        let arrow_col = out_col(src_slot[j]) + out_off(conn.from.node, src_slot[j]);
        put(&mut canvas, src_row, arrow_col, '>');
        put(&mut canvas, tgt_row, in_col(tgt_slot[j]) - 1, '<');
    }

    // ── Phase 2: Complete Port Drops, Vertical Lanes & Corners ──
    for (j, conn) in connections.iter().enumerate() {
        if !valid[j] {
            continue;
        }

        let src_row = block_row[conn.from.node] + 2;
        let tgt_row = block_row[conn.to.node] + 1;
        let src = out_col(src_slot[j]) + out_off(conn.from.node, src_slot[j]);
        let tgt = in_col(tgt_slot[j]) - 1;
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

        let src = out_col(src_slot[j]) + out_off(conn.from.node, src_slot[j]);
        let tgt = in_col(tgt_slot[j]) - 1;
        let lane = (lane_x0 + j * lane_step).min(width - 1);

        // East run from source port to lane
        for slot in canvas[src_track[j]][(src + 1)..lane.min(width)].iter_mut() {
            *slot = match *slot {
                '│' => '┼',
                ' ' => '─',
                other => other,
            };
        }

        // West run from lane to target port
        for slot in canvas[tgt_track[j]][(tgt + 1)..lane].iter_mut().rev() {
            *slot = match *slot {
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
