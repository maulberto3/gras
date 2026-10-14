//! Mermaid flowchart rendering for topologies.

use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;

/// Mermaid flowchart of a [`Topology`] — renders natively in GitHub/GitLab/Notion
/// and can be exported via `mmdc`.
pub fn topology_mermaid(graph: &Topology) -> String {
    let mut out = String::new();
    out.push_str("```mermaid\ngraph LR\n");
    // Per-net hidden-dim range for edge-thickness normalization (min/max
    // interpolate, same spirit as the planned ASCII box sizing).
    let dims: Vec<usize> = graph.nodes.iter().filter_map(|n| n.hidden_dim).collect();
    let (min_dim, max_dim) = (
        dims.iter().min().copied().unwrap_or(1),
        dims.iter().max().copied().unwrap_or(1),
    );
    let dim_range = (max_dim.saturating_sub(min_dim)).max(1);

        // Node definitions — COMPACT one-line labels; Mermaid auto-sizes boxes, so
        // multi-line labels balloon every box.
    for node in &graph.nodes {
        let label = match node.kind {
            NodeKind::Input => {
                format!(
                    "n{}[\"in {}\"]",
                    node.id,
                    graph.options.input_dim.unwrap_or(1)
                )
            }
            NodeKind::Hidden => {
                let dim = node.hidden_dim.unwrap_or(8);
                    // Activation moves OFF the label (per-port ops belong on the
                    // wire); label = combine + std + dim, kept one line.
                let combine = node
                    .combine_op
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "add".into());
                let std = node
                    .standardize
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "identity".into());
                format!("n{}[\"H{} {}·{} {}\"]", node.id, node.id, combine, std, dim)
            }
            NodeKind::Output => {
                    // Same combine·std·dim shape, prefixed `out`; activation omitted
                    // (fixed identity readout).
                let combine = node
                    .combine_op
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "add".into());
                let std = node
                    .standardize
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "identity".into());
                format!(
                    "n{}[\"out {}·{} {}\"]",
                    node.id,
                    combine,
                    std,
                    graph.options.output_dim.unwrap_or(1)
                )
            }
        };
        out.push_str(&format!("    {}\n", label));
    }

        // Node borders — stroke-width from the node's hidden_dim (per-net
        // normalized); one inline `style` line per node.
    const MIN_BORDER: f32 = 0.5;
    /// Thickest border stroke (widest hidden dim in the net).
    const MAX_BORDER: f32 = 5.0;
    for node in &graph.nodes {
        let dim = node.hidden_dim.unwrap_or(min_dim);
        let t = dim.saturating_sub(min_dim) as f32 / dim_range as f32;
        let width = MIN_BORDER + (MAX_BORDER - MIN_BORDER) * t.clamp(0.0, 1.0);
        out.push_str(&format!(
            "    style n{} stroke-width:{:.1}px\n",
            node.id, width
        ));
    }

        // Edges — stroke-width scaled by HOP DISTANCE, so long-range skips render
        // as thick trunks; normalized per net.
    const MIN_WIDTH: f32 = 0.5;
    /// Thickest edge stroke (longest hop in the net).
    const MAX_WIDTH: f32 = 5.0;
    let max_hop = graph
        .connections
        .iter()
        .map(|c| c.to.node.saturating_sub(c.from.node))
        .max()
        .unwrap_or(1)
        .max(1);
    for (idx, conn) in graph.connections.iter().enumerate() {
            // Edge label = the SOURCE PORT's activation: the wire carries the
            // non-linearity, so the signal name belongs on the arrow.
        let edge_act = graph.nodes[conn.from.node].port_activation(conn.from.index);
        let act = format!("{:#?}", edge_act).to_lowercase();
            // No port annotation: dedup guarantees at most one wire per
            // (source → target, activation), so parallel wires differ by label.
        out.push_str(&format!(
            "    n{} -- {} --> n{}\n",
            conn.from.node, act, conn.to.node
        ));
        let hop = conn.to.node.saturating_sub(conn.from.node).max(1);
        let t = (hop - 1) as f32 / (max_hop - 1).max(1) as f32;
        let width = MIN_WIDTH + (MAX_WIDTH - MIN_WIDTH) * t.clamp(0.0, 1.0);
        out.push_str(&format!(
            "    linkStyle {} stroke-width:{:.1}px\n",
            idx, width
        ));
    }

        // Orphaned ports — the mermaid counterpart of the ASCII `*` marks;
        // markers styled invisible (bare dashed whiskers).
    let mut orphan_idx = graph.connections.len();
    let output_id = graph
        .nodes
        .iter()
        .find(|n| n.kind == NodeKind::Output)
        .map(|n| n.id);
    for node in &graph.nodes {
            // The Output node's port always "dangles" by design (it's the exit, not
            // an unused port) — skip it so mermaid doesn't invent an orphan.
        let is_output = Some(node.id) == output_id;
        for port in 0..node.num_inputs {
            let wired = graph
                .connections
                .iter()
                .any(|c| c.to.node == node.id && c.to.index == port);
            if !wired {
                let marker = format!("o{}i{}", node.id, port);
                // Show WHAT the orphaned port would have applied — the port's
                // own activation — so the stub reads like `selu (input)`.
                let act = format!("{:#?}", node.port_activation(port)).to_lowercase();
                // marker → node: the stub ENTERS the box (input side).
                out.push_str(&format!(
                    "    {}(( )) -- {} (input) --> n{}\n",
                    marker, act, node.id
                ));
                out.push_str(&format!("    style {} fill:none,stroke:none\n", marker));
                out.push_str(&format!(
                    "    linkStyle {} stroke-dasharray:3 3\n",
                    orphan_idx
                ));
                orphan_idx += 1;
            }
        }
        for port in 0..node.num_outputs {
            if is_output {
                continue; // the net's exit port — never an orphan
            }
            let wired = graph
                .connections
                .iter()
                .any(|c| c.from.node == node.id && c.from.index == port);
            if !wired {
                let marker = format!("o{}o{}", node.id, port);
                let act = format!("{:#?}", node.port_activation(port)).to_lowercase();
                // node → marker: the stub LEAVES the box (output side).
                out.push_str(&format!(
                    "    n{} -- {} (orphan) --> {}(( ))\n",
                    node.id, act, marker
                ));
                out.push_str(&format!("    style {} fill:none,stroke:none\n", marker));
                out.push_str(&format!(
                    "    linkStyle {} stroke-dasharray:3 3\n",
                    orphan_idx
                ));
                orphan_idx += 1;
            }
        }
    }

    out.push_str("```\n");
    out
}
