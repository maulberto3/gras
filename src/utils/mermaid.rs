//! Mermaid flowchart rendering for topologies.

use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;

/// Mermaid flowchart of a [`Topology`].
/// Renders natively in GitHub markdown, GitLab, Notion, etc.
/// Can also be exported to PNG/SVG via `mmdc` (mermaid-cli).
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

    // Node definitions — COMPACT one-line labels. Mermaid auto-sizes each
    // box to fit its label, so multi-line labels balloon every box; one
    // short line keeps boxes small and roughly uniform. Full detail lives
    // in the Nodes table above.
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
                let act = format!("{:#?}", node.activation).to_lowercase();
                format!("n{}[\"H{} {} {}\"]", node.id, node.id, act, dim)
            }
            NodeKind::Output => {
                format!(
                    "n{}[\"out {}\"]",
                    node.id,
                    graph.options.output_dim.unwrap_or(1)
                )
            }
        };
        out.push_str(&format!("    {}\n", label));
    }

    // Node borders — stroke-width driven by the node's hidden_dim (per-net
    // normalized: min dims get the thinnest border, max the thickest). One
    // inline `style` line per node; Mermaid renders it as the box outline.
    const MIN_BORDER: f32 = 0.5;
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

    // Edges — stroke-width scaled by HOP DISTANCE (to.node − from.node):
    // the farther a wire travels, the thicker it renders, so long-range
    // skips (the interesting structural feature) are thick trunks while
    // immediate neighbor-to-neighbor wires stay thin. Normalized per net:
    // 1 hop = thinnest, max hop in this graph = thickest.
    const MIN_WIDTH: f32 = 0.5;
    const MAX_WIDTH: f32 = 5.0;
    let max_hop = graph
        .connections
        .iter()
        .map(|c| c.to.node.saturating_sub(c.from.node))
        .max()
        .unwrap_or(1)
        .max(1);
    for (idx, conn) in graph.connections.iter().enumerate() {
        out.push_str(&format!("    n{} --> n{}\n", conn.from.node, conn.to.node));
        let hop = conn.to.node.saturating_sub(conn.from.node).max(1);
        let t = (hop - 1) as f32 / (max_hop - 1).max(1) as f32;
        let width = MIN_WIDTH + (MAX_WIDTH - MIN_WIDTH) * t.clamp(0.0, 1.0);
        out.push_str(&format!(
            "    linkStyle {} stroke-width:{:.1}px\n",
            idx, width
        ));
    }

    out.push_str("```\n");
    out
}
