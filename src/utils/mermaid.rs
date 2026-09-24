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
                // Node-level activation moves OFF the label: with per-port
                // activations it's no longer "the" node's activation — each
                // outgoing wire carries its own (rendered on the edge label).
                // Label = combine + std + dim.
                // Kept one-line — Mermaid auto-sizes boxes, so a slightly wider
                // label is free, but a second line would balloon every box.
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
                // Same combine·std·dim shape as hidden nodes, prefixed `out`.
                // The activation is fixed identity (pure logits readout) so
                // it's omitted — nothing variable to show.
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
        // Edge label = the SOURCE PORT's activation: with per-port
        // activations, the wire itself carries the non-linearity, so the
        // signal name belongs on the arrow, not the box.
        let edge_act = graph.nodes[conn.from.node].port_activation(conn.from.index);
        let act = format!("{:#?}", edge_act).to_lowercase();
        // No port annotation: dedup guarantees at most one wire per
        // (source → target, activation) — so parallel wires always differ
        // by their activation label alone, and the source/target boxes are
        // already visible in the graph.
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

    // Orphaned ports — the mermaid counterpart of the ASCII `*` marks.
    // Direction mirrors the ASCII convention (inputs enter the box's left,
    // outputs leave its right): an orphaned INPUT renders as a dashed stub
    // flowing INTO the node (from a floating marker), an orphaned OUTPUT as
    // a stub flowing OUT. Markers are styled invisible (no fill/stroke) —
    // they read as bare dashed whiskers hanging off the box.
    let mut orphan_idx = graph.connections.len();
    let output_id = graph
        .nodes
        .iter()
        .find(|n| n.kind == NodeKind::Output)
        .map(|n| n.id);
    for node in &graph.nodes {
        // The Output node's own output port always "dangles" by design — it
        // is the net's exit, not an unused port (ASCII renders it bare, no
        // `*`). Skip it so mermaid doesn't invent an orphan the ASCII
        // (correctly) doesn't show.
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
