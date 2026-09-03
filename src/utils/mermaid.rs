//! Mermaid flowchart rendering for topologies.

use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;

/// Mermaid flowchart of a [`Topology`].
/// Renders natively in GitHub markdown, GitLab, Notion, etc.
/// Can also be exported to PNG/SVG via `mmdc` (mermaid-cli).
pub fn topology_mermaid(graph: &Topology) -> String {
    let mut out = String::new();
    out.push_str("```mermaid\ngraph LR\n");

    // Node definitions
    for node in &graph.nodes {
        let label = match node.kind {
            NodeKind::Input => {
                let dim = node.hidden_dim.unwrap_or(graph.options.hidden_dim);
                format!("n{}[\"Input<br/>{}→{}\"]", node.id, graph.options.input_dim, dim)
            }
            NodeKind::Hidden => {
                let dim = node.hidden_dim.unwrap_or(graph.options.hidden_dim);
                let act = format!("{:#?}", node.activation).to_lowercase();
                let combine = node.combine_op.map(|c| format!("{:#?}", c).to_lowercase()).unwrap_or_default();
                let std_label = node.standardize.map(|s| format!("{:#?}", s).to_lowercase()).unwrap_or_default();
                format!("n{}[\"H{}<br/>{}<br/>{}<br/>{}<br/>{}\"]", node.id, node.id, dim, act, combine, std_label)
            }
            NodeKind::Output => {
                let dim = node.hidden_dim.unwrap_or(graph.options.output_dim);
                format!("n{}[\"Output<br/>{}→{}\"]", node.id, dim, graph.options.output_dim)
            }
        };
        out.push_str(&format!("    {}\n", label));
    }

    // Edges — stroke-width scaled by port count
    // Thinnest edges: 0.5px, thickest: 5.0px, proportional in between
    const MIN_WIDTH: f32 = 0.5;
    const MAX_WIDTH: f32 = 5.0;
    for (idx, conn) in graph.connections.iter().enumerate() {
        out.push_str(&format!("    n{} --> n{}\n", conn.from.node, conn.to.node));
        // Target node port count (more ports = thicker line)
        let port_count = graph
            .nodes
            .iter()
            .find(|n| n.id == conn.to.node)
            .map(|n| n.num_inputs as f32)
            .unwrap_or(1.0);
        let max_ports = graph
            .options
            .max_hidden_inputs_per_node
            .max(1) as f32;
        // Linear interpolation: MIN_WIDTH at 0 ports, MAX_WIDTH at max_ports
        let t = (port_count / max_ports).clamp(0.0, 1.0);
        let width = MIN_WIDTH + (MAX_WIDTH - MIN_WIDTH) * t;
        out.push_str(&format!("    linkStyle {} stroke-width:{:.1}px\n", idx, width));
    }

    out.push_str("```\n");
    out
}
