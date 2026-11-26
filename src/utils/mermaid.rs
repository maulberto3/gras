//! Mermaid flowchart rendering for topologies.

use crate::graph::node::NodeKind;
use crate::graph::topology::{FeatureSplitPolicy, Topology};

/// How a blueprint's feature split names itself in a doc: the policy, when one is recorded, else
/// `rows` (masks present without a policy is a legacy record the load path reconciles; naming them
pub(crate) fn split_policy_name(policy: Option<FeatureSplitPolicy>) -> &'static str {
    match policy {
        Some(FeatureSplitPolicy::Exclusive) => "exclusive",
        Some(FeatureSplitPolicy::Random) => "random",
        Some(FeatureSplitPolicy::Chunks) => "chunks",
        // PROVENANCE: mixed rows came from crossover, which is not a drawable pool entry.
        Some(FeatureSplitPolicy::Crossover) => "crossover",
        // No policy + no rows is `full`; old rows with no recorded label are still just rows.
        None => "full",
    }
}

/// The Input node's feature split as one line — `split random · 8p · 392/784 seen` — or `None`
/// the net has none.
pub(crate) fn feature_split_cell(graph: &Topology) -> Option<String> {
    let masks = graph.input_feature_masks()?;
    let input_dim = graph.options.input_dim.unwrap_or(1).max(1);
    let seen = (0..input_dim)
        .filter(|f| masks.iter().any(|m| m.get(*f).copied().unwrap_or(0) != 0))
        .count();
    Some(format!(
        "split {} · {}p · {}/{} seen",
        split_policy_name(graph.options.feature_split),
        masks.len(),
        seen,
        input_dim
    ))
}

/// Mermaid flowchart of a [`Topology`] — renders natively in GitHub/GitLab/Notion
/// and can be exported via `mmdc`.
pub fn topology_mermaid(graph: &Topology) -> String {
    let mut out = String::new();
    out.push_str("```mermaid\n");
    // The SAME curve the lineage graph asks for (`lineage.rs`): without an explicit config every viewer picks its own, so one card showed the smooth `Δfit` wires of the birth…
    out.push_str("---\nconfig:\n  layout: dagre\n  flowchart:\n    curve: basis\n---\n");
    out.push_str("flowchart LR\n");
    // Per-net hidden-dim range for edge-thickness normalization (min/max
    // interpolate, same spirit as the planned ASCII box sizing).
    let dims: Vec<usize> = graph.nodes.iter().filter_map(|n| n.hidden_dim).collect();
    let (min_dim, max_dim) = (
        dims.iter().min().copied().unwrap_or(1),
        dims.iter().max().copied().unwrap_or(1),
    );
    let dim_range = (max_dim.saturating_sub(min_dim)).max(1);

    // Per-node `(in, out)` so the Input box can name its PROJECTION width, the number the
    // ASCII diagram prints as `4o ->64` and every other box already carries.
    let node_dims = graph.node_dims();
    let looping: Vec<usize> = crate::utils::graph_utils::looping_nodes(graph)
        .into_iter()
        .map(|(i, _)| i)
        .collect();

    // Node definitions — COMPACT one-line labels; Mermaid auto-sizes boxes,
    // multi-line labels balloon every box.
    for node in &graph.nodes {
        let label = match node.kind {
            NodeKind::Input => {
                // The Input box carries three facts, one per line: the DATASET width (`in 784`),
                // the fan-out and projection width (`4p → 64`) — without it a box with four
                // outgoing wires read as a single 784-wide vector, and its width was the only
                // one missing from the diagram — and, when the net has one, the feature split
                // (the split is genome on the INPUT's ports, so this is its home).
                let raw = graph.options.input_dim.unwrap_or(1);
                let ports = node.num_outputs.max(1);
                let projected = node_dims.get(node.id).map(|&(_, out)| out).unwrap_or(raw);
                match feature_split_cell(graph) {
                    Some(split) => format!(
                        "n{}[\"in {}<br/>{}p → {}<br/>{}\"]",
                        node.id, raw, ports, projected, split
                    ),
                    None => format!(
                        "n{}[\"in {}<br/>{}p → {}\"]",
                        node.id, raw, ports, projected
                    ),
                }
            }
            NodeKind::Hidden => {
                let dim = node.hidden_dim.unwrap_or(8);
                // Activation moves OFF the label (per-port ops belong on the wire); the label
                // keeps the INTERNAL ops in execution order — combine → Linear → std — then width.
                let combine = node
                    .combine_op
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "add".into());
                let std = node
                    .standardize
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "identity".into());
                // `↻` on the box, the same fact the Nodes table's `Recur` column states.
                let loop_mark = if looping.contains(&node.id) {
                    " ↻"
                } else {
                    ""
                };
                format!(
                    "n{}[\"H{} {}→lin→{} {}{loop_mark}\"]",
                    node.id, node.id, combine, std, dim
                )
            }
            NodeKind::Output => {
                // Same ordered shape, prefixed `out`; the readout's activation is
                // the port-0 allele, so it belongs on the wire like every other port.
                let combine = node
                    .combine_op
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "add".into());
                let std = node
                    .standardize
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "identity".into());
                format!(
                    "n{}[\"out {}→lin→{} {}\"]",
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
        // One border shape for every box: the loop is drawn by the self-link alone.
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
    let levels = graph.depths();
    let max_hop = graph
        .connections
        .iter()
        .map(|c| levels[c.to.node].saturating_sub(levels[c.from.node]).max(1))
        .max()
        .unwrap_or(1)
        .max(1);
    for (idx, conn) in graph.connections.iter().enumerate() {
        // Edge label = the SOURCE PORT's activation: the wire carries
        // non-linearity, so the signal name belongs on the arrow.
        let edge_act = graph.nodes[conn.from.node].port_activation(conn.from.index);
        let act = format!("{:#?}", edge_act).to_lowercase();
        // No port annotation: dedup guarantees at most one wire per
        // (source → target, activation), so parallel wires differ by label.
        out.push_str(&format!(
            "    n{} -- {} --> n{}\n",
            conn.from.node, act, conn.to.node
        ));
        let hop = levels[conn.to.node]
            .saturating_sub(levels[conn.from.node])
            .max(1);
        let t = (hop - 1) as f32 / (max_hop - 1).max(1) as f32;
        let width = MIN_WIDTH + (MAX_WIDTH - MIN_WIDTH) * t.clamp(0.0, 1.0);
        out.push_str(&format!(
            "    linkStyle {} stroke-width:{:.1}px\n",
            idx, width
        ));
    }

    // Recurrence — a dashed self-link per looping node, emitted AFTER every real wire so
    // the `linkStyle` indices above keep pointing at their edges. It is not a wire: edge
    // counts, hop widths and orphan detection all ignore it. No arrowhead: a loop has no
    // direction to point at, and the box already carries the `↻`.
    let mut extra_idx = graph.connections.len();
    for id in &looping {
        out.push_str(&format!("    n{} -.- n{}\n", id, id));
        out.push_str(&format!(
            "    linkStyle {} stroke-dasharray:4 2,stroke:#888\n",
            extra_idx
        ));
        extra_idx += 1;
    }

    // Orphaned ports — the mermaid counterpart of the ASCII `*` marks;
    // markers styled invisible (bare dashed whiskers).
    let mut orphan_idx = extra_idx;
    let output_id = graph
        .nodes
        .iter()
        .find(|n| n.kind == NodeKind::Output)
        .map(|n| n.id);
    for node in &graph.nodes {
        // The Output node's port always "dangles" by design (it's the exit,
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
