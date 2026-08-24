//! 🧬 The topology guide — the *blueprint* side of gras: nodes, wires,
//! scaffolds, orphans, JSON round-trip. Execution → `network_guide`.
//! The node type → `node_guide`.
//!
//! Run with: `source env_setup.sh && cargo run --example topology_guide`

use flodl::nn::Module;
use flodl::{DType, Device, Tensor, TensorOptions, Variable};

use gras::network::Network;
use gras::node::{Activation, Node, NodeKind};
use gras::spec::Spec;
use gras::topology::{Topology, TopologyOptions};

fn rand_input(batch: i64, input_dim: usize) -> Variable {
    Variable::new(
        Tensor::randn(
            &[batch, input_dim as i64],
            TensorOptions {
                dtype: DType::Float32,
                device: Device::CPU,
            },
        )
        .unwrap(),
        false,
    )
}

fn summarize(name: &str, g: &Topology) {
    g.validate().expect("graph must be valid");
    let (oi, oo) = g.orphan_counts();
    let pi: usize = g.nodes.iter().map(|n| n.num_inputs).sum();
    let po: usize = g.nodes.iter().map(|n| n.num_outputs).sum();
    println!(
        "  {name}: {} nodes · {pi} in-ports · {po} out-ports · {} wires · {oi} orphan-in / {oo} orphan-out · valid ✓\n",
        g.nodes.len(),
        g.connections.len()
    );
}

fn main() {
    // ── 1. Minimal pipeline (engine-style) ──────────────────────────────────
    println!("═ 1. Minimal pipeline ═");
    let mut graph = Topology::new(0, None);
    graph.create_random_hidden_nodes(5);
    graph.refresh_labels();
    graph.finalize();
    graph.validate().unwrap();
    summarize("random graph", &graph);
    let net = Network::build(&graph, Device::CPU).unwrap();
    let input = rand_input(2, graph.options.input_dim);
    let output = net.forward(&input).unwrap();
    println!(
        "  forward: {:?} -> {:?} ({} param tensors)\n",
        input.shape(),
        output.shape(),
        net.parameters().len()
    );

    // ── 2. Hand-built graph ──────────────────────────────────────────────────
    println!("═ 2. Hand-built graph ═");
    let mut manual = Topology::new(1, None);
    manual.nodes.push(Node::new_input(0, 2));
    manual
        .nodes
        .push(Node::new_hidden(1, 3, 2).with_activation(Activation::GeLU));
    manual
        .nodes
        .push(Node::new_hidden(2, 2, 1).with_hidden_dim(32));
    manual.nodes.push(Node::new_output(3, 1, 1));
    manual.refresh_labels();
    manual.finalize();
    manual.validate().unwrap();
    summarize("manual graph", &manual);

    // ── 3. Random generation + scaffolding ───────────────────────────────────
    println!("═ 3. Random generation + scaffolding ═");
    let mut bare = Topology::new(5, None);
    bare.create_random_hidden_nodes(2);
    println!(
        "  hidden-only: {} nodes, no Input/Output yet",
        bare.nodes.len()
    );
    bare.ensure_scaffold();
    bare.refresh_labels();
    bare.finalize();
    summarize("scaffolded graph", &bare);
    // Multiple Output nodes are rejected by validate() — use exactly one.
    println!("  multiple Output nodes → validation error\n");

    // ── 4. Random activations (GP hook) ──────────────────────────────────────
    println!("═ 4. Random activations + activation pool ═");
    let mut rng = fastrand::Rng::with_seed(7);
    let pool = [Activation::ReLU, Activation::GeLU, Activation::SELU];
    for node in &mut graph.nodes {
        if node.kind == NodeKind::Hidden {
            node.activation = pool[rng.usize(0..pool.len())];
        }
    }
    let acts: Vec<String> = graph
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Hidden)
        .map(|n| n.activation.to_string())
        .collect();
    println!("  inline pool draw → hidden acts: [{}]\n", acts.join(", "));

    // ── 5. Custom options ────────────────────────────────────────────────────
    println!("═ 5. Custom options ═");
    let opts = TopologyOptions {
        seed: 42,
        min_hidden_num_nodes: 2,
        max_hidden_num_nodes: 5,
        min_hidden_inputs_per_node: 2,
        max_hidden_inputs_per_node: 3,
        min_hidden_outputs_per_node: 2,
        max_hidden_outputs_per_node: 3,
        input_dim: 4,
        hidden_dim: 16,
        output_dim: 2,
    };
    let mut custom = Topology::new(3, Some(opts));
    custom.create_random_hidden_nodes(4);
    custom.refresh_labels();
    custom.finalize();
    summarize("custom graph", &custom);

    // ── 6. Wiring + orphan introspection ─────────────────────────────────────
    println!("═ 6. Wiring introspection ═");
    for (from, to) in manual.connection_labels() {
        println!("  {from} -> {to}");
    }
    let (oi, oo) = manual.orphan_counts();
    println!("  orphans: {oi} inputs (fed net_input), {oo} outputs\n");

    // ── 7. JSON round-trip ───────────────────────────────────────────────────
    println!("═ 7. JSON round-trip ═");
    let json = custom.to_json().unwrap();
    let out_dir = std::path::Path::new("results/topology_guide");
    std::fs::create_dir_all(out_dir).unwrap();
    let out_path = out_dir.join("topology.json");
    std::fs::write(&out_path, &json).unwrap();
    println!("  saved → {} ({} bytes)", out_path.display(), json.len());
    let reloaded = Topology::from_json(&json).unwrap();
    assert_eq!(Spec::from(&reloaded), Spec::from(&custom));
    summarize("reloaded graph", &reloaded);

    println!("  ✅ topology guide complete — every section ran");
}
