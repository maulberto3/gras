//! 🧬 The topology guide — the *blueprint* side of gras: nodes, wires,
//! scaffolds, orphans, JSON round-trip. **Execution** (compile → forward →
//! dims → combine ops) → `examples/network_guide.rs`; the node type →
//! `examples/node_guide.rs`.
//!
//! Run with: `source env_setup.sh && cargo run --example topology_guide`
//! Sections: 1 minimal pipeline · 2 hand-built · 3 random graphs (+ activations)
//!           3b scaffolding · 4 custom options · 5 wiring + de-orphan
//!           5b per-node insights · 6 persistence

use flodl::nn::Module;
use flodl::{DType, Device, Tensor, TensorOptions, Variable};

use gras::network::Network;
use gras::node::{Activation, Node, NodeKind};
use gras::spec::Spec;
use gras::topology::{CombineOp, Connection, Port, Topology, TopologyOptions};

// ── helpers ─────────────────────────────────────────────────────────────────

/// A random [batch, input_dim] input tensor.
fn rand_input(batch: i64, input_dim: usize) -> Variable {
    let opts = TensorOptions {
        dtype: DType::Float32,
        device: Device::CPU,
    };
    Variable::new(
        Tensor::randn(&[batch, input_dim as i64], opts).unwrap(),
        false,
    )
}

/// Topology-level summary: nodes, ports, wires, orphans, validity.
fn summarize(name: &str, graph: &Topology) {
    graph.validate().expect("graph must be valid");
    let (oi, oo) = graph.orphan_counts();
    let ports_in: usize = graph.nodes.iter().map(|n| n.num_inputs).sum();
    let ports_out: usize = graph.nodes.iter().map(|n| n.num_outputs).sum();
    println!(
        "  {name}: {} nodes · {ports_in} in-ports · {ports_out} out-ports · {} wires · {oi} orphaned-in / {oo} orphaned-out · valid ✓\n",
        graph.nodes.len(),
        graph.connections.len(),
    );
}

/// Per-node wiring: kind/ports/dim/activation/degrees/depth, then every input
/// port's sources (or `*` = orphaned → fed `net_input`) and output port's
/// sinks. Public topology API only.
fn log_insights(name: &str, graph: &Topology) {
    let hidden_dim = graph.options.hidden_dim;
    let depths = graph.depths();
    let degrees = graph.degrees();
    println!("  ── {name} insights ──");
    for node in &graph.nodes {
        let (in_deg, out_deg) = degrees[node.id];
        println!(
            "  n{} {:<7} in:{:<2} out:{:<2} · out_dim:{:<3} · act:{:<8} · depth:{} · indeg:{:<2} outdeg:{}",
            node.id,
            format!("{:?}", node.kind).to_lowercase(),
            node.num_inputs,
            node.num_outputs,
            node.hidden_dim.unwrap_or(hidden_dim),
            node.activation,
            depths[node.id],
            in_deg,
            out_deg,
        );
        for i in 0..node.num_inputs {
            let port = Port {
                node: node.id,
                index: i,
            };
            let sources: Vec<String> = graph
                .connections
                .iter()
                .filter(|c| c.to == port)
                .map(|c| c.from_label())
                .collect();
            if sources.is_empty() {
                println!("      i{i} * orphaned → net_input");
            } else {
                println!("      i{i} ← {}", sources.join(", "));
            }
        }
        for o in 0..node.num_outputs {
            let port = Port {
                node: node.id,
                index: o,
            };
            let sinks: Vec<String> = graph
                .connections
                .iter()
                .filter(|c| c.from == port)
                .map(|c| c.to_label())
                .collect();
            if sinks.is_empty() {
                println!("      o{o} * unused");
            } else {
                println!("      o{o} → {}", sinks.join(", "));
            }
        }
    }
    println!();
}

fn main() {
    // ── 1. Minimal pipeline (engine-style) ──────────────────────────────────
    println!("═ 1. Minimal pipeline (engine-style) ═");
    let mut graph = Topology::new(0, None);
    graph.create_random_hidden_nodes(5); // 🎲 random port counts from the ranges
    graph.refresh_labels(); //      📇 one label per port (rendering)
    graph.finalize(); //       🔗 auto-scaffold Input/Output + wire + auto-de-orphan
    graph.validate().unwrap(); //       🛡️ executable?
    summarize("random graph", &graph);

    let model = Network::build(&graph, Device::CPU).unwrap();
    let input = rand_input(2, graph.options.input_dim);
    let output = model.forward(&input).unwrap();
    println!(
        "  handoff → forward: {:?} -> {:?} ({} param tensors = weight+bias per layer)\n",
        input.shape(),
        output.shape(),
        model.parameters().len()
    );

    // ── 2. Hand-built graphs ────────────────────────────────────────────────
    println!("═ 2. Hand-built graph (explicit nodes + builders) ═");
    let mut manual = Topology::new(1, None);
    manual.nodes.push(Node::new_input(0, 2)); // 📥 feeds layers 1 and 2
    manual
        .nodes
        .push(Node::new_hidden(1, 3, 2).with_activation(Activation::GeLU));
    manual
        .nodes
        .push(Node::new_hidden(2, 2, 1).with_hidden_dim(32)); // 🕶️ wider channel
    manual.nodes.push(Node::new_output(3, 1, 1)); // 📤 network output
    manual.refresh_labels();
    manual.finalize();
    manual.validate().unwrap();
    summarize("manual graph", &manual);
    log_insights("manual graph", &manual);

    // ── 3. Random graphs + activations ──────────────────────────────────────
    println!("═ 3. Random generation + activation pool ═");
    let mut random = Topology::new(2, None);
    random.create_random_hidden_node();
    random.create_random_hidden_nodes(3);
    random.refresh_labels();
    random.finalize();
    random.validate().unwrap();
    summarize("random graph", &random);

    // The GP hook: assign every hidden node a random activation from a pool.
    // The engine does exactly this per individual, seeded from its seed chain.
    let mut rng = fastrand::Rng::with_seed(7);
    random.randomize_activations(
        &[Activation::ReLU, Activation::GeLU, Activation::SELU],
        &mut rng,
    );
    let acts: Vec<String> = random
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Hidden)
        .map(|n| n.activation.to_string())
        .collect();
    println!(
        "  randomize_activations(pool) → hidden acts: [{}]",
        acts.join(", ")
    );
    println!();

    // ── 3b. Scaffolding ─────────────────────────────────────────────────────
    println!("═ 3b. Scaffolding (auto Input/Output) ═");
    let mut bare = Topology::new(5, None);
    bare.create_random_hidden_nodes(2);
    println!(
        "  hidden-only graph: {} nodes, no Input/Output yet",
        bare.nodes.len()
    );
    bare.ensure_scaffold();
    let before = bare.nodes.clone();
    bare.ensure_scaffold();
    assert_eq!(bare.nodes, before, "ensure_scaffold must be idempotent");
    println!(
        "  ensure_scaffold → n0 is {:?}, last is {:?}; calling it again is a no-op ✓",
        bare.nodes[0].kind,
        bare.nodes.last().unwrap().kind
    );
    bare.refresh_labels();
    bare.finalize();
    summarize("scaffolded graph", &bare);

    // More than one Output node → merge_multi_outputs stacks them into one
    // projection node (the counterpart of input_proj).
    println!("  multi-output merge (merge_multi_outputs):");
    let mut multi = Topology::new(6, None);
    multi.nodes.push(Node::new_input(0, 1));
    multi.nodes.push(Node::new_hidden(1, 1, 1));
    multi.nodes.push(Node::new_output(2, 1, 1));
    multi.nodes.push(Node::new_output(3, 1, 1)); // second output node
    multi.finalize();
    let n_out = multi
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Output)
        .count();
    assert_eq!(n_out, 1);
    println!("  2 output nodes before → exactly {n_out} after finalize");
    summarize("multi-output merged graph", &multi);

    // ── 4. Custom options ───────────────────────────────────────────────────
    println!("═ 4. Custom options ═");
    let opts = TopologyOptions {
        seed: 42,         // 🎲 same seed → same random graph
        min_num_nodes: 2, // (reserved for future generation knobs)
        max_num_nodes: 5,
        min_inputs_per_node: 2, // 🔽 random hidden nodes: 2..=3 inputs
        max_inputs_per_node: 3,
        min_outputs_per_node: 2, // 🔼 ... and 2..=3 outputs
        max_outputs_per_node: 3,
        num_outputs_net: 1,          // (reserved for future graph-output knobs)
        input_dim: 4,                // 📥 network input is [batch, 4]
        hidden_dim: 16,              // 🧠 internal channel width
        combine_op: CombineOp::Mean, // ➗ average incoming tensors (vs Add)
    };
    let mut custom = Topology::new(3, Some(opts));
    custom.create_random_hidden_nodes(4);
    custom.refresh_labels();
    custom.finalize();
    summarize("custom graph", &custom);

    // ── 5. Wiring, up close ─────────────────────────────────────────────────
    println!("═ 5. Wiring introspection ═");
    for (from, to) in manual.connection_labels() {
        println!("  {from} -> {to}");
    }
    let (oi, oo) = manual.orphan_counts();
    println!("  manual graph orphans: {oi} inputs (fed net_input), {oo} outputs");

    // Manual wiring: push connections by hand instead of randomly.
    println!("  manual wiring (push connections yourself):");
    let mut wired = Topology::new(4, None);
    wired.nodes.push(Node::new_input(0, 2));
    wired
        .nodes
        .push(Node::new_hidden(1, 2, 1).with_activation(Activation::SELU));
    wired.nodes.push(Node::new_output(2, 1, 1));
    wired.refresh_labels();
    wired.connections.push(Connection {
        from: Port { node: 0, index: 0 },
        to: Port { node: 1, index: 0 },
    });
    wired.connections.push(Connection {
        from: Port { node: 0, index: 1 },
        to: Port { node: 1, index: 1 },
    });
    wired.connections.push(Connection {
        from: Port { node: 1, index: 0 },
        to: Port { node: 2, index: 0 },
    });
    wired.validate().unwrap();
    println!("{wired}");
    summarize("wired graph", &wired);

    // de-orphan, up close: leave one output orphaned, then watch it rewired.
    println!("  de-orphan, up close (rewire_orphaned_outputs):");
    let mut messy = Topology::new(7, None);
    messy.nodes.push(Node::new_input(0, 2));
    messy.nodes.push(Node::new_hidden(1, 3, 2));
    messy.nodes.push(Node::new_hidden(2, 2, 1));
    messy.nodes.push(Node::new_output(3, 1, 1));
    messy.connections.push(Connection {
        from: Port { node: 0, index: 0 },
        to: Port { node: 1, index: 2 },
    });
    messy.connections.push(Connection {
        from: Port { node: 0, index: 1 },
        to: Port { node: 2, index: 0 },
    });
    messy.connections.push(Connection {
        from: Port { node: 1, index: 0 },
        to: Port { node: 2, index: 1 },
    });
    messy.connections.push(Connection {
        from: Port { node: 1, index: 1 },
        to: Port { node: 3, index: 0 },
    });
    messy.validate().unwrap();
    let (oi, oo) = messy.orphan_counts();
    println!("  before: {oi} orphaned inputs (fed net_input), {oo} orphaned outputs");
    let added = messy.rewire_orphaned_outputs();
    println!("  rewire_orphaned_outputs() wired {added} orphaned output(s) into later nodes");
    let (oi, oo) = messy.orphan_counts();
    println!("  after: {oi} orphaned inputs, {oo} orphaned outputs");
    summarize("de-orphaned graph", &messy);
    log_insights("de-orphaned graph", &messy);

    // ── 6. Persistence ──────────────────────────────────────────────────────
    println!("═ 6. JSON round-trip ═");
    let json = custom.to_json().unwrap();
    std::fs::create_dir_all("saved").unwrap();
    std::fs::write("saved/topology.json", &json).unwrap();
    println!(
        "  saved blueprint → saved/topology.json ({} bytes)",
        json.len()
    );
    let reloaded = Topology::from_json(&json).unwrap();
    assert_eq!(Spec::from(&reloaded), Spec::from(&custom));
    summarize("reloaded graph", &reloaded);

    println!("  ✅ topology guide complete — every section ran");
}
