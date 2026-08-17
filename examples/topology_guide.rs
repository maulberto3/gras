//! 🧬 The full gras guide — every API, one walkthrough.
//!
//! Run with: `source env_setup.sh && cargo run --example topology_guide`
//!
//! Sections:
//!   1. The minimal pipeline      — the exact path the engine will run per individual
//!   2. Hand-built graphs         — full manual control (builders, activations, dims)
//!   3. Random graphs             — automatic generation from options + ranges
//!
//!   3b. Scaffolding              — auto Input/Output, multi-output merge
//!   4. Custom options            — seed, dims, port ranges, CombineOp
//!   5. The wiring, up close      — connection_pairs, orphan_counts, de-orphan
//!
//!   5b. Per-node insights        — per-port sources/orphans, degrees, dims
//!   6. Compile & run             — Network::build + forward + parameters
//!   7. Persistence               — to_json / from_json (architecture only, no weights)

use flodl::nn::Module;
use flodl::{DType, Device, Tensor, TensorOptions, Variable};

use gras::network::Network;
use gras::node::{Activation, Node, NodeKind};
use gras::spec::Spec;
use gras::topology::{CombineOp, Connection, Port, Topology, TopologyOptions};

// ── helpers ─────────────────────────────────────────────────────────────────

/// A random [batch, input_dim] input tensor, ready to run through a module.
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

/// Compile, run, and summarize — the payoff of every graph in this guide.
fn build_and_run(name: &str, graph: &Topology, batch: i64) {
    graph.validate().expect("graph must be valid");
    let model = Network::build(graph, Device::CPU).unwrap();

    println!("  {name}:");
    println!("{model}");
    let input = rand_input(batch, graph.options.input_dim);
    let output = model.forward(&input).unwrap();
    // parameters() counts *parameter tensors*: each Linear = weight + bias.
    println!(
        "  forward {:?} -> {:?}  ({} param tensors = weight+bias per layer)\n",
        input.shape(),
        output.shape(),
        model.parameters().len()
    );
}

/// Per-node wiring insights: for every node, its kind / ports / output dim /
/// activation / in-out degrees, then **every input port's sources** (or `*`
/// = orphaned → fed `net_input`) and **every output port's sinks** (or `*` =
/// unused). Everything below uses only the public topology API.
fn log_insights(name: &str, graph: &Topology) {
    let hidden_dim = graph.options.hidden_dim;
    println!("  ── {name} insights ──");
    for node in &graph.nodes {
        let out_dim = node.hidden_dim.unwrap_or(hidden_dim);
        let in_deg = graph
            .connections
            .iter()
            .filter(|c| c.to.node == node.id)
            .count();
        let out_deg = graph
            .connections
            .iter()
            .filter(|c| c.from.node == node.id)
            .count();
        println!(
            "  n{} {:<7} ports in:{:<2} out:{:<2} · out_dim:{:<3} · act:{:<8} · indeg:{:<2} outdeg:{}",
            node.id,
            format!("{:?}", node.kind).to_lowercase(),
            node.num_inputs,
            node.num_outputs,
            out_dim,
            node.activation,
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
    // ═══════════════════════════════════════════════════════════════════════
    // 1. The minimal pipeline — what the engine will do per individual
    // ═══════════════════════════════════════════════════════════════════════
    println!("═ 1. Minimal pipeline (engine-style) ═");
    // Topology::new(id, None) → default options (seed 16, hidden_dim 8, ...).
    let mut graph = Topology::new(0, None);
    graph.create_random_hidden_nodes(5); // 🎲 random port counts from the ranges
    graph.set_topology(); //      📇 mint one label per port (rendering)
    graph.set_network(); //       🔗 auto-scaffold Input/Output + wire + auto-de-orphan
    graph.validate().unwrap(); //       🛡️ check it's executable
    build_and_run("random graph", &graph, 2);

    // ═══════════════════════════════════════════════════════════════════════
    // 2. Hand-built graphs — full manual control
    // ═══════════════════════════════════════════════════════════════════════
    println!("═ 2. Hand-built graph (explicit nodes + builders) ═");
    let mut manual = Topology::new(1, None);
    manual.nodes.push(Node::new_input(0, 2)); // 📥 feeds layers 1 and 2
    // 🧠 Builder style: activation / per-node dim chained at construction.
    manual
        .nodes
        .push(Node::new_hidden(1, 3, 2).with_activation(Activation::GeLU));
    manual
        .nodes
        .push(Node::new_hidden(2, 2, 1).with_hidden_dim(32)); // 🕶️ wider channel
    manual.nodes.push(Node::new_output(3, 1, 1)); // 📤 network output
    manual.set_topology();
    manual.set_network();
    build_and_run("manual graph", &manual, 3);
    log_insights("manual graph", &manual);

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Random graphs — the NAS bread-and-butter
    // ═══════════════════════════════════════════════════════════════════════
    println!("═ 3. Random generation ═");
    let mut random = Topology::new(2, None);
    // One node at a time, or a batch — ids stay contiguous 0..n either way.
    random.create_random_hidden_node();
    random.create_random_hidden_nodes(3);
    println!(
        "  created {} nodes ({} hidden)",
        random.nodes.len(),
        random
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Hidden)
            .count()
    );
    random.set_topology();
    random.set_network();
    println!(
        "  after set_network: {} nodes — auto-scaffolded into Input → hidden → Output",
        random.nodes.len()
    );
    build_and_run("random graph", &random, 2);

    // ═══════════════════════════════════════════════════════════════════════
    // 3b. Scaffolding — the canonical Input → … → Output skeleton
    // ═══════════════════════════════════════════════════════════════════════
    println!("═ 3b. Scaffolding (auto Input/Output) ═");
    // Random graphs only create hidden nodes; set_network scaffolds the
    // skeleton. ensure_scaffold is public for manual use and idempotent.
    let mut bare = Topology::new(5, None);
    bare.create_random_hidden_nodes(2);
    println!(
        "  hidden-only graph: {} nodes, no Input/Output yet",
        bare.nodes.len()
    );
    bare.ensure_scaffold();
    println!(
        "  after ensure_scaffold: {} nodes — n0 is {:?}, last is {:?}",
        bare.nodes.len(),
        bare.nodes[0].kind,
        bare.nodes.last().unwrap().kind
    );
    let before = bare.nodes.clone();
    bare.ensure_scaffold();
    assert_eq!(bare.nodes, before, "ensure_scaffold must be idempotent");
    println!("  calling ensure_scaffold again is a no-op ✓");
    bare.set_topology();
    bare.set_network();
    build_and_run("scaffolded graph", &bare, 2);

    // More than one Output node → de_multi_outputs stacks them into a single
    // projection node (the output projection, counterpart of input_proj).
    println!("  multi-output merge (de_multi_outputs):");
    let mut multi = Topology::new(6, None);
    multi.nodes.push(Node::new_input(0, 1));
    multi.nodes.push(Node::new_hidden(1, 1, 1));
    multi.nodes.push(Node::new_output(2, 1, 1));
    multi.nodes.push(Node::new_output(3, 1, 1)); // second output node
    let n_out_before = multi
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Output)
        .count();
    println!("  {n_out_before} output nodes before set_network →");
    multi.set_network();
    let n_out = multi
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Output)
        .count();
    assert_eq!(n_out, 1);
    println!("  after set_network: exactly {n_out} output node (the stacked projection)");
    build_and_run("multi-output merged graph", &multi, 2);

    // ═══════════════════════════════════════════════════════════════════════
    // 4. Custom options — full control over generation & execution
    // ═══════════════════════════════════════════════════════════════════════
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
    custom.set_topology();
    custom.set_network();
    build_and_run("custom graph", &custom, 5);

    // ═══════════════════════════════════════════════════════════════════════
    // 5. The wiring, up close
    // ═══════════════════════════════════════════════════════════════════════
    println!("═ 5. Wiring introspection ═");
    // Human-readable wire list.
    for (from, to) in manual.connection_pairs() {
        println!("  {from} -> {to}");
    }
    // (orphaned_inputs, orphaned_outputs) — orphaned *inputs* are fed the
    // network input at runtime (legal); orphaned *outputs* are rewired into
    // later nodes automatically by set_network, so this should be 0.
    let (oi, oo) = manual.orphan_counts();
    println!("  manual graph orphans: {oi} inputs (fed net_input), {oo} outputs");

    // Manual wiring — you can build connections by hand instead of randomly.
    println!("  manual wiring (push connections yourself):");
    let mut wired = Topology::new(4, None);
    wired.nodes.push(Node::new_input(0, 2));
    wired
        .nodes
        .push(Node::new_hidden(1, 2, 1).with_activation(Activation::SELU));
    wired.nodes.push(Node::new_output(2, 1, 1));
    wired.set_topology(); // mints the labels below; wiring is separate
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
    build_and_run("wired graph", &wired, 2);

    // de-orphan, up close: wire most ports by hand, leave one output
    // orphaned, then watch de_orphan_outputs rewire it into a later node.
    println!("  de-orphan, up close (de_orphan_outputs):");
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
    println!("  before de-orphan: {oi} orphaned inputs (fed net_input), {oo} orphaned outputs");
    let added = messy.de_orphan_outputs();
    println!("  de_orphan_outputs() wired {added} orphaned output(s) into later nodes");
    let (oi, oo) = messy.orphan_counts();
    println!("  after: {oi} orphaned inputs, {oo} orphaned outputs");
    build_and_run("de-orphaned graph", &messy, 2);
    log_insights("de-orphaned graph", &messy);

    // ═══════════════════════════════════════════════════════════════════════
    // 6. Persistence — architecture only, never weights
    // ═══════════════════════════════════════════════════════════════════════
    println!("═ 6. JSON round-trip ═");
    let json = custom.to_json().unwrap();
    std::fs::create_dir_all("saved").unwrap();
    std::fs::write("saved/topology.json", &json).unwrap();
    println!(
        "  saved blueprint → saved/topology.json ({} bytes)",
        json.len()
    );

    // Reload: the RNG is re-seeded from options.seed, so the loaded graph
    // regenerates wiring identically to a fresh graph with the same options.
    let reloaded = Topology::from_json(&json).unwrap();
    assert_eq!(Spec::from(&reloaded), Spec::from(&custom));
    build_and_run("reloaded graph", &reloaded, 5);

    println!("  ✅ topology guide complete — every section ran");
}
