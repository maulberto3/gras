//! 🏭 The network guide — the execution side of gras: compiling a blueprint
//! into real flodl layers and running it.
//!
//! Run with: `source env_setup.sh && cargo run --example network_guide`
//! Sections: 1 compile · 2 forward · 3 parameters · 4 derived dims ·
//!           5 wiring at runtime · 6 combine ops · 7 rebuild · 8 devices
//! Blueprint → `topology_guide`; the node type → `node_guide`.

use flodl::nn::Module;
use flodl::{DType, Device, Tensor, TensorOptions, Variable};

use gras::network::Network;
use gras::node::Node;
use gras::topology::{CombineOp, Connection, Port, Topology};

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

/// Real parameter count: every learnable tensor's element count, summed.
fn total_params(net: &Network) -> i64 {
    net.parameters()
        .iter()
        .map(|p| p.variable.data().numel())
        .sum()
}

/// A small hand-built chain: input → hidden → output, fully wired.
fn chain() -> Topology {
    let mut g = Topology::new(0, None);
    g.nodes.push(Node::new_input(0, 1));
    g.nodes.push(Node::new_hidden(1, 1, 1));
    g.nodes.push(Node::new_output(2, 1, 1));
    g.connections.push(Connection {
        from: Port { node: 0, index: 0 },
        to: Port { node: 1, index: 0 },
    });
    g.connections.push(Connection {
        from: Port { node: 1, index: 0 },
        to: Port { node: 2, index: 0 },
    });
    g.refresh_labels();
    g
}

fn main() {
    // ── 1. Compile ──────────────────────────────────────────────────────────
    println!("═ 1. Compile — Network::build ═");
    let mut random = Topology::new(0, None);
    random.create_random_hidden_nodes(4);
    random.refresh_labels();
    random.finalize();
    random.validate().unwrap();
    let net = Network::build(&random, Device::CPU).unwrap();
    println!(
        "  built a {} from a {} blueprint",
        net.name(),
        random.nodes.len()
    );
    println!("  the compact net view (dims, activations, incoming ops):");
    println!("{net}");
    println!();

    // ── 2. Forward ──────────────────────────────────────────────────────────
    println!("═ 2. Forward — shapes across batches ═");
    for batch in [1i64, 4, 16] {
        let input = rand_input(batch, random.options.input_dim);
        let output = net.forward(&input).unwrap();
        println!(
            "  batch {batch:<2}: {:?} -> {:?}",
            input.shape(),
            output.shape()
        );
    }
    println!("  the 🏁 line in the net table marks the output node — forward returns its tensor");
    println!();

    // ── 3. Parameters ───────────────────────────────────────────────────────
    println!("═ 3. Parameters — tensor count vs element count ═");
    let params = net.parameters();
    println!(
        "  {} nodes + input_proj = {} linears × 2 (weight + bias) = {} param tensors",
        random.nodes.len(),
        random.nodes.len() + 1,
        params.len()
    );
    for p in &params {
        println!(
            "    {:<22} {:?} ({} elements)",
            p.name,
            p.variable.data().shape(),
            p.variable.data().numel()
        );
    }
    println!("  total learnable elements: {}", total_params(&net));
    println!();

    // ── 4. Derived dims ─────────────────────────────────────────────────────
    println!("═ 4. Derived dims — the node_dims ripple ═");
    // out_dim = node.hidden_dim override, else the graph's hidden_dim.
    // in_dim  = the (common) out_dim of wired sources, else hidden_dim.
    // Widen the middle node and watch every downstream layer re-derive:
    let mut wide = chain();
    wide.nodes[1].hidden_dim = Some(32); // 8 → 32 channels
    wide.validate().unwrap();
    let wide_net = Network::build(&wide, Device::CPU).unwrap();
    println!("  chain with hidden_dim 32 on n1 — the derived in/out dims:");
    println!("{wide_net}");
    println!(
        "  params: {} → {} elements (the wider layer dominates)",
        total_params(&Network::build(&chain(), Device::CPU).unwrap()),
        total_params(&wide_net)
    );
    println!();

    // ── 5. Wiring at runtime ────────────────────────────────────────────────
    println!("═ 5. Wiring at runtime — per input port: source(s) or orphan ═");
    println!("  (build precomputes this table once — forward never scans wires)");
    for node in &random.nodes {
        let mut ports: Vec<String> = Vec::new();
        for i in 0..node.num_inputs {
            let target = Port {
                node: node.id,
                index: i,
            };
            let sources: Vec<String> = random
                .connections
                .iter()
                .filter(|c| c.to == target)
                .map(|c| c.from_label())
                .collect();
            ports.push(if sources.is_empty() {
                format!("i{i}* (orphan → net_input)")
            } else {
                format!("i{i}←{}", sources.join("+"))
            });
        }
        println!(
            "  n{:<2} {:<8} {}",
            node.id,
            format!("{:?}", node.kind).to_lowercase(),
            if ports.is_empty() {
                "no input ports".to_string()
            } else {
                ports.join(" · ")
            }
        );
    }
    println!();

    // ── 6. Combine ops ──────────────────────────────────────────────────────
    println!("═ 6. Combine ops — Add vs Mean ═");
    // A fan-in node: two sources enter n2, so the combine op matters.
    let fan_in = |op: CombineOp| {
        let mut g = Topology::new(0, None);
        g.options.combine_op = op;
        g.nodes.push(Node::new_input(0, 2));
        g.nodes.push(Node::new_hidden(1, 1, 1));
        g.nodes.push(Node::new_output(2, 2, 1));
        g.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        g.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 1 },
        });
        g
    };
    for op in [CombineOp::Add, CombineOp::Mean] {
        let g = fan_in(op);
        g.validate().unwrap();
        let net = Network::build(&g, Device::CPU).unwrap();
        let out = net.forward(&rand_input(2, g.options.input_dim)).unwrap();
        println!(
            "  {op:?}: header shows `combine: {op:?}`, forward {:?} ✓",
            out.shape()
        );
    }
    // Per-node override: the graph says Add, but the fan-in node overrides.
    let mut g = fan_in(CombineOp::Add);
    g.nodes[2].combine_op = Some(CombineOp::Mean);
    g.validate().unwrap();
    let out = Network::build(&g, Device::CPU)
        .unwrap()
        .forward(&rand_input(2, g.options.input_dim))
        .unwrap();
    println!(
        "  per-node override: graph Add but node's combine_op = Some(Mean) → forward {:?} ✓",
        out.shape()
    );
    println!("  Add sums the incoming tensors; Mean averages them (÷ source count)");
    println!();

    // ── 7. Rebuild ──────────────────────────────────────────────────────────
    println!("═ 7. Uniqueness & rebuild ═");
    let a = Network::build(&random, Device::CPU).unwrap();
    let b = Network::build(&random, Device::CPU).unwrap();
    assert_ne!(a.name(), b.name());
    println!(
        "  two builds of the same blueprint get distinct names: {} vs {}",
        a.name(),
        b.name()
    );
    // Blueprint = source of truth: save → reload → rebuild (fresh weights).
    let json = random.to_json().unwrap();
    let rebuilt = Network::build(&Topology::from_json(&json).unwrap(), Device::CPU).unwrap();
    assert_eq!(rebuilt.parameters().len(), a.parameters().len());
    println!(
        "  JSON round-trip → rebuilt module: {} param tensors (same architecture, fresh weights)",
        rebuilt.parameters().len()
    );
    println!();

    // ── 8. Devices ──────────────────────────────────────────────────────────
    println!("═ 8. Devices ═");
    println!("  everything above builds on Device::CPU — the default.");
    println!(
        "  with `--features cuda`, build the same blueprint on the GPU:\n    \
         let gpu = Network::build(&topology, Device::CUDA(0))?;  // input must be on CUDA too"
    );

    println!("\n  ✅ network guide complete — every section ran");
}
