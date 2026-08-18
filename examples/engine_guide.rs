//! 🏭 The engine guide — the flagship NAS API in one walkthrough.
//!
//! Run with: `source env_setup.sh && cargo run --example engine_guide`
//! Scoring details → `fitness_guide`; blueprint/execution → `topology_guide` / `network_guide`.

use flodl::nn::Module;
use flodl::{Device, Variable};

use gras::data;
use gras::engine::{Engine, EngineOptions, Fitness};
use gras::fitness;
use gras::network::Network;
use gras::node::Activation;
use gras::topology::{CombineOp, Topology};

fn main() {
    // The engine logs via the `log` crate — init a compact logger.
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // 1. Setup — data (tensors dir) + options (flat builder, GP pools).
    let data_dir = std::path::Path::new("data/sine");
    if std::fs::read_dir(data_dir).is_err() {
        let ds = fitness::synthetic_sine(256, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }
    let opts = EngineOptions::builder()
        .set_pop_size(8)
        .set_num_generations(6)
        .set_input_dim(1) // must match the dataset's [n, input_dim]
        // GP pools: hidden dim per INDIVIDUAL (per-node dims would break
        // fan-in merging); combine op + activation PER NODE.
        .set_hidden_dim_pool(8..=16)
        .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
        .set_activation_pool(vec![
            Activation::Identity,
            Activation::ReLU,
            Activation::GeLU,
            Activation::SELU,
        ])
        // Budget: 4 batches × 32 rows per epoch (0 batches = whole dataset once).
        .set_num_epochs(1)
        .set_num_batches(4)
        .set_batch_size(32)
        .set_num_threads(3)
        .build()
        .unwrap();
    println!("═ 1. Options — the full cascade (engine → topology → network) ═");
    let (topo, net) = (opts.topology, opts.network);
    println!("  ┌ engine   {opts}");
    println!("  ├─ topology {topo}");
    println!("  └─ network  {net}");
    println!("  (seed None = randomized per run, recorded as run_seed in engine.json)");
    println!();

    // 2. Engine::new — a population of random blueprints, one per individual.
    println!("═ 2. Engine::new — seeding the population ═");
    let mut engine = Engine::new(opts.clone(), data_dir, Fitness::mse()).unwrap();
    println!("  run {} → {}", engine.run_id, engine.run_dir.display());
    for (i, g) in engine.pop.iter().enumerate() {
        println!(
            "    pop[{i}] {} nodes · {} wires · {} in-ports",
            g.nodes.len(),
            g.connections.len(),
            g.graph_inputs.len()
        );
    }
    println!();

    // 3. Run — the NAS loop: evaluate → log → evolve.
    println!("═ 3. Engine::run ═");
    engine.run().unwrap();
    let best = engine.best.as_ref().unwrap();
    println!(
        "  best after {} gens: fitness {:.4} (lower = better)",
        engine.generation, best.fitness
    );
    for line in std::fs::read_to_string(engine.run_dir.join("log.txt"))
        .unwrap()
        .lines()
    {
        println!("    {line}");
    }
    // The raw scores are queryable too — `scores()[i]` matches `pop[i]`.
    println!(
        "  raw scores (index-aligned with pop): {:?}",
        engine.scores()
    );
    println!();

    // 4. Fitness routes — built-in vs a drop-in closure, same path.
    println!("═ 4. Fitness routes — built-in vs drop-in ═");
    let mut engine2 = Engine::new(
        EngineOptions {
            pop_size: 4,
            num_generations: 3,
            ..opts.clone()
        },
        data_dir,
        // Minimal closure: prediction vs target — the engine runs the
        // forward pass, so the user only writes the metric.
        Fitness::custom(|pred: &Variable, y: &Variable| {
            let diff = pred.data().sub(&y.data())?;
            diff.abs()?.mean()?.item() // mean absolute error
        }),
    )
    .unwrap();
    engine2.run().unwrap();
    println!(
        "  custom MAE run best: {:.4}",
        engine2.best.as_ref().unwrap().fitness
    );
    println!();

    // 5. The run folder — improvements/ pair + log.txt + engine.json.
    println!("═ 5. The run folder ═");
    let run_dir = engine.run_dir.clone();
    let mut entries: Vec<String> = std::fs::read_dir(&run_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    println!("  {} — {}", run_dir.display(), entries.join(", "));
    let imp_dir = run_dir.join("improvements");
    if let Ok(files) = std::fs::read_dir(&imp_dir) {
        let mut names: Vec<String> = files
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        println!("  improvements/ — recipe + .net.json facts pair per best-improvement:");
        for n in &names {
            println!("    {n}");
        }
    }
    println!();

    // 6. Replication — engine.json alone rebuilds the best network.
    println!("═ 6. Replication — Engine::to_json → rebuild ═");
    let envelope: serde_json::Value = serde_json::from_str(&engine.to_json().unwrap()).unwrap();
    println!(
        "  engine.json: run_seed {} · best_fitness {:.4} · data {}",
        envelope["run_seed"].as_u64().unwrap(),
        envelope["best_fitness"].as_f64().unwrap(),
        envelope["data_path"].as_str().unwrap(),
    );
    let facts: serde_json::Value =
        serde_json::from_str(envelope["best_net_facts"].as_str().unwrap()).unwrap();
    println!(
        "  best_net_facts: {} nodes · {} wires · {} param elements",
        facts["num_nodes"].as_u64().unwrap(),
        facts["num_wires"].as_u64().unwrap(),
        facts["param_elements"].as_i64().unwrap(),
    );
    // Rebuild the best from the file alone.
    let best_graph = Topology::from_json(envelope["best_topology"].as_str().unwrap()).unwrap();
    let best_net = Network::build(&best_graph, Device::CPU).unwrap();
    let opts_t = flodl::TensorOptions {
        dtype: flodl::DType::Float32,
        device: Device::CPU,
    };
    let input = Variable::new(
        flodl::Tensor::randn(&[2, best_graph.options.input_dim as i64], opts_t).unwrap(),
        false,
    );
    let output = best_net.forward(&input).unwrap();
    println!(
        "  rebuilt best: {} nodes · {} param tensors · forward {:?} -> {:?}",
        best_graph.nodes.len(),
        best_net.parameters().len(),
        input.shape(),
        output.shape()
    );
    println!();

    // 7. Genetics — select() ready; crossover/mutate are documented no-ops.
    println!("═ 7. Genetics — select() ready, wiring pending ═");
    engine.crossover();
    engine.mutate();
    println!("  crossover()/mutate() are no-op stubs; select() (elitism + tournament)");
    println!("  is implemented in src/genetics.rs but not yet wired into next_generation().");

    println!("\n  ✅ engine guide complete — every section ran");
}
