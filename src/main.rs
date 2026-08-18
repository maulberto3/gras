//! 🧬 gras — the minimal pipeline, now engine-driven.
//!
//! The tight loop (random graph → wire → validate → build → forward) used to
//! live here by hand; the engine owns it now. This file shows the whole crate
//! in one run: tensors on disk → `Engine::new` seeds a random population →
//! `run()` scores every blueprint in parallel → the best is checkpointed.
//!
//! For the full API tour, see the guides:
//! `topology_guide` / `network_guide` / `fitness_guide` / `engine_guide`
//!
//! Run with: `source env_setup.sh && cargo run`
//!
//! ── ✅ what the crate offers today ─────────────────────────────────────────
//! • random topologies from ONE seed — a deterministic chain seeds every
//!   individual (the resolved `run_seed` is recorded in engine.json, so any
//!   run is reproducible)
//! • GP over BOTH layers — topology structure (port ranges, node counts)
//!   AND network values (hidden dim, combine op, per-node activations),
//!   sampled per individual from the pools in `EngineOptions`
//! • deterministic weight init — the base seed is baked into both the
//!   topology (blueprint) and the network options (weights), so same options
//!   ⇒ the exact same built model and the exact same run
//! • flat builder — `EngineOptions::builder().set_*()` routes into engine,
//!   topology, GP pools, and network options (no nested struct literals)
//! • engine: parallel population evaluation (rayon), built-in + custom
//!   fitness, direction-aware best, per-improvement checkpoint pairs,
//!   engine.json experiment envelope
//! • fitness: 4 continuous + 4 categorical built-ins, plus custom closures
//!   and trait-based scorers
//! • genetics::select (elitism + tournament) implemented + tested
//!
//! ── ⏳ still pending ───────────────────────────────────────────────────────
//! • crossover() / mutate()    — documented no-op stubs
//! • select()                  — implemented, not yet wired into the loop
//! • stop criteria             — only max generations (TargetFitness /
//!                               NoImprovement is a TODO in `Engine::run`)
//! • population checkpoints    — weights are never stored (fitness is a
//!                               random-init forward pass), so a run restarts
//!                               from seeds, not from saved state
//! • GPU (CUDA) device         — the engine builds on any `Device`; the CUDA
//!                               path is untested end to end
//! • data loaders              — tensors only (the flodl-native contract)

use flodl::Device;
use flodl::nn::Module;

use gras::data;
use gras::engine::{Direction, Engine, EngineOptions, Fitness};
use gras::fitness;
use gras::network::Network;
use gras::node::{Activation, NodeKind};
use gras::topology::CombineOp;

fn main() {
    // 1. Data — the engine's contract is a path to tensors written by
    //    `data::save_dataset` (inputs.bin + targets.bin + meta.json).
    let data_dir = std::path::Path::new("data/sine");
    if std::fs::read_dir(data_dir).is_err() {
        let ds = fitness::synthetic_sine(128, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
        println!("  saved synthetic sine dataset → {}/", data_dir.display());
    } else {
        println!("  found existing dataset → {}/", data_dir.display());
    }

    // 2. Options — the flat builder. Each set_* routes into the right layer:
    //    engine knobs, the topology template, the GP pools (sampled per
    //    individual), or the network options.
    let opts = EngineOptions::builder()
        .set_pop_size(8)
        .set_num_generations(1)
        .set_seed(Some(42)) // fixed → every run reproduces the same population
        .set_input_dim(1) // must match the dataset's [n, input_dim]
        .set_hidden_dim_pool(8..=16) // 🎛️ GP: per-individual hidden dim
        .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean]) // 🎛️ GP
        .set_activation_pool(vec![
            Activation::Identity,
            Activation::ReLU,
            Activation::GeLU,
            Activation::SELU,
        ]) // 🎛️ GP: per-node activations
        .set_num_threads(3) // ⚡ parallel population evaluation
        // Budget — how much of the dataset each fitness evaluation sees:
        // 1 epoch × 4 batches of 64 rows (of the 128-row sine set). Batching
        // keeps memory bounded and makes each score a sampled estimate.
        .set_num_epochs(1)
        .set_num_batches(4)
        .set_batch_size(64)
        .build()
        .unwrap();

    // 3. The engine — the options cascade. `EngineOptions` embeds BOTH the
    //    topology template and the network options; each flows inward:
    //    every individual clones the template (overriding only its derived
    //    seed), and every network is built with the network options.
    println!("   engine options — the full cascade:");
    let (topo, net) = (opts.topology, opts.network);
    println!("   ┌ engine   {opts}");
    println!("   ├─ topology {topo}");
    println!("   └─ network  {net}");

    // 4. Engine::new — a population of random blueprints, one per individual.
    let mut engine = Engine::new(opts, data_dir, Fitness::mse()).unwrap();
    println!("🧬 run {} → {}/", engine.run_id, engine.run_dir.display());
    println!("   population of {} random blueprints:", engine.pop.len());
    for (i, g) in engine.pop.iter().enumerate() {
        println!(
            "     pop[{i}] id {} · {} nodes · {} wires · {} in-ports",
            g.id,
            g.nodes.len(),
            g.connections.len(),
            g.graph_inputs.len()
        );
    }

    // 5. The flow, made visible: every individual is a (topology, network)
    // pair sampled from the pools — a derived seed, a GP-chosen hidden dim
    // (per-INDIVIDUAL: per-node dims would break fan-in merging, so they
    // stay uniform within a graph), and PER-NODE combine ops + activations
    // (each hidden node draws its own; the individual's base is the
    // fallback for combines).
    println!("   flow check — GP-sampled per individual/node:");
    for (i, g) in engine.pop.iter().enumerate() {
        let combines: Vec<String> = g
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Hidden)
            .map(|n| format!("{:?}", n.combine_op.unwrap_or(g.options.combine_op)))
            .collect();
        let acts: Vec<String> = g
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Hidden)
            .map(|n| n.activation.to_string())
            .collect();
        println!(
            "     pop[{i}] seed {} · hidden {} · combines [{}] · acts [{}]",
            g.options.seed,
            g.options.hidden_dim,
            combines.join(", "),
            acts.join(", ")
        );
    }
    println!(
        "   every network is built on device {:?} (the network options)",
        engine.options.network.device
    );

    // 6. Run — 1 generation of fitness evaluation. Each individual is
    //    scored on the budget set above: 1 epoch × 4 batches of 64 rows,
    //    the same sampled batches reused across the population so scores
    //    are comparable (num_batches = 0 would mean one full pass instead).
    engine.run().unwrap();

    // 7. Every score, made legible — the whole population ranked by fitness.
    //    Mse is a LOSS (minimize direction), so the best is the LOWEST
    //    score; `engine.best` picks it direction-aware, we just show why.
    let dir = engine.fitness.direction();
    let better = if dir == Direction::Minimize {
        "lowest"
    } else {
        "highest"
    };
    let mut ranked: Vec<(usize, f64)> = engine.scores().iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| {
        if dir == Direction::Minimize {
            a.1.total_cmp(&b.1)
        } else {
            b.1.total_cmp(&a.1)
        }
    });
    let best_idx = ranked[0].0;
    println!(
        "   population fitness — fitness {:?} ({better} = better):",
        engine.options.fitness
    );
    for (rank, (i, s)) in ranked.iter().enumerate() {
        let marker = if *i == best_idx { "  ← best" } else { "" };
        println!("     #{rank} pop[{i}]  {s:.4}{marker}");
    }

    // 8. What we got — the direction-aware best, its compact log line, the
    //    rebuildable blueprint, and the checkpointed artifacts.
    let best = engine.best.as_ref().unwrap();
    println!(
        "   best = pop[{best_idx}] · fitness {:.4} ({} = better) after {} gen(s)",
        best.fitness, better, engine.generation
    );
    for line in std::fs::read_to_string(engine.run_dir.join("log.txt"))
        .unwrap()
        .lines()
    {
        println!("     {line}");
    }
    let net = Network::build(&best.topology, Device::CPU).unwrap();
    println!(
        "   best blueprint: {} nodes · {} wires · {} param tensors (recipe → Network::build)",
        best.topology.nodes.len(),
        best.topology.connections.len(),
        net.parameters().len()
    );
    // 8b. The artifacts, annotated — each file's role in the experiment.
    println!("   artifacts → {}/", engine.run_dir.display());
    let mut entries: Vec<String> = std::fs::read_dir(engine.run_dir.clone())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    for e in entries {
        // Inline notes per artifact: what lives inside, and how to use it.
        let note = match e.as_str() {
            "engine.json" => {
                "  # whole experiment: options + run_seed + best → from_json replicates the run"
            }
            "log.txt" => "  # compact per-gen log (best / mean / worst)",
            "improvements" => "  # one pair per best-improvement (annotated below)",
            _ => "",
        };
        println!("     {e}{note}");
    }
    if let Ok(files) = std::fs::read_dir(engine.run_dir.join("improvements")) {
        let mut names: Vec<String> = files
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for n in names {
            // The pair: `.json` = the winning blueprint recipe, `.net.json` =
            // the built network's nutrition facts (dims, degrees, orphans…).
            let note = if n.ends_with(".net.json") {
                "  # built-network facts → Network::from_json"
            } else if n.ends_with(".json") {
                "  # best-topology recipe → Topology::from_json"
            } else {
                ""
            };
            println!("     improvements/{n}{note}");
        }
    }

    // 9. What the crate does NOT offer yet — the evolution roadmap.
    println!();
    println!("⏳ pending in the crate:");
    println!("   • crossover() / mutate() — documented no-op stubs");
    println!("   • select() — implemented, not yet wired into next_generation()");
    println!("   • stop criteria — only max generations (TargetFitness / NoImprovement TODO)");
    println!("   • population checkpoints — weights are never stored (fitness is a random-init");
    println!("     forward pass), so runs restart from the seed chain, not from saved state");
    println!("   • GPU (CUDA) device — engine builds on any Device; CUDA untested end to end");
    println!("   • data loaders — tensors only (the flodl-native contract)");
}
