# gras

> Genetic Programming for Neural Architecture Search in Rust.

[![Crates.io](https://img.shields.io/crates/v/gras.svg)](https://crates.io/crates/gras)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Evolves neural network topologies using genetic algorithms. The engine builds, trains, and scores populations of random architectures in parallel, then selects, crosses over, and mutates the best — repeating until convergence.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Overview](#api-overview)
- [Examples](#examples)
- [License](#license)

## Features

- **Evolutionary NAS** — evolve hidden layer count, dims, activations, combine ops, standardize ops per node
- **Variable hidden dims** — each node draws its own output width; projections bridge mismatched dims automatically
- **Dual fitness** — separate training loss (backward) from evolution ranking (fitness) via `from_loss_with_diff`
- **Parallel evaluation** — rayon-powered population scoring
- **Deterministic** — one seed reproduces the entire run: population, weights, batches
- **Flat builder** — `EngineOptions::builder().set_*()` routes into engine, topology, GP pools, network
- **Built-in metrics** — MSE, MAE, RMSE, R2, accuracy, F1, precision, cross-entropy (4 continuous + 4 categorical)
- **Custom fitness** — drop-in closures for any metric
- **Checkpointing** — engine.json + per-generation best/worst snapshots with full topology + rebuild snippet
- **Deduplication** — removes duplicate topologies via full Spec comparison
- **Selection** — tournament with elitism (best always survives)
- **Crossover** — two-point (matching-node pivot) or uniform (per-node swap)
- **Mutation** — one random type per individual: activation, combine op, standardize, hidden dim
- **Dropout** — configurable per-hidden-node regularization
- **Proportional batching** — class-balanced sampling for categorical data

## Installation

```toml
[dependencies]
gras = "0.1.2"
```

Optional CUDA support:

```toml
gras = { version = "0.1.2", features = ["cuda"] }
```

## Quick Start

```rust
use flodl::Device;
use gras::{data, Engine, EngineOptions, Fitness, Direction};

fn main() {
    // 1. Data — tensors on disk (inputs.bin + targets.bin + meta.json)
    let data_dir = std::path::Path::new("data/sine");
    if std::fs::read_dir(data_dir).is_err() {
        let ds = gras::synthetic::synthetic_sine(256, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    // 2. Options
    let opts = EngineOptions::builder()
        .set_pop_size(50)
        .set_num_generations(5)
        .set_hidden_dim_pool(8..=16)
        .set_num_batches(4)
        .set_batch_size(32)
        .set_num_epochs(1)
        .build()
        .unwrap();

    // 3. Run
    let fitness = Fitness::from_loss(
        |p, y| flodl::nn::loss::mse_loss(p, y),
        Direction::Minimize,
        "mse",
    );
    let mut engine = Engine::new(opts, data_dir, fitness).unwrap();
    engine.run().unwrap();

    // 4. Best architecture
    let best = engine.best.as_ref().unwrap();
    println!("best fitness: {:.4}", best.fitness);
    println!("{} nodes", best.topology.nodes.len());
}
```

Run with: `source env_setup.sh && cargo run --example quick_showcase`

## Architecture

```
Engine::new(options, data_path, fitness)
    validate_and_fill_options  — fill empty pools, check constraints
    resolve_seed               — user-provided or random
    load_data                  — read tensors, bind input_dim/output_dim
    create_population          — random topologies from seed
    dedup_population           — remove duplicates (if enabled)
    log_initialization         — print all resolved options

Engine::run()
    for each generation:
        evaluate_population    — parallel: build net, train, score
        save_snapshots         — best + worst to improvements/
        next_generation
            select             — tournament with elitism
            crossover          — two-point or uniform
            dedup_population   — remove crossover duplicates
            mutate             — one random type per individual
    log_run_summary            — winner, artifacts, rebuild snippet
```

## API Overview

### Engine Options

```rust
EngineOptions::builder()
    // Engine
    .set_seed(Some(42))
    .set_pop_size(200)
    .set_num_generations(10)
    .set_num_threads(0)           // 0 = auto-detect
    .set_results_dir("results")
    .set_progress_interval(50)    // print every N individuals
    .set_dedup_pop(true)          // remove duplicate topologies
    // Selection
    .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
    // Crossover
    .set_crossover(CrossoverKind::TwoPoint { action_prob: 0.5 })
    // .set_crossover(CrossoverKind::Uniform { action_prob: 0.7, swap_prob: 0.5 })
    // Mutation
    .set_mutation(MutationKind { mut_prob: 0.1, ..Default::default() })
    // GP pools (empty = all built-in ops)
    .set_hidden_dim_pool(8..=32)
    .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
    .set_activation_pool(vec![Activation::ReLU, Activation::GeLU])
    .set_standardize_op_pool(vec![StandardizeOp::LayerNorm])
    // Topology
    .set_min_hidden_num_nodes(3)
    .set_max_hidden_num_nodes(10)
    .set_min_hidden_inputs_per_node(1)
    .set_max_hidden_inputs_per_node(4)
    .set_min_hidden_outputs_per_node(1)
    .set_max_hidden_outputs_per_node(4)
    // Budget
    .set_num_batches(16)
    .set_batch_size(32)
    .set_num_epochs(1)
    .set_y_proportional_batches(true)  // class-balanced sampling
    // Training
    .set_learning_rate(1e-3)
    .set_optimizer(OptimizerKind::Adam)
    .set_grad_clip(1.0)
    .set_dropout_prob(0.1)
    // Network
    .set_device(Device::CPU)
    .set_dtype(DType::Float32)
```

### Fitness

Same function for training and ranking:

```rust
Fitness::from_loss(|p, y| mse_loss(p, y), Direction::Minimize, "mse")
```

Separate training loss from evolution fitness:

```rust
Fitness::from_loss_with_diff(
    |pred, y| f1_score(pred, y),       // evolution ranks on this
    Direction::Maximize,
    "f1",
    |pred, y| cross_entropy(pred, y),   // training minimizes this
    Direction::Minimize,
    "cross_entropy",
)
```

### Rebuild from checkpoint

```rust
use gras::{Topology, Network};

let v: serde_json::Value = serde_json::from_str(
    &std::fs::read_to_string("<path_to_json>").unwrap()).unwrap();
let topo = Topology::from_json(v["best_topology"].as_str().unwrap()).unwrap();
let net = Network::build(&topo, Device::CPU).unwrap();
```

Any `engine.json` or `improvements/*.json` file works — same format.

## Examples

| Example | What it shows |
|---------|--------------|
| `quick_showcase` | Minimal end-to-end run |
| `main` | Full MNIST test with F1 + crossover + mutation (not published) |

```sh
source env_setup.sh && cargo run --example quick_showcase
source env_setup.sh && cargo run --example main -- --seed 42
```

## License

MIT
