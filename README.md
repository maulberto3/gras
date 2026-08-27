<div align="center">

<!-- Replace with your logo: place it in assets/logo.png and uncomment the line below -->
<!-- <img src="assets/logo.png" alt="gras logo" width="200" /> -->

# 🧬 gras: Genetic Programming for Neural Architecture Search in Rust

[![Crates.io](https://img.shields.io/crates/v/gras.svg)](https://crates.io/crates/gras)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Evolves neural network topologies using genetic algorithms.
The engine builds, trains, and scores populations of random architectures in parallel,
then selects, crosses over, and mutates the best — repeating until convergence.

</div>

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Options Reference](#options-reference)
  - [Mandatory Fields](#mandatory-fields)
  - [Defaults](#defaults)
- [CUDA](#cuda)
- [Fitness](#fitness)
- [Data Format](#data-format)
  - [Required Layout](#required-layout)
  - [Binary Tensor Format](#binary-tensor-format)
  - [Framing Your Problem](#framing-your-problem)
- [Synthetic Datasets](#synthetic-datasets)
- [Rebuild from Checkpoint](#rebuild-from-checkpoint)
- [Architecture](#architecture)
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
- **Crossover** — one-point (matching-node pivot, subtree-like on DAG) or uniform (per-node swap)
- **Mutation** — one random type per individual: activation, combine op, or standardize op
- **16 activations** — Identity, ReLU, GeLU, SiLU, SELU, Tanh, Sigmoid, Mish, LeakyReLU, ELU, GeluTanh, Softplus, HardSwish, HardSigmoid, Sin, Cos
- **4 combine ops** — Add, Mean, Max, Min
- **2 standardize ops** — Identity, LayerNorm
- **Dropout** — configurable per-hidden-node regularization
- **Proportional batching** — class-balanced sampling for categorical data
- **Generation history** — optional per-gen best/worst metrics in engine.json

## Installation

```toml
[dependencies]
gras = "0.1"
```

Optional CUDA support:

```toml
gras = { version = "0.1", features = ["cuda"] }
```

## Setup

### Requirements

- **Rust** (edition 2024+) — install via [rustup](https://rustup.rs)
- **C compiler** — gcc/clang (needed by the flodl build system)
- **libtorch** — the PyTorch C++ runtime, precompiled for your platform

### CPU

gras works out of the box on CPU. Set the `LIBTORCH_PATH` environment variable to point at your libtorch installation, then build:

```bash
export LIBTORCH_PATH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
cargo build
```

### CUDA

For GPU support, you need:
1. **NVIDIA driver** — `nvidia-smi` should work
2. **CUDA toolkit** — matching your driver version
3. **libtorch (CUDA variant)** — compiled against the same CUDA version

Then build with the `cuda` feature:

```bash
export LIBTORCH_PATH=/path/to/libtorch-cuda
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
cargo build --features cuda
```

`auto_device()` will return `Device::CUDA(0)` when compiled with `--features cuda`, and `Device::CPU` otherwise.

### Verify

Run the quick showcase to verify everything works:

```bash
cargo run --example quick_showcase
```

> For detailed setup instructions (WSSL2, Ubuntu, fdl tool): [SETUP.md](SETUP.md)

## Quick Start

```rust
use flodl::Device;
use gras::{data, Engine, EngineOptions, Fitness, Direction, MutationMethod, SelectionMethod, CrossoverMethod};

fn main() {
    // 1. Data — tensors on disk (inputs.bin + targets.bin)
    let data_dir = std::path::Path::new("data/sine");
    if std::fs::read_dir(data_dir).is_err() {
        let ds = gras::synthetic::synthetic_sine(256, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    // 2. Options — mandatory: pop_size, num_generations, selection, crossover, mutation
    let opts = EngineOptions::builder()
        .set_pop_size(50)
        .set_num_generations(5)
        .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
        .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
        .set_mutation(MutationMethod::Activation { prob: 0.1 })
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

See [Examples](#examples) below for how to run this and other demos.

## Options Reference

### Mandatory Fields

These **must** be set or the builder rejects with an error:

| Field | Builder method | Why required |
|-------|---------------|--------------|
| `pop_size` | `.set_pop_size(n)` | Crossover needs ≥ 2 individuals |
| `num_generations` | `.set_num_generations(n)` | How many evolution cycles to run |
| `selection` | `.set_selection(kind)` | Must pick a selection strategy |
| `crossover` | `.set_crossover(kind)` | Must pick a crossover strategy |
| `mutation` | `.set_mutation(kind)` | Must pick at least one mutation strategy |

```rust
let opts = EngineOptions::builder()
    .set_pop_size(50)                          // ≥ 2
    .set_num_generations(10)                   // ≥ 1
    .set_selection(SelectionMethod::Tournament { tournament_size: 2 })  // required
    .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })       // required
    .set_mutation(MutationMethod::Activation { prob: 0.1 })               // required
    .build()
    .unwrap();
```

If you omit any of these, `build()` returns an error:

```
set_selection() is required — choose SelectionMethod::Tournament
set_crossover() is required — choose CrossoverMethod::OnePoint or Uniform
set_mutation() is required — choose MutationMethod::Activation, CombineOp, or Standardize
```

### Defaults

Everything else has conservative defaults. Set only what your experiment needs:

| Option | Default | Builder method |
|--------|---------|---------------|
| `seed` | None (random) | `.set_seed(Some(42))` |
| `num_threads` | 1 | `.set_num_threads(0)` (0 = auto) |
| `num_batches` | 4 | `.set_num_batches(16)` |
| `batch_size` | 32 | `.set_batch_size(64)` |
| `num_epochs` | 1 | `.set_num_epochs(5)` |
| `dropout_prob` | 0.05 | `.set_dropout_prob(0.1)` |
| `device` | CPU | `.set_device(Device::CUDA(0))` |
| `dtype` | Float32 | `.set_dtype(DType::Float32)` |
| `selection` | — (required) | `.set_selection(SelectionMethod::Tournament { tournament_size: 2 })` |
| `crossover` | — (required) | `.set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })` |
| `dedup_pop_and_fill` | false | `.set_dedup_pop_and_fill(true)` |
| `y_proportional_batches` | false | `.set_y_proportional_batches(true)` |
| `gens_history` | false | `.set_gens_history(true)` |
| `hidden_dim_pool` | 4..=8 | `.set_hidden_dim_pool(8..=32)` |
| `combine_op_pool` | all built-ins | `.set_combine_op_pool(vec![CombineOp::Add])` |
| `activation_pool` | all built-ins | `.set_activation_pool(vec![Activation::ReLU])` |
| `standardize_op_pool` | all built-ins | `.set_standardize_op_pool(vec![StandardizeOp::LayerNorm])` |
| `min_hidden_num_nodes` | 3 | `.set_min_hidden_num_nodes(5)` |
| `max_hidden_num_nodes` | 10 | `.set_max_hidden_num_nodes(20)` |
| `min_hidden_inputs_per_node` | 1 | `.set_min_hidden_inputs_per_node(2)` |
| `max_hidden_inputs_per_node` | 4 | `.set_max_hidden_inputs_per_node(6)` |
| `min_hidden_outputs_per_node` | 1 | `.set_min_hidden_outputs_per_node(2)` |
| `max_hidden_outputs_per_node` | 4 | `.set_max_hidden_outputs_per_node(6)` |

**Pools auto-fill:** When `combine_op_pool`, `activation_pool`, or `standardize_op_pool` are left empty, all built-in ops are included. Override with `set_*_pool(vec![...])` to restrict the search space.

### Built-in Ops

**Activations** (16): Identity, ReLU, GeLU, SiLU, SELU, Tanh, Sigmoid, Mish, LeakyReLU, ELU, GeluTanh, Softplus, HardSwish, HardSigmoid, Sin, Cos

**Combine ops** (4): Add, Mean, Max, Min

**Standardize ops** (2): Identity, LayerNorm

## CUDA

Enable CUDA with the cargo feature flag:

```toml
[dependencies]
gras = { version = "0.1", features = ["cuda"] }
```

Then build with the `cuda` feature:

```bash
cargo build --features cuda
```

The `auto_device()` helper detects the compiled feature automatically:

```rust
use gras::auto_device;

let opts = EngineOptions::builder()
    .set_device(auto_device())  // CUDA(0) when compiled with -F cuda, CPU otherwise
    .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
    .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
    .set_mutation(MutationMethod::Activation { prob: 0.1 })
    .build()
    .unwrap();
```

No code changes needed to switch between CPU and CUDA — just the feature flag at compile time.

### Environment

Set `LIBTORCH_PATH` and `LD_LIBRARY_PATH` to point at your libtorch installation before building. The repo includes an `env_setup.sh` convenience script for this.

## Fitness

### Simple (same function for training and ranking)

Use when the training loss IS the fitness metric:

```rust
Fitness::from_loss(
    |pred, y| flodl::nn::loss::mse_loss(pred, y),
    Direction::Minimize,
    "mse",
)
```

### Dual (separate training loss from evolution fitness)

Use when you want to rank on one metric but train on another. For example, rank by F1 but minimize cross-entropy during training:

```rust
Fitness::from_loss_with_diff(
    |pred, y| f1_score(pred, y),         // evolution ranks on this
    Direction::Maximize,
    "f1",
    |pred, y| cross_entropy_onehot_loss(pred, y),  // training minimizes this
    Direction::Minimize,
    "cross_entropy",
)
```

### Built-in scorers

**Continuous** (targets `[n, 1]`):

| Function | Description | Direction |
|----------|-------------|-----------|
| `mse_loss_score` | Mean squared error | Minimize |
| `rmse_score` | Root mean squared error | Minimize |
| `l1_loss_score` | Mean absolute error | Minimize |
| `r2_score` | R² (coefficient of determination) | Maximize |

**Categorical** (targets `[n, C]` one-hot):

| Function | Description | Direction |
|----------|-------------|-----------|
| `accuracy_score` | Fraction correct (argmax) | Maximize |
| `f1_score` | Macro-averaged F1 | Maximize |
| `precision_score` | Macro-averaged precision | Maximize |
| `cross_entropy_onehot` | Cross-entropy score | Minimize |
| `cross_entropy_onehot_loss` | Cross-entropy (Variable, for backward) | Minimize |

**Custom** — any closure `fn(&Variable, &Variable) -> Result<f32>`:

```rust
Fitness::from_loss(
    |pred, y| {
        let my_metric = custom_computation(pred, y)?;
        Ok(my_metric)
    },
    Direction::Maximize,
    "my_metric",
)
```

## Data Format

### Required Layout

The engine expects a directory containing two tensor files:

```
data/your_problem/
├── inputs.bin     # [n, input_dim]  — feature tensor
└── targets.bin    # [n, output_dim] — label tensor
```

- **inputs** must be 2-D: `[n_samples, input_dim]`
- **targets** must be 2-D: `[n_samples, output_dim]`
- Both are `f32` binary tensors (flodl native format)
- `input_dim` and `output_dim` are **auto-detected** from the tensor shapes — no need to specify them

### Binary Tensor Format

Each `.bin` file is a self-contained tensor:

```
magic "GRA1" (4 bytes) | dtype tag (1 byte) | ndim (u64 LE)
| shape (ndim × u64 LE) | raw bytes
```

Use `data::save_tensor` / `data::load_tensor` to work with individual tensors, or `data::save_dataset` / `data::load_dataset` for the full dataset.

### Framing Your Problem

**Regression** — 1 output, continuous target:

```text
inputs:  [n, features]    e.g. [256, 1] for sine
targets: [n, 1]           e.g. [256, 1] y = sin(2πx)
```

```rust
use gras::data::{save_dataset, Dataset};
use flodl::{Tensor, Device};

let inputs = Tensor::from_f32(&x_vals, &[256, 1], Device::CPU).unwrap();
let targets = Tensor::from_f32(&y_vals, &[256, 1], Device::CPU).unwrap();
save_dataset(Path::new("data/my_reg"), &Dataset { inputs, targets }).unwrap();
```

**Binary classification** — 1 output, target is 0.0 or 1.0:

```text
inputs:  [n, features]    e.g. [1000, 4]
targets: [n, 1]           e.g. [1000, 1] each row is 0.0 or 1.0
```

Use with `sigmoid` activation and `mse_loss` or `cross_entropy`.

**Multi-class** — C outputs, one-hot targets:

```text
inputs:  [n, features]    e.g. [1024, 784] (flattened 28×28)
targets: [n, C]           e.g. [1024, 10]  one-hot (each row sums to 1)
```

Use `f1_score` / `cross_entropy_onehot_loss` for fitness, and `.set_y_proportional_batches(true)` for class-balanced sampling.

```rust
// Quick one-hot encoding helper
use gras::synthetic::one_hot;
let class_indices = vec![0, 2, 1, 3, ...];
let targets = one_hot(&class_indices, 10, Device::CPU).unwrap();
```

**Multi-output regression** — multiple continuous targets:

```text
inputs:  [n, features]    e.g. [500, 3]
targets: [n, outputs]     e.g. [500, 4] — 4 target values per sample
```

## Synthetic Datasets

Built-in generators for quick testing (no data preparation needed):

| Generator | Task | Input dim | Target dim |
|-----------|------|-----------|------------|
| `synthetic_sine(n, seed, device)` | Regression | 1 | 1 |
| `synthetic_poly3(n, seed, device)` | Regression | 1 | 1 |
| `synthetic_sigmoid(n, seed, device)` | Regression | 1 | 1 |
| `synthetic_multi_sine(n, seed, device)` | Regression | 5 | 1 |
| `synthetic_xor(n, seed, device)` | Classification | 2 | 2 (one-hot) |
| `synthetic_blobs(n, seed, device)` | Classification | 2 | 3 (one-hot) |
| `synthetic_spiral(n, seed, device)` | Classification | 2 | 3 (one-hot) |
| `synthetic_iris_like(n, seed, device)` | Classification | 4 | 3 (one-hot) |
| `synthetic_classification(n, in_dim, out_dim, seed, device)` | Classification | custom | custom (one-hot) |

```rust
use gras::synthetic;
use gras::data::save_dataset;

// Quick smoke test — sine regression
let ds = synthetic::synthetic_sine(256, 42, Device::CPU).unwrap();
save_dataset(Path::new("data/sine"), &ds).unwrap();

// Custom classification
let ds = synthetic::synthetic_classification(1024, 784, 10, 42, Device::CPU).unwrap();
save_dataset(Path::new("data/mnist/train"), &ds).unwrap();
```

## Rebuild from Checkpoint

Every run saves `engine.json` and per-generation snapshots to `improvements/`. All share the same JSON format.

### Generation history

Enable with `.set_gens_history(true)`. The `engine.json` will include a `history` array with per-gen best/worst metrics:

```json
{
  "history": [
    { "generation": 0, "best_score": 0.4193, "best_loss": 1.8688, "worst_score": 0.0168, "worst_loss": 2.2691 },
    { "generation": 1, "best_score": 0.5488, "best_loss": 1.4182, "worst_score": 0.0564, "worst_loss": 2.2498 }
  ]
}
```

When disabled (default), `"history": null`.

### Load best topology from engine.json

```rust
use gras::{Topology, Network};
use flodl::Device;

let v: serde_json::Value = serde_json::from_str(
    &std::fs::read_to_string("results/1787795454685/engine.json").unwrap()
).unwrap();
let topo = Topology::from_json(v["best_topology"].as_str().unwrap()).unwrap();
let net = Network::build(&topo, Device::CPU).unwrap();
```

### Load any snapshot (best or worst from any generation)

```rust
use gras::{Topology, Network};
use flodl::Device;

let v: serde_json::Value = serde_json::from_str(
    &std::fs::read_to_string("results/1787795454685/improvements/gen09_best_0.5401.json").unwrap()
).unwrap();
let topo = Topology::from_json(v["best_topology"].as_str().unwrap()).unwrap();
let net = Network::build(&topo, Device::CPU).unwrap();
```

### Rebuild with custom options

```rust
use gras::{Topology, Network, NetworkOptions};
use flodl::{Device, DType};

let net = Network::build_with_options(&topo, &NetworkOptions {
    device: Device::CUDA(0),
    dtype: DType::Float32,
    seed: 42,
    dropout_prob: 0.0,
}).unwrap();
```

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
            crossover          — one-point or uniform
            dedup_population   — remove crossover duplicates
            refill_population  — fill back to pop_size (if enabled)
            mutate             — one random type per individual
    log_run_summary            — winner, artifacts, rebuild snippet
```

## Examples

| Example | What it shows |
|---------|--------------|
| `quick_showcase` | Minimal end-to-end run with all mandatory options |

## Contributing

Contributions are welcome! Please open an issue or PR on [GitHub](https://github.com/maulberto3/gras).

## License

MIT
