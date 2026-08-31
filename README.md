<div align="center">

<!-- <img src="assets/logo.png" alt="gras logo" width="200" /> -->

# 🧬 gras: Neural Architecture Search via Genetic Programming

[![Crates.io](https://img.shields.io/crates/v/gras.svg)](https://crates.io/crates/gras)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Evolves neural network topologies using genetic algorithms.
The engine builds, trains, and scores populations of random architectures in parallel,
then selects, crosses over, and mutates the best — repeating until convergence.

</div>

## Features

- **Evolutionary NAS** — evolve hidden layer count, dims, activations, combine ops, standardize ops per node
- **Variable hidden dims** — each node draws its own output width with configurable pool and stride
- **Parallel evaluation** — rayon-powered population scoring
- **Deterministic** — one seed reproduces the entire run: population, weights, batches
- **Built-in metrics** — MSE, MAE, RMSE, R2, accuracy, F1, precision, cross-entropy
- **Custom fitness** — drop-in closures for any metric
- **Checkpointing** — engine.json + per-generation snapshots with full topology
- **Deduplication** — removes duplicate topologies via full Spec comparison
- **Elitism** — configurable number of top individuals preserved each generation
- **Selection** — tournament with elitism
- **Crossover** — one-point (matching-node pivot, subtree-like on DAG) or uniform (per-node swap)
- **Mutation** — one random type per individual: activation, combine op, or standardize op
- **16 activations** — Identity, ReLU, GeLU, SiLU, SELU, Tanh, Sigmoid, Mish, LeakyReLU, ELU, GeluTanh, Softplus, HardSwish, HardSigmoid, Sin, Cos
- **4 combine ops** — Add, Mean, Max, Min
- **2 standardize ops** — Identity, LayerNorm
- **Dropout** — configurable per-hidden-node regularization
- **CSV support** — auto-detects CSV and converts to binary on first run
- **Robustness tracking** — tracks topology appearances, fitness stats, identifies most/least reliable architectures

## Installation

```toml
[dependencies]
gras = "0.1"

# Optional CUDA
gras = { version = "0.1", features = ["cuda"] }
```

## Setup

### Requirements

- **Rust** (edition 2024+) — install via [rustup](https://rustup.rs)
- **C compiler** — gcc/clang (needed by the flodl build system)
- **libtorch** — the PyTorch C++ runtime, precompiled for your platform

### CPU

```bash
export LIBTORCH_PATH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
cargo build
```

### CUDA

1. **NVIDIA driver** — `nvidia-smi` should work
2. **CUDA toolkit** — matching your driver version
3. **libtorch (CUDA variant)** — compiled against the same CUDA version

```bash
export LIBTORCH_PATH=/path/to/libtorch-cuda
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
cargo build --features cuda
```

`auto_device()` returns `Device::CUDA(0)` when compiled with `--features cuda`, `Device::CPU` otherwise.

## Quick Start

```rust
use std::path::Path;
use flodl::Device;
use gras::{data, Engine, EngineOptions, Fitness, Direction, MutationMethod, SelectionMethod, CrossoverMethod};
use gras::trainer::{SupervisedTrainer, TrainingConfig};

fn main() {
    // 1. Data
    let data_dir = Path::new("data/sine");
    if !data_dir.exists() {
        let ds = gras::synthetic::synthetic_sine(256, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    // 2. Fitness — pure scoring for ranking
    let fitness = Fitness::new(
        |p, y| {
            let diff = p.data().sub(&y.data()).unwrap();
            let sq = diff.mul(&diff).unwrap();
            Ok(sq.mean().unwrap().item().unwrap() as f32)
        },
        Direction::Minimize,
        "mse",
    );

    // 3. Trainer — owns data, loss, split, everything
    let trainer = SupervisedTrainer::new(
        data_dir, 1, 1,
        TrainingConfig { num_epochs: 1, ..Default::default() },
        |p, y| flodl::nn::loss::mse_loss(p, y),
    ).unwrap();

    // 4. Engine
    let opts = EngineOptions::builder()
        .set_pop_size(50)
        .set_num_generations(5)
        .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
        .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
        .set_mutation(MutationMethod::Activation { prob: 0.1 })
        .build()
        .unwrap();

    let mut engine = Engine::new(opts, fitness, Box::new(trainer)).unwrap();
    engine.run().unwrap();

    // 5. Inspect robustness
    engine.show_robustness(10, "both");
}
```

## Options Reference

### Mandatory Fields

| Field | Builder method | Constraint |
|-------|---------------|------------|
| `pop_size` | `.set_pop_size(n)` | ≥ 2 |
| `num_generations` | `.set_num_generations(n)` | ≥ 1 |
| `selection` | `.set_selection(kind)` | Required |
| `crossover` | `.set_crossover(kind)` | Required |
| `mutation` | `.set_mutation(kind)` | Required |

### Defaults

Everything else has conservative defaults. Set only what your experiment needs:

| Option | Default | Builder method |
|--------|---------|---------------|
| `seed` | None (random) | `.set_seed(Some(42))` |
| `num_threads` | 1 | `.set_num_threads(0)` (0 = auto) |
| `dropout_prob` | 0.0 | `.set_dropout_prob(0.1)` |
| `elite_count` | 2 | `.set_elite_count(5)` |
| `dedup_pop_and_fill` | false | `.set_dedup_pop_and_fill(true)` |

#### Topology

| Option | Default | Builder method |
|--------|---------|---------------|
| `hidden_dim_pool` | 8..=64 | `.set_hidden_dim_pool(16..=64)` |
| `hidden_dim_stride` | 8 | `.set_hidden_dim_stride(16)` |
| `min_hidden_num_nodes` | 2 | `.set_min_hidden_num_nodes(5)` |
| `max_hidden_num_nodes` | 8 | `.set_max_hidden_num_nodes(20)` |
| `min_hidden_inputs_per_node` | 1 | `.set_min_hidden_inputs_per_node(5)` |
| `max_hidden_inputs_per_node` | 8 | `.set_max_hidden_inputs_per_node(20)` |
| `min_hidden_outputs_per_node` | 1 | `.set_min_hidden_outputs_per_node(5)` |
| `max_hidden_outputs_per_node` | 8 | `.set_max_hidden_outputs_per_node(20)` |

#### Warm Start

| Option | Default | Builder method |
|--------|---------|---------------|
| `prior_topology` | None | `.set_prior_topology("engine.json")` |
| `prior_topologies` | None | `.set_prior_topologies(vec!["a.json", "b.json"])` |

## Fitness

### Built-in scorers

**Continuous** (targets `[n, 1]`):

| Function | Direction |
|----------|-----------|
| `mse_loss_score` | Minimize |
| `rmse_score` | Minimize |
| `l1_loss_score` | Minimize |
| `r2_score` | Maximize |

**Categorical** (targets `[n, C]` one-hot):

| Function | Direction |
|----------|-----------|
| `accuracy_score` | Maximize |
| `f1_score` | Maximize |
| `precision_score` | Maximize |
| `cross_entropy_onehot` | Minimize |
| `cross_entropy_onehot_loss` | Minimize (Variable, for backward) |

**Custom** — any closure `fn(&Variable, &Variable) -> Result<f32>`:

```rust
Fitness::new(
    |pred, y| {
        let my_metric = custom_computation(pred, y)?;
        Ok(my_metric)
    },
    Direction::Maximize,
    "my_metric",
)
```

## Data Format

### Supported Formats

| Format | Detection | Conversion |
|--------|-----------|------------|
| `.bin` (native) | `inputs.bin` + `targets.bin` exist | None — used directly |
| `.csv` | `inputs.csv` + `targets.csv` exist | Auto-converts to `.bin` in `flodl_data/` |

### Required Layout

```
data/your_problem/
├── inputs.csv     # [n, input_dim]  — feature tensor (CSV or .bin)
├── targets.csv    # [n, output_dim] — label tensor (CSV or .bin)
└── flodl_data/    # auto-created on first run (cached .bin)
    ├── inputs.bin
    └── targets.bin
```

- `input_dim` and `output_dim` are **auto-detected** from data shapes
- CSV headers are auto-detected (skipped if non-numeric)
- Lines starting with `#` are treated as comments

### Framing Your Problem

**Regression** — 1 output, continuous target:
```text
inputs:  [n, features]    targets: [n, 1]
```

**Binary classification** — 1 output, target 0.0 or 1.0:
```text
inputs:  [n, features]    targets: [n, 1]
```

**Multi-class** — C outputs, one-hot targets:
```text
inputs:  [n, features]    targets: [n, C]
```

**Multi-output regression** — multiple continuous targets:
```text
inputs:  [n, features]    targets: [n, outputs]
```

## Synthetic Datasets

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
| `synthetic_classification(n, in, out, seed, dev)` | Classification | custom | custom |

```rust
use gras::synthetic;
use gras::data::save_dataset;

let ds = synthetic::synthetic_sine(256, 42, Device::CPU).unwrap();
save_dataset(Path::new("data/sine"), &ds).unwrap();
```

## Outputs

### Run Directory Structure

```
results/<run_id>/
├── engine.json                    # Full run config + best topology + robustness
├── improvements/
│   ├── gen_00.json                # All individuals (seed, fitness, loss, params, topology)
│   ├── gen_00.md                  # Markdown for best topology
│   └── ...
```

### Robustness Tracking

At run end, the engine logs repeated topologies:

```
── repeated topologies (top 20) ──
  rank   appearances      mean   std_dev      min      max    params  topo_id
  #1             44    0.6103    0.0873   0.4611   0.7855   121610  3a7f2b1c
```

## Examples

| Example | Description |
|---------|-------------|
| `generate_md_from_gen` | Generate markdown for any topology from gen_XX.json |
| `train_from_gen` | Fully train a specific network and see results |

```bash
# Generate MD for a specific topology
cargo run --example generate_md_from_gen -- results/.../improvements/gen_00.json --best

# Via results directory + topo hash
cargo run --example generate_md_from_gen -- results/... 5e639bae

# Fully train a specific network
cargo run --example train_from_gen -- results/.../improvements/gen_00.json --best
```

## Contributing

Contributions are welcome! Please open an issue or PR on [GitHub](https://github.com/maulberto3/gras).

## License

MIT
