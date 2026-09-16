# gras

### Neural Architecture Search via a Continuous Step-Race

`gras` is a lightweight, high-performance Genetic Programming library for Neural Architecture Search in Rust. It evolves neural network topologies under a **step-race**: a unified, continuous evolutionary loop where all networks train and evaluate synchronously on a shared deterministic stream, culling and birthing individuals dynamically.

## Why gras?

Hand-designing network architectures is slow, error-prone, and biased. 
`gras` automates discovery: each hidden node independently evolves its own width, activation function, merge (combine) operation, and normalization structure—all dynamically wired and evaluated in real-time.

### Representation: Hand-designed vs. Evolved

**Hand-designed** — uniform, straight pipeline:

```mermaid
graph LR
    I([Input 100]) --> H1[64 · relu]
    H1 --> H2[64 · relu]
    H2 --> O([Output 1])
```

**gras evolved** — diverse widths, heterogeneous activations, complex multi-path routing:

```mermaid
graph LR
    I([Input 100]) ==> H1[64 · relu]
    H1 ==> H2[[96 · gelu]]
    H1 ==> H3(32 · silu)
    I ==> H4[48 · mish]
    H2 ==> H4
    H3 ==> H4
    H2 ==> H5[64 · tanh]
    H4 ==> H5
    H5 ==> O([Output 1])
```

---

### How the Step-Race Loop Works

Evolution is continuous, always active, and gated by a **historical checkpoint ledger**:

```mermaid
graph TD
    A([Seed Initial Population]) --> B[Group Step:<br/>Every live net trains on same batch]
    B --> C[Every Step:<br/>Trigger Crossover & Mutation Rolls]
    C --> D{Crossover child<br/>clears checkpoint gates?}
    D -->|Yes| E[Cull worst net &<br/>Insert crossover child]
    D -->|No| F[Discard child<br/>pop unchanged]
    C --> G[Mutation Roll:<br/>Select victim via inverse fitness]
    G --> H[Cull victim &<br/>Insert random immigrant]
    E --> I{clock > 0 &&<br/>clock % checkpoint_every == 0?}
    F --> I
    H --> I
    I -->|Yes| J[Write pop-mean fitness<br/>to checkpoints.json ledger]
    I -->|No| B
    J --> B
```

---

## Features

- **Continuous Step-Race Engine** — Dynamic culling and insertion on every step instead of rigid generation boundaries.
- **Checkpoint-Gated Crossover** — Crossover children must prove themselves during a history-replay catch-up by clearing a gate of historical population means.
- **Data-Agnostic Trainer Interface** — The evolutionary loop handles selection, culling, and metadata. You own the training recipe (Adam, SGD, schedules, custom losses).
- **100% Deterministic** — Identical `run_seed` and config guarantees identical weight initializations, batches, and evolutionary histories.
- **Bulletproof Resume** — `RaceEngine::resume` reloads active networks, ignores culled tombstones, validates population counts, and replays each net with bit-identical metric parity checked upon revival.
- **16 Activations** — Identity, ReLU, GeLU, SiLU, SELU, Tanh, Sigmoid, Mish, LeakyReLU, ELU, GeluTanh, Softplus, HardSwish, HardSigmoid, Sin, Cos.
- **Combine (Merge) Operators** — Add, Mean, Max, Min (+ Multiply, Subtract, Divide).
- **Standardize Operators** — Identity, LayerNorm.
- **Rich Diagnostic Visualization** — Topology markdown tables, edge lists, ASCII diagrams, and native Mermaid flowchart generation.

---

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
gras = "1.0"

# Compile with CUDA support
gras = { version = "1.0", features = ["cuda"] }
```

### Requirements

- **Rust** (edition 2024+)
- **C/C++ compiler** — gcc/clang (needed by the underlying `flodl` autograd compilation)
- **libtorch** — PyTorch C++ runtime precompiled for your platform

Before compiling or running, initialize the environment:
```bash
# CPU setup
source env_setup.sh
cargo build

# CUDA setup
source env_setup.sh cuda
cargo build $GRAS_FEATURES
```

---

## Quick Start Showcase (`examples/mnist.rs`)

We ship a complete, first-class documented showcase example under `examples/mnist.rs` illustrating the full capability of the library and config builder. Examples are CLI-free — every knob is a named constant at the top of the file; edit and run:

```bash
# Run the MNIST race showcase on CPU
source env_setup.sh && cargo run --release --example mnist

# Run on GPU (if CUDA features are enabled)
source env_setup.sh cuda && cargo run --release -F cuda --example mnist
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed N` | random (recorded) | Run seed for 100% reproducible execution |
| `--pop N` | 20 | Population size (active live networks in the race) |
| `--steps N` | 10 | Maximum step budget for the race |
| `--checkpoint-every N` | 10 | Cadence (in steps) of checkpoint gate logs |
| `--log-level L` | "summ" | Verbosity level (`summ`, `minimal`, `full`, `none`) |

---

## Library Quick Start

```rust
use gras::engine::{Direction, Fitness, RaceConfig, RaceEngine, RunMode};
use gras::trainer::TabularTrainer;
use gras::utils::{tabular_data, score};

fn main() -> flodl::tensor::Result<()> {
    // 1. Resolve dataset — the dir must exist (no synthetic fallback)
    let data_dir = std::path::Path::new("data/mnist"); // train/ + test/ splits
    let (train, test) = tabular_data::resolve_train_test_datasets(data_dir)?;

    // 2. Define Ranking Fitness (engine culling metric)
    let fitness = Fitness::new(
        score::accuracy_score,
        Direction::Maximize,
        "accuracy",
    );

    // 3. Define Supervised Trainer (Adam, gradient clipping)
    let trainer = TabularTrainer::new(|pred, y| {
        score::label_smoothing_cross_entropy_loss(pred, y, 0.1)
    })
    .with_learning_rate(1e-3)
    .with_grad_clip(1.0);

    // 4. Configure the Race
    let config = RaceConfig::builder()
        .set_pop_size(10)
        .set_max_steps(100)
        .set_crossover_gate_checkpoint_every(10)

        // --- Operation Pools (boilerplate-free &[&str] API) ---
        .set_network_combine_ops(&["Mean", "Min", "Max"])
        .set_network_activations(&["ReLU", "SELU", "GELU"])
        .set_network_standardize_ops(&["Identity"])

        // --- Gating, Mode & Sidecar exports ---
        .set_crossover_gate(gras::engine::config::CrossoverGate::Soft) // Checkpoint gate strictness for crossover children
        .set_mode(RunMode::Tabular)                       // Paradigm mode
        .set_csv_export(true)                             // Writes the history.csv unified event log (metric + attempt rows)

        // --- Structural Constraints ---
        .set_topology_min_hidden_num_nodes(5)
        .set_topology_max_hidden_num_nodes(10)
        .build();

    // 5. Build and Run the RaceEngine
    let mut engine = RaceEngine::new(gras::engine::RunSpec {
        data_dir: data_dir.to_path_buf(),
        config,
        fitness,
        trainer: Box::new(trainer),
        seed: Some(42),
        run_dir: None,
    })?;

    let stop_reason = engine.run()?;
    println!("Race stopped: {:?}", stop_reason);
    Ok(())
}
```

---

## Detailed Option Surface

### 1. Budgets and Halts
First budget hit cleanly stops the race:

| Option | Default | Builder method | Description |
|--------|---------|----------------|-------------|
| `max_steps` | None | `.set_max_steps(n)` | Total global step budget limit |
| `max_target_fitness` | None | `.set_max_target_fitness(v)` | Stop once best network hits this fitness |

**Stop criteria are mutually exclusive.** Set exactly ONE of `max_steps` or `max_target_fitness` — both would fight for different things (a step budget vs. a quality bar), so the engine rejects a config with both (`build()` panics with a clear message). `custom_stop` is independent and always evaluated last, joining whichever criterion you picked.

**Post-race pruner (optional).** With `.set_pruner_pop(true)` + `.set_pruner_method(PopPrunerMethod::Hard)` + `.set_pruner_steps(steps)`, a stop reason becomes a **transition** instead of an exit: the engine culls every net except the top-`elite_count` (default 1: the champion; each cull is recorded as a `pruner` attempt row + a `pruned` tombstone) and keeps training the survivors for `steps` more steps — same trainer, optimizer state, LR, and shared stream; evolution and stop criteria are off (they are evolution-phase concerns). Every solo step lands in `history.csv` and the survivors' net JSONs like a race step.

### 2. Genetic Evolution Parameters

| Option | Default | Builder method | Description |
|--------|---------|----------------|-------------|
| `pop_size` | 5 | `.set_pop_size(n)` | Number of active networks in population |
| `checkpoint_every` | 10 | `.set_crossover_gate_checkpoint_every(n)` | Step cadence of gate checkpoints (feeds the crossover gate bars) |
| `crossover_rolls` | 1 | `.set_crossover_rolls(n)` | Crossover attempts rolled per step |
| `crossover_retries` | 0 | `.set_crossover_retries(n)` | Extra full retries per crossover roll after a gate rejection (fresh parents + generate + gate each retry; every attempt recorded in `history.csv`) |
| `mutate_rolls` | 1 | `.set_mutate_rolls(n)` | Immigrant rolls processed per step |
| `crossover_prob` | 0.5 | `.set_crossover_prob(v)` | Crossover execution probability |
| `mutate_prob` | 0.2 | `.set_mutate_prob(v)` | Mutation execution probability |
| `elite_count` | 1 | `.set_elite_count(k)` | Elite guard size: top-k nets immune to ALL culls (minimum 1 — the champion is always guarded). Elites hold rank, not identity |

---

## Data-Agnostic "Bring Your Own Trainer"

The `RaceEngine` acts as an orchestrator. It knows nothing about backpropagation or loss computations. By implementing the `Trainer` trait (`src/trainer/mod.rs`), you can configure custom training schemas entirely from user space:

```rust
pub trait Trainer {
    /// Make the optimizer for a newly built or reloaded network.
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer>;

    /// Train the network for one clock-step in place.
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &StepContext<'_>,
    ) -> flodl::tensor::Result<StepReport>;
}
```

Contract rules enforced at runtime:
1. **Replay Determinism** — Call `seed_step_randomness(ctx.net_seed, step, 0)` in `train_step` to synchronize random dropout masks.
2. **In-place updates** — The engine owns the networks in memory; you borrow, apply gradients, and return.

We ship `TabularTrainer` out of the box as a reference recipe, and `examples/custom_trainer.rs` showcases momentum, linear warmup schedules, and delayed evaluation strategies.

---

## Data Contract — Two Resolvers

Point the engine at a data directory; one of two layouts must be present
(**no synthetic fallback** — a missing/invalid dataset fails loudly at start,
never silently trains on made-up data):

**1. `resolve_train_test_datasets(dir)` — explicit splits (MNIST-style).**
The directory holds `train/` and `test/` subdirs, each with `inputs.csv|bin`
+ `targets.csv|bin` (what `data/mnist_data.rs` produces). Train rows come
only from `train/`, eval rows only from `test/` — eval is genuinely unseen.

**2. `resolve_inputs_targets_datasets(dir)` — single pool (Kaggle-style).**
The directory holds `inputs.csv|bin` + `targets.csv|bin` side by side. The
engine splits the pool internally with a seeded shuffle into
train/eval/gating pools. (`resolve_dataset` is an alias for this.)

CSV is converted to the native `.bin` format on first load and cached under
`<dir>/flodl_data/` — subsequent runs load bins directly. See
`data/kaggle_ev_s6e9/DATA.md` for a worked external-data recipe.

---

## Outputs and Tooling

### Run Directory Layout
Each run creates a dedicated run directory containing:
```
results/<timestamp>/
├── engine.json         # Run identity + the COMPLETE config snapshot (every knob)
├── checkpoints.json    # The evolution gate ledger history
├── history.csv         # Unified event log, typed by the `type` column.
│                       #   Individual identity = (hash, net_seed): the same
│                       #   topology hash can reappear across eras (re-born or
│                       #   regenerated children); net_seed disambiguates.
│                       #   `metric` rows (one per step × live net): step, hash,
│                       #   net_seed, origin, entered_at_step, train_loss,
│                       #   eval_loss, fitness, <informative metrics...>
│                       #   `attempt` rows (one per evolution event): branch,
│                       #   attempt, outcome (inserted / rejected_gate),
│                       #   gate_index, child_fitness, bar, victim,
│                       #   victim_net_seed, pop_size
└── nets/
    ├── <hash>.json         # Live net: topology, net_seed, step, last_metrics, meta
    └── <culled_hash>.json  # Tombstone: is_alive=false, culled_at_step, cull_reason,
                            #   final_smoothed_fitness
```

**Precision.** Every numeric field in these artifacts is the raw `f32` (serde /
`Display` round-trip, i.e. full precision). Only the console log rounds —
per-step metrics to 2 decimals, as `mean ± std`. There is no `options.csv`: run
settings live in `engine.json`, and the unified history lives in `history.csv`,
which covers **every net that ever lived** (culled ones included, via `metric`
rows) **and every evolution attempt** (inserted or gate-rejected, via `attempt`
rows) — not just the surviving frontier.

### Stop summary & champion topology

At stop (any log level), the run writes extra things:

- **`elite-<hash>.md`** — the champion's (elite rank #1) full topology as
  Markdown: nodes table, edge list, ASCII wiring diagram, and a Mermaid
  flowchart. Written unconditionally — it's an artifact, not a log line. In
  the Mermaid diagram, each hidden node's label carries its full op signature
  (`combine·std·activation dim`, e.g. `H1 add·layernorm·gelu 32`), **box
  borders scale with the node's `hidden_dim`** and **wire thickness scales
  with hop distance** (long-range skips render thick).
- **`elite-<hash>.safetensors`** — the champion's trained weights in the
  Hugging Face safetensors format (PyTorch loads it natively:
  `safetensors.torch.load_file`). One `node<N>.weight`/`node<N>.bias` pair
  per graph node. Also unconditional. For any OTHER net (tombstones
  included), export via the example:
  `cargo run --example export_champion -- <run_dir> <full-hash> <data_dir>`
  (rebuilds + replays the net's history through the deterministic stream).
  No training ever happens in Python — the format is weights-out for
  external verification/reuse only.
- A condensed stop report: frontier snapshot count, the resume command, and
  the final elite list (top-k by smoothed fitness with origin, birth step,
  and parameter count).

### Solo-Net Recovery Tooling (`examples/train_by_hash.rs`)
Old engines required complex analysis to train an interesting candidate further. In `gras`, simply point our training tool to any run directory and network hash. It will automatically load the blueprint, rebuild the network, replay the batch stream deterministically to restore its weights, and continue training solo:

```bash
cargo run --example train_by_hash [RUN_DIR] [NET_HASH]
```

---

## Profiling

Numbers-first (no profiler needed — works everywhere, incl. WSL2 without a PMU):
```bash
cargo bench --bench stream        # stream vs train/eval cost split
```

Flamegraph (native Linux, `perf` required — full setup in **SETUP.md §4**):
```bash
sudo apt install -y linux-tools-common linux-tools-generic   # generic perf (WSL2-safe)
cargo install flamegraph
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid        # allow sampling

source env_setup.sh
cargo flamegraph --profile profiling --bench flamegraph -- --steps 300          # bare group step
cargo flamegraph --profile profiling --bench flamegraph -- --steps 300 --evolve # + catch-up replay
```
The example runs a bounded, self-contained race with logging and file I/O off,
so the profile shows compute, not formatting. `--evolve` answers the evolution
path; the default answers the batch-stream question.

---

## License

`gras` is licensed under the MIT License.
