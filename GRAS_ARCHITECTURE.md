# 🧬 GRAS Architecture

> Source of truth is the code; this doc is the map.

## Two-phase design

| Phase | Module | What it holds | Runs? |
|-------|--------|---------------|-------|
| **Blueprint** | `topology.rs` | nodes (port counts, kind, dim/activation/combine overrides) + connections, options | No |
| **Execution** | `network.rs` | flodl `Linear` per node, orphan projections, port projections, wiring table | Yes |

`Network::build(&topology, device)` borrows the blueprint, derives everything else.

## Invariants

1. **Node ids are contiguous `0..n`** — id = execution order.
2. **Edges only go forward** (`from.node < to.node`) — ascending order = topological sort.
3. **All output ports emit the same tensor** — fan-out is free.
4. **Orphaned inputs legal** — fed via per-orphan projection from raw input.
5. **Orphaned outputs auto-rewired** by `finalize()`.

## The pipeline

```rust
let mut g = Topology::new(0, None);
g.create_random_hidden_nodes(5);
g.refresh_labels();
g.finalize();
g.validate().unwrap();
let net = Network::build(&g, Device::CPU).unwrap();
let y = net.forward(&x).unwrap();
```

## Forward — 4 steps per node

```text
raw input ──▶ orphan_proj ──▶ node ──▶ port_proj (if needed) ──▶ next node
```

1. **Gather** — each input port: wired source output (via port proj if dims mismatch) or orphan projection from raw input.
2. **Combine** — Add/Mean over gathered tensors.
3. **Transform** — `layers[node](combined)`.
4. **Activate** — node's activation + standardize.

**Variable hidden dims:** each hidden node draws its own output width from `hidden_dim_pool`. Port projections bridge mismatched dims automatically. `effective_hidden_dim()` computes the max output dim across all nodes — used as fallback for input node and orphan projections.

## Fitness — score/loss split

```rust
Fitness::new_with_loss(
    accuracy_score,              // score: ranking metric (↑ maximize)
    cross_entropy_onehot_loss,   // loss: training signal (↓ minimize)
    Direction::Maximize,
    Direction::Minimize,
    "accuracy",
)
```

- `score_fn`: `(pred, y) → f32` — ranks individuals on **eval** batches.
- `loss_fn`: `(pred, y) → Variable` — backward on **train** batches. Defaults to MSE if None.
- Engine samples non-overlapping train + eval batches per generation.

## Engine

`EngineOptions::builder()` routes knobs into engine, topology, GP pools, and network:

```rust
EngineOptions::builder()
    .set_pop_size(100)
    .set_num_generations(10)
    .set_hidden_dim_pool(8..=16)      // per-node variable width
    .set_combine_op_pool(vec![...])    // if omitted → all ops
    .set_activation_pool(vec![...])    // if omitted → all activations
    .set_num_batches(16)
    .set_num_epochs(1)
    .build()
```

**GP pools:** hidden_dim, combine_op, activation, standardize — drawn per individual. If a pool is empty, all built-in options are used.

**Seeding:** `seed: Option<u64>` — `None` = random per run (recorded as `run_seed`); `Some` = deterministic. Each individual derives its own seed via golden-ratio splitmix chain.

**Run output** in `results/<ts>/`:
- `improvements/` — one topology recipe per best-improvement
- `engine.json` — full option chain + run_seed + best topology + net facts

## Serialization

`Topology::to_json()` / `from_json()` stores the blueprint (no weights). Rebuild = `from_json` → `Network::build`.

## Scoring helpers (public in `gras::fitness`)

`accuracy_score`, `r2_score`, `f1_score`, `precision_score`, `cross_entropy_onehot`, `mse_loss_score`, `l1_loss_score`, `rmse_score`, `f1_from_vecs`, `precision_from_vecs`, `argmax_classes`, `cross_entropy_onehot_loss`.

## Modules

| Module | Role |
|--------|------|
| `topology.rs` | Blueprint: nodes, connections, validation, GP randomization |
| `network.rs` | Execution: build, forward, orphan/port projections |
| `engine.rs` | NAS loop: population, evaluation, selection, checkpointing |
| `fitness.rs` | Score/loss split, scoring helpers, Direction |
| `selection.rs` | Elitism + tournament (tested, wired into engine) |
| `trainer.rs` | Training loop: epochs, batches, optimizer, grad clip |
| `node.rs` | Node type: activations (8), standardize ops, NodeKind |
| `data.rs` | Tensor I/O: save/load datasets |
| `spec.rs` | JSON serialization for topology and network facts |
| `error.rs` | Error types |
| `utils/` | ASCII rendering, logging, synthetic data |

## Examples

- `cargo run` — minimal pipeline
- `cargo run --example engine_guide` — NAS walkthrough
- `cargo run --example fitness_guide` — scoring/loss API
- `cargo run --example topology_guide` — blueprint side
- `cargo run --example network_guide` — execution side
- `cargo run --example node_guide` — node type details
