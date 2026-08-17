# 🧬 GRAS Architecture — how `Topology`, `Network`, and `Engine` fit together

> A developer's walkthrough of the graph model. Source of truth is the code;
> this doc is the map. Modules: `src/topology.rs` (blueprint), `src/network.rs`
> (execution), `src/node.rs` (node type), `src/spec.rs` (JSON), `src/engine.rs`
> (NAS loop), `src/fitness.rs` (scorers), `src/data.rs` (tensor I/O),
> `src/utils.rs` + `src/display.rs` (rendering), `src/error.rs` (errors).

## The two-phase design 🪜

A gras network is described in **two layers**:

| Layer | Module | What it holds | Runs? |
|---|---|---|---|
| **Blueprint** | [`Topology`](src/topology.rs) | nodes (port counts, kind, dim/activation overrides) + connections, options, seeded RNG. Pure data, no tensors. | No — inert |
| **Engine** | [`Network`](src/network.rs) | flodl `Linear` per node + input projection + precomputed wiring table | Yes — `forward()` |

`Network::build(&topology, device)` **borrows** the blueprint, clones what it
must own (connections, dims), and derives the rest (wiring table, per-node
dims). The blueprint stays the single source of truth: same graph, CPU or GPU,
serialized, or in a population.

## The invariants that make execution trivial

`forward` is a plain loop *because* these hold — all checked by
[`Topology::validate`](src/topology.rs); `build` refuses anything that fails:

1. **Node ids are contiguous `0..n`** — id = execution order = index into `layers`.
2. **Edges only go forward** (`from.node < to.node`) — ascending id *is* a
   topological order; no cycle detection at runtime.
3. **Output ports feed ≤ 1 input; input ports may hold many wires** — the node
   combines all incoming tensors (Add/Mean), so de-orphaning can stack wires.
4. **All output ports of a node emit the same tensor** — fan-out is free.
5. **Orphaned input ports are legal** — fed the network input (`net_input`) at
   execution; orphaned *outputs* are auto-rewired by `set_network`.

## The pipeline

```rust
let mut graph = Topology::new(0, None);                 // 1. empty blueprint
graph.create_random_hidden_nodes(5);                    // 2. random hidden nodes
graph.set_topology();                                   // 3. port labels (rendering)
graph.set_network();                                    // 4. scaffold + wire + de-orphan
graph.validate().unwrap();                              // 5. refuse broken graphs
let model = Network::build(&graph, Device::CPU).unwrap(); // 6. compile to linears
let y = model.forward(&input).unwrap();                 // 7. run it
```

- **`set_network()`** does everything in one call: `ensure_scaffold()` (prepend
  an `Input` node, guarantee exactly one `Output` node — random graphs only
  create hidden nodes), random 1:1 wiring from earlier output ports to later
  input ports, then `de_orphan_outputs()` (rewire unused outputs into
  compatible later nodes). Deterministic per `options.seed`.
- **`orphan_counts()`** → `(orphaned_inputs, orphaned_outputs)` for diagnostics.
- **`de_multi_outputs()`** merges > 1 `Output` nodes into one stacked
  projection node (the output projection, counterpart of `input_proj`).

## Inside `forward` — why a loop, not a `layers!` macro

Every node is a linear over its combined inputs, so a runtime loop *is* the
unrolling — a macro can't follow edges the RNG produced at runtime.
`build` precomputes the wiring once (`node_sources`: per input port, its
sources, or orphan); `forward` does **4 steps per node**:

```text
   x ──▶ input_proj ──▶ n0_out ──┐
                                 ▼
                            combined ──▶ layers[1] ──▶ act ──▶ n1_out = y
```

1. **Project** — `net_input = input_proj(x)` once; feeds every orphan/input node.
2. **Gather** — resolve each port via `node_sources` (`node_outputs[src]` if
   wired, `net_input` if orphan), summing into `combined`.
3. **Combine** — divide by source count if `CombineOp::Mean`, else leave the sum.
4. **Transform + activate** — `layers[node](combined)` then `activation.apply`.

## NAS evolution hooks 🧠

- `Node.hidden_dim: Option<usize>` — per-node output-width override.
- `Node.activation: Activation` — `Identity/ReLU/GeLU/SiLU/SELU/Tanh/Sigmoid/Mish`.
- `Topology::validate` — the guardrail mutation operators must not break.

## Serialization 🗂️

`Topology::to_json()` / `from_json()` (via `Spec` in `src/spec.rs`) store the
blueprint — options, nodes, labels, connections. **Weights are never stored.**
The RNG is re-seeded from `options.seed` on load, so a loaded graph rewires
identically. Rebuild = `from_json` → `Network::build`. Every topology JSON on
disk (`saved/`, `results/*/improvements/*.json`, `engine.json["best_topology"]`)
uses this same format, so any of them can be loaded by `Topology::from_json`.

## The engine 🏭

`Engine` (src/engine.rs) is the flagship public API: seed a population of
random topologies, score with a [`Fitness`](src/fitness.rs) (built-in `Mse`, a
drop-in closure via `Fitness::custom`, or a named `FitnessScorer`), track the
best, checkpoint, log compactly. Data is a **path to tensors**
(`data::save_dataset` format). Per run, `results/<ts>/` holds:

- `improvements/` — one blueprint JSON per best-improvement (`0000_gen00_fitness….json`)
- `log.txt` — one compact line per generation
- `engine.json` — `Engine::to_json()`: options + data path + best fitness +
  best topology, the single file to replicate the experiment

`Engine::new_seeded(opts, path, fitness, &[json])` starts a run from saved
topologies (e.g. a `saved/` file or the latest improvement) — they become the
first population entries, the rest are random. Genetics (`crossover`/`mutate`)
are documented no-op stubs, fully wired into the loop.

**Two views, one graph:** `Display` for a `Topology` is the Manhattan wire
*diagram* (nodes + wires + orphan markers); `Display` for a `Network` is the
compact *derived* table — per node, the layer dims (`in → out`), activation,
and incoming-input ops (per input port: its sources, or `*` when orphaned).
Both render from the same blueprint data via `src/utils.rs`.

## Missing data — catalog (not yet captured) 📋

Things a downstream process (NAS heuristics, logging, validation diagnostics,
rendering) might want that the blueprint/engine don't carry today. **Listed,
not implemented** — each is derivable from the existing data (nodes +
connections) or cheap to compute:

**Per node:**
- `orphaned_inputs: Vec<bool>` / `orphaned_outputs: Vec<bool>` — per-port
  orphan flags (currently only aggregate `orphan_counts()` exists).
- `in_degree` / `out_degree` — wired port counts (currently recomputed on the
  fly by callers).
- `depth` — longest path from the Input node; groups nodes into topological
  *levels* for printing and depth-biased NAS.
- `on_critical_path: bool` — whether the node lies on the Input → Output
  chain (useful for mutation targeting).
- `param_count` — `in_dim * out_dim + out_dim` for its layer.

**Per topology:**
- `depth_map: HashMap<usize, usize>` and level-grouped node ids.
- counts by kind (`Input`/`Hidden`/`Output`) and by activation.
- total parameter estimate (all nodes + input projection) without building.
- orphan summary per node (which input ports are orphaned) for validation
  diagnostics.

**Why:** fan-in/out and depth power NAS heuristics (prefer deeper/wider),
level groups make the ASCII renderer simpler (row = level instead of
id × 4), and per-port orphan flags make the debug views self-explanatory.

## Examples 🧪

- **`cargo run`** — the minimal pipeline (random → wire → validate → build → forward).
- **`cargo run --example topology_guide`** — the full API tour (hand-built,
  random, custom options, wiring introspection, JSON round-trip → `saved/`).
- **`cargo run --example engine_demo`** — both fitness routes on synthetic `x²`.
- **`cargo run --example mnist_data`** — download MNIST once, save as tensor
  datasets (reuses already-downloaded assets; `--force` to redo).
