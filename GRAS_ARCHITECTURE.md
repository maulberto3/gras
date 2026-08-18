# 🧬 GRAS Architecture — how `Topology`, `Network`, and `Engine` fit together

> A developer's walkthrough of the graph model. Source of truth is the code;
> this doc is the map. Modules: `src/topology.rs` (blueprint), `src/network.rs`
> (execution), `src/node.rs` (node type), `src/spec.rs` (JSON), `src/engine.rs`
> (NAS loop), `src/genetics.rs` (selection), `src/fitness.rs` (scorers),
> `src/data.rs` (tensor I/O), `src/utils.rs` + `src/display.rs` (rendering),
> `src/error.rs` (errors).

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
   execution; orphaned *outputs* are auto-rewired by `finalize`.

## The pipeline

```rust
let mut graph = Topology::new(0, None);                 // 1. empty blueprint
graph.create_random_hidden_nodes(5);                    // 2. random hidden nodes
graph.refresh_labels();                                  // 3. port labels (rendering)
graph.finalize();                                       // 4. scaffold + wire + de-orphan
graph.validate().unwrap();                              // 5. refuse broken graphs
let model = Network::build(&graph, Device::CPU).unwrap(); // 6. compile to linears
let y = model.forward(&input).unwrap();                 // 7. run it
```

- **`finalize()`** does everything in one call: `ensure_scaffold()` (prepend
  an `Input` node, guarantee exactly one `Output` node — random graphs only
  create hidden nodes), random 1:1 wiring from earlier output ports to later
  input ports, then `rewire_orphaned_outputs()` (rewire unused outputs into
  compatible later nodes). Deterministic per `options.seed`.
- **`orphan_counts()`** → `(orphaned_inputs, orphaned_outputs)` for diagnostics.
- **`merge_multi_outputs()`** merges > 1 `Output` nodes into one stacked
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
random topologies, score with a [`Fitness`](src/fitness.rs), track the best,
checkpoint, log compactly. Data is a **path to tensors**
(`data::save_dataset` format), normalized to f32 on load.

**Scoring — 4+4 built-ins + custom.** [`FitnessKind`](src/fitness.rs) has four
continuous kinds (`Mse`/`Mae`/`Rmse`/`R2`, targets `[n, 1]`) and four
categorical kinds (`Accuracy`/`CrossEntropy`/`F1`/`Precision`, targets
**one-hot** `[n, C]`). Every scorer declares its [`Direction`]
(Minimize/Maximize) — the engine compares candidates with it and logs the
raw user-space score with a `↓`/`↑` arrow. Custom routes are **minimal**:
the scorer only sees `(prediction, target)` — the engine runs the forward
pass and feeds the batches, so users just write the metric. `Fitness::custom`
(closure, defaults Minimize), `Fitness::custom_directed` /
`Fitness::scorer_directed` for maximizers, or a named `FitnessScorer`
(`Send + Sync`, so the population can be scored in parallel).

**Options chain, one source of truth.** `EngineOptions` embeds the full
`TopologyOptions` **template** and a `network: NetworkOptions` link, so the
whole engine → topology → network chain is one struct and `engine.json`
spells it out (`topology_options` + `options.network`; `Device` and `DType`
serialize as strings — neither implements serde). Configure it flat instead
of nested literals: `EngineOptions::builder().set_pop_size(..).set_hidden_dim_pool(8..=16)..build()`
routes each knob into the right layer (`build_engine(data, fitness)` also
starts the run). `NetworkOptions` holds the **execution** knobs — device,
`dtype` (Float32 default), and `seed` (weight init) — while architecture
values live blueprint-side.

**Deterministic weight init — the run reproduces end to end.** The engine
bakes the base seed into **both** layers: `topology.seed` (blueprint) and
`network.seed` (init). Every build derives its weights in Rust from the
individual's seed via a local RNG (replicating flodl's
`kaiming_uniform`/`uniform_bias` distributions), so **same options ⇒ the
exact same built model** — and since each build uses its own RNG, the
determinism holds under rayon parallel evaluation. `NetworkOptions.seed =
None` falls back to flodl's internal RNG (fresh weights per build, the
pre-seeding behavior).

**GP over both layers.** Beyond the topology template (port ranges, node
counts), `EngineOptions` carries the **GP pools** — `hidden_dim_pool:
RangeInclusive<usize>`, `combine_op_pool: Vec<CombineOp>`, `activation_pool:
Vec<Activation>` — sampled **per individual** at population creation from
the derived seed chain: each individual gets its own hidden dim, combine op,
and per-node activations (`Topology::randomize_activations`, hidden nodes
only). Same seed ⇒ same population, dims included — so the search space
covers network values, not just graph structure.

**Seeding.** `seed: Option<u64>`: `None` → a random base seed is derived per
run and recorded as `run_seed` in `engine.json` (fresh exploration every
launch, still re-launchable); `Some` → fully deterministic. Each individual
derives its own seed via a splitmix chain (`derive_seed(run_seed, i)`), so
one seed seeds child seeds — the whole population reproduces from
`run_seed` alone.

**Evaluation budget + parallelism.** `num_batches` random batches of
`batch_size` rows per `num_epochs`, averaged; `num_batches = 0` = the whole
dataset once, **chunked into `batch_size` slices** (memory-bounded, default
128). Batches are sampled once per generation (seeded `run_seed +
generation`) and reused for every individual, so scores stay comparable and
runs reproduce. `num_threads` (default 3, `0` = auto) drives **rayon parallel
evaluation** — one task per individual (batches travel as `Tensor` pairs;
flodl `Variable` is Rc-based and gets wrapped fresh per task). Per run,
`results/<ts>/` holds:

- `improvements/` — a **pair** per best-improvement: the blueprint recipe
  (`0000_gen00_fitness….json`) **plus the materialized-net facts**
  (`….net.json` — dims, wiring stats, real param counts via `Network::to_json`)
- `log.txt` — one compact line per generation
- `engine.json` — `Engine::to_json()`: the resolved `run_seed`, options
  (incl. the shared `topology_options` template) + data path + best fitness +
  best topology + `best_net_facts` — the single file to replicate the
  experiment

`Network::to_json()` is the **nutrition label of the built module**: name,
input/hidden dims, output node, combine op, node/wire counts, real parameter
tensors/elements, per-node dims, degrees, depths, orphan counts, kind and
activation histograms. It shares the *same* derived-diagnostics functions as
the blueprint ([`Topology::orphan_counts`], `degrees`, `depths`, …), so both
sides always agree. Weights stay out — a rebuilt `Network` has the same
architecture, fresh weights (by design).

A run starts fresh via `Engine::new(options, path, fitness)`; every artifact
it needs is captured in the run folder (`engine.json` carries the full option
chain + `run_seed`), so a future folder-based resume can rebuild everything
from there — no separate seeding entry point (removed). `run()` has a
stop-criteria TODO beyond max generations. Genetics live in `src/genetics.rs`:
**selection (elitism + tournament) is implemented and tested** but not yet
wired into the loop (pending design review); `crossover()`/`mutate()` remain
documented no-op stubs until the operators land.
**Two views, one graph:** `Display` for a `Topology` is the Manhattan wire
*diagram* (nodes + wires + orphan markers); `Display` for a `Network` is the
compact *derived* table — per node, the layer dims (`in → out`), activation,
and incoming-input ops (per input port: its sources, or `*` when orphaned).
Both render from the same blueprint data via `src/utils.rs`.

## Derived diagnostics 📊 — data now captured on the blueprint

All derived from nodes + connections (no build needed):

- **`Topology::node_dims()`** → `Vec<(in_dim, out_dim)>` — the same derivation
  `Network::build` uses (shared wiring table).
- **`Topology::orphan_ports(node)`** → `(orphaned input indices, orphaned
  output indices)` per node (raw; `orphan_counts()` stays the aggregate).
- **`Topology::degrees()`** → `Vec<(in_degree, out_degree)>` per node (per wire).
- **`Topology::depths()`** → topological levels: longest path from an `Input`
  node (0 for inputs; orphaned ports read `net_input`, so they add no depth).
  Powers depth-biased NAS and level-based rendering.
- **`Topology::param_estimate()`** → total learnable elements (input_proj +
  every node's `in·out + out`), matching a built network exactly (proptest).
- **`Topology::kind_counts()`** / **`Topology::activation_counts()`** →
  histograms by kind and activation.

**Still listed, not captured:** `on_critical_path` per node (depth + reverse
reachability — left for mutation targeting later) and level-grouped id maps
(trivially derived from `depths()`).

**Why they matter:** fan-in/out and depth power NAS heuristics (prefer
deeper/wider), `param_estimate` lets the engine rank candidates by cost
without building, and per-port orphan flags make debug views
self-explanatory.

## Examples 🧪

- **`cargo run`** — the minimal pipeline (random → wire → validate → build → forward).
- **`cargo run --example topology_guide`** — the blueprint side, topology-only:
  hand-built/random graphs, scaffolding, wiring introspection, per-node
  insights, JSON round-trip → `saved/`. One compile→forward handoff, then
  execution is left to `network_guide`.
- **`cargo run --example node_guide`** — the node type: constructors, builders,
  all 8 activations applied, `NodeKind` roles, NAS knobs, JSON.
- **`cargo run --example network_guide`** — execution: compile, forward,
  parameters (tensor vs element counts), derived dims, runtime wiring,
  combine ops, rebuild from JSON, devices.
- **`cargo run --example fitness_guide`** — scoring: the 4+4 built-in family
  (regression + one-hot classification), `Direction` (↓/↑), custom closure +
  `custom_directed`, a named `FitnessScorer`, and the synthetic dataset
  path contract.
- **`cargo run --example engine_guide`** — the engine walkthrough: population
  seeding, the run loop + checkpoints, fitness routes, seeding from saved,
  the run folder, and end-to-end replication of the best.
- **`cargo run --example mnist_data`** — download MNIST once, save as tensor
  datasets (reuses already-downloaded assets; `--force` to redo).
