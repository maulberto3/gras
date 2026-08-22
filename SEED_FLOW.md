# Seed Flow Through a Gras Experiment

## The chain in one picture

```
User seed (42 or random)
    │
    ▼
run_seed (42)
    │
    ├──▶ topology_options.seed = 42     (template)
    │
    ├──▶ network.seed = 42             (template, usize)
    │
    └──▶ per individual i:
         │
         ind_seed = run_seed + i × 0x9E37...   (golden ratio multiply)
         │
         ├──▶ Topology::new(seed=ind_seed)
         │       └──▶ self.rng = Rng::with_seed(ind_seed)
         │               ├── create_random_hidden_nodes  (port counts)
         │               ├── finalize → shuffle + wire   (connections)
         │               └── rewire_orphaned_outputs     (de-orphaning)
         │
         ├──▶ per-node GP draws (from the same rng):
         │       ├── hidden_dim    (from hidden_dim_pool)
         │       ├── activation    (from activation_pool)
         │       ├── combine_op    (from combine_op_pool)
         │       └── standardize   (from standardize_op_pool)
         │
         └──▶ Network::build(seed=ind_seed)
                 └──▶ rng = Rng::with_seed(ind_seed)
                         ├── orphan_projections  (Linear weights)
                         ├── node layers         (Linear weights)
                         └── port_projections    (Linear weights)

    Per-generation seeds (independent chain):
         │
         ├──▶ batch sampling:  run_seed + generation
         ├──▶ selection:       run_seed + generation + 0xCAFE
         └──▶ mutation:        run_seed + generation + 0xBEEF
```

## Step by step

1. **User sets seed** — `set_seed(Some(42))` or `None` (random per run).
2. **Engine resolves `run_seed`** — user's seed, or entropy-based if `None`.
3. **Bake into two layers** — `topology_options.seed = run_seed`, `network.seed = run_seed`. Both recorded in `engine.json`.
4. **Per-individual derivation** — `ind_seed = run_seed + i × golden_ratio`. Each individual gets a unique seed via `derive_seed()`.
5. **Topology creation** — `Topology::new(seed=ind_seed)` creates one `Rng` from that seed. This RNG drives: hidden node creation (port counts), finalize shuffling, wiring, and de-orphaning — all in sequence.
6. **Per-node GP draws** — from the same individual RNG, the engine draws `hidden_dim`, `activation`, `combine_op`, and `standardize` for each hidden node. Input/output nodes keep defaults.
7. **Batch sampling** — `run_seed + generation` seeds which batches each generation sees. All individuals in a generation see the same batches.
8. **Selection** — `run_seed + generation + 0xCAFE` seeds the tournament RNG. Deterministic per generation.
9. **Mutation** — `run_seed + generation + 0xBEEF` seeds the activation-swap RNG. Deterministic per generation.
10. **Weight init** — `Network::build(seed=ind_seed)` creates a fresh `Rng` from the same seed that drove the topology. Same seed → same weights.
11. **Training** — gradient descent modifies weights. No seed involved; the initial weights came from step 10.

## Key invariant

**One seed → one topology → one set of weights → one score.** Reproducible end to end. Changing the seed changes everything downstream deterministically.

## Seed offsets summary

| Purpose | Formula | Constant |
|---|---|---|
| Individual topology + weights | `run_seed + i × golden` | `0x9E37_79B9_7F4A_7C15` |
| Batch sampling | `run_seed + generation` | — |
| Selection (tournament) | `run_seed + generation + 0xCAFE` | `0xCAFE` |
| Mutation (activation swap) | `run_seed + generation + 0xBEEF` | `0xBEEF` |

The per-generation seeds are independent of each other (different offsets), so batch sampling, selection, and mutation never interfere.
