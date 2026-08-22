# Dtype Flow Through a Gras Experiment

## The chain in one picture

```
User sets dtype (Float32 by default)
    │
    ▼
EngineOptions.network.dtype = Float32
    │
    ├──▶ data loading:
    │       load_dataset(path) → .to_dtype(dtype)
    │       inputs.bin and targets.bin are cast on load
    │
    ├──▶ network build:
    │       ├── orphan_projections: Linear(input_dim → in_dim, dtype)
    │       ├── node layers:        Linear(in_dim → out_dim, dtype)
    │       └── port_projections:   Linear(src_dim → in_dim, dtype)
    │
    ├──▶ fitness scoring:
    │       pred and target tensors are already dtype
    │       (no explicit cast — fitness operates on whatever it receives)
    │
    └──▶ engine.json:
            network.dtype → "Float32" or "Float64"
            (serialized via Display impl)
```

## Step by step

1. **User sets dtype** — `set_dtype(DType::Float32)` (the default) or `DType::Float64`.
2. **Engine resolves data** — on `Engine::new`, the dataset is loaded and cast to the engine's dtype via `data.to_dtype(options.network.dtype)`. Both `inputs` and `targets` tensors are converted.
3. **Network build** — every `Linear` layer is created with the engine's dtype. This includes orphan projections, node layers, and port projections. The dtype determines the weight tensor precision.
4. **Forward pass** — all tensor operations (matmul, activation, combine, standardize) run in the network's dtype. No implicit casts.
5. **Fitness scoring** — the fitness function receives predictions and targets in the network's dtype. Built-in scorers (MSE, CrossEntropy, etc.) work in whatever precision the tensors have. Custom closures follow the same contract.
6. **engine.json** — the dtype is recorded as a string (`"Float32"` or `"Float64"`) for reproducibility.

## Where dtype matters

| Component | How dtype flows |
|---|---|
| **Dataset** | Cast on load: `load_dataset(path).to_dtype(dtype)` |
| **Orphan projections** | `Linear(input_dim → in_dim, dtype)` — bridges raw input to node dims |
| **Node layers** | `Linear(in_dim → out_dim, dtype)` — the main compute layers |
| **Port projections** | `Linear(src_dim → in_dim, dtype)` — bridges mismatched source dims |
| **Activations** | Run on tensors in the network's dtype (no explicit dtype param) |
| **Standardize ops** | LayerNorm uses the tensor's dtype for mean/var computation |
| **Fitness** | Operates on whatever precision the tensors carry |
| **engine.json** | Serialized as `"Float32"` or `"Float64"` |

## Default behavior

- Default dtype: `DType::Float32`
- If the user doesn't call `set_dtype`, everything runs in Float32
- The dataset is always cast on load, so it doesn't matter what dtype the files were saved in

## Key invariant

**One dtype → one precision for the entire pipeline.** The engine sets it once, and every component (data, network, fitness) inherits it. There is no mixed-precision path — tensors are either all Float32 or all Float64.
