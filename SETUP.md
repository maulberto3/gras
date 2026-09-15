# gras — setup

```
your Rust code → flodl crate → libtorch (cpu | cu128) → CUDA toolkit → NVIDIA driver
                                                          (CUDA only)     (CUDA only)
```

## 1. Fresh machine

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# then reopen the terminal so `cargo` is on PATH

# Build tools + CUDA 12.8 toolkit + NCCL + /usr/local/cuda symlink
# (~3GB — CUDA only; skip on a CPU-only machine)
sudo bash setup_ubuntu.sh

# fdl CLI
cargo install flodl-cli

# libtorch → shared global store. Run from $HOME, NOT from this project.
cd ~
fdl libtorch download --cuda 12.8 --path ~/.flodl/libtorch    # ~2GB, CUDA
fdl libtorch download --cpu       --path ~/.flodl/libtorch    # ~200MB, CPU
fdl diagnose
```

Already have libtorch in `~/.flodl/`? Skip this section — copy `env_setup.sh`
into the new project and go to §2.

## 2. Every shell

```bash
source env_setup.sh          # CPU
source env_setup.sh cuda     # CUDA
```

Sets `LIBTORCH_PATH`, `CUDA_HOME`, `LD_LIBRARY_PATH`, `LIBRARY_PATH`,
`CARGO_TARGET_DIR`, `FLODL_VARIANT`, `GRAS_FEATURES`.

## 3. Build / test / run

`gras` is a library + 8 examples — **there is no binary target**, so plain
`cargo run` fails. `$GRAS_FEATURES` is `--features cuda` on CUDA, empty on CPU.

```bash
cargo build $GRAS_FEATURES --all-targets
cargo test  $GRAS_FEATURES
cargo run   $GRAS_FEATURES --release --example mnist -- --steps 100
cargo run   $GRAS_FEATURES --example train_by_hash -- [RUN_DIR] [NET_HASH]
cargo run   $GRAS_FEATURES --bench flamegraph -- --steps 300
cargo bench $GRAS_FEATURES --bench stream
```

Verify CUDA actually runs:

```bash
source env_setup.sh cuda && cargo run $GRAS_FEATURES --release -F cuda --example mnist -- --steps 1
grep '"device"' results/*/engine.json     # expect "cuda:0"
```

Clean one target without touching the other: `rm -rf target_cpu/` or `target_cuda/`.

## 4. Profiling (flamegraph)

| Piece | Use |
|---|---|
| `benches/stream.rs` — stream vs train/eval numbers, no profiler needed | `cargo bench --bench stream`, or `make benc` — **start here** |
| `benches/flamegraph.rs` — the profiling workload | default = bare group step; `--evolve` adds catch-up replay |
| `[profile.profiling]` / `[profile.bench]` — debug symbols | already in `Cargo.toml` |
| `make prof` | the capture command below |

The full flow, in order:

```bash
# 1. capture tools. WSL2: `uname -r` is a Microsoft kernel with NO Ubuntu package,
#    so do not install linux-tools-$(uname -r) — it fails with "Unable to locate
#    package linux-tools-<ver>-microsoft-standard-WSL2". The generic tools work.
sudo apt install -y linux-tools-common linux-tools-generic
cargo install flamegraph

# 2. WSL2 only: /usr/bin/perf is a wrapper that refuses the Microsoft kernel
#    version. Point `perf` at the generic binary the package actually installed.
sudo ln -sf "$(ls -d /usr/lib/linux-tools/*/perf | tail -1)" /usr/local/bin/perf
perf --version

# 3. allow unprivileged sampling (use 0, not 1)
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

# 4. sanity-check that sampling works BEFORE trusting any graph:
perf stat -e cpu-clock true     # software event — works without a PMU
perf stat -e cycles true        # hardware event — WSL2 usually has no PMU
# If BOTH fail, this machine cannot sample → skip to the fallback below.

# 5. capture → results/prof/flamegraph.svg
source env_setup.sh
make prof
cargo flamegraph --profile profiling -o results/prof/flamegraph.svg \
  --bench flamegraph -- --steps 300 --evolve
```

Without `-o`, `cargo flamegraph` writes `flamegraph.svg` (and the intermediate
`perf.data`) into the current directory. Both are gitignored at the repo root.

`--evolve` adds checkpoint catch-up replay to the profile; omit it to profile
the bare group step.

**No-PMU fallback** (numbers instead of a graph — works everywhere, incl. WSL2):
```bash
cargo bench --bench stream     # or: make benc
```

Check sampling works before trusting a graph:

```bash
perf stat -e cpu-clock true     # software event — works without a PMU
perf stat -e cycles true        # hardware event — WSL2 usually has no PMU
```

If both fail, WSL2 cannot sample; profile on native Linux, or get the numbers
without a profiler:

```bash
cargo bench --bench stream     # or: make benc
```

### Memory profiling (dhat — Rust heap)

CPU profiles answer "where does time go"; the memory bench answers "where do
bytes go" — churn (total allocated), peak live, and per-call-site attribution.
No system install: it's a dev-dependency, same flow as the benches.

```bash
source env_setup.sh
cargo bench --bench memgraph -- --steps 300            # headline numbers on stdout
cargo bench --bench memgraph -- --steps 300 --evolve   # + evolution churn
cargo bench --bench memgraph -- --steps 300            # writes dhat-heap.json
```

Then load `dhat-heap.json` into the interactive tree:
<https://nnethercote.github.io/dh_view/dh_view.html> — same flamegraph idea,
but allocation bytes instead of samples.

**Scope:** dhat sees the **Rust heap only**. libtorch's C++ tensor arena and
CUDA memory are invisible; for those use `heaptrack` (intercepts malloc at the
system level, works out of the box on Linux):
```bash
sudo apt install -y heaptrack heaptrack-gui
heaptrack cargo bench --bench flamegraph -- --steps 300
heaptrack_gui heaptrack.*.gz
```

**Read the numbers:** high churn + low peak/churn % = allocation churn (the
autograd graph rebuild case — flodl arena work); churn ≈ peak = live tensors
dominate, nothing to tune.

## 5. Where things live

| Component | Location | Installed by |
|---|---|---|
| libtorch `.so` + headers | `~/.flodl/libtorch/precompiled/{cpu,cu128}/` | `fdl libtorch download --path ~/.flodl/libtorch` |
| CUDA toolkit headers + libs | `/usr/local/cuda-12.8/` (+ `/usr/local/cuda` symlink) | `sudo bash setup_ubuntu.sh` |
| NCCL headers | `/usr/include/nccl.h` | `sudo bash setup_ubuntu.sh` |
| NVIDIA driver | kernel module (WSL: host driver) | pre-installed |
| build output | `target_cpu/` or `target_cuda/` | `cargo` via `CARGO_TARGET_DIR` |

## Notes

- Run `fdl` from `~`, not from a project, or pass `--path ~/.flodl/libtorch` — inside a project it targets a project-local `libtorch/`.
- Ubuntu 26.04: CUDA 12.8 ships only in NVIDIA's **24.04** repo; `setup_ubuntu.sh` adds that repo for you.
- `env_setup.sh` exports into the whole shell session. Use a fresh shell per flow, or unset `CARGO_TARGET_DIR LIBTORCH_PATH LD_LIBRARY_PATH LIBRARY_PATH GRAS_FEATURES`.
- A `libtorch/` dir inside the project is a stray duplicate — delete it.
- No `fdl.yml` in this repo: the `env_setup.sh` + Makefile flow is the only one.
