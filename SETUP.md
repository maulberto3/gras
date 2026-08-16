# flodl + CUDA Setup Reference

## What You Need (3 layers)

```
Your Rust code
    ↓
flodl crate  (cargo add flodl --features cuda)
    ↓
libtorch/cu128  (~/.flodl/libtorch/precompiled/cu128/)   ← fdl setup
    ↓
CUDA toolkit headers  (/usr/local/cuda-12.8/)             ← apt install
    ↓
NVIDIA driver  (nvidia-smi works = you're fine)
```

---

## Quickstart (ordered steps)

**Fresh machine (first time):**

1. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
   (then reopen the terminal so `cargo` is on PATH)
2. One-time system setup — **must come before `cargo install`** (it installs gcc,
   which Rust needs to compile, plus unzip, which fdl needs to extract libtorch):
   ```bash
   sudo bash setup_ubuntu.sh
   ```
3. Install fdl CLI: `cargo install flodl-cli`
4. Download libtorch into the **shared global store** — run from outside the
   project so fdl targets `~/.flodl/` (in-project it would create a local copy):
   ```bash
   cd ~
   fdl libtorch download --cuda 12.8   # ~2GB, GPU
   fdl libtorch download --cpu         # ~200MB, fallback
   ```
5. In the project, build:
   ```bash
   source env_setup.sh cuda && cargo build --features cuda
   ```
6. Verify the GPU works at runtime:
   ```bash
   source env_setup.sh cuda && cargo run --features cuda --example gpu_check
   # Expected: CUDA matmul OK — result device: CUDA(0), shape: [4, 4]
   ```

**New project on an already-set-up machine:** libtorch is already in
`~/.flodl/`, so **fdl is not needed again** — copy `env_setup.sh` and the
`Cargo.toml` block from §5 into the new project, then go straight to step 5-6.

---

## 0. Prerequisites (fresh WSL)

Two things SETUP.md originally assumed were already present:

- **Rust via rustup** — `~/.cargo/bin` must be on PATH. New shells pick it up from
  `~/.bashrc` automatically, but the *current* shell needs `source ~/.cargo/bin/env`
  (or just close and reopen the terminal).
- **C compiler / build tools** — without gcc, `cargo install` fails to compile almost
  any crate (build scripts for `libc`, `serde`, etc. need a C compiler):

```bash
sudo apt update && sudo apt install -y build-essential pkg-config cmake unzip
```

> `unzip` is required by `fdl setup` to extract libtorch archives.

---

## Fast path (recommended)

All system-level steps (build tools + NVIDIA repo + CUDA headers) are bundled into
one script — run it with sudo once and skip straight to section 1:

```bash
sudo bash setup_ubuntu.sh
```

The script targets Ubuntu 24.04/26.04 and verifies the headers at the end.

---

## 1. Install fdl CLI

```bash
cargo install flodl-cli
```

## 2. Download libtorch (shared global location)

libtorch lives in **fdl's global config dir — `~/.flodl/libtorch/`** — so every
project on this machine shares the same CPU/CUDA variants (no per-project copies).

```bash
# Download CUDA 12.8 variant into the shared location (~2GB)
fdl libtorch download --cuda 12.8 --path ~/.flodl/libtorch

# CPU variant too, for CPU-only builds (~200MB)
fdl libtorch download --cpu --path ~/.flodl/libtorch

fdl diagnose   # verify GPU + libtorch compatibility
```

> **Why `--path`:** in a project (a dir with `Cargo.toml`), fdl defaults to a
> *project-local* `libtorch/` and only sees that one. `--path ~/.flodl/libtorch`
> installs into the global location that `env_setup.sh` points at. Run downloads
> from outside any cargo project and the `--path` can be dropped (standalone mode
> uses `~/.flodl/` by default).

## 3. Install CUDA toolkit headers (system level)

```bash
# Add NVIDIA apt repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Full toolkit (meta) — bundles nvcc + cudart + cuBLAS + cuSPARSE + cuSOLVER +
# cuFFT + cuRAND + CUPTI + Nsight in one shot (~3GB installed).
# Important: libtorch's headers include cusparse.h/cublas.h etc., so the minimal
# cuda-nvcc-12-8 install alone is NOT enough and causes missing-header build
# errors — prefer the full toolkit to avoid install-whack-a-mole.
sudo apt install -y cuda-toolkit-12-8 libnccl-dev

# cuda-nvcc-12-8 does NOT create the /usr/local/cuda symlink — make it so CUDA_HOME works
sudo ln -s /usr/local/cuda-12.8 /usr/local/cuda
```

Verify:
```bash
ls /usr/local/cuda/include/cuda_runtime.h   # must exist
ls /usr/local/cuda/include/crt/host_config.h # must exist
```

### Ubuntu version gotchas

- **Ubuntu 26.04:** CUDA 12.8 packages are only published in NVIDIA's **24.04** repo
  (they aren't built for 26.04). The `cuda-keyring_1.1-1_all.deb` above installs
  exactly that repo, so the commands work unchanged on 26.04 — the packages are
  glibc-backward-compatible. `fdl diagnose` may still report the CUDA 12.8 toolkit
  correctly; the toolkit version doesn't have to match your driver.
- **Driver newer than toolkit is fine:** WSL NVIDIA drivers are backward compatible.
  E.g. a driver reporting CUDA UMD 13.3 works fine with a 12.8 toolkit.

### WSL2: fdl can't auto-detect your GPU

`fdl setup` / `fdl diagnose` report **No GPU devices available** on WSL2 even though
`nvidia-smi` works. This is cosmetic — detection relies on `/dev/nvidia*` nodes that
WSL2 doesn't expose. The GPU still works at runtime via `libcuda.so`. Workaround:
force the variant explicitly instead of relying on auto-detect:

```bash
fdl libtorch download --cuda 12.8 --path ~/.flodl/libtorch   # ~2GB, bypasses auto-detect
```

### libtorch location (fdl 0.8+)

libtorch variants live in **fdl's global config dir — `~/.flodl/libtorch/`**
(precompiled variants under `~/.flodl/libtorch/precompiled/{cpu,cu128}`). All
projects on the machine share these — that's what `env_setup.sh` points at.

Caveat: inside a **flodl project** (detected by walking up from cwd — a
`Cargo.toml` mentioning `flodl`, or a `libtorch/.active` present), fdl's
`setup` / `libtorch download` / `diagnose` / `list` commands target a
**project-local** `libtorch/` instead of the global one. Non-flodl Rust
projects fall through to the global location. So:

- run fdl commands from **outside the project** (e.g. `cd ~ && fdl libtorch
  list`) — standalone mode uses `~/.flodl/` and sees your shared variants, or
- pass `--path ~/.flodl/libtorch` to `fdl libtorch download`, or
- if a stray project-local `libtorch/` ever appears, just delete it — it's a
  duplicate and gitignored anyway.

> `fdl setup` never asks where to install libtorch — it auto-detects. The
> "Install fdl globally?" prompt at the end of the wizard installs the **fdl
> binary** to `~/.local/bin/fdl`, which is unrelated to libtorch.

---

## 4. env_setup.sh (one per project)

```bash
#!/bin/bash
# source env_setup.sh       → CPU
# source env_setup.sh cuda  → CUDA

FLODL_BASE=/home/$USER/.flodl/libtorch/precompiled

if [ "${1}" = "cuda" ]; then
    export LIBTORCH_PATH="${FLODL_BASE}/cu128"
    export CUDA_HOME="/usr/local/cuda"
    export LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${LIBTORCH_PATH}/lib:${CUDA_HOME}/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CARGO_TARGET_DIR="target_cuda"
    export FLODL_VARIANT="cuda"
    echo "Environment set: CUDA — libtorch at ${LIBTORCH_PATH}"
else
    export LIBTORCH_PATH="${FLODL_BASE}/cpu"
    export LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${LIBTORCH_PATH}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CARGO_TARGET_DIR="target_cpu"
    export FLODL_VARIANT="cpu"
    echo "Environment set: CPU — libtorch at ${LIBTORCH_PATH}"
fi
```

> The repo's `env_setup.sh` uses `FLODL_BASE="$HOME/.flodl/libtorch/precompiled"`
> — the shared global location — so it works for any user and any project.

### env_setup.sh vs fdl — do they collide?

Short answer: **no, they're complementary** — `env_setup.sh` is a thin, read-only
layer over fdl's managed layout. One foot-gun (env leakage) to know about:

| Aspect | `env_setup.sh` (custom) | fdl 0.8 (tool) |
|---|---|---|
| libtorch layout | reads `~/.flodl/libtorch/precompiled/{cpu,cu128}` (fdl's global dir, read-only) | owns/writes `libtorch/` + `libtorch/.active` (project-local, or `$FLODL_HOME` standalone) |
| Env vars for cargo | `LIBTORCH_PATH`, `CUDA_HOME`, `CARGO_TARGET_DIR` (what flodl-sys's build.rs reads) | `LIBTORCH_HOST_PATH`, `CUDA_VERSION`, `CUDA_TAG` (only via `./fdl` + `fdl.yml`) |
| "Which variant" | chosen by argument (`cuda` / none) | chosen by `libtorch/.active` |

- **Files:** `env_setup.sh` never writes to `libtorch/`, so there's no file-level
  conflict. Variants in `~/.flodl/libtorch/precompiled/` (downloaded with
  `--path ~/.flodl/libtorch`) are exactly what `env_setup.sh` points at.
- **Two "active variant" selectors:** `libtorch/.active` (fdl) and
  `env_setup.sh`'s argument are independent — e.g. `fdl libtorch activate
  precompiled/cpu` changes fdl's pointer, but `source env_setup.sh cuda` still
  builds CUDA. For direct `cargo` builds, `env_setup.sh`'s argument is the source
  of truth; `.active` only matters if you adopt `fdl.yml` / `./fdl`.
- **Foot-gun — env leakage:** `source env_setup.sh` sets persistent vars for the
  whole shell session (`CARGO_TARGET_DIR=target_cuda`, `LD_LIBRARY_PATH`,
  `LIBRARY_PATH`, `LIBTORCH_PATH`). They leak into *any* `cargo`/`fdl` command
  in that shell — including other projects. Fix: use a fresh shell per flow, or
  `unset CARGO_TARGET_DIR LIBTORCH_PATH LD_LIBRARY_PATH LIBRARY_PATH` before
  switching.
- **This repo has no `fdl.yml`** — the custom flow (`env_setup.sh` + Makefile) is
  the only one wired up. If you ever add `fdl.yml`, fdl's native `./fdl build`
  derives the same env from `.active`, making `env_setup.sh`/Makefile redundant —
  pick one flow per shell, don't mix them.

---

## 5. Cargo.toml

```toml
[dependencies]
flodl = "0.7.0"

[features]
cuda = ["flodl/cuda"]
```

---

## 6. Device-aware code pattern

```rust
fn main_device() -> Device {
    #[cfg(feature = "cuda")]
    { Device::CUDA(0) }
    #[cfg(not(feature = "cuda"))]
    { Device::CPU }
}

// Always use on_device — never new() for CUDA projects
let model = FlowBuilder::from(Linear::on_device(in_dim, 64, device)?)
    .through(SELU::new())
    .through(Linear::on_device(64, out_dim, device)?)
    .build()?;
```

---

## 7. Build & run commands

```bash
# CPU
source env_setup.sh && cargo build
source env_setup.sh && cargo test my_test
source env_setup.sh && cargo run

# CUDA
source env_setup.sh cuda && cargo build  --features cuda
source env_setup.sh cuda && cargo test   --features cuda my_test
source env_setup.sh cuda && cargo run    --features cuda

# Clean one target without touching the other
rm -rf target_cpu/    # or: source env_setup.sh && cargo clean
rm -rf target_cuda/   # or: source env_setup.sh cuda && cargo clean

# Verify the GPU actually works at runtime (prints the result tensor's device)
source env_setup.sh cuda && cargo run --features cuda --example gpu_check
# Expected: CUDA matmul OK — result device: CUDA(0), shape: [4, 4]
```

> `libtorch/` (a project-local variant if one ever appears) and `target_cpu/` /
> `target_cuda/` are gitignored — the real ~7GB lives in `~/.flodl/libtorch/`.
> If a stray project-local `libtorch/` appears, delete it (it's a duplicate).

---

## What lives where

| Component | Location | Installed by |
|---|---|---|
| libtorch `.so` files | `~/.flodl/libtorch/precompiled/cu128/lib/` | `fdl libtorch download --cuda 12.8 --path ~/.flodl/libtorch` |
| Torch C++ headers | `~/.flodl/libtorch/precompiled/cu128/include/` | `fdl libtorch download --cuda 12.8 --path ~/.flodl/libtorch` |
| CUDA toolkit headers + libs | `/usr/local/cuda-12.8/` | `apt install cuda-toolkit-12-8` |
| NCCL headers | `/usr/include/nccl.h` | `apt install libnccl-dev` |
| NVIDIA driver | kernel module (WSL: host driver) | pre-installed |
