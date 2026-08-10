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

## 1. Install fdl CLI

```bash
cargo install flodl-cli
```

## 2. Download libtorch (auto-detects your GPU)

```bash
fdl setup
fdl diagnose   # verify GPU + libtorch compatibility
```

## 3. Install CUDA toolkit headers (system level)

```bash
# Add NVIDIA apt repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install headers + compiler (not the full 3GB toolkit)
sudo apt install -y cuda-nvcc-12-8 libnccl-dev
```

Verify:
```bash
ls /usr/local/cuda/include/cuda_runtime.h   # must exist
ls /usr/local/cuda/include/crt/host_config.h # must exist
```

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
```

---

## What lives where

| Component | Location | Installed by |
|---|---|---|
| libtorch `.so` files | `~/.flodl/libtorch/precompiled/cu128/lib/` | `fdl setup` |
| Torch C++ headers | `~/.flodl/libtorch/precompiled/cu128/include/` | `fdl setup` |
| CUDA toolkit headers | `/usr/local/cuda-12.8/include/` | `apt install cuda-nvcc-12-8` |
| NCCL headers | `/usr/include/nccl.h` | `apt install libnccl-dev` |
| NVIDIA driver | kernel module | pre-installed |