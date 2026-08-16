#!/bin/bash
# One-time system setup for the GRAS project (tested on Ubuntu 26.04 WSL).
# Run with:  sudo bash setup_ubuntu.sh
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: run with sudo:  sudo bash setup_ubuntu.sh" >&2
    exit 1
fi

echo "==> [1/4] Updating apt"
apt-get update

echo "==> [2/4] Installing build essentials (needed to compile Rust crates)"
apt-get install -y build-essential pkg-config cmake unzip

echo "==> [3/4] Adding NVIDIA CUDA apt repo (CUDA 12.8 lives in the 24.04 repo)"
KEYRING_DEB=$(mktemp /tmp/cuda-keyring.XXXXXX.deb)
wget -q -O "$KEYRING_DEB" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i "$KEYRING_DEB"
rm -f "$KEYRING_DEB"
apt-get update

echo "==> [4/4] Installing full CUDA 12.8 toolkit + NCCL"
# Full toolkit (meta): nvcc + cudart + cuBLAS + cuSPARSE + cuSOLVER + cuFFT +
# cuRAND + CUPTI + Nsight, all in one shot (~3GB installed). This avoids the
# missing-header whack-a-mole (cusparse.h, cublas.h, ...) that libtorch's
# headers trigger when only cuda-nvcc-12-8 is installed.
apt-get install -y cuda-toolkit-12-8 libnccl-dev

echo "==> Ensuring /usr/local/cuda symlink exists"
if [ ! -e /usr/local/cuda ]; then
    ln -s /usr/local/cuda-12.8 /usr/local/cuda
    echo "Created /usr/local/cuda -> /usr/local/cuda-12.8"
fi

echo "==> Verifying"
for f in /usr/local/cuda/include/cuda_runtime.h \
         /usr/local/cuda/include/crt/host_config.h \
         /usr/include/nccl.h; do
    if [ -f "$f" ]; then
        echo "OK   $f"
    else
        echo "FAIL $f is missing" >&2
        exit 1
    fi
done

echo ""
echo "All checks passed — build tools + CUDA 12.8 headers ready."
