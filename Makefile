SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c

################################################################################
# 📊 Development & Analysis
################################################################################
dev-size:
	clear && du ./target_cpu/debug/libgras.rlib -h 2>/dev/null || du ./target/debug/libgras.rlib -h

prod-size:
	clear && du ./target_cpu/release/libgras.rlib -h 2>/dev/null || du ./target/release/libgras.rlib -h

################################################################################
# 🧹 Quality & Testing (CI)
################################################################################
clean:
	cargo cache --autoclean && cargo clean

lint:
	source env_setup.sh && cargo fmt && cargo clippy --all-targets -- -D warnings

test:
	source env_setup.sh && cargo test

cove:
	source env_setup.sh && cargo tarpaulin --out Html

################################################################################
# 🔨 Build & Documentation
################################################################################
prep:
	source env_setup.sh && cargo build
	# Compiles lib + bins + examples + benches + tests

docu:
	source env_setup.sh && cargo doc

all: clean lint test cove prep docu

build:
	clear && make all

################################################################################
# ⚡ Performance & Profiling
################################################################################
benc:
	clear && source env_setup.sh && cargo bench --bench stream

prof:
	clear && source env_setup.sh && mkdir -p results/prof && \
	cargo flamegraph --profile profiling -o results/prof/flamegraph.svg --example flamegraph -- --steps 300

samp:
	# To grant temporary access before using samply
	# echo '1' | sudo tee /proc/sys/kernel/perf_event_paranoid
	clear && source env_setup.sh && samply record cargo run --release --bin ask_tell

dry_publ:
	source env_setup.sh && cargo release
	
publ:
	source env_setup.sh && cargo release --execute