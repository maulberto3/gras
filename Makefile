################################################################################
# 📊 Development & Analysis
################################################################################
dev-size:
	clear && du ./target/debug/libharu_cmaes.rlib -h

prod-size:
	clear && du ./target/release/libharu_cmaes.rlib -h

tree:
	cargo tree

graph-dep:
	cargo depgraph --dedup-transitive-deps | dot -Tpng > dependencies_graph_of_current_cargo_toml.png

deps: tree graph-dep
	@echo "✓ Dependency analysis complete"

################################################################################
# 🧹 Quality & Testing (CI)
################################################################################
clean:
	cargo cache --autoclean && cargo clean

lint:
	cargo fmt && cargo clippy -- -D warnings

test:
	cargo test

cove:
	cargo tarpaulin --out Html

################################################################################
# 🔨 Build & Documentation
################################################################################
prep:
	cargo build --all-targets 
	# Compiles lib + bins + examples + benches + tests

docu:
	cargo doc

exam:
	cargo run --release --bin express_use

all: clean lint test prep exam docu

build:
	clear && make all

################################################################################
# ⚡ Performance & Profiling
################################################################################
benc:
	clear && cargo bench --bench mine

prof:
	clear && cargo run --release --example flamegraph

samp:
	# To grant temporary access before using samply
	# echo '1' | sudo tee /proc/sys/kernel/perf_event_paranoid
	clear && samply record cargo run --release --bin ask_tell

################################################################################
# 🚀 Release & Publishing
################################################################################
VERSION := $(shell awk -F ' = ' '/^version/ {gsub(/"/, "", $$2); print $$2}' Cargo.toml)
clif:
	# Generate the changelog and commit it in the same step
	git cliff -t v$(VERSION) -o CHANGELOG.md
	git add CHANGELOG.md
	git diff-index --quiet --cached HEAD || git commit -m "Update changelog for v$(VERSION)"

tag:
	@git diff-index --quiet HEAD || { echo "Error: Uncommitted changes! Commit before publishing."; exit 1; }
	@echo "✓ Working tree clean"
	@git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@echo "🏷️  Tagged as v$(VERSION)"
	@git push --tags
	@echo "📤 Pushed tags to remote"
	@git push origin dev
	@echo "📤 Pushed dev branch to remote"

publish:
	@cargo publish
	@echo "✅ Successfully published v$(VERSION) to crates.io"

publ: clif tag publish
	@clear
	@echo "✅ Release v$(VERSION) complete!"