SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c

# Every recipe that links or loads libtorch runs through env_setup.sh, which
# exports LIBTORCH_PATH and pins CARGO_TARGET_DIR=target_cpu.
ENV := source env_setup.sh &&

.DEFAULT_GOAL := help
.PHONY: help all prep docu clean lint test cove ci benc prof mem samp dry_publ publ

################################################################################
# 🔨 Build, docs & tests
################################################################################
clean: ## prune the cargo registry cache and remove all target dirs
	cargo cache --autoclean
	rm -rf target 
# 	target_cpu target_cuda

prep: ## compile lib + bins + examples + benches + tests
	$(ENV) cargo build

docu: ## build the Rust docs
	$(ENV) cargo doc


test: ## run the Rust test suite
	$(ENV) cargo test

cove: ## HTML coverage report (cargo tarpaulin)
	$(ENV) cargo tarpaulin --out Html

lint: clean prep docu ## rustfmt + clippy (warnings reported, not fatal)
	$(ENV) cargo fmt && cargo clippy --all-targets

# Local mirror of .github/workflows/ci.yml. `lint` + `test` are prerequisites,
# so this one command covers the workflow's quality gate and packaging.
# CI checks out clean; a local tree usually is not, hence --allow-dirty.
ci: lint test cove ## CI rehearsal: lint+test, packaging dry run, wheel/sdist, smoke test
	$(ENV) cargo publish -p gras --dry-run --locked --allow-dirty
	$(ENV) rm -rf dist && uvx maturin@1.15.0 build --release --out dist && \
		uvx maturin@1.15.0 sdist --out dist
	uvx twine check dist/*
	rm -rf /tmp/gras-ci-smoke && uv venv /tmp/gras-ci-smoke --python 3.12 && \
		uv pip install --python /tmp/gras-ci-smoke/bin/python dist/*.whl && \
		/tmp/gras-ci-smoke/bin/python -c "import gras; print('gras', gras.__version__)"

################################################################################
# ⚡ Performance & profiling
################################################################################
benc: ## stream benchmark
	$(ENV) cargo bench --bench stream

prof: ## flamegraph of the profiling workload
	$(ENV) mkdir -p results/prof && \
	cargo flamegraph --profile profiling -o results/prof/flamegraph.svg --bench flamegraph -- --steps 300

mem: ## heap profile (dhat) of the benchmark workload
	$(ENV) cargo bench --bench memgraph -- --steps 300 --evolve

# Needs perf access: echo '1' | sudo tee /proc/sys/kernel/perf_event_paranoid
samp: ## record a sampled profile of the ask_tell binary
	$(ENV) samply record cargo run --release --bin ask_tell

################################################################################
# 🚀 Release (manual — CI only dry-runs packaging)
#
# LEVEL is required: cargo-release's default level is `release`, which only
# strips a pre-release suffix and keeps the current version — it would try to
# publish 2.1.1 again instead of bumping. Pass patch | minor | major |
# alpha/rc/beta, or an explicit version, e.g. `make dry_publ LEVEL=minor`.
################################################################################
dry_publ: ## plan a release (LEVEL=patch|minor|major|X.Y.Z)
	@test -n "$(LEVEL)" || { echo "usage: make $@ LEVEL=patch|minor|major|X.Y.Z"; exit 1; }
	$(ENV) cargo release $(LEVEL)

publ: ## release for real: bump, commit, tag, push, publish (LEVEL=...)
	@test -n "$(LEVEL)" || { echo "usage: make $@ LEVEL=patch|minor|major|X.Y.Z"; exit 1; }
	$(ENV) cargo release $(LEVEL) --execute

################################################################################
# 📖 Help
################################################################################
help: ## list targets
	@echo "Targets:"
	@grep -hE '^[a-z][a-z0-9_-]*:.*## ' $(MAKEFILE_LIST) | sort \
		| awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'
