SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c

# Every recipe that links or loads libtorch runs through env_setup.sh, which
# exports LIBTORCH_PATH and pins CARGO_TARGET_DIR=target_cpu.
ENV := source env_setup.sh &&

.DEFAULT_GOAL := help
.PHONY: help all prep docu clean fmt fmt_check lint test cove ci benc prof mem samp dry_publ publ

################################################################################
# 🔨 Build, docs & tests
################################################################################
# Only the stray `target/` is removed: it appears when a build runs without
# env_setup.sh (which pins CARGO_TARGET_DIR=target_cpu). target_cpu/ and
# target_cuda/ are deliberately kept so builds stay incremental.
clean: ## prune the cargo cache and remove the stray target/ dir
	cargo cache --autoclean
	rm -rf target

prep: ## compile lib + bins + examples + benches + tests
	$(ENV) cargo build

docu: ## build the Rust docs
	$(ENV) cargo doc

fmt: ## rewrite formatting with rustfmt (the fixer)
	$(ENV) cargo fmt --all

# What CI enforces: fails instead of rewriting, so it can never "pass" by
# leaving the tree dirty and then be undone by the next checkout.
fmt_check: ## verify formatting the way CI does (no rewrite)
	$(ENV) cargo fmt --all -- --check


test: ## run the Rust test suite
	$(ENV) cargo test

cove: ## HTML coverage report (cargo tarpaulin)
	$(ENV) cargo tarpaulin --out Html

lint: clean prep docu fmt ## rustfmt + clippy (warnings reported, not fatal)
	$(ENV) cargo clippy --all-targets

# Local mirror of .github/workflows/ci.yml. `lint` + `test` are prerequisites,
# so this one command covers the workflow's quality gate and packaging. `lint`
# runs the rustfmt FIXER (`make fmt`) and `fmt_check` then verifies the result
# the way CI does — a locally green `make ci` leaves a tree Actions accepts.
# CI checks out clean; a local tree usually is not, hence --allow-dirty.
ci: lint fmt_check test cove ## CI rehearsal: fmt+lint+test+cove, packaging dry run, wheel/sdist, py tests
	$(ENV) cargo publish -p gras --dry-run --locked --allow-dirty
	$(ENV) rm -rf dist && uvx maturin@1.15.0 build --release --out dist && \
		uvx maturin@1.15.0 sdist --out dist
	uvx twine check dist/*
	rm -rf /tmp/gras-ci-smoke && uv venv /tmp/gras-ci-smoke --python 3.12 && \
		uv pip install --python /tmp/gras-ci-smoke/bin/python dist/*.whl && \
		/tmp/gras-ci-smoke/bin/python -c "import gras; print('gras', gras.__version__)"
	uv pip install --python /tmp/gras-ci-smoke/bin/python pytest pandas
	/tmp/gras-ci-smoke/bin/python python/tests/test_determinism.py
	/tmp/gras-ci-smoke/bin/python -m pytest python/tests -q

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
# 🚀 Release (local by default; CI also deploys on a push to main)
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
