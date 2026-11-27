SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c

# ── Backend: CPU (default) or CUDA ─────────────────────────────────────────────
# `make … BACKEND=cuda` sources `env_setup.sh cuda`, so LIBTORCH_PATH points at
# the CUDA libtorch, builds land in target_cuda, and $${GRAS_FEATURES} carries
# `--features cuda` into every cargo/maturin command. Never mix the two in one
# shell: separate invocations and separate target dirs (AGENTS.md).
BACKEND ?= cpu
ifeq ($(filter $(BACKEND),cpu cuda),)
$(error BACKEND must be cpu or cuda, got '$(BACKEND)')
endif

# Every recipe that links or loads libtorch runs through env_setup.sh, which
# exports LIBTORCH_PATH and pins CARGO_TARGET_DIR. GRAS_QUIET silences its
# banner; the silence itself comes from the tools (`-q`), not from make's `@`.
BACKEND_ARG := $(if $(filter cuda,$(BACKEND)),cuda,)
ENV := export GRAS_QUIET=1; source env_setup.sh$(if $(BACKEND_ARG), $(BACKEND_ARG)) &&
# For the one target that must always be CUDA regardless of BACKEND (`ci-cuda`).
CUDA_ENV := export GRAS_QUIET=1; source env_setup.sh cuda &&

.DEFAULT_GOAL := help
.PHONY: help all prep clean fmt lint clippy test cove docu py pymod notebook ci ci-cuda benc prof mem samp dry-publ _require_level

################################################################################
# 🧹 Clean
################################################################################
# The light cleanup `ci` runs: prune the cargo cache and remove the stray
# `target/` a build without env_setup.sh leaves behind — that folder has the
# WRONG backend and must never be reused. target_cpu/ and target_cuda/ are kept
# on purpose; the big `cargo clean` is yours to run by hand.
clean: ## prune the cargo cache and remove the stray default target/ dir
	cargo cache --autoclean
	rm -rf target

################################################################################
# 🔨 Build
################################################################################
prep: ## compile lib + bins + examples + benches + tests
	$(ENV) cargo build -q $${GRAS_FEATURES}

################################################################################
# 📐 Format & lint
################################################################################
fmt: ## rewrite formatting with rustfmt (the fixer)
	$(ENV) cargo fmt --all -q

# Split out from `lint` so a backend pass that does not want the clean/rebuild
# (`ci-cuda`) can still get clippy.
clippy: ## clippy only (warnings reported, not fatal)
	$(ENV) cargo clippy --all-targets -q $${GRAS_FEATURES}

# `fmt` is the fixer CI's `--check` would demand, so `ci` runs it and never needs
# a second `--check` pass: a locally green run leaves no format diff behind.
lint: clean prep fmt clippy ## clean + build + rustfmt, then clippy (warnings reported)

################################################################################
# 🧪 Tests & coverage
################################################################################
test: ## run the Rust test suite
	$(ENV) cargo test -q $${GRAS_FEATURES}

cove: lint test ## HTML coverage report (cargo tarpaulin)
	$(ENV) cargo tarpaulin --out Html $${GRAS_FEATURES}

################################################################################
# 📚 Docs
################################################################################
# Rustdoc for the crate itself. Kept OFF the local `lint`/`ci` path: rustdoc
# recompiles the dependency graph in doc mode, which is minutes of laptop time for
# a check CI already runs (and docs.rs regenerates on publish). Run it on demand.
docu: ## build the Rust docs (CI runs this; slow locally)
	$(ENV) cargo doc -q --no-deps $${GRAS_FEATURES}

################################################################################
# 🐍 Python
################################################################################
# Refresh the EDITABLE extension (python/gras/_core.abi3.so, gitignored) that the
# notebook and a bare `import gras` read. CI builds its own wheel from the pushed
# source, so without this the local artifact stays at whatever Rust it last saw.
# Only the maturin step: `ci-cuda` rebuilds the extension without dragging the
# coverage run behind it, while `py` keeps its full-check prerequisite.
pymod: ## rebuild the editable Python extension only (maturin develop)
	$(ENV) VIRTUAL_ENV="$(PWD)/.venv" .venv/bin/maturin develop -q --manifest-path gras-py/Cargo.toml $${GRAS_FEATURES}

py: cove pymod ## rebuild the editable Python extension into .venv (maturin develop) after the full check

# The notebook is committed WITH its outputs: execute every code cell in a fresh
# namespace (reverse order, per AGENTS.md rule 18) and rewrite the stored outputs
# with the extension `py` just built. A failing cell aborts without writing.
notebook: py ## execute the tour's code cells and refresh its stored outputs
	$(ENV) .venv/bin/python python/run_notebook.py

################################################################################
# 🚦 CI rehearsal (mirrors .github/workflows/ci.yml)
################################################################################
# `py` + `lint` + `test` are prerequisites, so this one command covers the
# workflow's quality gate and packaging. `py` refreshes the editable extension
# FIRST, so anything run afterward against `.venv` (the notebook, a manual REPL)
# sees the Rust in this tree. `lint` runs the rustfmt FIXER: a locally green
# `make ci` leaves a tree Actions accepts. `docu` is NOT here — CI runs
# `cargo doc`, and rustdoc is slow on a laptop.
# CI checks out clean; a local tree usually is not, hence --allow-dirty.
# Only the sdist is PyPI-acceptable: the wheel is a plain `linux` tag (it does
# not bundle libtorch), and PyPI requires manylinux/musllinux. Same split the
# workflow uses, so a local `make ci` stages exactly what a push would upload.
#
# This is the CPU pass on purpose. A CUDA pass compiles into target_cuda, sharing
# no artifacts with target_cpu, so running `ci` with BACKEND=cuda rebuilds the
# entire graph a second time (coverage and notebook included). The CUDA half is
# its own optional gate: `make ci-cuda`.
ci: notebook ## py rebuild, clean+build+fmt+clippy, tests+cove, packaging, py tests
	$(ENV) cargo publish -p gras -q --dry-run --locked --allow-dirty
	$(ENV) rm -rf dist && uvx maturin@1.15.0 build -q --release --out dist $${GRAS_FEATURES}
	uvx maturin@1.15.0 sdist --out dist
	uvx twine check dist/*
	rm -rf dist/pypi && mkdir -p dist/pypi && cp dist/*.tar.gz dist/pypi/
	$(ENV) rm -rf /tmp/gras-ci-smoke && uv venv -q /tmp/gras-ci-smoke --python 3.12 && \
		uv pip install -q --python /tmp/gras-ci-smoke/bin/python dist/*.whl && \
		/tmp/gras-ci-smoke/bin/python -c "import gras; print('gras', gras.__version__)"
	uv pip install -q --python /tmp/gras-ci-smoke/bin/python pytest pandas
	$(ENV) /tmp/gras-ci-smoke/bin/python python/tests/test_determinism.py
	$(ENV) /tmp/gras-ci-smoke/bin/python -m pytest python/tests -q

# ── Explicit CUDA gate ───────────────────────────────────────────────────────
# The CUDA equivalent of the parts of `ci` that can actually differ per backend:
# compile, lint, tests and the extension. Kept OFF the `ci` chain and free of
# coverage/notebook/packaging, because those are identical work that would only
# be repeated in a second target dir. Run it before `make ci`, or on its own when
# only the CUDA side changed.
ci-cuda: ## optional CUDA gate: build + clippy + tests + CUDA extension (no coverage/notebook)
	$(MAKE) prep fmt clippy test pymod BACKEND=cuda
	$(CUDA_ENV) .venv/bin/python -c "import gras; assert gras.built_with_cuda, 'extension is not a CUDA build — check that --features cuda reached maturin'; print('gras', gras.__version__, 'built with cuda')"

################################################################################
# ⚡ Performance & profiling
################################################################################
benc: ## stream benchmark
	$(ENV) cargo bench -q $${GRAS_FEATURES} --bench stream

prof: ## flamegraph of the profiling workload
	$(ENV) mkdir -p results/prof && \
	cargo flamegraph $${GRAS_FEATURES} --profile profiling -o results/prof/flamegraph.svg --bench flamegraph -- --steps 300

mem: ## heap profile (dhat) of the benchmark workload
	$(ENV) cargo bench -q $${GRAS_FEATURES} --bench memgraph -- --steps 300 --evolve

# Needs perf access: echo '1' | sudo tee /proc/sys/kernel/perf_event_paranoid
samp: ## record a sampled profile of the ask_tell binary
	$(ENV) samply record cargo run -q $${GRAS_FEATURES} --release --bin ask_tell

################################################################################
# 🚀 Release rehearsal (publishing happens in GitHub Actions on a push to main)
#
# LEVEL is required: cargo-release's default level is `release`, which only
# strips a pre-release suffix and keeps the current version. Pass patch | minor | major |
# alpha/rc/beta, or an explicit version, e.g. `make dry-publ LEVEL=minor`.
#
# There is deliberately no local `publ`: the workflow is the only publisher,
# and it deploys crates.io + PyPI only from a push to `main` after the same
# job's checks pass.
#
# `dry-publ` is the MAXIMAL local rehearsal: it runs the whole `ci` mirror
# first, then cargo-release's plan for the crate, then the PyPI plan (what
# `dist/pypi` would upload and whether that version already exists). It
# commits, tags, pushes and publishes nothing.
################################################################################
_require_level:
	@test -n "$(LEVEL)" || { echo "usage: make dry-publ LEVEL=patch|minor|major|X.Y.Z"; exit 1; }

dry-publ: _require_level ci ## plan a release after the full CI rehearsal + PyPI plan (LEVEL=...)
	$(ENV) cargo release $(LEVEL)
	$(ENV) .venv/bin/python python/dry_publish.py


################################################################################
# 📖 Help
################################################################################
help: ## list targets
	@echo "Targets:"
	@grep -hE '^[a-z][a-z0-9_-]*:.*## ' $(MAKEFILE_LIST) | sort \
		| awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'
