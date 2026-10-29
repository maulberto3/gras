"""Determinism is the product: same seed + config ⇒ identical fitness stream.

Run with the project venv (needs the built extension and libtorch):

    source env_setup.sh
    $HOME/gras/.venv/bin/python3 python/tests/test_determinism.py

No pytest dependency: plain asserts, exit code 0 = pass. Each engine gets its
own temp run dir, so nothing under `results/` is touched (AGENTS.md build rules).
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import tempfile
import types
from pathlib import Path

from gras import (
    RaceConfig,
    RaceEngine,
    algos,
    fitnesses,
    losses,
    metrics,
    tasks,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "blobs"


def run_once(seed: int, steps: int = 3, pop: int = 4) -> str:
    """Run a tiny race in a fresh dir; return the history.csv digest."""
    with tempfile.TemporaryDirectory(prefix="gras-py-test-") as tmp:
        # `none`: the default is `summ`, whose per-step tables would bury this
        # test's one-line verdict.
        cfg = RaceConfig(pop_size=pop, elite_count=1, max_steps=steps, log_level="none")
        engine = RaceEngine(DATA_DIR, cfg, seed=seed, run_dir=tmp)
        assert engine.run_seed == seed, f"seed recorded: {engine.run_seed}"
        reason = engine.run()
        assert reason == "MaxSteps", f"expected a step-budget stop, got {reason}"
        history = Path(tmp) / "history.csv"
        assert history.is_file(), "history.csv must land in the run dir"
        assert (Path(tmp) / "engine.json").is_file(), "engine.json must land too"
        rows = list(csv.reader(history.open()))
        # Header + at least one row per clock; the stream itself is what we hash.
        assert len(rows) > 1, "history.csv must carry metric rows"
        return hashlib.sha256(history.read_bytes()).hexdigest()


def check_log_level_knob() -> None:
    """`summ` is the default and an unknown level is refused, not guessed."""
    assert RaceConfig().log_level == "summ", "the unit tests above assume the summ default"
    assert RaceConfig(log_level="minimal").log_level == "minimal"
    try:
        RaceEngine(DATA_DIR, RaceConfig(log_level="loud"), seed=0)
    except ValueError as exc:
        assert "log_level" in str(exc), exc
    else:
        raise AssertionError("an unknown log level must be refused")


def check_trainer_knobs() -> None:
    """The recipe is picked by name, and the run records the one it actually ran."""
    assert "mse" in losses(), losses()
    assert "r2" in metrics(), metrics()
    assert "mae" in fitnesses(), fitnesses()
    with tempfile.TemporaryDirectory(prefix="gras-py-knobs-") as tmp:
        cfg = RaceConfig(pop_size=4, elite_count=1, max_steps=2, log_level="none")
        engine = RaceEngine(
            DATA_DIR,
            cfg,
            seed=7,
            run_dir=tmp,
            loss="mse",
            fitness="mae",
            metrics=["r2"],
            lr=2e-3,
        )
        assert engine.run() == "MaxSteps"
        # An informative metric becomes its own column in the recorded stream...
        header = next(csv.reader((Path(tmp) / "history.csv").open()))
        assert "r2" in header, header
        # ...and engine.json names the recipe, knobs and all, so a resume diffs it.
        recorded = json.loads((Path(tmp) / "engine.json").read_text())["trainer"]
        assert recorded["loss"] == "mse", recorded
        assert abs(recorded["learning_rate"] - 2e-3) < 1e-6, recorded


def check_engine_knobs() -> None:
    """The engine's own families reach the run, and start neutral.

    Neutral means the values that change NOTHING about a ranking: λ = 0 (plain
    window mean) and Ω = 0 (no lottery), plus no elite inheritance. Everything
    else starts at the crate's `RaceConfig::defaults()`.
    """
    cfg = RaceConfig()
    assert (cfg.risk_aversion, cfg.train_merit, cfg.elite_previous_load) == (0.0, 0.0, False), cfg
    assert (cfg.pop_size, cfg.elite_count, cfg.smoothing_window) == (5, 1, 10), cfg
    assert (cfg.crossover_prob, cfg.mutate_prob) == (0.5, 0.5), cfg
    assert (cfg.crossover_cull_policy, cfg.mutation_cull_policy) == ("worst", "worst"), cfg
    assert cfg.crossover_selection == "roulette", cfg
    assert cfg.activation_pool == [] and cfg.feature_split is None, cfg
    # The topology shape behind the width range: unset means the crate's 2..=5
    # for node count and both fan ranges, and dropout off.
    assert (
        cfg.hidden_nodes_pool,
        cfg.hidden_inputs_pool,
        cfg.hidden_outputs_pool,
        cfg.dropout_prob,
    ) == (None, None, None, 0.0), cfg
    # Durability + probation are NOT neutral-at-zero: `0` removes a protection, so
    # these inherit the crate's shipped values instead of starting bare.
    assert cfg.checkpoint_every == 10, cfg
    assert cfg.crossover_ops_pool == [], cfg
    assert (
        cfg.mutation_probation_steps,
        cfg.crossover_probation_steps,
        cfg.elite_join_probation_steps,
        cfg.elite_crown_lifetime_steps,
    ) == (5, 5, 5, 5), cfg

    with tempfile.TemporaryDirectory(prefix="gras-py-engine-knobs-") as tmp:
        cfg = RaceConfig(
            pop_size=4,
            elite_count=1,
            max_steps=2,
            log_level="none",
            crossover_prob=0.25,
            mutate_prob=0.75,
            freeze_elites=False,
            crossover_cull_policy="random",
            mutation_cull_policy="inverse_fitness",
            crossover_selection="tournament",
            tournament_size=3,
            crossover_ops_pool=["uniform"],
            checkpoint_every=3,
            mutation_probation_steps=2,
            crossover_probation_steps=3,
            elite_join_probation_steps=1,
            elite_crown_lifetime_steps=7,
            hidden_dim_pool=(8, 16),
            hidden_dim_stride=8,
            hidden_nodes_pool=(3, 4),
            hidden_inputs_pool=(1, 3),
            hidden_outputs_pool=(2, 4),
            dropout_prob=0.1,
            activation_pool=["relu"],
            combine_op_pool=["add"],
            standardize_op_pool=["layernorm"],
            feature_split="exclusive",
            trace_file=True,
            run_name="knob-check",
            guardrail_matches=0,
        )
        assert RaceEngine(DATA_DIR, cfg, seed=13, run_dir=tmp).run() == "MaxSteps"
        meta = json.loads((Path(tmp) / "engine.json").read_text())
        sub = meta["config"]
        # Evolution knobs, under `config`.
        assert sub["crossover_prob"] == 0.25, sub
        assert sub["mutate_prob"] == 0.75, sub
        assert sub["freeze_elites"] is False, sub
        assert sub["crossover_cull_policy"] == "random", sub
        assert sub["mutation_cull_policy"] == "inverse_fitness", sub
        assert sub["crossover_selection"].startswith("tournament"), sub
        assert sub["crossover_ops_pool"] == ["uniform"], sub
        # Durability + probation, also under `config`.
        assert sub["checkpoint_every"] == 3, sub
        assert sub["mutation_probation_steps"] == 2, sub
        assert sub["crossover_probation_steps"] == 3, sub
        assert sub["elite_join_probation_steps"] == 1, sub
        assert sub["elite_crown_lifetime_steps"] == 7, sub
        # Artifacts + label.
        assert sub["trace_file"] is True, sub
        assert sub["run_name"] == "knob-check", sub
        # Search space, recorded at the header root.
        assert meta["hidden_dim_pool"] == "8..=16", meta["hidden_dim_pool"]
        assert meta["hidden_dim_stride"] == 8, meta["hidden_dim_stride"]
        # The header records `topology_options` whole, so the node/fan ranges and
        # dropout are readable back without a second field.
        topo = meta["topology_options"]
        assert topo["min_hidden_num_nodes"] == 3 and topo["max_hidden_num_nodes"] == 4, topo
        assert topo["min_hidden_inputs_per_node"] == 1, topo
        assert topo["max_hidden_inputs_per_node"] == 3, topo
        assert topo["min_hidden_outputs_per_node"] == 2, topo
        assert topo["max_hidden_outputs_per_node"] == 4, topo
        assert abs(topo["dropout_prob"] - 0.1) < 1e-6, topo
        assert meta["activation_pool"] == ["relu"], meta["activation_pool"]
        assert meta["combine_op_pool"] == ["add"], meta["combine_op_pool"]
        assert meta["standardize_op_pool"] == ["layernorm"], meta["standardize_op_pool"]
        assert sub["feature_split"] == "split_exclusive", sub["feature_split"]

    # A name that the engine would silently fall back on mid-run is refused here.
    for bad in (
        {"crossover_cull_policy": "nope"},
        {"mutation_cull_policy": "nope"},
        {"crossover_selection": "nope"},
        {"feature_split": "nope"},
        {"activation_pool": ["nope"]},
        {"crossover_ops_pool": ["nope"]},
        {"hidden_dim_pool": (8, 4)},
        {"hidden_nodes_pool": (4, 2)},
        {"hidden_nodes_pool": (0, 4)},
        {"hidden_inputs_pool": (3, 1)},
        {"hidden_outputs_pool": (4, 2)},
        {"dropout_prob": 1.5},
    ):
        try:
            RaceEngine(DATA_DIR, RaceConfig(log_level="none", **bad), seed=0)
        except ValueError as exc:
            assert next(iter(bad)) in str(exc), exc
        else:
            raise AssertionError(f"{bad} must be refused, not guessed")


def check_unknown_names_are_refused() -> None:
    """A typo fails before a single step, naming the offender and the valid set."""
    for kwarg in ({"loss": "nope"}, {"fitness": "nope"}, {"metrics": ["nope"]}):
        try:
            RaceEngine(DATA_DIR, RaceConfig(log_level="none"), seed=0, **kwarg)
        except ValueError as exc:
            assert "nope" in str(exc), exc
            assert "expected one of" in str(exc), exc
        else:
            raise AssertionError(f"{kwarg} must be refused, not guessed")


def check_from_arrays() -> None:
    """NumPy is a first-class source: the on-ramp writes the NATIVE files, and
    the directory form then reads the identical stream."""
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 3)).astype(np.float32)
    onehot = np.zeros((16, 2), dtype=np.float32)
    onehot[np.arange(16), rng.integers(0, 2, size=16)] = 1.0

    def digest_from_arrays(y, dtype_x):
        with tempfile.TemporaryDirectory(prefix="gras-py-arrays-") as tmp:
            data = Path(tmp) / "data"
            cfg = RaceConfig(pop_size=4, elite_count=1, max_steps=2, log_level="none")
            engine = RaceEngine.from_arrays(
                dtype_x(x), y, data, cfg, seed=11, run_dir=tmp
            )
            assert engine.run() == "MaxSteps"
            # The NATIVE pair, not a CSV: `inputs.bin` is what a Rust caller writes
            # and what resume replays.
            for name in ("inputs.bin", "targets.bin", "meta.json"):
                assert (data / name).is_file(), f"from_arrays must write {name}"
            assert not (data / "inputs.csv").exists(), "no CSV round trip"
            return hashlib.sha256((Path(tmp) / "history.csv").read_bytes()).hexdigest()

    a = digest_from_arrays(onehot, lambda v: v)
    b = digest_from_arrays(onehot, lambda v: v)
    assert a == b, "same seed + same arrays must reproduce the same stream"

    # float64 is NumPy's default; it must work without a cast, and the result
    # must be the same dataset as the float32 spelling (up to f32 rounding of
    # exactly-representable values).
    a64 = digest_from_arrays(onehot.astype(np.float64), lambda v: v.astype(np.float64))
    assert a64 == a, "float64 input must write the same dataset as float32"

    # The directory form reads what the on-ramp wrote — one data format, two doors.
    with tempfile.TemporaryDirectory(prefix="gras-py-arrays-dir-") as tmp:
        data = Path(tmp) / "data"
        cfg = RaceConfig(pop_size=4, elite_count=1, max_steps=2, log_level="none")
        a_dir, b_dir = Path(tmp) / "a", Path(tmp) / "b"
        a_dir.mkdir()
        b_dir.mkdir()
        RaceEngine.from_arrays(x, onehot, data, cfg, seed=11, run_dir=a_dir).run()
        RaceEngine(data, cfg, seed=11, run_dir=b_dir).run()
        by_dir = hashlib.sha256((Path(tmp) / "b" / "history.csv").read_bytes()).hexdigest()
    assert by_dir == a, "a dir built by from_arrays must replay like any dataset dir"

    # A dtype is refused by name rather than reinterpreted as noise.
    try:
        digest_from_arrays(onehot, lambda v: v.astype(np.int64))
    except ValueError as exc:
        assert "int64" in str(exc) and "float32" in str(exc), exc
    else:
        raise AssertionError("an int64 X must be refused, not reinterpreted")

    # Row alignment is an invariant, not an assumption.
    try:
        digest_from_arrays(onehot[:-1], lambda v: v)
    except ValueError as exc:
        assert "rows" in str(exc), exc
    else:
        raise AssertionError("a short Y must be refused")


class ScriptedEnv:
    """A gym-shaped world with no gym dependency: 2 discrete actions, 1 float obs.

    `reset(seed=…)` picks which action scores, so an environment that ignored its
    seed would make two same-seed runs differ — which is exactly what the RL
    determinism check below relies on.
    """

    def __init__(self, length: int = 6):
        self.action_space = types.SimpleNamespace(n=2)
        self.spec = types.SimpleNamespace(max_episode_steps=length)
        self._length = length
        self._t = 0
        self._target = 0

    def reset(self, seed=None):
        self._t = 0
        self._target = random.Random(seed).randrange(2)
        return [0.0], {}

    def step(self, action):
        self._t += 1
        reward = 1.0 if action == self._target else 0.0
        return [float(self._t)], reward, self._t >= self._length, False, {}


def check_tasks_and_algos() -> None:
    """Discovery tables exist, and `task=` presets the loss it names."""
    assert tasks() == ["classification", "regression"], tasks()
    assert algos() == ["reinforce"], algos()

    with tempfile.TemporaryDirectory(prefix="gras-py-task-") as tmp:
        cfg = RaceConfig(pop_size=4, elite_count=1, max_steps=2, log_level="none")
        engine = RaceEngine(DATA_DIR, cfg, seed=3, run_dir=tmp, task="regression")
        assert engine.run() == "MaxSteps"
        recorded = json.loads((Path(tmp) / "engine.json").read_text())["trainer"]
        assert recorded["loss"] == "mse", f"task=regression must preset mse, got {recorded}"

    try:
        RaceEngine(DATA_DIR, RaceConfig(log_level="none"), seed=0, task="nope")
    except ValueError as exc:
        assert "classification" in str(exc), exc
    else:
        raise AssertionError("an unknown task must be refused")


def check_scripted_env() -> None:
    """RL from Python: a gym-shaped env object, the crate's `Reinforce` underneath."""

    def digest() -> str:
        with tempfile.TemporaryDirectory(prefix="gras-py-rl-") as tmp:
            cfg = RaceConfig(pop_size=2, elite_count=1, max_steps=2, log_level="none")
            engine = RaceEngine.gym(ScriptedEnv(), cfg, seed=5, run_dir=tmp, lr=2e-3)
            assert engine.run() == "MaxSteps"
            meta = json.loads((Path(tmp) / "engine.json").read_text())
            assert meta["engine_mode"] == "rl", meta["engine_mode"]
            assert meta["trainer"]["trainer"] == "reinforce", meta["trainer"]
            assert meta["trainer"]["env"]["env"] == "ScriptedEnv", meta["trainer"]["env"]
            assert meta["trainer"]["env"]["actions"] == 2, meta["trainer"]["env"]
            assert meta["trainer"]["env"]["max_turns"] == 6, meta["trainer"]["env"]
            return hashlib.sha256((Path(tmp) / "history.csv").read_bytes()).hexdigest()

    a, b = digest(), digest()
    assert a == b, "same seed + same env must reproduce (reset(seed) must decide the episode)"

    # The objective knob reaches the crate: a discount switches the signal from
    # survival turns to discounted episode return (and records that it did).
    with tempfile.TemporaryDirectory(prefix="gras-py-rl-return-") as tmp:
        cfg = RaceConfig(pop_size=2, elite_count=1, max_steps=1, log_level="none")
        engine = RaceEngine.gym(
            ScriptedEnv(), cfg, seed=5, run_dir=tmp, discount=0.9, challenge_prob=0.0
        )
        assert engine.run() == "MaxSteps"
        trainer = json.loads((Path(tmp) / "engine.json").read_text())["trainer"]
        assert trainer["signal"]["kind"] == "return", trainer["signal"]
        assert trainer["challenge_prob"] == 0.0, trainer["challenge_prob"]

    try:
        RaceEngine.gym(ScriptedEnv(), RaceConfig(log_level="none"), algo="ppo")
    except ValueError as exc:
        assert "reinforce" in str(exc), exc
    else:
        raise AssertionError("an unknown algo must be refused")


def check_cartpole_if_available() -> None:
    """Real gymnasium when it is installed; the suite still passes without it."""
    try:
        import gymnasium
    except ImportError:
        return
    with tempfile.TemporaryDirectory(prefix="gras-py-cartpole-") as tmp:
        # One clock, two nets: enough to prove the adapter drives a real
        # environment (ndarray observations, 5-tuple steps, spec.max_episode_steps).
        cfg = RaceConfig(pop_size=2, elite_count=1, max_steps=1, log_level="none")
        engine = RaceEngine.gym(gymnasium.make("CartPole-v1"), cfg, seed=11, run_dir=tmp)
        assert engine.run() == "MaxSteps"
        recorded = json.loads((Path(tmp) / "engine.json").read_text())["trainer"]
        assert recorded["env"]["env"] == "CartPole-v1", recorded["env"]
        assert recorded["env"]["max_turns"] == 500, recorded["env"]
        assert recorded["env"]["actions"] == 2, recorded["env"]


# ── one smoke run per menu entry ──────────────────────────────────────────
# The published tables (`losses()`, `metrics()`, `fitnesses()`, `tasks()`,
# `algos()`) are swept FROM the table, so an entry added there is covered the
# day it lands. The config-string menus are not published — `RaceConfig` parses
# them against the crate's own spellings — so they are mirrored here. A spelling
# accepted there but absent here is a coverage gap, not a failure; every name
# listed below must actually train, which is what this sweep proves.
ACTIVATIONS = [
    "identity",
    "relu",
    "gelu",
    "silu",
    "selu",
    "tanh",
    "sigmoid",
    "mish",
    "leaky_relu",
    "elu",
    "gelu_tanh",
    "softplus",
    "hardswish",
    "hardsigmoid",
    "sin",
    "cos",
    "softmax",
    "log_softmax",
]
COMBINE_OPS = ["add", "mean", "max", "min", "mul", "sub", "div"]
STANDARDIZE_OPS = ["identity", "layernorm", "rmsnorm", "instancenorm", "groupnorm"]


def smoke(cfg_kw=None, engine_kw=None, env=None) -> tuple[dict, list[str]]:
    """The cheapest run that still trains: one clock, two nets, fresh dir.

    A menu entry that parses but cannot build a net or take a step fails here
    and nowhere else — the engine's error paths all assume a name it accepted
    is a name it can run. Returns `(engine.json, history.csv header)`.
    """
    with tempfile.TemporaryDirectory(prefix="gras-py-menu-") as tmp:
        cfg = RaceConfig(
            pop_size=2, elite_count=1, max_steps=1, log_level="none", **(cfg_kw or {})
        )
        if env is None:
            engine = RaceEngine(DATA_DIR, cfg, seed=1, run_dir=tmp, **(engine_kw or {}))
        else:
            engine = RaceEngine.gym(env, cfg, seed=1, run_dir=tmp, **(engine_kw or {}))
        assert engine.run() == "MaxSteps", f"{cfg_kw or engine_kw}: did not finish a clock"
        meta = json.loads((Path(tmp) / "engine.json").read_text())
        header = next(csv.reader((Path(tmp) / "history.csv").open()))
        return meta, header


def check_menu_entries() -> None:
    """Every accepted spelling trains for one clock (Stage D, item 19)."""
    for loss in losses():
        smoke(engine_kw={"loss": loss})
    for fitness in fitnesses():
        smoke(engine_kw={"fitness": fitness})
    for metric in metrics():
        # An informative metric is recorded per step as its own column, so a
        # name the trainer cannot actually score would leave the column out.
        _, header = smoke(engine_kw={"metrics": [metric]})
        assert metric in header, header
    for task in tasks():
        smoke(engine_kw={"task": task})
    for algo in algos():
        meta, _ = smoke(engine_kw={"algo": algo}, env=ScriptedEnv())
        assert meta["engine_mode"] == "rl", meta["engine_mode"]

    # One-entry pools force every net through that entry.
    for name in ACTIVATIONS:
        meta, _ = smoke(cfg_kw={"activation_pool": [name]})
        assert len(meta["activation_pool"]) == 1, name
    for name in COMBINE_OPS:
        smoke(cfg_kw={"combine_op_pool": [name]})
    for name in STANDARDIZE_OPS:
        smoke(cfg_kw={"standardize_op_pool": [name]})
    for op in ("one_point", "uniform"):
        smoke(cfg_kw={"crossover_ops_pool": [op]})

    for policy in ("worst", "random"):
        smoke(cfg_kw={"crossover_cull_policy": policy})
    for policy in ("worst", "inverse_fitness", "random"):
        smoke(cfg_kw={"mutation_cull_policy": policy})
    for method in ("roulette", "tournament"):
        smoke(cfg_kw={"crossover_selection": method})
    for split in ("random", "exclusive", "split_random", "split_exclusive"):
        smoke(cfg_kw={"feature_split": split})


def main() -> None:
    check_log_level_knob()
    check_trainer_knobs()
    check_engine_knobs()
    check_unknown_names_are_refused()
    check_from_arrays()
    check_tasks_and_algos()
    check_scripted_env()
    check_cartpole_if_available()
    check_menu_entries()
    a = run_once(seed=42)
    b = run_once(seed=42)
    assert a == b, "same seed + config must reproduce the same fitness stream"
    c = run_once(seed=43)
    assert c != a, "a different seed must produce a different stream"
    print(f"determinism ok: seed42={a[:12]} seed43={c[:12]}")


if __name__ == "__main__":
    main()
