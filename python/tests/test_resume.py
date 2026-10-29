"""Resume from Python: a split race must replay as the SAME race.

The claim is unusually strong, so the test is unusually blunt: run 6 clocks in
one process, then run the same 6 as 3 + resume(3), and compare the two
`history.csv` files byte for byte. Nothing is loaded from disk in the second
case — the engine replays each net from its blueprint + weight seed and
continues — so a digest match says the replay reproduced the weights, the
rolling windows, the culls and the guardrail's exam, not merely "it ran again".

Run with the project venv (needs the built extension and libtorch):

    source env_setup.sh
    .venv/bin/python python/tests/test_resume.py

No pytest dependency: plain asserts, exit code 0 = pass. Every run dir is a
temp dir, so nothing under `results/` is touched (AGENTS.md build rules).
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import tempfile
import types
from pathlib import Path

from gras import RaceConfig, RaceEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "blobs"

SEED = 4242
POP = 4


def cfg(steps: int, **kw) -> RaceConfig:
    """A tiny race; `none` keeps the per-step tables out of the test output."""
    return RaceConfig(pop_size=POP, elite_count=1, max_steps=steps, log_level="none", **kw)


def digest(run_dir: Path) -> str:
    return hashlib.sha256((run_dir / "history.csv").read_bytes()).hexdigest()


def live_hashes(run_dir: Path) -> set[str]:
    """The living population, read from the run's own net envelopes."""
    alive = set()
    for path in (run_dir / "nets").glob("*.json"):
        state = json.loads(path.read_text())
        if state.get("is_alive"):
            alive.add(state["hash"])
    return alive


def one_shot(root: Path, steps: int, **engine_kw) -> Path:
    """The reference: the whole race in one process."""
    run = root / "one-shot"
    run.mkdir()
    engine = RaceEngine(DATA_DIR, cfg(steps), seed=SEED, run_dir=run, **engine_kw)
    assert engine.run() == "MaxSteps"
    return run


def split(root: Path, first: int, total: int, **engine_kw) -> tuple[Path, RaceEngine]:
    """The claim: `first` clocks, the process ends, resume for the rest."""
    run = root / "split"
    run.mkdir()
    started = RaceEngine(DATA_DIR, cfg(first), seed=SEED, run_dir=run, **engine_kw)
    assert started.run() == "MaxSteps"
    assert started.step_clock == first, f"first half clock: {started.step_clock}"
    del started  # nothing of the first process survives the "restart"

    resumed = RaceEngine.resume(run, DATA_DIR, cfg(total))
    # The seed is RECORDED, not chosen: there is no seed argument on resume, and
    # the exam's pool split plus every replayed step key off this value.
    assert resumed.run_seed == SEED, f"resume must reuse the recorded seed: {resumed.run_seed}"
    assert resumed.step_clock == first, f"resume must pick up at the recorded clock: {resumed.step_clock}"
    assert resumed.live_count == POP, f"live population after replay: {resumed.live_count}"
    assert resumed.run() == "MaxSteps"
    assert resumed.step_clock == total, f"total clock: {resumed.step_clock}"
    return run, resumed


def check_twin_history() -> None:
    """3 + resume(3) == 6, down to the bytes of history.csv."""
    with tempfile.TemporaryDirectory(prefix="gras-py-resume-") as tmp:
        root = Path(tmp)
        a = one_shot(root, 6)
        b, resumed = split(root, 3, 6)
        da, db = digest(a), digest(b)
        assert da == db, (
            "a resumed race must replay bit-identically — the split run's history "
            f"differs ({da[:12]} vs {db[:12]}); diff {a}/history.csv {b}/history.csv"
        )
        # The same population survived, by identity (hash), not just by count.
        assert live_hashes(a) == live_hashes(b), "the resumed population must be the same nets"

        # The exam is part of the race, so it has to match too: a veredict scored
        # on a pool split from the WRONG seed would drift here and nowhere else.
        one = RaceEngine.resume(a, DATA_DIR, cfg(6)).guardrail()
        two = resumed.guardrail()
        assert one is not None and two is not None, "both runs must take the exam"
        assert one["hash"] == two["hash"], f"champion differs: {one['hash']} vs {two['hash']}"
        assert abs(one["holdout_mean"] - two["holdout_mean"]) < 1e-6, (
            one["holdout_mean"],
            two["holdout_mean"],
        )
        print(f"twin history ok: 3+3 == 6 ({da[:12]}) champion {one['hash'][:8]}")


def check_recipe_comes_from_engine_json() -> None:
    """A resumed run restates no recipe: loss, LR, clip and batch sizes are read
    back from the recorded `trainer` blob, and the result still matches a twin."""
    with tempfile.TemporaryDirectory(prefix="gras-py-resume-recipe-") as tmp:
        root = Path(tmp)
        recipe = {
            "loss": "mse",
            "fitness": "mae",
            "lr": 2e-3,
            "grad_clip": 0.5,
            "batch_size": 8,
            "eval_batch_size": 8,
        }
        a = one_shot(root, 4, **recipe)
        # NOTE: not one recipe kwarg is passed — that is the whole point.
        b, _ = split(root, 2, 4, **recipe)
        assert digest(a) == digest(b), "the recorded recipe must rebuild the same trainer"
        recorded = json.loads((b / "engine.json").read_text())["trainer"]
        assert recorded["loss"] == "mse", recorded
        assert abs(recorded["learning_rate"] - 2e-3) < 1e-9, recorded
        assert recorded["batch_size"] == 8 and recorded["eval_batch_size"] == 8, recorded
        print(f"recipe from engine.json ok: {recorded['loss']} lr={recorded['learning_rate']}")


def check_drift_is_refused() -> None:
    """A changed replay-relevant knob is refused BY NAME, not replayed anyway."""
    with tempfile.TemporaryDirectory(prefix="gras-py-resume-drift-") as tmp:
        root = Path(tmp)
        run = root / "drift"
        run.mkdir()
        RaceEngine(DATA_DIR, cfg(2), seed=SEED, run_dir=run, lr=1e-3).run()

        # Trainer recipe: the crate's own blob diff, which names the key.
        for kwarg, needle in (({"lr": 9e-3}, "learning_rate"), ({"batch_size": 32}, "batch_size")):
            try:
                RaceEngine.resume(run, DATA_DIR, cfg(4), **kwarg)
            except RuntimeError as exc:
                assert needle in str(exc), exc
                assert "mismatch" in str(exc), exc
            else:
                raise AssertionError(f"{kwarg} must be refused, not silently applied")

        # Engine knob: λ decides the ranking, so the engine refuses the change.
        try:
            RaceEngine.resume(run, DATA_DIR, cfg(4, risk_aversion=1.0))
        except RuntimeError as exc:
            assert "risk_aversion" in str(exc), exc
        else:
            raise AssertionError("a changed λ must be refused")

        # The metric set is frozen: one CSV cannot gain a column mid-file.
        r2 = root / "r2"
        r2.mkdir()
        RaceEngine(DATA_DIR, cfg(2), seed=SEED, run_dir=r2, metrics=["r2"]).run()
        # Omitted ⇒ the recorded set, so the file stays rectangular...
        assert RaceEngine.resume(r2, DATA_DIR, cfg(4)).run() == "MaxSteps"
        # ...an explicit identical set is fine...
        assert RaceEngine.resume(r2, DATA_DIR, cfg(4), metrics=["r2"]).run() == "MaxSteps"
        # ...and a different one is a caller error, not a silent second header.
        try:
            RaceEngine.resume(r2, DATA_DIR, cfg(4), metrics=["mse"])
        except ValueError as exc:
            assert "frozen" in str(exc), exc
        else:
            raise AssertionError("a changed metric set must be refused")
        print("drift refused ok: learning_rate, batch_size, risk_aversion, metrics")


class ScriptedEnv:
    """A gym-shaped world with no gym dependency (mirrors test_determinism)."""

    def __init__(self, length: int = 4):
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


def check_wrong_doors_are_named() -> None:
    """The two refusals a caller will actually hit name the run's mode / the file."""
    with tempfile.TemporaryDirectory(prefix="gras-py-resume-doors-") as tmp:
        root = Path(tmp)
        rl = root / "rl"
        rl.mkdir()
        rl_cfg = RaceConfig(pop_size=2, elite_count=1, max_steps=1, log_level="none")
        RaceEngine.gym(ScriptedEnv(), rl_cfg, seed=SEED, run_dir=rl).run()
        # An RL trainer IS its environment; engine.json cannot rebuild a Python object.
        try:
            RaceEngine.resume(rl, DATA_DIR, cfg(2))
        except ValueError as exc:
            assert "rl" in str(exc) and "RlEngine::resume" in str(exc), exc
        else:
            raise AssertionError("an RL run must not resume through the tabular door")

        # A directory that is not a run has no engine.json — say so, don't trace.
        empty = root / "not-a-run"
        empty.mkdir()
        try:
            RaceEngine.resume(empty, DATA_DIR, cfg(2))
        except RuntimeError as exc:
            assert "engine.json" in str(exc), exc
        else:
            raise AssertionError("a directory with no engine.json must be refused")
        print("wrong doors ok: rl mode named, missing engine.json named")


def check_final_population_table() -> None:
    """What a finished run leaves for a reader who does NOT resume it.

    ELI5: the run dir is the report. Every net that ever lived — alive or culled —
    carries its own final reading, and `history.csv` has every per-step one, so a
    Python caller can rank the WHOLE population (not only the survivors) and see
    who died when and why, with no new binding surface.
    """
    with tempfile.TemporaryDirectory(prefix="gras-py-resume-table-") as tmp:
        run = Path(tmp) / "table"
        run.mkdir()
        # Culls are gated by probation, so this race turns the windows down and the
        # rolls up: the point is a run dir with BOTH kinds of net in it.
        churn = cfg(
            5,
            crossover_rolls=2,
            mutate_rolls=1,
            crossover_prob=1.0,
            mutate_prob=1.0,
            crossover_probation_steps=1,
            mutation_probation_steps=1,
            elite_join_probation_steps=1,
        )
        engine = RaceEngine(DATA_DIR, churn, seed=SEED, run_dir=run)
        engine.run()
        # The exam is a CALL, not a stop hook: a Python run that never asks records
        # no champion (`guardrail: null`) and so seeds nothing. Ask, then read.
        assert engine.guardrail() is not None, "the exam must score this run's elites"
        nets = [json.loads(p.read_text()) for p in (run / "nets").glob("*.json")]
        alive = [n for n in nets if n.get("is_alive")]
        dead = [n for n in nets if not n.get("is_alive")]
        assert alive and dead, f"a 3-clock race must both keep and cull nets: {len(alive)}/{len(dead)}"
        # Every net that ever TRAINED carries the same three-legged final reading:
        # the readable window mean, its spread, and the risk-adjusted score it
        # ranked by. A net born on the last clock has none — and says so with
        # `samples = 0` instead of a zero that would read as "scored badly".
        for net in nets:
            if net["last_metrics"] is None:
                assert net["final_fitness_mean"] is None, net["hash"]
                assert net["final_fitness_samples"] == 0, net["hash"]
                continue
            assert net["final_fitness_mean"] is not None, net["hash"]
            assert net["final_risk_score"] is not None, net["hash"]
            assert net["final_fitness_samples"] > 0, net["hash"]
        for net in dead:
            assert net["culled_at_step"] is not None and net["cull_reason"], net["hash"]
        # One row per net per clock it trained — the full trace, not just the end.
        rows = list(csv.DictReader((run / "history.csv").open()))
        assert len(rows) >= len(nets), (len(rows), len(nets))
        assert {"hash", "step", "fitness", "fitness_smooth", "fitness_smooth_std"} <= set(rows[0])
        # And the exam's rank order: with elite_count=1 the champion is the best
        # LIVE net by the score the race ranked on, in the run's own direction.
        meta = json.loads((run / "engine.json").read_text())
        record = meta["guardrail"]
        assert record is not None, "the run must have taken the exam"
        scored = [n for n in alive if n["final_risk_score"] is not None]
        best = (min if meta["fitness_direction"] == "Minimize" else max)(
            scored, key=lambda n: n["final_risk_score"]
        )
        assert best["hash"] == record["champion_hash"], (
            f"champion {record['champion_hash'][:8]} != table's best {best['hash'][:8]}"
        )
        print(
            "final table ok: "
            f"{len(alive)} live + {len(dead)} culled nets, {len(rows)} history rows, "
            f"{len(record['elites'])} examined"
        )


def main() -> None:
    check_twin_history()
    check_recipe_comes_from_engine_json()
    check_drift_is_refused()
    check_wrong_doors_are_named()
    check_final_population_table()
    print("resume ok: replay is bit-identical, recipe is recorded, drift is refused")


if __name__ == "__main__":
    main()
