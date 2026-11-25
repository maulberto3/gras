"""The engine can be asked about the race it just ran — and the answer must
agree with the artifacts, not with a second implementation.

`live_count` is checked against the `is_alive` flags the engine wrote into
`nets/<hash>.json`; `smoothed_fitness_of(hash)` against the `fitness_smooth`
column of `history.csv` (which stores the risk-adjusted score, not the raw
window mean) and against the `λ` it is supposed to charge. Both are read back
from disk, so a binding that recomputes a number its own way fails here.

`history.csv` is a RECORD and keeps every net that ever ran a clock; only the
nets still in the population answer `smoothed_fitness_of`. A culled net's
rolling window goes with it, and its last numbers live in `nets/<hash>.json`
as `final_fitness_mean` / `final_fitness_std`. The checks below therefore read
the live intersection: a live net must agree with the CSV, a culled one must
answer None — never a stale number.

Run with the project venv (needs the built extension and libtorch):

    source env_setup.sh
    $HOME/gras/.venv/bin/python3 python/tests/test_introspection.py

No pytest dependency: plain asserts, exit code 0 = pass. Every engine gets its
own temp run dir, so nothing under `results/` is touched (AGENTS.md rule 16).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from gras import RaceConfig, RaceEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOBS = REPO_ROOT / "data" / "blobs"

POP = 4
STEPS = 3  # under the 5-clock newborn shield: nothing is culled, all stay live
SEED = 21


def run(
    tmp: Path, name: str, steps: int = STEPS, **overrides
) -> tuple[RaceEngine, Path, pd.DataFrame]:
    """One blobs race; return (engine, run dir, history)."""
    run_dir = tmp / name
    cfg = RaceConfig(pop_size=POP, champion_count=1, max_steps=steps,
                     log_level="none", **overrides)
    engine = RaceEngine(BLOBS, cfg, seed=SEED, run_dir=run_dir)
    engine.run()
    return engine, run_dir, pd.read_csv(run_dir / "history.csv")


def alive_hashes(run_dir: Path) -> set[str]:
    """The hashes the artifacts mark as alive — the engine's own record."""
    out = set()
    for path in (run_dir / "nets").glob("*.json"):
        if json.loads(path.read_text())["is_alive"]:
            out.add(path.stem)
    return out


def last_reading(history: pd.DataFrame, hash: str) -> pd.Series:
    return history[history["hash"] == hash].iloc[-1]


def test_live_count_matches_the_artifacts() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-introspect-live-") as tmp:
        run_dir = Path(tmp) / "live"
        engine = RaceEngine(
            BLOBS,
            RaceConfig(pop_size=POP, champion_count=1, max_steps=STEPS,
                       log_level="none"),
            seed=SEED,
            run_dir=run_dir,
        )
        assert engine.live_count == POP, (
            f"a fresh engine reports {engine.live_count} live nets, expected pop_size={POP}"
        )
        engine.run()
        alive = alive_hashes(run_dir)
        assert engine.live_count == len(alive), (
            f"live_count={engine.live_count} but nets/*.json marks {len(alive)} alive"
        )
        assert engine.live_count == POP, (
            f"no net should be culled in {STEPS} clocks (newborn shield), got {engine.live_count}"
        )


def test_smoothed_fitness_is_the_recorded_score() -> None:
    """`history.csv`'s `fitness_smooth` IS the window-centre score — the same
    number, or the binding has implemented a second formula."""
    with tempfile.TemporaryDirectory(prefix="gras-introspect-smooth-") as tmp:
        engine, run_dir, history = run(Path(tmp), "smooth")
        live = alive_hashes(run_dir)
        measured = [h for h in history["hash"].unique() if h in live]
        assert measured, "no net still in the population carried a reading into history.csv"
        for hash in measured:
            row = last_reading(history, hash)
            got = engine.smoothed_fitness_of(hash)
            assert got is not None, f"{hash}: no reading after a run that recorded one"
            assert abs(got - row["fitness_smooth"]) <= 1e-5, (
                f"{hash}: smoothed_fitness_of={got} but history.csv says "
                f"fitness_smooth={row['fitness_smooth']}"
            )
        # The rest of the record is tombstones: no current reading, no stale number.
        for hash in set(history["hash"].unique()) - live:
            assert engine.smoothed_fitness_of(hash) is None, (
                f"{hash}: a culled net must not report a current reading"
            )


def test_mean_type_reaches_the_ranking() -> None:
    """`mean_type="median"` is the ranking's centre, not a config field that is
    only echoed: the same seed ranks on a different centre, and the recorded
    score is still the one `smoothed_fitness_of` reports."""
    with tempfile.TemporaryDirectory(prefix="gras-introspect-mean-") as tmp:
        # `arithmetic` is named explicitly: the crate's DEFAULT is now `median`,
        # so a run that leaves it out is the same experiment, not a control.
        #
        # The family can only move a score when a window holds more than one
        # reading on the same weights. With the default policy-scoped window each
        # epoch contributes exactly one reading (mean == median by definition), so
        # bring the scoping off and run five clocks: a real multi-reading window.
        knobs = {"policy_scoped_fitness": False}
        plain, plain_dir, _ = run(Path(tmp), "arith", steps=5,
                                  mean_type="arithmetic", **knobs)
        median, run_dir, hist_b = run(Path(tmp), "median", steps=5,
                                      mean_type="median", **knobs)
        recorded = json.loads((run_dir / "engine.json").read_text())["config"]
        assert recorded["mean_type"] == "median", recorded

        live_b = alive_hashes(run_dir)
        live_plain = alive_hashes(plain_dir)
        shared = [h for h in hist_b["hash"].unique() if h in live_b and h in live_plain]
        assert shared, "the two runs no longer share a live, measured net"
        moved = 0
        for hash in shared:
            score = median.smoothed_fitness_of(hash)
            assert score is not None, f"{hash}: no reading with mean_type=median"
            row = last_reading(hist_b, hash)
            assert abs(score - row["fitness_smooth"]) <= 1e-5, (
                f"{hash}: smoothed_fitness_of={score} but history.csv says "
                f"fitness_smooth={row['fitness_smooth']}"
            )
            if plain.smoothed_fitness_of(hash) not in (None, score):
                moved += 1
        assert moved, "median and arithmetic ranked every net the same — the knob never arrived"


def test_unknown_hash_has_no_reading() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-introspect-none-") as tmp:
        engine, run_dir, history = run(Path(tmp), "none")
        assert engine.smoothed_fitness_of("f" * 16) is None, (
            "an unknown hash must answer None, not a default number"
        )
        assert engine.smoothed_fitness_of("") is None
        # A LIVE hash still answers — the None above is the lookup, not a dead method.
        live = alive_hashes(run_dir) & set(history["hash"])
        assert live, "no live net was measured, so the lookup cannot be probed"
        assert engine.smoothed_fitness_of(sorted(live)[0]) is not None


if __name__ == "__main__":
    test_live_count_matches_the_artifacts()
    test_smoothed_fitness_is_the_recorded_score()
    test_lambda_charges_the_window_spread()
    test_mean_type_reaches_the_ranking()
    test_unknown_hash_has_no_reading()
    print("test_introspection: ok (5 checks)")
