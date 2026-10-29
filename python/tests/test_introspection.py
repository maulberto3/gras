"""The engine can be asked about the race it just ran — and the answer must
agree with the artifacts, not with a second implementation.

`live_count` is checked against the `is_alive` flags the engine wrote into
`nets/<hash>.json`; `smoothed_fitness_of(hash)` against the `fitness_smooth`
column of `history.csv` (which stores the risk-adjusted score, not the raw
window mean) and against the `λ` it is supposed to charge. Both are read back
from disk, so a binding that recomputes a number its own way fails here.

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


def run(tmp: Path, name: str, **overrides) -> tuple[RaceEngine, Path, pd.DataFrame]:
    """One blobs race; return (engine, run dir, history)."""
    run_dir = tmp / name
    cfg = RaceConfig(pop_size=POP, elite_count=1, max_steps=STEPS,
                     guardrail_matches=0, log_level="none", **overrides)
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
            RaceConfig(pop_size=POP, elite_count=1, max_steps=STEPS,
                       guardrail_matches=0, log_level="none"),
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
    """`history.csv`'s `fitness_smooth` IS the risk-adjusted score — the same
    number, or the binding has implemented a second formula."""
    with tempfile.TemporaryDirectory(prefix="gras-introspect-smooth-") as tmp:
        engine, _, history = run(Path(tmp), "smooth")
        for hash in history["hash"].unique():
            row = last_reading(history, hash)
            got = engine.smoothed_fitness_of(hash)
            assert got is not None, f"{hash}: no reading after a run that recorded one"
            assert abs(got - row["fitness_smooth"]) <= 1e-5, (
                f"{hash}: smoothed_fitness_of={got} but history.csv says "
                f"fitness_smooth={row['fitness_smooth']}"
            )


def test_lambda_charges_the_window_spread() -> None:
    """Two λ, same seed: the difference must be exactly `Δλ × σ`.

    `score = mean ∓ λ·σ`, so a net's score under λ=2 is its λ=0 score plus
    `2σ` on a Minimize race — the spread it was charged for, nothing else.
    """
    with tempfile.TemporaryDirectory(prefix="gras-introspect-lambda-") as tmp:
        base, _, hist0 = run(Path(tmp), "lam0", risk_aversion=0.0)
        charged, run_dir, hist2 = run(Path(tmp), "lam2", risk_aversion=2.0)
        direction = json.loads((run_dir / "engine.json").read_text())["fitness_direction"]
        # "Worse side" is HIGHER for Minimize (loss) and LOWER for Maximize (reward).
        sign = 1.0 if direction == "Minimize" else -1.0

        checked = 0
        for hash in hist0["hash"].unique():
            if hash not in set(hist2["hash"]):
                continue  # a net that did not appear in both races proves nothing
            sd = last_reading(hist2, hash)["fitness_smooth_std"]
            if not sd > 1e-6:
                continue  # a flat window has no charge to observe
            charged_score = charged.smoothed_fitness_of(hash)
            base_score = base.smoothed_fitness_of(hash)
            assert charged_score is not None and base_score is not None
            expected = base_score + sign * 2.0 * sd
            assert abs(charged_score - expected) <= 1e-4, (
                f"{hash}: λ=2 score {charged_score}, λ=0 score {base_score}, σ {sd} — "
                f"expected {expected} ({direction}: mean {'+' if sign > 0 else '−'} λσ)"
            )
            checked += 1
        assert checked, "no net had a non-flat window — the λ charge was never observable"


def test_unknown_hash_has_no_reading() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-introspect-none-") as tmp:
        engine, _, history = run(Path(tmp), "none")
        assert engine.smoothed_fitness_of("f" * 16) is None, (
            "an unknown hash must answer None, not a default number"
        )
        assert engine.smoothed_fitness_of("") is None
        # A real hash still answers — the None above is the lookup, not a dead method.
        assert engine.smoothed_fitness_of(history["hash"].iloc[0]) is not None


if __name__ == "__main__":
    test_live_count_matches_the_artifacts()
    test_smoothed_fitness_is_the_recorded_score()
    test_lambda_charges_the_window_spread()
    test_unknown_hash_has_no_reading()
    print("test_introspection: ok (4 checks)")
