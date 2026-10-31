"""A run directory reproduces its race: `RaceEngine.resume(dir)` replays the same
nets and continues from the recorded clock — the Python twin of the Rust
`--resume` path, for BOTH arms (a tabular dir needs nothing else; an RL dir needs
its environment handed back).

Three things are asserted against the artifacts rather than against the binding's
own bookkeeping: the seeded identity (`run_seed`, `run_id`), the clock the run
stopped on (`step_clock` vs the last `history.csv` row), and the PREFIX property
(a resumed run extends its history — it never re-emits a row the first half
already wrote). The recipe is part of the record too, so naming a different loss
must be refused by the crate instead of quietly replaying another experiment.

Run with the project venv (needs the built extension and libtorch):

    source env_setup.sh
    $HOME/gras/.venv/bin/python3 python/tests/test_resume.py

No pytest dependency: plain asserts, exit code 0 = pass. Every engine gets its
own temp run dir, so nothing under `results/` is touched (AGENTS.md rule 16).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from gras import RaceConfig, RaceEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOBS = REPO_ROOT / "data" / "blobs"

POP = 4
STEPS = 3  # under the 5-clock newborn shield: nothing is culled, all stay live
SEED = 21
MORE = 5  # the ABSOLUTE budget the resume asks for


def first_run(tmp: Path) -> tuple[Path, str]:
    """One blobs race; return (run dir, the history it wrote)."""
    run_dir = tmp / "origin"
    cfg = RaceConfig(pop_size=POP, elite_count=1, max_steps=STEPS, log_level="none")
    RaceEngine(BLOBS, cfg, seed=SEED, run_dir=run_dir).run()
    return run_dir, (run_dir / "history.csv").read_text()


def test_resume_continues_the_recorded_run() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-resume-continue-") as tmp:
        origin, before = first_run(Path(tmp))
        header = json.loads((origin / "engine.json").read_text())

        engine = RaceEngine.resume(str(origin), max_steps=MORE)
        # The RECORD wins over the process: seed, id and clock come from the dir.
        assert engine.run_seed == SEED, (
            f"resumed seed {engine.run_seed}, expected the recorded {SEED}"
        )
        assert engine.run_id == header["run_id"], (
            f"resumed id {engine.run_id!r}, expected {header['run_id']!r}"
        )
        assert engine.step_clock == STEPS, (
            f"resumed clock {engine.step_clock}, expected the stop clock {STEPS}"
        )
        assert engine.live_count == POP, (
            f"resume restored {engine.live_count} live nets, expected pop_size={POP}"
        )

        assert engine.run() == "MaxSteps", "the extended budget should stop on MaxSteps"
        assert engine.step_clock == MORE, (
            f"after resume the clock is {engine.step_clock}, expected the absolute budget {MORE}"
        )
        after = (origin / "history.csv").read_text()
        # The first half's rows are a PREFIX of the resumed file: no rewriting, no
        # re-emitted (clock, hash) row.
        assert after.startswith(before), (
            "resume did not extend history.csv — the rows it already held changed"
        )


def test_a_different_recipe_is_refused() -> None:
    """The recipe is data: naming another loss must fail loudly, not replay a
    different experiment under the old one's name."""
    with tempfile.TemporaryDirectory(prefix="gras-resume-recipe-") as tmp:
        origin, _ = first_run(Path(tmp))
        try:
            RaceEngine.resume(str(origin), loss="mse")
        except RuntimeError as exc:
            assert "loss" in str(exc).lower(), (
                f"refusal should name the loss, got: {exc}"
            )
        else:
            raise AssertionError("resuming with a different loss must raise")


def test_rl_resume_continues_the_recorded_run() -> None:
    """An RL directory resumes too — the arm is the RECORD's, and the environment is
    the one thing on disk cannot hold. Everything else (signal, both episode counts,
    the challenge knobs, lr, clip, the cap) is read back out of `engine.json`."""
    try:
        import gymnasium
    except ImportError:  # the tabular checks above still run on a bare venv
        print("test_rl_resume: skipped (gymnasium not installed)")
        return

    with tempfile.TemporaryDirectory(prefix="gras-resume-rl-") as tmp:
        run_dir = Path(tmp) / "rl"
        cfg = RaceConfig(pop_size=2, elite_count=1, max_steps=1, log_level="none")
        RaceEngine.gym(gymnasium.make("CartPole-v1"), cfg, seed=SEED, run_dir=run_dir).run()
        before = (run_dir / "history.csv").read_text()

        # A FRESH env object: the resume must not need the instance the run used.
        resumed = RaceEngine.resume(
            str(run_dir), env=gymnasium.make("CartPole-v1"), max_steps=2
        )
        assert resumed.run_seed == SEED, f"resumed seed {resumed.run_seed}"
        assert resumed.step_clock == 1, f"resumed clock {resumed.step_clock}"
        assert resumed.live_count == 2, f"resumed {resumed.live_count} live nets"
        assert resumed.run() == "MaxSteps", "the extended budget should stop on MaxSteps"
        assert resumed.step_clock == 2, f"after resume the clock is {resumed.step_clock}"
        after = (run_dir / "history.csv").read_text()
        assert after.startswith(before), (
            "RL resume did not extend history.csv — the rows it already held changed"
        )


def test_rl_resume_needs_its_environment() -> None:
    """The one thing the record cannot hold is the world: without it the refusal
    names `env=`, and with a tabular kwarg the refusal names that kwarg."""
    try:
        import gymnasium
    except ImportError:
        print("test_rl_resume_needs_its_environment: skipped (gymnasium not installed)")
        return

    with tempfile.TemporaryDirectory(prefix="gras-resume-rl-env-") as tmp:
        run_dir = Path(tmp) / "rl"
        cfg = RaceConfig(pop_size=2, elite_count=1, max_steps=1, log_level="none")
        RaceEngine.gym(gymnasium.make("CartPole-v1"), cfg, seed=SEED, run_dir=run_dir).run()
        try:
            RaceEngine.resume(str(run_dir))
        except ValueError as exc:
            assert "env=" in str(exc), f"expected the missing env to be named, got: {exc}"
        else:
            raise AssertionError("an RL dir must not resume without env=")
        try:
            RaceEngine.resume(str(run_dir), env=gymnasium.make("CartPole-v1"), lr=0.5)
        except (RuntimeError, ValueError) as exc:
            assert "lr" in str(exc), f"the refusal should name lr, got: {exc}"
        else:
            raise AssertionError("a tabular recipe kwarg on an RL dir must be refused")


def test_resume_reports_a_missing_run() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-resume-missing-") as tmp:
        try:
            RaceEngine.resume(str(Path(tmp) / "not-a-run"))
        except RuntimeError as exc:
            assert "engine.json" in str(exc), (
                f"a dir with no engine.json should say so, got: {exc}"
            )
        else:
            raise AssertionError("resuming a dir with no engine.json must raise")


if __name__ == "__main__":
    test_resume_continues_the_recorded_run()
    test_a_different_recipe_is_refused()
    test_rl_resume_continues_the_recorded_run()
    test_rl_resume_needs_its_environment()
    test_resume_reports_a_missing_run()
    print("test_resume: ok (5 checks)")
