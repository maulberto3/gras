"""Interrupt a race for real, then resume it — the durability acceptance test.

Three scenarios, each a SIGKILL or SIGINT to a live writer process:

1. **SIGKILL with evolution on.** Before the frontier manifest, the net files culled
   after the last tick overwrote the frontier with tombstones: 8 live nets became 2
   and `resume` refused as a `pop_size` mismatch. The manifest is the frontier SET,
   so the resume must now accept and restore all 8.
2. **SIGKILL with `history_flush_each`.** Rows are written every clock while the
   frontier moves every `checkpoint_every`, so the CSV can run ahead; before the
   rewind, resuming re-emitted those rows (measured 16 and 60 duplicates). Now the
   CSV is truncated to the manifest clock first, so there must be none.
3. **SIGINT (Ctrl+C).** A graceful stop writes the frontier at the stop clock, so
   the dir is already a prefix — resume must accept, continue, and duplicate nothing.

Run with the project venv (needs the built extension and libtorch):

    source env_setup.sh
    .venv/bin/python python/tests/test_interrupt.py

The probes that found the original defects live in `notes/probes/`.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from gras import RaceConfig, RaceEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "blobs"
SEED = 17

# The writer: a race long enough that the parent's signal always lands mid-run.
WRITER = r'''
import json, sys
from pathlib import Path
from gras import RaceConfig, RaceEngine
run, kw, seed, data = Path(sys.argv[1]), json.loads(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
reason = RaceEngine(data, RaceConfig(**kw), seed=seed, run_dir=run).run()
print("REASON:", reason)
'''


def run_rows(run_dir: Path) -> list[tuple[int, str]]:
    lines = (run_dir / "history.csv").read_text().splitlines()
    head = lines[0].split(",")
    i_step, i_hash = head.index("step"), head.index("hash")
    return [(int(r.split(",")[i_step]), r.split(",")[i_hash]) for r in lines[1:]]


def manifest_of(run_dir: Path) -> dict:
    return json.loads((run_dir / "ckpt.json").read_text())


def kill_mid_run(tmp: Path, run_dir: Path, cfg: dict, sig: signal.Signals, after: float = 2.0):
    """Start a writer, signal it mid-race, return `(exit_code, output)`."""
    script = tmp / "writer.py"
    script.write_text(WRITER)
    proc = subprocess.Popen(
        [sys.executable, str(script), str(run_dir), json.dumps(cfg), str(SEED), str(DATA_DIR)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(after)
    proc.send_signal(sig)
    out, _ = proc.communicate(timeout=60)
    return proc.returncode, out


def check_sigkill_with_evolution_restores_the_set() -> None:
    """P1: the manifest restores all `pop_size` nets, not just the survivors."""
    cfg = dict(
        pop_size=8,
        elite_count=1,
        max_steps=100_000,
        checkpoint_every=5,
        log_level="none",
        # Heavy churn: one net dies and one is born almost every clock, which is what
        # buried the frontier under tombstones.
        crossover_rolls=2,
        mutate_rolls=1,
        crossover_prob=1.0,
        mutate_prob=1.0,
        crossover_probation_steps=1,
        mutation_probation_steps=1,
        elite_join_probation_steps=1,
    )
    with tempfile.TemporaryDirectory(prefix="gras-interrupt-churn-") as tmp:
        tmp = Path(tmp)
        run = tmp / "run"
        run.mkdir()
        code, _ = kill_mid_run(tmp, run, cfg, signal.SIGKILL)
        assert code != 0, "SIGKILL must not be a clean exit"

        manifest = manifest_of(run)
        assert len(manifest["nets"]) == 8, f"manifest must list the frontier set: {manifest}"
        live_now = [json.loads(p.read_text()) for p in (run / "nets").glob("*.json")]
        alive_now = [n for n in live_now if n["is_alive"]]
        assert len(alive_now) < 8, (
            "this scenario is only interesting when post-tick culls have overwritten the "
            f"frontier: {len(alive_now)} alive of {len(live_now)} files"
        )

        resume_cfg = RaceConfig(**{**cfg, "max_steps": manifest["next_clock"] + 2})
        engine = RaceEngine.resume(run, DATA_DIR, resume_cfg)
        assert engine.live_count == 8, f"the manifest restores the WHOLE set: {engine.live_count}"
        assert engine.step_clock == manifest["next_clock"], engine.step_clock
        assert engine.run() == "MaxSteps"
        # The clock continues from the manifest, not from the youngest survivor.
        assert engine.step_clock == manifest["next_clock"] + 2, engine.step_clock
        print(
            f"churn ok: {len(alive_now)} files looked alive, manifest restored 8, "
            f"resumed at {manifest['next_clock']}"
        )


def check_flush_ahead_has_no_duplicate_rows() -> None:
    """P2: the CSV is rewound to the checkpoint, so nothing is written twice."""
    cfg = dict(
        pop_size=4,
        elite_count=1,
        max_steps=100_000,
        # Coarse ticks + per-clock flushes: the CSV is guaranteed to run AHEAD.
        checkpoint_every=50,
        history_flush_each=True,
        log_level="none",
        crossover_rolls=0,
        mutate_rolls=0,
    )
    with tempfile.TemporaryDirectory(prefix="gras-interrupt-flush-") as tmp:
        tmp = Path(tmp)
        run = tmp / "run"
        run.mkdir()
        kill_mid_run(tmp, run, cfg, signal.SIGKILL)

        manifest = manifest_of(run)
        frontier = manifest["next_clock"] - 1
        ahead = max(step for step, _ in run_rows(run)) - frontier
        assert ahead > 0, (
            "this scenario needs rows written after the checkpoint "
            f"(frontier clock {frontier}, history {max(s for s, _ in run_rows(run))})"
        )

        resume_cfg = RaceConfig(**{**cfg, "max_steps": manifest["next_clock"] + 5})
        engine = RaceEngine.resume(run, DATA_DIR, resume_cfg)
        assert engine.run() == "MaxSteps"
        rows = run_rows(run)
        duplicates = sum(c - 1 for c in Counter(rows).values() if c > 1)
        assert duplicates == 0, (
            f"{duplicates} duplicate (step, hash) rows — the CSV was not rewound to the "
            f"checkpoint ({ahead} rows sat ahead of it before the resume)"
        )
        assert max(step for step, _ in rows) == engine.step_clock - 1, engine.step_clock
        print(f"flush-ahead ok: {ahead} rows sat ahead of the frontier, 0 duplicates after resume")


def check_sigint_is_a_graceful_prefix() -> None:
    """Ctrl+C writes the frontier at the stop clock, so the dir is already a prefix."""
    cfg = dict(
        pop_size=4,
        elite_count=1,
        max_steps=100_000,
        checkpoint_every=10,
        log_level="none",
        crossover_rolls=0,
        mutate_rolls=0,
    )
    with tempfile.TemporaryDirectory(prefix="gras-interrupt-sigint-") as tmp:
        tmp = Path(tmp)
        run = tmp / "run"
        run.mkdir()
        code, out = kill_mid_run(tmp, run, cfg, signal.SIGINT)
        assert code == 0, f"a graceful stop must exit 0, got {code}\n{out[-800:]}"
        assert "Interrupted" in out, out[-800:]

        manifest = manifest_of(run)
        rows = run_rows(run)
        # The stop wrote the frontier at the clock it stopped on: no gap, no overlap.
        assert max(step for step, _ in rows) == manifest["next_clock"] - 1, (
            manifest["next_clock"],
            max(step for step, _ in rows),
        )
        resume_cfg = RaceConfig(**{**cfg, "max_steps": manifest["next_clock"] + 3})
        engine = RaceEngine.resume(run, DATA_DIR, resume_cfg)
        assert engine.step_clock == manifest["next_clock"], engine.step_clock
        assert engine.run() == "MaxSteps"
        after = run_rows(run)
        duplicates = sum(c - 1 for c in Counter(after).values() if c > 1)
        assert duplicates == 0, f"{duplicates} duplicate rows after a Ctrl+C resume"
        print(f"sigint ok: stopped at {manifest['next_clock'] - 1}, resumed clean, 0 duplicates")


def main() -> None:
    check_sigkill_with_evolution_restores_the_set()
    check_flush_ahead_has_no_duplicate_rows()
    check_sigint_is_a_graceful_prefix()
    print("interrupt ok: SIGKILL restores the frontier set, CSVs rewind, Ctrl+C is a prefix")


if __name__ == "__main__":
    main()
