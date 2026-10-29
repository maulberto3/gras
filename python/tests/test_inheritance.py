"""A run is inheritable only after its champion is certified.

`RaceEngine.guardrail()` is the call that records the champion in `engine.json`;
`RaceConfig(elite_previous_load=True, elite_previous_dir=<run>)` seeds the next
run's founder batch from that record. Both halves are asserted here on the
ARTIFACTS (no log parsing), so the test says nothing about formatting.

Run with the project venv (needs the built extension and libtorch):

    source env_setup.sh
    $HOME/gras/.venv/bin/python3 python/tests/test_inheritance.py

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
STEPS = 3
EXAMS = 4  # guardrail_matches: draws per elite


def header(run_dir: Path) -> dict:
    return json.loads((run_dir / "engine.json").read_text())


def topologies(run_dir: Path) -> dict[str, dict]:
    """hash → blueprint, for every net this run wrote."""
    out = {}
    for path in (run_dir / "nets").glob("*.json"):
        doc = json.loads(path.read_text())
        topo = doc["topology"]
        out[path.stem] = json.loads(topo) if isinstance(topo, str) else topo
    return out


def certified_run(tmp: Path) -> tuple[Path, str, str]:
    """Run + certify; return (dir, run_id, champion hash)."""
    run_dir = tmp / "source"
    engine = RaceEngine(
        BLOBS,
        RaceConfig(pop_size=POP, elite_count=1, max_steps=STEPS,
                   guardrail_matches=EXAMS, log_level="none"),
        seed=11,
        run_dir=run_dir,
    )
    engine.run()
    verdict = engine.guardrail()
    assert verdict is not None, "guardrail() returned None with guardrail_matches > 0"
    assert len(verdict["hash"]) == 16, f"champion hash is not a full hash: {verdict['hash']}"
    assert verdict["games"] == EXAMS, (
        f"guardrail played {verdict['games']} exams, expected {EXAMS}"
    )
    assert verdict["holdout_mean"] is not None
    return run_dir, engine.run_id, verdict["hash"]


def test_guardrail_records_the_champion() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-inherit-") as tmp:
        run_dir, _, champion = certified_run(Path(tmp))
        record = header(run_dir)["guardrail"]
        assert record is not None, "engine.json has no guardrail block after guardrail()"
        assert record["champion_hash"] == champion, "recorded champion ≠ the verdict's net"
        assert record["matches_scored"] == EXAMS, "recorded exam count ≠ guardrail_matches"
        assert record["holdout_mean"] is not None, "no holdout mean recorded"


def test_guardrail_off_records_nothing() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-inherit-off-") as tmp:
        run_dir = Path(tmp) / "off"
        engine = RaceEngine(
            BLOBS,
            RaceConfig(pop_size=POP, elite_count=1, max_steps=STEPS,
                       guardrail_matches=0, log_level="none"),
            seed=11,
            run_dir=run_dir,
        )
        engine.run()
        assert engine.guardrail() is None, "guardrail_matches=0 must skip the exam"
        assert header(run_dir)["guardrail"] is None, "a skipped exam must record nothing"


def test_certified_source_seeds_the_next_run() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-inherit-seed-") as tmp:
        source, source_id, champion = certified_run(Path(tmp))
        champion_topo = topologies(source)[champion]

        child_dir = Path(tmp) / "child"
        child = RaceEngine(
            BLOBS,
            RaceConfig(pop_size=POP, elite_count=1, max_steps=STEPS,
                       guardrail_matches=0, log_level="none",
                       elite_previous_load=True, elite_previous_dir=source),
            seed=12,
            run_dir=child_dir,
        )
        child.run()

        seeded = {h: t for h, t in topologies(child_dir).items() if t == champion_topo}
        assert seeded, "the certified champion's blueprint did not reach the next run"
        # Seed slots carry a `created_from` lineage: which run, which net. The
        # crate records the SHORT hash (8 chars), the same one the log prints.
        want = f"seed:{source_id}:{champion[:8]}"
        for hash in seeded:
            born = json.loads((child_dir / "nets" / f"{hash}.json").read_text())["created_from"]
            assert born == want, f"inherited net's lineage is {born!r}, expected {want!r}"


def test_uncertified_source_seeds_nothing() -> None:
    """The knob was always wired — the gate had nothing to pass."""
    with tempfile.TemporaryDirectory(prefix="gras-inherit-none-") as tmp:
        source = Path(tmp) / "uncertified"
        RaceEngine(
            BLOBS,
            RaceConfig(pop_size=POP, elite_count=1, max_steps=STEPS,
                       guardrail_matches=EXAMS, log_level="none"),
            seed=11,
            run_dir=source,
        ).run()  # deliberately NOT certified

        child_dir = Path(tmp) / "child"
        RaceEngine(
            BLOBS,
            RaceConfig(pop_size=POP, elite_count=1, max_steps=STEPS,
                       guardrail_matches=0, log_level="none",
                       elite_previous_load=True, elite_previous_dir=source),
            seed=12,
            run_dir=child_dir,
        ).run()

        born = [json.loads(p.read_text())["created_from"]
                for p in (child_dir / "nets").glob("*.json")]
        assert not [b for b in born if b.startswith("seed:")], (
            "an uncertified source must inherit nothing, but the new run has seeded nets"
        )


if __name__ == "__main__":
    test_guardrail_records_the_champion()
    test_guardrail_off_records_nothing()
    test_certified_source_seeds_the_next_run()
    test_uncertified_source_seeds_nothing()
    print("test_inheritance: ok (4 checks)")
