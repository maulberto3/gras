"""Inheritance reads the source run's FINAL TABLE.

Every run ends with `elites.json` — `[{rank, hash, ...}]`, best first, ranked by the
risk-adjusted fitness the run printed. `RaceConfig(elite_previous_load=True,
elite_previous_dir=<run>)` seeds the next run's founder slots from exactly that list,
topology only (weights restart from the blueprint's seed). There is no exam and no
certification step: the table is the verdict.

Run with the project venv (needs the built extension and libtorch):

    source env_setup.sh
    uv run --with pytest pytest python/tests/test_inheritance.py

Every engine gets its own temp run dir, so nothing under `results/` is touched.
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


def topologies(run_dir: Path) -> dict[str, dict]:
    """hash → blueprint, for every net this run wrote."""
    out = {}
    for path in (run_dir / "nets").glob("*.json"):
        doc = json.loads(path.read_text())
        topo = doc["topology"]
        out[path.stem] = json.loads(topo) if isinstance(topo, str) else topo
    return out


def make_run(run_dir: Path, seed: int, inherit_from: Path | None = None) -> RaceEngine:
    return RaceEngine(
        BLOBS,
        RaceConfig(
            pop_size=POP,
            elite_count=1,
            max_steps=STEPS,
            log_level="none",
            elite_previous_load=inherit_from is not None,
            elite_previous_dir=inherit_from,
        ),
        seed=seed,
        run_dir=run_dir,
    )


def test_the_final_table_is_written_and_ranked() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-inherit-") as tmp:
        run_dir = Path(tmp) / "source"
        engine = make_run(run_dir, 11)
        engine.run()
        index = json.loads((run_dir / "elites.json").read_text())
        assert index, "a finished run must write its final table"
        assert index[0]["rank"] == 1, f"the table is not ranked: {index[:2]}"
        assert len(index[0]["hash"]) == 16, f"not a full hash: {index[0]['hash']}"


def test_the_table_seeds_the_next_run() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-inherit-seed-") as tmp:
        source = Path(tmp) / "source"
        source_engine = make_run(source, 11)
        source_engine.run()
        source_id = source_engine.run_id
        champion = json.loads((source / "elites.json").read_text())[0]["hash"]
        champion_topo = topologies(source)[champion]

        child_dir = Path(tmp) / "child"
        make_run(child_dir, 12, inherit_from=source).run()

        seeded = {h: t for h, t in topologies(child_dir).items() if t == champion_topo}
        assert seeded, "rank 1 of the source table did not reach the next run"
        want = f"seed:{source_id}:{champion[:8]}"
        for hash in seeded:
            born = json.loads((child_dir / "nets" / f"{hash}.json").read_text())["created_from"]
            assert born == want, f"inherited net's lineage is {born!r}, expected {want!r}"


def test_a_source_without_a_table_seeds_nothing() -> None:
    with tempfile.TemporaryDirectory(prefix="gras-inherit-none-") as tmp:
        source = Path(tmp) / "no-table"
        make_run(source, 11).run()
        (source / "elites.json").unlink()

        child_dir = Path(tmp) / "child"
        make_run(child_dir, 12, inherit_from=source).run()

        born = [
            json.loads(p.read_text())["created_from"]
            for p in (child_dir / "nets").glob("*.json")
        ]
        assert not [b for b in born if b.startswith("seed:")], (
            "a source with no elites.json must inherit nothing"
        )


if __name__ == "__main__":
    test_the_final_table_is_written_and_ranked()
    test_the_table_seeds_the_next_run()
    test_a_source_without_a_table_seeds_nothing()
    print("test_inheritance: ok (3 checks)")
