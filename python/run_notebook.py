"""Execute every code cell of the Python tour and rewrite the notebook's outputs.

`make notebook` runs this after `make py`, so the stored tables come from the
extension just built. AGENTS.md rule 18 says each code cell is self-contained:
the check therefore runs them in REVERSE order, every cell in its own globals.
An exception aborts the run and leaves the notebook untouched, so a borrowed
name (or a stale cell) fails the make target instead of being committed.
"""

from __future__ import annotations

import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "python" / "notebooks" / "python_bindings_tour.ipynb"


def stream(text: str, name: str) -> dict:
    """One Jupyter stream output; `text` is stored one entry per line."""
    return {
        "output_type": "stream",
        "name": name,
        "text": text.splitlines(keepends=True),
    }


def execute(source: str) -> list[dict]:
    """Run one cell in a FRESH namespace and capture what it printed."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        exec(compile(source, "<notebook-cell>", "exec"), {"__name__": "__main__"})
    outputs = []
    if out.getvalue():
        outputs.append(stream(out.getvalue(), "stdout"))
    if err.getvalue():
        outputs.append(stream(err.getvalue(), "stderr"))
    return outputs


def main() -> int:
    os.chdir(ROOT)  # the cells read `data/…` relative to the repo root
    nb = json.loads(NOTEBOOK.read_text())
    code = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    for count, index in enumerate(code, start=1):
        nb["cells"][index]["execution_count"] = count
    for index in reversed(code):
        cell = nb["cells"][index]
        source = "".join(cell["source"])
        if not source.strip():
            cell["outputs"] = []
            continue
        print(f"cell {index}…", flush=True)
        try:
            cell["outputs"] = execute(source)
        except Exception:
            traceback.print_exc()
            print(f"cell {index} failed; {NOTEBOOK.name} was NOT written", file=sys.stderr)
            return 1
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(
        f"executed {len(code)} code cell(s) in a fresh namespace, reverse order → "
        f"{NOTEBOOK.relative_to(ROOT)}"
    )
    return 0


raise SystemExit(main())
