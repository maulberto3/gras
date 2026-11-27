"""The PyPI leg's local dry run: what `dist/pypi` would upload, and whether that
version is already on PyPI.

CI publishes ONLY the sdist on a push to `main` (the wheel is a plain `linux`
build and stays a GitHub artifact), with `skip-existing: true`. PyPI has no
dry-run upload command, so this prints the decision the workflow's guard would
reach — without touching the network, the registry or git.
"""

from __future__ import annotations

import json
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDIST_DIR = ROOT / "dist" / "pypi"
NAME = "gras-engine"


def package_version() -> str:
    """The version maturin stamps on the sdist, from its single source of truth."""
    with (ROOT / "gras-py" / "Cargo.toml").open("rb") as f:
        return tomllib.load(f)["package"]["version"]


def published_versions() -> set[str] | None:
    """PyPI's release keys, or `None` when the index cannot be read."""
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{NAME}/json", timeout=10
        ) as response:
            return set(json.load(response)["releases"])
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        print(f"  could not read the PyPI index ({exc}); CI's skip-existing decides")
        return None
    except Exception as exc:  # offline, DNS, proxy…
        print(f"  could not read the PyPI index ({exc}); CI's skip-existing decides")
        return None


def main() -> int:
    version = package_version()
    print(f"PyPI plan: {NAME} {version} — sdist only; the wheel is a CI artifact")
    sdist = SDIST_DIR / f"gras_engine-{version}.tar.gz"
    if not sdist.is_file():
        print(f"  missing {sdist.relative_to(ROOT)} — `make ci` builds and stages it")
        return 1
    print(f"  {sdist.relative_to(ROOT)} — {sdist.stat().st_size / 1024:.0f} KiB")
    versions = published_versions()
    if versions is None:
        return 0
    if version in versions:
        print(f"  {NAME} {version} is already on PyPI — CI's publish step would skip it")
    else:
        print(f"  {NAME} {version} is NOT on PyPI — CI would upload this sdist")
    return 0


raise SystemExit(main())
