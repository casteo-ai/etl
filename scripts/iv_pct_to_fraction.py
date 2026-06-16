"""
One-off migration: rescale the option ``mark_iv`` column from percent to a
decimal fraction (52.95 -> 0.5295) in already-stored CSV / CSV.GZ files.

Usage
-----
    python scripts/iv_pct_to_fraction.py <file_or_dir> [<file_or_dir> ...]

Each path may be a single ``*.csv`` / ``*.csv.gz`` file or a directory (walked
recursively for both extensions). Compression is inferred from the extension,
so files are rewritten in the same format. Run this only once per file —
re-running would divide by 100 again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_COLUMN = "mark_iv"


def _iter_files(paths: list[str]):
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            yield from sorted(p.rglob("*.csv"))
            yield from sorted(p.rglob("*.csv.gz"))
        elif p.is_file():
            yield p


def migrate_file(path: Path) -> int:
    """Divide ``mark_iv`` by 100 in one file. Returns rows rescaled."""
    df = pd.read_csv(path)  # compression inferred from .gz suffix
    if _COLUMN not in df.columns:
        print(f"skip  {path} (no {_COLUMN} column)")
        return 0
    rescaled = df[_COLUMN].notna().sum()
    df[_COLUMN] = df[_COLUMN] / 100.0
    df.to_csv(path, index=False)  # same path/extension -> same compression
    print(f"ok    {path} ({rescaled:,} rows rescaled)")
    return int(rescaled)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    total = 0
    for path in _iter_files(argv):
        total += migrate_file(path)
    print(f"done: {total:,} mark_iv values rescaled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
