"""
One-off migration: fold legacy ``{ASSET}`` order-book folders into the new
``{ASSET}_{QUOTE}`` convention used by ``OrderBookCollector``.

Background
----------
The collector used to name each asset's directory with only the base asset
(``BTC``, ``AVAX``, ...). It now writes ``{ASSET}_{QUOTE}`` (``BTC_USDT``, ...).
Every legacy download was quoted in USDT, so each legacy folder maps to
``{ASSET}_USDT``.

What it does
------------
For every legacy folder under ``data/binance/order_book`` (a name with no
underscore), each daily file is merged into the matching ``{ASSET}_USDT`` day
file: rows are concatenated, de-duplicated on ``snapshot_time`` (keeping the
existing/target row), column-unioned (so a deeper ladder is preserved), and the
target is rewritten as a single-member ``.csv.gz``. The emptied legacy folder is
removed.

Run this only AFTER stopping any live ``run_orderbook`` collector, so the live
writer does not re-create the legacy folders while the merge runs.

Usage
-----
    python scripts/merge_orderbook_folders.py [data_dir] [--quote USDT]

``data_dir`` defaults to ``data/binance/order_book``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = "/home/poucho/code/etl/data/binance/order_book"
SNAPSHOT_COLUMN = "snapshot_time"
_LEADING_COLUMNS = [SNAPSHOT_COLUMN, "asset", "symbol"]
# Per-level ladder order must match OrderBookCollector exactly so the collector
# can keep appending to a merged day file without column drift.
_LADDER_RE = re.compile(r"^(bid|ask)_(price|size)_(\d+)$")
_SIDE_ORDER = {"bid": 0, "ask": 1}
_FIELD_ORDER = {"price": 0, "size": 1}
_METRIC_COLUMNS = [
    "mid_price", "spread", "spread_bps", "microprice",
    "bid_volume", "ask_volume", "total_volume", "imbalance",
]
_DAY_RE = re.compile(r"^(\d{8})\.csv(?:\.gz)?$")


def _is_legacy_folder(path: Path) -> bool:
    """Legacy folders are a bare asset code, e.g. ``BTC`` (no ``_QUOTE``)."""
    return path.is_dir() and "_" not in path.name


def _ordered_columns(columns: list[str]) -> list[str]:
    """Canonical column order: lead, ladder (by level then side), then metrics."""
    cols = set(columns)
    ordered = [c for c in _LEADING_COLUMNS if c in cols]

    def _ladder_key(c: str) -> tuple[int, int, int]:
        side, field, level = _LADDER_RE.match(c).groups()
        return (int(level), _SIDE_ORDER[side], _FIELD_ORDER[field])

    ordered += sorted((c for c in cols if _LADDER_RE.match(c)), key=_ladder_key)
    ordered += [c for c in _METRIC_COLUMNS if c in cols]
    ordered += [c for c in columns if c not in ordered]  # any unexpected extras
    return ordered


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)  # compression inferred from suffix


def _merge_day(legacy_file: Path, target: Path) -> int:
    """Merge one legacy day file into its ``_USDT`` target. Returns rows added."""
    legacy = _read(legacy_file)
    if target.exists():
        existing = _read(target)
        before = len(existing)
        combined = pd.concat([existing, legacy], ignore_index=True)
    else:
        before = 0
        combined = legacy

    combined = combined.drop_duplicates(subset=[SNAPSHOT_COLUMN], keep="first")
    combined = combined.sort_values(SNAPSHOT_COLUMN).reset_index(drop=True)
    combined = combined.reindex(columns=_ordered_columns(list(combined.columns)))

    target.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target, index=False, compression="gzip")
    return len(combined) - before


def migrate(data_dir: Path, quote: str) -> int:
    total_added = 0
    for folder in sorted(data_dir.iterdir()):
        if not _is_legacy_folder(folder):
            continue
        target_dir = data_dir / f"{folder.name}_{quote.upper()}"
        for legacy_file in sorted(folder.iterdir()):
            match = _DAY_RE.match(legacy_file.name)
            if match is None:
                print(f"skip  {legacy_file} (not a YYYYMMDD day file)")
                continue
            target = target_dir / f"{match.group(1)}.csv.gz"
            added = _merge_day(legacy_file, target)
            total_added += added
            print(f"ok    {legacy_file} -> {target} (+{added} rows)")
        shutil.rmtree(folder)
        print(f"rm    {folder}")
    return total_added


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", default=DEFAULT_DATA_DIR)
    parser.add_argument("--quote", default="USDT")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        parser.error(f"{data_dir} is not a directory")

    total = migrate(data_dir, args.quote)
    print(f"done: {total:,} rows folded into *_{args.quote.upper()} folders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
