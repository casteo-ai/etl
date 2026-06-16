"""
OptionSurfaceCollector — periodic snapshots of the Deribit option surface.

Each :meth:`collect_once` call fetches the full live option surface across every
currency and appends the rows to daily per-currency CSVs. The snapshot/CSV
machinery (file layout, ``snapshot_time`` stamping, append semantics) lives in
:class:`~src.collectors.base.BaseSnapshotCollector`; this module only wires in
the option-surface data source.

File layout
-----------
``{data_dir}/{ASSET}_{SETTLE}/{YYYYMMDD}.csv`` — e.g. ``BTC_BTC/20260615.csv``.
See :mod:`src.collectors.base` for the full description.
"""

from __future__ import annotations

import pandas as pd

from src.collectors.base import SNAPSHOT_COLUMN, BaseSnapshotCollector, _safe_code
from src.feeds.deribit import OPTION_SURFACE_COLUMNS

__all__ = [
    "OptionSurfaceCollector",
    "DEFAULT_DATA_DIR",
    "SNAPSHOT_COLUMN",
    "SNAPSHOT_COLUMNS",
    "_safe_code",
]

# Default output directory (see project CSV conventions / data_dir layout).
DEFAULT_DATA_DIR = "/home/poucho/code/etl/data/deribit/options/volatility"

# Full, stable column order for every CSV produced by this collector.
SNAPSHOT_COLUMNS = [SNAPSHOT_COLUMN, *OPTION_SURFACE_COLUMNS]


class OptionSurfaceCollector(BaseSnapshotCollector):
    """Collect and persist Deribit option surfaces as daily per-currency CSVs."""

    BASE_COLUMNS = OPTION_SURFACE_COLUMNS
    DEFAULT_DATA_DIR = DEFAULT_DATA_DIR
    SNAPSHOT_LABEL = "option surface"

    def _fetch(self, reload: bool) -> pd.DataFrame:
        return self.client.fetch_option_surface(reload=reload)
