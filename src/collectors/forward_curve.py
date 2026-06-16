"""
ForwardCurveCollector — periodic snapshots of the Deribit forward curve.

Each :meth:`collect_once` call fetches the full live forward curve across every
currency (perpetual swap + every dated future) and appends the rows to daily
per-currency CSVs. The snapshot/CSV machinery (file layout, ``snapshot_time``
stamping, append semantics) lives in
:class:`~src.collectors.base.BaseSnapshotCollector`; this module only wires in
the forward-curve data source.

Each daily file is the term structure sampled once per snapshot: the perpetual
(``is_perpetual=True``, ``expiry`` NaT) plus every dated future ordered by
expiry, with mark/index prices, basis and annualized basis.

File layout
-----------
``{data_dir}/{ASSET}_{SETTLE}/{YYYYMMDD}.csv`` — e.g. ``BTC_BTC/20260615.csv``.
See :mod:`src.collectors.base` for the full description.
"""

from __future__ import annotations

import pandas as pd

from src.collectors.base import SNAPSHOT_COLUMN, BaseSnapshotCollector, _safe_code
from src.feeds.deribit import FORWARD_CURVE_COLUMNS

__all__ = [
    "ForwardCurveCollector",
    "DEFAULT_DATA_DIR",
    "SNAPSHOT_COLUMN",
    "SNAPSHOT_COLUMNS",
    "_safe_code",
]

# Default output directory (see project CSV conventions / data_dir layout).
DEFAULT_DATA_DIR = "/home/poucho/code/etl/data/deribit/futures/forward_curve"

# Full, stable column order for every CSV produced by this collector.
SNAPSHOT_COLUMNS = [SNAPSHOT_COLUMN, *FORWARD_CURVE_COLUMNS]


class ForwardCurveCollector(BaseSnapshotCollector):
    """Collect and persist Deribit forward curves as daily per-currency CSVs."""

    BASE_COLUMNS = FORWARD_CURVE_COLUMNS
    DEFAULT_DATA_DIR = DEFAULT_DATA_DIR
    SNAPSHOT_LABEL = "forward curve"

    def _fetch(self, reload: bool) -> pd.DataFrame:
        return self.client.fetch_forward_curve(reload=reload)
