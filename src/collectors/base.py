"""
Shared persistence for periodic Deribit snapshot collectors.

A snapshot collector fetches a wide market-data DataFrame from
:class:`~src.feeds.deribit.DeribitClient` (an option surface, a forward curve,
...), stamps every row with a single ``snapshot_time`` and appends the rows to
daily CSV files — one file per ``(asset, settlement_currency)`` pair per UTC
day.

File layout (shared by every subclass)
--------------------------------------
``{data_dir}/{ASSET}_{SETTLE}/{YYYYMMDD}.csv``

For example::

    BTC_BTC/20260615.csv
    ETH_ETH/20260615.csv
    SOL_USDC/20260615.csv

The day is derived from ``snapshot_time`` (UTC), so a long-running process rolls
onto a new file at midnight UTC with no special handling. Files are append-only,
so a full day's file is a long-format time series that loads directly with::

    df = pd.read_csv(path, parse_dates=["snapshot_time", "expiry"])

The first column is always ``snapshot_time`` and the integer index is never
written, in line with the project's CSV conventions.

Subclasses provide the data source by implementing :meth:`_fetch` and setting
:attr:`BASE_COLUMNS` (the underlying DataFrame's column order) and
:attr:`DEFAULT_DATA_DIR`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Optional, Union

import pandas as pd

from src.feeds.deribit import DeribitClient

logger = logging.getLogger(__name__)

# Column written to every snapshot row; placed first so the CSV's leading
# column is a meaningful key rather than an integer index.
SNAPSHOT_COLUMN = "snapshot_time"

# Columns used to split a snapshot into one file per currency pair.
_PAIR_KEYS = ["asset", "settlement_currency"]


def _safe_code(value: object) -> str:
    """Sanitise a currency/asset code for use in a path.

    ``None``/``NaN`` becomes ``"UNKNOWN"`` so rows are never silently dropped;
    everything else is upper-cased with path-unsafe characters replaced.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "UNKNOWN"
    code = str(value).strip().upper()
    if not code:
        return "UNKNOWN"
    for ch in ("/", "\\", " ", ":"):
        code = code.replace(ch, "_")
    return code


class BaseSnapshotCollector:
    """Base class: stamp a Deribit snapshot and append it to daily per-pair CSVs.

    Parameters
    ----------
    client:
        A :class:`~src.feeds.deribit.DeribitClient`. When ``None`` one is built
        from the environment (the snapshot endpoints are public — no
        credentials needed).
    data_dir:
        Directory the daily CSV files are written to. Defaults to the subclass's
        :attr:`DEFAULT_DATA_DIR`. Created on first write.
    """

    #: Column order of the DataFrame returned by :meth:`_fetch` (no snapshot_time).
    BASE_COLUMNS: ClassVar[list[str]] = []
    #: Default output directory; subclasses must set this.
    DEFAULT_DATA_DIR: ClassVar[str] = ""
    #: Human label used in log lines (e.g. "option surface").
    SNAPSHOT_LABEL: ClassVar[str] = "snapshot"

    def __init__(
        self,
        client: Optional[DeribitClient] = None,
        data_dir: Optional[Union[str, Path]] = None,
        compress: bool = True,
    ) -> None:
        self.client = client if client is not None else DeribitClient()
        self._data_dir = Path(data_dir if data_dir is not None else self.DEFAULT_DATA_DIR)
        # When True, files are gzip-compressed (``.csv.gz``). Appends produce a
        # valid multi-member gzip that pandas/zcat read transparently.
        self._compress = compress

    @property
    def snapshot_columns(self) -> list[str]:
        """Full, stable column order for every CSV (snapshot_time first)."""
        return [SNAPSHOT_COLUMN, *self.BASE_COLUMNS]

    # ------------------------------------------------------------------
    # Data source (subclass hook)
    # ------------------------------------------------------------------
    def _fetch(self, reload: bool) -> pd.DataFrame:
        """Return the live snapshot DataFrame. Implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _csv_path(self, asset: object, settlement_currency: object, day: str) -> Path:
        """Resolve the daily CSV path for one ``(asset, settle)`` pair.

        Each pair gets its own ``ASSET_SETTLE`` directory; the file inside is
        just the UTC day as ``YYYYMMDD.csv`` (``.csv.gz`` when compressed).
        """
        folder = f"{_safe_code(asset)}_{_safe_code(settlement_currency)}"
        ext = "csv.gz" if self._compress else "csv"
        return self._data_dir / folder / f"{day}.{ext}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _append_group(
        self,
        group: pd.DataFrame,
        asset: object,
        settlement_currency: object,
        day: str,
    ) -> Path:
        """Append one currency-pair block to its daily CSV, writing the header
        only when the file is created."""
        path = self._csv_path(asset, settlement_currency, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Reindex to the canonical column order so every append is schema-stable.
        block = group.reindex(columns=self.snapshot_columns)
        write_header = not path.exists()
        compression = "gzip" if self._compress else None
        block.to_csv(
            path, mode="a", header=write_header, index=False, compression=compression
        )
        return path

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def collect_once(
        self,
        snapshot_time: Optional[Union[str, pd.Timestamp]] = None,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Fetch one snapshot across all currencies and append it to daily CSVs.

        Parameters
        ----------
        snapshot_time:
            Timestamp stamped on every row of this snapshot. Defaults to now
            (UTC). Naive values are assumed UTC. The UTC date selects the daily
            file each pair is written to.
        reload:
            Force the underlying client to reload the CCXT market map first, so
            newly listed instruments/expiries are picked up. Cheap to skip on
            most minutes; the runner reloads on a slow cadence.

        Returns
        -------
        DataFrame of the snapshot that was written, with ``snapshot_time`` as
        the first column followed by the source columns. Empty (but
        correctly-columned) when Deribit returns nothing.
        """
        ts = pd.Timestamp(snapshot_time) if snapshot_time is not None else pd.Timestamp.now("UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        data = self._fetch(reload=reload)
        if data.empty:
            logger.warning("Deribit returned an empty %s at %s", self.SNAPSHOT_LABEL, ts)
            return pd.DataFrame(columns=self.snapshot_columns)

        snapshot = data.copy()
        snapshot.insert(0, SNAPSHOT_COLUMN, ts)

        day = ts.strftime("%Y%m%d")
        written = 0
        for (asset, settle), group in snapshot.groupby(_PAIR_KEYS, dropna=False):
            path = self._append_group(group, asset, settle, day)
            written += len(group)
            logger.debug("Wrote %d rows to %s", len(group), path)

        logger.info(
            "Snapshot %s: %d %s rows across %d currency files",
            ts,
            written,
            self.SNAPSHOT_LABEL,
            snapshot[_PAIR_KEYS].drop_duplicates().shape[0],
        )
        return snapshot

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the underlying client's network resources."""
        self.client.close()

    def __enter__(self) -> "BaseSnapshotCollector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
