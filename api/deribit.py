"""FastAPI routes for serving stored Deribit historical market data.

Two datasets are served straight from the local CSV archive written by the
collectors (see ``src/collectors``):

* ``term-structure`` — historical futures forward curves, stored under
  ``data/deribit/futures/forward_curve/{ASSET}_{SETTLE}/{YYYYMMDD}.csv[.gz]``.
* ``volatility-surface`` — historical option volatility surfaces, stored under
  ``data/deribit/options/volatility/{ASSET}_{SETTLE}/{YYYYMMDD}.csv[.gz]``.

Each instrument is an ``{ASSET}_{SETTLE}`` directory (e.g. ``BTC_USDC``,
``BTC_BTC``, ``ETH_ETH``) holding one daily, gzip-compressed CSV partition per
UTC day. The first column of every file is ``snapshot_time``.

Callers select a window either with explicit ``start_date``/``end_date`` (UTC
days, inclusive) or with ``days`` (the most recent N daily partitions). The two
modes are mutually exclusive; when neither is given, the most recent
``_DEFAULT_DAYS`` (90) days are returned.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/deribit", tags=["deribit"])

_DERIBIT_DIR = Path(__file__).resolve().parents[1] / "data" / "deribit"
_TERM_STRUCTURE_DIR = _DERIBIT_DIR / "futures" / "forward_curve"
_VOLATILITY_DIR = _DERIBIT_DIR / "options" / "volatility"

# Instrument directories are ``{ASSET}_{SETTLE}`` (uppercase letters/digits).
_INSTRUMENT_RE = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$")
_PARTITION_RE = re.compile(r"^(\d{8})\.csv(?:\.gz)?$")
_CHUNK_SIZE = 50_000

# Window used when the caller specifies neither a date range nor a day count.
_DEFAULT_DAYS = 90


def _partition_date(path: Path) -> Optional[date]:
    """Parse the UTC day from a ``YYYYMMDD.csv[.gz]`` partition filename."""
    match = _PARTITION_RE.match(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _instrument_partitions(instrument_dir: Path) -> dict[date, Path]:
    """Map each UTC day to its partition file, preferring ``.csv.gz``."""
    partitions: dict[date, Path] = {}
    for path in instrument_dir.iterdir():
        if not path.is_file():
            continue
        day = _partition_date(path)
        if day is None:
            continue
        existing = partitions.get(day)
        # Prefer the gzip partition when both forms are present for a day.
        if existing is None or path.suffix == ".gz":
            partitions[day] = path
    return partitions


def _list_instruments(dataset_dir: Path) -> list[dict]:
    """Summarise every instrument directory under a dataset."""
    if not dataset_dir.exists():
        return []

    instruments: list[dict] = []
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_dir() or not _INSTRUMENT_RE.fullmatch(path.name):
            continue
        days = sorted(_instrument_partitions(path))
        if not days:
            continue
        asset, settlement_currency = path.name.split("_", 1)
        instruments.append(
            {
                "instrument": path.name,
                "asset": asset,
                "settlement_currency": settlement_currency,
                "available_days": len(days),
                "first_date": days[0].isoformat(),
                "last_date": days[-1].isoformat(),
            }
        )
    return instruments


def _resolve_instrument_dir(dataset_dir: Path, instrument: str) -> Path:
    """Validate ``instrument`` and return its directory, or raise 404/400."""
    name = instrument.strip().upper().replace("/", "_").replace("-", "_")
    if not _INSTRUMENT_RE.fullmatch(name):
        raise HTTPException(
            status_code=400,
            detail="Instrument must look like 'ASSET_SETTLE', e.g. 'BTC_USDC'.",
        )
    path = dataset_dir / name
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"No stored data for {name}.")
    return path


def _select_partitions(
    instrument_dir: Path,
    start_date: Optional[date],
    end_date: Optional[date],
    days: Optional[int],
) -> list[tuple[date, Path]]:
    """Pick the partition files matching the requested window, oldest first."""
    if days is not None and (start_date is not None or end_date is not None):
        raise HTTPException(
            status_code=400,
            detail="Use either start_date/end_date or days, not both.",
        )
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=400, detail="start_date must be on or before end_date."
        )

    partitions = sorted(_instrument_partitions(instrument_dir).items())
    if days is not None:
        return partitions[-days:]

    selected = partitions
    if start_date is not None:
        selected = [(d, p) for d, p in selected if d >= start_date]
    if end_date is not None:
        selected = [(d, p) for d, p in selected if d <= end_date]
    return selected


def _read_partition_records(
    files: list[tuple[date, Path]],
    limit: int,
    offset: int,
) -> tuple[list[dict], list[str]]:
    """Read selected partitions in date order, applying offset/limit.

    Files are read in chunks so a single large daily surface never has to be
    loaded fully into memory once ``limit`` rows have been collected.
    """
    records: list[dict] = []
    columns: list[str] = []
    skipped = 0
    for _, path in files:
        for chunk in pd.read_csv(path, chunksize=_CHUNK_SIZE):
            if not columns:
                columns = chunk.columns.tolist()
            if skipped < offset:
                remaining = offset - skipped
                if len(chunk) <= remaining:
                    skipped += len(chunk)
                    continue
                chunk = chunk.iloc[remaining:]
                skipped = offset

            needed = limit - len(records)
            chunk = chunk.head(needed)
            chunk = chunk.where(pd.notna(chunk), None)
            records.extend(chunk.to_dict(orient="records"))
            if len(records) >= limit:
                return records, columns
    return records, columns


def _serve_dataset(
    dataset: str,
    dataset_dir: Path,
    instrument: str,
    start_date: Optional[date],
    end_date: Optional[date],
    days: Optional[int],
    limit: int,
    offset: int,
) -> dict:
    """Shared handler for the two historical Deribit datasets."""
    instrument_dir = _resolve_instrument_dir(dataset_dir, instrument)
    # With no window specified, default to the most recent _DEFAULT_DAYS days.
    if start_date is None and end_date is None and days is None:
        days = _DEFAULT_DAYS
    files = _select_partitions(instrument_dir, start_date, end_date, days)
    records, columns = _read_partition_records(files, limit=limit, offset=offset)
    return {
        "exchange": "deribit",
        "dataset": dataset,
        "instrument": instrument_dir.name,
        "columns": columns,
        "start_date": start_date.isoformat() if start_date is not None else None,
        "end_date": end_date.isoformat() if end_date is not None else None,
        "days": days,
        "limit": limit,
        "offset": offset,
        "count": len(records),
        "data": records,
    }


@router.get("/term-structure/instruments", operation_id="list_deribit_term_structure_instruments")
def list_term_structure_instruments() -> dict:
    """List instruments with stored historical futures term structures."""
    instruments = _list_instruments(_TERM_STRUCTURE_DIR)
    return {
        "exchange": "deribit",
        "dataset": "term-structure",
        "count": len(instruments),
        "instruments": instruments,
    }


@router.get("/volatility-surface/instruments", operation_id="list_deribit_volatility_surface_instruments")
def list_volatility_surface_instruments() -> dict:
    """List instruments with stored historical option volatility surfaces."""
    instruments = _list_instruments(_VOLATILITY_DIR)
    return {
        "exchange": "deribit",
        "dataset": "volatility-surface",
        "count": len(instruments),
        "instruments": instruments,
    }


@router.get("/term-structure", operation_id="get_deribit_term_structure")
def get_term_structure(
    instrument: str = Query(..., description='Instrument directory, e.g. "BTC_USDC".'),
    start_date: Optional[date] = Query(None, description="Inclusive UTC start day, e.g. 2026-06-15."),
    end_date: Optional[date] = Query(None, description="Inclusive UTC end day, e.g. 2026-06-17."),
    days: Optional[int] = Query(None, ge=1, description="Return the most recent N days instead of a date range. Defaults to 90 when no date range is given."),
    limit: int = Query(10_000, ge=1, le=200_000, description="Maximum rows to return."),
    offset: int = Query(0, ge=0, description="Number of matching rows to skip before returning data."),
) -> dict:
    """Serve the stored historical futures term structure for one instrument."""
    return _serve_dataset(
        "term-structure",
        _TERM_STRUCTURE_DIR,
        instrument,
        start_date,
        end_date,
        days,
        limit,
        offset,
    )


@router.get("/volatility-surface", operation_id="get_deribit_volatility_surface")
def get_volatility_surface(
    instrument: str = Query(..., description='Instrument directory, e.g. "BTC_USDC".'),
    start_date: Optional[date] = Query(None, description="Inclusive UTC start day, e.g. 2026-06-15."),
    end_date: Optional[date] = Query(None, description="Inclusive UTC end day, e.g. 2026-06-17."),
    days: Optional[int] = Query(None, ge=1, description="Return the most recent N days instead of a date range. Defaults to 90 when no date range is given."),
    limit: int = Query(10_000, ge=1, le=200_000, description="Maximum rows to return."),
    offset: int = Query(0, ge=0, description="Number of matching rows to skip before returning data."),
) -> dict:
    """Serve the stored historical option volatility surface for one instrument."""
    return _serve_dataset(
        "volatility-surface",
        _VOLATILITY_DIR,
        instrument,
        start_date,
        end_date,
        days,
        limit,
        offset,
    )
