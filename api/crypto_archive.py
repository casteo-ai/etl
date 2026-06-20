"""FastAPI routes for serving local crypto archive CSV data."""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/crypto-archive", tags=["crypto_archive"])

_ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "data" / "crypto_archive"
_TICK_DIR = _ARCHIVE_DIR / "tick"
_MINUTE_DIR = _ARCHIVE_DIR / "minute"
_EPHEMERAL_DIR = Path(__file__).resolve().parents[1] / "data" / "ephimeral" / "crypto_archive"

_TICK_COLUMNS = ["trade_id", "price", "quantity", "timestamp", "is_buyer_maker"]
_MINUTE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "num_trades",
]
_PAIR_SYMBOL_RE = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$")
_CHUNK_SIZE = 100_000
_DEFAULT_TICK_DAYS = 30
_DEFAULT_MINUTE_DAYS = 365


def _list_archive_files(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return sorted(path.name for path in folder.glob("*.csv.gz") if path.is_file())


def _archive_file(folder: Path, symbol: str) -> Path:
    api_symbol = symbol.strip().upper()
    if not _PAIR_SYMBOL_RE.fullmatch(api_symbol):
        raise HTTPException(
            status_code=400,
            detail='Symbol must use BASE_QUOTE format, for example "BTC_USDC".',
        )

    path = folder / f"{api_symbol}.csv.gz"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No archive file found for {api_symbol}.")
    return path


def _archive_symbol(path: Path) -> str:
    return path.name.removesuffix(".csv.gz")


def _date_to_epoch(value: date, unit: str, end_of_day: bool = False) -> int:
    day_time = time.max if end_of_day else time.min
    dt = datetime.combine(value, day_time, tzinfo=timezone.utc)
    seconds = dt.timestamp()
    return int(seconds * 1000) if unit == "ms" else int(seconds)


def _resolve_timestamp_window(
    path: Path,
    columns: list[str],
    timestamp_column: str,
    timestamp_unit: str,
    start_date: Optional[date],
    end_date: Optional[date],
    days: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    if days is not None and (start_date is not None or end_date is not None):
        raise HTTPException(
            status_code=400,
            detail="Use either start_date/end_date or days, not both.",
        )
    if (start_date is None) != (end_date is None):
        raise HTTPException(
            status_code=400,
            detail="start_date and end_date must be supplied together.",
        )
    if start_date is not None and end_date is not None:
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date must be on or before end_date.")
        return (
            _date_to_epoch(start_date, timestamp_unit),
            _date_to_epoch(end_date, timestamp_unit, end_of_day=True),
        )
    if days is not None:
        latest = _latest_timestamp(path, columns, timestamp_column)
        if latest is None:
            return None, None
        span_seconds = days * 86400
        span = span_seconds * 1000 if timestamp_unit == "ms" else span_seconds
        return latest - span, latest
    return None, None


def _latest_timestamp(
    path: Path,
    columns: list[str],
    timestamp_column: str,
) -> Optional[int]:
    latest = None
    for chunk in pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        compression="gzip",
        usecols=[timestamp_column],
        chunksize=_CHUNK_SIZE,
    ):
        if not chunk.empty:
            latest = int(chunk[timestamp_column].iloc[-1])
    return latest


def _read_archive_csv(
    path: Path,
    columns: list[str],
    timestamp_column: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
    limit: int,
    offset: int,
) -> list[dict]:
    records: list[dict] = []
    skipped = 0
    for chunk in pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        compression="gzip",
        chunksize=_CHUNK_SIZE,
    ):
        if start_ts is not None:
            chunk = chunk[chunk[timestamp_column] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk[timestamp_column] <= end_ts]
        if chunk.empty:
            continue

        if skipped < offset:
            remaining_offset = offset - skipped
            if len(chunk) <= remaining_offset:
                skipped += len(chunk)
                continue
            chunk = chunk.iloc[remaining_offset:]
            skipped = offset

        needed = limit - len(records)
        chunk = chunk.head(needed)
        chunk = chunk.where(pd.notna(chunk), None)
        records.extend(chunk.to_dict(orient="records"))
        if len(records) >= limit:
            break
    return records


def _ephemeral_path(dataset: str, symbol: str) -> Path:
    today = date.today().strftime("%Y%m%d")
    suffix = uuid.uuid4().hex
    return _EPHEMERAL_DIR / f"{today}_{dataset}_{symbol}_{suffix}.csv"


def _write_archive_csv(
    path: Path,
    output_path: Path,
    columns: list[str],
    timestamp_column: str,
    start_ts: Optional[int],
    end_ts: Optional[int],
    offset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    wrote_header = False
    for chunk in pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        compression="gzip",
        chunksize=_CHUNK_SIZE,
    ):
        if start_ts is not None:
            chunk = chunk[chunk[timestamp_column] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk[timestamp_column] <= end_ts]
        if chunk.empty:
            continue

        if skipped < offset:
            remaining_offset = offset - skipped
            if len(chunk) <= remaining_offset:
                skipped += len(chunk)
                continue
            chunk = chunk.iloc[remaining_offset:]
            skipped = offset

        chunk.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True

    if not wrote_header:
        pd.DataFrame(columns=columns).to_csv(output_path, index=False)


def _effective_days(
    start_date: Optional[date],
    end_date: Optional[date],
    days: Optional[int],
    default_days: int,
) -> Optional[int]:
    if start_date is None and end_date is None and days is None:
        return default_days
    return days


@router.get("/tick/files", operation_id="list_crypto_archive_tick_files")
def list_tick_files() -> dict:
    """List available tick archive files."""
    files = _list_archive_files(_TICK_DIR)
    return {"dataset": "tick", "count": len(files), "files": files}


@router.get("/minute/files", operation_id="list_crypto_archive_minute_files")
def list_minute_files() -> dict:
    """List available minute archive files."""
    files = _list_archive_files(_MINUTE_DIR)
    return {"dataset": "minute", "count": len(files), "files": files}


@router.get("/tick", operation_id="get_crypto_archive_tick_data")
def get_tick_data(
    symbol: str = Query(..., description='Archive symbol in BASE_QUOTE format, for example "BTC_USDC".'),
    start_date: Optional[date] = Query(None, description="Inclusive UTC start date, for example 2026-01-01."),
    end_date: Optional[date] = Query(None, description="Inclusive UTC end date, for example 2026-01-31."),
    days: Optional[int] = Query(None, ge=1, description="Return the most recent N days from the end of the file. Defaults to 30 for tick data."),
    as_file: bool = Query(False, description="Write matching rows to an ephemeral CSV and return its path."),
    limit: int = Query(1_000, ge=1, le=100_000, description="Maximum rows to return."),
    offset: int = Query(0, ge=0, description="Number of matching rows to skip before returning data."),
) -> dict | str:
    """Serve paginated tick data from the local archive."""
    path = _archive_file(_TICK_DIR, symbol)
    effective_days = _effective_days(start_date, end_date, days, _DEFAULT_TICK_DAYS)
    start_ts, end_ts = _resolve_timestamp_window(
        path,
        _TICK_COLUMNS,
        "timestamp",
        "ms",
        start_date,
        end_date,
        effective_days,
    )
    archive_symbol = _archive_symbol(path)
    if as_file:
        output_path = _ephemeral_path("tick", archive_symbol)
        _write_archive_csv(
            path,
            output_path,
            _TICK_COLUMNS,
            "timestamp",
            start_ts,
            end_ts,
            offset=offset,
        )
        return str(output_path)

    records = _read_archive_csv(
        path,
        _TICK_COLUMNS,
        "timestamp",
        start_ts,
        end_ts,
        limit=limit,
        offset=offset,
    )
    return {
        "dataset": "tick",
        "symbol": archive_symbol,
        "columns": _TICK_COLUMNS,
        "timestamp_unit": "milliseconds",
        "start_date": start_date.isoformat() if start_date is not None else None,
        "end_date": end_date.isoformat() if end_date is not None else None,
        "days": effective_days,
        "limit": limit,
        "offset": offset,
        "count": len(records),
        "data": records,
    }


@router.get("/minute", operation_id="get_crypto_archive_minute_data")
def get_minute_data(
    symbol: str = Query(..., description='Archive symbol in BASE_QUOTE format, for example "ETH_USDT".'),
    start_date: Optional[date] = Query(None, description="Inclusive UTC start date, for example 2026-01-01."),
    end_date: Optional[date] = Query(None, description="Inclusive UTC end date, for example 2026-01-31."),
    days: Optional[int] = Query(None, ge=1, description="Return the most recent N days from the end of the file. Defaults to 365 for minute data."),
    as_file: bool = Query(False, description="Write matching rows to an ephemeral CSV and return its path."),
    limit: int = Query(1_000, ge=1, le=100_000, description="Maximum rows to return."),
    offset: int = Query(0, ge=0, description="Number of matching rows to skip before returning data."),
) -> dict | str:
    """Serve paginated minute data from the local archive."""
    path = _archive_file(_MINUTE_DIR, symbol)
    effective_days = _effective_days(start_date, end_date, days, _DEFAULT_MINUTE_DAYS)
    start_ts, end_ts = _resolve_timestamp_window(
        path,
        _MINUTE_COLUMNS,
        "timestamp",
        "s",
        start_date,
        end_date,
        effective_days,
    )
    archive_symbol = _archive_symbol(path)
    if as_file:
        output_path = _ephemeral_path("minute", archive_symbol)
        _write_archive_csv(
            path,
            output_path,
            _MINUTE_COLUMNS,
            "timestamp",
            start_ts,
            end_ts,
            offset=offset,
        )
        return str(output_path)

    records = _read_archive_csv(
        path,
        _MINUTE_COLUMNS,
        "timestamp",
        start_ts,
        end_ts,
        limit=limit,
        offset=offset,
    )
    return {
        "dataset": "minute",
        "symbol": archive_symbol,
        "columns": _MINUTE_COLUMNS,
        "timestamp_unit": "seconds",
        "start_date": start_date.isoformat() if start_date is not None else None,
        "end_date": end_date.isoformat() if end_date is not None else None,
        "days": effective_days,
        "limit": limit,
        "offset": offset,
        "count": len(records),
        "data": records,
    }
