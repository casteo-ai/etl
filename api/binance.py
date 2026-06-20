"""FastAPI routes for serving Binance market data."""

from __future__ import annotations

import re
import uuid
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from src.feeds.binance import BinanceClient

router = APIRouter(prefix="/binance", tags=["binance"])

_OHLCV_DIR = Path(__file__).resolve().parents[1] / "data" / "binance" / "ohlcv"
_EPHEMERAL_DIR = Path("/tmp") / "etl" / "binance" / "ohlcv"
_SAFE_SYMBOL_RE = re.compile(r"[^A-Z0-9_]+")

# Shared minute-OHLCV header, identical to the crypto-archive minute endpoint
# (api/crypto_archive.py::_MINUTE_COLUMNS). ``timestamp`` is epoch SECONDS.
OHLCV_COLUMNS = [
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


@lru_cache(maxsize=1)
def get_binance_client() -> BinanceClient:
    """Return the process-wide Binance client."""
    return BinanceClient()


def _records_from_frame(frame: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame with pandas timestamps into JSON-friendly records."""
    data = frame.reset_index()
    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = data[column].astype(str)
    data = data.where(pd.notna(data), None)
    return data.to_dict(orient="records")


def _utc_timestamp(value: str) -> pd.Timestamp:
    text = str(value).strip()
    if text.isdigit():
        return pd.Timestamp(int(text), unit="ms", tz="UTC")
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reshape the client's OHLCV frame to the shared minute schema.

    The client returns a frame indexed by ``open_time`` (UTC datetime) with an
    extra ``close_time`` column. To match the crypto-archive minute endpoint we
    emit ``timestamp`` as epoch seconds, drop ``close_time``, and fix the column
    order to ``OHLCV_COLUMNS``.
    """
    if frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    data = frame.reset_index()
    if "open_time" in data.columns:
        open_time = pd.to_datetime(data["open_time"], utc=True)
        # Whole seconds since the epoch (matches the archive's integer stamps).
        # Timedelta floor-division is independent of the datetime resolution
        # (pandas may parse as ns or us), unlike a hard-coded ns divisor.
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        data["timestamp"] = (open_time - epoch) // pd.Timedelta(seconds=1)
    return data.reindex(columns=OHLCV_COLUMNS)


def _ohlcv_records(frame: pd.DataFrame) -> list[dict]:
    data = _ohlcv_frame(frame)
    data = data.where(pd.notna(data), None)
    return data.to_dict(orient="records")


def _day_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start_day = start.floor("D")
    end_day = end.floor("D")
    if end == end_day and end > start:
        end_day -= pd.Timedelta(days=1)
    if end_day < start_day:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(start=start_day, end=end_day, freq="D", tz="UTC")


def _load_ohlcv_window(
    client: BinanceClient,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe: str,
) -> pd.DataFrame:
    if start >= end:
        raise HTTPException(status_code=400, detail="since must be before until.")
    if timeframe != "1m":
        raise HTTPException(
            status_code=400,
            detail="The partitioned Binance OHLCV dataset currently supports timeframe=1m only.",
        )

    parts: list[pd.DataFrame] = []
    for day in _day_range(start, end):
        day_end = min(day + pd.Timedelta(days=1), end)
        try:
            data = client.fetch_ohlcv_daily_gzip(
                symbol=symbol,
                day=day,
                until=day_end,
                timeframe=timeframe,
                data_dir=_OHLCV_DIR,
                force=False,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - API should surface a clean response
            raise HTTPException(status_code=502, detail=f"Failed to load Binance OHLCV for {symbol}: {exc}") from exc
        if not data.empty:
            parts.append(data)

    if not parts:
        return pd.DataFrame(columns=[column for column in OHLCV_COLUMNS if column != "timestamp"]).rename_axis("open_time")

    frame = pd.concat(parts).sort_index()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.loc[(frame.index >= start) & (frame.index < end)].copy()


def _cleanup_ephemeral_dir() -> None:
    if not _EPHEMERAL_DIR.exists():
        return
    today = date.today().strftime("%Y%m%d")
    for path in _EPHEMERAL_DIR.glob("*.csv"):
        if path.is_file() and not path.name.startswith(f"{today}_"):
            path.unlink(missing_ok=True)


def _ephemeral_path(symbol: str) -> Path:
    today = date.today().strftime("%Y%m%d")
    safe_symbol = _SAFE_SYMBOL_RE.sub("_", symbol.replace("/", "_").upper()).strip("_")
    return _EPHEMERAL_DIR / f"{today}_ohlcv_{safe_symbol}_{uuid.uuid4().hex}.csv"


def _write_ohlcv_csv(frame: pd.DataFrame, output_path: Path) -> None:
    _cleanup_ephemeral_dir()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ohlcv_frame(frame).to_csv(output_path, index=False)


@router.get("/ohlcv", operation_id="get_binance_ohlcv")
def get_ohlcv(
    symbol: str = Query(..., description='CCXT symbol, for example "BTC/USDT".'),
    since: str = Query(..., description="Inclusive window start as an ISO timestamp or epoch milliseconds."),
    until: Optional[str] = Query(None, description="Exclusive window end as an ISO timestamp or epoch milliseconds."),
    timeframe: str = Query("1m", description='CCXT timeframe, for example "1m", "5m", or "1h".'),
    as_file: bool = Query(False, description="Write matching rows to an ephemeral CSV in /tmp and return its path."),
    client: BinanceClient = Depends(get_binance_client),
) -> dict | str:
    """Serve Binance OHLCV data from daily gzip files, filling missing data first."""
    start_ts = _utc_timestamp(since)
    end_ts = _utc_timestamp(until) if until is not None else pd.Timestamp.now("UTC")
    data = _load_ohlcv_window(
        client=client,
        symbol=symbol,
        start=start_ts,
        end=end_ts,
        timeframe=timeframe,
    )
    if as_file:
        output_path = _ephemeral_path(symbol)
        _write_ohlcv_csv(data, output_path)
        return str(output_path)

    records = _ohlcv_records(data)
    return {
        "exchange": "binance",
        "symbol": symbol,
        "timeframe": timeframe,
        "since": start_ts.isoformat(),
        "until": end_ts.isoformat(),
        "columns": OHLCV_COLUMNS,
        "count": len(records),
        "ohlcv": records,
    }


@router.get("/order-book", operation_id="get_binance_order_book")
def get_order_book(
    symbol: str = Query(..., description='CCXT symbol, for example "BTC/USDT".'),
    limit: int = Query(100, ge=1, description="Depth per side requested from Binance."),
    client: BinanceClient = Depends(get_binance_client),
) -> dict:
    """Serve a Binance level-2 order-book snapshot."""
    data = client.fetch_order_book(symbol=symbol, limit=limit)
    return {
        "exchange": "binance",
        "symbol": symbol,
        "limit": limit,
        "count": len(data),
        "order_book": _records_from_frame(data),
    }
