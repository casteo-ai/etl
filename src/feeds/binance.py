"""
BinanceClient — stateful CCXT-backed client for Binance market data.

Fetches klines (default 1-minute) with the full Binance extended field set:
    open, high, low, close, base volume, quote volume, number of trades,
    taker-buy base volume, taker-buy quote volume.
    Taker-sell volumes are derived (total − taker-buy).

Persistence
-----------
Results are saved to CSV under ``data_dir`` (default ``data/binance/ohlcv/``).
File names follow the pattern ``{SYMBOL}_{timeframe}.csv``
(e.g. ``BTC_USDT_1m.csv``).  On every call the client:
  1. Loads the existing CSV into the in-memory cache if the cache is cold.
  2. Fetches only the portions of the requested window that are absent from
     the cache (before or after the cached range).
  3. Appends the new rows to the CSV — or rewrites it when data is prepended
     before the existing start.

Credentials (only required for private/account endpoints):
    BINANCE_API_KEY    — Binance API key
    BINANCE_API_SECRET — Binance API secret
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

try:
    import ccxt
except ImportError as exc:
    raise ImportError("ccxt is required for the Binance client: pip install ccxt") from exc


logger = logging.getLogger(__name__)

_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_volume", "taker_buy_quote_volume", "_ignore",
]

_MAX_CANDLES = 1000          # Binance hard limit per request
_FLUSH_SIZE  = 10_000        # persist to CSV every N candles during a large fetch
_BINANCE_ARCHIVE_DAILY_URL = "https://data.binance.vision/data/spot/daily/klines"
_BINANCE_ARCHIVE_MONTHLY_URL = "https://data.binance.vision/data/spot/monthly/klines"
_ARCHIVE_TIMEOUT = 30
_MICROSECOND_EPOCH_THRESHOLD = 10_000_000_000_000

_ORDER_BOOK_COLUMNS = ["timestamp", "symbol", "side", "level", "price", "amount"]
_FLOAT_COLS = (
    "open", "high", "low", "close", "volume", "quote_volume",
    "taker_buy_volume", "taker_buy_quote_volume",
    "taker_sell_volume", "taker_sell_quote_volume",
)


def _as_utc_timestamp(t: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _to_ms(t: Union[str, datetime, int, float, pd.Timestamp]) -> int:
    """Convert a timestamp to epoch milliseconds (UTC)."""
    if isinstance(t, (int, float)):
        return int(t)
    if isinstance(t, str):
        return int(pd.Timestamp(t, tz="UTC").value // 1_000_000)
    if isinstance(t, pd.Timestamp):
        return int(t.value // 1_000_000)
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return int(t.timestamp() * 1000)
    raise TypeError(f"Cannot convert {type(t)} to epoch ms")


class BinanceClient:
    """
    Stateful Binance client for historical OHLCV and instrument metadata.

    Cache design
    ------------
    ``_cache`` maps ``(symbol, timeframe)`` → DataFrame indexed by open_time.
    ``_csv_last_ts`` maps the same key → last timestamp written to the CSV,
    used to compute the append delta without re-reading the file.

    On the first ``fetch_ohlcv`` call for a given (symbol, timeframe) the
    CSV (if it exists) is loaded into ``_cache`` so subsequent calls in the
    same session skip the disk read.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        enable_rate_limit: bool = True,
        data_dir: Union[str, Path] = "/home/poucho/code/etl/data/binance/ohlcv",
    ) -> None:
        api_key = api_key or os.getenv("BINANCE_API_KEY")
        api_secret = api_secret or os.getenv("BINANCE_API_SECRET")

        config: dict[str, Any] = {
            "enableRateLimit": enable_rate_limit,
            "options": {"defaultType": "spot"},
        }
        if api_key:
            config["apiKey"] = api_key
        if api_secret:
            config["secret"] = api_secret

        self.exchange = ccxt.binance(config)
        if testnet:
            self.exchange.set_sandbox_mode(True)
        self.testnet = testnet

        self._data_dir = Path(data_dir)
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._csv_last_ts: dict[tuple[str, str], pd.Timestamp] = {}
        # Signals that the on-disk CSV's last row must be removed before the
        # next append — it may have been an incomplete candle when saved.
        self._csv_needs_trim: dict[tuple[str, str], bool] = {}

    # ------------------------------------------------------------------
    # Instruments / assets
    # ------------------------------------------------------------------

    def list_instruments(
        self,
        quote: Optional[str] = None,
        base: Optional[str] = None,
        market_type: Optional[str] = None,
        reload: bool = False,
    ) -> pd.DataFrame:
        """
        List all active instruments on Binance spot.

        Parameters
        ----------
        quote:
            Filter by quote currency (e.g. ``"USDT"``, ``"BTC"``).
        base:
            Filter by base asset (e.g. ``"BTC"``, ``"ETH"``).
        market_type:
            Filter by CCXT market type: ``"spot"``, ``"margin"``, etc.
        reload:
            Force a fresh load of the market map.

        Returns
        -------
        DataFrame of CCXT market records (one row per instrument), sorted by
        symbol. Key fields: ``symbol``, ``base``, ``quote``, ``type``,
        ``active``, ``precision``, ``limits``. Empty DataFrame when nothing
        matches.
        """
        markets = self.exchange.load_markets(reload)
        results = [
            m for m in markets.values()
            if m.get("active")
            and (quote is None or m.get("quote", "").upper() == quote.upper())
            and (base is None or m.get("base", "").upper() == base.upper())
            and (market_type is None or m.get("type") == market_type)
        ]
        results.sort(key=lambda m: m.get("symbol") or "")
        return pd.DataFrame(results)

    def list_assets(self, reload: bool = False) -> pd.DataFrame:
        """Single-column (``asset``) DataFrame of unique active base assets on Binance spot."""
        markets = self.exchange.load_markets(reload)
        assets = sorted({m["base"] for m in markets.values() if m.get("active")})
        return pd.DataFrame(assets, columns=["asset"])

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _csv_path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "_")
        return self._data_dir / f"{safe}_{timeframe}.csv"

    def _daily_ohlcv_path(
        self,
        symbol: str,
        day: Union[str, datetime, pd.Timestamp],
        data_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        safe = symbol.replace("/", "_")
        root = Path(data_dir) if data_dir is not None else self._data_dir
        return root / safe / f"{_as_utc_timestamp(day).strftime('%Y%m%d')}.csv.gz"

    @staticmethod
    def _archive_symbol(symbol: str) -> str:
        """Binance archive symbol format: ``BTC/USDC`` -> ``BTCUSDC``."""
        return symbol.replace("/", "").replace("_", "").upper()

    def _archive_url(self, symbol: str, timeframe: str, day: pd.Timestamp) -> str:
        archive_symbol = self._archive_symbol(symbol)
        day_str = day.strftime("%Y-%m-%d")
        return (
            f"{_BINANCE_ARCHIVE_DAILY_URL}/{archive_symbol}/{timeframe}/"
            f"{archive_symbol}-{timeframe}-{day_str}.zip"
        )

    def _monthly_archive_url(self, symbol: str, timeframe: str, month: pd.Timestamp) -> str:
        archive_symbol = self._archive_symbol(symbol)
        month_str = month.strftime("%Y-%m")
        return (
            f"{_BINANCE_ARCHIVE_MONTHLY_URL}/{archive_symbol}/{timeframe}/"
            f"{archive_symbol}-{timeframe}-{month_str}.zip"
        )

    @staticmethod
    def _archive_csv_to_df(payload: bytes) -> pd.DataFrame:
        """Convert a Binance public archive CSV payload to the client OHLCV frame."""
        df = pd.read_csv(io.BytesIO(payload), header=None, names=_KLINE_COLUMNS)
        open_raw = pd.to_numeric(df["open_time"], errors="coerce")
        df = df.loc[open_raw.notna()].copy()
        if df.empty:
            empty_cols = [c for c in _KLINE_COLUMNS if c != "_ignore"]
            return pd.DataFrame(columns=empty_cols).set_index("open_time")

        df["open_time"] = open_raw.loc[df.index].astype("int64")
        close_raw = pd.to_numeric(df["close_time"], errors="coerce")
        df["close_time"] = close_raw.loc[df.index].astype("int64")

        open_unit = "us" if df["open_time"].abs().max() >= _MICROSECOND_EPOCH_THRESHOLD else "ms"
        close_unit = "us" if df["close_time"].abs().max() >= _MICROSECOND_EPOCH_THRESHOLD else "ms"
        df["open_time"] = pd.to_datetime(df["open_time"], unit=open_unit, utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit=close_unit, utc=True)

        for col in ("open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_volume", "taker_buy_quote_volume"):
            df[col] = df[col].astype(float)
        df["num_trades"] = df["num_trades"].astype(int)
        df.drop(columns=["_ignore"], inplace=True)
        df.set_index("open_time", inplace=True)
        df.sort_index(inplace=True)
        return df

    def _should_try_bulk_archive(
        self,
        timeframe: str,
        range_start: pd.Timestamp,
        range_end: pd.Timestamp,
    ) -> bool:
        """Use the public archive only when it should beat API pagination."""
        full_day = (
            range_start == range_start.floor("D")
            and range_end == range_start + pd.Timedelta(days=1)
        )
        if not full_day:
            return False

        # Daily archive files are published after the UTC day has completed.
        if range_end > pd.Timestamp.now("UTC").floor("D"):
            return False

        step_ms = self.exchange.parse_timeframe(timeframe) * 1000
        candles = max(0, (_to_ms(range_end) - _to_ms(range_start)) // step_ms)
        api_requests = (candles + _MAX_CANDLES - 1) // _MAX_CANDLES
        return api_requests > 1

    def _fetch_archive_klines(
        self,
        symbol: str,
        timeframe: str,
        range_start: pd.Timestamp,
        range_end: pd.Timestamp,
        timeout: int = _ARCHIVE_TIMEOUT,
    ) -> Optional[pd.DataFrame]:
        """Fetch one full UTC day from Binance's public ZIP archive, if present."""
        url = self._archive_url(symbol, timeframe, range_start)
        req = Request(url, headers={"User-Agent": "etl-binance-client"})
        try:
            with urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
        except HTTPError as exc:
            if exc.code == 404:
                return None
            logger.warning("Binance archive request failed for %s: HTTP %s", url, exc.code)
            return None
        except (OSError, URLError, TimeoutError) as exc:
            logger.warning("Binance archive request failed for %s: %s", url, exc)
            return None

        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not csv_names:
                    return None
                with archive.open(csv_names[0]) as fh:
                    data = self._archive_csv_to_df(fh.read())
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            logger.warning("Could not parse Binance archive %s: %s", url, exc)
            return None

        data = data.loc[(data.index >= range_start) & (data.index < range_end)].copy()
        data.attrs["source"] = "bulk-archive"
        return data

    def _fetch_monthly_archive_klines(
        self,
        symbol: str,
        timeframe: str,
        month_start: pd.Timestamp,
        month_end: pd.Timestamp,
        timeout: int = _ARCHIVE_TIMEOUT,
    ) -> tuple[Optional[pd.DataFrame], str]:
        """Fetch one monthly Binance public ZIP archive.

        Returns ``(frame, status)`` where status is ``ok``, ``missing`` for a
        404 archive, or ``error`` for transient/request/parse failures.
        """
        url = self._monthly_archive_url(symbol, timeframe, month_start)
        req = Request(url, headers={"User-Agent": "etl-binance-client"})
        try:
            with urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
        except HTTPError as exc:
            if exc.code == 404:
                return None, "missing"
            logger.warning("Binance monthly archive request failed for %s: HTTP %s", url, exc.code)
            return None, "error"
        except (OSError, URLError, TimeoutError) as exc:
            logger.warning("Binance monthly archive request failed for %s: %s", url, exc)
            return None, "error"

        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not csv_names:
                    return pd.DataFrame(), "ok"
                with archive.open(csv_names[0]) as fh:
                    data = self._archive_csv_to_df(fh.read())
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            logger.warning("Could not parse Binance monthly archive %s: %s", url, exc)
            return None, "error"

        data = data.loc[(data.index >= month_start) & (data.index < month_end)].copy()
        data.attrs["source"] = "monthly-bulk-archive"
        return data, "ok"

    def fetch_ohlcv_monthly_gzip(
        self,
        symbol: str,
        month: Union[str, datetime, pd.Timestamp],
        *,
        start: Optional[Union[str, datetime, pd.Timestamp]] = None,
        until: Optional[Union[str, datetime, pd.Timestamp]] = None,
        timeframe: str = "1m",
        data_dir: Optional[Union[str, Path]] = None,
        force: bool = False,
    ) -> dict:
        """Fetch one monthly public archive and split it into daily gzip files.

        This is an archive-only fast path. ``status='missing'`` means Binance
        does not publish that symbol/month archive, usually because the pair had
        no data yet.
        """
        month_start = _as_utc_timestamp(month).replace(day=1).floor("D")
        month_end = month_start + pd.DateOffset(months=1)
        range_start = max(_as_utc_timestamp(start).floor("D"), month_start) if start is not None else month_start
        range_end = min(_as_utc_timestamp(until), month_end) if until is not None else month_end
        if range_start >= range_end:
            return {"status": "empty", "rows": 0, "written_days": []}

        data, status = self._fetch_monthly_archive_klines(symbol, timeframe, month_start, month_end)
        if status != "ok":
            return {"status": status, "rows": 0, "written_days": []}
        if data is None or data.empty:
            return {"status": "ok", "rows": 0, "written_days": []}

        data = data.loc[(data.index >= range_start) & (data.index < range_end)]
        written_days: list[str] = []
        for day, day_frame in data.groupby(data.index.floor("D")):
            if day_frame.empty:
                continue
            path = self._daily_ohlcv_path(symbol, day, data_dir=data_dir)
            if path.exists() and not force:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            day_frame = day_frame[~day_frame.index.duplicated(keep="last")].sort_index()
            day_frame.to_csv(path, compression="gzip")
            written_days.append(pd.Timestamp(day).strftime("%Y-%m-%d"))

        return {"status": "ok", "rows": len(data), "written_days": written_days}

    def _fetch_missing_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        range_start: pd.Timestamp,
        range_end: pd.Timestamp,
        verbose: bool,
    ) -> pd.DataFrame:
        if self._should_try_bulk_archive(timeframe, range_start, range_end):
            archived = self._fetch_archive_klines(symbol, timeframe, range_start, range_end)
            if archived is not None:
                return archived

        return self._fetch_raw_klines(
            symbol=symbol,
            timeframe=timeframe,
            since_ms=_to_ms(range_start),
            until_ms=_to_ms(range_end),
            verbose=verbose,
        )

    def _load_from_csv(self, symbol: str, timeframe: str) -> None:
        """
        Read the CSV into the cache, dropping the last row.

        The last persisted candle may have been incomplete (still-open candle
        captured mid-minute).  We always exclude it from the cache so that
        ``fetch_ohlcv`` re-downloads it from Binance and gets the final values.
        The on-disk file is trimmed (``_csv_needs_trim``) before the next append
        so the stale row is replaced rather than duplicated.
        """
        path = self._csv_path(symbol, timeframe)
        if not path.exists():
            return
        key = (symbol, timeframe)
        try:
            df = pd.read_csv(path, index_col="open_time")
            df.index = pd.to_datetime(df.index, format="ISO8601", utc=True)
            if "close_time" in df.columns:
                df["close_time"] = pd.to_datetime(df["close_time"], format="ISO8601", utc=True)
            for col in _FLOAT_COLS:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            if "num_trades" in df.columns:
                df["num_trades"] = df["num_trades"].astype(int)
            df = df[~df.index.duplicated(keep="last")].sort_index()

            if df.empty:
                self._cache[key] = df
                print(f"[{symbol} {timeframe}] CSV is empty, will fetch fresh")
                return

            if len(df) == 1:
                # Single row — delete the file entirely and start fresh rather
                # than leaving a header-only file that complicates appends.
                path.unlink(missing_ok=True)
                self._csv_last_ts.pop(key, None)
                self._cache[key] = df.iloc[:0]
                print(f"[{symbol} {timeframe}] single-row CSV removed, will fetch fresh")
                return

            # Drop the last row: it may be incomplete. Mark the file for trim
            # so _append_to_csv removes that row from disk before writing new data.
            self._csv_last_ts[key] = df.index[-2]   # second-to-last becomes new horizon
            self._cache[key] = df.iloc[:-1]
            self._csv_needs_trim[key] = True
            print(
                f"[{symbol} {timeframe}] loaded {len(df) - 1:,} candles from {path} "
                f"(last row at {df.index[-1]} will be refreshed)"
            )
        except Exception as exc:
            print(f"[{symbol} {timeframe}] warning: could not load CSV ({exc}), fetching fresh")

    @staticmethod
    def _trim_last_csv_row(path: Path) -> None:
        """Remove the last data row from a CSV file in-place using binary seek."""
        if not path.exists():
            return
        with open(path, "r+b") as f:
            f.seek(0, 2)
            pos = f.tell()
            if pos == 0:
                return
            # Skip a trailing newline if present
            f.seek(pos - 1)
            if f.read(1) == b"\n":
                pos -= 1
            # Walk backward byte-by-byte to find the newline ending the previous row
            while pos > 0:
                f.seek(pos - 1)
                if f.read(1) == b"\n":
                    break
                pos -= 1
            f.truncate(pos)

    def _append_to_csv(self, symbol: str, timeframe: str, new_df: pd.DataFrame) -> None:
        """Append rows newer than the last saved timestamp to the CSV.

        On the first call after a CSV load the stale last row is trimmed from
        disk before any new data is written.
        """
        if new_df.empty:
            return
        key = (symbol, timeframe)
        path = self._csv_path(symbol, timeframe)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Remove the stale last row from disk (once per session, first append only)
        if self._csv_needs_trim.get(key):
            self._trim_last_csv_row(path)
            self._csv_needs_trim[key] = False

        last_ts = self._csv_last_ts.get(key)
        to_write = new_df if last_ts is None else new_df.loc[new_df.index > last_ts]
        if to_write.empty:
            return

        if path.exists():
            to_write.to_csv(path, mode="a", header=False)
        else:
            to_write.to_csv(path)

        self._csv_last_ts[key] = to_write.index.max()

    def _rewrite_csv(self, symbol: str, timeframe: str) -> None:
        """Rewrite the full CSV from the current cache (used when data is prepended)."""
        key = (symbol, timeframe)
        df = self._cache.get(key)
        if df is None or df.empty:
            return
        path = self._csv_path(symbol, timeframe)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        self._csv_last_ts[key] = df.index.max()
        self._csv_needs_trim[key] = False   # full rewrite produces a clean file

    # ------------------------------------------------------------------
    # Internal: paginated raw kline fetch
    # ------------------------------------------------------------------

    @staticmethod
    def _rows_to_df(rows: list[list]) -> pd.DataFrame:
        """Convert a list of raw Binance kline rows to a typed, indexed DataFrame."""
        df = pd.DataFrame(rows, columns=_KLINE_COLUMNS).drop(columns=["_ignore"])
        df["open_time"]  = pd.to_datetime(df["open_time"].astype("int64"),  unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_volume", "taker_buy_quote_volume"):
            df[col] = df[col].astype(float)
        df["num_trades"] = df["num_trades"].astype(int)
        df.set_index("open_time", inplace=True)
        df.sort_index(inplace=True)
        return df

    def _fetch_raw_klines(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int,
        verbose: bool = True,
        on_flush: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Fetch all klines in [since_ms, until_ms) from Binance, paginating
        automatically in chunks of 1 000.

        Every ``_FLUSH_SIZE`` candles the accumulated buffer is converted to a
        DataFrame and passed to ``on_flush(df)`` (if provided) so the caller
        can persist intermediate results without waiting for the full fetch.

        Parameters
        ----------
        on_flush:
            Optional callable ``(pd.DataFrame) -> None`` invoked after each
            ``_FLUSH_SIZE``-candle batch is ready.
        """
        # Ensure markets are loaded before resolving the symbol — required by
        # CCXT before calling exchange.market().
        self.exchange.load_markets()

        market = self.exchange.market(symbol)
        binance_symbol = market["id"]
        interval = self.exchange.timeframes.get(timeframe, timeframe)
        step_ms = self.exchange.parse_timeframe(timeframe) * 1000
        estimated_total = max(1, (until_ms - since_ms) // step_ms)

        buffer: list[list] = []
        chunks: list[pd.DataFrame] = []
        total_fetched = 0
        cursor = since_ms

        while cursor < until_ms:
            batch = self.exchange.publicGetKlines({
                "symbol":    binance_symbol,
                "interval":  interval,
                "startTime": cursor,
                "endTime":   until_ms - 1,   # Binance endTime is inclusive
                "limit":     _MAX_CANDLES,
            })
            if not batch:
                break

            buffer.extend(batch)
            total_fetched += len(batch)

            if verbose:
                pct = min(100.0, total_fetched / estimated_total * 100)
                print(
                    f"\r[{symbol} {timeframe}] fetching… "
                    f"{total_fetched:,} / ~{estimated_total:,} candles ({pct:.1f}%)",
                    end="", flush=True,
                )

            # Flush buffer to disk every _FLUSH_SIZE candles
            if on_flush is not None and len(buffer) >= _FLUSH_SIZE:
                chunk = self._rows_to_df(buffer)
                on_flush(chunk)
                chunks.append(chunk)
                buffer = []

            if len(batch) < _MAX_CANDLES:
                break
            cursor = int(batch[-1][0]) + step_ms

        if verbose and total_fetched > 0:
            print()   # newline after progress line

        # Process any remaining rows in the buffer
        if buffer:
            chunk = self._rows_to_df(buffer)
            if on_flush is not None:
                on_flush(chunk)
            chunks.append(chunk)

        if not chunks:
            empty_cols = [c for c in _KLINE_COLUMNS if c != "_ignore"]
            return pd.DataFrame(columns=empty_cols).set_index("open_time")

        return pd.concat(chunks).sort_index()

    # ------------------------------------------------------------------
    # Public OHLCV with incremental cache + CSV persistence
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        since: Union[str, datetime, int],
        until: Optional[Union[str, datetime, int]] = None,
        timeframe: str = "1m",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV klines with extended Binance volume fields.

        On first call for a (symbol, timeframe) pair the CSV is loaded into
        memory. Only the window portions absent from the cache are pulled from
        the API. New rows are persisted to the CSV before returning.

        Parameters
        ----------
        symbol:
            CCXT symbol, e.g. ``"BTC/USDT"``.
        since:
            Start of the window (inclusive). ISO string, ``datetime``, or epoch ms.
        until:
            End of the window (exclusive). Defaults to now (UTC).
        timeframe:
            CCXT timeframe string. Default ``"1m"``.
        verbose:
            Print fetch progress. Recommended for multi-year downloads.

        Returns
        -------
        DataFrame indexed by ``open_time`` (UTC) with columns:
            open, high, low, close
            volume                  base-asset volume
            close_time              candle close timestamp (UTC)
            quote_volume            total quote-asset turnover
            num_trades              number of individual trades
            taker_buy_volume        taker-buy base-asset volume
            taker_buy_quote_volume  taker-buy quote-asset volume
            taker_sell_volume       derived: volume − taker_buy_volume
            taker_sell_quote_volume derived: quote_volume − taker_buy_quote_volume
        """
        since_ms = _to_ms(since)
        until_ms = _to_ms(until) if until is not None else _to_ms(pd.Timestamp.now("UTC"))
        step_ms  = self.exchange.parse_timeframe(timeframe) * 1000
        key      = (symbol, timeframe)

        # Cold-start: populate cache from CSV if available
        if key not in self._cache:
            self._load_from_csv(symbol, timeframe)

        cached = self._cache.get(key)
        before = after = None

        if cached is not None and not cached.empty:
            cached_start = _to_ms(cached.index.min())
            cached_end   = _to_ms(cached.index.max())

            if since_ms < cached_start:
                if verbose:
                    print(f"[{symbol} {timeframe}] fetching data before cached range…")
                before = self._fetch_raw_klines(symbol, timeframe, since_ms, cached_start, verbose)

            if until_ms > cached_end + step_ms:
                if verbose:
                    print(f"[{symbol} {timeframe}] fetching data after cached range…")
                # Flush intermediate chunks straight to CSV as they arrive.
                after = self._fetch_raw_klines(
                    symbol, timeframe, cached_end + step_ms, until_ms, verbose,
                    on_flush=lambda df: self._append_to_csv(symbol, timeframe, df),
                )

            # Merge new data into cache
            new_parts = [p for p in (before, after) if p is not None and not p.empty]
            if new_parts:
                merged = pd.concat([cached] + new_parts)
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                self._cache[key] = merged

            # Prepend triggers a full rewrite (rare); append already flushed
            # incrementally so only a final dedup write is needed if anything
            # slipped through (e.g. the last partial chunk).
            if before is not None and not before.empty:
                self._rewrite_csv(symbol, timeframe)
            elif after is not None and not after.empty:
                # Ensure the very last chunk (< _FLUSH_SIZE) is persisted.
                self._append_to_csv(symbol, timeframe, after)

        else:
            # Nothing cached at all: fetch the full window, flushing to CSV
            # every _FLUSH_SIZE candles so a timeout doesn't lose everything.
            fetched = self._fetch_raw_klines(
                symbol, timeframe, since_ms, until_ms, verbose,
                on_flush=lambda df: self._append_to_csv(symbol, timeframe, df),
            )
            self._cache[key] = fetched
            # Persist any remaining rows not yet flushed by on_flush.
            self._append_to_csv(symbol, timeframe, fetched)

        # Return the requested slice
        full = self._cache[key]
        start_ts = pd.Timestamp(since_ms, unit="ms", tz="UTC")
        end_ts   = pd.Timestamp(until_ms, unit="ms", tz="UTC")
        return full.loc[(full.index >= start_ts) & (full.index < end_ts)].copy()

    def fetch_ohlcv_daily_gzip(
        self,
        symbol: str,
        day: Union[str, datetime, pd.Timestamp],
        until: Optional[Union[str, datetime, pd.Timestamp]] = None,
        timeframe: str = "1m",
        data_dir: Optional[Union[str, Path]] = None,
        force: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Fetch one UTC day of OHLCV and save it as ``BASE_QUOTE/YYYYMMDD.csv.gz``.

        The output root defaults to this client's ``data_dir``. Empty days are
        returned as empty DataFrames and do not create files, which covers pairs
        that were not listed yet.
        """
        day_ts = _as_utc_timestamp(day).floor("D")
        until_ts = _as_utc_timestamp(until) if until is not None else day_ts + pd.Timedelta(days=1)
        end_ts = min(day_ts + pd.Timedelta(days=1), until_ts)
        path = self._daily_ohlcv_path(symbol, day_ts, data_dir=data_dir)
        existing = pd.DataFrame()

        if path.exists() and not force:
            existing = pd.read_csv(path, index_col="open_time")
            existing.index = pd.to_datetime(existing.index, format="ISO8601", utc=True)
            if "close_time" in existing.columns:
                existing["close_time"] = pd.to_datetime(existing["close_time"], format="ISO8601", utc=True)
            existing = existing[~existing.index.duplicated(keep="last")].sort_index()
            existing = existing.loc[(existing.index >= day_ts) & (existing.index < end_ts)]

        fetches: list[pd.DataFrame] = []
        if existing.empty:
            ranges = [(day_ts, end_ts)]
        else:
            step = pd.Timedelta(seconds=self.exchange.parse_timeframe(timeframe))
            ranges = []
            cursor = day_ts
            for existing_ts in existing.index:
                if existing_ts >= end_ts:
                    break
                if existing_ts > cursor:
                    ranges.append((cursor, min(existing_ts, end_ts)))
                cursor = max(cursor, existing_ts + step)
                if cursor >= end_ts:
                    break
            if cursor < end_ts:
                ranges.append((cursor, end_ts))

        if not ranges and not force:
            existing.attrs["downloaded_rows"] = 0
            return existing

        for range_start, range_end in ranges:
            if range_start >= range_end:
                continue
            fetches.append(
                self._fetch_missing_ohlcv_range(
                    symbol=symbol,
                    timeframe=timeframe,
                    range_start=range_start,
                    range_end=range_end,
                    verbose=verbose,
                )
            )

        downloaded_rows = sum(len(part) for part in fetches)
        parts = [part for part in [existing, *fetches] if not part.empty]
        data = pd.concat(parts).sort_index() if parts else pd.DataFrame()
        if not data.empty:
            data = data[~data.index.duplicated(keep="last")].sort_index()
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, compression="gzip")
        data.attrs["downloaded_rows"] = downloaded_rows
        return data

    # ------------------------------------------------------------------
    # Level-2 order book
    # ------------------------------------------------------------------
    def fetch_order_book(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch a level-2 order-book snapshot as a tidy DataFrame.

        Parameters
        ----------
        symbol:
            CCXT symbol, e.g. ``"BTC/USDT"``.
        limit:
            Depth per side. Binance accepts 5, 10, 20, 50, 100, 500, 1000 or
            5000; other values are rounded up by the exchange. Default 100.

        Returns
        -------
        DataFrame with columns ``timestamp, symbol, side, level, price, amount``,
        one row per book entry. ``side`` is ``bid``/``ask`` and ``level`` is the
        0-based rank from the top of book (Binance spot depth carries no
        timestamp, so the capture time is used). Empty when no book is returned.
        """
        ob = self.exchange.fetch_order_book(symbol, limit)
        ts_ms = ob.get("timestamp")
        ts = (
            pd.Timestamp(ts_ms, unit="ms", tz="UTC")
            if ts_ms is not None
            else pd.Timestamp.now("UTC")
        )
        rows = []
        for side in ("bid", "ask"):
            for rank, entry in enumerate(ob.get(side + "s") or []):
                rows.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "side": side,
                    "level": rank,
                    "price": entry[0],
                    "amount": entry[1],
                })
        return pd.DataFrame(rows, columns=_ORDER_BOOK_COLUMNS)

    def clear_cache(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> None:
        """
        Evict in-memory cache entries (does not delete CSV files).

        - Both given → remove that specific (symbol, timeframe).
        - Only symbol → remove all timeframes for that symbol.
        - Neither → clear the entire cache.
        """
        if symbol and timeframe:
            self._cache.pop((symbol, timeframe), None)
            self._csv_last_ts.pop((symbol, timeframe), None)
            self._csv_needs_trim.pop((symbol, timeframe), None)
        elif symbol:
            for k in [k for k in self._cache if k[0] == symbol]:
                del self._cache[k]
                self._csv_last_ts.pop(k, None)
                self._csv_needs_trim.pop(k, None)
        else:
            self._cache.clear()
            self._csv_last_ts.clear()
            self._csv_needs_trim.clear()

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release network resources held by CCXT."""
        fn = getattr(self.exchange, "close", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    def __enter__(self) -> "BinanceClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
