"""
BitpandaClient — stateful CCXT-backed client for Bitpanda exchange data.

Bitpanda's pro/exchange venue is exposed in CCXT under the id ``onetrading``
(Bitpanda Pro was rebranded to One Trading; the REST host is
``api.onetrading.com``). This client wraps it for:

* **Historical OHLCV** (``fetch_ohlcv``) — minute candles by default, with the
  same incremental CSV cache + gap-filling design as :class:`BinanceClient`:
  on the first call for a ``(symbol, timeframe)`` pair the existing CSV is
  loaded, and only the window portions missing from the cache are pulled from
  the API. Bitpanda candle volume is the base-asset ``total_amount`` (pass
  ``quote_volume=True`` for quote turnover instead).
* **Level-2 order book** (``fetch_order_book``) — a compiled book snapshot
  (``level=2``) as a tidy DataFrame, when the venue exposes it for the symbol.

Public market data (candles, order book, instruments) does not require
credentials. A key is only needed for higher rate limits / private endpoints.

Credentials
-----------
The API key is read from the ``BITPANDA_API_KEY`` environment variable, which is
loaded from a project-root ``.env`` file if present (no extra dependency — a
minimal loader is built in). One Trading uses an API key only (no secret).

    # .env
    BITPANDA_API_KEY=your_key_here
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

try:
    import ccxt
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("ccxt is required for the Bitpanda client: pip install ccxt") from exc


# CCXT exchange id for the Bitpanda (One Trading) exchange.
_EXCHANGE_ID = "onetrading"

_OHLCV_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
_FLOAT_COLS = ("open", "high", "low", "close", "volume")

_PAGE = 1000            # candles requested per window (One Trading default cap is 1500)
_FLUSH_SIZE = 10_000    # persist to CSV every N candles during a large fetch

_ORDER_BOOK_COLUMNS = ["timestamp", "symbol", "side", "level", "price", "amount"]


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


def _load_dotenv(env_file: Optional[Union[str, Path]] = None) -> None:
    """Minimal ``.env`` loader: populate ``os.environ`` from a key=value file.

    Existing environment variables are never overwritten. Lines that are blank,
    comments (``#``), or lack ``=`` are ignored; surrounding quotes are stripped.
    Defaults to the project-root ``.env`` (three levels up from this file).
    """
    if env_file is None:
        env_file = Path(__file__).resolve().parents[2] / ".env"
    path = Path(env_file)
    if not path.is_file():
        return
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


class BitpandaClient:
    """Stateful Bitpanda (One Trading) client for OHLCV and order-book data.

    Cache design mirrors :class:`BinanceClient`: ``_cache`` maps
    ``(symbol, timeframe)`` → DataFrame indexed by ``open_time``, and the last
    persisted candle is treated as potentially incomplete (dropped on load,
    re-fetched, and trimmed from the CSV before the next append).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_rate_limit: bool = True,
        data_dir: Union[str, Path] = "/home/poucho/code/etl/data/bitpanda",
        quote_volume: bool = False,
        env_file: Optional[Union[str, Path]] = None,
    ) -> None:
        _load_dotenv(env_file)
        api_key = api_key or os.getenv("BITPANDA_API_KEY")

        config: dict[str, Any] = {
            "enableRateLimit": enable_rate_limit,
            "options": {},
        }
        if quote_volume:
            # One Trading exposes base volume as `total_amount` (CCXT default)
            # and quote turnover as `volume`.
            config["options"]["fetchOHLCV"] = {"volume": "volume"}
        if api_key:
            config["apiKey"] = api_key

        self.exchange = getattr(ccxt, _EXCHANGE_ID)(config)
        self.quote_volume = quote_volume

        self._data_dir = Path(data_dir)
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._csv_last_ts: dict[tuple[str, str], pd.Timestamp] = {}
        self._csv_needs_trim: dict[tuple[str, str], bool] = {}

    # ------------------------------------------------------------------
    # Instruments / assets
    # ------------------------------------------------------------------
    def list_instruments(
        self,
        quote: Optional[str] = None,
        base: Optional[str] = None,
        reload: bool = False,
    ) -> pd.DataFrame:
        """List active instruments, optionally filtered by quote/base currency.

        Returns a DataFrame of CCXT market records (one row per instrument),
        sorted by symbol. Empty when nothing matches.
        """
        markets = self.exchange.load_markets(reload)
        results = [
            m for m in markets.values()
            if m.get("active") is not False
            and (quote is None or (m.get("quote") or "").upper() == quote.upper())
            and (base is None or (m.get("base") or "").upper() == base.upper())
        ]
        results.sort(key=lambda m: m.get("symbol") or "")
        return pd.DataFrame(results)

    def list_assets(self, reload: bool = False) -> pd.DataFrame:
        """Single-column (``asset``) DataFrame of unique active base assets."""
        markets = self.exchange.load_markets(reload)
        assets = sorted({m["base"] for m in markets.values() if m.get("active") is not False})
        return pd.DataFrame(assets, columns=["asset"])

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------
    def _csv_path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "_")
        return self._data_dir / f"{safe}_{timeframe}.csv"

    def _load_from_csv(self, symbol: str, timeframe: str) -> None:
        """Read the CSV into the cache, dropping the (possibly incomplete) last row."""
        path = self._csv_path(symbol, timeframe)
        if not path.exists():
            return
        key = (symbol, timeframe)
        try:
            df = pd.read_csv(path, index_col="open_time")
            df.index = pd.to_datetime(df.index, format="ISO8601", utc=True)
            for col in _FLOAT_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[~df.index.duplicated(keep="last")].sort_index()

            if df.empty:
                self._cache[key] = df
                return
            if len(df) == 1:
                path.unlink(missing_ok=True)
                self._csv_last_ts.pop(key, None)
                self._cache[key] = df.iloc[:0]
                return

            self._csv_last_ts[key] = df.index[-2]
            self._cache[key] = df.iloc[:-1]
            self._csv_needs_trim[key] = True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[{symbol} {timeframe}] warning: could not load CSV ({exc}), fetching fresh")

    @staticmethod
    def _trim_last_csv_row(path: Path) -> None:
        """Remove the last data row from a CSV file in-place using a binary seek."""
        if not path.exists():
            return
        with open(path, "r+b") as f:
            f.seek(0, 2)
            pos = f.tell()
            if pos == 0:
                return
            f.seek(pos - 1)
            if f.read(1) == b"\n":
                pos -= 1
            while pos > 0:
                f.seek(pos - 1)
                if f.read(1) == b"\n":
                    break
                pos -= 1
            f.truncate(pos)

    def _append_to_csv(self, symbol: str, timeframe: str, new_df: pd.DataFrame) -> None:
        """Append rows newer than the last saved timestamp to the CSV."""
        if new_df.empty:
            return
        key = (symbol, timeframe)
        path = self._csv_path(symbol, timeframe)
        self._data_dir.mkdir(parents=True, exist_ok=True)

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
        """Rewrite the full CSV from cache (used when data is prepended)."""
        key = (symbol, timeframe)
        df = self._cache.get(key)
        if df is None or df.empty:
            return
        path = self._csv_path(symbol, timeframe)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        self._csv_last_ts[key] = df.index.max()
        self._csv_needs_trim[key] = False

    # ------------------------------------------------------------------
    # Internal: paginated raw OHLCV fetch
    # ------------------------------------------------------------------
    def _rows_to_df(self, rows: list[list]) -> pd.DataFrame:
        """Convert raw CCXT OHLCV rows ([ts,o,h,l,c,v]) to a typed, indexed frame."""
        df = pd.DataFrame(rows, columns=_OHLCV_COLUMNS)
        df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.set_index("open_time", inplace=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def _fetch_raw_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int,
        verbose: bool = True,
        on_flush: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Fetch all candles in [since_ms, until_ms) from One Trading.

        One Trading returns candles for a ``[from, to]`` window of
        ``limit * timeframe`` regardless of how many actually traded, so the
        cursor advances by the full window each step — robust to sparse,
        low-liquidity markets where many minutes have no candle.
        """
        self.exchange.load_markets()
        step_ms = self.exchange.parse_timeframe(timeframe) * 1000
        window_ms = _PAGE * step_ms
        estimated_total = max(1, (until_ms - since_ms) // step_ms)

        buffer: list[list] = []
        chunks: list[pd.DataFrame] = []
        total_fetched = 0
        cursor = since_ms

        while cursor < until_ms:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=_PAGE)
            batch = [r for r in (batch or []) if since_ms <= int(r[0]) < until_ms]
            if batch:
                buffer.extend(batch)
                total_fetched += len(batch)
                if verbose:
                    pct = min(100.0, total_fetched / estimated_total * 100)
                    print(
                        f"\r[{symbol} {timeframe}] fetching… {total_fetched:,} candles ({pct:.1f}%)",
                        end="", flush=True,
                    )
                if on_flush is not None and len(buffer) >= _FLUSH_SIZE:
                    chunk = self._rows_to_df(buffer)
                    on_flush(chunk)
                    chunks.append(chunk)
                    buffer = []
            cursor += window_ms

        if verbose and total_fetched > 0:
            print()

        if buffer:
            chunk = self._rows_to_df(buffer)
            if on_flush is not None:
                on_flush(chunk)
            chunks.append(chunk)

        if not chunks:
            return pd.DataFrame(columns=_OHLCV_COLUMNS).set_index("open_time")
        return pd.concat(chunks)[lambda d: ~d.index.duplicated(keep="last")].sort_index()

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
        """Fetch historical OHLCV candles with an incremental CSV cache.

        Only the window portions absent from the cache are pulled from the API;
        new rows are persisted to ``{data_dir}/{SYMBOL}_{timeframe}.csv`` before
        returning.

        Parameters
        ----------
        symbol:
            CCXT symbol, e.g. ``"BTC/EUR"`` or ``"BTC/USD"``.
        since:
            Start of the window (inclusive). ISO string, ``datetime``, or epoch ms.
        until:
            End of the window (exclusive). Defaults to now (UTC).
        timeframe:
            CCXT timeframe (``1m``, ``5m``, ``15m``, ``30m``, ``1h``, ``4h``,
            ``1d``, ``1w``, ``1M``). Default ``1m``.

        Returns
        -------
        DataFrame indexed by ``open_time`` (UTC) with columns
        ``open, high, low, close, volume`` (base-asset volume unless the client
        was built with ``quote_volume=True``).
        """
        since_ms = _to_ms(since)
        until_ms = _to_ms(until) if until is not None else _to_ms(pd.Timestamp.now("UTC"))
        step_ms = self.exchange.parse_timeframe(timeframe) * 1000
        key = (symbol, timeframe)

        if key not in self._cache:
            self._load_from_csv(symbol, timeframe)

        cached = self._cache.get(key)
        before = after = None

        if cached is not None and not cached.empty:
            cached_start = _to_ms(cached.index.min())
            cached_end = _to_ms(cached.index.max())

            if since_ms < cached_start:
                before = self._fetch_raw_ohlcv(symbol, timeframe, since_ms, cached_start, verbose)
            if until_ms > cached_end + step_ms:
                after = self._fetch_raw_ohlcv(
                    symbol, timeframe, cached_end + step_ms, until_ms, verbose,
                    on_flush=lambda df: self._append_to_csv(symbol, timeframe, df),
                )

            new_parts = [p for p in (before, after) if p is not None and not p.empty]
            if new_parts:
                merged = pd.concat([cached] + new_parts)
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                self._cache[key] = merged

            if before is not None and not before.empty:
                self._rewrite_csv(symbol, timeframe)
            elif after is not None and not after.empty:
                self._append_to_csv(symbol, timeframe, after)
        else:
            fetched = self._fetch_raw_ohlcv(
                symbol, timeframe, since_ms, until_ms, verbose,
                on_flush=lambda df: self._append_to_csv(symbol, timeframe, df),
            )
            self._cache[key] = fetched
            self._append_to_csv(symbol, timeframe, fetched)

        full = self._cache[key]
        if full.empty:
            return full.copy()
        start_ts = pd.Timestamp(since_ms, unit="ms", tz="UTC")
        end_ts = pd.Timestamp(until_ms, unit="ms", tz="UTC")
        return full.loc[(full.index >= start_ts) & (full.index < end_ts)].copy()

    # ------------------------------------------------------------------
    # Level-2 order book
    # ------------------------------------------------------------------
    def fetch_order_book(self, symbol: str, level: int = 2) -> pd.DataFrame:
        """Fetch a level-2 order-book snapshot as a tidy DataFrame.

        Parameters
        ----------
        symbol:
            CCXT symbol, e.g. ``"BTC/USD"``.
        level:
            One Trading book level — 1 (best bid/ask), 2 (compiled book to
            market precision, default), or 3 (full book).

        Returns
        -------
        DataFrame with columns ``timestamp, symbol, side, level, price, amount``,
        one row per book entry. ``side`` is ``bid``/``ask`` and ``level`` is the
        0-based rank from the top of book. Empty when the venue returns no book.
        """
        ob = self.exchange.fetch_order_book(symbol, params={"level": level})
        ts_ms = ob.get("timestamp")
        ts = (
            pd.Timestamp(ts_ms, unit="ms", tz="UTC")
            if ts_ms is not None
            else pd.Timestamp.now("UTC")
        )
        rows = []
        for side in ("bid", "ask"):
            entries = ob.get(side + "s") or []
            for rank, entry in enumerate(entries):
                price, amount = entry[0], entry[1]
                rows.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "side": side,
                    "level": rank,
                    "price": price,
                    "amount": amount,
                })
        return pd.DataFrame(rows, columns=_ORDER_BOOK_COLUMNS)

    # ------------------------------------------------------------------
    def clear_cache(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> None:
        """Evict in-memory cache entries (does not delete CSV files)."""
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

    def close(self) -> None:
        """Release network resources held by CCXT."""
        fn = getattr(self.exchange, "close", None)
        if callable(fn):
            try:
                fn()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    def __enter__(self) -> "BitpandaClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
