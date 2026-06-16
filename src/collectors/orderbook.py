"""
OrderBookCollector — per-minute Binance order-book snapshots for the Deribit
derivative universe.

Universe
--------
The set of assets is the union of every underlying that has Deribit options or
term futures (:meth:`DeribitClient.derivative_underlyings`). Each asset is mapped
to its Binance spot market (``{ASSET}/{quote}``, default quote ``USDT``); assets
without a matching Binance spot market (e.g. Deribit's ``BTCDVOL`` vol index) are
skipped.

Output
------
One gzipped CSV per asset per UTC day, appended once per snapshot:
``{data_dir}/{ASSET}/{YYYYMMDD}.csv.gz`` (default ``data/binance/order_book/``).
Appends produce a valid multi-member gzip that pandas/zcat read transparently,
and the file rolls over automatically at midnight UTC.

Each row is a single snapshot in wide form:
``snapshot_time, asset, symbol``,
the top-N ladder (``bid_price_1..N, bid_size_1..N, ask_price_1..N, ask_size_1..N``,
1 = best), and aggregate metrics
(``mid_price, spread, spread_bps, microprice, bid_volume, ask_volume,
total_volume, imbalance``). ``snapshot_time`` is the first column and the integer
index is never written, per the project's CSV conventions. Load with::

    df = pd.read_csv(path, parse_dates=["snapshot_time"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.collectors.base import _safe_code
from src.feeds.binance import BinanceClient
from src.feeds.deribit import DeribitClient

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/home/poucho/code/etl/data/binance/order_book"
DEFAULT_DEPTH = 10
DEFAULT_QUOTE = "USDT"

SNAPSHOT_COLUMN = "snapshot_time"
# Binance accepts these depth limits; we request the smallest that covers depth.
_BINANCE_LIMITS = (5, 10, 20, 50, 100, 500, 1000, 5000)

_METRIC_COLUMNS = [
    "mid_price", "spread", "spread_bps", "microprice",
    "bid_volume", "ask_volume", "total_volume", "imbalance",
]


def _ladder_columns(depth: int) -> list[str]:
    cols: list[str] = []
    for i in range(1, depth + 1):
        cols += [f"bid_price_{i}", f"bid_size_{i}", f"ask_price_{i}", f"ask_size_{i}"]
    return cols


def snapshot_columns(depth: int) -> list[str]:
    """Full, stable column order for the per-asset CSV at a given depth."""
    return [SNAPSHOT_COLUMN, "asset", "symbol", *_ladder_columns(depth), *_METRIC_COLUMNS]


def _binance_limit(depth: int) -> int:
    for lim in _BINANCE_LIMITS:
        if lim >= depth:
            return lim
    return _BINANCE_LIMITS[-1]


def _features_from_book(
    book: pd.DataFrame,
    asset: str,
    symbol: str,
    snapshot_time: pd.Timestamp,
    depth: int,
) -> dict:
    """Build one wide snapshot row (ladder + metrics) from a tidy L2 book.

    ``book`` is the long-form DataFrame returned by
    :meth:`BinanceClient.fetch_order_book` (columns side/level/price/amount).
    """
    bids = book[book["side"] == "bid"].sort_values("level")
    asks = book[book["side"] == "ask"].sort_values("level")

    row: dict = {SNAPSHOT_COLUMN: snapshot_time, "asset": asset, "symbol": symbol}
    for i in range(depth):
        b = bids.iloc[i] if i < len(bids) else None
        a = asks.iloc[i] if i < len(asks) else None
        row[f"bid_price_{i + 1}"] = float(b["price"]) if b is not None else None
        row[f"bid_size_{i + 1}"] = float(b["amount"]) if b is not None else None
        row[f"ask_price_{i + 1}"] = float(a["price"]) if a is not None else None
        row[f"ask_size_{i + 1}"] = float(a["amount"]) if a is not None else None

    if len(bids) and len(asks):
        best_bid = float(bids.iloc[0]["price"])
        best_ask = float(asks.iloc[0]["price"])
        best_bid_sz = float(bids.iloc[0]["amount"])
        best_ask_sz = float(asks.iloc[0]["amount"])
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        bid_vol = float(bids["amount"].head(depth).sum())
        ask_vol = float(asks["amount"].head(depth).sum())
        top_sz = best_bid_sz + best_ask_sz
        row.update({
            "mid_price": mid,
            "spread": spread,
            "spread_bps": (spread / mid * 1e4) if mid else None,
            # microprice: best prices weighted by the *opposite* top size.
            "microprice": (best_bid * best_ask_sz + best_ask * best_bid_sz) / top_sz if top_sz else None,
            "bid_volume": bid_vol,
            "ask_volume": ask_vol,
            "total_volume": bid_vol + ask_vol,
            "imbalance": (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) else None,
        })
    return row


class OrderBookCollector:
    """Collect Binance order books for the Deribit derivative universe.

    Parameters
    ----------
    deribit_client / binance_client:
        Built from the environment when not supplied (public endpoints only).
    data_dir:
        Output directory for the per-asset CSVs.
    depth:
        Number of book levels per side to store (default 10).
    quote:
        Binance quote currency for the spot mapping (default ``USDT``).
    """

    def __init__(
        self,
        deribit_client: Optional[DeribitClient] = None,
        binance_client: Optional[BinanceClient] = None,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        depth: int = DEFAULT_DEPTH,
        quote: str = DEFAULT_QUOTE,
    ) -> None:
        self.deribit = deribit_client if deribit_client is not None else DeribitClient()
        self.binance = binance_client if binance_client is not None else BinanceClient()
        self._data_dir = Path(data_dir)
        self.depth = int(depth)
        self.quote = quote.upper()
        self._book_limit = _binance_limit(self.depth)
        self.columns = snapshot_columns(self.depth)
        # asset -> binance symbol (e.g. "BTC" -> "BTC/USDT")
        self._symbols: dict[str, str] = {}

    @property
    def symbols(self) -> dict[str, str]:
        return dict(self._symbols)

    # ------------------------------------------------------------------
    # Universe resolution
    # ------------------------------------------------------------------
    def refresh_universe(self, reload: bool = False) -> dict[str, str]:
        """Resolve the asset universe and map each to a Binance spot symbol."""
        assets = list(self.deribit.derivative_underlyings(reload=reload)["asset"])
        markets = self.binance.exchange.load_markets(reload)
        available = {
            m["base"]: m["symbol"]
            for m in markets.values()
            if m.get("active") and m.get("spot") and (m.get("quote") or "").upper() == self.quote
        }
        resolved, skipped = {}, []
        for asset in assets:
            if asset in available:
                resolved[asset] = available[asset]
            else:
                skipped.append(asset)
        self._symbols = dict(sorted(resolved.items()))
        if skipped:
            logger.info("No Binance %s spot market for: %s (skipped)", self.quote, ", ".join(skipped))
        logger.info("Order-book universe: %d assets -> %s", len(self._symbols), list(self._symbols.values()))
        return self.symbols

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _csv_path(self, asset: str, day: str) -> Path:
        """Daily gzipped CSV for one asset: ``{data_dir}/{ASSET}/{YYYYMMDD}.csv.gz``."""
        return self._data_dir / _safe_code(asset) / f"{day}.csv.gz"

    def _append_row(self, asset: str, row: dict, day: str) -> None:
        path = self._csv_path(asset, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([row]).reindex(columns=self.columns)
        write_header = not path.exists()
        frame.to_csv(path, mode="a", header=write_header, index=False, compression="gzip")

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def collect_once(
        self,
        snapshot_time: Optional[Union[str, pd.Timestamp]] = None,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Fetch each asset's Binance order book once and append to its CSV.

        Returns the combined snapshot (one row per asset). ``reload`` re-resolves
        the asset universe / market map before collecting.
        """
        ts = pd.Timestamp(snapshot_time) if snapshot_time is not None else pd.Timestamp.now("UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        if reload or not self._symbols:
            self.refresh_universe(reload=reload)

        day = ts.strftime("%Y%m%d")
        rows = []
        for asset, symbol in self._symbols.items():
            try:
                book = self.binance.fetch_order_book(symbol, limit=self._book_limit)
                row = _features_from_book(book, asset, symbol, ts, self.depth)
                self._append_row(asset, row, day)
                rows.append(row)
            except Exception:  # noqa: BLE001 — one bad symbol must not abort the pass
                logger.exception("Order-book snapshot failed for %s (%s)", asset, symbol)

        logger.info("Order-book snapshot %s: %d/%d assets captured", ts, len(rows), len(self._symbols))
        return pd.DataFrame(rows, columns=self.columns)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release both clients' network resources."""
        for client in (self.binance, self.deribit):
            try:
                client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    def __enter__(self) -> "OrderBookCollector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
