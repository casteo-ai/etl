"""Unit tests for src.collectors.orderbook (OrderBookCollector).

No network: fake Deribit/Binance clients return canned universes and books.
CSV persistence runs against ``tmp_path``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.collectors.orderbook import (
    OrderBookCollector,
    _binance_limit,
    _features_from_book,
    snapshot_columns,
)


TS = pd.Timestamp("2026-06-16 12:00:00", tz="UTC")


def _tidy_book(symbol, bids, asks, ts=TS):
    rows = []
    for lvl, (p, a) in enumerate(bids):
        rows.append({"timestamp": ts, "symbol": symbol, "side": "bid", "level": lvl, "price": p, "amount": a})
    for lvl, (p, a) in enumerate(asks):
        rows.append({"timestamp": ts, "symbol": symbol, "side": "ask", "level": lvl, "price": p, "amount": a})
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "side", "level", "price", "amount"])


class FakeDeribit:
    def __init__(self, assets):
        self._assets = assets
        self.closed = False

    def derivative_underlyings(self, reload=False):
        return pd.DataFrame(self._assets, columns=["asset"])

    def close(self):
        self.closed = True


class FakeBinance:
    def __init__(self, markets, books, fail=()):
        self._markets = markets
        self._books = books          # symbol -> (bids, asks)
        self._fail = set(fail)       # symbols that raise
        self.closed = False

        class _Ex:
            def __init__(self, mkts):
                self._m = mkts

            def load_markets(self, reload=False):
                return self._m

        self.exchange = _Ex(markets)

    def fetch_order_book(self, symbol, limit=100):
        if symbol in self._fail:
            raise RuntimeError("boom")
        bids, asks = self._books[symbol]
        return _tidy_book(symbol, bids, asks)

    def close(self):
        self.closed = True


def _markets(*symbols):
    out = {}
    for s in symbols:
        base, quote = s.split("/")
        out[s] = {"symbol": s, "base": base, "quote": quote, "active": True, "spot": True}
    return out


@pytest.fixture
def collector(tmp_path):
    markets = _markets("BTC/USDT", "ETH/USDT")
    books = {
        "BTC/USDT": ([[100.0, 2.0], [99.0, 1.0]], [[101.0, 3.0], [102.0, 1.0]]),
        "ETH/USDT": ([[10.0, 5.0]], [[10.2, 4.0]]),
    }
    deribit = FakeDeribit(["BTC", "ETH", "BTCDVOL"])  # BTCDVOL has no market
    binance = FakeBinance(markets, books)
    return OrderBookCollector(deribit_client=deribit, binance_client=binance, data_dir=tmp_path, depth=2)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_binance_limit_rounds_up(self):
        assert _binance_limit(10) == 10
        assert _binance_limit(7) == 10
        assert _binance_limit(1) == 5

    def test_snapshot_columns_shape(self):
        cols = snapshot_columns(2)
        assert cols[:3] == ["snapshot_time", "asset", "symbol"]
        assert "bid_price_1" in cols and "ask_size_2" in cols
        assert cols[-1] == "imbalance"

    def test_features_metrics(self):
        book = _tidy_book("BTC/USDT", [[100.0, 2.0], [99.0, 1.0]], [[101.0, 3.0], [102.0, 1.0]])
        row = _features_from_book(book, "BTC", "BTC/USDT", TS, depth=2)
        assert row["bid_price_1"] == 100.0 and row["ask_price_1"] == 101.0
        assert row["mid_price"] == 100.5
        assert row["spread"] == 1.0
        assert row["microprice"] == pytest.approx((100.0 * 3.0 + 101.0 * 2.0) / 5.0)  # 100.4
        assert row["bid_volume"] == 3.0 and row["ask_volume"] == 4.0
        assert row["imbalance"] == pytest.approx((3.0 - 4.0) / 7.0)

    def test_features_pads_short_side(self):
        book = _tidy_book("ETH/USDT", [[10.0, 5.0]], [[10.2, 4.0]])
        row = _features_from_book(book, "ETH", "ETH/USDT", TS, depth=2)
        assert row["bid_price_1"] == 10.0
        assert row["bid_price_2"] is None and row["ask_size_2"] is None


# ---------------------------------------------------------------------------
# universe resolution
# ---------------------------------------------------------------------------
class TestUniverse:
    def test_maps_and_skips_missing(self, collector):
        syms = collector.refresh_universe()
        assert syms == {"BTC": "BTC/USDT", "ETH": "ETH/USDT"}  # BTCDVOL skipped

    def test_quote_override(self, tmp_path):
        markets = _markets("BTC/USDC")
        binance = FakeBinance(markets, {"BTC/USDC": ([[1, 1]], [[2, 1]])})
        col = OrderBookCollector(FakeDeribit(["BTC"]), binance, data_dir=tmp_path, depth=2, quote="USDC")
        assert col.refresh_universe() == {"BTC": "BTC/USDC"}


# ---------------------------------------------------------------------------
# collect_once
# ---------------------------------------------------------------------------
class TestCollectOnce:
    def test_writes_one_gz_per_asset_per_day(self, collector, tmp_path):
        collector.collect_once(snapshot_time=TS)
        paths = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.csv.gz"))
        assert paths == ["BTC/20260616.csv.gz", "ETH/20260616.csv.gz"]

    def test_row_schema_and_first_column(self, collector, tmp_path):
        collector.collect_once(snapshot_time=TS)
        df = pd.read_csv(tmp_path / "BTC" / "20260616.csv.gz", parse_dates=["snapshot_time"])
        assert list(df.columns) == collector.columns
        assert df.columns[0] == "snapshot_time"
        assert len(df) == 1
        assert df.loc[0, "symbol"] == "BTC/USDT"
        assert df.loc[0, "mid_price"] == 100.5

    def test_appends_each_minute(self, collector, tmp_path):
        collector.collect_once(snapshot_time=TS)
        collector.collect_once(snapshot_time=TS + pd.Timedelta(minutes=1))
        df = pd.read_csv(tmp_path / "BTC" / "20260616.csv.gz", parse_dates=["snapshot_time"])
        assert len(df) == 2
        assert df["snapshot_time"].nunique() == 2  # no duplicate header row

    def test_rolls_over_by_utc_day(self, collector, tmp_path):
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-16 23:59:00", tz="UTC"))
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-17 00:00:00", tz="UTC"))
        days = sorted(p.name for p in (tmp_path / "BTC").glob("*.csv.gz"))
        assert days == ["20260616.csv.gz", "20260617.csv.gz"]

    def test_returns_combined_frame(self, collector):
        out = collector.collect_once(snapshot_time=TS)
        assert len(out) == 2
        assert set(out["asset"]) == {"BTC", "ETH"}

    def test_one_failing_symbol_does_not_abort(self, tmp_path):
        markets = _markets("BTC/USDT", "ETH/USDT")
        books = {"ETH/USDT": ([[10.0, 5.0]], [[10.2, 4.0]])}
        binance = FakeBinance(markets, books, fail=["BTC/USDT"])
        col = OrderBookCollector(FakeDeribit(["BTC", "ETH"]), binance, data_dir=tmp_path, depth=2)
        out = col.collect_once(snapshot_time=TS)
        assert list(out["asset"]) == ["ETH"]  # BTC failed, ETH captured
        assert (tmp_path / "ETH" / "20260616.csv.gz").exists()
        assert not (tmp_path / "BTC").exists()

    def test_close_propagates(self, collector):
        collector.close()
        assert collector.deribit.closed and collector.binance.closed
