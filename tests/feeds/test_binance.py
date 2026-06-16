"""Unit tests for src.feeds.binance (BinanceClient + helpers).

No network access: ``ccxt.binance`` is constructed (no I/O) but its
``parse_timeframe`` is the only real method used, and the network-bound
``_fetch_raw_klines`` is monkeypatched. CSV persistence is exercised against
``tmp_path``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.feeds.binance import BinanceClient, _to_ms


STEP_MS = 60_000  # 1-minute candles
T0 = int(pd.Timestamp("2025-01-01 00:00:00", tz="UTC").value // 1_000_000)


def make_kline_rows(start_ms: int, n: int, step_ms: int = STEP_MS) -> list[list]:
    """Build ``n`` synthetic raw Binance kline rows starting at ``start_ms``."""
    rows = []
    for i in range(n):
        ot = start_ms + i * step_ms
        rows.append([
            ot,                 # open_time
            100.0 + i,          # open
            101.0 + i,          # high
            99.0 + i,           # low
            100.5 + i,          # close
            10.0 + i,           # volume
            ot + step_ms - 1,   # close_time
            1000.0 + i,         # quote_volume
            5 + i,              # num_trades
            4.0 + i,            # taker_buy_volume
            400.0 + i,          # taker_buy_quote_volume
            "0",                # _ignore
        ])
    return rows


@pytest.fixture
def client(tmp_path):
    """BinanceClient writing CSVs under an isolated tmp dir."""
    return BinanceClient(data_dir=tmp_path)


# ---------------------------------------------------------------------------
# _to_ms
# ---------------------------------------------------------------------------
class TestToMs:
    def test_int_passthrough(self):
        assert _to_ms(1234) == 1234

    def test_float_truncates(self):
        assert _to_ms(1234.9) == 1234

    def test_iso_string(self):
        expected = int(pd.Timestamp("2025-01-01", tz="UTC").value // 1_000_000)
        assert _to_ms("2025-01-01") == expected

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2025-01-01", tz="UTC")
        assert _to_ms(ts) == int(ts.value // 1_000_000)

    def test_naive_datetime_assumed_utc(self):
        dt = datetime(2025, 1, 1, 0, 0, 0)
        expected = int(pd.Timestamp("2025-01-01", tz="UTC").value // 1_000_000)
        assert _to_ms(dt) == expected

    def test_aware_datetime(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert _to_ms(dt) == int(dt.timestamp() * 1000)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            _to_ms(object())


# ---------------------------------------------------------------------------
# _rows_to_df
# ---------------------------------------------------------------------------
class TestRowsToDf:
    def test_columns_and_index(self, client):
        df = client._rows_to_df(make_kline_rows(T0, 3))
        assert "_ignore" not in df.columns
        assert df.index.name == "open_time"
        assert str(df.index.tz) == "UTC"
        assert {"open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_volume", "taker_buy_quote_volume", "close_time",
                "num_trades"} <= set(df.columns)

    def test_dtypes(self, client):
        df = client._rows_to_df(make_kline_rows(T0, 3))
        assert df["close"].dtype == float
        assert df["num_trades"].dtype == int

    def test_sorted_by_open_time(self, client):
        rows = make_kline_rows(T0, 3)
        df = client._rows_to_df(list(reversed(rows)))
        assert list(df.index) == sorted(df.index)


# ---------------------------------------------------------------------------
# CSV path / cache eviction
# ---------------------------------------------------------------------------
class TestListMethods:
    @staticmethod
    def _markets():
        return {
            "BTC/USDT": {"symbol": "BTC/USDT", "base": "BTC", "quote": "USDT",
                         "type": "spot", "active": True},
            "ETH/USDT": {"symbol": "ETH/USDT", "base": "ETH", "quote": "USDT",
                         "type": "spot", "active": True},
            "OLD/USDT": {"symbol": "OLD/USDT", "base": "OLD", "quote": "USDT",
                         "type": "spot", "active": False},
        }

    def test_list_instruments_returns_dataframe(self, client):
        client.exchange = MagicMock()
        client.exchange.load_markets.return_value = self._markets()
        df = client.list_instruments(quote="USDT")
        assert isinstance(df, pd.DataFrame)
        # Inactive instruments excluded, sorted by symbol.
        assert list(df["symbol"]) == ["BTC/USDT", "ETH/USDT"]

    def test_list_instruments_base_filter(self, client):
        client.exchange = MagicMock()
        client.exchange.load_markets.return_value = self._markets()
        df = client.list_instruments(base="BTC")
        assert list(df["symbol"]) == ["BTC/USDT"]

    def test_list_assets_returns_dataframe(self, client):
        client.exchange = MagicMock()
        client.exchange.load_markets.return_value = self._markets()
        df = client.list_assets()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["asset"]
        assert list(df["asset"]) == ["BTC", "ETH"]  # active only, sorted


class TestCsvPath:
    def test_slash_replaced(self, client):
        path = client._csv_path("BTC/USDT", "1m")
        assert path.name == "BTC_USDT_1m.csv"


class TestClearCache:
    def _seed(self, client):
        for key in (("BTC/USDT", "1m"), ("BTC/USDT", "1h"), ("ETH/USDT", "1m")):
            client._cache[key] = pd.DataFrame()
            client._csv_last_ts[key] = pd.Timestamp("2025-01-01", tz="UTC")
            client._csv_needs_trim[key] = True

    def test_clear_specific(self, client):
        self._seed(client)
        client.clear_cache("BTC/USDT", "1m")
        assert ("BTC/USDT", "1m") not in client._cache
        assert ("BTC/USDT", "1h") in client._cache

    def test_clear_by_symbol(self, client):
        self._seed(client)
        client.clear_cache("BTC/USDT")
        assert not any(k[0] == "BTC/USDT" for k in client._cache)
        assert ("ETH/USDT", "1m") in client._cache

    def test_clear_all(self, client):
        self._seed(client)
        client.clear_cache()
        assert not client._cache
        assert not client._csv_last_ts
        assert not client._csv_needs_trim


# ---------------------------------------------------------------------------
# CSV persistence round-trip
# ---------------------------------------------------------------------------
class TestCsvPersistence:
    def test_append_then_load_drops_last_row(self, client):
        df = client._rows_to_df(make_kline_rows(T0, 5))
        client._append_to_csv("BTC/USDT", "1m", df)

        path = client._csv_path("BTC/USDT", "1m")
        assert path.exists()
        assert client._csv_last_ts[("BTC/USDT", "1m")] == df.index.max()

        # Fresh client loads from disk and drops the (possibly partial) last row.
        fresh = BinanceClient(data_dir=client._data_dir)
        fresh._load_from_csv("BTC/USDT", "1m")
        cached = fresh._cache[("BTC/USDT", "1m")]
        assert len(cached) == 4
        assert fresh._csv_needs_trim[("BTC/USDT", "1m")] is True
        assert fresh._csv_last_ts[("BTC/USDT", "1m")] == df.index[-2]

    def test_append_only_writes_newer_rows(self, client):
        df = client._rows_to_df(make_kline_rows(T0, 5))
        client._append_to_csv("BTC/USDT", "1m", df)
        # Re-appending the same frame writes nothing (all <= last_ts).
        client._append_to_csv("BTC/USDT", "1m", df)
        reloaded = pd.read_csv(client._csv_path("BTC/USDT", "1m"))
        assert len(reloaded) == 5

    def test_trim_last_csv_row(self, client):
        df = client._rows_to_df(make_kline_rows(T0, 5))
        path = client._csv_path("BTC/USDT", "1m")
        df.to_csv(path)
        before = pd.read_csv(path)
        client._trim_last_csv_row(path)
        after = pd.read_csv(path)
        assert len(after) == len(before) - 1

    def test_rewrite_csv_marks_clean(self, client):
        key = ("BTC/USDT", "1m")
        client._cache[key] = client._rows_to_df(make_kline_rows(T0, 5))
        client._csv_needs_trim[key] = True
        client._rewrite_csv("BTC/USDT", "1m")
        assert client._csv_needs_trim[key] is False
        reloaded = pd.read_csv(client._csv_path("BTC/USDT", "1m"))
        assert len(reloaded) == 5

    def test_single_row_csv_removed_on_load(self, client):
        df = client._rows_to_df(make_kline_rows(T0, 1))
        path = client._csv_path("BTC/USDT", "1m")
        df.to_csv(path)
        client._load_from_csv("BTC/USDT", "1m")
        assert not path.exists()
        assert client._cache[("BTC/USDT", "1m")].empty


# ---------------------------------------------------------------------------
# fetch_ohlcv (network mocked)
# ---------------------------------------------------------------------------
class TestFetchOhlcv:
    def test_cold_start_fetches_and_persists(self, client, monkeypatch):
        client.exchange = type("E", (), {"parse_timeframe": staticmethod(lambda tf: 60)})()

        full = client._rows_to_df(make_kline_rows(T0, 5))

        def fake_fetch(symbol, timeframe, since_ms, until_ms, verbose=True, on_flush=None):
            return full.copy()

        monkeypatch.setattr(client, "_fetch_raw_klines", fake_fetch)

        out = client.fetch_ohlcv("BTC/USDT", since=T0, until=T0 + 5 * STEP_MS)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 5
        # Cache populated and CSV written.
        assert ("BTC/USDT", "1m") in client._cache
        assert client._csv_path("BTC/USDT", "1m").exists()

    def test_slice_bounds_since_inclusive_until_exclusive(self, client, monkeypatch):
        client.exchange = type("E", (), {"parse_timeframe": staticmethod(lambda tf: 60)})()
        full = client._rows_to_df(make_kline_rows(T0, 10))
        monkeypatch.setattr(
            client, "_fetch_raw_klines",
            lambda *a, **k: full.copy(),
        )
        # Request candles [T0+2min, T0+5min): expect indices 2,3,4 -> 3 rows.
        out = client.fetch_ohlcv(
            "BTC/USDT", since=T0 + 2 * STEP_MS, until=T0 + 5 * STEP_MS
        )
        assert len(out) == 3
        assert out.index.min() == pd.Timestamp(T0 + 2 * STEP_MS, unit="ms", tz="UTC")
        assert out.index.max() == pd.Timestamp(T0 + 4 * STEP_MS, unit="ms", tz="UTC")

    def test_warm_cache_no_refetch(self, client, monkeypatch):
        client.exchange = type("E", (), {"parse_timeframe": staticmethod(lambda tf: 60)})()
        key = ("BTC/USDT", "1m")
        client._cache[key] = client._rows_to_df(make_kline_rows(T0, 10))

        def boom(*a, **k):
            raise AssertionError("_fetch_raw_klines should not be called")

        monkeypatch.setattr(client, "_fetch_raw_klines", boom)
        out = client.fetch_ohlcv("BTC/USDT", since=T0, until=T0 + 5 * STEP_MS)
        assert len(out) == 5


# ---------------------------------------------------------------------------
# fetch_order_book (level 2)
# ---------------------------------------------------------------------------
class TestFetchOrderBook:
    def test_returns_tidy_l2_frame(self, client):
        client.exchange.fetch_order_book = lambda symbol, limit=100: {
            "timestamp": None,
            "bids": [[100.0, 1.0], [99.5, 2.0]],
            "asks": [[100.5, 1.5], [101.0, 3.0]],
        }
        df = client.fetch_order_book("BTC/USDT", limit=10)
        assert list(df.columns) == ["timestamp", "symbol", "side", "level", "price", "amount"]
        assert len(df) == 4
        assert set(df["side"]) == {"bid", "ask"}
        top_bid = df[(df.side == "bid") & (df.level == 0)].iloc[0]
        assert top_bid["price"] == 100.0 and top_bid["amount"] == 1.0
        assert df["timestamp"].notna().all()  # no exchange ts -> capture time

    def test_limit_forwarded(self, client):
        captured = {}

        def fake_ob(symbol, limit=100):
            captured["limit"] = limit
            return {"timestamp": 1700000000000, "bids": [], "asks": []}

        client.exchange.fetch_order_book = fake_ob
        client.fetch_order_book("BTC/USDT", limit=50)
        assert captured["limit"] == 50

    def test_uses_exchange_timestamp_when_present(self, client):
        client.exchange.fetch_order_book = lambda symbol, limit=100: {
            "timestamp": 1700000000000, "bids": [[1.0, 1.0]], "asks": [],
        }
        df = client.fetch_order_book("BTC/USDT")
        assert df["timestamp"].iloc[0] == pd.Timestamp(1700000000000, unit="ms", tz="UTC")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------
class TestContextManager:
    def test_enter_returns_self_and_exit_closes(self, client):
        closed = {"v": False}
        client.exchange = type("E", (), {"close": lambda self: closed.__setitem__("v", True)})()
        with client as ctx:
            assert ctx is client
        assert closed["v"] is True
