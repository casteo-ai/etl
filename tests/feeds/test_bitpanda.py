"""Unit tests for src.feeds.bitpanda (BitpandaClient + helpers).

No network access: ``ccxt.onetrading`` is constructed (no I/O) but its market
loading and ``fetch_ohlcv`` / ``fetch_order_book`` are monkeypatched. CSV
persistence is exercised against ``tmp_path``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.feeds.bitpanda import BitpandaClient, _load_dotenv, _to_ms


STEP_MS = 60_000
T0 = int(pd.Timestamp("2026-01-01 00:00:00", tz="UTC").value // 1_000_000)


def make_ohlcv_rows(start_ms: int, n: int, step_ms: int = STEP_MS) -> list[list]:
    """Synthetic CCXT OHLCV rows: [ts, open, high, low, close, volume]."""
    rows = []
    for i in range(n):
        ts = start_ms + i * step_ms
        rows.append([ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0 + i])
    return rows


@pytest.fixture
def client(tmp_path, monkeypatch):
    """BitpandaClient with a mocked exchange and CSVs under tmp_path."""
    monkeypatch.delenv("BITPANDA_API_KEY", raising=False)
    c = BitpandaClient(data_dir=tmp_path)
    # parse_timeframe must behave like the real ccxt method.
    c.exchange.parse_timeframe = lambda tf: {"1m": 60, "5m": 300, "1h": 3600}[tf]
    c.exchange.load_markets = lambda reload=False: {}
    return c


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class TestToMs:
    def test_int_passthrough(self):
        assert _to_ms(1234) == 1234

    def test_iso_string(self):
        assert _to_ms("2026-01-01") == int(pd.Timestamp("2026-01-01", tz="UTC").value // 1_000_000)

    def test_naive_datetime_assumed_utc(self):
        dt = datetime(2026, 1, 1)
        assert _to_ms(dt) == int(pd.Timestamp("2026-01-01", tz="UTC").value // 1_000_000)


class TestLoadDotenv:
    def test_loads_without_override(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text('BITPANDA_API_KEY="abc123"\n# comment\nIGNORED line\nFOO=bar\n')
        monkeypatch.delenv("BITPANDA_API_KEY", raising=False)
        monkeypatch.delenv("FOO", raising=False)
        _load_dotenv(env)
        import os
        assert os.environ["BITPANDA_API_KEY"] == "abc123"
        assert os.environ["FOO"] == "bar"

    def test_does_not_override_existing(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("BITPANDA_API_KEY=fromfile\n")
        monkeypatch.setenv("BITPANDA_API_KEY", "fromenv")
        _load_dotenv(env)
        import os
        assert os.environ["BITPANDA_API_KEY"] == "fromenv"

    def test_missing_file_is_noop(self, tmp_path):
        _load_dotenv(tmp_path / "nope.env")  # must not raise


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------
class TestFetchOhlcv:
    def test_fetches_and_persists(self, client, tmp_path):
        rows = make_ohlcv_rows(T0, 5)
        client.exchange.fetch_ohlcv = lambda *a, **k: rows  # window covers all
        df = client.fetch_ohlcv("BTC/EUR", since=T0, until=T0 + 5 * STEP_MS, verbose=False)

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "open_time"
        assert len(df) == 5
        # persisted to CSV
        path = tmp_path / "BTC_EUR_1m.csv"
        assert path.exists()
        saved = pd.read_csv(path)
        assert saved.columns[0] == "open_time"
        assert len(saved) == 5

    def test_incremental_cache_skips_refetch(self, client, monkeypatch):
        calls = {"n": 0}

        def fake_fetch(*a, **k):
            calls["n"] += 1
            return make_ohlcv_rows(T0, 5)

        client.exchange.fetch_ohlcv = fake_fetch
        client.fetch_ohlcv("BTC/EUR", since=T0, until=T0 + 5 * STEP_MS, verbose=False)
        first = calls["n"]
        # Same window again: cache satisfies it, no further API pages.
        client.fetch_ohlcv("BTC/EUR", since=T0, until=T0 + 5 * STEP_MS, verbose=False)
        assert calls["n"] == first

    def test_gap_fill_appends_after(self, client, tmp_path):
        # First fetch 5 candles, then extend the window by 3 more.
        client.exchange.fetch_ohlcv = lambda *a, **k: make_ohlcv_rows(T0, 5)
        client.fetch_ohlcv("BTC/EUR", since=T0, until=T0 + 5 * STEP_MS, verbose=False)

        client.exchange.fetch_ohlcv = lambda *a, **k: make_ohlcv_rows(T0 + 5 * STEP_MS, 3)
        df = client.fetch_ohlcv("BTC/EUR", since=T0, until=T0 + 8 * STEP_MS, verbose=False)
        assert len(df) == 8
        saved = pd.read_csv(tmp_path / "BTC_EUR_1m.csv")
        assert len(saved) == 8  # appended, no duplicate header/rows

    def test_empty_result(self, client):
        client.exchange.fetch_ohlcv = lambda *a, **k: []
        df = client.fetch_ohlcv("BTC/EUR", since=T0, until=T0 + 5 * STEP_MS, verbose=False)
        assert df.empty

    def test_quote_volume_option_sets_ccxt_option(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BITPANDA_API_KEY", raising=False)
        c = BitpandaClient(data_dir=tmp_path, quote_volume=True)
        assert c.exchange.options.get("fetchOHLCV") == {"volume": "volume"}


# ---------------------------------------------------------------------------
# fetch_order_book
# ---------------------------------------------------------------------------
class TestFetchOrderBook:
    def test_returns_tidy_l2_frame(self, client):
        client.exchange.fetch_order_book = lambda symbol, params=None: {
            "timestamp": T0,
            "bids": [[100.0, 1.0], [99.5, 2.0]],
            "asks": [[100.5, 1.5], [101.0, 3.0]],
        }
        df = client.fetch_order_book("BTC/USD", level=2)
        assert list(df.columns) == ["timestamp", "symbol", "side", "level", "price", "amount"]
        assert len(df) == 4
        assert set(df["side"]) == {"bid", "ask"}
        top_bid = df[(df.side == "bid") & (df.level == 0)].iloc[0]
        assert top_bid["price"] == 100.0 and top_bid["amount"] == 1.0

    def test_level_param_forwarded(self, client):
        captured = {}

        def fake_ob(symbol, params=None):
            captured["params"] = params
            return {"timestamp": T0, "bids": [], "asks": []}

        client.exchange.fetch_order_book = fake_ob
        client.fetch_order_book("BTC/USD", level=3)
        assert captured["params"] == {"level": 3}

    def test_missing_timestamp_defaults_now(self, client):
        client.exchange.fetch_order_book = lambda symbol, params=None: {
            "timestamp": None, "bids": [[1.0, 1.0]], "asks": [],
        }
        df = client.fetch_order_book("BTC/USD")
        assert df["timestamp"].notna().all()


# ---------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------
class TestInstruments:
    def _markets(self):
        return {
            "BTC/EUR": {"symbol": "BTC/EUR", "base": "BTC", "quote": "EUR", "active": True},
            "ETH/EUR": {"symbol": "ETH/EUR", "base": "ETH", "quote": "EUR", "active": True},
            "BTC/USD": {"symbol": "BTC/USD", "base": "BTC", "quote": "USD", "active": True},
        }

    def test_list_instruments_filters_quote(self, client):
        client.exchange.load_markets = lambda reload=False: self._markets()
        df = client.list_instruments(quote="EUR")
        assert sorted(df["symbol"]) == ["BTC/EUR", "ETH/EUR"]

    def test_list_assets_unique_sorted(self, client):
        client.exchange.load_markets = lambda reload=False: self._markets()
        df = client.list_assets()
        assert list(df["asset"]) == ["BTC", "ETH"]
        assert list(df.columns) == ["asset"]
