"""Tests for InteractiveBrokersClient (no live TWS; the IB socket is faked).

The client only talks to TWS/Gateway through ``self.ib`` (an ``ib_insync.IB``
instance), so every test injects a :class:`FakeIB` and never opens a socket.
``BarData`` objects are real ib_insync objects fed through ``util.df`` exactly as
the live path does.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from ib_insync import BarData, Contract, ContractDetails, Stock

from src.feeds.interactive_brokers import (
    _CANDLE_COLUMNS,
    _FUTURE_EXPIRY_COLUMNS,
    _OPTION_EXPIRY_COLUMNS,
    InteractiveBrokersClient,
    _parse_ib_expiry,
)


def _bar(minute: int, **over) -> BarData:
    """Build a 1-minute BarData on 2026-06-19 14:MM UTC."""
    base = dict(open=1.0, high=2.0, low=0.5, close=1.5, volume=100, average=1.4, barCount=10)
    base.update(over)
    return BarData(date=dt.datetime(2026, 6, 19, 14, minute, tzinfo=dt.timezone.utc), **base)


class FakeIB:
    """Minimal stand-in for ``ib_insync.IB``: records calls, returns canned bars."""

    def __init__(
        self,
        bars: list[BarData] | None = None,
        details: list[ContractDetails] | None = None,
        connected: bool = True,
    ):
        self._connected = connected
        self.bars = bars if bars is not None else []
        self.details = details if details is not None else []
        self.hist_calls: list[dt.datetime] = []
        self.detail_calls: list[Contract] = []
        self.connect_calls = 0

    def isConnected(self) -> bool:
        return self._connected

    def connect(self, *a, **k) -> None:
        self.connect_calls += 1
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def qualifyContracts(self, contract):
        return [contract]

    def reqHistoricalData(self, contract, *, endDateTime, durationStr, barSizeSetting, **kwargs):
        self.hist_calls.append(endDateTime)
        return list(self.bars)

    def reqContractDetails(self, contract):
        self.detail_calls.append(contract)
        return list(self.details)


def _fut_detail(month: str, *, trading_class="MBT", local=None, con_id=1, multiplier="0.1"):
    c = Contract(
        secType="FUT", symbol="MBT", lastTradeDateOrContractMonth=month,
        tradingClass=trading_class, localSymbol=local or f"MBT{month}",
        multiplier=multiplier, exchange="CME", currency="USD", conId=con_id,
    )
    return ContractDetails(contract=c)


def _opt_detail(month, strike, right, *, sec="FOP", trading_class="WM", multiplier="0.1"):
    c = Contract(
        secType=sec, symbol="MBT", lastTradeDateOrContractMonth=month,
        strike=strike, right=right, tradingClass=trading_class,
        multiplier=multiplier, exchange="CME", currency="USD",
    )
    return ContractDetails(contract=c)


@pytest.fixture
def client() -> InteractiveBrokersClient:
    c = InteractiveBrokersClient(connect=False)
    c.ib = FakeIB()
    return c


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_defaults_when_env_unset(monkeypatch):
    for var in ("IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID"):
        monkeypatch.delenv(var, raising=False)
    c = InteractiveBrokersClient(connect=False)
    assert (c.host, c.port, c.client_id, c.readonly) == ("127.0.0.1", 7497, 1, True)


def test_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("IBKR_HOST", "10.0.0.5")
    monkeypatch.setenv("IBKR_PORT", "4002")
    monkeypatch.setenv("IBKR_CLIENT_ID", "7")
    c = InteractiveBrokersClient(connect=False)
    assert (c.host, c.port, c.client_id) == ("10.0.0.5", 4002, 7)


def test_explicit_args_beat_env(monkeypatch):
    monkeypatch.setenv("IBKR_PORT", "4002")
    c = InteractiveBrokersClient(port=7496, connect=False)
    assert c.port == 7496


# ---------------------------------------------------------------------------
# Contract building
# ---------------------------------------------------------------------------
def test_build_contract_routes_each_security_type():
    c = InteractiveBrokersClient(connect=False)
    assert c.build_contract("aapl", "stock").secType == "STK"
    assert c.build_contract("EUR/USD", "fx").pair() == "EURUSD"
    assert c.build_contract("SPX", "index", exchange="CBOE").secType == "IND"
    assert c.build_contract("BTC", "crypto").secType == "CRYPTO"
    fut = c.build_contract("ES", "future", exchange="CME", last_trade_date_or_contract_month="202609")
    assert fut.secType == "FUT" and fut.lastTradeDateOrContractMonth == "202609"


def test_future_without_expiry_raises():
    c = InteractiveBrokersClient(connect=False)
    with pytest.raises(ValueError, match="last_trade_date_or_contract_month"):
        c.build_contract("ES", "future", exchange="CME")


def test_unsupported_security_type_raises():
    c = InteractiveBrokersClient(connect=False)
    with pytest.raises(ValueError, match="Unsupported"):
        c.build_contract("AAPL", "warrant")


# ---------------------------------------------------------------------------
# Bar -> DataFrame conversion
# ---------------------------------------------------------------------------
def test_bars_to_dataframe_shape_and_dtypes():
    df = InteractiveBrokersClient._bars_to_dataframe([_bar(30), _bar(31)])
    assert df.index.name == "open_time"
    assert list(df.columns) == _CANDLE_COLUMNS
    assert str(df.index.tz) == "UTC"
    assert df["bar_count"].iloc[0] == 10
    assert df["average_price"].iloc[0] == pytest.approx(1.4)


def test_bars_to_dataframe_empty_is_well_formed():
    df = InteractiveBrokersClient._bars_to_dataframe([])
    assert df.empty
    assert df.index.name == "open_time"
    assert list(df.columns) == _CANDLE_COLUMNS


# ---------------------------------------------------------------------------
# fetch_1min_candles
# ---------------------------------------------------------------------------
def test_fetch_requires_contract_or_symbol(client):
    with pytest.raises(ValueError, match="contract= or symbol="):
        client.fetch_1min_candles()


def test_fetch_single_request_when_no_start(client):
    client.ib = FakeIB(bars=[_bar(30), _bar(31)])
    out = client.fetch_1min_candles(contract=Stock("AAPL", "SMART", "USD"), qualify=False)
    assert len(out) == 2
    assert len(client.ib.hist_calls) == 1  # one request, no paging


def test_fetch_start_after_end_returns_empty(client):
    out = client.fetch_1min_candles(
        contract=Stock("AAPL", "SMART", "USD"),
        start="2026-06-20",
        end="2026-06-19",
        qualify=False,
    )
    assert out.empty
    assert list(out.columns) == _CANDLE_COLUMNS


def test_fetch_paginates_then_filters_window(client):
    # Bars span 14:30-14:31 on 2026-06-19; the window covers them, and the
    # oldest bar <= start so paging terminates after the first chunk.
    client.ib = FakeIB(bars=[_bar(30), _bar(31)])
    out = client.fetch_1min_candles(
        contract=Stock("AAPL", "SMART", "USD"),
        start="2026-06-19 14:00",
        end="2026-06-19 15:00",
        qualify=False,
    )
    assert len(out) == 2
    assert out.index.is_monotonic_increasing
    assert out.index.min() >= pd.Timestamp("2026-06-19 14:00", tz="UTC")
    assert out.index.max() <= pd.Timestamp("2026-06-19 15:00", tz="UTC")


def test_fetch_deduplicates_overlapping_bars(client):
    client.ib = FakeIB(bars=[_bar(30), _bar(30), _bar(31)])
    out = client.fetch_1min_candles(
        contract=Stock("AAPL", "SMART", "USD"),
        start="2026-06-19 14:00",
        end="2026-06-19 15:00",
        qualify=False,
    )
    assert len(out) == 2  # duplicate 14:30 collapsed


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------
def test_connect_is_skipped_when_already_connected(client):
    client.connect()
    assert client.ib.connect_calls == 0  # FakeIB starts connected


def test_disconnect_flips_state(client):
    client.disconnect()
    assert client.is_connected is False


# ---------------------------------------------------------------------------
# Expiry parsing
# ---------------------------------------------------------------------------
def test_parse_ib_expiry_full_day_and_month():
    assert _parse_ib_expiry("20260925") == pd.Timestamp("2026-09-25")
    assert _parse_ib_expiry("202609") == pd.Timestamp("2026-09-01")


def test_parse_ib_expiry_invalid_is_nat():
    assert pd.isna(_parse_ib_expiry(""))
    assert pd.isna(_parse_ib_expiry("not-a-date"))


# ---------------------------------------------------------------------------
# list_future_expiries
# ---------------------------------------------------------------------------
def test_list_future_expiries_shape_and_sort(client):
    client.ib = FakeIB(details=[
        _fut_detail("20260925", con_id=2),
        _fut_detail("20260626", con_id=1),
    ])
    df = client.list_future_expiries("mbt", "CME")
    assert list(df.columns) == _FUTURE_EXPIRY_COLUMNS
    # Sorted by expiry ascending.
    assert list(df["contract_month"]) == ["20260626", "20260925"]
    assert df["expiry"].iloc[0] == pd.Timestamp("2026-06-26")
    assert df["con_id"].iloc[0] == 1
    # The query was a FUT contract for the upper-cased symbol.
    sent = client.ib.detail_calls[0]
    assert sent.secType == "FUT" and sent.symbol == "MBT"


def test_list_future_expiries_empty_is_well_formed(client):
    client.ib = FakeIB(details=[])
    df = client.list_future_expiries("MBT", "CME")
    assert df.empty
    assert list(df.columns) == _FUTURE_EXPIRY_COLUMNS


# ---------------------------------------------------------------------------
# list_option_expiries
# ---------------------------------------------------------------------------
def test_list_option_expiries_aggregates_strikes_and_rights(client):
    client.ib = FakeIB(details=[
        _opt_detail("20260626", 60000, "C"),
        _opt_detail("20260626", 65000, "P"),
        _opt_detail("20260626", 70000, "C"),
        _opt_detail("20260925", 60000, "C"),
    ])
    df = client.list_option_expiries("MBT", "CME", security_type="fop")
    assert list(df.columns) == _OPTION_EXPIRY_COLUMNS
    assert list(df["contract_month"]) == ["20260626", "20260925"]

    front = df.iloc[0]
    assert front["num_strikes"] == 3
    assert front["min_strike"] == 60000 and front["max_strike"] == 70000
    assert front["right_types"] == "CP"
    # FOP maps to the IBKR FOP secType.
    assert client.ib.detail_calls[0].secType == "FOP"


def test_list_option_expiries_option_maps_to_opt(client):
    client.ib = FakeIB(details=[_opt_detail("20260626", 100, "C", sec="OPT", trading_class="AAPL")])
    client.list_option_expiries("AAPL", "SMART", security_type="option")
    assert client.ib.detail_calls[0].secType == "OPT"


def test_list_option_expiries_rejects_bad_security_type(client):
    with pytest.raises(ValueError, match="security_type must be one of"):
        client.list_option_expiries("MBT", "CME", security_type="swap")


def test_list_option_expiries_empty_is_well_formed(client):
    client.ib = FakeIB(details=[])
    df = client.list_option_expiries("MBT", "CME", security_type="fop")
    assert df.empty
    assert list(df.columns) == _OPTION_EXPIRY_COLUMNS
