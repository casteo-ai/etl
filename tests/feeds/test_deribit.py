"""Unit tests for src.feeds.deribit (DeribitClient + helpers).

All ccxt network access is mocked. The real ``ccxt.deribit`` constructor is
used (it performs no I/O), but ``load_markets`` and the ``publicGet*`` calls
are replaced with stubs returning canned data.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.feeds.deribit import (
    DeribitClient,
    _as_bool,
    _to_float,
    _to_fraction,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
class TestAsBool:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " on "])
    def test_truthy(self, value):
        assert _as_bool(value) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", None, "maybe"])
    def test_falsy(self, value):
        assert _as_bool(value) is False


class TestToFloat:
    def test_none_returns_none(self):
        assert _to_float(None) is None

    def test_numeric_string(self):
        assert _to_float("1.5") == 1.5

    def test_int_to_float(self):
        result = _to_float(3)
        assert result == 3.0 and isinstance(result, float)

    def test_invalid_returns_none(self):
        assert _to_float("not-a-number") is None
        assert _to_float([1, 2]) is None


class TestToFraction:
    def test_percent_to_fraction(self):
        assert _to_fraction(37.3) == pytest.approx(0.373)
        assert _to_fraction("52.95") == pytest.approx(0.5295)

    def test_none_returns_none(self):
        assert _to_fraction(None) is None
        assert _to_fraction("not-a-number") is None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def client(monkeypatch):
    """Unauthenticated DeribitClient with no env credentials leaking in."""
    monkeypatch.delenv("DERIBIT_API_KEY", raising=False)
    monkeypatch.delenv("DERIBIT_API_SECRET", raising=False)
    monkeypatch.delenv("DERIBIT_TESTNET", raising=False)
    return DeribitClient()


def _markets():
    """A small canned CCXT market map keyed by symbol."""
    return {
        "ALPHA/EURR": {
            "symbol": "ALPHA/EURR", "id": "ALPHA_EURR", "type": "spot",
            "base": "ALPHA", "quote": "EURR", "active": True,
        },
        "ALPHA-PERP": {
            "symbol": "ALPHA-PERP", "id": "ALPHA_EURR-PERPETUAL", "type": "swap",
            "base": "ALPHA", "quote": "EURR", "settle": "EURR", "active": True,
        },
        "ALPHA-FUT": {
            "symbol": "ALPHA-FUT", "id": "ALPHA_EURR-27JUN25", "type": "future",
            "base": "ALPHA", "quote": "EURR", "settle": "EURR", "active": True,
        },
        "ALPHA-OPT": {
            "symbol": "ALPHA-OPT", "id": "ALPHA_EURR-27JUN25-100-C", "type": "option",
            "base": "ALPHA", "quote": "EURR", "settle": "EURR", "active": True,
            "strike": 100.0, "optionType": "call",
        },
        "BTC-PERP": {
            "symbol": "BTC-PERP", "id": "BTC-PERPETUAL", "type": "swap",
            "base": "BTC", "quote": "USD", "settle": "BTC", "active": True,
        },
        "BTC-FUT": {
            "symbol": "BTC-FUT", "id": "BTC-27JUN25", "type": "future",
            "base": "BTC", "quote": "USD", "settle": "BTC", "active": True,
        },
        "ETH-OPT": {
            "symbol": "ETH-OPT", "id": "ETH-27JUN25-3000-C", "type": "option",
            "base": "ETH", "quote": "ETH", "settle": "ETH", "active": True,
            "strike": 3000.0, "optionType": "call",
        },
        "SOL-OLD": {
            "symbol": "SOL-OLD", "id": "SOL_USDC-OLD", "type": "future",
            "base": "SOL", "quote": "USDC", "settle": "USDC", "active": False,
        },
    }


# ---------------------------------------------------------------------------
# Construction / auth
# ---------------------------------------------------------------------------
class TestInit:
    def test_unauthenticated_by_default(self, client):
        assert client.authenticated is False

    def test_authenticated_with_explicit_creds(self):
        c = DeribitClient(api_key="k", api_secret="s")
        assert c.authenticated is True

    def test_credentials_from_env(self, monkeypatch):
        monkeypatch.setenv("DERIBIT_API_KEY", "envk")
        monkeypatch.setenv("DERIBIT_API_SECRET", "envs")
        assert DeribitClient().authenticated is True

    def test_testnet_from_env(self, monkeypatch):
        monkeypatch.setenv("DERIBIT_TESTNET", "true")
        assert DeribitClient().testnet is True

    def test_testnet_explicit_overrides(self, monkeypatch):
        monkeypatch.setenv("DERIBIT_TESTNET", "true")
        assert DeribitClient(testnet=False).testnet is False


class TestCurrencyDiscovery:
    def test_settlement_currencies_are_loaded_from_markets(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())
        df = client.list_settlement_currencies()

        assert {"BTC", "ETH", "EURR"} <= set(df["settlement_currency"])
        assert set(df["instrument_family"]) == {
            "perpetual",
            "term_future",
            "option",
        }

    def test_spot_assets_are_distinct_from_settlement(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())
        df = client.list_spot_assets()

        assert list(df.columns) == ["asset", "quote_currency", "instrument_count"]
        row = df.set_index(["asset", "quote_currency"]).loc[("ALPHA", "EURR")]
        assert row["instrument_count"] == 1

    def test_derivative_assets_include_settlement_currency(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())

        perps = client.list_perpetual_assets()
        futures = client.list_term_future_assets()
        options = client.list_option_assets()

        assert ("ALPHA", "EURR") in set(
            zip(perps["asset"], perps["settlement_currency"])
        )
        assert ("ALPHA", "EURR") in set(
            zip(futures["asset"], futures["settlement_currency"])
        )
        assert ("ALPHA", "EURR") in set(
            zip(options["asset"], options["settlement_currency"])
        )


class TestRequireAuth:
    def test_raises_when_unauthenticated(self, client):
        with pytest.raises(PermissionError):
            client.fetch_positions()

    def test_balance_requires_auth(self, client):
        with pytest.raises(PermissionError):
            client.fetch_balance()


class TestFetchPositions:
    def _authed(self):
        c = DeribitClient(api_key="k", api_secret="s")
        c.exchange = MagicMock(apiKey="k", secret="s")
        return c

    def test_returns_dataframe_and_drops_empty(self):
        c = self._authed()
        c.exchange.fetch_positions.return_value = [
            {"symbol": "BTC-PERP", "contracts": 5, "side": "long"},
            {"symbol": "ETH-PERP", "contracts": 0, "side": "long"},
        ]
        df = c.fetch_positions()
        assert isinstance(df, pd.DataFrame)
        assert list(df["symbol"]) == ["BTC-PERP"]  # zero-size dropped

    def test_include_empty_keeps_all(self):
        c = self._authed()
        c.exchange.fetch_positions.return_value = [
            {"symbol": "BTC-PERP", "contracts": 5},
            {"symbol": "ETH-PERP", "contracts": 0},
        ]
        df = c.fetch_positions(include_empty=True)
        assert len(df) == 2

    def test_flat_returns_empty_dataframe(self):
        c = self._authed()
        c.exchange.fetch_positions.return_value = []
        df = c.fetch_positions()
        assert isinstance(df, pd.DataFrame) and df.empty


class TestFetchBalance:
    def test_returns_per_currency_dataframe(self):
        c = DeribitClient(api_key="k", api_secret="s")
        c.exchange = MagicMock(apiKey="k", secret="s")
        c.exchange.fetch_balance.return_value = {
            "info": {"raw": 1},
            "timestamp": 1,
            "datetime": "2025-01-01",
            "free": {"BTC": 1.0},          # aggregate meta key — must be skipped
            "used": {"BTC": 0.0},
            "total": {"BTC": 1.0},
            "BTC": {"free": 1.0, "used": 0.0, "total": 1.0},
            "ETH": {"free": 2.0, "used": 0.5, "total": 2.5},
        }
        df = c.fetch_balance()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["currency", "free", "used", "total"]
        assert set(df["currency"]) == {"BTC", "ETH"}
        eth = df.set_index("currency").loc["ETH"]
        assert eth["total"] == 2.5


# ---------------------------------------------------------------------------
# list_instruments and convenience wrappers
# ---------------------------------------------------------------------------
class TestListInstruments:
    def test_invalid_kind_raises(self, client):
        with pytest.raises(ValueError):
            client.list_instruments(kind="bogus")

    def test_returns_dataframe(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())
        df = client.list_instruments()
        assert isinstance(df, pd.DataFrame)
        assert "symbol" in df.columns

    def test_active_only_filters_inactive(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())
        symbols = set(client.list_instruments()["symbol"])
        assert "SOL-OLD" not in symbols  # inactive
        assert "BTC-PERP" in symbols

    def test_include_inactive(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())
        symbols = set(client.list_instruments(active_only=False)["symbol"])
        assert "SOL-OLD" in symbols

    def test_kind_filter(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())
        futures = client.list_instruments(kind="future", asset="BTC")
        assert list(futures["symbol"]) == ["BTC-FUT"]

    def test_legacy_currency_matches_base_quote_or_settle(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())
        eth = client.list_instruments(currency="ETH")
        assert list(eth["symbol"]) == ["ETH-OPT"]

    def test_explicit_asset_quote_and_settlement_filters(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())

        spot = client.list_spot(asset="ALPHA", quote_currency="EURR")
        perps = client.list_perpetuals(settlement_currency="EURR")

        assert list(spot["symbol"]) == ["ALPHA/EURR"]
        assert list(perps["symbol"]) == ["ALPHA-PERP"]

    def test_results_sorted_by_symbol(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())
        symbols = list(client.list_instruments()["symbol"])
        assert symbols == sorted(symbols)

    def test_instrument_symbols_wrapper(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())
        df = client.instrument_symbols(kind="option", asset="ETH")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["symbol"]
        assert list(df["symbol"]) == ["ETH-OPT"]

    def test_list_options_wrapper(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: _markets())
        df = client.list_options(asset="ETH")
        assert isinstance(df, pd.DataFrame)
        assert list(df["symbol"]) == ["ETH-OPT"]


# ---------------------------------------------------------------------------
# fetch_option_surface
# ---------------------------------------------------------------------------
class TestOptionSurface:
    def _setup(self, client, monkeypatch, summaries):
        markets = {
            "ETH-OPT": {
                "id": "ETH-27JUN25-3000-C",
                "type": "option",
                "base": "ETH",
                "quote": "ETH",
                "settle": "ETH",
                "active": True,
                "expiry": int(pd.Timestamp("2025-06-27 08:00:00", tz="UTC").value // 1_000_000),
                "strike": 3000.0,
                "optionType": "call",
            }
        }
        monkeypatch.setattr(client, "load_markets", lambda reload=False: markets)
        client.exchange = MagicMock()
        client.exchange.publicGetGetBookSummaryByCurrency.return_value = {"result": summaries}

    def test_returns_dataframe_with_expected_columns(self, client, monkeypatch):
        self._setup(client, monkeypatch, [{
            "instrument_name": "ETH-27JUN25-3000-C",
            "bid_price": 0.05, "ask_price": 0.06, "mark_price": 0.055,
            "mark_iv": 52.95, "underlying_price": 3050.0,
            "open_interest": 123.0, "volume": 7.0,
        }])
        df = client.fetch_option_surface("ETH")
        assert isinstance(df, pd.DataFrame)
        assert {"instrument_id", "asset", "settlement_currency",
                "expiry", "strike", "option_type",
                "bid", "ask", "mark_price", "mark_iv",
                "underlying_price", "open_interest", "volume"} <= set(df.columns)
        row = df.iloc[0]
        assert row["asset"] == "ETH"
        assert row["settlement_currency"] == "ETH"
        assert row["strike"] == 3000.0
        assert row["option_type"] == "call"
        # Deribit's 52.95% is stored as the decimal fraction 0.5295.
        assert row["mark_iv"] == pytest.approx(0.5295)

    def test_empty_result_returns_empty_dataframe(self, client, monkeypatch):
        self._setup(client, monkeypatch, [])
        df = client.fetch_option_surface("ETH")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_expiry_date_filter_excludes_other_expiries(self, client, monkeypatch):
        self._setup(client, monkeypatch, [{
            "instrument_name": "ETH-27JUN25-3000-C",
            "bid_price": 0.05, "ask_price": 0.06, "mark_price": 0.055,
            "mark_iv": 52.95, "underlying_price": 3050.0,
            "open_interest": 1.0, "volume": 1.0,
        }])
        df = client.fetch_option_surface("ETH", expiry_date="2030-01-01")
        assert df.empty

    def test_asset_filter_uses_market_metadata_not_prefix_table(self, client, monkeypatch):
        markets = {
            "ALPHA-OPT": {
                "id": "ALPHA_EURR-27JUN25-100-C",
                "type": "option",
                "base": "ALPHA",
                "quote": "EURR",
                "settle": "EURR",
                "active": True,
                "strike": 100.0,
                "optionType": "call",
            },
            "BETA-OPT": {
                "id": "BETA_EURR-27JUN25-100-C",
                "type": "option",
                "base": "BETA",
                "quote": "EURR",
                "settle": "EURR",
                "active": True,
            },
        }
        monkeypatch.setattr(client, "load_markets", lambda reload=False: markets)
        client.exchange = MagicMock()
        client.exchange.publicGetGetBookSummaryByCurrency.return_value = {"result": [
            {"instrument_name": "ALPHA_EURR-27JUN25-100-C", "mark_iv": 80.0},
            {"instrument_name": "BETA_EURR-27JUN25-100-C", "mark_iv": 50.0},
        ]}
        df = client.fetch_option_surface(asset="ALPHA")

        client.exchange.publicGetGetBookSummaryByCurrency.assert_called_once_with(
            {"currency": "EURR", "kind": "option"}
        )
        assert list(df["instrument_id"]) == ["ALPHA_EURR-27JUN25-100-C"]


# ---------------------------------------------------------------------------
# derivative_underlyings
# ---------------------------------------------------------------------------
class TestDerivativeUnderlyings:
    def test_union_of_option_and_future_assets(self, client, monkeypatch):
        markets = {
            "AAA-OPT": {"symbol": "AAA-OPT", "id": "AAA_USDC-X", "type": "option",
                        "base": "AAA", "quote": "USDC", "settle": "USDC", "active": True},
            "BBB-FUT": {"symbol": "BBB-FUT", "id": "BBB_USDC-X", "type": "future",
                        "base": "BBB", "quote": "USDC", "settle": "USDC", "active": True},
            "AAA-FUT": {"symbol": "AAA-FUT", "id": "AAA_USDC-Y", "type": "future",
                        "base": "AAA", "quote": "USDC", "settle": "USDC", "active": True},
            "CCC-PERP": {"symbol": "CCC-PERP", "id": "CCC", "type": "swap",
                         "base": "CCC", "quote": "USDC", "settle": "USDC", "active": True},
        }
        monkeypatch.setattr(client, "load_markets", lambda reload=False: markets)
        df = client.derivative_underlyings()
        # AAA (option+future) and BBB (future); CCC is swap-only -> excluded. Sorted.
        assert list(df.columns) == ["asset"]
        assert list(df["asset"]) == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# fetch_forward_curve
# ---------------------------------------------------------------------------
class TestForwardCurve:
    def test_perpetual_and_dated_rows(self, client, monkeypatch):
        future_expiry = pd.Timestamp.now("UTC") + pd.Timedelta(days=90)
        expiry_ms = int(future_expiry.value // 1_000_000)
        markets = {
            "BTC-PERP": {
                "id": "BTC-PERPETUAL",
                "type": "swap",
                "base": "BTC",
                "quote": "USD",
                "settle": "BTC",
                "active": True,
            },
            "BTC-FUT": {
                "id": "BTC-27JUN25",
                "type": "future",
                "base": "BTC",
                "quote": "USD",
                "settle": "BTC",
                "active": True,
                "expiry": expiry_ms,
            },
        }
        monkeypatch.setattr(client, "load_markets", lambda reload=False: markets)
        client.exchange = MagicMock()
        client.exchange.publicGetGetBookSummaryByCurrency.return_value = {"result": [
            {"instrument_name": "BTC-PERPETUAL", "mark_price": 100100.0,
             "estimated_delivery_price": 100000.0},
            {"instrument_name": "BTC-27JUN25", "mark_price": 102000.0,
             "estimated_delivery_price": 100000.0},
        ]}
        df = client.fetch_forward_curve("BTC")
        assert isinstance(df, pd.DataFrame)
        client.exchange.publicGetGetBookSummaryByCurrency.assert_called_once_with(
            {"currency": "BTC", "kind": "future"}
        )
        # Perp sorts first (column is numpy bool, so compare truthily).
        assert bool(df.iloc[0]["is_perpetual"]) is True
        assert pd.isna(df.iloc[0]["expiry"])
        # None upcasts to NaN once the column also holds a float (the dated row).
        assert pd.isna(df.iloc[0]["days_to_expiry"])

        dated = df[~df["is_perpetual"]].iloc[0]
        assert dated["asset"] == "BTC"
        assert dated["settlement_currency"] == "BTC"
        assert dated["days_to_expiry"] == pytest.approx(90, abs=1)
        assert dated["basis"] == pytest.approx(2000.0)
        # Contango => positive annualized basis.
        assert dated["annualized_basis_pct"] > 0

    def test_forward_curve_uses_non_static_settlement_metadata(self, client, monkeypatch):
        expiry = pd.Timestamp.now("UTC") + pd.Timedelta(days=30)
        expiry_ms = int(expiry.value // 1_000_000)
        markets = {
            "ALPHA-PERP": {
                "id": "ALPHA_EURR-PERPETUAL",
                "type": "swap",
                "base": "ALPHA",
                "quote": "EURR",
                "settle": "EURR",
                "active": True,
            },
            "ALPHA-FUT": {
                "id": "ALPHA_EURR-27JUN25",
                "type": "future",
                "base": "ALPHA",
                "quote": "EURR",
                "settle": "EURR",
                "active": True,
                "expiry": expiry_ms,
            },
            "BETA-FUT": {
                "id": "BETA_EURR-27JUN25",
                "type": "future",
                "base": "BETA",
                "quote": "EURR",
                "settle": "EURR",
                "active": True,
                "expiry": expiry_ms,
            },
        }
        monkeypatch.setattr(client, "load_markets", lambda reload=False: markets)
        client.exchange = MagicMock()
        client.exchange.publicGetGetBookSummaryByCurrency.return_value = {"result": [
            {"instrument_name": "ALPHA_EURR-PERPETUAL", "mark_price": 10.0,
             "estimated_delivery_price": 10.0},
            {"instrument_name": "ALPHA_EURR-27JUN25", "mark_price": 11.0,
             "estimated_delivery_price": 10.0},
            {"instrument_name": "BETA_EURR-27JUN25", "mark_price": 20.0,
             "estimated_delivery_price": 20.0},
        ]}

        df = client.fetch_forward_curve(asset="ALPHA")

        client.exchange.publicGetGetBookSummaryByCurrency.assert_called_once_with(
            {"currency": "EURR", "kind": "future"}
        )
        assert set(df["instrument_id"]) == {
            "ALPHA_EURR-PERPETUAL",
            "ALPHA_EURR-27JUN25",
        }
        assert set(df["settlement_currency"]) == {"EURR"}

    def test_empty_returns_empty_dataframe(self, client, monkeypatch):
        monkeypatch.setattr(client, "load_markets", lambda reload=False: {})
        client.exchange = MagicMock()
        client.exchange.publicGetGetBookSummaryByCurrency.return_value = {"result": []}
        df = client.fetch_forward_curve("BTC")
        assert isinstance(df, pd.DataFrame) and df.empty


# ---------------------------------------------------------------------------
# fetch_repo_spreads
# ---------------------------------------------------------------------------
class TestRepoSpreads:
    def test_decomposition_against_fed_curve(self, client, monkeypatch):
        expiry = pd.Timestamp.now("UTC") + pd.Timedelta(days=365)
        fwd = pd.DataFrame([
            {"instrument_id": "BTC-FUT", "asset": "BTC", "settlement_currency": "BTC",
             "expiry": expiry, "is_perpetual": False,
             "days_to_expiry": 365.0, "mark_price": 105000.0, "index_price": 100000.0},
            {"instrument_id": "BTC-PERP", "asset": "BTC", "settlement_currency": "BTC",
             "expiry": pd.NaT, "is_perpetual": True,
             "days_to_expiry": None, "mark_price": 100100.0, "index_price": 100000.0},
        ])
        monkeypatch.setattr(
            client,
            "fetch_forward_curve",
            lambda currency=None, reload=False, asset=None, settlement_currency=None: fwd,
        )

        fed = MagicMock()
        fed.get_discount_factor.return_value = 0.96  # ~4.08% cc over 1y

        df = client.fetch_repo_spreads("BTC", fed_client=fed)
        assert isinstance(df, pd.DataFrame)
        # Only the dated future is decomposed (perp is excluded).
        assert list(df["instrument_id"]) == ["BTC-FUT"]

        row = df.iloc[0]
        assert row["asset"] == "BTC"
        assert row["settlement_currency"] == "BTC"
        T = 1.0
        implied = -math.log(100000.0 / 105000.0) / T
        rf = -math.log(0.96) / T
        assert row["implied_rate_pct"] == pytest.approx(implied * 100, abs=1e-2)
        assert row["risk_free_rate_pct"] == pytest.approx(rf * 100, abs=1e-2)
        assert row["repo_spread_bps"] == pytest.approx((implied - rf) * 10_000, abs=1.0)

    def test_empty_when_no_dated_futures(self, client, monkeypatch):
        fwd = pd.DataFrame([
            {"instrument_id": "BTC-PERP", "asset": "BTC", "settlement_currency": "BTC",
             "expiry": pd.NaT, "is_perpetual": True,
             "days_to_expiry": None, "mark_price": 100100.0, "index_price": 100000.0},
        ])
        monkeypatch.setattr(
            client,
            "fetch_forward_curve",
            lambda currency=None, reload=False, asset=None, settlement_currency=None: fwd,
        )
        fed = MagicMock()
        df = client.fetch_repo_spreads("BTC", fed_client=fed)
        assert isinstance(df, pd.DataFrame) and df.empty


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------
class TestOrders:
    @pytest.fixture
    def authed(self):
        c = DeribitClient(api_key="k", api_secret="s")
        c.exchange = MagicMock(apiKey="k", secret="s")
        c.exchange.create_order.return_value = {"id": "order-1"}
        return c

    def test_buy_at_market(self, authed):
        out = authed.buy_at_market("BTC-PERP", 1.0)
        assert out == {"id": "order-1"}
        authed.exchange.create_order.assert_called_once_with(
            "BTC-PERP", "market", "buy", 1.0, params={}
        )

    def test_sell_at_market(self, authed):
        authed.sell_at_market("BTC-PERP", 2.0)
        authed.exchange.create_order.assert_called_once_with(
            "BTC-PERP", "market", "sell", 2.0, params={}
        )

    def test_buy_at_limit_passes_price_and_params(self, authed):
        authed.buy_at_limit("BTC-PERP", 1.0, 95000.0, params={"post_only": True})
        authed.exchange.create_order.assert_called_once_with(
            "BTC-PERP", "limit", "buy", 1.0, 95000.0, params={"post_only": True}
        )

    def test_sell_at_limit(self, authed):
        authed.sell_at_limit("BTC-PERP", 1.0, 99000.0)
        authed.exchange.create_order.assert_called_once_with(
            "BTC-PERP", "limit", "sell", 1.0, 99000.0, params={}
        )

    def test_order_requires_auth(self, client):
        with pytest.raises(PermissionError):
            client.buy_at_market("BTC-PERP", 1.0)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------
class TestContextManager:
    def test_close_called_on_exit(self, monkeypatch):
        c = DeribitClient()
        c.exchange = MagicMock()
        with c as ctx:
            assert ctx is c
        c.exchange.close.assert_called_once()
