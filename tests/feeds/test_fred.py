"""Unit tests for src.feeds.fred (FedClient + helpers).

External FRED HTTP access is fully mocked: ``FedClient._fred.get_series`` is
replaced so no network call is ever made.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.feeds.fred import _PILLARS, _CACHE_TTL, FedClient, _interp_df


# Realistic, all-positive par rates (percent) keyed by FRED series id, so the
# bootstrapped discount curve is strictly decreasing in tenor.
RATES_PCT = {
    "SOFR": 5.30,
    "DGS1MO": 5.25,
    "DGS3MO": 5.20,
    "DGS6MO": 5.10,
    "DGS1": 4.90,
    "DGS2": 4.60,
    "DGS3": 4.40,
    "DGS5": 4.20,
    "DGS10": 4.10,
}


def _fake_get_series(series_id: str) -> pd.Series:
    """Mimic fredapi: a daily series with a trailing NaN to exercise dropna()."""
    return pd.Series([RATES_PCT[series_id], float("nan")])


@pytest.fixture
def client(monkeypatch):
    """A FedClient whose underlying Fred object never touches the network."""
    c = FedClient(api_key="dummy-key")
    monkeypatch.setattr(c._fred, "get_series", _fake_get_series)
    return c


# ---------------------------------------------------------------------------
# _interp_df
# ---------------------------------------------------------------------------
class TestInterpDf:
    def test_zero_or_negative_tenor_returns_one(self):
        assert _interp_df(0, [(1.0, 0.95)]) == 1.0
        assert _interp_df(-1.0, [(1.0, 0.95)]) == 1.0

    def test_empty_curve_returns_one(self):
        assert _interp_df(2.0, []) == 1.0

    def test_exact_pillar_returns_pillar_df(self):
        curve = [(1.0, 0.95), (2.0, 0.90)]
        assert _interp_df(2.0, curve) == pytest.approx(0.90)

    def test_log_linear_interpolation_within_range(self):
        curve = [(1.0, 0.95), (2.0, 0.90)]
        # Midpoint in log space => geometric mean of the two discount factors.
        expected = math.sqrt(0.95 * 0.90)
        assert _interp_df(1.5, curve) == pytest.approx(expected)

    def test_flat_zero_rate_extrapolation_beyond_last_pillar(self):
        curve = [(1.0, 0.95), (2.0, 0.90)]
        z_last = -math.log(0.90) / 2.0
        expected = math.exp(-z_last * 3.0)
        assert _interp_df(3.0, curve) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Construction / credentials
# ---------------------------------------------------------------------------
class TestInit:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(ValueError):
            FedClient()

    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "env-key")
        c = FedClient()
        assert c._fred is not None

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "env-key")
        # Should not raise; explicit key is accepted.
        assert FedClient(api_key="explicit") is not None


# ---------------------------------------------------------------------------
# get_rates + caching
# ---------------------------------------------------------------------------
class TestGetRates:
    def test_rates_converted_to_decimal(self, client):
        df = client.get_rates()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["series_id", "tenor_years", "rate"]
        assert set(df["series_id"]) == set(_PILLARS)
        rates = dict(zip(df["series_id"], df["rate"]))
        assert rates["SOFR"] == pytest.approx(0.053)
        assert rates["DGS10"] == pytest.approx(0.041)

    def test_rates_sorted_by_tenor(self, client):
        df = client.get_rates()
        assert list(df["tenor_years"]) == sorted(df["tenor_years"])

    def test_results_cached(self, client, monkeypatch):
        calls = {"n": 0}

        def counting(series_id):
            calls["n"] += 1
            return _fake_get_series(series_id)

        monkeypatch.setattr(client._fred, "get_series", counting)

        client.get_rates()
        first = calls["n"]
        assert first == len(_PILLARS)

        client.get_rates()  # served from cache
        assert calls["n"] == first

    def test_force_refresh_refetches(self, client, monkeypatch):
        calls = {"n": 0}

        def counting(series_id):
            calls["n"] += 1
            return _fake_get_series(series_id)

        monkeypatch.setattr(client._fred, "get_series", counting)
        client.get_rates()
        client.get_rates(force_refresh=True)
        assert calls["n"] == 2 * len(_PILLARS)

    def test_cache_stale_property(self, client):
        assert client._cache_stale is True  # nothing fetched yet
        client.get_rates()
        assert client._cache_stale is False
        # Pretend the fetch happened longer ago than the TTL.
        client._fetched_at = datetime.now(timezone.utc) - _CACHE_TTL - timedelta(minutes=1)
        assert client._cache_stale is True


# ---------------------------------------------------------------------------
# bootstrap_discount_factors
# ---------------------------------------------------------------------------
class TestBootstrap:
    def test_returns_one_row_per_pillar(self, client):
        as_of = pd.Timestamp("2025-06-01").date()
        df = client.bootstrap_discount_factors(as_of=as_of)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["maturity_date", "discount_factor"]
        assert len(df) == len(_PILLARS)

    def test_all_discount_factors_in_unit_interval(self, client):
        df = client.bootstrap_discount_factors()
        for value in df["discount_factor"]:
            assert 0.0 < value <= 1.0

    def test_maturities_sorted(self, client):
        df = client.bootstrap_discount_factors()
        assert list(df["maturity_date"]) == sorted(df["maturity_date"])

    def test_curve_decreasing_with_tenor(self, client):
        # With a positive rate curve, discount factors must fall as tenor rises.
        tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        dfs = [client.get_discount_factor(t) for t in tenors]
        for earlier, later in zip(dfs, dfs[1:]):
            assert later < earlier


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------
class TestQueryInterface:
    def test_discount_factor_short_tenor_near_one(self, client):
        df = client.get_discount_factor(1 / 365)
        assert 0.99 < df <= 1.0

    def test_zero_rate_zero_tenor(self, client):
        assert client.get_zero_rate(0) == 0.0
        assert client.get_zero_rate(-1.0) == 0.0

    def test_zero_rate_positive_for_positive_tenor(self, client):
        z = client.get_zero_rate(2.0)
        assert z > 0
        # Sanity: continuously compounded zero rate is in a plausible band.
        assert 0.01 < z < 0.10

    def test_zero_rate_matches_discount_factor(self, client):
        t = 3.0
        df = client.get_discount_factor(t)
        z = client.get_zero_rate(t)
        assert df == pytest.approx(math.exp(-z * t))
