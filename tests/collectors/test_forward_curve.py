"""Unit tests for src.collectors.forward_curve (ForwardCurveCollector).

Persistence is shared with the option collector (see test_base.py /
test_option_surface.py); this module focuses on forward-curve specifics: the
right Deribit endpoint, the forward-curve column schema, and that the perpetual
plus dated futures are all persisted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.collectors.forward_curve import (
    SNAPSHOT_COLUMN,
    SNAPSHOT_COLUMNS,
    ForwardCurveCollector,
)
from src.feeds.deribit import FORWARD_CURVE_COLUMNS


def _curve(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FORWARD_CURVE_COLUMNS)


def _canned_curve() -> pd.DataFrame:
    """BTC inverse (perp + 1 dated) and SOL USDC-linear (perp + 1 dated)."""
    return _curve(
        [
            {
                "instrument_id": "BTC-PERPETUAL", "asset": "BTC",
                "settlement_currency": "BTC", "expiry": pd.NaT,
                "is_perpetual": True, "instrument_family": "perpetual",
                "days_to_expiry": None, "mark_price": 100500.0,
                "index_price": 100490.0, "basis": 10.0, "annualized_basis_pct": None,
            },
            {
                "instrument_id": "BTC-27JUN25", "asset": "BTC",
                "settlement_currency": "BTC",
                "expiry": pd.Timestamp("2025-06-27", tz="UTC"),
                "is_perpetual": False, "instrument_family": "term_future",
                "days_to_expiry": 12.0, "mark_price": 101000.0,
                "index_price": 100490.0, "basis": 510.0, "annualized_basis_pct": 15.4,
            },
            {
                "instrument_id": "SOL_USDC-PERPETUAL", "asset": "SOL",
                "settlement_currency": "USDC", "expiry": pd.NaT,
                "is_perpetual": True, "instrument_family": "perpetual",
                "days_to_expiry": None, "mark_price": 148.2,
                "index_price": 148.0, "basis": 0.2, "annualized_basis_pct": None,
            },
        ]
    )


class FakeDeribitClient:
    def __init__(self, curve=None):
        self._curve = curve if curve is not None else _canned_curve()
        self.reload_calls: list[bool] = []
        self.surface_calls = 0

    def fetch_forward_curve(self, reload=False):
        self.reload_calls.append(reload)
        return self._curve.copy()

    def fetch_option_surface(self, reload=False):  # guard: must not be used
        self.surface_calls += 1
        raise AssertionError("forward-curve collector must not call fetch_option_surface")

    def close(self):
        pass


@pytest.fixture
def collector(tmp_path):
    return ForwardCurveCollector(client=FakeDeribitClient(), data_dir=tmp_path, compress=False)


class TestCollectOnce:
    def test_uses_forward_curve_endpoint(self, collector):
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-15", tz="UTC"), reload=True)
        assert collector.client.reload_calls == [True]
        assert collector.client.surface_calls == 0

    def test_one_folder_per_pair(self, collector, tmp_path):
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-15 12:00", tz="UTC"))
        paths = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.csv"))
        assert paths == ["BTC_BTC/20260615.csv", "SOL_USDC/20260615.csv"]

    def test_columns_are_forward_curve_schema(self, collector, tmp_path):
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-15 12:00", tz="UTC"))
        df = pd.read_csv(tmp_path / "BTC_BTC" / "20260615.csv")
        assert list(df.columns) == SNAPSHOT_COLUMNS
        assert df.columns[0] == SNAPSHOT_COLUMN

    def test_perp_and_dated_both_stored(self, collector, tmp_path):
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-15 12:00", tz="UTC"))
        btc = pd.read_csv(tmp_path / "BTC_BTC" / "20260615.csv")
        assert len(btc) == 2
        assert set(btc["is_perpetual"]) == {True, False}

    def test_reloadable_into_pandas(self, collector, tmp_path):
        ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
        collector.collect_once(snapshot_time=ts)
        df = pd.read_csv(
            tmp_path / "BTC_BTC" / "20260615.csv",
            parse_dates=["snapshot_time", "expiry"],
        )
        assert df["snapshot_time"].iloc[0] == ts
        # Perpetual row has no expiry.
        perp = df[df["is_perpetual"]]
        assert perp["expiry"].isna().all()

    def test_returns_stamped_snapshot(self, collector):
        ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
        out = collector.collect_once(snapshot_time=ts)
        assert list(out.columns) == SNAPSHOT_COLUMNS
        assert len(out) == 3
        assert (out[SNAPSHOT_COLUMN] == ts).all()


class TestEdgeCases:
    def test_empty_curve_writes_nothing(self, tmp_path):
        client = FakeDeribitClient(curve=_curve([]))
        col = ForwardCurveCollector(client=client, data_dir=tmp_path, compress=False)
        out = col.collect_once(snapshot_time=pd.Timestamp("2026-06-15", tz="UTC"))
        assert out.empty
        assert list(out.columns) == SNAPSHOT_COLUMNS
        assert list(tmp_path.rglob("*.csv")) == []

    def test_default_data_dir(self):
        col = ForwardCurveCollector(client=FakeDeribitClient())
        assert str(col._data_dir).endswith("deribit/futures/forward_curve")

    def test_default_writes_csv_gz(self, tmp_path):
        col = ForwardCurveCollector(client=FakeDeribitClient(), data_dir=tmp_path)
        col.collect_once(snapshot_time=pd.Timestamp("2026-06-15 12:00", tz="UTC"))
        path = tmp_path / "BTC_BTC" / "20260615.csv.gz"
        assert path.exists()
        df = pd.read_csv(path)
        assert list(df.columns) == SNAPSHOT_COLUMNS
        assert len(df) == 2
