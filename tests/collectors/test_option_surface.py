"""Unit tests for src.collectors.option_surface (OptionSurfaceCollector).

No network access: the DeribitClient is replaced by a small fake whose
``fetch_option_surface`` returns canned surfaces. CSV persistence is exercised
against ``tmp_path``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.collectors.option_surface import (
    SNAPSHOT_COLUMN,
    SNAPSHOT_COLUMNS,
    OptionSurfaceCollector,
    _safe_code,
)
from src.feeds.deribit import OPTION_SURFACE_COLUMNS


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------
def _surface(rows: list[dict]) -> pd.DataFrame:
    """Build an option-surface DataFrame with the canonical column order."""
    return pd.DataFrame(rows, columns=OPTION_SURFACE_COLUMNS)


def _canned_surface() -> pd.DataFrame:
    """Two assets / settlement pairs: BTC-inverse and SOL-USDC-linear."""
    return _surface(
        [
            {
                "instrument_id": "BTC-27JUN25-100000-C",
                "asset": "BTC",
                "settlement_currency": "BTC",
                "expiry": pd.Timestamp("2025-06-27", tz="UTC"),
                "strike": 100000.0,
                "option_type": "call",
                "bid": 0.01, "ask": 0.02, "mark_price": 0.015, "mark_iv": 52.0,
                "underlying_price": 100500.0, "open_interest": 10.0, "volume": 1.0,
            },
            {
                "instrument_id": "BTC-27JUN25-100000-P",
                "asset": "BTC",
                "settlement_currency": "BTC",
                "expiry": pd.Timestamp("2025-06-27", tz="UTC"),
                "strike": 100000.0,
                "option_type": "put",
                "bid": 0.03, "ask": 0.04, "mark_price": 0.035, "mark_iv": 55.0,
                "underlying_price": 100500.0, "open_interest": 8.0, "volume": 2.0,
            },
            {
                "instrument_id": "SOL_USDC-27JUN25-150-C",
                "asset": "SOL",
                "settlement_currency": "USDC",
                "expiry": pd.Timestamp("2025-06-27", tz="UTC"),
                "strike": 150.0,
                "option_type": "call",
                "bid": 1.0, "ask": 1.2, "mark_price": 1.1, "mark_iv": 80.0,
                "underlying_price": 148.0, "open_interest": 100.0, "volume": 5.0,
            },
        ]
    )


class FakeDeribitClient:
    """Records reload calls and returns a preset surface (or a sequence)."""

    def __init__(self, surface=None, surfaces=None):
        self._surface = surface if surface is not None else _canned_surface()
        self._surfaces = list(surfaces) if surfaces is not None else None
        self.reload_calls: list[bool] = []
        self.closed = False

    def fetch_option_surface(self, reload=False):
        self.reload_calls.append(reload)
        if self._surfaces is not None:
            return self._surfaces.pop(0)
        return self._surface.copy()

    def close(self):
        self.closed = True


@pytest.fixture
def collector(tmp_path):
    return OptionSurfaceCollector(client=FakeDeribitClient(), data_dir=tmp_path, compress=False)


# ---------------------------------------------------------------------------
# _safe_code
# ---------------------------------------------------------------------------
class TestSafeCode:
    def test_upper_cases(self):
        assert _safe_code("btc") == "BTC"

    def test_none_and_nan_become_unknown(self):
        assert _safe_code(None) == "UNKNOWN"
        assert _safe_code(float("nan")) == "UNKNOWN"
        assert _safe_code(pd.NA) == "UNKNOWN"

    def test_empty_becomes_unknown(self):
        assert _safe_code("   ") == "UNKNOWN"

    def test_path_unsafe_chars_replaced(self):
        assert _safe_code("a/b c") == "A_B_C"


# ---------------------------------------------------------------------------
# collect_once — file layout & content
# ---------------------------------------------------------------------------
class TestCollectOnce:
    def test_one_file_per_asset_settlement_pair(self, collector, tmp_path):
        ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
        collector.collect_once(snapshot_time=ts)

        paths = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.csv"))
        assert paths == [
            "BTC_BTC/20260615.csv",
            "SOL_USDC/20260615.csv",
        ]

    def test_snapshot_time_is_first_column(self, collector, tmp_path):
        ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
        collector.collect_once(snapshot_time=ts)

        df = pd.read_csv(tmp_path / "BTC_BTC" / "20260615.csv")
        assert list(df.columns) == SNAPSHOT_COLUMNS
        assert df.columns[0] == SNAPSHOT_COLUMN

    def test_no_integer_index_written(self, collector, tmp_path):
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-15 12:00", tz="UTC"))
        first_line = (tmp_path / "BTC_BTC" / "20260615.csv").read_text().splitlines()[0]
        assert first_line.split(",")[0] == SNAPSHOT_COLUMN

    def test_rows_split_by_pair(self, collector, tmp_path):
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-15 12:00", tz="UTC"))
        btc = pd.read_csv(tmp_path / "BTC_BTC" / "20260615.csv")
        sol = pd.read_csv(tmp_path / "SOL_USDC" / "20260615.csv")
        assert len(btc) == 2
        assert len(sol) == 1
        assert set(btc["asset"]) == {"BTC"}
        assert set(sol["asset"]) == {"SOL"}

    def test_reloadable_into_pandas(self, collector, tmp_path):
        ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
        collector.collect_once(snapshot_time=ts)
        df = pd.read_csv(
            tmp_path / "BTC_BTC" / "20260615.csv",
            parse_dates=["snapshot_time", "expiry"],
        )
        assert df["snapshot_time"].iloc[0] == ts
        assert pd.api.types.is_datetime64_any_dtype(df["expiry"])

    def test_returns_stamped_snapshot(self, collector):
        ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
        out = collector.collect_once(snapshot_time=ts)
        assert list(out.columns) == SNAPSHOT_COLUMNS
        assert len(out) == 3
        assert (out[SNAPSHOT_COLUMN] == ts).all()

    def test_naive_snapshot_assumed_utc(self, collector, tmp_path):
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-15 12:00:00"))
        assert (tmp_path / "BTC_BTC" / "20260615.csv").exists()

    def test_default_snapshot_time_is_now(self, collector):
        out = collector.collect_once()
        assert out[SNAPSHOT_COLUMN].dt.tz is not None
        # Stamped with "now" — within a minute of test execution.
        assert (pd.Timestamp.now("UTC") - out[SNAPSHOT_COLUMN].iloc[0]) < pd.Timedelta(minutes=1)


# ---------------------------------------------------------------------------
# collect_once — appending across snapshots
# ---------------------------------------------------------------------------
class TestAppend:
    def test_second_snapshot_appends_without_duplicate_header(self, collector, tmp_path):
        t1 = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
        t2 = pd.Timestamp("2026-06-15 12:01:00", tz="UTC")
        collector.collect_once(snapshot_time=t1)
        collector.collect_once(snapshot_time=t2)

        path = tmp_path / "BTC_BTC" / "20260615.csv"
        lines = path.read_text().splitlines()
        # 1 header + 2 rows + 2 rows
        assert len(lines) == 5
        assert lines[0].startswith(SNAPSHOT_COLUMN)

        df = pd.read_csv(path, parse_dates=["snapshot_time"])
        assert set(df["snapshot_time"]) == {t1, t2}

    def test_snapshots_on_different_days_go_to_different_files(self, collector, tmp_path):
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-15 23:59:00", tz="UTC"))
        collector.collect_once(snapshot_time=pd.Timestamp("2026-06-16 00:00:00", tz="UTC"))
        names = sorted(p.name for p in (tmp_path / "BTC_BTC").glob("*.csv"))
        assert names == ["20260615.csv", "20260616.csv"]


# ---------------------------------------------------------------------------
# collect_once — edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_surface_writes_nothing(self, tmp_path):
        client = FakeDeribitClient(surface=_surface([]))
        col = OptionSurfaceCollector(client=client, data_dir=tmp_path, compress=False)
        out = col.collect_once(snapshot_time=pd.Timestamp("2026-06-15", tz="UTC"))
        assert out.empty
        assert list(out.columns) == SNAPSHOT_COLUMNS
        assert list(tmp_path.glob("*.csv")) == []

    def test_missing_asset_goes_to_unknown_file(self, tmp_path):
        surface = _surface(
            [
                {
                    "instrument_id": "X", "asset": None, "settlement_currency": None,
                    "expiry": pd.NaT, "strike": 1.0, "option_type": "call",
                    "bid": 0.0, "ask": 0.0, "mark_price": 0.0, "mark_iv": 0.0,
                    "underlying_price": 0.0, "open_interest": 0.0, "volume": 0.0,
                }
            ]
        )
        col = OptionSurfaceCollector(client=FakeDeribitClient(surface=surface), data_dir=tmp_path, compress=False)
        col.collect_once(snapshot_time=pd.Timestamp("2026-06-15", tz="UTC"))
        assert (tmp_path / "UNKNOWN_UNKNOWN" / "20260615.csv").exists()

    def test_reload_flag_forwarded(self, tmp_path):
        client = FakeDeribitClient()
        col = OptionSurfaceCollector(client=client, data_dir=tmp_path, compress=False)
        col.collect_once(snapshot_time=pd.Timestamp("2026-06-15", tz="UTC"), reload=True)
        assert client.reload_calls == [True]


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------
class TestContextManager:
    def test_close_propagates(self, tmp_path):
        client = FakeDeribitClient()
        with OptionSurfaceCollector(client=client, data_dir=tmp_path, compress=False):
            pass
        assert client.closed is True


# ---------------------------------------------------------------------------
# Compression (gzip is the default)
# ---------------------------------------------------------------------------
class TestCompression:
    def test_default_writes_csv_gz(self, tmp_path):
        col = OptionSurfaceCollector(client=FakeDeribitClient(), data_dir=tmp_path)
        col.collect_once(snapshot_time=pd.Timestamp("2026-06-15 12:00", tz="UTC"))
        path = tmp_path / "BTC_BTC" / "20260615.csv.gz"
        assert path.exists()
        df = pd.read_csv(path)  # compression inferred from .gz
        assert list(df.columns) == SNAPSHOT_COLUMNS
        assert len(df) == 2
