"""Unit tests for src.collectors.base (BaseSnapshotCollector + _safe_code)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.collectors.base import SNAPSHOT_COLUMN, BaseSnapshotCollector, _safe_code


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
# A minimal concrete collector to exercise the base directly.
# ---------------------------------------------------------------------------
class _DummyCollector(BaseSnapshotCollector):
    BASE_COLUMNS = ["asset", "settlement_currency", "value"]
    SNAPSHOT_LABEL = "dummy"

    def __init__(self, data_dir, frame, compress=False):
        # Bypass DeribitClient construction — no client needed for these tests.
        self.client = None
        self._data_dir = data_dir
        self._compress = compress
        self._frame = frame

    def _fetch(self, reload):
        self._last_reload = reload
        return self._frame.copy()


class TestBaseBehaviour:
    def test_fetch_not_implemented(self):
        base = BaseSnapshotCollector.__new__(BaseSnapshotCollector)
        with pytest.raises(NotImplementedError):
            base._fetch(reload=False)

    def test_snapshot_columns_prepends_snapshot_time(self, tmp_path):
        col = _DummyCollector(tmp_path, pd.DataFrame())
        assert col.snapshot_columns == [SNAPSHOT_COLUMN, "asset", "settlement_currency", "value"]

    def test_collect_once_writes_per_pair(self, tmp_path):
        frame = pd.DataFrame(
            [
                {"asset": "A", "settlement_currency": "X", "value": 1},
                {"asset": "B", "settlement_currency": "Y", "value": 2},
            ]
        )
        col = _DummyCollector(tmp_path, frame)
        out = col.collect_once(snapshot_time=pd.Timestamp("2026-06-15", tz="UTC"))
        assert len(out) == 2
        paths = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.csv"))
        assert paths == ["A_X/20260615.csv", "B_Y/20260615.csv"]
        assert col._last_reload is False


class TestCompression:
    def _frame(self):
        return pd.DataFrame(
            [
                {"asset": "A", "settlement_currency": "X", "value": 1},
                {"asset": "A", "settlement_currency": "X", "value": 2},
            ]
        )

    def test_gzip_writes_csv_gz_and_round_trips(self, tmp_path):
        col = _DummyCollector(tmp_path, self._frame(), compress=True)
        col.collect_once(snapshot_time=pd.Timestamp("2026-06-15", tz="UTC"))
        path = tmp_path / "A_X" / "20260615.csv.gz"
        assert path.exists()
        back = pd.read_csv(path)  # compression inferred from .gz
        assert len(back) == 2
        assert list(back.columns) == [SNAPSHOT_COLUMN, "asset", "settlement_currency", "value"]

    def test_gzip_append_is_multi_member_readable(self, tmp_path):
        col = _DummyCollector(tmp_path, self._frame(), compress=True)
        col.collect_once(snapshot_time=pd.Timestamp("2026-06-15 00:00", tz="UTC"))
        col.collect_once(snapshot_time=pd.Timestamp("2026-06-15 00:01", tz="UTC"))
        back = pd.read_csv(tmp_path / "A_X" / "20260615.csv.gz", parse_dates=[SNAPSHOT_COLUMN])
        assert len(back) == 4  # two snapshots appended
        assert back[SNAPSHOT_COLUMN].nunique() == 2

    def test_plain_csv_when_disabled(self, tmp_path):
        col = _DummyCollector(tmp_path, self._frame(), compress=False)
        col.collect_once(snapshot_time=pd.Timestamp("2026-06-15", tz="UTC"))
        assert (tmp_path / "A_X" / "20260615.csv").exists()
        assert not (tmp_path / "A_X" / "20260615.csv.gz").exists()
