"""Unit tests for src.archive.nas_archiver (NasArchiver + helpers).

All filesystem work happens under ``tmp_path``; the mount guard is disabled
(``require_mount=False``) except where the guard itself is under test.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.archive import nas_archiver as mod
from src.archive.nas_archiver import NasArchiver, _dated_from_name, _is_mounted


NOW = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)


def _write(path: Path, text: str, mtime: datetime | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(path, (ts, ts))
    return path


@pytest.fixture
def roots(tmp_path):
    src = tmp_path / "data"
    nas = tmp_path / "nas"
    src.mkdir()
    nas.mkdir()
    return src, nas


@pytest.fixture
def archiver(roots):
    src, nas = roots
    return NasArchiver(source_root=src, dest_root=nas, max_age_days=30, require_mount=False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_dated_from_name(self):
        assert _dated_from_name(Path("a/20260501.csv.gz")) == date(2026, 5, 1)
        assert _dated_from_name(Path("a/20260501.csv")) == date(2026, 5, 1)

    def test_dated_from_name_non_dated(self):
        assert _dated_from_name(Path("a/BTC_USDT_1m.csv")) is None
        assert _dated_from_name(Path("a/99999999.csv")) is None  # invalid date

    def test_is_mounted_false_when_only_root_is_mount(self, tmp_path, monkeypatch):
        # Only the filesystem root reports as a mountpoint -> NAS is not mounted.
        monkeypatch.setattr(mod.os.path, "ismount", lambda p: str(p) == os.sep)
        assert _is_mounted(tmp_path / "data") is False

    def test_is_mounted_true_when_ismount(self, tmp_path, monkeypatch):
        target = (tmp_path / "data")
        target.mkdir()
        resolved = target.resolve()
        monkeypatch.setattr(mod.os.path, "ismount", lambda p: Path(p) == resolved)
        assert _is_mounted(target) is True


# ---------------------------------------------------------------------------
# age selection
# ---------------------------------------------------------------------------
class TestAgeSelection:
    def test_old_dated_file_archived(self, archiver, roots):
        src, nas = roots
        _write(src / "deribit/BTC_BTC/20260501.csv.gz", "old")  # 46 days < cutoff
        df = archiver.archive_once(now=NOW)
        assert (nas / "deribit/BTC_BTC/20260501.csv.gz").read_text() == "old"
        assert not (src / "deribit/BTC_BTC/20260501.csv.gz").exists()
        assert df.loc[0, "status"] == "archived"

    def test_recent_dated_file_kept(self, archiver, roots):
        src, nas = roots
        _write(src / "deribit/BTC_BTC/20260616.csv.gz", "today")
        df = archiver.archive_once(now=NOW)
        assert (src / "deribit/BTC_BTC/20260616.csv.gz").exists()
        assert not (nas / "deribit/BTC_BTC/20260616.csv.gz").exists()
        assert df.empty

    def test_exactly_30_days_kept(self, archiver, roots):
        src, _ = roots
        # cutoff = 2026-05-17; a file dated exactly on the cutoff is NOT older.
        _write(src / "deribit/X_Y/20260517.csv.gz", "edge")
        df = archiver.archive_once(now=NOW)
        assert df.empty
        assert (src / "deribit/X_Y/20260517.csv.gz").exists()

    def test_non_dated_uses_mtime_old(self, archiver, roots):
        src, nas = roots
        old = _write(src / "binance/BTC_USDT_1m.csv", "x", mtime=NOW - timedelta(days=40))
        archiver.archive_once(now=NOW)
        assert (nas / "binance/BTC_USDT_1m.csv").exists()
        assert not old.exists()

    def test_non_dated_uses_mtime_recent(self, archiver, roots):
        src, nas = roots
        _write(src / "binance/ETH_USDT_1m.csv", "x", mtime=NOW - timedelta(days=5))
        archiver.archive_once(now=NOW)
        assert (src / "binance/ETH_USDT_1m.csv").exists()
        assert not (nas / "binance/ETH_USDT_1m.csv").exists()


# ---------------------------------------------------------------------------
# mirroring / safety
# ---------------------------------------------------------------------------
class TestMirrorAndSafety:
    def test_structure_is_mirrored(self, archiver, roots):
        src, nas = roots
        _write(src / "deribit/options/volatility/SOL_USDC/20260401.csv.gz", "s")
        archiver.archive_once(now=NOW)
        assert (nas / "deribit/options/volatility/SOL_USDC/20260401.csv.gz").read_text() == "s"

    def test_content_preserved(self, archiver, roots):
        src, nas = roots
        payload = "snapshot_time,mark_iv\n2026-04-01,0.5\n" * 100
        _write(src / "deribit/A_B/20260401.csv", payload)
        archiver.archive_once(now=NOW)
        assert (nas / "deribit/A_B/20260401.csv").read_text() == payload

    def test_existing_nas_folders_untouched(self, archiver, roots):
        src, nas = roots
        _write(nas / "bitpanda/keep.txt", "keep")  # unrelated NAS folder
        _write(src / "deribit/A_B/20260401.csv.gz", "x")
        archiver.archive_once(now=NOW)
        assert (nas / "bitpanda/keep.txt").read_text() == "keep"

    def test_dry_run_moves_nothing(self, archiver, roots):
        src, nas = roots
        _write(src / "deribit/A_B/20260401.csv.gz", "x")
        df = archiver.archive_once(dry_run=True, now=NOW)
        assert (src / "deribit/A_B/20260401.csv.gz").exists()
        assert not (nas / "deribit/A_B/20260401.csv.gz").exists()
        assert df.loc[0, "status"] == "would-archive"

    def test_already_archived_drops_local(self, archiver, roots):
        src, nas = roots
        _write(src / "deribit/A_B/20260401.csv.gz", "same")
        _write(nas / "deribit/A_B/20260401.csv.gz", "same")  # identical already on NAS
        df = archiver.archive_once(now=NOW)
        assert df.loc[0, "status"] == "already-archived"
        assert not (src / "deribit/A_B/20260401.csv.gz").exists()
        assert (nas / "deribit/A_B/20260401.csv.gz").read_text() == "same"

    def test_empty_dirs_pruned(self, archiver, roots):
        src, _ = roots
        _write(src / "deribit/A_B/20260401.csv.gz", "x")
        archiver.archive_once(now=NOW)
        assert not (src / "deribit/A_B").exists()  # emptied dir removed
        assert src.exists()  # root kept

    def test_no_partial_files_left(self, archiver, roots):
        src, nas = roots
        _write(src / "deribit/A_B/20260401.csv.gz", "x")
        archiver.archive_once(now=NOW)
        assert list(nas.rglob("*.partial")) == []


# ---------------------------------------------------------------------------
# mount guard + return value
# ---------------------------------------------------------------------------
class TestMountGuardAndManifest:
    def test_require_mount_raises_when_not_mounted(self, roots, monkeypatch):
        src, nas = roots
        _write(src / "deribit/A_B/20260401.csv.gz", "x")
        monkeypatch.setattr(mod, "_is_mounted", lambda p: False)
        arch = NasArchiver(source_root=src, dest_root=nas, require_mount=True)
        with pytest.raises(RuntimeError, match="not on a mounted filesystem"):
            arch.archive_once(now=NOW)

    def test_manifest_columns_and_size(self, archiver, roots):
        src, _ = roots
        _write(src / "deribit/A_B/20260401.csv.gz", "abcd")
        df = archiver.archive_once(now=NOW)
        assert list(df.columns) == mod.MANIFEST_COLUMNS
        assert df.loc[0, "size_bytes"] == 4
        assert df.loc[0, "relative_path"] == "deribit/A_B/20260401.csv.gz"

    def test_missing_source_returns_empty(self, tmp_path):
        arch = NasArchiver(
            source_root=tmp_path / "missing", dest_root=tmp_path / "nas", require_mount=False
        )
        df = arch.archive_once(now=NOW)
        assert df.empty
        assert list(df.columns) == mod.MANIFEST_COLUMNS
