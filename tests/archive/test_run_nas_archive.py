"""Unit tests for the NAS-archive entrypoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.archive import run_nas_archive as entry


def test_main_builds_archiver_and_runs(monkeypatch):
    captured = {}

    class FakeArchiver:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def archive_once(self, dry_run=False):
            captured["dry_run"] = dry_run
            return pd.DataFrame(
                [{"relative_path": "a", "size_bytes": 1, "source_path": "s",
                  "dest_path": "d", "status": "copied"}]
            )

    monkeypatch.setattr(entry, "NasArchiver", FakeArchiver)
    rc = entry.main(["--source", "/src", "--nas", "/nas", "--max-age-days", "10", "--dry-run"])
    assert rc == 0
    assert captured["source_root"] == "/src"
    assert captured["dest_root"] == "/nas"
    assert captured["max_age_days"] == 10
    assert captured["require_mount"] is True
    assert captured["dry_run"] is True


def test_main_no_require_mount_flag(monkeypatch):
    captured = {}
    fake = MagicMock()
    fake.archive_once.return_value = pd.DataFrame(columns=["status"])
    monkeypatch.setattr(entry, "NasArchiver", lambda **k: captured.update(k) or fake)
    entry.main(["--no-require-mount"])
    assert captured["require_mount"] is False


def test_main_returns_1_on_failure(monkeypatch):
    fake = MagicMock()
    fake.archive_once.return_value = pd.DataFrame(
        [{"relative_path": "a", "size_bytes": 1, "source_path": "s",
          "dest_path": "d", "status": "error: boom"}]
    )
    monkeypatch.setattr(entry, "NasArchiver", lambda **k: fake)
    rc = entry.main([])
    assert rc == 1
