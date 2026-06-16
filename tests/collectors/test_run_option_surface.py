"""Unit tests for the option-surface entrypoint (delegates to runner.run)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.collectors import run_option_surface as entry
from src.collectors.option_surface import DEFAULT_DATA_DIR, OptionSurfaceCollector


def test_main_delegates_to_runner(monkeypatch):
    run = MagicMock(return_value=0)
    monkeypatch.setattr(entry, "run", run)

    rc = entry.main(["--mode", "once"])
    assert rc == 0
    args, _ = run.call_args
    assert args[0] is OptionSurfaceCollector
    assert args[2] == DEFAULT_DATA_DIR
    # argv is forwarded through.
    assert run.call_args.args[3] == ["--mode", "once"]
