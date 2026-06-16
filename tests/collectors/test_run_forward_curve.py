"""Unit tests for the forward-curve entrypoint (delegates to runner.run)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.collectors import run_forward_curve as entry
from src.collectors.forward_curve import DEFAULT_DATA_DIR, ForwardCurveCollector


def test_main_delegates_to_runner(monkeypatch):
    run = MagicMock(return_value=0)
    monkeypatch.setattr(entry, "run", run)

    rc = entry.main(["--mode", "once"])
    assert rc == 0
    args, _ = run.call_args
    assert args[0] is ForwardCurveCollector
    assert args[2] == DEFAULT_DATA_DIR
    assert run.call_args.args[3] == ["--mode", "once"]
