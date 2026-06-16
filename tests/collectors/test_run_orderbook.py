"""Unit tests for the order-book entrypoint."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.collectors import run_orderbook as entry


def _patch_clients(monkeypatch):
    for name in ("DeribitClient", "BinanceClient"):
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        monkeypatch.setattr(entry, name, lambda *a, **k: client)


def test_once_mode(monkeypatch):
    _patch_clients(monkeypatch)
    collector = MagicMock()
    captured = {}
    monkeypatch.setattr(entry, "OrderBookCollector", lambda **k: captured.update(k) or collector)
    run_once = MagicMock()
    monkeypatch.setattr(entry, "run_once", run_once)

    rc = entry.main(["--mode", "once", "--data-dir", "/tmp/ob", "--depth", "20", "--quote", "USDC"])
    assert rc == 0
    assert captured["data_dir"] == "/tmp/ob"
    assert captured["depth"] == 20
    assert captured["quote"] == "USDC"
    run_once.assert_called_once_with(collector, reload=True)


def test_loop_mode(monkeypatch):
    _patch_clients(monkeypatch)
    monkeypatch.setattr(entry, "OrderBookCollector", lambda **k: MagicMock())
    run_loop = MagicMock()
    monkeypatch.setattr(entry, "run_loop", run_loop)

    rc = entry.main(["--mode", "loop", "--interval", "30", "--reload-every", "5"])
    assert rc == 0
    _, kwargs = run_loop.call_args
    assert kwargs["interval"] == 30.0
    assert kwargs["reload_every"] == 5
