"""Unit tests for src.collectors.runner (shared scheduling + CLI driver)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.collectors import runner


# ---------------------------------------------------------------------------
# _seconds_until_next
# ---------------------------------------------------------------------------
class TestSecondsUntilNext:
    def test_aligns_to_minute_boundary(self):
        # 1012.3 % 60 = 52.3s past the minute -> 7.7s until the next boundary.
        assert runner._seconds_until_next(60.0, clock=lambda: 1012.3) == pytest.approx(7.7)

    def test_full_interval_when_on_boundary(self):
        assert runner._seconds_until_next(60.0, clock=lambda: 1200.0) == 60.0

    def test_non_positive_interval(self):
        assert runner._seconds_until_next(0.0, clock=lambda: 5.0) == 0.0


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------
class TestRunOnce:
    def test_calls_collect_once_with_reload(self):
        collector = MagicMock()
        collector.collect_once.return_value = [1, 2, 3]
        n = runner.run_once(collector, reload=True)
        collector.collect_once.assert_called_once_with(reload=True)
        assert n == 3


# ---------------------------------------------------------------------------
# run_loop
# ---------------------------------------------------------------------------
class TestRunLoop:
    def test_runs_max_iterations(self):
        collector = MagicMock()
        sleep_fn = MagicMock()
        n = runner.run_loop(
            collector, interval=60, reload_every=60,
            max_iterations=3, sleep_fn=sleep_fn, clock=lambda: 0.0,
        )
        assert n == 3
        assert collector.collect_once.call_count == 3
        # Sleeps between iterations only, not after the last one.
        assert sleep_fn.call_count == 2

    def test_reload_cadence(self):
        collector = MagicMock()
        runner.run_loop(
            collector, interval=60, reload_every=2,
            max_iterations=4, sleep_fn=MagicMock(), clock=lambda: 0.0,
        )
        reloads = [c.kwargs["reload"] for c in collector.collect_once.call_args_list]
        assert reloads == [True, False, True, False]

    def test_survives_collect_errors(self):
        collector = MagicMock()
        collector.collect_once.side_effect = [RuntimeError("boom"), None]
        n = runner.run_loop(
            collector, interval=60, max_iterations=2,
            sleep_fn=MagicMock(), clock=lambda: 0.0,
        )
        assert n == 2  # error on iteration 1 did not abort the loop

    def test_keyboard_interrupt_propagates(self):
        collector = MagicMock()
        collector.collect_once.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            runner.run_loop(collector, max_iterations=1, sleep_fn=MagicMock())


# ---------------------------------------------------------------------------
# run (arg parsing + driving a collector class)
# ---------------------------------------------------------------------------
class TestRun:
    def _patch_client(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = False
        monkeypatch.setattr(runner, "DeribitClient", lambda *a, **k: fake_client)
        return fake_client

    def test_once_mode(self, monkeypatch):
        self._patch_client(monkeypatch)
        collector = MagicMock()
        collector_cls = MagicMock(return_value=collector)
        run_once = MagicMock()
        monkeypatch.setattr(runner, "run_once", run_once)

        rc = runner.run(
            collector_cls, "desc", "/default/dir",
            argv=["--mode", "once", "--data-dir", "/tmp/x", "--no-reload"],
        )
        assert rc == 0
        # Collector built with the parsed data-dir.
        _, kwargs = collector_cls.call_args
        assert kwargs["data_dir"] == "/tmp/x"
        run_once.assert_called_once_with(collector, reload=False)

    def test_loop_mode(self, monkeypatch):
        self._patch_client(monkeypatch)
        collector_cls = MagicMock(return_value=MagicMock())
        run_loop = MagicMock()
        monkeypatch.setattr(runner, "run_loop", run_loop)

        rc = runner.run(
            collector_cls, "desc", "/default/dir",
            argv=["--mode", "loop", "--interval", "30", "--reload-every", "5"],
        )
        assert rc == 0
        _, kwargs = run_loop.call_args
        assert kwargs["interval"] == 30.0
        assert kwargs["reload_every"] == 5

    def test_default_data_dir(self, monkeypatch):
        self._patch_client(monkeypatch)
        collector_cls = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(runner, "run_loop", MagicMock())

        runner.run(collector_cls, "desc", "/default/dir", argv=[])
        _, kwargs = collector_cls.call_args
        assert kwargs["data_dir"] == "/default/dir"
