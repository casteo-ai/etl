"""Unit tests for src.collectors.binance_ohlcv.

No network: a fake Binance client writes deterministic daily gzip files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.collectors.binance_ohlcv import BinanceOhlcvCollector, DEFAULT_WORKERS


class FakeExchange:
    def __init__(self, markets):
        self._markets = markets

    def load_markets(self, reload=False):
        return self._markets


class FakeBinance:
    def __init__(self, markets):
        self.exchange = FakeExchange(markets)
        self.calls = []
        self.closed = False

    def fetch_ohlcv_daily_gzip(
        self,
        symbol,
        day,
        until=None,
        timeframe="1m",
        data_dir=None,
        force=False,
        verbose=True,
    ):
        self.calls.append((symbol, pd.Timestamp(day), until, timeframe, Path(data_dir), force, verbose))
        index = pd.DatetimeIndex([pd.Timestamp(day)], name="open_time")
        data = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [10.0],
            },
            index=index,
        )
        path = Path(data_dir) / symbol.replace("/", "_") / f"{pd.Timestamp(day).strftime('%Y%m%d')}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, compression="gzip")
        return data

    def fetch_ohlcv_monthly_gzip(self, *args, **kwargs):
        return {"status": "error", "rows": 0, "written_days": []}

    def close(self):
        self.closed = True


def _write_universe(path: Path) -> None:
    pd.DataFrame(
        [
            {"Rank": 1, "Symbol": "BTC", "Data Points": 10},
            {"Rank": 2, "Symbol": "ETH", "Data Points": 5},
            {"Rank": 3, "Symbol": "USDC", "Data Points": 4},
            {"Rank": 4, "Symbol": "USDT", "Data Points": 3},
            {"Rank": 5, "Symbol": "MISSING", "Data Points": 1},
        ]
    ).to_csv(path, index=False)


class TestBinanceOhlcvCollector:
    def test_default_workers_is_four(self, tmp_path):
        collector = BinanceOhlcvCollector(client=FakeBinance({}), universe_path=tmp_path / "missing.csv")
        assert collector.workers == DEFAULT_WORKERS == 4

    def test_load_universe_maps_assets_to_usdc(self, tmp_path):
        universe = tmp_path / "binance.csv"
        _write_universe(universe)
        collector = BinanceOhlcvCollector(
            client=FakeBinance({}),
            universe_path=universe,
            data_dir=tmp_path / "out",
        )

        out = collector.load_universe()

        assert set(out["asset"]) == {"BTC", "ETH", "MISSING"}
        assert set(out["symbol"]) == {"BTC/USDC", "ETH/USDC", "MISSING/USDC"}

    def test_collect_all_prefers_usdc_then_falls_back_to_usdt(self, tmp_path):
        universe = tmp_path / "binance.csv"
        _write_universe(universe)
        client = FakeBinance({"BTC/USDC": {}, "BTC/USDT": {}, "ETH/USDT": {}})
        collector = BinanceOhlcvCollector(
            client=client,
            universe_path=universe,
            data_dir=tmp_path / "out",
        )

        manifest = collector.collect_all(
            start="2026-01-01",
            until="2026-01-02",
            verbose=False,
        )

        paths = sorted(str(path.relative_to(tmp_path / "out")) for path in (tmp_path / "out").rglob("*.csv.gz"))
        assert paths == ["BTC_USDC/20260101.csv.gz", "ETH_USDT/20260101.csv.gz"]
        assert len(client.calls) == 2
        assert [call[0] for call in client.calls] == ["BTC/USDC", "ETH/USDT"]
        assert set(manifest["status"]) == {"written", "missing-market"}
        assert list(manifest[manifest["status"] == "missing-market"]["symbol"]) == ["MISSING/USDC"]

    def test_collect_all_parallel_uses_worker_local_clients(self, tmp_path):
        universe = tmp_path / "binance.csv"
        _write_universe(universe)
        markets = {"BTC/USDC": {}, "ETH/USDT": {}}
        discovery_client = FakeBinance(markets)
        worker_clients = []

        def client_factory():
            client = FakeBinance(markets)
            worker_clients.append(client)
            return client

        collector = BinanceOhlcvCollector(
            client=discovery_client,
            client_factory=client_factory,
            universe_path=universe,
            data_dir=tmp_path / "out",
            workers=2,
        )

        manifest = collector.collect_all(
            start="2026-01-01",
            until="2026-01-02",
            verbose=False,
        )

        assert len(worker_clients) == 2
        assert len(discovery_client.calls) == 0
        assert sum(len(client.calls) for client in worker_clients) == 2
        assert sorted(call[0] for client in worker_clients for call in client.calls) == ["BTC/USDC", "ETH/USDT"]
        assert set(manifest["status"]) == {"written", "missing-market"}
        assert all(client.closed for client in worker_clients)

    def test_collect_asset_monthly_archive_skips_daily_calls(self, tmp_path):
        class MonthlyFakeBinance(FakeBinance):
            def fetch_ohlcv_monthly_gzip(
                self,
                symbol,
                month,
                start=None,
                until=None,
                timeframe="1m",
                data_dir=None,
                force=False,
            ):
                self.monthly_call = (symbol, month, start, until, timeframe, Path(data_dir), force)
                root = Path(data_dir) / symbol.replace("/", "_")
                root.mkdir(parents=True, exist_ok=True)
                for day in ("20260101", "20260102"):
                    index = pd.DatetimeIndex([pd.Timestamp(day, tz="UTC")], name="open_time")
                    pd.DataFrame({"open": [1.0]}, index=index).to_csv(root / f"{day}.csv.gz", compression="gzip")
                return {"status": "ok", "rows": 2, "written_days": ["2026-01-01", "2026-01-02"]}

            def fetch_ohlcv_daily_gzip(self, *args, **kwargs):
                raise AssertionError("daily fetch should not run after monthly archive prefill")

        client = MonthlyFakeBinance({"BTC/USDC": {}})
        collector = BinanceOhlcvCollector(client=client, universe_path=tmp_path / "missing.csv", data_dir=tmp_path / "out")

        manifest = collector.collect_asset("BTC", start="2026-01-01", until="2026-02-01", verbose=False)

        assert client.monthly_call[0] == "BTC/USDC"
        assert set(manifest["status"]) == {"archive-prefilled", "empty"}
        assert (tmp_path / "out" / "BTC_USDC" / "20260101.csv.gz").exists()

    def test_close_propagates(self, tmp_path):
        client = FakeBinance({})
        collector = BinanceOhlcvCollector(client=client, universe_path=tmp_path / "missing.csv")
        collector.close()
        assert client.closed is True
