"""Tests for the Binance FastAPI routes (no network; client dependency mocked).

Focus: the ``/binance/ohlcv`` payload must share an identical header with the
crypto-archive minute endpoint and stay consistent with that header.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi import HTTPException

import api.binance as binance_api
from api.binance import OHLCV_COLUMNS, get_ohlcv
from api.crypto_archive import _MINUTE_COLUMNS


class FakeBinanceClient:
    """Returns a canned OHLCV frame shaped like the real client's output."""

    def __init__(self, frame=None):
        self.frame = frame if frame is not None else self._default_frame()
        self.calls = []

    @staticmethod
    def _default_frame():
        index = pd.to_datetime(
            ["2020-12-31 00:00:00+00:00", "2020-12-31 00:01:00+00:00"], utc=True
        )
        frame = pd.DataFrame(
            {
                "open": [28894.88, 28925.69],
                "high": [28925.69, 28960.0],
                "low": [28893.51, 28900.0],
                "close": [28925.69, 28950.0],
                "volume": [1.770822, 2.5],
                # close_time must be dropped by the route.
                "close_time": pd.to_datetime(
                    ["2020-12-31 00:00:59.999000+00:00", "2020-12-31 00:01:59.999000+00:00"],
                    utc=True,
                ),
                "quote_volume": [51176.83, 72000.0],
                "num_trades": [31, 40],
                "taker_buy_volume": [1.199272, 1.5],
                "taker_buy_quote_volume": [34661.64, 43000.0],
            },
            index=index,
        )
        frame.index.name = "open_time"
        return frame

    def fetch_ohlcv_daily_gzip(
        self,
        symbol,
        day,
        until=None,
        timeframe="1m",
        data_dir=None,
        force=False,
        verbose=False,
    ):
        self.calls.append((symbol, day, until, timeframe, data_dir, force, verbose))
        return self.frame.copy()


def test_ohlcv_header_matches_crypto_archive_minute():
    body = get_ohlcv(
        symbol="BTC/USDC",
        since="2020-12-31",
        until="2021-01-01",
        timeframe="1m",
        as_file=False,
        client=FakeBinanceClient(),
    )
    assert body["columns"] == OHLCV_COLUMNS == _MINUTE_COLUMNS
    # Every record's keys equal the declared header, in order.
    for row in body["ohlcv"]:
        assert list(row.keys()) == _MINUTE_COLUMNS


def test_ohlcv_data_consistent_with_header():
    body = get_ohlcv(
        symbol="BTC/USDC",
        since="2020-12-31",
        until="2021-01-01",
        timeframe="1m",
        as_file=False,
        client=FakeBinanceClient(),
    )
    first = body["ohlcv"][0]
    # timestamp is epoch SECONDS (int), matching the archive convention.
    assert first["timestamp"] == 1609372800
    assert isinstance(first["timestamp"], int)
    # 1-minute spacing is preserved.
    assert body["ohlcv"][1]["timestamp"] - first["timestamp"] == 60
    # close_time is not leaked into the shared schema.
    assert "close_time" not in first
    assert first["open"] == 28894.88
    assert first["num_trades"] == 31


def test_ohlcv_empty_frame_keeps_header():
    fake = FakeBinanceClient(
        pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "close_time",
                     "quote_volume", "num_trades", "taker_buy_volume", "taker_buy_quote_volume"]
        ).rename_axis("open_time")
    )
    body = get_ohlcv(
        symbol="BTC/USDC",
        since="2020-12-31",
        until="2021-01-01",
        timeframe="1m",
        as_file=False,
        client=fake,
    )
    assert body["count"] == 0
    assert body["ohlcv"] == []
    assert body["columns"] == _MINUTE_COLUMNS


def test_ohlcv_route_uses_daily_partition_client():
    fake = FakeBinanceClient()

    get_ohlcv(
        symbol="BTC/USDC",
        since="2020-12-31",
        until="2021-01-01",
        timeframe="1m",
        as_file=False,
        client=fake,
    )

    assert len(fake.calls) == 1
    symbol, day, until, timeframe, data_dir, force, verbose = fake.calls[0]
    assert symbol == "BTC/USDC"
    assert day == pd.Timestamp("2020-12-31", tz="UTC")
    assert until == pd.Timestamp("2021-01-01", tz="UTC")
    assert timeframe == "1m"
    assert data_dir == binance_api._OHLCV_DIR
    assert force is False
    assert verbose is False


def test_ohlcv_as_file_returns_ephemeral_csv_path(tmp_path, monkeypatch):
    monkeypatch.setattr(binance_api, "_EPHEMERAL_DIR", tmp_path)
    stale = tmp_path / "20000101_ohlcv_BTC_USDC_stale.csv"
    stale.write_text("old\n")
    fake = FakeBinanceClient()

    path = get_ohlcv(
        symbol="BTC/USDC",
        since="2020-12-31",
        until="2021-01-01",
        timeframe="1m",
        as_file=True,
        client=fake,
    )

    output = pd.read_csv(path)
    assert str(path).startswith(str(tmp_path / "202"))
    assert list(output.columns) == _MINUTE_COLUMNS
    assert len(output) == 2
    assert not stale.exists()


def test_ohlcv_rejects_non_minute_timeframe():
    with pytest.raises(HTTPException) as exc:
        get_ohlcv(
            symbol="BTC/USDC",
            since="2020-12-31",
            until="2021-01-01",
            timeframe="5m",
            as_file=False,
            client=FakeBinanceClient(),
        )
    assert exc.value.status_code == 400
