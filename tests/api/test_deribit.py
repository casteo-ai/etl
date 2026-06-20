"""Tests for the Deribit historical-data FastAPI routes (filesystem-backed)."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.deribit as deribit_mod
from api.deribit import router

_TERM_COLUMNS = [
    "snapshot_time",
    "instrument_id",
    "asset",
    "settlement_currency",
    "expiry",
    "is_perpetual",
    "instrument_family",
    "days_to_expiry",
    "mark_price",
    "index_price",
    "basis",
    "annualized_basis_pct",
]


def _write_partition(folder, day, rows):
    """Write one ``YYYYMMDD.csv.gz`` partition with the given dict rows."""
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / f"{day}.csv.gz", index=False, compression="gzip")


def _term_row(snapshot_time, instrument_id, **overrides):
    row = {
        "snapshot_time": snapshot_time,
        "instrument_id": instrument_id,
        "asset": "BTC",
        "settlement_currency": "USDC",
        "expiry": "2026-09-25 08:00:00+00:00",
        "is_perpetual": False,
        "instrument_family": "term_future",
        "days_to_expiry": 99.6,
        "mark_price": 66476.0,
        "index_price": 65870.0,
        "basis": 606.0,
        "annualized_basis_pct": 3.37,
    }
    row.update(overrides)
    return row


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Build a fake on-disk Deribit archive and point the routes at it."""
    term_dir = tmp_path / "futures" / "forward_curve"
    vol_dir = tmp_path / "options" / "volatility"

    # Term structure: BTC_USDC over three days, ETH_ETH over one day.
    _write_partition(
        term_dir / "BTC_USDC",
        "20260615",
        [_term_row("2026-06-15 00:00:00+00:00", "BTC_USDC-25SEP26", mark_price=64000.0)],
    )
    _write_partition(
        term_dir / "BTC_USDC",
        "20260616",
        [_term_row("2026-06-16 00:00:00+00:00", "BTC_USDC-25SEP26", mark_price=65000.0)],
    )
    _write_partition(
        term_dir / "BTC_USDC",
        "20260617",
        [
            _term_row("2026-06-17 00:00:00+00:00", "BTC_USDC-25SEP26", mark_price=66000.0),
            _term_row(
                "2026-06-17 00:00:00+00:00",
                "BTC_USDC-PERPETUAL",
                is_perpetual=True,
                instrument_family="perpetual",
                expiry=None,
                days_to_expiry=None,
                annualized_basis_pct=None,
            ),
        ],
    )
    _write_partition(
        term_dir / "ETH_ETH",
        "20260617",
        [_term_row("2026-06-17 00:00:00+00:00", "ETH_ETH-25SEP26", asset="ETH", settlement_currency="ETH")],
    )
    # A non-instrument directory and a stray file should be ignored.
    (term_dir / "_scratch").mkdir(parents=True)
    (term_dir / "BTC_USDC" / "README.txt").write_text("ignore me")

    # Volatility surface: a single BTC_USDC day with two strikes.
    _write_partition(
        vol_dir / "BTC_USDC",
        "20260617",
        [
            {
                "snapshot_time": "2026-06-17 00:00:00+00:00",
                "instrument_id": "BTC_USDC-25SEP26-57000-C",
                "asset": "BTC",
                "settlement_currency": "USDC",
                "expiry": "2026-09-25 08:00:00+00:00",
                "strike": 57000.0,
                "option_type": "call",
                "bid": 5.0,
                "ask": None,
                "mark_price": 8618.1,
                "mark_iv": 0.7388,
                "underlying_price": 65612.0,
                "open_interest": 0.0,
                "volume": 0.0,
            }
        ],
    )

    monkeypatch.setattr(deribit_mod, "_TERM_STRUCTURE_DIR", term_dir)
    monkeypatch.setattr(deribit_mod, "_VOLATILITY_DIR", vol_dir)
    return tmp_path


@pytest.fixture
def client(data_dir):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ----------------------------------------------------------------------
# Instrument listing
# ----------------------------------------------------------------------
def test_list_term_structure_instruments(client):
    body = client.get("/deribit/term-structure/instruments").json()
    assert body["dataset"] == "term-structure"
    assert body["count"] == 2
    by_name = {i["instrument"]: i for i in body["instruments"]}
    assert set(by_name) == {"BTC_USDC", "ETH_ETH"}
    btc = by_name["BTC_USDC"]
    assert btc["asset"] == "BTC"
    assert btc["settlement_currency"] == "USDC"
    assert btc["available_days"] == 3
    assert btc["first_date"] == "2026-06-15"
    assert btc["last_date"] == "2026-06-17"


def test_list_volatility_surface_instruments(client):
    body = client.get("/deribit/volatility-surface/instruments").json()
    assert body["dataset"] == "volatility-surface"
    assert [i["instrument"] for i in body["instruments"]] == ["BTC_USDC"]


# ----------------------------------------------------------------------
# Date-range vs. days window selection
# ----------------------------------------------------------------------
def test_default_window_is_last_90_days(client):
    body = client.get("/deribit/term-structure", params={"instrument": "BTC_USDC"}).json()
    # No window given -> defaults to the most recent 90 days, which covers all
    # three fixture days (1 + 1 + 2 rows).
    assert body["days"] == 90
    assert body["count"] == 4
    assert body["columns"] == _TERM_COLUMNS
    assert body["instrument"] == "BTC_USDC"


def test_days_returns_most_recent(client):
    body = client.get(
        "/deribit/term-structure", params={"instrument": "BTC_USDC", "days": 1}
    ).json()
    assert body["days"] == 1
    assert body["count"] == 2  # only the latest day (two rows)
    assert {r["snapshot_time"] for r in body["data"]} == {"2026-06-17 00:00:00+00:00"}


def test_date_range_inclusive(client):
    body = client.get(
        "/deribit/term-structure",
        params={"instrument": "BTC_USDC", "start_date": "2026-06-15", "end_date": "2026-06-16"},
    ).json()
    assert body["count"] == 2
    marks = sorted(r["mark_price"] for r in body["data"])
    assert marks == [64000.0, 65000.0]


def test_open_ended_start(client):
    body = client.get(
        "/deribit/term-structure",
        params={"instrument": "BTC_USDC", "start_date": "2026-06-16"},
    ).json()
    assert body["count"] == 3  # 16th (1) + 17th (2)


# ----------------------------------------------------------------------
# Pagination
# ----------------------------------------------------------------------
def test_limit_and_offset(client):
    first = client.get(
        "/deribit/term-structure", params={"instrument": "BTC_USDC", "limit": 2}
    ).json()
    assert first["count"] == 2
    rest = client.get(
        "/deribit/term-structure", params={"instrument": "BTC_USDC", "limit": 10, "offset": 2}
    ).json()
    assert rest["count"] == 2


# ----------------------------------------------------------------------
# Volatility surface payload
# ----------------------------------------------------------------------
def test_volatility_surface_nan_becomes_null(client):
    body = client.get(
        "/deribit/volatility-surface", params={"instrument": "BTC_USDC", "days": 1}
    ).json()
    assert body["count"] == 1
    row = body["data"][0]
    assert row["ask"] is None
    assert row["option_type"] == "call"
    assert row["mark_iv"] == 0.7388


# ----------------------------------------------------------------------
# Validation / errors
# ----------------------------------------------------------------------
def test_days_and_dates_conflict(client):
    resp = client.get(
        "/deribit/term-structure",
        params={"instrument": "BTC_USDC", "days": 1, "start_date": "2026-06-15"},
    )
    assert resp.status_code == 400


def test_start_after_end(client):
    resp = client.get(
        "/deribit/term-structure",
        params={"instrument": "BTC_USDC", "start_date": "2026-06-17", "end_date": "2026-06-15"},
    )
    assert resp.status_code == 400


def test_unknown_instrument_404(client):
    resp = client.get("/deribit/term-structure", params={"instrument": "DOGE_USDC"})
    assert resp.status_code == 404


def test_malformed_instrument_400(client):
    resp = client.get("/deribit/term-structure", params={"instrument": "not a symbol!"})
    assert resp.status_code == 400


def test_instrument_normalised_from_slash(client):
    body = client.get("/deribit/term-structure", params={"instrument": "btc/usdc", "days": 1}).json()
    assert body["instrument"] == "BTC_USDC"
