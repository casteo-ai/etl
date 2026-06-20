"""Tests for the Bitpanda FastAPI route (no network; client dependency mocked)."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.bitpanda import get_bitpanda_client, router


class FakeBitpandaClient:
    """Returns a canned universe and records the reload flag."""

    def __init__(self):
        self.reload_calls = []

    def tradeable_assets(self, reload=False):
        self.reload_calls.append(reload)
        return pd.DataFrame(
            [
                {"asset": "BTC", "name": "Bitcoin", "id": "1"},
                {"asset": "AAPL", "name": "Apple", "id": "2"},
                {"asset": "XAU", "name": "Gold", "id": "3"},
            ]
        )


@pytest.fixture
def fake_client():
    return FakeBitpandaClient()


@pytest.fixture
def client(fake_client):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_bitpanda_client] = lambda: fake_client
    return TestClient(app)


def test_returns_full_universe(client):
    resp = client.get("/bitpanda/assets")
    assert resp.status_code == 200
    body = resp.json()
    assert body["exchange"] == "bitpanda"
    assert body["count"] == 3
    assert body["assets"][0] == {"asset": "BTC", "name": "Bitcoin", "id": "1"}
    assert {a["asset"] for a in body["assets"]} == {"BTC", "AAPL", "XAU"}


def test_uses_cache_by_default(client, fake_client):
    client.get("/bitpanda/assets")
    assert fake_client.reload_calls == [False]


def test_refresh_forces_reload(client, fake_client):
    resp = client.get("/bitpanda/assets", params={"refresh": "true"})
    assert resp.status_code == 200
    assert fake_client.reload_calls == [True]


def test_nan_becomes_null(client, fake_client):
    fake_client.tradeable_assets = lambda reload=False: pd.DataFrame(
        [{"asset": "FOO", "name": None, "id": "9"}]
    )
    body = client.get("/bitpanda/assets").json()
    assert body["assets"][0]["name"] is None
