"""FastAPI routes for serving USD FRED rate data."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.feeds.fred import FedClient

router = APIRouter(prefix="/fred", tags=["fred"])

_ROUTE_CACHE_TTL = timedelta(hours=3)
_rates_cache: dict[str, tuple[datetime, dict]] = {}
_discount_factor_cache: dict[str, tuple[datetime, dict]] = {}


def _cached(cache: dict[str, tuple[datetime, dict]], key: str) -> Optional[dict]:
    entry = cache.get(key)
    if entry is None:
        return None

    fetched_at, payload = entry
    if datetime.now(timezone.utc) - fetched_at > _ROUTE_CACHE_TTL:
        cache.pop(key, None)
        return None
    return payload


def _store(cache: dict[str, tuple[datetime, dict]], key: str, payload: dict) -> dict:
    cache[key] = (datetime.now(timezone.utc), payload)
    return payload


@lru_cache(maxsize=1)
def get_fed_client() -> FedClient:
    """Return the process-wide FRED client backed by FRED_API_KEY."""
    return FedClient()


@router.get("/rates", operation_id="get_fred_usd_rates")
def get_usd_rates(
    force_refresh: bool = Query(False, description="Refresh FRED rates instead of using the feed cache."),
    client: FedClient = Depends(get_fed_client),
) -> dict:
    """Serve the latest USD interest-rate pillars from FRED."""
    cache_key = "usd_rates"
    if not force_refresh:
        payload = _cached(_rates_cache, cache_key)
        if payload is not None:
            return payload

    rates = client.get_rates(force_refresh=force_refresh)
    payload = {
        "currency": "USD",
        "rates": rates.to_dict(orient="records"),
    }
    if force_refresh:
        _discount_factor_cache.clear()
    return _store(_rates_cache, cache_key, payload)


@router.get("/discount-factors", operation_id="get_fred_usd_discount_factors")
def get_usd_discount_factors(
    as_of: Optional[date] = Query(None, description="Curve date. Defaults to today's local date."),
    client: FedClient = Depends(get_fed_client),
) -> dict:
    """Serve bootstrapped USD discount factors from the FRED rate curve."""
    curve_date = as_of or date.today()
    cache_key = curve_date.isoformat()
    payload = _cached(_discount_factor_cache, cache_key)
    if payload is not None:
        return payload

    discount_factors = client.bootstrap_discount_factors(as_of=curve_date)
    discount_factors["maturity_date"] = discount_factors["maturity_date"].astype(str)
    payload = {
        "currency": "USD",
        "as_of": curve_date.isoformat(),
        "discount_factors": discount_factors.to_dict(orient="records"),
    }
    return _store(_discount_factor_cache, cache_key, payload)
