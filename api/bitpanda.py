"""FastAPI routes for serving Bitpanda reference data."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, Depends, Query

from src.feeds.bitpanda import BitpandaClient

router = APIRouter(prefix="/bitpanda", tags=["bitpanda"])


@lru_cache(maxsize=1)
def get_bitpanda_client() -> BitpandaClient:
    """Return the process-wide Bitpanda client."""
    return BitpandaClient()


def _records_from_frame(frame: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame into JSON-friendly records (NaN -> None)."""
    data = frame.where(pd.notna(frame), None)
    return data.to_dict(orient="records")


@router.get("/assets", operation_id="list_bitpanda_assets")
def get_assets(
    refresh: bool = Query(False, description="Force a refresh, bypassing the 24h cache."),
    client: BitpandaClient = Depends(get_bitpanda_client),
) -> dict:
    """List the entire Bitpanda tradeable-asset universe.

    Returns every tradeable asset (crypto, metals, stocks, ETFs, indices) —
    ~13-14k entries with ``symbol``/``name``/``id``. Backed by the client's
    24h-cached ``tradeable_assets`` call; nothing is stored to disk.
    """
    data = client.tradeable_assets(reload=refresh)
    return {
        "exchange": "bitpanda",
        "count": len(data),
        "assets": _records_from_frame(data),
    }
