"""FastAPI application for the ETL data API.

The MCP server is exposed at ``/mcp`` (streamable HTTP transport) via
``fastapi-mcp``: every API endpoint is published as an MCP tool named after its
``operation_id`` (e.g. ``get_binance_ohlcv``, ``get_deribit_term_structure``).
The MCP instance is built after all routers are mounted so it captures the full
route set.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from api.binance import router as binance_router
from api.bitpanda import router as bitpanda_router
from api.crypto_archive import router as crypto_archive_router
from api.deribit import router as deribit_router
from api.fred import router as fred_router

app = FastAPI(title="ETL API")
app.include_router(binance_router)
app.include_router(bitpanda_router)
app.include_router(crypto_archive_router)
app.include_router(deribit_router)
app.include_router(fred_router)

# Expose every endpoint above as an MCP tool at /mcp. Built last so all routes
# are already registered on the app.
mcp = FastApiMCP(
    app,
    name="ETL API",
    description="Casteo.ai market-data ETL: Binance, Bitpanda, Deribit, FRED and crypto-archive datasets.",
)
mcp.mount_http()
