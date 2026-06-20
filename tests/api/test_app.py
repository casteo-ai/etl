"""Smoke tests for the FastAPI app: MCP mount + endpoint operation_ids.

These guard the MCP integration (``api/app.py``). Every API endpoint is exposed
as an MCP tool named after its ``operation_id``; a router added without an
explicit ``operation_id`` would silently get FastAPI's auto-generated name
(e.g. ``get_foo_bar_get``), so these tests fail loudly when that happens.
"""

from __future__ import annotations

from fastapi.routing import APIRoute

from api.app import app, mcp

# Expected MCP tool names (= endpoint operation_ids), source-prefixed.
EXPECTED_TOOLS = {
    "get_binance_ohlcv",
    "get_binance_order_book",
    "list_bitpanda_assets",
    "list_deribit_term_structure_instruments",
    "list_deribit_volatility_surface_instruments",
    "get_deribit_term_structure",
    "get_deribit_volatility_surface",
    "get_fred_usd_rates",
    "get_fred_usd_discount_factors",
    "list_crypto_archive_tick_files",
    "list_crypto_archive_minute_files",
    "get_crypto_archive_tick_data",
    "get_crypto_archive_minute_data",
}


def _api_routes() -> list[APIRoute]:
    """Data endpoints only — exclude the MCP mount and FastAPI's own routes."""
    return [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and not route.path.startswith("/mcp")
    ]


def test_mcp_is_mounted():
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/mcp" in paths


def test_mcp_exposes_expected_tools():
    assert {tool.name for tool in mcp.tools} == EXPECTED_TOOLS


def test_every_endpoint_has_an_explicit_operation_id():
    # Auto-generated operation_ids end with the HTTP method (e.g. ``..._get``);
    # an explicit one never does. This catches a new route missing operation_id.
    for route in _api_routes():
        assert route.operation_id is not None, f"{route.path} has no operation_id"
        for method in (m.lower() for m in route.methods):
            assert not route.operation_id.endswith(
                f"_{method}"
            ), f"{route.path} looks auto-generated: {route.operation_id}"


def test_endpoint_operation_ids_match_mcp_tools():
    operation_ids = {route.operation_id for route in _api_routes()}
    assert operation_ids == EXPECTED_TOOLS
