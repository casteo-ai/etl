"""
Deribit client built on top of CCXT.

Capabilities:
    * Connect to Deribit (mainnet or testnet), with optional authentication.
    * Query open positions in the portfolio.
    * List every instrument available live on the exchange.
    * Discover settlement currencies and instrument assets from live metadata.
    * Fetch option volatility surfaces.
    * Fetch forward curves across perpetual and term futures.
    * Place market and limit orders (buy / sell).

The client deliberately does not keep a static list of Deribit currencies,
assets, or settlement routes.  Deribit changes listings over time, so the
source of truth is the CCXT market map returned by ``load_markets()``:

    settlement currency  -> derivative ``settle`` / raw settlement metadata
    spot asset/currency  -> spot ``base`` and ``quote``
    perpetual asset      -> swap ``base`` plus derivative settlement currency
    term futures asset   -> future ``base`` plus derivative settlement currency
    options asset        -> option ``base`` plus derivative settlement currency

For market-data methods, prefer explicit keyword arguments:

    client.fetch_forward_curve(asset="<asset>")
    client.fetch_forward_curve(asset="<asset>", settlement_currency="<settle>")
    client.fetch_forward_curve(settlement_currency="<settle>")

The legacy positional ``currency`` argument is still accepted.  When market
metadata shows that the value is an asset, it is treated as an asset; when it is
only a derivative settlement currency, it is treated as a settlement currency.

Credentials are read from the environment by default (never hard-code them):
    DERIBIT_API_KEY     - Deribit API client id
    DERIBIT_API_SECRET  - Deribit API client secret
    DERIBIT_TESTNET     - "1"/"true" to use the testnet (test.deribit.com)

Authentication is only required for account-specific calls such as
``fetch_positions`` / ``fetch_balance`` / order placement.
Listing instruments, the option surface, and the forward curve are all public.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Optional

import pandas as pd

try:
    import ccxt
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ccxt is required for the Deribit client. Install it with "
        "`pip install ccxt`."
    ) from exc


# CCXT market `type` values exposed by Deribit.
INSTRUMENT_KINDS = ("spot", "future", "swap", "option")
DERIVATIVE_KINDS = ("future", "swap", "option")
FORWARD_KINDS = ("future", "swap")

INSTRUMENT_FAMILIES = {
    "spot": "spot",
    "swap": "perpetual",
    "future": "term_future",
    "option": "option",
}

OPTION_SURFACE_COLUMNS = [
    "instrument_id",
    "asset",
    "settlement_currency",
    "expiry",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mark_price",
    "mark_iv",
    "underlying_price",
    "open_interest",
    "volume",
]

FORWARD_CURVE_COLUMNS = [
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

REPO_SPREAD_COLUMNS = [
    "instrument_id",
    "asset",
    "settlement_currency",
    "expiry",
    "days_to_expiry",
    "mark_price",
    "index_price",
    "df_riskfree",
    "df_implied",
    "implied_rate_pct",
    "risk_free_rate_pct",
    "repo_spread_bps",
]


def _as_bool(value: Optional[str]) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _to_float(value: Any) -> Optional[float]:
    """Coerce a value to float, returning None for null/empty."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_fraction(value: Any) -> Optional[float]:
    """Coerce a percent value to a decimal fraction (Deribit 37.3 -> 0.373)."""
    pct = _to_float(value)
    return pct / 100.0 if pct is not None else None


def _normalise_code(value: Any) -> Optional[str]:
    """Return an upper-case exchange code, or None for empty values."""
    if value is None:
        return None
    code = str(value).strip().upper()
    return code or None


def _first_code(*values: Any) -> Optional[str]:
    for value in values:
        code = _normalise_code(value)
        if code is not None:
            return code
    return None


def _market_info(market: dict[str, Any]) -> dict[str, Any]:
    info = market.get("info")
    return info if isinstance(info, dict) else {}


def _market_type(market: dict[str, Any]) -> Optional[str]:
    market_type = market.get("type")
    return str(market_type) if market_type is not None else None


def _market_asset(market: dict[str, Any]) -> Optional[str]:
    """Asset/underlying side of a Deribit market."""
    info = _market_info(market)
    return _first_code(
        market.get("base"),
        market.get("baseId"),
        info.get("base_currency"),
        info.get("baseCurrency"),
        info.get("base"),
    )


def _market_quote_currency(market: dict[str, Any]) -> Optional[str]:
    info = _market_info(market)
    return _first_code(
        market.get("quote"),
        market.get("quoteId"),
        info.get("quote_currency"),
        info.get("quoteCurrency"),
        info.get("quote"),
    )


def _market_settlement_currency(market: dict[str, Any]) -> Optional[str]:
    """Derivative settlement currency. Spot markets normally do not have one."""
    info = _market_info(market)
    return _first_code(
        market.get("settle"),
        market.get("settleId"),
        market.get("settlement"),
        info.get("settlement_currency"),
        info.get("settlementCurrency"),
        info.get("settle_currency"),
        info.get("settleCurrency"),
        info.get("settle"),
    )


def _parse_deribit_instrument_id(
    instrument_id: Optional[str],
    api_settlement_currency: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Best-effort fallback for summary rows when CCXT metadata is unavailable.

    Deribit derivative ids expose the asset and, for linear contracts, the
    settlement leg before the first dash: ``ASSET-...`` or ``ASSET_SETTLE-...``.
    The parser is intentionally generic over code values.
    """
    if not instrument_id:
        return None, _normalise_code(api_settlement_currency)

    prefix = str(instrument_id).split("-", 1)[0]
    if "_" in prefix:
        asset, settle = prefix.rsplit("_", 1)
        return _normalise_code(asset), _normalise_code(settle)

    asset = _normalise_code(prefix)
    settle = _normalise_code(api_settlement_currency) or asset
    return asset, settle


class DeribitClient:
    """Thin, purpose-built wrapper around ``ccxt.deribit``."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        enable_rate_limit: bool = True,
        extra_options: Optional[dict] = None,
    ) -> None:
        api_key = api_key if api_key is not None else os.getenv("DERIBIT_API_KEY")
        api_secret = (
            api_secret if api_secret is not None else os.getenv("DERIBIT_API_SECRET")
        )
        if testnet is None:
            testnet = _as_bool(os.getenv("DERIBIT_TESTNET"))

        config: dict[str, Any] = {
            "enableRateLimit": enable_rate_limit,
            "options": {"adjustForTimeDifference": True},
        }
        if api_key:
            config["apiKey"] = api_key
        if api_secret:
            config["secret"] = api_secret
        if extra_options:
            config["options"].update(extra_options)

        self.testnet = bool(testnet)
        self.exchange = ccxt.deribit(config)
        if self.testnet:
            # Routes requests to test.deribit.com.
            self.exchange.set_sandbox_mode(True)

    # ------------------------------------------------------------------
    # Connection / metadata
    # ------------------------------------------------------------------
    @property
    def authenticated(self) -> bool:
        """True when API credentials are configured (private calls allowed)."""
        return bool(self.exchange.apiKey and self.exchange.secret)

    def load_markets(self, reload: bool = False) -> dict[str, Any]:
        """Load (and cache) the full market map from Deribit."""
        return self.exchange.load_markets(reload)

    # ------------------------------------------------------------------
    # Instruments (public)
    # ------------------------------------------------------------------
    def _list_instruments(
        self,
        kind: Optional[str] = None,
        currency: Optional[str] = None,
        asset: Optional[str] = None,
        quote_currency: Optional[str] = None,
        settlement_currency: Optional[str] = None,
        active_only: bool = True,
        reload: bool = False,
    ) -> list[dict[str, Any]]:
        """Filtered list of raw CCXT market dicts (internal form)."""
        if kind is not None and kind not in INSTRUMENT_KINDS:
            raise ValueError(
                f"Unknown kind {kind!r}; expected one of {INSTRUMENT_KINDS}."
            )

        markets = self.load_markets(reload=reload)
        results: list[dict[str, Any]] = []
        cur = _normalise_code(currency)
        requested_asset = _normalise_code(asset)
        requested_quote = _normalise_code(quote_currency)
        requested_settle = _normalise_code(settlement_currency)
        for market in markets.values():
            if active_only and market.get("active") is False:
                continue
            if kind is not None and _market_type(market) != kind:
                continue

            market_asset = _market_asset(market)
            market_quote = _market_quote_currency(market)
            market_settle = _market_settlement_currency(market)
            if requested_asset is not None and market_asset != requested_asset:
                continue
            if requested_quote is not None and market_quote != requested_quote:
                continue
            if requested_settle is not None and market_settle != requested_settle:
                continue
            if cur is not None and cur not in {
                code
                for code in (market_asset, market_quote, market_settle)
                if code is not None
            }:
                continue
            results.append(market)

        results.sort(key=lambda m: m.get("symbol") or "")
        return results

    def list_instruments(
        self,
        kind: Optional[str] = None,
        currency: Optional[str] = None,
        asset: Optional[str] = None,
        quote_currency: Optional[str] = None,
        settlement_currency: Optional[str] = None,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """
        List every instrument currently available on Deribit.

        Parameters
        ----------
        kind:
            Optional filter on instrument type, one of ``INSTRUMENT_KINDS``
            (``"future"``, ``"swap"``, ``"option"``, ``"spot"``).
        currency:
            Legacy broad filter matching any of asset, quote currency, or
            settlement currency. Prefer the explicit filters below.
        asset:
            Optional base asset / derivative underlying filter.
        quote_currency:
            Optional spot or derivative quote currency filter.
        settlement_currency:
            Optional derivative settlement currency filter.
        active_only:
            When True (default) only return instruments flagged active/live.
        reload:
            Force a fresh fetch instead of using the cached market map.

        Returns
        -------
        DataFrame of CCXT market records, one row per instrument, with columns
        such as ``symbol``, ``id`` (Deribit instrument name), ``type``,
        ``base``, ``quote``, ``settle``, ``expiry``, ``strike`` and
        ``optionType``. Empty DataFrame when nothing matches.
        """
        return pd.DataFrame(
            self._list_instruments(
                kind=kind,
                currency=currency,
                asset=asset,
                quote_currency=quote_currency,
                settlement_currency=settlement_currency,
                active_only=active_only,
                reload=reload,
            )
        )

    def instrument_symbols(self, **kwargs: Any) -> pd.DataFrame:
        """Single-column (``symbol``) DataFrame of matching instrument symbols."""
        markets = self._list_instruments(**kwargs)
        return pd.DataFrame([m["symbol"] for m in markets], columns=["symbol"])

    def _asset_summary(
        self,
        kind: str,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        if kind not in INSTRUMENT_KINDS:
            raise ValueError(
                f"Unknown kind {kind!r}; expected one of {INSTRUMENT_KINDS}."
            )

        markets = self._list_instruments(
            kind=kind, active_only=active_only, reload=reload
        )
        columns = (
            ["asset", "quote_currency", "instrument_count"]
            if kind == "spot"
            else ["asset", "settlement_currency", "instrument_count"]
        )
        rows_by_key: dict[tuple[Optional[str], Optional[str]], int] = {}
        for market in markets:
            asset = _market_asset(market)
            if asset is None:
                continue

            if kind == "spot":
                other = _market_quote_currency(market)
            else:
                other = _market_settlement_currency(market)

            key = (asset, other)
            rows_by_key[key] = rows_by_key.get(key, 0) + 1

        rows = [
            {
                columns[0]: key[0],
                columns[1]: key[1],
                "instrument_count": count,
            }
            for key, count in sorted(rows_by_key.items())
        ]
        return pd.DataFrame(rows, columns=columns)

    def list_settlement_currencies(
        self,
        kind: Optional[str] = None,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """
        List derivative settlement currencies currently present in metadata.

        Settlement currencies are distinct from tradable assets. They come from
        Deribit's derivative ``settle`` metadata and are grouped by instrument
        family (perpetual, term futures, options).
        """
        if kind == "spot":
            return pd.DataFrame(
                columns=[
                    "settlement_currency",
                    "instrument_family",
                    "instrument_count",
                ]
            )
        if kind is not None and kind not in DERIVATIVE_KINDS:
            raise ValueError(
                f"Unknown derivative kind {kind!r}; expected one of {DERIVATIVE_KINDS}."
            )

        kinds = {kind} if kind is not None else set(DERIVATIVE_KINDS)
        markets = self.load_markets(reload=reload)
        rows_by_key: dict[tuple[str, str], int] = {}
        for market in markets.values():
            market_kind = _market_type(market)
            if market_kind not in kinds:
                continue
            if active_only and market.get("active") is False:
                continue
            settle = _market_settlement_currency(market)
            if settle is None:
                continue
            family = INSTRUMENT_FAMILIES[market_kind]
            key = (settle, family)
            rows_by_key[key] = rows_by_key.get(key, 0) + 1

        rows = [
            {
                "settlement_currency": settle,
                "instrument_family": family,
                "instrument_count": count,
            }
            for (settle, family), count in sorted(rows_by_key.items())
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "settlement_currency",
                "instrument_family",
                "instrument_count",
            ],
        )

    def list_spot_assets(
        self,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """List spot base assets and their quote currencies from live metadata."""
        return self._asset_summary("spot", active_only=active_only, reload=reload)

    def list_spot_currencies(
        self,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Alias for ``list_spot_assets`` for callers using currency wording."""
        return self.list_spot_assets(active_only=active_only, reload=reload)

    def list_perpetual_assets(
        self,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """List perpetual-swap assets and settlement currencies."""
        return self._asset_summary("swap", active_only=active_only, reload=reload)

    def list_perpetual_currencies(
        self,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Alias for ``list_perpetual_assets`` for currency-oriented callers."""
        return self.list_perpetual_assets(active_only=active_only, reload=reload)

    def list_term_future_assets(
        self,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """List dated futures assets and settlement currencies."""
        return self._asset_summary("future", active_only=active_only, reload=reload)

    def list_term_future_currencies(
        self,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Alias for ``list_term_future_assets`` for currency-oriented callers."""
        return self.list_term_future_assets(active_only=active_only, reload=reload)

    def list_option_assets(
        self,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """List option assets and settlement currencies."""
        return self._asset_summary("option", active_only=active_only, reload=reload)

    def list_option_currencies(
        self,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Alias for ``list_option_assets`` for currency-oriented callers."""
        return self.list_option_assets(active_only=active_only, reload=reload)

    def derivative_underlyings(
        self,
        active_only: bool = True,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Unique underlying assets that have Deribit options or term futures.

        The union of option underlyings and dated-futures underlyings (e.g.
        ``BTC``, ``ETH``, ``SOL``, ...), returned as a single-column ``asset``
        DataFrame sorted alphabetically. Intended as the instrument universe for
        downstream cross-exchange collection.
        """
        options = self._asset_summary("option", active_only=active_only, reload=reload)
        futures = self._asset_summary("future", active_only=active_only, reload=False)
        assets = set(options["asset"].dropna()) | set(futures["asset"].dropna())
        return pd.DataFrame(sorted(assets), columns=["asset"])

    def list_options(
        self,
        currency: Optional[str] = None,
        asset: Optional[str] = None,
        settlement_currency: Optional[str] = None,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Return active option instruments with explicit asset/settlement filters."""
        return self.list_instruments(
            kind="option",
            currency=currency,
            asset=asset,
            settlement_currency=settlement_currency,
            active_only=True,
            reload=reload,
        )

    def list_futures(
        self,
        currency: Optional[str] = None,
        asset: Optional[str] = None,
        settlement_currency: Optional[str] = None,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Return active dated futures with explicit asset/settlement filters."""
        return self.list_instruments(
            kind="future",
            currency=currency,
            asset=asset,
            settlement_currency=settlement_currency,
            active_only=True,
            reload=reload,
        )

    def list_perpetuals(
        self,
        currency: Optional[str] = None,
        asset: Optional[str] = None,
        settlement_currency: Optional[str] = None,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Return active perpetual swaps with explicit asset/settlement filters."""
        return self.list_instruments(
            kind="swap",
            currency=currency,
            asset=asset,
            settlement_currency=settlement_currency,
            active_only=True,
            reload=reload,
        )

    def list_spot(
        self,
        currency: Optional[str] = None,
        asset: Optional[str] = None,
        quote_currency: Optional[str] = None,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Return active spot instruments with explicit asset/quote filters."""
        return self.list_instruments(
            kind="spot",
            currency=currency,
            asset=asset,
            quote_currency=quote_currency,
            active_only=True,
            reload=reload,
        )

    # ------------------------------------------------------------------
    # Portfolio (private — requires credentials)
    # ------------------------------------------------------------------
    def _require_auth(self, action: str) -> None:
        if not self.authenticated:
            raise PermissionError(
                f"{action} requires API credentials. Set DERIBIT_API_KEY and "
                "DERIBIT_API_SECRET (or pass them to DeribitClient)."
            )

    def fetch_positions(
        self,
        symbols: Optional[Iterable[str]] = None,
        currency: Optional[str] = None,
        kind: Optional[str] = None,
        include_empty: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch open positions in the portfolio.

        Parameters
        ----------
        symbols:
            Optional iterable of CCXT symbols to restrict the query to.
        currency:
            Optional Deribit settlement currency. When set,
            only that currency's positions are requested.
        kind:
            Optional Deribit kind filter ("future", "option", ...).
        include_empty:
            When False (default) positions with zero size are dropped.

        Returns
        -------
        DataFrame of CCXT positions, one row each, with columns such as
        ``symbol``, ``side``, ``contracts``, ``notional``, ``entryPrice``,
        ``markPrice`` and ``unrealizedPnl``. Empty DataFrame when flat.
        """
        self._require_auth("Fetching positions")

        params: dict[str, Any] = {}
        if currency:
            params["currency"] = currency.upper()
        if kind:
            params["kind"] = kind

        symbol_list = list(symbols) if symbols is not None else None
        positions = self.exchange.fetch_positions(symbol_list, params)

        if not include_empty:
            positions = [p for p in positions if (p.get("contracts") or 0) != 0]
        return pd.DataFrame(positions)

    def fetch_balance(self, currency: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch account balance as a per-currency DataFrame.

        Returns
        -------
        DataFrame with columns ``currency``, ``free``, ``used``, ``total`` —
        one row per funded currency. Empty DataFrame when no balances.
        """
        self._require_auth("Fetching balance")
        params: dict[str, Any] = {}
        if currency:
            params["currency"] = currency.upper()
        raw = self.exchange.fetch_balance(params)

        # ccxt mixes per-currency sub-dicts with aggregate/meta keys; skip the
        # latter and keep only currency-code → {free, used, total} entries.
        skip = {"info", "timestamp", "datetime", "free", "used", "total"}
        rows = [
            {
                "currency": code,
                "free": bal.get("free"),
                "used": bal.get("used"),
                "total": bal.get("total"),
            }
            for code, bal in raw.items()
            if code not in skip and isinstance(bal, dict)
        ]
        return pd.DataFrame(rows, columns=["currency", "free", "used", "total"])

    # ------------------------------------------------------------------
    # Market-data routing helpers
    # ------------------------------------------------------------------
    def _markets_for_kinds(
        self,
        markets: dict[str, Any],
        kinds: Iterable[str],
        asset: Optional[str] = None,
        settlement_currency: Optional[str] = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        kind_set = set(kinds)
        requested_asset = _normalise_code(asset)
        requested_settle = _normalise_code(settlement_currency)
        results: list[dict[str, Any]] = []

        for market in markets.values():
            if active_only and market.get("active") is False:
                continue
            if _market_type(market) not in kind_set:
                continue
            if requested_asset is not None and _market_asset(market) != requested_asset:
                continue
            if (
                requested_settle is not None
                and _market_settlement_currency(market) != requested_settle
            ):
                continue
            results.append(market)

        return results

    def _resolve_derivative_request(
        self,
        currency: Optional[str],
        asset: Optional[str],
        settlement_currency: Optional[str],
        markets: dict[str, Any],
        kinds: Iterable[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Resolve the old ``currency`` argument without hard-coded routes.

        If the value appears as an asset in relevant markets, asset wins.  If it
        appears only as a settlement currency, it is treated as settlement.
        Callers that need the settlement meaning for an ambiguous code should
        pass ``settlement_currency=...`` explicitly.
        """
        if currency is not None and asset is not None:
            raise ValueError("Pass either currency or asset, not both.")

        requested_asset = _normalise_code(asset)
        requested_settle = _normalise_code(settlement_currency)
        legacy = _normalise_code(currency)
        if legacy is None:
            return requested_asset, requested_settle

        if requested_settle is not None:
            return legacy, requested_settle

        kind_set = set(kinds)
        asset_matches = False
        settlement_matches = False
        for market in markets.values():
            if _market_type(market) not in kind_set:
                continue
            if _market_asset(market) == legacy:
                asset_matches = True
            if _market_settlement_currency(market) == legacy:
                settlement_matches = True

        if asset_matches:
            return legacy, None
        if settlement_matches:
            return None, legacy
        return legacy, None

    def _book_summary_scope(
        self,
        markets: dict[str, Any],
        kinds: Iterable[str],
        asset: Optional[str],
        settlement_currency: Optional[str],
    ) -> tuple[set[str], tuple[str, ...]]:
        target_markets = self._markets_for_kinds(
            markets,
            kinds,
            asset=asset,
            settlement_currency=settlement_currency,
        )
        target_ids = {
            str(market["id"]) for market in target_markets if market.get("id")
        }

        requested_settle = _normalise_code(settlement_currency)
        if requested_settle is not None:
            return target_ids, (requested_settle,)

        settlements = sorted(
            {
                settle
                for settle in (
                    _market_settlement_currency(market) for market in target_markets
                )
                if settle is not None
            }
        )
        return target_ids, tuple(settlements)

    def _summary_metadata(
        self,
        summary: dict[str, Any],
        id_to_market: dict[str, dict[str, Any]],
        api_settlement_currency: str,
    ) -> tuple[Optional[str], dict[str, Any], Optional[str], Optional[str]]:
        instrument_id = summary.get("instrument_name") or summary.get("id")
        inst_id = str(instrument_id) if instrument_id is not None else None
        market = id_to_market.get(inst_id, {}) if inst_id is not None else {}
        parsed_asset, parsed_settle = _parse_deribit_instrument_id(
            inst_id, api_settlement_currency
        )

        asset = _market_asset(market) or parsed_asset
        settle = _market_settlement_currency(market) or parsed_settle
        return inst_id, market, asset, settle

    def _summary_in_scope(
        self,
        instrument_id: Optional[str],
        asset: Optional[str],
        settlement_currency: Optional[str],
        target_ids: set[str],
        requested_asset: Optional[str],
        requested_settlement_currency: Optional[str],
    ) -> bool:
        if target_ids:
            return instrument_id in target_ids
        if requested_asset is not None and asset != requested_asset:
            return False
        if (
            requested_settlement_currency is not None
            and settlement_currency != requested_settlement_currency
        ):
            return False
        return True

    # ------------------------------------------------------------------
    # Market data — option surface (public)
    # ------------------------------------------------------------------
    def fetch_option_surface(
        self,
        currency: Optional[str] = None,
        expiry_date: Optional[str] = None,
        reload: bool = False,
        *,
        asset: Optional[str] = None,
        settlement_currency: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch option book summaries grouped into a volatility surface.

        Parameters
        ----------
        currency:
            Legacy filter. Prefer ``asset=`` for the option underlying or
            ``settlement_currency=`` for the derivative settlement currency.
        expiry_date:
            Optional ISO-date string (``"2025-06-27"``) to restrict output to
            one expiry slice.
        reload:
            Force a reload of the CCXT market map.
        asset:
            Option underlying/base asset. This is distinct from settlement.
        settlement_currency:
            Derivative settlement currency used for the Deribit API query.

        Returns
        -------
        DataFrame with columns:
            instrument_id, asset, settlement_currency, expiry (UTC), strike,
            option_type (call/put), bid, ask, mark_price, mark_iv (fraction),
            underlying_price, open_interest, volume

        Notes
        -----
        Deribit returns option summaries by settlement currency.  This method
        discovers the required settlement currency or currencies from live CCXT
        market metadata, then filters returned summaries by exact instrument id.
        ``mark_iv`` is the Black-Scholes implied vol as a decimal fraction
        (e.g. 0.5295 = 52.95%); Deribit's percent value is divided by 100.
        """
        markets = self.load_markets(reload=reload)
        id_to_market = {m["id"]: m for m in markets.values() if m.get("id")}
        requested_asset, requested_settle = self._resolve_derivative_request(
            currency,
            asset,
            settlement_currency,
            markets,
            ("option",),
        )
        target_ids, settlements = self._book_summary_scope(
            markets,
            ("option",),
            asset=requested_asset,
            settlement_currency=requested_settle,
        )

        rows = []
        for settle in settlements:
            resp = self.exchange.publicGetGetBookSummaryByCurrency(
                {"currency": settle, "kind": "option"}
            )
            for summary in resp.get("result", []):
                inst_id, market, summary_asset, summary_settle = self._summary_metadata(
                    summary, id_to_market, settle
                )
                if not self._summary_in_scope(
                    inst_id,
                    summary_asset,
                    summary_settle,
                    target_ids,
                    requested_asset,
                    requested_settle,
                ):
                    continue

                expiry_ts = market.get("expiry")
                expiry = (
                    pd.Timestamp(expiry_ts, unit="ms", tz="UTC")
                    if expiry_ts
                    else pd.NaT
                )

                if expiry_date is not None:
                    if pd.isna(expiry) or str(expiry.date()) != expiry_date:
                        continue

                rows.append(
                    {
                        "instrument_id": inst_id,
                        "asset": summary_asset,
                        "settlement_currency": summary_settle,
                        "expiry": expiry,
                        "strike": market.get("strike"),
                        "option_type": market.get("optionType"),
                        "bid": _to_float(summary.get("bid_price")),
                        "ask": _to_float(summary.get("ask_price")),
                        "mark_price": _to_float(summary.get("mark_price")),
                        # Deribit quotes mark_iv in percent (52.95); store as a
                        # decimal fraction (0.5295) so vols are absolute numbers.
                        "mark_iv": _to_fraction(summary.get("mark_iv")),
                        "underlying_price": _to_float(
                            summary.get("underlying_price")
                        ),
                        "open_interest": _to_float(summary.get("open_interest")),
                        "volume": _to_float(summary.get("volume")),
                    }
                )

        df = pd.DataFrame(rows, columns=OPTION_SURFACE_COLUMNS)
        if df.empty:
            return df
        df.sort_values(["expiry", "strike", "option_type"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Market data — forward curve (public)
    # ------------------------------------------------------------------
    def fetch_forward_curve(
        self,
        currency: Optional[str] = None,
        reload: bool = False,
        *,
        asset: Optional[str] = None,
        settlement_currency: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch the forward curve: perpetual swap + all dated futures.

        Parameters
        ----------
        currency:
            Legacy filter. Prefer ``asset=`` for the derivative underlying or
            ``settlement_currency=`` for the settlement currency.
        reload:
            Force a reload of the CCXT market map.
        asset:
            Perpetual/term-future underlying or base asset.
        settlement_currency:
            Derivative settlement currency used for the Deribit API query.

        Returns
        -------
        DataFrame with columns:
            instrument_id, asset, settlement_currency, expiry (UTC, NaT for
            perp), is_perpetual, instrument_family, days_to_expiry (NaN for
            perp), mark_price, index_price, basis (mark - index),
            annualized_basis_pct

        Rows are sorted perpetual-first, then by ascending expiry.

        Notes
        -----
        Deribit returns futures and perpetual summaries by settlement currency.
        This method discovers settlement currencies from live CCXT metadata and
        filters returned summaries by exact instrument id.
        ``index_price`` is Deribit's ``estimated_delivery_price`` — the
        current spot index value used to estimate settlement.
        ``annualized_basis_pct`` = basis / index × 365 / days_to_expiry × 100.
        """
        markets = self.load_markets(reload=reload)
        id_to_market = {m["id"]: m for m in markets.values() if m.get("id")}
        requested_asset, requested_settle = self._resolve_derivative_request(
            currency,
            asset,
            settlement_currency,
            markets,
            FORWARD_KINDS,
        )
        target_ids, settlements = self._book_summary_scope(
            markets,
            FORWARD_KINDS,
            asset=requested_asset,
            settlement_currency=requested_settle,
        )

        now = pd.Timestamp.now("UTC")
        rows = []
        for settle in settlements:
            resp = self.exchange.publicGetGetBookSummaryByCurrency(
                {"currency": settle, "kind": "future"}
            )
            for summary in resp.get("result", []):
                inst_id, market, summary_asset, summary_settle = self._summary_metadata(
                    summary, id_to_market, settle
                )
                if not self._summary_in_scope(
                    inst_id,
                    summary_asset,
                    summary_settle,
                    target_ids,
                    requested_asset,
                    requested_settle,
                ):
                    continue

                market_kind = _market_type(market)
                is_perp = market_kind == "swap" or "PERPETUAL" in (inst_id or "")
                instrument_family = (
                    "perpetual"
                    if is_perp
                    else INSTRUMENT_FAMILIES.get(market_kind or "future", "term_future")
                )
                expiry_ts = market.get("expiry")
                expiry = (
                    pd.NaT
                    if is_perp or not expiry_ts
                    else pd.Timestamp(expiry_ts, unit="ms", tz="UTC")
                )

                mark_price = _to_float(summary.get("mark_price"))
                index_price = _to_float(summary.get("estimated_delivery_price"))
                basis = (
                    mark_price - index_price
                    if mark_price is not None and index_price is not None
                    else None
                )

                if not is_perp and pd.notna(expiry) and index_price:
                    days_to_expiry = (expiry - now).total_seconds() / 86400
                    ann_basis_pct = (
                        round(basis / index_price * 365 / days_to_expiry * 100, 4)
                        if basis is not None and days_to_expiry > 0
                        else None
                    )
                    days_to_expiry = round(days_to_expiry, 2)
                else:
                    days_to_expiry = None
                    ann_basis_pct = None

                rows.append(
                    {
                        "instrument_id": inst_id,
                        "asset": summary_asset,
                        "settlement_currency": summary_settle,
                        "expiry": expiry,
                        "is_perpetual": is_perp,
                        "instrument_family": instrument_family,
                        "days_to_expiry": days_to_expiry,
                        "mark_price": mark_price,
                        "index_price": index_price,
                        "basis": round(basis, 4) if basis is not None else None,
                        "annualized_basis_pct": ann_basis_pct,
                    }
                )

        df = pd.DataFrame(rows, columns=FORWARD_CURVE_COLUMNS)
        if df.empty:
            return df
        df.sort_values(
            ["is_perpetual", "expiry"], ascending=[False, True], inplace=True
        )
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Order placement (private — requires credentials)
    # ------------------------------------------------------------------
    def buy_at_market(
        self,
        symbol: str,
        amount: float,
        params: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Place a market buy order.

        Parameters
        ----------
        symbol:
            CCXT symbol for a Deribit perpetual, dated future, option, or spot
            instrument.
        amount:
            Order size in CCXT-normalised units for that instrument.
        params:
            Extra CCXT / Deribit params passed through to the API.
        """
        self._require_auth("Placing orders")
        return self.exchange.create_order(
            symbol, "market", "buy", amount, params=params or {}
        )

    def sell_at_market(
        self,
        symbol: str,
        amount: float,
        params: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Place a market sell order. See ``buy_at_market`` for parameter docs."""
        self._require_auth("Placing orders")
        return self.exchange.create_order(
            symbol, "market", "sell", amount, params=params or {}
        )

    def buy_at_limit(
        self,
        symbol: str,
        amount: float,
        price: float,
        params: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Place a limit buy order.

        Parameters
        ----------
        symbol:
            CCXT symbol.
        amount:
            Order size (same units as ``buy_at_market``).
        price:
            Limit price in the instrument's quote currency.
        params:
            Extra CCXT / Deribit params (e.g. ``{"post_only": True}``).
        """
        self._require_auth("Placing orders")
        return self.exchange.create_order(
            symbol, "limit", "buy", amount, price, params=params or {}
        )

    def sell_at_limit(
        self,
        symbol: str,
        amount: float,
        price: float,
        params: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Place a limit sell order. See ``buy_at_limit`` for parameter docs."""
        self._require_auth("Placing orders")
        return self.exchange.create_order(
            symbol, "limit", "sell", amount, price, params=params or {}
        )

    # ------------------------------------------------------------------
    # Repo spread decomposition (public data + FED curve)
    # ------------------------------------------------------------------
    def fetch_repo_spreads(
        self,
        currency: Optional[str] = None,
        fed_client: Any = None,
        reload: bool = False,
        *,
        asset: Optional[str] = None,
        settlement_currency: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Decompose each dated futures contract into a risk-free discount factor
        (from the FED SOFR/Treasury curve) and the residual repo spread baked
        into the futures price.

        The cost-of-carry model:
            F = S · exp(r_total · T)
            r_total  = implied_rate = ln(F/S) / T   [continuously compounded]
            r_rf     = risk-free zero rate at tenor T  (from FedClient)
            repo_spread = r_total − r_rf

        A positive repo_spread means the futures price is above the pure
        risk-free carry — the extra basis captures funding demand,
        convenience yield, and counterparty premia.

        Parameters
        ----------
        currency:
            Legacy filter. Prefer ``asset=`` or ``settlement_currency=``.
        fed_client:
            A ``FedClient`` instance.  If None, one is constructed automatically
            using the ``FRED_API_KEY`` environment variable.
        reload:
            Force a reload of the Deribit market map.
        asset:
            Dated futures underlying/base asset.
        settlement_currency:
            Derivative settlement currency.

        Returns
        -------
        DataFrame with columns:
            instrument_id, asset, settlement_currency, expiry, days_to_expiry,
            mark_price, index_price,
            df_riskfree, df_implied,
            implied_rate_pct, risk_free_rate_pct,
            repo_spread_bps
        Sorted by ascending days_to_expiry.
        """
        if fed_client is None:
            from src.feeds.fred import FedClient
            fed_client = FedClient()

        fwd = self.fetch_forward_curve(
            currency,
            reload=reload,
            asset=asset,
            settlement_currency=settlement_currency,
        )
        if fwd.empty:
            return pd.DataFrame(columns=REPO_SPREAD_COLUMNS)

        dated = fwd[
            ~fwd["is_perpetual"] & fwd["days_to_expiry"].notna()
        ].copy()

        rows = []
        for _, row in dated.iterrows():
            T = row["days_to_expiry"] / 365.0
            F = row["mark_price"]
            S = row["index_price"]

            if not (T > 0 and F and S and F > 0 and S > 0):
                continue

            # Implied (total) continuously compounded rate from futures price
            df_implied = S / F
            implied_rate = -math.log(df_implied) / T

            # Risk-free rate from bootstrapped FED curve
            df_riskfree = fed_client.get_discount_factor(T)
            rf_rate = -math.log(max(df_riskfree, 1e-12)) / T

            repo_spread = implied_rate - rf_rate

            rows.append(
                {
                    "instrument_id": row["instrument_id"],
                    "asset": row.get("asset"),
                    "settlement_currency": row.get("settlement_currency"),
                    "expiry": row["expiry"],
                    "days_to_expiry": row["days_to_expiry"],
                    "mark_price": F,
                    "index_price": S,
                    "df_riskfree": round(df_riskfree, 8),
                    "df_implied": round(df_implied, 8),
                    "implied_rate_pct": round(implied_rate * 100, 4),
                    "risk_free_rate_pct": round(rf_rate * 100, 4),
                    "repo_spread_bps": round(repo_spread * 10_000, 2),
                }
            )

        df = pd.DataFrame(rows, columns=REPO_SPREAD_COLUMNS)
        if not df.empty:
            df.sort_values("days_to_expiry", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release any underlying network resources held by CCXT."""
        close = getattr(self.exchange, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    def __enter__(self) -> "DeribitClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
