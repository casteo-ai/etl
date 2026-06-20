"""
Interactive Brokers client backed by ib_insync.

The client connects to a running TWS or IB Gateway session and fetches
historical 1-minute candles as pandas DataFrames.

Environment defaults
--------------------
IBKR_HOST       - TWS / Gateway host, default 127.0.0.1
IBKR_PORT       - TWS paper port is usually 7497; live TWS is usually 7496
IBKR_CLIENT_ID  - API client id, default 1
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Union

import pandas as pd

try:
    from ib_insync import (
        IB,
        Contract,
        Forex,
        Future,
        Index,
        Stock,
        util,
    )
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ib_insync is required for the Interactive Brokers client. "
        "Install it with `pip install ib_insync`."
    ) from exc


TimestampLike = Union[str, datetime, pd.Timestamp]

_CANDLE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "average_price",
    "bar_count",
]

_FUTURE_EXPIRY_COLUMNS = [
    "symbol",
    "expiry",
    "contract_month",
    "trading_class",
    "local_symbol",
    "multiplier",
    "exchange",
    "currency",
    "con_id",
]

_OPTION_EXPIRY_COLUMNS = [
    "symbol",
    "expiry",
    "contract_month",
    "trading_class",
    "right_types",
    "num_strikes",
    "min_strike",
    "max_strike",
    "multiplier",
    "exchange",
    "currency",
]

# Caller-facing option security types -> IBKR secType. ``option`` is a listed
# stock/index option (OPT); ``fop`` is an option on a future (FOP).
_OPTION_SECTYPES = {
    "option": "OPT",
    "opt": "OPT",
    "fop": "FOP",
    "future_option": "FOP",
    "futures_option": "FOP",
}


def _as_utc_timestamp(value: TimestampLike) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_ib_expiry(value: Optional[str]) -> pd.Timestamp:
    """Parse an IBKR expiry string to a tz-naive Timestamp (``NaT`` on failure).

    IBKR exposes the expiry as ``lastTradeDateOrContractMonth``, which is either
    a full day (``YYYYMMDD``) or a contract month (``YYYYMM``, mapped to the 1st).
    """
    if not value:
        return pd.NaT
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y%m"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return pd.NaT


class InteractiveBrokersClient:
    """
    Stateful Interactive Brokers market-data client.

    IBKR historical data requires a local TWS or IB Gateway process with API
    access enabled.  The class does not manage login or credentials; it only
    connects to the local API socket.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        readonly: bool = True,
        connect: bool = False,
        timeout: float = 10.0,
    ) -> None:
        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("IBKR_PORT", "7497"))
        self.client_id = int(client_id or os.getenv("IBKR_CLIENT_ID", "1"))
        self.readonly = readonly
        self.timeout = timeout
        self.ib = IB()

        if connect:
            self.connect()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.ib.isConnected()

    def connect(self) -> "InteractiveBrokersClient":
        """Connect to TWS or IB Gateway using the configured host/port."""
        if not self.is_connected:
            self.ib.connect(
                self.host,
                self.port,
                clientId=self.client_id,
                readonly=self.readonly,
                timeout=self.timeout,
            )
        return self

    def disconnect(self) -> None:
        """Disconnect from TWS or IB Gateway."""
        if self.is_connected:
            self.ib.disconnect()

    def _ensure_connected(self) -> None:
        if not self.is_connected:
            self.connect()

    def close(self) -> None:
        """Alias for :meth:`disconnect`; mirrors the other feed clients."""
        self.disconnect()

    def __enter__(self) -> "InteractiveBrokersClient":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------

    @staticmethod
    def stock_contract(
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        primary_exchange: Optional[str] = None,
    ) -> Contract:
        """Build an IBKR stock contract."""
        contract = Stock(symbol.upper(), exchange, currency.upper())
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        return contract

    @staticmethod
    def forex_contract(pair: str) -> Contract:
        """Build a cash FX contract, for example ``EURUSD``."""
        return Forex(pair.upper().replace("/", ""))

    @staticmethod
    def future_contract(
        symbol: str,
        last_trade_date_or_contract_month: str,
        exchange: str,
        currency: str = "USD",
    ) -> Contract:
        """Build an IBKR futures contract."""
        return Future(
            symbol.upper(),
            lastTradeDateOrContractMonth=last_trade_date_or_contract_month,
            exchange=exchange,
            currency=currency.upper(),
        )

    @staticmethod
    def index_contract(
        symbol: str,
        exchange: str,
        currency: str = "USD",
    ) -> Contract:
        """Build an IBKR index contract."""
        return Index(symbol.upper(), exchange, currency.upper())

    @staticmethod
    def crypto_contract(
        symbol: str,
        exchange: str = "PAXOS",
        currency: str = "USD",
    ) -> Contract:
        """Build an IBKR crypto contract where the account has crypto access."""
        return Contract(
            symbol=symbol.upper(),
            secType="CRYPTO",
            exchange=exchange,
            currency=currency.upper(),
        )

    def qualify_contract(self, contract: Contract) -> Contract:
        """Resolve an IBKR contract to its canonical conId metadata."""
        self._ensure_connected()
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError(f"IBKR could not qualify contract: {contract}")
        return qualified[0]

    def build_contract(
        self,
        symbol: str,
        security_type: str = "stock",
        exchange: str = "SMART",
        currency: str = "USD",
        primary_exchange: Optional[str] = None,
        last_trade_date_or_contract_month: Optional[str] = None,
    ) -> Contract:
        """Build a common IBKR contract from simple instrument fields."""
        kind = security_type.strip().lower()
        if kind in {"stock", "stk", "equity"}:
            return self.stock_contract(symbol, exchange, currency, primary_exchange)
        if kind in {"forex", "fx", "cash"}:
            return self.forex_contract(symbol)
        if kind in {"future", "fut"}:
            if not last_trade_date_or_contract_month:
                raise ValueError("Futures require last_trade_date_or_contract_month.")
            return self.future_contract(
                symbol,
                last_trade_date_or_contract_month,
                exchange,
                currency,
            )
        if kind in {"index", "ind"}:
            return self.index_contract(symbol, exchange, currency)
        if kind in {"crypto", "cryptocurrency"}:
            return self.crypto_contract(symbol, exchange, currency)
        raise ValueError(f"Unsupported IBKR security_type: {security_type}")

    # ------------------------------------------------------------------
    # Contract chains (expiries / option chains)
    # ------------------------------------------------------------------

    def list_future_expiries(
        self,
        symbol: str,
        exchange: str,
        currency: str = "USD",
        *,
        trading_class: Optional[str] = None,
        include_expired: bool = False,
    ) -> pd.DataFrame:
        """List the available futures expiries for an underlying.

        Resolves every listed futures contract for ``symbol`` on ``exchange``
        (e.g. ``"MBT"`` / ``"CME"`` for Micro Bitcoin) via IBKR contract
        details. One row per contract; pick a maturity by passing its
        ``contract_month`` to :meth:`fetch_1min_candles` as
        ``last_trade_date_or_contract_month``.

        Returns
        -------
        DataFrame with columns ``symbol``, ``expiry`` (parsed Timestamp),
        ``contract_month`` (raw IBKR string), ``trading_class``,
        ``local_symbol``, ``multiplier``, ``exchange``, ``currency``,
        ``con_id``, sorted by expiry. Empty when nothing resolves.
        """
        self._ensure_connected()
        contract = Future(symbol.upper(), exchange=exchange, currency=currency.upper())
        if trading_class:
            contract.tradingClass = trading_class
        contract.includeExpired = include_expired

        rows = []
        for details in self.ib.reqContractDetails(contract):
            c = details.contract
            rows.append(
                {
                    "symbol": c.symbol,
                    "expiry": _parse_ib_expiry(c.lastTradeDateOrContractMonth),
                    "contract_month": c.lastTradeDateOrContractMonth,
                    "trading_class": c.tradingClass,
                    "local_symbol": c.localSymbol,
                    "multiplier": c.multiplier,
                    "exchange": c.exchange,
                    "currency": c.currency,
                    "con_id": c.conId,
                }
            )

        df = pd.DataFrame(rows, columns=_FUTURE_EXPIRY_COLUMNS)
        if df.empty:
            return df
        return df.sort_values(["expiry", "trading_class"]).reset_index(drop=True)

    def list_option_expiries(
        self,
        symbol: str,
        exchange: str,
        currency: str = "USD",
        *,
        security_type: str = "option",
        trading_class: Optional[str] = None,
        include_expired: bool = False,
    ) -> pd.DataFrame:
        """List the option chain summarised by expiry for an underlying.

        Works for listed stock/index options (``security_type="option"`` → OPT)
        and options on futures (``security_type="fop"`` → FOP, e.g. options on
        the Micro Bitcoin future ``"MBT"`` / ``"CME"``). The raw chain has one
        contract per (expiry, strike, right); this collapses it to one row per
        ``(expiry, trading_class, multiplier)`` with the strike range and the
        available rights.

        Returns
        -------
        DataFrame with columns ``symbol``, ``expiry`` (parsed Timestamp),
        ``contract_month`` (raw IBKR string), ``trading_class``, ``right_types``
        (e.g. ``"CP"``), ``num_strikes``, ``min_strike``, ``max_strike``,
        ``multiplier``, ``exchange``, ``currency``, sorted by expiry. Empty when
        the underlying has no listed options.
        """
        sec_type = _OPTION_SECTYPES.get(security_type.strip().lower())
        if sec_type is None:
            raise ValueError(
                f"security_type must be one of {sorted(_OPTION_SECTYPES)}; got {security_type!r}."
            )

        self._ensure_connected()
        contract = Contract(
            symbol=symbol.upper(),
            secType=sec_type,
            exchange=exchange,
            currency=currency.upper(),
        )
        if trading_class:
            contract.tradingClass = trading_class
        contract.includeExpired = include_expired

        groups: dict[tuple, dict] = {}
        for details in self.ib.reqContractDetails(contract):
            c = details.contract
            key = (c.lastTradeDateOrContractMonth, c.tradingClass, c.multiplier)
            group = groups.setdefault(
                key,
                {"symbol": c.symbol, "strikes": set(), "rights": set(),
                 "exchange": c.exchange, "currency": c.currency},
            )
            if c.strike:
                group["strikes"].add(float(c.strike))
            if c.right:
                group["rights"].add(c.right)

        rows = []
        for (contract_month, trading_cls, multiplier), group in groups.items():
            strikes = sorted(group["strikes"])
            rows.append(
                {
                    "symbol": group["symbol"],
                    "expiry": _parse_ib_expiry(contract_month),
                    "contract_month": contract_month,
                    "trading_class": trading_cls,
                    "right_types": "".join(sorted(group["rights"])),
                    "num_strikes": len(strikes),
                    "min_strike": strikes[0] if strikes else None,
                    "max_strike": strikes[-1] if strikes else None,
                    "multiplier": multiplier,
                    "exchange": group["exchange"],
                    "currency": group["currency"],
                }
            )

        df = pd.DataFrame(rows, columns=_OPTION_EXPIRY_COLUMNS)
        if df.empty:
            return df
        return df.sort_values(["expiry", "trading_class"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Historical candles
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_to_dataframe(bars: object) -> pd.DataFrame:
        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame(columns=_CANDLE_COLUMNS).rename_axis("open_time")

        df = df.rename(
            columns={
                "date": "open_time",
                "average": "average_price",
                "barCount": "bar_count",
            }
        )
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time").sort_index()

        keep = [col for col in _CANDLE_COLUMNS if col in df.columns]
        df = df[keep].copy()
        for col in keep:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _request_1min_chunk(
        self,
        contract: Contract,
        end: pd.Timestamp,
        duration: str,
        what_to_show: str,
        use_rth: bool,
    ) -> pd.DataFrame:
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end.to_pydatetime(),
            durationStr=duration,
            barSizeSetting="1 min",
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,
            keepUpToDate=False,
        )
        return self._bars_to_dataframe(bars)

    def fetch_1min_candles(
        self,
        contract: Optional[Contract] = None,
        *,
        symbol: Optional[str] = None,
        security_type: str = "stock",
        exchange: str = "SMART",
        currency: str = "USD",
        primary_exchange: Optional[str] = None,
        last_trade_date_or_contract_month: Optional[str] = None,
        start: Optional[TimestampLike] = None,
        end: Optional[TimestampLike] = None,
        duration: str = "1 D",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        qualify: bool = True,
        max_requests: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch 1-minute historical candles.

        Pass either a pre-built ``ib_insync.Contract`` or enough fields to build
        one.  With ``start`` set, the method pages backward in 1-day chunks from
        ``end`` and returns one deduplicated DataFrame covering the requested
        window.  Without ``start``, it makes one IBKR request using ``duration``.
        """
        self._ensure_connected()

        if contract is None:
            if symbol is None:
                raise ValueError("Pass either contract= or symbol=.")
            contract = self.build_contract(
                symbol=symbol,
                security_type=security_type,
                exchange=exchange,
                currency=currency,
                primary_exchange=primary_exchange,
                last_trade_date_or_contract_month=last_trade_date_or_contract_month,
            )

        if qualify:
            contract = self.qualify_contract(contract)

        end_ts = _as_utc_timestamp(end or pd.Timestamp.now(tz="UTC"))

        if start is None:
            return self._request_1min_chunk(
                contract=contract,
                end=end_ts,
                duration=duration,
                what_to_show=what_to_show,
                use_rth=use_rth,
            )

        start_ts = _as_utc_timestamp(start)
        if start_ts >= end_ts:
            return pd.DataFrame(columns=_CANDLE_COLUMNS).rename_axis("open_time")

        chunks: list[pd.DataFrame] = []
        cursor = end_ts
        empty_step = pd.Timedelta(days=1)

        for _ in range(max_requests):
            if cursor <= start_ts:
                break

            frame = self._request_1min_chunk(
                contract=contract,
                end=cursor,
                duration="1 D",
                what_to_show=what_to_show,
                use_rth=use_rth,
            )

            if frame.empty:
                cursor -= empty_step
                continue

            chunks.append(frame)
            oldest = frame.index.min()
            if oldest <= start_ts:
                break

            next_cursor = oldest - pd.Timedelta(milliseconds=1)
            if next_cursor >= cursor:
                break
            cursor = next_cursor

        if not chunks:
            return pd.DataFrame(columns=_CANDLE_COLUMNS).rename_axis("open_time")

        out = pd.concat(chunks).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]
        return out
