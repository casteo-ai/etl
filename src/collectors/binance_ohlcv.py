"""Collect Binance 1-minute OHLCV for a local asset universe.

The universe is read from ``data/universe/binance.csv``. Each asset prefers a
USDC quote and falls back to USDT only when the USDC pair is unavailable. Data
is stored under:

    data/binance/ohlcv/{BASE_QUOTE}/{YYYYMMDD}.csv.gz

The date in the filename is the UTC candle date of the data in that file.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from src.feeds.binance import BinanceClient

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE_PATH = "/home/poucho/code/etl/data/universe/binance.csv"
DEFAULT_DATA_DIR = "/home/poucho/code/etl/data/binance/ohlcv"
DEFAULT_QUOTE = "USDC"
DEFAULT_FALLBACK_QUOTE = "USDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_YEARS = 5
DEFAULT_WORKERS = 4

MANIFEST_COLUMNS = [
    "asset",
    "symbol",
    "day",
    "path",
    "rows",
    "status",
]


def _normalise_asset(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    asset = str(value).strip().upper()
    return asset or None


def _utc_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


class BinanceOhlcvCollector:
    """Download daily 1-minute Binance OHLCV files for an asset universe."""

    def __init__(
        self,
        client: Optional[BinanceClient] = None,
        universe_path: Union[str, Path] = DEFAULT_UNIVERSE_PATH,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        quote: str = DEFAULT_QUOTE,
        fallback_quote: Optional[str] = DEFAULT_FALLBACK_QUOTE,
        timeframe: str = DEFAULT_TIMEFRAME,
        years: int = DEFAULT_YEARS,
        workers: int = DEFAULT_WORKERS,
        client_factory: Optional[Callable[[], BinanceClient]] = None,
    ) -> None:
        self.universe_path = Path(universe_path)
        self.data_dir = Path(data_dir)
        self.quote = quote.strip().upper()
        self.fallback_quote = fallback_quote.strip().upper() if fallback_quote else None
        self.timeframe = timeframe
        self.years = int(years)
        self.workers = max(1, int(workers))
        self._client_factory = client_factory
        self._owns_client = client is None
        self.client = client if client is not None else self._new_client()

    def _new_client(self) -> BinanceClient:
        if self._client_factory is not None:
            return self._client_factory()
        return BinanceClient(data_dir=self.data_dir)

    # ------------------------------------------------------------------
    # Universe / paths
    # ------------------------------------------------------------------
    def load_universe(self) -> pd.DataFrame:
        """Load the asset universe as ``asset, symbol`` rows."""
        data = pd.read_csv(self.universe_path)
        column = "Symbol" if "Symbol" in data.columns else data.columns[0]
        assets = [_normalise_asset(value) for value in data[column]]
        quote_assets = {quote for quote in (self.quote, self.fallback_quote) if quote}
        assets = sorted({asset for asset in assets if asset and asset not in quote_assets})
        rows = [{"asset": asset, "symbol": f"{asset}/{self.quote}"} for asset in assets]
        return pd.DataFrame(rows, columns=["asset", "symbol"])

    def _quote_candidates(self) -> tuple[str, ...]:
        candidates = [self.quote]
        if self.fallback_quote and self.fallback_quote not in candidates:
            candidates.append(self.fallback_quote)
        return tuple(candidates)

    def _resolve_symbol(self, asset: str, available: set[str]) -> Optional[str]:
        for quote in self._quote_candidates():
            symbol = f"{asset}/{quote}"
            if symbol in available:
                return symbol
        return None

    @staticmethod
    def _quote_from_symbol(symbol: str) -> str:
        return symbol.split("/")[-1].upper() if "/" in symbol else DEFAULT_QUOTE

    def _pair_code(self, asset: str, quote: Optional[str] = None) -> str:
        return f"{asset}_{quote or self.quote}"

    def _day_path(self, asset: str, day: pd.Timestamp, quote: Optional[str] = None) -> Path:
        return self.data_dir / self._pair_code(asset, quote) / f"{day.strftime('%Y%m%d')}.csv.gz"

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------
    def _available_symbols(self, reload: bool = False) -> set[str]:
        markets = self.client.exchange.load_markets(reload)
        return set(markets)

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def _fetch_day(
        self,
        symbol: str,
        day: pd.Timestamp,
        until: pd.Timestamp,
        force: bool,
        verbose: bool,
    ) -> pd.DataFrame:
        return self.client.fetch_ohlcv_daily_gzip(
            symbol=symbol,
            timeframe=self.timeframe,
            day=day,
            until=until,
            data_dir=self.data_dir,
            force=force,
            verbose=verbose,
        )

    def _month_starts(self, start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DatetimeIndex:
        start_month = pd.Timestamp(year=start_day.year, month=start_day.month, day=1, tz="UTC")
        end_month = pd.Timestamp(year=end_day.year, month=end_day.month, day=1, tz="UTC")
        return pd.date_range(start=start_month, end=end_month, freq="MS", tz="UTC")

    def _prefill_monthly_archives(
        self,
        symbol: str,
        asset: str,
        quote: str,
        start_day: pd.Timestamp,
        end_day: pd.Timestamp,
        until_ts: pd.Timestamp,
        force: bool,
    ) -> dict[pd.Timestamp, str]:
        """Use monthly bulk archives for fully missing historical months."""
        handled: dict[pd.Timestamp, str] = {}
        current_month = pd.Timestamp.now("UTC").replace(day=1).floor("D")

        for month_start in self._month_starts(start_day, end_day):
            month_end = month_start + pd.DateOffset(months=1)
            if month_start >= current_month:
                continue

            first_day = max(start_day, month_start)
            last_day = min(end_day, month_end - pd.Timedelta(days=1))
            if last_day < first_day:
                continue

            days = list(pd.date_range(start=first_day, end=last_day, freq="D", tz="UTC"))
            # Existing months may have partial daily files; leave those to the
            # daily gap-repair path instead of replacing them blindly.
            if not force and any(self._day_path(asset, day, quote).exists() for day in days):
                continue

            result = self.client.fetch_ohlcv_monthly_gzip(
                symbol=symbol,
                month=month_start,
                start=first_day,
                until=min(until_ts, month_end),
                timeframe=self.timeframe,
                data_dir=self.data_dir,
                force=force,
            )
            status = result.get("status")
            if status not in {"ok", "missing"}:
                continue

            written_days = len(result.get("written_days") or [])
            logger.info(
                "%s monthly archive %s: %s (%s rows, %s day files)",
                symbol,
                month_start.strftime("%Y-%m"),
                status,
                result.get("rows", 0),
                written_days,
            )
            for day in days:
                handled[day] = "archive-prefilled" if self._day_path(asset, day, quote).exists() else "empty"

        return handled

    def collect_asset(
        self,
        asset: str,
        *,
        start: Optional[Union[str, pd.Timestamp]] = None,
        until: Optional[Union[str, pd.Timestamp]] = None,
        force: bool = False,
        verbose: bool = True,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Collect one asset's resolved Binance quote 1-minute OHLCV history."""
        asset = _normalise_asset(asset)
        if asset is None:
            return pd.DataFrame(columns=MANIFEST_COLUMNS)

        symbol = symbol or f"{asset}/{self.quote}"
        quote = self._quote_from_symbol(symbol)
        until_ts = _utc_timestamp(until) if until is not None else pd.Timestamp.now("UTC")
        start_ts = _utc_timestamp(start) if start is not None else until_ts - pd.DateOffset(years=self.years)
        start_day = start_ts.floor("D")
        end_day = until_ts.floor("D")
        if until_ts == end_day and until_ts > start_ts:
            end_day -= pd.Timedelta(days=1)
        if end_day < start_day:
            return pd.DataFrame(columns=MANIFEST_COLUMNS)

        rows: list[dict] = []
        archive_day_status = self._prefill_monthly_archives(
            symbol,
            asset,
            quote,
            start_day,
            end_day,
            until_ts,
            force,
        )
        for day in pd.date_range(start=start_day, end=end_day, freq="D", tz="UTC"):
            path = self._day_path(asset, day, quote)
            if day in archive_day_status:
                status = archive_day_status[day]
                rows.append(
                    {
                        "asset": asset,
                        "symbol": symbol,
                        "day": day.strftime("%Y-%m-%d"),
                        "path": str(path),
                        "rows": None if status == "archive-prefilled" else 0,
                        "status": status,
                    }
                )
                continue

            existed_before = path.exists()
            try:
                data = self._fetch_day(symbol, day, until_ts, force=force, verbose=verbose)
                if data.empty:
                    status = "empty"
                    row_count = 0
                elif existed_before and not force and data.attrs.get("downloaded_rows", 0) == 0:
                    status = "exists"
                    row_count = len(data)
                elif existed_before and not force:
                    status = "updated"
                    row_count = len(data)
                else:
                    status = "written"
                    row_count = len(data)
            except Exception as exc:  # noqa: BLE001 - one pair/day must not abort the run
                logger.warning("Failed to collect %s for %s: %s", symbol, day.date(), exc)
                status = f"error: {exc}"
                row_count = 0

            rows.append(
                {
                    "asset": asset,
                    "symbol": symbol,
                    "day": day.strftime("%Y-%m-%d"),
                    "path": str(path),
                    "rows": row_count,
                    "status": status,
                }
            )
        return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    def _missing_market_frame(self, asset: str, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "asset": asset,
                    "symbol": symbol,
                    "day": None,
                    "path": None,
                    "rows": 0,
                    "status": "missing-market",
                }
            ],
            columns=MANIFEST_COLUMNS,
        )

    def _collect_asset_isolated(
        self,
        asset: str,
        symbol: str,
        *,
        start: Optional[Union[str, pd.Timestamp]],
        until: Optional[Union[str, pd.Timestamp]],
        force: bool,
        verbose: bool,
    ) -> pd.DataFrame:
        """Collect one asset using a worker-local client when possible."""
        if not self._owns_client and self._client_factory is None:
            return self.collect_asset(
                asset,
                symbol=symbol,
                start=start,
                until=until,
                force=force,
                verbose=verbose,
            )

        client = self._new_client()
        try:
            worker = BinanceOhlcvCollector(
                client=client,
                universe_path=self.universe_path,
                data_dir=self.data_dir,
                quote=self.quote,
                fallback_quote=self.fallback_quote,
                timeframe=self.timeframe,
                years=self.years,
                workers=1,
                client_factory=self._client_factory,
            )
            return worker.collect_asset(
                asset,
                symbol=symbol,
                start=start,
                until=until,
                force=force,
                verbose=verbose,
            )
        finally:
            client.close()

    def collect_all(
        self,
        *,
        start: Optional[Union[str, pd.Timestamp]] = None,
        until: Optional[Union[str, pd.Timestamp]] = None,
        force: bool = False,
        reload_markets: bool = False,
        verbose: bool = True,
        workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """Collect every asset, preferring USDC and falling back to USDT."""
        universe = self.load_universe()
        available = self._available_symbols(reload=reload_markets)
        worker_count = max(1, int(workers if workers is not None else self.workers))
        if worker_count > 1 and not self._owns_client and self._client_factory is None:
            logger.info("Using 1 worker because this collector was created with a shared client")
            worker_count = 1

        rows: list[tuple[int, pd.DataFrame]] = []
        pending: list[tuple[int, str, str]] = []

        for idx, item in universe.iterrows():
            asset = item["asset"]
            symbol = self._resolve_symbol(asset, available)
            if symbol is None:
                primary_symbol = item["symbol"]
                rows.append((idx, self._missing_market_frame(asset, primary_symbol)))
                logger.info("Skipping %s: market not found", primary_symbol)
                continue
            pending.append((idx, asset, symbol))

        fetch_verbose = verbose and worker_count == 1
        if worker_count == 1:
            for idx, asset, symbol in pending:
                logger.info("Collecting %s", symbol)
                rows.append(
                    (
                        idx,
                        self.collect_asset(
                            asset,
                            symbol=symbol,
                            start=start,
                            until=until,
                            force=force,
                            verbose=fetch_verbose,
                        ),
                    )
                )
        else:
            logger.info("Collecting %d Binance OHLCV asset(s) with %d workers", len(pending), worker_count)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        self._collect_asset_isolated,
                        asset,
                        symbol,
                        start=start,
                        until=until,
                        force=force,
                        verbose=fetch_verbose,
                    ): (idx, asset, symbol)
                    for idx, asset, symbol in pending
                }
                for future in as_completed(futures):
                    idx, asset, symbol = futures[future]
                    logger.info("Finished %s", symbol)
                    try:
                        rows.append((idx, future.result()))
                    except Exception as exc:  # noqa: BLE001 - one asset must not abort the run
                        logger.warning("Failed to collect %s: %s", symbol, exc)
                        rows.append(
                            (
                                idx,
                                pd.DataFrame(
                                    [
                                        {
                                            "asset": asset,
                                            "symbol": symbol,
                                            "day": None,
                                            "path": None,
                                            "rows": 0,
                                            "status": f"error: {exc}",
                                        }
                                    ],
                                    columns=MANIFEST_COLUMNS,
                                ),
                            )
                        )

        if not rows:
            return pd.DataFrame(columns=MANIFEST_COLUMNS)
        rows.sort(key=lambda item: item[0])
        return pd.concat([frame for _, frame in rows], ignore_index=True)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "BinanceOhlcvCollector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect 1-minute Binance OHLCV for data/universe/binance.csv against USDC.",
    )
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE_PATH, help=f"Universe CSV. Default: {DEFAULT_UNIVERSE_PATH}")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help=f"Output root. Default: {DEFAULT_DATA_DIR}")
    parser.add_argument("--quote", default=DEFAULT_QUOTE, help=f"Quote asset. Default: {DEFAULT_QUOTE}")
    parser.add_argument("--fallback-quote", default=DEFAULT_FALLBACK_QUOTE, help=f"Fallback quote when --quote is unavailable. Default: {DEFAULT_FALLBACK_QUOTE}.")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help=f"CCXT timeframe. Default: {DEFAULT_TIMEFRAME}")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help=f"Years back to collect. Default: {DEFAULT_YEARS}")
    parser.add_argument("--start", default=None, help="Override start timestamp/date.")
    parser.add_argument("--until", default=None, help="Override end timestamp/date.")
    parser.add_argument("--asset", default=None, help="Collect one asset instead of the full universe.")
    parser.add_argument("--force", action="store_true", help="Rewrite files that already exist.")
    parser.add_argument("--reload-markets", action="store_true", help="Reload Binance market metadata first.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Parallel asset workers. Default: {DEFAULT_WORKERS}.")
    parser.add_argument("--quiet", action="store_true", help="Disable Binance fetch progress output.")
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    with BinanceOhlcvCollector(
        universe_path=args.universe,
        data_dir=args.data_dir,
        quote=args.quote,
        fallback_quote=args.fallback_quote,
        timeframe=args.timeframe,
        years=args.years,
        workers=args.workers,
    ) as collector:
        if args.asset:
            available = collector._available_symbols(reload=args.reload_markets)
            asset = _normalise_asset(args.asset)
            symbol = collector._resolve_symbol(asset, available) if asset else None
            if symbol is None:
                symbol = f"{asset}/{collector.quote}" if asset else args.asset
                manifest = collector._missing_market_frame(asset or args.asset, symbol)
            else:
                logger.info("Collecting %s", symbol)
                manifest = collector.collect_asset(
                    args.asset,
                    symbol=symbol,
                    start=args.start,
                    until=args.until,
                    force=args.force,
                    verbose=not args.quiet,
                )
        else:
            manifest = collector.collect_all(
                start=args.start,
                until=args.until,
                force=args.force,
                reload_markets=args.reload_markets,
                verbose=not args.quiet,
                workers=args.workers,
            )

    written = int((manifest["status"] == "written").sum()) if not manifest.empty else 0
    errors = int(manifest["status"].astype(str).str.startswith("error:").sum()) if not manifest.empty else 0
    logger.info("Done: %d rows in manifest, %d files written, %d errors", len(manifest), written, errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
