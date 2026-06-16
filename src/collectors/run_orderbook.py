"""
Entrypoint for the Binance order-book collector (Deribit derivative universe).

Captures the Binance spot order book once per minute for every asset that has
Deribit options or term futures, appending one wide row per asset to
``{data_dir}/{ASSET}.csv``.

Modes (scheduling reused from :mod:`src.collectors.runner`):

* ``once`` — single snapshot then exit (cron).
* ``loop`` — daemon, one snapshot every ``--interval`` seconds.

Examples
--------
    python -m src.collectors.run_orderbook --mode loop --interval 60
    python -m src.collectors.run_orderbook --mode once
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

from src.collectors.orderbook import (
    DEFAULT_DATA_DIR,
    DEFAULT_DEPTH,
    DEFAULT_QUOTE,
    OrderBookCollector,
)
from src.collectors.runner import run_loop, run_once
from src.feeds.binance import BinanceClient
from src.feeds.deribit import DeribitClient

logger = logging.getLogger("src.collectors.run_orderbook")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Binance order books (top-N + metrics) for the Deribit derivative universe.",
    )
    parser.add_argument("--mode", choices=("once", "loop"), default="loop",
                        help="'once' for a single snapshot (cron), 'loop' for a daemon. Default: loop.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help=f"Output dir. Default: {DEFAULT_DATA_DIR}")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help=f"Book levels per side. Default: {DEFAULT_DEPTH}.")
    parser.add_argument("--quote", default=DEFAULT_QUOTE, help=f"Binance spot quote currency. Default: {DEFAULT_QUOTE}.")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between snapshots (loop). Default: 60.")
    parser.add_argument("--reload-every", type=int, default=60,
                        help="Re-resolve the asset universe every Nth loop iteration. Default: 60.")
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    with DeribitClient() as deribit, BinanceClient() as binance:
        collector = OrderBookCollector(
            deribit_client=deribit,
            binance_client=binance,
            data_dir=args.data_dir,
            depth=args.depth,
            quote=args.quote,
        )
        if args.mode == "once":
            run_once(collector, reload=True)
        else:
            logger.info("Starting order-book loop: interval=%ss depth=%s quote=%s -> %s",
                        args.interval, args.depth, args.quote, args.data_dir)
            try:
                run_loop(collector, interval=args.interval, reload_every=args.reload_every)
            except KeyboardInterrupt:
                logger.info("Interrupted; shutting down order-book loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
