"""
Generic scheduling + CLI for the Deribit snapshot collectors.

Both the option-surface and forward-curve collectors share the same runner: a
``once`` mode (single snapshot then exit, for cron) and a ``loop`` mode (a
daemon that snapshots every ``--interval`` seconds, aligned to the wall-clock
boundary). The thin ``run_*`` entrypoint modules just bind a collector class to
:func:`run`.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Callable, Optional, Sequence, Type

from src.collectors.base import BaseSnapshotCollector
from src.feeds.deribit import DeribitClient

logger = logging.getLogger("src.collectors.runner")


def _seconds_until_next(interval: float, clock: Callable[[], float] = time.time) -> float:
    """Seconds to sleep so the next wake aligns to an ``interval`` boundary.

    With ``interval=60`` this returns the time until the top of the next
    minute, keeping snapshots on tidy ``HH:MM:00`` timestamps.
    """
    if interval <= 0:
        return 0.0
    remainder = clock() % interval
    return interval - remainder if remainder else interval


def run_once(collector: BaseSnapshotCollector, reload: bool = True) -> int:
    """Take a single snapshot. Returns the number of rows written."""
    df = collector.collect_once(reload=reload)
    return len(df)


def run_loop(
    collector: BaseSnapshotCollector,
    interval: float = 60.0,
    reload_every: int = 60,
    max_iterations: Optional[int] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> int:
    """Snapshot every ``interval`` seconds until interrupted.

    Parameters
    ----------
    interval:
        Seconds between snapshots. Sleeps are aligned to wall-clock boundaries.
    reload_every:
        Reload the CCXT market map every Nth iteration (and on the first) so
        newly listed expiries are captured without paying the reload cost every
        minute.
    max_iterations:
        Stop after this many snapshots. ``None`` (default) runs forever; the
        bound exists mainly so the loop is testable.
    sleep_fn / clock:
        Injectable for testing.

    Returns
    -------
    The number of iterations actually run.
    """
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        reload = (reload_every > 0) and (iteration % reload_every == 0)
        try:
            collector.collect_once(reload=reload)
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001 — a daemon must survive transient errors
            logger.exception("Snapshot failed; continuing")
        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            break
        sleep_fn(_seconds_until_next(interval, clock))
    return iteration


def build_arg_parser(description: str, default_data_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        choices=("once", "loop"),
        default="loop",
        help="'once' for a single snapshot (cron), 'loop' for a daemon. Default: loop.",
    )
    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help=f"Output directory for the daily CSVs. Default: {default_data_dir}",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between snapshots in loop mode. Default: 60.",
    )
    parser.add_argument(
        "--reload-every",
        type=int,
        default=60,
        help="Reload the market map every Nth loop iteration. Default: 60.",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="In 'once' mode, skip the market-map reload before snapshotting.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Write plain .csv instead of gzip-compressed .csv.gz (default).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...). Default: INFO.",
    )
    return parser


def run(
    collector_cls: Type[BaseSnapshotCollector],
    description: str,
    default_data_dir: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Parse args and drive ``collector_cls`` in once/loop mode."""
    args = build_arg_parser(description, default_data_dir).parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    with DeribitClient() as client:
        collector = collector_cls(
            client=client, data_dir=args.data_dir, compress=not args.no_compress
        )
        if args.mode == "once":
            run_once(collector, reload=not args.no_reload)
        else:
            logger.info(
                "Starting %s loop: interval=%ss reload_every=%s -> %s",
                collector.SNAPSHOT_LABEL,
                args.interval,
                args.reload_every,
                args.data_dir,
            )
            try:
                run_loop(
                    collector,
                    interval=args.interval,
                    reload_every=args.reload_every,
                )
            except KeyboardInterrupt:
                logger.info("Interrupted; shutting down %s loop", collector.SNAPSHOT_LABEL)
    return 0
