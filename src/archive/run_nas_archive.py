"""
Entrypoint for the NAS archiver — runs one pass and exits.

Schedule this hourly to mirror new/updated local data to the NAS. Local files
older than the configured retention window are removed only after their NAS copy
is verified.

Examples
--------
Manual one-shot::

    python -m src.archive.run_nas_archive

Preview without moving anything::

    python -m src.archive.run_nas_archive --dry-run

Hourly via systemd timer (see deploy notes) or cron::

    5 * * * * cd /home/poucho/code/etl && \
        /home/poucho/miniconda3/envs/ml/bin/python -m src.archive.run_nas_archive \
        >> /home/poucho/code/etl/logs/nas_archive.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

from src.archive.nas_archiver import (
    DEFAULT_DEST_ROOT,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_SOURCE_ROOT,
    NasArchiver,
)

logger = logging.getLogger("src.archive.run_nas_archive")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mirror local data to the NAS and prune old verified local copies.",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE_ROOT, help=f"Local data root. Default: {DEFAULT_SOURCE_ROOT}")
    parser.add_argument("--nas", default=DEFAULT_DEST_ROOT, help=f"NAS mirror root. Default: {DEFAULT_DEST_ROOT}")
    parser.add_argument(
        "--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
        help=f"Prune local files older than this many days after NAS verification. Default: {DEFAULT_MAX_AGE_DAYS}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would move without copying or deleting.")
    parser.add_argument(
        "--no-require-mount", action="store_true",
        help="Do not require the NAS path to be a mountpoint (unsafe; for testing).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    archiver = NasArchiver(
        source_root=args.source,
        dest_root=args.nas,
        max_age_days=args.max_age_days,
        require_mount=not args.no_require_mount,
    )
    df = archiver.archive_once(dry_run=args.dry_run)

    moved = int(df["status"].str.contains("copied|replaced", regex=True).sum()) if not df.empty else 0
    failed = int(df["status"].str.startswith(("verify-failed", "error")).sum()) if not df.empty else 0
    logger.info("Done: %d candidates, %d moved, %d failed%s", len(df), moved, failed, " (dry-run)" if args.dry_run else "")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
