"""
Entrypoint for the Deribit option-surface collector.

Modes (see :mod:`src.collectors.runner`):

* ``once`` — single snapshot then exit (for cron, one process per minute).
* ``loop`` — daemon, one snapshot every ``--interval`` seconds.

Examples
--------
One-shot (for cron)::

    python -m src.collectors.run_option_surface --mode once

Long-running daemon, one snapshot per minute::

    python -m src.collectors.run_option_surface --mode loop --interval 60

Crontab line for one snapshot every minute::

    * * * * * cd /home/poucho/code/etl && \
        /path/to/conda/envs/etl/bin/python -m src.collectors.run_option_surface \
        --mode once >> /home/poucho/code/etl/logs/option_surface.log 2>&1
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.collectors.option_surface import DEFAULT_DATA_DIR, OptionSurfaceCollector
from src.collectors.runner import run

DESCRIPTION = "Collect Deribit option surfaces into daily per-currency CSVs."


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(OptionSurfaceCollector, DESCRIPTION, DEFAULT_DATA_DIR, argv)


if __name__ == "__main__":
    raise SystemExit(main())
