"""Pytest bootstrap: ensure the repository root is importable as `src.*`.

`src/feeds/deribit.py` does `from src.feeds.fred import FedClient`, so the repo
root must be on sys.path when tests run regardless of the invocation directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
