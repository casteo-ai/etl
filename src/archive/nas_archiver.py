"""
NasArchiver — move data older than N days from local disk to a NAS mirror.

Files under ``source_root`` (default ``/home/poucho/code/etl/data``) that are
older than ``max_age_days`` (default 30) are copied to ``dest_root`` (default
``/mnt/nas/market_data``) preserving the exact relative directory structure,
verified, and only then deleted locally. The NAS is a *mirror* of the local
tree: ``data/deribit/options/.../20260501.csv.gz`` →
``market_data/deribit/options/.../20260501.csv.gz``.

Design / safety
---------------
* **Copy-verify-delete.** Each file is copied to a ``.partial`` temp name, the
  copy is checksum-verified against the source (size + SHA-256), atomically
  renamed into place, and only then removed locally. A crash at any point never
  loses data and never leaves a truncated file that looks complete.
* **Idempotent.** If the destination already exists and matches the source
  checksum (e.g. a previous run copied it but didn't get to the delete), the
  local copy is simply removed.
* **Non-destructive to the NAS.** The archiver only ever writes the paths it
  archives. It never deletes anything on the NAS, so unrelated folders
  (``bitpanda``, ``crypto_archive``, ...) are left untouched.
* **Mount guard.** By default the run aborts unless ``dest_root`` sits on a real
  mountpoint, so a missing NAS mount can't cause files to be "archived" onto the
  local disk under ``/mnt`` and then deleted from their real location.

Age criterion
-------------
For the dated daily files (name is exactly ``YYYYMMDD`` before the extension,
e.g. ``20260501.csv.gz``) the embedded date is used, so rewriting/recompressing
a file doesn't reset its age. For everything else (e.g. Binance
``BTC_USDT_1m.csv``) the file mtime is used — which also means a file still being
appended to today is never archived.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_ROOT = "/home/poucho/code/etl/data"
DEFAULT_DEST_ROOT = "/mnt/nas/market_data"
DEFAULT_MAX_AGE_DAYS = 30

MANIFEST_COLUMNS = [
    "relative_path",
    "size_bytes",
    "source_path",
    "dest_path",
    "status",
]

# Daily snapshot files are named exactly YYYYMMDD before the extension.
_DATE_NAME = re.compile(r"^(\d{8})$")
_CHUNK = 1024 * 1024  # 1 MiB streaming hash


def _file_base(path: Path) -> str:
    """Filename with all extensions stripped (``20260501.csv.gz`` -> ``20260501``)."""
    return path.name.split(".", 1)[0]


def _dated_from_name(path: Path) -> Optional[date]:
    """Parse the embedded ``YYYYMMDD`` date, or None if the name isn't dated."""
    m = _DATE_NAME.match(_file_base(path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_mounted(path: Path) -> bool:
    """True if ``path`` or any parent (other than the filesystem root) is a
    mountpoint — i.e. the NAS is actually mounted."""
    p = path.resolve()
    while True:
        if p.exists() and os.path.ismount(p):
            return str(p) != p.anchor
        if p.parent == p:
            return False
        p = p.parent


class NasArchiver:
    """Move aged-out local data files to a mirrored NAS location.

    Parameters
    ----------
    source_root:
        Local data root to archive from.
    dest_root:
        NAS root to mirror into.
    max_age_days:
        Files older than this are archived. Files exactly this old are kept.
    require_mount:
        When True (default) abort if ``dest_root`` is not on a mountpoint.
    """

    def __init__(
        self,
        source_root: Union[str, Path] = DEFAULT_SOURCE_ROOT,
        dest_root: Union[str, Path] = DEFAULT_DEST_ROOT,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
        require_mount: bool = True,
    ) -> None:
        self.source_root = Path(source_root)
        self.dest_root = Path(dest_root)
        self.max_age_days = int(max_age_days)
        self.require_mount = require_mount

    # ------------------------------------------------------------------
    # Age test
    # ------------------------------------------------------------------
    def _cutoff_date(self, now: Optional[datetime] = None) -> date:
        now = now or datetime.now(timezone.utc)
        return now.date() - timedelta(days=self.max_age_days)

    def _is_old(self, path: Path, cutoff: date) -> bool:
        """True when the file is strictly older than the cutoff date."""
        dated = _dated_from_name(path)
        if dated is not None:
            return dated < cutoff
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()
        return mtime < cutoff

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    @staticmethod
    def _same_file(a: Path, b: Path) -> bool:
        try:
            if a.stat().st_size != b.stat().st_size:
                return False
        except FileNotFoundError:
            return False
        return _sha256(a) == _sha256(b)

    # ------------------------------------------------------------------
    # Per-file move
    # ------------------------------------------------------------------
    def _archive_file(self, src: Path, dest: Path, dry_run: bool) -> str:
        if dry_run:
            return "would-archive"

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Already archived (verified) by an earlier run — just drop the local copy.
        if dest.exists() and self._same_file(src, dest):
            src.unlink()
            return "already-archived"

        tmp = dest.with_name(dest.name + ".partial")
        try:
            shutil.copy2(src, tmp)  # preserves mtime/permissions
            if not self._same_file(src, tmp):
                tmp.unlink(missing_ok=True)
                return "verify-failed"
            os.replace(tmp, dest)  # atomic within the NAS filesystem
        except Exception:
            # Never delete the source if the copy/verify didn't complete.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        src.unlink()
        return "archived"

    # ------------------------------------------------------------------
    # Empty-dir cleanup
    # ------------------------------------------------------------------
    def _prune_empty_dirs(self) -> None:
        """Remove now-empty local subdirectories (never the source root)."""
        for dirpath, _dirnames, _files in os.walk(self.source_root, topdown=False):
            d = Path(dirpath)
            if d == self.source_root:
                continue
            try:
                if not any(d.iterdir()):
                    d.rmdir()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def archive_once(
        self, dry_run: bool = False, now: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Archive every aged-out file once.

        Returns
        -------
        Manifest DataFrame (one row per candidate file) with columns
        ``relative_path, size_bytes, source_path, dest_path, status``. ``status``
        is one of ``archived``, ``already-archived``, ``would-archive``
        (dry-run), ``verify-failed`` or ``error: ...``.
        """
        if not self.source_root.exists():
            logger.warning("Source root %s does not exist; nothing to archive", self.source_root)
            return pd.DataFrame(columns=MANIFEST_COLUMNS)

        if self.require_mount and not _is_mounted(self.dest_root):
            raise RuntimeError(
                f"Destination {self.dest_root} is not on a mounted filesystem — "
                "refusing to archive (pass require_mount=False to override)."
            )

        cutoff = self._cutoff_date(now)
        rows: list[dict] = []

        for src in sorted(self.source_root.rglob("*")):
            if not src.is_file():
                continue
            try:
                if not self._is_old(src, cutoff):
                    continue
            except OSError:
                continue

            rel = src.relative_to(self.source_root)
            dest = self.dest_root / rel
            try:
                size = src.stat().st_size
            except OSError:
                size = None

            try:
                status = self._archive_file(src, dest, dry_run)
            except Exception as exc:  # noqa: BLE001 — log and keep going
                logger.exception("Failed to archive %s", src)
                status = f"error: {exc}"

            rows.append(
                {
                    "relative_path": str(rel),
                    "size_bytes": size,
                    "source_path": str(src),
                    "dest_path": str(dest),
                    "status": status,
                }
            )
            if status in ("archived", "already-archived"):
                logger.info("%s -> %s (%s)", rel, dest, status)

        if not dry_run:
            self._prune_empty_dirs()

        df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        moved = int((df["status"].isin(["archived", "already-archived"])).sum()) if not df.empty else 0
        logger.info(
            "Archive pass complete: %d candidate(s), %d moved, cutoff < %s",
            len(df),
            moved,
            cutoff,
        )
        return df
