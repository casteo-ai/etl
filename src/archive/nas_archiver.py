"""
NasArchiver — mirror local market data to NAS, then prune aged local files.

Every run scans files under ``source_root`` (default
``/home/poucho/code/etl/data``) and mirrors new or updated local data into
``dest_root`` (default ``/mnt/nas/market_data``), preserving the exact relative
directory structure. The NAS is a durable mirror of the local tree:
``data/deribit/options/.../20260501.csv.gz`` →
``market_data/deribit/options/.../20260501.csv.gz``.

Design / safety
---------------
* **Copy-verify-promote.** Each file is copied to a ``.partial`` temp name,
  checksum-verified against the source (size + SHA-256), then atomically
  promoted into place. A crash never leaves a truncated file that looks
  complete.
* **Hourly mirror.** Missing NAS files are copied. Existing NAS files are
  replaced only when the local file was modified later than the NAS copy.
* **Backup on shrink.** If a newer local file is smaller than the NAS copy, the
  current NAS file is first preserved as ``file.ext.bckp`` before replacement.
  Existing distinct backups are never overwritten.
* **Non-destructive to the NAS.** The archiver never deletes data from the NAS.
  Old-file cleanup only removes verified local copies.
* **Mount guard.** By default the run aborts unless ``dest_root`` sits on a real
  mountpoint, so a missing NAS mount can't cause files to be "archived" onto the
  local disk under ``/mnt`` and then deleted from their real location.

Age criterion
-------------
For the dated daily files (name is exactly ``YYYYMMDD`` before the extension,
e.g. ``20260501.csv.gz``) the embedded date is used, so rewriting/recompressing
a file doesn't reset its age. For everything else (e.g. Binance
``BTC_USDT_1m.csv``) the file mtime is used — which also means a file still being
appended to today is never pruned locally.
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
    """Mirror local data files to NAS and prune aged-out local copies.

    Parameters
    ----------
    source_root:
        Local data root to archive from.
    dest_root:
        NAS root to mirror into.
    max_age_days:
        Local files older than this are removed after the NAS copy is verified.
        Files exactly this old are kept locally.
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
    # Per-file copy / backup
    # ------------------------------------------------------------------
    def _copy_verified(self, src: Path, dest: Path) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".partial")
        try:
            shutil.copy2(src, tmp)  # preserves mtime/permissions
            if not self._same_file(src, tmp):
                tmp.unlink(missing_ok=True)
                return "verify-failed"
            os.replace(tmp, dest)  # atomic within the NAS filesystem
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return "copied"

    def _backup_path(self, dest: Path) -> Path:
        backup = dest.with_name(dest.name + ".bckp")
        if not backup.exists():
            return backup
        if self._same_file(dest, backup):
            return backup

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        candidate = dest.with_name(dest.name + f".bckp.{stamp}")
        counter = 1
        while candidate.exists():
            candidate = dest.with_name(dest.name + f".bckp.{stamp}.{counter}")
            counter += 1
        return candidate

    def _backup_nas_file(self, dest: Path) -> Path:
        backup = self._backup_path(dest)
        if backup.exists() and self._same_file(dest, backup):
            return backup

        tmp = backup.with_name(backup.name + ".partial")
        try:
            shutil.copy2(dest, tmp)
            if not self._same_file(dest, tmp):
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"backup verification failed for {dest}")
            os.replace(tmp, backup)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return backup

    def _mirror_file(self, src: Path, dest: Path, dry_run: bool) -> str:
        """Mirror one local file to NAS without deleting either side."""
        if dest.exists() and self._same_file(src, dest):
            return "up-to-date"

        if not dest.exists():
            if dry_run:
                return "would-copy"
            return self._copy_verified(src, dest)

        src_stat = src.stat()
        dest_stat = dest.stat()
        if src_stat.st_mtime <= dest_stat.st_mtime:
            return "nas-newer-or-same"

        if dry_run:
            return (
                "would-backup-and-replace"
                if src_stat.st_size < dest_stat.st_size
                else "would-replace"
            )

        if src_stat.st_size < dest_stat.st_size:
            self._backup_nas_file(dest)
            status = self._copy_verified(src, dest)
            return "backup-and-replaced" if status == "copied" else status

        status = self._copy_verified(src, dest)
        return "replaced" if status == "copied" else status

    def _prune_local_file(self, src: Path, dest: Path, dry_run: bool) -> Optional[str]:
        """Remove an old local file only when the NAS copy is verified identical."""
        if not dest.exists() or not self._same_file(src, dest):
            return "local-prune-skipped"
        if dry_run:
            return "would-prune-local"
        src.unlink()
        return "local-pruned"

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
        """Mirror local files to NAS and prune aged-out verified local copies.

        Returns
        -------
        Manifest DataFrame (one row per local file) with columns
        ``relative_path, size_bytes, source_path, dest_path, status``. ``status``
        captures the mirror action, with ``+local-pruned`` appended when an old
        verified local file was removed.
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

            rel = src.relative_to(self.source_root)
            dest = self.dest_root / rel
            try:
                size = src.stat().st_size
            except OSError:
                size = None

            try:
                status = self._mirror_file(src, dest, dry_run)
                try:
                    is_old = self._is_old(src, cutoff)
                except OSError:
                    is_old = False
                if is_old and status not in ("verify-failed",):
                    if dry_run and status in (
                        "would-copy",
                        "would-replace",
                        "would-backup-and-replace",
                        "up-to-date",
                    ):
                        prune_status = "would-prune-local"
                    else:
                        prune_status = self._prune_local_file(src, dest, dry_run)
                    if prune_status is not None:
                        status = f"{status}+{prune_status}"
            except Exception as exc:  # noqa: BLE001 — log and keep going
                logger.exception("Failed to mirror %s", src)
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
            logger.info("%s -> %s (%s)", rel, dest, status)

        if not dry_run:
            self._prune_empty_dirs()

        df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        mirrored = int(
            df["status"].str.contains("copied|replaced", regex=True).sum()
        ) if not df.empty else 0
        pruned = int(df["status"].str.contains("local-pruned", regex=False).sum()) if not df.empty else 0
        logger.info(
            "Archive pass complete: %d file(s), %d mirrored, %d local pruned, cutoff < %s",
            len(df),
            mirrored,
            pruned,
            cutoff,
        )
        return df
