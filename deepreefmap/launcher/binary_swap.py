from __future__ import annotations

import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class BinarySwapError(RuntimeError):
    pass


def resolve_asset_name(platform: str | None = None) -> str:
    p = (platform or sys.platform).lower()
    if p.startswith("linux"):
        return "deepreefmap-linux-x64"
    if p.startswith("win"):
        return "deepreefmap-windows-x64.exe"
    raise BinarySwapError(f"No binary asset is built for platform {p!r}")


def find_asset_url(release: dict, asset_name: str) -> str:
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name:
            url = asset.get("browser_download_url")
            if url:
                return str(url)
    raise BinarySwapError(
        f"Release {release.get('tag_name', '?')} has no {asset_name!r} asset. "
        "This release may pre-date binary distribution. Pick a newer version."
    )


def download_to(
    url: str,
    dest_path: Path,
    progress_cb: Callable[[int, int], None] | None = None,
    chunk_size: int = 64 * 1024,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "deepreefmap-updater"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 — URL is from our GH release metadata
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with dest_path.open("wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress_cb is not None:
                    progress_cb(done, total)


def replace_binary(target_path: Path, src_path: Path) -> None:
    if not src_path.exists():
        raise BinarySwapError(f"Source binary missing: {src_path}")
    if sys.platform.startswith("win"):
        backup = target_path.with_suffix(target_path.suffix + ".old")
        if backup.exists():
            try:
                backup.unlink()
            except OSError:
                logger.debug("Could not remove stale backup %s", backup, exc_info=True)
        if target_path.exists():
            os.rename(target_path, backup)
        os.rename(src_path, target_path)
    else:
        os.chmod(src_path, 0o755)
        os.rename(src_path, target_path)


def cleanup_stale_backups(binary_path: Path) -> None:
    if not sys.platform.startswith("win"):
        return
    backup = binary_path.with_suffix(binary_path.suffix + ".old")
    if backup.exists():
        try:
            backup.unlink()
        except OSError:
            logger.debug("Failed to remove %s during startup cleanup", backup, exc_info=True)
