from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class BinarySwapError(RuntimeError):
    pass


def _is_rocm_build() -> bool:
    pyapp = os.environ.get("PYAPP")
    if pyapp and "rocm" in Path(pyapp).name:
        return True
    try:
        import torch

        # torch.version.hip is a version string on ROCm wheels and None on
        # CUDA/CPU wheels. The attribute always exists, so hasattr would match
        # every build and mislabel CUDA machines as ROCm.
        return torch.version.hip is not None
    except Exception:
        return False


def _cuda_variant_suffix() -> str:
    """``-cu130`` for a CUDA 13 build, else ``""``. Keeps an in-app update on its variant."""
    try:
        import torch

        cuda = getattr(torch.version, "cuda", None)
        if cuda:
            return "-cu130" if cuda.split(".")[0] == "13" else ""
    except Exception:
        pass
    pyapp = os.environ.get("PYAPP")
    if pyapp and "cu130" in Path(pyapp).name:
        return "-cu130"
    return ""


def resolve_asset_name(platform: str | None = None) -> str:
    p = (platform or sys.platform).lower()
    if p.startswith("linux"):
        if _is_rocm_build():
            return "deepreefmap-linux-x64-rocm"
        return f"deepreefmap-linux-x64{_cuda_variant_suffix()}"
    if p.startswith("win"):
        return f"deepreefmap-windows-x64{_cuda_variant_suffix()}.exe"
    if p.startswith("darwin"):
        return "deepreefmap-macos-arm64"
    raise BinarySwapError(f"No binary asset is built for platform {p!r}")


def match_asset_url(release: dict, asset_name: str) -> str | None:
    # Release assets carry a version label (deepreefmap-linux-x64-1.2.0[.exe])
    # while resolve_asset_name yields the bare platform name, so accept both.
    candidates = {asset_name}
    tag = str(release.get("tag_name", "")).lstrip("v")
    if tag:
        stem, dot, ext = asset_name.rpartition(".")
        if dot:
            candidates.add(f"{stem}-{tag}.{ext}")
        else:
            candidates.add(f"{asset_name}-{tag}")
    for asset in release.get("assets", []):
        if asset.get("name") in candidates:
            url = asset.get("browser_download_url")
            if url:
                return str(url)
    return None


def find_asset_url(release: dict, asset_name: str) -> str:
    url = match_asset_url(release, asset_name)
    if url is None:
        raise BinarySwapError(
            f"Release {release.get('tag_name', '?')} has no {asset_name!r} asset. "
            "This release may pre-date binary distribution. Pick a newer version."
        )
    return url


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


# --- Environment health / self-heal -----------------------------------------
# PyApp only checks that a version's env exists, not that it is intact, so an OS
# update or antivirus that deletes files inside it leaves a broken env in place.


def env_is_healthy(purelib: str | os.PathLike[str] | None = None) -> bool:
    """True if the heavy native deps look intact in the active environment.

    A stat-level check of the packages an OS update or antivirus is most likely
    to break. Returns True when the layout is unknown, so a false alarm never
    triggers a needless re-provision.
    """
    try:
        base = Path(purelib) if purelib is not None else Path(sysconfig.get_path("purelib"))
    except Exception:
        return True
    for pkg in ("torch", "PySide6"):
        if not (base / pkg).is_dir():
            return False
    torch_lib = base / "torch" / "lib"
    if torch_lib.is_dir() and not any(torch_lib.iterdir()):
        return False
    return True


def self_restore(binary_path: str | os.PathLike[str]) -> bool:
    """Reinstall the project into the env from the shared uv cache via PyApp's
    ``self restore``. Returns True on success."""
    try:
        subprocess.run([str(binary_path), "self", "restore"], check=True)
        return True
    except Exception:
        logger.exception("`self restore` failed for %s", binary_path)
        return False


# --- Previous-environment pruning -------------------------------------------
# Each version's env is a multi-GB directory PyApp never removes. Record the
# running env before an update swaps the binary, then drop it on the next launch.


def _env_dir_for_prefix(prefix: str | os.PathLike[str]) -> Path:
    # sys.prefix is ``.../<version>/python``; the version dir is its parent.
    return Path(prefix).parent


def record_previous_env(prefix: str | os.PathLike[str] | None = None) -> None:
    """Record the active version's env dir for removal after the next launch."""
    from deepreefmap.paths import env_prune_marker_path

    env_dir = _env_dir_for_prefix(prefix or sys.prefix)
    version = ""
    try:
        import importlib.metadata

        version = importlib.metadata.version("deepreefmap")
    except Exception:
        pass
    marker = env_prune_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"env_dir": str(env_dir), "version": version}))


def prune_previous_env(current_prefix: str | os.PathLike[str] | None = None) -> Path | None:
    """Remove the env recorded by a prior update, unless it is the active one.

    Runs at startup once the new version has provisioned. Never deletes the
    running env or anything outside a PyApp data dir. Returns the removed dir,
    or None if there was nothing to prune.
    """
    from deepreefmap.paths import env_prune_marker_path

    marker = env_prune_marker_path()
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text())
        old = data.get("env_dir")
    except Exception:
        marker.unlink(missing_ok=True)
        return None

    removed: Path | None = None
    current_dir = _env_dir_for_prefix(current_prefix or sys.prefix)
    if old:
        old_path = Path(old)
        if (
            old_path != current_dir
            and "pyapp" in old_path.parts
            and old_path.is_dir()
        ):
            shutil.rmtree(old_path, ignore_errors=True)
            removed = old_path
            logger.info("Pruned previous environment %s", old_path)
    marker.unlink(missing_ok=True)
    return removed


def perform_update(
    release: dict,
    binary_path: Path,
    target_version: str,
    progress_cb: Callable[[int, int], None] | None = None,
    line_cb: Callable[[str], None] | None = None,
) -> None:
    """Download the release's binary asset and swap it in place.

    Qt-free so the GUI worker, unit tests, and the e2e harness share one path.
    Records the running env for pruning just before the swap.
    """
    binary_path = Path(binary_path)

    def log(message: str) -> None:
        if line_cb is not None:
            line_cb(message)

    asset_name = resolve_asset_name()
    log(f"Looking up {asset_name} in release {release.get('tag_name')}…")
    url = find_asset_url(release, asset_name)
    log(f"Downloading {url}")
    staged = binary_path.with_name(binary_path.name + ".new")
    if staged.exists():
        staged.unlink()
    download_to(url, staged, progress_cb=progress_cb)
    log(f"Verifying download ({staged.stat().st_size} bytes)…")
    record_previous_env()
    log(f"Replacing binary at {binary_path}")
    replace_binary(binary_path, staged)
    log("Done. Relaunch to use the new version.")
