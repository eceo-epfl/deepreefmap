from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def pyapp_binary() -> str | None:
    """Path of the running PyApp binary, or None in a dev venv.

    PyApp exports ``PYAPP=<binary path>`` (older releases exported ``"1"``).
    """
    value = os.environ.get("PYAPP")
    if value and value != "1" and Path(value).exists():
        return value
    return None


class BinarySwapError(RuntimeError):
    pass


def _is_rocm_build() -> bool:
    # Raw env read: only the binary NAME matters here, and tests set PYAPP
    # to names that need not exist on disk.
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
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (URL is our GH release metadata)
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

    Returns True when the layout is unknown, so a false alarm never triggers a
    needless re-provision.
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


# --- Stale-environment pruning ------------------------------------------------
# Each version's env is a multi-GB directory PyApp never removes, and installer
# reinstalls leave the previous version's env behind. Sweep them on every launch.


def _env_dir_for_prefix(prefix: str | os.PathLike[str]) -> Path:
    # sys.prefix is ``.../<version>/python``; the version dir is its parent.
    return Path(prefix).parent


def prune_stale_envs(current_prefix: str | os.PathLike[str] | None = None) -> list[Path]:
    """Remove old version envs, keeping the running one plus one fallback.

    PyApp lays envs out as ``.../pyapp/deepreefmap/<version>/python``, so the running
    env's siblings are past versions. Outside a PyApp env (a dev venv) only the
    legacy-marker cleanup happens.
    """
    from deepreefmap.paths import env_prune_marker_path

    # Legacy marker from the pre-sweep prune mechanism.
    env_prune_marker_path().unlink(missing_ok=True)

    current = _env_dir_for_prefix(current_prefix or sys.prefix)
    if "pyapp" not in current.parts or not current.is_dir():
        return []
    siblings = [p for p in current.parent.iterdir() if p != current and p.is_dir()]
    siblings.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    removed: list[Path] = []
    # The newest sibling survives as a rollback target. Re-provisioning needs the
    # package index, so on an offline field laptop a deleted env is unrecoverable,
    # and a new version that provisions cleanly can still be broken at runtime.
    for stale in siblings[1:]:
        shutil.rmtree(stale, ignore_errors=True)
        removed.append(stale)
        logger.info("Pruned stale environment %s", stale)
    return removed


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def provision_env(
    binary_path: str | os.PathLike[str],
    line_cb: Callable[[str], None] | None = None,
) -> bool:
    """Provision the new binary's environment via ``self restore``, streaming
    install output to ``line_cb``. Returns False on failure instead of raising
    (the binary is already swapped; the next launch retries provisioning).
    """
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def log(message: str) -> None:
        if line_cb is not None:
            line_cb(message)

    try:
        proc = subprocess.Popen(
            [str(binary_path), "self", "restore"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            creationflags=creationflags,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            # uv redraws progress with \r and ANSI escapes; keep the final
            # segment of each line as the log entry.
            segment = _ANSI_RE.sub("", raw).split("\r")[-1].strip()
            if segment:
                log(segment)
        code = proc.wait()
    except Exception:
        logger.exception("Provisioning failed for %s", binary_path)
        log("Environment preparation failed; it will be retried on next launch.")
        return False
    if code != 0:
        logger.warning("`self restore` exited with %d for %s", code, binary_path)
        log("Environment preparation failed; it will be retried on next launch.")
        return False
    return True


def perform_update(
    release: dict,
    binary_path: Path,
    target_version: str,
    progress_cb: Callable[[int, int], None] | None = None,
    line_cb: Callable[[str], None] | None = None,
) -> None:
    """Download the release's asset, swap it in, then provision its environment.

    The old version's env is swept on the new version's first launch.
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
    log(f"Replacing binary at {binary_path}")
    replace_binary(binary_path, staged)
    log("Preparing the new version's environment…")
    provision_env(binary_path, line_cb=line_cb)
    log("Done. Relaunch to use the new version.")
