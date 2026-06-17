"""Storage accounting + retention for the field data-management UI.

Non-GUI so it stays unit-testable: disk capacity, per-run sizes, run enumeration (including compacted
runs), a per-run *protect* marker, and an age-based retention policy. Retention deletes whole runs
(each already a single ``.scene.zarr.zip`` once compacted) — it never touches a protected run, and the
caller decides when to apply it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from deepreefmap.io.scene_file import find_scene_file

_PROTECT_MARKER = ".protected"


@dataclass(frozen=True)
class DiskUsage:
    total: int
    used: int
    free: int

    @property
    def used_fraction(self) -> float:
        return 0.0 if self.total == 0 else self.used / self.total


@dataclass(frozen=True)
class RunInfo:
    path: Path
    name: str
    size_bytes: int
    timestamp: float  # epoch seconds (manifest run_timestamp, else manifest mtime)
    mode: str
    compacted: bool
    protected: bool

    def age_days(self, now: float) -> float:
        return max(0.0, (now - self.timestamp) / 86400.0)


def disk_usage(path: Path) -> DiskUsage:
    """Total/used/free bytes of the volume containing ``path`` (nearest existing parent)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    total, used, free = shutil.disk_usage(str(p))
    return DiskUsage(total=total, used=used, free=free)


def dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for f in Path(path).rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def is_protected(run_dir: Path) -> bool:
    return (Path(run_dir) / _PROTECT_MARKER).exists()


def set_protected(run_dir: Path, protected: bool) -> None:
    marker = Path(run_dir) / _PROTECT_MARKER
    if protected:
        marker.touch()
    else:
        marker.unlink(missing_ok=True)


def _run_timestamp(manifest: dict, manifest_path: Path) -> float:
    ts = manifest.get("run_timestamp")
    if isinstance(ts, str) and ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
    try:
        return manifest_path.stat().st_mtime
    except OSError:
        return 0.0


def iter_runs(root: Path) -> list[RunInfo]:
    """Enumerate runs under ``root`` (directories holding a ``run_manifest.json``), newest first.

    Covers both full and compacted runs — a compacted run is still a directory with the manifest and
    its scene file.
    """
    root = Path(root)
    runs: list[RunInfo] = []
    if not root.is_dir():
        return runs
    for child in root.iterdir():
        manifest_path = child / "run_manifest.json"
        if not (child.is_dir() and manifest_path.exists()):
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = {}
        name = (manifest.get("name") or "").strip() or child.name
        runs.append(
            RunInfo(
                path=child,
                name=name,
                size_bytes=dir_size_bytes(child),
                timestamp=_run_timestamp(manifest, manifest_path),
                mode=str(manifest.get("mode", "")) or "semantic",
                compacted=find_scene_file(child) is not None and not (child / "mapping_outputs.npz").exists(),
                protected=is_protected(child),
            )
        )
    runs.sort(key=lambda r: r.timestamp, reverse=True)
    return runs


def total_runs_bytes(root: Path) -> int:
    return sum(r.size_bytes for r in iter_runs(root))


def expired_runs(root: Path, max_age_days: float, now: float) -> list[RunInfo]:
    """Runs older than ``max_age_days`` and not protected — retention's deletion candidates."""
    return [r for r in iter_runs(root) if not r.protected and r.age_days(now) > max_age_days]


def delete_run(run_dir: Path) -> None:
    """Remove a run directory. Refuses a protected run."""
    run_dir = Path(run_dir)
    if is_protected(run_dir):
        raise PermissionError(f"run is protected: {run_dir}")
    shutil.rmtree(run_dir, ignore_errors=True)
