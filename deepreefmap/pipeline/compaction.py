"""Compact a completed run directory down to its single ``.scene.zarr.zip``.

After a run, the heavy plain artifacts (frames/labels/masks, ``mapping_outputs.npz``, the PLY clouds,
``ortho.npz``/``ortho.png``, ``benthic_cover.json``) are fully reconstructable from the scene file via
``deepreefmap extract``. Compaction prunes them, keeping only ``run_manifest.json`` (cheap discovery)
and the ``.scene.zarr.zip`` — but **only after verifying** the scene file loads and is structurally
complete, so a truncated or incomplete zip can never cause data loss.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from deepreefmap.io.scene_file import find_scene_file, load_scene_file

logger = logging.getLogger(__name__)

GEOMETRY_ONLY_MODE = "geometry_only"

# Heavy artifacts that are losslessly reconstructable from the scene file.
_PRUNE_DIRS = ("frames", "labels", "masks", ".cache", "videos")
_PRUNE_FILES = ("mapping_outputs.npz", "ortho.npz", "ortho.png", "benthic_cover.json")


class CompactionError(RuntimeError):
    """A run cannot be safely compacted (no scene file, or it failed verification)."""


def verify_scene_complete(scene_path: Path) -> None:
    """Raise :class:`CompactionError` unless the scene file is readable and carries every group needed
    to rehydrate exactly the artifacts this run wrote (per the manifest's ``output_files``)."""
    import json

    import zarr

    try:
        store = zarr.ZipStore(str(scene_path), mode="r")
    except Exception as exc:  # noqa: BLE001 - any read failure means "do not prune"
        raise CompactionError(f"scene file unreadable: {exc}") from exc
    try:
        root = zarr.open_group(store=store, mode="r")
        raw = root["meta"]["manifest"][()]
        raw = raw.item() if hasattr(raw, "item") else raw
        outputs = set(json.loads(raw).get("output_files", []) if isinstance(raw, str) else [])

        def _has(*path: str) -> bool:
            node = root
            for key in path:
                if key not in node:
                    return False
                node = node[key]
            return True

        missing = [g for g in ("meta", "classes", "mapping", "frames") if g not in root]
        # Each output the run produced must have the group needed to reproduce it on extract.
        if "semantic_reference_cloud.ply" in outputs and not _has("cloud"):
            missing.append("cloud")
        if "geometry_cloud.ply" in outputs and not _has("geometry"):
            missing.append("geometry")
        if "ortho.npz" in outputs and not _has("products", "ortho"):
            missing.append("products/ortho")
        if "benthic_cover.json" in outputs and not _has("products", "cover"):
            missing.append("products/cover")
        if "tsdf_cloud.ply" in outputs and not _has("tsdf", "geometry"):
            missing.append("tsdf/geometry")
        if "semantic_tsdf_cloud.ply" in outputs and not _has("tsdf", "semantic"):
            missing.append("tsdf/semantic")
        if missing:
            raise CompactionError(f"scene file missing groups {missing}; regenerate before compacting")
        # Sanity: the first frame must actually decode (catches a truncated zip).
        fg = root["frames"]
        if int(fg.attrs.get("n_frames", 0)) > 0:
            _ = fg["images_rgb"][0]
    finally:
        store.close()


def compact_run(run_dir: Path, *, verify: bool = True) -> Path:
    """Prune a run directory down to ``run_manifest.json`` + its scene file.

    Returns the scene file path. Raises :class:`CompactionError` (deleting nothing) if there is no
    complete scene file to fall back on.
    """
    run_dir = Path(run_dir)
    scene = find_scene_file(run_dir)
    if scene is None:
        raise CompactionError(f"no scene file in {run_dir}; cannot compact")
    if verify:
        loaded = load_scene_file(scene, run_dir=run_dir)
        if loaded is None:
            raise CompactionError(f"scene file failed to load/verify: {scene}")
        loaded.frame_accessor.close()
        verify_scene_complete(scene)

    removed = 0
    for d in _PRUNE_DIRS:
        p = run_dir / d
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    for f in _PRUNE_FILES:
        p = run_dir / f
        if p.exists():
            p.unlink()
            removed += 1
    for ply in run_dir.glob("*.ply"):
        ply.unlink()
        removed += 1
    logger.info("Compacted %s → %s (pruned %d artifacts)", run_dir, scene.name, removed)
    return scene


def is_compacted(run_dir: Path) -> bool:
    """True when a run's heavy source artifact is gone but its scene file remains."""
    return find_scene_file(Path(run_dir)) is not None and not (Path(run_dir) / "mapping_outputs.npz").exists()
