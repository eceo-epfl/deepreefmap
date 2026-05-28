from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from deepreefmap.config.classes import ClassConfig, load_classes, resolve_manifest_classes
from deepreefmap.io.exports import load_geometry_cloud
from deepreefmap.io.scene_file import (
    SCENE_FILE_SUFFIX,
    LazyFrameBatch,
    SceneFrameAccessor,
    find_scene_file,
    load_scene_file,
    scene_file_name,
)
from deepreefmap.pipeline import resume as resume_mod
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, SemanticPointCloud
from deepreefmap.pointcloud.filters import PointFilterConfig, build_semantic_reference_cloud

if TYPE_CHECKING:
    from deepreefmap.pointcloud.final_cloud_index import FinalCloudIndex

logger = logging.getLogger(__name__)

GEOMETRY_ONLY_MODE = "geometry_only"
SEMANTIC_MODE = "semantic"


@dataclass(frozen=True)
class LoadedRun:
    run_dir: Path
    manifest: dict[str, Any]
    classes_config: ClassConfig
    frame_batch: FrameBatch | LazyFrameBatch
    mapping_result: MappingSequenceResult
    output_files: list[str]
    mode: str = SEMANTIC_MODE
    reference_cloud: SemanticPointCloud = field(default_factory=SemanticPointCloud.empty)
    geometry_xyz: np.ndarray | None = None
    geometry_rgb: np.ndarray | None = None
    from_scene_file: bool = False
    scene_accessor: SceneFrameAccessor | None = None
    final_cloud_index: FinalCloudIndex | None = None
    world_points_warning: str | None = None


def _world_points_fallback_warning(
    manifest: dict[str, Any], mapping_result: MappingSequenceResult
) -> str | None:
    """Detect a LoGeR run whose cloud silently fell back to depth-unprojection.

    LoGeR emits per-point ``world_points``; before they were persisted to
    ``mapping_outputs.npz`` (commit 8f4d41d) a resumed run lost them and rebuilt
    geometry from depth+pose — degraded, not failed. A recorded
    ``geometry_source == "depth_unprojection"`` (e.g. intrinsics refinement) is
    intentional and not flagged.
    """
    if str(manifest.get("mapping_backend", "")) not in {"loger", "loger_star"}:
        return None
    if getattr(mapping_result, "world_points", None) is not None:
        return None
    if manifest.get("geometry_source") == "depth_unprojection":
        return None
    msg = (
        "LoGeR run loaded without per-point world geometry (mapping_outputs.npz "
        "predates world_points persistence); the cloud uses the depth-unprojection "
        "fallback. Re-run the reconstruction for full LoGeR geometry."
    )
    logger.warning(msg)
    return msg


def load_cached_run(
    run_dir: Path,
    *,
    point_filter_config: PointFilterConfig | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> LoadedRun:
    """Load a completed reconstruction folder.

    If a scene file exists and is up-to-date, loads from it (fast path).
    Otherwise falls back to the original slow path and generates the scene
    file in the background for next time.
    """

    def _step(stage: str, cur: int, tot: int) -> None:
        if progress_cb is not None:
            progress_cb(stage, cur, tot)

    run_dir = Path(run_dir)

    # --- Fast path: try scene file ---
    scene_path = find_scene_file(run_dir)
    if scene_path is not None:
        try:
            loaded = _load_from_scene_file(scene_path, run_dir, _step)
            if loaded is not None:
                logger.info("Loaded from scene file (fast path): %s", scene_path)
                return loaded
            logger.info("Scene file stale or incompatible, falling back to slow path")
        except Exception:
            logger.warning("Scene file load failed, falling back to slow path", exc_info=True)

    # Clean up stale .tmp files from interrupted background generation
    for tmp in run_dir.glob("*" + SCENE_FILE_SUFFIX + ".tmp"):
        try:
            tmp.unlink()
            logger.info("Cleaned up stale temp file: %s", tmp)
        except OSError:
            pass

    # --- Slow path ---
    result = _load_slow_path(run_dir, point_filter_config, _step)

    # Generate scene file in background for next time
    if result.mode == SEMANTIC_MODE and len(result.reference_cloud) > 0:
        _generate_scene_file_async(run_dir, result)

    return result


def _load_from_scene_file(
    scene_path: Path,
    run_dir: Path,
    step: Callable[[str, int, int], None],
) -> LoadedRun | None:
    scene = load_scene_file(scene_path, run_dir=run_dir, progress_cb=step)
    if scene is None:
        return None

    fb = LazyFrameBatch(scene.frame_accessor, scene.mapping_result.intrinsics)
    output_files = scene.manifest.get("output_files", [])

    return LoadedRun(
        run_dir=run_dir,
        manifest=scene.manifest,
        classes_config=scene.classes_config,
        frame_batch=fb,
        mapping_result=scene.mapping_result,
        output_files=output_files,
        mode=scene.run_mode,
        from_scene_file=True,
        scene_accessor=scene.frame_accessor,
        final_cloud_index=scene.final_cloud_index,
        world_points_warning=_world_points_fallback_warning(scene.manifest, scene.mapping_result),
    )


def _load_slow_path(
    run_dir: Path,
    point_filter_config: PointFilterConfig | None,
    _step: Callable[[str, int, int], None],
) -> LoadedRun:
    _step("manifest", 0, 1)
    manifest = _load_manifest(run_dir)
    _step("manifest", 1, 1)

    _step("classes", 0, 1)
    classes_config = load_classes(_resolve_classes_path(run_dir, manifest))
    _step("classes", 1, 1)

    _step("mapping", 0, 1)
    mapping_result = resume_mod.load_mapping_result(run_dir)
    if mapping_result is None:
        raise RuntimeError("Run folder is missing a readable mapping_outputs.npz artifact.")
    _step("mapping", 1, 1)

    sidecar = resume_mod.read_sidecar(run_dir, resume_mod.STAGE_PREPROCESS)
    if sidecar is None:
        sidecar = _preprocess_sidecar_from_manifest(manifest)
    frame_batch = resume_mod.load_prepared_frames(
        run_dir,
        sidecar,
        mapping_result.intrinsics,
        progress_cb=lambda done, total: _step("frames", done, total),
    )
    if frame_batch is None:
        raise RuntimeError(
            "Run folder is missing cached frames, labels, masks, or preprocess metadata required for viewing."
        )
    output_files = _output_files_from_manifest(manifest)
    mode = _resolve_mode(manifest)

    if mode == GEOMETRY_ONLY_MODE:
        geometry_path = run_dir / "geometry_cloud.ply"
        if not geometry_path.exists():
            raise RuntimeError(
                f"Geometry-only run is missing geometry_cloud.ply: {geometry_path}"
            )
        _step("geometry", 0, 1)
        geometry_xyz, geometry_rgb = load_geometry_cloud(geometry_path)
        _step("geometry", 1, 1)
        return LoadedRun(
            run_dir=run_dir,
            manifest=manifest,
            classes_config=classes_config,
            frame_batch=frame_batch,
            mapping_result=mapping_result,
            output_files=output_files,
            mode=mode,
            geometry_xyz=geometry_xyz,
            geometry_rgb=geometry_rgb,
            world_points_warning=_world_points_fallback_warning(manifest, mapping_result),
        )

    reference_cloud = build_semantic_reference_cloud(
        frame_batch,
        mapping_result,
        classes_config,
        point_filter_config,
        progress_cb=lambda done, total: _step("cloud", done, total),
        stage_cb=lambda name: _step(f"cloud_{name}", 0, 0),
    )

    return LoadedRun(
        run_dir=run_dir,
        manifest=manifest,
        classes_config=classes_config,
        frame_batch=frame_batch,
        mapping_result=mapping_result,
        output_files=output_files,
        mode=mode,
        reference_cloud=reference_cloud,
        world_points_warning=_world_points_fallback_warning(manifest, mapping_result),
    )


def _generate_scene_file_async(run_dir: Path, result: LoadedRun) -> None:
    """Build and save a scene file on a daemon thread so the next load is fast."""

    def _worker() -> None:
        try:
            from deepreefmap.io.scene_file import save_scene_file
            from deepreefmap.pointcloud.final_cloud_index import build_final_cloud_index

            frame_order = [int(f.frame_index) for f in result.frame_batch.frames]
            class_colors = result.classes_config.id_to_color
            fci = build_final_cloud_index(result.reference_cloud, frame_order, class_colors)

            fname = scene_file_name(result.manifest, run_dir)
            out = run_dir / fname
            save_scene_file(
                out,
                manifest=result.manifest,
                classes_config=result.classes_config,
                mapping_result=result.mapping_result,
                frame_batch=result.frame_batch,  # type: ignore[arg-type]  # TODO(stage2): unify FrameBatch/LazyFrameBatch
                final_cloud_index=fci,
                run_dir=run_dir,
            )
            logger.info("Scene file generated for next load: %s", out)
        except Exception:
            logger.warning("Background scene file generation failed", exc_info=True)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def _resolve_mode(manifest: dict[str, Any]) -> str:
    """Return the run mode, supporting schema_version=1 manifests via the magic segmentation_model value."""
    explicit = manifest.get("mode")
    if isinstance(explicit, str) and explicit:
        return explicit
    if manifest.get("segmentation_model") == "__skip__":
        return GEOMETRY_ONLY_MODE
    return SEMANTIC_MODE


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Run manifest must contain a JSON object: {manifest_path}")
    return payload


def _resolve_classes_path(run_dir: Path, manifest: dict[str, Any]) -> Path | None:
    return resolve_manifest_classes(manifest.get("classes"), run_dir)


def _preprocess_sidecar_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    frame_indices = manifest.get("frame_indices")
    clip_counts = manifest.get("clip_counts")
    if not frame_indices or clip_counts is None:
        raise RuntimeError("Run manifest lacks frame_indices/clip_counts and no preprocess cache sidecar exists.")
    return {"key": "", "frame_indices": frame_indices, "clip_counts": clip_counts}


def _output_files_from_manifest(manifest: dict[str, Any]) -> list[str]:
    output_files = manifest.get("output_files", [])
    if not isinstance(output_files, list) or not all(isinstance(p, str) for p in output_files):
        raise RuntimeError("Run manifest field output_files must be a list of strings.")
    return output_files
