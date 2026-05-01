from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from deepreefmap.config.classes import ClassConfig, load_classes
from deepreefmap.io.exports import load_geometry_cloud
from deepreefmap.pipeline import resume as resume_mod
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, SemanticPointCloud
from deepreefmap.pointcloud.filters import PointFilterConfig, build_semantic_reference_cloud


GEOMETRY_ONLY_MODE = "geometry_only"
SEMANTIC_MODE = "semantic"


@dataclass(frozen=True)
class LoadedRun:
    run_dir: Path
    manifest: dict[str, Any]
    classes_config: ClassConfig
    frame_batch: FrameBatch
    mapping_result: MappingSequenceResult
    output_files: list[str]
    mode: str = SEMANTIC_MODE
    reference_cloud: SemanticPointCloud = field(default_factory=SemanticPointCloud.empty)
    geometry_xyz: np.ndarray | None = None
    geometry_rgb: np.ndarray | None = None


def load_cached_run(
    run_dir: Path,
    *,
    point_filter_config: PointFilterConfig | None = None,
) -> LoadedRun:
    """Load a completed reconstruction folder into the objects expected by Viser."""

    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    classes_config = load_classes(_resolve_classes_path(run_dir, manifest))
    mapping_result = resume_mod.load_mapping_result(run_dir)
    if mapping_result is None:
        raise RuntimeError("Run folder is missing a readable mapping_outputs.npz artifact.")

    sidecar = resume_mod.read_sidecar(run_dir, resume_mod.STAGE_PREPROCESS)
    if sidecar is None:
        sidecar = _preprocess_sidecar_from_manifest(manifest)
    frame_batch = resume_mod.load_prepared_frames(run_dir, sidecar, mapping_result.intrinsics)
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
        geometry_xyz, geometry_rgb = load_geometry_cloud(geometry_path)
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
        )

    reference_cloud = build_semantic_reference_cloud(
        frame_batch,
        mapping_result,
        classes_config,
        point_filter_config,
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
    )


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


def _resolve_classes_path(run_dir: Path, manifest: dict[str, Any]) -> Path:
    classes_path = Path(str(manifest.get("classes", "configs/classes_coralscapes.yaml")))
    if classes_path.is_absolute() and classes_path.exists():
        return classes_path

    run_relative = run_dir / classes_path
    if run_relative.exists():
        return run_relative

    if classes_path.exists():
        return classes_path

    raise FileNotFoundError(f"Classes config not found for run viewer: {classes_path}")


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
