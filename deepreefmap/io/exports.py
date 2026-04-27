from __future__ import annotations

from pathlib import Path

import numpy as np

from deepreefmap.pipeline.artifacts import SemanticPointCloud
from deepreefmap.pointcloud.grid_ortho import OrthoGrid


def save_semantic_cloud(path: Path, cloud: SemanticPointCloud) -> None:
    payload: dict[str, np.ndarray] = {
        "xyz": cloud.xyz.astype(np.float32),
        "rgb": cloud.rgb.astype(np.uint8),
        "labels": cloud.labels.astype(np.int32),
    }
    if cloud.frame_indices is not None:
        payload["frame_indices"] = cloud.frame_indices.astype(np.int32)
    if cloud.confidence is not None:
        payload["confidence"] = cloud.confidence.astype(np.float32)
    if cloud.distance_to_camera is not None:
        payload["distance_to_camera"] = cloud.distance_to_camera.astype(np.float32)
    np.savez_compressed(path, **payload)


def save_ortho_grid(path: Path, grid: OrthoGrid) -> None:
    np.savez_compressed(
        path,
        rgb=grid.rgb,
        labels=grid.labels,
        height=grid.height,
        counts=grid.counts,
        frame_index=grid.frame_index,
        cell_size=np.asarray(grid.cell_size, dtype=np.float32),
        pixel_size_m=np.asarray(np.nan if grid.pixel_size_m is None else grid.pixel_size_m, dtype=np.float32),
    )
