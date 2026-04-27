from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from deepreefmap.pipeline.artifacts import SemanticPointCloud


@dataclass(frozen=True)
class OrthoGrid:
    rgb: np.ndarray
    labels: np.ndarray
    height: np.ndarray
    counts: np.ndarray
    frame_index: np.ndarray
    cell_size: float
    pixel_size_m: float | None = None


def aggregate_cloud_to_ortho_grid(
    cloud: SemanticPointCloud,
    bins: int = 2000,
    cell_size: float | None = None,
) -> OrthoGrid:
    if len(cloud) < 2 or _is_degenerate(cloud.xyz):
        empty_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        empty = np.zeros((1, 1), dtype=np.int32)
        return OrthoGrid(
            rgb=empty_rgb,
            labels=empty,
            height=empty.astype(np.float32),
            counts=empty,
            frame_index=empty,
            cell_size=1.0,
        )

    pca = PCA(n_components=2)
    xy = pca.fit_transform(cloud.xyz)
    z_axis = np.cross(pca.components_[0], pca.components_[1])
    z_axis /= max(np.linalg.norm(z_axis), 1e-8)
    heights = cloud.xyz @ z_axis
    xy -= xy.min(axis=0, keepdims=True)
    if cell_size is None:
        span = float(max(xy[:, 0].max(), xy[:, 1].max(), 1.0))
        cell_size = max(span / max(1, bins), 1e-6)
    coords = np.floor(xy / cell_size).astype(np.int32)
    width = int(coords[:, 0].max() + 1)
    height = int(coords[:, 1].max() + 1)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    labels = np.zeros((height, width), dtype=np.int32)
    z_img = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)
    frame_index = np.zeros((height, width), dtype=np.int32)

    keys = coords[:, 1].astype(np.int64) * width + coords[:, 0].astype(np.int64)
    order = np.argsort(keys)
    split_points = np.flatnonzero(np.diff(keys[order]) != 0) + 1
    groups = np.split(order, split_points)

    for group in groups:
        y = int(coords[group[0], 1])
        x = int(coords[group[0], 0])
        group_heights = heights[group]
        top_mask = group_heights >= group_heights.mean()
        top = group[top_mask] if top_mask.any() else group
        labels[y, x] = _mode_int(cloud.labels[top])
        z_img[y, x] = float(heights[top].mean())
        rgb[y, x] = np.clip(cloud.rgb[top].mean(axis=0), 0, 255).astype(np.uint8)
        counts[y, x] = int(group.size)
        if cloud.frame_indices is not None:
            frame_index[y, x] = int(round(float(cloud.frame_indices[top].mean())))

    return OrthoGrid(
        rgb=rgb,
        labels=labels,
        height=z_img,
        counts=counts,
        frame_index=frame_index,
        cell_size=float(cell_size),
    )


def _is_degenerate(xyz: np.ndarray) -> bool:
    if not np.all(np.isfinite(xyz)):
        return True
    spans = xyz.max(axis=0) - xyz.min(axis=0)
    return int((spans > 1e-9).sum()) < 2


def _mode_int(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    values = values.astype(np.int64)
    non_negative = values[values >= 0]
    if non_negative.size == values.size:
        return int(np.bincount(non_negative).argmax())
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])
