from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from deepreefmap.pipeline.artifacts import SemanticPointCloud


@dataclass(frozen=True)
class OrthoProjection:
    mean_xyz: np.ndarray
    components: np.ndarray
    xy_min: np.ndarray
    cell_size: float

    def project_cells(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = (np.asarray(xyz, dtype=np.float32) - self.mean_xyz) @ self.components.T
        xy -= self.xy_min
        coords = np.floor(xy / max(float(self.cell_size), 1e-9)).astype(np.int32)
        return coords[:, 0], coords[:, 1]


@dataclass(frozen=True)
class OrthoGrid:
    rgb: np.ndarray
    labels: np.ndarray
    height: np.ndarray
    counts: np.ndarray
    frame_index: np.ndarray
    cell_size: float
    pixel_size_m: float | None = None
    projection: OrthoProjection | None = None


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

    xyz_all = np.asarray(cloud.xyz, dtype=np.float32)
    pca = PCA(n_components=2)
    xy_raw = pca.fit_transform(xyz_all)
    z_axis = np.cross(pca.components_[0], pca.components_[1])
    z_axis /= max(np.linalg.norm(z_axis), 1e-8)
    heights = xyz_all @ z_axis
    xy_min = xy_raw.min(axis=0, keepdims=True)
    xy = xy_raw - xy_min
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
    distance_to_camera = _valid_distance_to_camera(cloud)

    keys = coords[:, 1].astype(np.int64) * width + coords[:, 0].astype(np.int64)
    order = np.argsort(keys)
    split_points = np.flatnonzero(np.diff(keys[order]) != 0) + 1
    groups = np.split(order, split_points)

    for group in groups:
        y = int(coords[group[0], 1])
        x = int(coords[group[0], 0])
        top = _camera_facing_group(group, heights, distance_to_camera)
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
        projection=OrthoProjection(
            mean_xyz=np.asarray(pca.mean_, dtype=np.float32),
            components=np.asarray(pca.components_, dtype=np.float32),
            xy_min=xy_min.reshape(2).astype(np.float32),
            cell_size=float(cell_size),
        ),
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


def _valid_distance_to_camera(cloud: SemanticPointCloud) -> np.ndarray | None:
    if cloud.distance_to_camera is None:
        return None
    dist = np.asarray(cloud.distance_to_camera, dtype=np.float32).reshape(-1)
    if dist.shape[0] != len(cloud):
        return None
    if not np.any(np.isfinite(dist)):
        return None
    return dist


def _camera_facing_group(group: np.ndarray, heights: np.ndarray, distance_to_camera: np.ndarray | None) -> np.ndarray:
    if distance_to_camera is not None:
        group_dist = distance_to_camera[group]
        finite = np.isfinite(group_dist)
        if np.any(finite):
            threshold = float(np.median(group_dist[finite]))
            top = group[finite & (group_dist <= threshold)]
            if top.size:
                return top

    group_heights = heights[group]
    top_mask = group_heights >= group_heights.mean()
    return group[top_mask] if top_mask.any() else group
