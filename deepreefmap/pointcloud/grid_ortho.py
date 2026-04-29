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

    distance_to_camera = _valid_distance_to_camera(cloud)

    keys = coords[:, 1].astype(np.int64) * width + coords[:, 0].astype(np.int64)
    flat_size = height * width

    if distance_to_camera is None:
        order = np.argsort(keys)
    else:
        order = np.lexsort((np.where(np.isfinite(distance_to_camera), distance_to_camera, np.inf), keys))

    keys_s = keys[order]
    group_starts = np.concatenate([[0], np.flatnonzero(np.diff(keys_s) != 0) + 1])
    group_sizes = np.diff(np.concatenate([group_starts, [keys_s.size]])).astype(np.int64)
    unique_keys = keys_s[group_starts]
    group_ids_s = np.repeat(np.arange(group_starts.size, dtype=np.int64), group_sizes)
    ranks_s = np.arange(keys_s.size, dtype=np.int64) - np.repeat(group_starts, group_sizes)

    heights_s = heights[order].astype(np.float32, copy=False)
    labels_s = np.asarray(cloud.labels, dtype=np.int32).reshape(-1)[order]
    rgb_s = np.asarray(cloud.rgb, dtype=np.uint8).reshape(-1, 3)[order]

    if distance_to_camera is not None:
        dist_s = distance_to_camera[order]
        finite_s = np.isfinite(dist_s)
        finite_counts = np.add.reduceat(finite_s.astype(np.int64), group_starts)
        finite_counts_s = np.repeat(finite_counts, group_sizes)
        top_counts_s = (finite_counts_s + 1) // 2
        top_mask = (finite_counts_s > 0) & finite_s & (ranks_s < top_counts_s)
        if np.any(finite_counts == 0):
            height_means = np.add.reduceat(heights_s, group_starts) / group_sizes
            height_means_s = np.repeat(height_means, group_sizes)
            top_mask |= (finite_counts_s == 0) & (heights_s >= height_means_s)
    else:
        height_means = np.add.reduceat(heights_s, group_starts) / group_sizes
        top_mask = heights_s >= np.repeat(height_means, group_sizes)

    top_group_ids = group_ids_s[top_mask]
    top_group_starts = np.concatenate([[0], np.flatnonzero(np.diff(top_group_ids) != 0) + 1])
    top_group_sizes = np.diff(np.concatenate([top_group_starts, [top_group_ids.size]])).astype(np.float32)
    top_unique_group_ids = top_group_ids[top_group_starts]
    top_keys = unique_keys[top_unique_group_ids]
    top_heights = heights_s[top_mask]
    top_rgb = rgb_s[top_mask].astype(np.float32)

    rgb_flat = np.zeros((flat_size, 3), dtype=np.uint8)
    labels_flat = np.zeros(flat_size, dtype=np.int32)
    z_flat = np.zeros(flat_size, dtype=np.float32)
    counts_flat = np.zeros(flat_size, dtype=np.int32)
    frame_flat = np.zeros(flat_size, dtype=np.int32)

    counts_flat[unique_keys] = group_sizes.astype(np.int32)
    z_flat[top_keys] = (np.add.reduceat(top_heights, top_group_starts) / top_group_sizes).astype(np.float32)
    rgb_flat[top_keys] = np.clip(
        np.add.reduceat(top_rgb, top_group_starts) / top_group_sizes[:, None],
        0,
        255,
    ).astype(np.uint8)

    if cloud.frame_indices is not None:
        frames_s = np.asarray(cloud.frame_indices, dtype=np.int32).reshape(-1)[order][top_mask].astype(np.float32)
        frame_flat[top_keys] = np.rint(np.add.reduceat(frames_s, top_group_starts) / top_group_sizes).astype(np.int32)

    labels_top = labels_s[top_mask]
    label_order = np.lexsort((labels_top, top_group_ids))
    label_groups = top_group_ids[label_order]
    label_values = labels_top[label_order]
    label_run_starts = np.concatenate(
        [[0], np.flatnonzero((np.diff(label_groups) != 0) | (np.diff(label_values) != 0)) + 1]
    )
    label_run_counts = np.diff(np.concatenate([label_run_starts, [label_values.size]])).astype(np.int64)
    label_run_groups = label_groups[label_run_starts]
    label_run_values = label_values[label_run_starts]
    label_group_starts = np.concatenate([[0], np.flatnonzero(np.diff(label_run_groups) != 0) + 1])
    max_counts = np.maximum.reduceat(label_run_counts, label_group_starts)
    max_counts_by_run = np.repeat(
        max_counts,
        np.diff(np.concatenate([label_group_starts, [label_run_groups.size]])),
    )
    mode_candidates = label_run_counts == max_counts_by_run
    candidate_indices = np.flatnonzero(mode_candidates)
    _, first_candidate_positions = np.unique(label_run_groups[candidate_indices], return_index=True)
    mode_indices = candidate_indices[first_candidate_positions]
    labels_flat[unique_keys[label_run_groups[mode_indices]]] = label_run_values[mode_indices]

    rgb = rgb_flat.reshape(height, width, 3)
    labels = labels_flat.reshape(height, width)
    z_img = z_flat.reshape(height, width)
    counts = counts_flat.reshape(height, width)
    frame_index = frame_flat.reshape(height, width)

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
