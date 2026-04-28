"""Pre-index the filtered reference cloud per semantic class for fast timeline slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from deepreefmap.pipeline.artifacts import SemanticPointCloud


def median_distance_to_camera(cloud: "SemanticPointCloud") -> float | None:
    """Median of finite positive `distance_to_camera` over the full reference cloud, or None if unavailable."""
    if cloud.distance_to_camera is None:
        return None
    d = np.asarray(cloud.distance_to_camera, dtype=np.float64).reshape(-1)
    if d.size != len(cloud):
        return None
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return None
    return float(np.median(d[valid]))


@dataclass(frozen=True)
class FinalCloudIndex:
    """Per-class point arrays sorted by timeline rank, plus prefix counts per slider step."""

    frame_order: tuple[int, ...]
    """Source frame indices in timeline order (length = num timeline steps)."""

    class_ids: tuple[int, ...]
    """Classes that appear in the cloud (sorted)."""

    xyz_by_class: dict[int, np.ndarray]
    rgb_by_class: dict[int, np.ndarray]
    semrgb_by_class: dict[int, np.ndarray]
    conf_by_class: dict[int, np.ndarray]
    """For class c, conf_by_class[c] is a 1-D float32 confidence per point (1.0 if cloud has no confidence)."""
    prefix_end_by_class: dict[int, np.ndarray]
    """For class c, prefix_end_by_class[c][t] = number of points with timeline_rank <= t."""


def build_final_cloud_index(
    cloud: "SemanticPointCloud",
    frame_order: list[int] | tuple[int, ...],
    class_colors: dict[int, tuple[int, int, int]],
) -> FinalCloudIndex:
    """Split cloud by label, sort points by timeline rank, build prefix-end arrays."""
    if len(cloud) == 0:
        fo = tuple(int(x) for x in frame_order)
        return FinalCloudIndex(
            frame_order=fo,
            class_ids=tuple(),
            xyz_by_class={},
            rgb_by_class={},
            semrgb_by_class={},
            conf_by_class={},
            prefix_end_by_class={},
        )

    xyz = np.asarray(cloud.xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(cloud.rgb, dtype=np.uint8).reshape(-1, 3)
    labels = np.asarray(cloud.labels, dtype=np.int32).reshape(-1)
    if cloud.frame_indices is None:
        raise ValueError("reference cloud must have frame_indices for timeline visualization")
    frame_indices = np.asarray(cloud.frame_indices, dtype=np.int32).reshape(-1)
    if cloud.confidence is not None:
        conf_all = np.asarray(cloud.confidence, dtype=np.float32).reshape(-1)
        if conf_all.shape[0] != xyz.shape[0]:
            conf_all = np.ones(xyz.shape[0], dtype=np.float32)
    else:
        conf_all = np.ones(xyz.shape[0], dtype=np.float32)

    dist = None
    if cloud.distance_to_camera is not None:
        dist = np.asarray(cloud.distance_to_camera, dtype=np.float32).reshape(-1)
        if dist.shape[0] != xyz.shape[0]:
            dist = None

    distance_cap = median_distance_to_camera(cloud)
    if distance_cap is not None and dist is not None:
        dist_keep = np.isfinite(dist) & (dist <= float(distance_cap))
    else:
        dist_keep = np.ones(xyz.shape[0], dtype=bool)

    xyz = xyz[dist_keep]
    rgb = rgb[dist_keep]
    labels = labels[dist_keep]
    frame_indices = frame_indices[dist_keep]
    conf_all = conf_all[dist_keep]

    frame_order_t = tuple(int(x) for x in frame_order)
    if not frame_order_t:
        return FinalCloudIndex(
            frame_order=frame_order_t,
            class_ids=tuple(),
            xyz_by_class={},
            rgb_by_class={},
            semrgb_by_class={},
            conf_by_class={},
            prefix_end_by_class={},
        )

    frame_to_rank = {fid: r for r, fid in enumerate(frame_order_t)}
    ranks = np.array([frame_to_rank.get(int(f), -1) for f in frame_indices.tolist()], dtype=np.int32)
    in_timeline = ranks >= 0
    xyz = xyz[in_timeline]
    rgb = rgb[in_timeline]
    labels = labels[in_timeline]
    ranks = ranks[in_timeline]
    conf_all = conf_all[in_timeline]

    n_steps = len(frame_order_t)
    xyz_by_class: dict[int, np.ndarray] = {}
    rgb_by_class: dict[int, np.ndarray] = {}
    semrgb_by_class: dict[int, np.ndarray] = {}
    conf_by_class: dict[int, np.ndarray] = {}
    prefix_end_by_class: dict[int, np.ndarray] = {}

    unique_labels = sorted(int(x) for x in np.unique(labels).tolist())
    for class_id in unique_labels:
        m = labels == int(class_id)
        if not np.any(m):
            continue
        xyz_c = xyz[m]
        rgb_c = rgb[m]
        ranks_c = ranks[m]
        conf_c = conf_all[m]
        order = np.argsort(ranks_c, kind="mergesort")
        xyz_c = xyz_c[order]
        rgb_c = rgb_c[order]
        ranks_c = ranks_c[order]
        conf_c = conf_c[order]

        color = class_colors.get(int(class_id), (128, 128, 128))
        sem = np.full_like(rgb_c, fill_value=0, dtype=np.uint8)
        sem[:, 0] = int(color[0])
        sem[:, 1] = int(color[1])
        sem[:, 2] = int(color[2])

        prefix_end = np.zeros(n_steps, dtype=np.int64)
        for t in range(n_steps):
            prefix_end[t] = int(np.searchsorted(ranks_c, t, side="right"))

        cid = int(class_id)
        xyz_by_class[cid] = xyz_c
        rgb_by_class[cid] = rgb_c
        semrgb_by_class[cid] = sem
        conf_by_class[cid] = conf_c.astype(np.float32, copy=False)
        prefix_end_by_class[cid] = prefix_end

    class_ids = tuple(sorted(xyz_by_class.keys()))
    return FinalCloudIndex(
        frame_order=frame_order_t,
        class_ids=class_ids,
        xyz_by_class=xyz_by_class,
        rgb_by_class=rgb_by_class,
        semrgb_by_class=semrgb_by_class,
        conf_by_class=conf_by_class,
        prefix_end_by_class=prefix_end_by_class,
    )
