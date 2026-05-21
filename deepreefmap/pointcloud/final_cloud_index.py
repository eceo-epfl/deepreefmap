"""Pre-index the filtered reference cloud per semantic class for fast timeline slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from deepreefmap.pipeline.artifacts import SemanticPointCloud


ProgressFn = Callable[[str, int, int], None]


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


def reconstruct_cloud_from_index(fci: FinalCloudIndex) -> "SemanticPointCloud":
    """Flatten a FinalCloudIndex back into a SemanticPointCloud."""
    from deepreefmap.pipeline.artifacts import SemanticPointCloud

    if not fci.class_ids:
        return SemanticPointCloud.empty()

    xyz_parts, rgb_parts, label_parts, conf_parts, fi_parts = [], [], [], [], []
    fo = np.asarray(fci.frame_order, dtype=np.int32)

    for cid in fci.class_ids:
        n = len(fci.xyz_by_class[cid])
        if n == 0:
            continue
        xyz_parts.append(fci.xyz_by_class[cid])
        rgb_parts.append(fci.rgb_by_class[cid])
        conf_parts.append(fci.conf_by_class[cid])
        label_parts.append(np.full(n, cid, dtype=np.int32))
        pe = fci.prefix_end_by_class[cid]
        point_fi = np.empty(n, dtype=np.int32)
        prev = 0
        for t in range(len(pe)):
            cur = int(pe[t])
            if cur > prev:
                point_fi[prev:cur] = fo[t] if t < len(fo) else 0
            prev = cur
        fi_parts.append(point_fi)

    return SemanticPointCloud(
        xyz=np.concatenate(xyz_parts),
        rgb=np.concatenate(rgb_parts),
        labels=np.concatenate(label_parts),
        frame_indices=np.concatenate(fi_parts),
        confidence=np.concatenate(conf_parts),
    )


def build_final_cloud_index(
    cloud: "SemanticPointCloud",
    frame_order: list[int] | tuple[int, ...],
    class_colors: dict[int, tuple[int, int, int]],
    progress: ProgressFn | None = None,
) -> FinalCloudIndex:
    """Split cloud by label, sort points by timeline rank, build prefix-end arrays.

    `progress(stage_label, current, total)` fires at major phases, with `total == 0`
    meaning indeterminate.
    """
    def _emit(stage: str, cur: int, tot: int) -> None:
        if progress is not None:
            try:
                progress(stage, cur, tot)
            except Exception:
                pass

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

    _emit("Indexing cloud", 0, 0)

    # Vectorised frame-index -> timeline-rank lookup. The previous Python
    # comprehension `[frame_to_rank.get(int(f), -1) for f in tolist()]`
    # was the dominant cost for large clouds (millions of points × dict.get).
    fo_arr = np.asarray(frame_order_t, dtype=np.int64)
    fi64 = frame_indices.astype(np.int64, copy=False)
    fi_min = int(fi64.min()) if fi64.size else 0
    fi_max = int(fi64.max()) if fi64.size else 0
    fo_min = int(fo_arr.min())
    fo_max = int(fo_arr.max())
    lo = min(fi_min, fo_min)
    hi = max(fi_max, fo_max)
    lookup = np.full(hi - lo + 1, -1, dtype=np.int32)
    lookup[fo_arr - lo] = np.arange(len(fo_arr), dtype=np.int32)
    ranks = lookup[fi64 - lo]
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
    step_targets = np.arange(n_steps, dtype=np.int32)
    n_classes = len(unique_labels)
    for ci, class_id in enumerate(unique_labels):
        _emit("Indexing classes", ci, n_classes)
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

        # One C-level searchsorted across all timeline steps replaces the
        # previous per-step Python loop.
        prefix_end = np.searchsorted(ranks_c, step_targets, side="right").astype(np.int64, copy=False)

        cid = int(class_id)
        xyz_by_class[cid] = xyz_c
        rgb_by_class[cid] = rgb_c
        semrgb_by_class[cid] = sem
        conf_by_class[cid] = conf_c.astype(np.float32, copy=False)
        prefix_end_by_class[cid] = prefix_end

    _emit("Indexing classes", n_classes, n_classes)

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
