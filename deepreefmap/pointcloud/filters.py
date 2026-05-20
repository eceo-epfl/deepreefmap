from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

from deepreefmap.config.classes import ClassConfig
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, SemanticPointCloud
from deepreefmap.pointcloud.unprojection import depth_to_points

_BIAS = np.int64(1 << 20)
_MASK = np.int64((1 << 21) - 1)


def _pack_voxel_keys_int64(keys: np.ndarray) -> np.ndarray | None:
    """Pack an ``(N, 3)`` int voxel-index array into one int64 per row, ix most significant.

    Returns ``None`` if any axis leaves the 21-bit-per-axis range so callers fall
    back to a per-axis lexsort. ix sits in the high bits so ascending packed order
    reproduces the old ix-major (ix, iy, iz) lexsort exactly, keeping the kept rows
    and their PLY byte-identical. At real voxel/replacement radii the range spans
    thousands of metres, so overflow never happens on real reef scans.
    """
    xa = keys[:, 0].astype(np.int64, copy=False) + _BIAS
    ya = keys[:, 1].astype(np.int64, copy=False) + _BIAS
    za = keys[:, 2].astype(np.int64, copy=False) + _BIAS
    in_range = (
        (xa >= 0) & (xa <= _MASK) & (ya >= 0) & (ya <= _MASK) & (za >= 0) & (za <= _MASK)
    )
    if not bool(np.all(in_range)):
        return None
    return (
        (xa.astype(np.uint64) << np.uint64(42))
        | (ya.astype(np.uint64) << np.uint64(21))
        | za.astype(np.uint64)
    ).astype(np.int64)


@dataclass(frozen=True)
class PointFilterConfig:
    min_depth: float = 0.05
    max_depth: float = 8.0
    confidence_percentile: float | None = 5.0
    min_confidence: float = 1e-5
    depth_edge_threshold: float | None = None
    voxel_size: float | None = 0.003
    replacement_radius_factor: float = 1.0
    replacement_radius_estimation_frames: int = 30
    replacement_radius_override: float | None = None


def build_semantic_reference_cloud(
    frame_batch: FrameBatch,
    mapping: MappingSequenceResult,
    classes_config: ClassConfig,
    config: PointFilterConfig | None = None,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
    stage_cb: Callable[[str], None] | None = None,
    max_workers: int = 6,
) -> SemanticPointCloud:
    """Lift the kept pixels of every frame into a filtered, voxelized semantic cloud."""
    def _emit_stage(name: str) -> None:
        if stage_cb is not None:
            try:
                stage_cb(name)
            except Exception:
                pass

    cfg = config or PointFilterConfig()
    ignore_labels = classes_config.ids_for_role("ignore_in_point_cloud")
    frame_lookup = {frame.frame_index: frame for frame in frame_batch.frames}
    active_radius = _resolve_replacement_radius(cfg, mapping.depth_maps)
    ignore_set = list(ignore_labels) if ignore_labels else []

    def _per_frame(result_i: int, frame_index: int):
        frame = frame_lookup.get(int(frame_index))
        if frame is None:
            return None
        depth = mapping.depth_maps[result_i].astype(np.float32)
        h, w = depth.shape
        labels = _resize_nearest(frame.labels, (w, h)).astype(np.int32)
        keep_mask = _resize_nearest(frame.keep_mask.astype(np.uint8), (w, h)) > 0
        rgb = _resize_rgb(frame.image_rgb, (w, h))
        confidence = None if mapping.confidence is None else mapping.confidence[result_i].astype(np.float32)
        if confidence is not None and confidence.shape != depth.shape:
            confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        if mapping.world_points is not None:
            xyz = mapping.world_points[result_i].reshape(-1, 3).astype(np.float32)
        else:
            xyz = depth_to_points(depth, mapping.intrinsics, mapping.poses_w_c[result_i]).astype(np.float32)

        valid = np.isfinite(depth)
        valid &= depth >= cfg.min_depth
        valid &= depth <= cfg.max_depth
        valid &= keep_mask
        if ignore_set:
            valid &= ~np.isin(labels, ignore_set)
        if cfg.depth_edge_threshold is not None:
            valid &= depth_edgeness(depth) <= cfg.depth_edge_threshold
        if confidence is not None:
            finite_conf = confidence[np.isfinite(confidence)]
            if finite_conf.size and cfg.confidence_percentile is not None:
                threshold = np.percentile(finite_conf, cfg.confidence_percentile)
            else:
                threshold = cfg.min_confidence
            valid &= confidence >= max(float(threshold), cfg.min_confidence)
        flat_valid = valid.reshape(-1)
        if not flat_valid.any():
            return None
        xyz_f = xyz[flat_valid]
        rgb_f = rgb.reshape(-1, 3)[flat_valid].astype(np.uint8)
        lab_f = labels.reshape(-1)[flat_valid].astype(np.int32)
        dist_f = depth.reshape(-1)[flat_valid].astype(np.float32)
        if confidence is not None:
            conf_f = confidence.reshape(-1)[flat_valid].astype(np.float32)
        else:
            conf_f = np.ones(int(flat_valid.sum()), dtype=np.float32)
        n = int(xyz_f.shape[0])
        frame_f = np.full(n, int(frame_index), dtype=np.int32)
        return xyz_f, rgb_f, lab_f, frame_f, conf_f, dist_f

    work = list(enumerate(mapping.frame_indices.tolist()))
    total = len(work)
    # Slot each result by submission position, not arrival: downstream tie-breaks
    # (the replacement lexsort, the voxel reduce) fall through to row order, so
    # completion order would leak thread scheduling into the cloud and its PLY.
    parts: list[tuple[np.ndarray, ...] | None] = [None] * total
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_pos = {ex.submit(_per_frame, ri, fi): ri for ri, fi in work}
        for fut in as_completed(fut_to_pos):
            parts[fut_to_pos[fut]] = fut.result()
            completed += 1
            if progress_cb is not None:
                progress_cb(completed, total)

    if all(part is None for part in parts):
        return SemanticPointCloud.empty()

    # For multi-million-point clouds this fill plus the replacement-radius
    # lexsort that follows are the silent gap the user sees after the
    # per-frame loop hits N/N, so surface them via stage_cb.
    _emit_stage("concatenating")
    n_total = sum(part[0].shape[0] for part in parts if part is not None)
    xyz = np.empty((n_total, 3), dtype=np.float32)
    rgb = np.empty((n_total, 3), dtype=np.uint8)
    labels = np.empty(n_total, dtype=np.int32)
    frame_indices = np.empty(n_total, dtype=np.int32)
    confidence = np.empty(n_total, dtype=np.float32)
    distance = np.empty(n_total, dtype=np.float32)
    # Releasing each part once copied keeps the peak at ~one cloud instead of
    # every part plus six concatenated copies co-resident.
    offset = 0
    for pos in range(total):
        part = parts[pos]
        if part is None:
            continue
        xyz_f, rgb_f, lab_f, frame_f, conf_f, dist_f = part
        stop = offset + xyz_f.shape[0]
        xyz[offset:stop] = xyz_f
        rgb[offset:stop] = rgb_f
        labels[offset:stop] = lab_f
        frame_indices[offset:stop] = frame_f
        confidence[offset:stop] = conf_f
        distance[offset:stop] = dist_f
        offset = stop
        parts[pos] = None
    cloud = SemanticPointCloud(
        xyz=xyz,
        rgb=rgb,
        labels=labels,
        frame_indices=frame_indices,
        confidence=confidence,
        distance_to_camera=distance,
    )

    if active_radius is not None:
        _emit_stage("replacing")
        cloud = nearest_camera_replace_semantic_cloud(
            cloud, active_radius,
            progress=lambda sub: _emit_stage(f"replacing_{sub}"),
        )

    if cfg.voxel_size is None or cfg.voxel_size <= 0:
        return cloud
    _emit_stage("voxelizing")
    return voxel_reduce_semantic_cloud(cloud, cfg.voxel_size)


def depth_edgeness(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    gx = np.zeros_like(depth)
    gy = np.zeros_like(depth)
    gx[:, :-1] += np.abs(depth[:, :-1] - depth[:, 1:])
    gx[:, 1:] += np.abs(depth[:, :-1] - depth[:, 1:])
    gy[:-1, :] += np.abs(depth[:-1, :] - depth[1:, :])
    gy[1:, :] += np.abs(depth[:-1, :] - depth[1:, :])
    return gx + gy


def voxel_reduce_semantic_cloud(cloud: SemanticPointCloud, voxel_size: float) -> SemanticPointCloud:
    if len(cloud) == 0:
        return cloud
    keys = np.floor(cloud.xyz / voxel_size).astype(np.int64)
    order = _voxel_sort_order(keys)
    keys_sorted = keys[order]
    group_starts = np.concatenate([[0], np.flatnonzero(np.any(np.diff(keys_sorted, axis=0) != 0, axis=1)) + 1])
    group_sizes = np.diff(np.concatenate([group_starts, [keys_sorted.shape[0]]])).astype(np.float32)

    xyz_s = cloud.xyz[order]
    centers = np.add.reduceat(xyz_s, group_starts, axis=0) / group_sizes[:, None]
    score = np.linalg.norm(xyz_s - np.repeat(centers, group_sizes.astype(np.int64), axis=0), axis=1)
    if cloud.confidence is not None:
        score -= cloud.confidence[order] * voxel_size
    if cloud.distance_to_camera is not None:
        score += cloud.distance_to_camera[order] * voxel_size * 0.01

    best = np.minimum.reduceat(score, group_starts)
    candidate_mask = score == np.repeat(best, group_sizes.astype(np.int64))
    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_groups = np.searchsorted(group_starts, candidate_indices, side="right") - 1
    _, first_candidate_positions = np.unique(candidate_groups, return_index=True)
    idx = order[candidate_indices[first_candidate_positions]]
    return SemanticPointCloud(
        xyz=cloud.xyz[idx],
        rgb=cloud.rgb[idx],
        labels=cloud.labels[idx],
        frame_indices=None if cloud.frame_indices is None else cloud.frame_indices[idx],
        confidence=None if cloud.confidence is None else cloud.confidence[idx],
        distance_to_camera=None if cloud.distance_to_camera is None else cloud.distance_to_camera[idx],
    )


def nearest_camera_replace_semantic_cloud(
    cloud: SemanticPointCloud,
    radius: float,
    *,
    progress: Callable[[str], None] | None = None,
) -> SemanticPointCloud:
    """Drop all but the nearest-to-camera point in each voxel of side `radius`.

    `progress` gets each sub-step name so callers can drive UI through the sort,
    which dominates on multi-million-point clouds.
    """
    def _emit(name: str) -> None:
        if progress is not None:
            try:
                progress(name)
            except Exception:
                pass

    if len(cloud) == 0 or radius <= 0 or not np.isfinite(radius):
        return cloud
    if cloud.distance_to_camera is None:
        return cloud
    _emit("keys")
    keys = np.floor(cloud.xyz / float(radius)).astype(np.int64)
    distance = np.asarray(cloud.distance_to_camera, dtype=np.float32).reshape(-1)
    _emit("sort")
    # Pack the three axes into one int64 (ix in the high bits) so a 2-key sort
    # reproduces the old (ix, iy, iz, distance) lexsort exactly: same nearest
    # point per voxel and same row order, so the cloud and its PLY stay
    # byte-identical while the sort roughly halves in time and peak memory.
    # lexsort is stable, so equal (voxel, distance) rows keep input order like
    # the old explicit arange key. Per-axis fallback if a voxel index overflows.
    packed = _pack_voxel_keys_int64(keys)
    if packed is not None:
        order = np.lexsort((distance, packed))
        packed_sorted = packed[order]
        selected = order[np.concatenate([[True], packed_sorted[1:] != packed_sorted[:-1]])]
    else:
        order = np.lexsort((distance, keys[:, 2], keys[:, 1], keys[:, 0]))
        keys_sorted = keys[order]
        selected = order[np.concatenate([[True], np.any(np.diff(keys_sorted, axis=0) != 0, axis=1)])]
    _emit("select")
    return SemanticPointCloud(
        xyz=cloud.xyz[selected],
        rgb=cloud.rgb[selected],
        labels=cloud.labels[selected],
        frame_indices=None if cloud.frame_indices is None else cloud.frame_indices[selected],
        confidence=None if cloud.confidence is None else cloud.confidence[selected],
        distance_to_camera=cloud.distance_to_camera[selected],
    )


def _voxel_sort_order(keys: np.ndarray) -> np.ndarray:
    # ix in the high bits makes ascending packed order match the per-axis
    # (ix, iy, iz) lexsort, with arange breaking ties to the original index, so
    # the reduced cloud keeps byte-identical row order. Per-axis on overflow.
    arange = np.arange(keys.shape[0], dtype=np.int64)
    packed = _pack_voxel_keys_int64(keys)
    if packed is not None:
        return np.lexsort((arange, packed))
    return np.lexsort((arange, keys[:, 2], keys[:, 1], keys[:, 0]))


def estimate_replacement_radius(
    depth_maps: np.ndarray,
    *,
    first_k: int,
    min_depth: float = 0.05,
    max_depth: float = 8.0,
) -> float | None:
    depth = np.asarray(depth_maps, dtype=np.float32)
    if depth.size == 0:
        return None
    k = max(1, int(first_k))
    sample = depth[:k]
    valid = np.isfinite(sample)
    valid &= sample >= float(min_depth)
    valid &= sample <= float(max_depth)
    if not np.any(valid):
        return None
    median_depth = float(np.median(sample[valid]))
    return float(np.clip(0.005 * median_depth, 0.002, 0.02))


def _resolve_replacement_radius(cfg: PointFilterConfig, depth_maps: np.ndarray) -> float | None:
    if cfg.replacement_radius_override is not None:
        r = float(cfg.replacement_radius_override)
        if not np.isfinite(r) or r <= 0:
            return None
        return r
    factor = float(cfg.replacement_radius_factor)
    if not np.isfinite(factor) or factor <= 0:
        return None
    base = estimate_replacement_radius(
        depth_maps,
        first_k=cfg.replacement_radius_estimation_frames,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
    )
    if base is None:
        return None
    return float(base * factor)


def _resize_nearest(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size_wh, interpolation=cv2.INTER_NEAREST)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size_wh, interpolation=cv2.INTER_AREA)
