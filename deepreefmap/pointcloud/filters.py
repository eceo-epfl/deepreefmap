from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from deepreefmap.config.classes import ClassConfig
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, SemanticPointCloud
from deepreefmap.pointcloud.unprojection import depth_to_points

_BIAS = np.int64(1 << 20)
_MASK = np.int64((1 << 21) - 1)


def _pack_voxel_key(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """Packed int keys where in-range; object array with int or tuple for overflow rows."""
    xa = ix.astype(np.int64, copy=False) + _BIAS
    ya = iy.astype(np.int64, copy=False) + _BIAS
    za = iz.astype(np.int64, copy=False) + _BIAS
    in_range = (xa >= 0) & (xa <= _MASK) & (ya >= 0) & (ya <= _MASK) & (za >= 0) & (za <= _MASK)
    n = int(ix.shape[0])
    out = np.empty(n, dtype=object)
    if np.any(in_range):
        xi = xa[in_range].astype(np.uint64)
        yi = ya[in_range].astype(np.uint64)
        zi = za[in_range].astype(np.uint64)
        packed = xi | (yi << np.uint64(21)) | (zi << np.uint64(42))
        ir = np.flatnonzero(in_range)
        out[ir] = packed.astype(object)
    if np.any(~in_range):
        ir = np.flatnonzero(~in_range)
        for j in ir:
            out[int(j)] = (int(ix[j]), int(iy[j]), int(iz[j]))
    return out


class _GrowableArray2D:
    def __init__(self, cols: int, dtype: np.dtype) -> None:
        self.cols = int(cols)
        self.dtype = dtype
        self._data = np.zeros((512, self.cols), dtype=dtype)
        self.size = 0

    def _ensure_capacity(self, need: int) -> None:
        if self.size + need <= len(self._data):
            return
        new_len = max(len(self._data) * 2, self.size + need)
        new_buf = np.zeros((new_len, self.cols), dtype=self.dtype)
        if self.size:
            new_buf[: self.size] = self._data[: self.size]
        self._data = new_buf

    def append_rows(self, rows: np.ndarray) -> None:
        n = int(rows.shape[0])
        if n == 0:
            return
        self._ensure_capacity(n)
        self._data[self.size : self.size + n] = rows
        self.size += n

    def set_row(self, i: int, row: np.ndarray) -> None:
        self._data[i] = row

    def view(self) -> np.ndarray:
        return self._data[: self.size]


class _GrowableArray1D:
    def __init__(self, dtype: np.dtype) -> None:
        self.dtype = dtype
        self._data = np.zeros(512, dtype=dtype)
        self.size = 0

    def _ensure_capacity(self, need: int) -> None:
        if self.size + need <= len(self._data):
            return
        new_len = max(len(self._data) * 2, self.size + need)
        new_buf = np.zeros(new_len, dtype=self.dtype)
        if self.size:
            new_buf[: self.size] = self._data[: self.size]
        self._data = new_buf

    def append(self, values: np.ndarray) -> None:
        n = int(values.shape[0])
        if n == 0:
            return
        self._ensure_capacity(n)
        self._data[self.size : self.size + n] = values
        self.size += n

    def __setitem__(self, i: int, v) -> None:
        self._data[i] = v

    def view(self) -> np.ndarray:
        return self._data[: self.size]


class NearestCameraVoxelMap:
    """Winner-takes-all store: one point per voxel cell; closer-to-camera replaces occupant."""

    def __init__(self, radius: float) -> None:
        if radius <= 0 or not np.isfinite(radius):
            raise ValueError("radius must be finite and positive")
        self._radius = float(radius)
        self._key_to_idx: dict[object, int] = {}
        self._xyz = _GrowableArray2D(3, np.float32)
        self._rgb = _GrowableArray2D(3, np.uint8)
        self._labels = _GrowableArray1D(np.int32)
        self._frame_indices = _GrowableArray1D(np.int32)
        self._confidence = _GrowableArray1D(np.float32)
        self._distance = _GrowableArray1D(np.float32)

    def add_points(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        labels: np.ndarray,
        frame_index: int,
        confidence: np.ndarray,
        distance: np.ndarray,
    ) -> None:
        n = int(xyz.shape[0])
        if n == 0:
            return
        r = self._radius
        ix = np.floor(xyz[:, 0] / r).astype(np.int64, copy=False)
        iy = np.floor(xyz[:, 1] / r).astype(np.int64, copy=False)
        iz = np.floor(xyz[:, 2] / r).astype(np.int64, copy=False)
        order = np.lexsort((distance, iz, iy, ix))
        cx, cy, cz = ix[order], iy[order], iz[order]
        same_cell = (np.diff(cx) == 0) & (np.diff(cy) == 0) & (np.diff(cz) == 0)
        first_in_cell = np.concatenate([[True], ~same_cell])
        w = order[first_in_cell]

        xyz_w = xyz[w]
        rgb_w = rgb[w]
        lab_w = labels[w]
        conf_w = confidence[w]
        dist_w = distance[w]
        keys = _pack_voxel_key(ix[w], iy[w], iz[w])

        fi = np.int32(frame_index)
        for i in range(int(w.shape[0])):
            raw = keys[i]
            key: object = raw if isinstance(raw, tuple) else int(raw)
            di = float(dist_w[i])
            existing = self._key_to_idx.get(key)
            if existing is None:
                j = self._xyz.size
                self._key_to_idx[key] = j
                self._xyz.append_rows(xyz_w[i : i + 1])
                self._rgb.append_rows(rgb_w[i : i + 1])
                self._labels.append(lab_w[i : i + 1])
                self._frame_indices.append(np.array([fi], dtype=np.int32))
                self._confidence.append(conf_w[i : i + 1])
                self._distance.append(np.array([di], dtype=np.float32))
            else:
                if di < float(self._distance.view()[existing]):
                    self._xyz.set_row(existing, xyz_w[i])
                    self._rgb.set_row(existing, rgb_w[i])
                    self._labels[existing] = lab_w[i]
                    self._frame_indices[existing] = fi
                    self._confidence[existing] = conf_w[i]
                    self._distance[existing] = di

    def to_semantic_cloud(self) -> SemanticPointCloud:
        if self._xyz.size == 0:
            return SemanticPointCloud.empty()
        return SemanticPointCloud(
            xyz=self._xyz.view().copy(),
            rgb=self._rgb.view().copy(),
            labels=self._labels.view().copy(),
            frame_indices=self._frame_indices.view().copy(),
            confidence=self._confidence.view().copy(),
            distance_to_camera=self._distance.view().copy(),
        )


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
) -> SemanticPointCloud:
    cfg = config or PointFilterConfig()
    ignore_labels = classes_config.ids_for_role("ignore_in_point_cloud")
    frame_lookup = {frame.frame_index: frame for frame in frame_batch.frames}
    active_radius = _resolve_replacement_radius(cfg, mapping.depth_maps)

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []
    dist_parts: list[np.ndarray] = []

    for result_i, frame_index in enumerate(mapping.frame_indices.tolist()):
        frame = frame_lookup.get(int(frame_index))
        if frame is None:
            continue
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
        if ignore_labels:
            valid &= ~np.isin(labels, list(ignore_labels))
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
            continue
        xyz_f = xyz[flat_valid]
        rgb_f = rgb.reshape(-1, 3)[flat_valid].astype(np.uint8)
        lab_f = labels.reshape(-1)[flat_valid].astype(np.int32)
        dist_f = depth.reshape(-1)[flat_valid].astype(np.float32)
        if confidence is not None:
            conf_f = confidence.reshape(-1)[flat_valid].astype(np.float32)
        else:
            conf_f = np.ones(int(flat_valid.sum()), dtype=np.float32)

        n = int(xyz_f.shape[0])
        xyz_parts.append(xyz_f)
        rgb_parts.append(rgb_f)
        label_parts.append(lab_f)
        frame_parts.append(np.full(n, int(frame_index), dtype=np.int32))
        conf_parts.append(conf_f)
        dist_parts.append(dist_f)

    if not xyz_parts:
        return SemanticPointCloud.empty()

    cloud = SemanticPointCloud(
        xyz=np.concatenate(xyz_parts, axis=0),
        rgb=np.concatenate(rgb_parts, axis=0),
        labels=np.concatenate(label_parts, axis=0),
        frame_indices=np.concatenate(frame_parts, axis=0),
        confidence=np.concatenate(conf_parts, axis=0),
        distance_to_camera=np.concatenate(dist_parts, axis=0),
    )

    if active_radius is not None:
        cloud = nearest_camera_replace_semantic_cloud(cloud, active_radius)

    if cfg.voxel_size is None or cfg.voxel_size <= 0:
        return cloud
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


def nearest_camera_replace_semantic_cloud(cloud: SemanticPointCloud, radius: float) -> SemanticPointCloud:
    if len(cloud) == 0 or radius <= 0 or not np.isfinite(radius):
        return cloud
    if cloud.distance_to_camera is None:
        return cloud
    keys = np.floor(cloud.xyz / float(radius)).astype(np.int64)
    distance = np.asarray(cloud.distance_to_camera, dtype=np.float32).reshape(-1)
    order = np.lexsort((np.arange(len(cloud), dtype=np.int64), distance, keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    selected = order[np.concatenate([[True], np.any(np.diff(keys_sorted, axis=0) != 0, axis=1)])]
    return SemanticPointCloud(
        xyz=cloud.xyz[selected],
        rgb=cloud.rgb[selected],
        labels=cloud.labels[selected],
        frame_indices=None if cloud.frame_indices is None else cloud.frame_indices[selected],
        confidence=None if cloud.confidence is None else cloud.confidence[selected],
        distance_to_camera=cloud.distance_to_camera[selected],
    )


def _voxel_sort_order(keys: np.ndarray) -> np.ndarray:
    return np.lexsort((np.arange(keys.shape[0], dtype=np.int64), keys[:, 2], keys[:, 1], keys[:, 0]))


def nearest_camera_filter(cloud: SemanticPointCloud, neighborhood_size: float) -> SemanticPointCloud:
    if len(cloud) == 0 or neighborhood_size <= 0:
        return cloud
    if cloud.distance_to_camera is None:
        return cloud
    keys = np.floor(cloud.xyz / neighborhood_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    split_points = np.flatnonzero(np.any(np.diff(keys_sorted, axis=0) != 0, axis=1)) + 1
    groups = np.split(order, split_points)
    selected: list[int] = []
    for group in groups:
        if group.size == 1:
            selected.append(int(group[0]))
            continue
        nearest_idx = int(group[int(np.argmin(cloud.distance_to_camera[group]))])
        selected.append(nearest_idx)
    idx = np.asarray(selected, dtype=np.int64)
    return SemanticPointCloud(
        xyz=cloud.xyz[idx],
        rgb=cloud.rgb[idx],
        labels=cloud.labels[idx],
        frame_indices=None if cloud.frame_indices is None else cloud.frame_indices[idx],
        confidence=None if cloud.confidence is None else cloud.confidence[idx],
        distance_to_camera=None if cloud.distance_to_camera is None else cloud.distance_to_camera[idx],
    )


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
