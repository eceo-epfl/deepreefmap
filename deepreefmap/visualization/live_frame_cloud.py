"""Lazy full-depth unprojection for the current timeline frame (LRU cache)."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import cv2
import numpy as np

from deepreefmap.pointcloud.unprojection import depth_to_points

if TYPE_CHECKING:
    from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult

# Match defaults in pointcloud/filters.py for consistency with reference cloud construction.
_DEFAULT_MIN_DEPTH = 0.05
_DEFAULT_MAX_DEPTH = 8.0


def build_enabled_label_lut(max_label_id: int, enabled_classes: set[int]) -> np.ndarray:
    """Boolean LUT: lut[label] == True iff label is enabled (label in 0..max_label_id)."""
    size = max(int(max_label_id), 0) + 1
    lut = np.zeros(size, dtype=bool)
    for c in enabled_classes:
        ci = int(c)
        if 0 <= ci < size:
            lut[ci] = True
    return lut


def mask_points_by_enabled_lut(
    labels: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    """Return boolean mask of points to keep (same length as labels)."""
    lab = np.asarray(labels, dtype=np.int32).reshape(-1)
    out = np.zeros(lab.shape[0], dtype=bool)
    if lut.size == 0:
        return out
    in_range = (lab >= 0) & (lab < lut.shape[0])
    out[in_range] = lut[lab[in_range]]
    return out


class LiveFrameCloudCache:
    """Unproject depth to world points for one timeline frame; LRU-cache by timeline index."""

    def __init__(
        self,
        frame_batch: "FrameBatch",
        mapping: "MappingSequenceResult",
        frame_order: tuple[int, ...],
        min_depth: float = _DEFAULT_MIN_DEPTH,
        max_depth: float = _DEFAULT_MAX_DEPTH,
        max_depth_for_viz: float | None = None,
        lru_size: int = 8,
    ) -> None:
        self._frame_order = frame_order
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._max_depth_for_viz = None if max_depth_for_viz is None else float(max_depth_for_viz)
        self._lru_size = max(1, int(lru_size))

        self._mapping_frame_indices = np.asarray(mapping.frame_indices, dtype=np.int32).reshape(-1)
        self._depth_maps = mapping.depth_maps
        self._poses_w_c = mapping.poses_w_c
        self._intrinsics = np.asarray(mapping.intrinsics, dtype=np.float64)
        self._world_points = None if mapping.world_points is None else mapping.world_points

        self._frame_lookup = {int(f.frame_index): f for f in frame_batch.frames}
        self._mapping_index: dict[int, int] = {}
        for i, fid in enumerate(self._mapping_frame_indices.tolist()):
            self._mapping_index[int(fid)] = int(i)

        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def mapping_index_for_timeline(self, timeline_t: int) -> int:
        if timeline_t < 0 or timeline_t >= len(self._frame_order):
            raise IndexError(f"timeline_t={timeline_t} out of range for frame_order len={len(self._frame_order)}")
        frame_idx = int(self._frame_order[timeline_t])
        if frame_idx not in self._mapping_index:
            raise KeyError(f"No mapping result for frame_index={frame_idx}")
        return int(self._mapping_index[frame_idx])

    def get_unmasked(self, timeline_t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (xyz Nx3 float32, rgb Nx3 uint8, labels N int32) for full valid depth pixels."""
        if timeline_t in self._cache:
            self._cache.move_to_end(timeline_t)
            return self._cache[timeline_t]

        frame_idx = int(self._frame_order[timeline_t])
        frame = self._frame_lookup.get(frame_idx)
        if frame is None:
            empty_xyz = np.zeros((0, 3), dtype=np.float32)
            empty_rgb = np.zeros((0, 3), dtype=np.uint8)
            empty_lab = np.zeros((0,), dtype=np.int32)
            self._cache[timeline_t] = (empty_xyz, empty_rgb, empty_lab)
            self._cache.move_to_end(timeline_t)
            self._trim()
            return self._cache[timeline_t]

        mi = self.mapping_index_for_timeline(timeline_t)
        depth = np.asarray(self._depth_maps[mi], dtype=np.float32)
        pose_w_c = np.asarray(self._poses_w_c[mi], dtype=np.float64)
        h_d, w_d = depth.shape

        labels_full = np.asarray(frame.labels, dtype=np.int32)
        rgb_full = np.asarray(frame.image_rgb, dtype=np.uint8)
        keep = np.asarray(frame.keep_mask, dtype=np.uint8)

        labels_d = cv2.resize(labels_full, (w_d, h_d), interpolation=cv2.INTER_NEAREST)
        rgb_d = cv2.resize(rgb_full, (w_d, h_d), interpolation=cv2.INTER_AREA)
        keep_d = cv2.resize(keep, (w_d, h_d), interpolation=cv2.INTER_NEAREST) > 0

        valid = np.isfinite(depth) & (depth >= self._min_depth) & (depth <= self._max_depth) & keep_d
        if self._max_depth_for_viz is not None:
            valid &= depth <= self._max_depth_for_viz
        flat = valid.reshape(-1)

        if self._world_points is not None:
            xyz_flat = np.asarray(self._world_points[mi], dtype=np.float32).reshape(-1, 3)
        else:
            xyz_flat = depth_to_points(depth, self._intrinsics, pose_w_c).reshape(-1, 3).astype(np.float32, copy=False)
        xyz_w = xyz_flat[flat]
        rgb_flat = rgb_d.reshape(-1, 3).astype(np.uint8, copy=False)[flat]
        lab_flat = labels_d.reshape(-1).astype(np.int32, copy=False)[flat]

        self._cache[timeline_t] = (xyz_w, rgb_flat, lab_flat)
        self._cache.move_to_end(timeline_t)
        self._trim()
        return xyz_w, rgb_flat, lab_flat

    def _trim(self) -> None:
        while len(self._cache) > self._lru_size:
            self._cache.popitem(last=False)
