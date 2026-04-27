from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from deepreefmap.config.taxonomy import Taxonomy
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, SemanticPointCloud
from deepreefmap.pointcloud.unprojection import depth_to_points


@dataclass(frozen=True)
class PointFilterConfig:
    min_depth: float = 0.05
    max_depth: float = 8.0
    confidence_percentile: float | None = 5.0
    min_confidence: float = 1e-5
    depth_edge_threshold: float | None = None
    stride: int = 4
    voxel_size: float | None = 0.003


def build_semantic_reference_cloud(
    frame_batch: FrameBatch,
    mapping: MappingSequenceResult,
    taxonomy: Taxonomy,
    config: PointFilterConfig | None = None,
) -> SemanticPointCloud:
    cfg = config or PointFilterConfig()
    ignore_labels = taxonomy.ids_for_role("ignore_in_point_cloud")
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []
    dist_parts: list[np.ndarray] = []
    frame_lookup = {frame.frame_index: frame for frame in frame_batch.frames}

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
        if cfg.stride > 1:
            stride_mask = np.zeros_like(valid, dtype=bool)
            stride_mask[:: cfg.stride, :: cfg.stride] = True
            valid &= stride_mask

        flat_valid = valid.reshape(-1)
        if not flat_valid.any():
            continue
        xyz_parts.append(xyz[flat_valid])
        rgb_parts.append(rgb.reshape(-1, 3)[flat_valid].astype(np.uint8))
        label_parts.append(labels.reshape(-1)[flat_valid].astype(np.int32))
        frame_parts.append(np.full(int(flat_valid.sum()), int(frame_index), dtype=np.int32))
        if confidence is not None:
            conf_parts.append(confidence.reshape(-1)[flat_valid].astype(np.float32))
        else:
            conf_parts.append(np.ones(int(flat_valid.sum()), dtype=np.float32))
        dist_parts.append(depth.reshape(-1)[flat_valid].astype(np.float32))

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
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    split_points = np.flatnonzero(np.any(np.diff(keys_sorted, axis=0) != 0, axis=1)) + 1
    groups = np.split(order, split_points)

    selected: list[int] = []
    confidence = cloud.confidence
    distances = cloud.distance_to_camera
    for group in groups:
        if group.size == 1:
            selected.append(int(group[0]))
            continue
        center = cloud.xyz[group].mean(axis=0)
        score = np.linalg.norm(cloud.xyz[group] - center, axis=1)
        if confidence is not None:
            score -= confidence[group] * voxel_size
        if distances is not None:
            score += distances[group] * voxel_size * 0.01
        selected.append(int(group[int(np.argmin(score))]))

    idx = np.asarray(selected, dtype=np.int64)
    return SemanticPointCloud(
        xyz=cloud.xyz[idx],
        rgb=cloud.rgb[idx],
        labels=cloud.labels[idx],
        frame_indices=None if cloud.frame_indices is None else cloud.frame_indices[idx],
        confidence=None if cloud.confidence is None else cloud.confidence[idx],
        distance_to_camera=None if cloud.distance_to_camera is None else cloud.distance_to_camera[idx],
    )


def _resize_nearest(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size_wh, interpolation=cv2.INTER_NEAREST)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size_wh, interpolation=cv2.INTER_AREA)
