from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from deepreefmap.pipeline.artifacts import SemanticPointCloud


def align_tsdf_to_reference(
    tsdf_xyz: np.ndarray,
    tsdf_rgb: np.ndarray,
    reference: SemanticPointCloud,
    max_distance: float | None = None,
) -> SemanticPointCloud:
    if tsdf_xyz.size == 0 or len(reference) == 0:
        return SemanticPointCloud.empty()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(reference.xyz)
    distances, indices = nn.kneighbors(tsdf_xyz.astype(np.float32), return_distance=True)
    nearest = indices[:, 0]
    keep = np.ones(tsdf_xyz.shape[0], dtype=bool)
    if max_distance is not None:
        keep &= distances[:, 0] <= max_distance

    if not keep.any():
        return SemanticPointCloud.empty()

    ref_idx = nearest[keep]
    return SemanticPointCloud(
        xyz=tsdf_xyz[keep].astype(np.float32),
        rgb=tsdf_rgb[keep].astype(np.uint8),
        labels=reference.labels[ref_idx].astype(np.int32),
        frame_indices=None
        if reference.frame_indices is None
        else reference.frame_indices[ref_idx].astype(np.int32),
        confidence=None
        if reference.confidence is None
        else reference.confidence[ref_idx].astype(np.float32),
        distance_to_camera=None
        if reference.distance_to_camera is None
        else reference.distance_to_camera[ref_idx].astype(np.float32),
    )
