from pathlib import Path

import numpy as np

from deepreefmap.config.classes import ClassConfig, SemanticClass
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame
from deepreefmap.pointcloud.filters import PointFilterConfig, build_semantic_reference_cloud, voxel_reduce_semantic_cloud
from deepreefmap.pipeline.artifacts import SemanticPointCloud


def _classes():
    return ClassConfig(
        classes=(
            SemanticClass(1, "reef", (10, 10, 10), frozenset()),
            SemanticClass(7, "human", (255, 0, 0), frozenset({"ignore_in_point_cloud"})),
        ),
        path=Path("test"),
    )


def test_build_semantic_reference_cloud_filters_labels_and_confidence():
    frame = PreparedFrame(
        frame_index=0,
        image_rgb=np.full((2, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 7], [1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255], [255, 255]], dtype=np.uint8),
    )
    mapping = MappingSequenceResult(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=np.ones((1, 2, 2), dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3),
        confidence=np.array([[[0.9, 0.9], [0.0, 0.9]]], dtype=np.float32),
    )
    batch = FrameBatch(frames=(frame,), intrinsics=np.eye(3, dtype=np.float32), image_size=(2, 2), clip_counts=(1,))

    cloud = build_semantic_reference_cloud(
        batch,
        mapping,
        _classes(),
        PointFilterConfig(stride=1, voxel_size=None, confidence_percentile=None, min_confidence=0.5),
    )

    assert len(cloud) == 2
    assert set(cloud.labels.tolist()) == {1}
    assert cloud.xyz.tolist() == [[0.0, 1.0, 2.0], [9.0, 10.0, 11.0]]


def test_voxel_reduce_is_deterministic():
    cloud = SemanticPointCloud(
        xyz=np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        rgb=np.zeros((3, 3), dtype=np.uint8),
        labels=np.array([1, 2, 3], dtype=np.int32),
        confidence=np.array([0.1, 0.9, 0.2], dtype=np.float32),
    )

    reduced = voxel_reduce_semantic_cloud(cloud, voxel_size=0.01)

    assert reduced.labels.tolist() == [2, 3]
