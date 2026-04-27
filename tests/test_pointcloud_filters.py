from pathlib import Path

import numpy as np

from deepreefmap.config.classes import ClassConfig, SemanticClass
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame
from deepreefmap.pointcloud.filters import (
    estimate_neighborhood_size_from_depth_maps,
    PointFilterConfig,
    build_semantic_reference_cloud,
    nearest_camera_filter,
    voxel_reduce_semantic_cloud,
)
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
        PointFilterConfig(voxel_size=None, confidence_percentile=None, min_confidence=0.5),
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


def test_nearest_camera_filter_keeps_nearest_per_neighborhood():
    cloud = SemanticPointCloud(
        xyz=np.array(
            [
                [0.000, 0.000, 0.0],
                [0.004, 0.000, 0.0],
                [0.020, 0.000, 0.0],
            ],
            dtype=np.float32,
        ),
        rgb=np.zeros((3, 3), dtype=np.uint8),
        labels=np.array([1, 2, 3], dtype=np.int32),
        distance_to_camera=np.array([2.0, 1.0, 0.5], dtype=np.float32),
    )

    reduced = nearest_camera_filter(cloud, neighborhood_size=0.01)

    assert reduced.labels.tolist() == [2, 3]


def test_build_semantic_reference_cloud_applies_nearest_camera_filter():
    frame = PreparedFrame(
        frame_index=0,
        image_rgb=np.full((2, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 1], [1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255], [255, 255]], dtype=np.uint8),
    )
    mapping = MappingSequenceResult(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=np.array([[[2.0, 1.0], [1.0, 1.0]]], dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=np.array(
            [
                [
                    [[0.000, 0.000, 0.0], [0.004, 0.000, 0.0]],
                    [[0.020, 0.000, 0.0], [0.030, 0.000, 0.0]],
                ]
            ],
            dtype=np.float32,
        ),
        confidence=np.ones((1, 2, 2), dtype=np.float32),
    )
    batch = FrameBatch(frames=(frame,), intrinsics=np.eye(3, dtype=np.float32), image_size=(2, 2), clip_counts=(1,))

    cloud = build_semantic_reference_cloud(
        batch,
        mapping,
        _classes(),
        PointFilterConfig(
            voxel_size=None,
            neighborhood_size_factor=1.0,
            confidence_percentile=None,
            min_confidence=0.0,
        ),
    )

    assert len(cloud) == 3
    assert np.allclose(
        cloud.xyz,
        np.array(
            [
                [0.004, 0.0, 0.0],
                [0.020, 0.0, 0.0],
                [0.030, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_estimate_neighborhood_size_from_depth_maps_uses_depth_statistics():
    depth_maps = np.array([[[1.0, 2.0], [3.0, np.nan]]], dtype=np.float32)
    size = estimate_neighborhood_size_from_depth_maps(depth_maps, min_depth=0.05, max_depth=8.0)
    assert size is not None
    # median([1,2,3]) == 2 -> 0.005 * 2 = 0.01 (within clamp range)
    assert np.isclose(size, 0.01)


def test_build_semantic_reference_cloud_uses_auto_neighborhood_default():
    frame = PreparedFrame(
        frame_index=0,
        image_rgb=np.full((2, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 1], [1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255], [255, 255]], dtype=np.uint8),
    )
    mapping = MappingSequenceResult(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=np.array([[[2.0, 2.0], [2.0, 2.0]]], dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=np.array(
            [
                [
                    [[0.000, 0.000, 0.0], [0.004, 0.000, 0.0]],
                    [[0.020, 0.000, 0.0], [0.030, 0.000, 0.0]],
                ]
            ],
            dtype=np.float32,
        ),
        confidence=np.ones((1, 2, 2), dtype=np.float32),
    )
    batch = FrameBatch(frames=(frame,), intrinsics=np.eye(3, dtype=np.float32), image_size=(2, 2), clip_counts=(1,))

    cloud = build_semantic_reference_cloud(
        batch,
        mapping,
        _classes(),
        PointFilterConfig(
            voxel_size=None,
            neighborhood_size_factor=None,
            neighborhood_filter_every_k_frames=30,
            confidence_percentile=None,
            min_confidence=0.0,
        ),
    )

    # auto neighborhood at depth=2m -> 0.01m, so first two points collide and one survives
    assert len(cloud) == 3
    assert np.allclose(
        cloud.xyz,
        np.array(
            [
                [0.000, 0.0, 0.0],
                [0.020, 0.0, 0.0],
                [0.030, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_semantic_reference_cloud_applies_factor_and_final_pass():
    frame0 = PreparedFrame(
        frame_index=0,
        image_rgb=np.full((1, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255]], dtype=np.uint8),
    )
    frame1 = PreparedFrame(
        frame_index=1,
        image_rgb=np.full((1, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255]], dtype=np.uint8),
    )
    frame2 = PreparedFrame(
        frame_index=2,
        image_rgb=np.full((1, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255]], dtype=np.uint8),
    )
    mapping = MappingSequenceResult(
        frame_indices=np.array([0, 1, 2], dtype=np.int32),
        depth_maps=np.array(
            [
                [[2.0, 2.0]],
                [[2.0, 2.0]],
                [[2.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        poses_w_c=np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0),
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=np.array(
            [
                [[[0.000, 0.0, 0.0], [0.004, 0.0, 0.0]]],
                [[[0.050, 0.0, 0.0], [0.054, 0.0, 0.0]]],
                [[[0.100, 0.0, 0.0], [0.104, 0.0, 0.0]]],
            ],
            dtype=np.float32,
        ),
        confidence=np.ones((3, 1, 2), dtype=np.float32),
    )
    batch = FrameBatch(
        frames=(frame0, frame1, frame2),
        intrinsics=np.eye(3, dtype=np.float32),
        image_size=(2, 1),
        clip_counts=(3,),
    )

    cloud = build_semantic_reference_cloud(
        batch,
        mapping,
        _classes(),
        PointFilterConfig(
            voxel_size=None,
            neighborhood_size_factor=2.0,
            neighborhood_filter_every_k_frames=2,
            confidence_percentile=None,
            min_confidence=0.0,
        ),
    )

    # base auto size at depth=2m is 0.01; factor 2.0 => 0.02.
    # Each frame has two points within same neighborhood, so exactly one survives each frame.
    assert len(cloud) == 3
    assert np.allclose(
        cloud.xyz,
        np.array(
            [
                [0.000, 0.0, 0.0],
                [0.050, 0.0, 0.0],
                [0.100, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
