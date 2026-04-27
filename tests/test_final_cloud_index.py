import numpy as np

from deepreefmap.pipeline.artifacts import SemanticPointCloud
from deepreefmap.visualization.final_cloud_index import build_final_cloud_index, median_distance_to_camera


def test_prefix_end_counts_by_timeline_rank() -> None:
    # frame_order: 10, 20, 30 — ranks 0,1,2
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    rgb = np.ones((4, 3), dtype=np.uint8) * 200
    labels = np.array([7, 7, 7, 7], dtype=np.int32)
    frame_indices = np.array([10, 10, 20, 30], dtype=np.int32)
    cloud = SemanticPointCloud(xyz=xyz, rgb=rgb, labels=labels, frame_indices=frame_indices)
    colors = {7: (255, 0, 0)}
    idx = build_final_cloud_index(cloud, [10, 20, 30], colors)
    assert idx.class_ids == (7,)
    pe = idx.prefix_end_by_class[7]
    assert pe[0] == 2  # two points at frame 10
    assert pe[1] == 3  # + frame 20
    assert pe[2] == 4  # + frame 30


def test_median_distance_to_camera_none_when_missing() -> None:
    cloud = SemanticPointCloud(
        xyz=np.zeros((2, 3), dtype=np.float32),
        rgb=np.zeros((2, 3), dtype=np.uint8),
        labels=np.zeros(2, dtype=np.int32),
        frame_indices=np.zeros(2, dtype=np.int32),
    )
    assert median_distance_to_camera(cloud) is None


def test_final_cloud_index_excludes_points_beyond_median_distance() -> None:
    """Reference distances [1,2,3,10] => median 2.5; keep only <= 2.5 (two points)."""
    xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
    rgb = np.ones((4, 3), dtype=np.uint8) * 50
    labels = np.array([5, 5, 5, 5], dtype=np.int32)
    frame_indices = np.zeros(4, dtype=np.int32)
    dist = np.array([1.0, 2.0, 3.0, 10.0], dtype=np.float32)
    cloud = SemanticPointCloud(
        xyz=xyz,
        rgb=rgb,
        labels=labels,
        frame_indices=frame_indices,
        distance_to_camera=dist,
    )
    idx = build_final_cloud_index(cloud, [0], {5: (1, 2, 3)})
    assert idx.class_ids == (5,)
    assert idx.xyz_by_class[5].shape[0] == 2


def test_accumulate_off_means_zero_prefix_semantics_via_prefix_array() -> None:
    """prefix_end[t] at max t equals total points in class."""
    xyz = np.random.randn(5, 3).astype(np.float32)
    rgb = np.random.randint(0, 255, (5, 3), dtype=np.uint8)
    labels = np.ones(5, dtype=np.int32)
    frame_indices = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    cloud = SemanticPointCloud(xyz=xyz, rgb=rgb, labels=labels, frame_indices=frame_indices)
    idx = build_final_cloud_index(cloud, [0, 1, 2], {1: (10, 20, 30)})
    pe = idx.prefix_end_by_class[1]
    assert int(pe[2]) == 5
