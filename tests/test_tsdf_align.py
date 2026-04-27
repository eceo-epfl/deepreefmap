import numpy as np

from deepreefmap.pipeline.artifacts import SemanticPointCloud
from deepreefmap.pointcloud.tsdf_align import align_tsdf_to_reference


def test_align_tsdf_to_reference_transfers_semantics():
    reference = SemanticPointCloud(
        xyz=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        rgb=np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8),
        labels=np.array([3, 5], dtype=np.int32),
        frame_indices=np.array([10, 11], dtype=np.int32),
        confidence=np.array([0.5, 0.9], dtype=np.float32),
    )
    tsdf_xyz = np.array([[1.01, 0, 0], [0.01, 0, 0]], dtype=np.float32)
    tsdf_rgb = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.uint8)

    aligned = align_tsdf_to_reference(tsdf_xyz, tsdf_rgb, reference, max_distance=0.1)

    assert aligned.labels.tolist() == [5, 3]
    assert aligned.frame_indices.tolist() == [11, 10]
    assert aligned.rgb.tolist() == [[10, 10, 10], [20, 20, 20]]


def test_align_tsdf_to_reference_respects_max_distance():
    reference = SemanticPointCloud(
        xyz=np.array([[0, 0, 0]], dtype=np.float32),
        rgb=np.zeros((1, 3), dtype=np.uint8),
        labels=np.array([3], dtype=np.int32),
    )

    aligned = align_tsdf_to_reference(
        np.array([[10, 0, 0]], dtype=np.float32),
        np.zeros((1, 3), dtype=np.uint8),
        reference,
        max_distance=0.1,
    )

    assert len(aligned) == 0
