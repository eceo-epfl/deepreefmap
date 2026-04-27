import numpy as np

from deepreefmap.io.exports import save_ortho_grid, save_semantic_cloud
from deepreefmap.pipeline.artifacts import SemanticPointCloud
from deepreefmap.pointcloud.grid_ortho import OrthoGrid


def test_save_semantic_cloud_round_trip(tmp_path):
    path = tmp_path / "cloud.npz"
    cloud = SemanticPointCloud(
        xyz=np.ones((2, 3), dtype=np.float32),
        rgb=np.zeros((2, 3), dtype=np.uint8),
        labels=np.array([1, 2], dtype=np.int32),
    )

    save_semantic_cloud(path, cloud)

    data = np.load(path)
    assert data["xyz"].shape == (2, 3)
    assert data["labels"].tolist() == [1, 2]


def test_save_ortho_grid_contains_expected_keys(tmp_path):
    path = tmp_path / "ortho.npz"
    grid = OrthoGrid(
        rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        labels=np.array([[1]], dtype=np.int32),
        height=np.array([[0.5]], dtype=np.float32),
        counts=np.array([[3]], dtype=np.int32),
        frame_index=np.array([[4]], dtype=np.int32),
        cell_size=0.1,
    )

    save_ortho_grid(path, grid)

    data = np.load(path)
    assert {"rgb", "labels", "height", "counts", "frame_index", "cell_size"} <= set(data.files)
