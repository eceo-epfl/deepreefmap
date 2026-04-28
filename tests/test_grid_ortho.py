from pathlib import Path

import numpy as np

from deepreefmap.config.classes import ClassConfig, SemanticClass
from deepreefmap.pipeline.artifacts import SemanticPointCloud
from deepreefmap.pointcloud.grid_ortho import OrthoGrid, aggregate_cloud_to_ortho_grid
from deepreefmap.pointcloud.transect_crop import crop_grid_around_transect
from deepreefmap.postproc.benthic_cover import compute_benthic_cover


def test_aggregate_cloud_uses_mode_class_per_cell():
    cloud = SemanticPointCloud(
        xyz=np.array(
            [[0, 0, 0], [0.01, 0.01, 0], [0.02, 0, 0], [1, 1, 0]],
            dtype=np.float32,
        ),
        rgb=np.array([[10, 0, 0], [20, 0, 0], [30, 0, 0], [0, 40, 0]], dtype=np.uint8),
        labels=np.array([2, 2, 3, 4], dtype=np.int32),
    )

    grid = aggregate_cloud_to_ortho_grid(cloud, cell_size=0.1)

    assert 2 in grid.labels
    assert 4 in grid.labels


def test_aggregate_cloud_returns_empty_grid_on_degenerate_input():
    cloud = SemanticPointCloud(
        xyz=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        rgb=np.zeros((3, 3), dtype=np.uint8),
        labels=np.array([1, 1, 1], dtype=np.int32),
    )

    grid = aggregate_cloud_to_ortho_grid(cloud, cell_size=0.1)

    assert grid.labels.shape == (1, 1)
    assert int(grid.labels.sum()) == 0


def test_benthic_cover_uses_classes_ignores_and_counts():
    classes_config = ClassConfig(
        classes=(
            SemanticClass(0, "unlabeled", (0, 0, 0), frozenset({"ignore_in_cover"})),
            SemanticClass(1, "sand", (194, 178, 128), frozenset()),
            SemanticClass(7, "human", (255, 0, 0), frozenset({"ignore_in_cover"})),
        ),
        path=Path("test"),
    )
    labels = np.array([[1, 1], [7, 0]], dtype=np.int32)
    counts = np.array([[2, 3], [100, 50]], dtype=np.int32)

    cover = compute_benthic_cover(labels, classes_config=classes_config, counts=counts)

    assert cover["denominator"] == 5
    assert cover["classes"]["1"]["name"] == "sand"
    assert cover["classes"]["1"]["fraction"] == 1.0


def test_crop_grid_around_transect_sets_pixel_size():
    grid = OrthoGrid(
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        labels=np.zeros((10, 10), dtype=np.int32),
        height=np.zeros((10, 10), dtype=np.float32),
        counts=np.ones((10, 10), dtype=np.int32),
        frame_index=np.zeros((10, 10), dtype=np.int32),
        cell_size=1.0,
    )
    grid.labels[5, 2:8] = 15

    cropped = crop_grid_around_transect(grid, 15, None, transect_length_m=5.0, crop_width_m=2.0)

    assert cropped.pixel_size_m is not None
    assert cropped.rgb.shape[0] < grid.rgb.shape[0]


def test_crop_grid_around_vertical_transect_uses_transect_direction():
    grid = OrthoGrid(
        rgb=np.zeros((12, 12, 3), dtype=np.uint8),
        labels=np.zeros((12, 12), dtype=np.int32),
        height=np.zeros((12, 12), dtype=np.float32),
        counts=np.ones((12, 12), dtype=np.int32),
        frame_index=np.zeros((12, 12), dtype=np.int32),
        cell_size=1.0,
    )
    grid.labels[2:9, 6] = 15

    cropped = crop_grid_around_transect(grid, 15, None, transect_length_m=6.0, crop_width_m=4.0)

    assert np.isclose(cropped.pixel_size_m, 1.0)
    assert cropped.labels.shape[0] > cropped.labels.shape[1]
    assert 15 in cropped.labels


def test_crop_grid_masks_pixels_outside_diagonal_transect_corridor():
    grid = OrthoGrid(
        rgb=np.ones((20, 20, 3), dtype=np.uint8) * 255,
        labels=np.ones((20, 20), dtype=np.int32),
        height=np.ones((20, 20), dtype=np.float32),
        counts=np.ones((20, 20), dtype=np.int32),
        frame_index=np.zeros((20, 20), dtype=np.int32),
        cell_size=1.0,
    )
    for idx in range(3, 17):
        grid.labels[idx, idx] = 15

    cropped = crop_grid_around_transect(
        grid,
        15,
        None,
        transect_length_m=float(np.sqrt(13**2 + 13**2)),
        crop_width_m=3.0,
    )

    assert cropped.counts[0, -1] == 0
    assert cropped.frame_index[0, -1] == -1
    assert cropped.counts[cropped.counts.shape[0] // 2, cropped.counts.shape[1] // 2] > 0
