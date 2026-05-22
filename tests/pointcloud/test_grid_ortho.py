from pathlib import Path

import numpy as np

from deepreefmap.config.classes import ClassConfig, SemanticClass
from deepreefmap.pipeline.artifacts import SemanticPointCloud
from deepreefmap.pointcloud.grid_ortho import OrthoGrid, aggregate_cloud_to_ortho_grid
from deepreefmap.pointcloud.transect_crop import (
    build_transect_crop_geometry,
    crop_grid_around_transect,
    point_mask_with_transect_geometry,
)
from deepreefmap.postproc.benthic_cover import aggregate_cover, compute_benthic_cover


def test_aggregate_cloud_uses_mode_class_per_cell() -> None:
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


def test_aggregate_cloud_prefers_camera_facing_points_when_available() -> None:
    cloud = SemanticPointCloud(
        xyz=np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 1.0], [0.01, 0.01, 1.0]],
            dtype=np.float32,
        ),
        rgb=np.zeros((4, 3), dtype=np.uint8),
        labels=np.array([1, 1, 2, 2], dtype=np.int32),
        distance_to_camera=np.array([10.0, 10.0, 1.0, 1.0], dtype=np.float32),
    )

    grid = aggregate_cloud_to_ortho_grid(cloud, cell_size=10.0)

    assert 2 in grid.labels


def test_aggregate_cloud_returns_empty_grid_on_degenerate_input() -> None:
    cloud = SemanticPointCloud(
        xyz=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        rgb=np.zeros((3, 3), dtype=np.uint8),
        labels=np.array([1, 1, 1], dtype=np.int32),
    )

    grid = aggregate_cloud_to_ortho_grid(cloud, cell_size=0.1)

    assert grid.labels.shape == (1, 1)
    assert int(grid.labels.sum()) == 0


def test_benthic_cover_uses_classes_ignores_and_counts() -> None:
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


def test_aggregate_cover_rolls_classes_into_groups() -> None:
    # Two fine classes share a coarse group; aggregation should sum their counts
    # and fractions while keeping a per-class breakdown at the fine level.
    classes_config = ClassConfig(
        classes=(
            SemanticClass(1, "branching alive", (1, 1, 1), frozenset(), "branching alive", "coral alive"),
            SemanticClass(2, "acropora alive", (2, 2, 2), frozenset(), "coral alive", "coral alive"),
            SemanticClass(3, "sand", (3, 3, 3), frozenset(), "sand", "sand"),
        ),
        path=Path("test"),
    )
    cover = {
        "classes": {
            "1": {"name": "branching alive", "count": 30.0, "fraction": 0.3},
            "2": {"name": "acropora alive", "count": 50.0, "fraction": 0.5},
            "3": {"name": "sand", "count": 20.0, "fraction": 0.2},
        },
        "denominator": 100.0,
    }

    coarse = aggregate_cover(cover, classes_config, "coarse")
    assert coarse["coral alive"]["count"] == 80.0
    assert coarse["coral alive"]["fraction"] == 0.8
    assert coarse["sand"]["fraction"] == 0.2

    intermediate = aggregate_cover(cover, classes_config, "intermediate")
    # Branching keeps its own bucket; acropora rolled up to "coral alive".
    assert intermediate["branching alive"]["fraction"] == 0.3
    assert intermediate["coral alive"]["fraction"] == 0.5


def test_crop_grid_around_transect_sets_pixel_size() -> None:
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


def test_crop_grid_around_vertical_transect_uses_transect_direction() -> None:
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


def test_crop_grid_masks_pixels_outside_diagonal_transect_corridor() -> None:
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


def test_point_mask_with_transect_geometry_filters_projected_points() -> None:
    cloud = SemanticPointCloud(
        xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [1.0, 5.0, 0.0],
                [2.0, 5.0, 0.0],
                [3.0, 5.0, 0.0],
            ],
            dtype=np.float32,
        ),
        rgb=np.zeros((8, 3), dtype=np.uint8),
        labels=np.array([15, 15, 15, 15, 1, 1, 1, 1], dtype=np.int32),
    )
    grid = aggregate_cloud_to_ortho_grid(cloud, cell_size=1.0)
    geometry = build_transect_crop_geometry(grid.labels, transect_label=15, transect_tools_label=None)

    keep = point_mask_with_transect_geometry(
        grid,
        cloud.xyz,
        geometry,
        transect_length_m=3.0,
        crop_width_m=1.0,
    )

    assert keep[:4].any()
    assert not keep[4:].any()
