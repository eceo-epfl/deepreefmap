from pathlib import Path

import numpy as np

from deepreefmap.config.classes import ClassConfig, SemanticClass
from deepreefmap.pointcloud.grid_ortho import OrthoGrid
from deepreefmap.postproc.ortho_outputs import TransectCropParams, apply_ortho_crop


def _classes() -> ClassConfig:
    return ClassConfig(
        classes=(
            SemanticClass(1, "reef", (10, 20, 30), frozenset()),
            SemanticClass(9, "transect", (255, 255, 255), frozenset({"transect_line"})),
        ),
        path=Path("test"),
    )


def _grid() -> OrthoGrid:
    labels = np.ones((10, 10), dtype=np.int32)
    labels[5, 2:8] = 9
    return OrthoGrid(
        rgb=np.full((10, 10, 3), 100, dtype=np.uint8),
        labels=labels,
        height=np.zeros((10, 10), dtype=np.float32),
        counts=np.ones((10, 10), dtype=np.int32),
        frame_index=np.zeros((10, 10), dtype=np.int32),
        cell_size=1.0,
    )


def test_apply_ortho_crop_defaults_to_uncropped_grid() -> None:
    base = _grid()

    outputs = apply_ortho_crop(base, _classes(), crop=None)

    assert outputs.grid is base
    assert outputs.cropped is False
    assert outputs.cover["denominator"] == 100.0


def test_apply_ortho_crop_recomputes_cover_for_transect_window() -> None:
    outputs = apply_ortho_crop(
        _grid(),
        _classes(),
        crop=TransectCropParams(transect_length_m=6.0, crop_width_m=2.0),
    )

    assert outputs.cropped is True
    assert outputs.grid.rgb.shape[0] < 10
    assert outputs.grid.rgb.shape[1] < 10
    assert outputs.cover["denominator"] == float(outputs.grid.counts.sum())
