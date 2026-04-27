import numpy as np

from deepreefmap.visualization.live_frame_cloud import build_enabled_label_lut, mask_points_by_enabled_lut


def test_lut_masks_labels() -> None:
    lut = build_enabled_label_lut(5, {1, 3})
    labels = np.array([0, 1, 2, 3, 5, 99], dtype=np.int32)
    m = mask_points_by_enabled_lut(labels, lut)
    expected = np.array([False, True, False, True, False, False], dtype=bool)
    assert np.array_equal(m, expected)


def test_lut_expands_for_large_label() -> None:
    lut = build_enabled_label_lut(3, {1})
    labels = np.array([10], dtype=np.int32)
    m = mask_points_by_enabled_lut(labels, lut)
    assert m.shape == (1,)
    assert not bool(m[0])
