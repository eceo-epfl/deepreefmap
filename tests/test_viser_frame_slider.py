import numpy as np

from deepreefmap.visualization.viser_app import ViserLiveApp


def _app_without_init() -> ViserLiveApp:
    return ViserLiveApp.__new__(ViserLiveApp)


def test_normalize_slider_position_rounds_and_clamps() -> None:
    app = _app_without_init()
    assert app._normalize_slider_position(2.49, max_pos=5) == 2
    assert app._normalize_slider_position(2.51, max_pos=5) == 3
    assert app._normalize_slider_position(-10.0, max_pos=5) == 0
    assert app._normalize_slider_position(999.0, max_pos=5) == 5


def test_normalize_slider_position_handles_non_finite_inputs() -> None:
    app = _app_without_init()
    assert app._normalize_slider_position(float("nan"), max_pos=7) == 7
    assert app._normalize_slider_position(float("inf"), max_pos=7) == 7
    assert app._normalize_slider_position(-float("inf"), max_pos=7) == 7
    assert app._normalize_slider_position("not-a-number", max_pos=7) == 7
    assert app._normalize_slider_position(None, max_pos=7) == 7


def test_colorize_depth_dims_pixels_beyond_max_depth() -> None:
    app = _app_without_init()
    depth = np.array([[0.5, 1.5], [2.0, float("nan")]], dtype=np.float32)
    rgb = ViserLiveApp._colorize_depth(app, depth, max_depth=1.0)
    assert tuple(rgb[0, 0].tolist()) != (40, 40, 40)
    assert tuple(rgb[0, 1]) == (40, 40, 40)
    assert tuple(rgb[1, 0]) == (40, 40, 40)
    assert tuple(rgb[1, 1].tolist()) == (0, 0, 0)
