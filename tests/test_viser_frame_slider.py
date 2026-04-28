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


def test_colorize_depth_maps_all_finite_pixels() -> None:
    """2D depth strip uses full finite range; depth viz cap applies only to 3D live cloud."""
    app = _app_without_init()
    depth = np.array([[0.5, 1.5], [2.0, float("nan")]], dtype=np.float32)
    rgb = ViserLiveApp._colorize_depth(app, depth)
    assert tuple(rgb[1, 1].tolist()) == (0, 0, 0)
    assert int(rgb[0, 0].sum()) > 0
    assert int(rgb[0, 1].sum()) > 0
    assert int(rgb[1, 0].sum()) > 0


def test_camera_view_params_use_pose_position_and_fov() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([1.0, 2.0, 3.0])

    position, wxyz, fov = ViserLiveApp._camera_view_params(pose, fov_y=0.75)

    assert position == (1.0, 2.0, 3.0)
    assert wxyz == (1.0, 0.0, 0.0, 0.0)
    assert fov == 0.75


def test_camera_view_params_backoff_moves_behind_camera() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([1.0, 2.0, 3.0])

    position, _wxyz, _fov = ViserLiveApp._camera_view_params(pose, fov_y=0.75, backoff=0.5)

    assert position == (1.0, 2.0, 2.5)
