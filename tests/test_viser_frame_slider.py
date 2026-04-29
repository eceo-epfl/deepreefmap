from pathlib import Path

import numpy as np

import deepreefmap.visualization.viser_app as viser_app_mod
from deepreefmap.config.classes import ClassConfig, SemanticClass
from deepreefmap.pointcloud.grid_ortho import OrthoGrid
from deepreefmap.postproc.ortho_outputs import OrthoOutputs
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


def test_ortho_preview_image_stacks_rgb_and_class_views() -> None:
    grid = OrthoGrid(
        rgb=np.full((2, 3, 3), 25, dtype=np.uint8),
        labels=np.array([[1, 2, 1], [2, 1, 2]], dtype=np.int32),
        height=np.zeros((2, 3), dtype=np.float32),
        counts=np.ones((2, 3), dtype=np.int32),
        frame_index=np.zeros((2, 3), dtype=np.int32),
        cell_size=1.0,
    )

    preview = ViserLiveApp._ortho_preview_image(grid, {1: (10, 20, 30), 2: (200, 210, 220)})

    assert preview.shape == (2, 6, 3)
    assert preview[0, 0].tolist() == [25, 25, 25]
    assert preview[0, 3].tolist() == [10, 20, 30]
    assert preview[0, 4].tolist() == [200, 210, 220]


def test_cover_summary_markdown_reports_crop_state_and_top_classes() -> None:
    grid = OrthoGrid(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        labels=np.ones((2, 2), dtype=np.int32),
        height=np.zeros((2, 2), dtype=np.float32),
        counts=np.ones((2, 2), dtype=np.int32),
        frame_index=np.zeros((2, 2), dtype=np.int32),
        cell_size=1.0,
    )
    cover = {
        "denominator": 4.0,
        "classes": {
            "1": {"name": "reef", "fraction": 0.75},
            "2": {"name": "sand", "fraction": 0.25},
        },
    }

    summary = ViserLiveApp._cover_summary_markdown(cover, grid, "cropped")

    assert "State: **cropped**" in summary
    assert "reef: `75.0%`" in summary
    assert "sand: `25.0%`" in summary


class _FakeHandle:
    def __init__(self, value=None):
        self.value = value
        self.image = None
        self.content = ""


class _FakeEvent:
    def __init__(self):
        self.called = False

    def set(self):
        self.called = True


def test_refresh_ortho_crop_preview_uses_toggle_and_slider_values(monkeypatch) -> None:
    app = _app_without_init()
    grid = OrthoGrid(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        labels=np.ones((2, 2), dtype=np.int32),
        height=np.zeros((2, 2), dtype=np.float32),
        counts=np.ones((2, 2), dtype=np.int32),
        frame_index=np.zeros((2, 2), dtype=np.int32),
        cell_size=1.0,
    )
    app._ortho_base_grid = grid
    app._ortho_classes_config = ClassConfig(
        classes=(SemanticClass(1, "reef", (10, 20, 30), frozenset()),),
        path=Path("test"),
    )
    app._ortho_crop_geometry = object()
    app._ortho_crop_selection = None
    app._ortho_image_handle = _FakeHandle()
    app._crop_summary_markdown_handle = _FakeHandle()
    app._crop_enabled_toggle = _FakeHandle(False)
    app._transect_length_slider = _FakeHandle(12.5)
    app._crop_width_slider = _FakeHandle(3.5)
    app._current_ortho_outputs = None
    app._active_crop_params = None
    app._crop_revision = 0
    app._dirty = _FakeEvent()
    calls = []

    def fake_apply_ortho_crop(base_grid, classes_config, *, crop, transect_geometry, transect_selection):
        calls.append(crop)
        return OrthoOutputs(grid=base_grid, cover={"classes": {}, "denominator": 4.0}, cropped=crop is not None)

    monkeypatch.setattr(viser_app_mod, "apply_ortho_crop", fake_apply_ortho_crop)
    monkeypatch.setattr(
        viser_app_mod,
        "build_transect_crop_selection",
        lambda geometry, *, transect_length_m, crop_width_m: object(),
    )

    ViserLiveApp._refresh_ortho_crop_preview(app)
    app._crop_enabled_toggle.value = True
    ViserLiveApp._refresh_ortho_crop_preview(app)

    assert calls[0] is None
    assert calls[1].transect_length_m == 12.5
    assert calls[1].crop_width_m == 3.5
    assert app._active_crop_params is calls[1]
    assert app._crop_revision == 2
    assert app._dirty.called is True
    assert app._ortho_image_handle.image is not None
    assert "State: **cropped**" in app._crop_summary_markdown_handle.content


def test_point_cloud_crop_filter_returns_none_when_crop_disabled() -> None:
    app = _app_without_init()
    app._active_crop_params = None
    app._ortho_base_grid = None

    assert ViserLiveApp._point_cloud_crop_filter(app) is None


def test_save_current_ortho_outputs_writes_grid_and_cover(tmp_path: Path) -> None:
    app = _app_without_init()
    grid = OrthoGrid(
        rgb=np.full((2, 2, 3), 50, dtype=np.uint8),
        labels=np.ones((2, 2), dtype=np.int32),
        height=np.zeros((2, 2), dtype=np.float32),
        counts=np.ones((2, 2), dtype=np.int32),
        frame_index=np.zeros((2, 2), dtype=np.int32),
        cell_size=1.0,
    )
    app._current_ortho_outputs = OrthoOutputs(
        grid=grid,
        cover={"classes": {}, "denominator": 4.0},
        cropped=False,
    )
    app._output_dir = tmp_path
    app._crop_summary_markdown_handle = _FakeHandle()

    ViserLiveApp._save_current_ortho_outputs(app)

    assert (tmp_path / "ortho.png").exists()
    assert (tmp_path / "ortho.npz").exists()
    assert (tmp_path / "benthic_cover.json").exists()
    assert "Saved:" in app._crop_summary_markdown_handle.content
