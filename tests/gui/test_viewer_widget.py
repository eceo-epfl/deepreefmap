"""The pyvistaqt viewer widget and its pure-numpy render/pick helpers.

Widget tests run offscreen; VTK-dependent ones are skipped when no OpenGL
context is available. The helper tests (colourisation, frustum geometry,
world-up estimation, pick-pixel selection) touch no Qt.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_viewer_widget_creates(qapp) -> None:
    from deepreefmap.gui.viewer.widget import QtPointCloudViewer

    viewer = QtPointCloudViewer(class_colors={1: (255, 0, 0)}, class_names={1: "test"})
    assert viewer.n_frames == 0
    assert not viewer.has_scene_data


def test_viewer_show_point_cloud(qapp) -> None:
    from deepreefmap.gui.viewer.widget import QtPointCloudViewer

    viewer = QtPointCloudViewer()
    xyz = np.random.rand(100, 3).astype(np.float32)
    rgb = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
    viewer.show_point_cloud(xyz, rgb)


def test_viewer_empty_cloud_noop(qapp) -> None:
    from deepreefmap.gui.viewer.widget import QtPointCloudViewer

    viewer = QtPointCloudViewer()
    viewer.show_point_cloud(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))



def _fake_geometry_scene():
    from types import SimpleNamespace

    frame = SimpleNamespace(frame_index=0, image_rgb=np.zeros((4, 4, 3), dtype=np.uint8))
    fb = SimpleNamespace(frames=[frame], frame_indices=[0], clip_counts=[1])
    mr = SimpleNamespace(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=np.ones((1, 4, 4), dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
    )
    xyz = np.random.rand(50, 3).astype(np.float32)
    rgb = np.random.randint(0, 255, (50, 3), dtype=np.uint8)
    return fb, mr, xyz, rgb


def test_geometry_scene_enables_timeline(qapp) -> None:
    from deepreefmap.gui.viewer.widget import QtPointCloudViewer

    viewer = QtPointCloudViewer()
    fb, mr, xyz, rgb = _fake_geometry_scene()
    viewer.load_geometry_scene(fb, mr, xyz, rgb)

    assert viewer.is_geometry_mode
    assert viewer.has_scene_data
    assert viewer.n_frames == 1
    # The timeline update must work without a FinalCloudIndex (no semantic data).
    viewer.apply_geometry_state(timeline_t=0, point_size=3.0, frustums_visible=True)


def test_geometry_scene_clears_back_to_empty(qapp) -> None:
    from deepreefmap.gui.viewer.widget import QtPointCloudViewer

    viewer = QtPointCloudViewer()
    fb, mr, xyz, rgb = _fake_geometry_scene()
    viewer.load_geometry_scene(fb, mr, xyz, rgb)
    viewer._clear_scene_data()

    assert not viewer.is_geometry_mode
    assert not viewer.has_scene_data
    assert viewer.n_frames == 0


def test_colorize_seg_maps_classes() -> None:
    from deepreefmap.gui.viewer.render import _colorize_seg

    labels = np.array([[1, 2], [2, 1]], dtype=np.int32)
    colors = {1: (255, 0, 0), 2: (0, 255, 0)}
    result = _colorize_seg(labels, colors)
    assert result[0, 0].tolist() == [255, 0, 0]
    assert result[0, 1].tolist() == [0, 255, 0]


def test_colorize_seg_fallback_gray() -> None:
    from deepreefmap.gui.viewer.render import _colorize_seg

    labels = np.array([[99]], dtype=np.int32)
    result = _colorize_seg(labels, {})
    assert result[0, 0].tolist() == [128, 128, 128]


def test_colorize_depth_handles_nan() -> None:
    from deepreefmap.gui.viewer.render import _colorize_depth

    depth = np.array([[float("nan"), 1.0], [2.0, float("nan")]], dtype=np.float32)
    result = _colorize_depth(depth)
    assert result.shape == (2, 2, 3)
    assert result[0, 0].sum() == 0
    assert result[0, 1].sum() > 0


def test_colorize_depth_all_nan() -> None:
    from deepreefmap.gui.viewer.render import _colorize_depth

    depth = np.full((3, 3), float("nan"), dtype=np.float32)
    result = _colorize_depth(depth)
    assert result.sum() == 0


@pytest.mark.parametrize(
    "rgb, expected_r",
    [
        (np.array([[255, 0, 128]], dtype=np.uint8), 1.0),
        (np.array([[0.5, 0.0, 1.0]], dtype=np.float32), 0.5),
    ],
)
def test_to_rgba_normalizes_by_dtype(rgb, expected_r) -> None:
    from deepreefmap.gui.viewer.render import _to_rgba

    rgba = _to_rgba(rgb)
    assert rgba.shape == (1, 4)
    assert abs(rgba[0, 0] - expected_r) < 0.01
    assert abs(rgba[0, 3] - 1.0) < 0.01


def test_build_frustum_lines_shape() -> None:
    from deepreefmap.gui.viewer.render import _build_frustum_lines

    pose = np.eye(4, dtype=np.float64)
    lines = _build_frustum_lines(pose, fov_y=1.0, aspect=1.5)
    # 4 edges from origin + 4 edges around rectangle = 8 line segments = 16 points
    assert lines.shape == (16, 3)


def test_build_frustum_lines_origin_at_pose_position() -> None:
    from deepreefmap.gui.viewer.render import _build_frustum_lines

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    lines = _build_frustum_lines(pose, fov_y=1.0, aspect=1.0)
    # First line segment starts at origin
    assert abs(lines[0, 0] - 10.0) < 0.01
    assert abs(lines[0, 1] - 20.0) < 0.01
    assert abs(lines[0, 2] - 30.0) < 0.01


def test_estimate_world_up_points_toward_cameras() -> None:
    from deepreefmap.gui.viewer.render import _estimate_world_up

    rng = np.random.default_rng(0)
    # Flat substrate in the XY plane; cameras hover above it along +Z.
    ground = np.column_stack(
        [rng.uniform(-5, 5, 400), rng.uniform(-5, 5, 400), rng.normal(0, 0.02, 400)]
    )
    cams_above = np.column_stack([rng.uniform(-5, 5, 30), rng.uniform(-5, 5, 30), np.full(30, 2.0)])
    up = _estimate_world_up(ground, cams_above)
    assert up[2] > 0.99  # ~ +Z, toward the cameras

    cams_below = cams_above.copy()
    cams_below[:, 2] = -2.0
    down = _estimate_world_up(ground, cams_below)
    assert down[2] < -0.99  # flips to -Z when cameras are on the other side

    assert _estimate_world_up(ground, None) == (0.0, 1.0, 0.0)  # fallback


def test_compute_transect_view_aligns_along_camera_path() -> None:
    from deepreefmap.gui.viewer.render import _compute_transect_view

    # Cameras drift along world +X from -5 to +5; points scatter around the line.
    cam_origins = np.stack([
        np.linspace(-5.0, 5.0, 11),
        np.full(11, 0.3),
        np.zeros(11),
    ], axis=1)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(500, 3))
    pts[:, 0] *= 5.0  # spread along X to match transect

    cam_pos, focal, up = _compute_transect_view(pts, cam_origins)
    cam_pos_a = np.asarray(cam_pos)
    focal_a = np.asarray(focal)
    up_a = np.asarray(up)

    assert up_a == pytest.approx(np.array([0.0, 1.0, 0.0]))
    forward = focal_a - cam_pos_a
    forward /= np.linalg.norm(forward)
    # Camera looks roughly along world Z (perpendicular to both X-transect and Y-up).
    assert abs(forward[0]) < 0.05
    assert abs(forward[1]) < 0.05
    assert abs(abs(forward[2]) - 1.0) < 0.05
    # Screen-right is cross(forward, up); end-minus-start must project positive.
    right = np.cross(forward, up_a)
    travel = cam_origins[-1] - cam_origins[0]
    assert float(travel @ right) > 0.0


def test_compute_transect_view_falls_back_for_degenerate_data() -> None:
    from deepreefmap.gui.viewer.render import _compute_transect_view

    # Single point, so PCA degenerates. The helper must still return finite numbers.
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    cam_pos, focal, up = _compute_transect_view(pts, None)
    assert focal == pytest.approx((1.0, 2.0, 3.0))
    assert up == pytest.approx((0.0, 1.0, 0.0))
    assert all(np.isfinite(cam_pos))


def test_select_pick_pixel_all_background_returns_none() -> None:
    from deepreefmap.gui.viewer.widget import QtPointCloudViewer

    z = np.ones((5, 5), dtype=np.float32)  # far plane everywhere == nothing drawn
    assert QtPointCloudViewer._select_pick_pixel(z, (2, 2)) is None


def test_select_pick_pixel_snaps_to_only_foreground_pixel() -> None:
    from deepreefmap.gui.viewer.widget import QtPointCloudViewer

    z = np.ones((5, 5), dtype=np.float32)
    z[1, 3] = 0.4  # one covered pixel at (col=3, row=1), cursor a few px away
    assert QtPointCloudViewer._select_pick_pixel(z, (2, 2)) == (3, 1)


def test_select_pick_pixel_prefers_pixel_under_cursor() -> None:
    from deepreefmap.gui.viewer.widget import QtPointCloudViewer

    z = np.full((5, 5), 0.5, dtype=np.float32)  # everything covered
    assert QtPointCloudViewer._select_pick_pixel(z, (2, 2)) == (2, 2)


def test_select_pick_pixel_breaks_distance_ties_by_depth() -> None:
    from deepreefmap.gui.viewer.widget import QtPointCloudViewer

    # Two covered pixels equidistant from the cursor; the front-most (smaller
    # depth) wins so picks land on the visible surface, not one behind it.
    z = np.ones((3, 3), dtype=np.float32)
    z[1, 0] = 0.8  # left, farther
    z[1, 2] = 0.3  # right, nearer
    assert QtPointCloudViewer._select_pick_pixel(z, (1, 1)) == (2, 1)
