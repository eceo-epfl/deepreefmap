"""Tests for the Qt launcher, viewer, model manager, and version checking.

Requires a real display because VTK (used by the 3D viewer) needs a working
OpenGL context. CI runs these under xvfb.
"""

from __future__ import annotations

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import numpy as np
import pytest


# --- Model manager ---

def test_model_list_has_all_expected_models():
    from deepreefmap.launcher.model_manager import ALL_MODELS

    names = {m.name for m in ALL_MODELS}
    assert "segformer-b2" in names
    assert "scsfmlearner" in names
    assert "coralscapes-vit-b-dpt" in names


def test_segformer_not_gated():
    from deepreefmap.launcher.model_manager import ALL_MODELS

    sf = next(m for m in ALL_MODELS if m.name == "segformer-b2")
    assert not sf.gated


def test_dinov3_gated():
    from deepreefmap.launcher.model_manager import ALL_MODELS

    dino = next(m for m in ALL_MODELS if m.name == "coralscapes-vit-b-dpt")
    assert dino.gated


def test_cache_detection_returns_false_for_nonexistent():
    from deepreefmap.launcher.model_manager import ModelInfo, is_model_cached

    fake = ModelInfo(
        name="fake",
        kind="test",
        hf_repos=["nonexistent-org/nonexistent-model-abc123"],
        gated=False,
        description="test",
    )
    assert not is_model_cached(fake)


# --- Version fetching ---

def test_fetch_versions_mock_env(monkeypatch):
    from deepreefmap.launcher.qt_app import _fetch_release_versions

    monkeypatch.setenv("DEEPREEFMAP_MOCK_VERSIONS", "2.0.0,1.5.0,1.0.1")
    versions = _fetch_release_versions()
    assert versions == ["2.0.0", "1.5.0", "1.0.1"]


def test_fetch_versions_mock_empty(monkeypatch):
    from deepreefmap.launcher.qt_app import _fetch_release_versions

    monkeypatch.setenv("DEEPREEFMAP_MOCK_VERSIONS", "")
    versions = _fetch_release_versions()
    assert versions == []


def test_fetch_versions_real_404(monkeypatch):
    from deepreefmap.launcher.qt_app import _fetch_release_versions

    monkeypatch.delenv("DEEPREEFMAP_MOCK_VERSIONS", raising=False)
    monkeypatch.setenv("DEEPREEFMAP_GH_REPO", "nonexistent-org-xyz/nonexistent-repo-abc")
    versions = _fetch_release_versions(timeout=5.0)
    assert versions is None


def test_fetch_versions_parses_github_response():
    """Spin up a local HTTP server returning fake GitHub releases JSON."""
    from deepreefmap.launcher.qt_app import _fetch_release_versions

    releases = [
        {"tag_name": "v2.0.0", "draft": False},
        {"tag_name": "v1.5.0", "draft": False},
        {"tag_name": "v1.0.0", "draft": True},
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(releases).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    import deepreefmap.launcher.qt_app_version as mod
    orig = mod._gh_releases_url
    mod._gh_releases_url = lambda: f"http://127.0.0.1:{port}/releases"
    try:
        versions = _fetch_release_versions(timeout=5.0)
    finally:
        mod._gh_releases_url = orig
        server.server_close()

    assert versions == ["2.0.0", "1.5.0"]


def test_fetch_releases_mock_synthesises_assets(monkeypatch):
    from deepreefmap.launcher.qt_app import _fetch_releases

    monkeypatch.setenv("DEEPREEFMAP_MOCK_VERSIONS", "2.0.0,1.0.1")
    releases = _fetch_releases()
    assert releases is not None
    assert [r["tag_name"] for r in releases] == ["v2.0.0", "v1.0.1"]
    names = {a["name"] for a in releases[0]["assets"]}
    assert "deepreefmap-linux-x64" in names
    assert "deepreefmap-windows-x64.exe" in names


def test_fetch_releases_keeps_assets_from_github_response():
    from deepreefmap.launcher.qt_app import _fetch_releases

    releases = [
        {
            "tag_name": "v2.0.0",
            "draft": False,
            "assets": [
                {
                    "name": "deepreefmap-linux-x64",
                    "browser_download_url": "https://example.invalid/v2.0.0/deepreefmap-linux-x64",
                },
            ],
        },
        {"tag_name": "v1.0.0", "draft": True, "assets": []},
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(releases).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    import deepreefmap.launcher.qt_app_version as mod
    orig = mod._gh_releases_url
    mod._gh_releases_url = lambda: f"http://127.0.0.1:{port}/releases"
    try:
        result = _fetch_releases(timeout=5.0)
    finally:
        mod._gh_releases_url = orig
        server.server_close()

    assert result is not None and len(result) == 1
    assert result[0]["tag_name"] == "v2.0.0"
    assert result[0]["assets"][0]["browser_download_url"].endswith("/deepreefmap-linux-x64")


# --- Binary swap helpers ---


def test_resolve_asset_name_linux():
    from deepreefmap.launcher.binary_swap import resolve_asset_name

    assert resolve_asset_name("linux") == "deepreefmap-linux-x64"


def test_resolve_asset_name_windows():
    from deepreefmap.launcher.binary_swap import resolve_asset_name

    assert resolve_asset_name("win32") == "deepreefmap-windows-x64.exe"


def test_resolve_asset_name_unsupported_raises():
    from deepreefmap.launcher.binary_swap import BinarySwapError, resolve_asset_name

    with pytest.raises(BinarySwapError):
        resolve_asset_name("darwin")


def test_find_asset_url_returns_match():
    from deepreefmap.launcher.binary_swap import find_asset_url

    rel = {
        "tag_name": "v1.0.0",
        "assets": [
            {"name": "deepreefmap-linux-x64", "browser_download_url": "https://x/y"},
            {"name": "other", "browser_download_url": "https://nope"},
        ],
    }
    assert find_asset_url(rel, "deepreefmap-linux-x64") == "https://x/y"


def test_find_asset_url_missing_raises():
    from deepreefmap.launcher.binary_swap import BinarySwapError, find_asset_url

    with pytest.raises(BinarySwapError):
        find_asset_url({"tag_name": "v0.5.0", "assets": []}, "deepreefmap-linux-x64")


def test_download_to_streams_chunks_and_reports_progress(tmp_path):
    from deepreefmap.launcher.binary_swap import download_to

    payload = b"x" * (200 * 1024)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    progress: list[tuple[int, int]] = []
    dest = tmp_path / "bin"
    try:
        download_to(
            f"http://127.0.0.1:{port}/bin", dest,
            lambda done, total: progress.append((done, total)),
            chunk_size=32 * 1024,
        )
    finally:
        server.server_close()

    assert dest.read_bytes() == payload
    assert progress and progress[-1] == (len(payload), len(payload))


def test_replace_binary_atomic_rename(tmp_path):
    from deepreefmap.launcher.binary_swap import replace_binary

    target = tmp_path / "current"
    target.write_bytes(b"old")
    src = tmp_path / "new"
    src.write_bytes(b"new")

    replace_binary(target, src)

    assert target.read_bytes() == b"new"
    assert not src.exists()


def test_current_version_returns_string():
    from deepreefmap.launcher.qt_app import _current_version

    v = _current_version()
    assert isinstance(v, str)
    assert v != ""


def test_pyapp_mock_path(monkeypatch):
    from deepreefmap.launcher.qt_app import _pyapp_binary_path

    monkeypatch.setenv("DEEPREEFMAP_MOCK_PYAPP", "1")
    monkeypatch.delenv("PYAPP", raising=False)
    assert _pyapp_binary_path() == "/tmp/mock-pyapp"


def test_pyapp_no_env(monkeypatch):
    from deepreefmap.launcher.qt_app import _pyapp_binary_path

    monkeypatch.delenv("DEEPREEFMAP_MOCK_PYAPP", raising=False)
    monkeypatch.delenv("PYAPP", raising=False)
    assert _pyapp_binary_path() is None


# --- Qt viewer protocol ---

def test_viewer_has_all_protocol_methods():
    from deepreefmap.visualization.qt_viewer import QtPointCloudViewer

    for method in (
        "start_run", "set_stage", "update_progress", "set_data",
        "mark_outputs_ready", "fail_run", "close", "wait_forever",
    ):
        assert hasattr(QtPointCloudViewer, method)


# --- Colorization helpers ---

def test_colorize_seg_maps_classes():
    from deepreefmap.visualization.qt_viewer import _colorize_seg

    labels = np.array([[1, 2], [2, 1]], dtype=np.int32)
    colors = {1: (255, 0, 0), 2: (0, 255, 0)}
    result = _colorize_seg(labels, colors)
    assert result[0, 0].tolist() == [255, 0, 0]
    assert result[0, 1].tolist() == [0, 255, 0]


def test_colorize_seg_fallback_gray():
    from deepreefmap.visualization.qt_viewer import _colorize_seg

    labels = np.array([[99]], dtype=np.int32)
    result = _colorize_seg(labels, {})
    assert result[0, 0].tolist() == [128, 128, 128]


def test_colorize_depth_handles_nan():
    from deepreefmap.visualization.qt_viewer import _colorize_depth

    depth = np.array([[float("nan"), 1.0], [2.0, float("nan")]], dtype=np.float32)
    result = _colorize_depth(depth)
    assert result.shape == (2, 2, 3)
    assert result[0, 0].sum() == 0
    assert result[0, 1].sum() > 0


def test_colorize_depth_all_nan():
    from deepreefmap.visualization.qt_viewer import _colorize_depth

    depth = np.full((3, 3), float("nan"), dtype=np.float32)
    result = _colorize_depth(depth)
    assert result.sum() == 0


def test_to_rgba_normalizes_uint8():
    from deepreefmap.visualization.qt_viewer import _to_rgba

    rgb = np.array([[255, 0, 128]], dtype=np.uint8)
    rgba = _to_rgba(rgb)
    assert rgba.shape == (1, 4)
    assert abs(rgba[0, 0] - 1.0) < 0.01
    assert abs(rgba[0, 3] - 1.0) < 0.01


def test_to_rgba_passthrough_float():
    from deepreefmap.visualization.qt_viewer import _to_rgba

    rgb = np.array([[0.5, 0.0, 1.0]], dtype=np.float32)
    rgba = _to_rgba(rgb)
    assert abs(rgba[0, 0] - 0.5) < 0.01


# --- Frustum geometry ---

def test_build_frustum_lines_shape():
    from deepreefmap.visualization.qt_viewer import _build_frustum_lines

    pose = np.eye(4, dtype=np.float64)
    lines = _build_frustum_lines(pose, fov_y=1.0, aspect=1.5)
    # 4 edges from origin + 4 edges around rectangle = 8 line segments = 16 points
    assert lines.shape == (16, 3)


def test_build_frustum_lines_origin_at_pose_position():
    from deepreefmap.visualization.qt_viewer import _build_frustum_lines

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    lines = _build_frustum_lines(pose, fov_y=1.0, aspect=1.0)
    # First line segment starts at origin
    assert abs(lines[0, 0] - 10.0) < 0.01
    assert abs(lines[0, 1] - 20.0) < 0.01
    assert abs(lines[0, 2] - 30.0) < 0.01


# --- Qt widget tests (offscreen) ---

@pytest.fixture(scope="module")
def qapp():
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_viewer_widget_creates(qapp):
    from deepreefmap.visualization.qt_viewer import QtPointCloudViewer

    viewer = QtPointCloudViewer(class_colors={1: (255, 0, 0)}, class_names={1: "test"})
    assert viewer.n_frames == 0
    assert not viewer.has_scene_data


def test_viewer_show_point_cloud(qapp):
    from deepreefmap.visualization.qt_viewer import QtPointCloudViewer

    viewer = QtPointCloudViewer()
    xyz = np.random.rand(100, 3).astype(np.float32)
    rgb = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
    viewer.show_point_cloud(xyz, rgb)


def test_viewer_empty_cloud_noop(qapp):
    from deepreefmap.visualization.qt_viewer import QtPointCloudViewer

    viewer = QtPointCloudViewer()
    viewer.show_point_cloud(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))


def test_window_creates(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes, DEFAULT_CLASSES_PATH
    from deepreefmap.launcher.qt_app import DeepReefMapWindow

    cc = load_classes(DEFAULT_CLASSES_PATH)
    window = DeepReefMapWindow(cc, DEFAULT_CLASSES_PATH)
    assert window.windowTitle() == "DeepReefMap"


# --- Batch CSV parser ---

def test_parse_timestamp_range_full():
    from deepreefmap.launcher.qt_app import _parse_timestamp_range

    assert _parse_timestamp_range("12-45.5") == (12.0, 45.5)


def test_parse_timestamp_range_open_end():
    from deepreefmap.launcher.qt_app import _parse_timestamp_range

    assert _parse_timestamp_range("30-") == (30.0, None)


def test_parse_timestamp_range_open_begin():
    from deepreefmap.launcher.qt_app import _parse_timestamp_range

    assert _parse_timestamp_range("-60") == (None, 60.0)


def test_parse_timestamp_range_empty():
    from deepreefmap.launcher.qt_app import _parse_timestamp_range

    assert _parse_timestamp_range("") == (None, None)


def test_parse_timestamp_range_single_value():
    from deepreefmap.launcher.qt_app import _parse_timestamp_range

    assert _parse_timestamp_range("15") == (15.0, None)


def test_load_batch_csv_parses_rows(tmp_path):
    from deepreefmap.launcher.qt_app import _load_batch_csv

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "videos,timestamps,transect_length,crop_width\n"
        "a.mp4,5-30,10,2\n"
        "b.mp4,-60,,1.5\n"
    )
    jobs = _load_batch_csv(csv_path)
    assert len(jobs) == 2
    assert jobs[0].video == "a.mp4"
    assert jobs[0].begin_s == 5.0
    assert jobs[0].end_s == 30.0
    assert jobs[0].transect_length == 10.0
    assert jobs[0].crop_width == 2.0
    assert jobs[0].name == "a"
    assert jobs[1].begin_s is None
    assert jobs[1].end_s == 60.0
    assert jobs[1].transect_length is None


def test_load_batch_csv_case_insensitive_columns(tmp_path):
    from deepreefmap.launcher.qt_app import _load_batch_csv

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "Videos,Timestamps,Transect_Length,Crop_Width\n"
        "x.mp4,0-10,5,1\n"
    )
    jobs = _load_batch_csv(csv_path)
    assert len(jobs) == 1


def test_load_batch_csv_rejects_missing_columns(tmp_path):
    from deepreefmap.launcher.qt_app import _load_batch_csv

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("videos,timestamps\nx.mp4,0-10\n")
    with pytest.raises(ValueError, match="missing required columns"):
        _load_batch_csv(csv_path)


def test_load_batch_csv_skips_blank_rows(tmp_path):
    from deepreefmap.launcher.qt_app import _load_batch_csv

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "videos,timestamps,transect_length,crop_width\n"
        ",,,,\n"
        "x.mp4,0-10,5,1\n"
    )
    jobs = _load_batch_csv(csv_path)
    assert len(jobs) == 1
    assert jobs[0].video == "x.mp4"


def test_load_batch_csv_rejects_excel(tmp_path):
    from deepreefmap.launcher.qt_app import _load_batch_csv

    bogus = tmp_path / "jobs.xlsx"
    bogus.write_bytes(b"not actually excel")
    with pytest.raises(ValueError, match="Excel"):
        _load_batch_csv(bogus)
