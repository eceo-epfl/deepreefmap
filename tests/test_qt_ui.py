"""Tests for the Qt launcher, viewer, model manager, and version checking.

Runs headless via QT_QPA_PLATFORM=offscreen (set in conftest.py when no
DISPLAY/WAYLAND_DISPLAY is present). VTK still needs a working OpenGL
context, so VTK-only tests are skipped when that can't be obtained.
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
    from deepreefmap.gui.model_manager import ALL_MODELS

    names = {m.name for m in ALL_MODELS}
    assert "segformer-b2" in names
    assert "scsfmlearner" in names
    assert "coralscapes-vit-b-dpt" in names


@pytest.mark.parametrize(
    "name, gated",
    [
        ("segformer-b2", False),
        ("coralscapes-vit-b-dpt", True),
    ],
)
def test_model_gated_flag(name, gated):
    from deepreefmap.gui.model_manager import ALL_MODELS

    info = next(m for m in ALL_MODELS if m.name == name)
    assert info.gated is gated


def test_cache_detection_returns_false_for_nonexistent():
    from deepreefmap.gui.model_manager import ModelInfo, is_model_cached

    fake = ModelInfo(
        name="fake",
        kind="test",
        hf_repos=["nonexistent-org/nonexistent-model-abc123"],
        gated=False,
        description="test",
    )
    assert not is_model_cached(fake)


def test_dinov3_dpt_entries_include_facebook_backbone():
    from deepreefmap.gui.model_manager import ALL_MODELS

    expected = {
        "coralscapes-vit-s-dpt": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "coralscapes-vit-b-dpt": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "coralscapes-vit-l-dpt": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }
    for name, backbone in expected.items():
        info = next(m for m in ALL_MODELS if m.name == name)
        assert backbone in info.hf_repos, (
            f"{name} must list {backbone} so offline laptops also cache the "
            "DINOv3 backbone that coralscapes_hub_model.py pulls in at load time"
        )


def test_loger_entries_materialise_into_ckpts_dir():
    from deepreefmap.gui.model_manager import MAPPING_MODELS
    from deepreefmap.mapping.registry import _LOGER_CKPTS

    by_name = {m.name: m for m in MAPPING_MODELS}
    assert "loger" in by_name and "loger_star" in by_name

    loger = by_name["loger"]
    assert loger.hf_repos == ["Junyi42/LoGeR"]
    assert loger.materialise_to[
        "LoGeR/latest.pt"
    ] == _LOGER_CKPTS / "LoGeR" / "latest.pt"

    star = by_name["loger_star"]
    assert star.materialise_to[
        "LoGeR_star/latest.pt"
    ] == _LOGER_CKPTS / "LoGeR_star" / "latest.pt"


def test_is_model_cached_requires_materialised_destinations(tmp_path, monkeypatch):
    from deepreefmap.gui import model_manager
    from deepreefmap.gui.model_manager import ModelInfo, is_model_cached

    fake_cache = tmp_path / "hf"
    repo_dir = fake_cache / "models--fake--repo"
    repo_dir.mkdir(parents=True)
    monkeypatch.setattr(model_manager, "_HF_CACHE_ROOT", fake_cache)

    dest = tmp_path / "ckpts" / "weight.pt"
    info = ModelInfo(
        name="materialise-fake",
        kind="test",
        hf_repos=["fake/repo"],
        gated=False,
        description="test",
        materialise_to={"weight.pt": dest},
    )
    assert not is_model_cached(info), "missing materialised file must read as not cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"x")
    assert is_model_cached(info)


def test_prefetch_refuses_when_disk_is_low(tmp_path, monkeypatch):
    from deepreefmap.gui import model_manager
    from deepreefmap.gui.model_manager import (
        InsufficientDiskSpace,
        ModelInfo,
        prefetch_model,
    )

    monkeypatch.setattr(model_manager, "_HF_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(
        model_manager.shutil,
        "disk_usage",
        lambda _p: type("U", (), {"total": 1, "used": 1, "free": 1})(),
    )
    info = ModelInfo(
        name="any",
        kind="test",
        hf_repos=["fake/repo"],
        gated=False,
        description="test",
    )
    with pytest.raises(InsufficientDiskSpace):
        prefetch_model(info)


# --- Version fetching ---

def test_fetch_versions_mock_env(monkeypatch):
    from deepreefmap.gui.app import _fetch_release_versions

    monkeypatch.setenv("DEEPREEFMAP_MOCK_VERSIONS", "2.0.0,1.5.0,1.0.1")
    versions = _fetch_release_versions()
    assert versions == ["2.0.0", "1.5.0", "1.0.1"]


def test_fetch_versions_mock_empty(monkeypatch):
    from deepreefmap.gui.app import _fetch_release_versions

    monkeypatch.setenv("DEEPREEFMAP_MOCK_VERSIONS", "")
    versions = _fetch_release_versions()
    assert versions == []


@pytest.mark.parametrize(
    "releases, current, expected",
    [
        ([{"tag_name": "v1.0.0"}, {"tag_name": "v0.9.0"}], "1.0.1", []),
        ([{"tag_name": "v1.0.0"}, {"tag_name": "v0.9.0"}], "1.0.0", []),
        (
            [{"tag_name": "v1.5.0"}, {"tag_name": "v2.0.0"}, {"tag_name": "v1.0.0"}],
            "1.0.0",
            ["v2.0.0", "v1.5.0"],
        ),
    ],
)
def test_newer_releases_orders_and_filters(releases, current, expected):
    from deepreefmap.gui.app import _newer_releases

    newer = _newer_releases(releases, current)
    assert [r["tag_name"] for r in newer] == expected


def test_newer_releases_unparseable_current_falls_back_to_inequality():
    from deepreefmap.gui.app import _newer_releases

    releases = [{"tag_name": "v1.0.0"}, {"tag_name": "v1.0.1"}]
    newer = _newer_releases(releases, "dev")
    assert {r["tag_name"] for r in newer} == {"v1.0.0", "v1.0.1"}


def test_fetch_versions_real_404(monkeypatch):
    from deepreefmap.gui.app import _fetch_release_versions

    monkeypatch.delenv("DEEPREEFMAP_MOCK_VERSIONS", raising=False)
    monkeypatch.setenv("DEEPREEFMAP_GH_REPO", "nonexistent-org-xyz/nonexistent-repo-abc")
    versions = _fetch_release_versions(timeout=5.0)
    assert versions is None


def test_fetch_versions_parses_github_response():
    """Spin up a local HTTP server returning fake GitHub releases JSON."""
    from deepreefmap.gui.app import _fetch_release_versions

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

    import deepreefmap.gui.version as mod
    orig = mod._gh_releases_url
    mod._gh_releases_url = lambda: f"http://127.0.0.1:{port}/releases"
    try:
        versions = _fetch_release_versions(timeout=5.0)
    finally:
        mod._gh_releases_url = orig
        server.server_close()

    assert versions == ["2.0.0", "1.5.0"]


@pytest.mark.parametrize(
    "include_older, expected",
    [
        (False, ["2.0.0"]),  # upgrades only
        (True, ["2.0.0", "1.0.0"]),  # all but current, newest first → rollback offered
    ],
)
def test_selectable_releases(include_older, expected):
    from deepreefmap.gui.version import _selectable_releases

    releases = [
        {"tag_name": "v1.0.0"},
        {"tag_name": "v2.0.0"},
        {"tag_name": "v1.5.0"},  # current
    ]
    got = _selectable_releases(releases, "1.5.0", include_older)
    assert [r["tag_name"].lstrip("v") for r in got] == expected


def test_gh_api_url_override(monkeypatch):
    from deepreefmap.gui.version import _gh_releases_url

    monkeypatch.setenv("DEEPREEFMAP_GH_API_URL", "http://127.0.0.1:9999/releases")
    monkeypatch.setenv("DEEPREEFMAP_GH_REPO", "owner/repo")  # override wins
    assert _gh_releases_url() == "http://127.0.0.1:9999/releases"

    monkeypatch.delenv("DEEPREEFMAP_GH_API_URL")
    assert _gh_releases_url() == "https://api.github.com/repos/owner/repo/releases"


def test_fetch_releases_mock_synthesises_assets(monkeypatch):
    from deepreefmap.gui.app import _fetch_releases

    monkeypatch.setenv("DEEPREEFMAP_MOCK_VERSIONS", "2.0.0,1.0.1")
    releases = _fetch_releases()
    assert releases is not None
    assert [r["tag_name"] for r in releases] == ["v2.0.0", "v1.0.1"]
    names = {a["name"] for a in releases[0]["assets"]}
    assert "deepreefmap-linux-x64" in names
    assert "deepreefmap-windows-x64.exe" in names


def test_fetch_releases_keeps_assets_from_github_response():
    from deepreefmap.gui.app import _fetch_releases

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

    import deepreefmap.gui.version as mod
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


@pytest.mark.parametrize(
    "platform, expected",
    [
        ("linux", "deepreefmap-linux-x64"),
        ("win32", "deepreefmap-windows-x64.exe"),
        ("darwin", "deepreefmap-macos-arm64"),
    ],
)
def test_resolve_asset_name(platform, expected, monkeypatch):
    from deepreefmap.gui import binary_swap

    # Pin the standard (CUDA/CPU) build so the linux base name is deterministic
    # regardless of the host's torch wheel.
    monkeypatch.setattr(binary_swap, "_is_rocm_build", lambda: False)
    assert binary_swap.resolve_asset_name(platform) == expected


def test_resolve_asset_name_unsupported_raises():
    from deepreefmap.gui.binary_swap import BinarySwapError, resolve_asset_name

    with pytest.raises(BinarySwapError):
        resolve_asset_name("freebsd")


def test_resolve_asset_name_rocm_linux(monkeypatch):
    from deepreefmap.gui import binary_swap

    monkeypatch.setattr(binary_swap, "_is_rocm_build", lambda: True)
    assert binary_swap.resolve_asset_name("linux") == "deepreefmap-linux-x64-rocm"


def test_is_rocm_build_from_pyapp_binary_name(monkeypatch):
    from deepreefmap.gui import binary_swap

    monkeypatch.setenv("PYAPP", "/opt/pyapp/deepreefmap-linux-x64-rocm")
    assert binary_swap._is_rocm_build() is True


def test_is_rocm_build_reads_torch_hip_version(monkeypatch):
    import torch

    from deepreefmap.gui import binary_swap

    monkeypatch.delenv("PYAPP", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert binary_swap._is_rocm_build() is False
    monkeypatch.setattr(torch.version, "hip", "6.3.42", raising=False)
    assert binary_swap._is_rocm_build() is True


def test_apply_theme_sets_dark_palette(qapp):
    # apply_theme mutates the shared app, so snapshot and restore to keep other
    # tests isolated. We assert on the palette (not style().objectName(), which
    # the global stylesheet wraps in an empty-named QStyleSheetStyle proxy).
    from PySide6.QtGui import QPalette

    from deepreefmap.gui.theme import apply_theme

    prev_style = qapp.style().objectName()
    prev_palette = QPalette(qapp.palette())
    prev_qss = qapp.styleSheet()
    try:
        apply_theme(qapp)
        win = qapp.palette().color(QPalette.ColorRole.Window)
        base = qapp.palette().color(QPalette.ColorRole.Base)
        assert win.red() < 80 and win.green() < 80 and win.blue() < 80
        assert base.lightness() < win.lightness()
    finally:
        qapp.setStyleSheet(prev_qss)
        qapp.setPalette(prev_palette)
        if prev_style:
            qapp.setStyle(prev_style)


def test_theme_semantic_constants_are_valid_hex():
    from PySide6.QtGui import QColor

    from deepreefmap.gui import theme

    for name in ("SUCCESS", "WARNING", "ERROR", "PRIMARY", "LINK", "UPDATE", "DANGER_BG"):
        assert QColor(getattr(theme, name)).isValid()


def test_find_asset_url_returns_match():
    from deepreefmap.gui.binary_swap import BinarySwapError, find_asset_url

    rel = {
        "tag_name": "v1.0.0",
        "assets": [
            {"name": "deepreefmap-linux-x64", "browser_download_url": "https://x/y"},
            {"name": "other", "browser_download_url": "https://nope"},
        ],
    }
    assert find_asset_url(rel, "deepreefmap-linux-x64") == "https://x/y"
    with pytest.raises(BinarySwapError):
        find_asset_url({"tag_name": "v0.5.0", "assets": []}, "deepreefmap-linux-x64")


def test_download_to_streams_chunks_and_reports_progress(tmp_path):
    from deepreefmap.gui.binary_swap import download_to

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
    from deepreefmap.gui.binary_swap import replace_binary

    target = tmp_path / "current"
    target.write_bytes(b"old")
    src = tmp_path / "new"
    src.write_bytes(b"new")

    replace_binary(target, src)

    assert target.read_bytes() == b"new"
    assert not src.exists()


def test_env_is_healthy_detects_missing_and_intact(tmp_path):
    from deepreefmap.gui.binary_swap import env_is_healthy

    # Missing torch/PySide6 → unhealthy.
    assert env_is_healthy(tmp_path) is False

    purelib = tmp_path / "site-packages"
    (purelib / "torch" / "lib").mkdir(parents=True)
    (purelib / "torch" / "lib" / "libtorch.so").write_bytes(b"\x00")
    (purelib / "PySide6").mkdir()
    assert env_is_healthy(purelib) is True

    # An emptied torch/lib (antivirus quarantine) → unhealthy.
    (purelib / "torch" / "lib" / "libtorch.so").unlink()
    assert env_is_healthy(purelib) is False


def test_prune_previous_env_removes_old_keeps_current(tmp_path, monkeypatch):
    from deepreefmap.gui import binary_swap

    pyapp_root = tmp_path / "pyapp" / "deepreefmap" / "hash"
    old_env = pyapp_root / "1.0.0"
    new_env = pyapp_root / "1.1.0"
    (old_env / "python").mkdir(parents=True)
    (new_env / "python").mkdir(parents=True)

    marker = tmp_path / "pending_env_prune.json"
    monkeypatch.setattr(
        "deepreefmap.paths.env_prune_marker_path", lambda: marker
    )

    # Recorded while the old version was running (prefix = <env>/python).
    binary_swap.record_previous_env(old_env / "python")
    assert marker.exists()

    # Now running the new version → old env pruned, new env untouched.
    removed = binary_swap.prune_previous_env(current_prefix=new_env / "python")

    assert removed == old_env
    assert not old_env.exists()
    assert new_env.exists()
    assert not marker.exists()


def test_prune_previous_env_no_marker_is_noop(tmp_path, monkeypatch):
    from deepreefmap.gui import binary_swap

    monkeypatch.setattr(
        "deepreefmap.paths.env_prune_marker_path",
        lambda: tmp_path / "absent.json",
    )
    assert binary_swap.prune_previous_env(current_prefix=tmp_path) is None


def test_prune_refuses_paths_outside_pyapp(tmp_path, monkeypatch):
    from deepreefmap.gui import binary_swap

    marker = tmp_path / "marker.json"
    monkeypatch.setattr("deepreefmap.paths.env_prune_marker_path", lambda: marker)
    victim = tmp_path / "not-pyapp" / "1.0.0"
    (victim / "python").mkdir(parents=True)
    marker.write_text(json.dumps({"env_dir": str(victim), "version": "1.0.0"}))

    removed = binary_swap.prune_previous_env(current_prefix=tmp_path / "cur")

    assert removed is None
    assert victim.exists()  # guard kept us from deleting a non-pyapp dir
    assert not marker.exists()


def test_perform_update_downloads_swaps_and_records_prune(tmp_path, monkeypatch):
    from deepreefmap.gui import binary_swap

    payload = b"NEW-BINARY-BYTES"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.handle_request, daemon=True).start()

    asset = binary_swap.resolve_asset_name()
    release = {
        "tag_name": "v1.1.0",
        "assets": [{"name": asset, "browser_download_url": f"http://127.0.0.1:{port}/bin"}],
    }
    target = tmp_path / "deepreefmap"
    target.write_bytes(b"OLD-BINARY")

    marker = tmp_path / "pending_env_prune.json"
    monkeypatch.setattr("deepreefmap.paths.env_prune_marker_path", lambda: marker)

    lines: list[str] = []
    try:
        binary_swap.perform_update(
            release, target, "1.1.0", line_cb=lines.append
        )
    finally:
        server.server_close()

    assert target.read_bytes() == payload
    assert not target.with_name(target.name + ".new").exists()
    assert marker.exists()  # next launch will consume this to prune the old env
    assert any("Replacing binary" in line for line in lines)


def test_update_then_prune_end_to_end(tmp_path, monkeypatch):
    """The container e2e as a fast headless test: real download + swap + marker,
    then prune of the previous env with the shared uv cache left intact."""
    from deepreefmap.gui import binary_swap

    payload = b"NEW-BINARY-BYTES"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.handle_request, daemon=True).start()

    # PyApp layout: .../pyapp/deepreefmap/<hash>/<version>/python
    pyapp_root = tmp_path / "pyapp" / "deepreefmap" / "hash"
    old_env = pyapp_root / "1.1.0"
    new_env = pyapp_root / "1.2.0"
    (old_env / "python").mkdir(parents=True)
    (new_env / "python").mkdir(parents=True)
    uv_cache = tmp_path / ".cache" / "uv"  # shared cache must survive the prune
    uv_cache.mkdir(parents=True)

    marker = tmp_path / "pending_env_prune.json"
    monkeypatch.setattr("deepreefmap.paths.env_prune_marker_path", lambda: marker)
    # While the old version runs, sys.prefix points into its env.
    monkeypatch.setattr(binary_swap.sys, "prefix", str(old_env / "python"))

    asset = binary_swap.resolve_asset_name()
    release = {
        "tag_name": "v1.2.0",
        "assets": [{"name": asset, "browser_download_url": f"http://127.0.0.1:{port}/bin"}],
    }
    target = tmp_path / "deepreefmap"
    target.write_bytes(b"OLD-BINARY")

    try:
        binary_swap.perform_update(release, target, "1.2.0")
    finally:
        server.server_close()

    assert target.read_bytes() == payload  # swapped in place
    assert marker.exists()  # old env recorded for pruning

    removed = binary_swap.prune_previous_env(current_prefix=str(new_env / "python"))

    assert removed == old_env
    assert not old_env.exists()
    assert new_env.exists()
    assert uv_cache.exists()
    assert not marker.exists()


def test_update_dialog_runs_perform_update(qapp, tmp_path, monkeypatch):
    """Guarantee the Install button's worker is wired to perform_update()."""
    from deepreefmap.gui import update_dialog

    calls = {}

    def fake_perform_update(release, binary_path, target_version, progress_cb=None, line_cb=None):
        calls["args"] = (release, binary_path, target_version)
        if line_cb is not None:
            line_cb("working")

    monkeypatch.setattr(update_dialog, "perform_update", fake_perform_update)
    monkeypatch.delenv("DEEPREEFMAP_MOCK_PYAPP", raising=False)

    binary = tmp_path / "binary"
    binary.write_bytes(b"x")
    dialog = update_dialog.UpdateProgressDialog(
        target_version="1.2.0",
        release={"tag_name": "v1.2.0", "assets": []},
        binary_path=binary,
    )
    done = []
    dialog._sig_done.connect(lambda ok, msg: done.append((ok, msg)))

    dialog._run_real()  # call the worker body directly (no thread, no exec)

    assert calls["args"][1] == binary
    assert calls["args"][2] == "1.2.0"
    assert done and done[0][0] is True


def test_self_restore_invokes_pyapp_self_restore(monkeypatch):
    from deepreefmap.gui import binary_swap

    calls = []
    monkeypatch.setattr(
        binary_swap.subprocess, "run", lambda cmd, check: calls.append((cmd, check))
    )
    assert binary_swap.self_restore("/path/bin") is True
    assert calls == [(["/path/bin", "self", "restore"], True)]

    def boom(cmd, check):
        raise OSError("restore failed")

    monkeypatch.setattr(binary_swap.subprocess, "run", boom)
    assert binary_swap.self_restore("/path/bin") is False


def test_bootstrap_self_heals_then_reexecs_when_env_broken(monkeypatch, tmp_path):
    import deepreefmap.bootstrap as bootstrap
    from deepreefmap.gui import binary_swap

    monkeypatch.setenv("PYAPP", str(tmp_path / "deepreefmap"))
    monkeypatch.delenv("DEEPREEFMAP_SELF_HEAL_ATTEMPTED", raising=False)
    monkeypatch.setattr(binary_swap, "env_is_healthy", lambda *a, **k: False)
    restored: list[str] = []
    monkeypatch.setattr(
        binary_swap, "self_restore", lambda b: bool(restored.append(b)) or True
    )

    class _Reexec(Exception):
        pass

    execs: list[tuple] = []

    def fake_execv(path, args):
        execs.append((path, args))
        raise _Reexec

    monkeypatch.setattr(bootstrap.os, "execv", fake_execv)

    try:
        with pytest.raises(_Reexec):
            bootstrap.main()
        assert restored, "self_restore should have been invoked"
        assert execs, "binary should be re-exec'd after restore"
        assert os.environ.get("DEEPREEFMAP_SELF_HEAL_ATTEMPTED") == "1"
    finally:
        os.environ.pop("DEEPREEFMAP_SELF_HEAL_ATTEMPTED", None)


def test_pyapp_mock_path(monkeypatch):
    from deepreefmap.gui.app import _pyapp_binary_path

    monkeypatch.setenv("DEEPREEFMAP_MOCK_PYAPP", "1")
    monkeypatch.delenv("PYAPP", raising=False)
    assert _pyapp_binary_path() == "/tmp/mock-pyapp"


def test_pyapp_no_env(monkeypatch):
    from deepreefmap.gui.app import _pyapp_binary_path

    monkeypatch.delenv("DEEPREEFMAP_MOCK_PYAPP", raising=False)
    monkeypatch.delenv("PYAPP", raising=False)
    assert _pyapp_binary_path() is None


# --- Colorization helpers ---

def test_colorize_seg_maps_classes():
    from deepreefmap.gui.viewer import _colorize_seg

    labels = np.array([[1, 2], [2, 1]], dtype=np.int32)
    colors = {1: (255, 0, 0), 2: (0, 255, 0)}
    result = _colorize_seg(labels, colors)
    assert result[0, 0].tolist() == [255, 0, 0]
    assert result[0, 1].tolist() == [0, 255, 0]


def test_colorize_seg_fallback_gray():
    from deepreefmap.gui.viewer import _colorize_seg

    labels = np.array([[99]], dtype=np.int32)
    result = _colorize_seg(labels, {})
    assert result[0, 0].tolist() == [128, 128, 128]


def test_colorize_depth_handles_nan():
    from deepreefmap.gui.viewer import _colorize_depth

    depth = np.array([[float("nan"), 1.0], [2.0, float("nan")]], dtype=np.float32)
    result = _colorize_depth(depth)
    assert result.shape == (2, 2, 3)
    assert result[0, 0].sum() == 0
    assert result[0, 1].sum() > 0


def test_colorize_depth_all_nan():
    from deepreefmap.gui.viewer import _colorize_depth

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
def test_to_rgba_normalizes_by_dtype(rgb, expected_r):
    from deepreefmap.gui.viewer import _to_rgba

    rgba = _to_rgba(rgb)
    assert rgba.shape == (1, 4)
    assert abs(rgba[0, 0] - expected_r) < 0.01
    assert abs(rgba[0, 3] - 1.0) < 0.01


# --- Frustum geometry ---

def test_build_frustum_lines_shape():
    from deepreefmap.gui.viewer import _build_frustum_lines

    pose = np.eye(4, dtype=np.float64)
    lines = _build_frustum_lines(pose, fov_y=1.0, aspect=1.5)
    # 4 edges from origin + 4 edges around rectangle = 8 line segments = 16 points
    assert lines.shape == (16, 3)


def test_build_frustum_lines_origin_at_pose_position():
    from deepreefmap.gui.viewer import _build_frustum_lines

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    lines = _build_frustum_lines(pose, fov_y=1.0, aspect=1.0)
    # First line segment starts at origin
    assert abs(lines[0, 0] - 10.0) < 0.01
    assert abs(lines[0, 1] - 20.0) < 0.01
    assert abs(lines[0, 2] - 30.0) < 0.01


def test_estimate_world_up_points_toward_cameras():
    from deepreefmap.gui.viewer import _estimate_world_up

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


def test_compute_transect_view_aligns_along_camera_path():
    from deepreefmap.gui.viewer import _compute_transect_view

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


def test_compute_transect_view_falls_back_for_degenerate_data():
    from deepreefmap.gui.viewer import _compute_transect_view

    # Single point — PCA degenerates; helper must still return finite numbers.
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    cam_pos, focal, up = _compute_transect_view(pts, None)
    assert focal == pytest.approx((1.0, 2.0, 3.0))
    assert up == pytest.approx((0.0, 1.0, 0.0))
    assert all(np.isfinite(cam_pos))


# --- Depth-buffer pick pixel selection ---

def test_select_pick_pixel_all_background_returns_none():
    from deepreefmap.gui.viewer import QtPointCloudViewer

    z = np.ones((5, 5), dtype=np.float32)  # far plane everywhere == nothing drawn
    assert QtPointCloudViewer._select_pick_pixel(z, (2, 2)) is None


def test_select_pick_pixel_snaps_to_only_foreground_pixel():
    from deepreefmap.gui.viewer import QtPointCloudViewer

    z = np.ones((5, 5), dtype=np.float32)
    z[1, 3] = 0.4  # one covered pixel at (col=3, row=1), cursor a few px away
    assert QtPointCloudViewer._select_pick_pixel(z, (2, 2)) == (3, 1)


def test_select_pick_pixel_prefers_pixel_under_cursor():
    from deepreefmap.gui.viewer import QtPointCloudViewer

    z = np.full((5, 5), 0.5, dtype=np.float32)  # everything covered
    assert QtPointCloudViewer._select_pick_pixel(z, (2, 2)) == (2, 2)


def test_select_pick_pixel_breaks_distance_ties_by_depth():
    from deepreefmap.gui.viewer import QtPointCloudViewer

    # Two covered pixels equidistant from the cursor; the front-most (smaller
    # depth) wins so picks land on the visible surface, not one behind it.
    z = np.ones((3, 3), dtype=np.float32)
    z[1, 0] = 0.8  # left, farther
    z[1, 2] = 0.3  # right, nearer
    assert QtPointCloudViewer._select_pick_pixel(z, (1, 1)) == (2, 1)


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
    from deepreefmap.gui.viewer import QtPointCloudViewer

    viewer = QtPointCloudViewer(class_colors={1: (255, 0, 0)}, class_names={1: "test"})
    assert viewer.n_frames == 0
    assert not viewer.has_scene_data


def test_viewer_show_point_cloud(qapp):
    from deepreefmap.gui.viewer import QtPointCloudViewer

    viewer = QtPointCloudViewer()
    xyz = np.random.rand(100, 3).astype(np.float32)
    rgb = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
    viewer.show_point_cloud(xyz, rgb)


def test_viewer_empty_cloud_noop(qapp):
    from deepreefmap.gui.viewer import QtPointCloudViewer

    viewer = QtPointCloudViewer()
    viewer.show_point_cloud(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))


def test_window_creates(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)
    assert window.windowTitle() == "DeepReefMap"


def test_updates_tab_dev_mode_vs_installed(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    window = DeepReefMapWindow(load_classes(), None)
    releases = [{"tag_name": "v2.0.0", "assets": []}, {"tag_name": "v1.0.0", "assets": []}]

    # Dev mode (no installer binary): explain it, hide the install controls.
    # isHidden() reflects the widget's own flag (the window is never shown, so
    # isVisible() would be False regardless).
    window._apply_update_check("1.1.0", releases, None)
    assert "development mode" in window._update_status_label.text().lower()
    assert window._update_version_combo.isHidden()
    assert window._update_show_all.isHidden()

    # Installed binary: install controls + rollback checkbox appear.
    window._apply_update_check("1.1.0", releases, "/tmp/mock-pyapp")
    assert not window._update_version_combo.isHidden()
    assert not window._update_show_all.isHidden()


def test_overlay_has_reset_button_and_r_shortcut_triggers_view_reset(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from PySide6.QtGui import QKeySequence, QShortcut

    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)
    assert window._reset_view_button is not None
    assert "Reset" in window._reset_view_button.text()

    calls: list[int] = []
    window._viewer.reset_view = lambda: calls.append(1)  # type: ignore[method-assign]

    window._reset_view_button.click()
    assert calls == [1]

    # QShortcut registered for R must exist and, when activated, fire the
    # same reset_view path. We trigger it via `activated.emit()` rather than
    # synthesising a key event — offscreen windows aren't "active", which
    # would suppress the natural shortcut dispatch.
    r_seq = QKeySequence("R")
    r_shortcuts = [
        s for s in window.findChildren(QShortcut)
        if s.key() == r_seq
    ]
    assert r_shortcuts, "expected an R shortcut on the window"
    r_shortcuts[0].activated.emit()
    assert len(calls) == 2


def test_legend_overlay_reorder_places_rows(qapp):
    from PySide6.QtWidgets import QWidget

    from deepreefmap.gui.viewer import LegendOverlay

    parent = QWidget()
    ov = LegendOverlay(parent)
    names = {1: "alpha", 2: "beta", 3: "gamma"}
    colors = {1: (1, 1, 1), 2: (2, 2, 2), 3: (3, 3, 3)}
    ov.rebuild([1, 2, 3], names, colors, on_toggle=lambda: None, class_counts={1: 10, 2: 20, 3: 30})
    ov.reorder([3, 1, 2])

    def row_of(cid: int) -> int:
        cb = ov._rows[cid][1]
        return ov._grid.getItemPosition(ov._grid.indexOf(cb))[0]

    assert row_of(3) < row_of(1) < row_of(2)


def test_legend_sort_selected_first_puts_checked_above_unchecked(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)
    window._build_legend()
    cids = list(window._legend_toggles.keys())
    assert len(cids) >= 3
    window._legend_toggles[cids[0]].setChecked(False)
    window._legend_toggles[cids[1]].setChecked(False)

    order = window._legend_sort_order()
    enabled = window._enabled_class_set()
    last_visible = max(i for i, c in enumerate(order) if c in enabled)
    first_hidden = min(i for i, c in enumerate(order) if c not in enabled)
    assert last_visible < first_hidden


def test_legend_sort_header_click_toggles_direction(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)
    window._build_legend()
    assert (window._legend_sort_mode, window._legend_sort_ascending) == ("selected", False)

    window._on_legend_sort_clicked("name")  # new column adopts its default (A–Z)
    assert (window._legend_sort_mode, window._legend_sort_ascending) == ("name", True)
    window._on_legend_sort_clicked("name")  # same column flips direction
    assert window._legend_sort_ascending is False
    window._on_legend_sort_clicked("size")  # new column adopts default (largest first)
    assert (window._legend_sort_mode, window._legend_sort_ascending) == ("size", False)


def test_pie_click_toggles_selection(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)
    window._build_legend()
    cids = list(window._legend_toggles.keys())
    assert len(cids) >= 2

    window._on_deselect_all_classes()
    assert window._enabled_class_set() == frozenset()
    window._on_sunburst_selection([cids[0]])  # add
    assert window._enabled_class_set() == frozenset({cids[0]})
    window._on_sunburst_selection([cids[1]])  # additive, keeps the first
    assert window._enabled_class_set() == frozenset({cids[0], cids[1]})
    window._on_sunburst_selection([cids[0]])  # re-click removes it
    assert window._enabled_class_set() == frozenset({cids[1]})


def test_master_checkbox_select_deselect_and_partial(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from PySide6.QtCore import Qt

    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)
    window._build_legend()
    present = frozenset(window._legend_toggles.keys())
    master = window._viewer.legend_overlay._master_check

    assert window._enabled_class_set() == present  # all on at build
    window._on_master_clicked()  # all -> none
    assert window._enabled_class_set() == frozenset()
    window._on_master_clicked()  # none -> all
    assert window._enabled_class_set() == present

    next(iter(window._legend_toggles.values())).setChecked(False)
    window._update_master_check()
    assert master.checkState() == Qt.CheckState.PartiallyChecked


def test_sunburst_reflects_selection(qapp, monkeypatch):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)
    # The sunburst sync runs through _on_viewer_control_changed, which only acts
    # once the viewer has scene data (as when a run is loaded). Patch via
    # monkeypatch so the class property is restored at teardown and later tests
    # (which may assert has_scene_data is False) aren't affected.
    window._viewer.apply_state = lambda **k: None
    monkeypatch.setattr(type(window._viewer), "has_scene_data", property(lambda self: True))
    window._build_legend()
    sb = window._cover_sunburst

    assert sb._selection_active is False  # all selected at build -> no dimming
    window._on_deselect_all_classes()
    assert sb._selection_active is True and sb._selected_ids == frozenset()
    cids = list(window._legend_toggles.keys())
    window._on_sunburst_selection([cids[0]])
    assert sb._selected_ids == frozenset({cids[0]}) and sb._selection_active is True
    window._on_show_all_classes()
    assert sb._selection_active is False


# --- Batch CSV parser ---

@pytest.mark.parametrize(
    "text, expected",
    [
        ("12-45.5", (12.0, 45.5)),
        ("30-", (30.0, None)),
        ("-60", (None, 60.0)),
        ("", (None, None)),
        ("15", (15.0, None)),
    ],
)
def test_parse_timestamp_range(text, expected):
    from deepreefmap.gui.app import _parse_timestamp_range

    assert _parse_timestamp_range(text) == expected


def test_load_batch_csv_parses_rows(tmp_path):
    from deepreefmap.gui.app import _load_batch_csv

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
    from deepreefmap.gui.app import _load_batch_csv

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "Videos,Timestamps,Transect_Length,Crop_Width\n"
        "x.mp4,0-10,5,1\n"
    )
    jobs = _load_batch_csv(csv_path)
    assert len(jobs) == 1


def test_load_batch_csv_rejects_missing_columns(tmp_path):
    from deepreefmap.gui.app import _load_batch_csv

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("videos,timestamps\nx.mp4,0-10\n")
    with pytest.raises(ValueError, match="missing required columns"):
        _load_batch_csv(csv_path)


def test_load_batch_csv_skips_blank_rows(tmp_path):
    from deepreefmap.gui.app import _load_batch_csv

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
    from deepreefmap.gui.app import _load_batch_csv

    bogus = tmp_path / "jobs.xlsx"
    bogus.write_bytes(b"not actually excel")
    with pytest.raises(ValueError, match="Excel"):
        _load_batch_csv(bogus)


# --- LoGeR GUI parity (WS2) ---

def test_loger_options_collected_from_form(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)

    assert window._collect_loger_options("scsfmlearner") is None

    window._loger_window_spin.setValue(16)
    window._loger_overlap_spin.setValue(2)
    window._loger_model_path_input.setText("")
    assert window._collect_loger_options("loger") == {
        "window_size": 16,
        "overlap_size": 2,
        "model_path": None,
    }

    window._loger_model_path_input.setText("/tmp/custom.pt")
    assert window._collect_loger_options("loger_star")["model_path"] == "/tmp/custom.pt"


def test_form_defaults_to_vit_b_and_loger_star(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow
    from deepreefmap.mapping.registry import loger_available

    window = DeepReefMapWindow(load_classes(), None)
    assert window._seg_combo.currentText() == "coralscapes-vit-b-dpt"
    assert window._map_combo.currentText() == (
        "loger_star" if loger_available() else "scsfmlearner"
    )
    # vit-b native (768,1376) → processing (1376,768); the Native preset feeds it unchanged.
    assert window._native_resolution == (1376, 768)


def test_loger_panel_visibility_follows_backend(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)

    window._map_combo.setCurrentText("scsfmlearner")
    assert window._loger_panel.isHidden()

    window._map_combo.setCurrentText("loger")
    assert not window._loger_panel.isHidden()


# --- Geometry-only viewer parity (WS3) ---

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


def test_geometry_scene_enables_timeline(qapp):
    from deepreefmap.gui.viewer import QtPointCloudViewer

    viewer = QtPointCloudViewer()
    fb, mr, xyz, rgb = _fake_geometry_scene()
    viewer.load_geometry_scene(fb, mr, xyz, rgb)

    assert viewer.is_geometry_mode
    assert viewer.has_scene_data
    assert viewer.n_frames == 1
    # The timeline update must work without a FinalCloudIndex (no semantic data).
    viewer.apply_geometry_state(timeline_t=0, point_size=3.0, frustums_visible=True)


def test_geometry_scene_clears_back_to_empty(qapp):
    from deepreefmap.gui.viewer import QtPointCloudViewer

    viewer = QtPointCloudViewer()
    fb, mr, xyz, rgb = _fake_geometry_scene()
    viewer.load_geometry_scene(fb, mr, xyz, rgb)
    viewer._clear_scene_data()

    assert not viewer.is_geometry_mode
    assert not viewer.has_scene_data
    assert viewer.n_frames == 0
