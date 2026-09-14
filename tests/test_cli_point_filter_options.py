from pathlib import Path
from unittest.mock import patch

from deepreefmap.camera import intrinsics
from deepreefmap.cli import main as cli_main
from deepreefmap.cli.main import _optional_filter_arg
from deepreefmap.pipeline import orchestrator as orchestrator_mod


def _profile_dir(tmp_path: Path) -> Path:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "reefcam.json").write_text("{}", encoding="utf-8")
    return profile_dir


def _run_reconstruct(tmp_path: Path, **overrides) -> dict:
    profile_dir = _profile_dir(tmp_path)
    captured: dict[str, object] = {}

    def _fake_run_reconstruction(**kwargs):
        captured.update(kwargs)

    with patch.object(intrinsics, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        orchestrator_mod, "run_reconstruction", _fake_run_reconstruction
    ):
        cli_main.reconstruct(
            videos="clip.mp4",
            camera_profile="reefcam",
            mapping="vggt_omega",
            **overrides,
        )
    return captured


def test_reconstruct_forwards_default_point_filter_options(tmp_path: Path):
    captured = _run_reconstruct(tmp_path)
    assert captured["confidence_percentile"] == 20.0
    assert captured["depth_edge_rtol"] == 0.03


def test_reconstruct_zero_disables_point_filters(tmp_path: Path):
    captured = _run_reconstruct(tmp_path, confidence_percentile=0.0, depth_edge_rtol=0.0)
    assert captured["confidence_percentile"] is None
    assert captured["depth_edge_rtol"] is None


def test_reconstruct_forwards_explicit_point_filter_options(tmp_path: Path):
    captured = _run_reconstruct(tmp_path, confidence_percentile=50.0, depth_edge_rtol=0.05)
    assert captured["confidence_percentile"] == 50.0
    assert captured["depth_edge_rtol"] == 0.05


def test_optional_filter_arg_semantics():
    # Real numbers: <= 0 disables, > 0 passes through as float.
    assert _optional_filter_arg(20.0, 20.0) == 20.0
    assert _optional_filter_arg(0.0, 20.0) is None
    assert _optional_filter_arg(-1.0, 20.0) is None
    # Non-numbers (e.g. Typer OptionInfo on direct calls) fall back to the default.
    assert _optional_filter_arg(object(), 0.03) == 0.03
    assert _optional_filter_arg(True, 0.03) == 0.03
