from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from deepreefmap.camera import intrinsics
from deepreefmap.cli import main as cli_main
from deepreefmap.pipeline import orchestrator as orchestrator_mod


def _profile_dir(tmp_path: Path) -> Path:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "reefcam.json").write_text("{}", encoding="utf-8")
    return profile_dir


def test_reconstruct_passes_lingbot_map_defaults(tmp_path: Path):
    profile_dir = _profile_dir(tmp_path)
    captured: dict[str, object] = {}

    def _fake_run_reconstruction(**kwargs):
        captured.update(kwargs)

    with patch.object(intrinsics, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        orchestrator_mod,
        "run_reconstruction",
        _fake_run_reconstruction,
    ):
        cli_main.reconstruct(
            videos="clip.mp4",
            camera_profile="reefcam",
            mapping="lingbot_map",
        )

    mapping_options = captured["mapping_options"]
    assert isinstance(mapping_options, dict)
    assert mapping_options["mode"] == "streaming"
    assert mapping_options["keyframe_interval"] is None
    assert mapping_options["window_size"] == 64
    assert mapping_options["overlap_keyframes"] is None
    assert mapping_options["attention"] == "auto"
    assert mapping_options["model_path"] is None
    # Flag not given -> None, so the orchestrator applies the backend default.
    assert captured["refine_intrinsics_from_mapper"] is None


def test_reconstruct_passes_explicit_no_refine_intrinsics(tmp_path: Path):
    profile_dir = _profile_dir(tmp_path)
    captured: dict[str, object] = {}

    def _fake_run_reconstruction(**kwargs):
        captured.update(kwargs)

    with patch.object(intrinsics, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        orchestrator_mod,
        "run_reconstruction",
        _fake_run_reconstruction,
    ):
        cli_main.reconstruct(
            videos="clip.mp4",
            camera_profile="reefcam",
            mapping="lingbot_map",
            refine_intrinsics_from_mapper=False,
        )

    assert captured["refine_intrinsics_from_mapper"] is False


def test_reconstruct_passes_lingbot_map_windowed_options(tmp_path: Path):
    profile_dir = _profile_dir(tmp_path)
    captured: dict[str, object] = {}

    def _fake_run_reconstruction(**kwargs):
        captured.update(kwargs)

    with patch.object(intrinsics, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        orchestrator_mod,
        "run_reconstruction",
        _fake_run_reconstruction,
    ):
        cli_main.reconstruct(
            videos="clip.mp4",
            camera_profile="reefcam",
            mapping="lingbot_map",
            lingbot_map_mode="windowed",
            lingbot_map_window_size=32,
            lingbot_map_overlap_keyframes=8,
            lingbot_map_keyframe_interval=2,
            lingbot_map_attention="sdpa",
        )

    mapping_options = captured["mapping_options"]
    assert mapping_options["mode"] == "windowed"
    assert mapping_options["window_size"] == 32
    assert mapping_options["overlap_keyframes"] == 8
    assert mapping_options["keyframe_interval"] == 2
    assert mapping_options["attention"] == "sdpa"


def test_reconstruct_rejects_bad_lingbot_map_mode(tmp_path: Path):
    profile_dir = _profile_dir(tmp_path)

    with patch.object(intrinsics, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        orchestrator_mod, "run_reconstruction", lambda **kwargs: None
    ):
        with pytest.raises(typer.Exit):
            cli_main.reconstruct(
                videos="clip.mp4",
                camera_profile="reefcam",
                mapping="lingbot_map",
                lingbot_map_mode="turbo",
            )


def test_reconstruct_rejects_bad_lingbot_map_keyframe_interval(tmp_path: Path):
    profile_dir = _profile_dir(tmp_path)

    with patch.object(intrinsics, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        orchestrator_mod, "run_reconstruction", lambda **kwargs: None
    ):
        with pytest.raises(typer.Exit):
            cli_main.reconstruct(
                videos="clip.mp4",
                camera_profile="reefcam",
                mapping="lingbot_map",
                lingbot_map_keyframe_interval=0,
            )


def test_reconstruct_rejects_missing_lingbot_map_checkpoint(tmp_path: Path):
    profile_dir = _profile_dir(tmp_path)
    missing = tmp_path / "nope.pt"

    with patch.object(intrinsics, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        orchestrator_mod, "run_reconstruction", lambda **kwargs: None
    ):
        with pytest.raises(typer.Exit):
            cli_main.reconstruct(
                videos="clip.mp4",
                camera_profile="reefcam",
                mapping="lingbot_map",
                lingbot_map_model_path=missing,
            )


def test_reconstruct_passes_lingbot_map_model_path(tmp_path: Path):
    profile_dir = _profile_dir(tmp_path)
    checkpoint_path = tmp_path / "lingbot.pt"
    checkpoint_path.write_bytes(b"placeholder")
    captured: dict[str, object] = {}

    def _fake_run_reconstruction(**kwargs):
        captured.update(kwargs)

    with patch.object(intrinsics, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        orchestrator_mod,
        "run_reconstruction",
        _fake_run_reconstruction,
    ):
        cli_main.reconstruct(
            videos="clip.mp4",
            camera_profile="reefcam",
            mapping="lingbot_map",
            lingbot_map_model_path=checkpoint_path,
        )

    mapping_options = captured["mapping_options"]
    assert mapping_options["model_path"] == str(checkpoint_path)
