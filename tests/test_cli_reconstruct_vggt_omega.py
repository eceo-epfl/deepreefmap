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


def test_reconstruct_passes_vggt_omega_options(tmp_path: Path):
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
            mapping="vggt_omega",
            vggt_omega_image_resolution=640,
        )

    mapping_options = captured["mapping_options"]
    assert isinstance(mapping_options, dict)
    assert mapping_options["image_resolution"] == 640
    assert mapping_options["model_path"] is None


def test_reconstruct_rejects_bad_vggt_omega_resolution(tmp_path: Path):
    profile_dir = _profile_dir(tmp_path)

    with patch.object(intrinsics, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        orchestrator_mod, "run_reconstruction", lambda **kwargs: None
    ):
        with pytest.raises(typer.Exit):
            cli_main.reconstruct(
                videos="clip.mp4",
                camera_profile="reefcam",
                mapping="vggt_omega",
                vggt_omega_image_resolution=500,
            )


def test_reconstruct_passes_vggt_omega_model_path(tmp_path: Path):
    profile_dir = _profile_dir(tmp_path)
    checkpoint_path = tmp_path / "vggt.pt"
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
            mapping="vggt_omega",
            vggt_omega_model_path=checkpoint_path,
        )

    mapping_options = captured["mapping_options"]
    assert mapping_options["model_path"] == str(checkpoint_path)
