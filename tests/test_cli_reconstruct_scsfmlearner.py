from pathlib import Path
from unittest.mock import patch

from deepreefmap.camera import intrinsics
from deepreefmap.cli import main as cli_main


def test_reconstruct_passes_default_scsfmlearner_resolution(tmp_path: Path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "reefcam.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"placeholder")

    def _fake_run_reconstruction(**kwargs):
        captured.update(kwargs)

    with patch.object(cli_main, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        intrinsics, "CAMERA_PROFILE_DIR", profile_dir
    ), patch.object(
        cli_main,
        "run_reconstruction",
        _fake_run_reconstruction,
    ):
        cli_main.reconstruct(
            videos="clip.mp4",
            camera_profile="reefcam",
            mapping="scsfmlearner",
            scsfmlearner_checkpoint_path=checkpoint_path,
            scsfmlearner_width=512,
            scsfmlearner_height=256,
        )

    mapping_options = captured["mapping_options"]
    assert isinstance(mapping_options, dict)
    assert mapping_options["target_width"] == 512
    assert mapping_options["target_height"] == 256


def test_reconstruct_passes_custom_scsfmlearner_resolution(tmp_path: Path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "reefcam.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"placeholder")

    def _fake_run_reconstruction(**kwargs):
        captured.update(kwargs)

    with patch.object(cli_main, "CAMERA_PROFILE_DIR", profile_dir), patch.object(
        intrinsics, "CAMERA_PROFILE_DIR", profile_dir
    ), patch.object(
        cli_main,
        "run_reconstruction",
        _fake_run_reconstruction,
    ):
        cli_main.reconstruct(
            videos="clip.mp4",
            camera_profile="reefcam",
            mapping="scsfmlearner",
            scsfmlearner_checkpoint_path=checkpoint_path,
            scsfmlearner_width=320,
            scsfmlearner_height=192,
        )

    mapping_options = captured["mapping_options"]
    assert isinstance(mapping_options, dict)
    assert mapping_options["target_width"] == 320
    assert mapping_options["target_height"] == 192
