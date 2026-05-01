from deepreefmap.camera.intrinsics import CameraProfile, available_profile_names
from deepreefmap.config.classes import load_classes


def test_default_classes_and_camera_profiles_load_outside_repo_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    classes = load_classes()
    profile = CameraProfile.load("gopro_hero_10")

    assert classes.classes
    assert profile.image_size == (1920, 1080)
    assert "gopro_hero_10" in available_profile_names()
