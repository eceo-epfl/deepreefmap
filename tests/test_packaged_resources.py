from pathlib import Path

from deepreefmap.camera.intrinsics import CameraProfile, available_profile_names
from deepreefmap.config.classes import COVER_LEVELS, load_classes


def test_default_classes_and_camera_profiles_load_outside_repo_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    classes = load_classes()
    profile = CameraProfile.load("gopro_hero_10")

    assert classes.classes
    assert profile.image_size == (1920, 1080)
    assert "gopro_hero_10" in available_profile_names()


def test_bundled_and_repo_classes_yaml_are_identical() -> None:
    # The bundled copy under deepreefmap/resources/configs is what ships with
    # the wheel; the top-level configs/ copy is what users edit in-tree. Drift
    # between them caused silent misclassification in the past.
    repo_path = Path("configs/classes_coralscapes.yaml")
    bundled = Path("deepreefmap/resources/configs/classes_coralscapes.yaml")
    assert repo_path.read_text() == bundled.read_text()


def test_classes_have_group_fields_for_all_levels() -> None:
    classes = load_classes()
    for cls in classes.classes:
        for level in COVER_LEVELS:
            group = classes.group_name_for_id(cls.id, level)
            assert isinstance(group, str) and group, (
                f"Class {cls.name} (id {cls.id}) missing valid group for level {level}"
            )


def test_bundled_fonts_are_present() -> None:
    # The launcher pins a global Inter font and a JetBrains Mono monospace; both
    # must ship as package data or the app silently falls back to per-OS system
    # fonts (the macOS/Linux size mismatch this was meant to fix).
    from importlib import resources

    from deepreefmap.launcher.fonts import _FONT_FILES

    fonts_dir = resources.files("deepreefmap.resources").joinpath("fonts")
    for name in _FONT_FILES:
        assert fonts_dir.joinpath(name).is_file(), f"missing bundled font {name}"
