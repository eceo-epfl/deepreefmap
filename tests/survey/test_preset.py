import pytest

from deepreefmap.survey.preset import (
    PRESET_KEYS,
    load_survey_preset,
    parse_preset,
    save_user_preset,
)

from deepreefmap.survey.preset import _bundled_defaults

SAMPLE = {
    **_bundled_defaults(),
    "fps": 4,
    "segmentation_name": "segformer-b2",
    "mapping_name": "scsfmlearner",
    "transect_crop_width": None,
    "skip_segmentation": True,
}

VALID = "schema_version: 2\n" + "\n".join(
    f"{key}: {value!r}" if isinstance(value, str) else f"{key}: {value}"
    for key, value in _bundled_defaults().items()
)

# What a machine that last ran the seven-key schema still has on disk.
V1_FILE = """
schema_version: 1
fps: 3
segmentation_name: segformer-b2
mapping_name: scsfmlearner
camera_profile_name: gopro_hero_10
transect_crop_width: 2.0
enable_tsdf: true
skip_segmentation: false
"""


def test_bundled_preset_loads(monkeypatch):
    monkeypatch.delenv("DEEPREEFMAP_SURVEY_PRESET", raising=False)
    preset = load_survey_preset()
    assert set(preset) == PRESET_KEYS
    assert isinstance(preset["fps"], int)


def test_env_override_wins(tmp_path, monkeypatch):
    override = tmp_path / "preset.yaml"
    override.write_text(VALID.replace("fps: 5", "fps: 3"))
    monkeypatch.setenv("DEEPREEFMAP_SURVEY_PRESET", str(override))
    assert load_survey_preset()["fps"] == 3


def test_parse_rejects_wrong_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        parse_preset(VALID.replace("schema_version: 2", "schema_version: 99"))


def test_parse_rejects_missing_and_unknown_keys():
    with pytest.raises(ValueError, match="missing"):
        parse_preset("schema_version: 2\nfps: 5\n")
    with pytest.raises(ValueError, match="unknown"):
        parse_preset(VALID + "\nsurprise: true\n")


def test_v1_file_upgrades_keeping_its_own_choices():
    """Scenario: a field laptop that last ran the seven-key schema.

    Expected behaviour: its settings survive, the new keys arrive as shipped.
    """
    preset = parse_preset(V1_FILE)
    assert set(preset) == PRESET_KEYS
    assert preset["fps"] == 3
    assert preset["mapping_name"] == "scsfmlearner"
    assert preset["enable_tsdf"] is True
    assert preset["grid_bins"] == _bundled_defaults()["grid_bins"]


def test_upgraded_v1_file_saves_back_as_v2(tmp_path, monkeypatch):
    target = tmp_path / "survey_preset.yaml"
    target.write_text(V1_FILE)
    monkeypatch.setattr("deepreefmap.survey.preset.survey_preset_path", lambda: target)
    monkeypatch.delenv("DEEPREEFMAP_SURVEY_PRESET", raising=False)
    save_user_preset(load_survey_preset())
    assert "schema_version: 2" in target.read_text()
    assert load_survey_preset()["fps"] == 3


def test_native_processing_size_stays_null_by_default():
    """A null size means follow the segmentation model, so the bundled preset
    must not pin numbers that would override it."""
    defaults = _bundled_defaults()
    assert defaults["processing_width"] is None
    assert defaults["processing_height"] is None


def test_save_user_preset_round_trips(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "survey_preset.yaml"
    monkeypatch.setattr("deepreefmap.survey.preset.survey_preset_path", lambda: target)
    monkeypatch.delenv("DEEPREEFMAP_SURVEY_PRESET", raising=False)
    path = save_user_preset(SAMPLE)
    assert path == target
    assert not target.with_name(target.name + ".tmp").exists()
    assert load_survey_preset() == SAMPLE


def test_save_user_preset_rejects_bad_keys():
    with pytest.raises(ValueError, match="unknown"):
        save_user_preset({**SAMPLE, "surprise": True})
    with pytest.raises(ValueError, match="missing"):
        save_user_preset({k: v for k, v in SAMPLE.items() if k != "fps"})
