import pytest

from deepreefmap.survey.preset import PRESET_KEYS, load_survey_preset, parse_preset

VALID = """
schema_version: 1
fps: 5
segmentation_name: coralscapes-vit-b-dpt
mapping_name: loger_star
camera_profile_name: gopro_hero_10
transect_crop_width: 1.0
enable_tsdf: false
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
        parse_preset(VALID.replace("schema_version: 1", "schema_version: 99"))


def test_parse_rejects_missing_and_unknown_keys():
    with pytest.raises(ValueError, match="missing"):
        parse_preset("schema_version: 1\nfps: 5\n")
    with pytest.raises(ValueError, match="unknown"):
        parse_preset(VALID + "surprise: true\n")
