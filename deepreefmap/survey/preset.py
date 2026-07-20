"""Bundled pipeline settings for survey mode, overridable per machine."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from deepreefmap.paths import survey_preset_path

PRESET_SCHEMA_VERSION = 1

# Exactly the run_reconstruction kwargs survey mode fixes; per-pass values
# (transect length, begin/end trim) come from the survey database.
PRESET_KEYS = {
    "fps",
    "segmentation_name",
    "mapping_name",
    "camera_profile_name",
    "transect_crop_width",
    "enable_tsdf",
    "skip_segmentation",
}


def load_survey_preset() -> dict[str, Any]:
    """Resolve the preset: $DEEPREEFMAP_SURVEY_PRESET, then user copy, then bundled."""
    override = os.environ.get("DEEPREEFMAP_SURVEY_PRESET")
    if override:
        return parse_preset(Path(override).read_text())
    user_copy = survey_preset_path()
    if user_copy.is_file():
        return parse_preset(user_copy.read_text())
    bundled = resources.files("deepreefmap.resources").joinpath("configs/survey_preset.yaml")
    return parse_preset(bundled.read_text())


def parse_preset(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Survey preset must be a YAML mapping.")
    version = data.pop("schema_version", None)
    if version != PRESET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported survey preset schema_version: {version}")
    missing = PRESET_KEYS - set(data)
    if missing:
        raise ValueError(f"Survey preset is missing keys: {', '.join(sorted(missing))}")
    unknown = set(data) - PRESET_KEYS
    if unknown:
        raise ValueError(f"Survey preset has unknown keys: {', '.join(sorted(unknown))}")
    return data
