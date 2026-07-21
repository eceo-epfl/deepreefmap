"""Bundled pipeline settings for survey mode, overridable per machine."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from deepreefmap.paths import survey_preset_path

PRESET_SCHEMA_VERSION = 2

# Schema 1 held only the seven core settings; a file that old is upgraded on read.
_SUPPORTED_VERSIONS = (1, 2)

# A snapshot of every run-form setting, so simple mode can offer the full form
# and have the choices survive a restart. Per-pass values (transect length,
# begin/end trim) come from the survey database, never from here.
PRESET_KEYS = {
    "fps",
    "segmentation_name",
    "mapping_name",
    "camera_profile_name",
    "transect_crop_width",
    "enable_tsdf",
    "skip_segmentation",
    "resolution_preset",
    "processing_width",
    "processing_height",
    "preprocess_batch_size",
    "grid_bins",
    "require_gravity_telemetry",
    "replacement_radius_factor",
    "replacement_radius_estimation_frames",
    "replacement_radius_override",
    "loger_window_size",
    "loger_overlap_size",
    "loger_model_path",
    "refine_intrinsics_from_mapper",
    "scs_target_width",
    "scs_target_height",
    "scs_checkpoint_path",
}


def _bundled_text() -> str:
    return resources.files("deepreefmap.resources").joinpath("configs/survey_preset.yaml").read_text()


def _bundled_defaults() -> dict[str, Any]:
    """The shipped settings, read without validation so upgrades can lean on them."""
    data = yaml.safe_load(_bundled_text())
    data.pop("schema_version", None)
    return data


def load_survey_preset() -> dict[str, Any]:
    """Resolve the preset: $DEEPREEFMAP_SURVEY_PRESET, then user copy, then bundled."""
    override = os.environ.get("DEEPREEFMAP_SURVEY_PRESET")
    if override:
        return parse_preset(Path(override).read_text())
    user_copy = survey_preset_path()
    if user_copy.is_file():
        return parse_preset(user_copy.read_text())
    return parse_preset(_bundled_text())


def save_user_preset(preset: dict[str, Any]) -> Path:
    """Write the preset to the per-machine override read back by load_survey_preset."""
    unknown = set(preset) - PRESET_KEYS
    if unknown:
        raise ValueError(f"Survey preset has unknown keys: {', '.join(sorted(unknown))}")
    missing = PRESET_KEYS - set(preset)
    if missing:
        raise ValueError(f"Survey preset is missing keys: {', '.join(sorted(missing))}")
    path = survey_preset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump({"schema_version": PRESET_SCHEMA_VERSION, **preset}, sort_keys=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def parse_preset(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Survey preset must be a YAML mapping.")
    version = data.pop("schema_version", None)
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported survey preset schema_version: {version}")
    if version < PRESET_SCHEMA_VERSION:
        # Keep the machine's own choices, take the shipped value for everything
        # the older schema had no field for.
        data = {**_bundled_defaults(), **data}
    missing = PRESET_KEYS - set(data)
    if missing:
        raise ValueError(f"Survey preset is missing keys: {', '.join(sorted(missing))}")
    unknown = set(data) - PRESET_KEYS
    if unknown:
        raise ValueError(f"Survey preset has unknown keys: {', '.join(sorted(unknown))}")
    return data
