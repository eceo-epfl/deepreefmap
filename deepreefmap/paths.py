"""Filesystem locations that must stay writable in a frozen PyApp binary."""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs


def loger_ckpts_dir() -> Path:
    """Directory holding LoGeR checkpoints (``latest.pt`` + ``original_config.yaml``).

    Must be user-writable: a deployed PyApp binary's install tree is read-only.
    ``DEEPREEFMAP_LOGER_CKPTS`` overrides it, but only if set before import.
    mapping/registry.py and gui/models/manager.py snapshot this at module level while
    loger_backend.py calls it per instance, so a later change splits them.
    """
    override = os.environ.get("DEEPREEFMAP_LOGER_CKPTS")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("deepreefmap", appauthor=False)) / "loger_ckpts"


def survey_preset_path() -> Path:
    """User override for the bundled survey-mode run preset."""
    return Path(platformdirs.user_data_dir("deepreefmap", appauthor=False)) / "survey_preset.yaml"


def tile_cache_dir() -> Path:
    """Persistent map tile cache; tiles land here only after being displayed."""
    return Path(platformdirs.user_cache_dir("deepreefmap", appauthor=False)) / "tiles"


def run_timings_path() -> Path:
    """Local timing/peak profile recorded per run, overridable for tests."""
    override = os.environ.get("DEEPREEFMAP_RUN_TIMINGS")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("deepreefmap", appauthor=False)) / "run_timings.json"


def env_prune_marker_path() -> Path:
    """Marker from the retired prune mechanism, kept so the launch-time sweep can unlink it."""
    return Path(platformdirs.user_data_dir("deepreefmap", appauthor=False)) / "pending_env_prune.json"
