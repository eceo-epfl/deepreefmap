"""Filesystem locations that must stay writable in a frozen PyApp binary."""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs


def loger_ckpts_dir() -> Path:
    """Directory holding LoGeR checkpoints (``latest.pt`` + ``original_config.yaml``).

    The LoGeR backend and the model-download flow both resolve checkpoints here, so
    it is the single source of truth for their location. It must be user-writable:
    in a deployed PyApp binary the install tree (where ``third_party/LoGeR/ckpts``
    lived in a source checkout) is read-only. Set ``DEEPREEFMAP_LOGER_CKPTS`` to
    override, e.g. to reuse an existing ``third_party/LoGeR/ckpts`` in development.
    """
    override = os.environ.get("DEEPREEFMAP_LOGER_CKPTS")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("deepreefmap", appauthor=False)) / "loger_ckpts"


def env_prune_marker_path() -> Path:
    """Marker file used by the retired marker-based env pruning.

    Superseded by the launch-time sweep (`binary_swap.prune_stale_envs`), which
    unlinks any leftover marker. Kept only for that cleanup.
    """
    return Path(platformdirs.user_data_dir("deepreefmap", appauthor=False)) / "pending_env_prune.json"
