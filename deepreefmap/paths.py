"""Filesystem locations that must stay writable in a frozen PyApp binary."""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs


def loger_ckpts_dir() -> Path:
    """Folder for LoGeR checkpoints, writable by the user.

    An installed GUI cannot write into its own program files, so checkpoints
    cannot sit in third_party/. Its model manager downloads them here instead.
    ``DEEPREEFMAP_LOGER_CKPTS`` points elsewhere for development checkouts
    that already have the files.
    """
    override = os.environ.get("DEEPREEFMAP_LOGER_CKPTS")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("deepreefmap", appauthor=False)) / "loger_ckpts"  # eg. ~/.local/share/deepreefmap on Linux
