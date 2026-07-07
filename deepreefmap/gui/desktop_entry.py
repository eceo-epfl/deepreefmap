"""Linux freedesktop menu integration for the packaged binary.

Writes a `.desktop` entry pointing at the running PyApp binary so the app shows
up in the applications menu / launcher. The in-app updater swaps the binary at
that same path, so the entry stays valid across updates and rollbacks.

Qt-free (like `binary_swap.py`) so unit tests and any future CLI hook share one
implementation.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ENTRY_NAME = "deepreefmap.desktop"
_ICON_NAME = "deepreefmap.png"


def _data_home() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    return Path(xdg) if xdg else Path.home() / ".local" / "share"


def _entry_path() -> Path:
    return _data_home() / "applications" / _ENTRY_NAME


def _icon_path() -> Path:
    return _data_home() / "icons" / "hicolor" / "512x512" / "apps" / _ICON_NAME


def _refresh_menu_database() -> None:
    """Ask the desktop environment to pick up the change. Best-effort."""
    exe = shutil.which("update-desktop-database")
    if exe is None:
        return
    try:
        subprocess.run(
            [exe, str(_data_home() / "applications")],
            check=False,
            capture_output=True,
        )
    except OSError:
        logger.debug("update-desktop-database failed", exc_info=True)


def desktop_entry_supported() -> bool:
    return sys.platform.startswith("linux")


def desktop_entry_installed() -> bool:
    return _entry_path().exists()


def install_desktop_entry(binary_path: str | os.PathLike[str]) -> Path:
    """Write the menu entry and icon for the binary. Returns the entry path."""
    binary = Path(binary_path).resolve()

    icon_dest = _icon_path()
    icon_dest.parent.mkdir(parents=True, exist_ok=True)
    from importlib import resources

    icon_src = resources.files("deepreefmap.resources") / "icon.png"
    icon_dest.write_bytes(icon_src.read_bytes())

    entry = _entry_path()
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=DeepReefMap\n"
        "Comment=Underwater 3D reconstruction and benthic cover mapping\n"
        f"Exec={binary}\n"
        # Theme-name lookup against the hicolor icon installed above; KDE's
        # launcher does not reliably render absolute Icon= paths.
        "Icon=deepreefmap\n"
        "Terminal=false\n"
        "Categories=Science;\n"
    )
    _refresh_menu_database()
    return entry


def remove_desktop_entry() -> None:
    """Delete the menu entry and icon written by `install_desktop_entry`."""
    _entry_path().unlink(missing_ok=True)
    _icon_path().unlink(missing_ok=True)
    _refresh_menu_database()
