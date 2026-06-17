"""Single source of truth for the app's persisted Qt settings.

Settings live in ``QSettings("ECEO", "deepreefmap")`` (``~/.config/ECEO/
deepreefmap.conf`` on Linux, registry/plist elsewhere), outside the install
tree, so they survive a PyApp reinstall or version swap. Key names are
defined once here; the existing strings are kept verbatim so values written
by earlier versions keep loading.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings

ORG = "ECEO"
APP = "deepreefmap"


class Keys:
    # Existing keys — strings kept verbatim so earlier versions' values load.
    LAST_VIDEO_PATH = "last_video_path"
    OUTPUT_ROOT_DIR = "output_root_dir"
    LAST_RUN_DIR = "last_run_dir"
    # Window geometry, persisted via Qt's QByteArray save/restore.
    WINDOW_GEOMETRY = "window_geometry"
    MAIN_SPLITTER_STATE = "main_splitter_state"


def settings() -> QSettings:
    return QSettings(ORG, APP)
