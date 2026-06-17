"""Entry point for the packaged (PyApp) binary.

Runs before the GUI on every launch to keep the environment in sync with the
binary: re-provision it if files were deleted (OS update / antivirus), and drop
the previous version's environment after an in-app update.

Imports stay stdlib-only at module load so this still runs when the heavy native
deps (torch / PySide6) are what got corrupted. The dev path
(`uv run deepreefmap launch`) calls `deepreefmap.gui.app:launch` directly and
skips all of it.
"""

from __future__ import annotations

import os
import sys

# Set before re-exec so a restore that doesn't fix things can't loop forever.
_HEAL_GUARD = "DEEPREEFMAP_SELF_HEAL_ATTEMPTED"


def main() -> None:
    from deepreefmap.gui.binary_swap import (
        cleanup_stale_backups,
        env_is_healthy,
        prune_previous_env,
        self_restore,
    )

    binary = os.environ.get("PYAPP")
    if (
        binary
        and binary != "1"
        and not os.environ.get(_HEAL_GUARD)
        and not env_is_healthy()
    ):
        if self_restore(binary):
            os.environ[_HEAL_GUARD] = "1"
            os.execv(binary, [binary, *sys.argv[1:]])
        # Restore failed: fall through so launch surfaces the real error.

    if binary and binary != "1":
        from pathlib import Path

        cleanup_stale_backups(Path(binary))

    # The new version has provisioned successfully (we got here), so it is safe
    # to drop the environment a prior in-app update left behind.
    prune_previous_env()

    from deepreefmap.gui.app import launch

    launch()


if __name__ == "__main__":
    main()
