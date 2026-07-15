"""Entry point for the packaged (PyApp) binary.

Imports stay stdlib-only at module load so this still runs when the heavy native
deps (torch / PySide6) are what got corrupted.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Set before re-exec so a restore that doesn't fix things can't loop forever.
_HEAL_GUARD = "DEEPREEFMAP_SELF_HEAL_ATTEMPTED"

# Inno Setup builds this key from AppId in scripts/installer.iss.
_UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\DeepReefMap_is1"


def _attach_parent_console() -> None:
    """On Windows, attach stdio to the invoking terminal for CLI use.

    The binary is built with PYAPP_IS_GUI so shortcuts open no console window; the
    cost is that GUI-subsystem processes start with no stdio.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        if not ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
            return
        sys.stdout = open("CONOUT$", "w", buffering=1)  # noqa: SIM115
        sys.stderr = open("CONOUT$", "w", buffering=1)  # noqa: SIM115
        sys.stdin = open("CONIN$")  # noqa: SIM115
    except Exception:
        pass


def _ensure_stdio_streams() -> None:
    """Guarantee stdout/stderr/stdin are never None.

    A GUI-subsystem process on Windows starts with no console, so all three are None
    and tqdm dies on its first refresh, killing the reconstruction.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
    if sys.stdin is None:
        sys.stdin = open(os.devnull)  # noqa: SIM115


def _refresh_uninstall_display_version() -> None:
    """Keep Add/Remove Programs in sync after an in-app update or rollback.

    The installer writes the uninstall key once; in-app updates swap the binary
    without re-running it, so the recorded version goes stale.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import importlib.metadata
        import winreg

        version = importlib.metadata.version("deepreefmap")
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _UNINSTALL_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, version)
    except FileNotFoundError:
        pass  # Portable or dev install, so there is no installer key.
    except OSError as exc:
        logger.warning("Could not refresh uninstall DisplayVersion: %s", exc)


def main() -> None:
    args = sys.argv[1:]
    if args:
        _attach_parent_console()
    _ensure_stdio_streams()

    from deepreefmap.packaging.binary_swap import (
        cleanup_stale_backups,
        env_is_healthy,
        prune_stale_envs,
        pyapp_binary,
        self_restore,
    )

    binary = pyapp_binary()
    if binary and not os.environ.get(_HEAL_GUARD) and not env_is_healthy():
        if self_restore(binary):
            os.environ[_HEAL_GUARD] = "1"
            os.execv(binary, [binary, *sys.argv[1:]])
        # Restore failed: fall through so launch surfaces the real error.

    if binary:
        from pathlib import Path

        cleanup_stale_backups(Path(binary))

    # This version has provisioned successfully (we got here), so old version envs,
    # left by in-app updates or installer reinstalls, can go. The newest one is kept
    # as an offline rollback target.
    prune_stale_envs()

    _refresh_uninstall_display_version()

    if args:
        from deepreefmap.cli.main import app

        app(args)
        return

    from deepreefmap.gui.app import launch

    launch()


if __name__ == "__main__":
    main()
