"""Entry point for the packaged (PyApp) binary.

Runs on every launch to keep the environment in sync with the binary:
re-provision it if files were deleted (OS update / antivirus), and drop the
previous version's environment after an in-app update. Then dispatches: no
arguments launches the GUI (double-click, desktop shortcut), any arguments go
to the Typer CLI, so the installed binary supports `deepreefmap reconstruct …`
and friends.

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


def _attach_parent_console() -> None:
    """On Windows, attach stdio to the invoking terminal for CLI use.

    The Windows binary is built with PYAPP_IS_GUI so shortcuts open no console
    window; the cost is that GUI-subsystem processes start with no stdio. When
    the user runs the binary from a terminal with arguments, attach to that
    terminal's console so Typer output is visible. Best-effort: silently a
    no-op off Windows or when there is no parent console (e.g. launched by a
    script with args but no terminal).
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

    A GUI-subsystem process on Windows starts with no console, so all three
    std streams are None. Anything that writes to them then crashes: tqdm
    defaults to file=sys.stderr and dies with 'NoneType has no attribute
    write' on its first refresh, killing the reconstruction. Point any missing
    stream at the null device so writes are silently discarded.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
    if sys.stdin is None:
        sys.stdin = open(os.devnull)  # noqa: SIM115


def _refresh_uninstall_display_version() -> None:
    """Keep Add/Remove Programs in sync after an in-app update or rollback.

    The Inno Setup installer writes the uninstall registry key once; in-app
    updates swap the binary without re-running it, so the recorded version goes
    stale. Rewrite DisplayVersion with the running version on every launch.
    No-op unless installed via the installer (key present). Best-effort.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import importlib.metadata
        import winreg

        version = importlib.metadata.version("deepreefmap")
        key_path = (
            r"Software\Microsoft\Windows\CurrentVersion\Uninstall\DeepReefMap_is1"
        )
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, version)
    except Exception:
        pass


def main() -> None:
    args = sys.argv[1:]
    if args:
        _attach_parent_console()
    _ensure_stdio_streams()

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

    _refresh_uninstall_display_version()

    if args:
        from deepreefmap.cli.main import app

        app(args)
        return

    from deepreefmap.gui.app import launch

    launch()


if __name__ == "__main__":
    main()
