"""Bootstrap dispatch and env self-healing.

Covers deepreefmap/bootstrap.py: routing to the GUI vs the CLI by argv,
the Windows uninstall-key refresh no-op elsewhere, and the self-heal path that
restores a broken env and re-execs the binary.
"""

from __future__ import annotations

import os
import sys

import pytest


def test_bootstrap_self_heals_then_reexecs_when_env_broken(monkeypatch, tmp_path) -> None:
    import deepreefmap.bootstrap as bootstrap
    from deepreefmap.packaging import binary_swap

    fake_binary = tmp_path / "deepreefmap"
    fake_binary.touch()  # pyapp_binary() only trusts PYAPP paths that exist
    monkeypatch.setenv("PYAPP", str(fake_binary))
    monkeypatch.delenv("DEEPREEFMAP_SELF_HEAL_ATTEMPTED", raising=False)
    monkeypatch.setattr(binary_swap, "env_is_healthy", lambda *a, **k: False)
    restored: list[str] = []
    monkeypatch.setattr(
        binary_swap, "self_restore", lambda b: bool(restored.append(b)) or True
    )

    class _Reexec(Exception):
        pass

    execs: list[tuple] = []

    def fake_execv(path, args):
        execs.append((path, args))
        raise _Reexec

    monkeypatch.setattr(bootstrap.os, "execv", fake_execv)

    try:
        with pytest.raises(_Reexec):
            bootstrap.main()
        assert restored, "self_restore should have been invoked"
        assert execs, "binary should be re-exec'd after restore"
        assert os.environ.get("DEEPREEFMAP_SELF_HEAL_ATTEMPTED") == "1"
    finally:
        os.environ.pop("DEEPREEFMAP_SELF_HEAL_ATTEMPTED", None)


def _quiet_bootstrap(monkeypatch, tmp_path):
    monkeypatch.delenv("PYAPP", raising=False)
    monkeypatch.delenv("DEEPREEFMAP_SELF_HEAL_ATTEMPTED", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


def test_bootstrap_no_args_launches_gui(monkeypatch, tmp_path) -> None:
    import deepreefmap.bootstrap as bootstrap
    import deepreefmap.gui.app as gui_app

    _quiet_bootstrap(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["deepreefmap"])
    launched = []
    monkeypatch.setattr(gui_app, "launch", lambda: launched.append(True))
    bootstrap.main()
    assert launched == [True]


def test_bootstrap_args_dispatch_to_cli(monkeypatch, tmp_path) -> None:
    import deepreefmap.bootstrap as bootstrap
    import deepreefmap.cli.main as cli_main
    import deepreefmap.gui.app as gui_app

    _quiet_bootstrap(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["deepreefmap", "list-models"])
    cli_calls = []
    monkeypatch.setattr(cli_main, "app", lambda args: cli_calls.append(args))
    monkeypatch.setattr(
        gui_app, "launch", lambda: pytest.fail("GUI must not launch for CLI args")
    )
    bootstrap.main()
    assert cli_calls == [["list-models"]]


def test_refresh_uninstall_display_version_noop_off_windows(monkeypatch) -> None:
    import deepreefmap.bootstrap as bootstrap

    # Must never raise on non-Windows or when the installer key is absent.
    bootstrap._refresh_uninstall_display_version()
