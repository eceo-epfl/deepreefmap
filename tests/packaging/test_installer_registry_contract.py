"""Inno Setup names the uninstall registry key after `AppId`, and bootstrap.py
rewrites `DisplayVersion` under that key on every launch so in-app updates don't
leave a stale version in Add/Remove Programs. Nothing at runtime ties the two
together: rename `AppId` and the write lands on a key that doesn't exist, which
is a silent no-op forever. This test drives the real refresh against a fake
registry and pins the key it opens to the AppId the installer actually ships.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from types import SimpleNamespace

_INSTALLER_ISS = Path(__file__).parents[2] / "scripts" / "installer.iss"


def _installer_app_id() -> str:
    """The AppId= value the installer's [Setup] section declares."""
    for line in _INSTALLER_ISS.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("AppId="):
            return stripped.split("=", 1)[1].strip()
    raise AssertionError(f"no AppId= in {_INSTALLER_ISS}")


class _FakeKey:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def __enter__(self) -> _FakeKey:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _fake_winreg(opened: list[str], values: dict[str, str]) -> SimpleNamespace:
    """Stand-in for the Windows-only winreg, recording which key was opened."""

    def open_key(root: object, path: str, reserved: int, access: int) -> _FakeKey:
        opened.append(path)
        return _FakeKey(values)

    def set_value_ex(
        key: _FakeKey, name: str, reserved: int, value_type: int, value: str
    ) -> None:
        key.values[name] = value

    return SimpleNamespace(
        HKEY_CURRENT_USER=object(),
        KEY_SET_VALUE=0x0002,
        REG_SZ=1,
        OpenKey=open_key,
        SetValueEx=set_value_ex,
    )


def test_uninstall_key_matches_installer_app_id(monkeypatch) -> None:
    import deepreefmap.bootstrap as bootstrap

    opened: list[str] = []
    values: dict[str, str] = {}
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(opened, values))

    bootstrap._refresh_uninstall_display_version()

    app_id = _installer_app_id()
    expected = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{app_id}_is1"
    assert opened == [expected], "bootstrap.py writes to a key the installer never makes"
    assert values["DisplayVersion"] == importlib.metadata.version("deepreefmap")
