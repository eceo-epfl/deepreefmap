"""Linux freedesktop menu integration (gui/desktop_entry.py)."""

from __future__ import annotations

import pytest

from deepreefmap.packaging import desktop_entry


@pytest.fixture
def xdg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(desktop_entry, "_refresh_menu_database", lambda: None)
    return tmp_path


def test_install_writes_entry_and_icon(xdg_home, tmp_path) -> None:
    binary = tmp_path / "bin" / "deepreefmap"
    binary.parent.mkdir()
    binary.write_bytes(b"\x7fELF")

    entry = desktop_entry.install_desktop_entry(binary)

    assert entry == xdg_home / "applications" / "deepreefmap.desktop"
    content = entry.read_text()
    assert f"Exec={binary.resolve()}" in content
    assert "Terminal=false" in content
    icon = xdg_home / "icons" / "hicolor" / "512x512" / "apps" / "deepreefmap.png"
    assert icon.exists()
    assert "Icon=deepreefmap\n" in content
    assert desktop_entry.desktop_entry_installed()


def test_remove_cleans_up(xdg_home, tmp_path) -> None:
    binary = tmp_path / "deepreefmap"
    binary.write_bytes(b"\x7fELF")
    desktop_entry.install_desktop_entry(binary)

    desktop_entry.remove_desktop_entry()

    assert not desktop_entry.desktop_entry_installed()
    assert not (xdg_home / "applications" / "deepreefmap.desktop").exists()
    assert not (
        xdg_home / "icons" / "hicolor" / "512x512" / "apps" / "deepreefmap.png"
    ).exists()


def test_remove_is_idempotent(xdg_home) -> None:
    desktop_entry.remove_desktop_entry()
    assert not desktop_entry.desktop_entry_installed()
