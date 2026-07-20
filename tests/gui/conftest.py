from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_ui_mode(qapp):
    """Keep the persisted mode and preview toggles from leaking between tests."""
    from PySide6.QtCore import QSettings

    settings = QSettings("ECEO", "deepreefmap")
    for key in ("ui_mode", "preview_3d"):
        settings.remove(key)
    yield
    for key in ("ui_mode", "preview_3d"):
        settings.remove(key)


@pytest.fixture(autouse=True)
def _offline_tiles(qapp):
    """Map widgets must never fetch tiles during tests."""
    from deepreefmap.gui.map.tile_cache import shared_tile_cache

    cache = shared_tile_cache()
    cache.network_enabled = False
    yield


@pytest.fixture
def make_window(qapp):
    """Factory building a fresh DeepReefMapWindow with the built-in classes.

    A factory rather than an instance so tests can set env vars (mock PyApp,
    timing-profile path) before construction.
    """
    pytest.importorskip("torch", reason="torch not loadable on this machine")

    def _make():
        from deepreefmap.config.classes import load_classes
        from deepreefmap.gui.app import DeepReefMapWindow

        return DeepReefMapWindow(load_classes(), None)

    return _make


@pytest.fixture
def window(make_window):
    """A fresh DeepReefMapWindow. Function-scoped: tests mutate form state,
    legend toggles and signals, so sharing an instance would leak state."""
    return make_window()
