from __future__ import annotations

import pytest


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
