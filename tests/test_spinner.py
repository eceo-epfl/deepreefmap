"""Spinner/stop control, status elapsed-time ticker, and indeterminate bar."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module")
def qapp():
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_spinner_timer_tracks_visibility(qapp):
    from deepreefmap.gui.spinner import SpinnerStopButton

    btn = SpinnerStopButton()
    assert not btn._timer.isActive()
    btn.show()
    assert btn._timer.isActive()
    btn.hide()
    assert not btn._timer.isActive()


def test_spinner_stopping_disables_button(qapp):
    from deepreefmap.gui.spinner import SpinnerStopButton

    btn = SpinnerStopButton()
    assert btn.isEnabled()
    btn.set_stopping(True)
    assert not btn.isEnabled()
    assert "Stopping" in btn.toolTip()
    btn.set_stopping(False)
    assert btn.isEnabled()


def test_spinner_emits_clicked(qapp):
    from deepreefmap.gui.spinner import SpinnerStopButton

    fired = []
    btn = SpinnerStopButton()
    btn.clicked.connect(lambda: fired.append(True))
    btn.click()
    assert fired == [True]


def _make_window(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    return DeepReefMapWindow(load_classes(), None)


def test_status_ticker_appends_elapsed_and_keeps_base(qapp, monkeypatch):
    import deepreefmap.gui.progress as progress_mod

    window = _make_window(qapp)
    window._begin_progress(window._recon_model)

    clock = [1000.0]
    monkeypatch.setattr(progress_mod.time, "monotonic", lambda: clock[0])

    window._apply_progress("mapping", "Mapping", current=3, total=10)
    assert window._status_label.text() == "Mapping… 3/10 · 0s"

    clock[0] += 74.0
    window._render_status()
    text = window._status_label.text()
    assert text.startswith("Mapping… 3/10")
    assert text.endswith("1m 14s")


def test_status_ticker_resets_per_stage(qapp, monkeypatch):
    import deepreefmap.gui.progress as progress_mod

    window = _make_window(qapp)
    window._begin_progress(window._recon_model)

    clock = [500.0]
    monkeypatch.setattr(progress_mod.time, "monotonic", lambda: clock[0])

    window._apply_progress("preprocess", "Preprocessing", current=1, total=5)
    clock[0] += 40.0
    window._apply_progress("mapping", "Mapping", current=1, total=5)
    # A new phase restarts the stopwatch, so elapsed is near zero, not 40s.
    window._render_status()
    assert window._status_label.text().endswith("0s")


def test_update_progress_zero_total_is_indeterminate(qapp):
    window = _make_window(qapp)
    window._begin_progress(window._recon_model)
    window._on_viewer_status(
        "update_progress", stage="mapping", current=0, total=0, message="LoGeR inference"
    )
    assert window._progress_bar.maximum() == 0
    assert "LoGeR inference" in window._status_label.text()
