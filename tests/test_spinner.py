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


def _plain(html: str) -> str:
    import re

    # The status label is two lines joined by <br>; render that as a space.
    return re.sub("<[^>]+>", "", re.sub(r"<br\s*/?>", " ", html)).strip()


def test_status_ticker_appends_elapsed_and_keeps_base(qapp, monkeypatch):
    import deepreefmap.gui.progress as progress_mod

    window = _make_window(qapp)
    window._begin_progress(window._recon_model)

    clock = [1000.0]
    monkeypatch.setattr(progress_mod.time, "monotonic", lambda: clock[0])

    window._apply_progress("mapping", "Mapping", current=3, total=10)
    assert _plain(window._status_label.text()) == "Mapping · Mapping 3/10 · 0s"

    clock[0] += 74.0
    window._render_status()
    text = _plain(window._status_label.text())
    assert "Mapping 3/10" in text
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
    # The line may carry a stage remainder after it, so pick the elapsed field.
    window._render_status()
    parts = _plain(window._status_label.text()).split(" · ")
    assert parts[2] == "0s"


def test_update_progress_zero_total_is_indeterminate(qapp):
    window = _make_window(qapp)
    window._begin_progress(window._recon_model)
    # Non-mapping stages keep the barber-pole for a zero (indeterminate) total.
    window._on_viewer_status(
        "update_progress", stage="outputs", current=0, total=0, message="Generating outputs"
    )
    assert window._progress_bar.maximum() == 0
    assert "Generating outputs" in window._status_label.text()


def test_mapping_zero_total_holds_the_continuous_bar(qapp):
    window = _make_window(qapp)
    window._begin_progress(window._recon_model)
    # Mapping is one continuous 0-100 bar, so its indeterminate sub-steps (prep,
    # GPU transfer, resume save) show a held determinate bar, not a reset.
    window._on_viewer_status(
        "update_progress", stage="mapping", current=0, total=0, message="LoGeR inference"
    )
    assert window._progress_bar.maximum() == 100
    assert "LoGeR inference" in window._status_label.text()


def test_start_button_disabled_when_form_invalid(qapp):
    window = _make_window(qapp)
    window._video_input.setText("")
    window._recompute_submit_state()
    assert not window._start_btn.isEnabled()
    assert "Cannot start" in window._start_btn.toolTip()


def test_run_controls_morph_setup_to_running(qapp):
    window = _make_window(qapp)
    # isHidden, not isVisible: the offscreen test window is never shown on screen.
    window._start_btn.setVisible(False)
    window._pause_btn.setVisible(True)
    window._spinner_stop.setVisible(True)
    window._end_run_controls()
    assert not window._start_btn.isHidden()
    assert window._pause_btn.isHidden()
    assert window._spinner_stop.isHidden()


def test_bars_carry_no_text_and_overall_estimate_is_visible(qapp, monkeypatch, tmp_path):
    import deepreefmap.gui.progress as progress_mod

    # Isolate the profile so the total slot's first-run state is deterministic.
    monkeypatch.setenv("DEEPREEFMAP_RUN_TIMINGS", str(tmp_path / "none.json"))
    window = _make_window(qapp)
    # The bars are graphical only; the numbers live in the status text and label.
    assert not window._progress_bar.isTextVisible()
    assert not window._total_progress_bar.isTextVisible()

    clock = [0.0]
    monkeypatch.setattr(progress_mod.time, "monotonic", lambda: clock[0])
    window._begin_progress(window._recon_model)
    window._render_eta()
    assert window._eta_total_label.text() == "estimating…"
    # Give the estimator history so the whole-run total is shown, not withheld.
    window._eta.priors = {"mapping": 0.5}
    window._apply_progress("mapping", "Mapping", current=1, total=100)
    clock[0] = 20.0
    window._apply_progress("mapping", "Mapping", current=25, total=100)
    assert "left" in window._eta_total_label.text()


def test_first_run_popup_hides_future_estimates_but_shows_measured(qapp, monkeypatch, tmp_path):
    import deepreefmap.gui.progress as progress_mod
    from PySide6.QtCore import QPointF

    # Isolate the timing profile so the host machine's real history can't leak in.
    monkeypatch.setenv("DEEPREEFMAP_RUN_TIMINGS", str(tmp_path / "none.json"))
    clock = [0.0]
    monkeypatch.setattr(progress_mod.time, "monotonic", lambda: clock[0])
    window = _make_window(qapp)
    window._begin_progress(window._recon_model)
    assert not window._eta.has_history
    window._apply_progress("mapping", "Mapping", current=1, total=100)
    clock[0] = 20.0
    window._apply_progress("mapping", "Mapping", current=25, total=100)
    window._on_total_bar_hover(QPointF(50.0, 50.0))
    text = window._timing_popup._label.text()
    assert "learning timings" in text
    assert "running" in text and "left" in text


def test_hover_popup_builds_rows_from_estimator(qapp, monkeypatch):
    import deepreefmap.gui.progress as progress_mod
    from PySide6.QtCore import QPointF

    window = _make_window(qapp)
    window._begin_progress(window._recon_model)
    clock = [100.0]
    monkeypatch.setattr(progress_mod.time, "monotonic", lambda: clock[0])
    window._apply_progress("preprocess", "Preprocess", current=1, total=10)
    clock[0] += 30.0
    window._apply_progress("mapping", "Mapping", current=2, total=10)
    window._on_total_bar_hover(QPointF(50.0, 50.0))
    assert window._timing_popup.isVisible()
    window._on_total_bar_hover(None)
    assert not window._timing_popup.isVisible()
