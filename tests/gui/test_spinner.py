"""Spinner/stop control, status elapsed-time ticker, and indeterminate bar."""

from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def _isolate_run_history(tmp_path, monkeypatch):
    # The estimator seeds priors from run_timings.json; point it at an empty temp
    # file so the status assertions don't depend on this machine's real history.
    monkeypatch.setenv("DEEPREEFMAP_RUN_TIMINGS", str(tmp_path / "run_timings.json"))


def test_spinner_timer_tracks_visibility(qapp) -> None:
    from deepreefmap.gui.core.spinner import SpinnerStopButton

    btn = SpinnerStopButton()
    assert not btn._timer.isActive()
    btn.show()
    assert btn._timer.isActive()
    btn.hide()
    assert not btn._timer.isActive()


def test_spinner_stopping_disables_button(qapp) -> None:
    from deepreefmap.gui.core.spinner import SpinnerStopButton

    btn = SpinnerStopButton()
    assert btn.isEnabled()
    btn.set_stopping(True)
    assert not btn.isEnabled()
    assert "Stopping" in btn.toolTip()
    btn.set_stopping(False)
    assert btn.isEnabled()


def test_spinner_emits_clicked(qapp) -> None:
    from deepreefmap.gui.core.spinner import SpinnerStopButton

    fired = []
    btn = SpinnerStopButton()
    btn.clicked.connect(lambda: fired.append(True))
    btn.click()
    assert fired == [True]


def _plain(html: str) -> str:
    import re

    # The status label is two lines joined by <br>; render that as a space.
    return re.sub("<[^>]+>", "", re.sub(r"<br\s*/?>", " ", html)).strip()


def test_status_ticker_appends_elapsed_and_keeps_base(make_window, monkeypatch) -> None:
    import deepreefmap.gui.runs.progress as progress_mod

    window = make_window()
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


def test_status_ticker_resets_per_stage(make_window, monkeypatch) -> None:
    import deepreefmap.gui.runs.progress as progress_mod

    window = make_window()
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


def test_update_progress_zero_total_is_indeterminate(make_window) -> None:
    window = make_window()
    window._begin_progress(window._recon_model)
    # A stage outside the continuous-fill set (here an ortho step) keeps the
    # barber-pole for a zero (indeterminate) total.
    window._on_viewer_status(
        "update_progress", stage="outputs", current=0, total=0, message="Computing PCA projection"
    )
    assert window._progress_bar.maximum() == 0
    assert "Computing PCA projection" in window._status_label.text()


def test_mapping_zero_total_holds_the_continuous_bar(make_window) -> None:
    window = make_window()
    window._begin_progress(window._recon_model)
    # Mapping is one continuous 0-100 bar, so its indeterminate sub-steps (prep,
    # GPU transfer, resume save) show a held determinate bar, not a reset.
    window._on_viewer_status(
        "update_progress", stage="mapping", current=0, total=0, message="LoGeR inference"
    )
    assert window._progress_bar.maximum() == 100
    assert "LoGeR inference" in window._status_label.text()


def test_cloud_subphases_hold_one_continuous_bar(make_window) -> None:
    window = make_window()
    window._begin_progress(window._recon_model)
    # The per-frame semantic-cloud loop only fills its slice of the Cloud stage,
    # not the whole bar. The replacement-radius tail owns most of the weight.
    window._on_viewer_status(
        "update_progress", stage="outputs", current=10, total=10, message="Building semantic cloud"
    )
    assert window._progress_bar.maximum() == 100
    filled = window._progress_bar.value()
    assert 0 < filled < 100  # not pinned at 100 when the cheap loop ends
    # The indeterminate lexsort holds a determinate bar advanced to its slice, so
    # the stage never reads as finished while it is still working.
    window._on_viewer_status(
        "set_stage", stage="outputs", status="running", message="Applying replacement radius"
    )
    assert window._progress_bar.maximum() == 100
    assert window._progress_bar.value() >= filled


def test_start_button_disabled_when_form_invalid(make_window) -> None:
    window = make_window()
    window._video_input.setText("")
    window._recompute_submit_state()
    assert not window._start_btn.isEnabled()
    assert "Cannot start" in window._start_btn.toolTip()


def test_run_controls_morph_setup_to_running(make_window) -> None:
    window = make_window()
    window._set_ui_mode("advanced")
    # isHidden, not isVisible: the offscreen test window is never shown on screen.
    window._begin_run_controls()
    assert window._start_btn.isHidden()
    assert not window._pause_btn.isHidden()
    assert not window._spinner_stop.isHidden()
    window._end_run_controls()
    assert not window._start_btn.isHidden()
    assert window._pause_btn.isHidden()
    assert window._spinner_stop.isHidden()


def test_simple_mode_never_shows_a_start_button(make_window) -> None:
    """Simple mode launches runs from its Run step, so start would be a decoy."""
    window = make_window()
    assert window._ui_mode == "simple"
    window._begin_run_controls()
    assert not window._pause_btn.isHidden()
    window._end_run_controls()
    assert window._start_btn.isHidden()


def test_bars_carry_no_text_and_overall_estimate_is_visible(make_window, monkeypatch, tmp_path) -> None:
    import deepreefmap.gui.runs.progress as progress_mod

    # Isolate the profile so the total slot's first-run state is deterministic.
    monkeypatch.setenv("DEEPREEFMAP_RUN_TIMINGS", str(tmp_path / "none.json"))
    window = make_window()
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


def test_first_run_popup_hides_future_estimates_but_shows_measured(make_window, monkeypatch, tmp_path) -> None:
    import deepreefmap.gui.runs.progress as progress_mod
    from PySide6.QtCore import QPointF

    # Isolate the timing profile so the host machine's real history can't leak in.
    monkeypatch.setenv("DEEPREEFMAP_RUN_TIMINGS", str(tmp_path / "none.json"))
    clock = [0.0]
    monkeypatch.setattr(progress_mod.time, "monotonic", lambda: clock[0])
    window = make_window()
    window._begin_progress(window._recon_model)
    assert not window._eta.has_history
    window._apply_progress("mapping", "Mapping", current=1, total=100)
    clock[0] = 20.0
    window._apply_progress("mapping", "Mapping", current=25, total=100)
    window._on_total_bar_hover(QPointF(50.0, 50.0))
    text = window._timing_popup._label.text()
    assert "learning timings" in text
    assert "running" in text and "left" in text


def test_hover_popup_builds_rows_from_estimator(make_window, monkeypatch) -> None:
    import deepreefmap.gui.runs.progress as progress_mod
    from PySide6.QtCore import QPointF

    window = make_window()
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


def test_progress_readouts_are_hidden_until_a_run_starts(window) -> None:
    """Scenario: an idle window.

    Expected behaviour: nothing reports progress until there is progress.
    """
    assert not window._progress_stack.isVisibleTo(window)
    assert not window._bottom_progress_bar.isVisibleTo(window)
    assert not window._eta_total_label.isVisibleTo(window)

    window._begin_progress(window._recon_model)
    assert window._progress_stack.isVisibleTo(window)
    assert window._bottom_progress_bar.isVisibleTo(window)

    window._reset_progress_bars()
    assert not window._progress_stack.isVisibleTo(window)
    assert not window._bottom_progress_bar.isVisibleTo(window)


def test_bottom_bar_mirrors_total_progress(window) -> None:
    window._begin_progress(window._recon_model)
    window._apply_progress("preprocess", "Preprocess", current=5, total=10)
    assert window._bottom_progress_bar.value() == window._total_progress_bar.value()


def test_status_and_transport_live_in_the_bottom_bar(window) -> None:
    for widget in (window._status_label, window._start_btn, window._spinner_stop):
        assert window._bottom_bar.isAncestorOf(widget)


def test_spinner_honours_a_larger_size(qapp) -> None:
    from deepreefmap.gui.core.spinner import SpinnerStopButton

    assert SpinnerStopButton(size=40).width() == 40
    assert SpinnerStopButton().width() == 26
