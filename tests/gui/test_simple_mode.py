def test_starts_in_simple_mode(window):
    assert window._ui_mode == "simple"
    assert window._left_stack.currentIndex() == 1
    assert window._mode_toggle_btn.text() == "Advanced"


def test_toggle_switches_to_advanced_and_back(window):
    window._mode_toggle_btn.click()
    assert window._ui_mode == "advanced"
    assert window._left_stack.currentIndex() == 0
    assert window._mode_toggle_btn.text() == "Simple"
    assert window._sidebar_tabs.currentIndex() == window._TAB_RUN
    window._mode_toggle_btn.click()
    assert window._ui_mode == "simple"
    assert window._left_stack.currentIndex() == 1


def test_mode_persists_across_windows(window, make_window):
    window._mode_toggle_btn.click()
    assert window._ui_mode == "advanced"
    other = make_window()
    assert other._ui_mode == "advanced"
    assert other._left_stack.currentIndex() == 0


def test_simple_nav_switches_sections(window):
    window._set_simple_section("run")
    assert window._simple_stack.currentIndex() == 1
    window._set_simple_section("analyse")
    assert window._simple_stack.currentIndex() == 2
    window._simple_nav_buttons["plan"].click()
    assert window._simple_stack.currentIndex() == 0


def test_app_mode_targets_sections_in_simple(window, monkeypatch):
    monkeypatch.setattr(window._viewer, "_ensure_plotter", lambda: None)
    window._set_app_mode("RUNNING")
    assert window._simple_stack.currentIndex() == 1
    window._set_app_mode("VIEWING")
    assert window._simple_stack.currentIndex() == 2
    window._set_app_mode("SETUP")
    assert window._simple_stack.currentIndex() == 2


def test_viewer_pane_follows_app_mode_in_simple(window, monkeypatch):
    monkeypatch.setattr(window._viewer, "_ensure_plotter", lambda: None)
    assert not window._viewer.isVisibleTo(window)
    window._set_app_mode("RUNNING")
    assert window._viewer.isVisibleTo(window)
    window._set_app_mode("SETUP")
    assert not window._viewer.isVisibleTo(window)


def test_advanced_run_controls_hidden_in_simple(window):
    assert not window._start_btn.isVisibleTo(window)
    assert not window._new_run_btn.isVisibleTo(window)
    window._mode_toggle_btn.click()
    assert window._start_btn.isVisibleTo(window)
    assert window._new_run_btn.isVisibleTo(window)


def test_simple_mode_creates_survey_db_under_root(window):
    from pathlib import Path

    from deepreefmap.survey.store import SURVEY_DB_NAME

    root = Path(window._out_root_input.text()).expanduser()
    assert (root / SURVEY_DB_NAME).exists()
