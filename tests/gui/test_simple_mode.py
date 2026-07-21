def test_starts_in_simple_mode(window):
    assert window._ui_mode == "simple"
    assert window._left_stack.currentIndex() == 1
    # The filled segment names the live mode, not the one you would switch to.
    assert window._mode_buttons["simple"].isChecked()
    assert not window._mode_buttons["advanced"].isChecked()


def test_toggle_switches_to_advanced_and_back(window):
    window._mode_buttons["advanced"].click()
    assert window._ui_mode == "advanced"
    assert window._left_stack.currentIndex() == 0
    assert window._mode_buttons["advanced"].isChecked()
    assert not window._mode_buttons["simple"].isChecked()
    assert window._sidebar_tabs.currentIndex() == window._TAB_RUN
    window._mode_buttons["simple"].click()
    assert window._ui_mode == "simple"
    assert window._left_stack.currentIndex() == 1


def test_mode_persists_across_windows(window, make_window):
    window._mode_buttons["advanced"].click()
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
    assert window._new_run_btn.isVisibleTo(window)
    window._mode_buttons["advanced"].click()
    assert window._start_btn.isVisibleTo(window)
    assert window._new_run_btn.isVisibleTo(window)
    assert not hasattr(window, "_past_runs_combo")


def test_simple_mode_creates_survey_db_under_root(window):
    from pathlib import Path

    from deepreefmap.survey.store import SURVEY_DB_NAME

    root = Path(window._out_root_input.text()).expanduser()
    assert (root / SURVEY_DB_NAME).exists()


def test_wizard_has_three_numbered_steps(window):
    assert list(window._simple_nav_buttons) == ["plan", "run", "analyse"]
    assert window._simple_nav_buttons["plan"].text() == "1. Plan"
    assert window._simple_nav_buttons["analyse"].text() == "3. Analyse"


def test_next_and_back_walk_the_steps(window):
    window._go_to_step("run")
    assert window._simple_stack.currentIndex() == 1
    window._wizard_back_buttons["run"].click()
    assert window._simple_stack.currentIndex() == 0
    window._wizard_next_buttons["plan"].click()
    assert window._simple_stack.currentIndex() == 1


def test_analyse_is_the_last_step(window):
    assert "analyse" not in window._wizard_next_buttons
    assert "plan" not in window._wizard_back_buttons


def test_navigation_locks_while_a_run_is_in_flight(window):
    window._set_wizard_navigation_enabled(False)
    assert not window._simple_nav_buttons["plan"].isEnabled()
    assert not window._wizard_next_buttons["plan"].isEnabled()
    window._set_wizard_navigation_enabled(True)
    assert window._simple_nav_buttons["plan"].isEnabled()
    assert window._wizard_next_buttons["plan"].isEnabled()
