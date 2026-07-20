import pytest

from PySide6.QtCore import QSettings


@pytest.fixture(autouse=True)
def _sandbox_out_root(tmp_path):
    settings = QSettings("ECEO", "deepreefmap")
    settings.setValue("output_root_dir", str(tmp_path))
    yield
    settings.remove("output_root_dir")


def test_starts_in_advanced_mode(window):
    tabs = window._sidebar_tabs
    assert window._ui_mode == "advanced"
    assert tabs.isTabVisible(window._TAB_RUN)
    assert tabs.isTabVisible(window._TAB_MODELS)
    assert not tabs.isTabVisible(window._TAB_PLAN)


def test_toggle_switches_to_survey_tabs(window):
    tabs = window._sidebar_tabs
    window._mode_toggle_btn.setChecked(True)
    assert window._ui_mode == "simple"
    assert tabs.isTabVisible(window._TAB_PLAN)
    assert tabs.isTabVisible(window._TAB_SURVEY)
    assert not tabs.isTabVisible(window._TAB_RUN)
    assert not tabs.isTabVisible(window._TAB_MODELS)
    assert tabs.currentIndex() == window._survey_home_tab()
    window._mode_toggle_btn.setChecked(False)
    assert window._ui_mode == "advanced"
    assert tabs.currentIndex() == window._TAB_RUN


def test_mode_persists_across_windows(make_window):
    first = make_window()
    first._mode_toggle_btn.setChecked(True)
    second = make_window()
    assert second._ui_mode == "simple"
    assert second._mode_toggle_btn.isChecked()


def test_app_mode_targets_survey_home_in_simple(window):
    window._set_ui_mode("simple")
    window._set_app_mode("SETUP")
    assert window._sidebar_tabs.currentIndex() == window._survey_home_tab()
    window._set_app_mode("VIEWING")
    assert window._sidebar_tabs.currentIndex() == window._TAB_RESULTS
