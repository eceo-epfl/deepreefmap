"""The simple-mode run settings dialog borrows the real advanced form.

The invariant every test guards: the form goes back to the sidebar, whatever
way the dialog closes.
"""

import pytest

from deepreefmap.gui.simple.settings_dialog import RunSettingsDialog


@pytest.fixture
def simple_window(window, tmp_path):
    window._out_root_input.setText(str(tmp_path))
    window._set_ui_mode("simple")
    return window


def per_run_widgets(window):
    return [
        window._video_row_widget,
        window._range_row_widget,
        window._run_name_widget,
        window._transect_length_widget,
    ]


def test_dialog_borrows_the_form_and_hands_it_back(simple_window):
    window = simple_window
    dialog = RunSettingsDialog(window, window._setup_page, per_run_widgets(window))
    assert dialog.isAncestorOf(window._setup_page)
    assert not window._video_row_widget.isVisibleTo(dialog)

    dialog.restore_form()
    assert window._run_tab.isAncestorOf(window._setup_page)
    assert not window._video_row_widget.isHidden()
    assert not window._run_name_widget.isHidden()


def test_every_exit_path_restores_the_form(simple_window):
    window = simple_window
    for close in (lambda d: d.accept(), lambda d: d.reject(), lambda d: d.close()):
        dialog = RunSettingsDialog(window, window._setup_page, per_run_widgets(window))
        close(dialog)
        assert window._run_tab.isAncestorOf(window._setup_page)


def test_restoring_twice_is_harmless(simple_window):
    window = simple_window
    dialog = RunSettingsDialog(window, window._setup_page, per_run_widgets(window))
    dialog.restore_form()
    dialog.restore_form()
    assert window._run_tab.isAncestorOf(window._setup_page)


def test_settings_edited_in_the_dialog_persist(simple_window, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "deepreefmap.survey.preset.survey_preset_path", lambda: tmp_path / "p.yaml"
    )
    window = simple_window
    monkeypatch.setattr(RunSettingsDialog, "exec", lambda self: window._grid_bins_spin.setValue(1234))
    window._on_edit_run_settings()
    assert window._run_tab.isAncestorOf(window._setup_page)
    assert window._survey_preset["grid_bins"] == 1234
    assert window._collect_run_settings()["grid_bins"] == 1234


def test_reset_defaults_button_restores_the_form_values(simple_window):
    window = simple_window
    window._grid_bins_spin.setValue(1234)
    window._reset_form_defaults()
    assert window._grid_bins_spin.value() == window._form_defaults["_grid_bins_spin"]


def test_mode_switch_is_blocked_while_the_dialog_holds_the_form(simple_window):
    window = simple_window
    window._settings_dialog_open = True
    window._request_ui_mode("advanced")
    assert window._ui_mode == "simple"
    assert "Close the run settings" in window._status_label.text()
    window._settings_dialog_open = False
    window._request_ui_mode("advanced")
    assert window._ui_mode == "advanced"


def test_settings_cannot_be_edited_mid_batch(simple_window):
    window = simple_window
    window._survey_worker_running = True
    window._on_edit_run_settings()
    assert window._run_tab.isAncestorOf(window._setup_page)
    assert "Wait for the current batch" in window._status_label.text()
