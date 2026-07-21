from deepreefmap.survey.preset import PRESET_KEYS


def test_preset_covers_every_settings_field(window):
    assert set(window._collect_preset_from_form()) == PRESET_KEYS


def test_reset_restores_defaults(window):
    window._batch_size_spin.setValue(window._batch_size_spin.value() + 1)
    window._loger_model_path_input.setText("/tmp/custom.pt")
    window._reset_form_defaults()
    assert window._batch_size_spin.value() == window._form_defaults["_batch_size_spin"]
    assert window._loger_model_path_input.text() == ""


def test_collect_and_populate_round_trip(window):
    window._fps_spin.setValue(3)
    window._seg_combo.setCurrentText("segformer-b2")
    window._map_combo.setCurrentText("scsfmlearner")
    window._grid_bins_spin.setValue(1234)
    window._loger_model_path_input.setText("/tmp/custom.pt")
    window._require_gravity_check.setChecked(True)
    preset = window._collect_preset_from_form()
    window._reset_form_defaults()
    window._populate_form_from_preset(preset)
    assert window._collect_preset_from_form() == preset
    assert window._grid_bins_spin.value() == 1234


def test_advanced_tweaks_survive_the_round_trip(window, tmp_path, monkeypatch):
    """Scenario: someone widens the grid bins in advanced, then goes back to simple.

    Expected behaviour: simple mode keeps the value instead of resetting it.
    """
    monkeypatch.setattr(
        "deepreefmap.survey.preset.survey_preset_path", lambda: tmp_path / "p.yaml"
    )
    window._mode_buttons["advanced"].click()
    window._grid_bins_spin.setValue(1234)
    window._mode_buttons["simple"].click()
    assert window._ui_mode == "simple"
    assert window._survey_preset["grid_bins"] == 1234


def test_native_processing_size_is_not_pinned(window):
    """A non-Custom resolution preset must not freeze the model's native size."""
    window._resolution_preset_combo.setCurrentText("Native")
    preset = window._collect_preset_from_form()
    assert preset["processing_width"] is None
    assert preset["processing_height"] is None


def test_custom_processing_size_round_trips(window):
    window._resolution_preset_combo.setCurrentText("Custom")
    window._proc_width_spin.setValue(800)
    window._proc_height_spin.setValue(600)
    preset = window._collect_preset_from_form()
    assert (preset["processing_width"], preset["processing_height"]) == (800, 600)
    window._reset_form_defaults()
    window._populate_form_from_preset(preset)
    assert (window._proc_width_spin.value(), window._proc_height_spin.value()) == (800, 600)


def test_preview_defaults_off_and_gates_canvas(window):
    viewer = window._viewer
    assert not window._preview_toggle_btn.isChecked()
    assert viewer._canvas_stack.currentWidget() is viewer._placeholder_container
    viewer._reveal_canvas()
    assert viewer._canvas_wanted
    assert viewer._canvas_stack.currentWidget() is viewer._placeholder_container


def test_allowing_preview_reveals_pending_scene(window, monkeypatch):
    viewer = window._viewer
    monkeypatch.setattr(viewer, "_ensure_plotter", lambda: None)
    viewer._reveal_canvas()
    window._preview_toggle_btn.setChecked(True)
    assert viewer._canvas_stack.currentWidget() is viewer._canvas_container
    window._preview_toggle_btn.setChecked(False)
    assert viewer._canvas_stack.currentWidget() is viewer._placeholder_container
    assert viewer._canvas_wanted


def test_preview_setting_persists(window, make_window):
    window._preview_toggle_btn.setChecked(True)
    other = make_window()
    assert other._preview_toggle_btn.isChecked()


def test_viewing_forces_preview_on(window, monkeypatch):
    monkeypatch.setattr(window._viewer, "_ensure_plotter", lambda: None)
    assert not window._preview_toggle_btn.isChecked()
    window._set_app_mode("VIEWING")
    assert window._preview_toggle_btn.isChecked()


def test_entering_advanced_expands_the_preset(window):
    window._mode_buttons["advanced"].click()
    preset = window._survey_preset
    assert preset is not None
    assert window._fps_spin.value() == preset["fps"]
    assert window._seg_combo.currentText() == preset["segmentation_name"]
    assert window._map_combo.currentText() == preset["mapping_name"]


def test_returning_to_simple_persists_tweaks(window, tmp_path, monkeypatch):
    from deepreefmap.survey.preset import parse_preset

    target = tmp_path / "survey_preset.yaml"
    monkeypatch.setattr("deepreefmap.survey.preset.survey_preset_path", lambda: target)
    window._mode_buttons["advanced"].click()
    window._fps_spin.setValue(3)
    window._seg_combo.setCurrentText("segformer-b2")
    window._mode_buttons["simple"].click()
    assert window._ui_mode == "simple"
    assert window._survey_preset["fps"] == 3
    assert window._survey_preset["segmentation_name"] == "segformer-b2"
    assert parse_preset(target.read_text())["fps"] == 3


def test_crop_width_zero_means_disabled(window):
    window._crop_width.setValue(0.0)
    assert window._collect_preset_from_form()["transect_crop_width"] is None
    window._populate_form_from_preset(
        {**window._collect_preset_from_form(), "transect_crop_width": 1.5}
    )
    assert window._crop_width.value() == 1.5
