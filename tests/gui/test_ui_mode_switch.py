from deepreefmap.survey.preset import PRESET_KEYS


def test_fresh_form_is_within_simple_bounds(window):
    assert window._form_outside_simple_bounds() == []


def test_perturbed_fields_are_flagged(window):
    window._grid_bins_spin.setValue(window._grid_bins_spin.value() + 10)
    window._require_gravity_check.setChecked(not window._require_gravity_check.isChecked())
    offending = window._form_outside_simple_bounds()
    assert "grid bins" in offending
    assert "require gravity telemetry" in offending


def test_reset_restores_defaults(window):
    window._batch_size_spin.setValue(window._batch_size_spin.value() + 1)
    window._loger_model_path_input.setText("/tmp/custom.pt")
    assert window._form_outside_simple_bounds()
    window._reset_non_preset_fields()
    assert window._form_outside_simple_bounds() == []


def test_collect_and_populate_round_trip(window):
    preset = {
        "fps": 3,
        "segmentation_name": "segformer-b2",
        "mapping_name": "scsfmlearner",
        "camera_profile_name": window._profile_combo.itemText(0),
        "transect_crop_width": None,
        "enable_tsdf": True,
        "skip_segmentation": True,
    }
    assert set(preset) == PRESET_KEYS
    window._populate_form_from_preset(preset)
    assert window._collect_preset_from_form() == preset


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
    window._mode_toggle_btn.click()
    preset = window._survey_preset
    assert preset is not None
    assert window._fps_spin.value() == preset["fps"]
    assert window._seg_combo.currentText() == preset["segmentation_name"]
    assert window._map_combo.currentText() == preset["mapping_name"]


def test_returning_to_simple_persists_tweaks(window, tmp_path, monkeypatch):
    from deepreefmap.survey.preset import parse_preset

    target = tmp_path / "survey_preset.yaml"
    monkeypatch.setattr("deepreefmap.survey.preset.survey_preset_path", lambda: target)
    window._mode_toggle_btn.click()
    window._fps_spin.setValue(3)
    window._seg_combo.setCurrentText("segformer-b2")
    window._mode_toggle_btn.click()
    assert window._ui_mode == "simple"
    assert window._survey_preset["fps"] == 3
    assert window._survey_preset["segmentation_name"] == "segformer-b2"
    assert parse_preset(target.read_text())["fps"] == 3


def test_out_of_bounds_state_can_keep_you_in_advanced(window, monkeypatch):
    window._mode_toggle_btn.click()
    window._grid_bins_spin.setValue(window._grid_bins_spin.value() + 5)
    monkeypatch.setattr(window, "_confirm_reset_for_simple", lambda offending: False)
    window._mode_toggle_btn.click()
    assert window._ui_mode == "advanced"


def test_out_of_bounds_state_resets_on_confirm(window, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "deepreefmap.survey.preset.survey_preset_path", lambda: tmp_path / "p.yaml"
    )
    window._mode_toggle_btn.click()
    default = window._grid_bins_spin.value()
    window._grid_bins_spin.setValue(default + 5)
    monkeypatch.setattr(window, "_confirm_reset_for_simple", lambda offending: True)
    window._mode_toggle_btn.click()
    assert window._ui_mode == "simple"
    assert window._grid_bins_spin.value() == default


def test_crop_width_zero_means_disabled(window):
    window._crop_width.setValue(0.0)
    assert window._collect_preset_from_form()["transect_crop_width"] is None
    window._populate_form_from_preset(
        {**window._collect_preset_from_form(), "transect_crop_width": 1.5}
    )
    assert window._crop_width.value() == 1.5
