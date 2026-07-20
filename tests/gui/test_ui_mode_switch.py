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


def test_crop_width_zero_means_disabled(window):
    window._crop_width.setValue(0.0)
    assert window._collect_preset_from_form()["transect_crop_width"] is None
    window._populate_form_from_preset(
        {**window._collect_preset_from_form(), "transect_crop_width": 1.5}
    )
    assert window._crop_width.value() == 1.5
