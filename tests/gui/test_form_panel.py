"""Run form behaviour: LoGeR options, defaults, backend panel visibility."""

from __future__ import annotations

import pytest


def test_loger_options_collected_from_form(window) -> None:

    assert window._collect_loger_options("scsfmlearner") is None

    window._loger_window_spin.setValue(16)
    window._loger_overlap_spin.setValue(2)
    window._loger_model_path_input.setText("")
    assert window._collect_loger_options("loger") == {
        "window_size": 16,
        "overlap_size": 2,
        "model_path": None,
    }

    window._loger_model_path_input.setText("/tmp/custom.pt")
    assert window._collect_loger_options("loger_star")["model_path"] == "/tmp/custom.pt"


def test_form_defaults_to_vit_b_and_loger_star(qapp) -> None:
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow
    from deepreefmap.mapping.registry import loger_available

    window = DeepReefMapWindow(load_classes(), None)
    assert window._seg_combo.currentText() == "coralscapes-vit-b-dpt"
    assert window._map_combo.currentText() == (
        "loger_star" if loger_available() else "scsfmlearner"
    )
    # vit-b native (768,1376) → processing (1376,768); the Native preset feeds it unchanged.
    assert window._native_resolution == (1376, 768)


def test_loger_panel_visibility_follows_backend(window) -> None:

    window._map_combo.setCurrentText("scsfmlearner")
    assert window._loger_panel.isHidden()

    window._map_combo.setCurrentText("loger")
    assert not window._loger_panel.isHidden()


def test_time_edit_parses_clamps_and_reverts(qapp) -> None:
    from deepreefmap.gui.form.time_edit import TimeSecondsEdit

    edit = TimeSecondsEdit()
    edit.setText("12.5")
    edit._commit()
    assert edit.value() == 12.5

    edit.setText("nonsense")
    edit._commit()
    assert edit.value() == 12.5
    assert edit.text() == "12.50"

    edit.setText("-3")
    edit._commit()
    assert edit.value() == 0.0

    edit.setMaximum(30.0)
    edit.setText("99")
    edit._commit()
    assert edit.value() == 30.0
    assert edit.text() == "30.00"


def test_begin_and_end_snap_together(window) -> None:
    window._end_spin.setValue(50.0)
    window._begin_spin.setValue(80.0)
    assert window._begin_spin.value() == 50.0

    window._begin_spin.setValue(20.0)
    window._end_spin.setValue(10.0)
    assert window._end_spin.value() == 20.0

