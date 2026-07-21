import pytest

from deepreefmap.survey.models import Transect, TransectPass, VideoAsset
from deepreefmap.survey.models.exporters import save_transects_csv


@pytest.fixture
def plan_window(window, tmp_path):
    window._out_root_input.setText(str(tmp_path))
    window._set_ui_mode("simple")
    return window


def make_transect(name="T1"):
    return Transect(
        name=name,
        start_lat=-17.5,
        start_lon=177.1,
        end_lat=-17.5005,
        end_lon=177.1005,
        length_m=50.0,
    )


def type_coord(window, which, text):
    """Type a coordinate the way a field worker pastes one off a GPS."""
    edit = window._tr_start_coord if which == "start" else window._tr_end_coord
    edit.setText(text)
    window._on_coords_edited()


def test_transect_autosaves_once_complete(plan_window):
    w = plan_window
    w._tr_name_input.setText("T1")
    type_coord(w, "start", "-17.5 177.1")
    assert w._survey_store().list_transects() == []
    type_coord(w, "end", "-17.5005, 177.1005")
    transects = w._survey_store().list_transects()
    assert len(transects) == 1
    assert transects[0].name == "T1"
    assert transects[0].start_lat == -17.5
    assert transects[0].end_lon == 177.1005
    w._tr_length.setValue(50.0)
    w._maybe_autosave()
    assert w._survey_store().list_transects()[0].length_m == 50.0
    assert w._transect_list.count() == 1
    assert "Geodesic" in w._tr_geodesic_label.text()


def test_draft_row_tracks_typing_before_save(plan_window):
    from PySide6.QtCore import Qt

    w = plan_window
    w._refresh_transect_list()
    assert w._transect_list.count() == 0
    w._tr_name_input.setText("Nor")
    assert w._transect_list.count() == 1
    item = w._transect_list.item(0)
    assert item.data(Qt.ItemDataRole.UserRole) == "draft"
    assert "Nor" in item.text()
    assert w._transect_list.currentRow() == 0
    w._tr_name_input.setText("North reef")
    assert "North reef" in w._transect_list.item(0).text()
    assert w._survey_store().list_transects() == []


def test_out_of_range_coordinate_is_reported(plan_window):
    """The range check that used to live on the Quick field still applies."""
    w = plan_window
    w._tr_name_input.setText("T1")
    type_coord(w, "start", "-17.5, 177.1")
    type_coord(w, "end", "95.0, 177.1")
    assert w._survey_store().list_transects() == []
    with pytest.raises(ValueError, match="out of range"):
        w._form_coordinates()


def test_garbage_coordinate_never_saves(plan_window):
    w = plan_window
    w._tr_name_input.setText("T1")
    type_coord(w, "start", "junk")
    type_coord(w, "end", "-17.5, 177.1")
    assert w._survey_store().list_transects() == []


def test_map_buttons_set_endpoints_one_shot(plan_window):
    w = plan_window
    w._map_start_btn.setChecked(True)
    w._plan_map.map_clicked.emit(-17.5, 177.1)
    assert w._tr_start_coord.text() == "-17.500000, 177.100000"
    assert not w._map_start_btn.isChecked()
    w._map_end_btn.setChecked(True)
    w._plan_map.map_clicked.emit(-17.5005, 177.1005)
    assert w._tr_end_coord.text() == "-17.500500, 177.100500"
    assert not w._map_end_btn.isChecked()


def test_map_buttons_are_mutually_exclusive(plan_window):
    w = plan_window
    w._map_start_btn.setChecked(True)
    w._map_end_btn.setChecked(True)
    assert not w._map_start_btn.isChecked()
    w._map_start_btn.setChecked(True)
    assert not w._map_end_btn.isChecked()


def test_copy_endpoint_puts_latlon_on_clipboard(plan_window, monkeypatch):
    # Stubbed clipboard: the real X11 selection is shared with the desktop and
    # races with clipboard managers.
    captured = []

    class _Clipboard:
        def setText(self, text):
            captured.append(text)

    class _App:
        @staticmethod
        def clipboard():
            return _Clipboard()

    monkeypatch.setattr("deepreefmap.gui.simple.plan.QGuiApplication", _App)
    w = plan_window
    w._tr_start_coord.setText("-17.500000, 177.100000")
    w._copy_endpoint("start")
    assert captured == ["-17.500000, 177.100000"]
    w._copy_endpoint("end")
    assert "No end point to copy" in w._status_label.text()


def test_map_click_without_armed_button_is_ignored(plan_window):
    w = plan_window
    w._plan_map.map_clicked.emit(-17.5, 177.1)
    assert w._tr_start_coord.text() == ""
    assert w._tr_end_coord.text() == ""


def test_draft_line_appears_once_both_endpoints_set(plan_window):
    w = plan_window
    w._map_start_btn.setChecked(True)
    w._plan_map.map_clicked.emit(-17.5, 177.1)
    assert not any(t.id == "draft" for t in w._plan_map._transects)
    w._map_end_btn.setChecked(True)
    w._plan_map.map_clicked.emit(-17.5005, 177.1005)
    assert any(t.id == "draft" for t in w._plan_map._transects)
    w._tr_name_input.setText("T1")
    w._on_transect_save()
    assert not any(t.id == "draft" for t in w._plan_map._transects)
    assert len(w._plan_map._transects) == 1


def test_duplicate_name_reports_and_keeps_one(plan_window):
    w = plan_window
    w._survey_store().add_transect(make_transect())
    w._tr_name_input.setText("T1")
    type_coord(w, "start", "-17.6 177.2")
    type_coord(w, "end", "-17.6005 177.2005")
    w._on_transect_save()
    assert "already exists" in w._status_label.text()
    assert len(w._survey_store().list_transects()) == 1


def test_edit_selected_transect_updates_row(plan_window):
    w = plan_window
    w._survey_store().add_transect(make_transect())
    w._refresh_transect_list()
    w._transect_list.setCurrentRow(0)
    assert w._tr_name_input.text() == "T1"
    w._tr_end_coord.setText("-17.502, 177.1005")
    w._on_transect_save()
    stored = w._survey_store().list_transects()
    assert len(stored) == 1
    assert stored[0].end_lat == -17.502


def test_delete_with_passes_is_blocked(plan_window):
    w = plan_window
    store = w._survey_store()
    transect = make_transect()
    video = VideoAsset(file_name="a.mp4", path="/a.mp4", hash="cd" * 16)
    store.add_transect(transect)
    store.upsert_video(video)
    store.add_pass(
        TransectPass(transect_id=transect.id, video_id=video.id, begin_s=0.0, end_s=30.0)
    )
    w._refresh_transect_list()
    w._transect_list.setCurrentRow(0)
    w._on_transect_delete()
    assert "cannot be deleted" in w._status_label.text()
    assert len(store.list_transects()) == 1


def test_import_csv_skips_existing(plan_window, tmp_path, monkeypatch):
    w = plan_window
    existing = make_transect()
    w._survey_store().add_transect(existing)
    csv_path = tmp_path / "in.csv"
    save_transects_csv(csv_path, [existing, make_transect("T2")])
    monkeypatch.setattr(
        "deepreefmap.gui.simple.plan.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(csv_path), "")),
    )
    w._on_transects_import()
    assert "Imported 1" in w._status_label.text()
    assert "Skipped 1" in w._status_label.text()
    assert [t.name for t in w._survey_store().list_transects()] == ["T1", "T2"]


def test_export_csv_round_trip(plan_window, tmp_path, monkeypatch):
    w = plan_window
    w._survey_store().add_transect(make_transect())
    out_path = tmp_path / "out.csv"
    monkeypatch.setattr(
        "deepreefmap.gui.simple.plan.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_path), "")),
    )
    w._on_transects_export()
    assert "T1" in out_path.read_text()


def test_pick_both_walks_start_then_end(plan_window):
    """Scenario: a new transect drawn entirely on the map.

    Expected behaviour: one button, two clicks, then it disarms itself.
    """
    w = plan_window
    w._pick_both_btn.setChecked(True)
    assert w._plan_map._pick_mode
    w._plan_map.map_clicked.emit(-17.5, 177.1)
    assert w._tr_start_coord.text() == "-17.500000, 177.100000"
    assert w._tr_end_coord.text() == ""
    assert "end" in w._status_label.text().lower()
    w._plan_map.map_clicked.emit(-17.5005, 177.1005)
    assert w._tr_end_coord.text() == "-17.500500, 177.100500"
    assert not w._pick_both_btn.isChecked()
    assert not w._plan_map._pick_mode


def test_single_endpoint_pick_disarms_pick_both(plan_window):
    w = plan_window
    w._pick_both_btn.setChecked(True)
    w._map_end_btn.setChecked(True)
    assert not w._pick_both_btn.isChecked()
    assert w._pick_stage is None
    w._plan_map.map_clicked.emit(-17.6, 177.2)
    assert w._tr_end_coord.text() == "-17.600000, 177.200000"
    assert w._tr_start_coord.text() == ""


def test_notes_round_trip_through_the_store(plan_window):
    w = plan_window
    w._tr_name_input.setText("T1")
    type_coord(w, "start", "-17.5, 177.1")
    type_coord(w, "end", "-17.5005, 177.1005")
    w._tr_description.setPlainText("tape run W→E\nviz ~12 m")
    w._maybe_autosave()
    stored = w._survey_store().list_transects()[0]
    assert stored.description == "tape run W→E\nviz ~12 m"
    w._on_transect_new()
    w._refresh_transect_list()
    w._transect_list.setCurrentRow(0)
    assert w._tr_description.toPlainText() == "tape run W→E\nviz ~12 m"


def test_selecting_a_transect_filters_the_browser(plan_window):
    w = plan_window
    transect = make_transect()
    w._survey_store().add_transect(transect)
    w._refresh_transect_list()
    w._transect_list.setCurrentRow(0)
    assert w._data_facet == "transects"
    assert w._data_selected_key == ("transect", str(transect.id))
