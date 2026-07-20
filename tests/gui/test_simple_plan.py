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


def enter_quick(window, text):
    window._tr_quick_input.setText(text)
    window._on_quick_entry()


def test_save_transect_via_quick_entry(plan_window):
    w = plan_window
    w._tr_name_input.setText("T1")
    enter_quick(w, "-17.5 177.1")
    enter_quick(w, "-17.5005, 177.1005")
    w._tr_length.setValue(50.0)
    w._on_transect_save()
    transects = w._survey_store().list_transects()
    assert len(transects) == 1
    assert transects[0].name == "T1"
    assert transects[0].start_lat == -17.5
    assert transects[0].end_lon == 177.1005
    assert transects[0].length_m == 50.0
    assert w._transect_list.count() == 1
    assert "Geodesic" in w._tr_geodesic_label.text()


def test_quick_entry_rejects_garbage(plan_window):
    enter_quick(plan_window, "junk")
    assert "Expected" in plan_window._status_label.text()


def test_duplicate_name_reports_and_keeps_one(plan_window):
    w = plan_window
    w._survey_store().add_transect(make_transect())
    w._tr_name_input.setText("T1")
    enter_quick(w, "-17.6 177.2")
    enter_quick(w, "-17.6005 177.2005")
    w._on_transect_save()
    assert "already exists" in w._status_label.text()
    assert len(w._survey_store().list_transects()) == 1


def test_edit_selected_transect_updates_row(plan_window):
    w = plan_window
    w._survey_store().add_transect(make_transect())
    w._refresh_transect_list()
    w._transect_list.setCurrentRow(0)
    assert w._tr_name_input.text() == "T1"
    w._tr_end_lat.setText("-17.502")
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
