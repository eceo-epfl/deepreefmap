import pytest

from deepreefmap.gui.simple.batch import _COL_STATUS, _COL_TRANSECT
from deepreefmap.survey.models import Transect


@pytest.fixture
def batch_window(window, tmp_path, monkeypatch):
    window._out_root_input.setText(str(tmp_path))
    window._set_ui_mode("simple")
    window._survey_store().add_transect(
        Transect(
            name="T1",
            start_lat=-17.5,
            start_lon=177.1,
            end_lat=-17.5005,
            end_lon=177.1005,
            length_m=50.0,
        )
    )
    window._refresh_survey_batch_tab()
    monkeypatch.setattr(window, "_survey_missing_models", lambda: [], raising=False)
    return window


def add_video(window, tmp_path, monkeypatch, name="GX010001.MP4"):
    path = tmp_path / name
    path.write_bytes(name.encode() * 4096)
    monkeypatch.setattr(
        "deepreefmap.gui.simple.batch._probe_video", lambda _path: (60.0, 30.0)
    )
    monkeypatch.setattr(
        "deepreefmap.gui.simple.batch.QFileDialog.getOpenFileNames",
        staticmethod(lambda *a, **k: ([str(path)], "")),
    )
    window._on_survey_add_videos()
    return path


def assign_transect(window, row_index):
    combo = window._survey_pass_table.cellWidget(row_index, _COL_TRANSECT)
    combo.setCurrentIndex(1)


def add_second_transect(window, name="T2"):
    window._survey_store().add_transect(
        Transect(
            name=name,
            start_lat=-17.6,
            start_lon=177.2,
            end_lat=-17.6005,
            end_lon=177.2005,
            length_m=50.0,
        )
    )
    window._refresh_survey_batch_tab()


def test_add_video_stays_unassigned_between_transects(batch_window, tmp_path, monkeypatch):
    add_second_transect(batch_window)
    add_video(batch_window, tmp_path, monkeypatch)
    assert batch_window._survey_pass_table.rowCount() == 1
    assert batch_window._survey_rows[0].transect_id is None
    assert batch_window._survey_rows[0].end_s == 60.0
    assert not batch_window._survey_start_btn.isEnabled()
    assert batch_window._survey_start_btn.text() == "Assign transects first (1 to do)"
    combo = batch_window._survey_pass_table.cellWidget(0, _COL_TRANSECT)
    assert combo.currentText() == "Not assigned yet"
    assert combo.styleSheet() != ""
    assert batch_window._survey_store().list_passes() == []


def test_add_video_preselects_the_only_transect(batch_window, tmp_path, monkeypatch):
    add_video(batch_window, tmp_path, monkeypatch)
    assert batch_window._survey_rows[0].transect_id is not None
    assert len(batch_window._survey_store().list_passes()) == 1
    combo = batch_window._survey_pass_table.cellWidget(0, _COL_TRANSECT)
    assert combo.currentText() == "T1"
    assert combo.styleSheet() == ""
    assert batch_window._survey_start_btn.isEnabled()


def test_assigning_transect_persists_pass(batch_window, tmp_path, monkeypatch):
    add_second_transect(batch_window)
    add_video(batch_window, tmp_path, monkeypatch)
    assert batch_window._survey_store().list_passes() == []
    assign_transect(batch_window, 0)
    passes = batch_window._survey_store().list_passes()
    assert len(passes) == 1
    assert passes[0].begin_s == 0.0
    assert passes[0].end_s == 60.0
    assert passes[0].direction == "forward"
    assert passes[0].batch_id is not None
    assert batch_window._survey_start_btn.isEnabled()
    assert "1" in batch_window._survey_start_btn.text()


def test_split_pass_duplicates_row(batch_window, tmp_path, monkeypatch):
    add_video(batch_window, tmp_path, monkeypatch)
    assign_transect(batch_window, 0)
    batch_window._survey_pass_table.setCurrentCell(0, 0)
    batch_window._on_survey_split_pass()
    assert batch_window._survey_pass_table.rowCount() == 2
    assert len(batch_window._survey_store().list_passes()) == 2


def test_run_batch_records_success_and_links_manifest(
    batch_window, tmp_path, monkeypatch, qapp
):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("deepreefmap.pipeline.orchestrator.run_reconstruction", fake_run)
    add_video(batch_window, tmp_path, monkeypatch)
    assign_transect(batch_window, 0)
    batch_window._on_survey_start()
    batch_window._pipeline_thread.join(timeout=10)
    qapp.processEvents()

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["viewer"] is batch_window._viewer
    assert kwargs["fps"] == 5
    assert kwargs["begin_s"] == 0.0
    assert kwargs["end_s"] == 60.0
    assert kwargs["transect_length"] == 50.0
    assert kwargs["output_dir"].parent == tmp_path
    survey = kwargs["manifest_extra"]["survey"]
    assert survey["transect"]["name"] == "T1"
    assert survey["pass"]["direction"] == "forward"
    runs = batch_window._survey_store().list_runs()
    assert [r.status for r in runs] == ["succeeded"]
    assert not batch_window._survey_start_btn.isEnabled()
    assert "0" in batch_window._survey_start_btn.text()
    assert batch_window._survey_pass_table.item(0, _COL_STATUS).text() == "succeeded"


def test_batch_lands_on_analysis_when_done(batch_window, tmp_path, monkeypatch, qapp):
    monkeypatch.setattr(
        "deepreefmap.pipeline.orchestrator.run_reconstruction", lambda **k: None
    )
    add_video(batch_window, tmp_path, monkeypatch)
    assign_transect(batch_window, 0)
    batch_window._on_survey_start()
    assert batch_window._app_mode == "RUNNING"
    assert batch_window._simple_stack.currentIndex() == 1
    batch_window._pipeline_thread.join(timeout=10)
    qapp.processEvents()
    assert batch_window._app_mode == "SETUP"
    assert batch_window._simple_stack.currentIndex() == 2


def test_failed_run_keeps_pass_remaining(batch_window, tmp_path, monkeypatch, qapp):
    def broken_run(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("deepreefmap.pipeline.orchestrator.run_reconstruction", broken_run)
    add_video(batch_window, tmp_path, monkeypatch)
    assign_transect(batch_window, 0)
    batch_window._on_survey_start()
    batch_window._pipeline_thread.join(timeout=10)
    qapp.processEvents()

    runs = batch_window._survey_store().list_runs()
    assert [r.status for r in runs] == ["failed"]
    assert "boom" in runs[0].error
    assert batch_window._survey_start_btn.isEnabled()
    assert "1" in batch_window._survey_start_btn.text()


def test_remove_pass_with_runs_is_blocked(batch_window, tmp_path, monkeypatch, qapp):
    monkeypatch.setattr(
        "deepreefmap.pipeline.orchestrator.run_reconstruction", lambda **k: None
    )
    add_video(batch_window, tmp_path, monkeypatch)
    assign_transect(batch_window, 0)
    batch_window._on_survey_start()
    batch_window._pipeline_thread.join(timeout=10)
    qapp.processEvents()
    batch_window._survey_pass_table.setCurrentCell(0, 0)
    batch_window._on_survey_remove_pass()
    assert "cannot be removed" in batch_window._status_label.text()
    assert batch_window._survey_pass_table.rowCount() == 1


def test_refresh_restores_batch_from_store(batch_window, tmp_path, monkeypatch):
    add_video(batch_window, tmp_path, monkeypatch)
    assign_transect(batch_window, 0)
    batch_name = batch_window._survey_batch_name.text()
    batch_window._survey_batch = None
    batch_window._survey_rows = []
    batch_window._survey_pass_table.setRowCount(0)
    batch_window._refresh_survey_batch_tab()
    assert batch_window._survey_pass_table.rowCount() == 1
    assert batch_window._survey_batch_name.text() == batch_name
    assert batch_window._survey_rows[0].transect_id is not None
