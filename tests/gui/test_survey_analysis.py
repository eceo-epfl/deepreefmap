import json

import pytest

from deepreefmap.survey.models import RunRecord, Transect, TransectPass, VideoAsset


@pytest.fixture
def analysis_window(window, tmp_path):
    window._out_root_input.setText(str(tmp_path))
    window._set_ui_mode("simple")
    store = window._survey_store()
    transect = Transect(
        name="T1",
        start_lat=-17.5,
        start_lon=177.1,
        end_lat=-17.5005,
        end_lon=177.1005,
    )
    store.add_transect(transect)
    video = store.upsert_video(VideoAsset(file_name="a.mp4", path="/a.mp4", hash="ab" * 16))
    pass_ = TransectPass(transect_id=transect.id, video_id=video.id, begin_s=0.0, end_s=60.0)
    store.add_pass(pass_)
    run = RunRecord(pass_id=pass_.id, run_dir_name="t1__p01", status="succeeded")
    store.add_run(run)
    cls = window._classes_config.classes[0]
    run_dir = tmp_path / "t1__p01"
    run_dir.mkdir()
    (run_dir / "benthic_cover.json").write_text(json.dumps({
        "classes": {str(cls.id): {"name": cls.name, "count": 30.0, "fraction": 0.3}},
        "denominator": 100.0,
    }))
    window._refresh_survey_analysis()
    return window


def test_analysis_populates_chart_table_and_runs(analysis_window):
    w = analysis_window
    assert w._analysis_transect_combo.count() == 1
    assert len(w._analysis_covers) == 1
    assert w._analysis_stats_table.rowCount() >= 1
    assert w._analysis_stats_table.item(0, 1).text() == "30.0%"
    assert w._analysis_runs_list.count() == 1
    assert "t1__p01" in w._analysis_runs_list.item(0).text()


def test_analysis_export_csv(analysis_window, tmp_path, monkeypatch):
    out_path = tmp_path / "repeat.csv"
    monkeypatch.setattr(
        "deepreefmap.gui.survey.analysis.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_path), "")),
    )
    analysis_window._on_analysis_export_csv()
    content = out_path.read_text()
    assert "mean_fraction" in content
    assert "t1__p01" in content


def test_analysis_level_switch_recomputes(analysis_window):
    w = analysis_window
    w._analysis_level_combo.setCurrentText("coarse")
    assert len(w._analysis_covers) == 1
    assert pytest.approx(sum(w._analysis_covers[0].cover.values())) == 0.3
