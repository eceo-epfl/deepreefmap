import json
from pathlib import Path

from deepreefmap.gui.runs.run_cards import RUN_META_ROLE
from deepreefmap.survey.catalogue import UNASSIGNED_TITLE
from deepreefmap.survey.models import RunRecord, Transect, TransectPass, VideoAsset
from deepreefmap.survey.models.convert import survey_manifest_block
from deepreefmap.survey.store import SurveyStore


def write_run(root: Path, dir_name: str, **overrides) -> Path:
    manifest = {
        "name": None,
        "mode": "semantic",
        "input_videos": ["/data/GX010001.MP4"],
        "video_hashes": ["ab" * 16],
        "run_timestamp": "2026-07-01T10:00:00+00:00",
        "begin_s": 0.0,
        "end_s": 60.0,
        "run_duration_s": 120.0,
        "semantic_reference_points": 1_000_000,
    }
    manifest.update(overrides)
    run_dir = root / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    return run_dir


def write_survey_run(root: Path, dir_name: str) -> Transect:
    store = SurveyStore(root / "survey.db")
    transect = Transect(
        name="T1",
        start_lat=-17.5,
        start_lon=177.1,
        end_lat=-17.5005,
        end_lon=177.1005,
        length_m=50.0,
    )
    store.add_transect(transect)
    video = store.upsert_video(
        VideoAsset(file_name="GX010001.MP4", path="/data/GX010001.MP4", hash="ab" * 16)
    )
    pass_ = TransectPass(transect_id=transect.id, video_id=video.id, begin_s=0.0, end_s=60.0)
    store.add_pass(pass_)
    run = RunRecord(pass_id=pass_.id, run_dir_name=dir_name, status="succeeded")
    store.add_run(run)
    write_run(root, dir_name, survey=survey_manifest_block(run, pass_, transect, None))
    store.close()
    return transect


def test_runs_facet_lists_all_runs(tmp_path, make_window):
    root = tmp_path / "out"
    write_run(root, "run_a")
    write_run(root, "run_b", run_timestamp="2026-07-02T10:00:00+00:00")
    window = make_window()
    assert window._data_facet == "runs"
    assert window._data_run_list.count() == 2
    assert window._data_run_list.item(0).text() == "run_b"


def test_transects_facet_groups_and_buckets_unassigned(tmp_path, make_window):
    root = tmp_path / "out"
    write_survey_run(root, "assigned")
    write_run(root, "loose", video_hashes=["cd" * 16])
    window = make_window()
    window._data_facet_buttons["transects"].click()
    titles = [
        window._data_tree.topLevelItem(i).text(0)
        for i in range(window._data_tree.topLevelItemCount())
    ]
    assert titles[0].startswith(UNASSIGNED_TITLE)
    assert any(t.startswith("T1") for t in titles)


def test_videos_facet_splits_time_windows(tmp_path, make_window):
    root = tmp_path / "out"
    write_run(root, "first", begin_s=0.0, end_s=60.0)
    write_run(root, "second", begin_s=60.0, end_s=120.0)
    window = make_window()
    window._data_facet_buttons["videos"].click()
    assert window._data_tree.topLevelItemCount() == 1
    assert window._data_tree.topLevelItem(0).childCount() == 2


def test_tree_selection_filters_run_list(tmp_path, make_window):
    root = tmp_path / "out"
    write_run(root, "first", begin_s=0.0, end_s=60.0)
    write_run(root, "second", begin_s=60.0, end_s=120.0)
    window = make_window()
    window._data_facet_buttons["videos"].click()
    assert window._data_run_list.count() == 2
    child = window._data_tree.topLevelItem(0).child(0)
    window._data_tree.setCurrentItem(child)
    assert window._data_run_list.count() == 1


def test_open_routes_through_auto_load(tmp_path, make_window, monkeypatch):
    root = tmp_path / "out"
    run_dir = write_run(root, "run_a")
    window = make_window()
    loaded = []
    monkeypatch.setattr(window, "_auto_load_run", lambda path: loaded.append(path))
    window._data_run_list.setCurrentRow(0)
    window._on_data_open_clicked()
    assert loaded == [run_dir]
    assert window._run_meta_banner.isVisibleTo(window)


def test_data_panel_moves_between_hosts(make_window):
    window = make_window()
    assert window._data_panel.parentWidget() is window._data_host_simple
    window._mode_toggle_btn.click()
    assert window._data_panel.parentWidget() is window._data_tab
    window._mode_toggle_btn.click()
    assert window._data_panel.parentWidget() is window._data_host_simple


def test_data_tab_and_nav_registered(make_window):
    window = make_window()
    assert window._sidebar_tabs.tabText(window._TAB_DATA) == "Data"
    assert window._sidebar_tabs.tabText(window._TAB_SYSTEM) == "System"
    assert "data" in window._simple_nav_buttons


def test_rename_updates_manifest_and_card(tmp_path, make_window, monkeypatch):
    root = tmp_path / "out"
    run_dir = write_run(root, "run_a")
    window = make_window()
    window._data_run_list.setCurrentRow(0)
    monkeypatch.setattr(
        "deepreefmap.gui.runs.data_manager.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("reef north", True)),
    )
    window._on_data_rename_clicked()
    on_disk = json.loads((run_dir / "run_manifest.json").read_text())
    assert on_disk["name"] == "reef north"
    assert window._data_run_list.item(0).text().startswith("reef north")


def test_delete_removes_run_after_confirmation(tmp_path, make_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    root = tmp_path / "out"
    run_dir = write_run(root, "doomed")
    window = make_window()
    window._data_run_list.setCurrentRow(0)
    monkeypatch.setattr(
        "deepreefmap.gui.runs.data_manager.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    window._on_data_delete_clicked()
    assert not run_dir.exists()
    assert window._data_run_list.count() == 0


def test_delete_refuses_open_run(tmp_path, make_window, monkeypatch):
    root = tmp_path / "out"
    run_dir = write_run(root, "active")
    window = make_window()
    window._active_run_dir = run_dir
    window._data_run_list.setCurrentRow(0)
    warnings = []
    monkeypatch.setattr(
        "deepreefmap.gui.runs.data_manager.QMessageBox.information",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    window._on_data_delete_clicked()
    assert run_dir.exists()
    assert warnings


def test_assign_moves_loose_run_under_transect(tmp_path, make_window, monkeypatch):
    root = tmp_path / "out"
    transect = write_survey_run(root, "assigned")
    write_run(root, "loose", video_hashes=["cd" * 16])
    window = make_window()
    window._data_run_list.setCurrentRow(0 if window._data_run_list.item(0).text() == "loose" else 1)
    monkeypatch.setattr(
        window, "_ask_assign_target", lambda transects: (transect.id, "forward")
    )
    window._on_data_assign_clicked()
    entries = {e.dir_name: e for e in window._data_entries}
    assert entries["loose"].transect_name == "T1"


def test_size_scan_slot_updates_label_and_cards(tmp_path, make_window):
    root = tmp_path / "out"
    write_run(root, "run_a")
    window = make_window()
    window._apply_run_sizes({"run_a": 2_000_000_000})
    assert "2.00 GB" in window._data_disk_label.text()
    assert "Space used" in window._data_disk_label.text()
    meta = window._data_run_list.item(0).data(RUN_META_ROLE)
    assert "2.00 GB" in meta["facts"]


def test_results_tab_has_no_rename_row(make_window):
    window = make_window()
    assert not hasattr(window, "_rename_btn")
    assert not hasattr(window, "_rename_edit")
