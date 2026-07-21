import json

import pytest

from deepreefmap.survey import catalogue
from deepreefmap.survey.catalogue import UNASSIGNED_TITLE
from deepreefmap.survey.models import RunRecord, Transect, TransectPass, VideoAsset
from deepreefmap.survey.models.convert import survey_manifest_block
from deepreefmap.survey.store import SurveyStore


@pytest.fixture
def out_root(tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    return root


@pytest.fixture
def store(out_root):
    return SurveyStore(out_root / "survey.db")


def make_transect(name="T1"):
    return Transect(
        name=name,
        start_lat=-17.5,
        start_lon=177.1,
        end_lat=-17.5005,
        end_lon=177.1005,
        length_m=50.0,
    )


def write_run(out_root, dir_name, **overrides):
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
    run_dir = out_root / dir_name
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    return run_dir


def seed_survey_run(store, out_root, dir_name, transect=None):
    transect = transect or make_transect()
    if store.get_transect(transect.id) is None:
        store.add_transect(transect)
    video = store.upsert_video(
        VideoAsset(file_name="GX010001.MP4", path="/data/GX010001.MP4", hash="ab" * 16)
    )
    pass_ = TransectPass(transect_id=transect.id, video_id=video.id, begin_s=0.0, end_s=60.0)
    store.add_pass(pass_)
    run = RunRecord(pass_id=pass_.id, run_dir_name=dir_name, status="succeeded")
    store.add_run(run)
    block = survey_manifest_block(run, pass_, transect, None)
    write_run(out_root, dir_name, survey=block)
    return transect, pass_, run


def scan(out_root, store=None):
    entries = catalogue.scan_out_root(out_root)
    if store is not None:
        catalogue.reconcile(entries, store)
    return entries


def test_scan_orders_newest_first(out_root):
    write_run(out_root, "older", run_timestamp="2026-07-01T10:00:00+00:00")
    write_run(out_root, "newer", run_timestamp="2026-07-02T10:00:00+00:00")
    assert [e.dir_name for e in catalogue.scan_out_root(out_root)] == ["newer", "older"]


def test_group_key_falls_back_from_pass_to_footage(out_root, store):
    seed_survey_run(store, out_root, "with_pass")
    write_run(out_root, "adhoc_a", begin_s=0.0, end_s=60.0)
    write_run(out_root, "adhoc_b", begin_s=0.0, end_s=60.0)
    write_run(out_root, "adhoc_other_window", begin_s=60.0, end_s=120.0)
    entries = {e.dir_name: e for e in scan(out_root, store)}
    assert catalogue.group_key(entries["with_pass"])[0] == "pass"
    assert catalogue.group_key(entries["adhoc_a"]) == catalogue.group_key(entries["adhoc_b"])
    assert catalogue.group_key(entries["adhoc_a"]) != catalogue.group_key(
        entries["adhoc_other_window"]
    )


def test_manifest_pass_id_groups_without_database(out_root, store):
    seed_survey_run(store, out_root, "original")
    fresh = SurveyStore(out_root / "fresh.db")
    entries = scan(out_root, fresh)
    assert catalogue.group_key(entries[0]) == ("pass", str(entries[0].manifest_pass_id))


def test_transects_facet_puts_unassigned_first(out_root, store):
    seed_survey_run(store, out_root, "assigned")
    write_run(out_root, "loose")
    groups = catalogue.transects_facet(scan(out_root, store), store.list_transects())
    assert groups[0].title == UNASSIGNED_TITLE
    assert [e.dir_name for e in groups[0].entries] == ["loose"]
    assert groups[1].title == "T1"
    assert groups[1].children[0].entries[0].dir_name == "assigned"


def test_transects_facet_lists_transects_without_runs(out_root, store):
    store.add_transect(make_transect("Empty"))
    groups = catalogue.transects_facet(scan(out_root, store), store.list_transects())
    assert [g.title for g in groups] == ["Empty"]
    assert groups[0].all_entries() == []


def test_videos_facet_separates_windows_on_shared_video(out_root, store):
    t1, t2 = make_transect("T1"), make_transect("T2")
    seed_survey_run(store, out_root, "first_half", transect=t1)
    store2_pass = TransectPass(
        transect_id=t2.id,
        video_id=store.list_videos()[0].id,
        begin_s=60.0,
        end_s=120.0,
    )
    store.add_transect(t2)
    store.add_pass(store2_pass)
    run = RunRecord(pass_id=store2_pass.id, run_dir_name="second_half", status="succeeded")
    store.add_run(run)
    write_run(out_root, "second_half", begin_s=60.0, end_s=120.0)
    groups = catalogue.videos_facet(scan(out_root, store))
    assert len(groups) == 1
    assert len(groups[0].children) == 2
    titles = {c.title for c in groups[0].children}
    assert any("T1" in t for t in titles) and any("T2" in t for t in titles)


def test_reconcile_database_wins_and_records_move(out_root, store):
    transect, pass_, _run = seed_survey_run(store, out_root, "run1")
    elsewhere = make_transect("Elsewhere")
    store.add_transect(elsewhere)
    pass_.transect_id = elsewhere.id
    store.update_pass(pass_)
    entry = scan(out_root, store)[0]
    assert entry.transect_name == "Elsewhere"
    assert entry.moved_from == "T1"


def test_rename_run_rewrites_manifest_in_place(out_root):
    run_dir = write_run(out_root, "run1")
    manifest = catalogue.rename_run(run_dir, "  reef north  ")
    assert manifest["name"] == "reef north"
    on_disk = json.loads((run_dir / "run_manifest.json").read_text())
    assert on_disk["name"] == "reef north"
    assert not (run_dir / "run_manifest.json.tmp").exists()


def test_delete_run_refuses_outside_root(out_root, tmp_path, store):
    stray = tmp_path / "elsewhere" / "run1"
    stray.mkdir(parents=True)
    (stray / "run_manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        catalogue.delete_run(out_root, stray, store)
    nested = write_run(out_root, "real")
    (nested / "no_manifest").mkdir()
    with pytest.raises(ValueError):
        catalogue.delete_run(out_root, nested / "no_manifest", store)


def test_delete_run_removes_directory_and_row(out_root, store):
    _t, _p, run = seed_survey_run(store, out_root, "doomed")
    catalogue.delete_run(out_root, out_root / "doomed", store)
    assert not (out_root / "doomed").exists()
    assert store.get_run(run.id) is None


def test_assign_moves_pass_with_sibling_reruns(out_root, store):
    transect, pass_, _run = seed_survey_run(store, out_root, "take1")
    store.add_run(RunRecord(pass_id=pass_.id, run_dir_name="take2", status="succeeded"))
    write_run(out_root, "take2")
    target = make_transect("Target")
    store.add_transect(target)
    entries = scan(out_root, store)
    catalogue.assign_to_transect(store, [e for e in entries if e.dir_name == "take1"], target.id)
    assert all(e.transect_name == "Target" for e in scan(out_root, store))


def test_assign_adopts_adhoc_runs_as_one_pass(out_root, store):
    write_run(out_root, "adhoc_a")
    write_run(out_root, "adhoc_b")
    target = make_transect("Target")
    store.add_transect(target)
    catalogue.assign_to_transect(store, scan(out_root, store), target.id)
    passes = store.list_passes(transect_id=target.id)
    assert len(passes) == 1
    assert {r.run_dir_name for r in store.runs_for_pass(passes[0].id)} == {"adhoc_a", "adhoc_b"}


def test_assign_derives_window_from_frames_when_untrimmed(out_root, store):
    write_run(out_root, "untrimmed", begin_s=None, end_s=None, frames_processed=900, fps=3)
    target = make_transect("Target")
    store.add_transect(target)
    catalogue.assign_to_transect(store, scan(out_root, store), target.id)
    pass_ = store.list_passes(transect_id=target.id)[0]
    assert pass_.begin_s == 0.0
    assert pass_.end_s == pytest.approx(300.0)


def test_assign_keeps_manifest_ids_so_rebuild_stays_idempotent(out_root, store):
    seed_survey_run(store, out_root, "copied")
    fresh = SurveyStore(out_root / "fresh.db")
    target = make_transect("Target")
    fresh.add_transect(target)
    entries = scan(out_root, fresh)
    catalogue.assign_to_transect(fresh, entries, target.id)
    report = fresh.rebuild_from_scan(out_root)
    assert report.transects == 1 and report.passes == 0 and report.runs == 0
    entry = scan(out_root, fresh)[0]
    assert entry.transect_name == "Target"
    assert entry.moved_from == "T1"


def test_group_stats_ranges(out_root):
    write_run(out_root, "small", run_duration_s=100.0, semantic_reference_points=1_000)
    write_run(out_root, "large", run_duration_s=300.0, semantic_reference_points=9_000)
    entries = catalogue.scan_out_root(out_root)
    entries[0].size_bytes = 5
    stats = catalogue.group_stats(entries)
    assert stats.run_count == 2
    assert stats.total_bytes == 5
    assert stats.duration_range == (100.0, 300.0)
    assert stats.point_range == (1_000, 9_000)
