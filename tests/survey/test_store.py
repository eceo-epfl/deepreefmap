import json
import sqlite3
import threading
import uuid

import pytest

from deepreefmap.survey.models import RunRecord, SurveyBatch, Transect, TransectPass, VideoAsset
from deepreefmap.survey.models.convert import survey_manifest_block
from deepreefmap.survey.store import SurveyStore


@pytest.fixture
def store(tmp_path):
    return SurveyStore(tmp_path / "survey.db")


def make_transect(name="T1"):
    return Transect(
        name=name,
        start_lat=-17.5,
        start_lon=177.1,
        end_lat=-17.5005,
        end_lon=177.1005,
        length_m=50.0,
    )


def make_video(content_hash="ab" * 16):
    return VideoAsset(file_name="GX010001.MP4", path="/data/GX010001.MP4", hash=content_hash)


def seed_pass(store, direction="forward"):
    transect, video = make_transect(), make_video()
    store.add_transect(transect)
    store.upsert_video(video)
    pass_ = TransectPass(
        transect_id=transect.id, video_id=video.id, begin_s=0.0, end_s=60.0, direction=direction
    )
    store.add_pass(pass_)
    return transect, video, pass_


def test_transect_crud_round_trip(store):
    transect = make_transect()
    store.add_transect(transect)
    assert store.get_transect(transect.id) == transect
    transect.end_lat = -17.501
    before = transect.updated_at
    transect.updated_at = "2000-01-01T00:00:00+00:00"
    store.update_transect(transect)
    stored = store.get_transect(transect.id)
    assert stored.end_lat == -17.501
    assert stored.updated_at >= before
    store.delete_transect(transect.id)
    assert store.get_transect(transect.id) is None


def test_transect_names_are_unique(store):
    store.add_transect(make_transect())
    with pytest.raises(sqlite3.IntegrityError):
        store.add_transect(make_transect())


def test_list_transects_orders_by_name(store):
    store.add_transect(make_transect("B"))
    store.add_transect(make_transect("A"))
    assert [t.name for t in store.list_transects()] == ["A", "B"]


def test_upsert_video_dedups_by_hash(store):
    first = store.upsert_video(make_video())
    again = make_video()
    again.path = "/moved/GX010001.MP4"
    again.duration_s = 61.5
    second = store.upsert_video(again)
    assert second.id == first.id
    assert len(store.list_videos()) == 1
    stored = store.get_video(first.id)
    assert stored.path == "/moved/GX010001.MP4"
    assert stored.duration_s == 61.5


def test_upsert_video_without_hash_always_inserts(store):
    store.upsert_video(make_video(content_hash=None))
    store.upsert_video(make_video(content_hash=None))
    assert len(store.list_videos()) == 2


def test_pass_filters(store):
    transect, video, pass_ = seed_pass(store)
    batch = SurveyBatch(name="Day 1")
    store.add_batch(batch)
    second = TransectPass(
        transect_id=transect.id,
        video_id=video.id,
        batch_id=batch.id,
        begin_s=70.0,
        end_s=120.0,
        direction="reverse",
    )
    store.add_pass(second)
    assert len(store.list_passes()) == 2
    assert store.list_passes(transect_id=transect.id) == [pass_, second]
    assert store.list_passes(batch_id=batch.id) == [second]


def test_delete_transect_with_passes_is_blocked(store):
    transect, _, _ = seed_pass(store)
    with pytest.raises(sqlite3.IntegrityError):
        store.delete_transect(transect.id)


def test_run_status_stamps_lifecycle(store):
    _, _, pass_ = seed_pass(store)
    run = RunRecord(pass_id=pass_.id, run_dir_name="t1__p01")
    store.add_run(run)
    store.set_run_status(run.id, "running")
    started = store.get_run(run.id)
    assert started.status == "running"
    assert started.started_at is not None
    assert started.finished_at is None
    store.set_run_status(run.id, "failed", error="boom")
    finished = store.get_run(run.id)
    assert finished.status == "failed"
    assert finished.error == "boom"
    assert finished.finished_at is not None


def test_run_status_rejects_unknown_run_and_status(store):
    _, _, pass_ = seed_pass(store)
    with pytest.raises(KeyError):
        store.set_run_status(uuid.uuid4(), "running")
    with pytest.raises(ValueError):
        run = RunRecord(pass_id=pass_.id, run_dir_name="run")
        store.add_run(run)
        store.set_run_status(run.id, "exploded")


def test_runs_for_transect_joins_passes(store):
    transect, _, pass_ = seed_pass(store)
    run = RunRecord(pass_id=pass_.id, run_dir_name="t1__p01")
    store.add_run(run)
    assert store.runs_for_transect(transect.id) == [store.get_run(run.id)]
    assert store.runs_for_pass(pass_.id) == [store.get_run(run.id)]


def test_run_lookup_by_dir_name_and_delete(store):
    _, _, pass_ = seed_pass(store)
    run = RunRecord(pass_id=pass_.id, run_dir_name="t1__p01")
    store.add_run(run)
    assert store.run_by_dir_name("t1__p01") == store.get_run(run.id)
    assert store.run_by_dir_name("missing") is None
    store.delete_run(run.id)
    assert store.get_run(run.id) is None


def test_json_export_import_round_trip(store, tmp_path):
    _, _, pass_ = seed_pass(store)
    batch = SurveyBatch(name="Day 1")
    store.add_batch(batch)
    store.add_run(RunRecord(pass_id=pass_.id, run_dir_name="t1__p01"))
    doc_path = tmp_path / "survey.json"
    store.export_json(doc_path)

    fresh = SurveyStore(tmp_path / "fresh.db")
    fresh.import_json(doc_path)
    assert fresh.list_transects() == store.list_transects()
    assert fresh.list_videos() == store.list_videos()
    assert fresh.list_passes() == store.list_passes()
    assert fresh.list_runs() == store.list_runs()

    fresh.import_json(doc_path)
    assert len(fresh.list_passes()) == 1


def write_manifest(out_root, run_dir_name, block, video_path="/data/GX010001.MP4"):
    run_dir = out_root / run_dir_name
    run_dir.mkdir(parents=True)
    manifest = {
        "name": run_dir_name,
        "input_videos": [video_path],
        "video_hashes": ["ab" * 16],
        "video_sizes": [1024],
        "video_mtimes": ["2026-07-20T09:00:00+00:00"],
        "run_timestamp": "2026-07-20T09:05:00+00:00",
        "survey": block,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))


def test_rebuild_from_scan_restores_everything(store, tmp_path):
    transect, video, pass_ = seed_pass(store, direction="reverse")
    batch = SurveyBatch(name="Day 1")
    store.add_batch(batch)
    run = RunRecord(pass_id=pass_.id, run_dir_name="t1__p01")
    store.add_run(run)

    out_root = tmp_path / "out"
    write_manifest(out_root, "t1__p01", survey_manifest_block(run, pass_, transect, batch))
    (out_root / "plain_run").mkdir()
    (out_root / "plain_run" / "run_manifest.json").write_text(json.dumps({"name": "plain"}))

    fresh = SurveyStore(tmp_path / "rebuilt.db")
    report = fresh.rebuild_from_scan(out_root)
    assert (report.transects, report.videos, report.batches, report.passes, report.runs) == (
        1, 1, 1, 1, 1,
    )
    assert report.skipped == ["plain_run"]
    restored = fresh.get_transect(transect.id)
    assert restored.name == transect.name
    assert restored.start_lat == transect.start_lat
    restored_pass = fresh.get_pass(pass_.id)
    assert restored_pass.direction == "reverse"
    assert restored_pass.end_s == 60.0
    restored_run = fresh.get_run(run.id)
    assert restored_run.status == "succeeded"
    assert restored_run.run_dir_name == "t1__p01"

    again = fresh.rebuild_from_scan(out_root)
    assert (again.transects, again.runs) == (0, 0)


def test_store_reopens_existing_database(tmp_path):
    path = tmp_path / "survey.db"
    first = SurveyStore(path)
    first.add_transect(make_transect())
    first.close()
    second = SurveyStore(path)
    assert len(second.list_transects()) == 1


def test_worker_thread_writes_are_visible(store):
    _, _, pass_ = seed_pass(store)
    errors = []

    def worker():
        try:
            run = RunRecord(pass_id=pass_.id, run_dir_name="t1__p01")
            store.add_run(run)
            store.set_run_status(run.id, "running")
            store.set_run_status(run.id, "succeeded")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert not errors
    assert [r.status for r in store.runs_for_pass(pass_.id)] == ["succeeded"]
