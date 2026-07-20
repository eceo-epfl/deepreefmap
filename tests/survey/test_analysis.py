import json

import pytest

from deepreefmap.config.classes import load_classes
from deepreefmap.survey.analysis import (
    assemble_transect_covers,
    cover_labels,
    repeatability_stats,
    reproducibility_groups,
)
from deepreefmap.survey.models import RunRecord, Transect, TransectPass, VideoAsset
from deepreefmap.survey.store import SurveyStore


@pytest.fixture(scope="module")
def classes_config():
    return load_classes()


@pytest.fixture
def store(tmp_path):
    return SurveyStore(tmp_path / "survey.db")


def seed_transect(store):
    transect = Transect(
        name="T1",
        start_lat=-17.5,
        start_lon=177.1,
        end_lat=-17.5005,
        end_lon=177.1005,
    )
    store.add_transect(transect)
    return transect


def seed_run(store, out_root, transect, run_dir_name, fractions, *,
             content_hash="ab" * 16, begin_s=0.0, end_s=60.0, status="succeeded",
             classes_config=None):
    """One pass + run whose run dir holds a benthic_cover.json with the given
    fine-class fractions (mapped onto the first N configured classes)."""
    video = store.upsert_video(
        VideoAsset(file_name="a.mp4", path="/a.mp4", hash=content_hash)
    )
    pass_ = TransectPass(
        transect_id=transect.id, video_id=video.id, begin_s=begin_s, end_s=end_s
    )
    store.add_pass(pass_)
    run = RunRecord(pass_id=pass_.id, run_dir_name=run_dir_name, status=status)
    store.add_run(run)
    run_dir = out_root / run_dir_name
    run_dir.mkdir(parents=True)
    classes = {}
    for cls, fraction in zip(classes_config.classes, fractions):
        classes[str(cls.id)] = {"name": cls.name, "count": fraction * 100, "fraction": fraction}
    (run_dir / "benthic_cover.json").write_text(
        json.dumps({"classes": classes, "denominator": 100.0})
    )
    return pass_, run


def test_assemble_reads_succeeded_runs_only(store, tmp_path, classes_config):
    transect = seed_transect(store)
    out_root = tmp_path / "out"
    seed_run(store, out_root, transect, "run_a", [0.3, 0.5], classes_config=classes_config)
    seed_run(store, out_root, transect, "run_b", [0.4, 0.4], status="failed",
             classes_config=classes_config)
    covers = assemble_transect_covers(store, out_root, transect.id, classes_config, level="fine")
    assert len(covers) == 1
    assert covers[0].run_dir_name == "run_a"
    assert covers[0].video_hash == "ab" * 16
    assert pytest.approx(sum(covers[0].cover.values())) == 0.8


def test_missing_cover_file_is_skipped(store, tmp_path, classes_config):
    transect = seed_transect(store)
    out_root = tmp_path / "out"
    _, run = seed_run(store, out_root, transect, "run_a", [0.3], classes_config=classes_config)
    (out_root / "run_a" / "benthic_cover.json").unlink()
    covers = assemble_transect_covers(store, out_root, transect.id, classes_config)
    assert covers == []


def test_repeatability_stats_math(store, tmp_path, classes_config):
    transect = seed_transect(store)
    out_root = tmp_path / "out"
    seed_run(store, out_root, transect, "run_a", [0.3], classes_config=classes_config)
    seed_run(store, out_root, transect, "run_b", [0.5], classes_config=classes_config)
    covers = assemble_transect_covers(store, out_root, transect.id, classes_config, level="fine")
    label = classes_config.classes[0].name
    stats = repeatability_stats(covers)[label]
    assert stats["mean"] == pytest.approx(0.4)
    assert stats["std"] == pytest.approx(0.1414, abs=1e-3)
    assert stats["range"] == pytest.approx(0.2)
    assert stats["cv"] == pytest.approx(stats["std"] / 0.4)


def test_reproducibility_groups_by_hash_and_trim(store, tmp_path, classes_config):
    transect = seed_transect(store)
    out_root = tmp_path / "out"
    seed_run(store, out_root, transect, "run_a", [0.3], classes_config=classes_config)
    seed_run(store, out_root, transect, "run_b", [0.32], classes_config=classes_config)
    seed_run(store, out_root, transect, "run_c", [0.5], begin_s=60.0, end_s=120.0,
             classes_config=classes_config)
    covers = assemble_transect_covers(store, out_root, transect.id, classes_config, level="fine")
    groups = reproducibility_groups(covers)
    assert len(groups) == 1
    assert {c.run_dir_name for c in groups[0]} == {"run_a", "run_b"}


def test_cover_labels_orders_by_mean_and_filters(store, tmp_path, classes_config):
    transect = seed_transect(store)
    out_root = tmp_path / "out"
    seed_run(store, out_root, transect, "run_a", [0.1, 0.6, 0.001],
             classes_config=classes_config)
    covers = assemble_transect_covers(store, out_root, transect.id, classes_config, level="fine")
    labels = cover_labels(covers, minimum_fraction=0.005)
    names = [cls.name for cls in classes_config.classes[:3]]
    assert labels[0] == names[1]
    assert names[2] not in labels
