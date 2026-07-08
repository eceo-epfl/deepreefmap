"""Batch job CSV parsing.

Covers deepreefmap/gui/batch.py: the timestamp-range mini-parser and the CSV
loader (column detection, blank-row skipping, Excel rejection).
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "text, expected",
    [
        ("12-45.5", (12.0, 45.5)),
        ("30-", (30.0, None)),
        ("-60", (None, 60.0)),
        ("", (None, None)),
        ("15", (15.0, None)),
    ],
)
def test_parse_timestamp_range(text, expected) -> None:
    from deepreefmap.gui.form.batch import _parse_timestamp_range

    assert _parse_timestamp_range(text) == expected


def test_load_batch_csv_parses_rows(tmp_path) -> None:
    from deepreefmap.gui.form.batch import _load_batch_csv

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "videos,timestamps,transect_length,crop_width\n"
        "a.mp4,5-30,10,2\n"
        "b.mp4,-60,,1.5\n"
    )
    jobs = _load_batch_csv(csv_path)
    assert len(jobs) == 2
    assert jobs[0].video == "a.mp4"
    assert jobs[0].begin_s == 5.0
    assert jobs[0].end_s == 30.0
    assert jobs[0].transect_length == 10.0
    assert jobs[0].crop_width == 2.0
    assert jobs[0].name == "a"
    assert jobs[1].begin_s is None
    assert jobs[1].end_s == 60.0
    assert jobs[1].transect_length is None


def test_load_batch_csv_case_insensitive_columns(tmp_path) -> None:
    from deepreefmap.gui.form.batch import _load_batch_csv

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "Videos,Timestamps,Transect_Length,Crop_Width\n"
        "x.mp4,0-10,5,1\n"
    )
    jobs = _load_batch_csv(csv_path)
    assert len(jobs) == 1


def test_load_batch_csv_rejects_missing_columns(tmp_path) -> None:
    from deepreefmap.gui.form.batch import _load_batch_csv

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("videos,timestamps\nx.mp4,0-10\n")
    with pytest.raises(ValueError, match="missing required columns"):
        _load_batch_csv(csv_path)


def test_load_batch_csv_skips_blank_rows(tmp_path) -> None:
    from deepreefmap.gui.form.batch import _load_batch_csv

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "videos,timestamps,transect_length,crop_width\n"
        ",,,,\n"
        "x.mp4,0-10,5,1\n"
    )
    jobs = _load_batch_csv(csv_path)
    assert len(jobs) == 1
    assert jobs[0].video == "x.mp4"


def test_load_batch_csv_rejects_excel(tmp_path) -> None:
    from deepreefmap.gui.form.batch import _load_batch_csv

    bogus = tmp_path / "jobs.xlsx"
    bogus.write_bytes(b"not actually excel")
    with pytest.raises(ValueError, match="Excel"):
        _load_batch_csv(bogus)
