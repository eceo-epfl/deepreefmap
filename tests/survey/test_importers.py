import uuid

import pytest

from deepreefmap.survey.models import Transect
from deepreefmap.survey.models.exporters import save_transects_csv
from deepreefmap.survey.models.importers import (
    import_transects_csv,
    import_transects_gpx,
    parse_latlon,
)

GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="-17.51" lon="177.11"><name>T2 start</name></wpt>
  <wpt lat="-17.512" lon="177.112"></wpt>
  <trk>
    <name>Reef flat</name>
    <trkseg>
      <trkpt lat="-17.5" lon="177.1"></trkpt>
      <trkpt lat="-17.5002" lon="177.1002"></trkpt>
      <trkpt lat="-17.5005" lon="177.1005"></trkpt>
    </trkseg>
  </trk>
</gpx>
"""


def test_parse_latlon_accepts_space_and_comma():
    assert parse_latlon("-17.5 177.1") == (-17.5, 177.1)
    assert parse_latlon("-17.5, 177.1") == (-17.5, 177.1)
    assert parse_latlon("-17.5,177.1") == (-17.5, 177.1)


def test_parse_latlon_rejects_garbage():
    with pytest.raises(ValueError):
        parse_latlon("-17.5")
    with pytest.raises(ValueError):
        parse_latlon("95 10")
    with pytest.raises(ValueError):
        parse_latlon("ten twenty")


def test_csv_import_case_insensitive_headers(tmp_path):
    path = tmp_path / "transects.csv"
    path.write_text(
        "Name,Start_Lat,Start_Lon,End_Lat,End_Lon,Length_M\n"
        "T1,-17.5,177.1,-17.5005,177.1005,50\n"
        ",,,,,\n"
    )
    transects = import_transects_csv(path)
    assert len(transects) == 1
    assert transects[0].name == "T1"
    assert transects[0].length_m == 50.0
    assert transects[0].depth_m is None


def test_csv_import_reports_row_number_on_error(tmp_path):
    path = tmp_path / "transects.csv"
    path.write_text(
        "name,start_lat,start_lon,end_lat,end_lon\n"
        "T1,-17.5,177.1,-17.5005,177.1005\n"
        "T2,not-a-number,177.1,-17.5,177.1\n"
    )
    with pytest.raises(ValueError, match="Row 3"):
        import_transects_csv(path)


def test_csv_import_rejects_missing_columns(tmp_path):
    path = tmp_path / "transects.csv"
    path.write_text("name,start_lat\nT1,-17.5\n")
    with pytest.raises(ValueError, match="end_lat"):
        import_transects_csv(path)


def test_csv_round_trip_preserves_ids(tmp_path):
    original = Transect(
        name="T1",
        start_lat=-17.5,
        start_lon=177.1,
        end_lat=-17.5005,
        end_lon=177.1005,
        depth_m=8.0,
        description="north edge",
    )
    path = tmp_path / "transects.csv"
    save_transects_csv(path, [original])
    imported = import_transects_csv(path)[0]
    assert imported.id == original.id
    assert imported.name == original.name
    assert imported.length_m is None
    assert imported.depth_m == 8.0
    assert imported.description == "north edge"


def test_gpx_import_reads_tracks_and_waypoint_pairs(tmp_path):
    path = tmp_path / "site.gpx"
    path.write_text(GPX)
    transects = import_transects_gpx(path)
    by_name = {t.name: t for t in transects}
    track = by_name["Reef flat"]
    assert (track.start_lat, track.start_lon) == (-17.5, 177.1)
    assert (track.end_lat, track.end_lon) == (-17.5005, 177.1005)
    pair = by_name["T2 start"]
    assert (pair.end_lat, pair.end_lon) == (-17.512, 177.112)
    assert all(isinstance(t.id, uuid.UUID) for t in transects)


def test_gpx_import_rejects_non_gpx(tmp_path):
    path = tmp_path / "junk.gpx"
    path.write_text("not xml at all")
    with pytest.raises(ValueError):
        import_transects_gpx(path)
