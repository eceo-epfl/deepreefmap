"""Transect imports: quick decimal-degree text entry, CSV, and GPX."""

from __future__ import annotations

import csv
import uuid
from pathlib import Path
from xml.etree import ElementTree

from deepreefmap.survey.models.transect import Transect

_CSV_REQUIRED = {"name", "start_lat", "start_lon", "end_lat", "end_lon"}


def parse_latlon(text: str) -> tuple[float, float]:
    """Parse "lat lon" or "lat, lon" in decimal degrees."""
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"Expected 'lat lon', got: {text!r}")
    lat, lon = float(parts[0]), float(parts[1])
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude out of range: {lat}")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude out of range: {lon}")
    return lat, lon


def import_transects_csv(path: Path) -> list[Transect]:
    """Read transects from a CSV with case-insensitive headers.

    Required columns: name, start_lat, start_lon, end_lat, end_lon.
    Optional: length_m, depth_m, description, id (a UUID kept for round-trips).
    """
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        norm = {fn.strip().lower(): fn for fn in reader.fieldnames}
        missing = _CSV_REQUIRED - set(norm)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        transects = []
        for n, row in enumerate(reader, start=2):
            name = _cell(row, norm, "name")
            if not name:
                continue
            try:
                transects.append(_transect_from_csv_row(row, norm, name))
            except ValueError as exc:
                raise ValueError(f"Row {n}: {exc}") from exc
    if not transects:
        raise ValueError("No usable rows in CSV.")
    return transects


def _cell(row: dict[str, str], norm: dict[str, str], key: str) -> str:
    if key not in norm:
        return ""
    return (row.get(norm[key], "") or "").strip()


def _optional_float(raw: str) -> float | None:
    return float(raw) if raw else None


def _transect_from_csv_row(row: dict[str, str], norm: dict[str, str], name: str) -> Transect:
    transect = Transect(
        name=name,
        start_lat=float(_cell(row, norm, "start_lat")),
        start_lon=float(_cell(row, norm, "start_lon")),
        end_lat=float(_cell(row, norm, "end_lat")),
        end_lon=float(_cell(row, norm, "end_lon")),
        length_m=_optional_float(_cell(row, norm, "length_m")),
        depth_m=_optional_float(_cell(row, norm, "depth_m")),
        description=_cell(row, norm, "description"),
    )
    raw_id = _cell(row, norm, "id")
    if raw_id:
        transect.id = uuid.UUID(raw_id)
    return transect


def import_transects_gpx(path: Path) -> list[Transect]:
    """Read transects from a GPX file.

    Each track or route becomes a transect from its first to its last point;
    bare waypoints pair up in file order (start, end, start, end, ...).
    """
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"Not a valid GPX file: {exc}") from exc
    transects = []
    for segment_tag, point_tag in (("trk", "trkpt"), ("rte", "rtept")):
        for i, segment in enumerate(root.findall(f".//{{*}}{segment_tag}"), start=1):
            points = segment.findall(f".//{{*}}{point_tag}")
            if len(points) < 2:
                continue
            name = _gpx_name(segment) or f"{path.stem} {segment_tag} {i}"
            transects.append(_transect_from_gpx_points(name, points[0], points[-1]))
    waypoints = root.findall(".//{*}wpt")
    for i in range(0, len(waypoints) - 1, 2):
        name = _gpx_name(waypoints[i]) or f"{path.stem} pair {i // 2 + 1}"
        transects.append(_transect_from_gpx_points(name, waypoints[i], waypoints[i + 1]))
    if not transects:
        raise ValueError("No tracks, routes, or waypoint pairs in GPX file.")
    return transects


def _gpx_name(element: ElementTree.Element) -> str:
    node = element.find("{*}name")
    return (node.text or "").strip() if node is not None else ""


def _transect_from_gpx_points(
    name: str, start: ElementTree.Element, end: ElementTree.Element
) -> Transect:
    def coords(point: ElementTree.Element) -> tuple[float, float]:
        try:
            return float(point.attrib["lat"]), float(point.attrib["lon"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"GPX point in {name!r} has no valid lat/lon") from exc

    start_lat, start_lon = coords(start)
    end_lat, end_lon = coords(end)
    return Transect(
        name=name,
        start_lat=start_lat,
        start_lon=start_lon,
        end_lat=end_lat,
        end_lon=end_lon,
    )
