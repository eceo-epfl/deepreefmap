"""Web Mercator tile arithmetic. Pure functions, no Qt."""

from __future__ import annotations

import math

TILE_SIZE = 256
MAX_LATITUDE = 85.0511
MIN_ZOOM = 1
MAX_ZOOM = 19


def deg2tile(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Fractional tile coordinates of a WGS84 point."""
    lat = max(-MAX_LATITUDE, min(MAX_LATITUDE, lat))
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile2deg(x: float, y: float, zoom: int) -> tuple[float, float]:
    """WGS84 point of fractional tile coordinates."""
    n = 2.0**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def clamp_zoom(zoom: int) -> int:
    return max(MIN_ZOOM, min(MAX_ZOOM, zoom))


def fit_zoom(
    points: list[tuple[float, float]], width_px: int, height_px: int, padding_px: int = 40
) -> int:
    """Largest zoom at which all points fit inside the given pixel viewport."""
    if not points or width_px <= 0 or height_px <= 0:
        return MIN_ZOOM
    usable_w = max(1, width_px - 2 * padding_px)
    usable_h = max(1, height_px - 2 * padding_px)
    for zoom in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
        xs, ys = zip(*(deg2tile(lat, lon, zoom) for lat, lon in points))
        span_w = (max(xs) - min(xs)) * TILE_SIZE
        span_h = (max(ys) - min(ys)) * TILE_SIZE
        if span_w <= usable_w and span_h <= usable_h:
            return zoom
    return MIN_ZOOM
