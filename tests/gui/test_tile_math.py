import pytest

from deepreefmap.gui.map.tile_math import clamp_zoom, deg2tile, fit_zoom, tile2deg


def test_origin_is_tile_grid_center():
    assert deg2tile(0.0, 0.0, 4) == (8.0, 8.0)


def test_round_trip():
    lat, lon = -17.5123, 177.1456
    x, y = deg2tile(lat, lon, 15)
    back_lat, back_lon = tile2deg(x, y, 15)
    assert back_lat == pytest.approx(lat, abs=1e-6)
    assert back_lon == pytest.approx(lon, abs=1e-6)


def test_polar_latitudes_clamp_into_grid():
    _, y = deg2tile(89.9, 0.0, 3)
    assert 0.0 <= y <= 8.0


def test_clamp_zoom_bounds():
    assert clamp_zoom(0) == 1
    assert clamp_zoom(25) == 19
    assert clamp_zoom(10) == 10


def test_fit_zoom_scales_with_extent():
    reef = [(-17.5, 177.1), (-17.5005, 177.1005)]
    world = [(-40.0, -100.0), (45.0, 150.0)]
    assert fit_zoom(reef, 400, 300) >= 15
    assert fit_zoom(world, 400, 300) <= 3
    assert fit_zoom([], 400, 300) == 1
