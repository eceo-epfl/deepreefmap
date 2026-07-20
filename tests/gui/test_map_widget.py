import pytest

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest

from deepreefmap.gui.map.layers import OSM_LAYER
from deepreefmap.gui.map.overlays import OverlayTransect
from deepreefmap.gui.map.tile_cache import TileCache
from deepreefmap.gui.map.widget import SlippyMapWidget


@pytest.fixture
def map_widget(qapp):
    cache = TileCache(OSM_LAYER)
    cache.network_enabled = False
    widget = SlippyMapWidget(cache=cache)
    widget.resize(400, 300)
    return widget


def make_overlay(selected=False):
    return OverlayTransect(
        id="11111111-1111-1111-1111-111111111111",
        start=(-17.5, 177.1),
        end=(-17.5005, 177.1005),
        color=QColor(0, 170, 170),
        selected=selected,
    )


def test_latlon_at_center_matches_view(map_widget):
    map_widget.set_view(-17.5, 177.1, 15)
    lat, lon = map_widget.latlon_at(QPointF(200, 150))
    assert lat == pytest.approx(-17.5, abs=1e-4)
    assert lon == pytest.approx(177.1, abs=1e-4)


def test_px_and_latlon_are_inverse(map_widget):
    map_widget.set_view(-17.5, 177.1, 16)
    point = map_widget._px_of(-17.5002, 177.1003)
    lat, lon = map_widget.latlon_at(point)
    assert lat == pytest.approx(-17.5002, abs=1e-6)
    assert lon == pytest.approx(177.1003, abs=1e-6)


def test_fit_transects_centers_on_overlay(map_widget):
    map_widget.set_transects([make_overlay()])
    map_widget.fit_transects()
    lat, lon = map_widget.latlon_at(QPointF(200, 150))
    assert lat == pytest.approx(-17.50025, abs=1e-3)
    assert lon == pytest.approx(177.10025, abs=1e-3)


def test_click_on_empty_map_emits_coordinates(map_widget):
    map_widget.set_view(-17.5, 177.1, 15)
    clicks = []
    map_widget.map_clicked.connect(lambda lat, lon: clicks.append((lat, lon)))
    QTest.mouseClick(map_widget, Qt.MouseButton.LeftButton, pos=QPoint(200, 150))
    assert len(clicks) == 1
    assert clicks[0][0] == pytest.approx(-17.5, abs=1e-3)


def test_click_on_transect_emits_id(map_widget):
    overlay = make_overlay()
    map_widget.set_transects([overlay])
    map_widget.fit_transects()
    center = map_widget._px_of(
        (overlay.start[0] + overlay.end[0]) / 2, (overlay.start[1] + overlay.end[1]) / 2
    )
    hits = []
    map_widget.transect_clicked.connect(hits.append)
    QTest.mouseClick(
        map_widget, Qt.MouseButton.LeftButton, pos=QPoint(int(center.x()), int(center.y()))
    )
    assert hits == [overlay.id]


def test_endpoint_hit_requires_editable(map_widget):
    overlay = make_overlay(selected=True)
    map_widget.set_transects([overlay])
    map_widget.fit_transects()
    start_px = map_widget._px_of(*overlay.start)
    assert map_widget._endpoint_at(start_px) is None
    map_widget.set_editable(overlay.id)
    hit = map_widget._endpoint_at(start_px)
    assert hit is not None
    assert hit[1] == "start"


def test_render_without_network_is_safe(map_widget):
    map_widget.set_transects([make_overlay(selected=True)])
    map_widget.fit_transects()
    image = map_widget.grab()
    assert not image.isNull()
