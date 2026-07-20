"""A minimal Leaflet-style map widget on QPainter and cached raster tiles."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from deepreefmap.gui.map.overlays import (
    ENDPOINT_HIT_PX,
    LINE_HIT_PX,
    OverlayTransect,
    segment_distance_px,
)
from deepreefmap.gui.map.tile_cache import TileCache, shared_tile_cache
from deepreefmap.gui.map.tile_math import TILE_SIZE, clamp_zoom, deg2tile, fit_zoom, tile2deg


class SlippyMapWidget(QWidget):
    """Pan/zoom raster map with transect line overlays.

    Works offline from the persistent tile cache; tiles never seen online are
    drawn as an empty grid.
    """

    map_clicked = Signal(float, float)
    transect_clicked = Signal(str)
    transect_endpoint_moved = Signal(str, str, float, float)
    view_changed = Signal()

    def __init__(self, cache: TileCache | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cache = cache if cache is not None else shared_tile_cache()
        self._cache.tile_ready.connect(lambda *_: self.update())
        self._center = (0.0, 0.0)
        self._zoom = 2
        self._transects: list[OverlayTransect] = []
        self._editable_id: str | None = None
        self._press_pos: QPointF | None = None
        self._press_center_tile: tuple[float, float] | None = None
        self._dragging_endpoint: tuple[OverlayTransect, str] | None = None
        self._moved = False
        self.setMinimumHeight(240)

    # --- view state ---

    def set_view(self, lat: float, lon: float, zoom: int) -> None:
        self._center = (lat, lon)
        self._zoom = clamp_zoom(zoom)
        self.update()
        self.view_changed.emit()

    def set_transects(self, transects: list[OverlayTransect]) -> None:
        self._transects = list(transects)
        self.update()

    def set_editable(self, transect_id: str | None) -> None:
        self._editable_id = transect_id
        self.update()

    def fit_transects(self) -> None:
        points = [p for t in self._transects for p in (t.start, t.end)]
        if not points:
            return
        zoom = fit_zoom(points, self.width() or 400, self.height() or 240)
        lat = sum(p[0] for p in points) / len(points)
        lon = sum(p[1] for p in points) / len(points)
        self.set_view(lat, lon, zoom)

    def latlon_at(self, pos: QPointF) -> tuple[float, float]:
        cx, cy = deg2tile(*self._center, self._zoom)
        tx = cx + (pos.x() - self.width() / 2) / TILE_SIZE
        ty = cy + (pos.y() - self.height() / 2) / TILE_SIZE
        return tile2deg(tx, ty, self._zoom)

    def _px_of(self, lat: float, lon: float) -> QPointF:
        cx, cy = deg2tile(*self._center, self._zoom)
        tx, ty = deg2tile(lat, lon, self._zoom)
        return QPointF(
            self.width() / 2 + (tx - cx) * TILE_SIZE,
            self.height() / 2 + (ty - cy) * TILE_SIZE,
        )

    # --- painting ---

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(42, 45, 48))
        self._paint_tiles(painter)
        self._paint_transects(painter)
        self._paint_attribution(painter)

    def _paint_tiles(self, painter: QPainter) -> None:
        cx, cy = deg2tile(*self._center, self._zoom)
        half_w = self.width() / 2
        half_h = self.height() / 2
        first_x = math.floor(cx - half_w / TILE_SIZE)
        last_x = math.floor(cx + half_w / TILE_SIZE)
        first_y = math.floor(cy - half_h / TILE_SIZE)
        last_y = math.floor(cy + half_h / TILE_SIZE)
        grid_pen = QPen(QColor(70, 74, 78))
        for tx in range(first_x, last_x + 1):
            for ty in range(first_y, last_y + 1):
                left = half_w + (tx - cx) * TILE_SIZE
                top = half_h + (ty - cy) * TILE_SIZE
                pixmap = self._cache.pixmap(self._zoom, tx, ty)
                if pixmap is not None:
                    painter.drawPixmap(int(left), int(top), pixmap)
                else:
                    painter.setPen(grid_pen)
                    painter.drawRect(QRectF(left, top, TILE_SIZE, TILE_SIZE))

    def _paint_transects(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for transect in self._transects:
            p1 = self._px_of(*transect.start)
            p2 = self._px_of(*transect.end)
            width = 4 if transect.selected else 3
            painter.setPen(QPen(transect.color, width))
            painter.drawLine(p1, p2)
            editable = transect.id == self._editable_id
            radius = 6.0 if editable else 4.0
            painter.setBrush(transect.color)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5 if editable else 0.5))
            painter.drawEllipse(p1, radius, radius)
            painter.drawEllipse(p2, radius, radius)

    def _paint_attribution(self, painter: QPainter) -> None:
        text = self._cache.layer.attribution
        metrics = painter.fontMetrics()
        pad = 4
        rect = QRectF(
            self.width() - metrics.horizontalAdvance(text) - 2 * pad - 2,
            self.height() - metrics.height() - 2 * pad,
            metrics.horizontalAdvance(text) + 2 * pad,
            metrics.height() + pad,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 110))
        painter.drawRect(rect)
        painter.setPen(QColor(235, 235, 235))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # --- interaction ---

    def _endpoint_at(self, pos: QPointF) -> tuple[OverlayTransect, str] | None:
        for transect in self._transects:
            if transect.id != self._editable_id:
                continue
            for which, latlon in (("start", transect.start), ("end", transect.end)):
                point = self._px_of(*latlon)
                if math.hypot(pos.x() - point.x(), pos.y() - point.y()) <= ENDPOINT_HIT_PX:
                    return transect, which
        return None

    def _transect_at(self, pos: QPointF) -> OverlayTransect | None:
        for transect in self._transects:
            p1 = self._px_of(*transect.start)
            p2 = self._px_of(*transect.end)
            if segment_distance_px(pos, p1, p2) <= LINE_HIT_PX:
                return transect
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._press_pos = event.position()
        self._press_center_tile = deg2tile(*self._center, self._zoom)
        self._dragging_endpoint = self._endpoint_at(event.position())
        self._moved = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._press_pos is None:
            return
        delta = event.position() - self._press_pos
        if not self._moved and math.hypot(delta.x(), delta.y()) < 3:
            return
        self._moved = True
        if self._dragging_endpoint is not None:
            transect, which = self._dragging_endpoint
            latlon = self.latlon_at(event.position())
            if which == "start":
                transect.start = latlon
            else:
                transect.end = latlon
            self.update()
            return
        assert self._press_center_tile is not None
        cx = self._press_center_tile[0] - delta.x() / TILE_SIZE
        cy = self._press_center_tile[1] - delta.y() / TILE_SIZE
        self._center = tile2deg(cx, cy, self._zoom)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._press_pos is None:
            return
        if self._dragging_endpoint is not None and self._moved:
            transect, which = self._dragging_endpoint
            latlon = transect.start if which == "start" else transect.end
            self.transect_endpoint_moved.emit(transect.id, which, latlon[0], latlon[1])
        elif not self._moved:
            hit = self._transect_at(event.position())
            if hit is not None:
                self.transect_clicked.emit(hit.id)
            else:
                lat, lon = self.latlon_at(event.position())
                self.map_clicked.emit(lat, lon)
        else:
            self.view_changed.emit()
        self._press_pos = None
        self._press_center_tile = None
        self._dragging_endpoint = None
        self._moved = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        step = 1 if event.angleDelta().y() > 0 else -1
        new_zoom = clamp_zoom(self._zoom + step)
        if new_zoom == self._zoom:
            return
        anchor = event.position()
        lat, lon = self.latlon_at(anchor)
        self._zoom = new_zoom
        # Keep the point under the cursor fixed while zooming.
        tx, ty = deg2tile(lat, lon, self._zoom)
        cx = tx - (anchor.x() - self.width() / 2) / TILE_SIZE
        cy = ty - (anchor.y() - self.height() / 2) / TILE_SIZE
        self._center = tile2deg(cx, cy, self._zoom)
        self.update()
        self.view_changed.emit()
