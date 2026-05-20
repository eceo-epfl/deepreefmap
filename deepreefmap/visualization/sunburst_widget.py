from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from deepreefmap.config.classes import ClassConfig
from deepreefmap.postproc.benthic_cover import aggregate_cover


_FINE_RING_OUTER = 1.00
_FINE_RING_INNER = 0.62
_COARSE_RING_OUTER = 0.60
_COARSE_RING_INNER = 0.34
_LABEL_MIN_FRACTION = 0.04  # Don't try to label slivers smaller than this.


@dataclass(frozen=True)
class _Slice:
    name: str
    fraction: float
    color: QColor
    start_deg: float
    span_deg: float


class SunburstWidget(QWidget):
    """Two-ring sunburst of benthic cover: outer = fine, inner = coarse.

    The widget repaints from `set_cover` data. No interactive editing — it's a
    read-only summary visualization that updates live with the transect crop.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fine_slices: tuple[_Slice, ...] = ()
        self._coarse_slices: tuple[_Slice, ...] = ()
        self._title: str = ""
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setToolTipDuration(8000)

    def set_cover(self, cover: dict[str, object] | None, classes_config: ClassConfig) -> None:
        if not cover or not isinstance(cover, dict):
            self._fine_slices = ()
            self._coarse_slices = ()
            self.update()
            return
        self._fine_slices = self._build_fine_slices(cover, classes_config)
        self._coarse_slices = self._build_coarse_slices(cover, classes_config)
        self.update()

    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def render_pixmap(self, size: int = 512) -> QPixmap:
        """Render the sunburst to an off-screen pixmap for export to PNG."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        try:
            self._paint(painter, QRectF(0, 0, size, size))
        finally:
            painter.end()
        return pixmap

    # ----- internal -----

    @staticmethod
    def _build_fine_slices(
        cover: dict[str, object], classes_config: ClassConfig
    ) -> tuple[_Slice, ...]:
        # Sort by fraction descending so the biggest slices land on the right
        # side of the ring — easier to read at a glance.
        items: list[tuple[int, str, float, tuple[int, int, int]]] = []
        for class_id_str, entry in (cover.get("classes") or {}).items():
            try:
                cid = int(class_id_str)
            except (TypeError, ValueError):
                continue
            frac = float(entry.get("fraction", 0.0))
            if frac <= 0:
                continue
            items.append(
                (cid, str(entry.get("name", f"class_{cid}")), frac, classes_config.color_for_id(cid))
            )
        items.sort(key=lambda r: r[2], reverse=True)
        return _angles_from_items([(name, frac, color) for _, name, frac, color in items])

    @staticmethod
    def _build_coarse_slices(
        cover: dict[str, object], classes_config: ClassConfig
    ) -> tuple[_Slice, ...]:
        grouped = aggregate_cover(cover, classes_config, "coarse")
        items = [
            (
                name,
                float(payload["fraction"]),
                classes_config.group_color_for_name(name, "coarse"),
            )
            for name, payload in grouped.items()
            if float(payload["fraction"]) > 0
        ]
        items.sort(key=lambda r: r[1], reverse=True)
        return _angles_from_items(items)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            self._paint(painter, QRectF(self.rect()))
        finally:
            painter.end()

    def _paint(self, painter: QPainter, bounds: QRectF) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        side = min(bounds.width(), bounds.height()) - 8
        if side <= 16:
            return
        cx = bounds.x() + bounds.width() / 2
        cy = bounds.y() + bounds.height() / 2
        outer_rect = _centered_square(cx, cy, side * _FINE_RING_OUTER)
        inner_rect = _centered_square(cx, cy, side * _FINE_RING_INNER)
        coarse_outer_rect = _centered_square(cx, cy, side * _COARSE_RING_OUTER)
        coarse_inner_rect = _centered_square(cx, cy, side * _COARSE_RING_INNER)

        pen = QPen(QColor(20, 20, 20))
        pen.setWidthF(0.7)
        painter.setPen(pen)

        # Outer ring: draw full pie then carve out the center via a background
        # pie to leave an annulus.
        for slc in self._fine_slices:
            painter.setBrush(slc.color)
            painter.drawPie(outer_rect, int(slc.start_deg * 16), int(slc.span_deg * 16))
        painter.setBrush(self.palette().window())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(inner_rect)
        painter.setPen(pen)

        # Inner ring (coarse) — same trick: full pie then carve out center.
        for slc in self._coarse_slices:
            painter.setBrush(slc.color)
            painter.drawPie(coarse_outer_rect, int(slc.start_deg * 16), int(slc.span_deg * 16))
        painter.setBrush(self.palette().window())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(coarse_inner_rect)

        # Slice labels for the largest fine slices (anything below 4% is too
        # narrow to read without overlap).
        font = QFont(painter.font())
        font.setPointSize(max(7, int(side * 0.018)))
        painter.setFont(font)
        painter.setPen(QPen(QColor(20, 20, 20)))
        for slc in self._fine_slices:
            if slc.fraction < _LABEL_MIN_FRACTION:
                continue
            mid_deg = slc.start_deg + slc.span_deg / 2
            label_radius = (
                side * (_FINE_RING_OUTER + _FINE_RING_INNER) / 2 / 2
            )
            from math import cos, radians, sin

            theta = radians(mid_deg)
            lx = cx + label_radius * cos(theta)
            ly = cy - label_radius * sin(theta)
            text = f"{slc.name} {slc.fraction * 100:.1f}%"
            metrics = painter.fontMetrics()
            tw = metrics.horizontalAdvance(text)
            th = metrics.height()
            painter.drawText(QPointF(lx - tw / 2, ly + th / 4), text)

        if self._title:
            painter.setPen(QPen(self.palette().text().color()))
            title_font = QFont(painter.font())
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.drawText(
                bounds,
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom),
                self._title,
            )

    # ----- hover tooltip -----

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        from math import atan2, degrees, hypot

        pos = event.position()
        bounds = QRectF(self.rect())
        cx = bounds.x() + bounds.width() / 2
        cy = bounds.y() + bounds.height() / 2
        side = min(bounds.width(), bounds.height()) - 8
        if side <= 16:
            return
        dx = pos.x() - cx
        dy = cy - pos.y()
        radius = hypot(dx, dy)
        # Convert to degrees in the same orientation QPainter uses (0 at 3
        # o'clock, increasing counter-clockwise).
        angle = degrees(atan2(dy, dx))
        if angle < 0:
            angle += 360.0

        outer_max = side * _FINE_RING_OUTER / 2
        outer_min = side * _FINE_RING_INNER / 2
        coarse_max = side * _COARSE_RING_OUTER / 2
        coarse_min = side * _COARSE_RING_INNER / 2

        slc: _Slice | None = None
        if outer_min < radius < outer_max:
            slc = _slice_at_angle(self._fine_slices, angle)
        elif coarse_min < radius < coarse_max:
            slc = _slice_at_angle(self._coarse_slices, angle)

        if slc is not None:
            self.setToolTip(f"{slc.name}: {slc.fraction * 100:.2f}%")
        else:
            self.setToolTip("")
        super().mouseMoveEvent(event)


def _angles_from_items(
    items: Sequence[tuple[str, float, tuple[int, int, int]]],
) -> tuple[_Slice, ...]:
    total = sum(max(0.0, frac) for _, frac, _ in items)
    if total <= 0:
        return ()
    slices: list[_Slice] = []
    cursor = 90.0  # Start at 12 o'clock so the largest slice runs into the top-right.
    for name, frac, color_rgb in items:
        if frac <= 0:
            continue
        span = -360.0 * (frac / total)  # Negative span = clockwise to read naturally.
        slices.append(
            _Slice(
                name=name,
                fraction=frac / total,
                color=QColor(*color_rgb),
                start_deg=cursor,
                span_deg=span,
            )
        )
        cursor += span
    return tuple(slices)


def _slice_at_angle(slices: Sequence[_Slice], angle_deg: float) -> _Slice | None:
    # QPainter angles increase counter-clockwise; our spans are negative
    # (clockwise). Reduce both to a normalized [0, 360) frame and test.
    for slc in slices:
        end = slc.start_deg + slc.span_deg
        lo = min(slc.start_deg, end) % 360.0
        hi = max(slc.start_deg, end) % 360.0
        if lo == hi:
            continue
        if lo < hi:
            if lo <= angle_deg < hi:
                return slc
        else:
            if angle_deg >= lo or angle_deg < hi:
                return slc
    return None


def _centered_square(cx: float, cy: float, side: float) -> QRectF:
    return QRectF(cx - side / 2, cy - side / 2, side, side)
