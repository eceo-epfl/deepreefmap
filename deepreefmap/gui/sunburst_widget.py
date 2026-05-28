from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, SupportsFloat, cast

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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
    class_ids: tuple[int, ...] = field(default_factory=tuple)


class SunburstWidget(QWidget):
    """Two-ring sunburst of benthic cover: outer = fine, inner = coarse.

    Hover dims non-hovered slices. Click emits selection_clicked with the
    class ids the slice covers (one id for a fine slice, group members for a
    coarse slice). Empty list = no slice under cursor.
    """

    selection_clicked = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fine_slices: tuple[_Slice, ...] = ()
        self._coarse_slices: tuple[_Slice, ...] = ()
        self._title: str = ""
        self._hover_ids: frozenset[int] = frozenset()
        # Persistent selection highlight: slices outside the selection are
        # dimmed so the donut mirrors the legend's current query. Inactive when
        # everything is selected (nothing to distinguish).
        self._selected_ids: frozenset[int] = frozenset()
        self._selection_active: bool = False
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setToolTipDuration(8000)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_cover(self, cover: dict[str, object] | None, classes_config: ClassConfig) -> None:
        if not cover or not isinstance(cover, dict):
            self._fine_slices = ()
            self._coarse_slices = ()
            self.update()
            return
        self._fine_slices = self._build_fine_slices(cover, classes_config)
        self._coarse_slices = self._build_coarse_slices(cover, classes_config)
        self.update()

    def has_data(self) -> bool:
        return bool(self._fine_slices)

    def set_selection(self, selected_ids: frozenset[int], active: bool) -> None:
        """Highlight the selected slices; dim the rest when `active`.

        `active` is False when the whole cloud is selected (nothing to
        distinguish), so the donut paints at full colour.
        """
        if (selected_ids, active) == (self._selected_ids, self._selection_active):
            return
        self._selected_ids = frozenset(selected_ids)
        self._selection_active = active
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
        for class_id_str, entry in cast("dict[str, dict[str, object]]", cover.get("classes") or {}).items():
            try:
                cid = int(class_id_str)
            except (TypeError, ValueError):
                continue
            frac = float(cast(SupportsFloat, entry.get("fraction", 0.0)))
            if frac <= 0:
                continue
            items.append(
                (cid, str(entry.get("name", f"class_{cid}")), frac, classes_config.color_for_id(cid))
            )
        items.sort(key=lambda r: r[2], reverse=True)
        return _angles_from_items(
            [(name, frac, color, (cid,)) for cid, name, frac, color in items]
        )

    @staticmethod
    def _build_coarse_slices(
        cover: dict[str, object], classes_config: ClassConfig
    ) -> tuple[_Slice, ...]:
        grouped = aggregate_cover(cover, classes_config, "coarse")
        group_members = {
            name: tuple(
                cls.id
                for cls in classes_config.classes
                if classes_config.group_name_for_id(cls.id, "coarse") == name
            )
            for name in grouped
        }
        items = [
            (
                name,
                float(payload["fraction"]),
                classes_config.group_color_for_name(name, "coarse"),
                group_members.get(name, ()),
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
            painter.setBrush(self._slice_brush(slc))
            painter.drawPie(outer_rect, int(slc.start_deg * 16), int(slc.span_deg * 16))
        painter.setBrush(self.palette().window())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(inner_rect)
        painter.setPen(pen)

        # Inner ring (coarse) — same trick: full pie then carve out center.
        for slc in self._coarse_slices:
            painter.setBrush(self._slice_brush(slc))
            painter.drawPie(coarse_outer_rect, int(slc.start_deg * 16), int(slc.span_deg * 16))
        painter.setBrush(self.palette().window())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(coarse_inner_rect)

        # Slice labels for the largest fine slices (anything below 4% is too
        # narrow to read without overlap). Drawn as light text with a dark
        # halo so they stay legible over any slice colour, dark or light.
        from math import cos, radians, sin

        font = QFont(painter.font())
        font.setPointSizeF(max(8.0, side * 0.020))
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for slc in self._fine_slices:
            if slc.fraction < _LABEL_MIN_FRACTION:
                continue
            mid_deg = slc.start_deg + slc.span_deg / 2
            label_radius = side * (_FINE_RING_OUTER + _FINE_RING_INNER) / 2 / 2
            theta = radians(mid_deg)
            lx = cx + label_radius * cos(theta)
            ly = cy - label_radius * sin(theta)
            text = f"{slc.name} {slc.fraction * 100:.1f}%"
            tw = metrics.horizontalAdvance(text)
            th = metrics.height()
            dimmed = bool(self._hover_ids) and not any(
                cid in self._hover_ids for cid in slc.class_ids
            )
            self._draw_halo_text(painter, lx - tw / 2, ly + th / 4, text, dimmed=dimmed)

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

    @staticmethod
    def _draw_halo_text(
        painter: QPainter, x: float, y: float, text: str, *, dimmed: bool = False
    ) -> None:
        """Draw `text` as light glyphs ringed by a dark halo for legibility."""
        fg = QColor(235, 238, 242, 130 if dimmed else 255)
        halo = QColor(0, 0, 0, 90 if dimmed else 200)
        painter.setPen(QPen(halo))
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)):
            painter.drawText(QPointF(x + dx, y + dy), text)
        painter.setPen(QPen(fg))
        painter.drawText(QPointF(x, y), text)

    # ----- hover + click -----

    @staticmethod
    def _dim(color: QColor, alpha: int) -> QColor:
        out = QColor(color)
        out.setAlpha(alpha)
        return out

    def _slice_brush(self, slc: _Slice) -> QColor:
        # Hover preview takes precedence over the persistent selection state.
        if self._hover_ids:
            if any(cid in self._hover_ids for cid in slc.class_ids):
                return slc.color
            return self._dim(slc.color, 70)
        if self._selection_active:
            in_sel = sum(1 for cid in slc.class_ids if cid in self._selected_ids)
            if in_sel == 0:
                return self._dim(slc.color, 55)
            if in_sel == len(slc.class_ids):
                return slc.color
            return self._dim(slc.color, 140)  # group only partly selected
        return slc.color

    def _slice_at(self, pos: QPointF) -> _Slice | None:
        from math import atan2, degrees, hypot

        bounds = QRectF(self.rect())
        cx = bounds.x() + bounds.width() / 2
        cy = bounds.y() + bounds.height() / 2
        side = min(bounds.width(), bounds.height()) - 8
        if side <= 16:
            return None
        dx = pos.x() - cx
        dy = cy - pos.y()
        radius = hypot(dx, dy)
        angle = degrees(atan2(dy, dx))
        if angle < 0:
            angle += 360.0

        outer_max = side * _FINE_RING_OUTER / 2
        outer_min = side * _FINE_RING_INNER / 2
        coarse_max = side * _COARSE_RING_OUTER / 2
        coarse_min = side * _COARSE_RING_INNER / 2

        if outer_min < radius < outer_max:
            return _slice_at_angle(self._fine_slices, angle)
        if coarse_min < radius < coarse_max:
            return _slice_at_angle(self._coarse_slices, angle)
        return None

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        slc = self._slice_at(event.position())
        ids = frozenset(slc.class_ids) if slc else frozenset()
        if ids != self._hover_ids:
            self._hover_ids = ids
            self.update()
        self.setToolTip(f"{slc.name}: {slc.fraction * 100:.2f}%" if slc else "")
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        if self._hover_ids:
            self._hover_ids = frozenset()
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            slc = self._slice_at(event.position())
            if slc is not None and slc.class_ids:
                self.selection_clicked.emit(list(slc.class_ids))
        super().mousePressEvent(event)


def _angles_from_items(
    items: Sequence[tuple[str, float, tuple[int, int, int], tuple[int, ...]]],
) -> tuple[_Slice, ...]:
    total = sum(max(0.0, frac) for _, frac, _, _ in items)
    if total <= 0:
        return ()
    slices: list[_Slice] = []
    cursor = 90.0  # Start at 12 o'clock so the largest slice runs into the top-right.
    for name, frac, color_rgb, class_ids in items:
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
                class_ids=tuple(int(c) for c in class_ids),
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
