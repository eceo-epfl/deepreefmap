"""QPainter-rendered icons for consistent, DPI-aware toolbar buttons."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap


def _px(size: int = 24, bg: QColor | None = None) -> tuple[QPixmap, QPainter]:
    pm = QPixmap(size, size)
    pm.fill(bg or Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    return pm, p


def crosshair_icon(size: int = 24, color: QColor | None = None) -> QIcon:
    c = color or QColor(230, 230, 230)
    pm, p = _px(size)
    pen = QPen(c, 1.6)
    p.setPen(pen)
    cx, cy = size / 2, size / 2
    r = size * 0.32
    gap = size * 0.08
    p.drawEllipse(QPointF(cx, cy), r, r)
    p.drawLine(QPointF(cx, cy - r - gap), QPointF(cx, cy - gap))
    p.drawLine(QPointF(cx, cy + gap), QPointF(cx, cy + r + gap))
    p.drawLine(QPointF(cx - r - gap, cy), QPointF(cx - gap, cy))
    p.drawLine(QPointF(cx + gap, cy), QPointF(cx + r + gap, cy))
    p.end()
    return QIcon(pm)


def refresh_icon(size: int = 24, color: QColor | None = None) -> QIcon:
    c = color or QColor(230, 230, 230)
    pm, p = _px(size)
    pen = QPen(c, 1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    cx, cy = size / 2, size / 2
    r = size * 0.3
    p.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r), 60 * 16, 270 * 16)
    ax, ay = cx + r * 0.5, cy - r
    a = size * 0.12
    p.drawLine(QPointF(ax - a, ay - a * 0.3), QPointF(ax, ay))
    p.drawLine(QPointF(ax, ay), QPointF(ax - a * 0.3, ay + a))
    p.end()
    return QIcon(pm)


def fit_icon(size: int = 24, color: QColor | None = None) -> QIcon:
    c = color or QColor(230, 230, 230)
    pm, p = _px(size)
    pen = QPen(c, 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    m = size * 0.2
    s = size - 2 * m
    corner_len = s * 0.3
    rect = QRectF(m, m, s, s)
    for cx, cy, dx, dy in [
        (rect.left(), rect.top(), 1, 1),
        (rect.right(), rect.top(), -1, 1),
        (rect.left(), rect.bottom(), 1, -1),
        (rect.right(), rect.bottom(), -1, -1),
    ]:
        p.drawLine(QPointF(cx, cy), QPointF(cx + dx * corner_len, cy))
        p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * corner_len))
    p.end()
    return QIcon(pm)


def help_icon(size: int = 24, color: QColor | None = None) -> QIcon:
    c = color or QColor(170, 170, 170)
    pm, p = _px(size)
    pen = QPen(c, 1.6)
    p.setPen(pen)
    cx, cy = size / 2, size / 2
    r = size * 0.36
    p.drawEllipse(QPointF(cx, cy), r, r)
    from PySide6.QtGui import QFont

    f = QFont()
    f.setPixelSize(int(size * 0.45))
    f.setBold(True)
    p.setFont(f)
    p.drawText(QRectF(0, -size * 0.03, size, size), Qt.AlignmentFlag.AlignCenter, "?")
    p.end()
    return QIcon(pm)


def check_icon(size: int = 16, color: QColor | None = None) -> QIcon:
    c = color or QColor(74, 170, 74)
    pm, p = _px(size)
    pen = QPen(c, 2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    m = size * 0.22
    p.drawLine(QPointF(m, size * 0.52), QPointF(size * 0.42, size - m))
    p.drawLine(QPointF(size * 0.42, size - m), QPointF(size - m, m))
    p.end()
    return QIcon(pm)


def download_icon(size: int = 16, color: QColor | None = None) -> QIcon:
    c = color or QColor(232, 160, 74)
    pm, p = _px(size)
    pen = QPen(c, 1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    cx = size / 2
    top = size * 0.15
    bot = size * 0.62
    p.drawLine(QPointF(cx, top), QPointF(cx, bot))
    a = size * 0.18
    p.drawLine(QPointF(cx - a, bot - a), QPointF(cx, bot))
    p.drawLine(QPointF(cx + a, bot - a), QPointF(cx, bot))
    base_y = size * 0.82
    p.drawLine(QPointF(size * 0.22, base_y), QPointF(size * 0.78, base_y))
    p.end()
    return QIcon(pm)


def lock_icon(size: int = 16, color: QColor | None = None) -> QIcon:
    c = color or QColor(232, 160, 74)
    pm, p = _px(size)
    pen = QPen(c, 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    cx = size / 2
    bw = size * 0.44
    bh = size * 0.34
    by = size * 0.52
    p.drawRect(QRectF(cx - bw / 2, by, bw, bh))
    ar = size * 0.22
    p.drawArc(QRectF(cx - ar, by - ar * 1.6, ar * 2, ar * 2), 0, 180 * 16)
    p.end()
    return QIcon(pm)
