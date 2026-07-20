"""Transect overlay model and pixel-space hit testing."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor

ENDPOINT_HIT_PX = 12.0
LINE_HIT_PX = 8.0


@dataclass
class OverlayTransect:
    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    color: QColor
    selected: bool = False


def segment_distance_px(point: QPointF, a: QPointF, b: QPointF) -> float:
    """Distance from ``point`` to the segment a-b, in pixels."""
    ax, ay = a.x(), a.y()
    dx, dy = b.x() - ax, b.y() - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return math.hypot(point.x() - ax, point.y() - ay)
    t = ((point.x() - ax) * dx + (point.y() - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(point.x() - (ax + t * dx), point.y() - (ay + t * dy))
