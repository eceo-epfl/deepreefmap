"""QPainter chart widgets for the survey analysis tab."""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget


def pass_color(direction: str, index: int) -> QColor:
    """Forward passes in teal shades, reverse in orange, darkening per pass."""
    hue = 190 if direction == "forward" else 25
    return QColor.fromHsv(hue, 180, max(110, 230 - index * 25))


class GroupedBarChart(QWidget):
    """Grouped bars: one group per label, one bar per series."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels: list[str] = []
        self._series: list[tuple[str, dict[str, float], QColor]] = []
        self.setMinimumHeight(240)

    def set_data(
        self, labels: list[str], series: list[tuple[str, dict[str, float], QColor]]
    ) -> None:
        self._labels = list(labels)
        self._series = list(series)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        text_color = self.palette().windowText().color()
        grid_color = QColor(text_color)
        grid_color.setAlpha(50)
        if not self._labels or not self._series:
            painter.setPen(text_color)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No completed passes yet.")
            return

        margin_left, margin_right, margin_top, margin_bottom = 44, 8, 26, 70
        plot = QRectF(
            margin_left,
            margin_top,
            max(1, self.width() - margin_left - margin_right),
            max(1, self.height() - margin_top - margin_bottom),
        )
        peak = max(
            (values.get(label, 0.0) for _, values, _ in self._series for label in self._labels),
            default=0.0,
        )
        top = max(0.05, math.ceil(peak * 20) / 20)

        painter.setPen(text_color)
        steps = 4
        for step in range(steps + 1):
            value = top * step / steps
            y = plot.bottom() - plot.height() * step / steps
            painter.setPen(grid_color)
            painter.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))
            painter.setPen(text_color)
            painter.drawText(
                QRectF(0, y - 8, margin_left - 6, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{value * 100:.0f}%",
            )

        group_width = plot.width() / len(self._labels)
        bar_width = group_width * 0.8 / len(self._series)
        for label_index, label in enumerate(self._labels):
            group_left = plot.left() + label_index * group_width + group_width * 0.1
            for series_index, (_, values, color) in enumerate(self._series):
                fraction = values.get(label, 0.0)
                height = plot.height() * (fraction / top)
                bar = QRectF(
                    group_left + series_index * bar_width,
                    plot.bottom() - height,
                    max(1.0, bar_width - 1),
                    height,
                )
                painter.fillRect(bar, color)
            painter.setPen(text_color)
            painter.save()
            painter.translate(group_left + group_width * 0.4, plot.bottom() + 6)
            painter.rotate(-45)
            painter.drawText(
                QRectF(-margin_bottom, 0, margin_bottom, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                painter.fontMetrics().elidedText(label, Qt.TextElideMode.ElideRight, margin_bottom),
            )
            painter.restore()

        legend_x = plot.left()
        for name, _, color in self._series:
            painter.fillRect(QRectF(legend_x, 6, 10, 10), color)
            text_width = painter.fontMetrics().horizontalAdvance(name) + 8
            painter.setPen(text_color)
            painter.drawText(
                QRectF(legend_x + 14, 2, text_width, 18),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                name,
            )
            legend_x += 14 + text_width + 8
