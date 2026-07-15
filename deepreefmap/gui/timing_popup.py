"""Compact stacked stage/total bars plus the floating per-stage breakdown.

The stage bar (top) and total bar (bottom) sit in one column so the two
percentages read as a unit and take half the width. Hovering the column pops the
breakdown, where measured stages show their real duration, the running stage its
live remainder, and pending stages a "~" estimate on later runs.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from deepreefmap.gui.eta import StageRow, format_duration
from deepreefmap.gui.theme import BORDER, GROOVE, PRIMARY, SUCCESS, TEXT_MUTED, WINDOW_TEXT

_STATE_COLOUR = {"done": SUCCESS, "running": PRIMARY, "pending": TEXT_MUTED}
_BAR_CELLS = 10  # width of the per-stage fill bar, in block glyphs


def _bar(frac: float, colour: str) -> str:
    """A small two-tone fill bar as block glyphs, coloured by stage state."""
    filled = max(0, min(_BAR_CELLS, round(_BAR_CELLS * frac)))
    return (
        f"<span style='color:{colour}'>{'█' * filled}</span>"
        f"<span style='color:{TEXT_MUTED}'>{'░' * (_BAR_CELLS - filled)}</span>"
    )


class HoverColumn(QWidget):
    """Container reporting cursor hover so the breakdown popup can follow the mouse.

    `hovered` carries the global cursor position while over the column and None on
    leave. Children set WA_TransparentForMouseEvents so the column sees the moves.
    """

    hovered = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)

    def enterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.hovered.emit(event.globalPosition())
        super().enterEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.hovered.emit(event.globalPosition())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.hovered.emit(None)
        super().leaveEvent(event)


class TimingPopup(QWidget):
    """Frameless popup rendering an estimator's stage rows as rich text."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        self._label = QLabel()
        self._label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._label)
        self.setStyleSheet(
            f"QWidget {{ background-color: {GROOVE}; border: 1px solid {BORDER};"
            f" border-radius: 4px; }} QLabel {{ color: {WINDOW_TEXT}; }}"
        )

    def set_rows(
        self,
        rows: list[StageRow],
        total_remaining_s: float | None,
        has_history: bool = True,
    ) -> None:
        cells = []
        elapsed_total = 0.0
        for row in rows:
            colour = _STATE_COLOUR.get(row.state, TEXT_MUTED)
            note = row.state
            if row.state in ("done", "running"):
                elapsed_total += row.seconds
                time_text = format_duration(row.seconds)
                # The running stage's remainder is measured, so show it always.
                if row.state == "running" and row.remaining is not None:
                    note = f"running · ~{format_duration(row.remaining)} left"
            elif has_history:
                time_text = f"~{format_duration(row.seconds)}" if row.seconds > 0 else "—"
            else:
                # First run: the pending "estimates" are not yet trustworthy, so
                # show the stages without inventing times for them.
                time_text = ""
            cells.append(
                f"<tr><td style='padding-right:12px'>{row.label}</td>"
                f"<td style='padding-right:12px;font-family:monospace'>{_bar(row.frac, colour)}</td>"
                # Fixed-width monospace so the count never reflows the column as
                # it ticks from 9s to 14s to 2m 03s and the popup stops jumping.
                f"<td align='right' width='64' "
                f"style='padding-right:14px;font-family:monospace'>{time_text}</td>"
                f"<td style='color:{colour}'>{note}</td></tr>"
            )
        if has_history:
            tail = (
                f" · ~{format_duration(total_remaining_s)} remaining"
                if total_remaining_s is not None
                else " · estimating…"
            )
        else:
            tail = " · learning timings on this machine"
        total_line = (
            f"<div style='margin-top:6px;color:{TEXT_MUTED}'>"
            f"Total {format_duration(elapsed_total)} elapsed{tail}</div>"
        )
        self._label.setText(f"<table cellspacing='2'>{''.join(cells)}</table>{total_line}")
        self.adjustSize()
