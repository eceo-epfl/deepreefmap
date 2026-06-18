"""Floating semantic legend overlay pinned to the 3D viewer canvas."""

from __future__ import annotations

from typing import Callable, cast

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.gui.core.theme import OVERLAY_TEXT
from deepreefmap.gui.viewer.render import _format_point_count


class LegendOverlay(QWidget):
    """Floating semi-transparent legend pinned to the top-right of the 3D canvas."""

    sort_clicked = Signal(str)
    master_clicked = Signal()
    # Without a host redraw on layout changes, stale pixels ghost through the
    # translucent panel until the camera next moves.
    repaint_requested = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # WA_StyledBackground lets the QSS background-color paint. Do NOT add
        # WA_TranslucentBackground: it is for top-level windows, and on a child over
        # a QOpenGLWidget on X11 it kills the QSS paint, leaving the overlay grey.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"""
            LegendOverlay {{
                background-color: rgba(20, 20, 20, 200);
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 6px;
            }}
            LegendOverlay QLabel#legend_title {{
                color: {OVERLAY_TEXT};
                font-size: 11px;
                font-weight: bold;
            }}
            LegendOverlay QLabel#legend_count {{
                color: #b8b8b8;
                font-size: 10px;
            }}
            LegendOverlay QToolButton#sort_header {{
                color: #cfd6dd;
                background: transparent;
                border: none;
                font-size: 10px;
                padding: 0px 2px;
            }}
            LegendOverlay QToolButton#sort_header:hover {{ color: #ffffff; }}
            LegendOverlay QCheckBox {{ color: {OVERLAY_TEXT}; font-size: 11px; spacing: 4px; }}
            LegendOverlay QCheckBox::indicator {{ width: 12px; height: 12px; }}
            LegendOverlay QScrollArea {{ background: transparent; border: none; }}
            LegendOverlay QWidget#legend_inner {{ background: transparent; }}
            LegendOverlay QToolButton {{
                color: {OVERLAY_TEXT};
                background-color: rgba(255, 255, 255, 20);
                border: 1px solid rgba(255, 255, 255, 60);
                border-radius: 3px;
                font-size: 10px;
                padding: 0px;
            }}
            LegendOverlay QToolButton:hover {{ background-color: rgba(255, 255, 255, 50); }}
            LegendOverlay QToolButton:pressed {{ background-color: rgba(255, 255, 255, 80); }}
            """
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self._title_label = QLabel("Legend")
        self._title_label.setObjectName("legend_title")
        header.addWidget(self._title_label, 1)
        self._minimize_btn = QToolButton()
        self._minimize_btn.setText("−")
        self._minimize_btn.setFixedSize(16, 16)
        self._minimize_btn.setToolTip("Collapse legend")
        self._minimize_btn.clicked.connect(self._toggle_minimized)
        header.addWidget(self._minimize_btn, 0)
        outer.addLayout(header)

        # Column headers above the list, laid out on the same grid as the rows
        # so they line up: [master checkbox + Name] | Points (over the counts).
        # The master checkbox toggles select-all/deselect-all (tri-state). Name
        # and Points are clickable sort headers (click again flips asc/desc;
        # active one is underlined with a ▲/▼ arrow).
        self._sort_row = QWidget()
        self._sort_grid = QGridLayout(self._sort_row)
        self._sort_grid.setContentsMargins(0, 0, 0, 0)
        self._sort_grid.setHorizontalSpacing(6)
        self._sort_grid.setColumnStretch(1, 1)
        self._sort_headers: dict[str, tuple[QToolButton, str]] = {}
        # Fixed 12px spacer matching the row swatch column so col 1 (the master
        # checkbox) lines up exactly with the row checkboxes below.
        col0_spacer = QWidget()
        col0_spacer.setFixedWidth(12)
        self._sort_grid.addWidget(col0_spacer, 0, 0)
        name_cell = QWidget()
        name_cell_layout = QHBoxLayout(name_cell)
        name_cell_layout.setContentsMargins(0, 0, 0, 0)
        name_cell_layout.setSpacing(4)
        self._master_check = QCheckBox()
        self._master_check.setTristate(True)
        self._master_check.setToolTip("Show all / hide all classes")
        self._master_check.clicked.connect(lambda _checked=False: self.master_clicked.emit())
        name_cell_layout.addWidget(self._master_check, 0)
        name_cell_layout.addWidget(self._make_sort_header("name", "Name"), 0)
        name_cell_layout.addStretch(1)
        self._sort_grid.addWidget(name_cell, 0, 1)
        self._sort_grid.addWidget(
            self._make_sort_header("size", "Points"), 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        # Empty col-3 cell mirroring the row "Only" button column so "Points"
        # lines up over the counts; its width is set from a real button in
        # rebuild(). Without a widget here the grid wouldn't reserve the column.
        self._sort_only_spacer = QWidget()
        self._sort_grid.addWidget(self._sort_only_spacer, 0, 3)
        outer.addWidget(self._sort_row, 0)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._inner = QWidget()
        self._inner.setObjectName("legend_inner")
        self._grid = QGridLayout(self._inner)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(2)
        self._grid.setColumnStretch(1, 1)
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll, 1)

        self._sunburst: QWidget | None = None
        self._sunburst_was_visible = False
        self._minimized = False
        # Per-class row widgets (swatch, checkbox, count, solo) so reorder() can
        # re-lay them out without recreating, which preserves checkbox state.
        self._rows: dict[int, tuple[QWidget, QCheckBox, QLabel, QToolButton]] = {}
        self.hide()

    def set_sunburst(self, widget: QWidget) -> None:
        """Dock a cover sunburst above the legend rows, inside this overlay."""
        if self._sunburst is widget:
            return
        widget.setParent(self)
        # Fixed height keeps the donut compact and makes the height budgeting in
        # reposition() deterministic, so it can never overlap the rows below.
        widget.setFixedHeight(170)
        cast("QVBoxLayout", self.layout()).insertWidget(1, widget, 0)
        self._sunburst = widget

    def _make_sort_header(self, key: str, label: str) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("sort_header")
        btn.setText(label)
        btn.setToolTip(f"Sort by {label.lower()} (click again to reverse)")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda _checked=False, k=key: self.sort_clicked.emit(k))
        self._sort_headers[key] = (btn, label)
        return btn

    def set_master_check_state(self, state: Qt.CheckState) -> None:
        """Set the header checkbox display state (blocked so it doesn't re-emit)."""
        self._master_check.blockSignals(True)
        self._master_check.setCheckState(state)
        self._master_check.blockSignals(False)

    def set_sort_indicator(self, key: str, ascending: bool) -> None:
        """Underline the active sort header and show its ▲/▼ direction arrow."""
        arrow = "▲" if ascending else "▼"
        for k, (btn, label) in self._sort_headers.items():
            font = btn.font()
            active = k == key
            btn.setText(f"{label} {arrow}" if active else label)
            font.setUnderline(active)
            btn.setFont(font)

    def _toggle_minimized(self) -> None:
        self._minimized = not self._minimized
        if self._minimized:
            # Remember whether the sunburst was showing (it's hidden on
            # geometry-only runs) so expanding restores that, not a blank donut.
            if self._sunburst is not None:
                self._sunburst_was_visible = self._sunburst.isVisibleTo(self)
                self._sunburst.setVisible(False)
            self._sort_row.setVisible(False)
            self._scroll.setVisible(False)
        else:
            self._scroll.setVisible(True)
            self._sort_row.setVisible(True)
            if self._sunburst is not None:
                self._sunburst.setVisible(self._sunburst_was_visible)
        self._minimize_btn.setText("+" if self._minimized else "−")
        self._minimize_btn.setToolTip(
            "Expand legend" if self._minimized else "Collapse legend"
        )
        self.reposition()

    def reorder(self, ordered_ids: list[int]) -> None:
        """Re-lay out the existing rows in `ordered_ids` order, no recreation."""
        for row_widgets in self._rows.values():
            for w in row_widgets:
                self._grid.removeWidget(w)
        row = 0
        for cid in ordered_ids:
            widgets = self._rows.get(cid)
            if widgets is None:
                continue
            for col, w in enumerate(widgets):
                self._grid.addWidget(w, row, col)
            row += 1
        self._inner.update()
        self.reposition()

    @staticmethod
    def _purge_grid(grid: QGridLayout) -> None:
        """Empty a grid, detaching widgets from the view now."""
        while grid.count():
            item = grid.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                # deleteLater alone leaves old rows painted until the event loop
                # runs, so a smaller rebuilt list ghosts on top of them.
                w.setParent(None)
                w.deleteLater()

    def clear(self) -> None:
        self._purge_grid(self._grid)

    def rebuild(
        self,
        class_ids: list[int],
        class_names: dict[int, str],
        class_colors: dict[int, tuple[int, int, int]],
        on_toggle: Callable[[], None],
        on_solo: Callable[[int], None] | None = None,
        class_counts: dict[int, int] | None = None,
    ) -> tuple[dict[int, QCheckBox], dict[int, QToolButton]]:
        """Populate one row per class present in the cloud; return (toggles, solo_buttons)."""
        self.clear()
        self._rows = {}
        toggles: dict[int, QCheckBox] = {}
        solo_buttons: dict[int, QToolButton] = {}
        if class_counts is not None:
            visible_ids = [cid for cid in class_ids if int(class_counts.get(cid, 0)) > 0]
        else:
            visible_ids = list(class_ids)
        for row, cid in enumerate(visible_ids):
            name = class_names.get(cid, str(cid))
            r, g, b = class_colors.get(cid, (128, 128, 128))
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); "
                "border: 1px solid rgba(255,255,255,80);"
            )
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.toggled.connect(on_toggle)
            count_label = QLabel()
            count_label.setObjectName("legend_count")
            count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if class_counts is not None and cid in class_counts:
                n = int(class_counts[cid])
                count_label.setText(_format_point_count(n))
                count_label.setToolTip(f"{n:,} points")
            solo = QToolButton()
            solo.setText("Only")
            solo.setFixedHeight(18)
            solo.setMinimumWidth(38)
            solo.setToolTip("Show only this class (click again to restore all)")
            if on_solo is not None:
                solo.clicked.connect(lambda _checked=False, c=cid: on_solo(c))
            self._grid.addWidget(swatch, row, 0)
            self._grid.addWidget(cb, row, 1)
            self._grid.addWidget(count_label, row, 2)
            self._grid.addWidget(solo, row, 3)
            self._rows[cid] = (swatch, cb, count_label, solo)
            toggles[cid] = cb
            solo_buttons[cid] = solo

        # Drive the scroll area's natural width from the inner content so
        # adjustSize() in reposition() picks up the correct width instead of
        # collapsing to QScrollArea's tiny default size hint.
        sb_w = self._scroll.verticalScrollBar().sizeHint().width()
        self._scroll.setMinimumWidth(self._inner.sizeHint().width() + sb_w + 4)

        # Align the column headers to the rows: reserve the "Only" column width
        # so "Points" sits over the counts, and reserve the scrollbar width on
        # the right so the header doesn't drift when the list scrolls.
        if visible_ids:
            first_solo = next(iter(solo_buttons.values()))
            self._sort_grid.setColumnMinimumWidth(3, first_solo.sizeHint().width())
        self._sort_grid.setContentsMargins(0, 0, sb_w, 0)

        # Record the full content height so reposition() can grow the scroll
        # area up to it when there's room, and give the scroll a small minimum
        # so it always yields under the height cap (scrolls instead of
        # overlapping the sunburst/pinned section above it).
        n_rows = len(visible_ids)
        self._grid.activate()
        inner_h = max(1, self._inner.sizeHint().height())
        self._list_content_h = inner_h + 4
        self._list_row_h = max(18, inner_h // n_rows) if n_rows else 18
        self._scroll.setMinimumHeight(min(self._list_content_h, 2 * self._list_row_h))
        return toggles, solo_buttons

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        cap_h = max(60, int(parent.height() * 0.85))
        self.setMaximumHeight(cap_h)
        self.setMaximumWidth(max(140, int(parent.width() * 0.5)))
        # Budget the main list's height to whatever remains under the cap once
        # the header and sunburst have taken their (bounded) space, so the stack
        # can't overflow the cap and overlap.
        if not self._minimized:
            chrome = 12 + 4  # outer top/bottom margins + a little spacing slack
            header_h = max(
                self._minimize_btn.sizeHint().height(), self._title_label.sizeHint().height()
            )
            used = chrome + header_h
            if self._sort_row.isVisibleTo(self):
                used += self._sort_row.sizeHint().height() + 4
            if self._sunburst is not None and self._sunburst.isVisibleTo(self):
                used += self._sunburst.height() + 4
            content_h = getattr(self, "_list_content_h", cap_h)
            floor = 2 * getattr(self, "_list_row_h", 18)
            scroll_h = max(floor, min(content_h, cap_h - used))
            self._scroll.setFixedHeight(scroll_h)
        self.adjustSize()
        margin = 8
        self.move(parent.width() - self.width() - margin, margin)
        self.raise_()
        # The overlay is translucent over the GL canvas; nudge the host to
        # re-render so a shrunk/regrouped layout doesn't ghost stale pixels.
        self.repaint_requested.emit()

    def showEvent(self, event):  # type: ignore[override]
        super().showEvent(event)
        self.reposition()
