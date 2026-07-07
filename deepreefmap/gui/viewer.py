from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, SupportsInt, cast

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

    from deepreefmap.config.classes import ClassConfig
    from deepreefmap.pipeline.artifacts import (
        FrameBatch,
        MappingSequenceResult,
        SemanticPointCloud,
    )
from PySide6.QtCore import QEvent, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.pointcloud.final_cloud_index import FinalCloudIndex, build_final_cloud_index
from deepreefmap.gui.live_frame_cloud import (
    LiveFrameCloudCache,
    build_enabled_label_lut,
    mask_points_by_enabled_lut,
)

logger = logging.getLogger(__name__)

_EMPTY_XYZ = np.zeros((0, 3), dtype=np.float32)


def _to_rgba(rgb: np.ndarray) -> np.ndarray:
    f = np.ascontiguousarray(rgb, dtype=np.float32)
    if f.max() > 1.0:
        f = f / 255.0
    return np.column_stack([f, np.ones(len(f), dtype=np.float32)])


def _colorize_seg(labels: np.ndarray, class_colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    h, w = labels.shape[:2]
    out = np.full((h, w, 3), 128, dtype=np.uint8)
    for cid, color in class_colors.items():
        out[labels == cid] = color
    return out


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    import cv2

    valid = np.isfinite(depth)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return out
    vals = depth[valid]
    lo, hi = np.percentile(vals, [2, 98])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    out[valid] = colored[valid][:, ::-1]
    out[~valid] = 0
    return out


def _build_frustum_lines(pose_w_c: np.ndarray, fov_y: float, aspect: float, scale: float = 0.04) -> np.ndarray:
    hy = np.tan(fov_y / 2) * scale
    hx = hy * aspect
    corners_cam = np.array([
        [-hx, -hy, scale],
        [hx, -hy, scale],
        [hx, hy, scale],
        [-hx, hy, scale],
    ], dtype=np.float64)
    R = pose_w_c[:3, :3]
    t = pose_w_c[:3, 3]
    corners_world = (R @ corners_cam.T).T + t
    origin = t
    lines = []
    for i in range(4):
        lines.append(origin)
        lines.append(corners_world[i])
    for i in range(4):
        lines.append(corners_world[i])
        lines.append(corners_world[(i + 1) % 4])
    return np.array(lines, dtype=np.float32)


def _estimate_world_up(
    positions: np.ndarray, cam_origins: np.ndarray | None
) -> tuple[float, float, float]:
    """Estimate the world "up" axis so camera frustums sit above the reef.

    The substrate is a roughly planar sheet, so its least-variance PCA axis is
    the surface normal. The recording cameras are physically above it, so we
    sign the normal to point from the cloud toward the camera origins. This is
    convention-independent: it works whether or not the poses were
    gravity-aligned, and is what keeps frustums on top in the default view.
    Falls back to +Y when there's not enough to infer a direction.
    """
    fallback = (0.0, 1.0, 0.0)
    if cam_origins is None:
        return fallback
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    cams = np.asarray(cam_origins, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3 or cams.shape[0] < 1:
        return fallback
    if pts.shape[0] > 50000:  # subsample huge clouds for a cheap SVD
        pts = pts[:: pts.shape[0] // 50000]
    centred = pts - pts.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return fallback
    normal = vh[-1]
    n = float(np.linalg.norm(normal))
    if n < 1e-9:
        return fallback
    normal = normal / n
    if float((cams.mean(axis=0) - pts.mean(axis=0)) @ normal) < 0.0:
        normal = -normal  # point toward the cameras (up)
    return (float(normal[0]), float(normal[1]), float(normal[2]))


def _compute_transect_view(
    positions: np.ndarray,
    cam_origins: np.ndarray | None,
    world_up: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return (camera_position, focal_point, up) for a transect-lengthwise view.

    The view is oriented so that the transect runs left-to-right on screen
    (start of recording on the left, end on the right) with world_up as
    screen-up — frustums, which sit at the recording camera origins, end up
    above the reef pointcloud when poses are gravity-aligned.

    Falls back to looking along world +Z if PCA fails or there isn't enough
    data to infer a direction.
    """
    up = np.asarray(world_up, dtype=np.float64)
    up_n = up / max(float(np.linalg.norm(up)), 1e-9)

    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), tuple(up_n.tolist()))  # type: ignore[return-value]
    center = pts.mean(axis=0)
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if not np.isfinite(extent) or extent <= 0.0:
        extent = 1.0

    def _principal_direction(samples: np.ndarray) -> np.ndarray | None:
        if samples.shape[0] < 2:
            return None
        centred = samples - samples.mean(axis=0)
        # Drop the up-component first so we only PCA the horizontal spread.
        centred = centred - np.outer(centred @ up_n, up_n)
        try:
            _, _, vh = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        v = vh[0]
        v = v - (v @ up_n) * up_n
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return None
        return v / n

    along: np.ndarray | None = None
    if cam_origins is not None:
        co = np.asarray(cam_origins, dtype=np.float64).reshape(-1, 3)
        along = _principal_direction(co)
        if along is not None and co.shape[0] >= 2:
            travel = co[-1] - co[0]
            travel = travel - (travel @ up_n) * up_n
            if float(travel @ along) < 0.0:
                along = -along
    if along is None:
        along = _principal_direction(pts)
    if along is None:
        return (
            tuple((center + np.array([0.0, 0.0, extent * 1.5])).tolist()),  # type: ignore[return-value]
            tuple(center.tolist()),  # type: ignore[return-value]
            tuple(up_n.tolist()),  # type: ignore[return-value]
        )

    # forward = camera look direction (into the scene). Right-handed:
    # right_world × up_world = forward_world. We want screen-right = along.
    forward = np.cross(up_n, along)
    n_fwd = float(np.linalg.norm(forward))
    if n_fwd < 1e-9:
        return (
            tuple((center + np.array([0.0, 0.0, extent * 1.5])).tolist()),  # type: ignore[return-value]
            tuple(center.tolist()),  # type: ignore[return-value]
            tuple(up_n.tolist()),  # type: ignore[return-value]
        )
    forward = forward / n_fwd
    cam_pos = center - extent * 1.5 * forward
    return (
        tuple(cam_pos.tolist()),  # type: ignore[return-value]
        tuple(center.tolist()),  # type: ignore[return-value]
        tuple(up_n.tolist()),  # type: ignore[return-value]
    )


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    f = arr.astype(np.float32)
    if f.size and f.max() <= 1.0 + 1e-6:
        f = f * 255.0
    return np.ascontiguousarray(np.clip(f, 0, 255).astype(np.uint8))


def _make_point_polydata(xyz: np.ndarray, rgb: np.ndarray) -> pv.PolyData:
    import pyvista as pv

    pts = np.ascontiguousarray(xyz, dtype=np.float32)
    pd = pv.PolyData(pts)
    pd["colors"] = _as_uint8_rgb(rgb)
    return pd


def _make_line_segments_polydata(points: np.ndarray) -> pv.PolyData:
    import pyvista as pv

    pts = np.ascontiguousarray(points, dtype=np.float32)
    n_segments = len(pts) // 2
    cells = np.empty((n_segments, 3), dtype=np.int64)
    cells[:, 0] = 2
    cells[:, 1] = np.arange(0, n_segments * 2, 2)
    cells[:, 2] = np.arange(1, n_segments * 2, 2)
    pd = pv.PolyData(pts)
    pd.lines = cells.ravel()
    return pd


def _format_point_count(n: int) -> str:
    """Compact human-readable count: 1234567 -> '1.23M', 4500 -> '4.5K'."""
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.0f}K"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class LegendOverlay(QWidget):
    """Floating semi-transparent legend pinned to the top-right of the 3D canvas.

    Populated by `rebuild()` with one row per class (swatch + checkbox + optional
    count + solo button). A header strip carries a sort selector and a
    minimize/expand toggle. Rows can be re-ordered in place via `reorder()`
    without recreating them. `reposition()` anchors the overlay to its parent's
    top-right corner and clamps the height to a fraction of the parent.

    `sort_clicked` fires with a column key ("name"/"size") when a sort header is
    clicked; `master_clicked` fires when the header's tri-state checkbox is
    clicked (the host decides select-all vs deselect-all). `repaint_requested`
    fires whenever the layout changes, so the host can force a redraw of the
    OpenGL canvas underneath (otherwise stale pixels ghost through the
    translucent panel until the camera moves).
    """

    sort_clicked = Signal(str)
    master_clicked = Signal()
    repaint_requested = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # WA_StyledBackground lets the QSS background-color paint. We
        # deliberately do NOT set WA_TranslucentBackground here: that
        # attribute is for top-level windows, and on child widgets layered
        # over a QOpenGLWidget (the pyvistaqt canvas) on Linux/X11 it
        # suppresses the QSS background paint, leaving the overlay showing
        # the default widget grey.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            LegendOverlay {
                background-color: rgba(20, 20, 20, 200);
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 6px;
            }
            LegendOverlay QLabel#legend_title {
                color: #e8e8e8;
                font-size: 11px;
                font-weight: bold;
            }
            LegendOverlay QLabel#legend_count {
                color: #b8b8b8;
                font-size: 10px;
            }
            LegendOverlay QToolButton#sort_header {
                color: #cfd6dd;
                background: transparent;
                border: none;
                font-size: 10px;
                padding: 0px 2px;
            }
            LegendOverlay QToolButton#sort_header:hover { color: #ffffff; }
            LegendOverlay QCheckBox { color: #e8e8e8; font-size: 11px; spacing: 4px; }
            LegendOverlay QCheckBox::indicator { width: 12px; height: 12px; }
            LegendOverlay QScrollArea { background: transparent; border: none; }
            LegendOverlay QWidget#legend_inner { background: transparent; }
            LegendOverlay QToolButton {
                color: #e8e8e8;
                background-color: rgba(255, 255, 255, 20);
                border: 1px solid rgba(255, 255, 255, 60);
                border-radius: 3px;
                font-size: 10px;
                padding: 0px;
            }
            LegendOverlay QToolButton:hover { background-color: rgba(255, 255, 255, 50); }
            LegendOverlay QToolButton:pressed { background-color: rgba(255, 255, 255, 80); }
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

        # Frustum visibility toggle used to live here; it's now in the
        # canvas toolbar overlay (qt_app_viewer_ctl._build_pick_mode_overlay).
        # The attribute is kept as a plain bool so existing code that reads
        # legend_overlay._frustum_check.isChecked() still works via the shim.
        self._frustum_visible = True

        self._sunburst: QWidget | None = None
        self._sunburst_was_visible = False
        self._minimized = False
        # Per-class row widgets (swatch, checkbox, count, solo) so reorder() can
        # re-lay them out without recreating — preserves checkbox state.
        self._rows: dict[int, tuple[QWidget, QCheckBox, QLabel, QToolButton]] = {}
        self.hide()

    def set_sunburst(self, widget: QWidget) -> None:
        """Dock a cover sunburst above the legend rows, inside this overlay.

        Stacked between the header (index 0) and the scroll area so clicking a
        pie slice and toggling a legend row share one panel. The donut is
        height-bounded so it stays compact and the scroll area keeps the rest.
        """
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
        """Empty a grid, detaching widgets from the view *now*.

        ``deleteLater`` alone leaves the old rows painted at their previous
        spots until the event loop deletes them, so a freshly rebuilt (smaller)
        list ghosts on top of them. Reparenting to None removes them from the
        display immediately; deleteLater then frees them.
        """
        while grid.count():
            item = grid.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
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
        """Populate one row per class; return (toggles, solo_buttons).

        `class_counts`, if given, adds a right-aligned count cell per row with
        a tooltip containing the full unformatted number. Classes with a
        zero (or missing) count are omitted entirely — the legend only shows
        classes actually present in the loaded cloud.
        """
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


class QtPointCloudViewer(QWidget):
    _sig_start_run = Signal(str, str)
    _sig_set_stage = Signal(str, str, object)
    _sig_update_progress = Signal(str, int, object, object, object)
    _sig_data_ready = Signal(object)
    _sig_mark_outputs = Signal(str, object)
    _sig_fail_run = Signal(str, str)
    _sig_close = Signal()

    point_picked = Signal(object)
    point_picked_clear = Signal()
    canvas_resized = Signal()
    frustum_picked = Signal(int)
    pick_mode_changed = Signal(bool)

    def __init__(
        self,
        class_colors: dict[int, tuple[int, int, int]] | None = None,
        class_names: dict[int, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._class_colors = class_colors or {}
        self._class_names = class_names or {}
        self._output_dir: Path | None = None

        self._rgb_label = QLabel()
        self._rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rgb_label.setMinimumHeight(120)
        self._rgb_label.setStyleSheet("background-color: #1a1a1a;")
        self._seg_label = QLabel()
        self._seg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._seg_label.setMinimumHeight(120)
        self._seg_label.setStyleSheet("background-color: #1a1a1a;")
        self._depth_label = QLabel()
        self._depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._depth_label.setMinimumHeight(120)
        self._depth_label.setStyleSheet("background-color: #1a1a1a;")
        self._frames_panel = QWidget()
        frames_outer = QVBoxLayout(self._frames_panel)
        frames_outer.setContentsMargins(0, 0, 0, 0)
        frames_outer.setSpacing(0)
        frames_row = QWidget()
        frames_layout = QHBoxLayout(frames_row)
        frames_layout.setContentsMargins(0, 0, 0, 0)
        frames_layout.setSpacing(0)
        frames_layout.addWidget(self._rgb_label, 1)
        frames_layout.addWidget(self._seg_label, 1)
        frames_layout.addWidget(self._depth_label, 1)
        frames_outer.addWidget(frames_row, 1)

        # Slider bar: a fat, hard-to-miss timeline control with a Frame N / N
        # readout to the right. The slider is the primary way the user scrubs
        # through the reconstruction, so we give it a tall handle, a clear
        # groove, and tick marks.
        slider_row = QWidget()
        slider_row.setStyleSheet("background-color: #202020;")
        slider_layout = QHBoxLayout(slider_row)
        slider_layout.setContentsMargins(8, 4, 8, 6)
        slider_layout.setSpacing(8)
        slider_label = QLabel("Frame")
        slider_label.setStyleSheet("color: #ccc; font-weight: bold;")
        slider_layout.addWidget(slider_label)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setMinimumHeight(34)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frame_slider.setTickInterval(0)
        self.frame_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 10px;
                background: #3a3a3a;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QSlider::sub-page:horizontal {
                background: #4aa3ff;
                border: 1px solid #2a78c8;
                border-radius: 5px;
            }
            QSlider::add-page:horizontal {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #f0f0f0;
                border: 2px solid #2a78c8;
                width: 18px;
                height: 26px;
                margin: -10px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal:hover { background: #ffffff; }
            QSlider::tick:horizontal { background: #777; }
            """
        )
        slider_layout.addWidget(self.frame_slider, 1)
        self._frame_readout = QLabel("0 / 0")
        self._frame_readout.setStyleSheet(
            'color: #e8e8e8; font-family: "JetBrains Mono"; min-width: 80px;'
        )
        self._frame_readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider_layout.addWidget(self._frame_readout)
        # Keep the readout in sync with the slider regardless of who moves it.
        self.frame_slider.valueChanged.connect(self._update_frame_readout)
        self.frame_slider.rangeChanged.connect(
            lambda _lo, _hi: self._update_frame_readout(self.frame_slider.value())
        )
        frames_outer.addWidget(slider_row)

        self._main_splitter = QSplitter(Qt.Orientation.Vertical)
        self._canvas_container = QWidget()
        self._canvas_layout = QVBoxLayout(self._canvas_container)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._main_splitter.addWidget(self._canvas_container)
        self._main_splitter.addWidget(self._frames_panel)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 1)
        self._canvas_revealed = False
        self._canvas_container.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._main_splitter)

        # Floating legend pinned to the canvas's top-right corner. Hidden until
        # _build_legend in the launcher populates it after a run loads.
        self.legend_overlay = LegendOverlay(self._canvas_container)
        self.legend_overlay.repaint_requested.connect(self._render_canvas_safe)
        self._canvas_container.installEventFilter(self)

        self._plotter: Any = None

        self._simple_actor: Any = None
        self._live_actor: Any = None
        self._live_polydata: pv.PolyData | None = None
        self._class_actors: dict[int, Any] = {}
        self._class_polydata: dict[int, pv.PolyData] = {}
        self._frustum_actors: dict[int, Any] = {}
        self._frustum_batch_actor: Any = None
        self._frustum_batch_pd: Any = None
        self._frustum_highlight_actor: Any = None
        self._frustum_highlight_pd: Any = None
        self._frustum_frame_ids: list[int] = []
        self._frustum_fid_to_idx: dict[int, int] = {}
        self._frustum_all_pts: list[np.ndarray] = []
        self._frustum_pts_per: int = 16

        self._final_index: FinalCloudIndex | None = None
        self._live_cache: LiveFrameCloudCache | None = None
        self._frame_batch: FrameBatch | None = None
        self._mapping_result: MappingSequenceResult | None = None
        self._max_label_id = 0
        # Geometry-only mode: a single static RGB cloud with a frustum/image
        # timeline but no semantic per-class partitioning (no FinalCloudIndex).
        self._geometry_mode = False
        self._geometry_frame_order: list[int] = []
        self._geometry_xyz: np.ndarray | None = None

        # Cached (position, focal_point, up) for the default transect view,
        # computed once at load so reset_view can reapply it without re-running
        # the SVD-based orientation fit over the whole cloud on every click.
        self._fit_camera_params: (
            tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
            | None
        ) = None

        self._last_t: int | None = None
        self._last_accumulate: bool | None = None
        self._last_semantic: bool | None = None
        self._last_enabled: frozenset[int] | None = None
        self._last_confidence: float | None = None
        self._last_point_size: float | None = None

        self._point_filter: Callable[[np.ndarray], np.ndarray] | None = None

        self._pick_2d_actors: list[object] = []
        self._pick_line_sources: list[Any] = []
        self._pick_ring_sources: list[Any] = []
        # Crosshair ticks stored as (line_source, ox1, oy1, ox2, oy2): pixel
        # offsets from the anchor so update_pick_anchor can reposition them.
        self._pick_tick_sources: list[tuple[Any, float, float, float, float]] = []
        self._picked_xyz: tuple[float, float, float] | None = None
        self._picked_color: tuple[int, int, int] = (255, 220, 60)
        self._picked_leader_target: tuple[float, float] | None = None
        self._pick_camera_obs_id: int | None = None
        self._pick_mode_enabled: bool = False
        self._pick_press_pos: tuple[int, int] | None = None
        self._pick_drag_detected: bool = False

        self._frame_panel_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        self._sig_start_run.connect(self._on_start_run)
        self._sig_set_stage.connect(self._on_set_stage)
        self._sig_update_progress.connect(self._on_update_progress)
        self._sig_data_ready.connect(self._on_data_ready)
        self._sig_mark_outputs.connect(self._on_mark_outputs)
        self._sig_fail_run.connect(self._on_fail_run)
        self._sig_close.connect(self._on_close)

        self._status_callback: Callable[..., None] | None = None

    def _update_frame_readout(self, value: int) -> None:
        total = max(0, self.frame_slider.maximum())
        # Display 1-indexed (matches video-frame numbering users expect) but
        # only when a range is available; otherwise show 0 / 0.
        if total <= 0:
            self._frame_readout.setText("0 / 0")
        else:
            self._frame_readout.setText(f"{int(value) + 1} / {total + 1}")

    def set_status_callback(self, cb: Callable[..., None]) -> None:
        self._status_callback = cb

    def set_point_filter(
        self, fn: Callable[[np.ndarray], np.ndarray] | None
    ) -> None:
        """Install an xyz→bool mask filter applied to every point cloud update."""
        self._point_filter = fn
        # Invalidate apply_state's idempotency cache so the next call repaints
        # every actor through the new filter.
        self._last_t = None

    _PICK_DRAG_THRESHOLD = 4
    # Gap-forgiveness radius (px) for the depth pick: if the exact click pixel
    # falls between point sprites, the nearest covered pixel to the cursor
    # within this window is used, so picks stay accurate without needing a
    # pixel-perfect hit.
    _PICK_PIXEL_RADIUS = 8

    # Selected-point reticle (fixed screen-space pixels): a thin hollow ring
    # plus 4 crosshair ticks with an open centre, so the picked point and its
    # neighbours stay visible instead of being hidden under a filled blob.
    _PICK_RING_RADIUS = 11.0
    _PICK_TICK_INNER = 6.0
    _PICK_TICK_OUTER = 14.0

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self._canvas_container and event.type() == QEvent.Type.Resize:
            self.legend_overlay.reposition()
            self.canvas_resized.emit()
            self._schedule_canvas_repaint()
        if obj is self.window() and event.type() in (
            QEvent.Type.Move,
            QEvent.Type.Resize,
            QEvent.Type.WindowActivate,
        ):
            self._schedule_canvas_repaint()
        if (
            obj is self._canvas_container
            and event.type() == QEvent.Type.MouseButtonDblClick
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._recenter_on_click(event)

        # Pick-mode click-vs-drag detection on the plotter widget. The pick
        # itself runs on release (only if the mouse didn't move more than
        # _PICK_DRAG_THRESHOLD pixels) so it can't be mistaken for an orbit.
        if obj is self._plotter and self._pick_mode_enabled:
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                pos = event.position().toPoint()
                self._pick_press_pos = (pos.x(), pos.y())
                self._pick_drag_detected = False
            elif etype == QEvent.Type.MouseMove and self._pick_press_pos is not None:
                pos = event.position().toPoint()
                dx = pos.x() - self._pick_press_pos[0]
                dy = pos.y() - self._pick_press_pos[1]
                if abs(dx) + abs(dy) > self._PICK_DRAG_THRESHOLD:
                    self._pick_drag_detected = True
            elif etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if not self._pick_drag_detected:
                    pos = event.position().toPoint()
                    res = self._pick_at(pos.x(), pos.y())
                    if res is not None:
                        self._process_pick(*res)
                    else:
                        self._on_pick_miss()
                self._pick_press_pos = None
        return super().eventFilter(obj, event)

    def _recenter_on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-center the orbit pivot on the world point under the cursor."""
        if self._plotter is None:
            return
        try:
            from vtkmodules.vtkRenderingCore import vtkWorldPointPicker

            qt_x, qt_y = event.position().x(), event.position().y()
            h = self._plotter.renderer.GetSize()[1]
            vtk_x, vtk_y = float(qt_x), float(h - qt_y)
            wp = vtkWorldPointPicker()
            wp.Pick(vtk_x, vtk_y, 0.0, self._plotter.renderer)
            picked = np.asarray(wp.GetPickPosition(), dtype=np.float64)
            if np.all(np.abs(picked) < 1e-12):
                return
            self._plotter.camera.focal_point = tuple(picked.tolist())
            self._plotter.reset_camera_clipping_range()
            self._plotter.render()
        except Exception:
            logger.debug("Double-click re-center failed", exc_info=True)

    def _ensure_plotter(self):
        if self._plotter is not None:
            return self._plotter
        from pyvistaqt import QtInteractor

        self._plotter = QtInteractor(self._canvas_container)
        self._plotter.set_background("#141414")
        self._plotter.iren.enable_custom_trackball_style(
            left="rotate",
            shift_left="pan",
            control_left="dolly",
            middle="pan",
            right="pan",
            shift_right="pan",
            control_right="dolly",
        )
        try:
            self._plotter.enable_eye_dome_lighting()
        except Exception:
            logger.debug("Eye dome lighting unavailable", exc_info=True)
        try:
            self._plotter.add_axes(
                interactive=False,
                line_width=3,
                color="white",
                x_color="#ff5a5a",
                y_color="#5aff7a",
                z_color="#5aaaff",
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
                viewport=(0.82, 0.0, 1.0, 0.18),
            )
        except Exception:
            logger.debug("Axes widget unavailable", exc_info=True)
        self._canvas_layout.addWidget(self._plotter)
        self._plotter.installEventFilter(self)
        # Keep the legend on top after the plotter is added below it.
        self.legend_overlay.raise_()
        self._install_scroll_zoom()
        # QtInteractor paints its GL surface straight to screen, so moving or
        # resizing the top-level window leaves the translucent overlays' old
        # background smeared over the viewport (Qt never recomposites the area
        # behind them). Watch the window for move/resize to force a full VTK
        # re-render, which repaints the whole viewport and clears the trails.
        window = self.window()
        if window is not None and not getattr(self, "_window_filter_installed", False):
            window.installEventFilter(self)
            self._window_filter_installed = True
        return self._plotter

    def _schedule_canvas_repaint(self) -> None:
        """Coalesce a burst of move/resize events into one delayed repaint.

        Move/resize fire continuously while the user drags; rendering a large
        cloud on each would stutter. A short single-shot timer collapses the
        burst so we re-render once the motion settles.
        """
        timer = getattr(self, "_repaint_timer", None)
        if timer is None:
            from PySide6.QtCore import QTimer

            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(40)
            timer.timeout.connect(self._force_canvas_repaint)
            self._repaint_timer = timer
        timer.start()

    def _force_canvas_repaint(self) -> None:
        """Repaint the VTK viewport and re-raise the overlays on top of it."""
        if self._plotter is None:
            return
        try:
            self._plotter.render()
        except Exception:
            return
        self.legend_overlay.raise_()
        self.legend_overlay.update()
        for child in self._canvas_container.findChildren(QWidget):
            if child.objectName() == "pick_mode_overlay":
                child.raise_()
                child.update()

    def _install_scroll_zoom(self) -> None:
        """Scroll-wheel zoom toward the cursor position instead of screen center.

        Observers are on the interactor (not the style) so zoom works in both
        navigate and pick mode.
        """
        plotter = self._plotter
        if plotter is None:
            return
        zoom_speed = 0.10

        def _zoom(obj, event):  # type: ignore[no-untyped-def]
            forward = event == "MouseWheelForwardEvent"
            direction = 1.0 if forward else -1.0
            camera = plotter.camera
            cam_pos = np.asarray(camera.position, dtype=np.float64)
            focal = np.asarray(camera.focal_point, dtype=np.float64)

            x, y = plotter.iren.interactor.GetEventPosition()
            try:
                from vtkmodules.vtkRenderingCore import vtkWorldPointPicker

                wp = vtkWorldPointPicker()
                wp.Pick(float(x), float(y), 0.0, plotter.renderer)
                picked = np.asarray(wp.GetPickPosition(), dtype=np.float64)
            except Exception:
                picked = focal

            target = picked if np.any(np.abs(picked) > 1e-12) else focal

            view_vec = cam_pos - target
            dist = float(np.linalg.norm(view_vec))
            if dist < 1e-9:
                return
            step = dist * zoom_speed * direction
            new_dist = max(dist - step, dist * 0.01)
            camera.position = tuple((target + (view_vec / dist) * new_dist).tolist())

            # Keep focal point between camera and target to prevent orbit inversion.
            cam_to_focal = np.asarray(camera.focal_point, dtype=np.float64) - np.asarray(
                camera.position, dtype=np.float64
            )
            if float(np.dot(cam_to_focal, view_vec)) > 0:
                camera.focal_point = tuple(
                    (np.asarray(camera.position, dtype=np.float64) + cam_to_focal * 0.5).tolist()
                )

            plotter.reset_camera_clipping_range()
            plotter.render()

        iren = plotter.iren
        iren.add_observer("MouseWheelForwardEvent", _zoom)
        iren.add_observer("MouseWheelBackwardEvent", _zoom)

    def _iter_pickable_actors(self):
        """Yield actors that _process_pick can resolve: frustums + class clouds.

        These hold the final, indexed cloud; the live/simple actors aren't in
        _process_pick's lookup tables, so picking them never resolved anyway.
        """
        if self._frustum_batch_actor is not None:
            yield self._frustum_batch_actor
        yield from self._frustum_actors.values()
        yield from self._class_actors.values()

    @staticmethod
    def _select_pick_pixel(
        z: np.ndarray, cursor_local: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Choose the foreground z-buffer pixel nearest the cursor.

        ``z`` has shape ``(ny, nx)``; ``z[j, i]`` is the depth at local column
        ``i``, row ``j`` (VTK's bottom-left origin). ``cursor_local`` is the
        click's ``(col, row)`` within the window. A pixel is foreground when
        its depth is ``< 1.0`` (something was rendered there). Returns the
        chosen ``(col, row)`` or ``None`` if the whole window is background.
        Ties in pixel distance break toward the nearer (smaller depth) pixel.
        """
        foreground = z < 1.0 - 1e-6
        if not foreground.any():
            return None
        ny, nx = z.shape
        ci, cj = cursor_local
        jj, ii = np.mgrid[0:ny, 0:nx]
        dist2 = (ii - ci).astype(np.float64) ** 2 + (jj - cj).astype(np.float64) ** 2
        # Primary key pixel distance; +depth as a sub-unit tiebreak (z in [0,1)
        # so it never outweighs an integer-spaced distance difference).
        score = np.where(foreground, dist2 + 1e-3 * z, np.inf)
        j, i = np.unravel_index(int(np.argmin(score)), score.shape)
        return int(i), int(j)

    def _pick_at(self, qt_x, qt_y):  # type: ignore[no-untyped-def]
        """Find the rendered point nearest the cursor via the depth buffer.

        Reads the z-buffer in a small window around the click, picks the
        foreground pixel closest to the cursor, unprojects it to a world point,
        and snaps to the nearest point across the visible pickable actors.
        Returns ``(dataset, point_id)`` for _process_pick, or ``None`` on a
        miss (click landed on the background). Unlike a ray+tolerance picker,
        this honours occlusion and the view angle.
        """
        if self._plotter is None:
            return None
        try:
            from vtkmodules.util.numpy_support import vtk_to_numpy
            from vtkmodules.vtkCommonCore import vtkFloatArray
            from vtkmodules.vtkRenderingCore import vtkWorldPointPicker

            ren = self._plotter.renderer
            win = self._plotter.render_window
            w, h = ren.GetSize()
            if w <= 0 or h <= 0:
                return None
            cx, cy = int(qt_x), int(h - int(qt_y))  # Qt top-left -> VTK bottom-left
            r = self._PICK_PIXEL_RADIUS
            x0, x1 = max(0, cx - r), min(w - 1, cx + r)
            y0, y1 = max(0, cy - r), min(h - 1, cy + r)
            if x1 < x0 or y1 < y0:
                return None
            nx, ny = x1 - x0 + 1, y1 - y0 + 1
            zarr = vtkFloatArray()
            win.GetZbufferData(x0, y0, x1, y1, zarr)
            raw = vtk_to_numpy(zarr)
            if raw.size != nx * ny:
                return None
            z = raw.reshape(ny, nx)  # row 0 == y0 (bottom)
            sel = self._select_pick_pixel(z, (cx - x0, cy - y0))
            if sel is None:
                return None
            wp = vtkWorldPointPicker()
            wp.Pick(float(x0 + sel[0]), float(y0 + sel[1]), 0.0, ren)
            world = np.asarray(wp.GetPickPosition(), dtype=np.float64)

            best = None  # (dist_sq, dataset, point_id) of the closest point so far
            for actor in self._iter_pickable_actors():
                try:
                    if actor is None or not actor.GetVisibility():
                        continue
                    mapper = actor.GetMapper()
                    ds = mapper.GetInput() if mapper is not None else None
                    if ds is None or ds.GetNumberOfPoints() == 0:
                        continue
                    pid = ds.FindPoint((float(world[0]), float(world[1]), float(world[2])))
                    if pid < 0:
                        continue
                    pt = np.asarray(ds.GetPoint(int(pid)), dtype=np.float64) - world
                    d2 = float(pt @ pt)
                    if best is None or d2 < best[0]:
                        best = (d2, ds, int(pid))
                except Exception:
                    continue
            if best is None:
                return None
            return best[1], best[2]
        except Exception:
            logger.debug("depth pick failed", exc_info=True)
            return None

    def _render_canvas_safe(self) -> None:
        """Force a GL re-render (e.g. after the legend overlay relayouts).

        Without this, the translucent overlay leaves stale pixels ghosting on
        the GL surface until the camera moves and triggers a redraw.
        """
        if self._plotter is None:
            return
        try:
            self._plotter.render()
        except Exception:
            pass

    def set_pick_mode(self, enabled: bool) -> None:
        """Toggle pick mode.

        While enabled: left-click picks a point (distinguished from drag by a
        pixel threshold in eventFilter). Orbit, pan, and zoom remain active —
        the trackball style is NOT swapped out. Cursor shows a crosshair to
        indicate that clicks will pick.
        """
        enabled = bool(enabled)
        if enabled == self._pick_mode_enabled:
            return
        if enabled:
            try:
                self._canvas_container.setCursor(Qt.CursorShape.CrossCursor)
            except Exception:
                pass
            self._pick_mode_enabled = True
        else:
            self._pick_press_pos = None
            try:
                self._canvas_container.unsetCursor()
            except Exception:
                pass
            self._pick_mode_enabled = False
        self.pick_mode_changed.emit(self._pick_mode_enabled)

    def _on_pick_miss(self) -> None:
        """Left-click in pick mode landed on the background (no point under cursor)."""
        self.point_picked_clear.emit()
        if self._status_callback is not None:
            cb = self._status_callback
            cb("No point under cursor")
            QTimer.singleShot(1500, lambda: cb(""))

    def _process_pick(self, mesh, point_id) -> None:  # type: ignore[no-untyped-def]
        """Commit a deferred pick after confirming it was a click, not a drag."""
        if self._final_index is None or self._plotter is None:
            self.point_picked_clear.emit()
            return

        # Batched frustum pick: identify which frustum by point index
        if self._frustum_batch_actor is not None:
            try:
                mapper = self._frustum_batch_actor.GetMapper()
                if mapper is not None and mapper.GetInput() is mesh and point_id is not None:
                    frustum_idx = int(point_id) // self._frustum_pts_per
                    if 0 <= frustum_idx < len(self._frustum_frame_ids):
                        self.frustum_picked.emit(self._frustum_frame_ids[frustum_idx])
                        self.point_picked_clear.emit()
                        return
            except Exception:
                pass

        # Legacy per-actor frustum pick
        for fid, actor in self._frustum_actors.items():
            try:
                mapper = actor.GetMapper()
                if mapper is not None and mapper.GetInput() is mesh:
                    self.frustum_picked.emit(int(fid))
                    self.point_picked_clear.emit()
                    return
            except Exception:
                continue

        picked_cid: int | None = None
        for cid, actor in self._class_actors.items():
            try:
                mapper = actor.GetMapper()
                if mapper is None:
                    continue
                if mapper.GetInput() is mesh:
                    if actor.GetVisibility():
                        picked_cid = cid
                    break
            except Exception:
                continue
        if picked_cid is None:
            self._on_pick_miss()
            return

        fi = self._final_index
        pid = int(point_id)

        orig_idx_arr = None
        try:
            if "orig_idx" in mesh.point_data:
                orig_idx_arr = np.asarray(mesh.point_data["orig_idx"], dtype=np.int64)
        except Exception:
            orig_idx_arr = None
        if orig_idx_arr is not None and 0 <= pid < orig_idx_arr.shape[0]:
            local_id = int(orig_idx_arr[pid])
        else:
            local_id = pid

        xyz_c = fi.xyz_by_class.get(picked_cid)
        conf_c = fi.conf_by_class.get(picked_cid)
        if xyz_c is None or local_id < 0 or local_id >= xyz_c.shape[0]:
            self.point_picked_clear.emit()
            return

        xyz = xyz_c[local_id]
        confidence = (
            float(conf_c[local_id])
            if conf_c is not None and local_id < conf_c.shape[0]
            else float("nan")
        )

        prefix_end = fi.prefix_end_by_class.get(picked_cid)
        frame_index = -1
        if prefix_end is not None and len(fi.frame_order) > 0:
            t = int(np.searchsorted(prefix_end, local_id, side="right"))
            if 0 <= t < len(fi.frame_order):
                frame_index = int(fi.frame_order[t])

        color = self._class_colors.get(picked_cid, (180, 180, 180))
        name = self._class_names.get(picked_cid, f"class {picked_cid}")

        screen_xy = (0, 0)
        try:
            pos = self._plotter.iren.GetEventPosition()
            plotter_h = int(self._plotter.height())
            screen_xy = (int(pos[0]), max(0, plotter_h - int(pos[1])))
        except Exception:
            pass

        payload = {
            "class_id": int(picked_cid),
            "class_name": str(name),
            "color": (int(color[0]), int(color[1]), int(color[2])),
            "xyz": (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            "frame_index": int(frame_index),
            "confidence": confidence,
            "screen_xy": screen_xy,
        }
        self.point_picked.emit(payload)

    def set_picked_marker(
        self,
        xyz: tuple[float, float, float],
        color: tuple[int, int, int],
        anchor_display: tuple[float, float] | None = None,
        leader_target_display: tuple[float, float] | None = None,
    ) -> None:
        """Mark the picked point with a screen-space crosshair + hollow ring.

        The reticle's open centre keeps the picked point and its neighbours
        visible. An optional 2D leader line connects it to a screen-space
        tooltip card. The world XYZ + leader target are cached so a
        camera-modified observer can redraw the 2D overlay with a recomputed
        anchor as the user orbits.

        On repeated calls with the same picked XYZ this only mutates the
        cached 2D source geometry — no actor allocation, no recreate, no
        flicker.
        """
        if self._plotter is None:
            return
        new_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        actors_present = bool(self._pick_2d_actors)
        if (
            actors_present
            and self._picked_xyz is not None
            and self._picked_xyz == new_xyz
        ):
            if anchor_display is not None:
                self.update_pick_anchor(anchor_display, leader_target_display)
            return
        self._build_pick_actors(new_xyz, color, anchor_display, leader_target_display)

    def _build_pick_actors(
        self,
        xyz: tuple[float, float, float],
        color: tuple[int, int, int],
        anchor_display: tuple[float, float] | None,
        leader_target_display: tuple[float, float] | None,
    ) -> None:
        if self._plotter is None:
            return
        self.clear_picked_marker()
        r, g, b = color
        self._picked_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self._picked_color = (int(r), int(g), int(b))
        self._picked_leader_target = leader_target_display
        if anchor_display is not None:
            self._add_pick_2d_overlay(
                anchor_display,
                leader_target_display,
                (r, g, b),
            )
        self._install_pick_camera_observer()
        try:
            self._plotter.render()
            # Some pyvistaqt builds need an extra render-window flush before
            # the 2D actors actually appear on screen.
            self._plotter.render_window.Render()
        except Exception:
            pass

    def update_pick_anchor(
        self,
        anchor_display: tuple[float, float],
        leader_target_display: tuple[float, float] | None,
    ) -> None:
        """Light update path: mutate cached 2D source geometry in place.

        Used on every camera-modified / canvas-resize event while a pick is
        active. Skips the full actor teardown + recreate that
        ``_build_pick_actors`` does, so an orbit stays smooth.
        """
        if self._plotter is None:
            return
        if not (self._pick_line_sources or self._pick_ring_sources or self._pick_tick_sources):
            return
        ax, ay = float(anchor_display[0]), float(anchor_display[1])
        if leader_target_display is not None:
            tx, ty = float(leader_target_display[0]), float(leader_target_display[1])
        else:
            tx, ty = ax, ay
        self._picked_leader_target = leader_target_display
        for src in self._pick_line_sources:
            try:
                src.SetPoint1(ax, ay, 0.0)
                src.SetPoint2(tx, ty, 0.0)
                src.Modified()
            except Exception:
                continue
        for src in self._pick_ring_sources:
            try:
                src.SetCenter(ax, ay, 0.0)
                src.Modified()
            except Exception:
                continue
        for entry in self._pick_tick_sources:
            try:
                src, ox1, oy1, ox2, oy2 = entry
                src.SetPoint1(ax + ox1, ay + oy1, 0.0)
                src.SetPoint2(ax + ox2, ay + oy2, 0.0)
                src.Modified()
            except Exception:
                continue
        try:
            self._plotter.render()
        except Exception:
            pass

    def _install_pick_camera_observer(self) -> None:
        """Re-emit canvas_resized whenever the camera moves while a pick is active.

        The launcher's refresh handler then recomputes the anchor from the
        picked world XYZ and redraws the 2D leader line + ring, so the line
        stays visually connected to the moving point instead of floating at
        its original click position.
        """
        if self._plotter is None or self._pick_camera_obs_id is not None:
            return
        try:
            cam = self._plotter.renderer.GetActiveCamera()
        except Exception:
            return
        if cam is None:
            return

        def _on_camera_modified(_caller, _event):
            if self._picked_xyz is None:
                return
            self.canvas_resized.emit()

        try:
            self._pick_camera_obs_id = cam.AddObserver(
                "ModifiedEvent", _on_camera_modified
            )
        except Exception:
            logger.debug("Could not attach camera observer for pick marker", exc_info=True)

    def world_to_display(
        self, xyz: tuple[float, float, float]
    ) -> tuple[float, float] | None:
        """Project a world-space point to plotter display coords (bottom-origin)."""
        if self._plotter is None:
            return None
        try:
            renderer = self._plotter.renderer
            renderer.SetWorldPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
            renderer.WorldToDisplay()
            dx, dy, _dz = renderer.GetDisplayPoint()
            return (float(dx), float(dy))
        except Exception:
            return None

    @property
    def picked_xyz(self) -> tuple[float, float, float] | None:
        return self._picked_xyz

    def _add_pick_2d_overlay(
        self,
        anchor_display: tuple[float, float],
        leader_target_display: tuple[float, float] | None,
        color: tuple[int, int, int],
    ) -> None:
        """VTK 2D ring + leader line in the plotter's display coordinates.

        Rendered as `vtkActor2D` so the geometry lives inside the OpenGL
        frame — sidesteps the Qt child-widget transparency issues that
        affect QOpenGLWidget on Wayland+NVIDIA.
        """
        if self._plotter is None:
            return
        try:
            import vtk
        except Exception:
            logger.debug("VTK unavailable for pick overlay", exc_info=True)
            return

        renderer = self._plotter.renderer
        ax, ay = float(anchor_display[0]), float(anchor_display[1])
        r, g, b = color

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()

        def _add_line(p1, p2, rgb, width, opacity=1.0):
            line_src = vtk.vtkLineSource()
            line_src.SetPoint1(p1[0], p1[1], 0.0)
            line_src.SetPoint2(p2[0], p2[1], 0.0)
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputConnection(line_src.GetOutputPort())
            mapper.SetTransformCoordinate(coord)
            line_actor = vtk.vtkActor2D()
            line_actor.SetMapper(mapper)
            line_actor.GetProperty().SetColor(*rgb)
            line_actor.GetProperty().SetLineWidth(width)
            line_actor.GetProperty().SetOpacity(opacity)
            renderer.AddActor2D(line_actor)
            self._pick_2d_actors.append(line_actor)
            # Cache the source so `update_pick_anchor` can move the line
            # without reallocating the actor each camera frame.
            self._pick_line_sources.append(line_src)

        def _add_ring(radius, rgb, width, opacity=1.0):
            src = vtk.vtkRegularPolygonSource()
            src.SetNumberOfSides(64)
            src.SetRadius(radius)
            src.SetCenter(ax, ay, 0.0)
            src.GeneratePolygonOff()
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputConnection(src.GetOutputPort())
            mapper.SetTransformCoordinate(coord)
            ring_actor = vtk.vtkActor2D()
            ring_actor.SetMapper(mapper)
            ring_actor.GetProperty().SetColor(*rgb)
            ring_actor.GetProperty().SetLineWidth(width)
            ring_actor.GetProperty().SetOpacity(opacity)
            renderer.AddActor2D(ring_actor)
            self._pick_2d_actors.append(ring_actor)
            self._pick_ring_sources.append(src)

        def _add_tick(off1, off2, rgb, width, opacity=1.0):
            # A crosshair tick from anchor+off1 to anchor+off2. The offsets are
            # cached so update_pick_anchor can slide it as the camera orbits.
            tick_src = vtk.vtkLineSource()
            tick_src.SetPoint1(ax + off1[0], ay + off1[1], 0.0)
            tick_src.SetPoint2(ax + off2[0], ay + off2[1], 0.0)
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputConnection(tick_src.GetOutputPort())
            mapper.SetTransformCoordinate(coord)
            tick_actor = vtk.vtkActor2D()
            tick_actor.SetMapper(mapper)
            tick_actor.GetProperty().SetColor(*rgb)
            tick_actor.GetProperty().SetLineWidth(width)
            tick_actor.GetProperty().SetOpacity(opacity)
            renderer.AddActor2D(tick_actor)
            self._pick_2d_actors.append(tick_actor)
            self._pick_tick_sources.append(
                (tick_src, off1[0], off1[1], off2[0], off2[1])
            )

        if leader_target_display is not None:
            tgt = (float(leader_target_display[0]), float(leader_target_display[1]))
            # Black stroke under, white over: stays readable against any background.
            _add_line((ax, ay), tgt, (0.0, 0.0, 0.0), 4.0, 0.95)
            _add_line((ax, ay), tgt, (1.0, 1.0, 1.0), 1.8, 0.95)

        # Reticle: a thin hollow ring + 4 crosshair ticks with an open centre.
        # Black halos go down first, the class colour over them, so the marker
        # reads against any cloud colour without a filled blob hiding the point.
        rad = self._PICK_RING_RADIUS
        ri, ro = self._PICK_TICK_INNER, self._PICK_TICK_OUTER
        col = (r / 255.0, g / 255.0, b / 255.0)
        dirs = ((0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0))
        _add_ring(rad, (0.0, 0.0, 0.0), 3.0, 0.9)
        for dx, dy in dirs:
            _add_tick((dx * ri, dy * ri), (dx * ro, dy * ro), (0.0, 0.0, 0.0), 3.0, 0.9)
        _add_ring(rad, col, 1.5, 1.0)
        for dx, dy in dirs:
            _add_tick((dx * ri, dy * ri), (dx * ro, dy * ro), col, 1.5, 1.0)

    def clear_picked_marker(self) -> None:
        self._picked_xyz = None
        self._picked_leader_target = None
        if self._plotter is None:
            self._pick_2d_actors = []
            self._pick_line_sources = []
            self._pick_ring_sources = []
            self._pick_tick_sources = []
            self._pick_camera_obs_id = None
            return
        # Detach the camera observer first so a mid-clear ModifiedEvent can't
        # re-emit canvas_resized into a half-torn-down state.
        if self._pick_camera_obs_id is not None:
            try:
                cam = self._plotter.renderer.GetActiveCamera()
                if cam is not None:
                    cam.RemoveObserver(self._pick_camera_obs_id)
            except Exception:
                logger.debug("Could not remove pick camera observer", exc_info=True)
            self._pick_camera_obs_id = None
        renderer = self._plotter.renderer
        for actor in self._pick_2d_actors:
            try:
                renderer.RemoveActor2D(actor)
            except Exception:
                pass
        self._pick_2d_actors = []
        self._pick_line_sources = []
        self._pick_ring_sources = []
        self._pick_tick_sources = []
        try:
            self._plotter.render()
        except Exception:
            pass

    def _reveal_canvas(self) -> None:
        self._ensure_plotter()
        self._canvas_revealed = True
        self._canvas_container.setVisible(True)
        # Re-apply on every call: the early-bail used here previously meant the
        # second set_data (semantic, after geometry preview) and the later
        # sidebar switch to Results could leave the bottom panel oversized.
        total = max(self._main_splitter.height(), self._main_splitter.sizeHint().height(), 400)
        self._main_splitter.setSizes([int(total * 0.75), int(total * 0.25)])

    def _hide_canvas(self) -> None:
        self._canvas_revealed = False
        self._canvas_container.setVisible(False)

    @property
    def has_scene_data(self) -> bool:
        return self._final_index is not None or self._geometry_mode

    @property
    def is_geometry_mode(self) -> bool:
        return self._geometry_mode

    @property
    def n_frames(self) -> int:
        if self._final_index is not None:
            return len(self._final_index.frame_order)
        if self._geometry_mode:
            return len(self._geometry_frame_order)
        return 0

    def _timeline_frame_order(self) -> "tuple[int, ...] | list[int]":
        """Frame indices in timeline order for whichever mode is active."""
        if self._final_index is not None:
            return self._final_index.frame_order
        return self._geometry_frame_order

    def class_point_counts(self) -> dict[int, int]:
        """Point counts per class in the loaded semantic cloud (empty if none)."""
        if self._final_index is None:
            return {}
        return {
            int(cid): int(arr.shape[0])
            for cid, arr in self._final_index.xyz_by_class.items()
        }

    # --- Simple point cloud ---

    def show_point_cloud(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        point_size: float = 2.0,
        name: str = "cloud",
    ) -> None:
        if xyz.shape[0] == 0:
            return
        self._clear_scene_data()
        plotter = self._ensure_plotter()
        pd = _make_point_polydata(xyz, rgb)
        self._simple_actor = plotter.add_mesh(
            pd, scalars="colors", rgb=True, point_size=point_size,
            style="points", name="simple_cloud",
        )
        plotter.reset_camera()
        self._reveal_canvas()

    # --- Scene data ---

    def load_scene_data(
        self,
        frame_batch: FrameBatch,
        mapping_result: MappingSequenceResult,
        reference_cloud: SemanticPointCloud,
        classes_config: ClassConfig,
    ) -> None:
        import pyvista as pv

        self._clear_scene_data()
        self._seg_label.setVisible(True)
        self._depth_label.setVisible(True)
        plotter = self._ensure_plotter()
        self._frame_batch = frame_batch
        self._mapping_result = mapping_result

        frame_order = [int(f.frame_index) for f in frame_batch.frames]
        self._emit_setup("Indexing point cloud", 0, 0)
        self._final_index = build_final_cloud_index(
            reference_cloud, frame_order, self._class_colors,
            progress=self._emit_setup,
        )
        self._live_cache = LiveFrameCloudCache(
            frame_batch, mapping_result, self._final_index.frame_order,
        )
        self._max_label_id = max(
            (max(self._class_colors.keys(), default=0)),
            max(self._final_index.class_ids, default=0),
        )

        n_classes = len(self._final_index.class_ids)
        for i, cid in enumerate(self._final_index.class_ids):
            if i == 0 or (i & 0x3) == 0 or i == n_classes - 1:
                self._emit_setup("Preparing class actors", i, n_classes)
            empty = pv.PolyData(np.zeros((1, 3), dtype=np.float32))
            empty["colors"] = np.zeros((1, 3), dtype=np.uint8)
            actor = plotter.add_mesh(
                empty, scalars="colors", rgb=True, point_size=2.0,
                style="points", name=f"class_{cid}",
            )
            actor.SetVisibility(False)
            self._class_actors[cid] = actor
            self._class_polydata[cid] = empty
        self._emit_setup("Preparing class actors", n_classes, n_classes)

        self._build_frustums(frame_batch, mapping_result)

        if self._final_index.class_ids:
            self._emit_setup("Fitting camera", 0, 0)
            all_xyz = [
                self._final_index.xyz_by_class[c]
                for c in self._final_index.class_ids
                if c in self._final_index.xyz_by_class
            ]
            if all_xyz:
                combined = np.concatenate(all_xyz, axis=0)
                if combined.shape[0] > 0:
                    self._auto_fit_camera(combined)

        self._reveal_canvas()
        self._notify_status("scene_loaded")

    def load_scene_data_indexed(
        self,
        frame_batch: FrameBatch,
        mapping_result: MappingSequenceResult,
        final_cloud_index: FinalCloudIndex,
        classes_config: ClassConfig,
    ) -> None:
        """Like load_scene_data but accepts a pre-built FinalCloudIndex."""
        import pyvista as pv

        self._clear_scene_data()
        self._seg_label.setVisible(True)
        self._depth_label.setVisible(True)
        plotter = self._ensure_plotter()
        self._frame_batch = frame_batch
        self._mapping_result = mapping_result

        self._final_index = final_cloud_index
        self._live_cache = LiveFrameCloudCache(
            frame_batch, mapping_result, self._final_index.frame_order,
        )
        self._max_label_id = max(
            (max(self._class_colors.keys(), default=0)),
            max(self._final_index.class_ids, default=0),
        )

        n_classes = len(self._final_index.class_ids)
        for i, cid in enumerate(self._final_index.class_ids):
            if i == 0 or (i & 0x3) == 0 or i == n_classes - 1:
                self._emit_setup("Preparing class actors", i, n_classes)
            empty = pv.PolyData(np.zeros((1, 3), dtype=np.float32))
            empty["colors"] = np.zeros((1, 3), dtype=np.uint8)
            actor = plotter.add_mesh(
                empty, scalars="colors", rgb=True, point_size=2.0,
                style="points", name=f"class_{cid}",
            )
            actor.SetVisibility(False)
            self._class_actors[cid] = actor
            self._class_polydata[cid] = empty
        self._emit_setup("Preparing class actors", n_classes, n_classes)

        self._build_frustums(frame_batch, mapping_result)

        if self._final_index.class_ids:
            self._emit_setup("Fitting camera", 0, 0)
            all_xyz = [
                self._final_index.xyz_by_class[c]
                for c in self._final_index.class_ids
                if c in self._final_index.xyz_by_class
            ]
            if all_xyz:
                combined = np.concatenate(all_xyz, axis=0)
                if combined.shape[0] > 0:
                    self._auto_fit_camera(combined)

        self._reveal_canvas()
        self._notify_status("scene_loaded")

    def load_geometry_scene(
        self,
        frame_batch: FrameBatch,
        mapping_result: MappingSequenceResult,
        geometry_xyz: np.ndarray,
        geometry_rgb: np.ndarray,
    ) -> None:
        """Open a geometry-only (``--skip-segmentation``) run as a real timeline.

        Shows the full RGB geometry cloud plus camera frustums and drives a
        frame slider / playback over an RGB + depth image panel. There are no
        labels, so there is no FinalCloudIndex and no per-class legend; the
        cloud itself is static while the slider moves the frustum highlight and
        image panel.
        """
        self._clear_scene_data()
        self._seg_label.setVisible(False)
        self._depth_label.setVisible(True)
        plotter = self._ensure_plotter()
        self._frame_batch = frame_batch
        self._mapping_result = mapping_result
        self._geometry_mode = True
        self._geometry_frame_order = [int(f.frame_index) for f in frame_batch.frames]
        self._geometry_xyz = np.asarray(geometry_xyz, dtype=np.float32)

        if self._geometry_xyz.shape[0] > 0:
            pd = _make_point_polydata(self._geometry_xyz, geometry_rgb)
            self._simple_actor = plotter.add_mesh(
                pd, scalars="colors", rgb=True, point_size=2.0,
                style="points", name="geometry_cloud",
            )

        self._build_frustums(frame_batch, mapping_result)

        if self._geometry_xyz.shape[0] > 0:
            self._auto_fit_camera(self._geometry_xyz)

        self._reveal_canvas()
        self._notify_status("scene_loaded")

    def apply_geometry_state(
        self,
        timeline_t: int,
        point_size: float,
        *,
        frustums_visible: bool = True,
    ) -> None:
        """Timeline update for geometry mode: point size, frustum highlight, image panel."""
        if not self._geometry_mode:
            return
        n = len(self._geometry_frame_order)
        if n == 0:
            return
        t = int(np.clip(timeline_t, 0, n - 1))
        self._last_t = t
        if point_size != self._last_point_size:
            self._update_point_sizes(point_size)
        self._update_frustum_visibility(frustums_visible, t)
        self._update_geometry_image_panel(t)
        if self._plotter is not None:
            try:
                self._plotter.render()
            except Exception:
                pass

    def _compose_geometry_frame_panel(self, t: int) -> "tuple[np.ndarray, np.ndarray] | None":
        """RGB + colorized depth for geometry-only frame t (no segmentation labels)."""
        import cv2

        order = self._geometry_frame_order
        if not order or self._frame_batch is None or self._mapping_result is None:
            return None
        tt = int(np.clip(t, 0, len(order) - 1))
        frame_idx = int(order[tt])

        frame = None
        for f in self._frame_batch.frames:
            if int(f.frame_index) == frame_idx:
                frame = f
                break
        if frame is None:
            return None

        mapping_indices = np.asarray(self._mapping_result.frame_indices, dtype=np.int32).reshape(-1)
        mi = None
        for i, fid in enumerate(mapping_indices.tolist()):
            if int(fid) == frame_idx:
                mi = i
                break
        if mi is None:
            return None

        rgb = np.asarray(frame.image_rgb, dtype=np.uint8)
        depth = np.asarray(self._mapping_result.depth_maps[mi], dtype=np.float32)
        h, w = rgb.shape[:2]
        depth_color = _colorize_depth(
            cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST),
        )
        return rgb, depth_color

    def _update_geometry_image_panel(self, t: int) -> None:
        parts = self._compose_geometry_frame_panel(t)
        if parts is None:
            return
        rgb, depth = parts
        self._paint_label(self._rgb_label, rgb)
        self._paint_label(self._depth_label, depth)

    def _emit_setup(self, message: str, current: int, total: int) -> None:
        """Forward a one-off setup-progress event to the GUI status callback.

        Used for stages that run on the GUI thread (cloud indexing, actor
        creation, frustum build, initial GPU upload) where the user otherwise
        sees a frozen "Setting up viewer…" label.
        """
        self._notify_status(
            "setup_progress", message=message, current=int(current), total=int(total)
        )

    def _build_frustums(self, frame_batch: FrameBatch, mapping_result: MappingSequenceResult) -> None:
        if self._plotter is None:
            return
        self._emit_setup("Building camera frustums", 0, 1)
        mapping_indices = np.asarray(mapping_result.frame_indices, dtype=np.int32).reshape(-1)
        intrinsics = np.asarray(mapping_result.intrinsics, dtype=np.float64)
        depth_h, depth_w = mapping_result.depth_maps[0].shape
        fy = float(intrinsics[1, 1])
        fov_y = 2.0 * np.arctan(depth_h / (2.0 * fy))
        aspect = depth_w / max(depth_h, 1)

        mi_lookup = {int(fid): i for i, fid in enumerate(mapping_indices.tolist())}

        all_pts: list[np.ndarray] = []
        frustum_frame_ids: list[int] = []
        for f in frame_batch.frames:
            frame_idx = int(f.frame_index)
            mi = mi_lookup.get(frame_idx)
            if mi is None:
                continue
            pose_w_c = np.asarray(mapping_result.poses_w_c[mi], dtype=np.float64)
            all_pts.append(_build_frustum_lines(pose_w_c, fov_y, aspect))
            frustum_frame_ids.append(frame_idx)

        if not all_pts:
            self._emit_setup("Building camera frustums", 1, 1)
            return

        # Single batched mesh for all frustums (one add_mesh call).
        batched = np.concatenate(all_pts, axis=0)
        pd = _make_line_segments_polydata(batched)
        self._frustum_batch_actor = self._plotter.add_mesh(
            pd, color=(0.5, 0.5, 0.5), line_width=1, opacity=0.6,
            name="frustums_batch",
        )
        self._frustum_batch_pd = pd
        self._frustum_frame_ids = frustum_frame_ids
        self._frustum_fid_to_idx = {fid: i for i, fid in enumerate(frustum_frame_ids)}
        self._frustum_pts_per = 16  # 8 line segments × 2 endpoints

        # Separate actor for the highlighted (current) frustum.
        hl_pd = _make_line_segments_polydata(all_pts[0])
        self._frustum_highlight_actor = self._plotter.add_mesh(
            hl_pd, color=(1.0, 0.8, 0.25), line_width=2, opacity=0.9,
            name="frustum_highlight",
        )
        self._frustum_highlight_pd = hl_pd
        self._frustum_all_pts = all_pts
        self._emit_setup("Building camera frustums", 1, 1)

    def _clear_scene_data(self) -> None:
        if self._plotter is not None:
            # Batch removals with `render=False` and do a single render at the
            # end. Per-actor rendering was the cause of the multi-second freeze
            # on "New reconstruction" with hundreds of frustum actors.
            def _remove(actor: object) -> None:
                try:
                    self._plotter.remove_actor(actor, render=False)
                except TypeError:
                    # Older pyvista versions don't accept `render`.
                    try:
                        self._plotter.remove_actor(actor)
                    except Exception:
                        pass
                except Exception:
                    pass

            for actor in self._class_actors.values():
                _remove(actor)
            if hasattr(self, "_frustum_batch_actor") and self._frustum_batch_actor is not None:
                _remove(self._frustum_batch_actor)
            if hasattr(self, "_frustum_highlight_actor") and self._frustum_highlight_actor is not None:
                _remove(self._frustum_highlight_actor)
            for actor in self._frustum_actors.values():
                _remove(actor)
            if self._live_actor is not None:
                _remove(self._live_actor)
            if self._simple_actor is not None:
                _remove(self._simple_actor)
            self.clear_picked_marker()
            try:
                self._plotter.render()
            except Exception:
                pass
        self._class_actors.clear()
        self._class_polydata.clear()
        self._frustum_actors.clear()
        self._frustum_batch_actor = None
        self._frustum_batch_pd = None
        self._frustum_highlight_actor = None
        self._frustum_highlight_pd = None
        self._frustum_frame_ids = []
        self._frustum_fid_to_idx = {}
        self._frustum_all_pts = []
        self._live_actor = None
        self._live_polydata = None
        self._simple_actor = None
        self._final_index = None
        self._live_cache = None
        self._geometry_mode = False
        self._geometry_frame_order = []
        self._geometry_xyz = None
        self._fit_camera_params = None
        self._frame_batch = None
        self._mapping_result = None
        self._frame_panel_cache.clear()
        self._last_t = None
        self._last_accumulate = None
        self._last_semantic = None
        self._last_enabled = None
        self._last_confidence = None
        self._last_point_size = None
        self._point_filter = None

    def _auto_fit_camera(self, positions: np.ndarray) -> None:
        if self._plotter is None:
            return
        cam_origins: np.ndarray | None = None
        if self._mapping_result is not None:
            try:
                poses = np.asarray(self._mapping_result.poses_w_c, dtype=np.float64)
                if poses.ndim == 3 and poses.shape[0] >= 1 and poses.shape[1:] == (4, 4):
                    cam_origins = poses[:, :3, 3]
            except Exception:
                logger.debug("Could not extract camera origins for fit", exc_info=True)
                cam_origins = None
        world_up = _estimate_world_up(positions, cam_origins)
        cam_pos, focal, up = _compute_transect_view(positions, cam_origins, world_up)
        self._fit_camera_params = (cam_pos, focal, up)
        self._apply_camera_params(cam_pos, focal, up)

    def _apply_camera_params(
        self,
        cam_pos: tuple[float, float, float],
        focal: tuple[float, float, float],
        up: tuple[float, float, float],
    ) -> None:
        if self._plotter is None:
            return
        self._plotter.camera.position = cam_pos
        self._plotter.camera.focal_point = focal
        self._plotter.camera.up = up
        self._plotter.reset_camera()

    def reset_view(self) -> None:
        """Re-orient the camera to the default transect-lengthwise view.

        The orientation is computed once at load (`_auto_fit_camera`) and cached,
        so this just reapplies it — no re-concatenating the cloud or re-running
        the SVD fit on every click. Falls back to a recompute if the cache is
        missing.
        """
        if self._plotter is None:
            return
        if self._fit_camera_params is not None:
            self._apply_camera_params(*self._fit_camera_params)
            try:
                self._plotter.render()
            except Exception:
                pass
            return
        if self._geometry_mode:
            combined = self._geometry_xyz
        elif self._final_index is not None:
            all_xyz = [
                self._final_index.xyz_by_class[c]
                for c in self._final_index.class_ids
                if c in self._final_index.xyz_by_class
            ]
            combined = np.concatenate(all_xyz, axis=0) if all_xyz else None
        else:
            return
        if combined is None or combined.shape[0] == 0:
            return
        self._auto_fit_camera(combined)
        try:
            self._plotter.render()
        except Exception:
            pass

    def zoom_to_point(self, xyz: tuple[float, float, float], radius: float = 0.3) -> None:
        """Move the camera to look at *xyz* from *radius* metres away."""
        if self._plotter is None:
            return
        cam = self._plotter.camera
        pos = np.asarray(cam.position, dtype=np.float64)
        target = np.asarray(xyz, dtype=np.float64)
        direction = pos - target
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction /= norm
        cam.focal_point = tuple(target.tolist())
        cam.position = tuple((target + direction * radius).tolist())
        try:
            self._plotter.render()
        except Exception:
            pass

    def view_from_frame_pose(self, t: int, backoff_m: float = 0.0) -> bool:
        """Snap the 3D camera to frame `t`'s pose, optionally pulled back."""
        if self._plotter is None or self._mapping_result is None:
            return False
        frame_order = self._timeline_frame_order()
        if len(frame_order) == 0:
            return False
        tt = int(np.clip(t, 0, len(frame_order) - 1))
        target_frame = int(frame_order[tt])
        mapping_indices = np.asarray(self._mapping_result.frame_indices, dtype=np.int32).reshape(-1)
        mi = None
        for i, fid in enumerate(mapping_indices.tolist()):
            if int(fid) == target_frame:
                mi = i
                break
        if mi is None:
            return False
        pose_w_c = np.asarray(self._mapping_result.poses_w_c[mi], dtype=np.float64)
        # Columns of the rotation block are the camera basis vectors in world
        # coordinates: x=right, y=down, z=forward (looking along +z).
        origin = pose_w_c[:3, 3]
        down = pose_w_c[:3, 1]
        forward = pose_w_c[:3, 2]
        cam_pos = origin - float(backoff_m) * forward
        focal = origin + forward
        self._plotter.camera.position = tuple(cam_pos.tolist())
        self._plotter.camera.focal_point = tuple(focal.tolist())
        self._plotter.camera.up = tuple((-down).tolist())
        try:
            self._plotter.render()
        except Exception:
            pass
        return True

    # --- State application ---

    def apply_state(
        self,
        timeline_t: int,
        accumulate: bool,
        enabled_classes: frozenset[int],
        semantic_colors: bool,
        point_size: float,
        min_confidence: float = 0.0,
        frustums_visible: bool = True,
    ) -> None:
        if self._final_index is None or self._live_cache is None:
            return
        if self._plotter is None:
            return

        fi = self._final_index
        n_steps = len(fi.frame_order)
        if n_steps <= 0:
            return
        t = int(np.clip(timeline_t, 0, n_steps - 1))
        min_conf = float(np.clip(min_confidence, 0.0, 1.0))

        need_full = (
            self._last_t != t
            or self._last_accumulate != accumulate
            or self._last_semantic != semantic_colors
            or self._last_enabled != enabled_classes
            or self._last_confidence != min_conf
        )

        if not need_full:
            if self._last_point_size != point_size:
                self._update_point_sizes(point_size)
            self._update_frustum_visibility(frustums_visible, t)
            self._plotter.render()
            return

        # First paint after scene load: surface per-class GPU-upload progress
        # so the user sees the "Setting up viewer" stage advance instead of
        # freezing while every class' polydata is pushed to VTK.
        first_paint = self._last_t is None
        self._update_live_cloud(t, enabled_classes, semantic_colors, min_conf, point_size)
        self._update_class_clouds(
            t, accumulate, enabled_classes, semantic_colors, min_conf, point_size,
            report_progress=first_paint,
        )
        self._update_frustum_visibility(frustums_visible, t)
        self._update_image_panel(t)
        if first_paint:
            self._emit_setup("Finalising viewer", 1, 1)

        self._last_t = t
        self._last_accumulate = accumulate
        self._last_semantic = semantic_colors
        self._last_enabled = enabled_classes
        self._last_confidence = min_conf
        self._last_point_size = point_size

        self._plotter.render()

    def _update_live_cloud(
        self,
        t: int,
        enabled_classes: frozenset[int],
        semantic_colors: bool,
        min_conf: float,
        point_size: float,
    ) -> None:
        try:
            if self._live_cache is None:
                raise RuntimeError("live cache not initialised")
            xyz_u, rgb_u, lab_u, conf_u = self._live_cache.get_unmasked(t)
        except Exception:
            xyz_u = _EMPTY_XYZ
            rgb_u = np.zeros((0, 3), dtype=np.uint8)
            lab_u = np.zeros((0,), dtype=np.int32)
            conf_u = np.zeros((0,), dtype=np.float32)

        if xyz_u.shape[0] == 0:
            if self._live_actor is not None:
                self._live_actor.SetVisibility(False)
            return

        max_id = max(self._max_label_id, int(lab_u.max()) if lab_u.size else 0)
        lut = build_enabled_label_lut(max_id, set(enabled_classes))
        m = mask_points_by_enabled_lut(lab_u, lut)
        if min_conf > 0.0 and conf_u.size:
            m &= conf_u >= min_conf
        if self._point_filter is not None and xyz_u.shape[0] > 0:
            try:
                m &= np.asarray(self._point_filter(xyz_u), dtype=bool).reshape(-1)
            except Exception:
                logger.debug("Point filter failed on live cloud", exc_info=True)
        xyz_live = xyz_u[m]

        if xyz_live.shape[0] == 0:
            if self._live_actor is not None:
                self._live_actor.SetVisibility(False)
            return

        if semantic_colors:
            cols_live = np.full((xyz_live.shape[0], 3), 128, dtype=np.uint8)
            for cid, color in self._class_colors.items():
                cols_live[lab_u[m] == cid] = color
        else:
            cols_live = rgb_u[m]

        pd = _make_point_polydata(xyz_live, cols_live)
        if self._live_actor is None:
            self._live_actor = self._plotter.add_mesh(
                pd, scalars="colors", rgb=True, point_size=point_size,
                style="points", name="live_cloud",
            )
            self._live_polydata = pd
        else:
            self._live_actor.GetMapper().SetInputData(pd)
            self._live_polydata = pd
            self._live_actor.GetProperty().SetPointSize(point_size)
            self._live_actor.SetVisibility(True)

    def _update_class_clouds(
        self,
        t: int,
        accumulate: bool,
        enabled_classes: frozenset[int],
        semantic_colors: bool,
        min_conf: float,
        point_size: float,
        report_progress: bool = False,
    ) -> None:
        fi = self._final_index
        assert fi is not None
        total_actors = len(self._class_actors) if report_progress else 0
        for idx, (cid, actor) in enumerate(self._class_actors.items()):
            if report_progress:
                self._emit_setup("Uploading class points", idx, total_actors)
            if cid not in enabled_classes:
                actor.SetVisibility(False)
                continue
            xyz_c = fi.xyz_by_class.get(cid)
            if xyz_c is None or xyz_c.shape[0] == 0:
                actor.SetVisibility(False)
                continue
            n = int(fi.prefix_end_by_class[cid][t]) if accumulate else 0
            if n <= 0:
                actor.SetVisibility(False)
                continue
            src = fi.semrgb_by_class[cid] if semantic_colors else fi.rgb_by_class[cid]
            pts = xyz_c[:n]
            cols = src[:n]
            orig_idx: np.ndarray | None = None
            if min_conf > 0.0:
                conf_c = fi.conf_by_class.get(cid)
                if conf_c is not None:
                    keep = conf_c[:n] >= min_conf
                    pts = pts[keep]
                    cols = cols[keep]
                    orig_idx = np.nonzero(keep)[0].astype(np.int32)
            if self._point_filter is not None and pts.shape[0] > 0:
                try:
                    keep_pf = np.asarray(self._point_filter(pts), dtype=bool).reshape(-1)
                    pts = pts[keep_pf]
                    cols = cols[keep_pf]
                    if orig_idx is None:
                        orig_idx = np.nonzero(keep_pf)[0].astype(np.int32)
                    else:
                        orig_idx = orig_idx[keep_pf]
                except Exception:
                    logger.debug("Point filter failed on class %s", cid, exc_info=True)
            if pts.shape[0] == 0:
                actor.SetVisibility(False)
                continue
            pd = _make_point_polydata(pts, cols)
            if orig_idx is not None:
                pd.point_data["orig_idx"] = orig_idx
            actor.GetMapper().SetInputData(pd)
            actor.GetProperty().SetPointSize(point_size)
            actor.SetVisibility(True)
            self._class_polydata[cid] = pd
        if report_progress and total_actors:
            self._emit_setup("Uploading class points", total_actors, total_actors)

    def _update_point_sizes(self, point_size: float) -> None:
        for actor in self._class_actors.values():
            actor.GetProperty().SetPointSize(point_size)
        if self._live_actor is not None:
            self._live_actor.GetProperty().SetPointSize(point_size)
        if self._simple_actor is not None:
            self._simple_actor.GetProperty().SetPointSize(point_size)
        self._last_point_size = point_size

    def _update_frustum_visibility(self, visible: bool, t: int) -> None:
        frame_order = self._timeline_frame_order()
        current_frame = None
        if len(frame_order) > 0:
            tt = int(np.clip(t, 0, len(frame_order) - 1))
            current_frame = int(frame_order[tt])

        # Batched frustum path
        if self._frustum_batch_actor is not None:
            self._frustum_batch_actor.SetVisibility(bool(visible))
            if self._frustum_highlight_actor is not None:
                show_hl = visible and current_frame is not None and current_frame in self._frustum_fid_to_idx
                self._frustum_highlight_actor.SetVisibility(bool(show_hl))
                if show_hl and current_frame is not None:
                    idx = self._frustum_fid_to_idx[current_frame]
                    pts = self._frustum_all_pts[idx]
                    new_pd = _make_line_segments_polydata(pts)
                    self._frustum_highlight_pd.copy_from(new_pd)
            return

        # Legacy per-actor path
        for fid, actor in self._frustum_actors.items():
            actor.SetVisibility(bool(visible))
            if not visible:
                continue
            prop = actor.GetProperty()
            if fid == current_frame:
                prop.SetColor(1.0, 0.8, 0.25)
                prop.SetOpacity(0.9)
                prop.SetLineWidth(2.0)
            else:
                prop.SetColor(0.5, 0.5, 0.5)
                prop.SetOpacity(0.6)
                prop.SetLineWidth(1.0)

    # --- Image panel ---

    def current_frame_stack(self) -> "np.ndarray | None":
        """Return the RGB/seg/depth composite for exporting the current frame."""
        if self._last_t is None:
            return None
        if self._frame_batch is None or self._mapping_result is None:
            return None
        if self._final_index is None:
            return None
        t = int(self._last_t)
        parts = self._frame_panel_cache.get(t) or self._compose_frame_panel(t)
        if parts is None:
            return None
        self._frame_panel_cache[t] = parts
        rgb, seg, depth = parts
        return np.concatenate([rgb, seg, depth], axis=0)

    def _update_image_panel(self, t: int) -> None:
        if self._frame_batch is None or self._mapping_result is None:
            return
        if self._final_index is None:
            return

        parts = self._frame_panel_cache.get(t)
        if parts is None:
            parts = self._compose_frame_panel(t)
            if parts is not None:
                self._frame_panel_cache[t] = parts

        if parts is None:
            return

        rgb, seg, depth = parts
        self._paint_label(self._rgb_label, rgb)
        self._paint_label(self._seg_label, seg)
        self._paint_label(self._depth_label, depth)

    @staticmethod
    def _paint_label(label: QLabel, image: np.ndarray) -> None:
        h, w, _ = image.shape
        qimg = QImage(np.ascontiguousarray(image).data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        target = max(1, min(w, label.width() or w))
        label.setPixmap(pixmap.scaledToWidth(target, Qt.TransformationMode.SmoothTransformation))

    def _compose_frame_panel(
        self, t: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        import cv2

        fi = self._final_index
        if fi is None or len(fi.frame_order) == 0:
            return None
        if self._frame_batch is None or self._mapping_result is None:
            return None
        tt = int(np.clip(t, 0, len(fi.frame_order) - 1))
        frame_idx = int(fi.frame_order[tt])

        frame = None
        for f in self._frame_batch.frames:
            if int(f.frame_index) == frame_idx:
                frame = f
                break
        if frame is None:
            return None

        mapping_indices = np.asarray(self._mapping_result.frame_indices, dtype=np.int32).reshape(-1)
        mi = None
        for i, fid in enumerate(mapping_indices.tolist()):
            if int(fid) == frame_idx:
                mi = i
                break
        if mi is None:
            return None

        rgb = np.asarray(frame.image_rgb, dtype=np.uint8)
        labels = np.asarray(frame.labels, dtype=np.int32)
        depth = np.asarray(self._mapping_result.depth_maps[mi], dtype=np.float32)

        h, w = rgb.shape[:2]
        seg_color = _colorize_seg(
            cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST),
            self._class_colors,
        )
        depth_color = _colorize_depth(
            cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST),
        )
        return rgb, seg_color, depth_color

    def _show_live_preprocess_frame(self, frame_index: int) -> None:
        import cv2

        if self._output_dir is None:
            return
        stem = f"{frame_index:08d}"
        frame_path = self._output_dir / "frames" / f"{stem}.png"
        labels_path = self._output_dir / "labels" / f"{stem}.npy"
        if not frame_path.exists():
            return
        rgb = cv2.imread(str(frame_path))
        if rgb is None:
            return
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        self._paint_label(self._rgb_label, rgb)
        if labels_path.exists():
            labels = np.load(str(labels_path))
            h, w = rgb.shape[:2]
            seg_color = _colorize_seg(
                cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST),
                self._class_colors,
            )
            self._seg_label.setVisible(True)
            self._paint_label(self._seg_label, seg_color)
        else:
            self._seg_label.setVisible(False)

    # --- Viewer protocol ---

    def start_run(self, run_label: str, output_dir: str) -> None:
        self._sig_start_run.emit(run_label, output_dir)

    def set_stage(self, stage: str, status: str, message: str | None = None) -> None:
        self._sig_set_stage.emit(stage, status, message)

    def update_progress(
        self, stage: str, current: int, total: int | None = None,
        message: str | None = None, frame_index: int | None = None,
    ) -> None:
        self._sig_update_progress.emit(stage, current, total, message, frame_index)

    def set_data(self, **kwargs: object) -> None:
        self._sig_data_ready.emit(kwargs)

    def mark_outputs_ready(self, output_dir: str, output_files: list[str]) -> None:
        self._sig_mark_outputs.emit(output_dir, output_files)

    def fail_run(self, stage: str, error_message: str) -> None:
        self._sig_fail_run.emit(stage, error_message)

    def close(self) -> None:  # type: ignore[override]  # routes shutdown through a signal; bool return unused
        self._sig_close.emit()

    def wait_forever(self) -> None:
        pass

    # --- Slots ---

    @Slot(str, str)
    def _on_start_run(self, run_label: str, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._hide_canvas()
        self._depth_label.setVisible(False)
        self._notify_status("start_run", run_label=run_label, output_dir=output_dir)

    @Slot(str, str, object)
    def _on_set_stage(self, stage: str, status: str, message: object) -> None:
        self._notify_status("set_stage", stage=stage, status=status, message=message)

    @Slot(str, int, object, object, object)
    def _on_update_progress(self, stage: str, current: int, total: object, message: object, frame_index: object) -> None:
        if stage == "preprocess" and frame_index is not None and self._output_dir is not None:
            self._show_live_preprocess_frame(int(cast("SupportsInt", frame_index)))
        self._notify_status("update_progress", stage=stage, current=current, total=total, message=message)

    @Slot(object)
    def _on_data_ready(self, kwargs: dict) -> None:
        if "reference_cloud" in kwargs:
            self.load_scene_data(
                frame_batch=kwargs["frame_batch"],
                mapping_result=kwargs["mapping_result"],
                reference_cloud=kwargs["reference_cloud"],
                classes_config=kwargs["classes_config"],
            )
        elif "geometry_xyz" in kwargs:
            fb = kwargs.get("frame_batch")
            mr = kwargs.get("mapping_result")
            if kwargs.get("geometry_only") and fb is not None and mr is not None:
                self.load_geometry_scene(
                    frame_batch=fb,
                    mapping_result=mr,
                    geometry_xyz=kwargs["geometry_xyz"],
                    geometry_rgb=kwargs["geometry_rgb"],
                )
            else:
                self.show_point_cloud(kwargs["geometry_xyz"], kwargs["geometry_rgb"])
        self._notify_status("data_ready", **kwargs)

    @Slot(str, object)
    def _on_mark_outputs(self, output_dir: str, output_files: object) -> None:
        self._notify_status("mark_outputs", output_dir=output_dir, output_files=output_files)
        # The callback above switches the sidebar to Results, which reflows the
        # central pane and can leave setSizes stale. Defer a re-apply until the
        # event queue has caught up.
        if self._canvas_revealed:
            QTimer.singleShot(0, self._reveal_canvas)

    @Slot(str, str)
    def _on_fail_run(self, stage: str, error_message: str) -> None:
        self._notify_status("fail_run", stage=stage, error_message=error_message)

    @Slot()
    def _on_close(self) -> None:
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass

    def _notify_status(self, event: str, **kwargs: object) -> None:
        if self._status_callback is not None:
            try:
                self._status_callback(event, **kwargs)
            except Exception:
                logger.exception("Status callback error")
