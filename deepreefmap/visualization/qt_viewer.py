from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv
    from pyvistaqt import QtInteractor
from PySide6.QtCore import QEvent, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.visualization.final_cloud_index import FinalCloudIndex, build_final_cloud_index
from deepreefmap.visualization.live_frame_cloud import (
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
    count + solo button). A header strip carries a minimize/expand toggle so the
    user can shrink the overlay to just its title bar when it covers too much of
    the canvas. `reposition()` anchors the overlay to its parent's top-right
    corner and clamps the height to 60% of the parent.
    """

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
            LegendOverlay QCheckBox { color: #e8e8e8; font-size: 11px; spacing: 4px; }
            LegendOverlay QCheckBox::indicator { width: 12px; height: 12px; }
            LegendOverlay QScrollArea { background: transparent; border: none; }
            LegendOverlay QWidget#legend_inner { background: transparent; }
            LegendOverlay QToolButton {
                color: #e8e8e8;
                background-color: rgba(255, 255, 255, 20);
                border: 1px solid rgba(255, 255, 255, 50);
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

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._inner = QWidget()
        self._inner.setObjectName("legend_inner")
        self._grid = QGridLayout(self._inner)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(2)
        self._grid.setColumnStretch(1, 1)
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll, 1)

        self._minimized = False
        self.hide()

    def _toggle_minimized(self) -> None:
        self._minimized = not self._minimized
        self._scroll.setVisible(not self._minimized)
        self._minimize_btn.setText("+" if self._minimized else "−")
        self._minimize_btn.setToolTip(
            "Expand legend" if self._minimized else "Collapse legend"
        )
        self.reposition()

    def clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

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
            count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
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
            toggles[cid] = cb
            solo_buttons[cid] = solo

        # Drive the scroll area's natural width from the inner content so
        # adjustSize() in reposition() picks up the correct width instead of
        # collapsing to QScrollArea's tiny default size hint.
        sb_w = self._scroll.verticalScrollBar().sizeHint().width()
        self._scroll.setMinimumWidth(self._inner.sizeHint().width() + sb_w + 4)

        # Reserve space for ~10 rows by default so the legend doesn't collapse
        # to the first handful of classes. If there are fewer classes, the
        # scroll area sizes to its content; if more, the user scrolls.
        n_rows = len(visible_ids)
        if n_rows:
            self._grid.activate()
            inner_h = max(1, self._inner.sizeHint().height())
            row_h = max(18, inner_h // n_rows)
            visible_rows = min(10, n_rows)
            target_h = visible_rows * row_h + 4
            self._scroll.setMinimumHeight(target_h)
        return toggles, solo_buttons

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        # Height capped so the overlay scrolls internally rather than
        # overflowing the canvas. Width is allowed to grow to fit the longest
        # class name, but capped at half the canvas so it can't swallow the
        # whole view.
        self.setMaximumHeight(max(60, int(parent.height() * 0.6)))
        self.setMaximumWidth(max(140, int(parent.width() * 0.5)))
        self.adjustSize()
        margin = 8
        self.move(parent.width() - self.width() - margin, margin)
        self.raise_()

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

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumHeight(120)
        self._image_label.setStyleSheet("background-color: #1a1a1a;")

        self._main_splitter = QSplitter(Qt.Vertical)
        self._canvas_container = QWidget()
        self._canvas_layout = QVBoxLayout(self._canvas_container)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._main_splitter.addWidget(self._canvas_container)
        self._main_splitter.addWidget(self._image_label)
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
        self._canvas_container.installEventFilter(self)

        self._plotter: QtInteractor | None = None

        self._simple_actor = None
        self._live_actor = None
        self._live_polydata: pv.PolyData | None = None
        self._class_actors: dict[int, object] = {}
        self._class_polydata: dict[int, pv.PolyData] = {}
        self._frustum_actors: dict[int, object] = {}

        self._final_index: FinalCloudIndex | None = None
        self._live_cache: LiveFrameCloudCache | None = None
        self._frame_batch = None
        self._mapping_result = None
        self._max_label_id = 0

        self._last_t: int | None = None
        self._last_accumulate: bool | None = None
        self._last_semantic: bool | None = None
        self._last_enabled: frozenset[int] | None = None
        self._last_confidence: float | None = None
        self._last_point_size: float | None = None

        self._point_filter: Callable[[np.ndarray], np.ndarray] | None = None

        self._picking_enabled = False
        self._picked_actor_inner = None
        self._picked_actor_outer = None

        self._frame_panel_cache: dict[int, np.ndarray] = {}

        self._sig_start_run.connect(self._on_start_run)
        self._sig_set_stage.connect(self._on_set_stage)
        self._sig_update_progress.connect(self._on_update_progress)
        self._sig_data_ready.connect(self._on_data_ready)
        self._sig_mark_outputs.connect(self._on_mark_outputs)
        self._sig_fail_run.connect(self._on_fail_run)
        self._sig_close.connect(self._on_close)

        self._status_callback: Callable[..., None] | None = None

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

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self._canvas_container and event.type() == QEvent.Type.Resize:
            self.legend_overlay.reposition()
            self.canvas_resized.emit()
        return super().eventFilter(obj, event)

    def _ensure_plotter(self):
        if self._plotter is not None:
            return self._plotter
        from pyvistaqt import QtInteractor

        self._plotter = QtInteractor(self._canvas_container)
        self._plotter.set_background("#141414")
        try:
            self._plotter.enable_eye_dome_lighting()
        except Exception:
            logger.debug("Eye dome lighting unavailable", exc_info=True)
        self._canvas_layout.addWidget(self._plotter)
        # Keep the legend on top after the plotter is added below it.
        self.legend_overlay.raise_()
        self._enable_picking()
        return self._plotter

    def _enable_picking(self) -> None:
        if self._picking_enabled or self._plotter is None:
            return
        try:
            self._plotter.enable_point_picking(
                callback=self._on_point_picked,
                show_message=False,
                show_point=False,
                left_clicking=True,
                use_mesh=True,
            )
            self._picking_enabled = True
        except Exception:
            logger.debug("enable_point_picking unavailable", exc_info=True)

    def _on_point_picked(self, mesh, point_id) -> None:
        if self._final_index is None or self._plotter is None:
            self.point_picked_clear.emit()
            return
        if mesh is None or point_id is None or int(point_id) < 0:
            self.point_picked_clear.emit()
            return

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
            self.point_picked_clear.emit()
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
    ) -> None:
        """Place a haloed sphere marker at the picked world position.

        Renders as a small bright sphere in the class color over a slightly
        larger white sphere so the marker stays visible regardless of the
        underlying point cloud color.
        """
        if self._plotter is None:
            return
        import pyvista as pv

        self.clear_picked_marker()
        pd = pv.PolyData(np.asarray([xyz], dtype=np.float32))
        r, g, b = color
        self._picked_actor_outer = self._plotter.add_mesh(
            pd, color=(1.0, 1.0, 1.0), point_size=26.0,
            render_points_as_spheres=True, style="points",
            name="picked_marker_outer", pickable=False,
        )
        self._picked_actor_inner = self._plotter.add_mesh(
            pd, color=(r / 255.0, g / 255.0, b / 255.0), point_size=16.0,
            render_points_as_spheres=True, style="points",
            name="picked_marker_inner", pickable=False,
        )
        try:
            self._plotter.render()
        except Exception:
            pass

    def clear_picked_marker(self) -> None:
        if self._plotter is None:
            self._picked_actor_inner = None
            self._picked_actor_outer = None
            return
        for attr in ("_picked_actor_outer", "_picked_actor_inner"):
            actor = getattr(self, attr, None)
            if actor is None:
                continue
            try:
                self._plotter.remove_actor(actor, render=False)
            except TypeError:
                try:
                    self._plotter.remove_actor(actor)
                except Exception:
                    pass
            except Exception:
                pass
            setattr(self, attr, None)
        try:
            self._plotter.render()
        except Exception:
            pass

    def _reveal_canvas(self) -> None:
        self._ensure_plotter()
        if self._canvas_revealed:
            return
        self._canvas_revealed = True
        self._canvas_container.setVisible(True)
        total = max(self._main_splitter.height(), 1)
        self._main_splitter.setSizes([int(total * 0.75), int(total * 0.25)])

    def _hide_canvas(self) -> None:
        self._canvas_revealed = False
        self._canvas_container.setVisible(False)

    @property
    def has_scene_data(self) -> bool:
        return self._final_index is not None

    @property
    def n_frames(self) -> int:
        if self._final_index is not None:
            return len(self._final_index.frame_order)
        return 0

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
        frame_batch: object,
        mapping_result: object,
        reference_cloud: object,
        classes_config: object,
    ) -> None:
        import pyvista as pv

        self._clear_scene_data()
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

    def _emit_setup(self, message: str, current: int, total: int) -> None:
        """Forward a one-off setup-progress event to the GUI status callback.

        Used for stages that run on the GUI thread (cloud indexing, actor
        creation, frustum build, initial GPU upload) where the user otherwise
        sees a frozen "Setting up viewer…" label.
        """
        self._notify_status(
            "setup_progress", message=message, current=int(current), total=int(total)
        )

    def _build_frustums(self, frame_batch: object, mapping_result: object) -> None:
        if self._plotter is None:
            return
        mapping_indices = np.asarray(mapping_result.frame_indices, dtype=np.int32).reshape(-1)
        intrinsics = np.asarray(mapping_result.intrinsics, dtype=np.float64)
        depth_h, depth_w = mapping_result.depth_maps[0].shape
        fy = float(intrinsics[1, 1])
        fov_y = 2.0 * np.arctan(depth_h / (2.0 * fy))
        aspect = depth_w / max(depth_h, 1)

        mi_lookup = {int(fid): i for i, fid in enumerate(mapping_indices.tolist())}

        frame_ids = [int(f.frame_index) for f in frame_batch.frames]
        total = len(frame_ids)
        # Throttle progress emits so we don't spam processEvents for hundreds
        # of frames — every ~16 frames is plenty to keep the bar moving.
        emit_every = max(1, total // 32)
        for i, frame_idx in enumerate(frame_ids):
            if i % emit_every == 0 or i == total - 1:
                self._emit_setup("Building camera frustums", i, total)
            mi = mi_lookup.get(frame_idx)
            if mi is None:
                continue
            pose_w_c = np.asarray(mapping_result.poses_w_c[mi], dtype=np.float64)
            pts = _build_frustum_lines(pose_w_c, fov_y, aspect)
            pd = _make_line_segments_polydata(pts)
            actor = self._plotter.add_mesh(
                pd, color=(0.5, 0.5, 0.5), line_width=1, opacity=0.6,
                name=f"frustum_{frame_idx}",
            )
            self._frustum_actors[frame_idx] = actor
        if total:
            self._emit_setup("Building camera frustums", total, total)

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
            for actor in self._frustum_actors.values():
                _remove(actor)
            if self._live_actor is not None:
                _remove(self._live_actor)
            if self._simple_actor is not None:
                _remove(self._simple_actor)
            if self._picked_actor_inner is not None:
                _remove(self._picked_actor_inner)
            if self._picked_actor_outer is not None:
                _remove(self._picked_actor_outer)
            try:
                self._plotter.render()
            except Exception:
                pass
        self._class_actors.clear()
        self._class_polydata.clear()
        self._frustum_actors.clear()
        self._live_actor = None
        self._live_polydata = None
        self._simple_actor = None
        self._picked_actor_inner = None
        self._picked_actor_outer = None
        self._final_index = None
        self._live_cache = None
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
        center = positions.mean(axis=0)
        extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        self._plotter.camera.focal_point = tuple(center.tolist())
        cam_pos = center + np.array([0.0, 0.0, extent * 1.5], dtype=np.float64)
        self._plotter.camera.position = tuple(cam_pos.tolist())
        self._plotter.camera.up = (0.0, 1.0, 0.0)
        self._plotter.reset_camera()

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
        fi = self._final_index
        current_frame = None
        if fi is not None and len(fi.frame_order) > 0:
            tt = int(np.clip(t, 0, len(fi.frame_order) - 1))
            current_frame = int(fi.frame_order[tt])
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
        """Return the RGB/seg/depth composite shown in the image panel.

        Used by the GUI to export the current frame as a PNG. Returns the
        same uint8 RGB stack the panel last rendered, or None if the viewer
        has no frame data yet.
        """
        if self._last_t is None:
            return None
        if self._frame_batch is None or self._mapping_result is None:
            return None
        if self._final_index is None:
            return None
        t = int(self._last_t)
        cached = self._frame_panel_cache.get(t)
        if cached is not None:
            return cached
        stacked = self._compose_frame_panel(t)
        if stacked is not None:
            self._frame_panel_cache[t] = stacked
        return stacked

    def _update_image_panel(self, t: int) -> None:
        if self._frame_batch is None or self._mapping_result is None:
            return
        if self._final_index is None:
            return

        if t in self._frame_panel_cache:
            stacked = self._frame_panel_cache[t]
        else:
            stacked = self._compose_frame_panel(t)
            if stacked is not None:
                self._frame_panel_cache[t] = stacked

        if stacked is None:
            return

        h, w, _ = stacked.shape
        qimg = QImage(stacked.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaledToWidth(
            min(w, self._image_label.width()), Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def _compose_frame_panel(self, t: int) -> np.ndarray | None:
        import cv2

        fi = self._final_index
        if fi is None or len(fi.frame_order) == 0:
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
        return np.concatenate([rgb, seg_color, depth_color], axis=0)

    def _show_live_preprocess_frame(self, frame_index: int) -> None:
        import cv2

        stem = f"{frame_index:08d}"
        frame_path = self._output_dir / "frames" / f"{stem}.png"
        labels_path = self._output_dir / "labels" / f"{stem}.npy"
        if not frame_path.exists():
            return
        rgb = cv2.imread(str(frame_path))
        if rgb is None:
            return
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if labels_path.exists():
            labels = np.load(str(labels_path))
            h, w = rgb.shape[:2]
            seg_color = _colorize_seg(
                cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST),
                self._class_colors,
            )
            stacked = np.concatenate([rgb, seg_color], axis=0)
        else:
            stacked = rgb
        h, w, _ = stacked.shape
        qimg = QImage(np.ascontiguousarray(stacked).data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label_w = self._image_label.width()
        if label_w > 0:
            pixmap = pixmap.scaledToWidth(min(w, label_w), Qt.SmoothTransformation)
        self._image_label.setPixmap(pixmap)

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

    def close(self) -> None:
        self._sig_close.emit()

    def wait_forever(self) -> None:
        pass

    # --- Slots ---

    @Slot(str, str)
    def _on_start_run(self, run_label: str, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._hide_canvas()
        self._notify_status("start_run", run_label=run_label, output_dir=output_dir)

    @Slot(str, str, object)
    def _on_set_stage(self, stage: str, status: str, message: object) -> None:
        self._notify_status("set_stage", stage=stage, status=status, message=message)

    @Slot(str, int, object, object, object)
    def _on_update_progress(self, stage: str, current: int, total: object, message: object, frame_index: object) -> None:
        if stage == "preprocess" and frame_index is not None and self._output_dir is not None:
            self._show_live_preprocess_frame(int(frame_index))
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
            self.show_point_cloud(kwargs["geometry_xyz"], kwargs["geometry_rgb"])
        self._notify_status("data_ready", **kwargs)

    @Slot(str, object)
    def _on_mark_outputs(self, output_dir: str, output_files: object) -> None:
        self._notify_status("mark_outputs", output_dir=output_dir, output_files=output_files)

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
