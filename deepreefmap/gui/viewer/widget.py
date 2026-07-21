"""VTK point cloud viewer widget for the desktop app."""

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
    QHBoxLayout,
    QLabel,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.pointcloud.final_cloud_index import FinalCloudIndex, build_final_cloud_index
from deepreefmap.gui.viewer.live_frame_cloud import (
    LiveFrameCloudCache,
    build_enabled_label_lut,
    mask_points_by_enabled_lut,
)
from deepreefmap.gui.viewer.legend import LegendOverlay
from deepreefmap.gui.core.theme import (
    BORDER,
    CARD_BG,
    GROOVE,
    OVERLAY_TEXT,
    PREVIEW_BG,
    PRIMARY,
    PRIMARY_DARK,
    SLIDER_HANDLE,
    TEXT_SECONDARY,
)
from deepreefmap.gui.viewer.picking import ViewerPickingMixin
from deepreefmap.gui.viewer.render import (
    _build_frustum_lines,
    _colorize_depth,
    _colorize_seg,
    _compute_transect_view,
    _estimate_world_up,
    _make_line_segments_polydata,
    _make_point_polydata,
)

logger = logging.getLogger(__name__)

_EMPTY_XYZ = np.zeros((0, 3), dtype=np.float32)


class QtPointCloudViewer(ViewerPickingMixin, QWidget):
    """Native 3D point cloud viewer driving the orchestrator viewer protocol."""

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
        self._rgb_label.setStyleSheet(f"background-color: {PREVIEW_BG};")
        self._seg_label = QLabel()
        self._seg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._seg_label.setMinimumHeight(120)
        self._seg_label.setStyleSheet(f"background-color: {PREVIEW_BG};")
        self._depth_label = QLabel()
        self._depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._depth_label.setMinimumHeight(120)
        self._depth_label.setStyleSheet(f"background-color: {PREVIEW_BG};")
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
        slider_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: bold;")
        slider_layout.addWidget(slider_label)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setMinimumHeight(34)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frame_slider.setTickInterval(0)
        self.frame_slider.setStyleSheet(
            f"""
            QSlider::groove:horizontal {{
                height: 10px;
                background: {GROOVE};
                border: 1px solid {BORDER};
                border-radius: 5px;
            }}
            QSlider::sub-page:horizontal {{
                background: {PRIMARY};
                border: 1px solid {PRIMARY_DARK};
                border-radius: 5px;
            }}
            QSlider::add-page:horizontal {{
                background: {CARD_BG};
                border: 1px solid {BORDER};
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: {SLIDER_HANDLE};
                border: 2px solid {PRIMARY_DARK};
                width: 18px;
                height: 26px;
                margin: -10px 0;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal:hover {{ background: #ffffff; }}
            QSlider::tick:horizontal {{ background: #777; }}
            """
        )
        slider_layout.addWidget(self.frame_slider, 1)
        self._frame_readout = QLabel("0 / 0")
        self._frame_readout.setStyleSheet(
            f'color: {OVERLAY_TEXT}; font-family: "JetBrains Mono"; min-width: 80px;'
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
        # The top pane is either the 3D canvas or an injected placeholder (the
        # progress panel); the preview toggle gates which one shows.
        self._placeholder_container = QWidget()
        placeholder_layout = QVBoxLayout(self._placeholder_container)
        placeholder_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas_stack = QStackedWidget()
        self._canvas_stack.addWidget(self._placeholder_container)
        self._canvas_stack.addWidget(self._canvas_container)
        self._main_splitter.addWidget(self._canvas_stack)
        self._main_splitter.addWidget(self._frames_panel)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 1)
        self._canvas_revealed = False
        self._canvas_wanted = False
        self._canvas_allowed = True

        # Slim header row above the canvas for controls that belong to the
        # viewer itself (the 3D preview toggle). A header rather than a canvas
        # overlay so the controls stay reachable while the placeholder shows.
        self._header_row = QHBoxLayout()
        self._header_row.setContentsMargins(4, 2, 4, 2)
        self._header_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._header_row)
        layout.addWidget(self._main_splitter)

        # Floating legend pinned to the canvas's top-right corner. Hidden until
        # _build_legend in the viewer controls populates it after a run loads.
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
        """Coalesce a burst of move/resize events into one repaint once motion settles."""
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
        """Scroll-wheel zoom toward the cursor position instead of screen center."""
        # Observers go on the interactor, not the style, so zoom keeps working in
        # pick mode as well as navigate.
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

    def _render_canvas_safe(self) -> None:
        """Force a GL re-render; without it the translucent overlay ghosts stale pixels."""
        if self._plotter is None:
            return
        if self._canvas_stack.currentWidget() is not self._canvas_container:
            return
        try:
            self._plotter.render()
        except Exception:
            pass

    def set_placeholder_widget(self, widget: QWidget) -> None:
        """Install the widget shown in place of the 3D canvas while it is off."""
        layout = self._placeholder_container.layout()
        assert layout is not None
        layout.addWidget(widget)

    def add_header_widget(self, widget: QWidget) -> None:
        """Dock a control into the viewer's slim header row, right-aligned."""
        self._header_row.addWidget(widget)

    def set_canvas_allowed(self, allowed: bool) -> None:
        """Gate the 3D canvas. Scene data keeps flowing while disallowed; allowing
        mid-run reveals a canvas that has been fed all along."""
        self._canvas_allowed = allowed
        if allowed and self._canvas_wanted:
            self._reveal_canvas()
        elif not allowed:
            self._canvas_revealed = False
            self._canvas_stack.setCurrentWidget(self._placeholder_container)

    def _reveal_canvas(self) -> None:
        self._canvas_wanted = True
        if not self._canvas_allowed:
            return
        self._ensure_plotter()
        self._canvas_revealed = True
        self._canvas_stack.setCurrentWidget(self._canvas_container)
        # Re-apply on every call: an early bail leaves the bottom panel oversized
        # after the second set_data (semantic, following the geometry preview) and
        # after the sidebar switches to Results.
        total = max(self._main_splitter.height(), self._main_splitter.sizeHint().height(), 400)
        self._main_splitter.setSizes([int(total * 0.75), int(total * 0.25)])

    def _hide_canvas(self) -> None:
        self._canvas_wanted = False
        self._canvas_revealed = False
        self._canvas_stack.setCurrentWidget(self._placeholder_container)

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

        No labels, so no FinalCloudIndex and no legend: the cloud is static and the
        slider only moves the frustum highlight and image panel.
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
        """Forward a one-off setup-progress event so GUI-thread stages don't look frozen."""
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
        """Re-orient the camera to the default transect-lengthwise view."""
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
        labels_path = self._output_dir / "labels" / f"{stem}.png"
        if not frame_path.exists():
            return
        rgb = cv2.imread(str(frame_path))
        if rgb is None:
            return
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        self._paint_label(self._rgb_label, rgb)
        if labels_path.exists():
            labels = cv2.imread(str(labels_path), cv2.IMREAD_GRAYSCALE)
            if labels is None:
                return
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
