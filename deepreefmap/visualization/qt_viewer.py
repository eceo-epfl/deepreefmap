from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget
import vispy
vispy.use(gl="gl2")
from vispy import scene  # noqa: E402

from deepreefmap.visualization.final_cloud_index import FinalCloudIndex, build_final_cloud_index  # noqa: E402
from deepreefmap.visualization.live_frame_cloud import (  # noqa: E402
    LiveFrameCloudCache,
    build_enabled_label_lut,
    mask_points_by_enabled_lut,
)

logger = logging.getLogger(__name__)

_EMPTY_XYZ = np.zeros((0, 3), dtype=np.float32)
_EMPTY_RGBA = np.zeros((0, 4), dtype=np.float32)


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


class QtPointCloudViewer(QWidget):
    _sig_start_run = Signal(str, str)
    _sig_set_stage = Signal(str, str, object)
    _sig_update_progress = Signal(str, int, object, object, object)
    _sig_data_ready = Signal(object)
    _sig_mark_outputs = Signal(str, object)
    _sig_fail_run = Signal(str, str)
    _sig_close = Signal()

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

        self._canvas = scene.SceneCanvas(keys="interactive", show=False)
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.TurntableCamera(fov=60, distance=5.0)
        self._view.bgcolor = "#141414"

        self._simple_markers = scene.visuals.Markers(parent=self._view.scene)
        self._simple_markers.scaling = "fixed"
        self._live_markers = scene.visuals.Markers(parent=self._view.scene)
        self._live_markers.scaling = "fixed"
        self._class_markers: dict[int, scene.visuals.Markers] = {}
        self._frustum_visuals: dict[int, scene.visuals.Line] = {}

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

        self._frame_panel_cache: dict[int, np.ndarray] = {}

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumHeight(120)
        self._image_label.setStyleSheet("background-color: #1a1a1a;")

        splitter = QSplitter(Qt.Vertical)
        canvas_container = QWidget()
        cl = QVBoxLayout(canvas_container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self._canvas.native)
        splitter.addWidget(canvas_container)
        splitter.addWidget(self._image_label)
        splitter.setSizes([700, 200])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

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

    @property
    def has_scene_data(self) -> bool:
        return self._final_index is not None

    @property
    def n_frames(self) -> int:
        if self._final_index is not None:
            return len(self._final_index.frame_order)
        return 0

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
        positions = np.ascontiguousarray(xyz, dtype=np.float32)
        self._simple_markers.set_data(
            pos=positions, face_color=_to_rgba(rgb), edge_width=0,
            size=point_size,
        )
        self._simple_markers.visible = True
        self._auto_fit_camera(positions)

    # --- Scene data ---

    def load_scene_data(
        self,
        frame_batch: object,
        mapping_result: object,
        reference_cloud: object,
        classes_config: object,
    ) -> None:
        self._clear_scene_data()
        self._frame_batch = frame_batch
        self._mapping_result = mapping_result

        frame_order = [int(f.frame_index) for f in frame_batch.frames]
        self._final_index = build_final_cloud_index(
            reference_cloud, frame_order, self._class_colors,
        )
        self._live_cache = LiveFrameCloudCache(
            frame_batch, mapping_result, self._final_index.frame_order,
        )
        self._max_label_id = max(
            (max(self._class_colors.keys(), default=0)),
            max(self._final_index.class_ids, default=0),
        )

        for cid in self._final_index.class_ids:
            m = scene.visuals.Markers(parent=self._view.scene)
            m.scaling = "fixed"
            m.visible = False
            self._class_markers[cid] = m

        self._build_frustums(frame_batch, mapping_result)

        self._simple_markers.visible = False
        self._last_t = None

        if self._final_index.class_ids:
            all_xyz = [self._final_index.xyz_by_class[c] for c in self._final_index.class_ids if c in self._final_index.xyz_by_class]
            if all_xyz:
                combined = np.concatenate(all_xyz, axis=0)
                if combined.shape[0] > 0:
                    self._auto_fit_camera(combined)

        self._notify_status("scene_loaded")

    def _build_frustums(self, frame_batch: object, mapping_result: object) -> None:
        mapping_indices = np.asarray(mapping_result.frame_indices, dtype=np.int32).reshape(-1)
        intrinsics = np.asarray(mapping_result.intrinsics, dtype=np.float64)
        depth_h, depth_w = mapping_result.depth_maps[0].shape
        fy = float(intrinsics[1, 1])
        fov_y = 2.0 * np.arctan(depth_h / (2.0 * fy))
        aspect = depth_w / max(depth_h, 1)

        mi_lookup = {int(fid): i for i, fid in enumerate(mapping_indices.tolist())}

        for frame_idx in (int(f.frame_index) for f in frame_batch.frames):
            mi = mi_lookup.get(frame_idx)
            if mi is None:
                continue
            pose_w_c = np.asarray(mapping_result.poses_w_c[mi], dtype=np.float64)
            pts = _build_frustum_lines(pose_w_c, fov_y, aspect)
            line = scene.visuals.Line(
                pos=pts, color=(0.5, 0.5, 0.5, 0.6), width=1,
                connect="segments", parent=self._view.scene,
            )
            self._frustum_visuals[frame_idx] = line

    def _clear_scene_data(self) -> None:
        for m in self._class_markers.values():
            m.parent = None
        self._class_markers.clear()
        self._live_markers.set_data(pos=_EMPTY_XYZ[:, :3] if _EMPTY_XYZ.shape[1] >= 3 else np.zeros((0, 3), dtype=np.float32))
        self._live_markers.visible = False
        for v in self._frustum_visuals.values():
            v.parent = None
        self._frustum_visuals.clear()
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

    def _auto_fit_camera(self, positions: np.ndarray) -> None:
        center = positions.mean(axis=0)
        extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        self._view.camera.center = tuple(center.tolist())
        self._view.camera.distance = extent * 1.5

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
            return

        # Live cloud
        try:
            xyz_u, rgb_u, lab_u, conf_u = self._live_cache.get_unmasked(t)
        except Exception:
            xyz_u = _EMPTY_XYZ
            rgb_u = np.zeros((0, 3), dtype=np.uint8)
            lab_u = np.zeros((0,), dtype=np.int32)
            conf_u = np.zeros((0,), dtype=np.float32)

        if xyz_u.shape[0] > 0:
            max_id = max(self._max_label_id, int(lab_u.max()) if lab_u.size else 0)
            lut = build_enabled_label_lut(max_id, set(enabled_classes))
            m = mask_points_by_enabled_lut(lab_u, lut)
            if min_conf > 0.0 and conf_u.size:
                m &= conf_u >= min_conf
            xyz_live = xyz_u[m]
            if semantic_colors:
                cols_live = np.full((xyz_live.shape[0], 3), 128, dtype=np.uint8)
                for cid, color in self._class_colors.items():
                    cols_live[lab_u[m] == cid] = color
            else:
                cols_live = rgb_u[m]
            if xyz_live.shape[0] > 0:
                self._live_markers.set_data(
                    pos=np.ascontiguousarray(xyz_live, dtype=np.float32),
                    face_color=_to_rgba(cols_live), edge_width=0,
                    size=point_size,
                )
                self._live_markers.visible = True
            else:
                self._live_markers.visible = False
        else:
            self._live_markers.visible = False

        # Final cloud per class
        for cid, markers in self._class_markers.items():
            if cid not in enabled_classes:
                markers.visible = False
                continue
            xyz_c = fi.xyz_by_class.get(cid)
            if xyz_c is None or xyz_c.shape[0] == 0:
                markers.visible = False
                continue
            n = int(fi.prefix_end_by_class[cid][t]) if accumulate else 0
            if n <= 0:
                markers.visible = False
                continue
            src = fi.semrgb_by_class[cid] if semantic_colors else fi.rgb_by_class[cid]
            pts = xyz_c[:n]
            cols = src[:n]
            if min_conf > 0.0:
                conf_c = fi.conf_by_class.get(cid)
                if conf_c is not None:
                    keep = conf_c[:n] >= min_conf
                    pts = pts[keep]
                    cols = cols[keep]
            if pts.shape[0] == 0:
                markers.visible = False
                continue
            markers.set_data(
                pos=np.ascontiguousarray(pts, dtype=np.float32),
                face_color=_to_rgba(cols), edge_width=0,
                size=point_size,
            )
            markers.visible = True

        self._update_frustum_visibility(frustums_visible, t)
        self._update_image_panel(t)

        self._last_t = t
        self._last_accumulate = accumulate
        self._last_semantic = semantic_colors
        self._last_enabled = enabled_classes
        self._last_confidence = min_conf
        self._last_point_size = point_size

    def _update_point_sizes(self, point_size: float) -> None:
        self._last_point_size = point_size

    def _update_frustum_visibility(self, visible: bool, t: int) -> None:
        fi = self._final_index
        current_frame = None
        if fi is not None and len(fi.frame_order) > 0:
            tt = int(np.clip(t, 0, len(fi.frame_order) - 1))
            current_frame = int(fi.frame_order[tt])
        for fid, line in self._frustum_visuals.items():
            line.visible = visible
            if visible and fid == current_frame:
                line.set_data(color=(1.0, 0.8, 0.25, 0.9), width=2)
            elif visible:
                line.set_data(color=(0.5, 0.5, 0.5, 0.6), width=1)

    # --- Image panel ---

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
        self._notify_status("start_run", run_label=run_label, output_dir=output_dir)

    @Slot(str, str, object)
    def _on_set_stage(self, stage: str, status: str, message: object) -> None:
        self._notify_status("set_stage", stage=stage, status=status, message=message)

    @Slot(str, int, object, object, object)
    def _on_update_progress(self, stage: str, current: int, total: object, message: object, frame_index: object) -> None:
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
        pass

    def _notify_status(self, event: str, **kwargs: object) -> None:
        if self._status_callback is not None:
            try:
                self._status_callback(event, **kwargs)
            except Exception:
                logger.exception("Status callback error")
