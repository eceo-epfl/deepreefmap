from __future__ import annotations

from contextlib import suppress
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

from deepreefmap.io.exports import save_ortho_grid
from deepreefmap.pointcloud.grid_ortho import OrthoGrid, aggregate_cloud_to_ortho_grid
from deepreefmap.pointcloud.transect_crop import (
    TransectCropGeometry,
    TransectCropSelection,
    build_transect_crop_geometry,
    build_transect_crop_selection,
    point_mask_with_transect_selection,
)
from deepreefmap.postproc.ortho_outputs import OrthoOutputs, TransectCropParams, apply_ortho_crop
from deepreefmap.postproc.reports import save_cover_report
from deepreefmap.visualization.final_cloud_index import build_final_cloud_index, median_distance_to_camera
from deepreefmap.visualization.live_frame_cloud import LiveFrameCloudCache
from deepreefmap.visualization.viser_scene import ViserSceneController, rotation_to_wxyz

if TYPE_CHECKING:
    from deepreefmap.config.classes import ClassConfig
    from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, SemanticPointCloud

logger = logging.getLogger(__name__)

_VISER_NAN_GUARD_PATCHED = False


def _install_viser_nan_slider_guard() -> None:
    """Best-effort guard for transient NaN slider updates from the browser."""

    global _VISER_NAN_GUARD_PATCHED
    if _VISER_NAN_GUARD_PATCHED:
        return
    try:
        from viser import _gui_api
    except Exception:
        return

    gui_api_cls = getattr(_gui_api, "GuiApi", None)
    if gui_api_cls is None:
        gui_api_cls = getattr(_gui_api, "_GuiApi", None)
    if gui_api_cls is None:
        return

    original = getattr(gui_api_cls, "_handle_gui_updates", None)
    if original is None or getattr(original, "_deepreefmap_nan_guard", False):
        _VISER_NAN_GUARD_PATCHED = True
        return

    async def _guarded_handle_gui_updates(self, client_id, message):  # type: ignore[no-untyped-def]
        try:
            return await original(self, client_id, message)
        except ValueError as exc:
            text = str(exc)
            if "cannot convert float NaN to integer" in text:
                logger.debug("Dropped transient NaN slider GUI update from client_id=%s", client_id)
                return
            raise

    setattr(_guarded_handle_gui_updates, "_deepreefmap_nan_guard", True)
    gui_api_cls._handle_gui_updates = _guarded_handle_gui_updates
    _VISER_NAN_GUARD_PATCHED = True


class ViserLiveApp:
    """Live viser server: pipeline status + reconstruction scene (post `set_data`)."""

    def __init__(
        self,
        class_colors: dict[int, tuple[int, int, int]] | None = None,
        class_names: dict[int, str] | None = None,
        port: int = 8080,
    ) -> None:
        self.enabled = False
        self.startup_error: str | None = None
        self._server = None
        self._class_colors = class_colors or {}
        self._class_names = class_names or {}

        self._frame_order: tuple[int, ...] = ()
        self._frame_panel_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._stacked_image_cache: dict[int, np.ndarray] = {}
        self._seg_color_cache: dict[int, np.ndarray] = {}
        self._depth_color_cache: dict[int, np.ndarray] = {}

        self._semantic_color_toggle = None
        self._point_size_slider = None
        self._confidence_slider = None
        self._frame_slider = None
        self._playing_toggle = None
        self._fps_slider = None
        self._accumulate_toggle = None
        self._hide_frustums_toggle = None
        self._view_current_camera_button = None
        self._follow_camera_toggle = None
        self._camera_backoff_slider = None
        self._stacked_handle = None
        self._legend_toggles: dict[int, object] = {}
        self._download_stacked_button = None
        self._latest_stacked_image: np.ndarray | None = None
        self._output_dir: Path | None = None
        self._ortho_base_grid: OrthoGrid | None = None
        self._ortho_classes_config = None
        self._ortho_crop_geometry: TransectCropGeometry | None = None
        self._ortho_crop_selection: TransectCropSelection | None = None
        self._current_ortho_outputs: OrthoOutputs | None = None
        self._active_crop_params: TransectCropParams | None = None
        self._crop_revision = 0
        self._ortho_image_handle = None
        self._crop_enabled_toggle = None
        self._transect_length_slider = None
        self._crop_width_slider = None
        self._save_ortho_button = None
        self._crop_summary_markdown_handle = None
        self._suppress_frame_slider_callback = False
        self._camera_view_by_frame: dict[int, tuple[np.ndarray, float]] = {}
        self._status_markdown_handle = None
        self._outputs_markdown_handle = None
        self._preprocess_progress_slider = None
        self._mapping_progress_slider = None
        self._stage_states: dict[str, tuple[str, str]] = {}

        self._scene_controller: ViserSceneController | None = None
        self._scene_ready = False
        self._render_lock = threading.Lock()
        self._dirty = threading.Event()
        self._stop_render = threading.Event()
        self._render_thread: threading.Thread | None = None

        try:
            import viser

            _install_viser_nan_slider_guard()
            self._server = viser.ViserServer(port=port)
            with suppress(Exception):
                self._server.gui.configure_theme(
                    dark_mode=True,
                    control_width="large",
                    control_layout="collapsible",
                )
            with suppress(Exception):
                self._server.gui.configure_theme(dark_mode=True, control_width="large")
            self.enabled = True

            self._semantic_color_toggle = self._server.gui.add_checkbox("Semantic cloud colors", False)
            self._point_size_slider = self._server.gui.add_slider("Point size", 0.0001, 0.05, 0.0001, 0.002)
            self._confidence_slider = self._server.gui.add_slider(
                "Min confidence (%)", 0.0, 100.0, 0.1, 0.0
            )
            self._frame_slider = self._server.gui.add_slider("Frame", 0.0, 0.0, 1.0, 0.0, disabled=True)
            self._playing_toggle = self._server.gui.add_checkbox("Playing", False)
            self._fps_slider = self._server.gui.add_slider("FPS", 1.0, 60.0, 0.5, 8.0)
            self._accumulate_toggle = self._server.gui.add_checkbox("Accumulate", True)
            self._hide_frustums_toggle = self._server.gui.add_checkbox("Hide camera frustums", False)
            with self._server.gui.add_folder("Camera view"):
                self._view_current_camera_button = self._server.gui.add_button("View from current camera")
                self._follow_camera_toggle = self._server.gui.add_checkbox("Follow current camera", False)
                self._camera_backoff_slider = self._server.gui.add_slider("Camera backoff", 0.0, 2.0, 0.01, 0.0)
                self._view_current_camera_button.disabled = True
                with suppress(Exception):
                    self._follow_camera_toggle.disabled = True
            self._stacked_handle = self._server.gui.add_image(
                np.zeros((3, 3, 3), dtype=np.uint8),
                label="RGB / Segmentation / Depth (stacked)",
            )
            with self._server.gui.add_folder("Semantic legend"):
                for index, class_id in enumerate(sorted(self._class_colors)):
                    class_id_i = int(class_id)
                    order_base = float(index) * 2.0
                    self._server.gui.add_image(
                        self._legend_swatch_image(class_id_i),
                        order=order_base,
                    )
                    toggle = self._server.gui.add_checkbox(
                        self._legend_checkbox_label(class_id_i),
                        True,
                        order=order_base + 1.0,
                    )
                    self._legend_toggles[int(class_id)] = toggle

                    @toggle.on_update
                    def _(_u, _cid: int = int(class_id)) -> None:  # noqa: ARG001
                        logger.debug("Legend toggle for class %d → enabled=%s", _cid, self._legend_toggles[_cid].value)
                        self._mark_dirty()

            self._download_stacked_button = self._server.gui.add_button("Download stacked RGB/Seg/Depth")
            self._download_stacked_button.disabled = True
            with self._server.gui.add_folder("Ortho crop"):
                self._ortho_image_handle = self._server.gui.add_image(
                    np.zeros((3, 3, 3), dtype=np.uint8),
                    label="Ortho RGB / classes (stacked)",
                )
                self._crop_enabled_toggle = self._server.gui.add_checkbox("Enable transect crop", False)
                self._transect_length_slider = self._server.gui.add_slider(
                    "Transect length (m)", 0.1, 200.0, 0.1, 10.0
                )
                self._crop_width_slider = self._server.gui.add_slider("Crop width (m)", 0.1, 50.0, 0.1, 2.0)
                self._save_ortho_button = self._server.gui.add_button("Save current ortho + cover")
                self._crop_summary_markdown_handle = self._server.gui.add_markdown("Ortho: waiting for data")
            with self._server.gui.add_folder("Pipeline status"):
                self._status_markdown_handle = self._server.gui.add_markdown("Stage: idle")
                self._preprocess_progress_slider = self._server.gui.add_slider(
                    "Preprocess progress",
                    0.0,
                    100.0,
                    0.1,
                    0.0,
                    disabled=True,
                )
                self._mapping_progress_slider = self._server.gui.add_slider(
                    "Mapping progress",
                    0.0,
                    100.0,
                    0.1,
                    0.0,
                    disabled=True,
                )
                self._outputs_markdown_handle = self._server.gui.add_markdown("Outputs: pending")

            @self._semantic_color_toggle.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._mark_dirty()

            @self._point_size_slider.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._mark_dirty()

            @self._confidence_slider.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._mark_dirty()

            @self._frame_slider.on_update
            def _(_u) -> None:  # noqa: ARG001
                if self._suppress_frame_slider_callback:
                    return
                self._validate_frame_slider_value(strict=True)
                self._mark_dirty()

            @self._accumulate_toggle.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._mark_dirty()

            @self._hide_frustums_toggle.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._mark_dirty()

            @self._view_current_camera_button.on_click
            def _(_u) -> None:  # noqa: ARG001
                self._apply_camera_view_to_clients()

            @self._follow_camera_toggle.on_update
            def _(_u) -> None:  # noqa: ARG001
                if bool(self._follow_camera_toggle is not None and self._follow_camera_toggle.value):
                    self._apply_camera_view_to_clients()

            @self._camera_backoff_slider.on_update
            def _(_u) -> None:  # noqa: ARG001
                if bool(self._follow_camera_toggle is not None and self._follow_camera_toggle.value):
                    self._apply_camera_view_to_clients()

            @self._download_stacked_button.on_click
            def _(_u) -> None:  # noqa: ARG001
                self._save_stacked_image()

            if self._crop_enabled_toggle is not None:
                @self._crop_enabled_toggle.on_update
                def _(_u) -> None:  # noqa: ARG001
                    self._refresh_ortho_crop_preview()

            if self._transect_length_slider is not None:
                @self._transect_length_slider.on_update
                def _(_u) -> None:  # noqa: ARG001
                    self._refresh_ortho_crop_preview()

            if self._crop_width_slider is not None:
                @self._crop_width_slider.on_update
                def _(_u) -> None:  # noqa: ARG001
                    self._refresh_ortho_crop_preview()

            if self._save_ortho_button is not None:
                @self._save_ortho_button.on_click
                def _(_u) -> None:  # noqa: ARG001
                    self._save_current_ortho_outputs()
        except Exception as exc:
            self.startup_error = str(exc) or exc.__class__.__name__
            self.enabled = False

    def set_data(
        self,
        frame_batch: "FrameBatch",
        mapping_result: "MappingSequenceResult",
        reference_cloud: "SemanticPointCloud",
        classes_config: "ClassConfig",
        ortho_bins: int = 1000,
        ortho_cloud: "SemanticPointCloud | None" = None,
        ortho_grid: OrthoGrid | None = None,
    ) -> None:
        """Build scene graph and caches after reconstruction (single bulk load)."""
        if not self.enabled or self._server is None:
            return

        frame_order = tuple(int(x) for x in frame_batch.frame_indices)
        self._frame_order = frame_order
        self._frame_panel_data.clear()
        for frame in frame_batch.frames:
            try:
                est = mapping_result.estimate_for_index(int(frame.frame_index))
            except KeyError:
                continue
            self._frame_panel_data[int(frame.frame_index)] = (
                frame.image_rgb,
                frame.labels,
                np.asarray(est.depth, dtype=np.float32),
            )
        self._stacked_image_cache.clear()
        self._seg_color_cache.clear()
        self._depth_color_cache.clear()
        self._camera_view_by_frame.clear()

        depth_viz_cap = median_distance_to_camera(reference_cloud)
        final_index = build_final_cloud_index(
            reference_cloud,
            list(frame_order),
            classes_config.id_to_color,
        )
        live_cache = LiveFrameCloudCache(
            frame_batch,
            mapping_result,
            frame_order,
            max_depth_for_viz=depth_viz_cap,
        )
        if ortho_grid is None:
            ortho_source = reference_cloud if ortho_cloud is None else ortho_cloud
            ortho_grid = aggregate_cloud_to_ortho_grid(ortho_source, bins=ortho_bins)
        self._ortho_base_grid = ortho_grid
        self._ortho_classes_config = classes_config
        self._ortho_crop_geometry = build_transect_crop_geometry(
            labels=self._ortho_base_grid.labels,
            transect_label=classes_config.single_id_for_role("transect_line"),
            transect_tools_label=classes_config.single_id_for_role("transect_tools"),
        )

        k = np.asarray(mapping_result.intrinsics, dtype=np.float64)
        fy = float(max(k[1, 1], 1e-6))
        frustum_specs: list[tuple[int, np.ndarray, int, int, float]] = []
        for fid in frame_order:
            try:
                est = mapping_result.estimate_for_index(int(fid))
            except KeyError:
                continue
            frame = next((f for f in frame_batch.frames if int(f.frame_index) == int(fid)), None)
            if frame is None:
                continue
            h, w = frame.image_rgb.shape[:2]
            fov_y = float(2.0 * np.arctan(h / (2.0 * fy)))
            pose_w_c = np.asarray(est.pose_w_c, dtype=np.float64)
            frustum_specs.append((int(fid), pose_w_c, w, h, fov_y))
            self._camera_view_by_frame[int(fid)] = (pose_w_c, fov_y)

        with self._render_lock:
            if self._scene_controller is None:
                self._scene_controller = ViserSceneController(self._server.scene)
            else:
                self._scene_controller.clear_dynamic_nodes()

            ps = 0.002 if self._point_size_slider is None else float(self._point_size_slider.value)
            self._scene_controller.build(
                final_index,
                live_cache,
                classes_config.id_to_color,
                frustum_specs,
                initial_point_size=ps,
            )
            self._wire_frustum_clicks()
            self._update_frame_slider_limits()
            self._suppress_frame_slider_callback = True
            try:
                if self._frame_slider is not None and frame_order:
                    self._frame_slider.value = float(max(len(frame_order) - 1, 0))
            finally:
                self._suppress_frame_slider_callback = False
            self._scene_ready = True

        if self._download_stacked_button is not None:
            self._download_stacked_button.disabled = False
        if self._view_current_camera_button is not None:
            self._view_current_camera_button.disabled = False
        if self._follow_camera_toggle is not None:
            with suppress(Exception):
                self._follow_camera_toggle.disabled = False
        self._refresh_ortho_crop_preview()
        self._mark_dirty()

        if self._render_thread is None or not self._render_thread.is_alive():
            self._stop_render.clear()
            self._render_thread = threading.Thread(target=self._render_loop, name="viser-render", daemon=True)
            self._render_thread.start()

    def _wire_frustum_clicks(self) -> None:
        if self._scene_controller is None:
            return
        for fid, hnd in self._scene_controller.iter_frustum_handles():
            if not hasattr(hnd, "on_click"):
                continue
            fi = int(fid)

            @hnd.on_click
            def _on_click(_e, frame_index: int = fi) -> None:  # type: ignore[no-untyped-def]
                self._select_frame_by_source_index(frame_index)

    def _select_frame_by_source_index(self, frame_index: int) -> None:
        if int(frame_index) not in self._frame_order:
            return
        pos = self._frame_order.index(int(frame_index))
        if self._frame_slider is not None:
            self._suppress_frame_slider_callback = True
            try:
                self._frame_slider.value = float(int(pos))
            finally:
                self._suppress_frame_slider_callback = False
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self._dirty.set()

    def _render_loop(self) -> None:
        while not self._stop_render.is_set():
            if self._playing_toggle is not None and bool(self._playing_toggle.value) and self._scene_ready:
                if self._frame_slider is not None and self._frame_order:
                    max_pos = max(len(self._frame_order) - 1, 0)
                    pos = self._normalize_slider_position(self._frame_slider.value, max_pos)
                    next_pos = (int(pos) + 1) % len(self._frame_order)
                    self._suppress_frame_slider_callback = True
                    try:
                        self._frame_slider.value = float(int(next_pos))
                    finally:
                        self._suppress_frame_slider_callback = False
                self._dirty.set()
                fps = 8.0 if self._fps_slider is None else max(1.0, float(self._fps_slider.value))
                time.sleep(1.0 / fps)
            else:
                self._dirty.wait(timeout=0.05)

            if not self._dirty.is_set():
                continue
            self._dirty.clear()
            self._apply_state_once()

    def _apply_state_once(self) -> None:
        if not self._scene_ready or self._scene_controller is None:
            return
        with self._render_lock:
            t = self._safe_slider_value()
            if t is None:
                t = 0
            accumulate = bool(self._accumulate_toggle is not None and self._accumulate_toggle.value)
            semantic = bool(self._semantic_color_toggle is not None and self._semantic_color_toggle.value)
            hide_frustums = bool(self._hide_frustums_toggle is not None and self._hide_frustums_toggle.value)
            enabled = self._enabled_class_set()
            ps = 0.002 if self._point_size_slider is None else float(self._point_size_slider.value)
            min_conf = 0.0 if self._confidence_slider is None else float(self._confidence_slider.value) / 100.0
            self._scene_controller.apply_state(
                timeline_t=int(t),
                accumulate=accumulate,
                enabled_classes=enabled,
                semantic_colors=semantic,
                point_size=ps,
                frustums_visible=not hide_frustums,
                min_confidence=min_conf,
                crop_filter=self._point_cloud_crop_filter(),
                crop_version=self._crop_revision,
            )
            self._refresh_image_panel_for_timeline(int(t))
            if bool(self._follow_camera_toggle is not None and self._follow_camera_toggle.value):
                self._apply_camera_view_to_clients(int(t))

    @staticmethod
    def _camera_view_params(
        pose_w_c: np.ndarray,
        fov_y: float,
        backoff: float = 0.0,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float], float]:
        pose = np.asarray(pose_w_c, dtype=np.float64)
        position = pose[:3, 3].astype(np.float64, copy=True)
        if backoff > 0.0:
            forward = pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            norm = float(np.linalg.norm(forward))
            if norm > 1e-8:
                position -= (forward / norm) * float(backoff)
        wxyz = rotation_to_wxyz(pose[:3, :3])
        position_xyz = (float(position[0]), float(position[1]), float(position[2]))
        return position_xyz, wxyz, float(fov_y)

    def _apply_camera_view_to_clients(self, slider_pos: int | None = None) -> None:
        if self._server is None or not self._frame_order:
            return
        pos = self._safe_slider_value() if slider_pos is None else int(slider_pos)
        if pos is None or pos < 0 or pos >= len(self._frame_order):
            return
        frame_index = int(self._frame_order[pos])
        spec = self._camera_view_by_frame.get(frame_index)
        if spec is None:
            return
        backoff = 0.0 if self._camera_backoff_slider is None else float(self._camera_backoff_slider.value)
        position, wxyz, fov = self._camera_view_params(spec[0], spec[1], backoff=backoff)
        for client in self._server.get_clients().values():
            with suppress(Exception):
                client.camera.position = np.asarray(position, dtype=np.float64)
                client.camera.wxyz = wxyz
                client.camera.fov = fov

    def _enabled_class_set(self) -> frozenset[int]:
        if not self._legend_toggles:
            return frozenset()
        return frozenset(int(cid) for cid, toggle in self._legend_toggles.items() if bool(toggle.value))

    def _refresh_image_panel_for_timeline(self, slider_pos: int) -> None:
        if not self._frame_order or slider_pos < 0 or slider_pos >= len(self._frame_order):
            self._latest_stacked_image = None
            if self._stacked_handle is not None:
                self._stacked_handle.image = np.zeros((3, 3, 3), dtype=np.uint8)
            self._sync_download_button_state()
            return
        fid = int(self._frame_order[slider_pos])
        stacked = self._stacked_image_for_frame(fid)
        self._latest_stacked_image = stacked
        self._sync_download_button_state()
        if self._stacked_handle is not None:
            self._stacked_handle.image = stacked

    def _stacked_image_for_frame(self, frame_index: int) -> np.ndarray:
        cached = self._stacked_image_cache.get(int(frame_index))
        if cached is not None:
            return cached
        payload = self._frame_panel_data.get(int(frame_index))
        if payload is None:
            return np.zeros((3, 3, 3), dtype=np.uint8)
        image_rgb, seg, depth = payload
        seg_color = self._seg_color_cache.get(int(frame_index))
        if seg_color is None:
            seg_color = self._colorize_seg(seg)
            self._seg_color_cache[int(frame_index)] = seg_color
        depth_color = self._depth_color_cache.get(int(frame_index))
        if depth_color is None:
            depth_color = self._colorize_depth(depth)
            self._depth_color_cache[int(frame_index)] = depth_color
        stacked = self._compose_stacked(image_rgb, seg_color, depth_color)
        self._stacked_image_cache[int(frame_index)] = stacked
        return stacked

    def _compose_stacked(self, image_rgb: np.ndarray, seg_color: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
        target_h, target_w = image_rgb.shape[:2]
        seg_display = self._resize_to_shape(seg_color, (target_h, target_w), interpolation=cv2.INTER_NEAREST)
        depth_display = self._resize_to_shape(depth_color, (target_h, target_w), interpolation=cv2.INTER_NEAREST)
        return np.concatenate([image_rgb, seg_display, depth_display], axis=0)

    def _resize_to_shape(
        self,
        image: np.ndarray,
        shape_hw: tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        target_h, target_w = shape_hw
        h, w = image.shape[:2]
        if h == target_h and w == target_w:
            return image
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """False-color depth for the 2D panel (full range); 3D live cloud clipping stays in ``LiveFrameCloudCache``."""
        d = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(d)
        if valid.sum() == 0:
            return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
        lo, hi = np.percentile(d[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.zeros_like(d, dtype=np.float32)
        norm[valid] = np.clip((d[valid] - lo) / (hi - lo), 0.0, 1.0)
        color_bgr = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rgb[~valid] = 0
        return rgb

    def _colorize_seg(self, seg: np.ndarray) -> np.ndarray:
        s = np.asarray(seg, dtype=np.int32)
        out = np.full((s.shape[0], s.shape[1], 3), 128, dtype=np.uint8)
        if not self._class_colors:
            return out
        for class_id, rgb in self._class_colors.items():
            out[s == int(class_id)] = np.asarray(rgb, dtype=np.uint8)
        return out

    def _normalize_slider_position(self, raw_value: object, max_pos: int) -> int:
        try:
            slider_value = float(raw_value)
        except (TypeError, ValueError):
            slider_value = float(max_pos)
        if not np.isfinite(slider_value):
            slider_value = float(max_pos)
            logger.debug("Frame slider produced non-finite value; clamped to max index=%d", max_pos)
        return int(np.clip(np.rint(slider_value), 0, max_pos))

    def _safe_slider_value(self) -> int | None:
        if self._frame_slider is None:
            return None
        if not self._frame_order:
            return None
        max_pos = max(len(self._frame_order) - 1, 0)
        return self._normalize_slider_position(self._frame_slider.value, max_pos)

    def _validate_frame_slider_value(self, strict: bool = False) -> int | None:
        slider_pos = self._safe_slider_value()
        if slider_pos is None:
            return None
        if strict and self._frame_slider is not None:
            self._suppress_frame_slider_callback = True
            try:
                self._frame_slider.value = float(int(slider_pos))
            finally:
                self._suppress_frame_slider_callback = False
        return slider_pos

    def _update_frame_slider_limits(self) -> None:
        if self._frame_slider is None or not self._frame_order:
            if self._frame_slider is not None:
                self._frame_slider.disabled = True
            return
        self._frame_slider.disabled = False
        self._frame_slider.min = 0.0
        self._frame_slider.max = float(int(max(len(self._frame_order) - 1, 0)))
        self._frame_slider.step = 1.0
        self._validate_frame_slider_value(strict=True)

    def _sync_download_button_state(self) -> None:
        if self._download_stacked_button is None:
            return
        self._download_stacked_button.disabled = self._latest_stacked_image is None

    def _refresh_ortho_crop_preview(self) -> None:
        if self._ortho_base_grid is None or self._ortho_classes_config is None:
            return
        try:
            crop = self._current_crop_params()
            selection = (
                None
                if crop is None
                else build_transect_crop_selection(
                    self._ortho_crop_geometry,
                    transect_length_m=crop.transect_length_m,
                    crop_width_m=crop.crop_width_m,
                )
            )
            outputs = apply_ortho_crop(
                self._ortho_base_grid,
                self._ortho_classes_config,
                crop=crop,
                transect_geometry=self._ortho_crop_geometry,
                transect_selection=selection,
            )
        except Exception as exc:
            self._set_markdown_content(self._crop_summary_markdown_handle, f"### Ortho crop\nFailed: {exc}")
            return
        self._current_ortho_outputs = outputs
        self._active_crop_params = crop
        self._ortho_crop_selection = selection
        self._crop_revision += 1

        if self._ortho_image_handle is not None:
            self._ortho_image_handle.image = self._ortho_preview_image(
                outputs.grid,
                self._ortho_classes_config.id_to_color,
            )
        crop_status = "cropped" if outputs.cropped else "uncropped"
        self._set_markdown_content(
            self._crop_summary_markdown_handle,
            self._cover_summary_markdown(outputs.cover, outputs.grid, crop_status),
        )
        self._mark_dirty()

    def _point_cloud_crop_filter(self) -> Callable[[np.ndarray], np.ndarray] | None:
        crop = self._active_crop_params
        if crop is None or self._ortho_base_grid is None:
            return None

        base_grid = self._ortho_base_grid

        def _filter(xyz: np.ndarray) -> np.ndarray:
            return point_mask_with_transect_selection(
                grid=base_grid,
                xyz=xyz,
                selection=self._ortho_crop_selection,
            )

        return _filter

    def _save_current_ortho_outputs(self) -> None:
        if self._current_ortho_outputs is None:
            self._refresh_ortho_crop_preview()
        if self._current_ortho_outputs is None:
            self._set_markdown_content(self._crop_summary_markdown_handle, "### Ortho crop\nNothing to save yet.")
            return
        output_dir = self._output_dir or (Path.cwd() / "viser_exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        grid = self._current_ortho_outputs.grid
        cv2.imwrite(str(output_dir / "ortho.png"), cv2.cvtColor(grid.rgb, cv2.COLOR_RGB2BGR))
        save_ortho_grid(output_dir / "ortho.npz", grid)
        save_cover_report(output_dir / "benthic_cover.json", self._current_ortho_outputs.cover)
        self._set_markdown_content(
            self._crop_summary_markdown_handle,
            self._cover_summary_markdown(
                self._current_ortho_outputs.cover,
                grid,
                "cropped" if self._current_ortho_outputs.cropped else "uncropped",
            )
            + f"\n- Saved: `{output_dir}`",
        )

    def _current_crop_params(self) -> TransectCropParams | None:
        if not bool(self._crop_enabled_toggle is not None and self._crop_enabled_toggle.value):
            return None
        length_m = 10.0 if self._transect_length_slider is None else float(self._transect_length_slider.value)
        width_m = 2.0 if self._crop_width_slider is None else float(self._crop_width_slider.value)
        return TransectCropParams(transect_length_m=length_m, crop_width_m=width_m)

    @staticmethod
    def _ortho_preview_image(
        grid: OrthoGrid,
        class_colors: dict[int, tuple[int, int, int]],
        max_side: int = 900,
    ) -> np.ndarray:
        rgb = np.asarray(grid.rgb, dtype=np.uint8)
        labels = np.asarray(grid.labels, dtype=np.int32)
        class_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            class_rgb[labels == int(class_id)] = np.asarray(color, dtype=np.uint8)
        if rgb.shape[:2] != class_rgb.shape[:2]:
            class_rgb = cv2.resize(class_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        preview = np.concatenate([rgb, class_rgb], axis=1)
        h, w = preview.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w, 1)))
        if scale < 1.0:
            preview = cv2.resize(
                preview,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return preview

    @staticmethod
    def _cover_summary_markdown(cover: dict[str, object], grid: OrthoGrid, crop_status: str) -> str:
        classes = cover.get("classes", {})
        denom = float(cover.get("denominator", 0.0) or 0.0)
        rows = [
            "### Ortho crop",
            f"- State: **{crop_status}**",
            f"- Grid: `{grid.rgb.shape[1]} x {grid.rgb.shape[0]}` cells",
            f"- Cover denominator: `{denom:.1f}`",
        ]
        if not isinstance(classes, dict) or not classes:
            rows.append("- Cover: no valid cells")
            return "\n".join(rows)
        ranked = sorted(
            classes.values(),
            key=lambda item: float(item.get("fraction", 0.0)) if isinstance(item, dict) else 0.0,
            reverse=True,
        )
        for item in ranked[:8]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "unknown"))
            frac = float(item.get("fraction", 0.0) or 0.0)
            rows.append(f"- {name}: `{frac * 100.0:.1f}%`")
        return "\n".join(rows)

    def _set_markdown_content(self, handle: object, content: str) -> None:
        if handle is None:
            return
        with suppress(Exception):
            handle.content = content
            return
        with suppress(Exception):
            handle.value = content

    def _refresh_status_markdown(self) -> None:
        if not self._stage_states:
            self._set_markdown_content(self._status_markdown_handle, "Stage: idle")
            return
        rows = []
        for stage_name in ("startup", "preprocess", "mapping", "outputs"):
            state = self._stage_states.get(stage_name)
            if state is None:
                continue
            status, message = state
            rows.append(f"- **{stage_name}**: {status} - {message}")
        self._set_markdown_content(self._status_markdown_handle, "### Pipeline\n" + "\n".join(rows))

    def start_run(self, run_label: str, output_dir: str) -> None:
        self._stage_states.clear()
        self._output_dir = Path(output_dir)
        self._set_markdown_content(self._outputs_markdown_handle, f"Outputs: pending (`{output_dir}`)")
        if self._preprocess_progress_slider is not None:
            self._preprocess_progress_slider.value = 0.0
        if self._mapping_progress_slider is not None:
            self._mapping_progress_slider.value = 0.0
        self.set_stage("startup", "running", f"{run_label}")

    def set_stage(self, stage: str, status: str, message: str | None = None) -> None:
        detail = "" if message is None else str(message)
        self._stage_states[str(stage)] = (str(status), detail)
        self._refresh_status_markdown()

    def update_progress(
        self,
        stage: str,
        current: int,
        total: int | None = None,
        message: str | None = None,
        frame_index: int | None = None,
    ) -> None:
        pct = 0.0
        if total is not None and total > 0:
            pct = float(np.clip((float(current) / float(total)) * 100.0, 0.0, 100.0))
        detail = "" if message is None else str(message)
        if frame_index is not None:
            detail = f"{detail} (frame={int(frame_index)})".strip()
        self.set_stage(stage, "running", detail)
        if stage == "preprocess" and self._preprocess_progress_slider is not None:
            self._preprocess_progress_slider.value = pct
        if stage == "mapping" and self._mapping_progress_slider is not None:
            self._mapping_progress_slider.value = pct

    def mark_outputs_ready(self, output_dir: str, output_files: list[str]) -> None:
        rendered = "\n".join(f"- `{f}`" for f in output_files)
        self.set_stage("outputs", "completed", "Outputs ready")
        self._set_markdown_content(
            self._outputs_markdown_handle,
            f"### Outputs\nDirectory: `{output_dir}`\n{rendered}",
        )

    def fail_run(self, stage: str, error_message: str) -> None:
        self.set_stage(stage, "failed", error_message)

    def _save_stacked_image(self) -> None:
        if self._latest_stacked_image is None:
            return
        output_dir = Path.cwd() / "viser_exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sp = self._safe_slider_value()
        fid = -1
        if sp is not None and self._frame_order and 0 <= sp < len(self._frame_order):
            fid = int(self._frame_order[sp])
        output_path = output_dir / f"frame_{fid:06d}_rgb_seg_depth_stacked_{stamp}.png"
        bgr = cv2.cvtColor(self._latest_stacked_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)

    def _legend_checkbox_label(self, class_id: int) -> str:
        return self._legend_display_name(int(class_id))

    def _legend_swatch_image(self, class_id: int) -> np.ndarray:
        r, g, b = self._class_colors[int(class_id)]
        swatch = np.zeros((28, 28, 3), dtype=np.uint8)
        swatch[:, :] = np.asarray([int(r), int(g), int(b)], dtype=np.uint8)
        return swatch

    def _legend_display_name(self, class_id: int) -> str:
        class_name = self._class_names.get(int(class_id), f"Class {int(class_id)}")
        class_name_s = str(class_name)
        if "\n" in class_name_s or "\r" in class_name_s:
            raise ValueError(
                f"Legend class name contains line break for class_id={int(class_id)}: {class_name_s!r}"
            )
        return class_name_s

    def close(self) -> None:
        self._stop_render.set()
        self._dirty.set()
        if self._render_thread is not None and self._render_thread.is_alive():
            self._render_thread.join(timeout=1.0)
        if self._server is None:
            return
        with suppress(Exception):
            self._server.stop()
            time.sleep(0.2)
        self.enabled = False
        self._server = None
        self._scene_ready = False

    def wait_forever(self) -> None:
        if not self.enabled:
            return
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
