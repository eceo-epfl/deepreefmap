from __future__ import annotations

from contextlib import suppress
import logging
import time
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)

_VISER_NAN_GUARD_PATCHED = False


def _install_viser_nan_slider_guard() -> None:
    """Best-effort guard for transient NaN slider updates from the browser.

    Viser can receive intermediate `NaN` values while a numeric field is being
    edited, and some versions cast those updates directly to `int`, which raises
    before user callbacks run. We keep behavior stable by dropping only that
    specific update error and preserving the last valid control value.
    """

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
    def __init__(
        self,
        class_colors: dict[int, tuple[int, int, int]] | None = None,
        class_names: dict[int, str] | None = None,
        port: int = 8080,
    ) -> None:
        self.enabled = False
        self._server = None
        self._class_colors = class_colors or {}
        self._class_names = class_names or {}
        self._frame_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._selected_frame_index: int | None = None
        self._frame_order: list[int] = []
        self._frustums_by_frame: dict[int, object] = {}
        self._points_by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._accumulated_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._stacked_image_cache: dict[int, np.ndarray] = {}
        self._seg_color_cache: dict[int, np.ndarray] = {}
        self._depth_color_cache: dict[int, np.ndarray] = {}
        self._cloud_handle = None
        self._semantic_color_toggle = None
        self._point_size_slider = None
        self._frame_slider = None
        self._playing_toggle = None
        self._fps_slider = None
        self._accumulate_toggle = None
        self._stacked_handle = None
        self._legend_toggles: dict[int, object] = {}
        self._enabled_class_ids = np.asarray([], dtype=np.int32)
        self._download_stacked_button = None
        self._latest_stacked_image: np.ndarray | None = None
        self._last_cloud_frame: int | None = None
        self._last_cloud_accumulate: bool | None = None
        self._last_cloud_filter_signature: tuple[int, ...] | None = None
        self._suppress_frame_slider_callback = False
        self._user_selected_frame = False
        self._last_slider_pos: int | None = None
        self._status_markdown_handle = None
        self._outputs_markdown_handle = None
        self._preprocess_progress_slider = None
        self._mapping_progress_slider = None
        self._stage_states: dict[str, tuple[str, str]] = {}
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
            # Keep this slider float-backed so transient NaN UI states do not trigger
            # integer casting failures inside viser's websocket handler.
            self._frame_slider = self._server.gui.add_slider("Frame", 0.0, 0.0, 1.0, 0.0, disabled=True)
            self._playing_toggle = self._server.gui.add_checkbox("Playing", False)
            self._fps_slider = self._server.gui.add_slider("FPS", 1.0, 60.0, 0.5, 8.0)
            self._accumulate_toggle = self._server.gui.add_checkbox("Accumulate", True)
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
                        label=self._legend_swatch_label(class_id_i),
                        order=order_base,
                    )
                    toggle = self._server.gui.add_checkbox(
                        self._legend_checkbox_label(class_id_i),
                        True,
                        order=order_base + 1.0,
                    )
                    self._legend_toggles[int(class_id)] = toggle
                    @toggle.on_update
                    def _(_, class_id_for_toggle: int = int(class_id)) -> None:
                        _ = class_id_for_toggle
                        self._refresh_enabled_class_ids()
                        self._refresh_cloud_only()
            self._refresh_enabled_class_ids()
            self._download_stacked_button = self._server.gui.add_button("Download stacked RGB/Seg/Depth")
            self._download_stacked_button.disabled = True
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
            def _(_) -> None:
                self._refresh_cloud_only()

            @self._point_size_slider.on_update
            def _(_) -> None:
                if self._cloud_handle is not None:
                    self._cloud_handle.point_size = float(self._point_size_slider.value)

            @self._frame_slider.on_update
            def _(_) -> None:
                if self._suppress_frame_slider_callback:
                    return
                slider_pos = self._validate_frame_slider_value(strict=True)
                if slider_pos is None:
                    return
                next_frame = int(self._frame_order[slider_pos])
                if self._selected_frame_index == next_frame:
                    return
                self._user_selected_frame = True
                self._render_current_state()

            @self._accumulate_toggle.on_update
            def _(_) -> None:
                self._render_current_state()

            @self._download_stacked_button.on_click
            def _(_) -> None:
                self._save_stacked_image()
        except Exception:
            self.enabled = False

    def update_frame(self, frame_index: int, image_rgb: np.ndarray, seg: np.ndarray, depth: np.ndarray, pose_w_c: np.ndarray) -> None:
        if not self.enabled:
            return
        frame_index = int(frame_index)
        self._frame_data[frame_index] = (image_rgb, seg, depth)
        self._seg_color_cache.pop(frame_index, None)
        self._depth_color_cache.pop(frame_index, None)
        self._stacked_image_cache.pop(frame_index, None)
        if frame_index not in self._frame_order:
            self._frame_order.append(frame_index)
            self._frame_order.sort()
            self._update_frame_slider_limits()
        scene = self._server.scene

        h, w = image_rgb.shape[:2]
        frustum_handle = scene.add_camera_frustum(
            name=f"/camera/{frame_index:06d}",
            wxyz=_rotation_to_wxyz(pose_w_c[:3, :3]),
            position=tuple(pose_w_c[:3, 3].tolist()),
            fov=float(np.deg2rad(60.0)),
            aspect=float(w) / float(max(h, 1)),
            scale=0.04,
            color=(0.6, 0.6, 0.6),
        )
        self._frustums_by_frame[frame_index] = frustum_handle
        if hasattr(frustum_handle, "on_click"):
            @frustum_handle.on_click
            def _(_, selected_frame_index: int = frame_index) -> None:
                self._set_current_frame(selected_frame_index, user_initiated=True)

        if self._selected_frame_index is None or not self._user_selected_frame:
            self._set_current_frame(int(self._frame_order[-1]), user_initiated=False)
            return
        self._render_current_state()

    def _show_frame(self, frame_index: int) -> None:
        frame_data = self._frame_data.get(int(frame_index))
        if frame_data is None:
            # Explicitly clear stale image context when selected frame has no RGB data.
            self._latest_stacked_image = None
            self._sync_download_button_state()
            return
        self._selected_frame_index = int(frame_index)
        stacked = self._stacked_image_for_frame(int(frame_index))
        self._latest_stacked_image = stacked
        self._sync_download_button_state()
        if self._stacked_handle is not None:
            self._stacked_handle.image = stacked

    def _stacked_image_for_frame(self, frame_index: int) -> np.ndarray:
        cached = self._stacked_image_cache.get(int(frame_index))
        if cached is not None:
            return cached
        image_rgb, seg, depth = self._frame_data[int(frame_index)]
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
        # Mapping depth may be lower resolution than RGB/segmentation; unify panel sizes
        # for stacked display.
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
        d = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(d)
        if valid.sum() == 0:
            return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
        lo, hi = np.percentile(d[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        color_bgr = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    def _colorize_seg(self, seg: np.ndarray) -> np.ndarray:
        s = np.asarray(seg, dtype=np.int32)
        out = np.full((s.shape[0], s.shape[1], 3), 128, dtype=np.uint8)
        if not self._class_colors:
            return out
        for class_id, rgb in self._class_colors.items():
            out[s == int(class_id)] = np.asarray(rgb, dtype=np.uint8)
        return out

    def add_points(self, xyz: np.ndarray, rgb: np.ndarray, labels: np.ndarray, frame_indices: np.ndarray) -> None:
        if not self.enabled:
            return
        if xyz.size == 0:
            return
        xyz_f32 = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        rgb_u8 = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        labels_i32 = np.asarray(labels, dtype=np.int32).reshape(-1)
        frame_indices_raw = np.asarray(frame_indices).reshape(-1)
        if not (len(xyz_f32) == len(rgb_u8) == len(labels_i32) == len(frame_indices_raw)):
            return
        finite_points = np.isfinite(xyz_f32).all(axis=1)
        finite_frames = np.isfinite(frame_indices_raw.astype(np.float64, copy=False))
        valid = finite_points & finite_frames
        if not np.any(valid):
            return
        xyz_f32 = xyz_f32[valid]
        rgb_u8 = rgb_u8[valid]
        labels_i32 = labels_i32[valid]
        frame_indices_i32 = frame_indices_raw[valid].astype(np.int32, copy=False)
        unique_frames = np.unique(frame_indices_i32)
        for frame_index in unique_frames.tolist():
            mask = frame_indices_i32 == int(frame_index)
            xyz_part = xyz_f32[mask]
            rgb_part = rgb_u8[mask]
            labels_part = labels_i32[mask]
            if frame_index in self._points_by_frame:
                old_xyz, old_rgb, old_labels = self._points_by_frame[frame_index]
                xyz_part = np.concatenate([old_xyz, xyz_part], axis=0)
                rgb_part = np.concatenate([old_rgb, rgb_part], axis=0)
                labels_part = np.concatenate([old_labels, labels_part], axis=0)
            self._points_by_frame[frame_index] = (xyz_part, rgb_part, labels_part)
        self._accumulated_cache.clear()
        self._update_frame_slider_limits()
        if self._frame_order and (self._selected_frame_index is None or not self._user_selected_frame):
            self._set_current_frame(int(self._frame_order[-1]), user_initiated=False)
            return
        self._render_current_state()

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
            # Write the normalized integer back for consistent UI state.
            self._suppress_frame_slider_callback = True
            try:
                self._frame_slider.value = float(int(slider_pos))
            finally:
                self._suppress_frame_slider_callback = False
        self._last_slider_pos = int(slider_pos)
        return slider_pos

    def _current_frame(self) -> int | None:
        if not self._frame_order:
            return None
        slider_pos = self._validate_frame_slider_value(strict=True)
        if slider_pos is not None:
            return int(self._frame_order[slider_pos])
        if self._selected_frame_index in self._frame_order:
            return int(self._selected_frame_index)
        return int(self._frame_order[0])

    def _sync_download_button_state(self) -> None:
        if self._download_stacked_button is None:
            return
        self._download_stacked_button.disabled = self._latest_stacked_image is None

    def _set_current_frame(self, frame_index: int, user_initiated: bool = False) -> None:
        if frame_index not in self._frame_order:
            return
        if user_initiated:
            self._user_selected_frame = True
        self._selected_frame_index = int(frame_index)
        if self._frame_slider is not None:
            pos = self._frame_order.index(int(frame_index))
            self._suppress_frame_slider_callback = True
            try:
                self._frame_slider.value = float(int(pos))
            finally:
                self._suppress_frame_slider_callback = False
        else:
            self._render_current_state()
        self._render_current_state()

    def _update_frame_slider_limits(self) -> None:
        if self._frame_slider is None or not self._frame_order:
            return
        self._frame_slider.disabled = False
        self._frame_slider.min = 0.0
        self._frame_slider.max = float(int(max(len(self._frame_order) - 1, 0)))
        self._frame_slider.step = 1.0
        slider_value = self._validate_frame_slider_value(strict=True)
        if self._selected_frame_index is None:
            self._suppress_frame_slider_callback = True
            try:
                self._frame_slider.value = float(int(max(len(self._frame_order) - 1, 0)))
            finally:
                self._suppress_frame_slider_callback = False
            return
        if int(self._selected_frame_index) in self._frame_order:
            self._suppress_frame_slider_callback = True
            try:
                self._frame_slider.value = float(int(self._frame_order.index(int(self._selected_frame_index))))
            finally:
                self._suppress_frame_slider_callback = False
        elif slider_value is None:
            self._suppress_frame_slider_callback = True
            try:
                self._frame_slider.value = float(int(max(len(self._frame_order) - 1, 0)))
            finally:
                self._suppress_frame_slider_callback = False

    def _render_current_state(self) -> None:
        current_frame = self._current_frame()
        if current_frame is None:
            return
        self._selected_frame_index = int(current_frame)
        self._show_frame(current_frame)
        accumulate = bool(self._accumulate_toggle is not None and self._accumulate_toggle.value)
        self._set_frustum_visibility(current_frame, accumulate)
        self._set_cloud_state(current_frame, accumulate, force_points=True)

    def _set_frustum_visibility(self, current_frame: int, accumulate: bool) -> None:
        if current_frame not in self._frame_order:
            return
        current_pos = self._frame_order.index(int(current_frame))
        for frame_index, handle in self._frustums_by_frame.items():
            frame_pos = self._frame_order.index(frame_index) if frame_index in self._frame_order else -1
            handle.visible = frame_pos <= current_pos if accumulate else frame_index == current_frame

    def _set_cloud_state(self, current_frame: int, accumulate: bool, force_points: bool = False) -> None:
        xyz, rgb, labels = self._cloud_for_frame(current_frame, accumulate)
        xyz, rgb, labels = self._filter_cloud_by_enabled_classes(xyz, rgb, labels)
        if xyz.size == 0:
            if self._cloud_handle is not None:
                self._cloud_handle.visible = False
            return
        use_semantic_colors = bool(self._semantic_color_toggle is not None and self._semantic_color_toggle.value)
        colors = self._colorize_labels(labels) if use_semantic_colors else rgb
        point_size = 0.002 if self._point_size_slider is None else float(self._point_size_slider.value)
        filter_signature = tuple(self._enabled_class_ids.tolist())
        should_update_points = force_points or (
            self._last_cloud_frame != int(current_frame)
            or self._last_cloud_accumulate != bool(accumulate)
            or self._last_cloud_filter_signature != filter_signature
        )

        if self._cloud_handle is None:
            self._cloud_handle = self._server.scene.add_point_cloud(
                name="/cloud/points",
                points=xyz,
                colors=colors,
                point_size=point_size,
            )
            self._last_cloud_frame = int(current_frame)
            self._last_cloud_accumulate = bool(accumulate)
            self._last_cloud_filter_signature = filter_signature
            return
        self._cloud_handle.visible = True
        if should_update_points:
            self._cloud_handle.points = xyz
        self._cloud_handle.colors = colors
        self._cloud_handle.point_size = point_size
        self._last_cloud_frame = int(current_frame)
        self._last_cloud_accumulate = bool(accumulate)
        self._last_cloud_filter_signature = filter_signature

    def _refresh_cloud_only(self) -> None:
        current_frame = self._current_frame()
        if current_frame is None:
            return
        accumulate = bool(self._accumulate_toggle is not None and self._accumulate_toggle.value)
        self._set_cloud_state(current_frame, accumulate, force_points=False)

    def _refresh_enabled_class_ids(self) -> None:
        if not self._legend_toggles:
            self._enabled_class_ids = np.asarray([], dtype=np.int32)
            return
        enabled = [class_id for class_id, toggle in self._legend_toggles.items() if bool(toggle.value)]
        self._enabled_class_ids = np.asarray(sorted(enabled), dtype=np.int32)

    def _filter_cloud_by_enabled_classes(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._legend_toggles:
            return xyz, rgb, labels
        if self._enabled_class_ids.size == 0:
            empty_xyz = np.zeros((0, 3), dtype=xyz.dtype)
            empty_rgb = np.zeros((0, 3), dtype=rgb.dtype)
            empty_labels = np.zeros((0,), dtype=labels.dtype)
            return empty_xyz, empty_rgb, empty_labels
        mask = np.isin(labels, self._enabled_class_ids)
        return xyz[mask], rgb[mask], labels[mask]

    def _cloud_for_frame(self, current_frame: int, accumulate: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._points_by_frame:
            empty_xyz = np.zeros((0, 3), dtype=np.float32)
            empty_rgb = np.zeros((0, 3), dtype=np.uint8)
            empty_labels = np.zeros((0,), dtype=np.int32)
            return empty_xyz, empty_rgb, empty_labels
        if not accumulate:
            return self._points_by_frame.get(
                int(current_frame),
                (
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.uint8),
                    np.zeros((0,), dtype=np.int32),
                ),
            )
        if current_frame in self._accumulated_cache:
            return self._accumulated_cache[current_frame]
        if current_frame not in self._frame_order:
            empty_xyz = np.zeros((0, 3), dtype=np.float32)
            empty_rgb = np.zeros((0, 3), dtype=np.uint8)
            empty_labels = np.zeros((0,), dtype=np.int32)
            return empty_xyz, empty_rgb, empty_labels
        current_pos = self._frame_order.index(int(current_frame))
        frame_ids = [idx for idx in self._frame_order[: current_pos + 1] if idx in self._points_by_frame]
        if not frame_ids:
            empty_xyz = np.zeros((0, 3), dtype=np.float32)
            empty_rgb = np.zeros((0, 3), dtype=np.uint8)
            empty_labels = np.zeros((0,), dtype=np.int32)
            self._accumulated_cache[current_frame] = (empty_xyz, empty_rgb, empty_labels)
            return self._accumulated_cache[current_frame]
        xyz = np.concatenate([self._points_by_frame[idx][0] for idx in frame_ids], axis=0)
        rgb = np.concatenate([self._points_by_frame[idx][1] for idx in frame_ids], axis=0)
        labels = np.concatenate([self._points_by_frame[idx][2] for idx in frame_ids], axis=0)
        self._accumulated_cache[current_frame] = (xyz, rgb, labels)
        return self._accumulated_cache[current_frame]

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

    def _colorize_labels(self, labels: np.ndarray) -> np.ndarray:
        out = np.full((labels.shape[0], 3), 128, dtype=np.uint8)
        if not self._class_colors:
            return out
        for class_id, rgb in self._class_colors.items():
            out[labels == int(class_id)] = np.asarray(rgb, dtype=np.uint8)
        return out

    def _save_stacked_image(self) -> None:
        if self._latest_stacked_image is None:
            return
        output_dir = Path.cwd() / "viser_exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        frame_id = -1 if self._selected_frame_index is None else int(self._selected_frame_index)
        output_path = output_dir / f"frame_{frame_id:06d}_rgb_seg_depth_stacked_{stamp}.png"
        bgr = cv2.cvtColor(self._latest_stacked_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)

    def _legend_checkbox_label(self, class_id: int) -> str:
        class_name = self._legend_display_name(int(class_id))
        return class_name

    def _legend_swatch_label(self, class_id: int) -> str:
        r, g, b = self._class_colors[int(class_id)]
        return f"RGB({int(r)}, {int(g)}, {int(b)})"

    def _legend_swatch_image(self, class_id: int) -> np.ndarray:
        r, g, b = self._class_colors[int(class_id)]
        swatch = np.zeros((10, 28, 3), dtype=np.uint8)
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
        if self._server is None:
            return
        with suppress(Exception):
            self._server.stop()
            # Let viser/websocket background threads finish printing before
            # Python tears down stdout during interpreter shutdown.
            time.sleep(0.2)
        self.enabled = False
        self._server = None

    def wait_forever(self) -> None:
        if not self.enabled:
            return
        try:
            while True:
                if bool(self._playing_toggle is not None and self._playing_toggle.value):
                    current = self._current_frame()
                    if current is not None:
                        current_pos = self._frame_order.index(current)
                        next_pos = (current_pos + 1) % len(self._frame_order)
                        self._set_current_frame(int(self._frame_order[next_pos]))
                    fps = 8.0 if self._fps_slider is None else max(1.0, float(self._fps_slider.value))
                    time.sleep(1.0 / fps)
                else:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass


def _rotation_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-8)
    return tuple(float(v) for v in quat)
