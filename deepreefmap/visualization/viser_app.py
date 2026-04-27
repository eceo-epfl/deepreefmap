from __future__ import annotations

from contextlib import suppress
import time
from pathlib import Path

import numpy as np
import cv2


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
        try:
            import viser

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
            self._frame_slider = self._server.gui.add_slider("Frame", 0, 0, 1, 0, disabled=True)
            self._playing_toggle = self._server.gui.add_checkbox("Playing", False)
            self._fps_slider = self._server.gui.add_slider("FPS", 1.0, 60.0, 0.5, 8.0)
            self._accumulate_toggle = self._server.gui.add_checkbox("Accumulate", True)
            with self._server.gui.add_folder("Semantic legend"):
                self._add_legend_swatch_block()
                for class_id in sorted(self._class_colors):
                    class_name = self._class_names.get(int(class_id), f"Class {int(class_id)}")
                    toggle = self._server.gui.add_checkbox(class_name, True)
                    self._legend_toggles[int(class_id)] = toggle
                    @toggle.on_update
                    def _(_, class_id_for_toggle: int = int(class_id)) -> None:
                        _ = class_id_for_toggle
                        self._refresh_enabled_class_ids()
                        self._refresh_cloud_only()
            self._refresh_enabled_class_ids()
            self._download_stacked_button = self._server.gui.add_button("Download stacked RGB/Seg/Depth")

            @self._semantic_color_toggle.on_update
            def _(_) -> None:
                self._refresh_cloud_only()

            @self._point_size_slider.on_update
            def _(_) -> None:
                if self._cloud_handle is not None:
                    self._cloud_handle.point_size = float(self._point_size_slider.value)

            @self._frame_slider.on_update
            def _(_) -> None:
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
                self._set_current_frame(selected_frame_index)

        if self._selected_frame_index is None:
            self._set_current_frame(int(self._frame_order[-1]))
        else:
            self._render_current_state()

    def _show_frame(self, frame_index: int) -> None:
        frame_data = self._frame_data.get(int(frame_index))
        if frame_data is None:
            return
        self._selected_frame_index = int(frame_index)
        stacked = self._stacked_image_for_frame(int(frame_index))
        self._latest_stacked_image = stacked
        if self._stacked_handle is None:
            self._stacked_handle = self._server.gui.add_image(
                stacked,
                label="RGB / Segmentation / Depth (stacked)",
            )
            return
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
        depth_display = self._resize_to_shape(depth_color, (target_h, target_w), interpolation=cv2.INTER_LINEAR)
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
        frame_indices_i32 = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
        unique_frames = np.unique(frame_indices_i32)
        for frame_index in unique_frames.tolist():
            mask = frame_indices_i32 == int(frame_index)
            xyz_part = xyz[mask]
            rgb_part = rgb[mask]
            labels_part = labels[mask].astype(np.int32)
            if frame_index in self._points_by_frame:
                old_xyz, old_rgb, old_labels = self._points_by_frame[frame_index]
                xyz_part = np.concatenate([old_xyz, xyz_part], axis=0)
                rgb_part = np.concatenate([old_rgb, rgb_part], axis=0)
                labels_part = np.concatenate([old_labels, labels_part], axis=0)
            self._points_by_frame[frame_index] = (xyz_part, rgb_part, labels_part)
            if frame_index not in self._frame_order:
                self._frame_order.append(frame_index)
        self._frame_order.sort()
        self._accumulated_cache.clear()
        self._update_frame_slider_limits()
        self._render_current_state()

    def _current_frame(self) -> int | None:
        if not self._frame_order:
            return None
        if self._frame_slider is not None:
            current_value = int(self._frame_slider.value)
            if current_value in self._frame_order:
                return current_value
        if self._selected_frame_index in self._frame_order:
            return int(self._selected_frame_index)
        return int(self._frame_order[0])

    def _set_current_frame(self, frame_index: int) -> None:
        if frame_index not in self._frame_order:
            return
        self._selected_frame_index = int(frame_index)
        if self._frame_slider is not None:
            self._frame_slider.value = int(frame_index)
        else:
            self._render_current_state()

    def _update_frame_slider_limits(self) -> None:
        if self._frame_slider is None or not self._frame_order:
            return
        self._frame_slider.disabled = False
        self._frame_slider.min = int(self._frame_order[0])
        self._frame_slider.max = int(self._frame_order[-1])
        current_value = int(self._frame_slider.value)
        if self._selected_frame_index is None:
            self._frame_slider.value = int(self._frame_order[-1])
            return
        if current_value not in self._frame_order:
            self._frame_slider.value = int(self._selected_frame_index)

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
        for frame_index, handle in self._frustums_by_frame.items():
            handle.visible = frame_index <= current_frame if accumulate else frame_index == current_frame

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
        frame_ids = [idx for idx in self._frame_order if idx <= current_frame and idx in self._points_by_frame]
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

    def _add_legend_swatch_block(self) -> None:
        if not self._class_colors:
            return
        rows: list[str] = []
        for class_id in sorted(self._class_colors):
            class_name = self._class_names.get(int(class_id), f"Class {int(class_id)}")
            r, g, b = self._class_colors[int(class_id)]
            rows.append(
                (
                    f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
                    f'<span style="display:inline-block;width:11px;height:11px;'
                    f'background:rgb({int(r)},{int(g)},{int(b)});border:1px solid #888;"></span>'
                    f"<span>{class_name}</span></div>"
                )
            )
        with suppress(Exception):
            self._server.gui.add_markdown("\n".join(rows))

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
