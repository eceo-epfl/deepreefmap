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
        self._cloud_handle = None
        self._semantic_color_toggle = None
        self._point_size_slider = None
        self._frame_slider = None
        self._playing_toggle = None
        self._fps_slider = None
        self._accumulate_toggle = None
        self._next_button = None
        self._prev_button = None
        self._rgb_handle = None
        self._seg_handle = None
        self._depth_handle = None
        self._side_by_side_handle = None
        self._legend_toggles: dict[int, object] = {}
        self._download_side_by_side_button = None
        self._latest_side_by_side_image: np.ndarray | None = None
        try:
            import viser

            self._server = viser.ViserServer(port=port)
            with suppress(Exception):
                self._server.gui.configure_theme(dark_mode=True)
            self.enabled = True
            self._semantic_color_toggle = self._server.gui.add_checkbox("Semantic cloud colors", False)
            self._point_size_slider = self._server.gui.add_slider("Point size", 0.0001, 0.05, 0.0001, 0.002)
            self._frame_slider = self._server.gui.add_slider("Frame", 0, 0, 1, 0, disabled=True)
            self._playing_toggle = self._server.gui.add_checkbox("Playing", False)
            self._fps_slider = self._server.gui.add_slider("FPS", 1.0, 60.0, 0.5, 8.0)
            self._accumulate_toggle = self._server.gui.add_checkbox("Accumulate", False)
            self._next_button = self._server.gui.add_button("Next")
            self._prev_button = self._server.gui.add_button("Prev")
            with self._server.gui.add_folder("Semantic legend"):
                for class_id in sorted(self._class_colors):
                    color = self._class_colors[int(class_id)]
                    class_name = self._class_names.get(int(class_id), f"Class {int(class_id)}")
                    toggle = self._server.gui.add_checkbox(
                        f"[{int(class_id):02d}] {class_name} ({int(color[0])},{int(color[1])},{int(color[2])})",
                        True,
                    )
                    self._legend_toggles[int(class_id)] = toggle
                    @toggle.on_update
                    def _(_, class_id_for_toggle: int = int(class_id)) -> None:
                        _ = class_id_for_toggle
                        self._render_current_state()
            self._download_side_by_side_button = self._server.gui.add_button("Download side-by-side view")

            @self._semantic_color_toggle.on_update
            def _(_) -> None:
                self._render_current_state()

            @self._point_size_slider.on_update
            def _(_) -> None:
                self._render_current_state()

            @self._frame_slider.on_update
            def _(_) -> None:
                self._render_current_state()

            @self._accumulate_toggle.on_update
            def _(_) -> None:
                self._render_current_state()

            @self._next_button.on_click
            def _(_) -> None:
                self._step_frame(+1)

            @self._prev_button.on_click
            def _(_) -> None:
                self._step_frame(-1)

            @self._download_side_by_side_button.on_click
            def _(_) -> None:
                self._save_side_by_side_image()
        except Exception:
            self.enabled = False

    def update_frame(self, frame_index: int, image_rgb: np.ndarray, seg: np.ndarray, depth: np.ndarray, pose_w_c: np.ndarray) -> None:
        if not self.enabled:
            return
        frame_index = int(frame_index)
        self._frame_data[frame_index] = (image_rgb, seg, depth)
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
            self._set_current_frame(frame_index)
        else:
            self._render_current_state()

    def _show_frame(self, frame_index: int) -> None:
        frame_data = self._frame_data.get(int(frame_index))
        if frame_data is None:
            return
        image_rgb, seg, depth = frame_data
        self._selected_frame_index = int(frame_index)
        if self._rgb_handle is None:
            self._rgb_handle = self._server.gui.add_image(image_rgb, label="RGB")
            self._seg_handle = self._server.gui.add_image(self._colorize_seg(seg), label="Segmentation")
            self._depth_handle = self._server.gui.add_image(self._colorize_depth(depth), label="Depth")
            side_by_side = self._compose_side_by_side(image_rgb, self._colorize_seg(seg), self._colorize_depth(depth))
            self._latest_side_by_side_image = side_by_side
            self._side_by_side_handle = self._server.gui.add_image(
                side_by_side,
                label="RGB | Segmentation | Depth (side-by-side)",
            )
            return
        self._rgb_handle.image = image_rgb
        seg_color = self._colorize_seg(seg)
        depth_color = self._colorize_depth(depth)
        self._seg_handle.image = seg_color
        self._depth_handle.image = depth_color
        side_by_side = self._compose_side_by_side(image_rgb, seg_color, depth_color)
        self._latest_side_by_side_image = side_by_side
        self._side_by_side_handle.image = side_by_side

    def _compose_side_by_side(self, image_rgb: np.ndarray, seg_color: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
        return np.concatenate([image_rgb, seg_color, depth_color], axis=1)

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

    def _step_frame(self, step: int) -> None:
        current = self._current_frame()
        if current is None:
            return
        current_pos = self._frame_order.index(current)
        next_pos = (current_pos + int(step)) % len(self._frame_order)
        self._set_current_frame(int(self._frame_order[next_pos]))

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
        if current_value not in self._frame_order:
            self._frame_slider.value = int(self._frame_order[0])

    def _render_current_state(self) -> None:
        current_frame = self._current_frame()
        if current_frame is None:
            return
        self._selected_frame_index = int(current_frame)
        self._show_frame(current_frame)
        accumulate = bool(self._accumulate_toggle is not None and self._accumulate_toggle.value)
        self._set_frustum_visibility(current_frame, accumulate)
        self._set_cloud_state(current_frame, accumulate)

    def _set_frustum_visibility(self, current_frame: int, accumulate: bool) -> None:
        for frame_index, handle in self._frustums_by_frame.items():
            handle.visible = frame_index <= current_frame if accumulate else frame_index == current_frame

    def _set_cloud_state(self, current_frame: int, accumulate: bool) -> None:
        xyz, rgb, labels = self._cloud_for_frame(current_frame, accumulate)
        xyz, rgb, labels = self._filter_cloud_by_enabled_classes(xyz, rgb, labels)
        if xyz.size == 0:
            if self._cloud_handle is not None:
                self._cloud_handle.visible = False
            return
        use_semantic_colors = bool(self._semantic_color_toggle is not None and self._semantic_color_toggle.value)
        colors = self._colorize_labels(labels) if use_semantic_colors else rgb
        point_size = 0.002 if self._point_size_slider is None else float(self._point_size_slider.value)

        if self._cloud_handle is None:
            self._cloud_handle = self._server.scene.add_point_cloud(
                name="/cloud/points",
                points=xyz,
                colors=colors,
                point_size=point_size,
            )
            return
        self._cloud_handle.visible = True
        self._cloud_handle.points = xyz
        self._cloud_handle.colors = colors
        self._cloud_handle.point_size = point_size

    def _filter_cloud_by_enabled_classes(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._legend_toggles:
            return xyz, rgb, labels
        enabled_ids = {class_id for class_id, toggle in self._legend_toggles.items() if bool(toggle.value)}
        if not enabled_ids:
            empty_xyz = np.zeros((0, 3), dtype=xyz.dtype)
            empty_rgb = np.zeros((0, 3), dtype=rgb.dtype)
            empty_labels = np.zeros((0,), dtype=labels.dtype)
            return empty_xyz, empty_rgb, empty_labels
        mask = np.isin(labels, np.asarray(sorted(enabled_ids), dtype=np.int32))
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

    def _save_side_by_side_image(self) -> None:
        if self._latest_side_by_side_image is None:
            return
        output_dir = Path.cwd() / "viser_exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        frame_id = -1 if self._selected_frame_index is None else int(self._selected_frame_index)
        output_path = output_dir / f"frame_{frame_id:06d}_rgb_seg_depth_{stamp}.png"
        bgr = cv2.cvtColor(self._latest_side_by_side_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)

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
                    self._step_frame(+1)
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
