from __future__ import annotations

from contextlib import suppress
import time

import numpy as np
import cv2


class ViserLiveApp:
    def __init__(self, class_colors: dict[int, tuple[int, int, int]] | None = None, port: int = 8080) -> None:
        self.enabled = False
        self._server = None
        self._class_colors = class_colors or {}
        self._frame_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._selected_frame_index: int | None = None
        self._pts_xyz: list[np.ndarray] = []
        self._pts_rgb: list[np.ndarray] = []
        self._pts_labels: list[np.ndarray] = []
        self._cloud_handle = None
        self._semantic_color_toggle = None
        self._rgb_handle = None
        self._seg_handle = None
        self._depth_handle = None
        try:
            import viser

            self._server = viser.ViserServer(port=port)
            with suppress(Exception):
                self._server.gui.configure_theme(dark_mode=True)
            self.enabled = True
            self._semantic_color_toggle = self._server.gui.add_checkbox("Semantic cloud colors", False)

            @self._semantic_color_toggle.on_update
            def _(_) -> None:
                self._refresh_cloud_handle()
        except Exception:
            self.enabled = False

    def update_frame(self, frame_index: int, image_rgb: np.ndarray, seg: np.ndarray, depth: np.ndarray, pose_w_c: np.ndarray) -> None:
        if not self.enabled:
            return
        self._frame_data[int(frame_index)] = (image_rgb, seg, depth)
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
        if hasattr(frustum_handle, "on_click"):
            @frustum_handle.on_click
            def _(_, selected_frame_index: int = int(frame_index)) -> None:
                self._show_frame(selected_frame_index)

        if self._selected_frame_index is None:
            self._show_frame(int(frame_index))

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
            return
        self._rgb_handle.image = image_rgb
        self._seg_handle.image = self._colorize_seg(seg)
        self._depth_handle.image = self._colorize_depth(depth)

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

    def add_points(self, xyz: np.ndarray, rgb: np.ndarray, labels: np.ndarray) -> None:
        if not self.enabled:
            return
        self._pts_xyz.append(xyz)
        self._pts_rgb.append(rgb)
        self._pts_labels.append(labels)
        self._refresh_cloud_handle()

    def _refresh_cloud_handle(self) -> None:
        if not self._pts_xyz:
            return
        all_xyz = np.concatenate(self._pts_xyz, axis=0)
        all_rgb = np.concatenate(self._pts_rgb, axis=0)
        all_labels = np.concatenate(self._pts_labels, axis=0).astype(np.int32)
        use_semantic_colors = bool(self._semantic_color_toggle is not None and self._semantic_color_toggle.value)
        colors = self._colorize_labels(all_labels) if use_semantic_colors else all_rgb

        if self._cloud_handle is None:
            self._cloud_handle = self._server.scene.add_point_cloud(
                name="/cloud/points",
                points=all_xyz,
                colors=colors,
                point_size=0.002,
            )
            return
        self._cloud_handle.points = all_xyz
        self._cloud_handle.colors = colors

    def _colorize_labels(self, labels: np.ndarray) -> np.ndarray:
        out = np.full((labels.shape[0], 3), 128, dtype=np.uint8)
        if not self._class_colors:
            return out
        for class_id, rgb in self._class_colors.items():
            out[labels == int(class_id)] = np.asarray(rgb, dtype=np.uint8)
        return out

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
                time.sleep(1.0)
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
