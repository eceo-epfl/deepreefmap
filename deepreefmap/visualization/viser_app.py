from __future__ import annotations

from contextlib import suppress
import time

import numpy as np
import cv2


class ViserLiveApp:
    def __init__(self) -> None:
        self.enabled = False
        self._server = None
        self._pts_xyz: list[np.ndarray] = []
        self._pts_rgb: list[np.ndarray] = []
        self._rgb_handle = None
        self._seg_handle = None
        self._depth_handle = None
        try:
            import viser

            self._server = viser.ViserServer()
            self.enabled = True
        except Exception:
            self.enabled = False

    def update_frame(self, frame_index: int, image_rgb: np.ndarray, seg: np.ndarray, depth: np.ndarray, pose_w_c: np.ndarray) -> None:
        if not self.enabled:
            return
        scene = self._server.scene

        if self._rgb_handle is None:
            self._rgb_handle = self._server.gui.add_image(image_rgb, label="RGB")
            self._seg_handle = self._server.gui.add_image(self._colorize_seg(seg), label="Segmentation")
            self._depth_handle = self._server.gui.add_image(self._colorize_depth(depth), label="Depth")
        else:
            self._rgb_handle.image = image_rgb
            self._seg_handle.image = self._colorize_seg(seg)
            self._depth_handle.image = self._colorize_depth(depth)

        # Keep updates lightweight; camera frustums + point cloud in 3D.
        h, w = image_rgb.shape[:2]
        scene.add_camera_frustum(
            name=f"/camera/{frame_index:06d}",
            wxyz=_rotation_to_wxyz(pose_w_c[:3, :3]),
            position=tuple(pose_w_c[:3, 3].tolist()),
            fov=float(np.deg2rad(60.0)),
            aspect=float(w) / float(max(h, 1)),
            scale=0.04,
        )

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
        # Deterministic pseudo-color table for label IDs.
        r = (s * 53 + 37) % 255
        g = (s * 97 + 17) % 255
        b = (s * 193 + 71) % 255
        return np.stack([r, g, b], axis=-1).astype(np.uint8)

    def add_points(self, xyz: np.ndarray, rgb: np.ndarray) -> None:
        if not self.enabled:
            return
        self._pts_xyz.append(xyz)
        self._pts_rgb.append(rgb)
        all_xyz = np.concatenate(self._pts_xyz, axis=0)
        all_rgb = np.concatenate(self._pts_rgb, axis=0)
        self._server.scene.add_point_cloud(
            name="/cloud/rgb",
            points=all_xyz,
            colors=all_rgb,
            point_size=0.002,
        )

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
