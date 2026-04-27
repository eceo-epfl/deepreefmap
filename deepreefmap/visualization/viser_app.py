from __future__ import annotations

import numpy as np


class ViserLiveApp:
    def __init__(self) -> None:
        self.enabled = False
        self._server = None
        self._pts_xyz: list[np.ndarray] = []
        self._pts_rgb: list[np.ndarray] = []
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
        # Keep updates lightweight; overwrite the current camera frustum and image panels.
        h, w = image_rgb.shape[:2]
        scene.add_camera_frustum(
            name=f"/camera/{frame_index:06d}",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=tuple(pose_w_c[:3, 3].tolist()),
            fov=float(np.deg2rad(60.0)),
            aspect=float(w) / float(max(h, 1)),
            scale=0.04,
        )

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
