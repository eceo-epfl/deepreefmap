from __future__ import annotations

import cv2
import numpy as np

from deepreefmap.camera.intrinsics import CameraProfile


class Rectifier:
    def __init__(self, profile: CameraProfile) -> None:
        self.profile = profile
        self._map1 = None
        self._map2 = None
        self._roi: tuple[int, int, int, int] | None = None

    def _init_maps(self, image_size: tuple[int, int]) -> None:
        w, h = image_size
        model = str(getattr(self.profile, "distorted_model", "RADIAL")).upper()
        k = np.array(
            [
                [self.profile.radial["fx"], 0.0, self.profile.radial["cx"]],
                [0.0, self.profile.radial["fy"], self.profile.radial["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
            self._roi = None
            self._map1 = None
            self._map2 = None
            return

        if model in {"RADIAL", "SIMPLE_RADIAL"}:
            dist = np.array([self.profile.radial.get("k1", 0.0), self.profile.radial.get("k2", 0.0), 0.0, 0.0], dtype=np.float32)
            # Compute ROI for valid undistorted pixels (alpha=0 crops black borders).
            _, roi = cv2.getOptimalNewCameraMatrix(
                k,
                dist,
                (w, h),
                alpha=0.0,
                newImgSize=(w, h),
                centerPrincipalPoint=True,
            )
            self._roi = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                k,
                dist,
                R=np.eye(3, dtype=np.float32),
                newCameraMatrix=self.profile.k.astype(np.float32),
                size=(w, h),
                m1type=cv2.CV_32FC1,
            )
            return

        if model == "OPENCV_FISHEYE":
            dist = np.array(
                [
                    self.profile.radial.get("k1", 0.0),
                    self.profile.radial.get("k2", 0.0),
                    self.profile.radial.get("k3", 0.0),
                    self.profile.radial.get("k4", 0.0),
                ],
                dtype=np.float32,
            )
            self._roi = (0, 0, w, h)
            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                k,
                dist.reshape(4, 1),
                R=np.eye(3, dtype=np.float32),
                P=self.profile.k.astype(np.float32),
                size=(w, h),
                m1type=cv2.CV_32FC1,
            )
            return

        raise ValueError(f"Unsupported distorted camera model for rectification: {model}")

    def rectify(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        model = str(getattr(self.profile, "distorted_model", "RADIAL")).upper()
        if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
            return image_rgb
        if self._map1 is None or self._map2 is None:
            self._init_maps((w, h))
        rect = cv2.remap(image_rgb, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)
        if self._roi is not None:
            x, y, rw, rh = self._roi
            if rw > 0 and rh > 0:
                rect = rect[y : y + rh, x : x + rw]
                # Keep downstream shape stable while removing black borders.
                rect = cv2.resize(rect, (w, h), interpolation=cv2.INTER_LINEAR)
        return rect
