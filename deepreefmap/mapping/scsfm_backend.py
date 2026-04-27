from __future__ import annotations

import cv2
import numpy as np

from deepreefmap.mapping.base import FrameEstimate, MappingBackend


class SCSfMBackend(MappingBackend):
    """
    Lightweight, dependency-free SfM-style backend.

    This is an inference-time geometric backend that keeps the architecture clean
    and replaceable. It estimates relative pose from tracked features and an
    essential matrix, and provides a proxy depth map from image gradients.
    """

    def __init__(self) -> None:
        self.name = "scsfm"
        self.default_window_size = 3
        self._k = np.eye(3, dtype=np.float32)
        self._pose_w_c = np.eye(4, dtype=np.float32)
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None

    def initialize(self, image_size: tuple[int, int], intrinsics: np.ndarray) -> None:
        del image_size
        self._k = intrinsics.astype(np.float32)
        self._pose_w_c = np.eye(4, dtype=np.float32)
        self._prev_gray = None
        self._prev_pts = None

    def _estimate_depth_proxy(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        grad = cv2.GaussianBlur(grad, (5, 5), 0)
        # Strong gradients are usually closer structure in reef transects.
        depth = 1.0 / (0.2 + grad)
        return np.clip(depth, 0.2, 8.0).astype(np.float32)

    def _estimate_relative_pose(self, curr_gray: np.ndarray) -> np.ndarray:
        rel = np.eye(4, dtype=np.float32)
        if self._prev_gray is None:
            return rel

        if self._prev_pts is None or len(self._prev_pts) < 40:
            self._prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray,
                maxCorners=1200,
                qualityLevel=0.01,
                minDistance=6,
                blockSize=7,
            )
        if self._prev_pts is None:
            return rel

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, curr_gray, self._prev_pts, None)
        if curr_pts is None or status is None:
            return rel

        mask = status.reshape(-1) == 1
        p0 = self._prev_pts[mask].reshape(-1, 2)
        p1 = curr_pts[mask].reshape(-1, 2)
        if len(p0) < 30:
            self._prev_pts = cv2.goodFeaturesToTrack(curr_gray, maxCorners=1200, qualityLevel=0.01, minDistance=6, blockSize=7)
            return rel

        E, inlier_mask = cv2.findEssentialMat(p1, p0, self._k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            self._prev_pts = cv2.goodFeaturesToTrack(curr_gray, maxCorners=1200, qualityLevel=0.01, minDistance=6, blockSize=7)
            return rel
        _, R, t, _ = cv2.recoverPose(E, p1, p0, self._k, mask=inlier_mask)
        rel[:3, :3] = R.astype(np.float32)
        rel[:3, 3] = (0.03 * t[:, 0]).astype(np.float32)

        self._prev_pts = p1.reshape(-1, 1, 2).astype(np.float32)
        return rel

    def process_frame(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        curr_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        rel = self._estimate_relative_pose(curr_gray)
        self._pose_w_c = self._pose_w_c @ rel
        self._prev_gray = curr_gray
        depth = self._estimate_depth_proxy(image_rgb)
        return FrameEstimate(
            frame_index=frame_index,
            depth=depth,
            pose_w_c=self._pose_w_c.copy(),
            intrinsics=self._k,
        )
