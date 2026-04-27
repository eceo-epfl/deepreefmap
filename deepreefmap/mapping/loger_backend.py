from __future__ import annotations

from collections import deque
from pathlib import Path
import inspect
import yaml

import cv2
import numpy as np

from deepreefmap.mapping.base import FrameEstimate, MappingBackend


class LoGeRBackend(MappingBackend):
    """
    Optional LoGeR adapter.

    If the LoGeR submodule/environment is unavailable, this backend gracefully
    falls back to a deterministic geometric proxy to keep the pipeline runnable.
    """

    def __init__(
        self,
        window_size: int = 32,
        overlap_size: int = 3,
        model_path: str | None = None,
        config_path: str | None = None,
        target_resolution: tuple[int, int] = (448, 252),
    ) -> None:
        self.name = "loger"
        self.default_window_size = window_size
        self._overlap_size = overlap_size
        self._model_path = model_path
        self._config_path = config_path
        self._target_resolution = target_resolution
        self._k = np.eye(3, dtype=np.float32)
        self._pose_w_c = np.eye(4, dtype=np.float32)
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._index_buffer: deque[int] = deque(maxlen=window_size)
        self._model = None
        self._device = "cpu"
        self._torch = None
        self._model_ready = False
        self._try_init_loger()

    def _try_init_loger(self) -> None:
        try:
            import torch
            from loger.models.pi3 import Pi3

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs: dict[str, object] = {}
            if self._config_path and Path(self._config_path).exists():
                cfg = yaml.safe_load(Path(self._config_path).read_text()) or {}
                mcfg = cfg.get("model", {})
                sig = inspect.signature(Pi3.__init__)
                valid = {n for n in sig.parameters if n not in {"self", "args", "kwargs"}}
                for key, value in mcfg.items():
                    if key in valid:
                        model_kwargs[key] = value
            self._model = Pi3(**model_kwargs).to(self._device).eval()
            if self._model_path and Path(self._model_path).exists():
                ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                elif isinstance(ckpt, dict):
                    state_dict = ckpt
                else:
                    state_dict = None
                if state_dict is not None:
                    self._model.load_state_dict(state_dict, strict=False)
            self._model_ready = True
        except Exception:
            self._model = None
            self._model_ready = False

    def initialize(self, image_size: tuple[int, int], intrinsics: np.ndarray) -> None:
        del image_size
        self._k = intrinsics.astype(np.float32)
        self._pose_w_c = np.eye(4, dtype=np.float32)
        self._buffer.clear()
        self._index_buffer.clear()

    def _fallback_estimate(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        depth = np.clip(3.0 - 2.0 * gray, 0.2, 8.0).astype(np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = -0.015 * frame_index
        return FrameEstimate(frame_index=frame_index, depth=depth, pose_w_c=pose, intrinsics=self._k)

    def process_frame(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        self._buffer.append(image_rgb)
        self._index_buffer.append(frame_index)
        if self._model is None or not self._model_ready or len(self._buffer) < self.default_window_size:
            return self._fallback_estimate(frame_index, image_rgb)

        try:
            torch = self._torch
            assert torch is not None
            resized = [
                cv2.resize(
                    frm,
                    self._target_resolution,
                    interpolation=cv2.INTER_AREA,
                )
                for frm in self._buffer
            ]
            batch = np.stack(resized, axis=0).astype(np.float32) / 255.0
            batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).unsqueeze(0).to(self._device)
            with torch.no_grad():
                out = self._model(
                    batch_t,
                    window_size=self.default_window_size,
                    overlap_size=self._overlap_size,
                )

            pose = None
            if isinstance(out, dict) and "camera_poses" in out:
                pose_t = out["camera_poses"].squeeze(0)[-1]
                pose = pose_t.detach().cpu().float().numpy()
            if pose is None or pose.shape != (4, 4):
                pose = self._pose_w_c.copy()
                pose[2, 3] -= 0.01

            depth = None
            if isinstance(out, dict):
                for key in ("points", "local_points"):
                    if key in out and out[key] is not None:
                        pts = out[key].squeeze(0)[-1].detach().cpu().float().numpy()
                        if pts.ndim == 3 and pts.shape[-1] >= 3:
                            depth = np.abs(pts[..., 2]).astype(np.float32)
                            break
            if depth is None:
                depth = self._fallback_estimate(frame_index, image_rgb).depth

            # Resize depth back to current frame.
            depth = cv2.resize(depth, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            self._pose_w_c = pose.astype(np.float32)
            return FrameEstimate(frame_index=frame_index, depth=depth, pose_w_c=self._pose_w_c.copy(), intrinsics=self._k)
        except Exception:
            return self._fallback_estimate(frame_index, image_rgb)
