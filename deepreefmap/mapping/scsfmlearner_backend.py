from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from deepreefmap.mapping.base import FrameEstimate, MappingBackend
from deepreefmap.mapping.scsfmlearner.models import DispResNet, PoseResNet, pose_vec_to_matrix


class SCSfMLearnerBackend(MappingBackend):
    def __init__(
        self,
        *,
        checkpoint_path: str,
        pose_checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.name = "scsfmlearner"
        self.default_window_size = 3
        self._checkpoint_path = checkpoint_path
        self._pose_checkpoint_path = pose_checkpoint_path
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._k = np.eye(3, dtype=np.float32)
        self._pose_w_c = np.eye(4, dtype=np.float32)
        self._prev_tensor: torch.Tensor | None = None
        self._disp_net: DispResNet | None = None
        self._pose_net: PoseResNet | None = None

    def initialize(self, image_size: tuple[int, int], intrinsics: np.ndarray) -> None:
        del image_size
        self._k = intrinsics.astype(np.float32)
        self._pose_w_c = np.eye(4, dtype=np.float32)
        self._prev_tensor = None
        self._load_models()

    def _load_models(self) -> None:
        disp_path = Path(self._checkpoint_path)
        if not disp_path.exists():
            raise FileNotFoundError(f"SC-SfMLearner depth checkpoint not found: {disp_path}")

        pose_path = Path(self._pose_checkpoint_path) if self._pose_checkpoint_path else disp_path
        if not pose_path.exists():
            raise FileNotFoundError(f"SC-SfMLearner pose checkpoint not found: {pose_path}")

        self._disp_net = DispResNet(num_layers=18, pretrained=False).to(self._device)
        self._pose_net = PoseResNet(num_layers=18, pretrained=False).to(self._device)

        disp_ckpt = torch.load(disp_path, map_location=self._device)
        if "disp_state_dict" in disp_ckpt:
            self._disp_net.load_state_dict(disp_ckpt["disp_state_dict"], strict=True)
        elif "state_dict" in disp_ckpt:
            self._disp_net.load_state_dict(disp_ckpt["state_dict"], strict=True)
        else:
            self._disp_net.load_state_dict(disp_ckpt, strict=True)

        pose_ckpt = torch.load(pose_path, map_location=self._device)
        if "pose_state_dict" in pose_ckpt:
            self._pose_net.load_state_dict(pose_ckpt["pose_state_dict"], strict=True)
        elif "state_dict" in pose_ckpt:
            self._pose_net.load_state_dict(pose_ckpt["state_dict"], strict=True)
        else:
            self._pose_net.load_state_dict(pose_ckpt, strict=True)

        self._disp_net.eval()
        self._pose_net.eval()

    @staticmethod
    def _to_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
        x = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return x.to(device)

    def process_frame(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        assert self._disp_net is not None
        x = self._to_tensor(image_rgb, self._device)
        with torch.no_grad():
            disp = self._disp_net(x)
            if isinstance(disp, list):
                disp = disp[0]
            depth = (1.0 / disp.clamp(min=1e-6)).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

            rel = np.eye(4, dtype=np.float32)
            if self._prev_tensor is not None and self._pose_net is not None:
                pose_vec = self._pose_net(self._prev_tensor, x)
                rel = pose_vec_to_matrix(pose_vec).squeeze(0).cpu().numpy().astype(np.float32)
            self._prev_tensor = x

        self._pose_w_c = self._pose_w_c @ rel
        return FrameEstimate(
            frame_index=frame_index,
            depth=depth,
            pose_w_c=self._pose_w_c.copy(),
            intrinsics=self._k,
            scale_type="relative",
        )
