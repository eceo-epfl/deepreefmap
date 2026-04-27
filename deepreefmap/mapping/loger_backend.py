from __future__ import annotations

from pathlib import Path
import inspect
import logging
import sys
import yaml

import cv2
import numpy as np

from deepreefmap.mapping.base import FrameEstimate, MappingBackend
from deepreefmap.pipeline.artifacts import MappingSequenceResult

logger = logging.getLogger(__name__)

# LoGeR's Pi3 model returns `camera_poses` as camera-to-world (T_w_c) transforms,
# canonicalized so that the first camera in the window sits at the world origin.
# Downstream (orchestrator -> integrate_tsdf, viser frustums) assumes T_w_c. If a
# future LoGeR variant flips this, the identity-pose assertion in
# `_assert_pose_convention` will fail loudly at runtime instead of silently
# producing mirrored geometry.
_POSE_IDENTITY_TOLERANCE = 1e-3


# LoGeR upstream is a research repo with no pyproject.toml/setup.py; we vendor
# it as a submodule under third_party/LoGeR and put its package directory on
# sys.path so `import loger.*` resolves. Done at module import time so any
# helper script (not just the backend) that imports this file gets the fix.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOGER_PATH = _REPO_ROOT / "third_party" / "LoGeR"
if _LOGER_PATH.is_dir() and str(_LOGER_PATH) not in sys.path:
    sys.path.insert(0, str(_LOGER_PATH))


class LoGeRBackend(MappingBackend):
    """LoGeR sequence adapter.

    LoGeR's temporal memory lives inside one forward pass over a real sequence.
    This backend therefore exposes `process_sequence` as the production path and
    deliberately refuses per-frame proxy estimates.
    """

    def __init__(
        self,
        window_size: int = 32,
        overlap_size: int = 3,
        model_path: str | None = None,
        config_path: str | None = None,
        target_resolution: tuple[int, int] = (448, 252),
        se3: bool = False,
        sim3: bool = False,
        turn_off_ttt: bool = False,
        turn_off_swa: bool = False,
    ) -> None:
        self.name = "loger"
        self.default_window_size = window_size
        self._overlap_size = overlap_size
        self._model_path = model_path or str(_LOGER_PATH / "ckpts" / "LoGeR" / "latest.pt")
        self._config_path = config_path or str(_LOGER_PATH / "ckpts" / "LoGeR" / "original_config.yaml")
        self._target_resolution = target_resolution
        self._k = np.eye(3, dtype=np.float32)
        self._image_size: tuple[int, int] | None = None
        self._model = None
        self._device = "cuda"
        self._torch = None
        self._se3 = se3
        self._sim3 = sim3
        self._turn_off_ttt = turn_off_ttt
        self._turn_off_swa = turn_off_swa
        self._load_loger()

    def _load_loger(self) -> None:
        import torch
        from loger.models.pi3 import Pi3

        if not torch.cuda.is_available():
            raise RuntimeError(
                "LoGeR requires CUDA, but no CUDA device is available. "
                "Run on a GPU host or pick a different mapping backend."
            )
        self._torch = torch
        self._device = "cuda"
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"LoGeR checkpoint not found: {self._model_path}. "
                "Download it or pass --loger-model-path."
            )

        model_kwargs: dict[str, object] = {}
        self._config: dict[str, object] = {}
        if self._config_path and Path(self._config_path).exists():
            self._config = yaml.safe_load(Path(self._config_path).read_text()) or {}
            mcfg = self._config.get("model", {})
            sig = inspect.signature(Pi3.__init__)
            valid = {
                name
                for name, param in sig.parameters.items()
                if name not in {"self", "args", "kwargs"}
                and param.kind
                in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            for key, value in mcfg.items():
                if key in valid:
                    model_kwargs[key] = value

        model = Pi3(**model_kwargs)
        ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise RuntimeError(f"Checkpoint '{self._model_path}' does not contain a readable state_dict.")
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        self._model = model.to(self._device).eval()

    def initialize(self, image_size: tuple[int, int], intrinsics: np.ndarray) -> None:
        self._image_size = image_size
        self._k = intrinsics.astype(np.float32)

    def process_frame(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        del frame_index, image_rgb
        raise RuntimeError("LoGeR must be run with process_sequence(); per-frame proxy estimates are disabled.")

    def process_sequence(
        self,
        frame_indices: list[int],
        images_rgb: list[np.ndarray],
    ) -> MappingSequenceResult:
        if not images_rgb:
            raise RuntimeError("LoGeR cannot process an empty sequence")
        try:
            torch = self._torch
            model = self._model
            assert torch is not None and model is not None
            target_w, target_h = self._target_resolution
            target_w = _nearest_multiple(target_w, 14)
            target_h = _nearest_multiple(target_h, 14)
            resized = [cv2.resize(frm, (target_w, target_h), interpolation=cv2.INTER_AREA) for frm in images_rgb]
            batch = np.stack(resized, axis=0).astype(np.float32) / 255.0
            batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).unsqueeze(0).to(self._device)
            forward_kwargs = {
                "window_size": self.default_window_size,
                "overlap_size": self._overlap_size,
                "sim3": self._sim3,
                "se3": self._se3 or bool((self._config.get("model", {}) or {}).get("se3", False)),
                "turn_off_ttt": self._turn_off_ttt,
                "turn_off_swa": self._turn_off_swa,
            }
            capability = torch.cuda.get_device_capability(self._device)[0]
            dtype = torch.bfloat16 if capability >= 8 else torch.float16
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                out = model(
                    batch_t,
                    **forward_kwargs,
                )

            if not isinstance(out, dict):
                raise RuntimeError("LoGeR inference did not return a prediction dictionary")
            local_points = _tensor_to_numpy(out.get("local_points"))
            world_points = _tensor_to_numpy(out.get("points"))
            poses = _tensor_to_numpy(out.get("camera_poses"))
            if local_points is None and world_points is None:
                raise RuntimeError("LoGeR output missing both 'local_points' and 'points'")
            if local_points is None:
                local_points = world_points
            assert local_points is not None
            depth = np.abs(local_points[..., 2]).astype(np.float32)
            if poses is None or poses.shape[-2:] != (4, 4):
                raise RuntimeError("LoGeR output missing usable 'camera_poses'")

            n_in = len(frame_indices)
            n_out = depth.shape[0]
            if n_out != n_in or poses.shape[0] != n_in:
                raise RuntimeError(
                    f"LoGeR returned {n_out} depth maps and {poses.shape[0]} poses for "
                    f"{n_in} input frames; refusing to silently drop frames."
                )

            poses, world_points = _reanchor_to_first_camera(poses, world_points)
            _assert_pose_convention(poses)

            confidence = _tensor_to_numpy(out.get("conf"))
            if confidence is not None:
                confidence_t = torch.as_tensor(confidence)
                confidence = torch.sigmoid(confidence_t).numpy().astype(np.float32)
                confidence = np.squeeze(confidence, axis=-1) if confidence.ndim == 4 and confidence.shape[-1] == 1 else confidence
            input_h, input_w = images_rgb[0].shape[:2]
            image_size = self._image_size or (input_w, input_h)
            intrinsics = _scale_intrinsics(self._k, image_size, (target_w, target_h))
            return MappingSequenceResult(
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
                depth_maps=depth.astype(np.float32),
                poses_w_c=poses.astype(np.float32),
                intrinsics=intrinsics,
                world_points=None if world_points is None else world_points.astype(np.float32),
                local_points=local_points.astype(np.float32),
                confidence=confidence,
                scale_type="relative",
            )
        except Exception as exc:
            raise RuntimeError("LoGeR sequence inference failed") from exc


def _reanchor_to_first_camera(
    poses: np.ndarray,
    world_points: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Canonicalize the world frame so camera 0 sits at the origin.

    Pi3 emits `camera_poses` and `points` in some internal world frame; the
    upstream eval adapter (`third_party/LoGeR/eval/pi3_adapter.py`) re-anchors
    them by left-multiplying every pose by `inv(poses[0])` and applying the
    same transform to `points`. Doing this here gives us a reproducible world
    frame (camera 0 = identity) across runs and matches the convention our
    pose-convention assertion expects.
    """
    if poses.shape[0] == 0:
        return poses, world_points

    poses_f64 = poses.astype(np.float64)
    reference_inv = np.linalg.inv(poses_f64[0])
    rebased = reference_inv[None, :, :] @ poses_f64
    rebased_poses = rebased.astype(poses.dtype)

    if world_points is None:
        return rebased_poses, None

    wp_f64 = world_points.astype(np.float64)
    flat = wp_f64.reshape(-1, 3)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = homog @ reference_inv.T
    rebased_world = transformed[:, :3].reshape(world_points.shape).astype(world_points.dtype)
    return rebased_poses, rebased_world


def _assert_pose_convention(poses: np.ndarray) -> None:
    """Post-condition check on the re-anchored pose stack.

    After `_reanchor_to_first_camera`, `poses[0]` is identity by construction,
    so this assertion catches numerical instability in the re-anchoring step
    (non-finite values, near-singular `poses[0]`). It additionally checks that
    `poses[1]`'s rotation block has `det ≈ +1` and `R @ R.T ≈ I` — those
    properties are true for any proper T_w_c rotation and would be broken by a
    reflected frame or a corrupted output tensor.
    """
    if poses.shape[0] == 0:
        return
    zero = poses[0]
    if not np.allclose(zero, np.eye(4), atol=_POSE_IDENTITY_TOLERANCE):
        raise RuntimeError(
            "LoGeR pose[0] is not identity; T_w_c convention assumption broken. "
            f"Got:\n{zero}"
        )
    if poses.shape[0] > 1:
        rotation = poses[1, :3, :3]
        det = float(np.linalg.det(rotation))
        if not np.isfinite(det) or abs(det - 1.0) > 1e-2:
            raise RuntimeError(
                f"LoGeR pose[1] rotation has det={det}; expected +1 for a proper "
                "camera-to-world rotation."
            )
        ortho_err = float(np.linalg.norm(rotation @ rotation.T - np.eye(3)))
        if ortho_err > 1e-2:
            raise RuntimeError(
                f"LoGeR pose[1] rotation is not orthonormal (||R R^T - I|| = {ortho_err})."
            )


def _nearest_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _tensor_to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.squeeze(0).detach().cpu().float().numpy()
    return np.asarray(value)


def _scale_intrinsics(k: np.ndarray, original_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    scaled = k.astype(np.float32).copy()
    scaled[0, :] *= float(target_w) / max(float(orig_w), 1.0)
    scaled[1, :] *= float(target_h) / max(float(orig_h), 1.0)
    scaled[2] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return scaled
