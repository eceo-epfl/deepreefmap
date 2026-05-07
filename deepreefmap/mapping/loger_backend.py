from __future__ import annotations

from pathlib import Path
import inspect
import logging
import sys
import time
import yaml

import cv2
import numpy as np

from deepreefmap.camera.intrinsics import scale_intrinsics
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
        target_resolution: tuple[int, int] = (504, 280),
        se3: bool = False,
        sim3: bool = False,
        turn_off_ttt: bool = False,
        turn_off_swa: bool = False,
        backend_id: str = "loger",
    ) -> None:
        self.name = backend_id
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
        gravity_vectors: np.ndarray | None = None,
    ) -> MappingSequenceResult:
        if gravity_vectors is not None:
            logger.info("Gravity telemetry is available but LoGeR pose output is left unchanged.")
        if not images_rgb:
            raise RuntimeError("LoGeR cannot process an empty sequence")
        try:
            torch = self._torch
            model = self._model
            assert torch is not None and model is not None
            total_frames = len(images_rgb)
            logger.info(
                "LoGeR sequence run starting: %d prepared frames, window_size=%d, overlap_size=%d",
                total_frames,
                self.default_window_size,
                self._overlap_size,
            )
            target_w, target_h = self._target_resolution
            target_w = _nearest_multiple(target_w, 14)
            target_h = _nearest_multiple(target_h, 14)
            t_resize = time.monotonic()
            resized = [cv2.resize(frm, (target_w, target_h), interpolation=cv2.INTER_AREA) for frm in images_rgb]
            logger.info(
                "LoGeR input resize complete for %d/%d frames to %dx%d in %.1fs",
                len(resized),
                total_frames,
                target_w,
                target_h,
                time.monotonic() - t_resize,
            )
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
            logger.info(
                "LoGeR inference running on %s with autocast dtype=%s (%d frames)...",
                self._device,
                str(dtype).split(".")[-1],
                total_frames,
            )
            t_infer = time.monotonic()
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True, dtype=dtype):
                out = model(
                    batch_t,
                    **forward_kwargs,
                )
            logger.info("LoGeR inference finished in %.1fs", time.monotonic() - t_infer)

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
            logger.info("LoGeR outputs validated: %d/%d frames have depth and poses", n_out, n_in)

            poses, world_points = _reanchor_to_first_camera(poses, world_points)
            _assert_pose_convention(poses)

            confidence = _tensor_to_numpy(out.get("conf"))
            if confidence is not None:
                confidence_t = torch.as_tensor(confidence)
                confidence = torch.sigmoid(confidence_t).numpy().astype(np.float32)
                confidence = np.squeeze(confidence, axis=-1) if confidence.ndim == 4 and confidence.shape[-1] == 1 else confidence
            input_h, input_w = images_rgb[0].shape[:2]
            image_size = self._image_size or (input_w, input_h)
            intrinsics = scale_intrinsics(self._k, image_size, (target_w, target_h))
            logger.info("LoGeR sequence run complete for %d frames", n_in)
            return MappingSequenceResult(
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
                depth_maps=depth.astype(np.float32),
                poses_w_c=poses.astype(np.float32),
                intrinsics=intrinsics,
                world_points=None if world_points is None else world_points.astype(np.float32),
                local_points=local_points.astype(np.float32),
                confidence=confidence,
                scale_type="relative",
                gravity_vectors=None if gravity_vectors is None else gravity_vectors.astype(np.float32),
            )
        except Exception as exc:
            raise RuntimeError("LoGeR sequence inference failed") from exc

    def refine_intrinsics(self, mapping_result: MappingSequenceResult) -> np.ndarray | None:
        local_points = mapping_result.local_points
        if local_points is None or local_points.ndim != 4 or local_points.shape[0] == 0:
            logger.info("LoGeR intrinsics refinement skipped: local_points unavailable.")
            return None
        try:
            refined = _estimate_intrinsics_from_local_points(
                local_points=local_points.astype(np.float32),
                seed_intrinsics=mapping_result.intrinsics.astype(np.float32),
            )
        except Exception as exc:
            logger.warning("LoGeR intrinsics refinement failed: %s", exc)
            return None
        if refined is None:
            logger.info("LoGeR intrinsics refinement produced no valid estimate; keeping camera profile K.")
            return None
        return refined.astype(np.float32)


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


def _estimate_intrinsics_from_local_points(
    *,
    local_points: np.ndarray,
    seed_intrinsics: np.ndarray,
) -> np.ndarray | None:
    """Estimate pinhole intrinsics from per-pixel local 3D points.

    LoGeR predicts local points as camera-frame XYZ. We estimate focal lengths
    by robustly solving x/z=(u-cx)/fx and y/z=(v-cy)/fy over all frames.
    Principal point is kept from the seed intrinsics.
    """
    if local_points.shape[-1] != 3:
        return None
    n, h, w, _ = local_points.shape
    if h <= 0 or w <= 0 or n <= 0:
        return None
    k = seed_intrinsics.astype(np.float32).copy()
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    z = local_points[..., 2]
    x = local_points[..., 0]
    y = local_points[..., 1]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (np.abs(z) > 1e-6)
    if not np.any(valid):
        return None

    rx = x / z
    ry = y / z
    u = np.broadcast_to(np.arange(w, dtype=np.float32)[None, None, :], (n, h, w))
    v = np.broadcast_to(np.arange(h, dtype=np.float32)[None, :, None], (n, h, w))

    fx_samples = (u - cx) / np.where(np.abs(rx) > 1e-6, rx, np.nan)
    fy_samples = (v - cy) / np.where(np.abs(ry) > 1e-6, ry, np.nan)
    fx = _robust_positive_median(fx_samples[valid])
    fy = _robust_positive_median(fy_samples[valid])
    if fx is None and fy is None:
        return None
    if fx is None:
        fx = fy
    if fy is None:
        fy = fx
    assert fx is not None and fy is not None

    max_dim = float(max(h, w))
    fx = float(np.clip(fx, 0.1 * max_dim, 20.0 * max_dim))
    fy = float(np.clip(fy, 0.1 * max_dim, 20.0 * max_dim))
    k[0, 0] = fx
    k[1, 1] = fy
    k[0, 1] = 0.0
    k[1, 0] = 0.0
    k[2] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return k


def _robust_positive_median(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    positive = np.abs(finite)
    positive = positive[positive > 1e-6]
    if positive.size == 0:
        return None
    return float(np.median(positive))
