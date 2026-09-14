from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import logging
import threading
import time

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

from deepreefmap.camera.intrinsics import scale_intrinsics
from deepreefmap.mapping.base import FrameEstimate, MappingBackend, ProgressCallback

# VGGT-Omega ships no built-in sliding window: the whole sequence goes through a
# single forward pass, and the two pose helpers below canonicalize the world
# frame exactly as the LoGeR backend does (camera 0 at the origin, T_w_c poses).
from deepreefmap.mapping.loger_backend import (
    _assert_pose_convention,
    _reanchor_to_first_camera,
)
from deepreefmap.pipeline.artifacts import MappingSequenceResult, ReconstructionCancelled

logger = logging.getLogger(__name__)

# Gated Hugging Face repository holding the released checkpoints. The recommended
# in-the-wild model runs at 512 px; users must request access before download.
_DEFAULT_HF_REPO_ID = "facebook/VGGT-Omega"
_DEFAULT_HF_FILENAME = "vggt_omega_1b_512.pt"
_ACCESS_REQUEST_URL = "https://huggingface.co/facebook/VGGT-Omega"

# VGGT-Omega's ViT works in 16 px patches, so the model resolution and every
# resized side must be a multiple of this.
_PATCH_SIZE = 16

# The aggregator crops extreme aspect ratios into [0.5, 2.0]; we resize whole
# in-memory frames instead of cropping, so reject anything outside that band
# rather than silently distorting geometry.
_MIN_ASPECT_RATIO = 0.5
_MAX_ASPECT_RATIO = 2.0

# Rough peak-memory model from the upstream README (A100, 624x416 inputs):
# ~6 GB to hold the 1B model plus ~75 MB of activations per input frame.
_BASE_MEMORY_GB = 6.0
_PER_FRAME_MEMORY_GB = 0.075


class VGGTOmegaBackend(MappingBackend):
    """VGGT-Omega feed-forward camera + depth adapter.

    Like LoGeR, VGGT-Omega reasons over the whole sequence in one forward pass,
    so this backend exposes ``process_sequence`` as the production path and
    refuses per-frame proxy estimates. Unlike LoGeR it has no sliding window:
    peak GPU memory grows with the frame count (~6 GB + ~75 MB/frame), so the
    caller must bound the sequence length (``--fps``, ``--begin/--end``).
    """

    def __init__(
        self,
        model_path: str | None = None,
        image_resolution: int = 512,
        device: "torch.device | str | None" = None,
        backend_id: str = "vggt_omega",
    ) -> None:
        if image_resolution <= 0 or image_resolution % _PATCH_SIZE != 0:
            raise ValueError(
                f"image_resolution must be a positive multiple of {_PATCH_SIZE}, got {image_resolution}."
            )
        self.name = backend_id
        # VGGT-Omega has no temporal window; expose the frame count as a single
        # window so shared progress/telemetry code has a sane value.
        self.default_window_size = 1
        self._model_path = model_path
        self._image_resolution = int(image_resolution)
        self._requested_device = device
        self._k = np.eye(3, dtype=np.float32)
        self._image_size: tuple[int, int] | None = None
        self._model = None
        self._torch = None
        self._device = "cuda"
        self._predicted_intrinsics: np.ndarray | None = None
        self._load_model()

    def _load_model(self) -> None:
        import torch

        from deepreefmap.device import (
            disable_torch_compile_without_triton,
            resolve_device,
        )

        disable_torch_compile_without_triton()

        try:
            from vggt_omega.models import VGGTOmega
        except ImportError as exc:
            raise RuntimeError(
                "VGGT-Omega is not installed. Install it with `uv sync --extra vggt_omega` "
                "(pulls facebookresearch/vggt-omega from GitHub)."
            ) from exc

        if self._requested_device is not None:
            device = torch.device(self._requested_device)
        else:
            device = resolve_device()
        if device.type == "cpu":
            raise RuntimeError(
                "VGGT-Omega requires a GPU (CUDA, ROCm, or MPS), but only CPU is available."
            )
        self._torch = torch
        self._device = device

        checkpoint_path = self._resolve_checkpoint_path()
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"VGGT-Omega checkpoint not found: {checkpoint_path}. "
                "Download it or pass --vggt-omega-model-path."
            )

        # autocast=False: the model otherwise hardcodes a CUDA bf16 autocast in
        # its forward; we drive precision through deepreefmap.device instead so
        # ROCm/MPS follow the same policy as every other backend.
        model = VGGTOmega(autocast=False)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise RuntimeError(
                f"VGGT-Omega checkpoint '{checkpoint_path}' does not contain a readable state_dict."
            )
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        self._model = model.to(self._device).eval()

    def _resolve_checkpoint_path(self) -> str:
        if self._model_path:
            return self._model_path
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(repo_id=_DEFAULT_HF_REPO_ID, filename=_DEFAULT_HF_FILENAME)
        except Exception as exc:
            message = str(exc).lower()
            if "gated" in message or "401" in message or "403" in message or "access" in message:
                raise RuntimeError(
                    "VGGT-Omega checkpoints are gated on Hugging Face. Request access at "
                    f"{_ACCESS_REQUEST_URL}, run `hf auth login`, or pass --vggt-omega-model-path "
                    "to point at a local checkpoint."
                ) from exc
            raise RuntimeError(
                "Failed to download the default VGGT-Omega checkpoint from "
                f"{_DEFAULT_HF_REPO_ID}/{_DEFAULT_HF_FILENAME}. "
                "Provide --vggt-omega-model-path to use a local checkpoint."
            ) from exc

    def initialize(self, image_size: tuple[int, int], intrinsics: np.ndarray) -> None:
        self._image_size = image_size
        self._k = intrinsics.astype(np.float32)

    def process_frame(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        del frame_index, image_rgb
        raise RuntimeError(
            "VGGT-Omega must be run with process_sequence(); per-frame proxy estimates are disabled."
        )

    def process_sequence(
        self,
        frame_indices: list[int],
        images_rgb: list[np.ndarray],
        gravity_vectors: np.ndarray | None = None,
        cancel_event: threading.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> MappingSequenceResult:
        if gravity_vectors is not None:
            logger.info("Gravity telemetry is available but VGGT-Omega pose output is left unchanged.")
        if not images_rgb:
            raise RuntimeError("VGGT-Omega cannot process an empty sequence")

        # Validate the input shape before the try block: its errors are user-facing
        # configuration problems, not inference failures, so they must not be
        # wrapped by the "sequence inference failed" handler below.
        input_h, input_w = images_rgb[0].shape[:2]
        aspect_ratio = input_h / max(input_w, 1)
        if not (_MIN_ASPECT_RATIO <= aspect_ratio <= _MAX_ASPECT_RATIO):
            raise RuntimeError(
                f"VGGT-Omega only supports aspect ratios (H/W) in "
                f"[{_MIN_ASPECT_RATIO}, {_MAX_ASPECT_RATIO}], got {aspect_ratio:.3f} "
                f"for {input_w}x{input_h} frames."
            )
        target_h, target_w = _balanced_target_shape(aspect_ratio, self._image_resolution, _PATCH_SIZE)

        try:
            torch = self._torch
            model = self._model
            assert torch is not None and model is not None
            total_frames = len(images_rgb)
            logger.info(
                "VGGT-Omega sequence run starting: %d frames, resolution=%d -> %dx%d",
                total_frames,
                self._image_resolution,
                target_w,
                target_h,
            )
            estimated_gb = _BASE_MEMORY_GB + _PER_FRAME_MEMORY_GB * total_frames
            logger.info(
                "VGGT-Omega estimated peak GPU memory ~%.1f GB for %d frames "
                "(single forward pass, no sliding window). Reduce --fps or trim with "
                "--begin/--end if this exceeds available memory.",
                estimated_gb,
                total_frames,
            )

            if progress_callback is not None:
                progress_callback(0, 0, "Preparing frames for VGGT-Omega")  # 0/0 shows an indeterminate bar
            t_resize = time.monotonic()
            resized = [
                cv2.resize(frm, (target_w, target_h), interpolation=cv2.INTER_AREA)
                for frm in images_rgb
            ]
            logger.info(
                "VGGT-Omega input resize complete for %d/%d frames to %dx%d in %.1fs",
                len(resized),
                total_frames,
                target_w,
                target_h,
                time.monotonic() - t_resize,
            )
            batch = np.stack(resized, axis=0).astype(np.float32) / 255.0
            del resized
            batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).unsqueeze(0).to(self._device)
            del batch

            from deepreefmap.device import autocast_context, get_autocast_dtype

            # VGGT-family models guard their prediction heads with
            # autocast('cuda', enabled=False), which only takes effect on CUDA.
            # On MPS/CPU the heads would inherit the reduced-precision context and
            # emit fp16/bf16-quantized depth and pose_enc, so autocast is only
            # enabled on CUDA.
            use_autocast = self._device.type == "cuda"
            dtype = get_autocast_dtype(self._device)
            logger.info(
                "VGGT-Omega inference running on %s with %s (%d frames)...",
                self._device,
                f"autocast dtype={str(dtype).split('.')[-1]}" if use_autocast else "full fp32 (autocast disabled)",
                total_frames,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise ReconstructionCancelled("Cancelled before VGGT-Omega inference")
            if progress_callback is not None:
                progress_callback(0, 0, "VGGT-Omega inference")
            t_infer = time.monotonic()
            with torch.inference_mode(), autocast_context(self._device, enabled=use_autocast):
                try:
                    out = model(batch_t)
                except (NotImplementedError, RuntimeError) as exc:
                    if self._device.type != "mps" or "not currently implemented for the MPS device" not in str(exc):
                        raise
                    logger.warning("VGGT-Omega op unsupported on MPS, retrying on CPU: %s", exc)
                    model = model.cpu()
                    batch_t = batch_t.cpu()
                    self._device = torch.device("cpu")
                    # Autocast was already disabled for the MPS attempt, so this
                    # CPU retry runs in full fp32 as well.
                    out = model(batch_t)
            logger.info("VGGT-Omega inference finished in %.1fs", time.monotonic() - t_infer)

            if progress_callback is not None:
                progress_callback(0, 0, "Transferring depth + poses from GPU")
            if cancel_event is not None and cancel_event.is_set():
                raise ReconstructionCancelled("Cancelled after VGGT-Omega inference")

            if not isinstance(out, dict):
                raise RuntimeError("VGGT-Omega inference did not return a prediction dictionary")

            from vggt_omega.utils.pose_enc import encoding_to_camera

            pose_enc = out.get("pose_enc")
            depth_t = out.get("depth")
            if pose_enc is None or depth_t is None:
                raise RuntimeError("VGGT-Omega output missing 'pose_enc' or 'depth'")
            extrinsics_t, pred_intrinsics_t = encoding_to_camera(pose_enc, (target_h, target_w))

            extrinsics = _squeeze_batch(_to_numpy(extrinsics_t))  # [S, 3, 4]
            pred_intrinsics = _squeeze_batch(_to_numpy(pred_intrinsics_t))  # [S, 3, 3]
            depth = _squeeze_batch(_to_numpy(depth_t))  # [S, H, W, 1] or [S, H, W]
            if depth.ndim == 4 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            depth_conf = _to_numpy(out.get("depth_conf"))
            if depth_conf is not None:
                depth_conf = _squeeze_batch(depth_conf)
                if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
                    depth_conf = depth_conf[..., 0]

            return self._predictions_to_result(
                frame_indices=frame_indices,
                extrinsics=extrinsics,
                pred_intrinsics=pred_intrinsics,
                depth=depth,
                depth_conf=depth_conf,
                target_size=(target_w, target_h),
                gravity_vectors=gravity_vectors,
                progress_callback=progress_callback,
            )
        except ReconstructionCancelled:
            raise
        except Exception as exc:
            raise RuntimeError("VGGT-Omega sequence inference failed") from exc

    def _predictions_to_result(
        self,
        *,
        frame_indices: list[int],
        extrinsics: np.ndarray,
        pred_intrinsics: np.ndarray,
        depth: np.ndarray,
        depth_conf: np.ndarray | None,
        target_size: tuple[int, int],
        gravity_vectors: np.ndarray | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> MappingSequenceResult:
        if depth.ndim != 3:
            raise RuntimeError(f"VGGT-Omega depth must have shape (S, H, W), got {depth.shape}")
        n_in = len(frame_indices)
        n_out = depth.shape[0]
        if n_out != n_in or extrinsics.shape[0] != n_in or pred_intrinsics.shape[0] != n_in:
            raise RuntimeError(
                f"VGGT-Omega returned {n_out} depth maps, {extrinsics.shape[0]} poses and "
                f"{pred_intrinsics.shape[0]} intrinsics for {n_in} input frames; "
                "refusing to silently drop frames."
            )

        # camera-from-world extrinsics -> camera-to-world (T_w_c), then anchor
        # camera 0 at the world origin for a reproducible frame.
        poses_w_c = _invert_extrinsics(extrinsics)
        if progress_callback is not None:
            progress_callback(0, 0, "Aligning poses to world frame")
        poses_w_c, _ = _reanchor_to_first_camera(poses_w_c, None, progress_callback)
        _assert_pose_convention(poses_w_c)

        confidence = None if depth_conf is None else _confidence_from_depth_conf(depth_conf)

        target_w, target_h = target_size
        image_size = self._image_size or (target_w, target_h)
        intrinsics = scale_intrinsics(self._k, image_size, (target_w, target_h))

        # Kept for refine_intrinsics(): the predicted K is already in depth-scale
        # (principal point at the resized frame centre), so it needs no rescaling.
        self._predicted_intrinsics = pred_intrinsics.astype(np.float32)

        # No world_points: the model has no point head, and unprojecting here
        # with the per-frame predicted K would bake a K into the geometry that
        # can disagree with the one the orchestrator settles on (calibrated or
        # refined). Returning depth + poses only forces the cloud stage through
        # depth_to_points(depth, K, pose) with that single, shared K.
        logger.info("VGGT-Omega sequence run complete for %d frames", n_in)
        return MappingSequenceResult(
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            depth_maps=depth.astype(np.float32, copy=False),
            poses_w_c=poses_w_c.astype(np.float32),
            intrinsics=intrinsics,
            world_points=None,
            local_points=None,
            confidence=confidence,
            scale_type="relative",
            gravity_vectors=None if gravity_vectors is None else gravity_vectors.astype(np.float32),
        )

    def refine_intrinsics(self, mapping_result: MappingSequenceResult) -> np.ndarray | None:
        del mapping_result
        pred = self._predicted_intrinsics
        if pred is None or pred.ndim != 3 or pred.shape[0] == 0:
            logger.info("VGGT-Omega intrinsics refinement skipped: no predicted intrinsics available.")
            return None
        k = np.median(pred.astype(np.float64), axis=0).astype(np.float32)
        # Zero the shear/bottom-row drift a per-frame median can introduce.
        k[0, 1] = 0.0
        k[1, 0] = 0.0
        k[2] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return k


def _balanced_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    """Return the (H, W) VGGT-Omega expects for a frame of the given aspect ratio.

    Mirrors ``vggt_omega.utils.load_fn.load_and_preprocess_images`` in ``balanced``
    mode: keep the total token count near ``(image_resolution / patch_size) ** 2``
    while matching the aspect ratio, rounded to whole patches.
    """
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / max(aspect_ratio, 1e-6))
    h_patches = token_number / max(w_patches, 1e-6)
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _invert_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Invert camera-from-world extrinsics into camera-to-world (T_w_c) poses.

    Accepts ``(S, 3, 4)`` or ``(S, 4, 4)`` and returns ``(S, 4, 4)`` float64.
    """
    extr = np.asarray(extrinsics, dtype=np.float64)
    if extr.ndim != 3 or extr.shape[-2] not in (3, 4) or extr.shape[-1] != 4:
        raise RuntimeError(
            f"VGGT-Omega extrinsics must have shape (S, 3, 4) or (S, 4, 4), got {extr.shape}"
        )
    rotation = extr[:, :3, :3]
    translation = extr[:, :3, 3]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    poses = np.tile(np.eye(4, dtype=np.float64), (extr.shape[0], 1, 1))
    poses[:, :3, :3] = rotation_t
    poses[:, :3, 3] = -np.einsum("sij,sj->si", rotation_t, translation)
    return poses


def _confidence_from_depth_conf(depth_conf: np.ndarray) -> np.ndarray:
    """Squash VGGT-Omega's unbounded depth confidence (>= 1) into [0, 1].

    ``1 - 1/c`` maps ``[1, inf)`` onto ``[0, 1)``; the clip guards against any
    out-of-spec value below 1.
    """
    c = np.asarray(depth_conf, dtype=np.float32)
    conf = 1.0 - 1.0 / np.maximum(c, 1e-6)
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def _to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def _squeeze_batch(value: np.ndarray) -> np.ndarray:
    """Drop a leading batch dimension of size 1 left by the [B, S, ...] forward."""
    if value.ndim >= 2 and value.shape[0] == 1:
        return value[0]
    return value
