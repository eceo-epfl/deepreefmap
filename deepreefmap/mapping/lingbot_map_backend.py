from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable
import importlib.util
import logging
import math
import threading
import time

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

from deepreefmap.camera.intrinsics import scale_intrinsics
from deepreefmap.mapping.base import FrameEstimate, MappingBackend, ProgressCallback

# LingBot-Map canonicalizes its world frame exactly as the LoGeR and VGGT-Omega
# backends do (camera 0 at the origin, T_w_c poses).
from deepreefmap.mapping.loger_backend import (
    _assert_pose_convention,
    _reanchor_to_first_camera,
)

# LingBot-Map produces the same feed-forward depth + confidence tensors as
# VGGT-Omega, so that math is shared verbatim (same cross-backend private-import
# precedent as vggt_omega_backend importing from loger_backend above). The pose
# convention differs, see _poses_w_c_from_decoded_pose_enc.
from deepreefmap.mapping.vggt_omega_backend import (
    _confidence_from_depth_conf,
    _squeeze_batch,
    _to_numpy,
)
from deepreefmap.pipeline.artifacts import MappingSequenceResult, ReconstructionCancelled

logger = logging.getLogger(__name__)

# Public Hugging Face repository holding the released checkpoints; not gated.
_DEFAULT_HF_REPO_ID = "robbyant/lingbot-map"
_DEFAULT_HF_FILENAME = "lingbot-map.pt"

# LingBot-Map's DINOv2 ViT works in 14 px patches, so the model resolution and
# every resized side must be a multiple of this.
_PATCH_SIZE = 14

# Upstream demo defaults (demo.py): the first frames are processed together with
# bidirectional attention to estimate scale, and the KV cache is trained for a
# ~320-view RoPE range, past which a keyframe stride is required.
_NUM_SCALE_FRAMES = 8
_KV_CACHE_SLIDING_WINDOW = 64
_ROPE_TRAIN_VIEWS = 320
_MAX_FRAME_NUM = 1024
_CAMERA_NUM_ITERATIONS = 4

_MODES = ("streaming", "windowed")
_ATTENTION_BACKENDS = ("auto", "sdpa", "flashinfer")


class LingBotMapBackend(MappingBackend):
    """LingBot-Map streaming camera + depth adapter.

    Like VGGT-Omega, LingBot-Map is a feed-forward model that reasons over the
    whole sequence, so this backend exposes ``process_sequence`` as the
    production path and refuses per-frame proxy estimates. Unlike VGGT-Omega it
    streams frame-by-frame against a bounded KV cache and offloads per-frame
    predictions to CPU, so peak GPU memory does not grow with the frame count.
    """

    def __init__(
        self,
        model_path: str | None = None,
        mode: str = "streaming",
        keyframe_interval: int | None = None,
        window_size: int = 64,
        overlap_keyframes: int | None = None,
        attention: str = "auto",
        image_size: int = 518,
        device: "torch.device | str | None" = None,
        backend_id: str = "lingbot_map",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
        if attention not in _ATTENTION_BACKENDS:
            raise ValueError(
                f"attention must be one of {_ATTENTION_BACKENDS}, got {attention!r}."
            )
        if image_size <= 0 or image_size % _PATCH_SIZE != 0:
            raise ValueError(
                f"image_size must be a positive multiple of {_PATCH_SIZE}, got {image_size}."
            )
        if keyframe_interval is not None and keyframe_interval < 1:
            raise ValueError(f"keyframe_interval must be >= 1, got {keyframe_interval}.")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        self.name = backend_id
        self._mode = mode
        self._keyframe_interval = keyframe_interval
        self._window_size = int(window_size)
        self._overlap_keyframes = overlap_keyframes
        self._attention = attention
        # Windowed mode processes fixed-size windows; streaming has no temporal
        # window, so expose 1 like VGGT-Omega for shared progress/telemetry code.
        self.default_window_size = self._window_size if mode == "windowed" else 1
        self._model_path = model_path
        self._image_size = int(image_size)
        self._patch_size = _PATCH_SIZE
        self._requested_device = device
        self._k = np.eye(3, dtype=np.float32)
        self._input_image_size: tuple[int, int] | None = None
        self._model = None
        self._torch = None
        self._device = "cuda"
        self._use_sdpa = True
        self._predicted_intrinsics: np.ndarray | None = None
        self._load_model()

    def _load_model(self) -> None:
        import torch

        from deepreefmap.device import (
            disable_torch_compile_without_triton,
            get_autocast_dtype,
            resolve_device,
        )

        disable_torch_compile_without_triton()

        try:
            if self._mode == "windowed":
                from lingbot_map.models.gct_stream_window import GCTStream
            else:
                from lingbot_map.models.gct_stream import GCTStream
        except ImportError as exc:
            raise RuntimeError(
                "LingBot-Map is not installed. Install it with `uv sync --extra lingbot_map` "
                "(pulls robbyant/lingbot-map from GitHub)."
            ) from exc

        if self._requested_device is not None:
            device = torch.device(self._requested_device)
        else:
            device = resolve_device()
        if device.type == "mps":
            # LingBot-Map's 3D RoPE precomputes its rotary frequency table in
            # float64 and moves it to the model device at forward time; MPS does
            # not support float64, so it can never run there. Fall back to CPU.
            logger.warning(
                "LingBot-Map cannot run on MPS (its 3D RoPE uses float64, which the MPS "
                "backend does not support); falling back to CPU. This will be slow for the "
                "4.6GB model."
            )
            device = torch.device("cpu")
        elif device.type == "cpu":
            raise RuntimeError(
                "LingBot-Map requires a CUDA or ROCm GPU (MPS is unsupported), "
                "but only CPU is available."
            )
        self._torch = torch
        self._device = device
        self._use_sdpa = self._resolve_use_sdpa(device)

        model = GCTStream(
            img_size=self._image_size,
            patch_size=self._patch_size,
            enable_3d_rope=True,
            max_frame_num=_MAX_FRAME_NUM,
            kv_cache_sliding_window=_KV_CACHE_SLIDING_WINDOW,
            kv_cache_scale_frames=_NUM_SCALE_FRAMES,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=self._use_sdpa,
            camera_num_iterations=_CAMERA_NUM_ITERATIONS,
        )

        checkpoint_path = self._resolve_checkpoint_path()
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"LingBot-Map checkpoint not found: {checkpoint_path}. "
                "Download it or pass --lingbot-map-model-path."
            )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        # strict=False mirrors upstream demo.py: the released checkpoint carries
        # a few keys (e.g. DINOv2 masking) unused at inference time.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info("LingBot-Map checkpoint missing %d keys", len(missing))
        if unexpected:
            logger.info("LingBot-Map checkpoint has %d unexpected keys", len(unexpected))

        model = model.to(device).eval()
        # Cast the DINOv2-style aggregator trunk to the inference dtype on CUDA to
        # drop the redundant fp32 master weight copy (upstream memory saving). The
        # prediction heads are guarded upstream with autocast('cuda', enabled=False),
        # so on CUDA they stay fp32. That guard is CUDA-only, which is why
        # process_sequence never enables autocast on other devices.
        if device.type == "cuda":
            dtype = get_autocast_dtype(device)
            if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
                model.aggregator = model.aggregator.to(dtype=dtype)
        self._model = model

    def _resolve_use_sdpa(self, device: "torch.device") -> bool:
        """Decide whether to use SDPA attention (True) or FlashInfer (False).

        FlashInfer is CUDA-only and JIT-compiles kernels on first use, so it is
        only selected when explicitly requested or auto-detected on a genuine
        CUDA (non-HIP) device with the package importable; every other case
        falls back to SDPA.
        """
        flashinfer_installed = importlib.util.find_spec("flashinfer") is not None
        is_cuda = device.type == "cuda" and getattr(self._torch.version, "hip", None) is None

        if self._attention == "flashinfer":
            if not flashinfer_installed:
                raise RuntimeError(
                    "LingBot-Map attention 'flashinfer' requested but the flashinfer "
                    "package is not installed. Install it with `pip install flashinfer-python` "
                    "or use --lingbot-map-attention sdpa."
                )
            if not is_cuda:
                raise RuntimeError(
                    "LingBot-Map attention 'flashinfer' requires a CUDA device; "
                    f"got {device.type}. Use --lingbot-map-attention sdpa."
                )
            logger.info("LingBot-Map using FlashInfer paged-KV attention.")
            return False
        if self._attention == "sdpa":
            logger.info("LingBot-Map using SDPA attention.")
            return True
        # auto
        if flashinfer_installed and is_cuda:
            logger.info("LingBot-Map auto-selected FlashInfer paged-KV attention.")
            return False
        logger.info("LingBot-Map using SDPA attention (FlashInfer unavailable).")
        return True

    def _resolve_checkpoint_path(self) -> str:
        if self._model_path:
            return self._model_path
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(repo_id=_DEFAULT_HF_REPO_ID, filename=_DEFAULT_HF_FILENAME)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download the default LingBot-Map checkpoint from "
                f"{_DEFAULT_HF_REPO_ID}/{_DEFAULT_HF_FILENAME}. "
                "Provide --lingbot-map-model-path to use a local checkpoint."
            ) from exc

    def initialize(self, image_size: tuple[int, int], intrinsics: np.ndarray) -> None:
        self._input_image_size = image_size
        self._k = intrinsics.astype(np.float32)

    def process_frame(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        del frame_index, image_rgb
        raise RuntimeError(
            "LingBot-Map must be run with process_sequence(); per-frame proxy estimates are disabled."
        )

    def _resolve_keyframe_interval(self, total_frames: int) -> int:
        if self._keyframe_interval is not None:
            return int(self._keyframe_interval)
        return _auto_keyframe_interval(total_frames, self._mode)

    def _install_forward_progress(
        self,
        model,
        total_frames: int,
        cancel_event: threading.Event | None,
        progress_callback: ProgressCallback | None,
    ) -> Callable[[], None]:
        """Wrap ``model.forward`` to report progress and honor cancellation.

        LingBot-Map streams one (or a block of scale) frames per forward call, so
        counting the frames each call consumes gives a real per-frame progress
        bar and a cancellation point mid-sequence (mirrors how the LoGeR backend
        wraps ``model.decode``).
        """
        original_forward = model.forward
        state = {"done": 0}

        def _counting_forward(images, *args, **kwargs):
            if cancel_event is not None and cancel_event.is_set():
                raise ReconstructionCancelled("Cancelled during LingBot-Map inference")
            out = original_forward(images, *args, **kwargs)
            n = images.shape[1] if images.dim() == 5 else images.shape[0]
            state["done"] += int(n)
            if progress_callback is not None:
                progress_callback(min(state["done"], total_frames), total_frames, "LingBot-Map inference")
            return out

        model.forward = _counting_forward
        return lambda: setattr(model, "forward", original_forward)

    def process_sequence(
        self,
        frame_indices: list[int],
        images_rgb: list[np.ndarray],
        gravity_vectors: np.ndarray | None = None,
        cancel_event: threading.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> MappingSequenceResult:
        if gravity_vectors is not None:
            logger.info("Gravity telemetry is available but LingBot-Map pose output is left unchanged.")
        if not images_rgb:
            raise RuntimeError("LingBot-Map cannot process an empty sequence")

        # Validate the input shape before the try block: its errors are user-facing
        # configuration problems, not inference failures, so they must not be
        # wrapped by the "sequence inference failed" handler below.
        input_h, input_w = images_rgb[0].shape[:2]
        target_h, target_w = _crop_target_shape(input_h, input_w, self._image_size, self._patch_size)

        try:
            torch = self._torch
            model = self._model
            assert torch is not None and model is not None
            total_frames = len(images_rgb)
            keyframe_interval = self._resolve_keyframe_interval(total_frames)
            logger.info(
                "LingBot-Map %s run starting: %d frames, resolution=%d -> %dx%d, keyframe_interval=%d",
                self._mode,
                total_frames,
                self._image_size,
                target_w,
                target_h,
                keyframe_interval,
            )

            if progress_callback is not None:
                progress_callback(0, 0, "Preparing frames for LingBot-Map")  # 0/0 shows an indeterminate bar
            t_resize = time.monotonic()
            resized = [
                cv2.resize(frm, (target_w, target_h), interpolation=cv2.INTER_AREA)
                for frm in images_rgb
            ]
            logger.info(
                "LingBot-Map input resize complete for %d/%d frames to %dx%d in %.1fs",
                len(resized),
                total_frames,
                target_w,
                target_h,
                time.monotonic() - t_resize,
            )
            batch = np.stack(resized, axis=0).astype(np.float32) / 255.0
            del resized
            # Keep the sequence on CPU: inference_streaming/windowed slice one
            # frame (or window) at a time onto the model device, so peak GPU
            # memory stays bounded regardless of sequence length.
            images_t = torch.from_numpy(batch).permute(0, 3, 1, 2)
            del batch

            from deepreefmap.device import autocast_context, get_autocast_dtype

            # Upstream guards the camera/depth heads with
            # autocast('cuda', enabled=False), which only takes effect on CUDA.
            # Anywhere else the heads would inherit the reduced-precision context
            # and emit bf16/fp16-quantized depth and pose_enc (visible as depth
            # terracing), so autocast is only enabled on CUDA.
            use_autocast = self._device.type == "cuda"
            logger.info(
                "LingBot-Map inference running on %s with %s (%d frames)...",
                self._device,
                f"autocast dtype={str(get_autocast_dtype(self._device)).split('.')[-1]}"
                if use_autocast
                else "full fp32 (autocast disabled)",
                total_frames,
            )

            if cancel_event is not None and cancel_event.is_set():
                raise ReconstructionCancelled("Cancelled before LingBot-Map inference")
            if progress_callback is not None:
                progress_callback(0, 0, "LingBot-Map inference")

            cpu_device = torch.device("cpu")
            t_infer = time.monotonic()

            def _run(active_model, device):
                restore = self._install_forward_progress(
                    active_model, total_frames, cancel_event, progress_callback
                )
                try:
                    with torch.inference_mode(), autocast_context(device, enabled=use_autocast):
                        if self._mode == "windowed":
                            return active_model.inference_windowed(
                                images_t,
                                window_size=self._window_size,
                                overlap_keyframes=self._overlap_keyframes,
                                num_scale_frames=_NUM_SCALE_FRAMES,
                                keyframe_interval=keyframe_interval,
                                output_device=cpu_device,
                            )
                        return active_model.inference_streaming(
                            images_t,
                            num_scale_frames=_NUM_SCALE_FRAMES,
                            keyframe_interval=keyframe_interval,
                            output_device=cpu_device,
                        )
                finally:
                    restore()

            # MPS is coerced to CPU in _load_model (LingBot-Map's float64 3D RoPE
            # is unsupported on MPS), so self._device is only ever CUDA/ROCm/CPU here.
            out = _run(model, self._device)
            logger.info("LingBot-Map inference finished in %.1fs", time.monotonic() - t_infer)

            if progress_callback is not None:
                progress_callback(0, 0, "Transferring depth + poses from GPU")
            if cancel_event is not None and cancel_event.is_set():
                raise ReconstructionCancelled("Cancelled after LingBot-Map inference")

            if not isinstance(out, dict):
                raise RuntimeError("LingBot-Map inference did not return a prediction dictionary")

            from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

            pose_enc = out.get("pose_enc")
            depth_t = out.get("depth")
            if pose_enc is None or depth_t is None:
                raise RuntimeError("LingBot-Map output missing 'pose_enc' or 'depth'")
            extrinsics_t, pred_intrinsics_t = pose_encoding_to_extri_intri(
                pose_enc, (target_h, target_w)
            )

            # [S, 3, 4]. Despite the VGGT-inherited docstring on
            # pose_encoding_to_extri_intri, LingBot-Map's pose_enc decodes to a
            # camera-to-world [R|t] (see _poses_w_c_from_decoded_pose_enc).
            poses_c2w = _squeeze_batch(_to_numpy(extrinsics_t))
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
                poses_c2w=poses_c2w,
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
            raise RuntimeError("LingBot-Map sequence inference failed") from exc

    def _predictions_to_result(
        self,
        *,
        frame_indices: list[int],
        poses_c2w: np.ndarray,
        pred_intrinsics: np.ndarray,
        depth: np.ndarray,
        depth_conf: np.ndarray | None,
        target_size: tuple[int, int],
        gravity_vectors: np.ndarray | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> MappingSequenceResult:
        if depth.ndim != 3:
            raise RuntimeError(f"LingBot-Map depth must have shape (S, H, W), got {depth.shape}")
        n_in = len(frame_indices)
        n_out = depth.shape[0]
        if n_out != n_in or poses_c2w.shape[0] != n_in or pred_intrinsics.shape[0] != n_in:
            raise RuntimeError(
                f"LingBot-Map returned {n_out} depth maps, {poses_c2w.shape[0]} poses and "
                f"{pred_intrinsics.shape[0]} intrinsics for {n_in} input frames; "
                "refusing to silently drop frames."
            )

        # The decoded [R|t] is already camera-to-world (T_w_c): pad to 4x4 without
        # inverting, then anchor camera 0 at the world origin for a reproducible frame.
        poses_w_c = _poses_w_c_from_decoded_pose_enc(poses_c2w)
        if progress_callback is not None:
            progress_callback(0, 0, "Aligning poses to world frame")
        poses_w_c, _ = _reanchor_to_first_camera(poses_w_c, None, progress_callback)
        _assert_pose_convention(poses_w_c)

        confidence = None if depth_conf is None else _confidence_from_depth_conf(depth_conf)

        target_w, target_h = target_size
        image_size = self._input_image_size or (target_w, target_h)
        intrinsics = scale_intrinsics(self._k, image_size, (target_w, target_h))

        # Kept for refine_intrinsics(): the predicted K is already in depth-scale
        # (principal point at the resized frame centre), so it needs no rescaling.
        self._predicted_intrinsics = pred_intrinsics.astype(np.float32)

        # No world_points: the released checkpoint has no point head, and
        # unprojecting here with the per-frame predicted K would bake a K into
        # the geometry that can disagree with the one the orchestrator settles
        # on (calibrated or refined). Returning depth + poses only forces the
        # cloud stage through depth_to_points(depth, K, pose) with that single,
        # shared K.
        logger.info("LingBot-Map sequence run complete for %d frames", n_in)
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
            logger.info("LingBot-Map intrinsics refinement skipped: no predicted intrinsics available.")
            return None
        k = np.median(pred.astype(np.float64), axis=0).astype(np.float32)
        # Zero the shear/bottom-row drift a per-frame median can introduce.
        k[0, 1] = 0.0
        k[1, 0] = 0.0
        k[2] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return k


def _poses_w_c_from_decoded_pose_enc(poses_c2w: np.ndarray) -> np.ndarray:
    """Pad LingBot-Map's decoded ``[R|t]`` to 4x4 camera-to-world poses.

    Unlike VGGT / VGGT-Omega, whose ``pose_enc`` decodes to camera-from-world
    (OpenCV) extrinsics that must be inverted, LingBot-Map's ``pose_enc`` holds
    the camera centre and camera-to-world rotation directly. Upstream relies on
    this: ``gct_stream_window.py`` treats the decoded matrix as ``c2w`` when
    aligning windows ("Decompose C2W: center + quaternion"), and ``demo.py``
    inverts it once so the viewer, which inverts again, ends up unprojecting with
    the decoded matrix as camera-to-world. Inverting here (as the VGGT-Omega
    backend correctly does for its model) would hand downstream the world-to-camera
    transform labelled as T_w_c, which smears every frame across the scene.

    Accepts ``(S, 3, 4)`` or ``(S, 4, 4)`` and returns ``(S, 4, 4)`` float64.
    """
    c2w = np.asarray(poses_c2w, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[-2] not in (3, 4) or c2w.shape[-1] != 4:
        raise RuntimeError(
            f"LingBot-Map poses must have shape (S, 3, 4) or (S, 4, 4), got {c2w.shape}"
        )
    poses = np.tile(np.eye(4, dtype=np.float64), (c2w.shape[0], 1, 1))
    poses[:, :3, :4] = c2w[:, :3, :4]
    return poses


def _crop_target_shape(
    input_h: int, input_w: int, image_size: int, patch_size: int
) -> tuple[int, int]:
    """Return the (H, W) LingBot-Map expects for a frame of the given size.

    Mirrors ``lingbot_map.utils.load_fn.load_and_preprocess_images`` in ``crop``
    mode: fix the width to ``image_size`` and scale the height to preserve the
    aspect ratio, rounded to whole patches. Upstream would center-crop a height
    that exceeds ``image_size``; we refuse instead so portrait inputs are never
    silently cropped.
    """
    if input_w <= 0 or input_h <= 0:
        raise RuntimeError(f"LingBot-Map received a degenerate frame size: {input_w}x{input_h}.")
    target_w = image_size
    target_h = int(round(input_h * (target_w / input_w) / patch_size) * patch_size)
    target_h = max(patch_size, target_h)
    if target_h > image_size:
        raise RuntimeError(
            "LingBot-Map crop mode fixes width to "
            f"{image_size}px, which maps {input_w}x{input_h} frames to a "
            f"{target_h}px height that exceeds {image_size}px and would be centre-cropped. "
            "Feed landscape frames (W >= H) or reduce the processing height."
        )
    return target_h, target_w


def _auto_keyframe_interval(num_frames: int, mode: str) -> int:
    """Bound the KV cache the way upstream ``demo.py`` does when unset.

    Streaming caches every frame up to the ~320-view RoPE training range, then
    strides so at most ~320 keyframes are retained. Windowed mode resets the
    cache per window, so it defaults to every frame.
    """
    if mode == "windowed":
        return 1
    if num_frames > _ROPE_TRAIN_VIEWS:
        return math.ceil(num_frames / _ROPE_TRAIN_VIEWS)
    return 1
