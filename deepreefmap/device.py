from __future__ import annotations

import functools
import gc
import importlib.util
import logging
import os

import torch

logger = logging.getLogger(__name__)

_rocm_attention_notice_emitted = False
_triton_notice_emitted = False


def _enable_rocm_experimental_attention() -> None:
    """Permit AOTriton flash/mem-efficient SDPA on ROCm.

    LoGeR forces the flash SDPA backend for bf16 (third_party/LoGeR attention),
    which PyTorch's HIP build gates behind this flag on RDNA3 (e.g. gfx1100). Without
    it, the flash-only context manager has no fallback and aborts with "No available
    kernel". `setdefault` lets a user force it off with the same variable set to 0.
    """
    global _rocm_attention_notice_emitted
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    if not _rocm_attention_notice_emitted:
        logger.warning(
            "ROCm/HIP build detected: ROCm support is experimental. Enabling AOTriton "
            "flash/mem-efficient attention via TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=%s "
            "(set it to 0 to disable).",
            os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"],
        )
        _rocm_attention_notice_emitted = True


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None) is not None:
            _enable_rocm_experimental_attention()
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@functools.lru_cache(maxsize=None)
def _flash_sdpa_works() -> bool:
    """True if a bf16 flash-attention SDPA kernel actually runs on this GPU.

    LoGeR forces the flash-only SDPA backend for bfloat16 activations
    (third_party/LoGeR attention.py), so picking bf16 autocast on a build
    without flash aborts with "No available kernel". Windows CUDA wheels ship
    without flash (cu130/sm_120 in particular). A tiny probe beats build
    flags: a wheel can include flash for other architectures only.
    """
    if not torch.cuda.is_available():
        return False
    try:
        q = torch.zeros(1, 1, 8, 64, dtype=torch.bfloat16, device="cuda")
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        return True
    except Exception:
        logger.warning(
            "Flash-attention SDPA unavailable on this torch build/GPU; "
            "using float16 autocast instead of bfloat16."
        )
        return False


def disable_torch_compile_without_triton() -> None:
    """Run ``@torch.compile`` functions eagerly when Triton is missing.

    torch.compile's inductor backend needs Triton to generate GPU kernels, but
    Windows torch wheels ship without it, so LoGeR's compiled TTT functions
    (third_party/LoGeR ttt.py) abort with TritonMissing. Dynamo checks
    ``config.disable`` at call time, so flipping it after the decorators have
    run still forces eager execution — numerically identical, just slower.
    """
    global _triton_notice_emitted
    try:
        if importlib.util.find_spec("triton") is not None:
            return
        torch._dynamo.config.disable = True
        if not _triton_notice_emitted:
            logger.warning(
                "Triton unavailable on this torch build; running "
                "torch.compile-decorated functions eagerly."
            )
            _triton_notice_emitted = True
    except Exception:
        pass


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        try:
            capability = torch.cuda.get_device_capability(device)[0]
            return (
                torch.bfloat16
                if capability >= 8 and _flash_sdpa_works()
                else torch.float16
            )
        except Exception:
            return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def autocast_context(
    device: torch.device, enabled: bool = True
) -> torch.amp.autocast:
    return torch.amp.autocast(
        device_type=device.type,
        enabled=enabled,
        dtype=get_autocast_dtype(device),
    )


def estimate_segmentation_batch_size(
    device: torch.device,
    width: int = 1376,
    height: int = 768,
) -> int:
    """Suggest a batch size that fits in free VRAM at the given resolution.

    Thresholds are calibrated against SegFormer-B2 at 1376×768 (~1.5 GB per
    frame in attention activations + ~1 GB weights) and scale linearly with
    pixel count.
    """
    try:
        if device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(device)
        elif device.type == "mps":
            free = torch.mps.driver_allocated_memory()
            total = torch.mps.recommended_max_working_set_size()  # type: ignore[attr-defined]  # torch.mps stubs omit this
            free = max(0, total - free)
        else:
            free = 4 * 1024**3
    except Exception:
        free = 4 * 1024**3
    free_gb = free / (1024**3)
    _BASELINE_PIXELS = 1376 * 768
    scale = max((width * height) / _BASELINE_PIXELS, 0.25)
    if free_gb >= 12 * scale:
        return 4
    if free_gb >= 6 * scale:
        return 2
    return 1


def release_device_memory(device: torch.device) -> None:
    gc.collect()
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif device.type == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass
