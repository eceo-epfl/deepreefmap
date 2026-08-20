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
    """Permit AOTriton flash SDPA on ROCm: LoGeR's bf16 path aborts without it on RDNA3."""
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
    """Pick the graphics device the models run on.

    The GUI ships on NVIDIA, AMD and Apple laptops. The backends used to
    assume NVIDIA. One shared chooser keeps every model on the same hardware.
    """
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None) is not None:
            _enable_rocm_experimental_attention()
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@functools.lru_cache(maxsize=None)
def _flash_sdpa_works() -> bool:
    """True if a bfloat16 flash attention kernel actually runs on this GPU.

    Some CUDA builds ship without one, so a quick probe beats trusting the wheel.
    """
    if not torch.cuda.is_available():
        return False
    try:
        q = torch.zeros(1, 1, 8, 64, dtype=torch.bfloat16, device="cuda")  # smallest shape the kernel accepts
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

    Windows torch wheels ship without Triton, so LoGeR's compiled functions
    would abort. Same numbers either way, just slower.
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
    """Choose the reduced precision the models run inference in.

    bfloat16 where the GPU truly supports it. The GUI's Windows builds lack
    the flash attention LoGeR's bfloat16 path leans on, so they drop to
    float16 or crash mid-run. One policy here, not one per model.
    """
    if device.type == "cuda":
        try:
            capability = torch.cuda.get_device_capability(device)[0]  # 8 = Ampere, the first bfloat16 generation
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


def release_device_memory(device: torch.device) -> None:
    """Free leftover GPU memory between pipeline stages.

    Field laptops running the GUI have little GPU memory. PyTorch keeps a
    finished model's memory reserved, so segmentation can starve mapping even
    though only one model is loaded at a time.
    """
    gc.collect()
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif device.type == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass
