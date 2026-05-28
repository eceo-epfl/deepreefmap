from __future__ import annotations

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        try:
            capability = torch.cuda.get_device_capability(device)[0]
            return torch.bfloat16 if capability >= 8 else torch.float16
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


def estimate_segmentation_batch_size(device: torch.device) -> int:
    # SegFormer-B2 at 1376x768 uses ~1.5 GB per frame in attention activations
    # plus ~1 GB for weights. Thresholds leave headroom for fragmentation and
    # the desktop compositor. Not model-aware — treats B2 as the baseline.
    try:
        if device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(device)
        elif device.type == "mps":
            free = torch.mps.driver_allocated_memory()
            total = torch.mps.recommended_max_working_set_size()
            free = max(0, total - free)
        else:
            free = 4 * 1024**3
    except Exception:
        free = 4 * 1024**3
    free_gb = free / (1024**3)
    if free_gb >= 12:
        return 4
    if free_gb >= 6:
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
