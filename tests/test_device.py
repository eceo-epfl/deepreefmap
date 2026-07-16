from __future__ import annotations

from unittest.mock import patch

import torch

from deepreefmap.device import (
    autocast_context,
    get_autocast_dtype,
    release_device_memory,
    resolve_device,
)


def test_resolve_device_prefers_cuda():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device().type == "cuda"


def test_resolve_device_prefers_mps_over_cpu():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device().type == "mps"


def test_resolve_device_falls_back_to_cpu():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_device().type == "cpu"


def test_get_autocast_dtype_mps():
    assert get_autocast_dtype(torch.device("mps")) == torch.float16


def test_get_autocast_dtype_cpu():
    assert get_autocast_dtype(torch.device("cpu")) == torch.bfloat16


def test_get_autocast_dtype_safe_on_capability_error():
    with patch("torch.cuda.get_device_capability", side_effect=RuntimeError("no device")):
        assert get_autocast_dtype(torch.device("cuda")) == torch.float16


def test_get_autocast_dtype_cuda_bf16_needs_flash():
    with patch("torch.cuda.get_device_capability", return_value=(12, 0)):
        with patch("deepreefmap.device._flash_sdpa_works", return_value=True):
            assert get_autocast_dtype(torch.device("cuda")) == torch.bfloat16
        with patch("deepreefmap.device._flash_sdpa_works", return_value=False):
            assert get_autocast_dtype(torch.device("cuda")) == torch.float16


def test_get_autocast_dtype_old_gpu_skips_flash_probe():
    with (
        patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        patch(
            "deepreefmap.device._flash_sdpa_works",
            side_effect=AssertionError("probe must not run below capability 8"),
        ),
    ):
        assert get_autocast_dtype(torch.device("cuda")) == torch.float16


def test_autocast_context_returns_context_manager():
    ctx = autocast_context(torch.device("cpu"))
    assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")


def test_release_device_memory_cpu_no_error():
    release_device_memory(torch.device("cpu"))
