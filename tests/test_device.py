from __future__ import annotations

from unittest.mock import patch

import torch

from deepreefmap.device import (
    autocast_context,
    disable_torch_compile_without_triton,
    get_autocast_dtype,
    release_device_memory,
    resolve_device,
)


def test_resolve_device_prefers_cuda() -> None:
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device().type == "cuda"


def test_resolve_device_prefers_cuda_over_mps() -> None:
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device().type == "cuda"


def test_resolve_device_prefers_mps_over_cpu() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device().type == "mps"


def test_resolve_device_falls_back_to_cpu() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_device().type == "cpu"


def test_get_autocast_dtype_mps() -> None:
    assert get_autocast_dtype(torch.device("mps")) == torch.float16


def test_get_autocast_dtype_cpu() -> None:
    assert get_autocast_dtype(torch.device("cpu")) == torch.bfloat16


def test_get_autocast_dtype_safe_on_capability_error() -> None:
    with patch("torch.cuda.get_device_capability", side_effect=RuntimeError("no device")):
        assert get_autocast_dtype(torch.device("cuda")) == torch.float16


def test_get_autocast_dtype_cuda_bf16_needs_flash() -> None:
    with patch("torch.cuda.get_device_capability", return_value=(12, 0)):
        with patch("deepreefmap.device._flash_sdpa_works", return_value=True):
            assert get_autocast_dtype(torch.device("cuda")) == torch.bfloat16
        with patch("deepreefmap.device._flash_sdpa_works", return_value=False):
            assert get_autocast_dtype(torch.device("cuda")) == torch.float16


def test_get_autocast_dtype_old_gpu_skips_flash_probe() -> None:
    # The probe answers yes, so only the capability gate can force float16.
    with (
        patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        patch("deepreefmap.device._flash_sdpa_works", return_value=True),
    ):
        assert get_autocast_dtype(torch.device("cuda")) == torch.float16


def test_disable_torch_compile_without_triton_disables_dynamo() -> None:
    original = torch._dynamo.config.disable
    try:
        with patch("deepreefmap.device.importlib.util.find_spec", return_value=None):
            torch._dynamo.config.disable = False
            disable_torch_compile_without_triton()
            assert torch._dynamo.config.disable is True
    finally:
        torch._dynamo.config.disable = original


def test_disable_torch_compile_with_triton_leaves_dynamo_alone() -> None:
    original = torch._dynamo.config.disable
    try:
        with patch("deepreefmap.device.importlib.util.find_spec", return_value=object()):
            torch._dynamo.config.disable = False
            disable_torch_compile_without_triton()
            assert torch._dynamo.config.disable is False
    finally:
        torch._dynamo.config.disable = original


def test_autocast_context_returns_context_manager() -> None:
    ctx = autocast_context(torch.device("cpu"))
    assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")


def test_release_device_memory_cpu_no_error() -> None:
    release_device_memory(torch.device("cpu"))
