"""Pre-run memory estimate and the ok/warn/block verdict, including the M1 case."""

from __future__ import annotations

from deepreefmap.memory_estimate import estimate_peak_bytes, preflight_check
from deepreefmap.system_probe import GPU_CUDA, GPU_MPS, GPU_NONE, GpuInfo, SystemProfile

_GB = 1024**3


def _profile(*, avail_gb, total_gb=None, gpu=None):
    total_gb = total_gb or avail_gb
    gpu = gpu or GpuInfo(GPU_NONE, "CPU only", None, None)
    return SystemProfile(
        os_name="Linux", os_release="x", cpu_logical=8, cpu_physical=4,
        total_ram_bytes=int(total_gb * _GB), available_ram_bytes=int(avail_gb * _GB),
        gpu=gpu, disk_total_bytes=0, disk_free_bytes=0, disk_path="/",
    )


def test_measured_estimate_scales_linearly_with_frames():
    recorded = {"ram_bytes": 30 * _GB, "vram_bytes": 8 * _GB, "frames": 1000}
    est = estimate_peak_bytes(2000, 1376, 768, "loger_star", "seg", recorded=recorded)
    assert est.source == "measured"
    assert est.ram_bytes == 60 * _GB
    assert est.vram_bytes == 16 * _GB


def test_analytic_estimate_grows_with_frames_and_resolution():
    small = estimate_peak_bytes(500, 1376, 768, "loger_star", "seg")
    big = estimate_peak_bytes(2000, 1376, 768, "loger_star", "seg")
    assert big.ram_bytes > small.ram_bytes
    assert small.source == "analytic"
    hi_res = estimate_peak_bytes(500, 2752, 1536, "loger_star", "seg")
    assert hi_res.ram_bytes > small.ram_bytes


def test_comfortable_run_is_ok():
    est = estimate_peak_bytes(500, 1376, 768, "loger_star", "seg")  # ~a few GB
    verdict = preflight_check(_profile(avail_gb=64), est)
    assert verdict.level == "ok"


def test_m1_8gb_blocks_a_large_run():
    # 1890 frames on an 8GB unified-memory Mac: VRAM competes with RAM.
    est = estimate_peak_bytes(1890, 1376, 768, "loger_star", "seg")
    mps = GpuInfo(GPU_MPS, "Apple GPU", None, None)
    verdict = preflight_check(_profile(avail_gb=7, total_gb=8, gpu=mps), est)
    assert verdict.level == "block"
    assert "reduce" in verdict.message.lower() or "crash" in verdict.message.lower()


def test_tight_headroom_warns():
    recorded = {"ram_bytes": 27 * _GB, "vram_bytes": None, "frames": 1000}
    est = estimate_peak_bytes(1000, 1376, 768, "loger_star", "seg", recorded=recorded)
    verdict = preflight_check(_profile(avail_gb=31, total_gb=32), est)
    assert verdict.level == "warn"


def test_cuda_vram_shortfall_only_warns_never_blocks_on_ram():
    # RAM is comfortable but the estimated VRAM exceeds free GPU memory.
    recorded = {"ram_bytes": 4 * _GB, "vram_bytes": 20 * _GB, "frames": 1000}
    est = estimate_peak_bytes(1000, 1376, 768, "loger_star", "seg", recorded=recorded)
    cuda = GpuInfo(GPU_CUDA, "RTX", total_vram_bytes=24 * _GB, free_vram_bytes=8 * _GB)
    verdict = preflight_check(_profile(avail_gb=64, gpu=cuda), est)
    assert verdict.level == "warn"
    assert "vram" in verdict.message.lower()
