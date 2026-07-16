"""Pre-run memory estimate and the ok/warn/block verdict, including the M1 case."""

from __future__ import annotations

from deepreefmap.memory_estimate import estimate_peak_bytes, memory_risk, preflight_check
from deepreefmap.system_probe import GPU_CUDA, GPU_MPS, GPU_NONE, GpuInfo, SystemProfile

_GB = 1024**3


def _profile(*, avail_gb, total_gb=None, gpu=None, swap_gb=0):
    total_gb = total_gb or avail_gb
    gpu = gpu or GpuInfo(GPU_NONE, "CPU only", None, None)
    return SystemProfile(
        os_name="Linux", os_release="x", cpu_logical=8, cpu_physical=4,
        total_ram_bytes=int(total_gb * _GB), available_ram_bytes=int(avail_gb * _GB),
        total_swap_bytes=int(swap_gb * _GB), free_swap_bytes=int(swap_gb * _GB),
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
    # 27 GB peak on 34 GB RAM: within the usable budget but under the warn headroom,
    # so it warns (low) rather than blocking.
    recorded = {"ram_bytes": 27 * _GB, "vram_bytes": None, "frames": 1000}
    est = estimate_peak_bytes(1000, 1376, 768, "loger_star", "seg", recorded=recorded)
    verdict = preflight_check(_profile(avail_gb=33, total_gb=34), est)
    assert verdict.level == "warn"
    assert verdict.risk == "low"


def test_cuda_vram_shortfall_only_warns_never_blocks_on_ram():
    # RAM is comfortable but the estimated VRAM exceeds free GPU memory.
    recorded = {"ram_bytes": 4 * _GB, "vram_bytes": 20 * _GB, "frames": 1000}
    est = estimate_peak_bytes(1000, 1376, 768, "loger_star", "seg", recorded=recorded)
    cuda = GpuInfo(GPU_CUDA, "RTX", total_vram_bytes=24 * _GB, free_vram_bytes=8 * _GB)
    verdict = preflight_check(_profile(avail_gb=64, gpu=cuda), est)
    assert verdict.level == "warn"
    assert "vram" in verdict.message.lower()


def test_exceeding_ram_but_fitting_swap_warns_high_not_blocks():
    # 40 GB run on 32 GB RAM with 40 GB swap: it fits in RAM + swap so it does not
    # hard-block, but it overruns the RAM the run can actually claim, so it thrashes
    # swap and is a HIGH warning, not the old reassuring "runs slowly".
    recorded = {"ram_bytes": 40 * _GB, "vram_bytes": None, "frames": 1000}
    est = estimate_peak_bytes(1000, 1376, 768, "loger_star", "seg", recorded=recorded)
    verdict = preflight_check(_profile(avail_gb=30, total_gb=32, swap_gb=40), est)
    assert verdict.level == "warn"
    assert verdict.risk == "high"
    assert "swap" in verdict.message.lower()


def test_os_reserve_flips_a_near_full_run_from_low_to_high():
    # 30 GB peak on a 33 GB machine: naively 3 GB "spare", so the old model called
    # it low risk. But the OS and other apps hold several GB the run can never take,
    # so the real budget is under 30 GB and the run spills into swap -> high.
    recorded = {"ram_bytes": 30 * _GB, "vram_bytes": None, "frames": 1000}
    est = estimate_peak_bytes(1000, 1376, 768, "loger_star", "seg", recorded=recorded)
    verdict = preflight_check(_profile(avail_gb=28, total_gb=33, swap_gb=32), est)
    assert verdict.level == "warn"
    assert verdict.risk == "high"
    assert verdict.ram_need_bytes > verdict.ram_available_bytes  # over the usable budget


def test_exceeding_ram_and_swap_blocks():
    recorded = {"ram_bytes": 40 * _GB, "vram_bytes": None, "frames": 1000}
    est = estimate_peak_bytes(1000, 1376, 768, "loger_star", "seg", recorded=recorded)
    verdict = preflight_check(_profile(avail_gb=30, total_gb=32, swap_gb=4), est)
    assert verdict.level == "block"


def test_measured_peak_is_graded_against_total_not_free():
    # A measured peak is an absolute system-wide high-water mark: it already
    # includes the resident baseline, so it is judged against total RAM (minus the
    # OS/other-apps floor), not the 10 GB momentarily free. The 20 GB peak still
    # fits the 32 GB machine comfortably.
    recorded = {"ram_bytes": 20 * _GB, "vram_bytes": None, "frames": 1000}
    est = estimate_peak_bytes(1000, 1376, 768, "loger_star", "seg", recorded=recorded)
    assert est.source == "measured"
    verdict = preflight_check(_profile(avail_gb=10, total_gb=32), est)
    assert verdict.level == "ok"
    # Budget is total minus the OS reserve: well above the 10 GB free, below 32 GB.
    assert 10 * _GB < verdict.ram_available_bytes < 32 * _GB


def test_memory_risk_bands():
    total = 32 * _GB
    assert memory_risk(16 * _GB, total).band == "safe"        # 50%
    assert memory_risk(25 * _GB, total).band == "moderate"    # 78%
    assert memory_risk(30 * _GB, total).band == "high"        # 94%, on the edge
    assert memory_risk(33 * _GB, total).band == "severe"      # over RAM, swaps
    # Over RAM and swap combined is the crash case.
    assert memory_risk(40 * _GB, total, total_swap_bytes=4 * _GB).band == "severe"
    # Fits into RAM plus swap: severe (it thrashes) but a distinct label.
    over = memory_risk(36 * _GB, total, total_swap_bytes=16 * _GB)
    assert over.band == "severe" and "swap" in over.label.lower()


def test_memory_risk_counts_measured_swap_as_committed():
    # RAM alone sits at 94% (moderate-to-high), but the run spilled 8 GB into swap,
    # so committed = 38 GB > 32 GB RAM: it was thrashing, which is the real risk.
    total = 32 * _GB
    ram_only = memory_risk(30 * _GB, total, total_swap_bytes=32 * _GB, peak_swap_bytes=0)
    with_swap = memory_risk(30 * _GB, total, total_swap_bytes=32 * _GB, peak_swap_bytes=8 * _GB)
    assert ram_only.band == "high"
    assert with_swap.band == "severe" and "swap" in with_swap.label.lower()
    assert with_swap.percent > 100.0
