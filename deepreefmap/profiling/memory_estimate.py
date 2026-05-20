"""Estimate a run's peak memory and decide whether the machine can survive it.

A Linux RAM exhaustion is an uncatchable OOM kill, so this refuses before a long run
rather than crash into it.
"""

from __future__ import annotations

from dataclasses import dataclass

from deepreefmap.profiling.system_probe import GPU_CUDA, GPU_MPS, SystemProfile, format_bytes

# Analytic per-frame model, traced through orchestrator/loger_backend/filters. Two
# independent per-frame terms:
# - Prepared frames stay in RAM for the whole run at the PROCESSING resolution:
#   rgb uint8 (3) + labels uint8 (1) + keep_mask uint8 (1) = 5 B/px.
# - Mapping arrays live at LoGeR's own 504x280 inference grid, NOT the processing
#   resolution. The peak stage is Pi3's CPU window merge: the per-window parts
#   (points + local_points f32, 12 each, + conf 4 = 28) are still referenced while
#   torch.cat materialises the merged copies (another 28) = 56 B per mapping pixel.
#
# Conservative for scsfmlearner, which maps frame-by-frame. The cloud stage (~110 B
# per kept point) overtakes the merge only when filtering keeps an unusually large
# cloud; measured peaks catch that per machine.
_FRAME_BATCH_BYTES_PER_PIXEL = 5
# LoGeR's default target_resolution (loger_backend.py). If a backend override
# changes it this drifts, but measured peaks supersede after one recorded run.
_MAPPING_PIXELS_PER_FRAME = 504 * 280
_MERGE_BYTES_PER_MAPPING_PIXEL = 56
# Allocator slack, per-block float64 transients and the confidence array.
_OVERHEAD = 1.15
# Torch, model weights and interpreter working set, present regardless of frames.
_BASELINE_APP_BYTES = int(2 * 1024**3)
# Rough flat VRAM need (model weights + window activations); refined by measurement.
_ANALYTIC_VRAM_BYTES = int(6 * 1024**3)

# Headroom thresholds against available RAM. Conservative starting points; the
# measured path makes them meaningful per machine.
_BLOCK_HEADROOM = int(2 * 1024**3)
_WARN_HEADROOM = int(6 * 1024**3)

# The OS, desktop and background apps hold several GB the run can never claim.
# Grading a measured peak against full physical RAM assumes a bare machine: a 31 GB
# run on a 33 GB box read as "low risk" and was OOM-killed the moment anything else
# was open. Reserve ~12% of total, clamped to [3 GB, 8 GB].
_OS_RESERVE_FRACTION = 0.12
_OS_RESERVE_MIN = int(3 * 1024**3)
_OS_RESERVE_MAX = int(8 * 1024**3)


def _os_reserve(total_ram_bytes: int) -> int:
    """RAM to hold back for the OS and other apps, in bytes."""
    return int(min(_OS_RESERVE_MAX, max(_OS_RESERVE_MIN, total_ram_bytes * _OS_RESERVE_FRACTION)))


@dataclass(frozen=True)
class MemoryEstimate:
    ram_bytes: int
    vram_bytes: int | None
    source: str  # "measured" | "analytic"


@dataclass(frozen=True)
class Verdict:
    level: str  # "ok" | "warn" | "block"
    risk: str  # "none" | "low" | "medium" | "high"
    ram_need_bytes: int
    ram_available_bytes: int
    headroom_bytes: int
    percent: float  # peak need as a share of the RAM (+ swap) budget
    message: str


def estimate_peak_bytes(
    frames: int,
    width: int,
    height: int,
    mapping_backend: str,
    seg_model: str,
    *,
    recorded: dict | None = None,
) -> MemoryEstimate:
    """Peak RAM/VRAM this run is expected to reach.

    ``recorded`` is a measured peak from a comparable run, scaled by frame count. Far
    more reliable than the analytic fallback, so it wins whenever it exists.
    """
    del mapping_backend, seg_model  # reserved for future per-backend calibration
    if recorded and recorded.get("frames") and recorded.get("ram_bytes"):
        ratio = frames / float(recorded["frames"])
        rec_vram = recorded.get("vram_bytes")
        return MemoryEstimate(
            ram_bytes=int(recorded["ram_bytes"] * ratio),
            vram_bytes=int(rec_vram * ratio) if rec_vram else None,
            source="measured",
        )
    ram = _BASELINE_APP_BYTES + int(frames * _bytes_per_frame(width, height) * _OVERHEAD)
    return MemoryEstimate(ram_bytes=ram, vram_bytes=_ANALYTIC_VRAM_BYTES, source="analytic")


def _bytes_per_frame(width: int, height: int) -> int:
    """Per-frame resident bytes at the run's peak stage (see the model above)."""
    return (
        max(1, width * height) * _FRAME_BATCH_BYTES_PER_PIXEL
        + _MAPPING_PIXELS_PER_FRAME * _MERGE_BYTES_PER_MAPPING_PIXEL
    )


def preflight_check(profile: SystemProfile, est: MemoryEstimate) -> Verdict:
    """Grade a run against the machine: ok (silent), warn, block (likely crash)."""
    ram_need = est.ram_bytes
    if profile.gpu.kind == GPU_MPS and est.vram_bytes:
        ram_need += est.vram_bytes  # unified memory: the GPU draws from system RAM
    if est.source == "measured":
        # A measured peak is a system-wide high-water mark that already includes the
        # OS and this app's baseline, so it grades against usable RAM. An analytic
        # estimate is only the run's own footprint on top of what is already
        # resident, so it grades against free RAM. Swapping the two double-counts
        # the baseline and cries wolf.
        budget = max(0, profile.total_ram_bytes - _os_reserve(profile.total_ram_bytes))
    else:
        budget = profile.available_ram_bytes
    swap = profile.free_swap_bytes
    capacity = budget + swap  # everything the run can draw on before it crashes
    headroom = budget - ram_need
    pct = 100.0 * ram_need / capacity if capacity else 0.0
    cap_word = "RAM + swap" if swap else "RAM"

    tag = " (estimated)" if est.source == "analytic" else ""
    need = format_bytes(ram_need)
    if ram_need > capacity - _BLOCK_HEADROOM:
        level, risk = "block", "high"
        message = (
            f"~{need} needed, {pct:.0f}% of {cap_word}{tag}. "
            f"Likely to crash. Lower the fps or resolution."
        )
    elif headroom < 0:
        # Fits only by spilling into swap. On a large-allocation mapping run this
        # thrashes and can still OOM once other apps grow, so it is a strong warning.
        level, risk = "warn", "high"
        message = (
            f"~{need} needed, over the {format_bytes(budget)} of RAM the run can "
            f"claim{tag}. Spills into swap and may crash if other apps are open. "
            f"Lower the fps or resolution."
        )
    elif headroom < _WARN_HEADROOM:
        level, risk = "warn", "low"
        message = (
            f"~{need} needed, {pct:.0f}% of {cap_word}{tag}. "
            f"Could run out. Lower the fps or resolution."
        )
    else:
        level, risk = "ok", "none"
        message = f"~{need} needed, {pct:.0f}% of {cap_word}. Comfortable."

    # Discrete-GPU VRAM: a shortfall is recoverable, so at most warn.
    if profile.gpu.kind == GPU_CUDA and est.vram_bytes and profile.gpu.free_vram_bytes is not None:
        if est.vram_bytes > profile.gpu.free_vram_bytes and level == "ok":
            level, risk = "warn", "low"
            message = (
                f"~{format_bytes(est.vram_bytes)} VRAM needed, over "
                f"{format_bytes(profile.gpu.free_vram_bytes)} free on the GPU. "
                f"May hit a VRAM out-of-memory error. Lower the resolution or window size."
            )

    return Verdict(level, risk, ram_need, budget, headroom, pct, message)


@dataclass(frozen=True)
class Risk:
    """Crash-risk banding of a run's peak RAM against a machine's total RAM."""

    band: str  # "safe" | "moderate" | "high" | "severe"
    label: str  # short human phrase
    percent: float  # peak as a percent of total RAM
    color: str  # hex for the UI


# Bands on committed memory (RAM + swap) as a share of physical RAM: the kernel
# reclaims and (on Linux) OOM-kills as usage nears 100%.
def memory_risk(
    peak_ram_bytes: int, total_ram_bytes: int, total_swap_bytes: int = 0, peak_swap_bytes: int = 0
) -> Risk:
    """Band a measured peak against total RAM, counting swap as secondary RAM."""
    committed = peak_ram_bytes + peak_swap_bytes
    pct = 100.0 * committed / total_ram_bytes if total_ram_bytes else 0.0
    if total_ram_bytes and committed > total_ram_bytes + total_swap_bytes:
        return Risk("severe", "Exceeds RAM + swap", pct, "#e05050")
    if total_ram_bytes and committed > total_ram_bytes:
        return Risk("severe", f"In swap (+{format_bytes(committed - total_ram_bytes)})", pct, "#e05050")
    if pct >= 90.0:
        return Risk("high", "Near RAM limit", pct, "#e07030")
    if pct >= 75.0:
        return Risk("moderate", "Moderate", pct, "#e0a030")
    return Risk("safe", "Comfortable", pct, "#4caf7d")
