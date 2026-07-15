"""Estimate a run's peak memory and decide whether the machine can survive it.

The RAM check is the one that matters: a Linux RAM exhaustion is an uncatchable
OOM kill, so the goal is to refuse (or loudly warn) before a 10-30 min run rather
than crash into it. VRAM exhaustion raises a catchable CUDA error, so it is a
secondary warning, never a hard block.

Estimation is measurement-first. When the machine has recorded a comparable run
(perf_sampler peaks in run_timings.json), we scale that real peak by frame count.
Only with no history do we fall back to a deliberately conservative analytic model
whose constants are calibrated from observed runs and refined as peaks accrue.

Qt-free and dependency-light so it can be unit-tested headless.
"""

from __future__ import annotations

from dataclasses import dataclass

from deepreefmap.system_probe import GPU_CUDA, GPU_MPS, SystemProfile, format_bytes

# Analytic per-frame model, derived from the arrays the pipeline actually holds
# (traced through orchestrator/loger_backend/filters). It estimates the run's own
# RAM FOOTPRINT (allocations on top of whatever is already resident), which is why
# preflight_check grades it against FREE RAM. A measured peak from perf_sampler is
# different in kind: an absolute system-wide high-water mark graded against TOTAL
# RAM. On the 31 GB reference box a 1134-frame 3fps run footprints ~19 GB and rode
# to a 30 GB / 97% system peak; the analytic footprint matches, and once measured
# the absolute peak supersedes it.
#
# Two independent per-frame terms:
# - Prepared frames stay in RAM for the whole run at the PROCESSING resolution:
#   rgb uint8 (3) + labels int32 (4) + keep_mask uint8 (1) = 8 B/px.
# - Mapping arrays live at LoGeR's own 504x280 inference grid, NOT the
#   processing resolution. The peak stage is the pose re-anchor, where
#   local_points + world_points + rebased output (f32, 12 each) + depth (4)
#   are co-resident = 40 B per mapping pixel.
#
# The cloud stage (~110 B per kept point) overtakes the re-anchor only when
# point filtering keeps an unusually large cloud; measured peaks catch that
# case per machine. scsfmlearner maps frame-by-frame and is far lighter, so
# this model is conservative for it.
_FRAME_BATCH_BYTES_PER_PIXEL = 8
# LoGeR's default target_resolution (loger_backend.py). If a backend override
# changes it this drifts, but measured peaks supersede after one recorded run.
_MAPPING_PIXELS_PER_FRAME = 504 * 280
_REANCHOR_BYTES_PER_MAPPING_PIXEL = 40
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


@dataclass(frozen=True)
class MemoryEstimate:
    ram_bytes: int
    vram_bytes: int | None
    source: str  # "measured" | "analytic"


@dataclass(frozen=True)
class Verdict:
    level: str  # "ok" | "warn" | "block"
    ram_need_bytes: int
    ram_available_bytes: int
    headroom_bytes: int
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

    ``recorded`` (from run_history.load_expected_peaks) is a measured peak for a
    comparable run plus the frame count it ran at; when present the estimate is
    that peak scaled linearly by frames, which is far more reliable than the
    analytic fallback used on a cold machine.
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
        + _MAPPING_PIXELS_PER_FRAME * _REANCHOR_BYTES_PER_MAPPING_PIXEL
    )


def max_frames_for_ram(available_bytes: int, width: int, height: int, *, margin_bytes: int = _WARN_HEADROOM) -> int:
    """Rough analytic ceiling on frames that fit in RAM at a resolution.

    The inverse of the analytic RAM model, used by the System-tab benchmark to
    answer "how much can this machine take?" with no video loaded. Conservative:
    it keeps `margin_bytes` spare.
    """
    budget = available_bytes - _BASELINE_APP_BYTES - margin_bytes
    return max(0, int(budget / (_bytes_per_frame(width, height) * _OVERHEAD)))


def preflight_check(profile: SystemProfile, est: MemoryEstimate) -> Verdict:
    """Grade a run against the machine: ok (silent), warn, block (likely crash).

    A **measured** peak is an absolute system-wide high-water mark (it already
    includes the OS, other apps and this app's own baseline), so it is graded
    against total RAM. An **analytic** estimate is only the run's own footprint on
    top of whatever is already resident, so it is graded against free RAM. Grading
    a measured peak against free RAM would double-count the baseline and cry wolf.

    Swap is part of the budget: a run that exceeds the RAM budget but fits in that
    budget plus free swap will not be OOM-killed, it will thrash into swap and run
    slowly, so that is a warn, not a block. Only exceeding budget plus swap blocks.
    On Apple's unified memory the GPU draws from RAM, so the VRAM need is added to
    the RAM need. On a discrete CUDA GPU, VRAM is checked separately and can only
    escalate to a warning, since a VRAM OOM is recoverable.
    """
    ram_need = est.ram_bytes
    if profile.gpu.kind == GPU_MPS and est.vram_bytes:
        ram_need += est.vram_bytes
    if est.source == "measured":
        budget = profile.total_ram_bytes
        budget_word = "total RAM"
    else:
        budget = profile.available_ram_bytes
        budget_word = "free"
    swap = profile.free_swap_bytes
    headroom = budget - ram_need

    qualifier = " (estimated, no history yet)" if est.source == "analytic" else ""
    if ram_need > budget + swap - _BLOCK_HEADROOM:
        level = "block"
        swap_note = f" plus {format_bytes(swap)} swap" if swap else ""
        message = (
            f"This run is expected to reach about {format_bytes(ram_need)} of RAM, but the "
            f"machine has {format_bytes(budget)} {budget_word}{swap_note}{qualifier}. It will "
            f"very likely run out of memory and crash. Reduce the fps or the processing "
            f"resolution before running."
        )
    elif headroom < 0:
        # Fits only by spilling into swap: it will complete, but slowly.
        level = "warn"
        message = (
            f"This run is expected to reach about {format_bytes(ram_need)} of RAM, more than the "
            f"{format_bytes(budget)} {budget_word}{qualifier}. It will spill into swap and run "
            f"very slowly. A lower fps or resolution avoids the slowdown."
        )
    elif headroom < _WARN_HEADROOM:
        level = "warn"
        message = (
            f"This run is expected to reach about {format_bytes(ram_need)} of RAM, leaving only "
            f"{format_bytes(headroom)} spare{qualifier}. It may run out of memory. "
            f"Consider a lower fps or resolution."
        )
    else:
        level = "ok"
        message = f"Estimated peak RAM about {format_bytes(ram_need)}, comfortably within memory."

    # Discrete-GPU VRAM: a shortfall is recoverable, so at most warn.
    if profile.gpu.kind == GPU_CUDA and est.vram_bytes and profile.gpu.free_vram_bytes is not None:
        if est.vram_bytes > profile.gpu.free_vram_bytes and level == "ok":
            level = "warn"
            message = (
                f"Estimated VRAM about {format_bytes(est.vram_bytes)} exceeds the "
                f"{format_bytes(profile.gpu.free_vram_bytes)} free on the GPU. The run may fail "
                f"with an out-of-memory error; a lower resolution or window size helps."
            )

    return Verdict(level, ram_need, budget, headroom, message)


@dataclass(frozen=True)
class Risk:
    """Crash-risk banding of a run's peak RAM against a machine's total RAM."""

    band: str  # "safe" | "moderate" | "high" | "severe"
    label: str  # short human phrase
    percent: float  # peak as a percent of total RAM
    colour: str  # hex for the UI


# Crash-risk bands on peak-RAM-as-a-share-of-total. The metric is the memory
# high-water mark relative to physical RAM: standard practice for OOM risk, since
# the kernel starts reclaiming and (on Linux) OOM-killing as usage nears 100%.
# Below ~75% there is comfortable headroom; 75-90% is a working margin; past ~90%
# a run is riding the edge (a small overshoot tips it over); at/over 100% it can
# only proceed by paging into swap, which thrashes, and over RAM+swap it crashes.
def memory_risk(peak_used_bytes: int, total_ram_bytes: int, total_swap_bytes: int = 0) -> Risk:
    """Band a measured peak against total RAM, folding swap into the worst case."""
    pct = 100.0 * peak_used_bytes / total_ram_bytes if total_ram_bytes else 0.0
    if total_ram_bytes and peak_used_bytes > total_ram_bytes + total_swap_bytes:
        return Risk("severe", "Exceeds RAM and swap — would crash", pct, "#e05050")
    if pct >= 100.0:
        return Risk("severe", "Over RAM — spills into swap", pct, "#e05050")
    if pct >= 90.0:
        return Risk("high", "High — ran on the edge of RAM", pct, "#e07030")
    if pct >= 75.0:
        return Risk("moderate", "Moderate headroom", pct, "#e0a030")
    return Risk("safe", "Comfortable headroom", pct, "#4caf7d")
