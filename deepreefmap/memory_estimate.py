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
# RAM; once a comparable run has been recorded the measured peak supersedes the
# analytic footprint.
#
# Two independent per-frame terms:
# - Prepared frames stay in RAM for the whole run at the PROCESSING resolution:
#   rgb uint8 (3) + labels int32 (4) + keep_mask uint8 (1) = 8 B/px.
# - Mapping arrays live at LoGeR's own 504x280 inference grid, NOT the
#   processing resolution. The peak stage is Pi3's CPU window merge, where
#   the per-window parts (points + local_points f32, 12 each, + conf 4 =
#   28) are still referenced while torch.cat materialises the merged copies
#   (another 28) = 56 B per mapping pixel. The later re-anchor rebases
#   points in place and adds no full-size buffer (loger_backend.py).
#
# The cloud stage (~110 B per kept point) overtakes the merge only when
# point filtering keeps an unusually large cloud; measured peaks catch that
# case per machine. scsfmlearner maps frame-by-frame and is far lighter, so
# this model is conservative for it.
_FRAME_BATCH_BYTES_PER_PIXEL = 8
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
        + _MAPPING_PIXELS_PER_FRAME * _MERGE_BYTES_PER_MAPPING_PIXEL
    )


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
    pct = 100.0 * ram_need / budget if budget else 0.0

    tag = " (estimated)" if est.source == "analytic" else ""
    if ram_need > budget + swap - _BLOCK_HEADROOM:
        level = "block"
        swap_note = f" + {format_bytes(swap)} swap" if swap else ""
        message = (
            f"~{format_bytes(ram_need)} needed, over {format_bytes(budget)} {budget_word}"
            f"{swap_note}{tag}. Likely to crash. Lower the fps or resolution."
        )
    elif headroom < 0:
        # Fits only by spilling into swap: it completes, but thrashes.
        level = "warn"
        message = (
            f"~{format_bytes(ram_need)} needed: runs in RAM + swap (~{pct:.0f}% of RAM){tag}, "
            f"slowly. Lower the fps or resolution."
        )
    elif headroom < _WARN_HEADROOM:
        level = "warn"
        message = (
            f"~{format_bytes(ram_need)} needed, {format_bytes(headroom)} spare{tag}. "
            f"May run out of memory. Lower the fps or resolution."
        )
    else:
        level = "ok"
        message = f"Peak ~{format_bytes(ram_need)}, within memory."

    # Discrete-GPU VRAM: a shortfall is recoverable, so at most warn.
    if profile.gpu.kind == GPU_CUDA and est.vram_bytes and profile.gpu.free_vram_bytes is not None:
        if est.vram_bytes > profile.gpu.free_vram_bytes and level == "ok":
            level = "warn"
            message = (
                f"~{format_bytes(est.vram_bytes)} VRAM needed, over {format_bytes(profile.gpu.free_vram_bytes)} "
                f"free on the GPU. May hit a VRAM out-of-memory error. Lower the resolution or window size."
            )

    return Verdict(level, ram_need, budget, headroom, message)


@dataclass(frozen=True)
class Risk:
    """Crash-risk banding of a run's peak RAM against a machine's total RAM."""

    band: str  # "safe" | "moderate" | "high" | "severe"
    label: str  # short human phrase
    percent: float  # peak as a percent of total RAM
    colour: str  # hex for the UI


# Crash-risk bands on committed-memory (RAM + swap) as a share of physical RAM.
# The metric is the memory high-water mark relative to RAM: standard OOM-risk
# practice, since the kernel reclaims and (on Linux) OOM-kills as usage nears 100%.
# Below ~75% is comfortable; 75-90% a working margin; past ~90% headroom is thin;
# once committed exceeds RAM the run pages into swap and slows sharply; over
# RAM+swap it crashes. peak_swap_bytes folds a run's measured spill into the total.
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
