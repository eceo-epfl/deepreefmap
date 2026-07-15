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
# (traced through orchestrator/loger_backend/filters, validated against real
# runs: 1134 frames peaked ~17 GB and fit a 31 GB box, 1890 frames did not).
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
    """Grade a run against the machine: ok (silent), warn (confirm), block (likely crash).

    On Apple's unified memory the GPU draws from system RAM, so the VRAM need is
    added to the RAM need. On a discrete CUDA GPU, VRAM is checked separately and
    can only escalate to a warning, since a VRAM OOM is recoverable.
    """
    ram_need = est.ram_bytes
    if profile.gpu.kind == GPU_MPS and est.vram_bytes:
        ram_need += est.vram_bytes
    available = profile.available_ram_bytes
    headroom = available - ram_need

    if headroom < _BLOCK_HEADROOM:
        level = "block"
    elif headroom < _WARN_HEADROOM:
        level = "warn"
    else:
        level = "ok"

    qualifier = " (estimated, no history yet)" if est.source == "analytic" else ""
    if level == "block":
        message = (
            f"This run needs about {format_bytes(ram_need)} of RAM but only "
            f"{format_bytes(available)} is free{qualifier}. It will very likely run out of "
            f"memory and crash. Reduce the fps or the processing resolution before running."
        )
    elif level == "warn":
        message = (
            f"This run needs about {format_bytes(ram_need)} of RAM, leaving only "
            f"{format_bytes(headroom)} spare{qualifier}. It may run out of memory. "
            f"Consider a lower fps or resolution."
        )
    else:
        message = f"Estimated peak RAM about {format_bytes(ram_need)}, comfortably within free memory."

    # Discrete-GPU VRAM: a shortfall is recoverable, so at most warn.
    if profile.gpu.kind == GPU_CUDA and est.vram_bytes and profile.gpu.free_vram_bytes is not None:
        if est.vram_bytes > profile.gpu.free_vram_bytes and level == "ok":
            level = "warn"
            message = (
                f"Estimated VRAM about {format_bytes(est.vram_bytes)} exceeds the "
                f"{format_bytes(profile.gpu.free_vram_bytes)} free on the GPU. The run may fail "
                f"with an out-of-memory error; a lower resolution or window size helps."
            )

    return Verdict(level, ram_need, available, headroom, message)
