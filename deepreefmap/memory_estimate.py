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

# Analytic per-frame model (post-local_points-drop). Each retained frame holds
# world_points (HxWx3 f32 = 12B/px) + depth (4B) + confidence (4B) = 20B/px.
_BYTES_PER_PIXEL_FRAME = 20
# The cloud stage concatenates a second copy of the points while the source is
# still alive, so the true peak overlaps roughly two frame-sized sets.
_PEAK_CONCURRENCY = 2.0
# numpy temporaries and allocator fragmentation on top.
_OVERHEAD = 1.25
# Torch, model weights and interpreter working set, present regardless of frames.
_BASELINE_APP_BYTES = int(2.5 * 1024**3)
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
    pixels = max(1, width * height)
    ram = _BASELINE_APP_BYTES + int(frames * pixels * _BYTES_PER_PIXEL_FRAME * _PEAK_CONCURRENCY * _OVERHEAD)
    return MemoryEstimate(ram_bytes=ram, vram_bytes=_ANALYTIC_VRAM_BYTES, source="analytic")


def max_frames_for_ram(available_bytes: int, width: int, height: int, *, margin_bytes: int = _WARN_HEADROOM) -> int:
    """Rough analytic ceiling on frames that fit in RAM at a resolution.

    The inverse of the analytic RAM model, used by the System-tab benchmark to
    answer "how much can this machine take?" with no video loaded. Conservative:
    it keeps `margin_bytes` spare.
    """
    pixels = max(1, width * height)
    budget = available_bytes - _BASELINE_APP_BYTES - margin_bytes
    per_frame = pixels * _BYTES_PER_PIXEL_FRAME * _PEAK_CONCURRENCY * _OVERHEAD
    return max(0, int(budget / per_frame))


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
