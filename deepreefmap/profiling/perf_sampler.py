"""Background RAM/VRAM sampler: measure real peak memory per pipeline stage."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceSample:
    t: float  # time.monotonic() timestamp, comparable to the orchestrator's stage marks
    ram_bytes: int
    vram_bytes: int | None
    swap_bytes: int = 0  # system swap in use; secondary RAM once physical RAM fills


class ResourceSampler:
    """Poll memory use on a daemon thread until stopped, mirroring the viser loop pattern."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[ResourceSample] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="drm-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        from deepreefmap.profiling.system_probe import sample_utilisation

        while not self._stop.is_set():
            try:
                util = sample_utilisation()
                self.samples.append(
                    ResourceSample(
                        time.monotonic(), util.ram_used_bytes, util.vram_used_bytes, util.swap_used_bytes
                    )
                )
            except Exception:
                pass
            # Interruptible sleep: stop() returns promptly instead of waiting a full interval.
            self._stop.wait(self._interval)


def peaks_from_marks(
    samples: list[ResourceSample],
    spans: tuple[tuple[str, str, str], ...],
    marks: dict[str, float],
) -> dict[str, dict[str, int | None]]:
    """Peak RAM/VRAM/swap within each stage span, keyed like `_durations_from_marks`.

    Swap is captured alongside RAM because a run that fills physical RAM pins it near
    100% and shows its real demand as swap, so a stage's true peak is RAM plus swap.
    """
    peaks: dict[str, dict[str, int | None]] = {}
    for begin, end, stage in spans:
        if begin not in marks or end not in marks or marks[end] < marks[begin]:
            continue
        t0, t1 = marks[begin], marks[end]
        window = [s for s in samples if t0 <= s.t <= t1]
        if not window:
            continue
        vrams = [s.vram_bytes for s in window if s.vram_bytes is not None]
        peaks[stage] = {
            "ram_bytes": max(s.ram_bytes for s in window),
            "vram_bytes": max(vrams) if vrams else None,
            "swap_bytes": max(s.swap_bytes for s in window),
        }
    return peaks
