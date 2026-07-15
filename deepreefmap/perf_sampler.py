"""Background RAM/VRAM sampler: measure real peak memory per pipeline stage.

Peaks measured on this machine beat any analytic guess, so the pre-run memory
check (memory_estimate.py) is fed from what past runs actually used rather than a
formula. The sampler is a daemon thread that polls system_probe.sample_utilisation
at a steady rate into a timestamped buffer; folding that buffer against the
orchestrator's stage marks (`peaks_from_marks`) gives per-stage peak RAM and VRAM.

Qt-free so the pipeline can use it headless and it can be unit-tested by feeding
`peaks_from_marks` a hand-built sample list.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceSample:
    t: float  # time.monotonic() timestamp, comparable to the orchestrator's stage marks
    ram_bytes: int
    vram_bytes: int | None


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
        from deepreefmap.system_probe import sample_utilisation

        while not self._stop.is_set():
            try:
                util = sample_utilisation()
                self.samples.append(ResourceSample(time.monotonic(), util.ram_used_bytes, util.vram_used_bytes))
            except Exception:
                pass
            # Interruptible sleep: stop() returns promptly instead of waiting a full interval.
            self._stop.wait(self._interval)


def peaks_from_marks(
    samples: list[ResourceSample],
    spans: tuple[tuple[str, str, str], ...],
    marks: dict[str, float],
) -> dict[str, dict[str, int | None]]:
    """Peak RAM/VRAM within each stage span, keyed like `_durations_from_marks`.

    A stage is reported only when both its marks were reached and at least one
    sample landed inside the window, so a geometry-only run omits stages it never
    ran and a stage shorter than the sample interval is simply absent.
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
        }
    return peaks
