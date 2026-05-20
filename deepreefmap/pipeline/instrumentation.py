"""Stage timing marks and resource sampling for one reconstruction run."""

from __future__ import annotations

import time
from pathlib import Path

from deepreefmap.profiling.perf_sampler import ResourceSampler, peaks_from_marks

# Coarse pipeline stages, in order, that the timing marks bracket. The GUI reads
# these durations back to build a per-machine cost profile
# (profiling/run_history.py), so the keys match profiling/eta.py's stage keys.
STAGE_SPANS: tuple[tuple[str, str, str], ...] = (
    ("start", "preprocess", "startup"),
    ("preprocess", "mapping", "preprocess"),
    ("mapping", "cloud", "mapping"),
    ("cloud", "ortho", "cloud"),
    ("ortho", "save", "ortho"),
    ("save", "end", "save_view"),
    ("end", "scene_end", "scene_save"),
)


def durations_from_marks(marks: dict[str, float]) -> dict[str, float]:
    """Wall-clock seconds per coarse stage from sequential monotonic marks."""
    durations: dict[str, float] = {}
    for begin, end, stage in STAGE_SPANS:
        if begin in marks and end in marks and marks[end] >= marks[begin]:
            durations[stage] = marks[end] - marks[begin]
    return durations


class RunInstrumentation:
    """One run's stage marks, memory sampling and machine profile.

    The real per-stage peaks land in the manifest and feed the pre-run memory check.
    """

    def __init__(self, output_dir: Path) -> None:
        from deepreefmap.profiling.system_probe import probe_system

        self.marks: dict[str, float] = {"start": time.monotonic()}
        self.system_profile: dict = probe_system(output_dir).to_dict()
        self._sampler = ResourceSampler()
        self._sampler.start()

    def mark(self, name: str) -> None:
        self.marks[name] = time.monotonic()

    def stage_durations(self) -> dict[str, float]:
        return durations_from_marks(self.marks)

    def stage_peaks(self) -> dict[str, dict[str, int | None]]:
        return peaks_from_marks(self._sampler.samples, STAGE_SPANS, self.marks)

    def stop(self) -> None:
        """Join the sampler thread, which otherwise polls for the life of the process."""
        self._sampler.stop()
