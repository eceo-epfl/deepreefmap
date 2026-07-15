"""Per-machine timing profile used to seed remaining-time estimates.

Every finished run records how long each stage took and the frame and point
counts that produced it. Next time, those durations seed the estimate for stages
that have not started yet (see gui/eta.py). The file is local, per-machine, and
inspectable: the user can open it to see what the predictions are based on, or
delete it to reset the profile.

Kept dependency-free and Qt-free so it can be tested without a display.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from pathlib import Path

import platformdirs

from deepreefmap.gui.eta import STAGES

logger = logging.getLogger(__name__)

# One cold-cache or thermally throttled run should not skew the profile, so we
# keep a short rolling window and fit with the median rather than the mean.
_MAX_RUNS_PER_KEY = 10


def timings_path() -> Path:
    """Location of the local timing profile, overridable for tests."""
    override = os.environ.get("DEEPREEFMAP_RUN_TIMINGS")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("deepreefmap", appauthor=False)) / "run_timings.json"


def history_key(mapping_backend: str, seg_model: str, proc_w: int, proc_h: int) -> str:
    """Profile key grouping comparable runs.

    Backend and resolution dominate the hardware-bound per-frame cost, so a CPU
    run must not seed a GPU one and a 720p run must not seed a 1080p one. Keying
    on both keeps each seeded prediction drawn only from like-for-like runs.
    """
    return f"{mapping_backend}|{seg_model}|{proc_w}x{proc_h}"


def _load_all(path: Path) -> dict[str, list[dict]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def load_priors(key: str, path: Path | None = None) -> dict[str, float]:
    """Seconds-per-driver-unit per stage, median over stored runs for `key`.

    Absent stages (or a first-ever run) simply return no entry, and the estimator
    falls back to weight-based projection.
    """
    runs = _load_all(path or timings_path()).get(key, [])
    if not runs:
        return {}
    from deepreefmap.gui.eta import driver_denominator

    priors: dict[str, float] = {}
    for spec in STAGES:
        ratios: list[float] = []
        for run in runs:
            duration = run.get("stage_durations", {}).get(spec.key)
            if duration is None:
                continue
            denom = driver_denominator(spec.driver, run.get("frames", 0), run.get("points"))
            if denom and denom > 0:
                ratios.append(duration / denom)
        if ratios:
            priors[spec.key] = statistics.median(ratios)
    return priors


def record_run(
    key: str,
    stage_durations: dict[str, float],
    frames: int,
    points: int | None,
    path: Path | None = None,
) -> None:
    """Append one finished run to the profile, capped to the rolling window."""
    target = path or timings_path()
    all_runs = _load_all(target)
    entry = {"stage_durations": stage_durations, "frames": frames, "points": points}
    all_runs.setdefault(key, []).append(entry)
    all_runs[key] = all_runs[key][-_MAX_RUNS_PER_KEY:]
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(all_runs, indent=2))
    except OSError:
        logger.warning("Could not write run timing profile to %s", target, exc_info=True)
