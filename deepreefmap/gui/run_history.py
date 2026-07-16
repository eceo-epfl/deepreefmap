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

# Stamped on each entry so a future schema change can tell old (peak-less) runs
# from new ones. Bump when the entry shape changes incompatibly.
_ENTRY_VERSION = 1


def timings_path() -> Path:
    """Location of the local timing profile, overridable for tests."""
    override = os.environ.get("DEEPREEFMAP_RUN_TIMINGS")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("deepreefmap", appauthor=False)) / "run_timings.json"


def history_key(mapping_backend: str, seg_model: str, proc_w: int, proc_h: int, fps: int) -> str:
    """Profile key grouping comparable runs.

    Backend, models, resolution and fps dominate the hardware-bound per-frame cost
    and the memory regime (5fps can thrash RAM where 3fps does not, changing the
    per-frame time), so a prediction is only ever seeded from like-for-like runs.
    Other params ride along as inspectable metadata on each run, not in the key, so
    the short rolling history is not fragmented into singletons.
    """
    return f"{mapping_backend}|{seg_model}|{proc_w}x{proc_h}|{fps}fps"


def load_expected_points(key: str, path: Path | None = None) -> int | None:
    """Median final point count over stored runs for `key`, or None if unseen.

    Point-driven stages (cloud, ortho, saves) scale with the final cloud size,
    unknown until mapping ends. Same key means a comparable cloud, so past runs'
    point count is the best provisional N; set_points supersedes it with the real
    N once mapping produces the cloud.
    """
    points = [int(r["points"]) for r in _load_all(path or timings_path()).get(key, []) if r.get("points")]
    return int(statistics.median(points)) if points else None


def _strip_fps(key: str) -> str:
    """Drop the trailing `|Nfps` segment so keys group by resolution, not fps."""
    return key.rsplit("|", 1)[0] if key.endswith("fps") else key


def load_expected_peaks(key: str, path: Path | None = None) -> dict | None:
    """Median measured peak RAM/VRAM (over all stages) and frame count for `key`.

    Returned as `{ram_bytes, vram_bytes|None, frames}` so the pre-run memory check
    can scale a real peak by this run's frame count. None when no comparable run
    has recorded peaks yet, so the caller falls back to the analytic estimate.

    Peaks are pooled across every recorded fps at this backend/model/resolution,
    not just the exact key: peak memory tracks the frame count, which the estimate
    already scales, so a 3fps measurement legitimately informs a 5fps run. (The ETA
    profile, by contrast, stays keyed per fps because wall-clock time per frame
    changes once a run starts thrashing swap.)
    """
    prefix = _strip_fps(key)
    runs = [
        r
        for k, entries in _load_all(path or timings_path()).items()
        if _strip_fps(k) == prefix
        for r in entries
        if r.get("stage_peaks") and r.get("frames")
    ]
    if not runs:
        return None
    committed: list[int] = []
    vrams: list[int] = []
    frames: list[int] = []
    for run in runs:
        stages = run["stage_peaks"].values()
        # Peak committed memory per stage = RAM plus the swap it spilled into, then
        # the worst stage. A thrashing run pins RAM near 100% and shows its real
        # demand as swap, so RAM alone would understate the true peak.
        per_stage = [
            s["ram_bytes"] + (s.get("swap_bytes") or 0) for s in stages if s.get("ram_bytes")
        ]
        if not per_stage:
            continue
        committed.append(max(per_stage))
        vram = [s["vram_bytes"] for s in stages if s.get("vram_bytes")]
        if vram:
            vrams.append(max(vram))
        frames.append(int(run["frames"]))
    if not committed:
        return None
    return {
        "ram_bytes": int(statistics.median(committed)),
        "vram_bytes": int(statistics.median(vrams)) if vrams else None,
        "frames": int(statistics.median(frames)),
    }


def summarise_recorded_runs(path: Path | None = None) -> list[dict]:
    """One row per recorded run that captured peaks, newest first.

    Each row carries the run's config, frame/point counts, its absolute peak RAM
    and VRAM, and the machine totals it ran on, so the System tab can show what the
    run actually cost and how close to a crash it came. Runs without peaks (old
    entries) are skipped: there is nothing memory-wise to report for them.
    """
    rows: list[dict] = []
    for key, entries in _load_all(path or timings_path()).items():
        for entry in entries:
            peaks = entry.get("stage_peaks")
            if not peaks:
                continue
            rams = [s["ram_bytes"] for s in peaks.values() if s.get("ram_bytes")]
            swaps = [s["swap_bytes"] for s in peaks.values() if s.get("swap_bytes")]
            vrams = [s["vram_bytes"] for s in peaks.values() if s.get("vram_bytes")]
            profile = entry.get("system_profile") or {}
            gpu = profile.get("gpu") or {}
            rows.append(
                {
                    "key": key,
                    "params": entry.get("params") or {},
                    "frames": entry.get("frames"),
                    "points": entry.get("points"),
                    "peak_ram_bytes": max(rams) if rams else None,
                    "peak_swap_bytes": max(swaps) if swaps else 0,
                    # Distinguish "measured 0 swap" from "predates swap capture", so
                    # the UI can show "not recorded" rather than a misleading 0%.
                    "swap_recorded": any("swap_bytes" in s for s in peaks.values()),
                    "peak_vram_bytes": max(vrams) if vrams else None,
                    "total_ram_bytes": profile.get("total_ram_bytes"),
                    "total_swap_bytes": profile.get("total_swap_bytes") or 0,
                    "gpu_name": gpu.get("name"),
                    "gpu_total_vram_bytes": gpu.get("total_vram_bytes"),
                }
            )
    rows.reverse()  # newest run first
    return rows


def group_recorded_runs(path: Path | None = None) -> list[dict]:
    """Collapse repeat runs of the same config into one median-averaged entry.

    Runs are grouped by the workload signature (mapping backend, segmentation
    model, resolution, fps and frame count); within a group the peak RAM/swap/VRAM
    are the median across runs, and ``count`` records how many were folded in. The
    median matches load_expected_peaks, so the numbers shown are the ones the
    pre-run check reasons from. Groups stay newest-first.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in summarise_recorded_runs(path):
        p = row["params"]
        signature = (
            p.get("mapping_backend"), p.get("segmentation_model"),
            p.get("processing_width"), p.get("processing_height"),
            p.get("fps"), row["frames"],
        )
        groups.setdefault(signature, []).append(row)

    grouped: list[dict] = []
    for members in groups.values():
        rep = members[0]  # newest in the group; machine totals are identical
        rams = [m["peak_ram_bytes"] for m in members if m["peak_ram_bytes"]]
        swaps = [m["peak_swap_bytes"] for m in members if m.get("swap_recorded")]
        vrams = [m["peak_vram_bytes"] for m in members if m["peak_vram_bytes"]]
        grouped.append(
            {
                "params": rep["params"],
                "frames": rep["frames"],
                "count": len(members),
                "peak_ram_bytes": int(statistics.median(rams)) if rams else None,
                "peak_swap_bytes": int(statistics.median(swaps)) if swaps else 0,
                "swap_recorded": any(m.get("swap_recorded") for m in members),
                "peak_vram_bytes": int(statistics.median(vrams)) if vrams else None,
                "total_ram_bytes": rep["total_ram_bytes"],
                "total_swap_bytes": rep["total_swap_bytes"],
                "gpu_name": rep["gpu_name"],
                "gpu_total_vram_bytes": rep["gpu_total_vram_bytes"],
            }
        )
    return grouped


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
    params: dict | None = None,
    stage_peaks: dict | None = None,
    system_profile: dict | None = None,
    path: Path | None = None,
) -> None:
    """Append one finished run to the profile, capped to the rolling window.

    `params` records the full run configuration (fps, models, resolution, mapping
    options); `stage_peaks` the measured per-stage peak RAM/VRAM; `system_profile`
    the machine it ran on. Peaks + profile feed the pre-run memory check.
    """
    target = path or timings_path()
    all_runs = _load_all(target)
    entry: dict = {"version": _ENTRY_VERSION, "stage_durations": stage_durations, "frames": frames, "points": points}
    if params:
        entry["params"] = params
    if stage_peaks:
        entry["stage_peaks"] = stage_peaks
    if system_profile:
        entry["system_profile"] = system_profile
    all_runs.setdefault(key, []).append(entry)
    all_runs[key] = all_runs[key][-_MAX_RUNS_PER_KEY:]
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(all_runs, indent=2))
    except OSError:
        logger.warning("Could not write run timing profile to %s", target, exc_info=True)
