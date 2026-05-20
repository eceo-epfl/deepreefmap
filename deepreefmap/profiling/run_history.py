"""Per-machine timing profile used to seed remaining-time estimates."""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from deepreefmap.paths import run_timings_path as timings_path
from deepreefmap.profiling.eta import STAGES

logger = logging.getLogger(__name__)

# One cold-cache or thermally throttled run should not skew the profile, so we
# keep a short rolling window and fit with the median rather than the mean.
_MAX_RUNS_PER_KEY = 10

# Stamped on each entry so a future schema change can tell old (peak-less) runs
# from new ones. Bump when the entry shape changes incompatibly.
_ENTRY_VERSION = 1


def history_key(mapping_backend: str, seg_model: str, proc_w: int, proc_h: int, fps: int) -> str:
    """Profile key grouping runs comparable enough to seed each other's estimate."""
    # fps belongs in the key because it changes the memory regime: 5fps can thrash
    # RAM where 3fps does not, which changes the per-frame time. Other params stay
    # out so the short rolling history is not fragmented into singletons.
    return f"{mapping_backend}|{seg_model}|{proc_w}x{proc_h}|{fps}fps"


def load_expected_points(key: str, path: Path | None = None) -> int | None:
    """Median final point count over stored runs for `key`, or None if unseen."""
    points = [int(r["points"]) for r in _load_all(path or timings_path()).get(key, []) if r.get("points")]
    return int(statistics.median(points)) if points else None


def _strip_fps(key: str) -> str:
    """Drop the trailing `|Nfps` segment so keys group by resolution, not fps."""
    return key.rsplit("|", 1)[0] if key.endswith("fps") else key


def load_expected_peaks(key: str, path: Path | None = None) -> dict | None:
    """Worst recent peak as `{ram_bytes, vram_bytes, frames}`, or None if unseen."""
    # Worst run, not the median: this is a crash predictor, and the high-water run
    # says what happens on a busier machine. Peaks pool across every fps at this
    # backend/model/resolution (unlike the ETA priors, keyed per fps) because peak
    # memory tracks the frame count the caller scales by.
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
    worst_committed = 0
    worst_frames = 0
    vrams: list[int] = []
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
        committed = max(per_stage)
        vram = [s["vram_bytes"] for s in stages if s.get("vram_bytes")]
        if vram:
            vrams.append(max(vram))
        if committed > worst_committed:
            worst_committed = committed
            worst_frames = int(run["frames"])
    if not worst_committed:
        return None
    return {
        "ram_bytes": worst_committed,
        "vram_bytes": max(vrams) if vrams else None,
        "frames": worst_frames,
    }


def summarise_recorded_runs(path: Path | None = None) -> list[dict]:
    """One row per recorded run that captured peaks, for the System tab."""
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
            durations = entry.get("stage_durations") or {}
            # Sum of the timed stages = wall-clock cost of the run, the figure the
            # System tab reports and normalises per frame across configs.
            run_seconds = sum(float(v) for v in durations.values()) if durations else None
            rows.append(
                {
                    "key": key,
                    "params": entry.get("params") or {},
                    "frames": entry.get("frames"),
                    "points": entry.get("points"),
                    "run_seconds": run_seconds,
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
    # Newest-first within each key only. Entries carry no timestamp, so runs under
    # different keys cannot be ordered against each other.
    rows.reverse()
    return rows


def group_recorded_runs(path: Path | None = None) -> list[dict]:
    """Collapse repeat runs of the same config into one median-averaged entry.

    The typical-cost view for the System tab. The pre-run check instead reasons from
    the worst recent run (load_expected_peaks), which is the crash predictor.
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
        # The signature carries no machine identity, so the newest member's totals
        # stand in for the group: a re-run after a RAM upgrade reports the new total.
        rep = members[0]
        rams = [m["peak_ram_bytes"] for m in members if m["peak_ram_bytes"]]
        swaps = [m["peak_swap_bytes"] for m in members if m.get("swap_recorded")]
        vrams = [m["peak_vram_bytes"] for m in members if m["peak_vram_bytes"]]
        secs = [m["run_seconds"] for m in members if m.get("run_seconds")]
        # A group is one exact config (fps and frame count are both in the signature),
        # so this time is never pooled across fps, whose memory regime differs.
        run_seconds = int(statistics.median(secs)) if secs else None
        frames = rep["frames"]
        grouped.append(
            {
                "params": rep["params"],
                "frames": frames,
                "count": len(members),
                "run_seconds": run_seconds,
                # A per-frame throughput hint for eyeballing between cards. Rough
                # only: fps and scene complexity change per-frame cost, so it is not
                # an apples-to-apples figure across different fps.
                "seconds_per_frame": (run_seconds / frames) if run_seconds and frames else None,
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


def distinct_model_combinations(rows: list[dict]) -> list[tuple[str, str]]:
    """Unique (mapping_backend, segmentation_model) pairs feeding the System tab filter."""
    seen: dict[tuple[str, str], None] = {}
    for row in rows:
        p = row["params"]
        seen.setdefault((p.get("mapping_backend"), p.get("segmentation_model")), None)
    return list(seen)


def _load_all(path: Path) -> dict[str, list[dict]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def load_priors(key: str, path: Path | None = None) -> dict[str, float]:
    """Seconds-per-driver-unit per stage, median over stored runs for `key`."""
    runs = _load_all(path or timings_path()).get(key, [])
    if not runs:
        return {}
    from deepreefmap.profiling.eta import driver_denominator

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
    """Append one finished run to the profile, capped to the rolling window."""
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
