from __future__ import annotations

from pathlib import Path
from typing import Iterator

import imageio.v3 as iio
import numpy as np


def iter_video_frames(
    video_paths: list[Path],
    target_fps: int,
    begin_s: float | None = None,
    end_s: float | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (global_index, RGB_frame) from one or more videos at *target_fps*.

    *begin_s* / *end_s* are wall-clock offsets into the **concatenated** video
    stream.  Frames outside [begin_s, end_s) are skipped (but still decoded,
    because imageio streams sequentially).
    """
    global_idx = 0
    cumulative_time = 0.0
    next_sample_time = _first_sample_time(begin_s, target_fps)
    interval_end = float("inf") if end_s is None else max(0.0, float(end_s))

    for path in video_paths:
        meta = iio.immeta(path)
        src_fps = float(meta.get("fps", target_fps))
        src_fps = src_fps if src_fps > 0 else float(max(1, target_fps))
        local_count = 0

        for local_idx, frame in enumerate(iio.imiter(path)):
            local_count = local_idx + 1
            if next_sample_time >= interval_end:
                break
            t = cumulative_time + local_idx / src_fps
            if t + 1e-9 < next_sample_time:
                continue
            if t >= interval_end:
                break
            yield global_idx, frame
            global_idx += 1
            next_sample_time += 1.0 / max(1, target_fps)
            while next_sample_time <= t + 1e-9:
                next_sample_time += 1.0 / max(1, target_fps)

        cumulative_time += local_count / src_fps
        if next_sample_time >= interval_end:
            break


def selected_local_indices_for_clip(
    *,
    nframes: int,
    src_fps: float,
    target_fps: int,
    cumulative_time: float,
    next_sample_time: float,
    end_s: float | None,
) -> tuple[list[int], float]:
    """Return local frame indices selected by the timestamp sampler for one clip."""
    src_fps = src_fps if src_fps > 0 else float(max(1, target_fps))
    interval_end = float("inf") if end_s is None else max(0.0, float(end_s))
    selected: list[int] = []
    sample_time = next_sample_time
    for local_idx in range(max(0, int(nframes))):
        if sample_time >= interval_end:
            break
        t = cumulative_time + local_idx / src_fps
        if t + 1e-9 < sample_time:
            continue
        if t >= interval_end:
            break
        selected.append(local_idx)
        sample_time += 1.0 / max(1, target_fps)
        while sample_time <= t + 1e-9:
            sample_time += 1.0 / max(1, target_fps)
    return selected, sample_time


def _first_sample_time(begin_s: float | None, target_fps: int) -> float:
    start = 0.0 if begin_s is None else max(0.0, float(begin_s))
    fps = float(max(1, target_fps))
    return float(np.ceil(start * fps - 1e-9) / fps)
