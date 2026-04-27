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

    for path in video_paths:
        meta = iio.immeta(path)
        src_fps = float(meta.get("fps", target_fps))
        stride = max(1, int(round(src_fps / max(1, target_fps))))
        local_count = 0

        for local_idx, frame in enumerate(iio.imiter(path)):
            local_count = local_idx + 1
            if local_idx % stride != 0:
                continue
            t = cumulative_time + local_idx / src_fps
            if begin_s is not None and t < begin_s:
                continue
            if end_s is not None and t >= end_s:
                break
            yield global_idx, frame
            global_idx += 1

        cumulative_time += local_count / src_fps
