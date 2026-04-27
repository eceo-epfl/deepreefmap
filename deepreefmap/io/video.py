from pathlib import Path
from typing import Iterator

import imageio.v3 as iio
import numpy as np


def iter_video_frames(video_paths: list[Path], target_fps: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield RGB frames from multiple videos in order at approximately target_fps."""
    global_idx = 0
    for path in video_paths:
        meta = iio.immeta(path)
        src_fps = float(meta.get("fps", target_fps))
        stride = max(1, int(round(src_fps / max(1, target_fps))))
        for local_idx, frame in enumerate(iio.imiter(path)):
            if local_idx % stride != 0:
                continue
            yield global_idx, frame
            global_idx += 1
