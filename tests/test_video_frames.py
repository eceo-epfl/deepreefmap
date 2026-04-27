from pathlib import Path

import numpy as np

from deepreefmap.io import video


def test_iter_video_frames_applies_concatenated_time_window(monkeypatch):
    frames = {
        Path("a.mp4"): [np.full((1, 1, 3), i, dtype=np.uint8) for i in range(4)],
        Path("b.mp4"): [np.full((1, 1, 3), i + 4, dtype=np.uint8) for i in range(4)],
    }

    monkeypatch.setattr(video.iio, "immeta", lambda path: {"fps": 2})
    monkeypatch.setattr(video.iio, "imiter", lambda path: iter(frames[path]))

    result = list(video.iter_video_frames([Path("a.mp4"), Path("b.mp4")], target_fps=1, begin_s=1.0, end_s=3.0))

    assert [idx for idx, _ in result] == [0, 1]
    assert [int(frame[0, 0, 0]) for _, frame in result] == [2, 4]
