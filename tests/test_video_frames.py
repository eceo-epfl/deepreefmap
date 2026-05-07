from pathlib import Path

import numpy as np

from deepreefmap.io import video
from deepreefmap.pipeline import orchestrator


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


def test_estimated_frame_count_matches_iterator_for_end_exclusive_window(monkeypatch):
    path = Path("clip.mp4")
    frames = [np.full((1, 1, 3), i % 256, dtype=np.uint8) for i in range(300)]

    monkeypatch.setattr(video.iio, "immeta", lambda p: {"fps": 30, "nframes": len(frames)})
    monkeypatch.setattr(video.iio, "imiter", lambda p: iter(frames))
    monkeypatch.setattr(orchestrator.iio, "immeta", lambda p: {"fps": 30, "nframes": len(frames)})

    result = list(video.iter_video_frames([path], target_fps=10, begin_s=1.0, end_s=2.0))
    estimated = orchestrator._estimate_selected_frame_count([path], fps=10, begin_s=1.0, end_s=2.0)

    assert estimated == 10
    assert len(result) == estimated
    assert [int(frame[0, 0, 0]) for _, frame in result] == list(range(30, 60, 3))


def test_estimated_frame_count_matches_iterator_for_non_integer_source_fps(monkeypatch):
    path = Path("clip.mp4")
    frames = [np.full((1, 1, 3), i, dtype=np.uint8) for i in range(120)]

    monkeypatch.setattr(video.iio, "immeta", lambda p: {"fps": 29.97, "nframes": len(frames)})
    monkeypatch.setattr(video.iio, "imiter", lambda p: iter(frames))
    monkeypatch.setattr(orchestrator.iio, "immeta", lambda p: {"fps": 29.97, "nframes": len(frames)})

    result = list(video.iter_video_frames([path], target_fps=10, begin_s=0.0, end_s=1.0))
    estimated = orchestrator._estimate_selected_frame_count([path], fps=10, begin_s=0.0, end_s=1.0)

    assert len(result) == estimated
    assert len(result) == 10


def test_timestamp_sampler_honors_target_fps_when_source_is_not_multiple(monkeypatch):
    path = Path("clip.mp4")
    frames = [np.full((1, 1, 3), i % 256, dtype=np.uint8) for i in range(750)]

    monkeypatch.setattr(video.iio, "immeta", lambda p: {"fps": 25, "nframes": len(frames)})
    monkeypatch.setattr(video.iio, "imiter", lambda p: iter(frames))
    monkeypatch.setattr(orchestrator.iio, "immeta", lambda p: {"fps": 25, "nframes": len(frames)})

    result = list(video.iter_video_frames([path], target_fps=10, begin_s=10.0, end_s=20.0))
    estimated = orchestrator._estimate_selected_frame_count([path], fps=10, begin_s=10.0, end_s=20.0)

    assert estimated == 100
    assert len(result) == estimated
