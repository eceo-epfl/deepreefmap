from pathlib import Path

import numpy as np

from deepreefmap.telemetry import gopro


def test_extract_gravity_vectors_for_video_selection_matches_sampled_frames(monkeypatch):
    timestamps = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    streams = {
        Path("a.mp4"): (
            np.array(
                [[1, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]],
                dtype=np.float32,
            ),
            timestamps,
        ),
        Path("b.mp4"): (
            np.array(
                [[0, 0, 5], [0, 6, 0], [7, 0, 0], [8, 0, 0]],
                dtype=np.float32,
            ),
            timestamps,
        ),
    }

    monkeypatch.setattr(gopro.iio, "immeta", lambda path: {"fps": 2, "nframes": 4})
    monkeypatch.setattr(gopro, "_extract_gravity_stream", lambda path: streams[path])

    gravity = gopro.extract_gravity_vectors_for_video_selection(
        [Path("a.mp4"), Path("b.mp4")],
        target_fps=1,
        begin_s=1.0,
        end_s=3.0,
    )

    assert gravity is not None
    assert np.allclose(gravity, [[0, 1, 0], [0, 0, 1]])


def test_extract_gravity_vectors_for_video_selection_returns_none_when_stream_missing(monkeypatch):
    monkeypatch.setattr(gopro.iio, "immeta", lambda path: {"fps": 2, "nframes": 4})
    monkeypatch.setattr(gopro, "_extract_gravity_stream", lambda path: None)

    gravity = gopro.extract_gravity_vectors_for_video_selection([Path("a.mp4")], target_fps=1)

    assert gravity is None


def test_extract_gravity_vectors_for_video_selection_skips_unselected_clips(monkeypatch):
    def stream_for(path: Path):
        if path == Path("a.mp4"):
            return None
        return (
            np.array([[0, 0, 1], [0, 2, 0], [3, 0, 0], [0, 0, 4]], dtype=np.float32),
            np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
        )

    monkeypatch.setattr(gopro.iio, "immeta", lambda path: {"fps": 2, "nframes": 4})
    monkeypatch.setattr(gopro, "_extract_gravity_stream", stream_for)

    gravity = gopro.extract_gravity_vectors_for_video_selection(
        [Path("a.mp4"), Path("b.mp4")],
        target_fps=1,
        begin_s=2.0,
        end_s=3.0,
    )

    assert gravity is not None
    assert np.allclose(gravity, [[0, 0, 1]])
