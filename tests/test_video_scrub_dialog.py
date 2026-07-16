import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest


@pytest.fixture(scope="module")
def qapp():
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory) -> tuple[Path, float]:
    path = tmp_path_factory.mktemp("scrub") / "tiny.mp4"
    frames = np.zeros((20, 48, 64, 3), dtype=np.uint8)
    for i in range(20):
        frames[i, :, :, 0] = i * 12
    iio.imwrite(path, frames, fps=10)
    return path, 2.0


def test_defaults_span_the_full_video(qapp, tiny_video):
    from deepreefmap.gui.video_scrub_dialog import VideoScrubDialog

    path, duration = tiny_video
    dialog = VideoScrubDialog(path, duration)
    begin, end = dialog.time_range()
    assert begin == 0.0
    assert end == duration
    dialog.reject()


def test_end_at_slider_max_returns_exact_duration(qapp, tiny_video):
    # _effective_time_range() only collapses end to "full length" when it
    # matches the probed duration, so the max tick must not round away from it.
    from deepreefmap.gui.video_scrub_dialog import VideoScrubDialog

    path, _ = tiny_video
    duration = 2.0004999
    dialog = VideoScrubDialog(path, duration)
    dialog._end_slider.setValue(dialog._end_slider.maximum())
    assert dialog.time_range()[1] == duration
    dialog.reject()


def test_sliders_map_ticks_to_seconds(qapp, tiny_video):
    from deepreefmap.gui.video_scrub_dialog import VideoScrubDialog

    path, duration = tiny_video
    dialog = VideoScrubDialog(path, duration, begin_s=0.5, end_s=1.5)
    assert dialog.time_range() == (0.5, 1.5)
    dialog._begin_slider.setValue(80)
    assert dialog.time_range()[0] == 0.8
    dialog.reject()


def test_handles_clamp_each_other(qapp, tiny_video):
    from deepreefmap.gui.video_scrub_dialog import VideoScrubDialog

    path, duration = tiny_video
    dialog = VideoScrubDialog(path, duration, begin_s=0.5, end_s=1.0)
    dialog._begin_slider.setValue(150)
    assert dialog._end_slider.value() == 150

    dialog._end_slider.setValue(30)
    assert dialog._begin_slider.value() == 30
    dialog.reject()


def test_preview_paints_a_frame(qapp, tiny_video):
    from deepreefmap.gui.video_scrub_dialog import VideoScrubDialog

    path, duration = tiny_video
    dialog = VideoScrubDialog(path, duration)
    dialog._request_preview(1.0)
    dialog._show_pending_frame()
    pixmap = dialog._preview.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    dialog.reject()


def test_capture_released_on_close(qapp, tiny_video):
    from deepreefmap.gui.video_scrub_dialog import VideoScrubDialog

    path, duration = tiny_video
    dialog = VideoScrubDialog(path, duration)
    assert dialog._cap.isOpened()
    dialog.reject()
    assert not dialog._cap.isOpened()
