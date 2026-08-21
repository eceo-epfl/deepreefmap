"""Resource ownership across run_reconstruction's failure paths.

A failed run must leave no threads behind, close any viser server it built
itself, and leave an injected Qt viewer open, because the GUI keeps one window
across many runs.
"""

from __future__ import annotations

import inspect
import threading
from pathlib import Path

import pytest

from deepreefmap.pipeline.orchestrator import run_reconstruction


class _RecordingViewer:
    def __init__(self) -> None:
        self.closed = 0
        self.failures: list[tuple[str, str]] = []

    def start_run(self, run_label: str, output_dir: str) -> None:
        pass

    def set_stage(self, stage: str, status: str, message: str | None = None) -> None:
        pass

    def update_progress(self, stage, current, total=None, message=None, frame_index=None) -> None:
        pass

    def set_data(self, **kwargs: object) -> None:
        pass

    def mark_outputs_ready(self, output_dir: str, output_files: list[str]) -> None:
        pass

    def fail_run(self, stage: str, error_message: str) -> None:
        self.failures.append((stage, error_message))

    def close(self) -> None:
        self.closed += 1

    def wait_forever(self) -> None:
        pass


def _run_expecting_failure(
    tmp_path: Path, expected: type[Exception], **overrides: object
) -> None:
    kwargs: dict[str, object] = {
        "video_paths": [str(tmp_path / "missing.mp4")],
        "fps": 2,
        "segmentation_name": "segformer-b2",
        "mapping_name": "scsfmlearner",
        "camera_profile_name": "gopro",
        "output_dir": tmp_path / "out",
        "transect_length": None,
        "transect_crop_width": None,
    }
    kwargs.update(overrides)
    with pytest.raises(expected):
        run_reconstruction(**kwargs)  # type: ignore[arg-type]


def _run_expecting_deep_failure(tmp_path: Path) -> None:
    # Real clip, unknown backend: fails after frame preparation has run.
    _run_expecting_failure(
        tmp_path, ValueError,
        video_paths=[str(Path(__file__).parent / "data" / "reef_clip.mp4")],
        camera_profile_name="gopro_hero_10", mapping_name="no-such-backend",
        skip_segmentation=True, processing_width=344, processing_height=192, end_s=1.0,
    )


def test_failed_run_leaves_no_new_threads(tmp_path: Path) -> None:
    """A failure after frame preparation leaves no threads behind.

    The first run warms process-wide singletons (tqdm's monitor thread), which
    live for the interpreter, not the run.
    """
    _run_expecting_deep_failure(tmp_path)
    before = {t.ident for t in threading.enumerate()}

    for _ in range(2):
        _run_expecting_deep_failure(tmp_path)

    leaked = [t.name for t in threading.enumerate() if t.ident not in before]
    assert leaked == []


def test_failed_run_closes_the_viser_server_it_built(tmp_path: Path, monkeypatch) -> None:
    built = _RecordingViewer()
    monkeypatch.setattr(
        "deepreefmap.visualization.simple_viser_app.SimpleGeometryViserApp",
        lambda **kwargs: built,
    )

    _run_expecting_failure(
        tmp_path,
        FileNotFoundError,
        enable_viser=True,
        skip_segmentation=True,
        camera_profile_name="no-such-profile",
    )

    assert built.closed == 1
    assert [stage for stage, _ in built.failures] == ["startup"]


def test_failed_run_leaves_an_injected_viewer_open(tmp_path: Path) -> None:
    """The Qt GUI owns its viewer and reuses it for the next run."""
    injected = _RecordingViewer()

    _run_expecting_failure(
        tmp_path,
        FileNotFoundError,
        viewer=injected,
        enable_viser=True,
        keep_viser_open=False,
        camera_profile_name="no-such-profile",
    )

    assert injected.closed == 0
    assert [stage for stage, _ in injected.failures] == ["startup"]


def test_enable_viser_holds_its_v1_signature_position() -> None:
    params = list(inspect.signature(run_reconstruction).parameters.values())

    assert params[8].name == "enable_viser"
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params[:9])
    assert all(p.kind is p.KEYWORD_ONLY for p in params[9:])
