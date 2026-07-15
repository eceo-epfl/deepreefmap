"""Benthic cover and ortho exports on a loaded run.

Scenario: loading a cached run rebuilds an ortho grid for the preview, from a
viewer cloud that is distance-capped and pre-TSDF. The displayed cover then
self-corrects from benthic_cover.json, so a wrong export is invisible on screen.

Expected behaviour: exports carry the run's published artefacts, and only a
user-driven crop is allowed to produce different numbers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PySide6.QtWidgets import QFileDialog

from deepreefmap.config.classes import load_classes
from deepreefmap.io.exports import save_ortho_grid
from deepreefmap.pointcloud.grid_ortho import OrthoGrid

PUBLISHED_COVER = {
    "classes": {
        "1": {"name": "reef", "count": 900.0, "fraction": 0.9},
        "5": {"name": "sand", "count": 100.0, "fraction": 0.1},
    },
    "denominator": 1000.0,
}


def _grid_of(labels: np.ndarray) -> OrthoGrid:
    return OrthoGrid(
        rgb=np.zeros((*labels.shape, 3), dtype=np.uint8),
        labels=labels,
        height=np.zeros(labels.shape, dtype=np.float32),
        counts=np.ones(labels.shape, dtype=np.int32),
        frame_index=np.zeros(labels.shape, dtype=np.int32),
        cell_size=0.01,
    )


def _load_published_run(window, tmp_path: Path) -> Path:
    """Leave the window in the state a cached-run load leaves it in.

    The display grid is all sand while the published report is 90% reef, which
    stands in for the recompute disagreeing with the pipeline on a real run.
    """
    classes_path = tmp_path / "classes.yaml"
    classes_path.write_text(
        yaml.safe_dump(
            {
                "classes": [
                    {"id": 1, "name": "reef", "color": [10, 20, 30], "roles": []},
                    {"id": 5, "name": "sand", "color": [200, 200, 100], "roles": []},
                ]
            }
        )
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "benthic_cover.json").write_text(json.dumps(PUBLISHED_COVER))

    display_grid = _grid_of(np.full((10, 10), 5, dtype=np.int32))
    window._set_ortho_sources(None, display_grid, load_classes(classes_path))
    window._show_results(str(run_dir))
    return run_dir


def _fine_csv_fractions(path: Path) -> dict[str, float]:
    with path.open() as fh:
        return {row["name"]: float(row["fraction"]) for row in csv.DictReader(fh)}


def test_cover_export_matches_published_report(window, tmp_path, monkeypatch) -> None:
    _load_published_run(window, tmp_path)
    dest = tmp_path / "export"
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(dest))
    )

    window._on_export_cover_csv()

    assert _fine_csv_fractions(dest / "benthic_cover_fine.csv") == {"reef": 0.9, "sand": 0.1}


def test_ortho_npz_export_copies_published_grid(window, tmp_path, monkeypatch) -> None:
    run_dir = _load_published_run(window, tmp_path)
    published_labels = np.full((4, 4), 1, dtype=np.int32)
    save_ortho_grid(run_dir / "ortho.npz", _grid_of(published_labels))
    target = tmp_path / "exported.npz"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), ""))
    )

    window._on_export_ortho_npz()

    assert np.array_equal(np.load(target)["labels"], published_labels)


def test_user_crop_overrides_the_published_cover(window, tmp_path) -> None:
    _load_published_run(window, tmp_path)

    window._results_transect_length.setValue(1.0)
    window._results_crop_width.setValue(0.5)

    cover = window._current_cover_dict()
    assert cover is not None
    assert set(cover["classes"]) == {"5"}


def test_zeroed_crop_returns_to_the_published_cover(window, tmp_path) -> None:
    _load_published_run(window, tmp_path)
    window._results_transect_length.setValue(1.0)
    window._results_crop_width.setValue(0.5)

    window._results_crop_width.setValue(0.0)

    assert window._current_cover_dict() == PUBLISHED_COVER


def test_cover_falls_back_to_a_recompute_when_the_run_published_none(window, tmp_path) -> None:
    _load_published_run(window, tmp_path)
    (tmp_path / "run" / "benthic_cover.json").unlink()

    cover = window._current_cover_dict()
    assert cover is not None
    assert set(cover["classes"]) == {"5"}
