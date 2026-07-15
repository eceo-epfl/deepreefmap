"""Manifest-derived inputs for the cached-run ortho rebuild.

The rebuild must reproduce what the pipeline published rather than fall back on
the ortho builder's own defaults, which differ from the run's settings.
"""

from __future__ import annotations

from deepreefmap.gui.runs.loading import _manifest_grid_bins, _manifest_transect_crop


def test_grid_bins_come_from_the_manifest() -> None:
    assert _manifest_grid_bins({"grid_bins": 6000}) == 6000


def test_grid_bins_default_when_the_manifest_predates_the_param() -> None:
    assert _manifest_grid_bins({}) == 2000
    assert _manifest_grid_bins({"grid_bins": None}) == 2000
    assert _manifest_grid_bins({"grid_bins": 0}) == 2000


def test_applied_transect_crop_comes_from_the_manifest() -> None:
    crop = _manifest_transect_crop(
        {"transect": {"length": 5.0, "crop_width": 1.5, "applied": True}}
    )
    assert crop is not None
    assert (crop.transect_length_m, crop.crop_width_m) == (5.0, 1.5)


def test_unapplied_or_absent_transect_crop_is_none() -> None:
    assert _manifest_transect_crop({}) is None
    assert (
        _manifest_transect_crop({"transect": {"length": None, "crop_width": None, "applied": False}})
        is None
    )
    assert _manifest_transect_crop({"transect": {"applied": True}}) is None
