"""Default processing resolution is the segmentation model's native size.

DPT heads have no internal resize, so the orchestrator feeds each model its native
resolution. Camera-native is the fallback when segmentation is skipped or unknown.
"""

import pytest

from deepreefmap.pipeline.orchestrator import (
    _default_processing_size,
    _resolve_processing_image_size,
)

_PROFILE = (1920, 1080)  # camera-native (width, height)


def test_default_is_model_native_when_segmenting() -> None:
    assert _default_processing_size("coralscapes-vit-s-dpt", False, _PROFILE) == (688, 384)
    assert _default_processing_size("coralscapes-vit-b-dpt", False, _PROFILE) == (1376, 768)


def test_default_falls_back_to_profile_when_skipping_segmentation() -> None:
    assert _default_processing_size("coralscapes-vit-s-dpt", True, _PROFILE) == _PROFILE


def test_default_falls_back_to_profile_for_unknown_model() -> None:
    assert _default_processing_size("mystery-model", False, _PROFILE) == _PROFILE


def test_explicit_override_wins_over_native_default() -> None:
    native = _default_processing_size("coralscapes-vit-s-dpt", False, _PROFILE)
    assert _resolve_processing_image_size(
        native, processing_width=None, processing_height=None
    ) == (688, 384)
    assert _resolve_processing_image_size(
        native, processing_width=512, processing_height=288
    ) == (512, 288)


def test_one_sided_override_is_rejected() -> None:
    with pytest.raises(ValueError):
        _resolve_processing_image_size((688, 384), processing_width=512, processing_height=None)
