"""The unified total bar advances through mapping's inference/align/save steps."""

from __future__ import annotations

from deepreefmap.gui.progress import (
    _RECON_PHASES,
    _STAGE_MESSAGE_TO_PHASE,
    ProgressModel,
)


def test_mapping_substeps_are_ordered_phases():
    keys = [k for k, _ in _RECON_PHASES]
    assert keys.index("mapping") < keys.index("mapping_align") < keys.index("mapping_save")
    assert keys.index("mapping_save") < keys.index("outputs")


def test_backend_messages_route_to_the_new_phases():
    assert _STAGE_MESSAGE_TO_PHASE["Aligning poses to world frame"] == "mapping_align"
    assert _STAGE_MESSAGE_TO_PHASE["Saving depth + points for resume"] == "mapping_save"
    assert _STAGE_MESSAGE_TO_PHASE["Mapping complete"] == "mapping_save"


def test_total_bar_keeps_moving_through_align_and_save():
    model = ProgressModel(_RECON_PHASES)
    model.update("preprocess", 100, 100)
    after_inference = model.update("mapping", 100, 100)
    # The align step drives its own weight, so the bar advances past inference.
    during_align = model.update("mapping_align", 60, 100)
    assert during_align > after_inference
    after_align = model.update("mapping_align", 100, 100)
    # The save is indeterminate (0/0), so the bar holds until the next phase, but
    # never regresses.
    during_save = model.update("mapping_save", 0, 0)
    assert during_save >= after_align
    # Mapping complete banks the save weight; the cloud starting promotes it.
    done_save = model.update("mapping_save", 100, 100)
    assert done_save >= after_align
    after_cloud = model.update("outputs", 1, 10)
    assert after_cloud >= done_save
