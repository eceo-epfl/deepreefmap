"""The unified total bar advances through mapping's inference/align/save steps."""

from __future__ import annotations

import pytest

from deepreefmap.gui.runs.progress import (
    _MAPPING_SUBPHASE_SPANS,
    _RECON_PHASES,
    _STAGE_MESSAGE_TO_PHASE,
    ProgressModel,
)


def test_mapping_substeps_are_ordered_phases() -> None:
    keys = [k for k, _ in _RECON_PHASES]
    assert keys.index("mapping") < keys.index("mapping_align") < keys.index("mapping_save")
    assert keys.index("mapping_save") < keys.index("outputs")


def test_backend_messages_route_to_the_new_phases() -> None:
    assert _STAGE_MESSAGE_TO_PHASE["Aligning poses to world frame"] == "mapping_align"
    assert _STAGE_MESSAGE_TO_PHASE["Saving depth + points for resume"] == "mapping_save"
    assert _STAGE_MESSAGE_TO_PHASE["Mapping complete"] == "mapping_save"


def test_total_bar_keeps_moving_through_align_and_save() -> None:
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


def test_mapping_subphase_spans_tile_zero_to_one_in_order() -> None:
    spans = _MAPPING_SUBPHASE_SPANS
    assert spans["mapping"][0] == 0.0
    assert spans["mapping_save"][1] == pytest.approx(1.0)
    # Contiguous and non-overlapping, inference first.
    assert spans["mapping"][1] == spans["mapping_align"][0]
    assert spans["mapping_align"][1] == spans["mapping_save"][0]
    # Inference carries the largest slice (weight 15 vs 8 vs 2).
    width = {k: hi - lo for k, (lo, hi) in spans.items()}
    assert width["mapping"] > width["mapping_align"] > width["mapping_save"]


def _combined_detail_pct(phase_key: str, current: int, total: int, prev: float) -> float:
    # Mirror of the mixin's detail-bar formula: one monotonic 0-100 fill sliced by
    # the span weights, held on indeterminate (total<=0) sub-steps.
    lo, hi = _MAPPING_SUBPHASE_SPANS[phase_key]
    within = min(1.0, current / total) if total > 0 else 0.0
    return max(prev, 100.0 * (lo + (hi - lo) * within))


def test_mapping_detail_bar_never_regresses_across_substeps() -> None:
    # A real run's reported sequence: inference windows, the indeterminate GPU
    # transfer, the re-anchor point-blocks, the indeterminate save, then complete.
    reports = [
        ("mapping", 0, 8),
        ("mapping", 4, 8),
        ("mapping", 8, 8),
        ("mapping", 0, 0),          # GPU transfer, indeterminate
        ("mapping_align", 80, 160),
        ("mapping_align", 160, 160),
        ("mapping_save", 0, 0),     # resume save, indeterminate
        ("mapping_save", 8, 8),     # mapping complete
    ]
    prev = 0.0
    values = []
    for phase, cur, tot in reports:
        prev = _combined_detail_pct(phase, cur, tot, prev)
        values.append(prev)
    assert values == sorted(values)      # never snaps back
    assert values[3] == values[2]        # transfer holds at inference's end
    assert round(values[-1]) == 100      # complete fills the whole mapping bar
