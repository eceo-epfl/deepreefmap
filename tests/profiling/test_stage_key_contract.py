"""`profiling/eta.py` learns and projects per-stage timings keyed by stage name;
`pipeline/instrumentation.py` is what emits those durations from a run's timing
marks. The two key sets are written out independently and drift doesn't raise: a
stage the estimator knows but instrumentation never times just has no history,
so the ETA quietly degrades to weight-based projection. This pins them together.
"""

from __future__ import annotations

from deepreefmap.pipeline.instrumentation import STAGE_SPANS
from deepreefmap.profiling.eta import STAGES


def test_timed_spans_match_eta_stages() -> None:
    timed = {stage for _, _, stage in STAGE_SPANS}
    assert timed == {spec.key for spec in STAGES}, (
        "instrumentation.py STAGE_SPANS and eta.py STAGES disagree; a stage only "
        "one side names loses its history and falls back to a weight-based guess"
    )
