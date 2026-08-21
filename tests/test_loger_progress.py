"""LoGeR progress reporting: window counting and per-block re-anchor progress."""

import numpy as np
from reanchor_reference import single_shot_reanchor_world

from deepreefmap.mapping.loger_backend import _count_windows, _reanchor_to_first_camera


def test_count_windows_matches_pi3_sliding_scheme() -> None:
    # Whole sequence fits in one window → single pass, no countdown.
    assert _count_windows(10, window_size=32, overlap_size=3) == 1
    assert _count_windows(10, window_size=0, overlap_size=3) == 1
    # step = window - overlap = 29; windows start at 0, 29, 58, ... until N.
    assert _count_windows(100, window_size=32, overlap_size=3) == 4
    assert _count_windows(64, window_size=32, overlap_size=3) == 3


def test_reanchor_reports_monotonic_block_progress(monkeypatch) -> None:
    import deepreefmap.mapping.loger_backend as lb

    rng = np.random.default_rng(5)
    n, h, w = 4, 5, 7  # 140 points
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    poses[0, :3, 3] = (1.0, 0.5, -0.25)  # non-trivial transform, not a no-op check
    world = (rng.random((n, h, w, 3), dtype=np.float64).astype(np.float32) * 2.0 - 1.0)
    reference = single_shot_reanchor_world(poses, world)

    calls: list[tuple[int, int, str]] = []
    monkeypatch.setattr(lb, "_REANCHOR_POINT_BLOCK", 40)
    _, rebased_world = _reanchor_to_first_camera(
        poses, world, lambda cur, tot, msg: calls.append((cur, tot, msg))
    )
    # Progress reporting must not perturb the transform.
    assert np.array_equal(rebased_world, reference)
    # 140 points in blocks of 40 -> reports at 40, 80, 120, 140, all against 140.
    assert [c[0] for c in calls] == [40, 80, 120, 140]
    assert all(tot == 140 for _, tot, _ in calls)
    assert all(msg == "Aligning poses to world frame" for _, _, msg in calls)
