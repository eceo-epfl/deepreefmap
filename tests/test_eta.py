"""Remaining-time estimator: measurement-first, no number without signal."""

from __future__ import annotations

from deepreefmap.gui.eta import RunEtaEstimator, stage_for_phase


def test_no_estimate_before_any_signal():
    est = RunEtaEstimator(frames=100)
    assert est.total_remaining_s(now=0.0) is None


def test_running_stage_extrapolates_from_live_rate():
    est = RunEtaEstimator(frames=100)
    est.update("preprocess", current=1, total=100, now=0.0)
    est.update("preprocess", current=50, total=100, now=50.0)
    # Half done in 50s → roughly 50s left for preprocess. Other stages have no
    # signal yet, so the total is at least the running-stage remainder.
    remaining = est.total_remaining_s(now=50.0)
    assert remaining is not None
    assert 40.0 <= remaining <= 70.0


def test_running_stage_shows_prior_before_live_is_reliable():
    # Preprocess prior: 100 frames * 1 s/frame = 100s. At 4% done (below the 8%
    # live threshold) the step must still count down from the prior, not sit blank
    # waiting for the library's first real numbers.
    est = RunEtaEstimator(frames=100, priors={"preprocess": 1.0})
    est.update("preprocess", current=4, total=100, now=4.0)
    row = {r.key: r for r in est.stage_rows(now=4.0)}["preprocess"]
    assert row.state == "running"
    assert row.remaining is not None
    assert 90.0 <= row.remaining <= 100.0  # ~100 * (1 - 0.04)


def test_running_stage_hands_over_to_live_rate():
    # Prior says 100s; the stage actually runs at double speed. Past the handover
    # fraction the live rate wins, so it reads ~25s, not the ~50s prior remainder.
    est = RunEtaEstimator(frames=100, priors={"preprocess": 1.0})
    est.update("preprocess", current=1, total=100, now=0.0)
    est.update("preprocess", current=50, total=100, now=25.0)
    row = {r.key: r for r in est.stage_rows(now=25.0)}["preprocess"]
    assert row.remaining is not None
    assert 20.0 <= row.remaining <= 32.0


def test_status_stage_remaining_counts_down_from_prior():
    # The status line uses current_stage_remaining; it must fall back to the
    # prior like the total does, not blank out below the live threshold.
    est = RunEtaEstimator(frames=100, priors={"preprocess": 1.0})
    est.update("preprocess", current=4, total=100, now=4.0)
    remaining = est.current_stage_remaining(now=4.0)
    assert remaining is not None
    assert 90.0 <= remaining <= 100.0


def test_indeterminate_subphase_keeps_the_stage_remainder():
    # The resume save reports (0, 0). It must not zero mapping's fraction and
    # blank the countdown while the save runs.
    est = RunEtaEstimator(frames=100, priors={"mapping": 1.0})
    est.update("mapping", current=1, total=100, now=0.0)
    est.update("mapping", current=90, total=100, now=90.0)
    before = est.current_stage_remaining(now=90.0)
    est.update("mapping", current=0, total=0, now=91.0)
    after = est.current_stage_remaining(now=91.0)
    assert before is not None and after is not None
    assert abs(after - before) < 2.0


def test_subphase_unit_change_does_not_drag_frac_backwards():
    # The pose re-anchor reports point counts, restarting near zero in a new
    # unit. Mapping's fraction must stay monotonic so the countdown neither
    # blanks nor jumps back to the full prior.
    est = RunEtaEstimator(frames=100, priors={"mapping": 1.0})
    est.update("mapping", current=1, total=100, now=0.0)
    est.update("mapping", current=90, total=100, now=90.0)
    est.update("mapping_align", current=1_000, total=500_000, now=91.0)
    row = {r.key: r for r in est.stage_rows(now=91.0)}["mapping"]
    assert row.frac >= 0.9
    assert row.remaining is not None
    assert row.remaining < 30.0


def test_completed_stage_calibrates_pending_via_weights():
    est = RunEtaEstimator(frames=100)
    est.update("preprocess", current=1, total=100, now=0.0)
    # Preprocess finishes (weight 18) in 36s when mapping starts → 2s per weight.
    est.update("mapping", current=1, total=100, now=36.0)
    rows = {r.key: r for r in est.stage_rows(now=36.0)}
    assert rows["preprocess"].state == "done"
    # A pending point-stage is projected from seconds-per-weight (cloud weight 13).
    assert rows["cloud"].state == "pending"
    assert rows["cloud"].predicted
    assert rows["cloud"].seconds > 0


def test_history_prior_used_for_pending_frame_stage():
    est = RunEtaEstimator(frames=200, priors={"mapping": 0.5})
    est.update("preprocess", current=1, total=100, now=0.0)
    rows = {r.key: r for r in est.stage_rows(now=1.0)}
    # 200 frames * 0.5 s/frame from history.
    assert abs(rows["mapping"].seconds - 100.0) < 1e-6


def test_expected_points_lets_point_stage_predict_before_mapping_ends():
    # The 2nd-run bug: without a provisional N, cloud/ortho/save collapse to a few
    # seconds. Seeding N from history makes them predict properly pre-mapping.
    seeded = RunEtaEstimator(frames=100, priors={"cloud": 1e-6}, expected_points=5_000_000)
    unseeded = RunEtaEstimator(frames=100, priors={"cloud": 1e-6})
    seeded.update("preprocess", current=1, total=100, now=0.0)
    unseeded.update("preprocess", current=1, total=100, now=0.0)
    cloud_seeded = {r.key: r for r in seeded.stage_rows(now=1.0)}["cloud"]
    cloud_unseeded = {r.key: r for r in unseeded.stage_rows(now=1.0)}["cloud"]
    # 5e6 * log(5e6) * 1e-6 ~= 77s from the point prior, not a weight guess.
    assert cloud_seeded.seconds > 50.0
    assert cloud_unseeded.seconds == 0.0


def test_real_points_override_expected_points():
    est = RunEtaEstimator(frames=100, priors={"ortho": 1e-6}, expected_points=1_000_000)
    est.update("preprocess", current=1, total=100, now=0.0)
    est.set_points(9_000_000)
    ortho = {r.key: r for r in est.stage_rows(now=1.0)}["ortho"]
    # POINTS driver: 9e6 * 1e-6 = 9s from the true N, not the 1e6 seed.
    assert abs(ortho.seconds - 9.0) < 1e-6


def test_point_stage_waits_for_known_n_before_using_point_prior():
    est = RunEtaEstimator(frames=100, priors={"ortho": 1e-6})
    est.update("preprocess", current=1, total=100, now=0.0)
    est.update("mapping", current=1, total=100, now=20.0)
    before = {r.key: r for r in est.stage_rows(now=20.0)}["ortho"].seconds
    est.set_points(5_000_000)
    after = {r.key: r for r in est.stage_rows(now=20.0)}["ortho"].seconds
    # Before N is known the point prior can't apply, so the two differ once the
    # real point count arrives (5e6 * 1e-6 = 5s).
    assert abs(after - 5.0) < 1e-6
    assert before != after


def test_current_stage_remaining_is_measured_without_history():
    est = RunEtaEstimator(frames=100)
    assert est.current_stage_remaining(now=0.0) is None
    est.update("mapping", current=1, total=100, now=0.0)
    est.update("mapping", current=25, total=100, now=25.0)
    # 25% in 25s → ~75s left
    remaining = est.current_stage_remaining(now=25.0)
    assert remaining is not None
    assert 60.0 <= remaining <= 90.0


def test_visible_remaining_withheld_without_history():
    est = RunEtaEstimator(frames=100)
    est.update("mapping", current=1, total=100, now=0.0)
    est.update("mapping", current=25, total=100, now=25.0)
    # A measured stage remainder exists for the status line...
    assert est.current_stage_remaining(now=25.0) is not None
    # ...but with no per-machine history the whole-run total slot is withheld,
    # never filled with the stage remainder masquerading as a total.
    assert est.visible_remaining(now=25.0) is None


def test_visible_remaining_is_total_with_history():
    est = RunEtaEstimator(frames=100, priors={"mapping": 0.5})
    est.update("preprocess", current=1, total=100, now=0.0)
    est.update("mapping", current=25, total=100, now=25.0)
    assert est.visible_remaining(now=25.0) == est.total_remaining_s(now=25.0)


def test_stage_rows_carry_fill_fraction():
    est = RunEtaEstimator(frames=100)
    est.update("preprocess", current=1, total=100, now=0.0)
    est.update("mapping", current=40, total=100, now=10.0)
    rows = {r.key: r for r in est.stage_rows(now=10.0)}
    # Done fills the bar, the running stage fills to its live fraction, pending is empty.
    assert rows["preprocess"].frac == 1.0
    assert abs(rows["mapping"].frac - 0.4) < 1e-6
    assert rows["cloud"].frac == 0.0


def test_stage_label_for_phase():
    from deepreefmap.gui.eta import stage_label_for_phase

    assert stage_label_for_phase("ortho_pca") == "Ortho"
    assert stage_label_for_phase("preprocess") == "Preprocess"
    assert stage_label_for_phase("nonsense") is None


def test_phase_folding():
    assert stage_for_phase("ortho_pca") == "ortho"
    assert stage_for_phase("viewer_upload") == "save_view"
    assert stage_for_phase("cloud_replace") == "cloud"
    assert stage_for_phase("nonsense") is None


def test_mapping_substeps_fold_onto_the_one_mapping_stage():
    # The align and resume-save sub-phases are their own bars on the total, but
    # the estimator keeps them under a single learnable "mapping" stage so the
    # coarse status label stays "Mapping" and mapping isn't marked done early.
    from deepreefmap.gui.eta import stage_label_for_phase

    assert stage_for_phase("mapping_align") == "mapping"
    assert stage_for_phase("mapping_save") == "mapping"
    assert stage_label_for_phase("mapping_align") == "Mapping"
    assert stage_label_for_phase("mapping_save") == "Mapping"


def test_mapping_stage_not_marked_done_by_align_or_save():
    est = RunEtaEstimator(frames=100)
    est.update("mapping", current=1, total=100, now=0.0)
    est.update("mapping_align", current=1, total=1, now=30.0)
    est.update("mapping_save", current=0, total=0, now=45.0)
    rows = {r.key: r for r in est.stage_rows(now=45.0)}
    assert rows["mapping"].state == "running"
    # Only the cloud starting ends mapping.
    est.update("outputs", current=1, total=10, now=50.0)
    assert {r.key: r for r in est.stage_rows(now=50.0)}["mapping"].state == "done"
