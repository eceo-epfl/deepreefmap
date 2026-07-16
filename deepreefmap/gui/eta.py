"""Remaining-time estimation for a reconstruction run.

The estimate is measurement-first. A running stage starts from its prior scaled to
what is left, then glides onto its own live throughput as that rate settles, so a
countdown never blanks or snaps mid-stage. Before there is any real signal the
estimator returns ``None`` so the UI can say "estimating…" rather than show a
fabricated number.

This module is deliberately Qt-free so the logic can be tested without a display.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

# Cost driver per coarse stage. The reconstruction is a strictly sequential
# pipeline, so total remaining time is the running stage plus the sum of the
# pending ones.
#
# Why these shapes:
# - preprocess/mapping are per-frame Python loops, so cost is linear in the
#   selected frame count.
# - the cloud replacement radius and ortho cell sort are `np.lexsort`, which is
#   O(N log N) in point count.
# - ortho PCA is a fixed 2-component fit, dominated by an O(N) covariance pass.
# - point count N is unknown until mapping finishes, so point-driven stages fall
#   back to weight-based projection until `set_points` supplies the real N.
FIXED = "fixed"
FRAMES = "frames"
POINTS = "points"
POINTS_NLOGN = "points_nlogn"


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    driver: str
    weight: float  # relative share used only as a first-run fallback prior


# Coarse stages the user sees in the breakdown. Weights are the aggregated
# `_RECON_PHASES` shares (gui/progress.py) and are used only when there is no
# per-machine history yet.
STAGES: tuple[StageSpec, ...] = (
    StageSpec("startup", "Startup", FIXED, 1.0),
    StageSpec("preprocess", "Preprocess", FRAMES, 18.0),
    StageSpec("mapping", "Mapping", FRAMES, 25.0),
    StageSpec("cloud", "Cloud", POINTS_NLOGN, 13.0),
    StageSpec("ortho", "Ortho", POINTS, 22.0),
    StageSpec("save_view", "Save + view", POINTS, 7.0),
    # The scene .zarr.zip re-serialises the whole cloud + every frame, so it
    # scales with points and was the untimed "reconstruction complete" tail.
    StageSpec("scene_save", "Scene file", POINTS, 14.0),
)

_STAGE_BY_KEY = {s.key: s for s in STAGES}

# Fine per-step phase keys (gui/progress.py) folded onto the coarse stages above.
_PHASE_TO_STAGE = {
    "startup": "startup",
    "preprocess": "preprocess",
    "mapping": "mapping",
    # Align + resume-save are shown as their own bars on the total, but fold back
    # onto the one learnable "mapping" stage: we have no separate history for them
    # and the coarse status label should stay "Mapping" throughout.
    "mapping_align": "mapping",
    "mapping_save": "mapping",
    "outputs": "cloud",
    "cloud_concat": "cloud",
    "cloud_replace": "cloud",
    "cloud_voxel": "cloud",
    "ortho_pca": "ortho",
    "ortho_sort": "ortho",
    "ortho_aggregate": "ortho",
    "ortho_cover": "ortho",
    "viewer_index_cloud": "save_view",
    "viewer_index_classes": "save_view",
    "viewer_actors": "save_view",
    "viewer_frustums": "save_view",
    "viewer_camera": "save_view",
    "viewer_upload": "save_view",
    "viewer_finalise": "save_view",
    "ortho_save": "save_view",
    "scene_save": "scene_save",
}

# Only trust the live extrapolation once a stage has made enough progress that its
# rate has settled. Extrapolating from 1% done wildly overshoots.
_MIN_FRAC_FOR_LIVE = 0.08
# Over [_MIN_FRAC_FOR_LIVE, _LIVE_HANDOVER_FRAC] the estimate glides from the prior
# to the stage's own measured rate, so it does not snap (e.g. halve) the instant
# the library reports its first real numbers.
_LIVE_HANDOVER_FRAC = 0.4
_EMA_ALPHA = 0.3


def stage_for_phase(phase_key: str) -> str | None:
    """Coarse stage a fine progress phase belongs to, or None if unmapped."""
    return _PHASE_TO_STAGE.get(phase_key)


def stage_label_for_phase(phase_key: str) -> str | None:
    """Human label of the coarse stage a fine phase belongs to (matches the popup)."""
    stage = _PHASE_TO_STAGE.get(phase_key)
    return _STAGE_BY_KEY[stage].label if stage else None


def format_duration(seconds: float) -> str:
    """Render a duration as `<1s`, `37s`, `2m 14s`, or `1h 03m`.

    A genuinely sub-second stage reads `<1s` rather than `0s`, which otherwise
    looks like a broken or missing figure.
    """
    if 0 < seconds < 1:
        return "<1s"
    secs = int(seconds)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60:02d}s"
    return f"{secs // 3600}h {(secs % 3600) // 60:02d}m"


def _nlogn(n: float) -> float:
    return n * math.log(n) if n > 1 else 0.0


def driver_denominator(driver: str, frames: int, points: int | None) -> float | None:
    """Size that a stage's cost scales with, or None when it isn't known yet.

    Shared by the live estimator and the history fitter so a stored constant and
    a live prediction divide and multiply by the exact same quantity.
    """
    if driver == FIXED:
        return 1.0
    if driver == FRAMES:
        # frames == 0 means the count isn't known yet, same as points == None.
        return float(frames) if frames > 0 else None
    if points is None:
        return None
    return float(points) if driver == POINTS else _nlogn(points)


@dataclass
class _StageRun:
    state: str = "pending"  # pending | running | done
    started_at: float | None = None
    ended_at: float | None = None
    frac: float = 0.0
    rate: float | None = None  # EMA of frac per second, for smoothing

    def elapsed(self, now: float) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else now
        return max(0.0, end - self.started_at)


@dataclass
class StageRow:
    key: str
    label: str
    state: str
    seconds: float | None  # None on a pending stage with no basis yet ("estimating…")
    predicted: bool
    remaining: float | None = None  # live remainder for the running stage
    frac: float = 0.0  # 0..1 fill for the hover bar (done=1, running=live, pending=0)


@dataclass
class RunEtaEstimator:
    """Live remaining-time estimate for one reconstruction run.

    Fed stage transitions and progress fractions from the GUI. `priors` maps a
    stage key to seconds-per-unit-of-driver learned from this machine's past runs
    (see gui/run_history.py); absent keys fall back to weight-based projection.
    """

    frames: int
    priors: dict[str, float] = field(default_factory=dict)
    points: int | None = None
    expected_points: int | None = None
    _runs: dict[str, _StageRun] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._runs = {s.key: _StageRun() for s in STAGES}
        self._order = [s.key for s in STAGES]

    @property
    def has_history(self) -> bool:
        """True when this machine has learned timings for the selected backends.

        Without history the only "estimates" are weight-based extrapolations from
        a few seconds of the current run, which read as arbitrary. We hide them on
        a first run and just calibrate silently.
        """
        return bool(self.priors)

    def set_points(self, points: int) -> None:
        """Supply the true point count once mapping has produced the cloud."""
        self.points = points

    def update(self, phase_key: str, current: int, total: int, now: float) -> None:
        stage = stage_for_phase(phase_key)
        if stage is None:
            return
        run = self._runs[stage]
        # Any earlier stage still marked running is finished the moment a later
        # one reports, since the pipeline is sequential.
        idx = self._order.index(stage)
        for k in self._order[:idx]:
            prev = self._runs[k]
            if prev.state == "running":
                prev.state = "done"
                prev.ended_at = now
        if run.state == "pending":
            run.state = "running"
            run.started_at = now
        if run.state == "done":
            return
        if total <= 0:
            # Indeterminate sub-steps (align prep, the resume save) carry no
            # fraction; zeroing frac here would blank the remainder mid-stage.
            return
        # Sub-phases fold onto one stage but count different units (mapping
        # windows, then re-anchor points), so a later sub-phase restarting at 0
        # must not drag the stage's fraction backwards. Same monotonic fill as
        # the visible mapping bar.
        frac = max(run.frac, current / total)
        elapsed = run.elapsed(now)
        if frac > 0 and elapsed > 0:
            inst = frac / elapsed
            run.rate = inst if run.rate is None else _EMA_ALPHA * inst + (1 - _EMA_ALPHA) * run.rate
        run.frac = frac

    def _driver_value(self, spec: StageSpec) -> float | None:
        # Point-driven stages have no true N until mapping ends, so fall back to
        # the historical N from comparable runs; set_points supplies the real one.
        points = self.points if self.points is not None else self.expected_points
        return driver_denominator(spec.driver, self.frames, points)

    def _completed_seconds_per_weight(self, now: float) -> float | None:
        """Seconds-per-weight calibrated from stages already finished this run.

        This is the fallback prior when the machine has no stored history yet: it
        turns the relative phase weights into real seconds using whatever the run
        has actually measured so far.
        """
        num = den = 0.0
        for spec in STAGES:
            run = self._runs[spec.key]
            if run.state == "done":
                num += run.elapsed(now)
                den += spec.weight
        return num / den if den > 0 else None

    def _seconds_per_weight(self, now: float) -> float | None:
        """A per-weight rate for stages that lack their own driver-based prior.

        Prefer this run's completed stages (measured); before any stage finishes,
        fall back to the rate implied by the stages that DO carry a stored prior.
        This keeps a point-driven stage from reading 0 when its own prior or point
        count is not yet known: it gets a non-zero weight-based over-estimate from
        the start instead. None only when there is no basis at all (first run).
        """
        measured = self._completed_seconds_per_weight(now)
        if measured is not None:
            return measured
        ratios: list[float] = []
        for spec in STAGES:
            driver = self._driver_value(spec)
            const = self.priors.get(spec.key)
            if const is not None and driver is not None and spec.weight > 0:
                ratios.append((const * driver) / spec.weight)
        return statistics.median(ratios) if ratios else None

    def _prior_estimate(self, spec: StageSpec, now: float) -> float | None:
        driver = self._driver_value(spec)
        const = self.priors.get(spec.key)
        if const is not None and driver is not None:
            return const * driver
        spw = self._seconds_per_weight(now)
        if spw is not None:
            return spw * spec.weight
        return None

    def _live_remaining(self, spec: StageSpec) -> float | None:
        """Remainder from this stage's own measured throughput, or None if not yet reliable.

        Purely measured: no prior, no weights. This is the figure safe to show on
        a first run, the same way a tqdm bar reports a stage ETA.
        """
        run = self._runs[spec.key]
        if run.frac >= _MIN_FRAC_FOR_LIVE and run.rate:
            return max(0.0, (1.0 - run.frac) / run.rate)
        return None

    def _prior_remaining(self, spec: StageSpec, now: float) -> float | None:
        """The prior stage duration scaled to what is left, for a running stage.

        `_prior_estimate` is the whole-stage duration; a stage part-way through has
        only `(1 - frac)` of it left. This is what lets a running stage count down
        from our estimate before its own measured rate is trustworthy.
        """
        full = self._prior_estimate(spec, now)
        if full is None:
            return None
        return max(0.0, full * (1.0 - self._runs[spec.key].frac))

    def _live_confidence(self, spec: StageSpec) -> float:
        """0 at the live threshold, ramping to 1 by the handover fraction."""
        frac = self._runs[spec.key].frac
        if frac <= _MIN_FRAC_FOR_LIVE:
            return 0.0
        if frac >= _LIVE_HANDOVER_FRAC:
            return 1.0
        return (frac - _MIN_FRAC_FOR_LIVE) / (_LIVE_HANDOVER_FRAC - _MIN_FRAC_FOR_LIVE)

    def _running_remaining(self, spec: StageSpec, now: float) -> float | None:
        """Remaining for the running stage: prior first, gliding into the live rate.

        Before the stage's own rate has settled we show the prior scaled to what is
        left, so the step and the total both count down from the estimate we already
        had instead of sitting blank. Once the live rate is reliable we blend into
        it over a short window so the figure glides rather than snapping.
        """
        live = self._live_remaining(spec)
        prior = self._prior_remaining(spec, now)
        if live is None:
            return prior
        if prior is None:
            return live
        w = self._live_confidence(spec)
        return w * live + (1.0 - w) * prior

    def running_stage_label(self) -> str | None:
        """Label of the coarse stage currently running, or None if none is.

        The furthest-along running stage wins. The pipeline is sequential, so if a
        late fold-back event (a viewer-setup message arriving after the scene-file
        save started) leaves an earlier stage marked running too, the later one is
        the true current stage. The status line uses this instead of the
        last-reported fine phase, which can arrive out of order.
        """
        for spec in reversed(STAGES):
            if self._runs[spec.key].state == "running":
                return spec.label
        return None

    def current_stage_remaining(self, now: float) -> float | None:
        """Remainder for whichever stage is running, or None with no signal at all.

        Prior-first like the total, so the status line's countdown never blinks
        out early in a stage. On a first run with no priors it degrades to the
        purely measured figure, appearing once the live rate settles.
        """
        for spec in STAGES:
            if self._runs[spec.key].state == "running":
                return self._running_remaining(spec, now)
        return None

    def visible_remaining(self, now: float) -> float | None:
        """The whole-run figure for the always-visible total slot, or None.

        Only shown with per-machine history. On a first run the pending stages
        have no seed, so a total would either be a pure weight-guess or the
        running stage's own remainder masquerading as a whole-run countdown. We
        withhold it (the caller shows "estimating…") and surface the measured
        stage remainder on the status line via `current_stage_remaining` instead.
        """
        return self.total_remaining_s(now) if self.has_history else None

    def total_remaining_s(self, now: float) -> float | None:
        remaining = 0.0
        have_signal = False
        for spec in STAGES:
            run = self._runs[spec.key]
            if run.state == "done":
                continue
            if run.state == "running":
                part = self._running_remaining(spec, now)
            else:
                part = self._prior_estimate(spec, now)
            if part is None:
                continue
            have_signal = True
            remaining += part
        return remaining if have_signal else None

    def stage_rows(self, now: float) -> list[StageRow]:
        rows: list[StageRow] = []
        for spec in STAGES:
            run = self._runs[spec.key]
            if run.state == "done":
                rows.append(StageRow(spec.key, spec.label, "done", run.elapsed(now), False, frac=1.0))
            elif run.state == "running":
                rows.append(StageRow(
                    spec.key, spec.label, "running", run.elapsed(now), False,
                    remaining=self._running_remaining(spec, now), frac=run.frac,
                ))
            else:
                # None (no basis yet) is preserved, not coerced to 0, so the popup
                # can render "estimating…" rather than a misleading "0s".
                est = self._prior_estimate(spec, now)
                rows.append(StageRow(spec.key, spec.label, "pending", est, True))
        return rows
