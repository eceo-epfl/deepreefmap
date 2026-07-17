"""Remaining-time estimation for a reconstruction run."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

# Cost driver per coarse stage. The pipeline is strictly sequential, so remaining
# time is the running stage plus the sum of the pending ones. The shapes:
# - preprocess/mapping are per-frame Python loops, so linear in frame count.
# - the cloud replacement radius and ortho cell sort are `np.lexsort`, O(N log N).
# - ortho PCA is a fixed 2-component fit, dominated by an O(N) covariance pass.
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
# `_RECON_PHASES` shares (gui/runs/progress.py) and are used only when there is no
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

# Fine per-step phase keys (gui/runs/progress.py) folded onto the coarse stages above.
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
# A running stage whose fraction is exhausted (the mapping tail after the last
# window, folded sub-phases pinned at 1.0) has no extrapolation left; showing
# "~0s left" there is a lie, so the remainder is withheld instead.
_FRAC_EXHAUSTED = 0.999
# Once a stage has run this many times longer than its prior predicted with no
# measurable progress, the prior is falsified and its remainder is withheld.
_PRIOR_OVERRUN_FACTOR = 1.5


def stage_for_phase(phase_key: str) -> str | None:
    """Coarse stage a fine progress phase belongs to, or None if unmapped."""
    return _PHASE_TO_STAGE.get(phase_key)


def stage_label_for_phase(phase_key: str) -> str | None:
    """Human label of the coarse stage a fine phase belongs to (matches the popup)."""
    stage = _PHASE_TO_STAGE.get(phase_key)
    return _STAGE_BY_KEY[stage].label if stage else None


def format_duration(seconds: float) -> str:
    """Render a duration as `<1s`, `37s`, `2m 14s`, or `1h 03m`."""
    if 0 < seconds < 1:
        return "<1s"
    secs = int(seconds)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60:02d}s"
    return f"{secs // 3600}h {(secs % 3600) // 60:02d}m"


def format_remaining(seconds: float) -> str:
    """Render a remainder coarsely, rounded up: `~15s`, `~2m 30s`, `~11m 00s`.

    Estimates carry no second-level precision, so the display shouldn't either;
    the coarse buckets also stop the figure flapping between renders.
    """
    if seconds < 60:
        step = 5
    elif seconds < 600:
        step = 30
    else:
        step = 60
    return f"~{format_duration(max(step, math.ceil(seconds / step) * step))}"


def _nlogn(n: float) -> float:
    return n * math.log(n) if n > 1 else 0.0


def driver_denominator(driver: str, frames: int, points: int | None) -> float | None:
    """Size that a stage's cost scales with, or None when it isn't known yet."""
    # Shared by the live estimator and the history fitter so a stored constant and a
    # live prediction divide by the same quantity.
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
    frac0: float | None = None  # fraction at the first determinate event

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
    """Live remaining-time estimate for one reconstruction run."""

    frames: int
    # Stage key -> seconds per unit of driver, learned from this machine's past runs
    # (run_history.py). Absent keys fall back to weight-based projection.
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
        """True when this machine has learned timings for the selected backends."""
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
        if run.frac0 is None:
            # The extrapolation baseline: only progress earned after this point
            # counts, so a stage entered mid-slice doesn't divide by unearned frac.
            run.frac0 = frac
        run.frac = frac

    def _driver_value(self, spec: StageSpec) -> float | None:
        # Point-driven stages have no true N until mapping ends, so fall back to
        # the historical N from comparable runs; set_points supplies the real one.
        points = self.points if self.points is not None else self.expected_points
        return driver_denominator(spec.driver, self.frames, points)

    def _completed_seconds_per_weight(self, now: float) -> float | None:
        """Seconds-per-weight calibrated from stages already finished this run."""
        num = den = 0.0
        for spec in STAGES:
            run = self._runs[spec.key]
            if run.state == "done":
                num += run.elapsed(now)
                den += spec.weight
        return num / den if den > 0 else None

    def _seconds_per_weight(self, now: float) -> float | None:
        """A per-weight rate for stages that lack their own driver-based prior.

        Keeps a point-driven stage from reading 0 before its point count is known.
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

    def _live_remaining(self, spec: StageSpec, now: float) -> float | None:
        """Remainder from this stage's own throughput, or None if not yet reliable.

        Purely measured, so it is safe to show on a first run. Evaluated against
        the wall clock at query time — average time per unit of earned fraction,
        scaled by the fraction left — so it counts down between sparse progress
        events and grows honestly when the stage stalls, instead of freezing at
        whatever the rate was at the last event.
        """
        run = self._runs[spec.key]
        if run.frac0 is None:
            return None
        delta = run.frac - run.frac0
        elapsed = run.elapsed(now)
        if delta >= _MIN_FRAC_FOR_LIVE and elapsed > 0:
            return max(0.0, elapsed * (1.0 - run.frac) / delta)
        return None

    def _live_confidence(self, spec: StageSpec) -> float:
        """0 at the live threshold, ramping to 1 by the handover fraction."""
        run = self._runs[spec.key]
        delta = run.frac - run.frac0 if run.frac0 is not None else 0.0
        if delta <= _MIN_FRAC_FOR_LIVE:
            return 0.0
        if delta >= _LIVE_HANDOVER_FRAC:
            return 1.0
        return (delta - _MIN_FRAC_FOR_LIVE) / (_LIVE_HANDOVER_FRAC - _MIN_FRAC_FOR_LIVE)

    def _running_remaining(self, spec: StageSpec, now: float) -> float | None:
        """Remaining for the running stage: prior first, gliding into the live rate."""
        run = self._runs[spec.key]
        if run.frac >= _FRAC_EXHAUSTED:
            # Folded tail work (align, transfer, saves) after the fraction is
            # spent: no extrapolation left, so no number rather than "~0s left".
            return None
        live = self._live_remaining(spec, now)
        full = self._prior_estimate(spec, now)
        prior = max(0.0, full * (1.0 - run.frac)) if full is not None else None
        if live is None:
            if full is not None and run.elapsed(now) > _PRIOR_OVERRUN_FACTOR * full:
                return None
            return prior
        if prior is None:
            return live
        w = self._live_confidence(spec)
        return w * live + (1.0 - w) * prior

    def running_stage_label(self) -> str | None:
        """Label of the coarse stage currently running, or None if none is."""
        # The furthest-along running stage wins: a late fold-back event (viewer setup
        # arriving after the scene-file save started) can leave an earlier stage
        # marked running, and the pipeline is sequential, so the later one is current.
        for spec in reversed(STAGES):
            if self._runs[spec.key].state == "running":
                return spec.label
        return None

    def current_stage_remaining(self, now: float) -> float | None:
        """Remainder for whichever stage is running, or None with no signal at all."""
        for spec in STAGES:
            if self._runs[spec.key].state == "running":
                return self._running_remaining(spec, now)
        return None

    def visible_remaining(self, now: float) -> float | None:
        """The whole-run figure for the always-visible total slot, or None."""
        # Withheld without history: the pending stages have no seed, so a total would
        # be a guess masquerading as a countdown.
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
