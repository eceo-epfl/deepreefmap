from __future__ import annotations

import time

from deepreefmap.gui._window_protocol import MixinBase
from deepreefmap.gui.eta import (
    RunEtaEstimator,
    format_duration,
    stage_for_phase,
    stage_label_for_phase,
)
from deepreefmap.gui.theme import PRIMARY

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


class ProgressModel:
    """Weighted, ordered phase model that drives the unified total progress bar.

    Each phase contributes a fixed fraction of the total. Phases are reported
    forward-only: when a later phase begins, all earlier phases are promoted
    to 100% so the total bar never moves backwards within a run.
    """

    def __init__(self, phases: list[tuple[str, float]]) -> None:
        self._phases = phases
        self._idx_by_key = {k: i for i, (k, _) in enumerate(phases)}
        self._total_weight = sum(w for _, w in phases) or 1.0
        self._percents: dict[str, float] = {k: 0.0 for k, _ in phases}
        self._max_idx = -1

    def update(self, key: str, cur: int, tot: int) -> int:
        """Record progress for `key`. Returns the new total percent (0-100)."""
        idx = self._idx_by_key.get(key)
        if idx is not None:
            if idx > self._max_idx:
                # Promote the previously-active phase (if any) and every
                # phase we skipped past — they're all done. `max(0, ...)`
                # handles the initial state where _max_idx == -1.
                for i in range(max(0, self._max_idx), idx):
                    self._percents[self._phases[i][0]] = 100.0
                self._max_idx = idx
            if tot > 0:
                frac = max(0.0, min(1.0, float(cur) / float(tot)))
                new_pct = 100.0 * frac
                if new_pct > self._percents[key]:
                    self._percents[key] = new_pct
        return self.total_percent()

    def total_percent(self) -> int:
        s = sum(self._percents[k] / 100.0 * w for k, w in self._phases)
        return int(round(s / self._total_weight * 100))

    def reset(self) -> None:
        for k in self._percents:
            self._percents[k] = 0.0
        self._max_idx = -1


# Note on ortho_* weights: build_ortho_outputs is dominated by sklearn's
# PCA.fit_transform on the full point cloud. On large reefs (10M+ points
# — the 3.5GB-dataset case) that single step can be ~60% of total wall
# time, which is why ortho_pca carries the biggest individual weight.
_RECON_PHASES: list[tuple[str, float]] = [
    ("startup", 1.0),
    ("preprocess", 18.0),
    # Mapping is three real steps the user waits through: window inference, the
    # float64 pose re-anchor, and the resume-npz write. Splitting them stops the
    # bar pinning at 100% for minutes after the last inference window. Shares
    # reflect the measured post-optimisation run (inference ~87s, align ~55s,
    # save now seconds once the npz is uncompressed).
    ("mapping", 15.0),
    ("mapping_align", 8.0),
    ("mapping_save", 2.0),
    ("outputs", 2.0),
    ("cloud_concat", 2.0),
    ("cloud_replace", 10.0),
    ("cloud_voxel", 1.0),
    ("ortho_pca", 12.0),
    ("ortho_sort", 4.0),
    ("ortho_aggregate", 4.0),
    ("ortho_cover", 2.0),
    ("viewer_index_cloud", 1.0),
    ("viewer_index_classes", 4.0),
    ("viewer_actors", 1.0),
    ("viewer_frustums", 3.0),
    ("viewer_camera", 1.0),
    ("viewer_upload", 6.0),
    ("viewer_finalise", 1.0),
    ("ortho_save", 2.0),
    # The scene file is the last write and the slowest; give it real weight so the
    # total bar keeps moving instead of freezing at "Reconstruction complete".
    ("scene_save", 8.0),
]

# Some coarse stages are reported as several sub-phases but must read as one
# continuous 0-100 detail bar (and one estimator fraction), not several resets or
# a pin at 100%. Each sub-phase drives a slice sized by the same weights the total
# bar uses, so the fill flows straight through, and an indeterminate tail step
# holds at its slice start instead of leaving the stage stuck "done" while it runs.

# Mapping: window inference, pose re-anchor, resume save.
_MAPPING_PHASE_KEYS: tuple[str, ...] = ("mapping", "mapping_align", "mapping_save")

# Cloud: the per-frame unprojection loop (reported as "outputs") then the tail —
# concatenate, replacement-radius lexsort, voxel reduce. Without this split the
# cheap per-frame loop drove the stage to 100% and the long lexsort read "~0s
# left"; the tail owns most of the weight, matching where the wall time goes.
_CLOUD_PHASE_KEYS: tuple[str, ...] = ("outputs", "cloud_concat", "cloud_replace", "cloud_voxel")

# The per-item loop that legitimately carries a `cur/tot`; the other sub-phases
# report raw point totals or nothing and would read as noise on the status line.
_COUNTED_SUBPHASES: frozenset[str] = frozenset({"mapping", "outputs"})


def _subphase_spans(keys: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    weights = dict(_RECON_PHASES)
    total = sum(weights[k] for k in keys) or 1.0
    spans: dict[str, tuple[float, float]] = {}
    acc = 0.0
    for key in keys:
        share = weights[key] / total
        spans[key] = (acc, acc + share)
        acc += share
    return spans


_MAPPING_SUBPHASE_SPANS = _subphase_spans(_MAPPING_PHASE_KEYS)
_CLOUD_SUBPHASE_SPANS = _subphase_spans(_CLOUD_PHASE_KEYS)
# One lookup over every fine phase shown as part of a continuous stage fill.
_SUBPHASE_SPANS = {**_MAPPING_SUBPHASE_SPANS, **_CLOUD_SUBPHASE_SPANS}


# cloud_concat / cloud_replace / cloud_voxel are the silent post-frame steps
# inside build_semantic_reference_cloud (concatenate, replacement-radius
# lexsort, optional voxel reduce). On a 3.5GB dataset cloud_replace alone is
# multi-second wall time — hence the chunky weight. The ortho_* phases come
# from the live ortho preview built at the end of _apply_loaded_run.
_LOAD_PHASES: list[tuple[str, float]] = [
    ("manifest", 1.0),
    ("mapping_load", 6.0),
    ("frames_load", 18.0),
    ("cloud_build", 15.0),
    ("cloud_concat", 3.0),
    ("cloud_replace", 12.0),
    ("cloud_voxel", 2.0),
    ("ortho_pca", 8.0),
    ("ortho_sort", 2.0),
    ("ortho_aggregate", 1.0),
    ("ortho_cover", 1.0),
    ("viewer_index_cloud", 2.0),
    ("viewer_index_classes", 5.0),
    ("viewer_actors", 1.0),
    ("viewer_frustums", 4.0),
    ("viewer_camera", 1.0),
    ("viewer_upload", 17.0),
    ("viewer_finalise", 1.0),
]

# Maps setup_progress messages from qt_viewer to phase keys.
_SETUP_MESSAGE_TO_PHASE: dict[str, str] = {
    "Indexing point cloud": "viewer_index_cloud",
    "Indexing cloud": "viewer_index_cloud",
    "Indexing classes": "viewer_index_classes",
    "Preparing class actors": "viewer_actors",
    "Building camera frustums": "viewer_frustums",
    "Fitting camera": "viewer_camera",
    "Uploading class points": "viewer_upload",
    "Finalising viewer": "viewer_finalise",
}

# Maps view-run loader stage strings to phase keys. The `cloud_*` variants
# are emitted by run_loader's stage_cb after the per-frame loop reports
# N/N, so the bars don't freeze during concatenation / replacement /
# voxelization.
_LOAD_STAGE_TO_PHASE: dict[str, str] = {
    "manifest": "manifest",
    "classes": "manifest",
    "mapping": "mapping_load",
    "frames": "frames_load",
    "cloud": "cloud_build",
    "cloud_concatenating": "cloud_concat",
    "cloud_replacing": "cloud_replace",
    # The replacement-radius lexsort is the dominant cost of cloud_replace
    # on multi-million-point clouds; route its sub-steps to the same phase
    # so the total bar reflects them under cloud_replace's weight.
    "cloud_replacing_keys": "cloud_replace",
    "cloud_replacing_sort": "cloud_replace",
    "cloud_replacing_select": "cloud_replace",
    "cloud_voxelizing": "cloud_voxel",
    "geometry": "cloud_build",
}

# Maps the per-stage `set_stage(stage, status, message)` text to a finer
# phase key. Used so the "outputs" stage can drive distinct ortho_* phases
# from the messages the orchestrator emits while building the ortho grid
# and writing the final files.
_STAGE_MESSAGE_TO_PHASE: dict[str, str] = {
    "Aligning poses to world frame": "mapping_align",
    "Saving depth + points for resume": "mapping_save",
    "Mapping complete": "mapping_save",
    "Concatenating point arrays": "cloud_concat",
    "Applying replacement radius": "cloud_replace",
    "Replacement radius: computing voxel keys": "cloud_replace",
    "Replacement radius: sorting points": "cloud_replace",
    "Replacement radius: selecting representatives": "cloud_replace",
    "Reducing by voxel size": "cloud_voxel",
    "Computing PCA projection": "ortho_pca",
    "Sorting points into cells": "ortho_sort",
    "Aggregating ortho grid": "ortho_aggregate",
    "Computing benthic cover": "ortho_cover",
    "Saving semantic cloud": "ortho_save",
    "Saving TSDF cloud": "ortho_save",
    "Saving ortho image": "ortho_save",
    "Saving cover report": "ortho_save",
    "Writing run manifest": "ortho_save",
    "Saving outputs": "ortho_save",
    "Saving scene file": "scene_save",
    "Building geometry cloud": "outputs",
    "Generating outputs": "outputs",
}


class ProgressBarsMixin(MixinBase):
    """DeepReefMapWindow methods that drive the per-step + unified progress bars."""

    def _ensure_status_tick_timer(self) -> QTimer:
        # Mixins have no __init__, so the ticker is created lazily on first use.
        timer = getattr(self, "_status_tick_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(1000)
            timer.timeout.connect(self._render_status)
            self._status_tick_timer = timer
        return timer

    def _ensure_timing_popup(self):
        popup = getattr(self, "_timing_popup", None)
        if popup is None:
            from deepreefmap.gui.timing_popup import TimingPopup

            popup = TimingPopup(self)
            self._timing_popup = popup
        return popup

    def _connect_bar_hover(self) -> None:
        # The bar column reports hover so the breakdown can follow the cursor.
        # Connected once, before any hover can fire.
        if not getattr(self, "_hover_connected", False):
            self._progress_stack.hovered.connect(self._on_total_bar_hover)
            self._hover_connected = True

    def _begin_progress(self, model: ProgressModel) -> None:
        """Switch the active progress model and light up both bars from zero."""
        self._connect_bar_hover()
        model.reset()
        self._active_progress_model = model
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setEnabled(True)
        self._total_progress_bar.setRange(0, 100)
        self._total_progress_bar.setValue(0)
        self._total_progress_bar.setEnabled(True)
        self._status_base_text = ""
        self._status_count_text = ""
        self._status_phase_key = None
        self._status_phase_started = time.monotonic()
        self._stage_fill: dict[str, float] = {}
        # ETA only applies to a reconstruction; a cached-run load has its own model.
        self._eta = self._new_run_estimator() if model is self._recon_model else None
        self._ensure_status_tick_timer().start()

    def _new_run_estimator(self) -> RunEtaEstimator:
        """Estimator seeded from this machine's history for the selected backends."""
        from deepreefmap.gui.run_history import history_key, load_expected_points, load_priors

        try:
            key = history_key(
                self._map_combo.currentText(),
                self._seg_combo.currentText(),
                self._proc_width_spin.value(),
                self._proc_height_spin.value(),
                self._fps_spin.value(),
            )
            priors = load_priors(key)
            expected_points = load_expected_points(key)
        except Exception:
            priors = {}
            expected_points = None
        return RunEtaEstimator(frames=0, priors=priors, expected_points=expected_points)

    def _reset_progress_bars(self) -> None:
        # Bars stay visible but empty when idle so the top-right cluster always
        # reads as the run status area next to the play button.
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setEnabled(False)
        self._total_progress_bar.setRange(0, 100)
        self._total_progress_bar.setValue(0)
        self._total_progress_bar.setEnabled(False)
        self._eta_total_label.setText("")
        self._active_progress_model = None
        self._eta = None
        popup = getattr(self, "_timing_popup", None)
        if popup is not None:
            popup.hide()
        timer = getattr(self, "_status_tick_timer", None)
        if timer is not None:
            timer.stop()
        self._status_base_text = ""
        self._status_count_text = ""
        self._status_phase_key = None
        self._stage_fill = {}

    def _render_status(self) -> None:
        """Recompose the status label: coloured stage + label, then a metrics line.

        The stage name and its description stay on the first line; the frame
        count, stage elapsed, and stage remainder sit on a second line directly
        underneath so the figures never widen the first line as they change.
        """
        self._render_eta()
        base = getattr(self, "_status_base_text", "")
        if not base:
            return
        now = time.monotonic()
        started = getattr(self, "_status_phase_started", None)
        # Second line: count · stage elapsed · stage remainder. Kept off the
        # first line so the label text never jumps as the numbers grow. The
        # remainder is prior-first then live (same blend as the total), and
        # stage-scoped so it never masquerades as the whole-run total.
        parts: list[str] = []
        count = getattr(self, "_status_count_text", "")
        if count:
            parts.append(count)
        if started is not None:
            parts.append(format_duration(now - started))
        est = getattr(self, "_eta", None)
        stage_left = est.current_stage_remaining(now) if est is not None else None
        if stage_left is not None:
            parts.append(f"~{format_duration(stage_left)} left")
        metrics = " · ".join(parts)
        # Colour the active coarse stage so the left text names it (and the stage
        # name is dropped from the bars). During a reconstruction take it from the
        # estimator's running stage (monotonic), so a late viewer-setup event can't
        # regress the token to a finished stage while a later save is still running.
        # Fall back to the fine phase key when there is no estimator (cached load).
        stage = est.running_stage_label() if est is not None else None
        if not stage:
            stage = stage_label_for_phase(getattr(self, "_status_phase_key", "") or "")
        if stage:
            first = f'<b><span style="color:{PRIMARY}">{stage}</span></b> · {base}'
        else:
            first = base
        self._status_label.setText(f"{first}<br>{metrics}" if metrics else first)

    def _render_eta(self) -> None:
        """Refresh the visible overall-estimate label and the breakdown popup."""
        est = getattr(self, "_eta", None)
        if est is None:
            return
        now = time.monotonic()
        visible = est.visible_remaining(now)
        # Overall estimate shown plainly rather than buried in the hover. None
        # means no trustworthy figure yet (a first run still calibrating).
        self._eta_total_label.setText(
            f"~{format_duration(visible)} left" if visible is not None else "estimating…"
        )
        popup = getattr(self, "_timing_popup", None)
        if popup is not None and popup.isVisible():
            popup.set_rows(est.stage_rows(now), est.total_remaining_s(now), est.has_history)

    def _on_total_bar_hover(self, global_pos) -> None:
        est = getattr(self, "_eta", None)
        if global_pos is None or est is None:
            popup = getattr(self, "_timing_popup", None)
            if popup is not None:
                popup.hide()
            return
        popup = self._ensure_timing_popup()
        now = time.monotonic()
        popup.set_rows(est.stage_rows(now), est.total_remaining_s(now), est.has_history)
        popup.move(int(global_pos.x()) + 14, int(global_pos.y()) + 16)
        popup.show()

    def _apply_progress(
        self,
        phase_key: str,
        label: str,
        current: int = 0,
        total: int = 0,
        flush: bool = False,
    ) -> None:
        """Update the per-step bar/label and the unified total bar.

        - `total > 1`: per-step bar is determinate; status shows `cur/tot`.
        - `total == 1`: per-step bar shows the phase as complete.
        - `total <= 0`: per-step bar is indeterminate.
        Total bar always reflects the active model's weighted progress.
        """
        # Reset the stage stopwatch when the phase key changes so elapsed time
        # is per-stage, not per-run.
        now = time.monotonic()
        if getattr(self, "_status_phase_key", None) != phase_key:
            self._status_phase_key = phase_key
            self._status_phase_started = now

        # Mapping and cloud are each several sub-phases folded into one monotonic
        # 0-100 fill sized by their weights. Compute that combined value once so the
        # detail bar and the hover breakdown show the same number, and so an
        # indeterminate tail step (GPU transfer, the resume save, the cloud lexsort)
        # holds at its slice start rather than pinning the stage at 100% (which read
        # as "~0s left" while it was still working). Accumulate per coarse stage so
        # cloud does not inherit mapping's finished fill.
        span = _SUBPHASE_SPANS.get(phase_key)
        stage_combined: float | None = None
        if span is not None:
            coarse = stage_for_phase(phase_key) or phase_key
            lo, hi = span
            within = min(1.0, current / total) if total > 0 else 0.0
            combined = 100.0 * (lo + (hi - lo) * within)
            fills = getattr(self, "_stage_fill", None)
            if fills is None:
                fills = {}
                self._stage_fill = fills
            stage_combined = max(fills.get(coarse, 0.0), combined)
            fills[coarse] = stage_combined

        est = getattr(self, "_eta", None)
        if est is not None and stage_for_phase(phase_key) is not None:
            # The preprocess total is the selected frame count, the size the
            # per-frame stages scale with; capture it for pending predictions.
            if phase_key == "preprocess" and total > 0:
                est.frames = total
            if stage_combined is not None:
                # Feed the estimator the same combined fill the detail bar shows
                # (0-100), not the raw per-sub-phase fraction. Otherwise the hover
                # bar snaps to 100% when the determinate loop ends and its remainder
                # reads 0s while the indeterminate tail still runs.
                est.update(phase_key, int(round(stage_combined)), 100, now)
            else:
                est.update(phase_key, current, total, now)

        # The bars carry no text (the stage name is coloured in the status line);
        # they only show fill. Indeterminate for total <= 0. The frame count is
        # kept out of the base text so it can sit on the metrics line.
        if stage_combined is not None:
            if self._progress_bar.minimum() != 0 or self._progress_bar.maximum() != 100:
                self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(int(round(stage_combined)))
            self._status_base_text = label
            # Only the per-item loop carries a meaningful count; the tail sub-phases
            # report raw point totals or nothing and would read as noise.
            self._status_count_text = (
                f"{current}/{total}" if phase_key in _COUNTED_SUBPHASES and total > 1 else ""
            )
        elif total > 1:
            if self._progress_bar.minimum() != 0 or self._progress_bar.maximum() != total:
                self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
            self._status_base_text = label
            self._status_count_text = f"{current}/{total}"
        elif total == 1:
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(1)
            self._status_base_text = label
            self._status_count_text = ""
        else:
            self._progress_bar.setRange(0, 0)
            self._status_base_text = label
            self._status_count_text = ""
        self._render_status()
        self._progress_bar.setEnabled(True)

        if self._active_progress_model is not None:
            pct = self._active_progress_model.update(
                phase_key,
                current if total > 0 else 0,
                total if total > 0 else 1,
            )
            self._total_progress_bar.setRange(0, 100)
            self._total_progress_bar.setValue(pct)
            self._total_progress_bar.setEnabled(True)

        if flush:
            QApplication.processEvents()
