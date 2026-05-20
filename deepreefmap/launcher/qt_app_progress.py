from __future__ import annotations

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
    ("mapping", 25.0),
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
]

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
    "Building geometry cloud": "outputs",
    "Generating outputs": "outputs",
}


class ProgressBarsMixin:
    """DeepReefMapWindow methods that drive the per-step + unified progress bars."""

    def _begin_progress(self, model: ProgressModel) -> None:
        """Switch the active progress model and show both bars from zero."""
        model.reset()
        self._active_progress_model = model
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._total_progress_bar.setRange(0, 100)
        self._total_progress_bar.setValue(0)
        self._total_progress_bar.setVisible(True)

    def _reset_progress_bars(self) -> None:
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._total_progress_bar.setRange(0, 100)
        self._total_progress_bar.setValue(0)
        self._total_progress_bar.setVisible(False)
        self._active_progress_model = None

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
        if total > 1:
            if self._progress_bar.minimum() != 0 or self._progress_bar.maximum() != total:
                self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
            self._status_label.setText(f"{label}… {current}/{total}")
        elif total == 1:
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(1)
            self._status_label.setText(f"{label}…")
        else:
            self._progress_bar.setRange(0, 0)
            self._status_label.setText(f"{label}…")
        self._progress_bar.setVisible(True)

        if self._active_progress_model is not None:
            pct = self._active_progress_model.update(
                phase_key,
                current if total > 0 else 0,
                total if total > 0 else 1,
            )
            self._total_progress_bar.setRange(0, 100)
            self._total_progress_bar.setValue(pct)
            self._total_progress_bar.setVisible(True)

        if flush:
            QApplication.processEvents()
