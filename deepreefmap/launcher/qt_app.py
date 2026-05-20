from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from PySide6.QtCore import QSettings, QSize, QStandardPaths, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPixmap, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


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


_DEFAULT_GH_REPO = "eceo-epfl/deepreefmap"


def _gh_releases_url() -> str:
    repo = os.environ.get("DEEPREEFMAP_GH_REPO", _DEFAULT_GH_REPO)
    return f"https://api.github.com/repos/{repo}/releases"


def _pyapp_binary_path() -> str | None:
    if os.environ.get("DEEPREEFMAP_MOCK_PYAPP"):
        return "/tmp/mock-pyapp"
    value = os.environ.get("PYAPP")
    if value and value != "1" and Path(value).exists():
        return value
    return None


def _fetch_release_versions(timeout: float = 8.0) -> list[str] | None:
    import urllib.request

    mock = os.environ.get("DEEPREEFMAP_MOCK_VERSIONS")
    if mock is not None:
        return [v.strip() for v in mock.split(",") if v.strip()]
    try:
        req = urllib.request.Request(_gh_releases_url(), headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            releases = json.load(resp)
        versions = []
        for rel in releases:
            tag = rel.get("tag_name", "")
            if tag.startswith("v"):
                tag = tag[1:]
            if tag and not rel.get("draft"):
                versions.append(tag)
        return versions if versions else None
    except Exception as exc:
        logger.warning("Failed to fetch releases from GitHub: %s", exc)
        return None


def _current_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("deepreefmap")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


class DeepReefMapWindow(QMainWindow):
    _sig_update_check_done = Signal(str, object, object)
    _sig_model_status_done = Signal(object, object)
    _sig_pipeline_error = Signal(str)
    _sig_status_text = Signal(str)
    _sig_hf_auth_done = Signal(object, str)
    _sig_download_progress = Signal(str, int)
    _sig_run_loaded = Signal(object, str, str)
    _sig_load_progress = Signal(str, int, int)
    _sig_batch_progress = Signal(int, int, str)
    _sig_batch_done = Signal(int, int, str)

    def __init__(self, classes_config: object, classes_path: Path) -> None:
        super().__init__()
        self._classes_config = classes_config
        self._classes_path = classes_path
        self._pipeline_thread: threading.Thread | None = None
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        self._sig_update_check_done.connect(self._apply_update_check)
        self._sig_model_status_done.connect(self._apply_model_status)
        self._sig_pipeline_error.connect(self._on_pipeline_error)
        self._sig_status_text.connect(lambda t: self._status_label.setText(t))
        self._sig_hf_auth_done.connect(self._on_hf_auth_done)
        self._sig_download_progress.connect(self._on_download_progress)
        self._sig_run_loaded.connect(self._apply_loaded_run)
        self._sig_load_progress.connect(self._on_load_progress)
        self._sig_batch_progress.connect(self._on_batch_progress)
        self._sig_batch_done.connect(self._on_batch_done)

        self.setWindowTitle("DeepReefMap")
        self.resize(1400, 900)

        from deepreefmap.visualization.qt_viewer import QtPointCloudViewer

        self._viewer = QtPointCloudViewer(
            class_colors=classes_config.id_to_color,
            class_names=classes_config.id_to_name,
        )
        self._viewer.set_status_callback(self._on_viewer_status)

        # Build the form first so widgets it references (status_label, etc.)
        # are constructed before we wire them into the top toolbar.
        form_panel = self._build_form_panel()
        top_bar = self._build_top_bar()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(form_panel)
        splitter.addWidget(self._viewer)
        splitter.setSizes([380, 1020])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(6)

        # Banner below the toolbar that pops up the instant a past run is
        # clicked, with the manifest metadata. Hidden until populated.
        self._run_meta_banner = QLabel("")
        self._run_meta_banner.setWordWrap(True)
        self._run_meta_banner.setTextFormat(Qt.TextFormat.RichText)
        self._run_meta_banner.setStyleSheet(
            "background-color: #1f2a36; color: #d8e2ec;"
            " padding: 4px 12px; border-bottom: 1px solid #2f3f50;"
        )
        # Compact single-row format means we only need ~2 lines of height.
        self._run_meta_banner.setMaximumHeight(56)
        self._run_meta_banner.setVisible(False)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(top_bar)
        central_layout.addWidget(self._run_meta_banner)
        central_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _build_form_panel(self) -> QWidget:
        from deepreefmap.camera.intrinsics import available_profile_names
        from deepreefmap.mapping.registry import list_mapping_backends
        from deepreefmap.segmentation.registry import list_segmentation_models

        profiles = available_profile_names() or ["gopro_hero_10"]
        seg_models = list_segmentation_models()
        map_backends = list_mapping_backends()
        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        default_root = str(Path(documents or str(Path.home())) / "DeepReefMap")

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        # These widgets are owned by the top toolbar but constructed here so
        # initialization code (_refresh_past_runs_combo, etc.) can reference
        # them before the toolbar is laid out.
        self._past_runs_combo = QComboBox()
        self._past_runs_combo.setMinimumContentsLength(20)
        self._past_runs_combo.currentIndexChanged.connect(self._on_past_run_selected)
        # Custom delegate paints each dropdown item as a card with name +
        # facts + input video, so the user can preview metadata before clicking.
        self._past_runs_combo.setItemDelegate(_PastRunCardDelegate(self._past_runs_combo))
        view = self._past_runs_combo.view()
        view.setSpacing(0)
        # Popup minimum width is computed from font metrics so it scales with
        # system DPI / font size (Windows scaling, Linux Hi-DPI, etc.).
        em = max(1, view.fontMetrics().height())
        view.setMinimumWidth(em * 36)
        # Auto-resize horizontally if needed so long fact strings have room.
        self._past_runs_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )

        self._new_run_btn = QPushButton("New reconstruction")
        self._new_run_btn.setToolTip("Clear the viewer and start a fresh run")
        self._new_run_btn.clicked.connect(self._on_new_reconstruction)

        self._past_open_btn = QPushButton("Open")
        self._past_open_btn.setToolTip("Open the selected run's folder in the system file manager")
        self._past_open_btn.clicked.connect(self._open_selected_past_run)

        self._load_cancel_btn = QPushButton("Cancel")
        self._load_cancel_btn.setVisible(False)
        self._load_cancel_btn.clicked.connect(self._cancel_load)

        layout.addWidget(QLabel("<b>New reconstruction</b>"))

        video_row = QHBoxLayout()
        self._video_input = QLineEdit()
        self._video_input.setPlaceholderText("Path to video file")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_video)
        video_row.addWidget(self._video_input, 1)
        video_row.addWidget(browse_btn)
        layout.addLayout(video_row)

        layout.addWidget(QLabel("Camera profile"))
        self._profile_combo = QComboBox()
        self._profile_combo.addItems(profiles)
        layout.addWidget(self._profile_combo)

        layout.addWidget(QLabel("Segmentation"))
        self._seg_combo = QComboBox()
        self._seg_combo.addItems(seg_models)
        idx = self._seg_combo.findText("segformer-b2")
        if idx >= 0:
            self._seg_combo.setCurrentIndex(idx)
        layout.addWidget(self._seg_combo)

        layout.addWidget(QLabel("Mapping"))
        self._map_combo = QComboBox()
        self._map_combo.addItems(map_backends)
        idx = self._map_combo.findText("scsfmlearner")
        if idx >= 0:
            self._map_combo.setCurrentIndex(idx)
        layout.addWidget(self._map_combo)

        layout.addWidget(QLabel("Output root"))
        self._out_root_input = QLineEdit(default_root)
        layout.addWidget(self._out_root_input)
        root_btn_row = QHBoxLayout()
        root_btn_row.setContentsMargins(0, 0, 0, 0)
        root_browse_btn = QPushButton("Browse")
        root_browse_btn.clicked.connect(self._browse_output_root)
        root_btn_row.addWidget(root_browse_btn)
        root_open_btn = QPushButton("Open")
        root_open_btn.setToolTip("Open the output root folder in the system file manager")
        root_open_btn.clicked.connect(self._open_output_root)
        root_btn_row.addWidget(root_open_btn)
        root_default_btn = QPushButton("Default")
        root_default_btn.setToolTip("Reset to <Documents>/DeepReefMap")
        root_default_btn.clicked.connect(self._reset_output_root_to_default)
        root_btn_row.addWidget(root_default_btn)
        root_btn_row.addStretch(1)
        layout.addLayout(root_btn_row)

        layout.addWidget(QLabel("Run name"))
        from datetime import datetime

        self._run_name_input = QLineEdit(datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._run_name_input.setPlaceholderText("Friendly name (e.g. barrier-reef-2026-05-20)")
        layout.addWidget(self._run_name_input)

        self._effective_dir_label = QLabel("")
        self._effective_dir_label.setStyleSheet("color: #888;")
        self._effective_dir_label.setWordWrap(True)
        layout.addWidget(self._effective_dir_label)

        layout.addWidget(QLabel("FPS"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(10)
        layout.addWidget(self._fps_spin)

        # Begin/end timestamps in seconds. Max range is filled in once a video
        # is picked via Browse and we probe its duration with cv2.
        range_row = QHBoxLayout()
        range_row.setContentsMargins(0, 0, 0, 0)
        begin_col = QVBoxLayout()
        begin_col.setContentsMargins(0, 0, 0, 0)
        begin_col.addWidget(QLabel("Begin (s)"))
        self._begin_spin = QDoubleSpinBox()
        self._begin_spin.setDecimals(2)
        self._begin_spin.setRange(0.0, 1e9)
        self._begin_spin.setSingleStep(1.0)
        self._begin_spin.setValue(0.0)
        begin_col.addWidget(self._begin_spin)
        range_row.addLayout(begin_col, 1)

        end_col = QVBoxLayout()
        end_col.setContentsMargins(0, 0, 0, 0)
        end_col.addWidget(QLabel("End (s)"))
        self._end_spin = QDoubleSpinBox()
        self._end_spin.setDecimals(2)
        self._end_spin.setRange(0.0, 1e9)
        self._end_spin.setSingleStep(1.0)
        self._end_spin.setValue(0.0)
        end_col.addWidget(self._end_spin)
        range_row.addLayout(end_col, 1)
        layout.addLayout(range_row)

        self._video_duration_s: float | None = None

        self._advanced_toggle = QCheckBox("Advanced settings")
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        layout.addWidget(self._advanced_toggle)

        self._advanced_panel = QWidget()
        adv_layout = QVBoxLayout(self._advanced_panel)
        adv_layout.setContentsMargins(12, 0, 0, 0)
        adv_layout.addWidget(QLabel("Transect length (m) — 0 disables"))
        self._transect_length = QDoubleSpinBox()
        self._transect_length.setRange(0.0, 100.0)
        self._transect_length.setDecimals(2)
        self._transect_length.setSingleStep(0.1)
        self._transect_length.setValue(0.0)
        self._transect_length.setSuffix(" m")
        adv_layout.addWidget(self._transect_length)
        adv_layout.addWidget(QLabel("Crop width (m) — 0 disables"))
        self._crop_width = QDoubleSpinBox()
        self._crop_width.setRange(0.0, 50.0)
        self._crop_width.setDecimals(2)
        self._crop_width.setSingleStep(0.1)
        self._crop_width.setValue(0.0)
        self._crop_width.setSuffix(" m")
        adv_layout.addWidget(self._crop_width)
        self._tsdf_check = QCheckBox("Enable TSDF")
        adv_layout.addWidget(self._tsdf_check)
        self._skip_seg_check = QCheckBox("Skip segmentation")
        adv_layout.addWidget(self._skip_seg_check)
        self._advanced_panel.setVisible(False)
        layout.addWidget(self._advanced_panel)

        self._submit_btn = QPushButton("Start reconstruction")
        self._submit_btn.clicked.connect(self._on_submit)
        layout.addWidget(self._submit_btn)

        self._batch_btn = QPushButton("Batch reconstruction…")
        self._batch_btn.setToolTip(
            "Run a CSV of reconstructions sequentially. "
            "Columns: videos, timestamps (begin-end seconds), transect_length, crop_width."
        )
        self._batch_btn.clicked.connect(self._on_batch_clicked)
        layout.addWidget(self._batch_btn)

        self._submit_hint = QLabel("")
        self._submit_hint.setWordWrap(True)
        self._submit_hint.setStyleSheet("color: #c84; font-style: italic;")
        layout.addWidget(self._submit_hint)

        # Sticky banner for non-fatal quality warnings emitted during a run
        # (preprocess detected mostly-background frames, missing transect line,
        # etc.). Cleared at the start of each new run.
        self._warnings_label = QLabel("")
        self._warnings_label.setWordWrap(True)
        self._warnings_label.setTextFormat(Qt.TextFormat.RichText)
        self._warnings_label.setStyleSheet(
            "background-color: #4a3a14; color: #ffd98a;"
            " border: 1px solid #8a6b1a; padding: 6px; border-radius: 3px;"
        )
        self._warnings_label.setVisible(False)
        self._run_warnings: list[str] = []
        layout.addWidget(self._warnings_label)

        # Status label and progress bar are owned by the top toolbar but
        # constructed here so they exist before _recompute_submit_state runs.
        self._status_label = QLabel("Ready. Fill the form above and click Start.")
        self._status_label.setWordWrap(True)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setToolTip("Current step progress")

        # Second bar showing total weighted progress across all phases of
        # a reconstruction or cached-run load. Driven by the active
        # ProgressModel.
        self._total_progress_bar = QProgressBar()
        self._total_progress_bar.setRange(0, 100)
        self._total_progress_bar.setValue(0)
        self._total_progress_bar.setVisible(False)
        self._total_progress_bar.setFormat("Total %p%")
        self._total_progress_bar.setToolTip("Total progress across all phases")

        self._recon_model = ProgressModel(_RECON_PHASES)
        self._load_model = ProgressModel(_LOAD_PHASES)
        self._active_progress_model: ProgressModel | None = None

        layout.addWidget(_separator())


        self._viewer_controls_group = QGroupBox("Viewer controls")
        self._viewer_controls_group.setVisible(False)
        vc_layout = QVBoxLayout(self._viewer_controls_group)

        self._semantic_check = QCheckBox("Semantic colors")
        self._semantic_check.setChecked(True)
        self._semantic_check.toggled.connect(self._on_viewer_control_changed)
        vc_layout.addWidget(self._semantic_check)

        self._accumulate_check = QCheckBox("Accumulate frames")
        self._accumulate_check.setChecked(True)
        self._accumulate_check.toggled.connect(self._on_viewer_control_changed)
        vc_layout.addWidget(self._accumulate_check)

        self._hide_frustums_check = QCheckBox("Hide frustums")
        self._hide_frustums_check.toggled.connect(self._on_viewer_control_changed)
        vc_layout.addWidget(self._hide_frustums_check)

        vc_layout.addWidget(QLabel("Point size"))
        self._point_size_spin = QDoubleSpinBox()
        self._point_size_spin.setRange(0.5, 20.0)
        self._point_size_spin.setValue(2.0)
        self._point_size_spin.setSingleStep(0.5)
        self._point_size_spin.valueChanged.connect(self._on_viewer_control_changed)
        vc_layout.addWidget(self._point_size_spin)

        vc_layout.addWidget(QLabel("Min confidence (%)"))
        self._confidence_slider = QSlider(Qt.Horizontal)
        self._confidence_slider.setRange(0, 100)
        self._confidence_slider.setValue(0)
        self._confidence_slider.valueChanged.connect(self._on_viewer_control_changed)
        vc_layout.addWidget(self._confidence_slider)

        vc_layout.addWidget(QLabel("Frame"))
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self._on_viewer_control_changed)
        vc_layout.addWidget(self._frame_slider)

        play_row = QHBoxLayout()
        self._play_check = QCheckBox("Play")
        self._play_check.toggled.connect(self._on_play_toggled)
        play_row.addWidget(self._play_check)
        play_row.addWidget(QLabel("FPS:"))
        self._play_fps_spin = QSpinBox()
        self._play_fps_spin.setRange(1, 60)
        self._play_fps_spin.setValue(8)
        self._play_fps_spin.valueChanged.connect(self._on_play_fps_changed)
        play_row.addWidget(self._play_fps_spin)
        vc_layout.addLayout(play_row)

        layout.addWidget(self._viewer_controls_group)


        self._legend_group = QGroupBox("Semantic legend")
        self._legend_group.setVisible(False)
        self._legend_layout = QVBoxLayout(self._legend_group)
        self._legend_toggles: dict[int, QCheckBox] = {}
        layout.addWidget(self._legend_group)


        self._results_group = QGroupBox("Results")
        self._results_group.setVisible(False)
        res_layout = QVBoxLayout(self._results_group)
        self._metadata_label = QLabel("")
        self._metadata_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._metadata_label.setWordWrap(True)
        self._metadata_label.setTextFormat(Qt.TextFormat.RichText)
        res_layout.addWidget(self._metadata_label)

        ortho_row = QHBoxLayout()
        ortho_row.setSpacing(4)
        self._ortho_rgb_preview = QLabel("RGB ortho")
        self._ortho_rgb_preview.setAlignment(Qt.AlignCenter)
        self._ortho_rgb_preview.setMinimumSize(160, 100)
        self._ortho_rgb_preview.setStyleSheet("background-color: #1a1a1a; color: #666;")
        ortho_row.addWidget(self._ortho_rgb_preview, 1)
        self._ortho_seg_preview = QLabel("Seg ortho")
        self._ortho_seg_preview.setAlignment(Qt.AlignCenter)
        self._ortho_seg_preview.setMinimumSize(160, 100)
        self._ortho_seg_preview.setStyleSheet("background-color: #1a1a1a; color: #666;")
        ortho_row.addWidget(self._ortho_seg_preview, 1)
        res_layout.addLayout(ortho_row)

        self._ortho_label = QLabel()
        self._ortho_label.setAlignment(Qt.AlignCenter)
        self._ortho_label.setVisible(False)
        res_layout.addWidget(self._ortho_label)

        crop_box = QGroupBox("Transect crop (live)")
        crop_box.setVisible(False)
        crop_layout = QGridLayout(crop_box)
        crop_layout.addWidget(QLabel("Transect length (m)"), 0, 0)
        self._results_transect_length = QDoubleSpinBox()
        self._results_transect_length.setRange(0.0, 100.0)
        self._results_transect_length.setDecimals(2)
        self._results_transect_length.setSingleStep(0.1)
        self._results_transect_length.setValue(0.0)
        crop_layout.addWidget(self._results_transect_length, 0, 1)
        self._results_transect_slider = QSlider(Qt.Horizontal)
        self._results_transect_slider.setRange(0, 10000)
        crop_layout.addWidget(self._results_transect_slider, 1, 0, 1, 2)
        crop_layout.addWidget(QLabel("Crop width (m)"), 2, 0)
        self._results_crop_width = QDoubleSpinBox()
        self._results_crop_width.setRange(0.0, 50.0)
        self._results_crop_width.setDecimals(2)
        self._results_crop_width.setSingleStep(0.1)
        self._results_crop_width.setValue(0.0)
        crop_layout.addWidget(self._results_crop_width, 2, 1)
        self._results_crop_slider = QSlider(Qt.Horizontal)
        self._results_crop_slider.setRange(0, 5000)
        crop_layout.addWidget(self._results_crop_slider, 3, 0, 1, 2)
        self._crop_box = crop_box
        res_layout.addWidget(crop_box)

        self._results_transect_length.valueChanged.connect(self._on_results_transect_length_changed)
        self._results_crop_width.valueChanged.connect(self._on_results_crop_width_changed)
        self._results_transect_slider.valueChanged.connect(self._on_results_transect_slider_changed)
        self._results_crop_slider.valueChanged.connect(self._on_results_crop_slider_changed)

        self._cover_label = QLabel()
        self._cover_label.setWordWrap(True)
        res_layout.addWidget(self._cover_label)

        rename_row = QHBoxLayout()
        self._rename_btn = QPushButton("Rename…")
        self._rename_btn.clicked.connect(self._begin_rename)
        rename_row.addWidget(self._rename_btn)
        self._rename_edit = QLineEdit()
        self._rename_edit.setVisible(False)
        self._rename_edit.returnPressed.connect(self._commit_rename)
        rename_row.addWidget(self._rename_edit, 1)
        self._rename_ok_btn = QPushButton("OK")
        self._rename_ok_btn.setVisible(False)
        self._rename_ok_btn.clicked.connect(self._commit_rename)
        rename_row.addWidget(self._rename_ok_btn)
        self._rename_cancel_btn = QPushButton("Cancel")
        self._rename_cancel_btn.setVisible(False)
        self._rename_cancel_btn.clicked.connect(self._cancel_rename)
        rename_row.addWidget(self._rename_cancel_btn)
        res_layout.addLayout(rename_row)

        self._open_dir_btn = QPushButton("Open output directory")
        self._open_dir_btn.clicked.connect(self._open_output_dir)
        res_layout.addWidget(self._open_dir_btn)

        exports_grid = QGridLayout()
        exports_grid.setHorizontalSpacing(6)
        exports_grid.setVerticalSpacing(4)
        self._export_ortho_npz_btn = QPushButton("Save ortho (NPZ)")
        self._export_ortho_npz_btn.clicked.connect(self._on_export_ortho_npz)
        exports_grid.addWidget(self._export_ortho_npz_btn, 0, 0)
        self._export_ortho_png_btn = QPushButton("Save ortho preview (PNG)")
        self._export_ortho_png_btn.clicked.connect(self._on_export_ortho_png)
        exports_grid.addWidget(self._export_ortho_png_btn, 0, 1)
        self._export_cover_btn = QPushButton("Save benthic cover (CSV)")
        self._export_cover_btn.clicked.connect(self._on_export_cover_csv)
        exports_grid.addWidget(self._export_cover_btn, 1, 0)
        self._export_zip_btn = QPushButton("Zip output directory")
        self._export_zip_btn.clicked.connect(self._on_export_zip)
        exports_grid.addWidget(self._export_zip_btn, 1, 1)
        res_layout.addLayout(exports_grid)
        layout.addWidget(self._results_group)

        layout.addWidget(_separator())

        layout.addWidget(QLabel("<b>Tools</b>"))
        test_btn = QPushButton("Render test cloud")
        test_btn.clicked.connect(self._render_test_cloud)
        layout.addWidget(test_btn)
        load_btn = QPushButton("Load cached run...")
        load_btn.clicked.connect(self._load_cached_run)
        layout.addWidget(load_btn)


        layout.addWidget(_separator())
        models_group = QGroupBox("Models")
        self._models_layout = QVBoxLayout(models_group)

        auth_row = QHBoxLayout()
        self._hf_auth_label = QLabel("Checking Hugging Face login...")
        self._hf_auth_label.setWordWrap(True)
        auth_row.addWidget(self._hf_auth_label, 1)

        self._hf_auth_icon = QLabel()
        self._hf_auth_icon.setFixedWidth(14)
        self._hf_auth_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        auth_row.addWidget(self._hf_auth_icon)
        auth_row.addSpacing(6)

        self._hf_auth_btn = QPushButton("Log in...")
        self._hf_auth_btn.setFixedWidth(110)
        self._hf_auth_btn.clicked.connect(self._on_hf_auth_button)
        auth_row.addWidget(self._hf_auth_btn)
        self._models_layout.addLayout(auth_row)

        self._models_layout.addWidget(_separator())

        self._models_grid_host = QWidget()
        self._models_grid = QGridLayout(self._models_grid_host)
        self._models_grid.setContentsMargins(0, 4, 0, 0)
        self._models_grid.setHorizontalSpacing(10)
        self._models_grid.setVerticalSpacing(4)
        self._models_grid.setColumnStretch(0, 1)
        self._models_grid.setColumnStretch(1, 0)
        self._models_layout.addWidget(self._models_grid_host)

        self._hf_auth_user: str | None = None
        self._model_rows: dict[str, QWidget] = {}
        self._model_actions: dict[str, QWidget] = {}
        self._downloading: set[str] = set()
        self._delete_armed: dict[str, QPushButton] = {}
        self._last_model_states: list = []

        self._seg_combo.currentTextChanged.connect(self._on_required_models_changed)
        self._map_combo.currentTextChanged.connect(self._on_required_models_changed)
        self._skip_seg_check.toggled.connect(self._on_required_models_changed)
        self._video_input.textChanged.connect(self._recompute_submit_state)
        self._video_input.editingFinished.connect(self._on_video_input_committed)
        self._out_root_input.textChanged.connect(self._on_output_root_changed)
        self._run_name_input.textChanged.connect(self._on_run_name_changed)

        self._active_run_dir: Path | None = None
        self._active_run_manifest: dict | None = None
        self._load_cancelled = False

        self._base_ortho_grid: object | None = None
        self._ortho_cloud: object | None = None
        self._ortho_classes_config: object | None = None
        self._current_ortho_grid: object | None = None
        self._results_output_dir: Path | None = None
        self._ortho_crop_refresh_pending = False

        self._settings = QSettings("ECEO", "deepreefmap")
        last_video = self._settings.value("last_video_path", "", type=str)
        if last_video and Path(last_video).exists():
            self._video_input.setText(last_video)
            self._auto_probe_video_duration(last_video)
        saved_root = self._settings.value("output_root_dir", "", type=str)
        if saved_root:
            self._out_root_input.setText(saved_root)
        self._update_effective_dir_label()
        self._refresh_past_runs_combo()
        self._recompute_submit_state()

        # Past runs are listed in the top-bar combo newest-first; the user
        # can click one to load. We don't auto-load on startup so the app
        # opens instantly. A stale half-finished last_run_dir is cleared so
        # it doesn't appear at the top of the combo as the most recent entry
        # only to error out on click.
        last_run = self._settings.value("last_run_dir", "", type=str)
        if last_run:
            last_run_path = Path(last_run)
            if not ((last_run_path / "run_manifest.json").exists()
                    and (last_run_path / "mapping_outputs.npz").exists()):
                self._settings.remove("last_run_dir")
        layout.addWidget(models_group)
        threading.Thread(target=self._refresh_model_status, daemon=True).start()


        layout.addWidget(_separator())
        self._update_label = QLabel(f"Version: <b>{_current_version()}</b>. Checking for updates...")
        self._update_label.setWordWrap(True)
        layout.addWidget(self._update_label)
        self._update_btn = QPushButton("Install update")
        self._update_btn.setVisible(False)
        self._update_btn.clicked.connect(self._on_update)
        layout.addWidget(self._update_btn)

        threading.Thread(target=self._check_for_update, daemon=True).start()

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        # Allow the user to drag the splitter to collapse the form down to a
        # small minimum. Removing the hard min lets the 3D viewport take as
        # much space as they want.
        scroll.setMinimumWidth(0)
        return scroll

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("QWidget { background-color: #2a2a2a; } ")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 6, 8, 6)
        h.setSpacing(8)

        h.addWidget(QLabel("Past runs:"))
        h.addWidget(self._past_runs_combo, 2)
        h.addWidget(self._past_open_btn)
        h.addWidget(self._new_run_btn)

        # Vertical separator between navigation and status.
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #555;")
        h.addSpacing(6)
        h.addWidget(sep)
        h.addSpacing(6)

        self._status_label.setStyleSheet("color: #ccc;")
        h.addWidget(self._status_label, 3)
        self._progress_bar.setMaximumWidth(160)
        h.addWidget(self._progress_bar)
        self._total_progress_bar.setMaximumWidth(160)
        h.addWidget(self._total_progress_bar)
        h.addWidget(self._load_cancel_btn)

        return bar

    def _cancel_load(self) -> None:
        # Soft cancel: the worker thread can't be interrupted mid-read, but
        # we set a flag so _apply_loaded_run drops the result when it eventually
        # arrives. The thread is a daemon and will exit with the process.
        self._load_cancelled = True
        self._load_cancel_btn.setVisible(False)
        self._reset_progress_bars()
        self._status_label.setText("Load cancelled.")

    # --- Unified progress plumbing ---

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


    def _on_viewer_control_changed(self) -> None:
        if not self._viewer.has_scene_data:
            return
        self._viewer.apply_state(
            timeline_t=self._frame_slider.value(),
            accumulate=self._accumulate_check.isChecked(),
            enabled_classes=self._enabled_class_set(),
            semantic_colors=self._semantic_check.isChecked(),
            point_size=self._point_size_spin.value(),
            min_confidence=self._confidence_slider.value() / 100.0,
            frustums_visible=not self._hide_frustums_check.isChecked(),
        )

    def _enabled_class_set(self) -> frozenset[int]:
        return frozenset(int(cid) for cid, cb in self._legend_toggles.items() if cb.isChecked())

    def _on_play_toggled(self, playing: bool) -> None:
        if playing:
            interval = max(16, int(1000 / max(1, self._play_fps_spin.value())))
            self._playback_timer.start(interval)
        else:
            self._playback_timer.stop()

    def _on_play_fps_changed(self) -> None:
        if self._playback_timer.isActive():
            interval = max(16, int(1000 / max(1, self._play_fps_spin.value())))
            self._playback_timer.setInterval(interval)

    def _on_playback_tick(self) -> None:
        n = self._viewer.n_frames
        if n <= 0:
            return
        nxt = (self._frame_slider.value() + 1) % n
        self._frame_slider.setValue(nxt)

    def _show_viewer_controls(self) -> None:
        n = self._viewer.n_frames
        self._frame_slider.setRange(0, max(0, n - 1))
        self._frame_slider.setValue(n - 1)
        self._viewer_controls_group.setVisible(True)


    def _build_legend(self) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._legend_toggles.clear()

        cc = self._classes_config
        for cid in sorted(cc.id_to_name.keys()):
            name = cc.id_to_name[cid]
            color = cc.id_to_color.get(cid, (128, 128, 128))
            row = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(f"background-color: rgb({color[0]},{color[1]},{color[2]}); border: 1px solid #666;")
            row.addWidget(swatch)
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.toggled.connect(self._on_viewer_control_changed)
            row.addWidget(cb, 1)
            self._legend_toggles[cid] = cb
            container = QWidget()
            container.setLayout(row)
            self._legend_layout.addWidget(container)

        self._legend_group.setVisible(True)


    def _show_results(self, output_dir: str) -> None:
        out = Path(output_dir)
        self._results_output_dir = out

        manifest_path = out / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                self._metadata_label.setText(
                    self._format_run_metadata(manifest, out, include_disk_size=True)
                )
            except Exception:
                self._metadata_label.setText("")
        else:
            self._metadata_label.setText("")

        ortho_path = out / "ortho.png"
        if ortho_path.exists():
            pixmap = QPixmap(str(ortho_path))
            scaled = pixmap.scaledToWidth(min(340, pixmap.width()), Qt.SmoothTransformation)
            self._ortho_label.setPixmap(scaled)
            self._ortho_label.setVisible(True)

        cover_path = out / "benthic_cover.json"
        if cover_path.exists():
            try:
                with open(cover_path) as f:
                    cover = json.load(f)
                self._cover_label.setText(self._format_cover_html(cover))
            except Exception:
                pass

        self._results_group.setVisible(True)

    @staticmethod
    def _format_cover_html(cover: dict) -> str:
        classes = cover.get("classes", {}) if isinstance(cover, dict) else {}
        lines = ["<b>Benthic cover:</b><br>"]
        for cid_str, info in sorted(classes.items(), key=lambda x: -x[1].get("fraction", 0)):
            name = info.get("name", cid_str)
            frac = info.get("fraction", 0)
            if frac > 0.001:
                lines.append(f"{name}: {frac * 100:.1f}%<br>")
        return "".join(lines)

    def _set_ortho_sources(
        self,
        cloud: object | None,
        base_grid: object | None,
        classes_config: object | None,
    ) -> None:
        self._ortho_cloud = cloud
        self._base_ortho_grid = base_grid
        self._ortho_classes_config = classes_config
        self._current_ortho_grid = base_grid
        if base_grid is not None:
            self._crop_box.setVisible(True)
            self._refresh_ortho_preview(base_grid)
        else:
            self._crop_box.setVisible(False)

    def _refresh_ortho_preview(self, grid: object) -> None:
        rgb = getattr(grid, "rgb", None)
        labels = getattr(grid, "labels", None)
        if rgb is None or labels is None:
            return
        self._ortho_rgb_preview.setPixmap(self._numpy_rgb_to_pixmap(rgb))
        seg_rgb = self._labels_to_rgb(labels, self._ortho_classes_config)
        self._ortho_seg_preview.setPixmap(self._numpy_rgb_to_pixmap(seg_rgb))

    @staticmethod
    def _numpy_rgb_to_pixmap(rgb: object, max_width: int = 320) -> QPixmap:
        import numpy as np
        from PySide6.QtGui import QImage

        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return QPixmap()
        img = QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(img)
        if pix.width() > max_width:
            pix = pix.scaledToWidth(max_width, Qt.SmoothTransformation)
        return pix

    @staticmethod
    def _labels_to_rgb(labels: object, classes_config: object | None):
        import numpy as np

        labels_arr = np.asarray(labels, dtype=np.int32)
        out = np.zeros((labels_arr.shape[0], labels_arr.shape[1], 3), dtype=np.uint8)
        if classes_config is None:
            return out
        for cid, color in classes_config.id_to_color.items():
            out[labels_arr == int(cid)] = np.asarray(color, dtype=np.uint8)
        return out

    def _recompute_ortho_crop(self) -> None:
        if self._base_ortho_grid is None or self._ortho_classes_config is None:
            return
        from deepreefmap.postproc.ortho_outputs import TransectCropParams, apply_ortho_crop

        tl = float(self._results_transect_length.value())
        cw = float(self._results_crop_width.value())
        crop = (
            TransectCropParams(transect_length_m=tl, crop_width_m=cw)
            if tl > 0.0 and cw > 0.0
            else None
        )
        try:
            outputs = apply_ortho_crop(self._base_ortho_grid, self._ortho_classes_config, crop=crop)
        except Exception as exc:
            self._status_label.setText(f"Crop failed: {exc}")
            return
        self._current_ortho_grid = outputs.grid
        self._refresh_ortho_preview(outputs.grid)
        self._cover_label.setText(self._format_cover_html(outputs.cover))
        self._apply_viewer_crop_filter(crop)

    def _apply_viewer_crop_filter(self, crop: object | None) -> None:
        if self._base_ortho_grid is None:
            return
        if not hasattr(self._viewer, "set_point_filter"):
            return
        if crop is None:
            self._viewer.set_point_filter(None)
            self._on_viewer_control_changed()
            return

        from deepreefmap.pointcloud.transect_crop import (
            build_transect_crop_geometry,
            build_transect_crop_selection,
            point_mask_with_transect_selection,
        )

        geometry = build_transect_crop_geometry(
            labels=self._base_ortho_grid.labels,
            transect_label=self._ortho_classes_config.single_id_for_role("transect_line"),
            transect_tools_label=self._ortho_classes_config.single_id_for_role("transect_tools"),
        )
        try:
            selection = build_transect_crop_selection(
                geometry=geometry,
                transect_length_m=crop.transect_length_m,
                crop_width_m=crop.crop_width_m,
            )
        except ValueError:
            return
        grid_ref = self._base_ortho_grid

        def _filter(xyz):
            return point_mask_with_transect_selection(grid_ref, xyz, selection)

        self._viewer.set_point_filter(_filter)
        self._on_viewer_control_changed()

    def _on_results_transect_length_changed(self, value: float) -> None:
        self._results_transect_slider.blockSignals(True)
        self._results_transect_slider.setValue(int(value * 100))
        self._results_transect_slider.blockSignals(False)
        self._recompute_ortho_crop()

    def _on_results_crop_width_changed(self, value: float) -> None:
        self._results_crop_slider.blockSignals(True)
        self._results_crop_slider.setValue(int(value * 100))
        self._results_crop_slider.blockSignals(False)
        self._recompute_ortho_crop()

    def _on_results_transect_slider_changed(self, value: int) -> None:
        self._results_transect_length.blockSignals(True)
        self._results_transect_length.setValue(value / 100.0)
        self._results_transect_length.blockSignals(False)
        self._recompute_ortho_crop()

    def _on_results_crop_slider_changed(self, value: int) -> None:
        self._results_crop_width.blockSignals(True)
        self._results_crop_width.setValue(value / 100.0)
        self._results_crop_width.blockSignals(False)
        self._recompute_ortho_crop()

    def _default_export_dir(self) -> str:
        if self._results_output_dir is not None:
            return str(self._results_output_dir)
        return self._out_root_input.text() or str(Path.home())

    def _on_export_ortho_npz(self) -> None:
        if self._current_ortho_grid is None:
            self._status_label.setText("No ortho grid available to export.")
            return
        default = str(Path(self._default_export_dir()) / "ortho.npz")
        path, _ = QFileDialog.getSaveFileName(self, "Save ortho NPZ", default, "NumPy archive (*.npz)")
        if not path:
            return
        try:
            from deepreefmap.io.exports import save_ortho_grid

            save_ortho_grid(Path(path), self._current_ortho_grid)
            self._status_label.setText(f"Saved ortho NPZ to {path}")
        except Exception as exc:
            self._status_label.setText(f"Export failed: {exc}")
            logger.exception("Failed to save ortho NPZ")

    def _on_export_ortho_png(self) -> None:
        if self._current_ortho_grid is None:
            self._status_label.setText("No ortho preview available to export.")
            return
        default = str(Path(self._default_export_dir()) / "ortho_preview.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save ortho preview PNG", default, "PNG image (*.png)")
        if not path:
            return
        try:
            import numpy as np

            grid = self._current_ortho_grid
            rgb = np.asarray(grid.rgb, dtype=np.uint8)
            seg_rgb = self._labels_to_rgb(grid.labels, self._ortho_classes_config)
            if rgb.shape[:2] != seg_rgb.shape[:2]:
                seg_rgb = np.zeros_like(rgb)
            composite = np.concatenate([rgb, seg_rgb], axis=1)
            import cv2

            cv2.imwrite(path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            self._status_label.setText(f"Saved ortho preview to {path}")
        except Exception as exc:
            self._status_label.setText(f"Export failed: {exc}")
            logger.exception("Failed to save ortho preview PNG")

    def _on_export_cover_csv(self) -> None:
        cover = self._current_cover_dict()
        if cover is None:
            self._status_label.setText("No benthic cover available to export.")
            return
        default = str(Path(self._default_export_dir()) / "benthic_cover.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Save benthic cover CSV", default, "CSV file (*.csv)")
        if not path:
            return
        try:
            import csv

            classes = cover.get("classes", {}) if isinstance(cover, dict) else {}
            rows = sorted(
                ((cid, info) for cid, info in classes.items()),
                key=lambda x: -float(x[1].get("fraction", 0.0)),
            )
            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["class_id", "name", "fraction", "count"])
                for cid, info in rows:
                    writer.writerow([
                        cid,
                        info.get("name", ""),
                        f"{float(info.get('fraction', 0.0)):.6f}",
                        info.get("count", ""),
                    ])
            self._status_label.setText(f"Saved benthic cover CSV to {path}")
        except Exception as exc:
            self._status_label.setText(f"Export failed: {exc}")
            logger.exception("Failed to save cover CSV")

    def _on_export_zip(self) -> None:
        if self._results_output_dir is None or not Path(self._results_output_dir).exists():
            self._status_label.setText("No output directory to zip.")
            return
        default = str(Path(self._default_export_dir()).parent / f"{Path(self._results_output_dir).name}.zip")
        path, _ = QFileDialog.getSaveFileName(self, "Save output as zip", default, "Zip archive (*.zip)")
        if not path:
            return
        try:
            import shutil

            base = path[:-4] if path.endswith(".zip") else path
            archive_path = shutil.make_archive(
                base_name=base,
                format="zip",
                root_dir=str(self._results_output_dir.parent),
                base_dir=self._results_output_dir.name,
            )
            self._status_label.setText(f"Saved zip archive to {archive_path}")
        except Exception as exc:
            self._status_label.setText(f"Export failed: {exc}")
            logger.exception("Failed to zip output directory")

    def _current_cover_dict(self) -> dict | None:
        if self._current_ortho_grid is not None and self._ortho_classes_config is not None:
            try:
                from deepreefmap.postproc.benthic_cover import compute_benthic_cover

                grid = self._current_ortho_grid
                return compute_benthic_cover(
                    grid.labels, classes_config=self._ortho_classes_config, counts=grid.counts
                )
            except Exception:
                logger.debug("Failed to compute live benthic cover", exc_info=True)
        if self._results_output_dir is not None:
            cover_path = self._results_output_dir / "benthic_cover.json"
            if cover_path.exists():
                try:
                    return json.loads(cover_path.read_text())
                except Exception:
                    return None
        return None


    def _refresh_model_status(self) -> None:
        from deepreefmap.launcher.model_manager import ALL_MODELS, check_hf_auth, is_model_cached

        auth_user = check_hf_auth()
        model_states = [(m, is_model_cached(m)) for m in ALL_MODELS]
        self._sig_model_status_done.emit(auth_user, model_states)

    def _apply_model_status(self, auth_user: str | None, model_states: list) -> None:
        self._hf_auth_user = auth_user
        self._last_model_states = list(model_states)
        if auth_user:
            self._hf_auth_label.setText(f"Logged in to Hugging Face as <b>{auth_user}</b>")
            self._hf_auth_label.setToolTip(
                f"Signed in to Hugging Face as {auth_user}. Click Log out to remove the saved token."
            )
            self._hf_auth_icon.setText('<span style="color:#4a4; font-weight:bold">●</span>')
            self._hf_auth_icon.setToolTip("Signed in to Hugging Face")
            self._hf_auth_btn.setText("Log out")
            self._hf_auth_btn.setEnabled(True)
        else:
            required = self._required_model_names()
            gated_required = [
                info.name for info, _cached in model_states
                if info.gated and info.name in required
            ]
            label = "Not logged in to Hugging Face"
            if gated_required:
                label += (
                    f'  <span style="color:#e8a04a">— needed for '
                    f'{", ".join(gated_required)}</span>'
                )
            self._hf_auth_label.setText(label)
            self._hf_auth_label.setToolTip(
                "Some gated models need a Hugging Face account. "
                "Click Log in… to paste an access token from huggingface.co/settings/tokens."
            )
            self._hf_auth_icon.setText('<span style="color:#e8a04a; font-weight:bold">!</span>')
            self._hf_auth_icon.setToolTip(
                "Hugging Face login required to download gated models — "
                "click Log in… to paste an access token."
            )
            self._hf_auth_btn.setText("Log in...")
            self._hf_auth_btn.setEnabled(True)

        for w in self._model_rows.values():
            w.deleteLater()
        self._model_rows.clear()
        self._model_actions.clear()
        self._delete_armed.clear()
        while self._models_grid.count():
            item = self._models_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        required = self._required_model_names()
        ordered_states = sorted(model_states, key=lambda s: s[0].name not in required)
        for row, (info, cached) in enumerate(ordered_states):
            name_html = f'<span style="color:#cfd">{info.name}</span>'
            if info.name in required:
                name_html += (
                    '&nbsp;<span style="color:#e8a04a; '
                    'font-size:10px; font-weight:bold">REQUIRED</span>'
                )
            name_label = QLabel(name_html)
            self._models_grid.addWidget(name_label, row, 0)

            action = self._make_action_widget(info, cached, auth_user)
            self._models_grid.addWidget(action, row, 1)
            self._model_rows[info.name] = name_label
            self._model_actions[info.name] = action

        self._recompute_submit_state()

    def _required_model_names(self) -> set[str]:
        required = {self._map_combo.currentText()}
        if not self._skip_seg_check.isChecked():
            required.add(self._seg_combo.currentText())
        return required

    def _on_required_models_changed(self, _value: object = "") -> None:
        if self._last_model_states:
            self._apply_model_status(self._hf_auth_user, self._last_model_states)
        self._recompute_submit_state()

    def _on_advanced_toggled(self, checked: bool) -> None:
        self._advanced_panel.setVisible(checked)

    def _recompute_submit_state(self) -> None:
        reasons: list[str] = []
        video = self._video_input.text().strip()
        if not video:
            reasons.append("pick a video file")
        elif not Path(video).exists():
            reasons.append("video file not found")
        if not self._out_root_input.text().strip():
            reasons.append("set an output root")
        if not self._run_name_input.text().strip():
            reasons.append("set a run name")

        cached_names = {info.name for info, cached in self._last_model_states if cached}
        missing = [m for m in sorted(self._required_model_names()) if m not in cached_names]
        if missing and self._last_model_states:
            reasons.append(f"download required model{'s' if len(missing) > 1 else ''}: {', '.join(missing)}")

        ok = not reasons
        self._submit_btn.setEnabled(ok)
        if ok:
            self._submit_hint.setText("")
        else:
            self._submit_hint.setText("Cannot start: " + "; ".join(reasons) + ".")

    def _make_action_widget(self, info, cached: bool, auth_user: str | None) -> QWidget:
        if info.name in self._downloading:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("Downloading %p%")
            bar.setFixedWidth(150)
            return bar

        container = QWidget()
        hb = QHBoxLayout(container)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(6)

        icon = QLabel()
        icon.setFixedWidth(14)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if cached:
            icon.setText('<span style="color:#4a4; font-weight:bold">✓</span>')
            icon.setToolTip("cached")
        elif info.gated and not auth_user:
            icon.setText('<span style="color:#e8a04a; font-weight:bold">!</span>')
            icon.setToolTip(
                "Hugging Face login required — this is a gated model. "
                "Click Log in… above to paste an access token."
            )
        else:
            icon.setText('<span style="color:#888">○</span>')
            icon.setToolTip("not downloaded")
        hb.addWidget(icon)

        if cached:
            btn = QPushButton("Delete")
            btn.setFixedWidth(110)
            btn.setToolTip(f"Delete cached files for {info.name}")
            model_name = info.name
            btn.clicked.connect(lambda checked=False, n=model_name: self._on_delete_click(n))
        elif info.gated and not auth_user:
            btn = QPushButton("Log in")
            btn.setFixedWidth(110)
            btn.clicked.connect(self._on_hf_auth_button)
        else:
            btn = QPushButton("Download")
            btn.setFixedWidth(110)
            model_name = info.name
            btn.clicked.connect(lambda checked=False, n=model_name: self._download_model(n))
        hb.addWidget(btn)
        return container

    def _on_hf_auth_button(self) -> None:
        if self._hf_auth_user:
            self._hf_auth_btn.setEnabled(False)
            self._status_label.setText("Logging out of Hugging Face...")

            def _do_logout() -> None:
                from deepreefmap.launcher.model_manager import hf_logout

                try:
                    hf_logout()
                    self._sig_hf_auth_done.emit(None, "")
                except Exception as exc:
                    self._sig_hf_auth_done.emit(self._hf_auth_user, str(exc)[:200])

            threading.Thread(target=_do_logout, daemon=True).start()
            return

        dlg = _HfLoginDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        token = dlg.token()
        if not token:
            return

        self._hf_auth_btn.setEnabled(False)
        self._status_label.setText("Logging in to Hugging Face...")

        def _do_login() -> None:
            from deepreefmap.launcher.model_manager import hf_login

            try:
                user = hf_login(token)
                self._sig_hf_auth_done.emit(user, "")
            except Exception as exc:
                self._sig_hf_auth_done.emit(None, str(exc)[:200])

        threading.Thread(target=_do_login, daemon=True).start()

    def _on_delete_click(self, model_name: str) -> None:
        # First click arms the button; second click within 3 s executes.
        container = self._model_actions.get(model_name)
        if container is None:
            return
        btn = container.findChild(QPushButton)
        if btn is None:
            return
        if self._delete_armed.get(model_name) is btn:
            self._delete_armed.pop(model_name, None)
            self._execute_delete(model_name)
            return

        self._delete_armed[model_name] = btn
        btn.setText("Confirm?")
        btn.setStyleSheet("background-color: #8a2222; color: white; font-weight: bold;")

        def _revert() -> None:
            if self._delete_armed.get(model_name) is btn:
                self._delete_armed.pop(model_name, None)
                try:
                    btn.setText("Delete")
                    btn.setStyleSheet("")
                except RuntimeError:
                    pass  # widget was destroyed by a refresh

        QTimer.singleShot(3000, _revert)

    def _execute_delete(self, model_name: str) -> None:
        from deepreefmap.launcher.model_manager import ALL_MODELS, delete_model

        info = next((m for m in ALL_MODELS if m.name == model_name), None)
        if info is None:
            return
        self._status_label.setText(f"Deleting {model_name}...")

        def _do_delete() -> None:
            try:
                removed = delete_model(info)
                if removed:
                    self._sig_status_text.emit(f"Deleted cached files for {model_name}.")
                else:
                    self._sig_status_text.emit(f"No cached revisions found for {model_name}.")
            except Exception as exc:
                self._sig_status_text.emit(f"Delete failed: {str(exc)[:200]}")
            finally:
                threading.Thread(target=self._refresh_model_status, daemon=True).start()

        threading.Thread(target=_do_delete, daemon=True).start()

    def _swap_action_to_progress(self, model_name: str) -> None:
        old = self._model_actions.get(model_name)
        if old is None:
            return
        # Locate the cell so we can drop in the progress bar at the same spot.
        idx = self._models_grid.indexOf(old)
        if idx < 0:
            return
        row, col, _, _ = self._models_grid.getItemPosition(idx)
        self._models_grid.removeWidget(old)
        old.deleteLater()
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setFormat("Downloading %p%")
        bar.setFixedWidth(130)
        self._models_grid.addWidget(bar, row, col)
        self._model_actions[model_name] = bar

    def _on_download_progress(self, model_name: str, percent: int) -> None:
        widget = self._model_actions.get(model_name)
        if isinstance(widget, QProgressBar):
            widget.setValue(max(0, min(100, percent)))

    def _on_hf_auth_done(self, user: object, error: str) -> None:
        if error:
            self._status_label.setText(f"Hugging Face auth failed: {error}")
        elif user:
            self._status_label.setText(f"Logged in to Hugging Face as {user}.")
        else:
            self._status_label.setText("Logged out of Hugging Face.")
        threading.Thread(target=self._refresh_model_status, daemon=True).start()

    def _download_model(self, model_name: str) -> None:
        from deepreefmap.launcher.model_manager import ALL_MODELS, prefetch_model

        info = next((m for m in ALL_MODELS if m.name == model_name), None)
        if info is None or model_name in self._downloading:
            return
        self._status_label.setText(f"Downloading model {model_name}...")
        self._downloading.add(model_name)
        self._swap_action_to_progress(model_name)

        def _progress(n: int, total: int) -> None:
            if total <= 0:
                return
            self._sig_download_progress.emit(model_name, int(100 * n / total))

        def _do_download() -> None:
            try:
                prefetch_model(info, progress_cb=_progress)
                self._sig_status_text.emit(f"Model {model_name} downloaded.")
            except Exception as exc:
                msg = str(exc)[:200]
                self._sig_status_text.emit(f"Download failed: {msg}")
            finally:
                self._downloading.discard(model_name)
                threading.Thread(target=self._refresh_model_status, daemon=True).start()

        threading.Thread(target=_do_download, daemon=True).start()

    def _open_output_dir(self) -> None:
        d = getattr(self, "_results_output_dir", None) or self._viewer._output_dir
        if d and Path(d).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))


    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "", "Video files (*.mp4 *.MP4 *.avi *.mov *.mkv);;All files (*)"
        )
        if path:
            self._video_input.setText(path)
            self._auto_probe_video_duration(path)

    def _auto_probe_video_duration(self, video_path: str) -> None:
        """Probe with cv2 and fill the End spinbox so the user has a sane default."""
        duration = _probe_video_duration_s(video_path)
        if duration is None:
            return
        self._video_duration_s = duration
        # Cap is generous to allow concatenated streams beyond a single file.
        self._end_spin.setMaximum(max(duration, 1e9))
        self._begin_spin.setMaximum(max(duration, 1e9))
        self._begin_spin.setValue(0.0)
        self._end_spin.setValue(duration)

    def _on_video_input_committed(self) -> None:
        """Probe duration if the user typed/pasted a path bypassing Browse."""
        path = self._video_input.text().strip()
        if path and Path(path).exists():
            self._auto_probe_video_duration(path)

    def _browse_output_root(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select output root directory", self._out_root_input.text()
        )
        if path:
            self._out_root_input.setText(path)

    def _open_output_root(self) -> None:
        root = Path(self._out_root_input.text()).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    def _reset_output_root_to_default(self) -> None:
        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        default = str(Path(documents or str(Path.home())) / "DeepReefMap")
        self._out_root_input.setText(default)

    @staticmethod
    def _sanitize_run_name(name: str) -> str:
        import re
        from datetime import datetime

        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
        return cleaned.strip("._-") or datetime.now().strftime("%Y%m%d-%H%M%S")

    def _effective_run_dir(self) -> Path:
        root = Path(self._out_root_input.text()).expanduser()
        name = self._sanitize_run_name(self._run_name_input.text())
        return root / name

    def _update_effective_dir_label(self) -> None:
        try:
            target = self._effective_run_dir()
            self._effective_dir_label.setText(f"→ {target}")
        except Exception:
            self._effective_dir_label.setText("")

    def _on_output_root_changed(self, _text: str = "") -> None:
        self._update_effective_dir_label()
        self._recompute_submit_state()
        self._settings.setValue("output_root_dir", self._out_root_input.text())
        self._refresh_past_runs_combo()

    def _on_run_name_changed(self, _text: str = "") -> None:
        self._update_effective_dir_label()
        self._recompute_submit_state()

    def _refresh_past_runs_combo(self) -> None:
        root = Path(self._out_root_input.text()).expanduser()
        entries: list[tuple[Path, str, float, dict]] = []
        if root.exists() and root.is_dir():
            for child in root.iterdir():
                manifest = child / "run_manifest.json"
                if not (child.is_dir() and manifest.exists()):
                    continue
                display = child.name
                data: dict = {}
                try:
                    data = json.loads(manifest.read_text())
                    name = data.get("name")
                    if name:
                        display = f"{name}  ({child.name})"
                except Exception:
                    pass
                entries.append((child, display, manifest.stat().st_mtime, data))
        entries.sort(key=lambda e: e[2], reverse=True)

        # Block signals to avoid triggering _on_past_run_selected during repopulation.
        self._past_runs_combo.blockSignals(True)
        try:
            self._past_runs_combo.clear()
            self._past_runs_combo.addItem("— Select a past run —", userData=None)
            for path, display, _mtime, data in entries:
                self._past_runs_combo.addItem(display, userData=str(path))
                idx = self._past_runs_combo.count() - 1
                tooltip = self._format_run_metadata(data, path, include_disk_size=False)
                self._past_runs_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
                self._past_runs_combo.setItemData(
                    idx,
                    self._build_past_run_card_meta(data, path),
                    _PAST_RUN_META_ROLE,
                )
            if self._active_run_dir is not None:
                for i in range(1, self._past_runs_combo.count()):
                    if self._past_runs_combo.itemData(i) == str(self._active_run_dir):
                        self._past_runs_combo.setCurrentIndex(i)
                        break
        finally:
            self._past_runs_combo.blockSignals(False)

    @staticmethod
    def _format_run_metadata(manifest: dict, run_dir: Path, *, include_disk_size: bool) -> str:
        """Multi-line format used in tooltips and the sidebar Results block."""
        lines: list[str] = []
        name = (manifest.get("name") or "").strip() or run_dir.name
        lines.append(f"<b>{name}</b>  <i>({run_dir.name})</i>")
        mode = manifest.get("mode")
        if mode:
            lines.append(f"Mode: {mode}")
        seg = manifest.get("segmentation_model")
        if seg:
            lines.append(f"Segmentation: {seg}")
        mapping = manifest.get("mapping_backend")
        if mapping:
            lines.append(f"Mapping: {mapping}")
        profile = manifest.get("camera_profile")
        if profile:
            lines.append(f"Camera profile: {profile}")
        frames = manifest.get("frames_processed")
        if frames is not None:
            lines.append(f"Frames: {frames}")
        sem_pts = manifest.get("semantic_reference_points")
        if sem_pts:
            lines.append(f"Semantic points: {int(sem_pts):,}")
        metric_pts = manifest.get("metric_points")
        if metric_pts:
            lines.append(f"Metric points: {int(metric_pts):,}")
        videos = manifest.get("input_videos") or []
        if videos:
            lines.append(f"Input: {', '.join(Path(v).name for v in videos)}")
        if include_disk_size:
            disk = _format_disk_size(run_dir)
            if disk:
                lines.append(f"Disk: {disk}")
        return "<br>".join(lines)

    @staticmethod
    def _build_past_run_card_meta(manifest: dict, run_dir: Path) -> dict:
        """Build a flat dict the dropdown delegate uses to paint each card."""
        name = (manifest.get("name") or "").strip() or run_dir.name
        facts: list[str] = []
        mode = manifest.get("mode")
        if mode:
            facts.append(mode)
        frames = manifest.get("frames_processed")
        if frames is not None:
            facts.append(f"{frames}f")
        seg = manifest.get("segmentation_model")
        if seg and seg != "__skip__":
            facts.append(str(seg))
        mapping = manifest.get("mapping_backend")
        if mapping:
            facts.append(str(mapping))
        sem_pts = manifest.get("semantic_reference_points")
        if sem_pts:
            n = int(sem_pts)
            if n >= 1_000_000:
                facts.append(f"{n / 1_000_000:.1f}M pts")
            elif n >= 1_000:
                facts.append(f"{n / 1_000:.0f}k pts")
            else:
                facts.append(f"{n} pts")
        videos = manifest.get("input_videos") or []
        video_line = ""
        if videos:
            names = [Path(v).name for v in videos]
            if len(names) == 1:
                video_line = f"📹 {names[0]}"
            else:
                video_line = f"📹 {names[0]} (+ {len(names) - 1} more)"
        return {
            "title": name,
            "slug": "" if name == run_dir.name else f"({run_dir.name})",
            "facts": "  ·  ".join(facts),
            "video": video_line,
        }

    @staticmethod
    def _format_run_metadata_compact(manifest: dict, run_dir: Path, *, include_disk_size: bool) -> str:
        """Single-line wrapping format used in the inline top banner."""
        name = (manifest.get("name") or "").strip() or run_dir.name
        header = (
            f'<b style="font-size:13px">{name}</b>'
            f'&nbsp;<span style="color:#7a8a99">({run_dir.name})</span>'
        )
        facts: list[str] = []
        for label, key, fmt in (
            ("Mode", "mode", str),
            ("Frames", "frames_processed", str),
            ("Segmentation", "segmentation_model", str),
            ("Mapping", "mapping_backend", str),
            ("Camera", "camera_profile", str),
            ("Semantic pts", "semantic_reference_points", lambda v: f"{int(v):,}"),
            ("Metric pts", "metric_points", lambda v: f"{int(v):,}"),
            ("Input", "input_videos", lambda v: ", ".join(Path(p).name for p in v) if v else ""),
        ):
            v = manifest.get(key)
            if v is not None and v != "" and v != []:
                facts.append(
                    f'<span style="color:#8aa0b8">{label}:</span>&nbsp;'
                    f'<span style="color:#d8e2ec">{fmt(v)}</span>'
                )
        if include_disk_size:
            disk = _format_disk_size(run_dir)
            if disk:
                facts.append(
                    f'<span style="color:#8aa0b8">Disk:</span>&nbsp;'
                    f'<span style="color:#d8e2ec">{disk}</span>'
                )
        sep = '&nbsp;<span style="color:#4a5f74">·</span>&nbsp;'
        return f"{header}&nbsp;&nbsp;{sep.join(facts)}"

    def _on_past_run_selected(self, index: int) -> None:
        if index <= 0:
            self._hide_run_meta_banner()
            return
        run_dir = self._past_runs_combo.itemData(index)
        if not run_dir:
            return
        path = Path(run_dir)
        # Show the metadata banner *immediately* from the manifest, before the
        # potentially-slow load kicks off, so the user gets instant feedback.
        manifest_path = path / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                self._show_run_meta_banner(manifest, path, include_disk_size=False)
            except Exception:
                self._hide_run_meta_banner()
        self._auto_load_run(path)

    def _show_run_meta_banner(self, manifest: dict, run_dir: Path, *, include_disk_size: bool) -> None:
        self._run_meta_banner.setText(
            self._format_run_metadata_compact(manifest, run_dir, include_disk_size=include_disk_size)
        )
        self._run_meta_banner.setVisible(True)

    def _hide_run_meta_banner(self) -> None:
        self._run_meta_banner.setVisible(False)
        self._run_meta_banner.setText("")

    def _open_selected_past_run(self) -> None:
        index = self._past_runs_combo.currentIndex()
        run_dir = self._past_runs_combo.itemData(index) if index > 0 else None
        if not run_dir:
            run_dir = str(self._active_run_dir) if self._active_run_dir else self._out_root_input.text()
        path = Path(run_dir).expanduser()
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self._status_label.setText(f"Folder not found: {path}")

    def _begin_rename(self) -> None:
        if self._active_run_dir is None:
            return
        current = ""
        if self._active_run_manifest:
            current = str(self._active_run_manifest.get("name") or "")
        if not current:
            current = self._active_run_dir.name
        self._rename_edit.setText(current)
        self._rename_btn.setVisible(False)
        self._rename_edit.setVisible(True)
        self._rename_ok_btn.setVisible(True)
        self._rename_cancel_btn.setVisible(True)
        self._rename_edit.setFocus()
        self._rename_edit.selectAll()

    def _cancel_rename(self) -> None:
        self._rename_edit.setVisible(False)
        self._rename_ok_btn.setVisible(False)
        self._rename_cancel_btn.setVisible(False)
        self._rename_btn.setVisible(True)

    def _commit_rename(self) -> None:
        if self._active_run_dir is None:
            self._cancel_rename()
            return
        new_name = self._rename_edit.text().strip()
        if not new_name:
            self._cancel_rename()
            return
        manifest_path = self._active_run_dir / "run_manifest.json"
        try:
            data = json.loads(manifest_path.read_text())
            data["name"] = new_name
            tmp = manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, manifest_path)
            self._active_run_manifest = data
            self._status_label.setText(f"Renamed run to '{new_name}'.")
            self._refresh_past_runs_combo()
        except Exception as exc:
            self._status_label.setText(f"Rename failed: {exc}")
            logger.exception("Failed to rename run")
        finally:
            self._cancel_rename()

    def _on_new_reconstruction(self) -> None:
        self._viewer._clear_scene_data()
        self._results_group.setVisible(False)
        self._legend_group.setVisible(False)
        self._viewer_controls_group.setVisible(False)
        self._hide_run_meta_banner()
        self._clear_run_warnings()
        self._active_run_dir = None
        self._active_run_manifest = None
        self._set_ortho_sources(None, None, None)
        from datetime import datetime

        self._run_name_input.setText(datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._past_runs_combo.blockSignals(True)
        self._past_runs_combo.setCurrentIndex(0)
        self._past_runs_combo.blockSignals(False)
        self._status_label.setText("Ready. Fill the form above and click Start.")

    def _on_submit(self) -> None:
        video = self._video_input.text().strip()
        if not video:
            self._status_label.setText("Error: video path is required.")
            return
        video_path = Path(video).expanduser()
        if not video_path.exists():
            self._status_label.setText(f"Error: file not found: {video_path}")
            return

        run_name = self._sanitize_run_name(self._run_name_input.text())
        # Reflect the sanitised slug back so the user sees what's actually written.
        if run_name != self._run_name_input.text():
            self._run_name_input.setText(run_name)
        out_dir = Path(self._out_root_input.text()).expanduser() / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        self._settings.setValue("last_video_path", str(video_path))
        self._settings.setValue("output_root_dir", self._out_root_input.text())
        self._settings.setValue("last_run_dir", str(out_dir))

        transect_length = self._transect_length.value() or None
        transect_crop = self._crop_width.value() or None

        begin_s, end_s = self._effective_time_range()
        kwargs = {
            "video_paths": [str(video_path)],
            "fps": self._fps_spin.value(),
            "segmentation_name": self._seg_combo.currentText(),
            "mapping_name": self._map_combo.currentText(),
            "camera_profile_name": self._profile_combo.currentText(),
            "output_dir": out_dir,
            "transect_length": transect_length,
            "transect_crop_width": transect_crop,
            "enable_viser": False,
            "enable_tsdf": self._tsdf_check.isChecked(),
            "skip_segmentation": self._skip_seg_check.isChecked(),
            "classes_path": self._classes_path,
            "run_name": run_name,
            "begin_s": begin_s,
            "end_s": end_s,
        }

        self._set_form_enabled(False)
        self._begin_progress(self._recon_model)
        self._status_label.setText("Reconstruction starting…")

        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline, args=(kwargs,), daemon=True
        )
        self._pipeline_thread.start()

    def _run_pipeline(self, kwargs: dict) -> None:
        from deepreefmap.pipeline.orchestrator import run_reconstruction

        try:
            run_reconstruction(viewer=self._viewer, **kwargs)
        except Exception as exc:
            logger.exception("Reconstruction failed")
            msg = str(exc)
            if len(msg) > 300:
                msg = msg[:300] + "..."
            self._sig_pipeline_error.emit(msg)

    def _on_pipeline_error(self, msg: str) -> None:
        self._status_label.setText(f"Failed: {msg}")
        self._reset_progress_bars()
        self._set_form_enabled(True)

    def _on_batch_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select batch CSV",
            self._out_root_input.text(),
            "CSV files (*.csv);;All files (*)",
        )
        if not path_str:
            return
        try:
            jobs = _load_batch_csv(Path(path_str))
        except Exception as exc:
            self._status_label.setText(f"Batch CSV error: {exc}")
            logger.exception("Failed to load batch CSV")
            return

        # Outputs go to `batch_out/<job_name>/` under the user's chosen
        # output root so they don't collide with regular single runs.
        base_out = Path(self._out_root_input.text()).expanduser() / "batch_out"
        base_out.mkdir(parents=True, exist_ok=True)

        self._set_form_enabled(False)
        self._batch_btn.setEnabled(False)
        self._status_label.setText(f"Batch starting: {len(jobs)} job(s)")
        self._progress_bar.setRange(0, len(jobs))
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)

        # Snapshot the form once so a user editing it mid-batch doesn't
        # produce mixed configurations across jobs.
        common = {
            "fps": self._fps_spin.value(),
            "segmentation_name": self._seg_combo.currentText(),
            "mapping_name": self._map_combo.currentText(),
            "camera_profile_name": self._profile_combo.currentText(),
            "enable_viser": False,
            "enable_tsdf": self._tsdf_check.isChecked(),
            "skip_segmentation": self._skip_seg_check.isChecked(),
            "classes_path": self._classes_path,
        }
        self._pipeline_thread = threading.Thread(
            target=self._run_batch_worker,
            args=(jobs, base_out, common),
            daemon=True,
        )
        self._pipeline_thread.start()

    def _run_batch_worker(
        self, jobs: list[_BatchJob], base_out: Path, common: dict
    ) -> None:
        from deepreefmap.pipeline.orchestrator import run_reconstruction

        ok = 0
        last_error = ""
        for idx, job in enumerate(jobs, start=1):
            self._sig_batch_progress.emit(idx, len(jobs), job.name)
            video_path = Path(job.video).expanduser()
            if not video_path.exists():
                last_error = f"row {idx}: {video_path} not found"
                logger.error("Batch %s/%s: %s", idx, len(jobs), last_error)
                continue
            out_dir = base_out / self._sanitize_run_name(job.name)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_reconstruction(
                    video_paths=[str(video_path)],
                    output_dir=out_dir,
                    transect_length=job.transect_length,
                    transect_crop_width=job.crop_width,
                    begin_s=job.begin_s,
                    end_s=job.end_s,
                    run_name=job.name,
                    viewer=None,
                    **common,
                )
                ok += 1
            except Exception as exc:
                logger.exception("Batch job %s failed", job.name)
                last_error = f"{job.name}: {exc}"
        self._sig_batch_done.emit(ok, len(jobs), last_error[:300])

    def _on_batch_progress(self, idx: int, total: int, name: str) -> None:
        self._status_label.setText(f"Batch: job {idx} of {total} — {name}")
        if self._progress_bar.maximum() != total:
            self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(idx - 1)
        self._progress_bar.setVisible(True)

    def _on_batch_done(self, ok: int, total: int, last_error: str) -> None:
        self._progress_bar.setValue(total)
        self._reset_progress_bars()
        if ok == total:
            self._status_label.setText(f"Batch complete: {ok}/{total} job(s) succeeded.")
        elif last_error:
            self._status_label.setText(
                f"Batch finished: {ok}/{total} succeeded. Last error: {last_error}"
            )
        else:
            self._status_label.setText(f"Batch finished: {ok}/{total} succeeded.")
        self._set_form_enabled(True)
        self._batch_btn.setEnabled(True)
        self._refresh_past_runs_combo()

    def _set_form_enabled(self, enabled: bool) -> None:
        for w in (
            self._video_input, self._profile_combo, self._seg_combo,
            self._map_combo, self._out_root_input, self._run_name_input,
            self._fps_spin, self._begin_spin, self._end_spin,
            self._transect_length, self._crop_width,
            self._tsdf_check, self._skip_seg_check, self._submit_btn,
            self._batch_btn,
        ):
            w.setEnabled(enabled)

    def _effective_time_range(self) -> tuple[float | None, float | None]:
        """Translate the spinboxes into (begin_s, end_s) arguments.

        A zero begin means "from start" (None). An end that equals the probed
        video duration also means "to end" (None) so the orchestrator skips
        clamping and trusts ffmpeg.
        """
        begin = float(self._begin_spin.value())
        end = float(self._end_spin.value())
        begin_arg: float | None = begin if begin > 0.0 else None
        end_arg: float | None = end if end > 0.0 else None
        if end_arg is not None and self._video_duration_s is not None:
            if abs(end_arg - self._video_duration_s) < 1e-3:
                end_arg = None
        return begin_arg, end_arg

    def _render_test_cloud(self) -> None:
        import numpy as np

        try:
            n = 500_000
            theta = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
            phi = np.random.uniform(0, np.pi, n).astype(np.float32)
            r = np.random.uniform(0.8, 1.0, n).astype(np.float32)
            xyz = np.column_stack([r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)])
            rgb = ((xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0)) * 255).astype(np.uint8)
            self._viewer.show_point_cloud(xyz, rgb, point_size=2.0)
        except Exception as exc:
            self._status_label.setText(f"Test cloud error: {exc}")
            logger.exception("Render test cloud failed")

    def _load_cached_run(self) -> None:
        run_dir = QFileDialog.getExistingDirectory(self, "Select cached run directory")
        if not run_dir:
            return
        self._auto_load_run(Path(run_dir))

    def _auto_load_run(self, run_dir: Path) -> None:
        self._load_cancelled = False
        self._status_label.setText(f"Loading run from {run_dir.name}…")
        self._begin_progress(self._load_model)
        # Indeterminate per-step bar until the first stage callback arrives.
        self._progress_bar.setRange(0, 0)
        self._load_cancel_btn.setVisible(True)
        threading.Thread(target=self._load_run_worker, args=(run_dir,), daemon=True).start()

    def _load_run_worker(self, run_dir: Path) -> None:
        try:
            from deepreefmap.pipeline.run_loader import load_cached_run

            def _cb(stage: str, cur: int, tot: int) -> None:
                self._sig_load_progress.emit(stage, cur, tot)

            result = load_cached_run(run_dir, progress_cb=_cb)
            self._sig_run_loaded.emit(result, str(run_dir), "")
        except Exception as exc:
            logger.exception("Failed to load cached run")
            self._sig_run_loaded.emit(None, str(run_dir), str(exc)[:300])

    _STAGE_LABELS = {
        "manifest": "Reading manifest",
        "classes": "Loading classes",
        "mapping": "Loading mapping outputs",
        "frames": "Loading frames",
        "cloud": "Building semantic cloud",
        "cloud_concatenating": "Concatenating point arrays",
        "cloud_replacing": "Applying replacement radius",
        "cloud_replacing_keys": "Replacement radius: computing voxel keys",
        "cloud_replacing_sort": "Replacement radius: sorting points",
        "cloud_replacing_select": "Replacement radius: selecting representatives",
        "cloud_voxelizing": "Reducing by voxel size",
        "geometry": "Loading geometry cloud",
    }

    def _on_load_progress(self, stage: str, cur: int, tot: int) -> None:
        if self._load_cancelled:
            return
        label = self._STAGE_LABELS.get(stage, stage)
        phase_key = _LOAD_STAGE_TO_PHASE.get(stage, stage)
        self._apply_progress(phase_key, label, current=cur, total=tot)

    def _apply_loaded_run(self, result: object, run_dir_str: str, error: str) -> None:
        from deepreefmap.pipeline.run_loader import GEOMETRY_ONLY_MODE

        self._load_cancel_btn.setVisible(False)

        if self._load_cancelled:
            self._reset_progress_bars()
            return

        run_dir = Path(run_dir_str)
        if error or result is None:
            self._status_label.setText(f"Error loading run: {error}")
            self._reset_progress_bars()
            return

        # The post-cloud work below all runs on the GUI thread (PyVista actor
        # creation must); the viewer emits setup_progress events that drive
        # both the per-step and the total bar via _apply_progress.
        self._apply_progress("viewer_index_cloud", "Setting up viewer", 0, 0, flush=True)

        if result.mode == GEOMETRY_ONLY_MODE:
            self._viewer.show_point_cloud(result.geometry_xyz, result.geometry_rgb)
        else:
            cloud = result.reference_cloud
            fb = result.frame_batch
            mr = result.mapping_result
            if cloud is not None and fb is not None and mr is not None:
                self._viewer.load_scene_data(fb, mr, cloud, self._classes_config)
                self._build_legend()
                self._show_viewer_controls()
                self._on_viewer_control_changed()
            elif cloud is not None:
                self._viewer.show_point_cloud(cloud.xyz, cloud.rgb)

        # Build the live ortho preview BEFORE finalising — otherwise the bars
        # hide and the user stares at a frozen UI during the PCA/aggregate
        # work. Drive both bars through the same setup_progress event keys
        # by routing the build_ortho_outputs callback through _apply_progress.
        if (
            result.mode != GEOMETRY_ONLY_MODE
            and result.reference_cloud is not None
            and len(result.reference_cloud) > 1
        ):
            try:
                from deepreefmap.postproc.ortho_outputs import build_ortho_outputs

                def _ortho_load_progress(message: str) -> None:
                    phase = _STAGE_MESSAGE_TO_PHASE.get(message, "ortho_pca")
                    self._apply_progress(phase, message, 0, 0, flush=True)

                outputs = build_ortho_outputs(
                    result.reference_cloud,
                    result.classes_config,
                    progress=_ortho_load_progress,
                )
                self._set_ortho_sources(
                    result.reference_cloud, outputs.grid, result.classes_config
                )
                self._cover_label.setText(self._format_cover_html(outputs.cover))
            except Exception:
                logger.exception("Failed to build ortho preview for cached run")
            self._results_group.setVisible(True)

        self._apply_progress("viewer_finalise", "Finalising viewer", 1, 1)
        self._reset_progress_bars()

        self._active_run_dir = run_dir
        self._active_run_manifest = result.manifest
        display = result.manifest.get("name") or run_dir.name
        self._status_label.setText(f"Loaded run '{display}' from {run_dir}")
        # Refresh the banner now that the full load is done, including disk size.
        self._show_run_meta_banner(result.manifest, run_dir, include_disk_size=True)

        ortho_path = run_dir / "ortho.png"
        if ortho_path.exists():
            self._show_results(str(run_dir))
        else:
            # No ortho (e.g. geometry-only run) — still surface the metadata
            # block by showing the Results group with just the details.
            self._metadata_label.setText(
                self._format_run_metadata(result.manifest, run_dir, include_disk_size=True)
            )
            self._results_output_dir = run_dir


    def _add_run_warning(self, message: str) -> None:
        if message in self._run_warnings:
            return
        self._run_warnings.append(message)
        html = "<b>Quality warnings:</b><br>" + "<br>".join(
            f"• {w}" for w in self._run_warnings
        )
        self._warnings_label.setText(html)
        self._warnings_label.setVisible(True)

    def _clear_run_warnings(self) -> None:
        self._run_warnings = []
        self._warnings_label.setText("")
        self._warnings_label.setVisible(False)

    def _on_viewer_status(self, event: str, **kwargs: object) -> None:
        _STAGE_LABELS = {
            "startup": "Startup",
            "preprocess": "Preprocessing",
            "mapping": "Mapping",
            "outputs": "Building outputs",
        }
        if event == "start_run":
            self._clear_run_warnings()
            self._apply_progress("startup", "Starting reconstruction", 0, 0)
        elif event == "set_stage":
            stage = str(kwargs.get("stage", ""))
            status = str(kwargs.get("status", ""))
            message = str(kwargs.get("message", "") or "")
            # Finer phase routing when the orchestrator's set_stage message
            # names a known sub-step (e.g. "Computing PCA projection" inside
            # "outputs"); otherwise fall back to the top-level stage key.
            phase_key = _STAGE_MESSAGE_TO_PHASE.get(message, stage)
            stage_label = _STAGE_LABELS.get(stage, stage)
            label = message or stage_label
            if status == "completed":
                self._apply_progress(phase_key, f"{stage_label} complete", 1, 1)
            elif status == "warning":
                if message:
                    self._add_run_warning(str(message))
            else:
                self._apply_progress(phase_key, label, 0, 0)
        elif event == "update_progress":
            current = int(kwargs.get("current", 0) or 0)
            total = int(kwargs.get("total", 0) or 0)
            stage = str(kwargs.get("stage", ""))
            label = _STAGE_LABELS.get(stage, stage) or "Working"
            if total:
                self._apply_progress(stage, label, current, total)
        elif event == "data_ready":
            if self._viewer.has_scene_data:
                self._build_legend()
                self._show_viewer_controls()
                self._on_viewer_control_changed()
            self._apply_progress("viewer_finalise", "Reconstruction complete", 1, 1)
            ortho_cloud = kwargs.get("ortho_cloud")
            cc = kwargs.get("classes_config") or self._classes_config
            if ortho_cloud is not None and len(ortho_cloud) > 1:
                try:
                    from deepreefmap.postproc.ortho_outputs import build_ortho_outputs

                    outputs = build_ortho_outputs(ortho_cloud, cc)
                    self._set_ortho_sources(ortho_cloud, outputs.grid, cc)
                    self._cover_label.setText(self._format_cover_html(outputs.cover))
                except Exception:
                    logger.exception("Failed to build live ortho preview")
        elif event == "setup_progress":
            message = str(kwargs.get("message", "Setting up viewer"))
            current = int(kwargs.get("current", 0) or 0)
            total = int(kwargs.get("total", 0) or 0)
            phase_key = _SETUP_MESSAGE_TO_PHASE.get(message, "viewer_index_cloud")
            # flush=True because viewer-setup happens on the GUI thread: without
            # an explicit processEvents the user sees the bars freeze.
            self._apply_progress(phase_key, message, current, total, flush=True)
        elif event == "mark_outputs":
            output_dir = kwargs.get("output_dir", "")
            self._status_label.setText(f"Outputs saved to {output_dir}")
            self._reset_progress_bars()
            self._set_form_enabled(True)
            if output_dir:
                self._show_results(str(output_dir))
                self._active_run_dir = Path(str(output_dir))
                manifest_path = self._active_run_dir / "run_manifest.json"
                if manifest_path.exists():
                    try:
                        self._active_run_manifest = json.loads(manifest_path.read_text())
                    except Exception:
                        self._active_run_manifest = None
                self._settings.setValue("last_run_dir", str(self._active_run_dir))
                self._refresh_past_runs_combo()
        elif event == "fail_run":
            error = kwargs.get("error_message", "unknown error")
            self._status_label.setText(f"Failed: {error}")
            self._reset_progress_bars()
            self._set_form_enabled(True)


    def _check_for_update(self) -> None:
        current = _current_version()
        versions = _fetch_release_versions()
        pyapp_bin = _pyapp_binary_path()
        self._sig_update_check_done.emit(current, versions, pyapp_bin)

    def _apply_update_check(self, current: str, versions: list[str] | None, pyapp_bin: str | None) -> None:
        if versions is None:
            self._update_label.setText(f"Version: <b>{current}</b>. Couldn't reach GitHub.")
            return
        if not versions:
            self._update_label.setText(f"Version: <b>{current}</b>. No releases found.")
            return
        latest = versions[0]
        if latest == current:
            self._update_label.setText(f"Version: <b>{current}</b> (up to date).")
        elif pyapp_bin:
            self._update_label.setText(
                f"Version: <b>{current}</b>. Latest: <b>{latest}</b>. "
                f"Available: {', '.join(versions[:5])}"
            )
            self._update_btn.setVisible(True)
            self._update_btn.setText(f"Install {latest}")
            self._update_target_version = latest
        else:
            self._update_label.setText(
                f"Version: <b>{current}</b>. Latest: <b>{latest}</b> (not running from installer)."
            )

    def _on_update(self) -> None:
        import subprocess

        pyapp_bin = _pyapp_binary_path()
        if pyapp_bin is None:
            logger.warning("Update button clicked but no PyApp binary detected")
            return
        self._update_btn.setEnabled(False)
        if os.environ.get("DEEPREEFMAP_MOCK_PYAPP"):
            logger.info("would have run pyapp self update")
            QMessageBox.information(self, "Update", "Update simulated successfully (mock mode).")
            return
        try:
            subprocess.Popen([pyapp_bin, "self", "update"])
        except Exception as exc:
            logger.exception("Failed to launch pyapp self update")
            QMessageBox.warning(self, "Update", f"Failed to launch updater: {exc!r}")
            self._update_btn.setEnabled(True)
            return
        QMessageBox.information(
            self,
            "Update",
            "Update is running in the background. Please relaunch the app once it completes.",
        )


_PAST_RUN_META_ROLE = Qt.ItemDataRole.UserRole + 1


class _PastRunCardDelegate(QStyledItemDelegate):
    """Paints each past-run dropdown item as a multi-line card with metadata.

    All sizes are computed from QFontMetrics so the card scales correctly with
    system DPI / font size on both Linux and Windows. Facts wrap to multiple
    lines if the popup is narrower than the text, so nothing is silently lost.
    """

    PAD_X_EMS = 0.8
    PAD_Y_EMS = 0.35
    GAP_EMS = 0.15

    @staticmethod
    def _title_font(base: QFont) -> QFont:
        f = QFont(base)
        f.setBold(True)
        return f

    @staticmethod
    def _slug_font(base: QFont) -> QFont:
        f = QFont(base)
        pt = base.pointSize() if base.pointSize() > 0 else 10
        f.setPointSize(max(8, pt - 1))
        return f

    @staticmethod
    def _facts_font(base: QFont) -> QFont:
        return QFont(base)

    @staticmethod
    def _video_font(base: QFont) -> QFont:
        f = QFont(base)
        pt = base.pointSize() if base.pointSize() > 0 else 10
        f.setPointSize(max(8, pt - 1))
        f.setItalic(True)
        return f

    def _layout(self, option: QStyleOptionViewItem, meta: dict, avail_w: int) -> dict:
        base = option.font
        title_fm = option.fontMetrics  # used for em sizing
        em = max(1, title_fm.height())
        pad_x = int(self.PAD_X_EMS * em)
        pad_y = int(self.PAD_Y_EMS * em)
        gap = int(self.GAP_EMS * em)

        from PySide6.QtGui import QFontMetrics

        title_h = QFontMetrics(self._title_font(base)).height()
        slug_h = QFontMetrics(self._slug_font(base)).height() if meta.get("slug") else 0
        head_h = max(title_h, slug_h)

        facts_text = meta.get("facts") or ""
        facts_h = 0
        if facts_text:
            facts_fm = QFontMetrics(self._facts_font(base))
            inner_w = max(40, avail_w - pad_x * 2)
            facts_rect = facts_fm.boundingRect(
                0, 0, inner_w, 10_000,
                Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                facts_text,
            )
            facts_h = facts_rect.height()

        video_text = meta.get("video") or ""
        video_h = 0
        if video_text:
            video_h = QFontMetrics(self._video_font(base)).height()

        total_h = pad_y * 2 + head_h
        if facts_h:
            total_h += gap + facts_h
        if video_h:
            total_h += gap + video_h

        return {
            "pad_x": pad_x, "pad_y": pad_y, "gap": gap,
            "head_h": head_h, "title_h": title_h, "slug_h": slug_h,
            "facts_h": facts_h, "video_h": video_h, "total_h": total_h,
        }

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        meta = index.data(_PAST_RUN_META_ROLE)
        em = max(1, option.fontMetrics.height())
        # Preferred width: ~32 chars of body text plus padding. The view can
        # be wider; the layout will fill it. EM-based so it scales with DPI.
        preferred_w = int(em * 24)
        if meta is None:
            # Placeholder row stays one line tall.
            return QSize(preferred_w, em + int(self.PAD_Y_EMS * em) * 2)

        # Use the actual viewport width when available; fall back to preferred.
        avail_w = option.rect.width() if option.rect.width() > 0 else preferred_w
        layout = self._layout(option, meta, avail_w)
        return QSize(preferred_w, layout["total_h"])

    def paint(self, painter, option: QStyleOptionViewItem, index) -> None:
        meta = index.data(_PAST_RUN_META_ROLE)
        if meta is None:
            super().paint(painter, option, index)
            return

        painter.save()

        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if selected:
            painter.fillRect(option.rect, QColor("#4a7fb0"))
        elif hovered:
            painter.fillRect(option.rect, QColor("#3a5f8a"))
        else:
            painter.fillRect(option.rect, QColor("#2a2a2a"))

        layout = self._layout(option, meta, option.rect.width())
        pad_x = layout["pad_x"]
        pad_y = layout["pad_y"]
        gap = layout["gap"]
        r = option.rect.adjusted(pad_x, pad_y, -pad_x, -pad_y)

        base = option.font

        # Title.
        title_font = self._title_font(base)
        painter.setFont(title_font)
        painter.setPen(QColor("white" if (hovered or selected) else "#e8eef5"))
        title = meta.get("title", "")
        title_fm = painter.fontMetrics()
        title_w = title_fm.horizontalAdvance(title)
        baseline = r.top() + title_fm.ascent()
        painter.drawText(r.left(), baseline, title)

        # Slug, drawn on the same baseline as the title (or hidden if no room).
        slug = meta.get("slug", "")
        if slug:
            slug_font = self._slug_font(base)
            painter.setFont(slug_font)
            painter.setPen(QColor("#c5d0db" if (hovered or selected) else "#8aa0b8"))
            slug_fm = painter.fontMetrics()
            slug_x = r.left() + title_w + int(layout["title_h"] * 0.4)
            slug_max_w = r.right() - slug_x
            if slug_max_w > 0:
                elided_slug = slug_fm.elidedText(slug, Qt.TextElideMode.ElideRight, slug_max_w)
                painter.drawText(slug_x, baseline, elided_slug)

        # Facts (word-wrapped block).
        cursor_y = r.top() + layout["head_h"]
        facts_text = meta.get("facts", "")
        if facts_text:
            cursor_y += gap
            painter.setFont(self._facts_font(base))
            painter.setPen(QColor("#dfe6ee" if (hovered or selected) else "#c0cad6"))
            facts_rect = type(r)(r.left(), cursor_y, r.width(), layout["facts_h"])
            painter.drawText(
                facts_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                facts_text,
            )
            cursor_y += layout["facts_h"]

        # Input video (single elided line).
        video = meta.get("video", "")
        if video:
            cursor_y += gap
            painter.setFont(self._video_font(base))
            painter.setPen(QColor("#b5c2d0" if (hovered or selected) else "#7a8a99"))
            video_fm = painter.fontMetrics()
            elided = video_fm.elidedText(video, Qt.TextElideMode.ElideMiddle, r.width())
            painter.drawText(r.left(), cursor_y + video_fm.ascent(), elided)

        painter.restore()


class _BatchJob:
    """One row of a batch reconstruction CSV.

    Job name defaults to the video filename stem so outputs land in a
    predictable folder per row.
    """

    __slots__ = ("video", "begin_s", "end_s", "transect_length", "crop_width", "name")

    def __init__(
        self,
        video: str,
        begin_s: float | None,
        end_s: float | None,
        transect_length: float | None,
        crop_width: float | None,
        name: str,
    ) -> None:
        self.video = video
        self.begin_s = begin_s
        self.end_s = end_s
        self.transect_length = transect_length
        self.crop_width = crop_width
        self.name = name


def _parse_optional_float(raw: str) -> float | None:
    s = (raw or "").strip()
    if not s:
        return None
    return float(s)


def _parse_timestamp_range(raw: str) -> tuple[float | None, float | None]:
    """Parse "<begin>-<end>" in seconds. Either side may be empty.

    Accepts a single value like "30" as a begin with no end. The dash splits
    on the *last* `-` so that ranges with negative-zero inputs would still
    fail loudly via float() rather than silently parse.
    """
    s = (raw or "").strip()
    if not s:
        return None, None
    if "-" not in s:
        return _parse_optional_float(s), None
    head, _, tail = s.partition("-")
    return _parse_optional_float(head), _parse_optional_float(tail)


def _load_batch_csv(path: Path) -> list[_BatchJob]:
    """Read a CSV with case-insensitive columns and return parsed rows.

    Required columns: videos, timestamps, transect_length, crop_width.
    Raises ValueError on missing columns or unparseable values.
    """
    import csv

    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        raise ValueError(
            "Excel files aren't supported (pandas isn't a dependency). "
            "Save the sheet as CSV and try again."
        )
    required = {"videos", "timestamps", "transect_length", "crop_width"}
    jobs: list[_BatchJob] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        norm = {fn.strip().lower(): fn for fn in reader.fieldnames}
        missing = required - set(norm.keys())
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        for n, row in enumerate(reader, start=2):
            video = (row.get(norm["videos"], "") or "").strip()
            if not video:
                continue  # skip blank rows
            try:
                begin_s, end_s = _parse_timestamp_range(row.get(norm["timestamps"], ""))
                transect_length = _parse_optional_float(row.get(norm["transect_length"], ""))
                crop_width = _parse_optional_float(row.get(norm["crop_width"], ""))
            except ValueError as exc:
                raise ValueError(f"Row {n}: {exc}") from exc
            jobs.append(
                _BatchJob(
                    video=video,
                    begin_s=begin_s,
                    end_s=end_s,
                    transect_length=transect_length,
                    crop_width=crop_width,
                    name=Path(video).stem or f"job_{n - 1}",
                )
            )
    if not jobs:
        raise ValueError("No usable rows in CSV.")
    return jobs


def _probe_video_duration_s(video_path: str) -> float | None:
    """Return seconds via cv2 frame count / fps, or None on failure."""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        try:
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        if not fps or fps <= 0 or not frames or frames <= 0:
            return None
        return float(frames) / float(fps)
    except Exception:
        logger.warning("Failed to probe video duration", exc_info=True)
        return None


def _format_disk_size(run_dir: Path) -> str | None:
    try:
        total = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    except Exception:
        return None
    if total >= 1e9:
        return f"{total / 1e9:.2f} GB"
    return f"{total / 1e6:.1f} MB"


def _separator() -> QWidget:
    line = QWidget()
    line.setFixedHeight(1)
    line.setStyleSheet("background-color: #555;")
    return line


class _HfLoginDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Log in to Hugging Face")
        self.setModal(True)

        layout = QVBoxLayout(self)
        intro = QLabel(
            'Paste an access token from '
            '<a href="https://huggingface.co/settings/tokens">'
            'huggingface.co/settings/tokens</a>. '
            "A read token is enough for gated models."
        )
        intro.setOpenExternalLinks(True)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._token_edit = QLineEdit()
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setPlaceholderText("hf_...")
        layout.addWidget(self._token_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.resize(420, 140)

    def token(self) -> str:
        return self._token_edit.text().strip()


def launch(classes_path: Path | None = None, view_run_dir: Path | None = None) -> None:
    from deepreefmap.config.classes import DEFAULT_CLASSES_PATH, load_classes

    if classes_path is None:
        classes_path = DEFAULT_CLASSES_PATH
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ.setdefault("QT_OPENGL", "desktop")
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    qt_app = QApplication.instance() or QApplication(sys.argv)
    from importlib import resources
    icon_path = resources.files("deepreefmap.resources").joinpath("icon.png")
    qt_app.setWindowIcon(QIcon(str(icon_path)))
    classes_config = load_classes(classes_path)
    window = DeepReefMapWindow(classes_config, classes_path)
    window.show()
    if view_run_dir is not None:
        QTimer.singleShot(100, lambda: window._auto_load_run(view_run_dir))

    # Qt's exec() blocks in C++, so Python's SIGINT handler can't fire until
    # the event loop yields. Install a handler that closes the app, and run a
    # no-op timer to wake the interpreter every 200 ms so the handler runs.
    signal.signal(signal.SIGINT, lambda *_: qt_app.quit())
    sigint_heartbeat = QTimer()
    sigint_heartbeat.start(200)
    sigint_heartbeat.timeout.connect(lambda: None)

    sys.exit(qt_app.exec())
