from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.launcher.log_view import LogView, install_qt_log_handler
from deepreefmap.launcher.qt_app_past_runs import _PastRunCardDelegate
from deepreefmap.launcher.qt_app_progress import (
    _LOAD_PHASES,
    _RECON_PHASES,
    ProgressModel,
)
from deepreefmap.launcher.qt_app_version import _current_version
from deepreefmap.visualization.sunburst_widget import SunburstWidget

logger = logging.getLogger(__name__)


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


def _separator() -> QWidget:
    line = QWidget()
    line.setFixedHeight(1)
    line.setStyleSheet("background-color: #555;")
    return line


class FormPanelMixin:
    """DeepReefMapWindow methods that build and drive the sidebar form panel + top toolbar."""

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

        # Four-tab sidebar: Run (setup form / live log), Results (viewer
        # controls + results panel for a loaded run), Models (HF auth +
        # per-model download/delete), Tools (utilities + update check).
        self._TAB_RUN = 0
        self._TAB_RESULTS = 1
        self._TAB_MODELS = 2
        self._TAB_TOOLS = 3
        self._sidebar_tabs = QTabWidget()
        # Tabs expand to share the panel width equally so labels of different
        # length (Run / Results / Models / Tools) end up the same visible width.
        self._sidebar_tabs.tabBar().setExpanding(True)
        self._sidebar_tabs.setStyleSheet(
            "QTabBar::tab { min-width: 70px; padding: 6px 10px; }"
        )
        self._run_tab = QWidget()
        run_layout = QVBoxLayout(self._run_tab)
        run_layout.setContentsMargins(4, 6, 4, 4)
        run_layout.setAlignment(Qt.AlignTop)
        self._viewer_tab = QWidget()
        viewer_layout = QVBoxLayout(self._viewer_tab)
        viewer_layout.setContentsMargins(4, 6, 4, 4)
        viewer_layout.setAlignment(Qt.AlignTop)
        self._models_tab = QWidget()
        models_layout = QVBoxLayout(self._models_tab)
        models_layout.setContentsMargins(4, 6, 4, 4)
        models_layout.setAlignment(Qt.AlignTop)
        self._tools_tab = QWidget()
        tools_layout = QVBoxLayout(self._tools_tab)
        tools_layout.setContentsMargins(4, 6, 4, 4)
        tools_layout.setAlignment(Qt.AlignTop)
        self._sidebar_tabs.addTab(self._run_tab, "Run")
        self._sidebar_tabs.addTab(self._viewer_tab, "Results")
        self._sidebar_tabs.addTab(self._models_tab, "Models")
        self._sidebar_tabs.addTab(self._tools_tab, "Tools")
        # Results tab has nothing to show until a run loads — disable it so
        # the tab is greyed out and unclickable until _show_viewer_controls
        # runs.
        self._sidebar_tabs.setTabEnabled(self._TAB_RESULTS, False)
        layout.addWidget(self._sidebar_tabs)

        # Run tab swaps between setup form and the live log via this stack.
        # The VIEWING app mode does NOT swap the stack — it leaves the form
        # in place and switches the sidebar to the Results tab instead.
        self._mode_stack = QStackedWidget()
        self._setup_page = QWidget()
        setup_layout = QVBoxLayout(self._setup_page)
        setup_layout.setAlignment(Qt.AlignTop)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        self._running_page = QWidget()
        running_layout = QVBoxLayout(self._running_page)
        running_layout.setAlignment(Qt.AlignTop)
        running_layout.setContentsMargins(0, 0, 0, 0)
        self._mode_stack.addWidget(self._setup_page)
        self._mode_stack.addWidget(self._running_page)
        run_layout.addWidget(self._mode_stack)

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

        setup_layout.addWidget(QLabel("<b>New reconstruction</b>"))

        video_row = QHBoxLayout()
        self._video_input = QLineEdit()
        self._video_input.setPlaceholderText("Path to video file")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_video)
        video_row.addWidget(self._video_input, 1)
        video_row.addWidget(browse_btn)
        setup_layout.addLayout(video_row)

        setup_layout.addWidget(QLabel("Camera profile"))
        self._profile_combo = QComboBox()
        self._profile_combo.addItems(profiles)
        setup_layout.addWidget(self._profile_combo)

        setup_layout.addWidget(QLabel("Segmentation"))
        seg_row = QHBoxLayout()
        seg_row.setContentsMargins(0, 0, 0, 0)
        seg_row.setSpacing(4)
        self._seg_combo = QComboBox()
        self._seg_combo.addItems(seg_models)
        idx = self._seg_combo.findText("segformer-b2")
        if idx >= 0:
            self._seg_combo.setCurrentIndex(idx)
        seg_row.addWidget(self._seg_combo, 1)
        self._seg_status_btn = self._build_model_status_button(self._seg_combo)
        seg_row.addWidget(self._seg_status_btn)
        setup_layout.addLayout(seg_row)

        setup_layout.addWidget(QLabel("Mapping"))
        map_row = QHBoxLayout()
        map_row.setContentsMargins(0, 0, 0, 0)
        map_row.setSpacing(4)
        self._map_combo = QComboBox()
        self._map_combo.addItems(map_backends)
        idx = self._map_combo.findText("scsfmlearner")
        if idx >= 0:
            self._map_combo.setCurrentIndex(idx)
        map_row.addWidget(self._map_combo, 1)
        self._map_status_btn = self._build_model_status_button(self._map_combo)
        map_row.addWidget(self._map_status_btn)
        setup_layout.addLayout(map_row)

        setup_layout.addWidget(QLabel("Output root"))
        self._out_root_input = QLineEdit(default_root)
        setup_layout.addWidget(self._out_root_input)
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
        setup_layout.addLayout(root_btn_row)

        setup_layout.addWidget(QLabel("Run name"))
        from datetime import datetime

        self._run_name_input = QLineEdit(datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._run_name_input.setPlaceholderText("Friendly name (e.g. barrier-reef-2026-05-20)")
        setup_layout.addWidget(self._run_name_input)

        self._effective_dir_label = QLabel("")
        self._effective_dir_label.setStyleSheet("color: #888;")
        self._effective_dir_label.setWordWrap(True)
        setup_layout.addWidget(self._effective_dir_label)

        setup_layout.addWidget(QLabel("FPS"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(10)
        setup_layout.addWidget(self._fps_spin)

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
        setup_layout.addLayout(range_row)

        self._video_duration_s: float | None = None

        self._advanced_toggle = QCheckBox("Advanced settings")
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        setup_layout.addWidget(self._advanced_toggle)

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
        setup_layout.addWidget(self._advanced_panel)

        self._submit_btn = QPushButton("Start reconstruction")
        self._submit_btn.clicked.connect(self._on_submit)
        setup_layout.addWidget(self._submit_btn)

        self._batch_btn = QPushButton("Batch reconstruction…")
        self._batch_btn.setToolTip(
            "Run a CSV of reconstructions sequentially. "
            "Columns: videos, timestamps (begin-end seconds), transect_length, crop_width."
        )
        self._batch_btn.clicked.connect(self._on_batch_clicked)
        setup_layout.addWidget(self._batch_btn)

        self._submit_hint = QLabel("")
        self._submit_hint.setWordWrap(True)
        self._submit_hint.setStyleSheet("color: #c84; font-style: italic;")
        setup_layout.addWidget(self._submit_hint)

        # Sticky banner for non-fatal quality warnings emitted during a run
        # (preprocess detected mostly-background frames, missing transect line,
        # etc.). Cleared at the start of each new run. Lives on the Results
        # tab so it surfaces alongside the results it warns about, plus is
        # mirrored on the running page so the user sees them as they happen.
        self._warnings_label = QLabel("")
        self._warnings_label.setWordWrap(True)
        self._warnings_label.setTextFormat(Qt.TextFormat.RichText)
        self._warnings_label.setStyleSheet(
            "background-color: #4a3a14; color: #ffd98a;"
            " border: 1px solid #8a6b1a; padding: 6px; border-radius: 3px;"
        )
        self._warnings_label.setVisible(False)
        self._run_warnings: list[str] = []
        viewer_layout.addWidget(self._warnings_label)

        # Running-page content: a sibling warnings label (Qt widgets can only
        # have one parent, so we use a second label and keep both texts in sync
        # via _refresh_run_warnings_view) plus the live log panel.
        self._warnings_label_running = QLabel("")
        self._warnings_label_running.setWordWrap(True)
        self._warnings_label_running.setTextFormat(Qt.TextFormat.RichText)
        self._warnings_label_running.setStyleSheet(self._warnings_label.styleSheet())
        self._warnings_label_running.setVisible(False)
        running_layout.addWidget(self._warnings_label_running)

        running_layout.addWidget(QLabel("<b>Live log</b>"))
        self._log_view = LogView()
        running_layout.addWidget(self._log_view, 1)

        # The log handler streams every deepreefmap.* log line into the panel.
        # A per-run FileHandler is opened/closed in _begin_pipeline_run and
        # cleanup paths.
        self._qt_log_handler = install_qt_log_handler()
        self._qt_log_handler.line_signal.connect(self._log_view.append_line)
        self._run_log_file_handler = None

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

        # Results tab: viewer controls + results panel. The tab itself is
        # disabled until a run is loaded (greyed out and unclickable), so no
        # empty-state placeholder is needed inside. addStretch is appended at
        # the end after the results group is added below.
        viewer_layout.addWidget(self._viewer_controls_group)

        # The legend lives as a floating overlay on the 3D canvas; this dict
        # is populated by _build_legend and queried by _enabled_class_set.
        self._legend_toggles: dict[int, QCheckBox] = {}
        self._legend_solo_buttons: dict[int, object] = {}

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

        # Two-ring sunburst (outer = fine classes, inner = coarse groups)
        # appears above the HTML cover table so the user gets a visual sense of
        # composition at a glance. Updates live with the transect crop.
        self._cover_sunburst = SunburstWidget()
        res_layout.addWidget(self._cover_sunburst, 1)

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
        self._export_qc_video_btn = QPushButton("Render QC video (MP4)")
        self._export_qc_video_btn.clicked.connect(self._on_export_qc_video)
        exports_grid.addWidget(self._export_qc_video_btn, 2, 0)
        self._export_frame_btn = QPushButton("Save current frame (PNG)")
        self._export_frame_btn.clicked.connect(self._on_export_current_frame)
        exports_grid.addWidget(self._export_frame_btn, 2, 1)
        res_layout.addLayout(exports_grid)
        viewer_layout.addWidget(self._results_group)
        viewer_layout.addStretch()

        tools_layout.addWidget(QLabel("<b>Tools</b>"))
        test_btn = QPushButton("Render test cloud")
        test_btn.clicked.connect(self._render_test_cloud)
        tools_layout.addWidget(test_btn)
        load_btn = QPushButton("Load cached run...")
        load_btn.clicked.connect(self._load_cached_run)
        tools_layout.addWidget(load_btn)

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
        # Models groupbox lives in its own sidebar tab so the Run tab stays
        # focused on the setup form. The inline status icons next to the
        # seg/mapping dropdowns surface state without forcing the user to
        # switch tabs for common cases.
        self._models_group = models_group
        models_layout.addWidget(models_group)
        models_layout.addStretch()
        threading.Thread(target=self._refresh_model_status, daemon=True).start()


        tools_layout.addWidget(_separator())
        self._update_version_label = QLabel(f"Version: <b>{_current_version()}</b>")
        self._update_version_label.setWordWrap(True)
        tools_layout.addWidget(self._update_version_label)
        self._update_status_label = QLabel("Checking for updates…")
        self._update_status_label.setWordWrap(True)
        self._update_status_label.setStyleSheet("color: #aaa;")
        tools_layout.addWidget(self._update_status_label)
        update_row = QHBoxLayout()
        self._update_version_combo = QComboBox()
        self._update_version_combo.setVisible(False)
        update_row.addWidget(self._update_version_combo, 1)
        self._update_btn = QPushButton("Install")
        self._update_btn.setVisible(False)
        self._update_btn.clicked.connect(self._on_update)
        update_row.addWidget(self._update_btn)
        tools_layout.addLayout(update_row)
        self._available_releases: list[dict] = []

        threading.Thread(target=self._check_for_update, daemon=True).start()

        tools_layout.addStretch()

        # Start in SETUP — no run loaded yet. The mode flips to RUNNING in
        # _begin_pipeline_run and to VIEWING when a past run is selected or a
        # reconstruction completes.
        self._set_app_mode("SETUP")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        # NoFrame removes the QScrollArea's default beveled border so the
        # sidebar blends into the main window instead of looking like a panel
        # inside a panel.
        scroll.setFrameShape(QFrame.NoFrame)
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
