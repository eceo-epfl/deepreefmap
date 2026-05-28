from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QSettings, QSize, QStandardPaths, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QStandardItemModel
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
    QStyle,
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
from deepreefmap.launcher.theme import WARN_BG, WARN_BORDER, WARN_TEXT
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
        from deepreefmap.mapping.registry import (
            LOGER_INSTALL_HINT,
            list_mapping_backends,
            loger_available,
        )
        from deepreefmap.segmentation.registry import (
            get_model_resolution,
            list_segmentation_models,
        )

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
        self._TAB_UPDATES = 3
        self._sidebar_tabs = QTabWidget()
        # Tabs expand to share the panel width equally so labels of different
        # length (Run / Results / Models / Updates) end up the same visible width.
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
        self._updates_tab = QWidget()
        updates_layout = QVBoxLayout(self._updates_tab)
        updates_layout.setContentsMargins(4, 6, 4, 4)
        # No layout-level AlignTop here: it shrinks the layout to its size hint,
        # which makes word-wrapped labels wrap at a narrow heuristic width. The
        # trailing addStretch() keeps content top-aligned while letting labels
        # use the full panel width.
        self._sidebar_tabs.addTab(self._run_tab, "Run")
        self._sidebar_tabs.addTab(self._viewer_tab, "Results")
        self._sidebar_tabs.addTab(self._models_tab, "Models")
        self._sidebar_tabs.addTab(self._updates_tab, "Updates")
        # Results tab has nothing to show until a run loads — disable it so
        # the tab is greyed out and unclickable until _show_viewer_controls
        # runs.
        self._sidebar_tabs.setTabEnabled(self._TAB_RESULTS, False)
        layout.addWidget(self._sidebar_tabs)

        # Setup form is the only content on the Run tab. The live log lives in
        # a separate bottom panel built by _build_log_panel and assembled into
        # the main window's vertical splitter in qt_app.py, so it stays visible
        # alongside the form during a run instead of replacing it.
        self._setup_page = QWidget()
        setup_layout = QVBoxLayout(self._setup_page)
        setup_layout.setAlignment(Qt.AlignTop)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.addWidget(self._setup_page)

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
        # Adjust on first show only so a freshly-selected long path doesn't
        # widen the combo (and through it the top bar, and through it the whole
        # window) past a comfortable size. Long entries are elided instead.
        self._past_runs_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow
        )
        self._past_runs_combo.setMaximumWidth(360)

        # "+" icon button on the far left of the top bar starts a fresh
        # reconstruction (clears the viewer + resets the past-run selection).
        from deepreefmap.launcher.qt_icons import plus_icon

        self._new_run_btn = QPushButton()
        self._new_run_btn.setIcon(plus_icon(20))
        self._new_run_btn.setToolTip("New reconstruction")
        self._new_run_btn.setFixedSize(28, 28)
        self._new_run_btn.clicked.connect(self._on_new_reconstruction)

        # Log toggle button — checkable so the pressed state mirrors panel
        # visibility. Auto-opens when a run starts; user can collapse afterwards.
        self._log_toggle_btn = QPushButton("Log")
        self._log_toggle_btn.setToolTip("Show or hide the live log panel")
        self._log_toggle_btn.setCheckable(True)
        self._log_toggle_btn.setFixedHeight(24)
        self._log_toggle_btn.toggled.connect(self._set_log_panel_visible)

        self._load_cancel_btn = QPushButton("Cancel")
        self._load_cancel_btn.setVisible(False)
        self._load_cancel_btn.clicked.connect(self._cancel_load)

        self._warnings_label_running = QLabel("")
        self._warnings_label_running.setWordWrap(True)
        self._warnings_label_running.setTextFormat(Qt.TextFormat.RichText)
        self._warnings_label_running.setStyleSheet(
            f"background-color: {WARN_BG}; color: {WARN_TEXT};"
            f" border: 1px solid {WARN_BORDER}; padding: 6px; border-radius: 3px;"
        )
        self._warnings_label_running.setVisible(False)
        setup_layout.addWidget(self._warnings_label_running)

        # --- Input group ---
        input_group = QGroupBox("Input")
        ig = QVBoxLayout(input_group)

        video_row = QHBoxLayout()
        self._video_input = QLineEdit()
        self._video_input.setPlaceholderText("Path to video file")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_video)
        video_row.addWidget(self._video_input, 1)
        video_row.addWidget(browse_btn)
        ig.addLayout(video_row)

        ig.addWidget(QLabel("Camera profile"))
        self._profile_combo = QComboBox()
        self._profile_combo.addItems(profiles)
        ig.addWidget(self._profile_combo)

        ig.addWidget(QLabel("FPS"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(10)
        ig.addWidget(self._fps_spin)

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
        ig.addLayout(range_row)

        self._video_duration_s: float | None = None
        setup_layout.addWidget(input_group)

        # --- Models group ---
        models_group = QGroupBox("Models")
        mg = QVBoxLayout(models_group)

        mg.addWidget(QLabel("Segmentation"))
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
        mg.addLayout(seg_row)

        mg.addWidget(QLabel("Mapping"))
        map_row = QHBoxLayout()
        map_row.setContentsMargins(0, 0, 0, 0)
        map_row.setSpacing(4)
        self._map_combo = QComboBox()
        self._map_combo.addItems(map_backends)
        default_map = "loger" if loger_available() else "scsfmlearner"
        idx = self._map_combo.findText(default_map)
        if idx >= 0:
            self._map_combo.setCurrentIndex(idx)
        if not loger_available():
            map_model = self._map_combo.model()
            if isinstance(map_model, QStandardItemModel):
                for i in range(self._map_combo.count()):
                    if self._map_combo.itemText(i) in ("loger", "loger_star"):
                        item = map_model.item(i)
                        item.setEnabled(False)
                        item.setData(LOGER_INSTALL_HINT, Qt.ItemDataRole.ToolTipRole)
        map_row.addWidget(self._map_combo, 1)
        self._map_status_btn = self._build_model_status_button(self._map_combo)
        map_row.addWidget(self._map_status_btn)
        mg.addLayout(map_row)

        setup_layout.addWidget(models_group)

        # --- Output group ---
        output_group = QGroupBox("Output")
        og = QVBoxLayout(output_group)

        og.addWidget(QLabel("Run name"))
        from datetime import datetime

        self._run_name_input = QLineEdit(datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._run_name_input.setPlaceholderText("Friendly name (e.g. barrier-reef-2026-05-20)")
        og.addWidget(self._run_name_input)

        self._effective_dir_label = QLabel("")
        self._effective_dir_label.setStyleSheet("color: #888;")
        self._effective_dir_label.setWordWrap(True)
        self._effective_dir_label.setTextFormat(Qt.TextFormat.RichText)
        self._effective_dir_label.setOpenExternalLinks(True)
        og.addWidget(self._effective_dir_label)

        style = self.style()
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(4)
        label_row.addWidget(QLabel("Output root"))
        from deepreefmap.launcher.qt_icons import arrow_right_icon

        root_open_btn = QPushButton()
        root_open_btn.setIcon(arrow_right_icon(18))
        root_open_btn.setFixedSize(26, 24)
        root_open_btn.setToolTip("Open output root in file manager")
        root_open_btn.clicked.connect(self._open_output_root)
        label_row.addWidget(root_open_btn)
        label_row.addStretch(1)
        og.addLayout(label_row)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(4)
        self._out_root_input = QLineEdit(default_root)
        input_row.addWidget(self._out_root_input, 1)
        root_browse_btn = QPushButton()
        root_browse_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        root_browse_btn.setIconSize(QSize(18, 18))
        root_browse_btn.setFixedSize(28, 28)
        root_browse_btn.setToolTip("Browse for output root folder…")
        root_browse_btn.clicked.connect(self._browse_output_root)
        input_row.addWidget(root_browse_btn)
        root_default_btn = QPushButton()
        root_default_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        root_default_btn.setIconSize(QSize(18, 18))
        root_default_btn.setFixedSize(28, 28)
        root_default_btn.setToolTip("Reset to <Documents>/DeepReefMap")
        root_default_btn.clicked.connect(self._reset_output_root_to_default)
        input_row.addWidget(root_default_btn)
        og.addLayout(input_row)

        setup_layout.addWidget(output_group)

        self._advanced_toggle = QCheckBox("Advanced settings")
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        setup_layout.addWidget(self._advanced_toggle)
        self._vram_notice = QLabel()
        self._vram_notice.setWordWrap(True)
        self._vram_notice.setStyleSheet("color: #e0a030; font-size: 11px; margin: 2px 0 4px 0;")
        self._vram_notice.setVisible(False)
        setup_layout.addWidget(self._vram_notice)

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
        # --- Processing resolution ---
        default_seg = self._seg_combo.currentText()
        res = get_model_resolution(default_seg)
        self._native_resolution = (res[1], res[0]) if res else (1376, 768)
        self._is_dpt_model = "dpt" in default_seg

        adv_layout.addWidget(QLabel("Processing resolution"))
        self._resolution_preset_combo = QComboBox()
        self._resolution_preset_combo.addItems(["Native", "Half", "Quarter", "Custom"])
        self._resolution_preset_combo.setToolTip(
            "Resolution preset relative to the segmentation model's training resolution."
        )
        adv_layout.addWidget(self._resolution_preset_combo)

        res_row = QHBoxLayout()
        res_row.setContentsMargins(0, 0, 0, 0)
        res_row.setSpacing(4)
        w_col = QVBoxLayout()
        w_col.setContentsMargins(0, 0, 0, 0)
        w_col.addWidget(QLabel("Width"))
        self._proc_width_spin = QSpinBox()
        self._proc_width_spin.setRange(256, 3840)
        self._proc_width_spin.setSingleStep(32)
        self._proc_width_spin.setValue(self._native_resolution[0])
        self._proc_width_spin.setEnabled(False)
        w_col.addWidget(self._proc_width_spin)
        res_row.addLayout(w_col, 1)
        h_col = QVBoxLayout()
        h_col.setContentsMargins(0, 0, 0, 0)
        h_col.addWidget(QLabel("Height"))
        self._proc_height_spin = QSpinBox()
        self._proc_height_spin.setRange(256, 2160)
        self._proc_height_spin.setSingleStep(32)
        self._proc_height_spin.setValue(self._native_resolution[1])
        self._proc_height_spin.setEnabled(False)
        h_col.addWidget(self._proc_height_spin)
        res_row.addLayout(h_col, 1)
        adv_layout.addLayout(res_row)

        self._dpt_resolution_warning = QLabel(
            "DPT models have no internal resize — non-native resolution may reduce accuracy."
        )
        self._dpt_resolution_warning.setWordWrap(True)
        self._dpt_resolution_warning.setStyleSheet("color: #e0a030; font-size: 11px;")
        self._dpt_resolution_warning.setVisible(False)
        adv_layout.addWidget(self._dpt_resolution_warning)

        self._resolution_preset_combo.currentTextChanged.connect(
            self._on_resolution_preset_changed
        )
        self._proc_width_spin.valueChanged.connect(self._update_vram_warning)
        self._proc_height_spin.valueChanged.connect(self._update_vram_warning)

        # --- Batch size ---
        adv_layout.addWidget(QLabel("Segmentation batch size"))
        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(1, 16)
        self._batch_size_spin.setValue(4)
        self._batch_size_spin.setToolTip(
            "Frames segmented per GPU batch. Lower values use less VRAM."
        )
        self._batch_size_spin.valueChanged.connect(self._update_vram_warning)
        adv_layout.addWidget(self._batch_size_spin)
        self._vram_auto_label = QLabel()
        self._vram_auto_label.setWordWrap(True)
        self._vram_auto_label.setStyleSheet("color: #e0a030; font-size: 11px;")
        adv_layout.addWidget(self._vram_auto_label)
        self._reset_defaults_btn = QPushButton("Reset to defaults")
        self._reset_defaults_btn.clicked.connect(self._reset_advanced_defaults)
        adv_layout.addWidget(self._reset_defaults_btn)
        self._update_vram_warning()
        adv_layout.addWidget(QLabel("Grid bins (ortho resolution)"))
        self._grid_bins_spin = QSpinBox()
        self._grid_bins_spin.setRange(100, 10000)
        self._grid_bins_spin.setSingleStep(100)
        self._grid_bins_spin.setValue(2000)
        self._grid_bins_spin.setToolTip("Number of bins for the ortho projection grid.")
        adv_layout.addWidget(self._grid_bins_spin)
        adv_layout.addWidget(QLabel("Replacement radius factor — 0 = auto"))
        self._rr_factor_spin = QDoubleSpinBox()
        self._rr_factor_spin.setRange(0.0, 10.0)
        self._rr_factor_spin.setDecimals(2)
        self._rr_factor_spin.setSingleStep(0.1)
        self._rr_factor_spin.setValue(0.0)
        self._rr_factor_spin.setToolTip("Multiplier on the auto replacement radius. 0 = use auto estimate.")
        adv_layout.addWidget(self._rr_factor_spin)
        adv_layout.addWidget(QLabel("Replacement radius estimation frames"))
        self._rr_est_frames_spin = QSpinBox()
        self._rr_est_frames_spin.setRange(1, 200)
        self._rr_est_frames_spin.setValue(30)
        self._rr_est_frames_spin.setToolTip("Number of leading depth maps used to estimate the default replacement radius.")
        adv_layout.addWidget(self._rr_est_frames_spin)
        adv_layout.addWidget(QLabel("Replacement radius override (m) — 0 = auto"))
        self._rr_override_spin = QDoubleSpinBox()
        self._rr_override_spin.setRange(0.0, 10.0)
        self._rr_override_spin.setDecimals(4)
        self._rr_override_spin.setSingleStep(0.001)
        self._rr_override_spin.setValue(0.0)
        self._rr_override_spin.setToolTip("Absolute replacement voxel size in meters. 0 = use auto estimate.")
        adv_layout.addWidget(self._rr_override_spin)
        self._tsdf_check = QCheckBox("Enable TSDF")
        adv_layout.addWidget(self._tsdf_check)
        self._require_gravity_check = QCheckBox("Require gravity telemetry")
        self._require_gravity_check.setToolTip("Fail if GoPro gravity telemetry cannot be loaded.")
        adv_layout.addWidget(self._require_gravity_check)
        self._skip_seg_check = QCheckBox("Skip segmentation")
        adv_layout.addWidget(self._skip_seg_check)

        # SCSfMLearner-specific knobs — only shown when scsfmlearner is selected.
        self._scs_panel = QWidget()
        scs_layout = QVBoxLayout(self._scs_panel)
        scs_layout.setContentsMargins(0, 6, 0, 0)
        scs_layout.setSpacing(2)
        scs_layout.addWidget(QLabel("SCSfMLearner width"))
        self._scs_width_spin = QSpinBox()
        self._scs_width_spin.setRange(64, 2048)
        self._scs_width_spin.setSingleStep(32)
        self._scs_width_spin.setValue(512)
        scs_layout.addWidget(self._scs_width_spin)
        scs_layout.addWidget(QLabel("SCSfMLearner height"))
        self._scs_height_spin = QSpinBox()
        self._scs_height_spin.setRange(64, 2048)
        self._scs_height_spin.setSingleStep(32)
        self._scs_height_spin.setValue(256)
        scs_layout.addWidget(self._scs_height_spin)
        scs_layout.addWidget(QLabel("SCSfMLearner checkpoint (optional override)"))
        scs_ckpt_row = QHBoxLayout()
        scs_ckpt_row.setContentsMargins(0, 0, 0, 0)
        scs_ckpt_row.setSpacing(4)
        self._scs_checkpoint_input = QLineEdit()
        self._scs_checkpoint_input.setPlaceholderText("Default: EPFL-ECEO/deepreefmap-sfm-net")
        scs_ckpt_row.addWidget(self._scs_checkpoint_input, 1)
        scs_ckpt_btn = QPushButton()
        scs_ckpt_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        scs_ckpt_btn.setIconSize(QSize(18, 18))
        scs_ckpt_btn.setFixedSize(28, 28)
        scs_ckpt_btn.setToolTip("Browse for a SCSfMLearner .pt checkpoint…")
        scs_ckpt_btn.clicked.connect(self._browse_scs_checkpoint)
        scs_ckpt_row.addWidget(scs_ckpt_btn)
        scs_layout.addLayout(scs_ckpt_row)
        self._scs_panel.setVisible(False)
        adv_layout.addWidget(self._scs_panel)

        # LoGeR-specific knobs — only shown when a LoGeR backend is selected.
        self._loger_panel = QWidget()
        loger_layout = QVBoxLayout(self._loger_panel)
        loger_layout.setContentsMargins(0, 6, 0, 0)
        loger_layout.setSpacing(2)
        loger_layout.addWidget(QLabel("LoGeR window size"))
        self._loger_window_spin = QSpinBox()
        self._loger_window_spin.setRange(1, 256)
        self._loger_window_spin.setValue(32)
        loger_layout.addWidget(self._loger_window_spin)
        loger_layout.addWidget(QLabel("LoGeR overlap size"))
        self._loger_overlap_spin = QSpinBox()
        self._loger_overlap_spin.setRange(0, 64)
        self._loger_overlap_spin.setValue(3)
        loger_layout.addWidget(self._loger_overlap_spin)
        loger_layout.addWidget(QLabel("LoGeR checkpoint (optional override)"))
        loger_ckpt_row = QHBoxLayout()
        loger_ckpt_row.setContentsMargins(0, 0, 0, 0)
        loger_ckpt_row.setSpacing(4)
        self._loger_model_path_input = QLineEdit()
        self._loger_model_path_input.setPlaceholderText("Default: vendored checkpoint")
        loger_ckpt_row.addWidget(self._loger_model_path_input, 1)
        loger_ckpt_btn = QPushButton()
        loger_ckpt_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        loger_ckpt_btn.setIconSize(QSize(18, 18))
        loger_ckpt_btn.setFixedSize(28, 28)
        loger_ckpt_btn.setToolTip("Browse for a LoGeR .pt checkpoint…")
        loger_ckpt_btn.clicked.connect(self._browse_loger_checkpoint)
        loger_ckpt_row.addWidget(loger_ckpt_btn)
        loger_layout.addLayout(loger_ckpt_row)
        self._refine_intrinsics_check = QCheckBox("Refine intrinsics from mapper")
        loger_layout.addWidget(self._refine_intrinsics_check)
        self._loger_panel.setVisible(False)
        adv_layout.addWidget(self._loger_panel)

        self._advanced_panel.setVisible(False)
        setup_layout.addWidget(self._advanced_panel)

        self._gated_warning = QLabel()
        self._gated_warning.setWordWrap(True)
        self._gated_warning.setTextFormat(Qt.TextFormat.RichText)
        self._gated_warning.setOpenExternalLinks(True)
        self._gated_warning.setStyleSheet(
            f"background-color: {WARN_BG}; color: {WARN_TEXT};"
            f" border: 1px solid {WARN_BORDER}; padding: 6px; border-radius: 3px;"
            " font-size: 11px;"
        )
        self._gated_warning.setVisible(False)
        setup_layout.addWidget(self._gated_warning)

        self._submit_btn = QPushButton("Start reconstruction")
        self._submit_btn.clicked.connect(self._on_submit)
        setup_layout.addWidget(self._submit_btn)

        run_ctl_row = QHBoxLayout()
        run_ctl_row.setContentsMargins(0, 0, 0, 0)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setToolTip("Cancel the running reconstruction")
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        run_ctl_row.addWidget(self._stop_btn)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setToolTip(
            "Pause the reconstruction at the next safe checkpoint. "
            "Long mapping passes may take time to respond."
        )
        self._pause_btn.setCheckable(True)
        self._pause_btn.setVisible(False)
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        run_ctl_row.addWidget(self._pause_btn)
        setup_layout.addLayout(run_ctl_row)

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
            f"background-color: {WARN_BG}; color: {WARN_TEXT};"
            f" border: 1px solid {WARN_BORDER}; padding: 6px; border-radius: 3px;"
        )
        self._warnings_label.setVisible(False)
        self._run_warnings: list[str] = []
        viewer_layout.addWidget(self._warnings_label)

        # Live log view: lives in a bottom panel built by _build_log_panel and
        # assembled into the main window's vertical splitter in qt_app.py.
        # Construction is here so the panel is ready to receive log lines as
        # soon as the qt log handler installs (next block).
        self._log_view = LogView()

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

        # The frustum visibility toggle lives inside the legend overlay (see
        # LegendOverlay._frustum_check), alongside the other per-view controls.

        vc_layout.addWidget(QLabel("Point size"))
        self._point_size_spin = QDoubleSpinBox()
        self._point_size_spin.setRange(0.5, 20.0)
        self._point_size_spin.setValue(2.0)
        self._point_size_spin.setSingleStep(0.5)
        self._point_size_spin.valueChanged.connect(self._on_viewer_control_changed)
        vc_layout.addWidget(self._point_size_spin)

        self._confidence_box = QWidget()
        conf_layout = QVBoxLayout(self._confidence_box)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_layout.addWidget(QLabel("Min confidence (%)"))
        self._confidence_slider = QSlider(Qt.Horizontal)
        self._confidence_slider.setRange(0, 100)
        self._confidence_slider.setValue(0)
        self._confidence_slider.valueChanged.connect(self._on_viewer_control_changed)
        conf_layout.addWidget(self._confidence_slider)
        vc_layout.addWidget(self._confidence_box)

        self._frame_slider = self._viewer.frame_slider
        self._frame_slider.valueChanged.connect(self._on_viewer_control_changed)

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

        follow_row = QHBoxLayout()
        self._follow_camera_check = QCheckBox("Follow camera")
        self._follow_camera_check.toggled.connect(self._on_follow_camera_changed)
        follow_row.addWidget(self._follow_camera_check)
        self._view_from_camera_btn = QPushButton("Snap")
        self._view_from_camera_btn.setToolTip("Snap the 3D view to the current frame's camera")
        self._view_from_camera_btn.clicked.connect(self._on_view_from_camera)
        follow_row.addWidget(self._view_from_camera_btn)
        vc_layout.addLayout(follow_row)

        vc_layout.addWidget(QLabel("Camera backoff (m)"))
        self._camera_backoff_spin = QDoubleSpinBox()
        self._camera_backoff_spin.setRange(0.0, 5.0)
        self._camera_backoff_spin.setSingleStep(0.1)
        self._camera_backoff_spin.setValue(0.5)
        self._camera_backoff_spin.valueChanged.connect(self._on_follow_camera_changed)
        vc_layout.addWidget(self._camera_backoff_spin)

        # Results tab: viewer controls + results panel. The tab itself is
        # disabled until a run is loaded (greyed out and unclickable), so no
        # empty-state placeholder is needed inside. addStretch is appended at
        # the end after the results group is added below.
        viewer_layout.addWidget(self._viewer_controls_group)

        # The legend lives as a floating overlay on the 3D canvas; this dict
        # is populated by _build_legend and queried by _enabled_class_set.
        self._legend_toggles: dict[int, QCheckBox] = {}
        self._legend_solo_buttons: dict[int, object] = {}
        # Legend sort state: column "selected" (visibility), "name", or "size",
        # plus a direction. _legend_order_cache lets _apply_legend_sort skip
        # reflows when the resulting order is unchanged.
        self._legend_sort_mode: str = "selected"
        self._legend_sort_ascending: bool = False
        self._legend_order_cache: list[int] | None = None
        self._legend_sort_connected: bool = False

        self._results_group = QGroupBox("Results")
        self._results_group.setVisible(False)
        res_layout = QVBoxLayout(self._results_group)

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

        # Two-ring sunburst (outer = fine classes, inner = coarse groups) docks
        # above the legend rows in the canvas overlay so the user can show/hide
        # classes from either the pie or the rows. Updates live with the crop.
        self._cover_sunburst = SunburstWidget()
        self._cover_sunburst.selection_clicked.connect(self._on_sunburst_selection)
        self._cover_sunburst.setVisible(False)
        self._viewer.legend_overlay.set_sunburst(self._cover_sunburst)

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

        self._discover_btn = QPushButton("Check Hugging Face for new models")
        self._discover_btn.setToolTip(
            "Query the EPFL-ECEO organisation on Hugging Face for newly published "
            "models. Requires an internet connection."
        )
        self._discover_btn.clicked.connect(self._on_discover_clicked)
        self._models_layout.addWidget(self._discover_btn)

        self._hf_auth_user: str | None = None
        self._model_rows: dict[str, QWidget] = {}
        self._model_actions: dict[str, QWidget] = {}
        self._downloading: set[str] = set()
        self._download_cancel_requested: set[str] = set()
        self._delete_armed: dict[str, QPushButton] = {}
        self._last_model_states: list = []
        self._repo_access: dict[str, bool | None] = {}
        self._download_errors: dict[str, str] = {}

        self._seg_combo.currentTextChanged.connect(self._on_required_models_changed)
        self._seg_combo.currentTextChanged.connect(self._on_seg_model_changed)
        self._map_combo.currentTextChanged.connect(self._on_required_models_changed)
        self._map_combo.currentTextChanged.connect(self._on_mapping_backend_changed)
        self._skip_seg_check.toggled.connect(self._on_required_models_changed)
        self._video_input.textChanged.connect(self._recompute_submit_state)
        self._video_input.editingFinished.connect(self._on_video_input_committed)
        self._out_root_input.textChanged.connect(self._on_output_root_changed)
        self._run_name_input.textChanged.connect(self._on_run_name_changed)

        # Watch the output root directory so the past-runs combo reflects new
        # manifests appearing on disk (e.g. a sibling process completes a run)
        # in addition to user edits of the path text.
        self._out_root_watcher = QFileSystemWatcher(self)
        self._out_root_watcher.directoryChanged.connect(
            lambda _path: self._refresh_past_runs_combo()
        )

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
        self._update_out_root_watch()
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

        self._update_version_label = QLabel(f"Version: <b>{_current_version()}</b>")
        self._update_version_label.setWordWrap(True)
        updates_layout.addWidget(self._update_version_label)
        self._update_status_label = QLabel("Checking for updates…")
        self._update_status_label.setWordWrap(True)
        self._update_status_label.setStyleSheet("color: #aaa;")
        updates_layout.addWidget(self._update_status_label)
        update_row = QHBoxLayout()
        self._update_version_combo = QComboBox()
        self._update_version_combo.setVisible(False)
        update_row.addWidget(self._update_version_combo, 1)
        self._update_btn = QPushButton("Install")
        self._update_btn.setVisible(False)
        self._update_btn.clicked.connect(self._on_update)
        update_row.addWidget(self._update_btn)
        updates_layout.addLayout(update_row)
        self._available_releases: list[dict] = []

        threading.Thread(target=self._check_for_update, daemon=True).start()

        updates_layout.addStretch()

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

        # New-reconstruction "+" button is the leftmost element — it's the
        # primary action that resets the workspace for a fresh run.
        h.addWidget(self._new_run_btn)

        h.addWidget(QLabel("Past runs:"))
        h.addWidget(self._past_runs_combo, 2)

        h.addWidget(self._log_toggle_btn)

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

    def _build_log_panel(self) -> QWidget:
        """Bottom-of-window panel hosting the live log.

        Created lazily by qt_app.py so it can place the panel in the central
        vertical splitter (alongside the form + viewer above it). The panel
        is hidden initially and the log_toggle button in the top bar drives
        its visibility.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(4)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addWidget(QLabel("<b>Live log</b>"))
        header_row.addStretch(1)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setToolTip("Hide log panel")
        close_btn.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; }")
        close_btn.clicked.connect(lambda: self._set_log_panel_visible(False))
        header_row.addWidget(close_btn)
        layout.addLayout(header_row)
        layout.addWidget(self._log_view, 1)
        panel.setVisible(False)
        self._log_panel = panel
        return panel

    def _set_log_panel_visible(self, visible: bool) -> None:
        if not hasattr(self, "_log_panel"):
            return
        self._log_panel.setVisible(visible)
        # Keep the top-bar toggle in sync without re-triggering this slot.
        if self._log_toggle_btn.isChecked() != visible:
            self._log_toggle_btn.blockSignals(True)
            self._log_toggle_btn.setChecked(visible)
            self._log_toggle_btn.blockSignals(False)
        # When the user re-opens the panel after collapsing the splitter all
        # the way down, give it a reasonable default height again.
        if visible and hasattr(self, "_central_vsplitter"):
            sizes = self._central_vsplitter.sizes()
            if len(sizes) == 2 and sizes[1] < 40:
                total = sum(sizes) or 800
                bottom = max(200, total // 4)
                self._central_vsplitter.setSizes([total - bottom, bottom])
    def _on_advanced_toggled(self, checked: bool) -> None:
        self._advanced_panel.setVisible(checked)

    def _on_mapping_backend_changed(self, _value: object = "") -> None:
        backend = self._map_combo.currentText()
        self._loger_panel.setVisible(backend in ("loger", "loger_star"))
        self._scs_panel.setVisible(backend == "scsfmlearner")

    def _collect_loger_options(self, mapping_name: str) -> dict | None:
        """Build the LoGeR mapping_options dict from the form, or None for other backends."""
        if mapping_name not in ("loger", "loger_star"):
            return None
        model_path = self._loger_model_path_input.text().strip()
        return {
            "window_size": self._loger_window_spin.value(),
            "overlap_size": self._loger_overlap_spin.value(),
            "model_path": model_path or None,
        }

    def _browse_loger_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select LoGeR checkpoint", "", "Checkpoints (*.pt *.pth);;All files (*)"
        )
        if path:
            self._loger_model_path_input.setText(path)

    def _browse_scs_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SCSfMLearner checkpoint", "", "Checkpoints (*.pt *.pth);;All files (*)"
        )
        if path:
            self._scs_checkpoint_input.setText(path)

    def _on_seg_model_changed(self, name: str) -> None:
        from deepreefmap.segmentation.registry import get_model_resolution

        res = get_model_resolution(name)
        self._native_resolution = (res[1], res[0]) if res else (1376, 768)
        self._is_dpt_model = "dpt" in name
        preset = self._resolution_preset_combo.currentText()
        if preset != "Custom":
            self._apply_resolution_preset(preset)
        self._update_dpt_warning()
        self._update_vram_warning()

    def _on_resolution_preset_changed(self, preset: str) -> None:
        is_custom = preset == "Custom"
        self._proc_width_spin.setEnabled(is_custom)
        self._proc_height_spin.setEnabled(is_custom)
        if not is_custom:
            self._apply_resolution_preset(preset)
        self._update_dpt_warning()
        self._update_vram_warning()

    def _apply_resolution_preset(self, preset: str) -> None:
        nw, nh = self._native_resolution
        divisors = {"Native": 1, "Half": 2, "Quarter": 4}
        d = divisors.get(preset, 1)
        self._proc_width_spin.blockSignals(True)
        self._proc_height_spin.blockSignals(True)
        self._proc_width_spin.setValue(nw // d)
        self._proc_height_spin.setValue(nh // d)
        self._proc_width_spin.blockSignals(False)
        self._proc_height_spin.blockSignals(False)

    def _update_dpt_warning(self) -> None:
        nw, nh = self._native_resolution
        current_w = self._proc_width_spin.value()
        current_h = self._proc_height_spin.value()
        show = self._is_dpt_model and (current_w != nw or current_h != nh)
        self._dpt_resolution_warning.setVisible(show)

    def _update_vram_warning(self) -> None:
        w = self._proc_width_spin.value()
        h = self._proc_height_spin.value()
        batch = self._batch_size_spin.value()
        try:
            from deepreefmap.device import estimate_segmentation_batch_size, resolve_device

            suggested = estimate_segmentation_batch_size(resolve_device(), w, h)
        except Exception:
            suggested = 4
        if batch > suggested:
            self._vram_auto_label.setText(
                f"Batch size {batch} may exceed available VRAM at {w}×{h}. "
                f"Suggested for your GPU: {suggested}."
            )
            self._vram_auto_label.setVisible(True)
            self._vram_notice.setText(
                "VRAM warning — check batch size in advanced settings."
            )
            self._vram_notice.setVisible(True)
        else:
            self._vram_auto_label.setVisible(False)
            self._vram_notice.setVisible(False)
        self._update_dpt_warning()

    def _update_gated_warning(self) -> None:
        seg_name = self._seg_combo.currentText()
        if self._skip_seg_check.isChecked():
            self._gated_warning.setVisible(False)
            return
        info_match = None
        for info, cached in self._last_model_states:
            if info.name == seg_name:
                info_match = (info, cached)
                break
        if info_match is None or not info_match[0].gated:
            self._gated_warning.setVisible(False)
            return
        info, cached = info_match
        if cached:
            self._gated_warning.setVisible(False)
            return
        links = " ".join(
            f'<a href="https://huggingface.co/{repo}" style="color:{WARN_TEXT}">{repo}</a>'
            for repo in info.hf_repos
        )
        logged_in = self._hf_auth_user is not None
        if logged_in:
            msg = (
                f"<b>{seg_name}</b> requires license acceptance. "
                f"Visit each repo and click <i>Agree and access</i>: {links}"
            )
        else:
            msg = (
                f"<b>{seg_name}</b> requires Hugging Face login and license acceptance. "
                f"Log in on the Models tab, then visit each repo and click "
                f"<i>Agree and access</i>: {links}"
            )
        self._gated_warning.setText(msg)
        self._gated_warning.setVisible(True)

    def _reset_advanced_defaults(self) -> None:
        self._resolution_preset_combo.setCurrentText("Native")
        self._batch_size_spin.setValue(4)
        self._grid_bins_spin.setValue(2000)
        self._rr_factor_spin.setValue(0.0)
        self._rr_est_frames_spin.setValue(30)
        self._rr_override_spin.setValue(0.0)

    def _gpu_available(self) -> bool:
        cached = getattr(self, "_gpu_available_cache", None)
        if cached is None:
            try:
                from deepreefmap.device import resolve_device

                cached = resolve_device().type != "cpu"
            except Exception:
                cached = False
            self._gpu_available_cache = cached
        return cached

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

        if not self._last_model_states:
            reasons.append("checking model availability…")
        else:
            cached_names = {info.name for info, cached in self._last_model_states if cached}
            missing = [m for m in sorted(self._required_model_names()) if m not in cached_names]
            if missing:
                reasons.append(f"download required model{'s' if len(missing) > 1 else ''}: {', '.join(missing)}")

        if self._map_combo.currentText() in ("loger", "loger_star") and not self._gpu_available():
            reasons.append("LoGeR needs a GPU (none detected)")

        ok = not reasons
        self._submit_btn.setEnabled(ok)
        if ok:
            self._submit_hint.setText("")
        else:
            self._submit_hint.setText("Cannot start: " + "; ".join(reasons) + ".")

        self._update_gated_warning()
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
        except Exception:
            self._effective_dir_label.setText("")
            return
        if target.exists():
            self._effective_dir_label.setText(
                f'→ <a href="file://{target}" style="color:#9ecbff;">{target}</a>'
            )
        else:
            self._effective_dir_label.setText(f"→ {target}")

    def _on_output_root_changed(self, _text: str = "") -> None:
        self._update_effective_dir_label()
        self._recompute_submit_state()
        self._settings.setValue("output_root_dir", self._out_root_input.text())
        self._refresh_past_runs_combo()
        self._update_out_root_watch()

    def _update_out_root_watch(self) -> None:
        """Re-point the QFileSystemWatcher at the current output root.

        Watching the directory lets us pick up new manifests that appear from
        other processes (e.g. a parallel run, file-manager moves) without
        requiring the user to interact with the form.
        """
        if not hasattr(self, "_out_root_watcher"):
            return
        existing = self._out_root_watcher.directories()
        if existing:
            self._out_root_watcher.removePaths(existing)
        root = Path(self._out_root_input.text()).expanduser()
        if root.exists() and root.is_dir():
            self._out_root_watcher.addPath(str(root))

    def _on_run_name_changed(self, _text: str = "") -> None:
        self._update_effective_dir_label()
        self._recompute_submit_state()
