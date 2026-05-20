from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap, QSurfaceFormat
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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


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
    _sig_update_result = Signal(str)
    _sig_model_status_done = Signal(object, object)
    _sig_pipeline_error = Signal(str)
    _sig_status_text = Signal(str)
    _sig_hf_auth_done = Signal(object, str)
    _sig_download_progress = Signal(str, int)
    _sig_run_loaded = Signal(object, str, str)
    _sig_load_progress = Signal(str, int, int)

    def __init__(self, classes_config: object, classes_path: Path) -> None:
        super().__init__()
        self._classes_config = classes_config
        self._classes_path = classes_path
        self._pipeline_thread: threading.Thread | None = None
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        self._sig_update_check_done.connect(self._apply_update_check)
        self._sig_update_result.connect(lambda t: self._update_label.setText(t))
        self._sig_model_status_done.connect(self._apply_model_status)
        self._sig_pipeline_error.connect(self._on_pipeline_error)
        self._sig_status_text.connect(lambda t: self._status_label.setText(t))
        self._sig_hf_auth_done.connect(self._on_hf_auth_done)
        self._sig_download_progress.connect(self._on_download_progress)
        self._sig_run_loaded.connect(self._apply_loaded_run)
        self._sig_load_progress.connect(self._on_load_progress)

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

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(top_bar)
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

        self._advanced_toggle = QCheckBox("Advanced settings")
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        layout.addWidget(self._advanced_toggle)

        self._advanced_panel = QWidget()
        adv_layout = QVBoxLayout(self._advanced_panel)
        adv_layout.setContentsMargins(12, 0, 0, 0)
        adv_layout.addWidget(QLabel("Transect length (m)"))
        self._transect_length = QLineEdit()
        self._transect_length.setPlaceholderText("optional")
        adv_layout.addWidget(self._transect_length)
        adv_layout.addWidget(QLabel("Crop width (m)"))
        self._crop_width = QLineEdit()
        self._crop_width.setPlaceholderText("optional")
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

        self._submit_hint = QLabel("")
        self._submit_hint.setWordWrap(True)
        self._submit_hint.setStyleSheet("color: #c84; font-style: italic;")
        layout.addWidget(self._submit_hint)

        # Status label and progress bar are owned by the top toolbar but
        # constructed here so they exist before _recompute_submit_state runs.
        self._status_label = QLabel("Ready. Fill the form above and click Start.")
        self._status_label.setWordWrap(True)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)

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
        self._ortho_label = QLabel()
        self._ortho_label.setAlignment(Qt.AlignCenter)
        res_layout.addWidget(self._ortho_label)
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
        self._out_root_input.textChanged.connect(self._on_output_root_changed)
        self._run_name_input.textChanged.connect(self._on_run_name_changed)

        self._active_run_dir: Path | None = None
        self._active_run_manifest: dict | None = None
        self._load_cancelled = False

        self._settings = QSettings("ECEO", "deepreefmap")
        last_video = self._settings.value("last_video_path", "", type=str)
        if last_video and Path(last_video).exists():
            self._video_input.setText(last_video)
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
        self._progress_bar.setMaximumWidth(180)
        h.addWidget(self._progress_bar)
        h.addWidget(self._load_cancel_btn)

        return bar

    def _cancel_load(self) -> None:
        # Soft cancel: the worker thread can't be interrupted mid-read, but
        # we set a flag so _apply_loaded_run drops the result when it eventually
        # arrives. The thread is a daemon and will exit with the process.
        self._load_cancelled = True
        self._load_cancel_btn.setVisible(False)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._status_label.setText("Load cancelled.")


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

        cover_path = out / "benthic_cover.json"
        if cover_path.exists():
            try:
                with open(cover_path) as f:
                    cover = json.load(f)
                classes = cover.get("classes", {})
                lines = ["<b>Benthic cover:</b><br>"]
                for cid_str, info in sorted(classes.items(), key=lambda x: -x[1].get("fraction", 0)):
                    name = info.get("name", cid_str)
                    frac = info.get("fraction", 0)
                    if frac > 0.001:
                        lines.append(f"{name}: {frac * 100:.1f}%<br>")
                self._cover_label.setText("".join(lines))
            except Exception:
                pass

        self._results_group.setVisible(True)


    def _refresh_model_status(self) -> None:
        from deepreefmap.launcher.model_manager import ALL_MODELS, check_hf_auth, is_model_cached

        auth_user = check_hf_auth()
        model_states = [(m, is_model_cached(m)) for m in ALL_MODELS]
        self._sig_model_status_done.emit(auth_user, model_states)

    def _apply_model_status(self, auth_user: str | None, model_states: list) -> None:
        self._hf_auth_user = auth_user
        self._last_model_states = list(model_states)
        if auth_user:
            self._hf_auth_label.setText(f"Logged in as <b>{auth_user}</b>")
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
            if self._active_run_dir is not None:
                for i in range(1, self._past_runs_combo.count()):
                    if self._past_runs_combo.itemData(i) == str(self._active_run_dir):
                        self._past_runs_combo.setCurrentIndex(i)
                        break
        finally:
            self._past_runs_combo.blockSignals(False)

    @staticmethod
    def _format_run_metadata(manifest: dict, run_dir: Path, *, include_disk_size: bool) -> str:
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
        if include_disk_size:
            try:
                total = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
                if total >= 1e9:
                    lines.append(f"Disk: {total / 1e9:.2f} GB")
                else:
                    lines.append(f"Disk: {total / 1e6:.1f} MB")
            except Exception:
                pass
        return "<br>".join(lines)

    def _on_past_run_selected(self, index: int) -> None:
        if index <= 0:
            return
        run_dir = self._past_runs_combo.itemData(index)
        if not run_dir:
            return
        self._auto_load_run(Path(run_dir))

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
        self._active_run_dir = None
        self._active_run_manifest = None
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

        def parse_optional_float(s: str) -> float | None:
            s = s.strip()
            return float(s) if s else None

        try:
            transect_length = parse_optional_float(self._transect_length.text())
            transect_crop = parse_optional_float(self._crop_width.text())
        except ValueError as exc:
            self._status_label.setText(f"Error: {exc}")
            return

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
        }

        self._set_form_enabled(False)
        self._status_label.setText("Reconstruction starting...")
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)

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
        self._progress_bar.setVisible(False)
        self._set_form_enabled(True)

    def _set_form_enabled(self, enabled: bool) -> None:
        for w in (
            self._video_input, self._profile_combo, self._seg_combo,
            self._map_combo, self._out_root_input, self._run_name_input,
            self._fps_spin, self._transect_length, self._crop_width,
            self._tsdf_check, self._skip_seg_check, self._submit_btn,
        ):
            w.setEnabled(enabled)

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
        # Indeterminate ("busy") progress bar — setRange(0, 0) makes Qt animate
        # the chunk back and forth so the user sees the app isn't frozen.
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(True)
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
        "geometry": "Loading geometry cloud",
    }

    def _on_load_progress(self, stage: str, cur: int, tot: int) -> None:
        if self._load_cancelled:
            return
        label = self._STAGE_LABELS.get(stage, stage)
        if tot > 1:
            self._status_label.setText(f"{label}… {cur}/{tot}")
            if self._progress_bar.minimum() != 0 or self._progress_bar.maximum() != tot:
                self._progress_bar.setRange(0, tot)
            self._progress_bar.setValue(cur)
        else:
            self._status_label.setText(f"{label}…")
            # Bracket the indeterminate animation around fast stages so the
            # user sees the bar pulsing rather than freezing.
            self._progress_bar.setRange(0, 0)

    def _apply_loaded_run(self, result: object, run_dir_str: str, error: str) -> None:
        from deepreefmap.pipeline.run_loader import GEOMETRY_ONLY_MODE

        # Restore the progress bar to determinate-but-hidden so subsequent
        # reconstruction runs aren't stuck in indeterminate mode.
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._load_cancel_btn.setVisible(False)

        if self._load_cancelled:
            # User aborted while the worker was still reading from disk.
            return

        run_dir = Path(run_dir_str)
        if error or result is None:
            self._status_label.setText(f"Error loading run: {error}")
            return

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

        self._active_run_dir = run_dir
        self._active_run_manifest = result.manifest
        display = result.manifest.get("name") or run_dir.name
        self._status_label.setText(f"Loaded run '{display}' from {run_dir}")

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
            self._results_group.setVisible(True)


    def _on_viewer_status(self, event: str, **kwargs: object) -> None:
        if event == "start_run":
            self._status_label.setText("Starting reconstruction...")
            self._progress_bar.setValue(0)
            self._progress_bar.setVisible(True)
        elif event == "set_stage":
            stage = kwargs.get("stage", "")
            status = kwargs.get("status", "")
            message = kwargs.get("message", "")
            label = {"startup": "Startup", "preprocess": "Preprocessing", "mapping": "Mapping", "outputs": "Building outputs"}.get(stage, stage)
            if status == "completed":
                self._status_label.setText(f"{label} complete" + (f" — {message}" if message else ""))
            else:
                self._status_label.setText(f"{label}..." + (f" {message}" if message else ""))
        elif event == "update_progress":
            current = kwargs.get("current", 0)
            total = kwargs.get("total")
            stage = kwargs.get("stage", "")
            if total:
                pct = int(100 * int(current) / int(total))
                self._progress_bar.setValue(pct)
                self._progress_bar.setVisible(True)
                self._status_label.setText(f"{'Preprocessing' if stage == 'preprocess' else 'Mapping'}: {current}/{total} frames")
        elif event == "data_ready":
            self._status_label.setText("Reconstruction complete. Loading viewer...")
            self._progress_bar.setValue(100)
            if self._viewer.has_scene_data:
                self._build_legend()
                self._show_viewer_controls()
                self._on_viewer_control_changed()
                self._status_label.setText("Reconstruction complete.")
        elif event == "mark_outputs":
            output_dir = kwargs.get("output_dir", "")
            self._status_label.setText(f"Outputs saved to {output_dir}")
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
            self._progress_bar.setVisible(False)
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
        pyapp_bin = _pyapp_binary_path()
        if pyapp_bin is None:
            return
        self._update_btn.setEnabled(False)
        self._update_label.setText("Updating... this may take a few minutes.")
        threading.Thread(target=self._run_update, args=(pyapp_bin,), daemon=True).start()

    def _run_update(self, pyapp_bin: str) -> None:
        import subprocess

        if os.environ.get("DEEPREEFMAP_MOCK_PYAPP"):
            self._sig_update_result.emit("Mock update: simulated success. Close and reopen to apply.")
            return
        try:
            result = subprocess.run([pyapp_bin, "self", "update"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                text = "Update installed. Close this window and reopen the app."
            else:
                tail = (result.stderr or result.stdout)[-300:]
                text = f"Update failed: {tail}"
        except Exception as exc:
            text = f"Update failed: {exc!r}"
        self._sig_update_result.emit(text)


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
