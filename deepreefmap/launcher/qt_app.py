from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.launcher.qt_app_batch import (  # noqa: F401  re-exported for tests
    BatchMixin,
    _load_batch_csv,
    _parse_timestamp_range,
)
from deepreefmap.launcher.qt_app_form_panel import FormPanelMixin
from deepreefmap.launcher.qt_app_models import ModelManagementMixin
from deepreefmap.launcher.qt_app_past_runs import (
    PastRunsMixin,
)
from deepreefmap.launcher.qt_app_results import ResultsMixin
from deepreefmap.launcher.qt_app_run_loader import RunLoadingMixin
from deepreefmap.launcher.qt_app_viewer_ctl import ViewerControlsMixin
from deepreefmap.launcher.qt_app_progress import (
    ProgressBarsMixin,
)
from deepreefmap.launcher.qt_app_version import (  # noqa: F401  re-exported for tests
    VersionCheckMixin,
    _current_version,
    _fetch_release_versions,
    _fetch_releases,
    _pyapp_binary_path,
)

logger = logging.getLogger(__name__)


class DeepReefMapWindow(
    QMainWindow,
    BatchMixin,
    FormPanelMixin,
    ModelManagementMixin,
    PastRunsMixin,
    ProgressBarsMixin,
    ResultsMixin,
    RunLoadingMixin,
    ViewerControlsMixin,
    VersionCheckMixin,
):
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
    _sig_qc_render_progress = Signal(int, int)
    _sig_qc_render_done = Signal(bool, str)

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
        self._viewer.point_picked.connect(self._on_point_picked)
        self._viewer.point_picked_clear.connect(self._on_point_picked_clear)
        self._viewer.canvas_resized.connect(self._on_canvas_resized)
        self._viewer.frustum_picked.connect(self._on_frustum_picked)
        self._pick_card = None
        self._last_pick_payload = None

        # Build the form first so widgets it references (status_label, etc.)
        # are constructed before we wire them into the top toolbar.
        form_panel = self._build_form_panel()
        top_bar = self._build_top_bar()
        log_panel = self._build_log_panel()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(form_panel)
        splitter.addWidget(self._viewer)
        splitter.setSizes([440, 960])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(6)

        # Vertical splitter places the live log as a togglable section at the
        # bottom of the window, alongside the form + 3D viewer above it. Hidden
        # initially; the top-bar Log button drives visibility, and _on_submit
        # auto-opens it when a run starts.
        self._central_vsplitter = QSplitter(Qt.Vertical)
        self._central_vsplitter.addWidget(splitter)
        self._central_vsplitter.addWidget(log_panel)
        self._central_vsplitter.setSizes([700, 220])
        self._central_vsplitter.setStretchFactor(0, 1)
        self._central_vsplitter.setStretchFactor(1, 0)
        self._central_vsplitter.setChildrenCollapsible(True)
        self._central_vsplitter.setHandleWidth(6)

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
        central_layout.addWidget(self._central_vsplitter, 1)
        self.setCentralWidget(central)



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
