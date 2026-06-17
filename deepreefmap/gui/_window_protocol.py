"""Type-only base for the DeepReefMapWindow mixins.

The window is composed from several `*Mixin` classes (see `qt_app.py`), each of
which calls `self._foo` attributes and methods that are defined in *sibling*
mixins or in `DeepReefMapWindow.__init__`. Analysed in isolation, mypy cannot
see those, so every cross-mixin reference would be an `attr-defined` error.

`MixinBase` declares the shared surface once. Under `TYPE_CHECKING` it is a
`QWidget` subclass (so `self` also carries the Qt widget API and can be passed
where a `QWidget` is expected); at runtime it is plain `object`, so it adds no
base class, changes no MRO, and has zero runtime effect. Each mixin inherits
from it.

`QWidget` is deliberate rather than `QMainWindow`: `DeepReefMapWindow` lists
`QMainWindow` first among its own bases, so a `QMainWindow`-derived mixin base
would put `QMainWindow` ahead of its own subclasses and break C3 linearisation.
`QWidget` is a shared ancestor of both, which linearises cleanly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging
    import threading
    from pathlib import Path

    from PySide6.QtCore import QSettings, QTimer, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QSlider,
        QSpinBox,
        QTabWidget,
        QToolButton,
        QWidget,
    )

    from deepreefmap.config.classes import ClassConfig
    from deepreefmap.io.scene_file import SceneFrameAccessor
    from deepreefmap.gui.log_view import LogView
    from deepreefmap.gui.progress import ProgressModel
    from deepreefmap.gui.pick_tooltip import PickCard
    from deepreefmap.pipeline.artifacts import SemanticPointCloud
    from deepreefmap.pointcloud.grid_ortho import OrthoGrid
    from deepreefmap.gui.viewer import QtPointCloudViewer
    from deepreefmap.gui.sunburst_widget import SunburstWidget

    class MixinBase(QWidget):
        # --- plain state -------------------------------------------------
        _classes_config: ClassConfig
        _classes_path: Path | None
        _active_run_dir: Path | None
        _video_duration_s: float | None
        _settings: QSettings
        _playback_timer: QTimer
        _pipeline_thread: threading.Thread | None
        _active_progress_model: ProgressModel | None
        _active_run_manifest: dict | None
        _results_output_dir: Path | None
        _TAB_RUN: int
        _TAB_RESULTS: int
        _TAB_UPDATES: int
        _downloading: set[str]
        _download_cancel_requested: set[str]
        _download_errors: dict[str, str]
        _delete_armed: dict[str, QPushButton]
        _model_actions: dict[str, QWidget]
        _model_rows: dict[str, QWidget]
        _run_warnings: list[str]

        # --- composite widgets / models ----------------------------------
        _viewer: QtPointCloudViewer
        _log_view: LogView
        _cover_sunburst: SunburstWidget
        _load_model: ProgressModel
        _recon_model: ProgressModel

        # --- ortho preview state -----------------------------------------
        _ortho_cloud: SemanticPointCloud | None
        _base_ortho_grid: OrthoGrid | None
        _current_ortho_grid: OrthoGrid | None
        _ortho_classes_config: ClassConfig | None

        # --- viewer / legend / pick state --------------------------------
        _pick_card: PickCard | None
        _last_pick_payload: dict | None
        _pick_card_pinned_pos: tuple[int, int] | None
        _legend_sort_mode: str
        _legend_sort_ascending: bool
        _legend_sort_connected: bool
        _legend_order_cache: list[int] | None
        _legend_solo_buttons: dict[int, QToolButton]
        _legend_toggles: dict[int, QCheckBox]
        _run_log_file_handler: logging.FileHandler | None
        _scene_accessor: SceneFrameAccessor | None
        _available_releases: list[dict]
        _current_version_str: str

        # --- checkboxes --------------------------------------------------
        _accumulate_check: QCheckBox
        _follow_camera_check: QCheckBox
        _play_check: QCheckBox
        _refine_intrinsics_check: QCheckBox
        _require_gravity_check: QCheckBox
        _semantic_check: QCheckBox
        _skip_seg_check: QCheckBox
        _tsdf_check: QCheckBox
        _update_show_all: QCheckBox

        # --- buttons -----------------------------------------------------
        _batch_btn: QPushButton
        _discover_btn: QPushButton
        _hf_auth_btn: QPushButton
        _load_cancel_btn: QPushButton
        _pause_btn: QPushButton
        _rename_btn: QPushButton
        _rename_cancel_btn: QPushButton
        _rename_ok_btn: QPushButton
        _stop_btn: QPushButton
        _submit_btn: QPushButton
        _update_btn: QPushButton

        # --- spin boxes --------------------------------------------------
        _batch_size_spin: QSpinBox
        _fps_spin: QSpinBox
        _grid_bins_spin: QSpinBox
        _play_fps_spin: QSpinBox
        _proc_height_spin: QSpinBox
        _proc_width_spin: QSpinBox
        _rr_est_frames_spin: QSpinBox
        _scs_height_spin: QSpinBox
        _scs_width_spin: QSpinBox
        _begin_spin: QDoubleSpinBox
        _camera_backoff_spin: QDoubleSpinBox
        _crop_width: QDoubleSpinBox
        _end_spin: QDoubleSpinBox
        _point_size_spin: QDoubleSpinBox
        _results_crop_width: QDoubleSpinBox
        _results_transect_length: QDoubleSpinBox
        _rr_factor_spin: QDoubleSpinBox
        _rr_override_spin: QDoubleSpinBox
        _transect_length: QDoubleSpinBox

        # --- sliders -----------------------------------------------------
        _confidence_slider: QSlider
        _frame_slider: QSlider
        _results_crop_slider: QSlider
        _results_transect_slider: QSlider

        # --- labels ------------------------------------------------------
        _cover_label: QLabel
        _hf_auth_icon: QLabel
        _hf_auth_label: QLabel
        _ortho_rgb_preview: QLabel
        _ortho_seg_preview: QLabel
        _run_meta_banner: QLabel
        _status_label: QLabel
        _update_status_label: QLabel
        _update_version_label: QLabel
        _warnings_label: QLabel
        _warnings_label_running: QLabel

        # --- combos / line edits -----------------------------------------
        _map_combo: QComboBox
        _past_runs_combo: QComboBox
        _profile_combo: QComboBox
        _seg_combo: QComboBox
        _update_version_combo: QComboBox
        _out_root_input: QLineEdit
        _rename_edit: QLineEdit
        _run_name_input: QLineEdit
        _scs_checkpoint_input: QLineEdit
        _video_input: QLineEdit

        # --- containers / layouts ----------------------------------------
        _confidence_box: QWidget
        _crop_box: QGroupBox
        _results_group: QGroupBox
        _viewer_controls_group: QGroupBox
        _models_grid: QGridLayout
        _sidebar_tabs: QTabWidget
        _progress_bar: QProgressBar
        _total_progress_bar: QProgressBar

        # --- signals (defined as class attrs on DeepReefMapWindow) --------
        _sig_update_check_done = Signal(str, object, object)
        _sig_model_status_done = Signal(object, object)
        _sig_pipeline_error = Signal(str)
        _sig_pipeline_cancelled = Signal()
        _sig_status_text = Signal(str)
        _sig_hf_auth_done = Signal(object, str)
        _sig_download_progress = Signal(str, int)
        _sig_run_loaded = Signal(object, str, str)
        _sig_load_progress = Signal(str, int, int)
        _sig_batch_progress = Signal(int, int, str)
        _sig_batch_done = Signal(int, int, str)
        _sig_qc_render_progress = Signal(int, int)
        _sig_qc_render_done = Signal(bool, str)
        _sig_discovery_done = Signal(object, object)

        # --- cross-mixin methods -----------------------------------------
        def _add_run_warning(self, message: str) -> None: ...
        def _apply_progress(
            self,
            phase_key: str,
            label: str,
            current: int = 0,
            total: int = 0,
            flush: bool = False,
        ) -> None: ...
        def _auto_load_run(self, run_dir: Path) -> None: ...
        def _begin_progress(self, model: ProgressModel) -> None: ...
        def _begin_rename(self) -> None: ...
        def _build_legend(self) -> None: ...
        def _build_model_status_button(self, combo: QComboBox) -> QPushButton: ...
        def _cancel_load(self) -> None: ...
        def _cancel_rename(self) -> None: ...
        def _check_for_update(self) -> None: ...
        def _clear_run_warnings(self) -> None: ...
        def _collect_loger_options(self, mapping_name: str) -> dict | None: ...
        def _commit_rename(self) -> None: ...
        def _recompute_submit_state(self) -> None: ...
        def _refresh_model_status(self) -> None: ...
        def _refresh_past_runs_combo(self) -> None: ...
        def _refresh_run_warnings_view(self) -> None: ...
        def _required_model_names(self) -> set[str]: ...
        def _reset_progress_bars(self) -> None: ...
        def _reveal_legend_overlay(self) -> None: ...
        def _set_app_mode(self, mode: str) -> None: ...
        def _set_form_enabled(self, enabled: bool) -> None: ...
        def _set_log_panel_visible(self, visible: bool) -> None: ...
        def _set_ortho_sources(
            self,
            cloud: SemanticPointCloud | None,
            base_grid: OrthoGrid | None,
            classes_config: ClassConfig | None,
        ) -> None: ...
        def _set_semantic_only_controls_visible(self, visible: bool) -> None: ...
        def _show_results(self, output_dir: str) -> None: ...
        def _show_run_meta_banner(
            self, manifest: dict, run_dir: Path, *, include_disk_size: bool
        ) -> None: ...
        def _show_viewer_controls(self) -> None: ...
        def _update_effective_dir_label(self) -> None: ...

        # event handlers invoked across mixins
        def _on_batch_clicked(self) -> None: ...
        def _on_discover_clicked(self) -> None: ...
        def _on_export_cover_csv(self) -> None: ...
        def _on_export_current_frame(self) -> None: ...
        def _on_export_ortho_npz(self) -> None: ...
        def _on_export_ortho_png(self) -> None: ...
        def _on_export_qc_video(self) -> None: ...
        def _on_export_zip(self) -> None: ...
        def _on_follow_camera_changed(self) -> None: ...
        def _on_hf_auth_button(self) -> None: ...
        def _on_new_reconstruction(self) -> None: ...
        def _on_past_run_selected(self, index: int) -> None: ...
        def _on_pause_toggled(self, paused: bool) -> None: ...
        def _on_play_fps_changed(self) -> None: ...
        def _on_play_toggled(self, playing: bool) -> None: ...
        def _on_required_models_changed(self, _value: object = "") -> None: ...
        def _on_results_crop_slider_changed(self, value: int) -> None: ...
        def _on_results_crop_width_changed(self, value: float) -> None: ...
        def _on_results_transect_length_changed(self, value: float) -> None: ...
        def _on_results_transect_slider_changed(self, value: int) -> None: ...
        def _on_stop_clicked(self) -> None: ...
        def _on_submit(self) -> None: ...
        def _on_sunburst_selection(self, class_ids: list) -> None: ...
        def _on_toggle_show_all_versions(self, _checked: bool) -> None: ...
        def _on_update(self) -> None: ...
        def _on_viewer_control_changed(self) -> None: ...
        def _on_view_from_camera(self) -> None: ...

        @staticmethod
        def _format_cover_html(cover: dict) -> str: ...
        @staticmethod
        def _sanitize_run_name(name: str) -> str: ...

else:
    MixinBase = object
