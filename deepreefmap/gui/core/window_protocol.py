"""Declare the surface the DeepReefMapWindow mixins share, so mypy can resolve
cross-mixin `self._foo` references."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging
    import threading
    import uuid
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
        QListWidget,
        QProgressBar,
        QPushButton,
        QSlider,
        QSpinBox,
        QTableWidget,
        QTabWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    from deepreefmap.config.classes import ClassConfig
    from deepreefmap.io.scene_file import SceneFrameAccessor
    from deepreefmap.gui.system.log_view import LogView
    from deepreefmap.gui.runs.progress import ProgressModel
    from deepreefmap.gui.viewer.pick_tooltip import PickCard
    from deepreefmap.profiling.eta import RunEtaEstimator
    from deepreefmap.gui.core.spinner import SpinnerStopButton
    from deepreefmap.gui.runs.timing_popup import HoverColumn, TimingPopup
    from deepreefmap.pipeline.artifacts import SemanticPointCloud
    from deepreefmap.pointcloud.grid_ortho import OrthoGrid
    from deepreefmap.gui.viewer.widget import QtPointCloudViewer
    from deepreefmap.gui.runs.sunburst import SunburstWidget
    from deepreefmap.gui.survey.charts import GroupedBarChart
    from deepreefmap.survey.models import SurveyBatch
    from deepreefmap.survey.store import SurveyStore

    # QWidget, not QMainWindow: DeepReefMapWindow lists QMainWindow first among
    # its bases, so a QMainWindow base here breaks C3 linearisation.
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
        _status_tick_timer: QTimer
        _status_base_text: str
        _status_count_text: str
        _status_phase_key: str | None
        _status_phase_started: float
        _active_run_manifest: dict | None
        _results_output_dir: Path | None
        _TAB_RUN: int
        _TAB_PLAN: int
        _TAB_SURVEY: int
        _TAB_ANALYSIS: int
        _TAB_RESULTS: int
        _TAB_SYSTEM: int
        _TAB_MODELS: int
        _survey_tabs: list[int]
        _ui_mode: str
        _survey_store_obj: SurveyStore | None
        _transect_form_id: uuid.UUID | None
        _quick_entry_to_end: bool
        _survey_rows: list
        _survey_transects: list
        _survey_batch: SurveyBatch | None
        _survey_preset: dict | None
        _survey_cancel_event: threading.Event | None
        _survey_worker_running: bool
        _analysis_covers: list
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
        _desktop_entry_btn: QPushButton
        _discover_btn: QPushButton
        _hf_auth_btn: QPushButton
        _pause_btn: QPushButton
        _rename_btn: QPushButton
        _rename_cancel_btn: QPushButton
        _rename_ok_btn: QPushButton
        _scrub_btn: QPushButton
        _spinner_stop: SpinnerStopButton
        _start_btn: QPushButton
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
        _memory_notice: QLabel
        _memory_warn_icon: QLabel
        _recorded_runs_caption: QLabel
        _status_label: QLabel
        _update_status_label: QLabel
        _update_version_label: QLabel
        _warnings_label: QLabel
        _warnings_label_running: QLabel

        # --- survey mode -------------------------------------------------
        _mode_toggle_btn: QToolButton
        _survey_batch_name: QLineEdit
        _survey_preset_label: QLabel
        _survey_pass_table: QTableWidget
        _survey_start_btn: QPushButton
        _survey_stop_btn: QPushButton
        _analysis_transect_combo: QComboBox
        _analysis_level_combo: QComboBox
        _analysis_chart: GroupedBarChart
        _analysis_stats_table: QTableWidget
        _analysis_repro_label: QLabel
        _analysis_runs_list: QListWidget
        _transect_list: QListWidget
        _tr_name_input: QLineEdit
        _tr_quick_input: QLineEdit
        _tr_start_lat: QLineEdit
        _tr_start_lon: QLineEdit
        _tr_end_lat: QLineEdit
        _tr_end_lon: QLineEdit
        _tr_length: QDoubleSpinBox
        _tr_depth: QDoubleSpinBox
        _tr_description: QLineEdit
        _tr_geodesic_label: QLabel

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
        _progress_stack: HoverColumn
        _eta_total_label: QLabel
        _eta: RunEtaEstimator | None
        _timing_popup: TimingPopup

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
        _sig_survey_progress = Signal(int, int, str)
        _sig_survey_done = Signal(int, int, str)

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
        def _build_system_panel(self, layout: object) -> None: ...
        def _on_sidebar_tab_changed(self, index: int) -> None: ...
        def _refresh_recorded_runs(self) -> None: ...
        def _update_memory_profile_warning(self) -> None: ...
        def _cancel_load(self) -> None: ...
        def _cancel_rename(self) -> None: ...
        def _check_for_update(self) -> None: ...
        def _clear_run_warnings(self) -> None: ...
        def _collect_loger_options(self, mapping_name: str) -> dict | None: ...
        def _commit_rename(self) -> None: ...
        def _estimate_frame_count(self, fps: int) -> int | None: ...
        def _recompute_submit_state(self) -> None: ...
        def _refresh_desktop_entry_button(self) -> None: ...
        def _refresh_model_status(self) -> None: ...
        def _refresh_past_runs_combo(self) -> None: ...
        def _refresh_run_warnings_view(self) -> None: ...
        def _required_model_names(self) -> set[str]: ...
        def _reset_progress_bars(self) -> None: ...
        def _render_status(self) -> None: ...
        def _render_eta(self) -> None: ...
        def _end_run_controls(self) -> None: ...
        def _new_run_estimator(self) -> RunEtaEstimator: ...
        def _reveal_legend_overlay(self) -> None: ...
        def _set_app_mode(self, mode: str) -> None: ...
        def _set_form_enabled(self, enabled: bool) -> None: ...
        def _build_mode_toggle(self) -> QToolButton: ...
        def _build_plan_tab(self, layout: QVBoxLayout) -> None: ...
        def _init_ui_mode(self) -> None: ...
        def _set_ui_mode(self, mode: str) -> None: ...
        def _survey_home_tab(self) -> int: ...
        def _survey_store(self) -> SurveyStore: ...
        def _survey_data_changed(self) -> None: ...
        def _refresh_transect_list(self, select_id: uuid.UUID | None = None) -> None: ...
        def _build_survey_batch_tab(self, layout: QVBoxLayout) -> None: ...
        def _build_survey_analysis_tab(self, layout: QVBoxLayout) -> None: ...
        def _refresh_survey_analysis(self) -> None: ...
        def _refresh_survey_batch_tab(self) -> None: ...
        def _refresh_survey_transect_combos(self) -> None: ...
        def _refresh_survey_pass_statuses(self) -> None: ...
        def _recompute_survey_start(self) -> None: ...
        def _on_survey_progress(self, index: int, total: int, name: str) -> None: ...
        def _on_survey_done(self, ok: int, total: int, last_error: str) -> None: ...
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
        def _on_toggle_desktop_entry(self) -> None: ...
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
