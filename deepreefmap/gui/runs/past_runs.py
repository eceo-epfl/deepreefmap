"""Run banner and the workspace reset behind the toolbar "+" button."""

from __future__ import annotations

import logging
from pathlib import Path

from deepreefmap.gui.core.window_protocol import MixinBase
from deepreefmap.gui.runs.run_cards import format_run_metadata_compact

logger = logging.getLogger(__name__)


class PastRunsMixin(MixinBase):
    """DeepReefMapWindow methods for the run banner and the workspace reset."""

    def _show_run_meta_banner(self, manifest: dict, run_dir: Path, *, include_disk_size: bool) -> None:
        # A warm Data-section size cache saves the synchronous directory walk.
        cached_size = getattr(self, "_run_size_cache", {}).get(run_dir.name)
        self._run_meta_banner.setText(
            format_run_metadata_compact(
                manifest, run_dir, include_disk_size=include_disk_size, disk_bytes=cached_size
            )
        )
        self._run_meta_banner.setVisible(True)

    def _hide_run_meta_banner(self) -> None:
        self._run_meta_banner.setVisible(False)
        self._run_meta_banner.setText("")

    def _on_new_reconstruction(self) -> None:
        self._viewer._clear_scene_data()
        self._results_group.setVisible(False)
        self._viewer.legend_overlay.setVisible(False)
        self._viewer_controls_group.setVisible(False)
        self._sidebar_tabs.setTabEnabled(self._TAB_RESULTS, False)
        self._hide_run_meta_banner()
        self._clear_run_warnings()
        self._active_run_dir = None
        self._active_run_manifest = None
        self._set_ortho_sources(None, None, None)
        from datetime import datetime

        self._run_name_input.setText(datetime.now().strftime("%Y%m%d-%H%M%S"))
        if getattr(self, "_ui_mode", "advanced") == "simple":
            self._status_label.setText("Workspace cleared.")
        else:
            self._status_label.setText(self._idle_status_text())
        self._set_app_mode("SETUP")
