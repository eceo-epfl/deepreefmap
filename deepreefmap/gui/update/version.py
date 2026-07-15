"""Updates UI: release check worker, Updates-tab controls, desktop entry toggle."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtGui import QColor

from deepreefmap.gui.core.window_protocol import MixinBase
from deepreefmap.gui.core.theme import UPDATE
from deepreefmap.packaging.releases import (
    current_version,
    fetch_releases,
    newer_releases,
    parse_version,
    pyapp_binary_path,
    release_version,
    selectable_releases,
)

logger = logging.getLogger(__name__)

# Amber accent used to flag the Updates tab when a newer release exists.
_UPDATE_ACCENT = QColor(UPDATE)


class VersionCheckMixin(MixinBase):
    """DeepReefMapWindow methods for checking GitHub releases and installing updates."""

    def _check_for_update(self) -> None:
        current = current_version()
        releases = fetch_releases()
        pyapp_bin = pyapp_binary_path()
        self._sig_update_check_done.emit(current, releases, pyapp_bin)

    def _set_updates_tab_alert(self, latest: str | None) -> None:
        """Flag the System tab (which hosts updates) amber when `latest` is available.

        Passing None clears the alert and restores the default tab style.
        """
        bar = self._sidebar_tabs.tabBar()
        idx = self._TAB_SYSTEM
        if latest is None:
            bar.setTabText(idx, "System")
            bar.setTabTextColor(idx, QColor())  # invalid color falls back to theme default
            self._sidebar_tabs.setTabToolTip(idx, "")
            return
        bar.setTabText(idx, "System ●")
        bar.setTabTextColor(idx, _UPDATE_ACCENT)
        self._sidebar_tabs.setTabToolTip(idx, f"Version {latest} is available")

    def _apply_update_check(self, current: str, releases: list[dict] | None, pyapp_bin: str | None) -> None:
        self._current_version_str = current
        self._update_version_label.setText(f"Version: <b>{current}</b>")
        self._set_updates_tab_alert(None)
        self._update_show_all.setVisible(False)
        self._update_version_combo.setVisible(False)
        self._update_btn.setVisible(False)
        self._available_releases = list(releases or [])

        # Surface a newer release in the tab regardless of mode, as a nudge.
        newer = newer_releases(self._available_releases, current)
        if newer:
            self._set_updates_tab_alert(release_version(newer[0]))

        # Dev mode: running from source, not the installed binary. In-app
        # install/rollback swap the binary in place, which only makes sense for
        # the installed application, so the controls stay hidden here.
        if pyapp_bin is None:
            self._update_status_label.setText(
                "Running development mode. Launch from a binary to manage versions."
            )
            return

        if releases is None:
            self._update_status_label.setText("Couldn't reach GitHub.")
            return
        if not releases:
            self._update_status_label.setText("No releases found.")
            return
        # Installed binary: a rollback is only meaningful if there is any version
        # other than the current one.
        self._update_show_all.setVisible(
            any(release_version(r) != current for r in releases)
        )
        self._populate_update_versions()

    def _populate_update_versions(self) -> None:
        current = self._current_version_str
        include_older = self._update_show_all.isChecked()
        selectable = selectable_releases(self._available_releases, current, include_older)
        current_v = parse_version(current)
        self._update_version_combo.clear()
        for rel in selectable:
            version = release_version(rel)
            rv = parse_version(version)
            marker = ""
            if current_v is not None and rv is not None:
                marker = " ↑" if rv > current_v else " ↓"
            self._update_version_combo.addItem(f"{version}{marker}", rel)
        has_items = self._update_version_combo.count() > 0
        self._update_version_combo.setVisible(has_items)
        self._update_btn.setVisible(has_items)
        if not has_items:
            self._update_status_label.setText("Up to date.")
        elif include_older:
            self._update_status_label.setText("Pick a version to install or roll back to:")
        else:
            self._update_status_label.setText(
                f"Latest: <b>{release_version(selectable[0])}</b>. Pick a version to install:"
            )

    def _on_toggle_show_all_versions(self, _checked: bool) -> None:
        if self._available_releases:
            self._populate_update_versions()

    def _refresh_desktop_entry_button(self) -> None:
        from deepreefmap.packaging.desktop_entry import desktop_entry_installed

        if desktop_entry_installed():
            self._desktop_entry_btn.setText("Remove from applications menu")
        else:
            self._desktop_entry_btn.setText("Add to applications menu")

    def _on_toggle_desktop_entry(self) -> None:
        from deepreefmap.packaging.desktop_entry import (
            desktop_entry_installed,
            install_desktop_entry,
            remove_desktop_entry,
        )

        try:
            if desktop_entry_installed():
                remove_desktop_entry()
            else:
                pyapp_bin = pyapp_binary_path()
                if pyapp_bin is None:
                    return
                install_desktop_entry(pyapp_bin)
        except OSError:
            logger.exception("Desktop entry update failed")
        self._refresh_desktop_entry_button()

    def _on_update(self) -> None:
        from deepreefmap.gui.update.dialog import UpdateProgressDialog

        pyapp_bin = pyapp_binary_path()
        if pyapp_bin is None:
            logger.warning("Install clicked but no PyApp binary detected")
            return
        index = self._update_version_combo.currentIndex()
        if index < 0:
            return
        release = self._update_version_combo.itemData(index)
        if not isinstance(release, dict):
            logger.warning("Selected release has no metadata")
            return
        version = release_version(release)
        self._update_btn.setEnabled(False)
        try:
            dialog = UpdateProgressDialog(
                target_version=version,
                release=release,
                binary_path=Path(pyapp_bin),
                parent=self,
            )
            dialog.run()
        finally:
            self._update_btn.setEnabled(True)
