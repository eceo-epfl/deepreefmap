from __future__ import annotations

from deepreefmap.gui._window_protocol import MixinBase

import json
import logging
import os
from pathlib import Path

from PySide6.QtGui import QColor

from deepreefmap.gui.theme import UPDATE

logger = logging.getLogger(__name__)

# Amber accent used to flag the Updates tab when a newer release exists.
_UPDATE_ACCENT = QColor(UPDATE)


_DEFAULT_GH_REPO = "eceo-epfl/deepreefmap"


def _gh_releases_url() -> str:
    # Full-URL override points the release check at a local server, so the real
    # download + swap + provision + prune path can be validated without a public
    # release (see tests/e2e/update_interactive.sh). DEEPREEFMAP_GH_REPO swaps
    # only the owner/repo against the real GitHub host.
    override = os.environ.get("DEEPREEFMAP_GH_API_URL")
    if override:
        return override
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


def _fetch_releases(timeout: float = 8.0) -> list[dict] | None:
    """Return raw release records (with `assets`) for binary swap.

    Drafts and releases without a binary asset for this platform are dropped,
    so pre-binary releases (v1.0.0) and failed asset uploads are never offered.

    Mock via `DEEPREEFMAP_MOCK_VERSIONS`: synthesises records with one
    `deepreefmap-linux-x64` asset per version pointing at a placeholder URL.
    """
    import urllib.request

    from deepreefmap.gui.binary_swap import BinarySwapError, match_asset_url, resolve_asset_name

    mock = os.environ.get("DEEPREEFMAP_MOCK_VERSIONS")
    if mock is not None:
        records = []
        for v in (s.strip() for s in mock.split(",")):
            if not v:
                continue
            records.append({
                "tag_name": f"v{v}",
                "draft": False,
                "assets": [
                    {
                        "name": "deepreefmap-linux-x64",
                        "browser_download_url": f"https://example.invalid/v{v}/deepreefmap-linux-x64",
                    },
                    {
                        "name": "deepreefmap-windows-x64.exe",
                        "browser_download_url": f"https://example.invalid/v{v}/deepreefmap-windows-x64.exe",
                    },
                ],
            })
        return records if records else None
    try:
        req = urllib.request.Request(_gh_releases_url(), headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            releases = json.load(resp)
        kept = [rel for rel in releases if rel.get("tag_name") and not rel.get("draft")]
        try:
            asset_name = resolve_asset_name()
        except BinarySwapError:
            asset_name = None  # unsupported platform: dev mode, install controls stay hidden
        if asset_name is not None:
            kept = [rel for rel in kept if match_asset_url(rel, asset_name) is not None]
        # Empty list means "reached GitHub, nothing installable" (renders as
        # "No releases found."); None is reserved for fetch failures.
        return kept
    except Exception as exc:
        logger.warning("Failed to fetch release metadata from GitHub: %s", exc)
        return None


def _release_version(record: dict) -> str:
    tag = str(record.get("tag_name", ""))
    return tag[1:] if tag.startswith("v") else tag


def _parse_version(value: str):
    from packaging.version import InvalidVersion, Version

    try:
        return Version(value)
    except InvalidVersion:
        return None


def _newer_releases(releases: list[dict], current: str) -> list[dict]:
    """Releases strictly newer than `current`, newest first.

    Comparison is semantic (packaging.version), so a current build ahead of
    the newest published release reports nothing to install. Falls back to
    string inequality only when `current` itself can't be parsed, so an
    unparseable version still surfaces differing releases.
    """
    current_v = _parse_version(current)
    if current_v is None:
        return [r for r in releases if _release_version(r) != current]
    newer = []
    for rel in releases:
        rv = _parse_version(_release_version(rel))
        if rv is not None and rv > current_v:
            newer.append((rv, rel))
    newer.sort(key=lambda pair: pair[0], reverse=True)
    return [rel for _, rel in newer]


def _selectable_releases(releases: list[dict], current: str, include_older: bool) -> list[dict]:
    """Releases the user may install, newest first.

    With ``include_older`` off this is :func:`_newer_releases` (upgrades only).
    With it on, every release except the exact current version is offered, so a
    rollback to an older version is possible. Mechanically a downgrade is the
    same as an upgrade: the chosen asset is downloaded, swapped in, and the
    previous environment is pruned once the older version provisions.
    """
    if not include_older:
        return _newer_releases(releases, current)
    keyed = []
    for rel in releases:
        version = _release_version(rel)
        if version == current:
            continue
        keyed.append((_parse_version(version) or _parse_version("0"), rel))
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [rel for _, rel in keyed]


def _current_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("deepreefmap")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


class VersionCheckMixin(MixinBase):
    """DeepReefMapWindow methods for checking GitHub releases and installing updates."""

    def _check_for_update(self) -> None:
        current = _current_version()
        releases = _fetch_releases()
        pyapp_bin = _pyapp_binary_path()
        self._sig_update_check_done.emit(current, releases, pyapp_bin)

    def _set_updates_tab_alert(self, latest: str | None) -> None:
        """Flag the Updates tab amber with a dot when `latest` is available.

        Passing None clears the alert and restores the default tab style.
        """
        bar = self._sidebar_tabs.tabBar()
        idx = self._TAB_UPDATES
        if latest is None:
            bar.setTabText(idx, "Updates")
            bar.setTabTextColor(idx, QColor())  # invalid colour → theme default
            self._sidebar_tabs.setTabToolTip(idx, "")
            return
        bar.setTabText(idx, "Updates ●")
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
        newer = _newer_releases(self._available_releases, current)
        if newer:
            self._set_updates_tab_alert(_release_version(newer[0]))

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
            any(_release_version(r) != current for r in releases)
        )
        self._populate_update_versions()

    def _populate_update_versions(self) -> None:
        current = self._current_version_str
        include_older = self._update_show_all.isChecked()
        selectable = _selectable_releases(self._available_releases, current, include_older)
        current_v = _parse_version(current)
        self._update_version_combo.clear()
        for rel in selectable:
            version = _release_version(rel)
            rv = _parse_version(version)
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
                f"Latest: <b>{_release_version(selectable[0])}</b>. Pick a version to install:"
            )

    def _on_toggle_show_all_versions(self, _checked: bool) -> None:
        if self._available_releases:
            self._populate_update_versions()

    def _refresh_desktop_entry_button(self) -> None:
        from deepreefmap.gui.desktop_entry import desktop_entry_installed

        if desktop_entry_installed():
            self._desktop_entry_btn.setText("Remove from applications menu")
        else:
            self._desktop_entry_btn.setText("Add to applications menu")

    def _on_toggle_desktop_entry(self) -> None:
        from deepreefmap.gui.desktop_entry import (
            desktop_entry_installed,
            install_desktop_entry,
            remove_desktop_entry,
        )

        try:
            if desktop_entry_installed():
                remove_desktop_entry()
            else:
                pyapp_bin = _pyapp_binary_path()
                if pyapp_bin is None:
                    return
                install_desktop_entry(pyapp_bin)
        except OSError:
            logger.exception("Desktop entry update failed")
        self._refresh_desktop_entry_button()

    def _on_update(self) -> None:
        from deepreefmap.gui.update_dialog import UpdateProgressDialog

        pyapp_bin = _pyapp_binary_path()
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
        version = _release_version(release)
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
