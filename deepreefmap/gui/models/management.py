"""Models tab: per-weight cache status, download progress, and gated-login prompts."""

from __future__ import annotations

from deepreefmap.gui.core.window_protocol import MixinBase

import threading
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from deepreefmap.gui.models.hf_dialog import HfLoginDialog
from deepreefmap.gui.core.theme import BORDER, CARD_BG, DANGER_BG, SUCCESS, TEXT_DIM, WARNING

if TYPE_CHECKING:
    from deepreefmap.gui.models.manager import ModelInfo


class ModelManagementMixin(MixinBase):
    """DeepReefMapWindow methods for HF auth, model status, download, and delete."""

    def _find_model_state(self, model_name: str) -> tuple[ModelInfo | None, bool]:
        for state_info, state_cached in self._last_model_states:
            if state_info.name == model_name:
                return state_info, state_cached
        return None, False

    def _jump_to_model(self, model_name: str | None = None) -> None:
        """Switch the sidebar to the Models tab and reveal one row."""
        if hasattr(self, "_sidebar_tabs") and hasattr(self, "_TAB_MODELS"):
            self._sidebar_tabs.setCurrentIndex(self._TAB_MODELS)
        if not hasattr(self, "_models_group"):
            return
        scroll_area = self._models_group.parentWidget()
        while scroll_area is not None and not isinstance(scroll_area, QScrollArea):
            scroll_area = scroll_area.parentWidget()
        target = self._model_rows.get(model_name) if model_name else None
        if scroll_area is not None:
            scroll_area.ensureWidgetVisible(target or self._models_group, 0, 20)
        if target is not None:
            self._flash_model_row(target)

    def _flash_model_row(self, label: QWidget) -> None:
        prev = label.styleSheet()
        label.setStyleSheet(
            "QLabel { background-color: rgba(232, 160, 74, 60);"
            f" border: 1px solid {WARNING}; border-radius: 3px; padding: 2px; }}"
        )

        def _clear() -> None:
            try:
                label.setStyleSheet(prev)
            except RuntimeError:
                pass  # widget destroyed by an _apply_model_status refresh

        QTimer.singleShot(1500, _clear)

    def _build_model_status_button(self, combo: QComboBox) -> QPushButton:
        # Compact action button next to the model dropdown. The click
        # behaviour depends on the current state of the selected model:
        # ⬇ downloads, 🔒 opens the HF login dialog, ✓ jumps to the row in
        # the Models tab so the user can delete it.
        btn = QPushButton("…")
        btn.setFixedWidth(28)
        btn.setToolTip("Open Models")
        btn.clicked.connect(lambda: self._on_status_button_click(combo.currentText()))
        return btn

    def _on_status_button_click(self, model_name: str) -> None:
        if model_name in self._downloading:
            self._cancel_download(model_name)
            return
        info, cached = self._find_model_state(model_name)
        if info is None:
            self._jump_to_model(None)
            return
        if cached:
            self._jump_to_model(model_name)
        elif info.gated and self._hf_auth_user is None:
            self._on_hf_auth_button()
        else:
            self._download_model(model_name)

    def _update_model_status_button(
        self, btn: QPushButton, selected_name: str
    ) -> None:
        if selected_name in self._downloading:
            pct = int(btn.property("downloadPercent") or 0)
            self._apply_downloading_style(btn, selected_name, pct)
            return
        info, cached = self._find_model_state(selected_name)
        if info is None:
            btn.setText("…")
            btn.setToolTip("Open Models")
            btn.setStyleSheet("")
            return
        from deepreefmap.gui.core.icons import check_icon, download_icon, lock_icon

        if cached:
            btn.setText("")
            btn.setIcon(check_icon(16))
            btn.setToolTip(f"{selected_name} is downloaded. Click to manage cache.")
            btn.setStyleSheet("")
        elif info.gated and self._hf_auth_user is None:
            btn.setText("")
            btn.setIcon(lock_icon(16))
            btn.setToolTip(
                f"{selected_name} is gated. Click to log in to Hugging Face."
            )
            btn.setStyleSheet("")
        else:
            from deepreefmap.gui.models.manager import ModelStatus, model_status

            status, why = model_status(info)
            btn.setText("")
            btn.setIcon(download_icon(16))
            if status == ModelStatus.PARTIAL:
                btn.setToolTip(f"{selected_name} download incomplete ({why}). Click to re-download.")
            else:
                btn.setToolTip(f"{selected_name} not downloaded. Click to download.")
            btn.setStyleSheet("")

    def _apply_downloading_style(
        self, btn: QPushButton, model_name: str, percent: int
    ) -> None:
        # Render the inline status button as a cancel control (✕) with a
        # left-to-right green fill that tracks download percent. Clamp the
        # gradient stop just inside [0, 1] so qlineargradient stays well-formed
        # at the edges.
        pct = max(0, min(100, percent))
        stop = max(0.0001, min(0.9999, pct / 100.0))
        cancelling = model_name in self._download_cancel_requested
        if cancelling:
            btn.setText("…")
            btn.setToolTip(f"Cancelling download of {model_name}…")
            btn.setEnabled(False)
        else:
            btn.setText("✕")
            btn.setToolTip(f"Downloading {model_name} ({pct}%). Click to cancel.")
            btn.setEnabled(True)
        btn.setProperty("downloadPercent", pct)
        btn.setStyleSheet(
            "QPushButton {"
            " color: #fff; font-weight: bold;"
            f" background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f" stop:0 #3a7a3a, stop:{stop:.4f} #3a7a3a,"
            f" stop:{min(stop + 0.0001, 1.0):.4f} {CARD_BG}, stop:1 {CARD_BG});"
            f" border: 1px solid {BORDER}; border-radius: 3px;"
            "}"
        )

    def _form_status_buttons_for(self, model_name: str) -> list[QPushButton]:
        # The inline status button next to each combo always reflects the
        # currently-selected model in that combo, so this only matches form
        # buttons whose combo points at this model right now. If the user
        # changes the dropdown mid-download, _update_models_button_status
        # re-renders the affected button to whatever state its new model is in.
        out: list[QPushButton] = []
        if hasattr(self, "_seg_combo") and hasattr(self, "_seg_status_btn"):
            if self._seg_combo.currentText() == model_name:
                out.append(self._seg_status_btn)
        if hasattr(self, "_map_combo") and hasattr(self, "_map_status_btn"):
            if self._map_combo.currentText() == model_name:
                out.append(self._map_status_btn)
        return out

    def _update_models_button_status(self) -> None:
        """Refresh the per-dropdown model status icons."""
        if hasattr(self, "_seg_status_btn") and hasattr(self, "_seg_combo"):
            self._update_model_status_button(
                self._seg_status_btn, self._seg_combo.currentText()
            )
        if hasattr(self, "_map_status_btn") and hasattr(self, "_map_combo"):
            self._update_model_status_button(
                self._map_status_btn, self._map_combo.currentText()
            )

    def _refresh_model_status(self) -> None:
        from deepreefmap.gui.models.manager import all_known_models, check_hf_auth, is_model_cached

        auth_user, can_gated = check_hf_auth()
        self._can_read_gated = can_gated
        model_states = [(m, is_model_cached(m)) for m in all_known_models()]
        self._sig_model_status_done.emit(auth_user, model_states)

    def _on_discover_clicked(self) -> None:
        self._discover_btn.setEnabled(False)
        self._discover_btn.setText("Checking…")

        def _work() -> None:
            from deepreefmap.gui.models.manager import discover_models

            names, error = discover_models()
            self._sig_discovery_done.emit(names, error)

        threading.Thread(target=_work, daemon=True).start()

    def _on_discovery_done(self, new_names: object, error: object) -> None:
        self._discover_btn.setEnabled(True)
        self._discover_btn.setText("Check Hugging Face for new models")
        if error:
            self._status_label.setText(f"Discovery failed: {error}")
            return
        names = new_names if isinstance(new_names, list) else []
        if names:
            self._status_label.setText(f"Found new models: {', '.join(names)}")
            self._refresh_seg_combo_items()
        else:
            self._status_label.setText("No new models found.")
        # Re-render the Models tab so newly registered models get cards.
        threading.Thread(target=self._refresh_model_status, daemon=True).start()

    def _refresh_seg_combo_items(self) -> None:
        """Add any newly registered segmentation models to the dropdown without
        disturbing the current selection (discovery is segmentation-only)."""
        from deepreefmap.segmentation.registry import list_segmentation_models

        existing = {self._seg_combo.itemText(i) for i in range(self._seg_combo.count())}
        self._seg_combo.blockSignals(True)
        try:
            for name in list_segmentation_models():
                if name not in existing:
                    self._seg_combo.addItem(name)
        finally:
            self._seg_combo.blockSignals(False)

    def _apply_model_status(self, auth_user: str | None, model_states: list) -> None:
        self._hf_auth_user = auth_user
        self._last_model_states = list(model_states)
        self._update_models_button_status()
        can_gated = getattr(self, "_can_read_gated", True)
        if auth_user:
            if can_gated:
                self._hf_auth_label.setText(f"Logged in to Hugging Face as <b>{auth_user}</b>")
                self._hf_auth_label.setToolTip(
                    f"Signed in to Hugging Face as {auth_user}. Click Log out to remove the saved token."
                )
                self._hf_auth_icon.setText(f'<span style="color:{SUCCESS}; font-weight:bold">●</span>')
                self._hf_auth_icon.setToolTip("Signed in to Hugging Face")
            else:
                self._hf_auth_label.setText(
                    f'Logged in as <b>{auth_user}</b>. '
                    f'<span style="color:{WARNING}">Token lacks gated repo access. '
                    f'Edit your token at '
                    f'<a href="https://huggingface.co/settings/tokens" style="color:{WARNING}">'
                    f'huggingface.co/settings/tokens</a> and enable '
                    f'"Read access to contents of all public gated repos".</span>'
                )
                self._hf_auth_label.setTextFormat(Qt.TextFormat.RichText)
                self._hf_auth_label.setOpenExternalLinks(True)
                self._hf_auth_label.setToolTip(
                    "Your fine-grained token does not have the 'Read access to contents of all "
                    "public gated repos you can access' permission. Edit the token on "
                    "huggingface.co/settings/tokens to enable it."
                )
                self._hf_auth_icon.setText(f'<span style="color:{WARNING}; font-weight:bold">!</span>')
                self._hf_auth_icon.setToolTip("Token missing gated repo permission")
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
                    f'  <span style="color:{WARNING}">(needed for '
                    f'{", ".join(gated_required)})</span>'
                )
            self._hf_auth_label.setText(label)
            self._hf_auth_label.setToolTip(
                "Some gated models need a Hugging Face account. "
                "Click Log in… to paste an access token from huggingface.co/settings/tokens."
            )
            self._hf_auth_icon.setText(f'<span style="color:{WARNING}; font-weight:bold">!</span>')
            self._hf_auth_icon.setToolTip(
                "Hugging Face login required to download gated models. "
                "Click Log in… to paste an access token."
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
            if item is None:
                continue
            iw = item.widget()
            if iw is not None:
                iw.deleteLater()

        required = self._required_model_names()
        ordered_states = sorted(
            model_states,
            key=lambda s: (s[0].name in required, s[0].release_date or ""),
            reverse=True,
        )
        grid_row = 0
        for info, cached in ordered_states:
            # Name on line one, size/date/REQUIRED on a wrapped second line.
            # Keeping the metadata off the name line shrinks the row's minimum
            # width so the action button never gets clipped in a narrow sidebar.
            name_html = f'<span style="color:#cfd">{info.name}</span>'
            meta_parts: list[str] = []
            if info.approx_size_mb:
                size_text = (
                    f"~{info.approx_size_mb / 1024:.1f} GB"
                    if info.approx_size_mb >= 1024
                    else f"~{info.approx_size_mb} MB"
                )
                meta_parts.append(f'<span style="color:{TEXT_DIM}; font-size:10px">{size_text}</span>')
            if info.release_date:
                meta_parts.append(
                    f'<span style="color:{TEXT_DIM}; font-size:10px">({info.release_date})</span>'
                )
            if info.name in required:
                meta_parts.append(
                    f'<span style="color:{WARNING}; '
                    'font-size:10px; font-weight:bold">REQUIRED</span>'
                )
            if meta_parts:
                name_html += "<br>" + "&nbsp;".join(meta_parts)
            name_label = QLabel(name_html)
            name_label.setWordWrap(True)
            # Ignored horizontal policy lets the label give up width freely so
            # the fixed-width action button in column 1 is always visible.
            name_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self._models_grid.addWidget(name_label, grid_row, 0)

            action = self._make_action_widget(info, cached, auth_user)
            self._models_grid.addWidget(action, grid_row, 1)
            self._model_rows[info.name] = name_label
            self._model_actions[info.name] = action
            grid_row += 1

        self._recompute_submit_state()

    def _required_model_names(self) -> set[str]:
        required = {self._map_combo.currentText()}
        if not self._skip_seg_check.isChecked():
            seg = self._seg_combo.currentText()
            required.add(seg)
            from deepreefmap.gui.models.manager import DPT_BACKBONE_MAP

            backbone = DPT_BACKBONE_MAP.get(seg)
            if backbone:
                required.add(backbone)
        return required

    def _on_required_models_changed(self, _value: object = "") -> None:
        if self._last_model_states:
            self._apply_model_status(self._hf_auth_user, self._last_model_states)
        self._recompute_submit_state()

    def _make_action_widget(self, info, cached: bool, auth_user: str | None) -> QWidget:
        from deepreefmap.gui.models.manager import model_available

        if not model_available(info):
            # Model needs an install extra that isn't present (LoGeR today).
            # Show it greyed with a hint instead of a Download button.
            container = QWidget()
            hb = QHBoxLayout(container)
            hb.setContentsMargins(0, 0, 0, 0)
            from deepreefmap.mapping.registry import LOGER_INSTALL_HINT

            label = QLabel(f'<span style="color:{TEXT_DIM}">install required</span>')
            label.setToolTip(LOGER_INSTALL_HINT)
            hb.addWidget(label)
            return container

        if info.name in self._downloading:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("Downloading %p%")
            bar.setFixedWidth(150)
            return bar

        # A cache that exists but fails verification (interrupted/cancelled
        # download: config.json landed, weights or the custom loader did not).
        # Distinct from "never downloaded" so the row can prompt a repair
        # instead of a silent re-download that reads as "nothing happened".
        from deepreefmap.gui.models.manager import ModelStatus, model_status

        partial_reason = ""
        if not cached:
            status, why = model_status(info)
            if status == ModelStatus.PARTIAL:
                partial_reason = why

        container = QWidget()
        hb = QHBoxLayout(container)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(6)

        icon = QLabel()
        icon.setFixedWidth(14)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if cached:
            icon.setText(f'<span style="color:{SUCCESS}; font-weight:bold">✓</span>')
            icon.setToolTip("cached")
        elif partial_reason:
            icon.setText(f'<span style="color:{WARNING}; font-weight:bold">⚠</span>')
            icon.setToolTip(f"download incomplete: {partial_reason}")
        elif info.gated and not auth_user:
            icon.setText(f'<span style="color:{WARNING}; font-weight:bold">!</span>')
            icon.setToolTip(
                "Hugging Face login required for this gated model. "
                "Click Log in… above to paste an access token."
            )
        else:
            icon.setText(f'<span style="color:{TEXT_DIM}">○</span>')
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
            prior_error = self._download_errors.get(info.name)
            if partial_reason:
                btn_text = "Repair"
            elif prior_error:
                btn_text = "Retry"
            else:
                btn_text = "Download"
            btn = QPushButton(btn_text)
            btn.setFixedWidth(110)
            if partial_reason:
                # Re-download fills in the missing files. Flag it orange so an
                # incomplete cache doesn't masquerade as a fresh download.
                btn.setToolTip(
                    f"Cached files are incomplete ({partial_reason}). Click to re-download."
                )
                btn.setStyleSheet(f"QPushButton {{ color: {WARNING}; }}")
            elif prior_error:
                # Surface the failure at the row so it survives the next
                # status refresh, instead of disappearing from the shared
                # status bar the moment the user clicks anywhere else.
                btn.setToolTip(f"Previous download failed: {prior_error}\nClick to retry.")
                btn.setStyleSheet(f"QPushButton {{ color: {WARNING}; }}")
            model_name = info.name
            btn.clicked.connect(lambda checked=False, n=model_name: self._download_model(n))
        hb.addWidget(btn)
        return container

    def _on_hf_auth_button(self) -> None:
        if self._hf_auth_user:
            self._hf_auth_btn.setEnabled(False)
            self._status_label.setText("Logging out of Hugging Face...")

            def _do_logout() -> None:
                from deepreefmap.gui.models.manager import hf_logout

                try:
                    hf_logout()
                    self._sig_hf_auth_done.emit(None, "")
                except Exception as exc:
                    self._sig_hf_auth_done.emit(self._hf_auth_user, str(exc)[:200])

            threading.Thread(target=_do_logout, daemon=True).start()
            return

        dlg = HfLoginDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        token = dlg.token()
        if not token:
            return

        self._hf_auth_btn.setEnabled(False)
        self._status_label.setText("Logging in to Hugging Face...")

        def _do_login() -> None:
            from deepreefmap.gui.models.manager import hf_login

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
        btn.setStyleSheet(f"background-color: {DANGER_BG}; color: white; font-weight: bold;")

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
        from deepreefmap.gui.models.manager import all_known_models, delete_model

        info = next((m for m in all_known_models() if m.name == model_name), None)
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
        row, col, _, _ = cast("tuple[int, int, int, int]", self._models_grid.getItemPosition(idx))
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
        for btn in self._form_status_buttons_for(model_name):
            self._apply_downloading_style(btn, model_name, percent)

    def _on_hf_auth_done(self, user: object, error: str) -> None:
        if error:
            self._status_label.setText(f"Hugging Face auth failed: {error}")
        elif user:
            self._status_label.setText(f"Logged in to Hugging Face as {user}.")
        else:
            self._status_label.setText("Logged out of Hugging Face.")
        threading.Thread(target=self._refresh_model_status, daemon=True).start()

    def _download_model(self, model_name: str) -> None:
        from deepreefmap.gui.models.manager import (
            DownloadCancelled,
            all_known_models,
            prefetch_model,
        )

        info = next((m for m in all_known_models() if m.name == model_name), None)
        if info is None or model_name in self._downloading:
            return
        self._status_label.setText(f"Downloading model {model_name}...")
        self._downloading.add(model_name)
        self._download_cancel_requested.discard(model_name)
        self._download_errors.pop(model_name, None)
        self._swap_action_to_progress(model_name)
        for btn in self._form_status_buttons_for(model_name):
            self._apply_downloading_style(btn, model_name, 0)

        def _progress(n: int, total: int) -> None:
            if model_name in self._download_cancel_requested:
                raise DownloadCancelled()
            if total <= 0:
                return
            self._sig_download_progress.emit(model_name, int(100 * n / total))

        def _do_download() -> None:
            try:
                prefetch_model(info, progress_cb=_progress)
                self._sig_status_text.emit(f"Model {model_name} downloaded.")
            except DownloadCancelled:
                self._sig_status_text.emit(f"Download of {model_name} cancelled.")
            except Exception as exc:
                msg = str(exc)[:200]
                self._download_errors[model_name] = msg
                self._sig_status_text.emit(f"Download failed: {msg}")
            finally:
                self._downloading.discard(model_name)
                self._download_cancel_requested.discard(model_name)
                threading.Thread(target=self._refresh_model_status, daemon=True).start()

        threading.Thread(target=_do_download, daemon=True).start()

    def _cancel_download(self, model_name: str) -> None:
        if model_name not in self._downloading:
            return
        self._download_cancel_requested.add(model_name)
        self._status_label.setText(f"Cancelling download of {model_name}…")
        # Re-render any matching form buttons so they show the "…" cancelling
        # tooltip immediately, without waiting for the next progress tick.
        for btn in self._form_status_buttons_for(model_name):
            self._apply_downloading_style(
                btn, model_name, int(btn.property("downloadPercent") or 0)
            )
