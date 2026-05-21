from __future__ import annotations

import threading

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QWidget,
)

from deepreefmap.launcher.qt_app_hf_dialog import HfLoginDialog


class ModelManagementMixin:
    """DeepReefMapWindow methods for HF auth, model status, download, and delete."""

    def _find_model_state(self, model_name: str) -> tuple[object, bool]:
        for state_info, state_cached in self._last_model_states:
            if state_info.name == model_name:
                return state_info, state_cached
        return None, False

    def _jump_to_model(self, model_name: str | None = None) -> None:
        """Switch the sidebar to the Models tab and reveal one row.

        Used by the inline status buttons so the cached case ("Click ✓")
        takes the user straight to the Delete control for that model.
        """
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

    def _flash_model_row(self, label: QLabel) -> None:
        prev = label.styleSheet()
        label.setStyleSheet(
            "QLabel { background-color: rgba(232, 160, 74, 60);"
            " border: 1px solid #e8a04a; border-radius: 3px; padding: 2px; }"
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
        info, cached = self._find_model_state(selected_name)
        if info is None:
            btn.setText("…")
            btn.setToolTip("Open Models")
            btn.setStyleSheet("")
            return
        if cached:
            btn.setText("✓")
            btn.setToolTip(f"{selected_name} is downloaded. Click to manage cache.")
            btn.setStyleSheet("QPushButton { color: #4a4; font-weight: bold; }")
        elif info.gated and self._hf_auth_user is None:
            btn.setText("🔒")
            btn.setToolTip(
                f"{selected_name} is gated. Click to log in to Hugging Face."
            )
            btn.setStyleSheet("QPushButton { color: #e8a04a; font-weight: bold; }")
        else:
            btn.setText("⬇")
            btn.setToolTip(f"{selected_name} not downloaded. Click to download.")
            btn.setStyleSheet("QPushButton { color: #e8a04a; font-weight: bold; }")

    def _update_models_button_status(self) -> None:
        """Refresh the per-dropdown model status icons.

        Driven from _apply_model_status whenever the model cache or HF login
        state changes; also after the user picks a different model in either
        dropdown.
        """
        if hasattr(self, "_seg_status_btn") and hasattr(self, "_seg_combo"):
            self._update_model_status_button(
                self._seg_status_btn, self._seg_combo.currentText()
            )
        if hasattr(self, "_map_status_btn") and hasattr(self, "_map_combo"):
            self._update_model_status_button(
                self._map_status_btn, self._map_combo.currentText()
            )

    def _refresh_model_status(self) -> None:
        from deepreefmap.launcher.model_manager import ALL_MODELS, check_hf_auth, is_model_cached

        auth_user = check_hf_auth()
        model_states = [(m, is_model_cached(m)) for m in ALL_MODELS]
        self._sig_model_status_done.emit(auth_user, model_states)

    def _apply_model_status(self, auth_user: str | None, model_states: list) -> None:
        self._hf_auth_user = auth_user
        self._last_model_states = list(model_states)
        self._update_models_button_status()
        if auth_user:
            self._hf_auth_label.setText(f"Logged in to Hugging Face as <b>{auth_user}</b>")
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

        dlg = HfLoginDialog(self)
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
