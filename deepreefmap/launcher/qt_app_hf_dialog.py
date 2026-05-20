from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class HfLoginDialog(QDialog):
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
