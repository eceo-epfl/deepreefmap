from __future__ import annotations

from deepreefmap.memory_estimate import Verdict

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class PreflightDialog(QDialog):
    """Confirm/cancel gate shown before a run the memory check flags as risky.

    The estimate can be wrong, so the user can always override: a `block` verdict
    just defaults the focus to Cancel and warns harder, it never forbids the run.
    """

    def __init__(self, verdict: Verdict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        blocking = verdict.level == "block"
        self.setWindowTitle("Not enough memory?" if blocking else "Memory may be tight")
        self.setModal(True)

        layout = QVBoxLayout(self)
        label = QLabel(verdict.message)
        label.setWordWrap(True)
        layout.addWidget(label)

        buttons = QDialogButtonBox()
        run_btn = buttons.addButton("Run anyway", QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_btn = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # A block verdict defaults to Cancel; a warn defaults to proceeding.
        cancel_btn.setDefault(blocking)
        run_btn.setDefault(not blocking)
        layout.addWidget(buttons)

        self.resize(460, 150)

    def confirmed(self) -> bool:
        """True if the user chose to run despite the warning."""
        return self.exec() == QDialog.DialogCode.Accepted
