from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

logger = logging.getLogger(__name__)

# Structural greys, matching the dark look the app was already built around.
WINDOW = "#2b2b2b"
WINDOW_TEXT = "#e0e0e0"
BASE = "#1e1e1e"
ALT_BASE = "#262626"
BUTTON = "#353535"
BORDER = "#555555"
GROOVE = "#3a3a3a"
TEXT_MUTED = "#aaaaaa"
DISABLED_FG = "#6f6f6f"

# Named semantic accents. These consolidate several inconsistent spellings that
# were scattered across the GUI (e.g. success was both "#4a4" and
# QColor(74, 170, 74)); migrate call sites onto these so there's one value each.
SUCCESS = "#4aaa4a"
WARNING = "#e8a04a"
ERROR = "#ff6b5e"  # brighter than the old #c0392b/#c84 for contrast on dark
PRIMARY = "#4aa3ff"
LINK = "#9ecbff"
UPDATE = "#e0a030"
DANGER_BG = "#8a2222"  # filled "Confirm delete?" button; distinct from ERROR text

# Compound tokens that travel together (background + text + border).
BANNER_BG, BANNER_TEXT, BANNER_BORDER = "#1f2a36", "#d8e2ec", "#2f3f50"
WARN_BG, WARN_TEXT, WARN_BORDER = "#4a3a14", "#ffd98a", "#8a6b1a"

# Layered on top of Fusion + the palette for the few things the palette alone
# doesn't make consistent. Additive: existing per-widget stylesheets still
# cascade over this and win on conflicts.
GLOBAL_QSS = f"""
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: {WINDOW_TEXT};
}}
QToolTip {{
    color: {WINDOW_TEXT};
    background-color: {GROOVE};
    border: 1px solid {BORDER};
    padding: 3px 6px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    top: -1px;
}}
QTabBar::tab {{
    background: {BUTTON};
    color: {TEXT_MUTED};
    padding: 6px 10px;
    border: 1px solid {BORDER};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background: {WINDOW};
    color: {WINDOW_TEXT};
}}
QSplitter::handle {{
    background-color: {WINDOW};
}}
QSplitter::handle:hover {{
    background-color: {BORDER};
}}
*:focus {{
    outline: none;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus, QAbstractSpinBox:focus {{
    border: 1px solid {PRIMARY};
}}
"""


def apply_theme(app: QApplication) -> None:
    """Force a consistent dark theme regardless of the host OS appearance.

    Without this the app inherits each platform's palette — dark on a dark
    Linux desktop, but white on a macOS machine in Light mode, where standard
    widgets render light while the hardcoded-dark stylesheets stay dark. Fusion
    plus an explicit dark palette makes the UI identical on Linux/Windows/macOS.
    """
    try:
        app.setStyle("Fusion")
    except Exception:
        logger.warning("Could not set Fusion style; keeping platform style", exc_info=True)

    # Tell the platform we want dark so native chrome follows — notably the
    # macOS window titlebar and native dialogs, which Fusion + the palette
    # don't reach. Needs Qt 6.8+; harmless no-op on older builds.
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except (AttributeError, TypeError):
        logger.debug("setColorScheme unavailable (needs Qt 6.8+)", exc_info=True)

    role = QPalette.ColorRole
    group = QPalette.ColorGroup
    pal = QPalette()
    pal.setColor(role.Window, QColor(WINDOW))
    pal.setColor(role.WindowText, QColor(WINDOW_TEXT))
    pal.setColor(role.Base, QColor(BASE))
    pal.setColor(role.AlternateBase, QColor(ALT_BASE))
    pal.setColor(role.ToolTipBase, QColor(GROOVE))
    pal.setColor(role.ToolTipText, QColor(WINDOW_TEXT))
    pal.setColor(role.Text, QColor(WINDOW_TEXT))
    pal.setColor(role.PlaceholderText, QColor("#8a8a8a"))
    pal.setColor(role.Button, QColor(BUTTON))
    pal.setColor(role.ButtonText, QColor(WINDOW_TEXT))
    pal.setColor(role.BrightText, QColor("#ffffff"))
    pal.setColor(role.Link, QColor(LINK))
    pal.setColor(role.Highlight, QColor(PRIMARY))
    pal.setColor(role.HighlightedText, QColor("#101010"))

    pal.setColor(group.Disabled, role.Text, QColor(DISABLED_FG))
    pal.setColor(group.Disabled, role.WindowText, QColor(DISABLED_FG))
    pal.setColor(group.Disabled, role.ButtonText, QColor(DISABLED_FG))
    pal.setColor(group.Disabled, role.Highlight, QColor(GROOVE))
    pal.setColor(group.Disabled, role.HighlightedText, QColor(DISABLED_FG))
    app.setPalette(pal)

    app.setStyleSheet(GLOBAL_QSS)
