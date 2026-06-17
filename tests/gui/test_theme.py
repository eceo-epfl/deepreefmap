"""Theme application and token validity."""

from __future__ import annotations


def test_apply_theme_sets_dark_palette(qapp) -> None:
    # apply_theme mutates the shared app, so snapshot and restore to keep other
    # tests isolated. We assert on the palette (not style().objectName(), which
    # the global stylesheet wraps in an empty-named QStyleSheetStyle proxy).
    from PySide6.QtGui import QPalette

    from deepreefmap.gui.core.theme import apply_theme

    prev_style = qapp.style().objectName()
    prev_palette = QPalette(qapp.palette())
    prev_qss = qapp.styleSheet()
    try:
        apply_theme(qapp)
        win = qapp.palette().color(QPalette.ColorRole.Window)
        base = qapp.palette().color(QPalette.ColorRole.Base)
        assert win.red() < 80 and win.green() < 80 and win.blue() < 80
        assert base.lightness() < win.lightness()
    finally:
        qapp.setStyleSheet(prev_qss)
        qapp.setPalette(prev_palette)
        if prev_style:
            qapp.setStyle(prev_style)


def test_theme_semantic_constants_are_valid_hex() -> None:
    from PySide6.QtGui import QColor

    from deepreefmap.gui.core import theme

    for name in ("SUCCESS", "WARNING", "ERROR", "PRIMARY", "LINK", "UPDATE", "DANGER_BG"):
        assert QColor(getattr(theme, name)).isValid()

