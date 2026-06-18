"""Semantic legend, cover sunburst and viewer-overlay controls."""

from __future__ import annotations

import pytest


def test_overlay_has_reset_button_and_r_shortcut_triggers_view_reset(qapp) -> None:
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from PySide6.QtGui import QKeySequence, QShortcut

    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)
    assert window._reset_view_button is not None
    assert "Reset" in window._reset_view_button.text()

    calls: list[int] = []
    window._viewer.reset_view = lambda: calls.append(1)  # type: ignore[method-assign]

    window._reset_view_button.click()
    assert calls == [1]

    # QShortcut registered for R must exist and, when activated, fire the
    # same reset_view path. We trigger it via `activated.emit()` rather than
    # synthesising a key event, because offscreen windows aren't "active", which
    # would suppress the natural shortcut dispatch.
    r_seq = QKeySequence("R")
    r_shortcuts = [
        s for s in window.findChildren(QShortcut)
        if s.key() == r_seq
    ]
    assert r_shortcuts, "expected an R shortcut on the window"
    r_shortcuts[0].activated.emit()
    assert len(calls) == 2


def test_legend_overlay_reorder_places_rows(qapp) -> None:
    from PySide6.QtWidgets import QWidget

    from deepreefmap.gui.viewer.legend import LegendOverlay

    parent = QWidget()
    ov = LegendOverlay(parent)
    names = {1: "alpha", 2: "beta", 3: "gamma"}
    colors = {1: (1, 1, 1), 2: (2, 2, 2), 3: (3, 3, 3)}
    ov.rebuild([1, 2, 3], names, colors, on_toggle=lambda: None, class_counts={1: 10, 2: 20, 3: 30})
    ov.reorder([3, 1, 2])

    def row_of(cid: int) -> int:
        cb = ov._rows[cid][1]
        return ov._grid.getItemPosition(ov._grid.indexOf(cb))[0]

    assert row_of(3) < row_of(1) < row_of(2)


def test_legend_sort_selected_first_puts_checked_above_unchecked(window) -> None:
    window._build_legend()
    cids = list(window._legend_toggles.keys())
    assert len(cids) >= 3
    window._legend_toggles[cids[0]].setChecked(False)
    window._legend_toggles[cids[1]].setChecked(False)

    order = window._legend_sort_order()
    enabled = window._enabled_class_set()
    last_visible = max(i for i, c in enumerate(order) if c in enabled)
    first_hidden = min(i for i, c in enumerate(order) if c not in enabled)
    assert last_visible < first_hidden


def test_legend_sort_header_click_toggles_direction(window) -> None:
    window._build_legend()
    assert (window._legend_sort_mode, window._legend_sort_ascending) == ("selected", False)

    window._on_legend_sort_clicked("name")  # new column adopts its default (A–Z)
    assert (window._legend_sort_mode, window._legend_sort_ascending) == ("name", True)
    window._on_legend_sort_clicked("name")  # same column flips direction
    assert window._legend_sort_ascending is False
    window._on_legend_sort_clicked("size")  # new column adopts default (largest first)
    assert (window._legend_sort_mode, window._legend_sort_ascending) == ("size", False)


def test_pie_click_toggles_selection(window) -> None:
    window._build_legend()
    cids = list(window._legend_toggles.keys())
    assert len(cids) >= 2

    window._on_deselect_all_classes()
    assert window._enabled_class_set() == frozenset()
    window._on_sunburst_selection([cids[0]])  # add
    assert window._enabled_class_set() == frozenset({cids[0]})
    window._on_sunburst_selection([cids[1]])  # additive, keeps the first
    assert window._enabled_class_set() == frozenset({cids[0], cids[1]})
    window._on_sunburst_selection([cids[0]])  # re-click removes it
    assert window._enabled_class_set() == frozenset({cids[1]})


def test_master_checkbox_select_deselect_and_partial(qapp) -> None:
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from PySide6.QtCore import Qt

    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    cc = load_classes()
    window = DeepReefMapWindow(cc, None)
    window._build_legend()
    present = frozenset(window._legend_toggles.keys())
    master = window._viewer.legend_overlay._master_check

    assert window._enabled_class_set() == present  # all on at build
    window._on_master_clicked()  # all -> none
    assert window._enabled_class_set() == frozenset()
    window._on_master_clicked()  # none -> all
    assert window._enabled_class_set() == present

    next(iter(window._legend_toggles.values())).setChecked(False)
    window._update_master_check()
    assert master.checkState() == Qt.CheckState.PartiallyChecked


def test_sunburst_reflects_selection(window, monkeypatch) -> None:
    # The sunburst sync runs through _on_viewer_control_changed, which only acts
    # once the viewer has scene data (as when a run is loaded). Patch via
    # monkeypatch so the class property is restored at teardown and later tests
    # (which may assert has_scene_data is False) aren't affected.
    window._viewer.apply_state = lambda **k: None
    monkeypatch.setattr(type(window._viewer), "has_scene_data", property(lambda self: True))
    window._build_legend()
    sb = window._cover_sunburst

    assert sb._selection_active is False  # all selected at build -> no dimming
    window._on_deselect_all_classes()
    assert sb._selection_active is True and sb._selected_ids == frozenset()
    cids = list(window._legend_toggles.keys())
    window._on_sunburst_selection([cids[0]])
    assert sb._selected_ids == frozenset({cids[0]}) and sb._selection_active is True
    window._on_show_all_classes()
    assert sb._selection_active is False

