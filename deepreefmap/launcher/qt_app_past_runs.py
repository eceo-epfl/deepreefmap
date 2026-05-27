from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

logger = logging.getLogger(__name__)


_PAST_RUN_META_ROLE = Qt.ItemDataRole.UserRole + 1

_GEOMETRY_LABELS = {
    "world_points": "world points (full)",
    "depth_unprojection": "depth-unprojection",
}


def _format_timestamp(value: object) -> str:
    """Render an ISO-8601 run_timestamp as a short local date/time, else passthrough."""
    if not isinstance(value, str):
        return ""
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def _run_sort_key(manifest: dict, mtime: float) -> float:
    """Prefer the recorded run timestamp; fall back to the manifest file mtime."""
    ts = manifest.get("run_timestamp")
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts).timestamp()
        except ValueError:
            pass
    return mtime


def _format_disk_size(run_dir: Path) -> str | None:
    try:
        total = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    except Exception:
        return None
    if total >= 1e9:
        return f"{total / 1e9:.2f} GB"
    return f"{total / 1e6:.1f} MB"


class _PastRunCardDelegate(QStyledItemDelegate):
    """Paints each past-run dropdown item as a multi-line card with metadata.

    All sizes are computed from QFontMetrics so the card scales correctly with
    system DPI / font size on both Linux and Windows. Facts wrap to multiple
    lines if the popup is narrower than the text, so nothing is silently lost.
    """

    PAD_X_EMS = 0.8
    PAD_Y_EMS = 0.35
    GAP_EMS = 0.15

    @staticmethod
    def _title_font(base: QFont) -> QFont:
        f = QFont(base)
        f.setBold(True)
        return f

    @staticmethod
    def _slug_font(base: QFont) -> QFont:
        f = QFont(base)
        pt = base.pointSize() if base.pointSize() > 0 else 10
        f.setPointSize(max(8, pt - 1))
        return f

    @staticmethod
    def _facts_font(base: QFont) -> QFont:
        return QFont(base)

    @staticmethod
    def _video_font(base: QFont) -> QFont:
        f = QFont(base)
        pt = base.pointSize() if base.pointSize() > 0 else 10
        f.setPointSize(max(8, pt - 1))
        f.setItalic(True)
        return f

    def _layout(self, option: QStyleOptionViewItem, meta: dict, avail_w: int) -> dict:
        base = option.font
        title_fm = option.fontMetrics  # used for em sizing
        em = max(1, title_fm.height())
        pad_x = int(self.PAD_X_EMS * em)
        pad_y = int(self.PAD_Y_EMS * em)
        gap = int(self.GAP_EMS * em)

        from PySide6.QtGui import QFontMetrics

        title_h = QFontMetrics(self._title_font(base)).height()
        slug_h = QFontMetrics(self._slug_font(base)).height() if meta.get("slug") else 0
        head_h = max(title_h, slug_h)

        facts_text = meta.get("facts") or ""
        facts_h = 0
        if facts_text:
            facts_fm = QFontMetrics(self._facts_font(base))
            inner_w = max(40, avail_w - pad_x * 2)
            facts_rect = facts_fm.boundingRect(
                0, 0, inner_w, 10_000,
                Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                facts_text,
            )
            facts_h = facts_rect.height()

        video_text = meta.get("video") or ""
        video_h = 0
        if video_text:
            video_h = QFontMetrics(self._video_font(base)).height()

        total_h = pad_y * 2 + head_h
        if facts_h:
            total_h += gap + facts_h
        if video_h:
            total_h += gap + video_h

        return {
            "pad_x": pad_x, "pad_y": pad_y, "gap": gap,
            "head_h": head_h, "title_h": title_h, "slug_h": slug_h,
            "facts_h": facts_h, "video_h": video_h, "total_h": total_h,
        }

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        meta = index.data(_PAST_RUN_META_ROLE)
        em = max(1, option.fontMetrics.height())
        # Preferred width: ~32 chars of body text plus padding. The view can
        # be wider; the layout will fill it. EM-based so it scales with DPI.
        preferred_w = int(em * 24)
        if meta is None:
            # Placeholder row stays one line tall.
            return QSize(preferred_w, em + int(self.PAD_Y_EMS * em) * 2)

        # Use the actual viewport width when available; fall back to preferred.
        avail_w = option.rect.width() if option.rect.width() > 0 else preferred_w
        layout = self._layout(option, meta, avail_w)
        return QSize(preferred_w, layout["total_h"])

    def paint(self, painter, option: QStyleOptionViewItem, index) -> None:
        meta = index.data(_PAST_RUN_META_ROLE)
        if meta is None:
            super().paint(painter, option, index)
            return

        painter.save()

        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if selected:
            painter.fillRect(option.rect, QColor("#4a7fb0"))
        elif hovered:
            painter.fillRect(option.rect, QColor("#3a5f8a"))
        else:
            painter.fillRect(option.rect, QColor("#2a2a2a"))

        layout = self._layout(option, meta, option.rect.width())
        pad_x = layout["pad_x"]
        pad_y = layout["pad_y"]
        gap = layout["gap"]
        r = option.rect.adjusted(pad_x, pad_y, -pad_x, -pad_y)

        base = option.font

        # Title.
        title_font = self._title_font(base)
        painter.setFont(title_font)
        painter.setPen(QColor("white" if (hovered or selected) else "#e8eef5"))
        title = meta.get("title", "")
        title_fm = painter.fontMetrics()
        title_w = title_fm.horizontalAdvance(title)
        baseline = r.top() + title_fm.ascent()
        painter.drawText(r.left(), baseline, title)

        # Slug, drawn on the same baseline as the title (or hidden if no room).
        slug = meta.get("slug", "")
        if slug:
            slug_font = self._slug_font(base)
            painter.setFont(slug_font)
            painter.setPen(QColor("#c5d0db" if (hovered or selected) else "#8aa0b8"))
            slug_fm = painter.fontMetrics()
            slug_x = r.left() + title_w + int(layout["title_h"] * 0.4)
            slug_max_w = r.right() - slug_x
            if slug_max_w > 0:
                elided_slug = slug_fm.elidedText(slug, Qt.TextElideMode.ElideRight, slug_max_w)
                painter.drawText(slug_x, baseline, elided_slug)

        # Facts (word-wrapped block).
        cursor_y = r.top() + layout["head_h"]
        facts_text = meta.get("facts", "")
        if facts_text:
            cursor_y += gap
            painter.setFont(self._facts_font(base))
            painter.setPen(QColor("#dfe6ee" if (hovered or selected) else "#c0cad6"))
            facts_rect = type(r)(r.left(), cursor_y, r.width(), layout["facts_h"])
            painter.drawText(
                facts_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                facts_text,
            )
            cursor_y += layout["facts_h"]

        # Input video (single elided line).
        video = meta.get("video", "")
        if video:
            cursor_y += gap
            painter.setFont(self._video_font(base))
            painter.setPen(QColor("#b5c2d0" if (hovered or selected) else "#7a8a99"))
            video_fm = painter.fontMetrics()
            elided = video_fm.elidedText(video, Qt.TextElideMode.ElideMiddle, r.width())
            painter.drawText(r.left(), cursor_y + video_fm.ascent(), elided)

        painter.restore()


class PastRunsMixin:
    """DeepReefMapWindow methods for the past-runs dropdown, banner, and rename UI."""

    def _refresh_past_runs_combo(self) -> None:
        root = Path(self._out_root_input.text()).expanduser()
        entries: list[tuple[Path, str, float, dict]] = []
        if root.exists() and root.is_dir():
            for child in root.iterdir():
                manifest = child / "run_manifest.json"
                if not (child.is_dir() and manifest.exists()):
                    continue
                display = child.name
                data: dict = {}
                try:
                    data = json.loads(manifest.read_text())
                    name = data.get("name")
                    if name:
                        display = f"{name}  ({child.name})"
                except Exception:
                    pass
                entries.append((child, display, manifest.stat().st_mtime, data))
        entries.sort(key=lambda e: _run_sort_key(e[3], e[2]), reverse=True)

        # Block signals to avoid triggering _on_past_run_selected during repopulation.
        self._past_runs_combo.blockSignals(True)
        try:
            self._past_runs_combo.clear()
            self._past_runs_combo.addItem("— Select a past run —", userData=None)
            for path, display, _mtime, data in entries:
                self._past_runs_combo.addItem(display, userData=str(path))
                idx = self._past_runs_combo.count() - 1
                tooltip = self._format_run_metadata(data, path, include_disk_size=False)
                self._past_runs_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
                self._past_runs_combo.setItemData(
                    idx,
                    self._build_past_run_card_meta(data, path),
                    _PAST_RUN_META_ROLE,
                )
            if self._active_run_dir is not None:
                for i in range(1, self._past_runs_combo.count()):
                    if self._past_runs_combo.itemData(i) == str(self._active_run_dir):
                        self._past_runs_combo.setCurrentIndex(i)
                        break
        finally:
            self._past_runs_combo.blockSignals(False)

    @staticmethod
    def _format_run_metadata(manifest: dict, run_dir: Path, *, include_disk_size: bool) -> str:
        """Multi-line format used in tooltips and the sidebar Results block."""
        lines: list[str] = []
        name = (manifest.get("name") or "").strip() or run_dir.name
        lines.append(f"<b>{name}</b>  <i>({run_dir.name})</i>")
        mode = manifest.get("mode")
        if mode:
            lines.append(f"Mode: {mode}")
        seg = manifest.get("segmentation_model")
        if seg:
            lines.append(f"Segmentation: {seg}")
        mapping = manifest.get("mapping_backend")
        if mapping:
            lines.append(f"Mapping: {mapping}")
        mopts = manifest.get("mapping_options") or {}
        if mopts.get("window_size") is not None:
            lines.append(
                f"LoGeR window/overlap: {mopts.get('window_size')}/{mopts.get('overlap_size')}"
            )
        if manifest.get("refine_intrinsics_from_mapper"):
            lines.append("Intrinsics: refined from mapper")
        geom = manifest.get("geometry_source")
        if geom:
            lines.append(f"Geometry: {_GEOMETRY_LABELS.get(geom, geom)}")
        profile = manifest.get("camera_profile")
        if profile:
            lines.append(f"Camera profile: {profile}")
        frames = manifest.get("frames_processed")
        if frames is not None:
            fps = manifest.get("fps")
            lines.append(f"Frames: {frames}" + (f" @ {fps} fps" if fps else ""))
        pw, ph = manifest.get("processing_width"), manifest.get("processing_height")
        if pw and ph:
            lines.append(f"Processing size: {pw}×{ph}")
        sem_pts = manifest.get("semantic_reference_points")
        if sem_pts:
            lines.append(f"Semantic points: {int(sem_pts):,}")
        metric_pts = manifest.get("metric_points")
        if metric_pts:
            lines.append(f"Metric points: {int(metric_pts):,}")
        videos = manifest.get("input_videos") or []
        if videos:
            lines.append(f"Input: {', '.join(Path(v).name for v in videos)}")
        created = _format_timestamp(manifest.get("run_timestamp"))
        if created:
            lines.append(f"Created: {created}")
        if include_disk_size:
            disk = _format_disk_size(run_dir)
            if disk:
                lines.append(f"Disk: {disk}")
        return "<br>".join(lines)

    @staticmethod
    def _build_past_run_card_meta(manifest: dict, run_dir: Path) -> dict:
        """Build a flat dict the dropdown delegate uses to paint each card."""
        name = (manifest.get("name") or "").strip() or run_dir.name
        facts: list[str] = []
        mode = manifest.get("mode")
        if mode:
            facts.append(mode)
        frames = manifest.get("frames_processed")
        if frames is not None:
            facts.append(f"{frames}f")
        seg = manifest.get("segmentation_model")
        if seg and seg != "__skip__":
            facts.append(str(seg))
        mapping = manifest.get("mapping_backend")
        if mapping:
            facts.append(str(mapping))
        geom = manifest.get("geometry_source")
        if str(mapping) in {"loger", "loger_star"} and geom:
            facts.append("world-pts" if geom == "world_points" else "⚠ depth")
        sem_pts = manifest.get("semantic_reference_points")
        if sem_pts:
            n = int(sem_pts)
            if n >= 1_000_000:
                facts.append(f"{n / 1_000_000:.1f}M pts")
            elif n >= 1_000:
                facts.append(f"{n / 1_000:.0f}k pts")
            else:
                facts.append(f"{n} pts")
        videos = manifest.get("input_videos") or []
        video_line = ""
        if videos:
            names = [Path(v).name for v in videos]
            if len(names) == 1:
                video_line = f"📹 {names[0]}"
            else:
                video_line = f"📹 {names[0]} (+ {len(names) - 1} more)"
        return {
            "title": name,
            "slug": "" if name == run_dir.name else f"({run_dir.name})",
            "facts": "  ·  ".join(facts),
            "video": video_line,
        }

    @staticmethod
    def _format_run_metadata_compact(manifest: dict, run_dir: Path, *, include_disk_size: bool) -> str:
        """Single-line wrapping format used in the inline top banner."""
        name = (manifest.get("name") or "").strip() or run_dir.name
        header = (
            f'<b style="font-size:13px">{name}</b>'
            f'&nbsp;<span style="color:#7a8a99">({run_dir.name})</span>'
        )
        facts: list[str] = []
        for label, key, fmt in (
            ("Mode", "mode", str),
            ("Frames", "frames_processed", str),
            ("Segmentation", "segmentation_model", str),
            ("Mapping", "mapping_backend", str),
            ("Geometry", "geometry_source", lambda v: _GEOMETRY_LABELS.get(v, str(v))),
            ("Camera", "camera_profile", str),
            ("Semantic pts", "semantic_reference_points", lambda v: f"{int(v):,}"),
            ("Metric pts", "metric_points", lambda v: f"{int(v):,}"),
            ("Input", "input_videos", lambda v: ", ".join(Path(p).name for p in v) if v else ""),
            ("Created", "run_timestamp", _format_timestamp),
        ):
            v = manifest.get(key)
            if v is not None and v != "" and v != []:
                facts.append(
                    f'<span style="color:#8aa0b8">{label}:</span>&nbsp;'
                    f'<span style="color:#d8e2ec">{fmt(v)}</span>'
                )
        if include_disk_size:
            disk = _format_disk_size(run_dir)
            if disk:
                facts.append(
                    f'<span style="color:#8aa0b8">Disk:</span>&nbsp;'
                    f'<span style="color:#d8e2ec">{disk}</span>'
                )
        sep = '&nbsp;<span style="color:#4a5f74">·</span>&nbsp;'
        return f"{header}&nbsp;&nbsp;{sep.join(facts)}"

    def _on_past_run_selected(self, index: int) -> None:
        if index <= 0:
            self._hide_run_meta_banner()
            return
        run_dir = self._past_runs_combo.itemData(index)
        if not run_dir:
            return
        path = Path(run_dir)
        # Show the metadata banner *immediately* from the manifest, before the
        # potentially-slow load kicks off, so the user gets instant feedback.
        manifest_path = path / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                self._show_run_meta_banner(manifest, path, include_disk_size=False)
            except Exception:
                self._hide_run_meta_banner()
        self._auto_load_run(path)

    def _show_run_meta_banner(self, manifest: dict, run_dir: Path, *, include_disk_size: bool) -> None:
        self._run_meta_banner.setText(
            self._format_run_metadata_compact(manifest, run_dir, include_disk_size=include_disk_size)
        )
        self._run_meta_banner.setVisible(True)

    def _hide_run_meta_banner(self) -> None:
        self._run_meta_banner.setVisible(False)
        self._run_meta_banner.setText("")

    def _begin_rename(self) -> None:
        if self._active_run_dir is None:
            return
        current = ""
        if self._active_run_manifest:
            current = str(self._active_run_manifest.get("name") or "")
        if not current:
            current = self._active_run_dir.name
        self._rename_edit.setText(current)
        self._rename_btn.setVisible(False)
        self._rename_edit.setVisible(True)
        self._rename_ok_btn.setVisible(True)
        self._rename_cancel_btn.setVisible(True)
        self._rename_edit.setFocus()
        self._rename_edit.selectAll()

    def _cancel_rename(self) -> None:
        self._rename_edit.setVisible(False)
        self._rename_ok_btn.setVisible(False)
        self._rename_cancel_btn.setVisible(False)
        self._rename_btn.setVisible(True)

    def _commit_rename(self) -> None:
        if self._active_run_dir is None:
            self._cancel_rename()
            return
        new_name = self._rename_edit.text().strip()
        if not new_name:
            self._cancel_rename()
            return
        manifest_path = self._active_run_dir / "run_manifest.json"
        try:
            data = json.loads(manifest_path.read_text())
            data["name"] = new_name
            tmp = manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, manifest_path)
            self._active_run_manifest = data
            self._status_label.setText(f"Renamed run to '{new_name}'.")
            self._refresh_past_runs_combo()
        except Exception as exc:
            self._status_label.setText(f"Rename failed: {exc}")
            logger.exception("Failed to rename run")
        finally:
            self._cancel_rename()

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
        self._past_runs_combo.blockSignals(True)
        self._past_runs_combo.setCurrentIndex(0)
        self._past_runs_combo.blockSignals(False)
        self._status_label.setText("Ready. Fill the form above and click Start.")
        self._set_app_mode("SETUP")
