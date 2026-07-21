"""Run metadata formatting and the multi-line card delegate, shared by every
widget that lists runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from deepreefmap.gui.core.theme import BANNER_TEXT, CARD_BG
from deepreefmap.profiling.eta import format_duration
from deepreefmap.survey.catalogue import run_duration_s

RUN_META_ROLE = Qt.ItemDataRole.UserRole + 1

_GEOMETRY_LABELS = {
    "world_points": "world points (full)",
    "depth_unprojection": "depth-unprojection",
}


def format_timestamp(value: object) -> str:
    """Render an ISO-8601 run_timestamp as a short local date/time, else passthrough."""
    if not isinstance(value, str):
        return ""
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def related_run_counts(entries: list[tuple[Path, dict]]) -> dict[Path, int]:
    """Per run dir, how many sibling runs share a video_hash with it.

    Old manifests have no video_hashes and never count as related.
    """
    dirs_by_hash: dict[str, set[Path]] = {}
    for run_dir, manifest in entries:
        for h in manifest.get("video_hashes") or []:
            if h:
                dirs_by_hash.setdefault(h, set()).add(run_dir)
    counts: dict[Path, int] = {}
    for run_dir, manifest in entries:
        related: set[Path] = set()
        for h in manifest.get("video_hashes") or []:
            if h:
                related |= dirs_by_hash[h]
        related.discard(run_dir)
        counts[run_dir] = len(related)
    return counts


def format_bytes(total: float) -> str:
    if total >= 1e9:
        return f"{total / 1e9:.2f} GB"
    return f"{total / 1e6:.1f} MB"


def format_disk_size(run_dir: Path) -> str | None:
    try:
        total = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    except Exception:
        return None
    return format_bytes(total)


def video_details(manifest: dict, index: int = 0) -> list[str]:
    """Short hash, size, and recording date for one input video, where known."""
    details: list[str] = []
    hashes = manifest.get("video_hashes") or []
    sizes = manifest.get("video_sizes") or []
    mtimes = manifest.get("video_mtimes") or []
    if index < len(hashes) and hashes[index]:
        details.append(f"#{str(hashes[index])[:8]}")
    if index < len(sizes) and sizes[index]:
        details.append(format_bytes(float(sizes[index])))
    if index < len(mtimes) and mtimes[index]:
        stamp = format_timestamp(mtimes[index])
        if stamp:
            details.append(stamp)
    return details


def format_trim_range(manifest: dict) -> str | None:
    """The processed slice of the video, shown only when the run was trimmed."""
    begin = manifest.get("begin_s")
    end = manifest.get("end_s")
    if begin is None and end is None:
        return None
    begin_txt = f"{float(begin):.1f}" if begin is not None else "0"
    end_txt = f"{float(end):.1f}s" if end is not None else "end"
    return f"{begin_txt}–{end_txt}"


def _disk_label(run_dir: Path, disk_bytes: int | None) -> str | None:
    if disk_bytes is not None:
        return format_bytes(disk_bytes)
    return format_disk_size(run_dir)


def format_run_metadata(
    manifest: dict,
    run_dir: Path,
    *,
    include_disk_size: bool,
    disk_bytes: int | None = None,
) -> str:
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
    for i, v in enumerate(manifest.get("input_videos") or []):
        details = video_details(manifest, i)
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"Input: {Path(v).name}{suffix}")
    trim = format_trim_range(manifest)
    if trim:
        lines.append(f"Range: {trim}")
    dur = run_duration_s(manifest)
    if dur:
        lines.append(f"Runtime: {format_duration(dur)}")
    created = format_timestamp(manifest.get("run_timestamp"))
    if created:
        lines.append(f"Created: {created}")
    if include_disk_size:
        disk = _disk_label(run_dir, disk_bytes)
        if disk:
            lines.append(f"Disk: {disk}")
    return "<br>".join(lines)


def build_run_card_meta(manifest: dict, run_dir: Path, related_runs: int = 0) -> dict:
    """Build a flat dict the card delegate uses to paint each run."""
    name = (manifest.get("name") or "").strip() or run_dir.name
    facts: list[str] = []
    mode = manifest.get("mode")
    if mode:
        facts.append(mode)
    frames = manifest.get("frames_processed")
    if frames is not None:
        fps = manifest.get("fps")
        facts.append(f"{frames}f @ {fps}fps" if fps else f"{frames}f")
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
    trim = format_trim_range(manifest)
    if trim:
        facts.append(trim)
    dur = run_duration_s(manifest)
    if dur:
        facts.append(format_duration(dur))
    if related_runs:
        facts.append(f"{related_runs} related run{'s' if related_runs > 1 else ''}")
    videos = manifest.get("input_videos") or []
    video_line = ""
    if videos:
        names = [Path(v).name for v in videos]
        bits = [names[0], *video_details(manifest)]
        video_line = f"📹 {'  ·  '.join(bits)}"
        if len(names) > 1:
            video_line += f" (+ {len(names) - 1} more)"
    return {
        "title": name,
        "slug": "" if name == run_dir.name else f"({run_dir.name})",
        "facts": "  ·  ".join(facts),
        "video": video_line,
    }


def format_run_metadata_compact(
    manifest: dict,
    run_dir: Path,
    *,
    include_disk_size: bool,
    disk_bytes: int | None = None,
) -> str:
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
        ("FPS", "fps", str),
        ("Segmentation", "segmentation_model", str),
        ("Mapping", "mapping_backend", str),
        ("Geometry", "geometry_source", lambda v: _GEOMETRY_LABELS.get(v, str(v))),
        ("Camera", "camera_profile", str),
        ("Semantic pts", "semantic_reference_points", lambda v: f"{int(v):,}"),
        ("Metric pts", "metric_points", lambda v: f"{int(v):,}"),
        ("Input", "input_videos", lambda v: ", ".join(Path(p).name for p in v) if v else ""),
        ("Created", "run_timestamp", format_timestamp),
    ):
        v = manifest.get(key)
        if v is not None and v != "" and v != []:
            facts.append(
                f'<span style="color:#8aa0b8">{label}:</span>&nbsp;'
                f'<span style="color:{BANNER_TEXT}">{fmt(v)}</span>'
            )
    trim = format_trim_range(manifest)
    if trim:
        facts.append(
            f'<span style="color:#8aa0b8">Range:</span>&nbsp;'
            f'<span style="color:{BANNER_TEXT}">{trim}</span>'
        )
    dur = run_duration_s(manifest)
    if dur:
        facts.append(
            f'<span style="color:#8aa0b8">Runtime:</span>&nbsp;'
            f'<span style="color:{BANNER_TEXT}">{format_duration(dur)}</span>'
        )
    if include_disk_size:
        disk = _disk_label(run_dir, disk_bytes)
        if disk:
            facts.append(
                f'<span style="color:#8aa0b8">Disk:</span>&nbsp;'
                f'<span style="color:{BANNER_TEXT}">{disk}</span>'
            )
    sep = '&nbsp;<span style="color:#4a5f74">·</span>&nbsp;'
    return f"{header}&nbsp;&nbsp;{sep.join(facts)}"


class RunCardDelegate(QStyledItemDelegate):
    """Paints each run list item as a multi-line card with metadata."""

    # Sizes are em multiples off QFontMetrics, not pixels, so the card tracks
    # system DPI and font size on both Linux and Windows.

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
        meta = index.data(RUN_META_ROLE)
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
        meta = index.data(RUN_META_ROLE)
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
            painter.fillRect(option.rect, QColor(CARD_BG))

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
