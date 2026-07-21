from __future__ import annotations

from deepreefmap.gui.core.window_protocol import MixinBase

import logging
import threading
from pathlib import Path

from PySide6.QtWidgets import QFileDialog

logger = logging.getLogger(__name__)


class _BatchJob:
    """One row of a batch reconstruction CSV; name defaults to the video filename stem."""

    __slots__ = ("video", "begin_s", "end_s", "transect_length", "crop_width", "name")

    def __init__(
        self,
        video: str,
        begin_s: float | None,
        end_s: float | None,
        transect_length: float | None,
        crop_width: float | None,
        name: str,
    ) -> None:
        self.video = video
        self.begin_s = begin_s
        self.end_s = end_s
        self.transect_length = transect_length
        self.crop_width = crop_width
        self.name = name


def _parse_optional_float(raw: str) -> float | None:
    s = (raw or "").strip()
    if not s:
        return None
    return float(s)


def _parse_timestamp_range(raw: str) -> tuple[float | None, float | None]:
    """Parse "<begin>-<end>" in seconds, splitting at the first dash.

    Either side may be empty, and a bare "30" is a begin with no end.
    """
    s = (raw or "").strip()
    if not s:
        return None, None
    if "-" not in s:
        return _parse_optional_float(s), None
    head, _, tail = s.partition("-")
    return _parse_optional_float(head), _parse_optional_float(tail)


def _load_batch_csv(path: Path) -> list[_BatchJob]:
    """Read a CSV with case-insensitive columns and return parsed rows."""
    import csv

    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        raise ValueError(
            "Excel files aren't supported (pandas isn't a dependency). "
            "Save the sheet as CSV and try again."
        )
    required = {"videos", "timestamps", "transect_length", "crop_width"}
    jobs: list[_BatchJob] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        norm = {fn.strip().lower(): fn for fn in reader.fieldnames}
        missing = required - set(norm.keys())
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        for n, row in enumerate(reader, start=2):
            video = (row.get(norm["videos"], "") or "").strip()
            if not video:
                continue  # skip blank rows
            try:
                begin_s, end_s = _parse_timestamp_range(row.get(norm["timestamps"], ""))
                transect_length = _parse_optional_float(row.get(norm["transect_length"], ""))
                crop_width = _parse_optional_float(row.get(norm["crop_width"], ""))
            except ValueError as exc:
                raise ValueError(f"Row {n}: {exc}") from exc
            jobs.append(
                _BatchJob(
                    video=video,
                    begin_s=begin_s,
                    end_s=end_s,
                    transect_length=transect_length,
                    crop_width=crop_width,
                    name=Path(video).stem or f"job_{n - 1}",
                )
            )
    if not jobs:
        raise ValueError("No usable rows in CSV.")
    return jobs


class BatchMixin(MixinBase):
    """DeepReefMapWindow methods that drive CSV-driven batch reconstruction."""

    def _on_batch_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select batch CSV",
            self._out_root_input.text(),
            "CSV files (*.csv);;All files (*)",
        )
        if not path_str:
            return
        try:
            jobs = _load_batch_csv(Path(path_str))
        except Exception as exc:
            self._status_label.setText(f"Batch CSV error: {exc}")
            logger.exception("Failed to load batch CSV")
            return

        # Outputs go to `batch_out/<job_name>/` under the user's chosen
        # output root so they don't collide with regular single runs.
        base_out = Path(self._out_root_input.text()).expanduser() / "batch_out"
        base_out.mkdir(parents=True, exist_ok=True)

        self._set_form_enabled(False)
        self._batch_btn.setEnabled(False)
        self._status_label.setText(f"Batch starting: {len(jobs)} job(s)")
        self._progress_bar.setRange(0, len(jobs))
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)

        # Snapshot the form once so a user editing it mid-batch doesn't
        # produce mixed configurations across jobs.
        common = {
            "fps": self._fps_spin.value(),
            "segmentation_name": self._seg_combo.currentText(),
            "mapping_name": self._map_combo.currentText(),
            "camera_profile_name": self._profile_combo.currentText(),
            "enable_tsdf": self._tsdf_check.isChecked(),
            "skip_segmentation": self._skip_seg_check.isChecked(),
            "classes_path": self._classes_path,
        }
        self._pipeline_thread = threading.Thread(
            target=self._run_batch_worker,
            args=(jobs, base_out, common),
            daemon=True,
        )
        self._pipeline_thread.start()

    def _run_batch_worker(
        self, jobs: list[_BatchJob], base_out: Path, common: dict
    ) -> None:
        from deepreefmap.pipeline.orchestrator import run_reconstruction

        ok = 0
        last_error = ""
        for idx, job in enumerate(jobs, start=1):
            self._sig_batch_progress.emit(idx, len(jobs), job.name)
            video_path = Path(job.video).expanduser()
            if not video_path.exists():
                last_error = f"row {idx}: {video_path} not found"
                logger.error("Batch %s/%s: %s", idx, len(jobs), last_error)
                continue
            out_dir = base_out / self._sanitize_run_name(job.name)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_reconstruction(
                    video_paths=[str(video_path)],
                    output_dir=out_dir,
                    transect_length=job.transect_length,
                    transect_crop_width=job.crop_width,
                    begin_s=job.begin_s,
                    end_s=job.end_s,
                    run_name=job.name,
                    viewer=None,
                    **common,
                )
                ok += 1
            except Exception as exc:
                logger.exception("Batch job %s failed", job.name)
                last_error = f"{job.name}: {exc}"
        self._sig_batch_done.emit(ok, len(jobs), last_error[:300])

    def _on_batch_progress(self, idx: int, total: int, name: str) -> None:
        self._status_label.setText(f"Batch: job {idx} of {total}: {name}")
        if self._progress_bar.maximum() != total:
            self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(idx - 1)
        self._progress_bar.setVisible(True)

    def _on_batch_done(self, ok: int, total: int, last_error: str) -> None:
        self._progress_bar.setValue(total)
        self._reset_progress_bars()
        if ok == total:
            self._status_label.setText(f"Batch complete: {ok}/{total} job(s) succeeded.")
        elif last_error:
            self._status_label.setText(
                f"Batch finished: {ok}/{total} succeeded. Last error: {last_error}"
            )
        else:
            self._status_label.setText(f"Batch finished: {ok}/{total} succeeded.")
        self._set_form_enabled(True)
        self._batch_btn.setEnabled(True)
        self._refresh_data_manager()
