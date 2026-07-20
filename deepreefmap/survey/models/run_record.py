"""One execution of a pass through the reconstruction pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from deepreefmap.survey.models.common import utc_now_iso

RUN_STATUSES = ("pending", "running", "succeeded", "failed", "cancelled")
TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")


@dataclass(slots=True)
class RunRecord:
    """Re-running a pass creates a new record; repeats are the reproducibility data.

    ``run_dir_name`` is relative to the output root so a moved folder keeps working.
    Cover numbers stay in the run directory's benthic_cover.json, never in the database.
    """

    pass_id: uuid.UUID
    run_dir_name: str
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    error: str = ""
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES:
            raise ValueError(f"status must be one of {RUN_STATUSES}, got {self.status!r}")
        if not self.run_dir_name.strip():
            raise ValueError("run_dir_name must not be empty")
