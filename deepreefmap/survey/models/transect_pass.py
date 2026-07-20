"""One traversal of a transect: a time-trimmed segment of a single video."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from deepreefmap.survey.models.common import utc_now_iso

PASS_DIRECTIONS = ("forward", "reverse")


@dataclass(slots=True)
class TransectPass:
    """A pass never spans videos; one video may contain several passes."""

    transect_id: uuid.UUID
    video_id: uuid.UUID
    begin_s: float
    end_s: float
    direction: str = "forward"
    batch_id: uuid.UUID | None = None
    notes: str = ""
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if self.direction not in PASS_DIRECTIONS:
            raise ValueError(f"direction must be one of {PASS_DIRECTIONS}, got {self.direction!r}")
        if self.begin_s < 0:
            raise ValueError(f"begin_s must be >= 0, got {self.begin_s}")
        if self.end_s <= self.begin_s:
            raise ValueError(f"end_s must be greater than begin_s ({self.begin_s})")

    def duration_s(self) -> float:
        return self.end_s - self.begin_s
