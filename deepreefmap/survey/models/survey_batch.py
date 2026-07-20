"""A group of passes queued and executed together (typically one field day)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from deepreefmap.survey.models.common import utc_now_iso


@dataclass(slots=True)
class SurveyBatch:
    name: str
    preset_name: str = "survey_preset"
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Batch name must not be empty")
