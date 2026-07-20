"""An input video identified by content hash."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from deepreefmap.io.video_hash import describe_video
from deepreefmap.survey.models.common import utc_now_iso


@dataclass(slots=True)
class VideoAsset:
    """One clip on disk. Identity is ``hash``; ``path`` is advisory and may go stale."""

    file_name: str
    path: str
    hash: str | None = None
    size_bytes: int | None = None
    mtime: str | None = None
    duration_s: float | None = None
    fps: float | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_path(cls, path: Path) -> VideoAsset:
        """Describe a clip on disk. duration_s/fps need a decoder, so callers fill them."""
        info = describe_video(path)
        content_hash = info.get("hash")
        size_bytes = info.get("size_bytes")
        mtime = info.get("mtime")
        return cls(
            file_name=path.name,
            path=str(path),
            hash=content_hash if isinstance(content_hash, str) else None,
            size_bytes=size_bytes if isinstance(size_bytes, int) else None,
            mtime=mtime if isinstance(mtime, str) else None,
        )
