"""Fast content identity for input videos.

Uses imohash (Syncthing's constant-time algorithm): file size plus sampled chunks,
so a 4 GB clip hashes in well under a millisecond. Dedup-grade, not cryptographic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from imohash import hashfile

logger = logging.getLogger(__name__)


def hash_video(path: Path) -> str | None:
    """Return the 32-char hex imohash of ``path``, or None if hashing fails.

    A hash failure only loses run grouping, so it must never break a run.
    """
    try:
        return str(hashfile(str(path), hexdigest=True))
    except Exception:
        logger.warning("Could not hash %s", path, exc_info=True)
        return None


def hash_videos(paths: list[Path]) -> list[str | None]:
    """Hashes parallel to ``paths`` (None entries for unhashable files)."""
    return [hash_video(p) for p in paths]


def describe_video(path: Path) -> dict[str, object]:
    """Identity + display metadata for one clip: hash, size_bytes, mtime.

    mtime is UTC ISO-8601. A GoPro's mtime is the recording end time, which is what
    a diver recognises a clip by.
    """
    info: dict[str, object] = {"hash": hash_video(path), "size_bytes": None, "mtime": None}
    try:
        st = path.stat()
        info["size_bytes"] = st.st_size
        info["mtime"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        pass
    return info


def describe_videos(paths: list[Path]) -> list[dict[str, object]]:
    """Per-clip descriptions parallel to ``paths``."""
    return [describe_video(p) for p in paths]
