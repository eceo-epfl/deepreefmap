"""Fast content identity for input videos.

Uses imohash (Syncthing's constant-time algorithm): file size plus sampled
chunks from the start, middle, and end, so a 4 GB clip hashes in well under a
millisecond. Dedup-grade, not cryptographic — good for grouping runs of the
same clip, not for integrity verification.
"""

from __future__ import annotations

import logging
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
