"""GitHub release discovery and version comparison for the in-app updater."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_GH_REPO = "eceo-epfl/deepreefmap"


def gh_releases_url() -> str:
    # Full-URL override points the release check at a local server, so the real
    # download + swap + provision + prune path can be validated without a public
    # release (see tests/e2e/update_e2e.sh --interactive). DEEPREEFMAP_GH_REPO
    # swaps only the owner/repo against the real GitHub host.
    override = os.environ.get("DEEPREEFMAP_GH_API_URL")
    if override:
        return override
    repo = os.environ.get("DEEPREEFMAP_GH_REPO", _DEFAULT_GH_REPO)
    return f"https://api.github.com/repos/{repo}/releases"


def pyapp_binary_path() -> str | None:
    """Running PyApp binary path for update controls, honouring the test mock."""
    if os.environ.get("DEEPREEFMAP_MOCK_PYAPP"):
        return "/tmp/mock-pyapp"
    from deepreefmap.packaging.binary_swap import pyapp_binary

    return pyapp_binary()


def fetch_release_versions(timeout: float = 8.0) -> list[str] | None:
    import urllib.request

    mock = os.environ.get("DEEPREEFMAP_MOCK_VERSIONS")
    if mock is not None:
        return [v.strip() for v in mock.split(",") if v.strip()]
    try:
        req = urllib.request.Request(gh_releases_url(), headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            releases = json.load(resp)
        versions = []
        for rel in releases:
            tag = rel.get("tag_name", "")
            if tag.startswith("v"):
                tag = tag[1:]
            if tag and not rel.get("draft"):
                versions.append(tag)
        return versions if versions else None
    except Exception as exc:
        logger.warning("Failed to fetch releases from GitHub: %s", exc)
        return None


def fetch_releases(timeout: float = 8.0) -> list[dict] | None:
    """Raw release records (with `assets`) for binary swap.

    Drafts and releases with no asset for this platform are dropped, so pre-binary
    releases and failed uploads are never offered.
    """
    import urllib.request

    from deepreefmap.packaging.binary_swap import BinarySwapError, match_asset_url, resolve_asset_name

    mock = os.environ.get("DEEPREEFMAP_MOCK_VERSIONS")
    if mock is not None:
        records = []
        for v in (s.strip() for s in mock.split(",")):
            if not v:
                continue
            records.append({
                "tag_name": f"v{v}",
                "draft": False,
                "assets": [
                    {
                        "name": "deepreefmap-linux-x64",
                        "browser_download_url": f"https://example.invalid/v{v}/deepreefmap-linux-x64",
                    },
                    {
                        "name": "deepreefmap-windows-x64.exe",
                        "browser_download_url": f"https://example.invalid/v{v}/deepreefmap-windows-x64.exe",
                    },
                ],
            })
        return records if records else None
    try:
        req = urllib.request.Request(gh_releases_url(), headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            releases = json.load(resp)
        kept = [rel for rel in releases if rel.get("tag_name") and not rel.get("draft")]
        try:
            asset_name = resolve_asset_name()
        except BinarySwapError:
            asset_name = None  # unsupported platform: dev mode, install controls stay hidden
        if asset_name is not None:
            kept = [rel for rel in kept if match_asset_url(rel, asset_name) is not None]
        # Empty list means "reached GitHub, nothing installable" (renders as
        # "No releases found."); None is reserved for fetch failures.
        return kept
    except Exception as exc:
        logger.warning("Failed to fetch release metadata from GitHub: %s", exc)
        return None


def release_version(record: dict) -> str:
    tag = str(record.get("tag_name", ""))
    return tag[1:] if tag.startswith("v") else tag


def parse_version(value: str):
    from packaging.version import InvalidVersion, Version

    try:
        return Version(value)
    except InvalidVersion:
        return None


def newer_releases(releases: list[dict], current: str) -> list[dict]:
    """Releases strictly newer than `current`, newest first.

    Falls back to string inequality when `current` itself can't be parsed.
    """
    current_v = parse_version(current)
    if current_v is None:
        return [r for r in releases if release_version(r) != current]
    newer = []
    for rel in releases:
        rv = parse_version(release_version(rel))
        if rv is not None and rv > current_v:
            newer.append((rv, rel))
    newer.sort(key=lambda pair: pair[0], reverse=True)
    return [rel for _, rel in newer]


def selectable_releases(releases: list[dict], current: str, include_older: bool) -> list[dict]:
    """Releases the user may install, newest first.

    ``include_older`` offers every version but the current one: mechanically a
    rollback is the same as an upgrade.
    """
    if not include_older:
        return newer_releases(releases, current)
    keyed = []
    for rel in releases:
        version = release_version(rel)
        if version == current:
            continue
        keyed.append((parse_version(version) or parse_version("0"), rel))
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [rel for _, rel in keyed]


def current_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("deepreefmap")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"
