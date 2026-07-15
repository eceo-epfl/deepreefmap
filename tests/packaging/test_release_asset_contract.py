"""The in-app updater asks the GitHub Releases API for a per-platform binary by a
name built from `resolve_asset_name`. If `release.yml` stops publishing an asset the
updater will request, the update silently 404s. These tests pin the two files
together by driving the real `find_asset_url` against the names the release actually
publishes, so a rename on either side fails here instead of in the field.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

import pytest
import yaml

from deepreefmap.packaging import binary_swap

_FAKE_VERSION = "9.9.9"
_RELEASE_YML = Path(__file__).parents[2] / ".github" / "workflows" / "release.yml"

# (platform arg to resolve_asset_name, is_rocm build, cuda variant suffix).
# Mirrors the release.yml build matrix and tests/test_qt_ui.py's cases.
_RUN_VARIANTS = [
    ("linux", False, ""),
    ("linux", False, "-cu130"),
    ("linux", True, ""),
    ("win32", False, ""),
    ("win32", False, "-cu130"),
    ("darwin", False, ""),
]


def _published_asset_names() -> set[str]:
    """Basenames of every asset a release attaches.

    Derived from the build matrix (binary per entry, setup exe on Windows, dmg
    on macOS, mirroring the upload steps), then checked against the release
    step's files glob so nothing the matrix builds is silently dropped.
    """
    data = yaml.safe_load(_RELEASE_YML.read_text())
    names: set[str] = set()
    for entry in data["jobs"]["build"]["strategy"]["matrix"]["include"]:
        artifact = entry["artifact"]
        ext = entry.get("ext") or ""
        names.add(f"{artifact}-{_FAKE_VERSION}{ext}")
        if str(entry["os"]).startswith("windows"):
            names.add(artifact.replace("deepreefmap-", "deepreefmap-setup-", 1) + f"-{_FAKE_VERSION}.exe")
        if str(entry["os"]).startswith("macos"):
            names.add(f"{artifact}-{_FAKE_VERSION}.dmg")
    if not names:
        raise AssertionError("no build matrix entries found in release.yml")
    for step in data["jobs"]["release"]["steps"]:
        if not str(step.get("uses", "")).startswith("softprops/action-gh-release"):
            continue
        patterns = [
            os.path.basename(line.strip().replace("${{ steps.label.outputs.value }}", _FAKE_VERSION))
            for line in str(step["with"]["files"]).splitlines()
            if line.strip()
        ]
        unmatched = sorted(n for n in names if not any(fnmatch.fnmatch(n, p) for p in patterns))
        assert not unmatched, f"release.yml files glob does not attach: {unmatched}"
    return names


def _synthetic_release(names: set[str]) -> dict:
    return {
        "tag_name": f"v{_FAKE_VERSION}",
        "assets": [{"name": n, "browser_download_url": f"https://example.invalid/{n}"} for n in names],
    }


@pytest.mark.parametrize("platform, is_rocm, suffix", _RUN_VARIANTS)
def test_updater_requested_asset_is_published(platform, is_rocm, suffix, monkeypatch) -> None:
    monkeypatch.setattr(binary_swap, "_is_rocm_build", lambda: is_rocm)
    monkeypatch.setattr(binary_swap, "_cuda_variant_suffix", lambda: suffix)

    release = _synthetic_release(_published_asset_names())
    asset_name = binary_swap.resolve_asset_name(platform)
    # Raises BinarySwapError if release.yml publishes nothing matching this request.
    binary_swap.find_asset_url(release, asset_name)


def test_first_install_assets_are_published() -> None:
    published = _published_asset_names()
    for expected in (
        f"deepreefmap-setup-windows-x64-{_FAKE_VERSION}.exe",
        f"deepreefmap-setup-windows-x64-cu130-{_FAKE_VERSION}.exe",
        f"deepreefmap-macos-arm64-{_FAKE_VERSION}.dmg",
    ):
        assert expected in published, f"release.yml no longer attaches {expected}"
