"""Recognise EPFL-ECEO model repos by naming convention.

Discovery (the Models-tab "Check Hugging Face for new models" button) lists the
org's repos and turns each known-loadable one into a ModelInfo plus the args
needed to register a loader. A repo is "known-loadable" only when it matches one
of the families below; anything else is skipped so the UI never offers a model
the app can't actually run.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from deepreefmap.gui.model_manager import ModelInfo


@dataclass(frozen=True)
class _Family:
    kind: str
    family: str  # loader tag understood by segmentation.registry
    pattern: re.Pattern[str]
    gated: bool
    short_name: Callable[[re.Match[str]], str]
    resolution: Callable[[re.Match[str]], tuple[int, int]]
    backbone: Callable[[re.Match[str]], str | None]
    describe: Callable[[str], str]


_DPT = _Family(
    kind="segmentation",
    family="dpt",
    # Only s/b/l variants are loadable today; unknown variants don't match and
    # are skipped, which is the intended "known-loadable only" behaviour.
    pattern=re.compile(r"^coralscapes-vit-(?P<v>[sbl])-dpt$"),
    gated=True,
    short_name=lambda m: m.group(0),
    resolution=lambda m: (384, 688) if m.group("v") == "s" else (768, 1376),
    # The DPT head's loader pulls a Meta DINOv3 backbone at first use, so it must
    # be cached alongside the head for offline laptops to stay self-sufficient.
    backbone=lambda m: f"facebook/dinov3-vit{m.group('v')}16-pretrain-lvd1689m",
    describe=lambda name: f"DINOv3 DPT ({name}, requires HF login)",
)

_SEGFORMER = _Family(
    kind="segmentation",
    family="segformer",
    pattern=re.compile(r"^segformer-b(?P<n>\d+)-finetuned-coralscapes(?:-\d+-\d+)?$"),
    gated=False,
    short_name=lambda m: f"segformer-b{m.group('n')}",
    resolution=lambda m: (1024, 1024),
    backbone=lambda m: None,
    describe=lambda name: f"SegFormer ({name}, no auth required)",
)

_FAMILIES = (_DPT, _SEGFORMER)


def synthesize_model_info(
    repo_id: str,
) -> tuple[ModelInfo, tuple[int, int], str] | None:
    """Return (info, resolution, family) for a known-loadable repo, else None.

    `info.approx_size_mb` is left None to avoid a per-repo metadata round-trip;
    the Models tab already renders a missing size chip gracefully.
    """
    base = repo_id.split("/", 1)[-1]
    for fam in _FAMILIES:
        m = fam.pattern.match(base)
        if m is None:
            continue
        hf_repos = [repo_id]
        backbone = fam.backbone(m)
        if backbone:
            hf_repos.append(backbone)
        info = ModelInfo(
            name=fam.short_name(m),
            kind=fam.kind,
            hf_repos=hf_repos,
            gated=fam.gated,
            description=fam.describe(base),
            approx_size_mb=None,
        )
        return info, fam.resolution(m), fam.family
    return None
