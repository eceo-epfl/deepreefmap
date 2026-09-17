# Third-Party Notices

This file tracks third-party components that need review before an open-source
release.

## LoGeR

DeepReefMap can use LoGeR through the `third_party/LoGeR` git submodule. The
current submodule checkout does not include a clear top-level license file, and
some files inside it carry separate upstream notices, including non-commercial
terms.

Before publishing a public release that includes or depends on this submodule,
confirm the LoGeR license, model checkpoint terms, and compatibility with the
license chosen for DeepReefMap. If compatibility is unclear, keep LoGeR outside
release archives and document it as an optional user-managed integration.

## VGGT-Omega

DeepReefMap can use VGGT-Omega through the optional `vggt_omega` extra, which
installs the `vggt-omega` package from `https://github.com/facebookresearch/vggt-omega`.
It is not vendored into the wheel. VGGT-Omega is released under the FAIR
Noncommercial Research License, which restricts use to noncommercial research,
and its model checkpoints are gated on Hugging Face (`facebook/VGGT-Omega`) and
carry their own terms.

Before publishing a public release that depends on this package, confirm the
FAIR license and checkpoint terms are compatible with the intended use. Because
the license is noncommercial, keep VGGT-Omega an optional, user-managed
integration and do not bundle it or its checkpoints in release archives.

## LingBot-Map

DeepReefMap can use LingBot-Map through the optional `lingbot_map` extra, which
installs the `lingbot-map` package from `https://github.com/robbyant/lingbot-map`.
It is not vendored into the wheel. LingBot-Map is released under the Apache
License 2.0. Its model checkpoints are published separately on Hugging Face
(`robbyant/lingbot-map`) and are downloaded at runtime; they carry their own
terms.

Before publishing a public release that depends on this package, confirm the
LingBot-Map license and checkpoint terms are compatible with the intended use.
Keep LingBot-Map an optional, user-managed integration and do not bundle its
checkpoints in release archives.

## Model Checkpoints

Segmentation and mapping models are downloaded or loaded separately from the
source tree. Release notes should name each model source, pinned revision or
checksum, and license/usage terms.

