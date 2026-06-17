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

## Model Checkpoints

Segmentation and mapping models are downloaded or loaded separately from the
source tree. Release notes should name each model source, pinned revision or
checksum, and license/usage terms.

## Bundled Fonts

`deepreefmap/resources/fonts/` ships two families under the SIL Open Font
License 1.1, both unmodified and neither declaring a Reserved Font Name:

- Inter 4.001 (Regular, Medium, SemiBold, Bold). Copyright (c) 2016 The Inter
  Project Authors (https://github.com/rsms/inter).
- JetBrains Mono 2.304 (Regular, Bold). Copyright 2020 The JetBrains Mono
  Project Authors (https://github.com/JetBrains/JetBrainsMono).

The full license texts ship with the fonts as `Inter-LICENSE.txt` and
`JetBrainsMono-OFL.txt`. The OFL covers the fonts as a separate work and is
compatible with the Apache-2.0 license of DeepReefMap.
