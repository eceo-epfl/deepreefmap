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
