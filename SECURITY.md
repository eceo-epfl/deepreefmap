# Security Policy

DeepReefMap is not yet at a formal public-release security support level.

## Reporting

Please report potential vulnerabilities privately to the project maintainers
before opening a public issue. Include reproduction steps, affected versions or
commits, and whether the issue requires untrusted input such as checkpoints,
model repositories, manifests, videos, or camera profile names.

## ML Artifact Trust Boundary

Treat model checkpoints and Hugging Face model repositories as trusted code and
data unless they are pinned and verified. PyTorch checkpoints may execute code
during loading, and some model backends execute Python files from downloaded
model repositories.
