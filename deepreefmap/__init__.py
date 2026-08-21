"""DeepReefMap package."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("deepreefmap")
except PackageNotFoundError:  # running from a source checkout, nothing installed
    __version__ = "0.0.0"
