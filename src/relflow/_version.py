"""Installed RelFlow package version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

UNKNOWN_VERSION = "0+unknown"

try:
    __version__ = version("relflow")
except PackageNotFoundError:
    # Keep source-tree imports usable before the project is installed.
    __version__ = UNKNOWN_VERSION


__all__ = ["__version__"]
