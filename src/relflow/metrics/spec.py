"""Pluggy markers and hook specification for metric providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from relflow.metrics.base import MetricPlugin


hookspec = pluggy.HookspecMarker("metrics")
hookimpl = pluggy.HookimplMarker("metrics")


class PluginSpec:
    """Hooks implemented by registered metric providers."""

    @hookspec
    def metric(self) -> "MetricPlugin":
        """Return one registered metric definition."""
        raise NotImplementedError
