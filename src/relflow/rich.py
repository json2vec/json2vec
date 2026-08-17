"""Internal Rich rendering and bounded human-facing diagnostics.

This module is intentionally dependency-light so every RelFlow subsystem can
use the same console without creating import cycles.  It is not a replacement
for application logging or Lightning metrics, and it is not a supported
compatibility surface.
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from io import StringIO
from threading import Lock
from typing import Any, TypeAlias

from rich.console import Console
from rich.theme import Theme
from rich.traceback import install as rich_traceback_install

ANSI_SEQUENCE = re.compile(
    r"""
    \x1b
    (?:
        \][^\x07\x1b]*(?:\x07|\x1b\\)
        | [PX^_][^\x1b]*(?:\x1b\\)
        | \[[0-?]*[\x20-\x2f]*[@-~]
        | [@-_]
    )
    """,
    flags=re.DOTALL | re.VERBOSE,
)
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
HTML_TAG = re.compile(r"<[^>]*>")
STYLE_ATTRIBUTE = re.compile(r'\sstyle="(?P<style>[^"]*)"', flags=re.IGNORECASE)
COLOR_PROPERTIES = frozenset(
    {
        "background-color",
        "color",
        "text-decoration-color",
    }
)
DIM_COLORS = frozenset({"#7f7f7f", "#808080"})

theme = Theme(
    {
        "relflow.info": "cyan",
        "relflow.warning": "yellow",
        "relflow.error": "bold red",
        "relflow.name": "bold",
        "relflow.type": "yellow",
        "relflow.dim": "dim",
    }
)

console = Console(stderr=True, theme=theme)

verbose = False
tracebacks_installed = False
traceback_lock = Lock()


def set_verbose(enabled: bool) -> None:
    """Set internal diagnostic verbosity.

    Verbose mode permits occasional setup and phase diagnostics.  It never
    enables per-record, batch, step, or epoch output.
    """

    global verbose
    verbose = bool(enabled)


def is_verbose() -> bool:
    """Return whether explicitly requested diagnostic detail is enabled."""

    return verbose


def install_tracebacks() -> None:
    """Install Rich rendering for uncaught exceptions once per process.

    Installation is explicit because Rich also integrates with an active
    IPython shell.  Traceback locals stay hidden to avoid exposing records,
    predictions, credentials, or processor-bound values.
    """

    global tracebacks_installed
    if tracebacks_installed:
        return

    with traceback_lock:
        if tracebacks_installed:
            return
        rich_traceback_install(
            console=console,
            show_locals=False,
            locals_max_length=10,
            locals_max_string=80,
            max_frames=50,
        )
        tracebacks_installed = True


def render_text(renderable: object, *, width: int = 120) -> str:
    """Render an object to deterministic ANSI-free terminal text."""

    output = StringIO()
    capture = Console(
        file=output,
        record=True,
        width=width,
        force_terminal=False,
        force_jupyter=False,
        theme=theme,
    )
    capture.print(renderable)
    plain = capture.export_text(clear=False)
    return strip_control_sequences(plain).rstrip("\n")


def render_html(renderable: object, *, width: int = 120) -> str:
    """Render an object to a transparent inline-styled HTML fragment."""

    capture = Console(
        file=StringIO(),
        record=True,
        width=width,
        force_terminal=False,
        force_jupyter=False,
        theme=theme,
    )
    capture.print(renderable)
    html = capture.export_html(
        inline_styles=True,
        clear=False,
        code_format=(
            '<pre style="font-family: Menlo, Consolas, monospace; '
            "white-space: pre-wrap; margin: 0; padding: 0; border: 0; "
            'background: transparent;"><code>{code}</code></pre>'
        ),
    )
    return sanitize_style_attributes(strip_control_sequences(html))


def strip_control_sequences(value: str) -> str:
    """Remove terminal escape sequences and unsafe control characters."""

    without_ansi = ANSI_SEQUENCE.sub("", value)
    return CONTROL_CHARACTERS.sub("", without_ansi)


def sanitize_style_attributes(value: str) -> str:
    """Remove terminal palette colors from generated HTML style attributes."""

    def sanitize_tag(tag_match: re.Match[str]) -> str:
        def sanitize_style(style_match: re.Match[str]) -> str:
            declarations: list[str] = []
            dim = False
            has_opacity = False

            for raw_declaration in style_match.group("style").split(";"):
                declaration = raw_declaration.strip()
                if not declaration:
                    continue

                property_name, separator, property_value = declaration.partition(":")
                normalized_name = property_name.strip().lower()
                normalized_value = property_value.strip().lower()
                if separator and normalized_name in COLOR_PROPERTIES:
                    if normalized_name != "background-color" and normalized_value in DIM_COLORS:
                        dim = True
                    continue

                if normalized_name == "opacity":
                    has_opacity = True
                declarations.append(declaration)

            if dim and not has_opacity:
                declarations.append("opacity: 0.7")
            if not declarations:
                return ""
            return f' style="{"; ".join(declarations)};"'

        return STYLE_ATTRIBUTE.sub(sanitize_style, tag_match.group(0))

    return HTML_TAG.sub(sanitize_tag, value)


class MimeBundleDisplay:
    """Marimo display proxy that preserves HTML and plain-text MIME values."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def _mime_(self) -> tuple[str, dict[str, str]]:
        return "application/vnd.marimo+mimebundle", self.value._repr_mimebundle_()


IncidentPart: TypeAlias = str | int | float | bool | None
IncidentKey: TypeAlias = tuple[IncidentPart, ...]


@dataclass(frozen=True, slots=True)
class Incident:
    """Result of recording one occurrence of a bounded incident."""

    key: IncidentKey
    count: int
    suppressed: int
    emit: bool
    overflow: bool = False


class IncidentTracker:
    """Bound repeated diagnostics without retaining incident payloads.

    Only short scalar key parts and integer counters are retained. Each
    incident kind has its own unique-key budget, so a noisy source cannot
    suppress unrelated diagnostics. The number of kinds is bounded as well.
    """

    MAX_PART_LENGTH = 160
    MAX_COUNT = sys.maxsize
    MAX_EMISSIONS_PER_KEY = 3
    OVERFLOW_PART = "<additional-incidents>"
    KINDS_OVERFLOW_KEY: IncidentKey = ("<additional-kinds>",)

    def __init__(self, *, max_keys: int = 128, max_kinds: int = 16) -> None:
        if max_keys <= 0:
            raise ValueError("max_keys must be positive")
        if max_kinds <= 0:
            raise ValueError("max_kinds must be positive")
        self.max_keys = max_keys
        self.max_kinds = max_kinds
        self.counts: OrderedDict[str, OrderedDict[IncidentKey, int]] = OrderedDict()
        self.overflow_counts: dict[str, int] = {}
        self.kinds_overflow_count = 0
        self.lock = Lock()

    @classmethod
    def normalize_key(cls, key: IncidentKey) -> IncidentKey:
        if not key:
            raise ValueError("incident key must not be empty")

        normalized: list[IncidentPart] = []
        for part in key:
            if part is None or isinstance(part, (bool, int, float)):
                normalized.append(part)
                continue
            if isinstance(part, str):
                normalized.append(part[: cls.MAX_PART_LENGTH])
                continue
            raise TypeError("incident key parts must be short scalar values")
        return tuple(normalized)

    def record(self, kind: str, *key: IncidentPart, limit: int = 1) -> Incident:
        """Record an occurrence and say whether its diagnostic should emit."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        if not isinstance(kind, str):
            raise TypeError("incident kind must be a string")
        limit = min(limit, self.MAX_EMISSIONS_PER_KEY)
        normalized = self.normalize_key((kind, *key))
        normalized_kind = normalized[0]
        if not isinstance(normalized_kind, str):
            raise TypeError("incident kind must be a string")
        partition_key = normalized[1:]

        with self.lock:
            partition = self.counts.get(normalized_kind)
            if partition is None and len(self.counts) >= self.max_kinds:
                self.kinds_overflow_count = min(self.kinds_overflow_count + 1, self.MAX_COUNT)
                count = self.kinds_overflow_count
                stored_key = self.KINDS_OVERFLOW_KEY
                overflow = True
            else:
                if partition is None:
                    partition = OrderedDict()
                    self.counts[normalized_kind] = partition

                overflow = partition_key not in partition and len(partition) >= self.max_keys
                if overflow:
                    count = min(self.overflow_counts.get(normalized_kind, 0) + 1, self.MAX_COUNT)
                    self.overflow_counts[normalized_kind] = count
                    stored_key = (normalized_kind, self.OVERFLOW_PART)
                else:
                    count = min(partition.get(partition_key, 0) + 1, self.MAX_COUNT)
                    partition[partition_key] = count
                    stored_key = normalized

        return Incident(
            key=stored_key,
            count=count,
            suppressed=count if overflow else max(count - limit, 0),
            emit=False if overflow else count <= limit,
            overflow=overflow,
        )

    def snapshot(self) -> dict[IncidentKey, int]:
        """Return small occurrence counters for diagnostics and tests."""

        with self.lock:
            snapshot = {
                (kind, *key): count for kind, partition in self.counts.items() for key, count in partition.items()
            }
            for kind, count in self.overflow_counts.items():
                snapshot[(kind, self.OVERFLOW_PART)] = count
            if self.kinds_overflow_count:
                snapshot[self.KINDS_OVERFLOW_KEY] = self.kinds_overflow_count
            return snapshot

    def reset(self, kind: str | None = None) -> None:
        """Clear all process-local counters, or only one incident kind."""

        with self.lock:
            if kind is None:
                self.counts.clear()
                self.overflow_counts.clear()
                self.kinds_overflow_count = 0
                return

            normalized = self.normalize_key((kind,))[0]
            if not isinstance(normalized, str):
                raise TypeError("incident kind must be a string")
            self.counts.pop(normalized, None)
            self.overflow_counts.pop(normalized, None)


incidents = IncidentTracker()


def record_incident(kind: str, *key: IncidentPart, limit: int = 1) -> Incident:
    """Record an incident in RelFlow's shared process-local tracker."""

    return incidents.record(kind, *key, limit=limit)
