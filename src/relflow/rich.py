"""Internal Rich rendering and bounded human-facing diagnostics.

This module is intentionally dependency-light so every RelFlow subsystem can
use the same console without creating import cycles.  It is not a replacement
for application logging or Lightning metrics, and it is not a supported
compatibility surface.
"""

from __future__ import annotations

import os
import re
import sys
import weakref
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import StringIO
from threading import Lock
from types import TracebackType
from typing import Any, TypeAlias

import pydantic
from rich.console import Console
from rich.constrain import Constrain
from rich.segment import Segment
from rich.theme import Theme
from rich.traceback import Traceback
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

console = Console(
    stderr=True,
    theme=theme,
    log_path=False,
    log_time_format="[%Y-%m-%d %H:%M:%S]",
)

verbose = os.environ.get("RELFLOW_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
tracebacks_installed = False
traceback_lock = Lock()
previous_excepthook: Any = None

TRACEBACK_WIDTH = 88
TRACEBACK_CODE_WIDTH = 88
TRACEBACK_MESSAGE_LENGTH = 1600
TRACEBACK_MESSAGE_LINES = 20
TRACEBACK_NOTE_LIMIT = 8
TRACEBACK_STACK_LIMIT = 8
TRACEBACK_GROUP_LIMIT = 8
TRACEBACK_OPTIONS: dict[str, Any] = {
    "width": TRACEBACK_WIDTH,
    "code_width": TRACEBACK_CODE_WIDTH,
    "extra_lines": 1,
    "show_locals": False,
    "locals_max_length": 10,
    "locals_max_string": 80,
    "max_frames": 50,
    "suppress": (pydantic,),
}


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

    RelFlow calls this during package import. Traceback locals stay hidden to
    avoid exposing records, predictions, credentials, or processor-bound
    values.
    """

    global previous_excepthook, tracebacks_installed
    if tracebacks_installed:
        return

    with traceback_lock:
        if tracebacks_installed:
            return
        if sys.excepthook is not sys.__excepthook__:
            # The hosting application already owns process-level exception
            # reporting. Replacing or chaining it risks duplicate tracebacks.
            previous_excepthook = sys.excepthook
            tracebacks_installed = True
            return
        previous_excepthook = rich_traceback_install(console=console, **TRACEBACK_OPTIONS)
        rich_hook = sys.excepthook

        # Rich installs through IPython's display hooks when a shell is active.
        # Otherwise replace its sys hook with the whole-traceback constrained
        # renderer below. A pre-existing application hook returned above.
        if rich_hook is not previous_excepthook:

            def excepthook(
                exception_type: type[BaseException],
                exception: BaseException,
                traceback: TracebackType | None,
            ) -> None:
                print_exception(exception, traceback=traceback)

            sys.excepthook = excepthook
        tracebacks_installed = True


def print_exception(exception: BaseException, *, traceback: TracebackType | None = None) -> None:
    """Render one bounded, locals-hidden exception to the shared stderr console."""

    resolved_traceback = exception.__traceback__ if traceback is None else traceback
    rendered = Traceback.from_exception(
        type(exception),
        exception,
        resolved_traceback,
        **TRACEBACK_OPTIONS,
    )
    bound_traceback_messages(rendered)
    console.print(Constrain(SanitizedTraceback(rendered), width=TRACEBACK_WIDTH))


class SanitizedTraceback:
    """Strip untrusted terminal controls from Rich's final traceback segments."""

    def __init__(self, rendered: Traceback) -> None:
        self.rendered = rendered

    def __rich_console__(self, output: Console, options: Any):
        for segment in output.render(self.rendered, options):
            yield Segment(strip_control_sequences(segment.text), segment.style, segment.control)


def bound_traceback_messages(rendered: Traceback) -> None:
    """Bound exception messages and notes without changing their traceback frames."""

    def bound(value: str, *, length: int = TRACEBACK_MESSAGE_LENGTH) -> str:
        value = strip_control_sequences(value)
        if len(value) > length:
            value = f"{value[: length - 1]}…"
        lines = value.splitlines()
        if len(lines) > TRACEBACK_MESSAGE_LINES:
            omitted = len(lines) - TRACEBACK_MESSAGE_LINES + 1
            lines = [*lines[: TRACEBACK_MESSAGE_LINES - 1], f"… {omitted} more lines"]
        return "\n".join(lines)

    def visit(trace: Any) -> None:
        if len(trace.stacks) > TRACEBACK_STACK_LIMIT:
            head = TRACEBACK_STACK_LIMIT // 2
            tail = TRACEBACK_STACK_LIMIT - head
            omitted = len(trace.stacks) - TRACEBACK_STACK_LIMIT
            trace.stacks = [*trace.stacks[:head], *trace.stacks[-tail:]]
            trace.stacks[0].notes.insert(0, f"… {omitted} chained exceptions omitted")

        for stack in trace.stacks:
            if len(stack.exceptions) > TRACEBACK_GROUP_LIMIT:
                head = TRACEBACK_GROUP_LIMIT // 2
                tail = TRACEBACK_GROUP_LIMIT - head
                omitted = len(stack.exceptions) - TRACEBACK_GROUP_LIMIT
                stack.exceptions = [*stack.exceptions[:head], *stack.exceptions[-tail:]]
                stack.exc_value = f"{stack.exc_value}; {omitted} sub-exceptions omitted"

            stack.exc_type = bound(stack.exc_type, length=160)
            stack.exc_value = bound(stack.exc_value)
            notes = [bound(note, length=400) for note in stack.notes[:TRACEBACK_NOTE_LIMIT]]
            if len(stack.notes) > TRACEBACK_NOTE_LIMIT:
                notes.append(f"… {len(stack.notes) - TRACEBACK_NOTE_LIMIT} more notes")
            stack.notes = notes
            for frame in stack.frames:
                frame.filename = bound(frame.filename, length=240)
                frame.name = bound(frame.name, length=160)
                frame.line = bound(frame.line, length=1000)
            if stack.syntax_error is not None:
                stack.syntax_error.filename = bound(stack.syntax_error.filename, length=240)
                stack.syntax_error.line = bound(stack.syntax_error.line, length=1000)
                stack.syntax_error.msg = bound(stack.syntax_error.msg, length=400)
                syntax_notes = [bound(note, length=400) for note in stack.syntax_error.notes[:TRACEBACK_NOTE_LIMIT]]
                if len(stack.syntax_error.notes) > TRACEBACK_NOTE_LIMIT:
                    syntax_notes.append(f"… {len(stack.syntax_error.notes) - TRACEBACK_NOTE_LIMIT} more notes")
                stack.syntax_error.notes = syntax_notes
            for nested in stack.exceptions:
                visit(nested)

    visit(rendered.trace)


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


def middle_elide(value: str, limit: int) -> str:
    """Bound text while preserving both its identifying beginning and end."""

    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit == 1:
        return "…"

    head = (limit - 1 + 1) // 2
    tail = limit - 1 - head
    return f"{value[:head]}…{value[-tail:]}" if tail else f"{value[:head]}…"


def bounded_path(value: str, *, limit: int) -> str:
    """Return a safe bounded path that keeps its final components identifiable."""

    label = " ".join(strip_control_sequences(value).split()) or "<unnamed>"
    if len(label) <= limit:
        return label

    parts = [part for part in label.split("/") if part]
    if len(parts) < 2 or limit < 10:
        return middle_elide(label, limit)

    parent, leaf = parts[-2:]
    # Keep the two components nearest the selected value. The leaf receives
    # most of the budget, and each long component is elided in its middle so
    # sibling prefixes such as ``first_...`` and ``second_...`` stay distinct.
    component_budget = limit - len("…//")
    leaf_budget = min(len(leaf), max(8, component_budget * 2 // 3))
    parent_budget = component_budget - leaf_budget
    if parent_budget < 4:
        shift = 4 - parent_budget
        parent_budget += shift
        leaf_budget = max(1, leaf_budget - shift)

    return f"…/{middle_elide(parent, parent_budget)}/{middle_elide(leaf, leaf_budget)}"


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


@dataclass(frozen=True, slots=True)
class IncidentSummary:
    """Bounded aggregate for one incident kind within a logical lifecycle."""

    kind: str
    occurrences: int
    emitted: int
    suppressed: int
    unique: int
    overflowed: int


class IncidentTracker:
    """Bound and summarize diagnostics for one explicit logical lifecycle."""

    MAX_PART_LENGTH = 160
    MAX_COUNT = sys.maxsize
    OVERFLOW_PART = "<additional-incidents>"
    KINDS_OVERFLOW_KEY: IncidentKey = ("<additional-kinds>",)

    def __init__(self, *, max_keys: int = 32, max_kinds: int = 16) -> None:
        if max_keys <= 0:
            raise ValueError("max_keys must be positive")
        if max_kinds <= 0:
            raise ValueError("max_kinds must be positive")
        self.max_keys = max_keys
        self.max_kinds = max_kinds
        self.counts: OrderedDict[str, OrderedDict[IncidentKey, int]] = OrderedDict()
        self.overflow_counts: dict[str, int] = {}
        self.overflow_notices: set[str] = set()
        self.kinds_overflow_count = 0
        self.kinds_overflow_notice = False
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

    def record(self, kind: str, *key: IncidentPart, occurrences: int = 1) -> Incident:
        """Record an occurrence and say whether its diagnostic should emit."""

        if occurrences <= 0:
            raise ValueError("occurrences must be positive")
        if not isinstance(kind, str):
            raise TypeError("incident kind must be a string")
        normalized = self.normalize_key((kind, *key))
        normalized_kind = normalized[0]
        if not isinstance(normalized_kind, str):
            raise TypeError("incident kind must be a string")
        partition_key = normalized[1:]

        with self.lock:
            partition = self.counts.get(normalized_kind)
            if partition is None and len(self.counts) >= self.max_kinds:
                self.kinds_overflow_count = min(self.kinds_overflow_count + occurrences, self.MAX_COUNT)
                count = self.kinds_overflow_count
                stored_key = self.KINDS_OVERFLOW_KEY
                overflow = True
                emit = not self.kinds_overflow_notice
                self.kinds_overflow_notice = True
            else:
                if partition is None:
                    partition = OrderedDict()
                    self.counts[normalized_kind] = partition

                overflow = partition_key not in partition and len(partition) >= self.max_keys
                if overflow:
                    count = min(
                        self.overflow_counts.get(normalized_kind, 0) + occurrences,
                        self.MAX_COUNT,
                    )
                    self.overflow_counts[normalized_kind] = count
                    stored_key = (normalized_kind, self.OVERFLOW_PART)
                    emit = normalized_kind not in self.overflow_notices
                    self.overflow_notices.add(normalized_kind)
                else:
                    previous = partition.get(partition_key, 0)
                    count = min(previous + occurrences, self.MAX_COUNT)
                    partition[partition_key] = count
                    stored_key = normalized
                    emit = previous == 0

        return Incident(
            key=stored_key,
            count=count,
            suppressed=max(count - 1, 0),
            emit=emit,
            overflow=overflow,
        )

    def summary(self) -> tuple[IncidentSummary, ...]:
        """Return bounded aggregates without mutating this lifecycle."""

        with self.lock:
            summaries: list[IncidentSummary] = []
            for kind, partition in self.counts.items():
                overflowed = self.overflow_counts.get(kind, 0)
                occurrences = sum(partition.values()) + overflowed
                emitted = len(partition) + int(kind in self.overflow_notices)
                summaries.append(
                    IncidentSummary(
                        kind=kind,
                        occurrences=occurrences,
                        emitted=emitted,
                        suppressed=(
                            sum(max(count - 1, 0) for count in partition.values())
                            + max(overflowed - int(kind in self.overflow_notices), 0)
                        ),
                        unique=len(partition),
                        overflowed=overflowed,
                    )
                )

            if self.kinds_overflow_count:
                summaries.append(
                    IncidentSummary(
                        kind=str(self.KINDS_OVERFLOW_KEY[0]),
                        occurrences=self.kinds_overflow_count,
                        emitted=int(self.kinds_overflow_notice),
                        suppressed=max(self.kinds_overflow_count - int(self.kinds_overflow_notice), 0),
                        unique=0,
                        overflowed=self.kinds_overflow_count,
                    )
                )
            return tuple(summaries)

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
                self.overflow_notices.clear()
                self.kinds_overflow_count = 0
                self.kinds_overflow_notice = False
                return

            normalized = self.normalize_key((kind,))[0]
            if not isinstance(normalized, str):
                raise TypeError("incident kind must be a string")
            self.counts.pop(normalized, None)
            self.overflow_counts.pop(normalized, None)
            self.overflow_notices.discard(normalized)


@dataclass(slots=True)
class _ScopedIncidents:
    reference: weakref.ReferenceType[Any]
    tracker: IncidentTracker


class IncidentRegistry:
    """Weakly scope bounded trackers to live owners in this process."""

    SCOPE_OVERFLOW_PART = "<additional-scopes>"

    def __init__(self, *, max_scopes: int = 32, max_keys: int = 32, max_kinds: int = 16) -> None:
        self.max_scopes = max_scopes
        self.max_keys = max_keys
        self.max_kinds = max_kinds
        self.global_tracker = IncidentTracker(max_keys=max_keys, max_kinds=max_kinds)
        self.scope_overflow = IncidentTracker(max_keys=1, max_kinds=max_kinds)
        self.scopes: OrderedDict[int, _ScopedIncidents] = OrderedDict()
        self.lock = Lock()

    def _tracker(self, scope: object, *, create: bool) -> IncidentTracker | None:
        token = id(scope)
        with self.lock:
            state = self.scopes.get(token)
            if state is not None and state.reference() is scope:
                return state.tracker
            if not create or len(self.scopes) >= self.max_scopes:
                return None

            def remove(reference: weakref.ReferenceType[Any]) -> None:
                with self.lock:
                    current = self.scopes.get(token)
                    if current is not None and current.reference is reference:
                        self.scopes.pop(token, None)

            try:
                reference = weakref.ref(scope, remove)
            except TypeError as error:
                raise TypeError("incident scopes must support weak references") from error
            tracker = IncidentTracker(max_keys=self.max_keys, max_kinds=self.max_kinds)
            self.scopes[token] = _ScopedIncidents(reference=reference, tracker=tracker)
            return tracker

    def record(
        self,
        kind: str,
        *key: IncidentPart,
        scope: object | None = None,
        occurrences: int = 1,
    ) -> Incident:
        tracker = self.global_tracker if scope is None else self._tracker(scope, create=True)
        if tracker is not None:
            return tracker.record(kind, *key, occurrences=occurrences)

        incident = self.scope_overflow.record(
            kind,
            self.SCOPE_OVERFLOW_PART,
            occurrences=occurrences,
        )
        return Incident(
            key=(kind, self.SCOPE_OVERFLOW_PART),
            count=incident.count,
            suppressed=incident.suppressed,
            emit=incident.emit,
            overflow=True,
        )

    def summary(self, *, scopes: Iterable[object] | None = None) -> tuple[IncidentSummary, ...]:
        trackers: list[IncidentTracker]
        include_scope_overflow = scopes is None
        if scopes is None:
            with self.lock:
                trackers = [self.global_tracker, *(state.tracker for state in self.scopes.values())]
        else:
            trackers = []
            for scope in scopes:
                tracker = self._tracker(scope, create=False)
                if tracker is not None:
                    trackers.append(tracker)

        combined: dict[str, list[int]] = {}
        for tracker in trackers:
            for item in tracker.summary():
                values = combined.setdefault(item.kind, [0, 0, 0, 0, 0])
                values[0] += item.occurrences
                values[1] += item.emitted
                values[2] += item.suppressed
                values[3] += item.unique
                values[4] += item.overflowed

        if include_scope_overflow:
            for item in self.scope_overflow.summary():
                values = combined.setdefault(item.kind, [0, 0, 0, 0, 0])
                values[0] += item.occurrences
                values[1] += item.emitted
                values[2] += item.suppressed
                values[4] += item.occurrences

        return tuple(
            IncidentSummary(
                kind=kind,
                occurrences=values[0],
                emitted=values[1],
                suppressed=values[2],
                unique=values[3],
                overflowed=values[4],
            )
            for kind, values in sorted(combined.items())
        )

    def reset(self, kind: str | None = None, *, scopes: Iterable[object] | None = None) -> None:
        if scopes is None:
            with self.lock:
                trackers = [self.global_tracker, *(state.tracker for state in self.scopes.values())]
            trackers.append(self.scope_overflow)
        else:
            trackers = []
            for scope in scopes:
                tracker = self._tracker(scope, create=False)
                if tracker is not None:
                    trackers.append(tracker)
        for tracker in trackers:
            tracker.reset(kind)

    def snapshot(self) -> dict[IncidentKey, int]:
        """Return process-local counters for diagnostics and tests."""

        snapshot = self.global_tracker.snapshot()
        with self.lock:
            scoped = tuple(self.scopes.items())
        for token, state in scoped:
            for key, count in state.tracker.snapshot().items():
                snapshot[("scope", token, *key)] = count
        snapshot.update(self.scope_overflow.snapshot())
        return snapshot


incidents = IncidentRegistry()


def record_incident(
    kind: str,
    *key: IncidentPart,
    scope: object | None = None,
    occurrences: int = 1,
) -> Incident:
    """Record an incident in RelFlow's shared process-local tracker."""

    return incidents.record(kind, *key, scope=scope, occurrences=occurrences)


def log_incident_summaries(
    summaries: Iterable[IncidentSummary],
    *,
    message: str = "suppressed repeated RelFlow diagnostics",
    context: Mapping[str, Any] | None = None,
) -> bool:
    """Log one bounded lifecycle summary when any repeats were suppressed."""

    repeated = [item for item in summaries if item.suppressed > 0]
    if not repeated:
        return False

    payload: dict[str, Any] = dict(context or {})
    payload["incidents"] = {
        item.kind: {
            "occurrences": item.occurrences,
            "suppressed": item.suppressed,
            "unique": item.unique,
            "overflowed": item.overflowed,
        }
        for item in repeated
    }
    console.log(f"[relflow.warning]{message}[/]", payload)
    return True
