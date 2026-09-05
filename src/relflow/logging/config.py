from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping
from typing import Any, ClassVar, Self

from rich.console import Console, ConsoleRenderable
from rich.logging import RichHandler
from rich.text import Text

CONTEXT = "relflow_context"
LOG_LEVEL = os.getenv("RELFLOW_LOG_LEVEL", "DEBUG").upper()
console = Console(stderr=True)


class Handler(RichHandler):
    """Render structured RelFlow records as compact, styled terminal events."""

    styles: ClassVar[dict[int, str]] = {
        logging.DEBUG: "bold white on #475569",
        logging.INFO: "bold white on #2563eb",
        logging.WARNING: "bold #422006 on #fbbf24",
        logging.ERROR: "bold white on #dc2626",
        logging.CRITICAL: "bold white on #991b1b",
    }

    def render_message(self, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        context = getattr(record, CONTEXT, {})
        if not isinstance(context, Mapping):
            context = {}

        rendered = Text()
        component = context.get("component")
        if component is not None:
            label = str(component).replace("_", " ").upper()
            rendered.append(f" {label} ", style=self.styles.get(record.levelno, "bold white on #2563eb"))
            rendered.append("  ")
        rendered.append(message, style="bold" if record.levelno >= logging.WARNING else None)

        details = [(key, value) for key, value in context.items() if key != "component"]
        if details:
            rendered.append("\n")
            for index, (key, value) in enumerate(details):
                if index:
                    rendered.append("  •  ", style="dim")
                rendered.append(str(key), style="dim cyan")
                rendered.append("=", style="dim")
                rendered.append(self.value(value), style="cyan")
        return rendered

    @staticmethod
    def value(value: object) -> str:
        if value is None:
            return "∅"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:,.6g}"
        rendered = str(value).replace("\n", " ↵ ")
        return rendered if len(rendered) <= 120 else f"{rendered[:117]}…"


class Logger(logging.LoggerAdapter):
    """Standard logger with immutable structured context binding."""

    def bind(self, **context: object) -> Self:
        return type(self)(self.logger, {**self.extra, **context})

    def process(
        self,
        message: object,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[object, MutableMapping[str, Any]]:
        extra = dict(kwargs.get("extra") or {})
        supplied = extra.pop(CONTEXT, {})
        if not isinstance(supplied, Mapping):
            raise TypeError(f"logging {CONTEXT} must be a mapping, got {type(supplied).__name__}")
        extra[CONTEXT] = {**self.extra, **supplied}
        kwargs["extra"] = extra
        return message, kwargs


def configure(*, level: str | int | None = None, output: Console | None = None) -> Logger:
    """Install one Rich handler on the package logger and return its adapter."""

    selected = LOG_LEVEL if level is None else level
    if isinstance(selected, str):
        name = selected.upper()
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if name not in levels:
            choices = ", ".join(levels)
            raise ValueError(f"RelFlow log level must be one of {choices}, got {selected!r}")
        selected = levels[name]
    elif not isinstance(selected, int) or isinstance(selected, bool):
        raise TypeError(f"RelFlow log level must be a string or integer, got {type(selected).__name__}")

    core = logging.getLogger("relflow")
    core.setLevel(selected)
    core.propagate = False
    for existing in tuple(core.handlers):
        if isinstance(existing, Handler):
            core.removeHandler(existing)

    handler = Handler(
        level=selected,
        console=output or console,
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_word_wrap=True,
        tracebacks_show_locals=False,
        log_time_format="%H:%M:%S",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    core.addHandler(handler)
    return Logger(core, {})


logger = configure()

__all__ = ["CONTEXT", "Handler", "LOG_LEVEL", "Logger", "configure", "console", "logger"]
