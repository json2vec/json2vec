from __future__ import annotations

import os
import subprocess
import sys

import pytest
from rich.console import Console
from rich.text import Text

import relflow.rich as rich_support
from relflow.rich import IncidentTracker


def test_internal_console_is_themed_and_writes_to_stderr() -> None:
    assert isinstance(rich_support.console, Console)
    assert rich_support.console.stderr is True
    assert {
        "relflow.info",
        "relflow.warning",
        "relflow.error",
        "relflow.name",
        "relflow.type",
        "relflow.dim",
    } <= set(rich_support.theme.styles)


def test_diagnostics_are_not_exported_as_public_api() -> None:
    import relflow
    import relflow.logging

    assert "rich" not in relflow.__all__
    for name in ("console", "configure_console", "install_tracebacks"):
        assert not hasattr(relflow, name)
        assert not hasattr(relflow.logging, name)


def test_internal_verbosity_is_settable_without_replacing_console() -> None:
    try:
        rich_support.set_verbose(True)
        assert rich_support.is_verbose() is True
        rich_support.set_verbose(False)
        assert rich_support.is_verbose() is False
    finally:
        rich_support.set_verbose(False)


def test_console_can_be_captured_silenced_and_renders_values_literally() -> None:
    with rich_support.console.capture() as captured:
        rich_support.console.log("[relflow.warning]diagnostic[/]", {"value": "[literal]"})

    output = captured.get()
    assert "diagnostic" in output
    assert "[literal]" in output

    previous_quiet = rich_support.console.quiet
    try:
        rich_support.console.quiet = True
        with rich_support.console.capture() as silenced:
            rich_support.console.log("must not appear")
        assert silenced.get() == ""
    finally:
        rich_support.console.quiet = previous_quiet


def test_isolated_render_helpers_are_plain_and_theme_neutral() -> None:
    renderable = Text.assemble(
        ("bold", "bold"),
        " ",
        ("dim", "dim"),
        " ",
        ("italic", "italic"),
    )

    plain = rich_support.render_text(renderable)
    html = rich_support.render_html(renderable)

    assert plain == "bold dim italic"
    assert "\x1b" not in plain
    assert "color: #" not in html
    assert "background-color: #" not in html
    assert "font-weight: bold" in html
    assert "opacity: 0.7" in html
    assert "font-style: italic" in html


def test_html_palette_sanitizing_does_not_rewrite_rendered_text() -> None:
    literal = "literal color: #deadbeef; background-color: #abcdef; remains"

    html = rich_support.render_html(Text(literal, style="red on blue"))

    assert literal in html
    assert "color: #" not in html.split(literal, maxsplit=1)[0]
    assert "background-color: #" not in html.split(literal, maxsplit=1)[0]


def test_render_helpers_strip_terminal_and_control_sequences() -> None:
    unsafe = Text("before\x1b[31mred\x1b[0m\x1b]52;c;clipboard\x07after\x00")

    plain = rich_support.render_text(unsafe)
    html = rich_support.render_html(unsafe)

    for rendered in (plain, html):
        assert "\x1b" not in rendered
        assert "\x07" not in rendered
        assert "\x00" not in rendered
        assert "before" in rendered
        assert "red" in rendered
        assert "after" in rendered


def test_import_has_no_output_or_traceback_hook_side_effect() -> None:
    script = "import sys\nhook = sys.excepthook\nimport relflow\nassert sys.excepthook is hook\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_internal_traceback_install_is_explicit_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_install(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(rich_support, "tracebacks_installed", False)
    monkeypatch.setattr(rich_support, "rich_traceback_install", fake_install)

    rich_support.install_tracebacks()
    rich_support.install_tracebacks()

    assert len(calls) == 1
    assert calls[0]["console"] is rich_support.console
    assert calls[0]["show_locals"] is False
    assert calls[0]["max_frames"] == 50


def test_installed_traceback_hides_locals_and_writes_to_stderr() -> None:
    script = (
        "import os\n"
        "from relflow.rich import install_tracebacks\n"
        "install_tracebacks()\n"
        "secret = os.environ['RELFLOW_TEST_SECRET']\n"
        "raise RuntimeError('diagnostic boom')\n"
    )
    environment = {**os.environ, "RELFLOW_TEST_SECRET": "must-not-leak"}

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "RuntimeError: diagnostic boom" in result.stderr
    assert "must-not-leak" not in result.stderr


def test_incident_tracker_suppresses_repeats_and_retains_count() -> None:
    tracker = IncidentTracker(max_keys=4)

    occurrences = [tracker.record("number-clamp", "record/amount", limit=1000) for _ in range(1000)]

    assert sum(item.emit for item in occurrences) == 3
    assert occurrences[-1].count == 1000
    assert occurrences[-1].suppressed == 997
    assert tracker.snapshot() == {("number-clamp", "record/amount"): 1000}


def test_incident_tracker_bounds_many_unique_keys_and_emissions() -> None:
    tracker = IncidentTracker(max_keys=3)

    occurrences = [tracker.record("streaming-read", f"file-{index}", limit=2) for index in range(1000)]
    unrelated = tracker.record("number-clamp", 1, "record/amount", "nonfinite")
    snapshot = tracker.snapshot()

    assert sum(item.emit for item in occurrences) == 3
    assert unrelated.emit is True
    assert len(snapshot) == 5
    assert snapshot[("streaming-read", "<additional-incidents>")] == 997
    assert snapshot[("number-clamp", 1, "record/amount", "nonfinite")] == 1
    assert occurrences[-1].overflow is True
    assert occurrences[-1].suppressed == 997


def test_incident_tracker_bounds_unique_incident_kinds() -> None:
    tracker = IncidentTracker(max_keys=2, max_kinds=2)

    first = tracker.record("first-kind", "one")
    second = tracker.record("second-kind", "two")
    overflow = tracker.record("third-kind", "three")

    assert first.emit is True
    assert second.emit is True
    assert overflow.emit is False
    assert overflow.overflow is True
    assert tracker.snapshot()[("<additional-kinds>",)] == 1


def test_incident_tracker_can_reset_one_kind_without_affecting_others() -> None:
    tracker = IncidentTracker(max_keys=2)
    tracker.record("streaming-read", "file.parquet")
    tracker.record("number-clamp", 1, "record/amount")
    assert tracker.record("number-clamp", 1, "record/amount").emit is False

    tracker.reset("number-clamp")

    assert tracker.record("number-clamp", 2, "record/amount").emit is True
    assert tracker.snapshot()[("streaming-read", "file.parquet")] == 1


def test_incident_tracker_retains_only_bounded_scalar_keys() -> None:
    tracker = IncidentTracker(max_keys=2)
    long_value = "x" * 1000

    incident = tracker.record("kind", long_value)

    assert incident.key == ("kind", "x" * 160)
    with pytest.raises(TypeError, match="short scalar"):
        tracker.record("kind", object())
