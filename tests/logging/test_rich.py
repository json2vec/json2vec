from __future__ import annotations

import os
import re
import subprocess
import sys
from gc import collect

import pydantic
import pytest
from rich.console import Console
from rich.text import Text

import relflow.rich as rich_support
from relflow.rich import IncidentRegistry, IncidentTracker


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
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", output)
    assert "test_rich.py:" not in output

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


def test_import_installs_traceback_hook_without_output() -> None:
    script = "import sys\nhook = sys.excepthook\nimport relflow\nassert sys.excepthook is not hook\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_internal_traceback_install_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_install(**kwargs: object):
        calls.append(kwargs)
        previous = sys.excepthook
        sys.excepthook = lambda *_: None
        return previous

    monkeypatch.setattr(rich_support, "tracebacks_installed", False)
    monkeypatch.setattr(rich_support, "rich_traceback_install", fake_install)
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)

    rich_support.install_tracebacks()
    rich_support.install_tracebacks()

    assert len(calls) == 1
    assert calls[0]["console"] is rich_support.console
    assert calls[0]["width"] == 88
    assert calls[0]["code_width"] == 88
    assert calls[0]["extra_lines"] == 1
    assert calls[0]["show_locals"] is False
    assert calls[0]["max_frames"] == 50
    assert calls[0]["suppress"] == (pydantic,)


def test_import_preserves_an_existing_host_traceback_hook() -> None:
    script = (
        "import sys\n"
        "def host_hook(*args):\n"
        "    sys.stderr.write('host-hook\\n')\n"
        "sys.excepthook = host_hook\n"
        "import relflow\n"
        "assert sys.excepthook is host_hook\n"
        "raise RuntimeError('must-not-be-rendered-twice')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "host-hook\n"


def test_verbose_environment_switch_is_effective_at_import() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import relflow.rich as support; assert support.is_verbose()"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "RELFLOW_VERBOSE": "1"},
    )

    assert result.returncode == 0, result.stderr


def test_installed_traceback_hides_locals_and_writes_to_stderr() -> None:
    script = (
        "import os\nimport relflow\nsecret = os.environ['RELFLOW_TEST_SECRET']\nraise RuntimeError('diagnostic boom')\n"
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
    assert "╭" in result.stderr
    assert "Traceback (most recent call last)" in result.stderr
    assert "RuntimeError: diagnostic boom" in result.stderr
    assert "must-not-leak" not in result.stderr
    assert max(map(len, result.stderr.splitlines())) <= 88


def test_installed_traceback_constrains_long_messages_in_a_wide_terminal() -> None:
    script = "import relflow\nraise RuntimeError('x' * 180)\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "200"},
    )

    assert result.returncode != 0
    assert "RuntimeError" in result.stderr
    assert max(map(len, result.stderr.splitlines())) <= 88
    assert len(result.stderr.splitlines()) < 40


def test_installed_traceback_bounds_multiline_messages_and_notes() -> None:
    script = (
        "import relflow\n"
        "error = RuntimeError('\\n'.join(f'line-{index}' for index in range(100)))\n"
        "for index in range(20): error.add_note('note-' + str(index) + '-' + 'x' * 500)\n"
        "raise error\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "200"},
    )

    assert result.returncode != 0
    assert "more lines" in result.stderr
    assert "more notes" in result.stderr
    assert "line-99" not in result.stderr
    assert "note-19" not in result.stderr
    assert max(map(len, result.stderr.splitlines())) <= 88


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (
            "import relflow\n"
            "try:\n"
            "    raise ValueError('inner failure')\n"
            "except ValueError as error:\n"
            "    raise RuntimeError('outer failure') from error\n",
            ("ValueError: inner failure", "RuntimeError: outer failure"),
        ),
        (
            "import relflow\nraise ExceptionGroup('batch failure', [ValueError('first'), TypeError('second')])\n",
            ("ExceptionGroup: batch failure", "ValueError: first", "TypeError: second"),
        ),
    ],
    ids=["chain", "exception-group"],
)
def test_installed_traceback_preserves_exception_structure(
    script: str,
    expected: tuple[str, ...],
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "200"},
    )

    assert result.returncode != 0
    for text in expected:
        assert text in result.stderr
    assert max(map(len, result.stderr.splitlines())) <= 88


@pytest.mark.parametrize(
    "script",
    [
        (
            "import relflow\n"
            "def fail(depth):\n"
            "    if depth == 0: raise RuntimeError('chain-root')\n"
            "    try: fail(depth - 1)\n"
            "    except RuntimeError as error: raise RuntimeError(f'chain-{depth}') from error\n"
            "fail(100)\n"
        ),
        ("import relflow\nraise ExceptionGroup('many failures', [ValueError(f'item-{i}') for i in range(200)])\n"),
    ],
    ids=["long-chain", "wide-exception-group"],
)
def test_installed_traceback_bounds_exception_structure_cardinality(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "200"},
    )

    assert result.returncode != 0
    assert "omitted" in result.stderr
    assert len(result.stderr.splitlines()) < 200
    assert len(result.stderr) < 20_000
    assert max(map(len, result.stderr.splitlines())) <= 88


def test_installed_traceback_strips_controls_from_source_lines(tmp_path) -> None:
    script = tmp_path / "unsafe_source.py"
    script.write_text(
        "import relflow\nraise RuntimeError('boom')  # \x1b]8;;https://example.invalid\x07unsafe\x1b]8;;\x07\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "200"},
    )

    assert result.returncode != 0
    assert "RuntimeError: boom" in result.stderr
    assert "\x1b" not in result.stderr
    assert "\x07" not in result.stderr
    assert not {character for character in result.stderr if ord(character) < 32} - {"\n", "\t"}


def test_incident_tracker_suppresses_repeats_and_retains_count() -> None:
    tracker = IncidentTracker(max_keys=4)

    occurrences = [tracker.record("number-clamp", "record/amount") for _ in range(1000)]

    assert sum(item.emit for item in occurrences) == 1
    assert occurrences[-1].count == 1000
    assert occurrences[-1].suppressed == 999
    assert tracker.snapshot() == {("number-clamp", "record/amount"): 1000}
    assert tracker.summary()[0].suppressed == 999


def test_incident_tracker_bounds_many_unique_keys_and_emissions() -> None:
    tracker = IncidentTracker(max_keys=3)

    occurrences = [tracker.record("streaming-read", f"file-{index}") for index in range(1000)]
    unrelated = tracker.record("number-clamp", 1, "record/amount", "nonfinite")
    snapshot = tracker.snapshot()

    assert sum(item.emit for item in occurrences) == 4
    assert unrelated.emit is True
    assert len(snapshot) == 5
    assert snapshot[("streaming-read", "<additional-incidents>")] == 997
    assert snapshot[("number-clamp", 1, "record/amount", "nonfinite")] == 1
    assert occurrences[-1].overflow is True
    assert occurrences[-1].suppressed == 996
    assert tracker.summary()[0] == rich_support.IncidentSummary(
        kind="streaming-read",
        occurrences=1000,
        emitted=4,
        suppressed=996,
        unique=3,
        overflowed=997,
    )


def test_incident_tracker_bounds_unique_incident_kinds() -> None:
    tracker = IncidentTracker(max_keys=2, max_kinds=2)

    first = tracker.record("first-kind", "one")
    second = tracker.record("second-kind", "two")
    overflow = tracker.record("third-kind", "three")
    suppressed = tracker.record("fourth-kind", "four")

    assert first.emit is True
    assert second.emit is True
    assert overflow.emit is True
    assert overflow.overflow is True
    assert suppressed.emit is False
    assert tracker.snapshot()[("<additional-kinds>",)] == 2
    assert tracker.summary()[-1] == rich_support.IncidentSummary(
        kind="<additional-kinds>",
        occurrences=2,
        emitted=1,
        suppressed=1,
        unique=0,
        overflowed=2,
    )


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


def test_incident_registry_releases_scoped_trackers_with_their_owner() -> None:
    class Owner:
        pass

    registry = IncidentRegistry(max_scopes=2)
    owner = Owner()
    registry.record("kind", "key", scope=owner)
    assert len(registry.scopes) == 1

    del owner
    collect()

    assert len(registry.scopes) == 0


def test_incident_registry_reports_overflowed_owner_scopes_as_overflow() -> None:
    class Owner:
        pass

    registry = IncidentRegistry(max_scopes=1)
    retained = Owner()
    overflowed = Owner()
    registry.record("kind", "retained", scope=retained)
    first = registry.record("kind", "overflowed", scope=overflowed)
    repeated = registry.record("kind", "overflowed", scope=overflowed)

    assert first.emit is True
    assert first.overflow is True
    assert repeated.emit is False
    assert registry.summary() == (
        rich_support.IncidentSummary(
            kind="kind",
            occurrences=3,
            emitted=2,
            suppressed=1,
            unique=1,
            overflowed=2,
        ),
    )


def test_incident_summary_logs_only_when_occurrences_were_suppressed() -> None:
    tracker = IncidentTracker()
    tracker.record("kind", "key")

    with rich_support.console.capture() as first:
        emitted = rich_support.log_incident_summaries(tracker.summary())
    assert emitted is False
    assert first.get() == ""

    tracker.record("kind", "key")
    with rich_support.console.capture() as repeated:
        emitted = rich_support.log_incident_summaries(tracker.summary())
    assert emitted is True
    assert "occurrences" in repeated.get()
    assert "suppressed" in repeated.get()
