# Rich Console, Tracebacks, And Rendering Spec

- Status: Implemented
- Date: 2026-08-17
- Scope owner: Runtime and rendering maintainers

## Summary

RelFlow should use Rich directly for human-facing diagnostics, tracebacks, and
object display. Loguru should be removed.

This is intentionally not a one-for-one logging migration. The current logs are
an input to the redesign, not a compatibility contract. The implementation
should keep only messages that help a user make a decision, use
`rich.console.Console.log()` for those messages, and delete noisy lifecycle and
per-item debug output.

The implementation is deliberately tucked inside an internal Rich support
module. It adds no public RelFlow console, logging configuration, or traceback
installation API. Importing `relflow` installs one safe, process-wide Rich
uncaught-exception hook with locals hidden when the host still uses Python's
default hook; an existing host hook is preserved. Applications that want their
own console, or intentionally need to replace that traceback presentation,
configure Rich directly.

Rich remains a required core dependency. RelFlow does not introduce another
logging abstraction, structured event schema, custom levels, or JSON sink.

## Goals

- one internal shared, themed `Console` for RelFlow-owned human output;
- concise `console.log()` messages with Rich rendering of native values;
- bounded output for long-running jobs: routine epochs, steps, batches, and
  records never generate console logs;
- one idempotent, import-installed Rich hook for uncaught exceptions, plus Rich
  tracebacks at RelFlow-owned failure boundaries;
- useful direct and nested rendering for common RelFlow objects;
- executable Quarto examples that show the same Rich object views as notebooks,
  with a plain-text fallback rather than hand-authored output;
- no Loguru dependency, imports, sinks, formatting, or tests;
- safe behavior in terminals, redirected output, notebooks, Lightning, and
  serving;
- no accidental printing of raw observations, predictions, secrets, or
  traceback locals.

## Non-Goals

- preserving every existing log call, level, message, or bound field;
- machine-readable or remotely shipped logs;
- replacing an application's standard-library, Uvicorn, or platform logging;
- adding a new `Logger`, handler adapter, event catalog, or `TRACE` level;
- adding public console, verbosity, or traceback-configuration exports;
- using console logs as a heartbeat, progress stream, or metrics backend;
- automatically replacing Lightning progress or experiment metric logging;
- installing a global pretty-print hook during `import relflow`;
- rendering every private Torch or Lightning object.

## State Before This Change

- Rich is already a core dependency and already renders `Leaf`, `Branch`,
  `Schema`, `Model`, `Selection`, and `TensorFieldBase`.
- Loguru is a core dependency used across architecture, data, tensorfields, and
  one Lightning callback.
- `src/relflow/logging/config.py` is dormant unless imported directly. When it
  is imported, it removes global Loguru sinks and prints JSON-shaped Rich output
  to stdout.
- Loguru's default sink still makes ordinary library operations noisy, including
  model construction and processor configuration.
- No production call currently emits a logged exception traceback.
- Existing Rich display has correctness gaps: unnamed leaves and branches can
  fail, nested objects become a quoted multiline string, overlapping
  selections duplicate subtrees, and tensorfield display may copy or
  synchronize full device state.

The implementation should fix these behaviors rather than preserve them.

## Design

### Internal Shared Console

Add a dependency-light internal module at `src/relflow/rich.py`:

```python
from rich.console import Console
from rich.theme import Theme

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
```

The exact palette may evolve; semantic style names are an internal convention.

Requirements:

- do not export the console or its configuration from `relflow` or
  `relflow.logging`;
- write diagnostics to stderr so stdout remains available for predictions,
  benchmark JSON, and pipes;
- let Rich own terminal width, color detection, `NO_COLOR`, wrapping, and
  pretty rendering;
- create no secondary JSON console or logging handler;
- never interpolate user-controlled text into Rich markup; pass values as
  separate objects or `Text`/`Pretty` instances;
- keep `log_locals=False`;
- keep any enhanced-verbosity switch internal and available only to
  RelFlow-owned entry points and tests; it is not a compatibility surface;
- allow internal tests and owned application boundaries to capture or redirect
  the console through normal Rich APIs.

Internal modules should import the same console object. Renderable objects still
use the `console` supplied to `__rich_console__`; they should not reach back to
the global console while being rendered.

### What Counts As A Log

Use `console.log()` only for a human-visible operational event where the user
can respond. The shared console adds a timestamp, deliberately omits source
paths, and can render dicts, paths, enums, tensors, and RelFlow renderables
without converting everything to a string.

Prefer:

```python
console.log(
    "[relflow.warning]vocabulary is near capacity[/]",
    {"address": address, "size": size, "capacity": capacity},
)
```

Avoid a wrapper that recreates levels, binding, propagation, or structured
logging. If a callsite needs a label, use the shared semantic styles.

### Long-Running Job Policy

Console output must scale with unusual incidents and meaningful state changes,
not with job duration.

| Mode | Allowed output |
| --- | --- |
| Default | Actionable warnings, degraded/fallback behavior, unexpected exceptions, and other events where something materially went wrong |
| Enhanced internal diagnostics | Default output plus one-time setup summaries, explicit phase transitions, and occasional state changes useful for diagnosis |
| Lightning metrics/progress | Repeated loss, throughput, counters, epoch/step progress, and other time-series telemetry |

RelFlow must never emit a console line for every observation, batch, optimizer
step, or epoch. Enhanced verbosity does not change that rule. It may reveal
additional one-time context, but recurring telemetry still belongs in metrics
or progress callbacks.

Repeated incidents must also be bounded. A callsite that can fire in a hot loop
logs the first occurrence, at most one notice when its detail budget overflows,
and one suppressed-count summary at a natural lifecycle boundary. Per-address
detail must also have a cap so a stream
of millions of unique bad keys/files cannot create millions of lines. A final
aggregate summary may be emitted when a natural lifecycle boundary is already
available; it must not require an epoch-by-epoch message.

The general incident registry is process-local, bounded in memory, and stores
only small keys/counters—not tensors, observations, exceptions, or model
objects. Budgets are partitioned by a bounded set of stable incident kinds.
Model/schema lifecycles use weak owner scopes, so completed jobs release their
state without placing object IDs in keys. Streaming failures group on suffix and
exception class while retaining only the first credential-redacted path as
representative context. Their bounded counters belong to the data-module split
and are shared only with that split's loader workers, so the warning budget
survives loader and non-persistent worker recreation and emits at most one
suppression summary for the split lifecycle. If worker-shared diagnostic state
cannot be created, workers suppress those advisory messages rather than failing
or duplicating them; data loading and terminal read errors are unchanged.
`RELFLOW_VERBOSE=1` enables the sparse enhanced mode before import.
Internal verbose callsites check the configured flag before constructing
expensive context.
Application-owned Rich output is independent of this internal state.

### Current Log Triage

The migration should make an editorial decision at each callsite.

| Current area | Direction |
| --- | --- |
| Model initialization | Remove; constructing an object should not announce itself |
| Processor binding, normalization, provider resolution, and per-output `TRACE` | Remove; these are hot-path/internal details |
| Schema mutation audit messages | Remove by default; make the mutated objects and selections easy to inspect |
| Epoch lifecycle callback | Remove/deprecate; epoch repetition belongs to Lightning progress and metrics even in verbose mode |
| Checkpoint load and rollback | Routine load and configured best-checkpoint rollback are verbose-only |
| Streaming file skipped after a recoverable read error | Keep as a warning with one credential-redacted representative path, suffix, and exception class; never include a parser message that may echo raw rows; deduplicate by suffix/error class and summarize suppression |
| Vocabulary nearing/exceeding capacity | Log threshold transitions or first occurrence, not every synchronization |
| Unsafe number inputs being clamped | Log the first incident per field and meaningful severity changes; count repeated clamps without per-batch output |
| Cluster configuration that cannot train | Prefer a Python warning during configuration, or a concise console warning if it is truly runtime-specific |
| No trainable fields / zero loss | Warn once per affected run/schema, or fail if the contract should be strict; never repeat per step |
| Configuration startup message | Remove; configuration should not log itself |

Python `warnings.warn()` remains appropriate for API misuse, questionable model
configuration, and deprecation. Lightning's `Model.track()`,
`LightningModule.log()`, and `ThroughputLogger` remain metric logging and are
not routed through the Rich console.

A finite streaming read raises only when every attempted source failed with a
known recoverable read error. A valid empty file, or a chunk/record worker with
no locally assigned rows, is a successful empty result. Replacement sampling
instead fails after a bounded run of attempts without yielding because it
otherwise could retry bad or empty files forever.

### Tracebacks

Use the two Rich mechanisms for their intended jobs:

1. `console.print_exception(show_locals=False)` inside an `except` block when
   RelFlow owns and consumes or translates the failure at that boundary.
2. `rich.traceback.install(console=console, show_locals=False)` once during
   package initialization, providing a safe default for uncaught exceptions.

Do not wrap either mechanism in a public RelFlow convenience API. Importing
`relflow` installs the hook idempotently and emits no output. Applications that
own the process and deliberately need different traceback presentation may
replace the default with Rich's API directly.

Traceback rules:

- locals are hidden by default;
- frame count, causal-chain/group cardinality, exception messages, and
  exception notes are bounded;
- the entire traceback and its source panels have a stable 88-column maximum,
  while narrower terminals may constrain both;
- repeated imports or internal initialization do not stack hooks;
- an existing host-owned exception hook is preserved rather than chained;
- internal Pydantic frames are suppressed while the originating user frame and
  exception remain visible;
- exception chaining and exception groups remain visible;
- expected/recoverable errors get one short warning, not a traceback;
- unexpected errors get one traceback at the boundary that handles them;
- a boundary that re-raises an exception for the process hook does not print it
  first;
- HTTP responses never include tracebacks or locals;
- RelFlow-owned failure context never adds raw request bodies, observations,
  predictions, or processor-bound values;
- hiding locals does not redact text already embedded in an arbitrary exception
  message or source line; application code must not put credentials or raw
  records there;
- exception, path, function, and source text is stripped of untrusted terminal
  control sequences before output;
- notebook runtimes control how cell exceptions reach the hook, and Rich may
  integrate with a supported active shell; docs do not install another hook in
  a notebook cell.

### Serving

Importing `relflow` installs the internal safe Rich traceback hook before either
deployment API is used. `Deployment.app()` does not otherwise reconfigure
process output, and `Deployment.serve()` does not install a second hook. An
external ASGI runner remains responsible for its own logs and may intentionally
replace the process-wide traceback presentation.

Shared-console output remains incident-only; routine startup and status
messages stay with Uvicorn. Uvicorn can keep its native logging rather than
being forced through a custom adapter. At a RelFlow-owned exception boundary,
print one Rich traceback and return or raise according to the existing HTTP
contract; avoid printing the same exception again through both RelFlow and
Uvicorn or the process hook.

`Deployment.log_level` continues to configure Uvicorn. It does not need to
become a general Rich console level system.

### Lightning And Distributed Runs

- do not route metrics through `Console`;
- do not replace progress callbacks automatically;
- remove/deprecate `EpochLifecycleLogger`; RelFlow should not offer a callback
  whose purpose is one line per epoch;
- never add step-, batch-, or epoch-driven console output, including in verbose
  mode;
- keep existing rank-zero guards where they already prevent duplicated
  warnings;
- avoid adding distributed synchronization merely to decorate a message;
- accept that independent workers may interleave stderr output; cross-process
  aggregation belongs to the hosting application;
- verify that occasional `console.log()` output coexists with Lightning's
  default progress bar.

## Rich Object Rendering

### Shared Rendering Foundation

The same internal `rich.py` module should own semantic styles and small helpers for
plain-text/HTML capture. It must not import model, data, inference, or
tensorfield modules, so those subsystems can use it without cycles.

Use native Rich composition (`Tree`, `Table`, `Text`, `Pretty`) rather than
manually calling child `__rich_console__` methods and concatenating connectors.

Protocol expectations:

- `__rich_console__` provides the full direct view;
- `__rich_repr__` provides a compact nested view and never calls `str(self)`;
- `str(obj)` is deterministic plain text without ANSI codes;
- existing `repr(obj)` and serialization behavior stay intact unless a class
  explicitly needs a safer stable repr;
- Jupyter `_repr_mimebundle_` / `_repr_html_` and Marimo `_mime_` continue to
  work and escape content correctly.

RelFlow does not globally call `rich.pretty.install()`. Users may do so through
Rich; RelFlow objects should simply behave well when nested in ordinary lists
and dictionaries.

### Quarto And Marimo Rendering Contract

The rendered Quarto site is a supported presentation target for RelFlow's Rich
objects. Documentation must exercise the real renderer from executable cells;
it must not substitute copied terminal output, screenshots, or a separately
maintained diagram for the object being documented.

- A RelFlow renderable should normally be the final expression in a
  `{python .marimo}` cell. This lets `_repr_mimebundle_` / `_mime_` provide
  `text/html` plus `text/plain`. Built-in `print(model)` is not an equivalent
  example because it discards the HTML representation.
- `Renderable._display_()` returns a small internal MIME-bundle proxy. Marimo
  checks this protocol before its opinionated type formatters, so a RelFlow
  `Model` uses the production Rich tree instead of Marimo's generic PyTorch
  module viewer. The proxy has no Marimo import or runtime dependency and
  retains both HTML and plain-text representations.
- Put independently rendered values in separate cells. A deliberate Rich
  composition or a documentation-only capture helper is acceptable when one
  example must contain multiple values, but it must still use the production
  renderables.
- Static extraction must prefer the HTML member of the MIME bundle and retain
  plain text for non-HTML consumers. The generated page must remain readable
  without client-side execution.
- Representative docs must cover a model/schema tree, a selection, and a
  RelFlow object nested in a basic list or mapping. P1 data-module and processor
  views should be added to their existing guide pages as those renderers land.
- Object examples use MIME display rather than the internal diagnostic console,
  which intentionally writes to stderr. Any executable Rich console example
  constructs and captures an application-owned, cell-local console so build
  diagnostics stay clean and the example does not replace the import-installed
  process hook.
- Rich output must fit the documentation content column, provide intentional
  wrapping or horizontal overflow at narrow widths, contain no ANSI escape
  sequences, and escape user-controlled text.
- Light and dark Quarto themes must both remain legible. Documentation CSS uses
  semantic classes or variables and must not depend on exact inline hex colors
  emitted by one Rich version.

The canonical live examples are `core-concepts/model-tree.qmd` for the direct
tree, `guides/schema-mutation.qmd` for selections, and
`reference/public-api.qmd` for nested/basic-object behavior. In particular,
the existing `print(model)` example in the public API page should become MIME
display during implementation.

### Object Priorities

| Priority | Objects | Desired view |
| --- | --- | --- |
| P0 | `Leaf`, `Branch`, `Schema`, `Model` | Robust bound/unbound tree, meaningful flags/config, narrow-width wrapping, compact nested repr |
| P0 | `Selection` | One compact entry per selected address; no repeated ancestor subtrees |
| P0 | `TensorFieldBase` | Native indexing over named structural axes; preview the exact current 0D–2D slice, name/shape remaining axes and request a further slice above 2D, leave trailing payload axes untouched, and never copy/synchronize an accelerator by default |
| P0 | `EncodedInput` | Compact bounded address, state-shape, and structural-axis inventory; report metadata presence without rendering metadata values |
| P1 | `Mask` | Effective rate/count, window, scope, offset, and exclusions rather than only `masks=N` |
| P1 | `NodeAttribute`, `NodePredicate` | Stable readable expression; no lambda memory address |
| P1 | `Observation`, preprocessors, postprocessors | Bounded data shape/keys and callable signature/readiness; hide bound values by default |
| P1 | Public data modules | Split/config summary that does not build loaders, iterate data, or open files |
| P2 | `InferenceConfig`, `Deployment`, `Writer`, plugin/vocabulary/counter diagnostics | Compact user-oriented summaries when a documented inspection job needs them |
| Keep native | `Address`, string enums, `Tokens` | Preserve scalar behavior and style in surrounding views |

The listed P0 and P1 views are part of this change. P2 can follow without
inventing a new visual language.

### Rendering Safety

Default rendering must not:

- mutate a model or serialized configuration;
- build modules, loaders, or prediction outputs;
- iterate a dataset or open a path;
- move a tensor to CPU, call a synchronizing reduction/`.item()` on an
  accelerator tensor, or retain an autograd graph;
- render unbounded observations, vocabularies, predictions, or tensor values;
- expose processor-bound secrets in a normal repr or traceback local.

Unnamed or partially bound objects display a placeholder such as `<unnamed>`
rather than raising. Rendering must leave `model_dump`, `state_dict`, caches,
training mode, checkpoints, and pickle output unchanged.

## Source Areas

The implementation can touch:

- `pyproject.toml` and `uv.lock`;
- a new internal `src/relflow/rich.py` with lightweight verbosity/incident
  state and no package-root exports;
- current Loguru producers under `architecture/`, `data/`, `logging/`, and
  `tensorfields/`;
- current renderers in `structs/tree.py`, `structs/structure.py`,
  `structs/experiment.py`, `architecture/root.py`, and
  `tensorfields/base.py`;
- P1/P2 object owners in selectors, processors, datasets, helpers, inference,
  counter, and vocabulary modules;
- logging, rendering, serving, public API, environment, and example tests;
- Quarto/Marimo extraction and presentation in
  `docs/_extensions/marimo-team/marimo/extract.py` and
  `docs/assets/stylesheets/quarto-marimo.css`;
- README and executable documentation pages covering troubleshooting, public
  API, Lightning, performance, serving, model trees, mutations, and data
  modules.

`src/relflow/logging/throughput.py` remains a Lightning metric callback despite
its name. It counts actual encoded-batch cardinality and, in distributed runs,
reports the global observation count over the slowest rank's elapsed time. Its
epoch-boundary reductions and metric emission never go through the Rich
console.

## Dependencies And Breaking Change

- remove `loguru` from `pyproject.toml`;
- regenerate `uv.lock`, removing Loguru and its now-unneeded transitive packages;
- retain Rich as a core dependency and support the declared Rich 14 minimum
  unless the minimum is deliberately raised;
- verify built wheel metadata requires Rich and not Loguru;
- make no compatibility shim for the deleted Loguru configuration module,
  event schema, sinks, or epoch callback;
- keep stdout clean and non-TTY output free of ANSI control sequences.

This is a visual and diagnostic behavior change. Exact old messages, levels,
and bound fields are not compatibility requirements.

## Tests

Focused tests should cover:

- the internal shared Rich `Console` writes to stderr and is not exported;
- `relflow` has no `console`, `configure_console`, or `install_tracebacks`
  attributes;
- import emits no output, installs the safe traceback hook once with locals
  hidden, and does not install a pretty-print hook;
- default mode emits no routine epoch, step, batch, record, construction, or
  mutation chatter;
- verbose mode enables useful one-time context without enabling per-epoch or
  per-step output;
- thousands of repeated occurrences of one incident produce bounded output and
  retain a suppressed-occurrence count;
- incident deduplication remains memory-bounded under many unique keys and
  kinds, does not let one kind suppress another, and retains no tensors or
  observations;
- `console.log()` renders mappings, paths, enums, and brackets safely;
- redirected output, `NO_COLOR`, fixed-width capture, and no ANSI in a
  non-TTY;
- handled and import-installed uncaught tracebacks, with locals hidden;
- expected recoverable errors do not print a traceback;
- no raw secret/request value is added by RelFlow-owned diagnostic context or
  serving error handling, and traceback locals remain hidden;
- Uvicorn and Lightning output is not duplicated;
- unnamed/nested/narrow/overlapping object display cases;
- no tensor device copy or synchronization during default rendering;
- MIME/HTML escaping and Marimo compatibility;
- Marimo static extraction prefers a Rich HTML MIME member while preserving its
  plain-text fallback;
- representative generated Quarto pages contain the expected live Rich object
  content, no ANSI escapes or object memory addresses, and no quoted multiline
  fallback such as `Request('...')`;
- documentation source does not use built-in `print()` where a live RelFlow
  object representation is intended;
- documentation output remains legible under both light and dark theme CSS
  without selectors coupled to Rich's generated hex colors;
- rendering does not mutate serialization or runtime state;
- no Loguru imports remain in source/tests and no Loguru requirement remains in
  package metadata.

The existing Loguru-specific sink and `.bind()` tests should be deleted or
rewritten around observable diagnostic behavior, not recreated as a new event
schema.

## Documentation

Document behavior without advertising a RelFlow logging API:

- stderr versus stdout;
- the default incident-only policy and the scope of any internally enabled
  enhanced diagnostics;
- why repeated progress belongs in Lightning metrics/progress rather than
  console logs;
- `show_locals=False` and the privacy risk of enabling locals;
- the shared import-installed traceback default across `Deployment.serve()` and
  externally hosted `Deployment.app()`;
- the difference between Rich diagnostics and Lightning metric loggers;
- direct Rich/IPython pretty-printing for arbitrary Python containers;
- direct Rich configuration for applications that want their own console or
  intentionally need to replace the process-wide traceback presentation;
- the lack of a machine/JSON logging contract.

The object-rendering sections use executable Marimo cells and bare final
expressions so the published Quarto pages contain Rich HTML, not merely source
code or plain `print()` output. Application-owned Rich examples that execute
are captured locally; a docs build must not emit them as build-process stderr.

Generated documentation is updated through `make render`, not by editing
`docs/site` directly.

## Delivery Plan

1. Add the internal shared console/theme, verbosity state, and bounded
   incident tracking, with import/output tests and no public exports.
2. Triage current Loguru callsites: delete noise, convert useful diagnostics to
   `console.log()` or `warnings.warn()`, and remove Loguru.
3. Add handled Rich traceback behavior at owned boundaries and one safe,
   import-installed hook for uncaught exceptions.
4. Repair P0 object rendering and shared rendering helpers.
5. Add P1 summaries where they fall naturally out of the shared renderer.
6. Convert representative Quarto cells to live MIME display, make the static
   HTML styling theme-safe, and add focused source/generated-site assertions.
7. Regenerate dependencies, render docs, build the wheel, and run the full
   suite.

## Validation

```bash
uv lock --check
uv sync --locked
uv run ruff check src tests
uv run ty check src/relflow --output-format concise
uv run pytest tests/logging tests/structs/test_rich_display.py
uv run pytest
uv build
make render
```

## Acceptance Criteria

- Loguru is absent from code and built dependencies.
- RelFlow uses one internal themed console, exposes no diagnostics
  configuration API, and does not recreate a logging framework around it.
- Only concise, actionable diagnostics remain; processor and construction noise
  is gone.
- default and internally enhanced output remain bounded as epochs and steps
  grow; repeated incidents are deduplicated or summarized, and recurring
  telemetry stays in Lightning metrics/progress.
- stdout stays clean and redirected output is plain.
- importing `relflow` installs the bounded, locals-hidden Rich traceback hook
  once without emitting output or exposing a configuration API.
- unexpected failures have one bounded Rich traceback with locals hidden;
  expected failures remain concise and HTTP clients never receive stacks.
- Lightning metrics, Uvicorn ownership, Python warnings, and Rich diagnostics
  remain distinct.
- core RelFlow objects render correctly both directly and inside basic Python
  containers without hidden data/device work.
- the rendered Quarto site shows those production Rich representations from
  executable cells, including nested/basic-object display, with working light
  and dark themes and a plain-text MIME fallback.
- rendering changes no model, serialization, or runtime state.
- focused/full tests, lint, types, lock validation, wheel build, and docs render
  pass.

## References

- [Rich Console logging](https://rich.readthedocs.io/en/stable/console.html#logging)
- [Rich Console API](https://rich.readthedocs.io/en/stable/reference/console.html)
- [Rich tracebacks](https://rich.readthedocs.io/en/stable/traceback.html)
- [Rich pretty printing and repr protocol](https://rich.readthedocs.io/en/stable/pretty.html)

## Definition Of Done

RelFlow has a small, idiomatic Rich presentation layer: one console, a handful
of useful logs, safe default tracebacks with locals hidden, and objects that
are pleasant to inspect. There is no Loguru dependency and no replacement
logging framework.
