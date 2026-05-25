# Contributing

JSON2Vec is a schema-driven model factory with extension points for datatypes,
preprocessors, data readers, and serving. Contributions should keep the core
graph generic and move domain-specific behavior to the type or plugin that owns
it.

## Development Setup

Use Python 3.12 or newer.

```bash
uv sync
uv run pytest
uv run ruff check
```

For focused changes, run the smallest relevant test subset first, then run the
full suite before publishing a branch.

## Type-Centered Domain Modeling

Prefer type-centered domain modeling.

If logic exists to normalize, construct, expand, validate, interpret, or
dispatch behavior for one domain concept, put that behavior on the owning type.
Orchestration code should compose domain objects, not reimplement their rules.

Use:

- `classmethod` for named construction, normalization, expansion, or parsing.
- instance methods for behavior that depends on object state.
- `@property` for derived readable state.
- `enum.StrEnum` or `enum.IntEnum` when a value set is finite and meaningful.
- enum methods when behavior branches by enum value.
- Pydantic validators and fields for model boundary validation.
- `beartype` for runtime validation at public or cross-boundary functions.
- module-level helpers only for standalone algorithms or behavior that truly
  spans multiple owning types.

Avoid primitive obsession. Prefer enums, Pydantic models, request classes, and
small value-bearing types over repeated string/dict checks.

## Extension Boundaries

Tensorfield behavior belongs in tensorfield plugins.

Define tensorfield extensions with `json2vec.tensorfields.base.Plugin` and
register the extension components with `@plugin.register`:

- `Request`: schema/request model for the datatype.
- `TensorField`: tensorized values and trainable target state.
- `Embedder`: datatype-specific input embedding.
- `Decoder`: datatype-specific output head.
- `loss`: datatype-specific training objective and metrics.
- `write`: optional prediction serialization.
- `plot`: optional diagnostic rendering.

The core architecture should reach tensorfield behavior through `TENSORFIELDS`
and the plugin contract. Do not add imports or branches in core architecture for
a specific tensorfield type.

Technical requirements for a tensorfield also belong to the extension layer:

- Register Lightning callbacks with `plugin.callback(...)`.
- Put shared extension utilities under `json2vec.tensorfields.shared`.
- Expose worker/process encoding state through an embedder
  `interprocess_encoding_context` property when needed.
- Keep distributed synchronization, vocabulary updates, counters, cache
  flushing, and other extension-specific runtime concerns behind registered
  callbacks or extension-owned objects.

Core model code may aggregate plugin callbacks and pass interprocess encoding
context generically. It should not know that a category field needs a
vocabulary, that a numeric field needs a counter, or that a future media field
needs a loader.

## Plugin Contract

New tensorfield plugins must preserve the existing component contracts:

- Plugin names use lowercase letters, numbers, and underscores.
- `Request` subclasses the schema node/request base used by tensorfields.
- `TensorField` subclasses `TensorFieldBase` and implements `new`, `empty`,
  `mask`, and `target`.
- `Embedder` subclasses `EmbedderBase` and accepts `hyperparameters` and
  `address`.
- `Decoder` subclasses `DecoderBase` and accepts `hyperparameters` and
  `address`.
- `loss` accepts `module`, `prediction`, `batch`, and `strata`.
- `write` accepts `module` and `prediction`.
- `plot` accepts `module`, `address`, `branch`, and `detail`.

Optional dependencies should be imported inside the extension path that needs
them and should fail with a clear message. Do not make core imports depend on an
optional plugin dependency.

## Preprocessors And Data

Dataset-specific transformation belongs in preprocessors, not in data loaders or
the model graph. Register reusable preprocessors with `@preprocess`; keep
preprocessor inputs and outputs as JSON-like dict objects.

The data pipeline may fetch, shard, batch, shuffle, sample, preprocess, and call
generic encoding. It should not contain datatype-specific logic. Encoding should
dispatch through schema requests and tensorfield plugins.

## Runtime State And Distributed Training

Do not retain autograd graphs in long-lived state. Metric logging should detach
tensors before passing them to logging systems, and callbacks should avoid
storing tensors from the active graph unless they explicitly own detached state.

Callbacks must be safe to attach through `Model.configure_callbacks()`. Prefer
idempotent callbacks and extension registration over manual callback wiring in
training scripts.

Model schema mutation should go through `Model.update(...)`,
`Model.override(...)`, or deployment `update(...)` operations so runtime modules
stay synchronized with hyperparameters and mutation locks are respected.

## Public API

Expose stable user-facing types from `json2vec.__init__` only when they are part
of the supported API. Internal helpers should remain close to the subsystem that
owns them.

Prefer path-like APIs that accept `str | pathlib.Path` at boundaries and
normalize to `Path` internally.

## Tests

Add focused tests for every behavioral change.

Use existing test locations:

- `tests/tensorfields/` for plugin and tensorfield contracts.
- `tests/architecture/` for model graph, callback aggregation, and runtime
  behavior.
- `tests/data/` for dataset and iterable behavior.
- `tests/preprocessors/` for preprocessor registration and output behavior.
- `tests/inference/` for deployment and prediction serving behavior.
- `tests/structs/` for schema, enums, and validation behavior.

When adding a new extension, include tests for registration, encoding, loss,
write/plot defaults or custom behavior, callback registration, and any
distributed/shared-state behavior it introduces.

## Repository Hygiene

Keep changes scoped to the requested behavior. Avoid unrelated refactors in the
same branch.

Specs and exploratory planning notes are local working artifacts unless they are
explicitly intended to become durable documentation. Durable contributor-facing
rules belong in this file or in `docs/`.
