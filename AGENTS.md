# RelFlow Agent Guide

This file is the high-signal context for AI coding agents working in this repo. Prefer it over rediscovering package conventions from scattered tests.

## Package Mental Model

RelFlow builds PyTorch/Lightning models from JSON-like schemas. Users normally import the package as:

```python
import relflow as rf
```

The public surface should stay usable from `relflow` directly: `Model`, `Branch`, built-in tensorfields, data modules, preprocessors, inference writers, deployment helpers, mutation predicates, and extension base classes.

The schema is the architecture. `Model(...)` receives field constructors and branch nodes, then builds the root branch, tensorfield embedders, context encoders, decoders, losses, and prediction outputs.

## Common API Patterns

Minimal supervised model:

```python
model = rf.Model(
    d_model=64,
    n_layers=2,
    n_heads=4,
    amount=rf.Number,
    merchant=rf.Category(size=4096),
    label=rf.Category(target=True, size=2),
)
```

Nested repeated context:

```python
model = rf.Model(
    d_model=64,
    n_layers=2,
    n_heads=4,
    line_items=rf.Branch(
        length=32,
        sku=rf.Category(size=2048),
        quantity=rf.Number,
    ),
    returned=rf.Category(target=True, size=2),
)
```

Root branch naming is passed to `Model(...)` with `name=...`. The
generated root branch is always a singleton; use child `Branch(length=...)`
for repeated data.

```python
model = rf.Model(
    name="event",
    d_model=32,
    n_layers=1,
    n_heads=4,
    embed=True,
    amount=rf.Number,
)
```

## Gotchas

- Do not use a public `Struct(...)` constructor. Public examples should use `Model(...)` and `Branch(...)`.
- `Model(..., name="customer")` names the generated root branch. Older examples may say `root=...`; update them.
- Processed observation names and nesting match the schema by default. A node
  may opt into RelFlow's node-relative structural query syntax with paths such
  as `query="source.path"` or `query="items[-32:][*].sku"`; RelFlow never
  infers queries. Filters, joins, sorting, and derived values belong in an
  `rf.Preprocessor`.
- `Branch(name="transactions")` reads the same-named child collection, and its
  leaves read keys such as `amount` from each child mapping.
- `Branch(overflow="head")` is the default. Use `overflow="tail"` for recency-ordered histories and `overflow="error"` for strict schemas. The generated root branch uses internal `Overflow.error`.
- `target=True` is shorthand for `p_prune=1.0`; the field is hidden from input and decoded as a supervised target.
- `embed=True` emits an embedding in prediction output. It does not make the field a supervised target.
- `Hash` represents large identifiers with batch-salted hashes, preserving equality across fields in one encoded batch without learning a persistent vocabulary.
- `DateParts` is for calendar parts. If elapsed time or recency matters, derive a `Number`.
- Tensorfield plugins declare canonical Arrow atom compatibility families with
  `Plugin(types=...)`; custom Python atoms add plugin-owned physical matchers
  with `Plugin(..., arrow={Type: matcher})`. Keep datatype contracts out of
  `Request` and the shared ragged engine.
- Preprocessors accept and return identity-bearing `rf.Batch` objects before
  query/coalescing. Use them for source renaming, Arrow compute, windowing,
  joins, normalization, or explicit row expansion/grouping. Awkward is an
  optional transient view for nested transforms; persisted pipeline values and
  `RaggedField` members remain Arrow-backed.
- Postprocessors accept and return same-row, same-identity `rf.Batch` objects
  after prediction writing. Use them to reshape Arrow output for APIs or
  warehouses.

## Data And Training

Use `ArrowDataModule(...)` for canonical examples. Keep examples tiny:

```python
import pyarrow as pa

table = pa.Table.from_pylist(records)
datamodule = rf.ArrowDataModule(
    model=model,
    train=table,
    validate=table,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)
```

For quick examples, train with `max_epochs=1`, `limit_train_batches=1`, and `limit_val_batches=1`.

`model.predict(...)` returns a `pyarrow.Table`. It accepts Arrow inputs directly
and retains a nonempty sequence of mappings as a small interactive convenience;
pass a typed empty Arrow object when there are no rows. Canonical predictions
are stored under the table's `"predictions"` column.

## Inference

Top-level inference exports:

- `rf.Writer` writes batch prediction output.
- `rf.Postprocessor` is the postprocess callable type.
- `rf.Deployment`, `rf.Accelerator`, `rf.JSONBackend`, and related serving types are lazy exports that require `relflow[serving]`.

## Code Style

RelFlow code should read as a short sequence of domain operations.

- Prefer the shortest precise noun or verb, such as `Batch`, `Plan`, `compile`,
  `bind`, `query`, `coalesce`, and `write`. Use a longer name when one word
  would hide a distinction that matters. Do not repeat the surrounding module
  or class name in an identifier.
- Never prefix a function or class name with `_`; control exposure with
  explicit `__all__` and root exports. Python protocol methods such as
  `__post_init__` are the exception.
- Inline a one-use function when it only forwards a call, renames arguments,
  or hides a few incidental expressions. A named function should own a
  semantic phase, recursion, a reusable algorithm, callback identity, or an
  independently meaningful invariant.
- Prefer plain module functions for transformations. Add a class only when
  state and invariants travel together or a framework protocol requires it.
  Avoid `Manager`, `Helper`, and `Service` objects that merely relay calls.
- Keep control flow linear with validation and guard clauses followed by the
  main path. Avoid adapter chains, dispatcher pyramids, speculative
  abstractions, and parallel ways to perform the same operation.
- Comments and docstrings explain contracts, invariants, or non-obvious
  reasons. They do not narrate the syntax immediately below them.
- Error messages identify the affected domain object or address, the expected
  invariant, the actual value or type, and a useful remedy when one exists.
  Translate only exceptions whose meaning is understood, and preserve their
  cause.
- Do not preserve aliases, shims, or old and new execution paths unless
  compatibility is an explicit requirement. For an intentional breaking
  change, remove obsolete code, exports, dependencies, tests, and docs
  together.

Simplicity comes before cleverness. Vectorize work over the value axis when it
removes repeated Python traversal. A clear loop over schema nodes, addresses,
chunks, or a fixed number of tensor axes is acceptable; a recursive loop over
thousands of values is a signal to use Arrow, Awkward, NumPy, or Torch.

## Ownership Boundaries

- Keep one canonical representation through a subsystem. Arrow is the CPU data
  plane; Polars and Python values are ingress adapters, Awkward is a transient
  nested-operation view, Torch owns model computation, and Python objects
  reappear only at an extension-local library boundary that requires them or
  an explicit application/JSON boundary.
- Keep the batch dimension and `Batch` identity explicit through every row
  selection, expansion, grouping, shuffle, and postprocessing operation.
- Shared data code may understand Arrow containers, validity, offsets, shape,
  and lineage. It must not know a built-in tensorfield name, configuration
  attribute, or value interpretation.
- A tensorfield plugin owns its accepted Arrow families, semantic validation,
  tensorization, embedding, decoding, loss, output schema, writing, callbacks,
  and datatype-specific runtime resources.
- Reach optional behavior through a registered component or an explicit
  protocol. Do not infer a capability from a class name, string name, concrete
  extension import, or magic callable parameter.
- Registration must remain late-extensible. Avoid import-time snapshots of the
  registry in validation or serialization contracts.
- If adding a datatype requires a branch in shared architecture or data code,
  the extension boundary is incomplete.

## Review Heuristics

Before considering a change complete, ask:

- Can a third-party datatype use this path without editing shared data or
  architecture modules?
- Is each new helper a real phase or invariant, or just a one-use forwarding
  layer?
- Does any hot path materialize Arrow values as Python rows only to convert
  them back?
- Are there two representations, entry points, or compatibility paths where
  one would suffice?
- Are invariants enforced once at the boundary and then trusted internally?
- Do tests cover the architectural boundary as well as the built-in examples?

## Useful Commands

```bash
uv run ruff format --check
uv run ruff check
uv run ty check src/relflow --output-format concise
uv run pytest
uv run pytest tests/test_public_api.py
make render
```

## Documentation Entry Points

- `docs/index.qmd`
- `docs/getting-started.qmd`
- `docs/ai-quickstart.qmd`
- `docs/core-concepts/querypaths.qmd`
- `docs/core-concepts/data-types.qmd`

When adding docs, prefer runnable inline Python snippets and current public
imports. Keep Quarto pages self-contained; do not depend on external standalone
scripts.
