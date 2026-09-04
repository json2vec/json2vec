# Contributing

RelFlow is a schema-driven model factory with extension points for datatypes,
preprocessors, data sources, and serving. Contributions should make the main
path easier to read, keep one canonical representation per subsystem, and put
specialized behavior beside the concept that owns it.

## Development Setup

Use Python 3.12 or newer.

```bash
uv sync
uv run ruff format --check
uv run ruff check
uv run ty check src/relflow --output-format concise
uv run pytest
```

Run the smallest relevant test subset while developing. Before publishing a
branch, run the complete checks above and `make render` when documentation or a
public contract changed.

## Implementation Style

RelFlow code should read as a short sequence of domain operations. Prefer
direct composition over scaffolding.

### Names And Code Shape

- Prefer the shortest precise noun or verb. Existing vocabulary includes
  `Batch`, `Plan`, `Projection`, `compile`, `bind`, `query`, `scan`, `merge`,
  `regularize`, and `coalesce`.
- Use a longer name when one word would conceal an important distinction. Do
  not repeat the surrounding module or class name in every identifier.
- Never prefix a function or class name with `_`. Python protocol methods such
  as `__post_init__` are the exception. Control the public surface with
  explicit `__all__` and package exports.
- Inline a one-use function when it only forwards a call, renames arguments,
  or hides a few incidental expressions. A function earns a name when it owns
  a semantic phase, recursion, a reusable algorithm, callback identity, or an
  independently meaningful invariant.
- Prefer plain module functions for transformations. Create a class when state
  and invariants travel together or a framework protocol requires it—not to
  hide arguments behind a `Manager`, `Helper`, or `Service`.
- Keep control flow linear. Validate the boundary, use guard clauses for
  exceptional paths, and let the main transformation remain visible.
- Do not add abstractions in anticipation of possible reuse. Extract them when
  the shared concept is real and its contract can be named precisely.

Comments and docstrings explain contracts, invariants, or non-obvious reasons.
They should not narrate the syntax below them. Error messages should identify
the affected object or address, the expected invariant, the actual value or
type, and a useful remedy when one exists. Translate only exceptions whose
meaning is understood, and preserve their cause.

### Types And Invariants

Create a value type when data and invariants need to travel together. Small
immutable dataclasses with `frozen=True` and `slots=True` are a good fit for
compiled plans, projections, and other trusted intermediate values.

Keep transformation algorithms as plain functions unless their behavior
genuinely depends on instance or class state. Use enums when a finite value set
has shared meaning, Pydantic at schema boundaries, and explicit validation at
runtime boundaries. After construction or validation, downstream code should
be able to trust the object instead of repeating defensive checks.

## Canonical Data Flow

Arrow is RelFlow's canonical CPU data plane. Convert supported Python and
Polars inputs once at ingress. Keep datasets, preprocessors, queries,
coalescing, prediction output, postprocessors, and writers Arrow-backed.
Awkward is a transient view for nested transforms; Torch owns model
computation. Materialize Python objects only inside an extension-local library
boundary that requires them or at an explicit application/JSON boundary.

Prefer whole-array Arrow, Awkward, NumPy, or Torch operations over Python loops
across observations or leaf values. A clear loop over schema nodes, addresses,
Arrow chunks, or a fixed number of tensor axes is appropriate when those are
the real units of orchestration.

Every row-changing operation must preserve or deliberately derive `rf.Batch`
identity. Use `Batch.slice`, `take`, `order`, `filter`, `expand`, `explode`, and
`group` rather than separating payload rows from their lineage. Use `replace`
only for a same-row payload transformation. Sampling and shuffling must remain
deterministic from identity, seed, stratum, and epoch—not from source chunk
boundaries.

Keep one representation and one execution path wherever possible. Avoid
round-tripping Arrow through Python rows, maintaining parallel Arrow and
Awkward models, or adding several adapters for the same boundary.

## Ownership And Extension Boundaries

Shared orchestration composes contracts. It must not reproduce the rules of a
particular datatype.

- Shared data code may understand Arrow containers, validity, offsets, shape,
  branch overflow, and lineage.
- A tensorfield plugin owns accepted Arrow families, semantic validation,
  tensorization, embedding, decoding, loss, output schema, writing, callbacks,
  and datatype-specific runtime resources.
- Reach tensorfield behavior through `TENSORFIELDS` and registered components.
  Do not import a concrete extension or branch on names such as `"category"`
  or `"number"` in shared architecture or data code.
- Do not inspect a callable for a magic parameter to infer a datatype
  capability. Add a small explicit plugin contract when orchestration must
  provide optional lifecycle state.
- Registration must remain late-extensible. Validation and serialization must
  not freeze a snapshot of whichever plugins happened to be imported first.
- Adding a third-party datatype should not require editing shared data or
  architecture modules.

Optional dependencies belong in the extension path that needs them and should
fail with a clear installation message. Do not make core imports depend on an
optional plugin dependency.

## Tensorfield Contract

Define a tensorfield extension with `rf.Plugin`:

```python
import relflow as rf

plugin = rf.Plugin(
    name="example",
    types=(int | float,),
)
```

Plugin names use lowercase letters, numbers, and underscores. `Request`,
`TensorField`, `Embedder`, and `Decoder` subclass their corresponding RelFlow
base classes.

`types` declares compatible canonical Arrow atom families. A custom Python
atom supplies its physical Arrow matcher with
`Plugin(..., arrow={Type: matcher})`. Keep these datatype contracts out of
`Request`, `RaggedField`, and the shared query/coalescing engine.

Register the components owned by the extension:

- `Request`: the Pydantic schema options for the datatype.
- `TensorField`: `new(field, address, schema, strata)`,
  `empty(batch_size, address, schema)`, masking, targeting, and target state.
- `Embedder`: model input construction from `schema` and `address`.
- `Decoder`: model output construction from `schema` and `address`.
- `loss(module, prediction, batch, strata)`: the training objective and
  datatype metrics.
- `output(module, address)`: the stable Arrow prediction coordinate type, or
  `None` when the plugin has no decoded output.
- `write(module, prediction, datatype)`: a `pa.StructArray` matching the
  declared output type, or `None`.

Register Lightning callback factories with `plugin.callback(...)`. Put
utilities shared by several extensions under `relflow.tensorfields.shared`,
but keep extension-specific vocabulary, counters, synchronization, cache
flushing, and other lifecycle behavior behind plugin-owned objects.

## Preprocessors And Postprocessors

Preprocessors accept identity-bearing `rf.Batch` objects and return `Batch`, an
iterable of `Batch`, or `None`. Use them for Arrow-native renaming, derivation,
normalization, windowing, joins, or deliberate row expansion/grouping. Declare
required and produced columns when the processor contract depends on them.

Partition preprocessors must be safe on each source partition independently.
Use dataset scope only when the operation truly needs the materialized split.
Datatype encoding and model semantics still belong in tensorfield plugins.

Postprocessors accept and return one `rf.Batch`. They may reshape prediction
columns for an application or warehouse, but must preserve row count and exact
batch identity. JSON conversion belongs after postprocessing at the serving
boundary.

## Runtime State

Do not retain autograd graphs in long-lived state. Detach tensors before
passing scalar values to logging systems. Callbacks should not store tensors
from the active graph unless they explicitly own detached state.

Callbacks attached through `Model.configure_callbacks()` must be idempotent
and safe in distributed execution. Prefer extension registration over manual
callback wiring in training scripts.

Model schema mutation should go through `Model.update(...)`,
`Model.override(...)`, or deployment `update(...)` so runtime modules remain
synchronized with the schema and mutation locks.

## Public API And Documentation

Users normally write `import relflow as rf`. Expose supported user-facing
types from `relflow.__init__`; keep implementation helpers close to their
owning subsystem and control module exposure explicitly.

Path-like public boundaries should accept `str | pathlib.Path` and normalize
to `Path` once inside the boundary.

User-facing plumbing carries a documentation cost. When a change introduces a
new callable contract, Arrow shape, lifecycle hook, or failure mode, document:

- the smallest complete example;
- accepted inputs and exact outputs;
- where conversion and validation occur;
- identity, batching, and distributed behavior when relevant;
- one realistic nested example; and
- actionable errors and migration notes for a breaking change.

Prefer runnable inline examples using the current top-level API. Keep Quarto
pages self-contained and update code, tests, API references, and narrative docs
in the same change.

## Tests

Add focused tests for every behavior and invariant. Test the public contract,
not private implementation choreography.

Use the existing locations:

- `tests/tensorfields/` for plugin and tensorfield contracts.
- `tests/architecture/` for graph construction, runtime contracts, and
  callback aggregation.
- `tests/data/` for Arrow identity, queries, coalescing, datasets, sampling,
  and shuffling.
- `tests/preprocessors/` for processor registration, callable binding, and
  output contracts.
- `tests/inference/` for prediction writing and serving.
- `tests/structs/` for schema and validation behavior.

For an extension-boundary change, include a minimal third-party plugin test.
Prove that it can register, validate data, encode, and round-trip its schema
without adding its name or attributes to shared modules. Architectural tests
should guard against concrete extension imports and built-in-name dispatch,
not merely exercise every current built-in.

## Repository Hygiene

Keep changes scoped to the requested behavior. Do not combine an unrelated
refactor with a feature or fix.

Compatibility is a product requirement, not an automatic reason to preserve
two designs. When compatibility is required, isolate and test the adapter. For
an intentional breaking change, delete superseded paths, aliases, exports,
dependencies, tests, and docs rather than leaving both systems in place.

Specs and exploratory notes are working artifacts unless they are deliberately
accepted as durable documentation. Lasting contributor rules belong here or in
`AGENTS.md`; user-facing behavior belongs in `docs/`.
