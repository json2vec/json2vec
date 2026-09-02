# Awkward Array Preprocessing Design

- Status: Phase 1 implemented; Awkward-backed eager coalescing is the canonical
  tensorization path and `RaggedField` is the tensorfield extension boundary.
  Direct Arrow/Polars adapters and profile-driven fused kernels remain Phase 2
  work.
- Date: 2026-08-29
- Audit baseline: `e7cf72d`
- Priority: Simplification and standardization first, speed second
- Related design: [Unified Masking And Pruning](unified-mask-design-spec.md)

## Decision

Do not replace individual Python loops with isolated Awkward calls.

Adopt Awkward as RelFlow's one canonical CPU-side nested-array engine and a core
dependency. User preprocessing always produces a batch dimension, batch size is
tunable to available memory, and the expected nested payload contains thousands
of values. Under that workload, retaining a Python recursive engine as the
primary path optimizes for the wrong scale.

Standardize built-in tensorization around one internal `RaggedField` contract
implemented with Awkward. Do not ship an optional “fast path” next to a Python
engine: maintaining two permanent correctness paths would work against the
primary goal. Built-in and third-party tensorfields adopt the same new contract
in one change; there is no Python-values adapter.

The canonical design is one eager coalescing operation per processed batch. It
prepares every address selected for encoding before any datatype codec runs and
returns one `RaggedField` per encoded address. This avoids repeatedly
materializing and walking the same nested records and can reuse Arrow/Polars
buffers.

Direct schema-address projection is the default. A leaf may explicitly opt
into JMESPath with `query=...` when declarative source extraction is valuable.
No JMESPath expression is inferred for an omitted query. The explicit path
pays one extraction/materialization pass per queried leaf, then converges with
direct projection before the shared paired value/state regularization. It is
not a second padding, state, literal, overflow, or datatype engine.

Keep masking and model routing in PyTorch after tensorization. Most masking is
already tensorized; introducing Awkward there would add a backend transition
without simplifying the model contract.

## Goals

- Replace RelFlow's custom recursive list algebra with one maintained nested
  array abstraction where that makes the code smaller.
- Give every built-in tensorfield the same structure/state preparation path.
- Separate common structure handling from small datatype-specific leaf codecs.
- Dispatch raw atom preparation through each plugin's `types` declaration,
  without teaching the ragged core any concrete tensorfield type names.
- Define exact distinctions among missing paths, explicit null, padding,
  prediction masks, and structured leaf values.
- Avoid Python per-value loops where an array operation naturally expresses the
  same work.
- Preserve deliberate loops for stateful or plugin-owned behavior.
- Improve throughput for deep, wide, or Arrow-backed batches without regressing
  ordinary small batches unacceptably.

## Non-Goals

- Eliminating every `for` loop.
- Moving schema traversal or plugin dispatch into Awkward.
- Moving GPU tensor operations out of PyTorch.
- Replacing arbitrary user preprocessors with an array DSL.
- Reimplementing stateful online vocabulary semantics as a vectorized operation.
- Supporting two permanent tensorization backends.
- Treating initial synthetic benchmark numbers as production claims.

## Pre-Adoption Pipeline

At the audit baseline, the core path was approximately:

```text
Polars / Arrow / Python source
  -> Python dict per observation
  -> user preprocessor
  -> EncodedBatch: list[list[dict]] (batch x singleton generated root)
  -> loop over schema requests
  -> JMESPath/custom extraction into nested Python lists
  -> TensorField.new
       -> recursively locate "<MASK>"
       -> recursively coerce/encode values
       -> recursively clip/pad values
       -> recursively clip/pad literal-mask flags
       -> convert NumPy arrays to Torch
  -> apply Torch mask/prune policies
  -> model
```

The field loop is appropriate: every request is a plugin dispatch with its own
codec and state. The repeated per-value structure traversal inside each request
is the standardization opportunity.

## Repeated Code To Consolidate

The primary custom implementation was in `src/relflow/data/nested.py`:

- `apply` recursively transforms leaf values;
- `extract_mask_literals` recursively cleans and records prediction literals;
- `_iter_leaf_nodes` traverses nested values and applies overflow rules;
- `_write_leaf` separates null from valued state;
- `pad` allocates and fills dense value/state arrays.

All built-in `TensorField.new` implementations then repeated some version of:

```python
values, literal = extract_mask_literals(...)
encoded = apply(values, encode_leaf)
content, state = pad(encoded, ...)
literal, _ = pad(literal, ...)

return TensorField(
    content=torch.tensor(content),
    state=torch.tensor(state).masked_fill(torch.tensor(literal), Tokens.masked.value),
    trainable=torch.zeros(...),
    targets=TensorDict({}),
)
```

The repetition appears in Boolean, Number, Category, Cluster, Set, Vector,
DateParts, Hash, and Text. The exact content codec differs; the structure/state
work does not.

## Complete Loop Inventory

This inventory is exhaustive for runtime loops that traverse observations,
nested values, or converted prediction values at audit baseline `e7cf72d`.
Schema construction, layer iteration, and metric-registry loops that never walk
data values are excluded.

### Replace With Eager Coalescing / RaggedField

| Location | Current traversal | Replacement |
| --- | --- | --- |
| `data/nested.py:16-54` `apply` / `walk` | Recursive list-to-leaf callback | Flat `RaggedField.values`; keep only the datatype callback |
| `data/nested.py:57-73` `contains_mask_literal` | Recursive ndarray/list/tuple/mapping scan, including `tolist()` | Recognize literal state only at modeled `RaggedField` projections; stop scanning unmodeled metadata for the reserved string |
| `data/nested.py:76-138` `extract_mask_literals` / `walk` | Recursive cleaned-tree and parallel-flag construction | Literal state produced during canonical ingestion |
| `data/nested.py:141-188` `_iter_leaf_nodes` | Explicit stack traversal with overflow at each depth | Awkward clipping, offsets, presence, and regularization |
| `data/nested.py:190-202` `_write_leaf` | Per-leaf null/state/content write | Compiled Awkward kernels or one fused Numba placement kernel |
| `data/nested.py:205-235` `pad` | Allocate plus walk/scatter | Internal eager coalescing into `RaggedField` |
| `data/iterables.py:133-179` query compiler | Parses generated selectors and recursively projects ordinary fields | Replace generated/default selectors with direct schema-address projection; retain only cached JMESPath compilation for explicit `query` leaves |
| `data/iterables.py:156-177` direct-query `apply` | Recursive list projection and missing-child compaction | Schema-path projection with coordinate-preserving final state |
| `data/iterables.py:245-246` JMESPath evaluation | Evaluates arbitrary value expressions for every leaf | Run only for explicit `query`; join direct values before shared paired regularization |
| `data/iterables.py:187-208` resolution monitor | Stack walk to detect an empty result | Reduce `RaggedField.state` |
| `tensorfields/shared/vocabulary.py:192-208` `_tokens` | Recursive outer batch/branch flattening | Iterate `RaggedField.values`; retain iteration inside one Set value |
| `data/datasets/polars.py:73-119` | `to_dicts()` / named `iter_rows()` materialization | Arrow/Polars-to-coalescer adapter on eligible no-row-preprocessor paths |
| `data/datasets/streaming.py:151-250` | Arrow `RecordBatch.to_pylist()` | `ak.from_arrow` plus exactly one singleton root axis |

At the audit baseline, every built-in repeated literal extraction and usually
two structural passes (content/state plus literal flags). All nine call-site
groups now consume fields produced by the eager coalescer:

| Tensorfield | Audit-baseline preparation range |
| --- | --- |
| Boolean | `tensorfields/extensions/boolean.py:123-145` |
| Number | `tensorfields/extensions/number.py:124-146` |
| Category | `tensorfields/extensions/category.py:184-215` |
| Cluster | `tensorfields/extensions/cluster.py:228-259` |
| Set | `tensorfields/extensions/set.py:150-181` |
| Vector | `tensorfields/extensions/vector.py:100-130` |
| DateParts | `tensorfields/extensions/dateparts.py:249-276` |
| Hash | `tensorfields/extensions/hashable.py:94-116` |
| Text | `tensorfields/extensions/text.py:156-178` |

### Vectorize In Torch Or NumPy

| Location | Current loop | Replacement |
| --- | --- | --- |
| `tensorfields/base.py:324-331` | One exact-count branch-mask loop per flattened candidate row | Random score + masked `topk` + scatter in Torch |
| `tensorfields/base.py:356-365` | Apply/hide once per configured policy | Keep the small policy loop, union selections, call datatype `hide` once |
| `vector.py:132-137` | Dense object selection, `tolist()`, then `np.stack` | Flat validated matrix plus RaggedField scatter |
| `category.py:499-514` | Recursive top-k candidate packaging | Optional `ak.zip` + `ak.to_list` at the public output boundary |

### Keep: The Loop Carries Semantics

| Location | Why it stays |
| --- | --- |
| Schema-path projection | A tiny loop over address components is clearer than generated code. |
| `data/iterables.py:227-267` encode | One plugin/resource dispatch per schema request |
| `data/iterables.py:38-116`, `processors.py:213-269` | User preprocessing, 0..N outputs, batching, sampling, and shuffle are control flow |
| `datasets/polars.py`, `streaming.py`, `custom.py` source loops | File iteration, sharding, replacement sampling, buffers, and arbitrary iterables are incremental algorithms |
| `boolean.py:47-50` | Exact bool validation/error semantics |
| `category.py:192-193`, `cluster.py:236-237` | Ordered vocabulary mutation, locks, proposals, capacity, and first-seen IDs |
| `set.py:106-127` | Per-label stateful vocabulary lookup; scatter after lookup is array work |
| `vector.py:55-77` | Per-leaf semantic width/type validation |
| `dateparts.py:258-259` | Arbitrary configured `datetime.strptime` pattern |
| `hashable.py:121-135` | MessagePack + BLAKE3 per original scalar |
| `text.py:185-202` | String validation and one already-batched tokenizer call |
| `text.py:319-380` | Memory-bounded Hugging Face encoder microbatches |
| `shared/vocabulary.py:90-178,345-426` | Ordered dedupe, multiprocessing locks/proposals, capacity, and DDP merge |
| `shared/counter.py:44-63` | Already uses Torch flatten/mask/`bincount`; registry loops are per field |
| `data/iterables.py:330-372`, `architecture/runtime.py:58-118` | Per-plugin masking, graph routing, and decoder dispatch; values are already tensors |

### Recursive Loops Outside Input Preparation

These are inventoried so “remove recursion” does not turn into an indiscriminate
rewrite:

| Location | Decision |
| --- | --- |
| `structs/packages.py:23-110` prediction unbatch/serialize/squeeze | Keep for heterogeneous Python API output; an Arrow writer may bypass it |
| `inference/deployment.py:221-246` response splitting | Keep; one response object per request is the API contract |
| `category.py:499-514` candidate records | Awkward candidate noted above, but it is output optimization, not RaggedField |
| `set.py:424-434` threshold dictionary packing | Keep unless the public variable-dictionary output changes |
| `inference/callback.py:34-76` Arrow schema comparison | Keep; recursively compares types, never values |
| `architecture/contracts.py:148-166,448-468` tensor-tree traversal | Keep; periodic/cold validation over tensor leaves, not scalar data |

The practical deletion target is therefore precise: five `data.nested`
primitives, generated/default query extraction, nine repeated tensorfield
preparation groups, eligible Arrow/Polars row materialization, and one Torch
exact-count loop. Explicit opt-in JMESPath evaluation, stateful codecs, and
orchestration remain visible semantic work rather than being forced into
Awkward array algebra.

## Canonical Internal Boundary

Introduce one Awkward-backed field view:

```python
@dataclass(frozen=True)
class RaggedField:
    state: np.ndarray  # int64, dense model geometry
    values: ak.Array  # flat retained valued leaves
    placement: np.ndarray  # one dense destination per value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.state.shape

    def place(
        self,
        encoded: np.ndarray,
        *,
        fill: Any,
        value_shape: tuple[int, ...] = (),
    ) -> np.ndarray: ...
```

The coalescer gives each `RaggedField`:

- the dense `np.int64` RelFlow `state` array, including predict-literal masked
  state and ready for Torch embedding indices;
- every retained valued leaf in stable row-major order after overflow;
- one dense placement index for each retained value.

The invariants are deliberately strict:

- the internal coalescer receives batch × singleton-root leading axes and
  never asks a datatype codec to guess or insert either axis;
- `state` contains final `Tokens` values—`valued`, `null`, `padded`, or
  `masked`—rather than temporary routing codes;
- `values` excludes null, missing, literal-masked, and overflow-clipped leaves;
- `placement` is one-dimensional `np.int64`; `placement[i]` indexes the raveled
  dense destination for `values[i]`;
- plugin-declared raw atom compatibility is checked before Awkward construction;
  overflow validation/clipping then completes before any stateful resource or
  `TensorField.new` codec sees a retained value;
- `place` requires exactly `len(values)` encoded rows, validates the declared
  trailing value shape, scatters them, and leaves every other destination at
  `fill`.

Datatype implementations become leaf codecs. The internal coalescer runs once,
then the field for an address is passed to its codec. For Category:

```python
fields = coalesce(processed_batch, schema=schema, strata=strata)
field = fields[address]

vocabulary.reserve(field.values, learn=strata == Strata.train)
encoded = np.fromiter(
    (vocabulary.encode(value) for value in field.values),
    dtype=np.int64,
    count=len(field.values),
)
state = torch.from_numpy(field.state)  # RaggedField guarantees int64.

return TensorField(
    content=torch.from_numpy(field.place(encoded, fill=0)),
    state=state,
    trainable=torch.zeros(field.shape, dtype=torch.bool),
    regularized=torch.zeros(field.shape, dtype=torch.bool),
    present=state != Tokens.padded.value,
    targets=TensorDict({}),
)
```

Number becomes a dtype coercion plus `place`. Text passes the flat retained
strings to one tokenizer call and places token IDs/masks. Hash retains only its
flat BLAKE3 loop. Set retains only its label encoding/scatter. A clipped value
cannot enter a vocabulary, affect counts, invoke a tokenizer, or reach
`TensorField.new` codec validation because it is outside the configured model
shape. It has already satisfied the plugin's pre-Awkward atom contract needed
for safe coalescing.

`coalesce` is an internal module-level function, not another extension API.
`RaggedField` is the public extension boundary. The old iterator is deleted;
tests assert the semantic truth table rather than treating the old engine as an
oracle.

## Canonical Eager Coalescer

Awkward represents irregular list structure, option values, and record fields
without object-dtype NumPy arrays. Once preprocessing has produced a batch, one
internal call prepares every encoded field atomically:

```python
def coalesce(
    values: EncodedBatch,
    *,
    schema: Schema,
    strata: Strata,
) -> dict[Address, RaggedField]: ...
```

`coalesce` is deliberately internal. Tensorfield extensions receive the
resulting `RaggedField`; they do not construct, project, or regularize an
intermediate batch carrier.

For each address selected for encoding, the coalescer:

1. projects the same-named schema path or evaluates its explicit JMESPath;
2. applies the owning plugin's raw atom compatibility contract;
3. pairs the ragged values with a state sidecar containing final `Tokens`;
4. clips and pads the value/state pair together across declared branch axes;
5. emits flat retained values, dense final state, and placement as one
   `RaggedField`.

Direct and queried values therefore converge before regularization. JMESPath
remains an opt-in extraction step, but it cannot fork overflow, state, literal,
or placement semantics. Every field is successfully prepared before the first
`TensorField.new(...)` codec runs, so a structural/type failure cannot leave a
partially updated vocabulary, counter, tokenizer, or other field-owned state.

The sidecar uses `Tokens.valued`, `Tokens.null`, `Tokens.padded`, and
`Tokens.masked` directly. There is no intermediate `0/1/2` routing vocabulary.
Pairing values and final state before clipping/padding keeps their coordinates
aligned and prevents the two halves from applying overflow independently.

Projection preserves branch coordinates. For example,
`[{}, {"f": None}, {"f": 1}]` produces `[P, N, V]`; it never compacts to
`[N, V]`. Missing or null containers have zero children. Plugin atom checks and
all structural preparation complete before a tensorfield reserves, counts,
tokenizes, semantically validates, or encodes retained leaves.

The small schema-path loop is desirable. It replaces nested loops over every
record and event with column selection while keeping dynamic paths readable.
Filters, functions, indexing, slices, and multiselects may be expressed by an
explicit leaf JMESPath `query`; business logic, joins, or coordinated sibling
changes belong in a preprocessor. JMESPath projections may compact missing or
null leaves, so use null-preserving `map(&field, collection)` expressions for
aligned queried siblings or preprocess them together.

Only directly bound modeled values and materialized explicit-query results must
use values that Awkward can represent. UUID, Decimal, custom classes, and other
opaque values need normalization when a field binds to them; the same values
may remain untouched under unmodeled metadata keys. Any coalescing failure is
reported before datatype codecs begin, regardless of whether the field uses
direct projection or an explicit query.

Prediction metadata remains the original processed observation, stored
separately from modeled Awkward data. Unmodeled metadata is neither traversed
nor coerced during tensorization.

`ak.pad_none(..., clip=True)` turns a ragged axis into a fixed-width axis, and a
filled regular array can be converted to NumPy/Torch. See the official
[padding guide](https://awkward-array.org/doc/main/user-guide/how-to-restructure-pad.html).

## Columnar Source Adapters

The largest expected speed opportunity is avoiding conversion from Arrow to
Python rows and immediately back into arrays:

```python
# Current streaming path
records = record_batch.to_pylist()

# Proposed no-preprocessor/columnar path
records = ak.from_arrow(record_batch)
```

Awkward documents Arrow conversion as usually zero-copy, though this is not a
universal guarantee. Prefer record batches over repeatedly concatenating Arrow
chunks. See the official
[Arrow conversion guide](https://awkward-array.org/doc/main/user-guide/how-to-convert-arrow.html).

Observation preprocessors and explicit JMESPath queries receive Python objects,
so those source paths materialize rows before paired ragged regularization. A
no-preprocessor, direct-binding Arrow path can remain columnar. Prediction
metadata is materialized separately only at the public output boundary.

## Before And After: Padding And State

### Pre-Adoption Shape Traversal

The audit-baseline implementation allocated dense buffers, then visited each nested
leaf and writes its content/state:

```python
values = np.full(shape, pad_value, dtype=dtype)
state = np.full(shape, Tokens.padded.value, dtype=np.int8)

for index, value in _iter_leaf_nodes(nested, shape, overflows, address):
    if value is None:
        state[index] = Tokens.null.value
    else:
        values[index] = encode(value) if encode else value
        state[index] = Tokens.valued.value
```

This is correct and general, but every value crosses the Python interpreter.

### Awkward Structural Pass

This executable sketch regularizes two ragged axes while preserving explicit
null versus inserted padding:

```python
import awkward as ak
import numpy as np

from relflow.structs.enums import Tokens

values = ak.Array(
    [
        [[1.0, None, 3.0], [4.0]],
        [[5.0, 6.0]],
    ]
)

# Existing slot indices remain present even when the corresponding value is
# explicitly null. Newly padded slots have slot=None.
slots = ak.local_index(values, axis=2)


def fixed(array):
    array = ak.pad_none(array, 3, axis=1, clip=True)
    array = ak.fill_none(array, [], axis=1)
    return ak.pad_none(array, 4, axis=2, clip=True)


padded_values = fixed(values)
padded_slots = fixed(slots)

padding = ak.to_numpy(ak.is_none(padded_slots, axis=2))
null = ak.to_numpy(ak.is_none(padded_values, axis=2)) & ~padding
content = ak.to_numpy(ak.fill_none(padded_values, 0.0, axis=2))

state = np.full(content.shape, Tokens.valued.value, dtype=np.int64)
state[null] = Tokens.null.value
state[padding] = Tokens.padded.value
```

Result:

```text
content[0] = [[1, 0, 3, 0], [4, 0, 0, 0], [0, 0, 0, 0]]
state[0]   = [[V, N, V, P], [V, P, P, P], [P, P, P, P]]
```

The production helper must generalize this across shape rank, head/tail/error
overflow, missing parents, and structured leaf boundaries. The example is a
proof of the state technique, not the implementation itself.

The existing low-level `pad` helper uses compact `int8` state before built-ins
cast it for Torch. `RaggedField` instead standardizes its public boundary on
`np.int64`, so `torch.from_numpy(field.state)` is a valid `nn.Embedding` index
without another datatype-specific cast.

## Before And After: Exact Branch Counts

Awkward should not be introduced into masking merely to remove one Python loop.
The current exact-count branch selector iterates each flattened candidate row.
A PyTorch top-k scatter expresses it directly:

```python
# candidates: [..., branch_length] boolean tensor
count = int(mask.count)  # validated as non-null for a count policy
k = min(count, candidates.shape[-1])
scores = torch.rand(candidates.shape, device=candidates.device)
scores.masked_fill_(~candidates, 2.0)

indices = scores.topk(k, dim=-1, largest=False).indices
valid = candidates.gather(-1, indices)

selected = torch.zeros_like(candidates)
selected.scatter_(-1, indices, valid)
```

This selects exactly `min(count, available)` randomly per row without leaving
Torch or looping over rows. It is uniform in the absence of equal random scores;
finite-precision ties use `topk`'s tie behavior. Rate-based leaf and branch
masking is already tensorized and should remain there.

Under the unified masking/pruning design, policy selections should be unioned
and applied once. Structural `Prune` becomes parcel validity; it is not an
Awkward option value and not a new state token.

## Loops That Should Remain

The goal is to remove accidental per-value traversal, not semantic control
flow.

| Loop | Keep? | Reason |
| --- | ---: | --- |
| Per-field plugin dispatch | Yes | Plugins declare raw atom compatibility; tensorfields own semantic shape, codecs, and resources. |
| Schema/path-component traversal | Yes | Small, explicit, and proportional to schema depth. |
| User preprocessor invocation | Yes | A callable may emit zero, one, or many arbitrary observations. |
| Streaming, sharding, buffering, sampling | Yes | Incremental control-flow algorithms, not array math. |
| Vocabulary reservation/encoding | Yes | First-seen ordering, locks, proposals, and DDP synchronization are stateful. |
| BLAKE3/MessagePack Hash codec | Yes | Cryptographic operation per valued scalar. |
| Set label scatter | Yes | Stateful label lookup; Awkward supplies flat leaves and placement. |
| Vector validation/coercion | Yes | Enforces semantic width/type errors per structured leaf. |
| Datetime parsing | Yes | Arbitrary configured formats are not a simple universal ufunc. |
| Hugging Face tokenization/microbatching | Yes | Already batched external model logic. |
| Tensor masking and routing | Yes, in Torch | Already on the tensor backend used by the model. |
| Cold validation/diagnostic traversals | Usually | Clarity matters more than optimizing rare error paths. |

## Canonical Edge Semantics

### Prediction Literal Scope

`"<MASK>"` is interpreted and validated only at modeled fields and their
declared structured-leaf boundary. The coalescer does not recursively scan
unmodeled metadata for the reserved string. This removes a full raw-record
traversal and makes the wire contract follow what the model actually consumes.

### Missing Key Versus Explicit Null

Missing and explicit-null coordinates remain aligned but receive different
states. For a scalar field:

```python
[{}, {"f": None}, {"f": 1}]  # state = [padded, null, valued]
```

A direct Awkward record projection can represent both missing and explicit null
as option values. The coalescer pairs values with final `Tokens` state before
regularization, so clipping and padding never compact a missing coordinate or
misalign state from its value.

### Structured Leaves

Set values and Vector embeddings are lists semantically, but not branch axes.
Every operation stops at schema `leaf_depth`. `"<MASK>"` is recognized only
when the projected leaf itself is the sentinel; the same string inside a
structured leaf is ordinary codec input.

### Heterogeneous Leaves

The canonical carrier accepts Awkward-compatible strings, numbers, booleans,
records, lists, and option values for directly bound modeled leaves. Explicit
query results follow the same constraint when materialized. Preprocessors
normalize UUID, Decimal, custom classes, and device tensors only when a modeled
field or query selects them. Unmodeled metadata remains in the untouched source
observation rather than entering the Awkward layout.

Awkward may coerce heterogeneous Python scalars differently according to input
order. Each datatype therefore declares accepted raw atom compatibility
families when it creates its plugin. Separate tuple entries are incompatible;
a PEP 604 union is one family whose members may safely promote together:

```python
identity = Plugin(name="identity", types=(str, bytes))
numeric = Plugin(name="numeric", types=(int | float,))
```

The identity plugin accepts either registered type but rejects a field address
that mixes them. The numeric plugin permits `int` and `float` together. Plugin
construction rejects a type repeated across families. Matching prefers an
exact registered runtime type after canonicalization. Thus explicitly
registered `bool`/`int` and `datetime`/`date` remain distinct; accepted custom
subclasses are registered explicitly. List, tuple, ndarray, iterator, and
NumPy scalar classes are prepared structurally and are not valid family
entries. Record-valued extensions may register a concrete mapping type such
as `dict` as a terminal atom.

Before Awkward construction, the shared core converts NumPy scalar atoms to
their Python equivalents and recursively prepares atoms inside list, tuple,
and NumPy-array leaf values. It aggregates observed family indexes by field
address and rejects more than one family. This is generic plugin dispatch: the
core resolves `request.type` through the registry but never branches on it or
names a concrete tensorfield.

The `types` declaration describes carrier atoms, not semantic leaf shape. A
tensorfield still validates a Set container, Vector width, date grammar, or
other datatype meaning after overflow selects retained values. One-shot
iterators are rejected rather than consumed. `None` and a whole-leaf
prediction `"<MASK>"` are routing states outside plugin atom matching; the same
string nested inside a structured leaf is an ordinary `str` atom. Directly
bound leaves and explicit query results use the same contract.

### Overflow

- head overflow maps naturally to clipping before padding;
- tail overflow needs an axis-specific trailing slice/selection;
- error overflow uses `ak.num` validation and reports its address and axis;
- all-empty and incomplete parent nodes need explicit layout tests.

### Conversion Boundary

`ak.to_numpy` intentionally refuses irregular or irreducible object layouts,
which is useful after regularization. Fill option values and make all model axes
regular before conversion. Prefer `torch.from_numpy` at the final CPU boundary.
`ak.to_torch` supports only a subset of Awkward layouts; see its
[API contract](https://awkward-array.org/doc/main/reference/generated/ak.to_torch.html).

Keep Awkward in CPU data workers. CUDA placement, autocast, attention, and model
execution remain Lightning/PyTorch responsibilities.

### Custom Tensorfields

Every extension declares its atom families with
`Plugin(name=..., types=(...))`. The core invokes that plugin uniformly for
direct and explicit-query leaves and never dispatches on a built-in type name.

Every tensorfield then receives `RaggedField`, including third-party
extensions. It semantically validates or encodes the retained
`field.values`, then calls `field.place`. There is no nested-Python
`TensorField.new` signature or plugin-owned query path. The core converges
direct projection and explicit JMESPath extraction before paired
regularization, prepares every encoded field, and only then invokes tensorfield
codecs.

The runtime owns prediction metadata separately. Awkward layouts do not enter
model state or checkpoints.

## Directional Measurements

An early isolated benchmark compared a self-contained reference Python
extractor/padder with Awkward 2.13.0 on synthetic ragged numeric branches. It
asserted equal content/state only for that shared numeric head-overflow case;
it was not the semantic oracle for the new missing, overflow, or literal rules.

These numbers are directional only and were not collected on a controlled
benchmark host. Methods run in displayed order, so allocation and thermal order
effects remain possible. The host was Apple arm64 with Python 3.12.10, NumPy
2.4.6, PyArrow 24.0.0, and repository revision `e7cf72d`.
All field-count cases project the first fields from one shared maximum-width
dataset, so they have identical ragged lengths and shared-field null patterns.

Representative target case: 32 observations, capacity 1,536, mean nested length
1,086.8, p95 2,204.6, 8% explicit nulls, and five repeats. Median milliseconds:

| Fields | Legacy Python reference | Prototype Awkward from Python | Arrow to Python reference | Prototype Awkward from Arrow | Legacy extracted pad | Prototype Awkward pad |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.95 | 4.40 | 35.63 | 1.68 | 8.72 | 2.37 |
| 4 | 50.73 | 13.77 | 105.29 | 6.66 | 40.94 | 11.34 |
| 16 | 225.37 | 52.32 | 459.85 | 26.68 | 160.12 | 44.62 |

Small-control case: 32 observations, capacity 64, the same null rate, and seven
repeats. Median milliseconds:

| Fields | Legacy Python reference | Prototype Awkward from Python | Arrow to Python reference | Prototype Awkward from Arrow | Legacy extracted pad | Prototype Awkward pad |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.51 | 1.33 | 1.57 | 1.29 | 0.36 | 1.21 |
| 4 | 2.03 | 4.86 | 4.44 | 4.72 | 1.43 | 4.85 |
| 16 | 8.40 | 19.08 | 17.40 | 18.81 | 5.76 | 19.73 |

Interpretation:

- in this simplified prototype, Awkward was about 2.7--4.3x faster from a Python
  batch, 15.8--21.2x faster from Arrow, and 3.6--3.7x faster for the
  already-extracted regularization step;
- in the capacity-64 control, fixed Awkward setup made the Python path about
  2.3--2.6x slower and the already-extracted path about 3.3--3.4x slower; a
  one-row case is more extreme (about 1 ms of Awkward overhead per field here);
- replacing `pad` alone improved this input but offers much less architectural
  simplification than sharing the whole batch;
- sharing one eager Awkward coalescing pass amortized conversion across fields;
- avoiding Arrow `to_pylist()` dominated only the sufficiently large synthetic
  workload.

These figures are an upper-bound microbenchmark for simplified numeric
structure handling, not the measured speed of the canonical implementation.
The result supports a canonical batch/source boundary, not adding Awkward only
inside the existing `pad` function.

A Phase 1 benchmark reconstructed the exact audit-baseline numeric path and
compared it with the then-current canonical batch preparation plus all field
projections on the same 32-observation, capacity-1,536 workload. Median
milliseconds:

| Fields | Audit-baseline implementation | Canonical Phase 1 | Structural speedup |
| ---: | ---: | ---: | ---: |
| 1 | 24.46 | 25.21 | 0.97x |
| 4 | 73.21 | 59.76 | 1.23x |
| 16 | 281.20 | 208.84 | 1.35x |

The canonical path does more work than the simplified prototype: it preserves
coordinates and supports the complete missing/null/overflow/literal contract
while keeping unmodeled metadata outside Awkward. Treat these as directional
implementation results, not a release or end-to-end training claim. The eager
coalescer should be rebenchmarked separately rather than inheriting these
historical timings.

## General Value Ballpark

The prototype and implementation spot check measure numeric structure handling,
not an end-to-end loader. Use these results as scope indicators:

| Workload | Expected preprocessing value |
| --- | --- |
| Primitive or vocabulary-backed fields from Python | Roughly neutral for one field (0.97x here), then 1.23--1.35x structural speedup for 4--16 fields in the documented Phase 1 benchmark; remeasure other shapes |
| Arrow/Parquet with no row-wise preprocessor | The prototype's roughly 10--20x source-boundary result is an upper bound; the direct columnar adapter is Phase 2 |
| Hash, DateParts, Set, and other Python/stateful codecs | Structure may improve, but codec work remains and no total-field range is yet established |
| Text | Tokenization/encoder usually dominates; ingestion may improve without moving the dominant cost |
| Model step dominated by GPU compute | Little wall-clock gain unless preprocessing currently starves the GPU |

The training gain follows Amdahl's law and will be smaller than the isolated
structural gain unless preprocessing dominates the step. Validate it with
complete DataLoader/GPU-idle profiles.

The non-performance value is broader: nine built-in tensorfields converge on
one structure/state/literal/overflow implementation and schema-aware projection
becomes one maintained contract. Phase 2 source adapters can then remove nested
Arrow-to-Python round trips without introducing a second tensorization engine.

## Compilation And JIT

Awkward's normal array operations already execute through compiled low-level
kernels; ordinary `ak.pad_none`, `ak.fill_none`, projections, reducers, and
ufuncs do not run a Python callback per leaf. Start there. The official
[kernel specification](https://awkward-array.org/doc/main/reference/generated/kernels.html)
states that non-NumPy/CuPy array manipulation is performed by compiled kernels.

Use Numba only when profiling finds that a readable Awkward expression makes
several full passes or large intermediates. Awkward arrays can be passed into a
Numba-compiled imperative function, so a fused regularize/place kernel can look
like this:

```python
import numba as nb
import numpy as np


@nb.njit(cache=True, nogil=True)
def place_number(rows, capacity, padded, null, valued):
    content = np.zeros((len(rows), 1, capacity), dtype=np.float32)
    state = np.full((len(rows), 1, capacity), padded, dtype=np.int64)

    for batch_index in range(len(rows)):
        for root in rows[batch_index]:  # canonical singleton root
            width = min(len(root), capacity)
            for slot in range(width):
                value = root[slot]
                if value is None:
                    state[batch_index, 0, slot] = null
                else:
                    content[batch_index, 0, slot] = value
                    state[batch_index, 0, slot] = valued

    return content, state
```

This is still an Awkward design: `rows` is an `ak.Array`, its layout owns the
ragged structure, and only the hot fused traversal is imperative. Numba turns
those explicit loops into native code. It is a good fit for overflow + null
state + placement in one pass when the equivalent array expression allocates
several intermediates. The official
[Awkward/Numba guide](https://awkward-array.org/doc/main/user-guide/how-to-use-in-numba-intro.html)
shows the same tradeoff and a multi-pass example where a fused compiled loop is
substantially faster.

Important constraints:

- inside `@njit`, iterate and branch; do not call `ak.*`, NumPy ufuncs over an
  Awkward layout, or fancy Awkward slices. Those belong outside the compiled
  function. See the official
  [supported-features guide](https://awkward-array.org/doc/main/user-guide/how-to-use-in-numba-features.html);
- Awkward inputs are immutable in compiled code. Return NumPy buffers, a view
  into the input, or fill an `ak.ArrayBuilder` created outside the JIT;
- variable batch size and list lengths are runtime values and do not by
  themselves require a new specialization. Layout form and leaf dtypes do;
- eager coalescing must normalize each field to a schema-stable Awkward Form so
  all-empty, no-null, and some-null batches do not create a stream of
  specializations;
- preprocessors normalize opaque modeled values and explicit-query results
  before their Awkward construction, so nopython kernels see only
  schema-stable Awkward forms; opaque unmodeled metadata never enters them;
- `cache=True` writes compiled specializations to disk (see the
  [Numba JIT options](https://numba.readthedocs.io/en/stable/user/jit.html)),
  and worker startup should prewarm the finite set of schema forms. Compilation
  and cache-load time must be reported separately from steady state;
- JAX is not the path here: Awkward's
  [JAX backend is deprecated](https://awkward-array.org/doc/main/user-guide/how-to-specialize-differentiate-jax.html).
  Numba CUDA is possible, but CPU DataLoader workers plus one final
  NumPy-to-Torch boundary are simpler and avoid premature device transfers.
- Awkward TypeTracer predicts forms and touched buffers without data; it is a
  planning/testing tool, not execution acceleration. `torch.compile` begins
  only after the NumPy-to-Torch boundary and does not compile RaggedField.

Do not maintain vectorized and Numba production implementations indefinitely.
Start with Awkward kernels. If a fused transform demonstrably warrants Numba,
make that one compiled implementation canonical and keep a small reference
oracle only in tests.

## Prototype And Benchmark Plan

### Phase 1: Canonical Engine

1. Add Awkward as a core dependency.
2. Implement internal eager `coalesce(...)` and the `RaggedField` boundary
   together.
3. Change every built-in and third-party tensorfield contract to consume
   `RaggedField`.
4. Delete duplicated literal/pad/state code, generated/default JMESPath
   extraction, and the old recursive engine. Keep explicit queries as an
   extraction adapter that converges before paired regularization.
5. Run the full tensorfield, data, plugin, and checkpoint suites.
6. Record deleted code, concepts, and error paths as primary acceptance metrics.

The canonical engine must cover:

- ranks 1 through 5;
- empty and all-empty arrays;
- missing key versus explicit null;
- incomplete scalar at each branch level;
- head, tail, and error overflow at every axis;
- plugin-declared atom compatibility before Awkward, followed by overflow
  before `TensorField.new` semantic validation, vocabulary reservation,
  counting, and tokenization, including carrier-valid but codec-invalid clipped
  leaves;
- structured Set/Vector leaves;
- mixed Hash inputs;
- prediction literal placement and nested structured-leaf strings as codec
  input;
- vocabulary first-seen order and canonical exposure counts;
- documented address/axis/type errors;
- custom tensorfield conformance;
- direct and explicit-query inputs producing the same canonical field semantics,
  including overflow after extraction and null-preserving filtered siblings;
- one real TensorField/embedder forward proving state index dtype compatibility.

### Phase 2: Columnar And Compiled Paths

1. Add Arrow/Polars adapters that avoid row materialization when preprocessing
   permits it.
2. JIT only the fused irregular transforms that remain multi-pass bottlenecks.
3. Measure complete DataLoader and GPU-idle behavior, not only `pad`.

Benchmark:

- Python, Polars, and Parquet/Arrow sources;
- direct binding, explicit JMESPath binding, identity preprocessing, and
  one-to-many preprocessing;
- batch sizes 1, 32, 256, and 1024;
- 1, 8, and 32 fields;
- depths 1, 2, and 3;
- mean branch lengths 8, 64, 1,024, and 4,096, with batch size adjusted to a
  fixed memory budget;
- 0%, 10%, and 50% null/missing values;
- 0 and 4 workers.

Report records/s, valued leaves/s, median and p95 batch latency, peak
RSS/allocations, worker startup, and GPU idle time. Separate source conversion,
projection, regularization, codec, and Torch conversion timings. Plot against
total nested leaves as well as observation count; batch count alone is a poor
measure when one observation contains thousands of values.

## Acceptance Gates

The Awkward launch is complete only when all of these are true:

1. RelFlow has one production nested-array engine, not two.
2. `_iter_leaf_nodes`, `_write_leaf`, repeated literal padding, and duplicated
   built-in structure handling are removed.
3. Awkward-specific logic is centralized; individual datatype modules do not
   grow bespoke Awkward traversals.
4. Missing/null/padding/literal/overflow semantics match the canonical truth
   table in this design.
5. Every custom tensorfield uses the same `RaggedField` contract.
6. Omitted queries use direct schema-address projection; explicit queries
   converge with direct values before paired regularization without introducing
   a second structural engine.
7. Batches can scale with the configured memory budget without Python recursive
   traversal becoming the bottleneck.
8. Representative thousand-value nested and Arrow-backed workloads improve
   end-to-end throughput, rather than only a micro-operation.
9. The core dependency and supported platforms remain acceptable.
10. The ragged core contains no registry of concrete leaf type names; custom
    plugins participate through the same required `types` declaration as
    built-ins.

Small-batch overhead remains a benchmark dimension and can inform minimum batch
sizing, but it is not a reason to retain a second production tensorization
engine for a workload whose normal nested cardinality is above one thousand.

## Official References

- [Awkward padding and clipping](https://awkward-array.org/doc/main/user-guide/how-to-restructure-pad.html)
- [Awkward and Python conversion](https://awkward-array.org/doc/main/user-guide/how-to-convert-python.html)
- [Awkward and Arrow conversion](https://awkward-array.org/doc/main/user-guide/how-to-convert-arrow.html)
- [Awkward `ak.mask`](https://awkward-array.org/doc/main/reference/generated/ak.mask.html)
- [Awkward `ak.to_numpy`](https://awkward-array.org/doc/main/reference/generated/ak.to_numpy.html)
- [Awkward `ak.to_torch`](https://awkward-array.org/doc/main/reference/generated/ak.to_torch.html)
- [Awkward compiled kernel specification](https://awkward-array.org/doc/main/reference/generated/kernels.html)
- [Awkward with Numba](https://awkward-array.org/doc/main/user-guide/how-to-use-in-numba-intro.html)
- [Awkward features supported by Numba](https://awkward-array.org/doc/main/user-guide/how-to-use-in-numba-features.html)
- [Awkward TypeTracer reports](https://awkward-array.org/doc/main/reference/generated/ak.typetracer.typetracer_with_report.html)
- [Numba JIT options and cache](https://numba.readthedocs.io/en/stable/user/jit.html)
