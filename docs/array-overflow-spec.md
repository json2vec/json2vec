# Array Overflow Policy Spec

## Problem

`Array(max_length=...)` controls how many repeated items are retained in each
schema array dimension. The docs currently say extra items are truncated, but
they do not state which side is retained or whether users can make overflow an
error.

The behavior is also hard to infer from `pad(...)`:

- Python lists longer than the configured shape are silently truncated.
- The current truncation policy is **head**: indices starting at `0` are kept.
- No error is raised for overflow today.
- `pad(...)` currently treats only Python `list` objects as repeated
  dimensions. A NumPy ndarray that appears at a leaf value is not sliced by
  array padding. For `Vector`, the vector tensorfield validates `n_dim`
  separately and raises if the vector width is wrong.

This is too implicit for production schemas. Time-ordered histories often need
the most recent records (`tail`), small debugging datasets may need strict
overflow failures (`error`), and existing users need the current behavior
preserved (`head`).

## Goal

Add an overflow policy to each user-declared `Array`:

```python
overflow: Overflow = Overflow.head
```

The policy decides what happens when a source repeated dimension has more items
than the array's `max_length`.

| Policy | Behavior |
| --- | --- |
| `"head"` | Keep the first `max_length` items and drop the rest. This is the current behavior. |
| `"tail"` | Keep the last `max_length` items and drop the earlier items. Retained values are compacted into output slots `0..max_length-1`. |
| `"error"` | Raise a `ValueError` if the source contains more than `max_length` items. |

## Public API

### Nested Arrays

```python
model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Category("event_type", max_vocab_size=128),
        j2v.Number("amount"),
        name="events",
        max_length=128,
        overflow="tail",
    ),
    j2v.Category("label", target=True, max_vocab_size=2),
    d_model=64,
    n_layers=2,
    n_heads=4,
)
```

This keeps the last 128 `events` in the query result order.

### Strict Overflow

```python
j2v.Array(
    j2v.Category("sku", max_vocab_size=4096),
    name="line_items",
    max_length=32,
    overflow="error",
)
```

This raises when any observation contains more than 32 `line_items`.

### Root Array

Do not expose `overflow` on the generated root array. The root array is a
synthetic singleton context and is always length `1`. Internally, root overflow
must be `Overflow.error`: if the root dimension is ever larger than `1`, the
input has violated a core runtime invariant. User-facing overflow policy only
applies to declared repeated `Array(...)` nodes below the root.

## Semantics

Overflow applies only to schema array dimensions, not to tensorfield-native
inner dimensions.

- `Array(overflow=...)` controls repeated object slots.
- `Vector(n_dim=...)` still controls vector width and should continue to raise
  when a vector has the wrong length.
- `Text(max_length=...)` still controls tokenizer truncation and padding.
- `Set(...)` still controls set content through its vocabulary logic.

Array order is the order returned by the leaf query. Users who need a specific
order should sort, filter, or window in a preprocessor before tensorization.

For nested arrays, each dimension uses the closest owning `Array` policy. For
example:

```python
j2v.Array(
    j2v.Array(
        j2v.Number("amount"),
        name="transactions",
        max_length=32,
        overflow="tail",
    ),
    name="sessions",
    max_length=4,
    overflow="head",
)
```

This keeps the first 4 sessions, and within each retained session keeps the
last 32 transactions.

## Current Implementation Points

The relevant code paths today are:

- `src/json2vec/structs/structure.py`
  - `Array` defines `max_length`.
- `src/json2vec/structs/tree.py`
  - `Leaf.shape` derives repeated dimensions from array ancestors.
- `src/json2vec/structs/experiment.py`
  - `Hyperparameters.shapes` exposes leaf tensor shapes.
- `src/json2vec/data/processing.py`
  - `pad(...)` and `_iter_leaf_nodes(...)` materialize nested query results
    into fixed shapes.
- `src/json2vec/tensorfields/extensions/*.py`
  - Tensorfields call `pad(...)` with `(len(values), *array_shape)`.

The current truncation behavior lives in `_iter_leaf_nodes(...)`:

```python
limit = min(len(node), shape[depth])
for index in range(limit - 1, -1, -1):
    stack.append((node[index], depth + 1, base + (index * step)))
```

That means overflow keeps the head of the list and silently drops items after
`shape[depth]`.

## Proposed Internal Model

### Types

Add a shared enum in `src/json2vec/structs/enums.py`:

```python
class Overflow(enum.StrEnum):
    head = "head"
    tail = "tail"
    error = "error"
```

This should mirror other string-valued configuration enums in the repo. Pydantic
should coerce string inputs such as `"tail"` to `Overflow.tail` during schema
validation.

### Array

Add to `Array`:

```python
overflow: Overflow = Overflow.head
```

Validation should reject any other value through pydantic.

### Root Schema Constructors

Do not add `overflow` to `Hyperparameters.from_schema(...)` or
`Model.from_schema(...)`. The generated root array should always use the
default internal `Overflow.error` value and should not present a user-facing
overflow option.

Do not add `max_length` to `Hyperparameters.from_schema(...)` or
`Model.from_schema(...)`. The generated root array is always a singleton wrapper
around each observation; repeated records must be modeled with an explicit child
`Array(max_length=...)`.

`Hyperparameters` should force the top-level `fields` array to
`Overflow.error` during schema materialization. That keeps old serialized
schemas safe even if they do not contain an `overflow` field, and prevents
manual root-array payloads from changing the root invariant. User-declared
child arrays still default to `Overflow.head` unless configured otherwise.

### Leaf Overflow Paths

Add an array-overflow analogue to `Leaf.shape`.

```python
@functools.cached_property
def overflows(self) -> tuple[Overflow, ...]:
    out: list[Overflow] = []
    for node in self.path:
        if node.type == "array":
            out.append(node.overflow)
    return tuple(out)
```

Add a callable helper on `Hyperparameters` for tensorfield padding:

```python
def overflows(self, address: Address) -> tuple[Overflow, ...]:
    return (Overflow.error, *self.requests[Address(str(address))].overflows)
```

The returned tuple includes the batch overflow policy at dimension `0`, then
the generated root and declared schema array policies.

### Padding

Extend `pad(...)`:

```python
def pad(
    nested: Any,
    shape: tuple[int, ...],
    dtype: type | str = object,
    pad_value: Any = None,
    overflows: tuple[Overflow, ...] | None = None,
    address: Address | str | None = None,
    value_shape: tuple[int, ...] = (),
    encode: Callable[[Any], Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

Rules:

1. `overflows=None` means all dimensions use `Overflow.head` for backwards
   compatibility.
2. `len(overflows)` must equal `len(shape)` when provided.
3. `Overflow.head` keeps source indices `0:max_length`.
4. `Overflow.tail` keeps source indices `len(node)-max_length:len(node)` and writes
   them compactly into destination slots starting at `0`.
5. `Overflow.error` raises `ValueError` when `len(node) > max_length`.
6. `value_shape` appends fixed terminal content dimensions to the returned
   values array while keeping the state flags shaped as `shape`.
7. `encode` transforms non-null leaf values before assignment.

The first dimension passed to `pad(...)` by tensorfields is the batch dimension,
not a schema array dimension. The second dimension is the generated root array
and is also an internal invariant. Tensorfield calls should prepend strict
policies for both dimensions:

```python
data, flags = pad(
    nested=values,
    shape=(len(values), *array_shape),
    overflows=hyperparameters.overflows(address),
    address=address,
)
```

Tensorfields that expand one terminal value into a fixed content vector should
still delegate traversal and state flags to `pad(...)`:

```python
data, flags = pad(
    nested=values,
    shape=(len(values), *array_shape),
    dtype=np.float32,
    pad_value=0.0,
    value_shape=(n_tokens,),
    encode=encode_leaf,
    overflows=hyperparameters.overflows(address),
    address=address,
)
```

This replaces tensorfield-specific walkers such as the previous set
multi-hot padding path.

The batch dimension and generated root dimension should always use
`Overflow.error`. If either is larger than the expected shape, something is
critically wrong: batching or root wrapping has broken before model execution.
Declared arrays below the root use their configured policy.

### Error Messages

`pad(...)` should accept the leaf `Address` as an error-message hint, without
requiring full per-dimension array path context. It can identify overflow using
the dimension index:

- Dimension `0` is the batch dimension.
- Dimension `1` is the generated root node.
- Deeper dimensions report the dimension index plus the leaf address, expected
  shape, and provided shape.

## NumPy Array Handling

This change should avoid accidentally slicing tensorfield-native ndarray
content.

Recommended behavior:

- At non-leaf repeated dimensions, `pad(...)` may treat `list`, `tuple`, and
  `np.ndarray` as sequence-like containers for overflow purposes.
- At leaf depth, the node is the tensorfield value and must not be sliced by
  `Array.overflow`.
- `Vector` values should continue to be coerced and validated by the vector
  tensorfield before padding. A vector with width larger than `n_dim` should
  still raise the existing vector width error, not use `Array.overflow`.

This resolves the ambiguity: `Array.overflow` slices repeated records, not
feature vectors or token sequences.

## Tests

Add focused unit tests before broader integration tests.

### `tests/data/test_processing.py`

- `test_pad_overflow_head_is_default`
  - Input `[[1, 2, 3]]`, shape `(1, 2)`, no `overflows`.
  - Output keeps `[1, 2]`.
- `test_pad_overflow_tail_keeps_last_items`
  - Input `[[1, 2, 3]]`, shape `(1, 2)`, overflows
    `(Overflow.error, Overflow.tail)`.
  - Output keeps `[2, 3]`.
- `test_pad_overflow_error_raises`
  - Input `[[1, 2, 3]]`, shape `(1, 2)`, overflows
    `(Overflow.error, Overflow.error)`.
  - Raises `ValueError`.
- `test_pad_batch_overflow_raises`
  - Input has more items than the configured batch dimension.
  - `overflows=(Overflow.error, ...)` raises.
- `test_pad_root_overflow_raises`
  - A malformed encoded batch has more than one root item where root
    `max_length` is `1`.
  - The generated root `Overflow.error` raises.
- `test_pad_nested_overflow_policies_are_per_depth`
  - Parent keeps head, child keeps tail.
- `test_pad_overflow_tail_compacts_slots`
  - Ensure retained tail items land at destination slots `0..max_length-1`.
- `test_pad_overflow_does_not_slice_leaf_ndarray`
  - Leaf ndarray value at terminal depth is retained whole.

### `tests/structs/test_structure_model.py`

- `Array(..., overflow="tail")` validates and serializes.
- Invalid overflow values fail validation.
- `hyperparameters.overflows(address)` returns policies in padding order, with
  batch dimension `0` and the generated root fixed to `Overflow.error`.
- A manually supplied root overflow policy is normalized to `Overflow.error`.
- Cache clearing includes `overflows`.

### Tensorfield Integration

Use one simple tensorfield, such as `Number` or `Category`:

- Build `Array(Number("value"), max_length=2, overflow="tail")`.
- Encode input with three repeated values.
- Assert encoded content keeps the last two values.
- Build the same schema with `overflow="error"`.
- Assert encode raises before training starts.

### Public API Tests

- Nested `j2v.Array(..., overflow="tail")` is available from the package root
  import path.
- `Model.from_schema(..., overflow="tail")` is not part of the public API and
  should fail as an unexpected argument if attempted.
- `Model.from_schema(..., max_length=2)` is not part of the public API and
  should fail as an unexpected argument if attempted.

## Documentation Updates

Update:

- `docs/data-types/array.md`
  - Add `overflow` to the configuration table.
  - Replace the current vague "extra items are truncated" wording with explicit
    `head`, `tail`, and `error` behavior.
  - Add examples for time-ordered histories using `overflow="tail"` and strict
    schemas using `overflow="error"`.
- `docs/core-concepts/model-tree.md`
  - Mention that `max_length` and `overflow` define each array dimension.
- `docs/core-concepts/querypaths.md`
  - Remind users that overflow applies after query selection and preserves query
    order.
- `docs/guides/data-modules.md`
  - In the batch path, clarify that array overflow is resolved during query
    tensorization before masking and pruning.
- `docs/ai-quickstart.md`
  - Add a short note for repeated histories: use `overflow="tail"` when the
    newest events should be retained.
- `AGENTS.md`
  - Add a gotcha: `Array(overflow="head")` is the default; use `"tail"` for
    recency-ordered histories and `"error"` for strict schemas.

## Backwards Compatibility

Defaulting user-declared arrays to `"head"` preserves current behavior for
existing schemas and checkpoints. The root array is the exception: it is
normalized to `Overflow.error` because the root dimension is expected to be a
singleton.

Serialized schemas without an `overflow` field should load as `"head"`.
Serialized schemas with the new field should remain valid pydantic models and
round-trip through checkpoints.

## Acceptance Criteria

- Users can set `overflow` on user-declared repeated `Array(...)` nodes.
- Users cannot set `overflow` on the generated root array.
- Users cannot set `max_length` on the generated root array through
  `Model.from_schema(...)`.
- Existing schemas behave exactly as before when `overflow` is omitted.
- `overflow="head"` keeps the first `max_length` repeated items.
- `overflow="tail"` keeps the last `max_length` repeated items.
- `overflow="error"` raises a clear `ValueError` before model forward passes.
- Nested arrays apply overflow policies independently per depth.
- Leaf ndarray content, especially `Vector` values, is not truncated by
  `Array.overflow`.
- Array docs explicitly state which items are retained.
