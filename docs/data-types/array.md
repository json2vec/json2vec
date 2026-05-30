# Array

Use `Array` for repeated nested objects. An array groups child fields, gives
them a shared repeated shape, and builds a context encoder over the repeated
items.

```json
{
  "measurements": [
    {"name": "mean_radius", "value": 17.99},
    {"name": "mean_texture", "value": 10.38}
  ]
}
```

```python
import json2vec as j2v

measurements = j2v.Array(
    j2v.Category("name", max_vocab_size=32),
    j2v.Number("value"),
    name="measurements",
    max_length=8,
    n_layers=2,
)
```

Use a different type for scalar vectors (`Vector`) or unordered labels (`Set`).
If repeated rows are only a melted representation of a flat table, preprocessing
or flattening is usually more efficient.

## Input Values

`Array` expects a list of child objects at the source path named by `name`, unless
the child fields use explicit `query` expressions. Each child tensorfield is
padded or truncated to the configured `max_length`.

With `max_length=2`, two items are retained, extra items are truncated, and
missing slots are padded:

```json
[
  {"measurements": [{"name": "a", "value": 1.0}, {"name": "b", "value": 2.0}, {"name": "c", "value": 3.0}]},
  {"measurements": [{"name": "a", "value": 4.0}]}
]
```

The first record keeps `a` and `b`; `c` is truncated. The second record keeps
`a` and adds one padded slot.

For a top-level schema like:

```python
model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Category("name", max_vocab_size=32),
        j2v.Number("value"),
        name="measurements",
        max_length=8,
    ),
    d_model=32,
    n_layers=1,
    n_heads=4,
)
```

JSON2Vec infers child queries like `[*].measurements[*].name` and
`[*].measurements[*].value`.

!!! Note
    Field `name` controls the public schema name. `query` controls where values
    are read. When omitted, JSON2Vec infers the request query from the field and
    parent array names.

Use explicit queries when the source keys do not match the public schema names:

```python
items = j2v.Array(
    j2v.Category("sku", query="[*].line_items[*].product_sku", max_vocab_size=2048),
    j2v.Number("quantity", query="[*].line_items[*].qty"),
    name="items",
    max_length=32,
)
```

## Examples

Common array fields include:

- Transaction line items, order items, cart contents, or invoice rows.
- Time-ordered events in a session, visit, trip, or workflow.
- Repeated measurements, sensor readings, observations, or lab results.
- Related parties or counterparties attached to one record.

## Configuration

| Option | Default | Notes |
| --- | --- | --- |
| `name` | required | Public schema name and inferred source key. |
| `fields` | `[]` | Child arrays or tensorfields. Positional constructor arguments become `fields`. |
| `max_length` | `1` | Number of repeated slots retained per observation. Must be positive. |
| `attention` | `"mha"` | Attention implementation for the array encoder. |
| `n_layers` | `1` | Number of encoder layers for this array. |
| `n_heads` | `4` | Attention heads for this array. Must be even. |
| `n_linear` | `1` | Number of feed-forward linear layers in this array. |
| `dropout` | `None` | Optional dropout rate. |
| `p_mask` | `None` | Stored on the array node. Runtime masking is applied to active child tensorfields. |
| `p_prune` | `None` | Stored on the array node. Runtime pruning is applied to active child tensorfields. |
| `embed` | `False` | Includes this array node in `Model.predict(...)` outputs under `embedding`. |
| `description` | `None` | Optional schema metadata. |

## Nesting

Arrays can contain tensorfields or other arrays. Each nested array contributes
another repeated dimension to the child fields' shape.

```python
session = j2v.Array(
    j2v.Array(
        j2v.Number("amount"),
        name="transactions",
        max_length=32,
    ),
    name="sessions",
    max_length=4,
)
```

Deep nesting is useful when the source shape matters. If repeated records are
only an artifact of upstream storage, flattening or preprocessing the data can
be more efficient.

## Target And Prediction Behavior

`Array` itself is not a supervised target. Its child tensorfields can be
targets, and the array context is used to encode those children and route
information through the model.

Configure `embed=True` on an array when you want `Model.predict(...)` to return
a representation for the grouped context under that array address.
