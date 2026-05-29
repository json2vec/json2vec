# Built-In Data Types

Data types are the structural and typed nodes in a JSON2Vec schema. `Array`
groups repeated nested objects. Tensorfields are typed leaves that read values,
encode them into tensors, hide values during training, and decode targets when
requested.

Use constructor names from the package root in Python. Serialized schemas use
the lower-case `type` value.

The pages in this section cover built-in data types. To define a new data type,
see [Custom Data Types](../guides/tensorfields.ipynb).

## Choose A Data Type

| Source value | Recommended type | Use a different type when |
| --- | --- | --- |
| Continuous scalar | [`Number`](number.md) | Numeric value is an ID, code, or class label. |
| One label | [`Category`](category.md) | The label only needs equality matching within a repeated context; use `Entity`. |
| Zero or more labels | [`Set`](set.md) | Labels have attributes or order; use `Array`. |
| Repeated objects | [`Array`](array.md) | Repetition is only an upstream storage artifact; preprocess or flatten. |
| Timestamp/calendar value | [`DateParts`](dateparts.md) | Elapsed time or recency matters; derive a `Number`. |
| Local repeated identity | [`Entity`](entity.md) | The ID must be stable across training and prediction; use `Category`. |
| Precomputed dense vector | [`Vector`](vector.md) | JSON2Vec should compute embeddings from strings; use `Text`. |
| Free-form text | [`Text`](text.md) | The string is a bounded label; use `Category` or `Set`. |

Same raw value can need different types:

- `"12345"` is a `Number` only if distance and magnitude matter.
- `"12345"` is a `Category` if it is a stable global ID or code.
- `"12345"` is an `Entity` if only repeated equality inside the current
  repeated context matters.
- `["red", "sale"]` is a `Set` if the labels are unordered.
- `[{"name": "red"}, {"name": "sale"}]` is an `Array` if each item has fields.

## Prediction Support

| Type | Public `Model.predict(...)` content | State probabilities | Notes |
| --- | --- | --- | --- |
| `Number` | Yes, scalar value | Yes | Metrics are reported in original value scale. |
| `Category` | Yes, best label plus optional top-k candidates | Yes | Unknown bucket is internal only. |
| `Set` | Yes, per-label probabilities or thresholded labels | Yes | `threshold` can reduce API response size. |
| `Vector` | Yes, reconstructed vector | Yes | Non-valued predictions return zero-vector content. |
| `DateParts` | No | No public payload | Trains losses and accuracies only. |
| `Entity` | No | No public payload | Batch-local identity representation. |
| `Text` | No | No public payload | Reconstructs frozen encoder embeddings, not text. |
| `Array` | No direct payload | No | Child fields may emit predictions. |

## Shared Leaf Options

Every tensorfield inherits shared leaf options. Type-specific pages document
options unique to that tensorfield. `Array` has structural options because it
groups children instead of reading one source value.

### Schema Identity

| Option | Default | Notes |
| --- | --- | --- |
| `name` | required | Public schema name. If `query` is omitted, this is also the source key. |
| `query` | inferred | JMESPath expression for the source value. See [Schemas & Queries](../guides/model-schemas.md). |
| `description` | `None` | Optional schema metadata. |
| `active` | `True` | Inactive fields stay in the schema but are ignored by encoding, losses, and prediction. |

### Training Roles

| Option | Default | Notes |
| --- | --- | --- |
| `target` | `False` | Shorthand for `p_prune=1.0`; hides the field from input and trains reconstruction. |
| `p_mask` | `None` | Randomly hides individual values during training. |
| `p_prune` | `None` | Randomly hides whole field instances during training. |
| `weight` | `1.0` | Multiplier applied to this field's loss. |

### Outputs And Decoder Options

| Option | Default | Notes |
| --- | --- | --- |
| `embed` | `False` | Includes this node in `Model.embed(...)` outputs. It does not make the field a target. |
| `pooling` | `"query"` | Target decoder pooling: `"query"` or `"mean"`. |
| `n_heads` | `4` | Attention heads used by query pooling. Must be even. |
| `dropout` | `None` | Optional dropout rate for query pooling. |
| `n_linear` | `1` | Number of linear layers used by query pooling. |

## Value State

Tensorfields track value state separately from value content:

- `valued`: the source value exists and was encoded.
- `null`: the source value exists as `None`.
- `padded`: the configured array shape has a slot with no source value.
- `masked`: training or prediction intentionally hid the value.

```json
[
  {"tags": ["vip"]},
  {"tags": []},
  {"tags": null},
  {}
]
```

For `Set("tags")`, `["vip"]` and `[]` are both valued content. `null` is a
null field state. A missing repeated slot inside an `Array` is padded.

This separation matters most for numeric content. A sentinel such as `0`, `-1`,
or `999999` can collide with real values and distort normalization. `NaN` can
say "not a number", but it cannot distinguish null source data, padded array
slots, and training-time masks. The model predicts state separately, then
content is meaningful when the field is valued.
