# Schemas & Queries

A schema does two jobs at once. It describes where the pipeline should read values from a record (via `JMESPath`), and it defines the model modules built for those values.

Arrays become context encoders with pooling mechanisms, and leaf fields become typed tensorfields such as `Number`, `Category`, `Set`, `DateParts`, etc. See [Built-In Data Types](../data-types/index.md) for the behavior and configuration of each schema node.

Every leaf field has `active=True` by default. Set `active=False` when you want to keep a field in the schema tree but ignore it during encoding, forward passes, losses, and prediction. Reactivating the same field later rebuilds the compatible modules. Deactivating a leaf field is like deleting it, but it is a reversible operation.

## Top-Level Fields

For simple records, omit `query`. JSON2Vec infers the source path from the field name.

```python
record = {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "cultivar": "class_0",
}

model = j2v.Model.from_schema(
    j2v.Number("alcohol"),
    j2v.Number("malic_acid"),
    j2v.Number("ash", active=False),
    j2v.Category("cultivar", target=True, max_vocab_size=4),
    d_model=16,
    n_layers=1,
    n_heads=4,
)
```

The active inferred request queries are `[*].alcohol`, `[*].malic_acid`, and `[*].cultivar`. The inactive `ash` field remains selectable for later updates, but it is not encoded or trained while inactive.

## Nested Arrays

Use `Array` for repeated child objects. Child fields are read from the array named by the parent node.

```python
record = {
    "measurements": [
        {"name": "mean_radius", "value": 17.99},
        {"name": "mean_texture", "value": 10.38},
    ],
    "diagnosis": "malignant",
}

model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Category("name", max_vocab_size=16),
        j2v.Number("value"),
        name="measurements",
        max_length=8,
    ),
    j2v.Category("diagnosis", target=True, max_vocab_size=2),
    d_model=16,
    n_layers=1,
    n_heads=4,
)
```

The inferred child queries are `[*].measurements[*].name` and `[*].measurements[*].value`. The resulting schema addresses are `record/measurements/name` and `record/measurements/value`.

!!! Note
    The above schema is just an example. It is inefficient to use this melted version of what could be a flattened dataframe.

## Explicit Queries

Use explicit `query=...` when the source key does not match the public schema name, or when the source payload is awkward.

```python
model = j2v.Model.from_schema(
    j2v.Number("amount", query='[*].transaction."amount_usd"'),
    j2v.Category("merchant", query="[*].transaction.merchant_name", max_vocab_size=1024),
    j2v.Category("label", query="[*].outcome", target=True, max_vocab_size=2),
    d_model=16,
    n_layers=1,
    n_heads=4,
)
```

The schema name and source query do not need to match. This lets a model expose stable names while reading from versioned, nested, or awkward upstream payloads.

## Request Queries Versus Compiled Queries

Request queries are written relative to one processed observation. During encoding, JSON2Vec prepends the outer batch selector before running JMESPath.

```python
request_query = "[*].amount"
compiled_query = "[*][*].amount"
```

Most users should think in terms of the request query. Do not add both leading selectors yourself. A request query such as `[*][*].amount` becomes `[*][*][*].amount` at encode time and is over-nested for the normal encoded batch shape.

The normal encoded batch shape is:

```python
[
    [{"amount": 10.0}],
    [{"amount": 25.5}],
]
```

The outer list is the training or prediction batch. The inner list contains one or more records emitted for a processed observation. Preprocessors that split one source object into multiple observations still feed the same query convention.

## When To Use A Preprocessor Instead

Use `query` when the source shape is stable and can be selected declaratively. Use a [preprocessor](preprocessors.ipynb) when records need Python logic first: type coercion, vendor-specific normalization, windowing, session splitting, or derived fields.
