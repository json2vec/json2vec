# Model Schemas

Use direct tensorfield constructors inside `Model.from_schema(...)`.

A schema does two jobs at once. It describes the data JSON2Vec should read, and it defines the model modules that will be built for that data. Array nodes become context encoders, while leaf nodes become typed tensorfields such as `Number` or `Category`.

```python
model = j2v.Model.from_schema(
    j2v.Number("alcohol"),
    j2v.Number("malic_acid"),
    j2v.Number("color_intensity"),
    j2v.Number("proline"),
    j2v.Category("cultivar", target=True, max_vocab_size=4),
    d_model=128,
    n_layers=3,
    n_heads=4,
)
```

Use explicit `query=...` when the source key does not match the schema node name.

Queries are JMESPath expressions evaluated against the encoded batch. Tutorial notebooks use `ROOT = "[*][*]"` because a Polars data module batches records into a nested list shape before field extraction. The extra selectors make each field read from every record in every batch item.

## Self-Contained Example

```python
ROOT = "[*][*]"

model = j2v.Model.from_schema(
    j2v.Number("alcohol", query=f"{ROOT}.alcohol"),
    j2v.Number("malic_acid", query=f"{ROOT}.malic_acid"),
    j2v.Number("color_intensity", query=f"{ROOT}.color_intensity"),
    j2v.Number("proline", query=f"{ROOT}.proline"),
    j2v.Category("cultivar", query=f"{ROOT}.cultivar", target=True, max_vocab_size=4),
    d_model=16,
    n_layers=1,
    n_heads=4,
    batch_size=4,
)
```

The schema name and the source query do not have to be the same. This lets a public model use stable names while reading from upstream payloads whose field names are awkward, nested, or versioned.
