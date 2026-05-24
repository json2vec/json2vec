# Getting Started

Use the docs extra when you want the rendered notebooks, API reference, and local site.

```bash
uv sync --extra docs
uv run --extra docs mkdocs serve
```

Open the local URL printed by MkDocs and run the **Hello World** tutorial first. It trains one tiny model, calls `predict`, calls `embed`, and plots the schema tree. The example is intentionally small so the full workflow stays readable.

## Minimal Shape

For top-level records, field names are enough. JSON2Vec infers the request query from the field name.

```python
import json2vec as j2v

model = j2v.Model.from_schema(
    j2v.Number("sepal_length"),
    j2v.Number("petal_length"),
    j2v.Category("species", target=True, max_vocab_size=4),
    d_model=16,
    n_layers=1,
    n_heads=4,
    batch_size=8,
)
```

`target=True` withholds `species` from the input during supervised training and asks the model to decode it from the remaining fields.

## Nested Shape

Use `Array` when a record contains repeated child objects.

```python
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

This schema reads records shaped like:

```python
{
    "measurements": [
        {"name": "mean_radius", "value": 17.99},
        {"name": "mean_texture", "value": 10.38},
    ],
    "diagnosis": "malignant",
}
```

For source data whose keys do not match the schema names, add explicit `query=...` expressions. See [Schemas & Queries](guides/model-schemas.md) for the query rules and the batching details.

## Next Steps

- Continue with **Hello World** for the complete supervised loop.
- Jump to **Masked Pretraining** if you want a nested self-supervised example.
- Use [Model Updates](guides/model-update.ipynb) when you need to add, delete, reset, or temporarily override schema nodes after creating a model.
