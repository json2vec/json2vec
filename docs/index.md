# JSON2Vec

JSON2Vec builds neural encoders directly from nested JSON-like records. Instead of flattening histories, line items, sessions, or measurements into a fixed feature table first, you describe the record shape and JSON2Vec builds the matching model tree.

```python
record = {
    "customer_tier": "gold",
    "line_items": [
        {"sku": "A12", "quantity": 2, "price": 19.99},
        {"sku": "B07", "quantity": 1, "price": 45.50},
    ],
    "returned": False,
}

model = j2v.Model.from_schema(
    j2v.Category("customer_tier", max_vocab_size=16),
    j2v.Array(
        j2v.Category("sku", max_vocab_size=2048),
        j2v.Number("quantity"),
        j2v.Number("price"),
        name="line_items",
        max_length=32,
    ),
    j2v.Category("returned", target=True, max_vocab_size=2),
    d_model=64,
    n_layers=2,
    n_heads=4,
)
```

The schema is the model blueprint: arrays become context encoders, typed fields become tensorfields, targets become decoded outputs, and configured nodes can return embeddings.

## What To Read

- **New to the project:** start with [Getting Started](getting-started.md) for a runnable path.
- **Evaluating fit:** read [Why JSON2Vec](motivation.md) before investing in examples.
- **Working with record shapes:** use [Schemas & Queries](guides/model-schemas.md).
- **Changing model schemas after construction:** use [Model Updates](guides/model-update.ipynb).
- **Measuring field impact:** use [Field Ablation](guides/field-ablation.ipynb).
- **Preparing awkward source data:** use [Preprocessors](guides/preprocessors.ipynb).
- **Extending datatypes:** use [Tensorfield Extensions](guides/tensorfields.ipynb).

## Tutorial Path

The tutorials are ordered by workflow:

- **Hello World** runs the smallest supervised training loop.
- **Masked Pretraining** introduces nested arrays and self-supervised masking.
- **Nested Supervised Training** uses repeated measurement objects plus a root target.
- **Supervised Tabular Training** shows a compact flat classifier for comparison.
- **Serving** turns a saved schema into a deployment wrapper.
