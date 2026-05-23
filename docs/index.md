# JSON2Vec

JSON2Vec builds neural models directly from nested JSON-like records. The schema is the model: arrays become encoder nodes, typed fields become tensorfield requests, and the same graph is used for training, prediction, embedding, and serving.

Start with [Getting Started](getting-started.md), read the [Motivation](motivation.md), then work through the rendered notebooks under Tutorials. Each notebook is self-contained and uses the minimal API directly.

```bash
uv sync --extra docs
uv run --extra docs mkdocs serve
```

## What To Read First

The tutorial notebooks are ordered by workflow, not by feature list:

- **Hello World** shows the smallest useful supervised training loop.
- **End-to-End Training** keeps the same API shape but uses more input fields.
- **Pretraining** uses masking instead of a supervised label.
- **Fine-Tuning** adds a nested array of measurements and a supervised target.
- **Serving** turns a trained schema into a deployment object.

The guides are narrower references for the mechanics you will reuse across those workflows: schema queries, explicit model mutation, preprocessors, and custom tensorfields.

## Minimal API Shape

```python
import json2vec as j2v

model = j2v.Model.from_schema(
    j2v.Number("sepal_length"),
    j2v.Number("petal_length"),
    j2v.Category("species", target=True, max_vocab_size=4),
    d_model=16,
    n_layers=1,
    n_heads=4,
    batch_size=4,
)

model.set(j2v.where("name") == "record", embed=True)
model.plot(detail=True)
```

`Model.from_schema(...)` builds both the schema tree and the neural modules that match it. Fields default to reading from keys with the same name, while explicit `query=...` expressions can pull values from nested structures.

`model.set(...)` is intentionally explicit. It is how examples turn fields into targets, set masking rates, expose embeddings, or change serving behavior without relying on hidden rate resolution.

## Example Plot

The notebooks show their model plots inline as Rich console output, so the rendered docs match the same `model.plot(detail=True)` call used in examples.
