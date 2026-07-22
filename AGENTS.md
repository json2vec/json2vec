# JSON2Vec Agent Guide

This file is the high-signal context for AI coding agents working in this repo. Prefer it over rediscovering package conventions from scattered tests.

## Package Mental Model

JSON2Vec builds PyTorch/Lightning models from JSON-like schemas. Users normally import the package as:

```python
import json2vec as jv
```

The public surface should stay usable from `json2vec` directly: `Model`, `Branch`, built-in tensorfields, data modules, preprocessors, inference writers, deployment helpers, mutation predicates, and extension base classes.

The schema is the architecture. `Model(...)` receives field constructors and branch nodes, then builds the root branch, tensorfield embedders, context encoders, decoders, losses, and prediction outputs.

## Common API Patterns

Minimal supervised model:

```python
model = jv.Model(
    d_model=64,
    n_layers=2,
    n_heads=4,
    amount=jv.Number,
    merchant=jv.Category(size=4096),
    label=jv.Category(target=True, size=2),
)
```

Nested repeated context:

```python
model = jv.Model(
    d_model=64,
    n_layers=2,
    n_heads=4,
    line_items=jv.Branch(
        length=32,
        sku=jv.Category(size=2048),
        quantity=jv.Number,
    ),
    returned=jv.Category(target=True, size=2),
)
```

Root branch naming is passed to `Model(...)` with `name=...`. The
generated root branch is always a singleton; use child `Branch(length=...)`
for repeated data.

```python
model = jv.Model(
    name="event",
    d_model=32,
    n_layers=1,
    n_heads=4,
    embed=True,
    amount=jv.Number,
)
```

## Gotchas

- Do not use a public `Struct(...)` constructor. Public examples should use `Model(...)` and `Branch(...)`.
- `Model(..., name="customer")` names the generated root branch. Older examples may say `root=...`; update them.
- Inferred request queries are written from one processed observation: `[*].amount`, not `[*][*].amount`.
- `Branch(name="transactions")` makes child default queries like `[*].transactions[*].amount`.
- `Branch(overflow="head")` is the default. Use `overflow="tail"` for recency-ordered histories and `overflow="error"` for strict schemas. The generated root branch uses internal `Overflow.error`.
- `target=True` is shorthand for `p_prune=1.0`; the field is hidden from input and decoded as a supervised target.
- `embed=True` emits an embedding in prediction output. It does not make the field a supervised target.
- `Entity` is for local repeated-identity matching and requires more than one value per observation, usually under a `Branch(length>1)`.
- `DateParts` is for calendar parts. If elapsed time or recency matters, derive a `Number`.
- Preprocessors run before tensorization. Use them for Python logic, windowing, normalization, or splitting one raw record into multiple observations.
- Postprocessors run after prediction writing. Use them to reshape address-keyed outputs for APIs or warehouses.

## Data And Training

Use `PolarsDataModule(...)` for in-memory examples and docs. Keep examples tiny:

```python
datamodule = jv.PolarsDataModule(
    model=model,
    train=records,
    validate=records,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)
```

For quick examples, train with `max_epochs=1`, `limit_train_batches=1`, and `limit_val_batches=1`.

`model.predict([...])` accepts a list of raw dictionaries and returns an address-keyed dictionary. Configured embeddings appear under `"embedding"`.

## Inference

Top-level inference exports:

- `jv.Writer` writes batch prediction output.
- `jv.Postprocessor` is the postprocess callable type.
- `jv.Deployment`, `jv.API`, `jv.Accelerator`, and related serving types are lazy exports that require `json2vec[serving]`.

## Useful Commands

```bash
uv run pytest
uv run pytest tests/test_public_api.py
uv run ty check src/json2vec --output-format concise
make render
```

## Documentation Entry Points

- `docs/index.qmd`
- `docs/getting-started.qmd`
- `docs/ai-quickstart.qmd`
- `docs/core-concepts/querypaths.qmd`
- `docs/core-concepts/data-types.qmd`

When adding docs, prefer runnable inline Python snippets and current public
imports. Keep Quarto pages self-contained; do not depend on external standalone
scripts.
