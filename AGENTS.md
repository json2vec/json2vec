# RelFlow Agent Guide

This file is the high-signal context for AI coding agents working in this repo. Prefer it over rediscovering package conventions from scattered tests.

## Package Mental Model

RelFlow builds PyTorch/Lightning models from JSON-like schemas. Users normally import the package as:

```python
import relflow as rf
```

The public surface should stay usable from `relflow` directly: `Model`, `Branch`, built-in tensorfields, data modules, preprocessors, inference writers, deployment helpers, mutation predicates, and extension base classes.

The schema is the architecture. `Model(...)` receives field constructors and branch nodes, then builds the root branch, tensorfield embedders, context encoders, decoders, losses, and prediction outputs.

## Common API Patterns

Minimal supervised model:

```python
model = rf.Model(
    d_model=64,
    n_layers=2,
    n_heads=4,
    amount=rf.Number,
    merchant=rf.Category(size=4096),
    label=rf.Category(target=True, size=2),
)
```

Nested repeated context:

```python
model = rf.Model(
    d_model=64,
    n_layers=2,
    n_heads=4,
    line_items=rf.Branch(
        length=32,
        sku=rf.Category(size=2048),
        quantity=rf.Number,
    ),
    returned=rf.Category(target=True, size=2),
)
```

Root branch naming is passed to `Model(...)` with `name=...`. The
generated root branch is always a singleton; use child `Branch(length=...)`
for repeated data.

```python
model = rf.Model(
    name="event",
    d_model=32,
    n_layers=1,
    n_heads=4,
    embed=True,
    amount=rf.Number,
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
- `Hash` represents large identifiers with static CPU hashes and, by default, device-side salts derived from Lightning `global_step` during training and validation. `deterministic=True` disables that rotation for a persistent representation, which can increase overfitting. Test and prediction are always unsalted.
- `DateParts` is for calendar parts. If elapsed time or recency matters, derive a `Number`.
- Preprocessors run before tensorization. Use them for Python logic, windowing, normalization, or splitting one raw record into multiple observations.
- Postprocessors run after prediction writing. Use them to reshape address-keyed outputs for APIs or warehouses.

## Data And Training

Use `PolarsDataModule(...)` for in-memory examples and docs. Keep examples tiny:

```python
datamodule = rf.PolarsDataModule(
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

- `rf.Writer` writes batch prediction output.
- `rf.Postprocessor` is the postprocess callable type.
- `rf.Deployment`, `rf.API`, `rf.Accelerator`, and related serving types are lazy exports that require `relflow[serving]`.

## Useful Commands

```bash
uv run pytest
uv run pytest tests/test_public_api.py
uv run ty check src/relflow --output-format concise
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
