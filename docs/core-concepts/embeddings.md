# Embeddings & Self-Supervised Learning

`json2vec` can learn useful representations without a labeled target. Configure
reconstruction tasks with `p_mask` or `p_prune`, then request embeddings from the
schema nodes you want to export with `embed=True`.

The same decoder path is used for supervised and self-supervised training:

- `target=True` is exact shorthand for `p_prune=1.0`: it hides a field from
  input and trains a supervised prediction.
- `p_mask` randomly hides individual observed values during training and asks
  the model to reconstruct them.
- `p_prune` randomly removes fields or contexts during training and asks the
  model to reconstruct them.
- `embed=True` emits an embedding for that schema node during prediction. It
  does not make the field a target.

Think of `target=True` as the always-pruned supervised case. It is conceptually
similar to always asking the model to reconstruct the field, but use
`target=True` or `p_prune=1.0` for that behavior. `p_mask=1.0` is not a valid
configuration; masking is stochastic and uses rates lower than `1.0`.

## Root Embeddings

Set `embed=True` on `Model.from_schema(...)` to emit one representation for the
whole record.

```python
import json2vec as j2v

model = j2v.Model.from_schema(
    j2v.Number("amount", p_mask=0.15),
    j2v.Category("merchant", p_mask=0.15, max_vocab_size=4096),
    name="transaction",
    d_model=64,
    n_layers=2,
    n_heads=4,
    embed=True,
)
```

After training, `model.predict(...)` includes the root embedding:

```python
predictions = model.predict(records)
transaction_embedding = predictions[j2v.Address("transaction")]["embedding"]
```

Use root embeddings for whole-record retrieval, clustering, anomaly detection,
deduplication, or downstream models.

## Array Embeddings

Set `embed=True` on an `Array` to emit a representation for a repeated context.

```python
model = j2v.Model.from_schema(
    j2v.Category("customer_id", active=False, max_vocab_size=100_000),
    j2v.Array(
        j2v.Category("event_type", p_mask=0.15, max_vocab_size=128),
        j2v.Category("merchant", p_mask=0.15, max_vocab_size=4096),
        j2v.Number("amount", p_mask=0.15),
        name="events",
        max_length=64,
        embed=True,
    ),
    name="customer",
    d_model=128,
    n_layers=2,
    n_heads=4,
    embed=True,
)
```

This produces an embedding tree with at least two useful addresses:

```python
customer_embedding = predictions[j2v.Address("customer")]["embedding"]
events_embedding = predictions[j2v.Address("customer", "events")]["embedding"]
```

Use array embeddings when the repeated behavior matters independently from the
whole record, such as event streams, transactions, line items, sessions, or
measurements.

## Leaf Embeddings

Leaf fields can also request embeddings. This is useful when a field has local
semantic meaning and you want a representation at that address.

```python
model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Entity("device_id", embed=True),
        j2v.Category("event_type", p_mask=0.10, max_vocab_size=128),
        name="login_sessions",
        max_length=32,
        embed=True,
    ),
    name="customer",
    d_model=64,
    n_layers=2,
    n_heads=4,
    embed=True,
)
```

Use `Entity` for local repeated-identity matching, such as whether the same
device appears multiple times inside one observation. Use `Category` when the
identifier should have a persistent vocabulary across training and prediction.

## A Self-Supervised Schema

This sketch has no label. The model learns by reconstructing masked values and
emits both root and array embeddings. The complete inline version below trains
one tiny CPU batch from the bundled digit records.

```python
import lightning.pytorch as lit
import polars as pl
import torch

import json2vec as j2v

records = pl.read_ndjson("docs/data/digits.jsonl").head(24)

model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Category("row", max_vocab_size=8),
        j2v.Category("column", max_vocab_size=8),
        j2v.Number("intensity", p_mask=0.15),
        name="pixels",
        max_length=64,
        embed=True,
    ),
    name="digit",
    d_model=24,
    n_layers=1,
    n_heads=4,
    batch_size=8,
    embed=True,
    optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-2),
)

datamodule = j2v.PolarsDataModule(
    model=model,
    train=records,
    validate=records,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    observation_buffer_size=32,
    sample_rate=1.0,
)

trainer = lit.Trainer(
    accelerator="cpu",
    max_epochs=1,
    logger=False,
    enable_progress_bar=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    limit_train_batches=1,
    limit_val_batches=1,
)

trainer.fit(model=model, datamodule=datamodule)
predictions = model.predict(records.to_dicts()[:2])

digit = predictions[j2v.Address("digit")]["embedding"]
pixels = predictions[j2v.Address("digit", "pixels")]["embedding"]
```

## Mixed Supervised And Self-Supervised Training

You can combine a supervised target with auxiliary reconstruction tasks.

```python
model = j2v.Model.from_schema(
    j2v.Number("amount", p_mask=0.10),
    j2v.Category("merchant", p_mask=0.05, max_vocab_size=4096),
    j2v.Category("fraud", target=True, max_vocab_size=2),
    name="transaction",
    d_model=64,
    n_layers=2,
    n_heads=4,
    embed=True,
)
```

The target trains the task you care about directly. The masked fields add
reconstruction pressure that can improve the representation, especially when
labels are sparse.

## Exporting Embeddings

For small interactive workflows, call `Model.predict(...)` directly.

```python
predictions = model.predict(records)
root = predictions[j2v.Address("customer")]["embedding"]
events = predictions[j2v.Address("customer", "events")]["embedding"]
```

For larger offline jobs, use `j2v.Writer` with the normal Lightning prediction
loop so embeddings and metadata can be written to Parquet. Add a
[postprocessor](../guides/postprocessors.md) when you need to strip, rename, or flatten the
output for a downstream index.

## Comparing Embeddings

Embeddings from related schema addresses are the easiest to compare directly:
root to root, event-array to event-array, or the same leaf address across
records. Be careful comparing unrelated branches. A customer-level embedding and
a transaction-array embedding may have the same width but are trained for
different context roles.

Cosine similarity is a reasonable starting point for retrieval and clustering,
but validate it against the task. Embeddings are model outputs, not ground-truth
explanations.

## When Not To Use This Pattern

Do not use self-supervised embeddings as a substitute for a validated supervised
target when the deployment decision requires one. Do not add masking to fields
that are mostly noise or leakage. Do not expose embeddings from public APIs
unless downstream consumers need them and privacy review allows it.

Use embeddings when the representation itself is useful: retrieval, clustering,
anomaly detection, weak supervision, transfer learning, or diagnostics.

## Where Next

- Use [Built-In Data Types](data-types.md) to choose leaf fields.
- Use [Query Paths](querypaths.md) to bind source data to schema nodes.
- Use [Postprocessors](../guides/postprocessors.md) to reshape exported embeddings.
- See the [Device Tenure case study](../case-studies/device-tenure.md) for a
  nested risk-modeling example.
