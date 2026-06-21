<h1 align="center"><code>json2vec</code></h1>

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white" />
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-2E8B57" /></a>
  <a href="https://json2vec.github.io/json2vec/"><img alt="Documentation" src="https://img.shields.io/badge/docs-Quarto-39729E?logo=quarto&amp;logoColor=white" /></a>
  <!-- discord-invite:start -->
  <a href="https://discord.gg/DVyZUkvTFA"><img alt="Discord channel invite" src="https://img.shields.io/badge/discord-join%20the%20channel-5865F2?logo=discord&amp;logoColor=white" /></a>
  <!-- discord-invite:end -->
</p>

`json2vec` builds PyTorch/Lightning models directly from JSON-like schemas.
It is meant for predictive modeling on records that are not naturally flat:
customers with transactions, orders with line items, sessions with clickstream
events, devices recurring across histories, and mixed datatypes at every level.

Most ML pipelines flatten that shape first, then train on one fixed feature
row. `json2vec` takes the opposite path: describe the structured record, and
the schema becomes the model.

## Core Idea

A `json2vec` schema is both a data contract and an architecture blueprint.

- Leaf fields such as `Number`, `Category`, `Set`, `Entity`, `Text`, and
  `Vector` become datatype-specific tensorfields.
- `Branch` nodes define shared contexts for child fields, with optional local
  attention and pooling before the representation flows upward.
- Targets, masks, pruning, and embeddings are configured on the same schema
  tree.
- Prediction output is keyed by schema address, so decoded values and
  embeddings remain attached to the part of the record that produced them.

That gives one model surface for supervised prediction, masked reconstruction,
unsupervised embedding workflows, schema mutation, field importance, batch
inference, and serving.

## A Model From A Nested Record

```python
import json2vec as jv

model = jv.Model.from_tree(
    name="order",
    d_model=64,
    n_layers=2,
    n_heads=4,
    embed=True,
    customer_tier=jv.Category(size=16),
    line_items=jv.Branch(
        length=32,
        embed=True,
        sku=jv.Category(size=2048),
        quantity=jv.Number,
        price=jv.Number,
    ),
    returned=jv.Category(target=True, size=2),
)
```

This model reads records shaped like:

```python
{
    "customer_tier": "gold",
    "line_items": [
        {"sku": "A12", "quantity": 2, "price": 19.99},
        {"sku": "B07", "quantity": 1, "price": 45.50},
    ],
    "returned": "false",
}
```

The `line_items` branch has its own repeated context, `returned` is withheld
from input and decoded as a supervised target, and `embed=True` asks prediction
to emit embeddings at configured addresses.

## Train With Lightning

`jv.Model` is a LightningModule. `jv.PolarsDataModule` and
`jv.StreamingDataModule` are LightningDataModule implementations. The schema
defines the model tree, typed losses, prediction outputs, and embeddings;
Lightning runs `fit`, `validate`, `test`, and `predict`.

```python
import lightning.pytorch as lit
import polars as pl
import torch

import json2vec as jv

records = pl.read_ndjson("docs/data/iris.jsonl").head(36)

model = jv.Model.from_tree(
    d_model=16,
    n_layers=1,
    n_heads=4,
    batch_size=8,
    embed=True,
    optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-2),
    sepal_length=jv.Number,
    petal_length=jv.Number,
    species=jv.Category(target=True, size=4, topk=[2]),
)

datamodule = jv.PolarsDataModule(
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
    enable_model_summary=False,
    enable_checkpointing=False,
    limit_train_batches=1,
    limit_val_batches=1,
)

trainer.fit(model=model, datamodule=datamodule)
```

For larger jobs, the same model can run through normal Lightning callbacks,
checkpointing, precision settings, device placement, and distributed
strategies. See
[Training With Lightning](https://json2vec.github.io/json2vec/guides/lightning.html).

## Predict And Embed

For small interactive batches, call `model.predict(...)` with raw dictionaries.

```python
predictions = model.predict(records.to_dicts()[:3])

species = predictions[jv.Address("record", "species")]
record = predictions[jv.Address("record")]

print(species["content"]["value"])
print(species["content"]["probability"])
print(record["embedding"])
```

For larger offline jobs, configure a `predict` split on a data module and attach
`jv.Writer` to Lightning's prediction loop.

```python
writer = jv.Writer("predictions")

trainer = lit.Trainer(
    accelerator="cpu",
    callbacks=[writer],
    logger=False,
)

predict_datamodule = jv.PolarsDataModule(
    model=model,
    predict=records.drop("species"),
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)

trainer.predict(model=model, datamodule=predict_datamodule)
```

`Writer` creates rank-partitioned Parquet files such as
`predictions/rank-0.parquet`. Use a postprocessor when downstream systems need
flat columns, renamed addresses, redacted payloads, or fewer fields. See
[Batch Inference](https://json2vec.github.io/json2vec/guides/batch-inference.html)
and [Postprocessors](https://json2vec.github.io/json2vec/guides/postprocessors.html).

## Learning Modes

`json2vec` does not maintain separate supervised and self-supervised code
paths. Supervised learning is the special case where a target field is hidden
from the input 100% of the time and decoded from the remaining context.

| Setting | What the model sees | What prediction can emit |
| --- | --- | --- |
| plain input | value is visible | no decoded output unless otherwise configured |
| `target=True` | value is hidden | decoded supervised output |
| `p_mask` | some observed values are hidden during training | decoded reconstruction |
| `p_prune` | whole leaf instances are hidden during training | decoded reconstruction |
| `embed=True` | does not hide the value | embedding at that address |

`target=True` is exact shorthand for `p_prune=1.0`. Use `p_mask` for stochastic
value-level reconstruction with rates lower than `1.0`. Use `embed=True` when
you want a representation returned from prediction.

## Data Modules

Data modules load raw records, apply optional preprocessing, batch
observations, tensorize values from the model schema, apply training-time
masking and target pruning, and hand encoded batches to Lightning.

Choose the data module by where the records live:

| Use case | Module |
| --- | --- |
| Tutorials, tests, notebooks, in-memory Polars frames | `PolarsDataModule` |
| Many local files | `StreamingDataModule` |
| S3-backed datasets | `StreamingDataModule` |
| Distributed training or prediction over large inputs | `StreamingDataModule` |

`StreamingDataModule` supports local paths and `s3://...` roots with `ndjson`,
`parquet`, `feather`, `avro`, `csv`, `orc`, and `json` suffixes. Split
arguments are compiled regular expressions matched against discovered file
paths.

See [Data Modules](https://json2vec.github.io/json2vec/guides/data-modules.html)
for split configuration, sharding, sampling, buffers, and preprocessors.

## What Makes This Different

- **Hierarchical context encoding:** child records interact locally before
  their representation flows upward.
- **Extensible datatypes:** each field type owns validation, tensorization,
  missing-state handling, masking, decoding, loss, metrics, and output writing.
- **Unified training roles:** `target=True`, `p_prune`, and `p_mask` all use the
  same reconstruction path.
- **Embedding trees:** embeddings can come from the root, branches, or selected
  leaves.
- **Schema evolution:** fields can be added, removed, updated, reset, or
  temporarily overridden after construction.
- **Production missingness semantics:** `null`, `padded`, `masked`, and
  `valued` are distinct tensorfield states.
- **Training-serving parity:** queries, preprocessors, tensorization, model
  execution, prediction writing, and postprocessors stay on the same configured
  path.

## Where It Fits

Use `json2vec` when relationships inside the record matter: account histories,
fraud or risk snapshots, order and fulfillment events, flight itineraries,
operations telemetry, user sessions, repeated measurements, or mixed datatype
objects where flattening would discard useful structure.

Use a simpler tabular model when flattening loses no meaningful context. The
point is not to replace every table. The point is to model nested business data
without making a feature table the only representation the model can see.

## What It Does Not Do

`json2vec` stops at the representation and typed prediction layer. It is not a
feature store, governance system, rule engine, authorization layer,
decision-capture system, or audit platform. Those systems can consume
`json2vec` embeddings and predictions, but their policies and operational
controls remain separate concerns.

The open-source layer is the reusable encoder and runtime infrastructure. It
does not require users to publish data, schemas, checkpoints, or model
parameters.

## Install

For local development:

```bash
uv sync
```

The package requires Python `>=3.12`.

Optional extras:

```bash
uv sync --extra text
uv sync --extra serving
uv sync --extra docs
```

The `text` extra installs Hugging Face `transformers`. The `serving` extra
installs FastAPI-backed deployment dependencies. The `docs` extra installs the
Python packages used by the Quarto docs.

## Documentation Map

Start with:

- [Getting Started](https://json2vec.github.io/json2vec/getting-started.html)
- [AI / Expert Quickstart](https://json2vec.github.io/json2vec/ai-quickstart.html)
- [Model Tree](https://json2vec.github.io/json2vec/core-concepts/model-tree.html)
- [Query Paths](https://json2vec.github.io/json2vec/core-concepts/querypaths.html)
- [Built-In Data Types](https://json2vec.github.io/json2vec/core-concepts/data-types.html)
- [Learning Modes & Embeddings](https://json2vec.github.io/json2vec/core-concepts/embeddings.html)
- [Training With Lightning](https://json2vec.github.io/json2vec/guides/lightning.html)
- [Data Modules](https://json2vec.github.io/json2vec/guides/data-modules.html)
- [Batch Inference](https://json2vec.github.io/json2vec/guides/batch-inference.html)

Tutorials and guides:

- [Postprocessors](https://json2vec.github.io/json2vec/guides/postprocessors.html)
- [Field Importance](https://json2vec.github.io/json2vec/guides/field-importance.html)
- [Field Stacking](https://json2vec.github.io/json2vec/guides/field-stacking.html)
- [Branch](https://json2vec.github.io/json2vec/data-types/branch.html)
- [Number](https://json2vec.github.io/json2vec/data-types/number.html)
- [Category](https://json2vec.github.io/json2vec/data-types/category.html)
- [Set](https://json2vec.github.io/json2vec/data-types/set.html)
- [Entity](https://json2vec.github.io/json2vec/data-types/entity.html)
- [DateParts](https://json2vec.github.io/json2vec/data-types/dateparts.html)
- [Vector](https://json2vec.github.io/json2vec/data-types/vector.html)
- [Text](https://json2vec.github.io/json2vec/data-types/text.html)
- [Device Tenure Case Study](https://json2vec.github.io/json2vec/case-studies/device-tenure.html)

Build the docs locally with:

```bash
make render
```

## Repository Layout

- `src/json2vec/architecture`: model assembly, attention, pooling, and routing
- `src/json2vec/data`: dataset fetch/read/process/batch/encode pipeline
- `src/json2vec/inference`: serving and prediction callbacks
- `src/json2vec/logging`: runtime logging callbacks
- `src/json2vec/preprocessors`: preprocessor registry
- `src/json2vec/structs`: pydantic config models, enums, and tree nodes
- `src/json2vec/tensorfields`: tensorfield plugin system and built-in fields
- `tests/`: package test suite
- `docs/`: Quarto project, pages, guides, stylesheets, and sample data

## Development

Run tests:

```bash
uv run pytest
```

Run type and lint checks:

```bash
uv run ty check src/json2vec --output-format concise
uv run ruff check
```

## Community

Join the [`json2vec` Discord](https://discord.gg/DVyZUkvTFA) for questions,
design discussion, and release notes.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.

## References

- `BIBLIOGRAPHY.md`
- `CITATION.bib`
