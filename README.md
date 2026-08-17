<h1 align="center"><code>relflow</code></h1>

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white" />
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-2E8B57" /></a>
  <a href="https://relflow.github.io/relflow/"><img alt="Documentation" src="https://img.shields.io/badge/docs-Quarto-39729E?logo=quarto&amp;logoColor=white" /></a>
  <!-- discord-invite:start -->
  <a href="https://discord.gg/DVyZUkvTFA"><img alt="Discord channel invite" src="https://img.shields.io/badge/discord-join%20the%20channel-5865F2?logo=discord&amp;logoColor=white" /></a>
  <!-- discord-invite:end -->
</p>

RelFlow builds PyTorch/Lightning models directly from
JSON-like schemas.
It is meant for predictive modeling on records that are not naturally flat:
customers with transactions, orders with line items, sessions with clickstream
events, devices recurring across histories, and mixed datatypes at every level.

Most ML pipelines flatten that shape first, then train on one fixed feature
row. `relflow` takes the opposite path: describe the structured record, and
the schema becomes the model.

## Core Idea

A `relflow` schema is both a data contract and an architecture blueprint.

- Leaf fields such as `Number`, `Category`, `Set`, `Hash`, `Text`, and
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
import relflow as rf

model = rf.Model(
    name="order",
    d_model=64,
    n_layers=2,
    n_heads=4,
    embed=True,
    customer_tier=rf.Category(size=16),
    line_items=rf.Branch(
        length=32,
        embed=True,
        sku=rf.Category(size=2048),
        quantity=rf.Number,
        price=rf.Number,
    ),
    returned=rf.Category(target=True, size=2),
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

`rf.Model` is a LightningModule. `rf.PolarsDataModule` and
`rf.StreamingDataModule` are LightningDataModule implementations. The schema
defines the model tree, typed losses, prediction outputs, and embeddings;
Lightning runs `fit`, `validate`, `test`, and `predict`.

```python
import lightning.pytorch as lit
import polars as pl
import torch

import relflow as rf

records = pl.read_ndjson("docs/data/iris.jsonl").head(36).with_row_index()
train_records = records.filter((pl.col("index") % 3) != 2).drop("index")
validate_records = records.filter((pl.col("index") % 3) == 2).drop("index")

model = rf.Model(
    d_model=16,
    n_layers=1,
    n_heads=4,
    batch_size=8,
    embed=True,
    optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-2),
    sepal_length=rf.Number,
    petal_length=rf.Number,
    species=rf.Category(target=True, size=3, topk=[2]),
)

datamodule = rf.PolarsDataModule(
    model=model,
    train=train_records,
    validate=validate_records,
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

This tiny deterministic split is only a wiring example. Use a representative,
leakage-safe validation design before interpreting the metrics as model quality.

For larger jobs, the same model can run through normal Lightning callbacks,
checkpointing, precision settings, device placement, and distributed
strategies. See
[Training With Lightning](https://relflow.github.io/relflow/guides/lightning.html).

## Predict And Embed

For small interactive batches, call `model.predict(...)` with raw dictionaries.

```python
requests = validate_records.drop("species").head(3).to_dicts()
predictions = model.predict(requests)

species = predictions[rf.Address("record", "species")]
record = predictions[rf.Address("record")]

print(species["content"]["value"])
print(species["content"]["probability"])
print(record["embedding"])
```

For larger offline jobs, configure a `predict` split on a data module and attach
`rf.Writer` to Lightning's prediction loop.

```python
writer = rf.Writer("predictions")

trainer = lit.Trainer(
    accelerator="cpu",
    callbacks=[writer],
    logger=False,
)

predict_datamodule = rf.PolarsDataModule(
    model=model,
    predict=validate_records.drop("species"),
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)

trainer.predict(
    model=model,
    datamodule=predict_datamodule,
    return_predictions=False,
)
```

`Writer` creates rank-partitioned Parquet files such as
`predictions/rank-0.parquet`. Use a postprocessor when downstream systems need
flat columns, renamed addresses, redacted payloads, or fewer fields. See
[Batch Inference](https://relflow.github.io/relflow/guides/batch-inference.html)
and [Postprocessors](https://relflow.github.io/relflow/guides/postprocessors.html).

## Learning Modes

`relflow` does not maintain separate supervised and self-supervised code
paths. Supervised learning is the special case where a target field is hidden
from the input 100% of the time and decoded from the remaining context.

| Setting | What the model sees | What prediction can emit |
| --- | --- | --- |
| plain input | value is visible | no decoded output unless otherwise configured |
| `target=True` | value is hidden | decoded supervised output |
| `p_mask` | sampled configured leaf positions are hidden in train, validation, and test | decoded reconstruction |
| `p_prune` | whole leaf instances are hidden in train, validation, and test | decoded reconstruction |
| `embed=True` | does not hide the value | embedding at that address |

`target=True` is exact shorthand for `p_prune=1.0`. The current `p_mask`
implementation samples configured positions, including null and padded
positions, at rates lower than `1.0`; see
[Dynamic Masking](https://relflow.github.io/relflow/core-concepts/dynamic-masking.html)
for exact selection behavior. Use `embed=True` when you want a representation
returned from prediction.

## Data Modules

Data modules load raw records, apply optional preprocessing, batch
observations, tensorize values from the model schema, apply configured masking
and target pruning in non-predict loops, and hand encoded batches to Lightning.

Choose the data module by where the records live:

| Use case | Module |
| --- | --- |
| Tutorials, tests, notebooks, in-memory Polars frames | `PolarsDataModule` |
| Many local files | `StreamingDataModule` |
| S3-backed datasets | `StreamingDataModule` |
| Distributed training or prediction over large inputs | `StreamingDataModule` |

`StreamingDataModule` supports local `ndjson`, `parquet`, `feather`, `csv`,
`orc`, and `json` inputs. S3 roots use the PyArrow-backed formats—`parquet`,
`feather`, `csv`, `orc`, and `json`; `ndjson` is local-only. Avro is not
supported by the current reader. Split arguments are compiled regular
expressions matched against discovered file paths.

See [Data Modules](https://relflow.github.io/relflow/guides/data-modules.html)
for split configuration, sharding, sampling, buffers, and preprocessors.

## What Makes This Different

- **Hierarchical context encoding:** child records interact locally before
  their representation flows upward.
- **Typed datatype architecture:** each built-in field owns validation,
  tensorization, missing-state handling, masking, decoding, loss, metrics, and
  output writing. The external registration surface is experimental and
  same-process only in the current release.
- **Unified training roles:** `target=True`, `p_prune`, and `p_mask` all use the
  same reconstruction path.
- **Embedding trees:** embeddings can come from the root, branches, or selected
  leaves.
- **Schema evolution:** fields can be added, removed, updated, reset, or
  temporarily overridden after construction.
- **Production missingness semantics:** `valued`, `null`, `padded`, `masked`,
  and reserved `other` are distinct tensorfield states.
- **Training-serving parity:** queries, preprocessors, tensorization, model
  execution, prediction writing, and postprocessors stay on the same configured
  path.

## Where It Fits

Use `relflow` when relationships inside the record matter: account histories,
fraud or risk snapshots, order and fulfillment events, flight itineraries,
operations telemetry, user sessions, repeated measurements, or mixed datatype
objects where flattening would discard useful structure.

Use a simpler tabular model when flattening loses no meaningful context. The
point is not to replace every table. The point is to model nested business data
without making a feature table the only representation the model can see.

## What It Does Not Do

`relflow` stops at the representation and typed prediction layer. It is not a
feature store, governance system, rule engine, authorization layer,
decision-capture system, or audit platform. Those systems can consume
`relflow` embeddings and predictions, but their policies and operational
controls remain separate concerns.

The open-source layer is the reusable encoder and runtime infrastructure. It
does not require users to publish data, schemas, checkpoints, or model
parameters.

## Install

RelFlow requires Python `>=3.12`. The currently documented user installation
is directly from the GitHub repository:

```bash
python -m pip install "relflow @ git+https://github.com/relflow/relflow.git"
```

This follows the repository's default branch. Pin a tag or commit in the Git
URL for reproducible environments; the published documentation currently
tracks `main` and does not retain versioned snapshots.

Install optional functionality from the same source:

```bash
python -m pip install "relflow[text] @ git+https://github.com/relflow/relflow.git"
python -m pip install "relflow[serving] @ git+https://github.com/relflow/relflow.git"
```

Verify the environment:

```bash
python -c "import importlib.metadata; import relflow; print(importlib.metadata.version('relflow'))"
```

For a contributor checkout, use the locked development environment instead:

```bash
uv sync
```

Contributor extras:

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

- [Getting Started](https://relflow.github.io/relflow/getting-started.html)
- [AI / Expert Quickstart](https://relflow.github.io/relflow/ai-quickstart.html)
- [Model Tree](https://relflow.github.io/relflow/core-concepts/model-tree.html)
- [Data Flow](https://relflow.github.io/relflow/core-concepts/data-flow.html)
- [Binding Data](https://relflow.github.io/relflow/core-concepts/binding-data.html)
- [Query Paths](https://relflow.github.io/relflow/core-concepts/querypaths.html)
- [Built-In Data Types](https://relflow.github.io/relflow/core-concepts/data-types.html)
- [Learning Modes & Embeddings](https://relflow.github.io/relflow/core-concepts/embeddings.html)
- [Training With Lightning](https://relflow.github.io/relflow/guides/lightning.html)
- [Data Modules](https://relflow.github.io/relflow/guides/data-modules.html)
- [Evaluation And Metrics](https://relflow.github.io/relflow/guides/evaluation.html)
- [Model Lifecycle](https://relflow.github.io/relflow/guides/model-lifecycle.html)
- [Batch Inference](https://relflow.github.io/relflow/guides/batch-inference.html)
- [Serving](https://relflow.github.io/relflow/guides/serving.html)

Tutorials and guides:

- [Postprocessors](https://relflow.github.io/relflow/guides/postprocessors.html)
- [Field Importance](https://relflow.github.io/relflow/guides/field-importance.html)
- [Field Stacking](https://relflow.github.io/relflow/guides/field-stacking.html)
- [Model Configuration](https://relflow.github.io/relflow/guides/model-configuration.html)
- [Performance And Scaling](https://relflow.github.io/relflow/guides/performance.html)
- [Temporal Validation](https://relflow.github.io/relflow/guides/temporal-validation.html)
- [Schema Mutation](https://relflow.github.io/relflow/guides/schema-mutation.html)
- [Troubleshooting](https://relflow.github.io/relflow/guides/troubleshooting.html)
- [Experimental Custom Tensorfields](https://relflow.github.io/relflow/guides/custom-tensorfields.html)
- [Public API Map](https://relflow.github.io/relflow/reference/public-api.html)
- [Branch](https://relflow.github.io/relflow/data-types/branch.html)
- [Number](https://relflow.github.io/relflow/data-types/number.html)
- [Boolean](https://relflow.github.io/relflow/data-types/boolean.html)
- [Category](https://relflow.github.io/relflow/data-types/category.html)
- [Set](https://relflow.github.io/relflow/data-types/set.html)
- [Hash](https://relflow.github.io/relflow/data-types/hash.html)
- [DateParts](https://relflow.github.io/relflow/data-types/dateparts.html)
- [Vector](https://relflow.github.io/relflow/data-types/vector.html)
- [Text](https://relflow.github.io/relflow/data-types/text.html)
- [Reproducible Iris Case Study](https://relflow.github.io/relflow/case-studies/iris-reproducible.html)
- [Device Tenure Case Study](https://relflow.github.io/relflow/case-studies/device-tenure.html)

Build the docs locally with:

```bash
make render
uv run pytest tests/examples/test_e2e_examples.py
```

## Repository Layout

- `src/relflow/architecture`: model assembly, attention, pooling, and routing
- `src/relflow/data`: dataset fetch/read/process/batch/encode pipeline and preprocessor exports
- `src/relflow/rich.py`: internal Rich diagnostics and rendering support
- `src/relflow/inference`: serving and prediction callbacks
- `src/relflow/logging`: Lightning metric callbacks
- `src/relflow/structs`: pydantic config models, enums, and tree nodes
- `src/relflow/tensorfields`: tensorfield plugin system and built-in fields
- `tests/`: package test suite
- `docs/`: Quarto project, pages, guides, stylesheets, and sample data

## Development

Run tests:

```bash
uv run pytest
```

Run type and lint checks:

```bash
uv run ty check src/relflow --output-format concise
uv run ruff check
```

## Community

Join the [`relflow` Discord](https://discord.gg/DVyZUkvTFA) for questions,
design discussion, and release notes.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.

## References

- `BIBLIOGRAPHY.md`
- `CITATION.bib`
