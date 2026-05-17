# JSON2Vec

`json2vec` is a schema-driven framework for learning embeddings and task
heads directly from nested, semi-structured records without flattening them
into a fixed feature table first.

The central idea is that the schema is the encoder. A declared tree of
arrays and typed fields becomes an addressable neural graph: leaf tensorfield
plugins encode raw values, array nodes aggregate child embeddings with
rotary self-attention and learned-query cross-attention pooling, and
datatype-specific decoders reconstruct masked, targeted, or supervised targets
from the surrounding hierarchy.

This makes `json2vec` a factory for structure-aware encoders rather than a
single domain model. Customer/account/transaction data, flight itineraries,
order fulfillment events, clickstream sessions, and other nested records can
all use the same machinery while keeping their proprietary data, schemas, and
trained checkpoints private.

## What Makes This Different

- **Attributed-distance embeddings.** The model can emit embeddings at any
  configured field or array, not only at the root. That means two observations
  can be similar overall while still exposing which branch of the hierarchy
  accounts for the difference: customer profile, monthly statement, login
  session, transaction history, or any other declared array.
- **Target-trained counterfactuals.** Training can periodically remove whole
  fields, not just mask individual values. At inference time, the
  same mechanism supports zero-shot ablation questions such as "what changes if
  device data is unavailable?" without retraining a separate model for every
  feature-removal scenario.
- **One path for self-supervised and supervised learning.** Masked values,
  targeted fields, and explicit supervised targets all flow through the same
  datatype-specific heads. A new tensorfield type brings its own embedding,
  decoding, loss, and writing logic, so the framework stays reusable as schemas
  grow.
- **Schema evolution is a first-class workflow.** Because modules are addressed
  by the schema tree, fields can be added or removed and selected fields can be
  targeted through programmatic hyperparameters without rebuilding a separate
  feature pipeline.
- **Production semantics for missingness.** `null`, `padded`, `masked`,
  and `valued` are distinct states in the tensorfield type system.
  They are not collapsed into one generic missing-value bucket.
- **Online state lives with the model.** Stateful components such as category
  vocabularies, counters, and numeric normalization state are learned during
  streaming training and serialized with checkpoints, so deployment does not
  depend on a parallel tokenizer or normalizer artifact.
- **Training-serving parity.** The same configured graph is used for fitting,
  validation, testing, batch prediction, and LitServe-backed online inference.

The attributed embeddings and target-trained ablations are model-level
explanation primitives. They help answer where two records differ and how a
prediction changes when an information source is withheld. They are not a
complete compliance story by themselves, but they make governance and audit
layers easier to build on top of the representation layer.

## Where It Fits

Use `json2vec` when the hierarchy is part of the signal:

- customer, account, transaction, statement, device, and session records
- flight itineraries, legs, segments, and events
- orders, shipments, fulfillment events, and support histories
- entities with repeated sub-objects, evolving schemas, and mixed datatypes
- embedding retrieval, anomaly detection, counterfactual ablation, and
  multi-target prediction over nested records

## What It Does Not Do

`json2vec` stops at the representation and typed prediction layer. It does not
try to be a feature store, governance system, rule engine, authorization layer,
decision-capture system, or audit platform. Those systems can consume
`json2vec` embeddings and predictions, but their policies and operational
controls remain separate concerns.

It also does not require users to publish data, schemas, checkpoints, or model
parameters. The open-source layer is the reusable encoder and runtime
infrastructure. Your data stays yours, as does your parameters.
The framework works under the assumption that model parameters will not be shared.

## What Is In This Repository

This repository currently contains:

- the core library under `src/json2vec/`
- tensorfield plugins for `number`, `category`, `set`, `dateparts`, `entity`, `vector`, and `text`
- a processor registry for dataset-specific preprocessing
- a LitServe deployment entrypoint for serving from checkpoints
- a programmatic WandB/Lightning lifecycle example under `examples/`
- tests covering structure loading, data processing, tensorfields, training helpers, logging, and inference
- diagrams plus longer design docs in `docs/`

## Install

For local development:

```bash
uv sync
```

If you want an editable install:

```bash
pip install -e .
```

The package requires Python `>=3.12`.

## Core Concepts

- `Hyperparameters` defines the model tree plus masking, targeting, and embedding controls.
- `Array` nodes describe hierarchical grouping and aggregation.
- Field `Request` nodes declare a `type`, a `query`, and type-specific options.
- `Address` values are stable paths such as `root/account/transaction/amount`.
- `jmespath` queries extract values from each observation.
- `TensorField` instances preserve typed content plus state tokens such as
  `valued`, `null`, `padded`, and `masked`.
- `Parcel` objects carry embeddings from leaves to parent arrays and then up
  the tree.
- `heritage` is the path from a leaf to the root; decoders use that path when
  reconstructing masked, targeted, or supervised targets.
Supported dataset suffixes are:

- `ndjson`
- `parquet`
- `feather`
- `avro`
- `csv`
- `orc`
- `json`

Supported dataset roots are local paths and `s3://...` URIs. If `dataset.root` is `null`, the pipeline runs in processor-driven mode and expects the configured processor to generate observations.

## How The Graph Runs

For each batch:

1. Each field request extracts values with its `jmespath` query.
2. The matching tensorfield plugin tensorizes those values, updates any online
   state allowed for the current split, and records trainable targets when
   masking or targeting occurs.
3. Leaf embedders emit parcels to their parent arrays.
4. Array nodes run bottom-up. Each array concatenates available child
   parcels, optionally applies rotary transformer layers, compresses with
   learned-query cross-attention, and emits a new parcel to its parent.
5. Leaf decoders consume the parcel sequence along their heritage path to
   reconstruct trainable targets.

Random `p_mask` corrupts individual values. Random `p_target` removes whole
field instances across an observation. Hyperparameter-level `target` fields are always
withheld and become supervised targets; `embed` addresses are
serialized as embeddings during prediction.

## Minimal Training Workflow

Hyperparameters and datasets are defined programmatically. Users own the Lightning `Trainer`,
optimizer, scheduler, callbacks, and checkpoint policy.

```python
import lightning.pytorch as lit
import torch

from json2vec import Dataset, DefaultDataModule, Hyperparameters, JSON2Vec, Strata, Suffix

hyperparameters = Hyperparameters.model_validate({
    "d_model": 16,
    "p_mask": 0.15,
    "p_target": 0.05,
    "embed": ["root"],
    "fields": {
        "name": "root",
        "type": "array",
        "attention": "mha",
        "dropout": 0.1,
        "max_length": 1,
        "n_outputs": 1,
        "fields": [
            {
                "name": "identifier",
                "type": "category",
                "query": "[*].id",
                "pooling": "query",
                "max_vocab_size": 1024,
            }
        ],
    },
})

dataset = Dataset(
    root="/path/to/data",
    file_buffer_size=16,
    observation_buffer_size=16,
    processor="default",
    suffix=Suffix.ndjson,
    patterns={strata: r".*" for strata in Strata},
)

model = JSON2Vec(
    hyperparameters=hyperparameters,
    batch_size=32,
    optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-3),
)
data = DefaultDataModule.from_model(
    model,
    dataset=dataset,
    num_workers=4,
    chunk_batch_size=4096,
)

trainer = lit.Trainer(accelerator="auto")
trainer.fit(model=model, datamodule=data)
```

Array `attention` can be `mha`, `gqa`, `mqa`, or `none`. Leaf decoder
`pooling` can be `query` or `mean`.

For a fuller lifecycle with separate pretraining, finetuning, validation, and
testing datasets plus a shared `WandbLogger`, see
`examples/wandb_lifecycle.py`.

To turn a field into a supervised target, include its address in
`hyperparameters.target`. The model will
withhold that field from the encoder and use the same datatype-specific decoder
that is used for masked/targeted reconstruction. To export embeddings, include
field or array addresses in `hyperparameters.embed`.

## Inference And Serving

Batch prediction uses the same model and data module path as training. Prediction outputs are written to `tmp/predictions/`.

Checkpoints carry the Lightning weights, serialized hyperparameters, and
stateful tensorfield state such as online category vocabularies, numeric
normalization buffers, and class-frequency counters. This tight coupling is
intentional: the deployed model should not depend on a separate, manually
synchronized tokenizer or normalizer artifact.

For online serving, the repository exposes `json2vec.inference.deployment.Deployment`, which wraps a checkpoint-backed model in LitServe. Runtime configuration is environment-driven:

- `JSON2VEC_CHECKPOINT` or `CHECKPOINT`
- `JSON2VEC_MAX_BATCH_SIZE`
- `JSON2VEC_BATCH_TIMEOUT`
- `JSON2VEC_WORKERS_PER_DEVICE`
- `JSON2VEC_ACCELERATOR`
- `JSON2VEC_TRACK_REQUESTS`

A minimal serve entrypoint is:

```python
from json2vec.inference.deployment import Deployment

Deployment.serve()
```

## Processor Model

Processors are registered Python callables. The built-in `default` processor is a no-op and returns each observation unchanged.

Custom processors are registered with `@shim(yields=False)` for single-object transformations or `@shim(yields=True)` for generators.

- transformation processors must return a single `dict`
- generator processors may yield `dict` objects or return a `list[dict]`
- every emitted object is wrapped as a single-item root array before tensorization

Configured `dataset.kwargs` are passed into the processor, with unsupported keyword arguments automatically ignored.

## Tensorfield Plugins

The current built-in tensorfield types are:

- `number`
- `category`
- `set`
- `dateparts`
- `entity`
- `vector`
- `text`

Each tensorfield plugin provides a request schema plus the model components needed to encode values, decode predictions, compute losses, and optionally serialize outputs.

The `text` tensorfield requires the optional `transformers` dependency and is not installed by default.

## Runtime Environment

Training, dataloading, logging, callbacks, and checkpointing are ordinary
Python/Lightning concerns. Pass dataloader behavior directly to
`DefaultDataModule` with named arguments such as `num_workers`,
`persistent_workers`, `pin_memory`, `sharding`, and `chunk_batch_size`.

The serving entrypoint still supports deployment process settings:

- `JSON2VEC_CHECKPOINT`
- `JSON2VEC_MAX_BATCH_SIZE`
- `JSON2VEC_BATCH_TIMEOUT`
- `JSON2VEC_WORKERS_PER_DEVICE`
- `JSON2VEC_ACCELERATOR`
- `JSON2VEC_TRACK_REQUESTS`

Supported sharding strategies are `file`, `chunk`, and `record`.

## Repository Layout

- `src/json2vec/architecture`: model assembly, attention, pooling, and parcel routing
- `src/json2vec/data`: dataset fetch/read/process/batch/encode pipeline
- `src/json2vec/inference`: serving and prediction callbacks
- `src/json2vec/logging`: runtime logging callbacks
- `src/json2vec/processors`: processor registry and built-in extensions
- `src/json2vec/structs`: pydantic config models, enums, and tree nodes
- `src/json2vec/tensorfields`: tensorfield plugin system and built-in field types
- `examples/`: programmatic training and evaluation examples
- `tests/`: package test suite
- `docs/summary.typ` and `docs/whitepaper.typ`: longer written documentation

## Diagrams

The repository includes architecture and pipeline diagrams:

![Tree of encoding modules](docs/diagrams/tree.drawio.svg)

![Single array node](docs/diagrams/node.drawio.svg)

![Pipeline stages](docs/diagrams/pipeline.drawio.svg)

![Example configured module tree](docs/diagrams/modules.drawio.svg)

## Development

Run the test suite with:

```bash
uv run pytest
```

Run lint checks with:

```bash
uv run ruff check
```

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.

## References

- `BIBLIOGRAPHY.md`
- `CITATION.bib`
