<p align="center">
  <img src="docs/diagrams/json2vec.png" alt="JSON2Vec logo" width="180">
</p>

<h1 align="center">JSON2Vec</h1>

<p align="center">
  <img alt="Python >= 3.12" src="https://img.shields.io/badge/python-%3E%3D3.12-3776AB?logo=python&logoColor=white">
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-2E8B57"></a>
  <!-- discord-invite:start -->
  <a href="https://discord.gg/DVyZUkvTFA"><img alt="Discord channel invite" src="https://img.shields.io/badge/discord-join%20the%20channel-5865F2?logo=discord&logoColor=white"></a>
  <!-- discord-invite:end -->
</p>

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
infrastructure. Your data stays yours, and so do your parameters.
The framework works under the assumption that model parameters will not be shared.

## What Is In This Repository

This repository currently contains:

- the core library under `src/json2vec/`
- tensorfield plugins for `number`, `category`, `set`, `dateparts`, `entity`, `vector`, and `text`
- a processor registry for dataset-specific preprocessing
- a LitServe deployment entrypoint for serving from checkpoints
- a hello-world Lightning notebook under `examples/hello_world.ipynb`
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

## Hello World Notebook

`examples/hello_world.ipynb` trains a tiny model from an in-memory synthetic
dataset. It demonstrates the full loop: register a streaming processor, declare
a schema, train a supervised category target, then call `predict` and `embed`.

```python
import lightning.pytorch as lit
import torch
from rich.pretty import pprint

import json2vec as j2v


@j2v.shim(yields=True)
def hello_world_records(observation: dict, strata: j2v.Strata):
    records = [
        {"color": "red", "label": "warm"},
        {"color": "orange", "label": "warm"},
        {"color": "yellow", "label": "warm"},
        {"color": "blue", "label": "cool"},
        {"color": "green", "label": "cool"},
        {"color": "purple", "label": "cool"},
    ]

    yield from records


params = j2v.Hyperparameters(
    d_model=16,
    fields=j2v.Array(
        name="record",
        fields=[
            j2v.Category(name="color", query="[*].color", max_vocab_size=16),
            j2v.Category(name="label", query="[*].label", max_vocab_size=8, topk=[2]),
        ],
    ),
    target=j2v.Address("record", "label"),
    embed=j2v.Address("record"),
)

model = j2v.Architecture(
    hyperparameters=params,
    batch_size=4,
    optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-2),
)

datamodule = j2v.StreamingDataModule.from_model(
    model,
    dataset=j2v.Dataset(processor=hello_world_records),
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    file_buffer_size=1,
    observation_buffer_size=32,
    sample_rate=1.0,
)

trainer = lit.Trainer(
    accelerator="cpu",
    max_epochs=20,
    logger=False,
    enable_progress_bar=False,
    enable_model_summary=False,
    enable_checkpointing=False,
)

trainer.fit(model=model, datamodule=datamodule)

batch = [[{"color": "red"}], [{"color": "blue"}]]

pprint(model.predict(batch))
pprint(model.embed(batch))
```

The prediction call returns a typed result for `record/label`; after training on
the toy data, red-like records should classify as `warm` and blue-like records
as `cool`. The embedding call returns the configured `record` embedding for each
input observation.

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

## Processor Model

Processors are registered Python callables. The built-in `default` processor is a no-op and returns each observation unchanged.

Custom processors are registered with `@shim(yields=False)` for single-object transformations or `@shim(yields=True)` for generators.

- transformation processors must return a single `dict`
- generator processors may yield `dict` objects or return a `list[dict]`
- every emitted object is wrapped as a single-item root array before tensorization

Configured `dataset.kwargs` are passed into the processor, with unsupported keyword arguments automatically ignored.

## Tensorfield Plugins

Each tensorfield plugin provides a request schema plus the model components
needed to encode values, decode predictions, compute losses, and optionally
serialize outputs. Built-in tensorfields share the base leaf options `name`,
`query`, `pooling`, `weight`, `n_heads`, `n_linear`, `dropout`, `p_mask`, and
`p_target`.

| Type | Use It For | Key Options |
| --- | --- | --- |
| `number` | Scalar numeric values. Values are padded with explicit state tokens, normalized online during training, embedded with learned Fourier features, and decoded as regression targets. | `jitter`, `n_bands`, `offset`, `alpha`, `objective` (`mae`, `mse`, `huber`) |
| `category` | Single-label categorical values with an online vocabulary stored in the checkpoint. Unknown or overflow labels route to a reserved unavailable bucket instead of becoming `null`. Prediction output includes label probabilities and optional top-k candidates. | `max_vocab_size`, `n_bands`, `p_unavailable`, `topk` |
| `set` | Unordered collections of categorical labels, encoded as a multi-hot vector over an online vocabulary. Strings are treated as one-item sets, iterables as many-item sets, and unknown labels use the reserved unavailable bucket. | `max_vocab_size`, `p_unavailable` |
| `dateparts` | Datetime values represented through selected calendar/time components. Inputs may be native datetimes or strings parsed with a configured pattern. | `dateparts` (`day_of_year`, `week_of_year`, `month_of_year`, `day_of_month`, `week_of_month`, `day_of_week`, `hour_of_day`, `minute_of_hour`), `pattern` |
| `entity` | Hashable identifiers where the useful signal is equality or co-occurrence within the current observation rather than a global vocabulary. Values are re-indexed locally per observation and require at least two slots per observation. | `topk` |
| `vector` | Fixed-width numeric embeddings or dense feature vectors supplied by another model or system. Inputs may be lists, tuples, 1D NumPy arrays, or 1D Torch tensors and are projected into `d_model`. | `n_dim`, `objective` (`l1`, `l2`) |
| `text` | String values encoded by a frozen Hugging Face `AutoModel`, pooled, and projected into `d_model`. Masked or targeted text is trained by reconstructing the encoder representation rather than generating text. | `model_name`, `max_length`, `encoder_batch_size`, `encoder_pooling` (`cls`, `mean`, `pooler`), `objective` (`l1`, `l2`), `revision`, `local_files_only` |

The `text` tensorfield requires the optional `transformers` dependency and is
not installed by default:

```bash
uv sync --extra text
```

## Community

Join the Discord channel for questions, design discussion, and release notes:
<https://discord.gg/DVyZUkvTFA>

## Repository Layout

- `src/json2vec/architecture`: model assembly, attention, pooling, and parcel routing
- `src/json2vec/data`: dataset fetch/read/process/batch/encode pipeline
- `src/json2vec/inference`: serving and prediction callbacks
- `src/json2vec/logging`: runtime logging callbacks
- `src/json2vec/processors`: processor registry and built-in extensions
- `src/json2vec/structs`: pydantic config models, enums, and tree nodes
- `src/json2vec/tensorfields`: tensorfield plugin system and built-in field types
- `examples/`: hello-world training and inference notebook
- `tests/`: package test suite
- `docs/whitepaper.typ`: longer written documentation

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
